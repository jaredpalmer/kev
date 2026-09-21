"""Single-owner inference execution for the serving process.

The MLX model and its mutable prefix cache live on one worker thread. HTTP
handlers submit bounded jobs and wait on futures; they never touch model state
or hold a lock while inference is running.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
from collections import deque
import queue
import threading
import time
from typing import Any, Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


class InferenceBusy(RuntimeError):
    """The bounded inference queue has no admission capacity."""


class InferenceUnavailable(RuntimeError):
    """The worker is stopping or has failed."""


class InferenceTimeout(TimeoutError):
    """A request exceeded its serving deadline."""


class InferenceCancelled(asyncio.CancelledError):
    """The HTTP client disconnected before inference completed."""


@dataclass
class _Job(Generic[T]):
    fn: Callable[[], T]
    future: Future[T]
    submitted_at: float
    deadline: float | None
    batch_key: Hashable | None = None
    payload: Any = None
    started_at: float | None = None


@dataclass
class BatchItem(Generic[T]):
    """One result returned by a batch callback."""

    value: T | None = None
    error: BaseException | None = None
    fatal: bool = False


class InferenceWorker:
    """Own model execution and mutable inference state on one daemon thread.

    Batching is opt-in through ``batch_fn``. Jobs without a batch key always
    execute individually, which keeps the serial baseline available.
    """

    def __init__(self, *, max_queue: int = 64, request_timeout_s: float = 120.0,
                 batch_fn: Callable[[list[_Job]], list[BatchItem]] | None = None,
                 max_batch: int = 1, batch_wait_s: float = 0.0, name: str = "kev-inference"):
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        if batch_wait_s < 0:
            raise ValueError("batch_wait_s must be non-negative")
        self.max_queue = max_queue
        self.request_timeout_s = request_timeout_s
        self.batch_fn = batch_fn
        self.max_batch = max_batch
        self.batch_wait_s = batch_wait_s
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=max_queue)
        self._pending: deque[_Job] = deque()
        self._admitted = 0
        self._state = "STARTING"
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        with self._state_lock:
            self._state = "READY"

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def queue_depth(self) -> int:
        with self._state_lock:
            return self._admitted

    def submit(self, fn: Callable[[], T], *, timeout_s: float | None = None,
               batch_key: Hashable | None = None, payload: Any = None) -> tuple[Future[T], _Job[T]]:
        """Admit one job without blocking, returning its future and metadata."""
        now = time.monotonic()
        timeout_s = self.request_timeout_s if timeout_s is None else timeout_s
        if timeout_s <= 0:
            raise InferenceTimeout("inference deadline has expired")
        future: Future[T] = Future()
        job = _Job(fn=fn, future=future, submitted_at=now, deadline=now + timeout_s,
                   batch_key=batch_key, payload=payload)
        with self._state_lock:
            if self._state != "READY":
                raise InferenceUnavailable(f"inference worker is {self._state.lower()}")
            if self._admitted >= self.max_queue:
                raise InferenceBusy("inference queue is full")
            try:
                self._queue.put_nowait(job)
            except queue.Full as exc:
                raise InferenceBusy("inference queue is full") from exc
            self._admitted += 1
        return future, job

    def run(self, fn: Callable[[], T], *, timeout_s: float | None = None,
            batch_key: Hashable | None = None, payload: Any = None) -> T:
        """Submit and wait, attaching timing to the usual ``(value, meta)`` result."""
        future, job = self.submit(fn, timeout_s=timeout_s, batch_key=batch_key, payload=payload)
        timeout_s = self.request_timeout_s if timeout_s is None else timeout_s
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeout as exc:
            future.cancel()
            raise InferenceTimeout("inference request timed out") from exc
        return self._annotate(result, job)

    async def run_async(self, fn: Callable[[], T], *, timeout_s: float | None = None,
                        batch_key: Hashable | None = None, payload: Any = None,
                        is_disconnected: Callable[[], Any] | None = None) -> T:
        """Await a job without blocking the event loop and cancel on disconnect."""
        future, job = self.submit(fn, timeout_s=timeout_s, batch_key=batch_key, payload=payload)
        timeout_s = self.request_timeout_s if timeout_s is None else timeout_s
        wrapped = asyncio.wrap_future(future)
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    future.cancel()
                    raise InferenceTimeout("inference request timed out")
                if is_disconnected is not None and await is_disconnected():
                    future.cancel()
                    raise InferenceCancelled()
                done, _ = await asyncio.wait({wrapped}, timeout=min(0.05, remaining))
                if done:
                    return self._annotate(wrapped.result(), job)
        except asyncio.CancelledError:
            future.cancel()
            raise

    @staticmethod
    def _annotate(result: T, job: _Job[T]) -> T:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            meta = result[1]
            started = job.started_at or time.monotonic()
            meta.setdefault("queue_wait_ms", round(max(0.0, (started - job.submitted_at) * 1000), 1))
            meta.setdefault("worker_ms", round(max(0.0, (time.monotonic() - started) * 1000), 1))
            meta.setdefault("server_latency_ms", round(max(0.0, (time.monotonic() - job.submitted_at) * 1000), 1))
        return result

    def stop(self, *, timeout_s: float = 5.0) -> None:
        with self._state_lock:
            if self._state in {"STOPPED", "DRAINING"}:
                return
            self._state = "DRAINING"
        self._stop.set()
        self._thread.join(timeout=max(0.0, timeout_s))
        with self._state_lock:
            self._state = "FAILED" if self._thread.is_alive() else "STOPPED"

    def _run(self) -> None:
        try:
            while not self._stop.is_set() or not self._queue.empty() or self._pending:
                jobs = self._take_batch()
                if not jobs:
                    continue
                self._execute(jobs)
        except BaseException as exc:
            with self._state_lock:
                self._state = "FAILED"
            while self._pending:
                job = self._pending.popleft()
                if not job.future.done():
                    job.future.set_exception(InferenceUnavailable(f"inference worker failed: {exc}"))
                with self._state_lock:
                    self._admitted = max(0, self._admitted - 1)
                self._queue.task_done()
            while True:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    break
                if not job.future.done():
                    job.future.set_exception(InferenceUnavailable(f"inference worker failed: {exc}"))
                with self._state_lock:
                    self._admitted = max(0, self._admitted - 1)
                self._queue.task_done()

    def _take_one(self) -> _Job | None:
        if self._pending:
            return self._pending.popleft()
        try:
            return self._queue.get(timeout=0.05)
        except queue.Empty:
            return None

    def _take_batch(self) -> list[_Job]:
        first = self._take_one()
        if first is None:
            return []
        if self.batch_fn is None or first.batch_key is None or self.max_batch == 1:
            return [first]
        batch = [first]
        deferred = []
        deadline = time.monotonic() + self.batch_wait_s
        while len(batch) < self.max_batch:
            try:
                remaining = deadline - time.monotonic()
                candidate = self._queue.get(timeout=remaining) if remaining > 0 else self._queue.get_nowait()
            except queue.Empty:
                break
            if candidate.batch_key == first.batch_key:
                batch.append(candidate)
            else:
                deferred.append(candidate)
        for candidate in reversed(deferred):
            self._pending.appendleft(candidate)
        return batch

    def _finish(self, job: _Job) -> None:
        with self._state_lock:
            self._admitted = max(0, self._admitted - 1)
        # A job taken from the Queue must be marked done exactly once. Jobs in
        # _pending were taken from that queue already, so this is still valid.
        self._queue.task_done()

    def _execute(self, jobs: list[_Job]) -> None:
        ready = []
        now = time.monotonic()
        for job in jobs:
            try:
                if job.future.cancelled():
                    continue
                if job.deadline is not None and now >= job.deadline:
                    job.future.set_exception(InferenceTimeout("inference request timed out in queue"))
                    continue
                job.started_at = time.monotonic()
                ready.append(job)
            except Exception as exc:
                if not job.future.cancelled():
                    job.future.set_exception(exc)
            finally:
                if job.future.cancelled() or (job.deadline is not None and now >= job.deadline):
                    self._finish(job)
        if not ready:
            return
        try:
            if self.batch_fn is not None and len(ready) > 1 and ready[0].batch_key is not None:
                outcomes = self.batch_fn(ready)
                if len(outcomes) != len(ready):
                    raise RuntimeError("batch callback returned the wrong number of results")
                for job, outcome in zip(ready, outcomes):
                    if job.future.cancelled():
                        continue
                    if outcome.error is not None:
                        job.future.set_exception(InferenceUnavailable(str(outcome.error)) if outcome.fatal else outcome.error)
                    else:
                        job.future.set_result(outcome.value)
                fatal = next((outcome.error for outcome in outcomes if outcome.fatal and outcome.error is not None), None)
                if fatal is not None:
                    self._fail_worker(fatal)
            else:
                for job in ready:
                    if not job.future.cancelled():
                        job.future.set_result(job.fn())
        except Exception as exc:
            fatal = isinstance(exc, (MemoryError, SystemError, RuntimeError))
            for job in ready:
                if not job.future.cancelled() and not job.future.done():
                    job.future.set_exception(InferenceUnavailable(str(exc)) if fatal else exc)
            if fatal:
                self._fail_worker(exc)
        finally:
            for job in ready:
                self._finish(job)

    def _fail_worker(self, exc: BaseException) -> None:
        """Stop admission after a device/cache-level failure.

        The model and mutable caches are not safe to reuse after these errors;
        recovery requires restarting the serving process.
        """
        with self._state_lock:
            self._state = "FAILED"
        self._stop.set()
        failure = InferenceUnavailable(f"inference worker failed: {exc}")
        while self._pending:
            job = self._pending.popleft()
            if not job.future.done():
                job.future.set_exception(failure)
            self._queue.task_done()
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if not job.future.done():
                job.future.set_exception(failure)
            self._queue.task_done()
        with self._state_lock:
            self._admitted = 0
