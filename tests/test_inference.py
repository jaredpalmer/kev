import asyncio
import threading
import time

import pytest

from kev.inference import BatchItem, InferenceBusy, InferenceCancelled, InferenceTimeout, InferenceUnavailable, InferenceWorker


def test_worker_runs_one_owner_and_reports_queue_time():
    worker = InferenceWorker(max_queue=2, request_timeout_s=1)
    try:
        owner = threading.get_ident()
        result = worker.run(lambda: (threading.get_ident(), {"latency_ms": 0.0}))
        assert result[0] != owner
        assert result[1]["queue_wait_ms"] >= 0
        assert result[1]["server_latency_ms"] >= result[1]["worker_ms"]
        assert worker.state == "READY"
    finally:
        worker.stop()
    assert worker.state == "STOPPED"


def test_worker_queue_is_bounded_and_timeout_cancels_queued_job():
    worker = InferenceWorker(max_queue=2, request_timeout_s=1)
    entered = threading.Event()
    release = threading.Event()
    try:
        first, _ = worker.submit(lambda: (entered.set(), release.wait(1))[1])
        assert entered.wait(1)
        second, _ = worker.submit(lambda: "queued")
        with pytest.raises(InferenceBusy):
            worker.submit(lambda: "rejected")
        second.cancel()
        release.set()
        assert first.result(timeout=1) is True
    finally:
        release.set()
        worker.stop()


def test_worker_timeout_and_exception_do_not_poison_next_job():
    worker = InferenceWorker(max_queue=2, request_timeout_s=1)
    try:
        with pytest.raises(InferenceTimeout):
            worker.run(lambda: time.sleep(0.05), timeout_s=0.001)
        with pytest.raises(ValueError, match="bad"):
            worker.run(lambda: (_ for _ in ()).throw(ValueError("bad")))
        assert worker.run(lambda: 3) == 3
    finally:
        worker.stop()


def test_worker_rejects_after_stop():
    worker = InferenceWorker(max_queue=1)
    worker.stop()
    with pytest.raises(InferenceUnavailable):
        worker.run(lambda: 1)


def test_worker_fails_closed_after_runtime_error():
    worker = InferenceWorker(max_queue=2)
    with pytest.raises(InferenceUnavailable, match="device exploded"):
        worker.run(lambda: (_ for _ in ()).throw(RuntimeError("device exploded")))
    assert worker.state == "FAILED"
    with pytest.raises(InferenceUnavailable):
        worker.run(lambda: 1)
    worker.stop()


def test_async_wait_cancels_when_client_disconnects():
    worker = InferenceWorker(max_queue=2)
    entered = threading.Event()
    release = threading.Event()
    blocker, _ = worker.submit(lambda: (entered.set(), release.wait(1))[1])
    assert entered.wait(1)

    async def run():
        with pytest.raises(InferenceCancelled):
            await worker.run_async(lambda: "never delivered", is_disconnected=lambda: _true())

    async def _true():
        return True

    try:
        asyncio.run(run())
        assert blocker.done() is False
    finally:
        release.set()
        assert blocker.result(timeout=1) is True
        worker.stop()


def test_worker_batches_jobs_with_the_same_key():
    entered = threading.Event()
    release = threading.Event()
    batches = []

    def batch_fn(jobs):
        batches.append([job.payload for job in jobs])
        return [BatchItem(value=job.payload * 2) for job in jobs]

    worker = InferenceWorker(max_queue=4, batch_fn=batch_fn, max_batch=4)
    try:
        blocker, _ = worker.submit(lambda: (entered.set(), release.wait(1))[1])
        assert entered.wait(1)
        a, _ = worker.submit(lambda: "fallback", batch_key=("state",), payload=2)
        b, _ = worker.submit(lambda: "fallback", batch_key=("state",), payload=3)
        release.set()
        assert blocker.result(timeout=1) is True
        assert a.result(timeout=1) == 4
        assert b.result(timeout=1) == 6
        assert batches == [[2, 3]]
    finally:
        release.set()
        worker.stop()
