# MLX Serving Phase 2

## Objective

Increase concurrent request throughput while preserving current single-request
latency and question isolation. Keep one MLX model and one GPU execution owner.
Measure only the current MLX backend; the phase-2 reference is the current MLX
serial execution path with cross-request batching disabled. Within-request
question batching remains enabled in every reference run.

## Evidence And Limits

`runs/mlx-benchmark-restarted.json` records approximately 250 state tokens:

| Workload | HTTP P50 | HTTP P95 |
| --- | ---: | ---: |
| Unique state, 5 questions | 103.12 ms | 112.35 ms |
| Cached state, 5 questions | 47.38 ms | 49.30 ms |
| Cached state, 5 questions, concurrency 4 | 181.22 ms | 183.62 ms |

Before this implementation, `kev/serve.py::_probs` held a global lock during
inference and cache updates. Its latency clock started after lock acquisition,
so `latency_ms` excluded queue wait. The new worker exposes queue timing while
keeping `latency_ms` as model execution time. `kev/mlx_model.py::_branches`
batches questions within one request only.
The concurrency result is consistent with serialization, not a measurement of
GPU saturation. The existing benchmark has only 12 measured serial samples per
case, one text fixture, and 15 concurrent samples. It is a smoke baseline, not
complete latency, accuracy, or isolation acceptance.

## Implementation Checkpoint

The first checkpoint is implemented in `kev/inference.py` and integrated with
`kev/serve.py`: one worker owns the model and prefix cache, HTTP calls use a
bounded non-blocking admission queue, futures carry results and exceptions,
request deadlines are enforced, and shutdown drains or fails pending work.
Same-state cross-request batching is now enabled for MLX: compatible queued
jobs share one prefix computation and one branch batch, while different states
remain separate. Batches are bounded by request count, flattened branch rows,
and estimated prefix-plus-branch tokens; device/cache-level runtime failures
fail closed and require a process restart. Async HTTP routes cancel queued jobs
when the client disconnects; in-flight Metal work remains non-preemptive and its
result is discarded. A 1 ms configurable microbatch window lets concurrent
same-state requests arrive before dispatch; setting it to zero restores the
minimum-latency mode. The worker reports queue wait, worker time, and server time in the returned metadata;
`/api/info` reports worker state and queue depth.

The checkpoint is covered by `tests/test_inference.py`: worker ownership,
bounded admission, timeout cancellation, exception recovery, and shutdown are
tested without loading model weights. The selected backend-independent suite
currently passes with 17 tests (two unrelated tests are deselected). The live
MLX process must be restarted after this change before running API or latency
checks against the new worker.

With the cached current Kev-4B MLX weights, two same-state questions produced
the same branch probabilities as separate execution with maximum absolute
differences of `6.53e-5` and `0.0`. This is a direct model-path check, not yet
an HTTP load result; the running service must be restarted for that measurement.

## Architecture

HTTP validation -> bounded inference queue -> worker encoding -> scheduler
-> prefix resolution -> branch batch -> pointer readout -> per-request futures.

- One dedicated worker owns model initialization, execution, cache mutation,
  MLX array materialization, and teardown. Run one serving process per model/GPU;
  multiple ASGI workers would each allocate a model and scheduler.
- Convert every inference route, including separate and permutation helpers,
  to asynchronous submission. Keep bounded CPU encoding off the event loop.
  Complete event-loop futures through thread-safe callbacks; never mutate an
  asyncio future directly from the worker thread. Health/info routes remain
  responsive during inference and read synchronized metric snapshots.
- Every job carries a request ID, enqueue time, deadline, encoded state key,
  branch rows, and the original question indices. Never route results by state
  key alone: different requests can share a state and have different questions.
- Bound both preprocessing work and encoded queue occupancy. Admit by request,
  token, and estimated working-memory budgets; reject excess work with HTTP 503
  and Retry-After. Bound HTTP body bytes before full parsing and bound question
  and option counts before tokenization. Reserve admission capacity before
  encoding, adjust to actual token counts afterward, and release reservations
  exactly once on every completion/error/cancellation path. Use 413 for oversized
  bodies and 422 for invalid token/shape limits.
- Use strict encoding for this service: reject overlong states rather than
  silently truncate them. The current encoder counts a state delimiter and
  enforces state plus each branch <= max_branch (currently 8192); max_state
  and max_branch are not independent 8192-token allowances. Reject empty
  question/option lists and any job whose smallest execution unit cannot fit
  the configured working-memory budget.
- Preserve FIFO age priority. A bounded scan can find compatible rows, but an
  old job must get the next compatible turn after a configured maximum number
  of bypasses. Rotate split jobs after one bounded batch. A large request does
  not receive an unbounded dedicated turn; individually infeasible rows are
  rejected. Budget prefix work too, because a long prefix pass is non-preemptive.
- Start with zero intentional batch delay: batch jobs already waiting when the
  worker becomes available. Sweep 0/1/2 ms delays only if measured throughput
  gains justify the single-request latency cost.
- A request can span multiple bounded batches. Complete its future only when
  all rows have returned in original order; fail it explicitly on batch errors.
- Remove canceled/expired queued jobs. In-flight GPU work may finish, but its
  result must not be delivered to another request. On shutdown stop admission,
  drain within a deadline, then fail outstanding futures explicitly. Deadlines
  start at admission and include encoding and every queue visit; expiry returns
  504 when the connection is live. Disconnects discard results and free queued
  reservations without pretending to interrupt an in-flight GPU operation.
- Define worker states STARTING, READY, DRAINING, FAILED, and STOPPED. Invalid
  input is rejected before GPU execution. Unexpected GPU/cache errors fail the
  affected batch; fatal or uncertain device-state errors make readiness fail,
  stop admission, and fail pending jobs. Do not retry a possibly corrupted
  batch or reuse its mutable caches. Recovery requires model/worker restart.

## Prefix And Branch Execution

1. Resolve cache hits and group misses by exact encoded state within a model
   instance. Compute a missing prefix once for all waiting jobs using that state.
   Track resident hits, misses, and coalesced misses separately. Maintain an
   in-flight prefix registry independent of the resident LRU; retain a prefix
   until all dependent split jobs finish, even if it is not admitted to the LRU.
   Prefix failure completes all dependents with an error and removes the registry
   entry. Do not resolve all pending misses before running ready branches: select
   a bounded prefix work unit or ready branch batch using age and deadline. A
   single long prefix still blocks the GPU; long-input SLOs must reflect that.
2. First combine rows from requests sharing a state. This extends the existing
   prefix replication path with minimal new cache semantics.
3. Then combine different states of equal prefix length and similar branch
   length. Each row receives a private cache wrapper and correct recurrent/KV
   state. Never feed one request's mutated cache into another request.
   `ArraysCache.state` includes a mutable Python list, and `from_state` assigns
   that list directly in the installed dependency: wrapper cloning alone is
   not sufficient isolation. Merge into fresh batch-owned storage, clone mutable
   containers, and ensure writes cannot reach source arrays. Fingerprint resident
   cache tensors/metadata before and after execution in correctness tests.
4. Unequal prefix lengths require verified per-row attention offsets, masks,
   and recurrent cache behavior in the installed MLX-LM implementation. Enable
   `BatchKVCache.merge` already left-pads KV tensors and sets per-row offsets;
   recurrent `ArraysCache` has different semantics and must not inherit attention
   padding blindly. Test the combined hybrid model, not just cache helpers. Enable
   this only after full-pass parity checks; otherwise use separate batches by
   prefix length. Padding must not become semantic prefix content.
5. Initially run distinct cache-miss prefix passes serially; branch passes still
   batch across jobs. Batch prefix passes only if profiling shows they dominate
   and padded recurrent state extraction passes isolation tests.

Batch limits consider rows, padded branch tokens, prefix attention lengths, and
estimated cache replication memory. Tune limits using measurements; do not
assume larger batches are faster. Long branches use a separate scheduling class
with age promotion to limit padding and head-of-line blocking.

Replace the four-entry-only LRU with a byte budget plus an entry cap. Pin cache
objects referenced by active batches; eviction removes residency but must not
invalidate active references. Account for pinned and temporary memory as well
as resident bytes. Count shared resident/pinned allocations once and track allocator
reserved memory separately from logical tensor bytes. Reserve temporary working
memory before a batch, including hidden activations and attention workspace;
cache nbytes alone is not a peak-memory bound. Include model weights and a
measured safety margin in the total device budget. An oversized cache entry can
be used transiently without becoming resident if its execution fits the budget.
Materialize prefix state before publishing it to consumers.
Cache keys use exact token content and execution semantics; model reload creates
a new cache namespace. No raw state content is written to metric logs.

## Observability

Use monotonic timing and report encoding, queue wait, prefix execution, branch
execution, readout, and server total time. Shared GPU times belong to a batch;
do not sum them as independent per-request GPU consumption. Preserve a clearly
documented inference timing field and separately expose queue and total time.

Record completed requests/s, questions/s, latency percentiles, batch row counts,
padding ratio, queue depth/oldest age, hit/coalescing rates, failures, rejections,
timeouts, resident/pinned cache bytes, and peak GPU memory. HTTP wall time remains
the user-facing acceptance metric, including serialization and transport.

## Proposed Acceptance Targets

These are engineering targets, not claimed results. Pin the machine, model,
dtype, request corpus, length distribution, cache policy, and offered load.
First preserve the pre-scheduler current-MLX baseline (A). Then compare the new
worker with cross-request batching off (B) and on (C), with identical cache
budgets. A vs B detects scheduler overhead; B vs C measures batching gains.
Do not overwrite A or let a slower B redefine the single-request target.

| Metric | Target |
| --- | --- |
| Concurrency 1 HTTP P50/P95 | Each <= 110% of baseline A on the same workload |
| Unique state, about 250 tokens, 1-8 short questions | P50 <= 300 ms; P95 <= 500 ms |
| Concurrency 4, cached state, 5 short questions | P95 <= 150 ms; requests/s >= 1.5x B at the same concurrency and corpus |
| Mixed-state workload at a fixed arrival rate | P95 <= 500 ms; zero errors/rejections/timeouts in the finite acceptance run |
| Overload | Bounded queue/memory; explicit rejections; report offered and completed load |
| Correctness | Report probability max/mean/P99 drift and all choice changes vs serial MLX |
| Isolation | No cross-request result routing or mutable-cache contamination |

Freeze the mixed corpus and arrival rate before tuning C: use a versioned
50% unique / 50% reusable-state request schedule, with recorded state-length
and question-count distributions and a fixed hot set. Establish B's sustainable
rate using the SLO above, then test B and C at 80% of that rate. Separately sweep
C to find its sustainable rate; report throughput and queue growth, not just
latency at an arbitrarily low load. Long-input stress results are separate from
the approximately 250-token headline SLO.

For batching parity, use an initial maximum absolute probability drift gate of
0.005 and require every argmax change to be investigated. This threshold is a
proposed gate, not an observed numerical bound. A near-tie explanation must be
supported by serial probability margins. Freeze the gate before tuning: every
probability vector must be finite, nonnegative, correctly sized, and normalized
within 1e-5. Any argmax change blocks default enablement unless explicitly
accepted with a documented fixture and numerical analysis; being a near tie
does not automatically waive the gate. Report Noul/Score value and confidence
changes as well as Choice changes. Deterministic cross-request leakage
is always a failure. Current-only checks cannot establish fp32 reference parity.

## Benchmark And Correctness Matrix

- Run matched batching-off/on modes of the current MLX service sequentially.
  No old PyTorch backend is loaded. Persist raw per-request observations, seeds,
  workload checksums, actual token lengths, code revision plus dirty diff hash,
  dependency versions, backend settings, errors, and startup metadata. The
  existing runs JSON is only a smoke artifact, not the frozen acceptance corpus.
- Exercise the production `/v1/systemone` endpoint and internal probability
  endpoint. Include Choice, Score, and Noul response mapping.
- Use state lengths near 64/256/1024/4096 tokens; include boundary-length tests.
  Cover 1/3/5/8 questions, variable option counts, and short/long branch mixtures.
  Construct lengths from encoded tokens, including delimiters/options; test valid
  state-plus-branch limits and over-limit rejection separately.
- Separate unique states, one repeated state, a hot set larger than cache
  capacity, and mixed hit/miss traffic. Use run-specific unique state tokens so
  previous runs cannot accidentally warm the miss workload.
- Test concurrency 1/2/4/8/16 and scheduled arrivals independent of completions.
  Measure scheduled-arrival-to-completion as well as dispatch-to-completion, so
  client-side backlog is visible. Report errors and rejected work alongside
  latency; do not improve percentiles by silently excluding failures.
- For headline cases, collect at least 1,000 completed requests per repetition
  and three repetitions after warmup. Use smaller exploratory sweeps first.
  Include a 30-minute mixed-load memory/eviction soak at the selected load.
  Keep cache warmup outside measurement, establish the intended cache occupancy
  separately per case, alternate B/C run order, and check client saturation.
  Report duration and actual completed requests/questions divided by wall time;
  do not infer throughput from a latency percentile. Provide per-run percentiles
  and variation rather than pooling three runs into one apparent large sample.
- Compare each request against serial MLX with identical encoded inputs. Test
  question reordering, duplicate calls, interleaved states of identical length,
  cold/hot/evicted cache, concurrent same-state misses, heterogeneous padding,
  split requests, cancellation, timeout, and worker failure recovery.
- Compare cached/batched execution with the current MLX full-pass path for
  selected fixtures to detect errors shared by the cache-based serial reference.

## Delivery Order

1. Add trustworthy timing and workload benchmarks; freeze a current MLX baseline.
2. [In progress] Introduce a single-owner worker, bounded queue, futures, deadlines,
   and shutdown behavior. The initial queue/lifecycle checkpoint and same-state
   MLX batch path are implemented; byte/memory admission limits and deeper cache
   isolation tests remain. Validate unchanged predictions and latency against A;
   record B.
3. [In progress] Measure same-state cross-request batching, miss coalescing,
   row-budget limits, and result routing. Keep the path disabled for backends
   without the MLX batch methods.
4. Add different-state batches and refine fair scheduling under mixed load.
   Validate cache-type semantics before supporting unequal prefix lengths.
   Same-state batching is an intermediate milestone, not completion of mixed-state
   acceptance. Unequal-length prefix batching and prefix-pass batching remain
   optional experiments gated by correctness and measured benefit.
5. Tune batch limits and delay using the full matrix; publish an acceptance report
   with throughput/latency curves, probability drift, and memory measurements.

Code locations: `kev/serve.py` for HTTP integration,
`kev/inference.py` for scheduler/worker ownership, `kev/mlx_model.py` for batch
execution, `scripts/bench_mlx.py` for load measurement, and targeted scheduler
and MLX integration tests. Implement no quantization or model replacement in
this phase; those change the accuracy tradeoff independently of scheduling.
