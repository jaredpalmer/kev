# Indexed recurrent-state reads on Kev-4B

Baseline: `2074cc44256200bb37540103fa4572901faa10f3`.
Candidate implementation: `8e339ae34979678fc154366f1d1f06a50cb49e2b`.
Related to [#107](https://github.com/jaredpalmer/kev/issues/107).
Measured on 2026-10-24. Historical issue measurements are not used as results.

A question row currently gathers the document's recurrent state with
`index_select`, then copies that result into its row buffer. Multiple questions
repeat this work. The candidate passes the bank tensor and the row's bank-entry
index to an FLA 0.5.2 state-scan kernel. It changes only the initial-state address;
all question-local intermediates remain separate and final-state writes are off.
The index stride is explicit because entries are a column of the graph upload
buffer. The original gathered path remains for unfused backbones.

## Environment and scope

H100 80GB HBM3, driver 610.57.04, Torch 2.8.0 with CUDA 12.8,
Transformers 5.17.0, FLA and fla-core 0.5.2, Triton 3.8.0,
causal-conv1d 1.7.0. The repository image installs its locked dependencies, then
its existing FLA/Triton and convolution packages. The canonical kernel environment is recorded in `comparison-final.json`.

Checkpoint: `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52`.
Serving uses merged bf16 weights, fused layers and CUDA graphs. The initial
baseline launch resolved the public repo's default revision; the cached snapshot
was recorded before implementation, and subsequent reads pinned that revision.
Each controlled comparison uses one loaded checkpoint on one physical GPU for
both paths. Separate runs use separate H100 containers, not a guaranteed shared
physical GPU. CPU and network end-to-end latency are outside this measurement.

No 9B, 27B, other GPU architecture, backward pass, or training claim is made.
Convolution-state and attention-cache gathers remain. Row-buffer allocations
remain too. The vendored scan retains FLA's configurations and arithmetic and
requires maintenance when that pin changes; its MIT license is retained under
`docs/licenses/`.

## Criteria set before candidate results

The candidate must preserve probabilities bit for bit against the current
serving calculation, with zero answer flips. Kernel tests must preserve the bank
exactly and honor changed, duplicated and strided indices across graph replays.
Any numerical difference must be investigated rather than widening a tolerance.
Performance claims require measured improvement with identical inputs and a
shared environment. Correctness and speed are separate criteria.

## Controlled model measurements

`scripts/indexed_state_bench.py` calls the serving model's `probs_batch` on
identical encoded inputs with the original gathered path and with indexed reads.
It rebuilds graph buffers per path, warms three calls and captures pending graphs
before timing 15 calls. Only the final controlled measurement JSON is committed;
other runs are summarized here to avoid redundant artifacts. CUDA events bracket each call and synchronize its end;
wall times are saved too. Event intervals include upload and submission gaps,
not merely a profiler sum of kernel durations. Initialization, warmup and capture
wall time are recorded separately as `first_seconds`.

Requests use the existing six-question support-triage workload. Shared-document
batches repeat a ticket; distinct documents have different ticket numbers. The
long-branch case appends twenty copies of the same instruction to each question.
Cold calls recompute all prefixes; mixed calls reuse half. Three requests cover
padded state and row buckets. Two exploratory runs reverse the order of the gathered and
indexed paths. Their aggregate results are summarized below. Their recurrent kernel is
the final kernel; the gather loop was subsequently simplified without changing
its copies. A third confirmation, `comparison-final.json`, records source hashes
matching commit `8e339ae` exactly for all three changed core modules.

Median GPU milliseconds, baseline then candidate:

| Workload | Gathered first | Indexed second | Gathered second | Indexed first |
|---|---:|---:|---:|---:|
| 1 request, warm | 12.469 | 11.576 | 12.755 | 11.806 |
| 16 requests, shared document, warm | 105.542 | 92.184 | 106.872 | 93.597 |
| 3 distinct documents, warm | 31.300 | 28.122 | 31.839 | 28.899 |
| 16 distinct documents, warm | 106.064 | 92.352 | 107.808 | 94.303 |
| 16 distinct documents, longer branches | 322.256 | 307.778 | 319.337 | 305.147 |
| 16 distinct documents, cold | 111.467 | 98.590 | 111.406 | 98.160 |
| 16 distinct documents, mixed | 107.241 | 93.981 | 107.513 | 93.939 |

The warm 16-document case reduces median model latency by 12.9% and 12.5% in
the two orders. Its wall-time throughput is 150.8 to 173.2 requests/s in the first
run and 148.4 to 169.6 in the reverse-order run. Longer branches reduce the gain
to about 4.5%. These are two timing trials, not a population confidence interval.

Across all seven cases, all 504 question outputs match bit for bit in each run:
maximum absolute probability difference 0, answer flips 0/504. Every graph
capture reported success. The same question outputs are repeated in the second
run, so these are not 1,008 distinct examples.

Peak allocated memory in the warm 16-document case was 15,459,513,344 versus
15,492,871,168 bytes in the first run. In reverse order it was 15,492,543,488
versus 15,459,513,344 bytes. The second path retains about 33 MB more regardless
of which path it is; this does not establish a memory saving or regression.
This change removes copies, not row-buffer
allocations.

The final committed-source confirmation measures the warm 16-document batch at
106.050 ms gathered and 91.979 ms indexed, a 13.3% reduction. Wall-time throughput
is 150.8 versus 173.9 requests/s. It also preserves all 504 question outputs bit
for bit. This confirmation, together with its source hashes, is the primary
candidate measurement; the earlier two runs check consistency and order effects.

## Isolated copy baseline

`scripts/deltanet_gather_bench.py` reproduces the existing recurrent gather/copy
expression over all 24 DeltaNet layers, each with bank shape
`[16, 32, 128, 128]`, fp32 states, and seed 0. It warms ten calls, captures the
copy graph, warms ten replays, then measures seven samples of 100 replays with
CUDA events. It covers 1, 6, 8, 16, 32, 64, 96 and 128 rows, reading either one
or sixteen bank entries. At 96 rows the median is 5.062 ms for one entry and
5.367 ms for sixteen entries. This is recurrent copy cost only, with no attention,
convolution or model computation. The candidate removes this operation; it still
reads the state in its scan. The net result must be taken from the controlled
model comparison, not inferred from this isolated number.

## Reproduction

These commands use a minimal launcher around `modal_app.py`'s unchanged image
and benchmark functions, so a contributor workspace does not need the unrelated
snapshot mirror's `huggingface-secret`. Authenticate Modal first. Names must be
new on every run. GPU calls time out after one hour and do not retry.

```sh
uvx --python 3.13 --from 'modal==1.5.5' --with 'cbor2<6' modal run runs/indexed-delta-4b/launch.py --kind compare --name indexed-compare-NEW --flags '--reps 15'
uvx --python 3.13 --from 'modal==1.5.5' --with 'cbor2<6' modal run runs/indexed-delta-4b/launch.py --kind compare --name indexed-reversed-NEW --flags '--reps 15 --indexed-first'
uvx --python 3.13 --from 'modal==1.5.5' --with 'cbor2<6' modal run runs/indexed-delta-4b/launch.py --kind gather --name indexed-gather-NEW
uvx --python 3.13 --from 'modal==1.5.5' --with 'cbor2<6' modal run runs/indexed-delta-4b/launch.py --kind serving --name indexed-serving-NEW --flags '--n 8 --reps 5 --isolation'
```

For the original serving baseline, check out its SHA in another worktree and
copy `launch.py` into the same relative path there. Run the last command there
with a new name. The other scripts are candidate-side controlled comparisons.
Results are in Modal volume `kev-runs` under `/scripts/NAME/report.json` for
compare/gather and `/serving/NAME/report.json` for serving. Fetch the exact file:

```sh
uvx --python 3.13 --from 'modal==1.5.5' --with 'cbor2<6' modal volume get kev-runs /scripts/indexed-compare-NEW/report.json /tmp/indexed-compare.json
```

The local `cbor2<6` constraint is only an isolated CLI workaround for this Intel
Mac's missing Rust toolchain. No project dependency or lockfile was changed.
The pinned Torch release has no Intel macOS wheel, so verification ran on Linux.

## Full serving benchmark

The existing `scripts/serving_bench.py --n 8 --reps 5 --isolation` was run before
implementation and again on the final implementation. Both load the same
checkpoint revision. This is a separate-container comparison, with scheduler
batch formation free to vary. The aggregate results are summarized below.

| In-process workload | Baseline req/s | Candidate req/s |
|---|---:|---:|
| Six questions, new short state, 32 clients | 106.2 | 117.3 |
| Six questions, new short state, 64 clients | 112.8 | 120.1 |
| Mixed decision-v7, 32 clients | 132.7 | 131.4 |
| Mixed decision-v7, 64 clients | 128.3 | 121.2 |
| Five questions, 2,200-token state, 64 clients | 14.5 | 14.4 |

The mixed-traffic result is a regression in this run. It does not support a
universal serving-throughput improvement. Even the unchanged eager path was
slower in the candidate container, for example 92.8 to 107.3 ms on six questions
with a new short state. This suggests environment variation, but does not by
itself explain away a regression. An additional within-container ABBA mixed
traffic experiment investigates it.

Both reports have identical numerical reference/parity and isolation blocks.
The graph path differs from eager bf16 by at most 0.007879853, and from the fp32
reference by at most 0.016447008, with zero flips on the eight questions. These
are existing precision differences, identical in baseline and candidate. Direct
gathered-versus-indexed equality is established by the controlled comparison,
not by these unrelated-reference tolerances. Captures: 203 baseline and 200
candidate, with zero failed captures.

## Verification

- New CUDA kernel tests: 8 passed on H100, covering fp32 and bf16 inputs,
  lengths 1/17/64/129, batches 1/3/8/17, duplicated and strided indices, changed
  indices between replays, and byte-identical bank contents after each read.
- Existing `tests/test_model.py::test_cuda_graphs_match_eager`: 1 passed on H100
  with pinned Kev-0.8B revision `9a45d25eb2ab761841196625383fa1dff0e56c1e`.
- Required fast suites: 527 passed, 26 skipped on Linux CPU. The complete tracked
  repository was mounted because the research image omits some test fixtures.
  Optional FLA and causal-conv1d were removed only in that disposable CPU
  container: their CUDA convolution cannot accept the CPU tensors these tests
  exercise. GPU tests and measurements retain the original CUDA image.
- `uv lock --check` passes. The only `pyproject.toml` change includes the retained
  MIT license in distributions; dependency requirements and `uv.lock` are unchanged.
- `uv build --wheel` passes. Wheel inspection confirms both `kev/indexed_delta.py`
  and the FLA MIT license are included. Python compile and `git diff --check` pass.

CUDA test reproduction on the repository image:

```sh
uv run modal run modal_app.py::gpu_tests --gpu H100 --tests 'tests/test_indexed_delta.py tests/test_model.py::test_cuda_graphs_match_eager'
```

Fast suites, in a supported Python environment without the optional CUDA kernels:

```sh
uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py tests/test_documents_tools.py tests/test_hard_v1.py tests/test_devtools_v1.py tests/test_breadth_v1.py tests/test_rounds.py tests/test_skill_scripts.py -q
```

## Mixed-traffic investigation

The additional ABBA experiment uses one loaded model and GPU, with a fresh Server
and graph set per phase. Each phase runs the original seeded 256-record mixed
workload at 32 and 64 clients, warms/captures, and measures a second pass. The
first attempt swapped graphs on a running Server; its worker retains its startup
graph object, so idle capture never finished. That attempt was stopped and has
no reported throughput result. The corrected experiment starts a fresh Server
for every graph set.

Second-pass requests/s, in order, from the mixed-traffic investigation:

| Phase | Path | 32 clients | 64 clients |
|---|---|---:|---:|
| 1 | Gathered | 109.69 | 122.12 |
| 2 | Indexed | 113.22 | 120.07 |
| 3 | Indexed | 114.16 | 130.86 |
| 4 | Gathered | 110.57 | 120.01 |

The 64-client regression from the separate-container benchmark does not repeat
consistently in this experiment. It also does not establish a stable serving
throughput gain. These second passes meet new graph shapes: 21 to 27 captures
remain pending at the end, so these are not fully warmed steady-state measurements. The controlled fixed-shape experiment, which captures its
known shapes before timing, remains the primary performance claim.

Reproduce the investigation with a new name:

```sh
uvx --python 3.13 --from 'modal==1.5.5' --with 'cbor2<6' modal run runs/indexed-delta-4b/launch.py --kind compare --name indexed-mixed-NEW --flags '--mixed-throughput'
```

Structural review found no remaining blocker. The scan stays in its own module,
state-bank selection stays in CUDA graph construction, and measurement scripts
use the canonical device, environment and JSON helpers. The large vendored
kernel is retained rather than generated or patched at runtime so its change
against the pinned FLA source is inspectable. No deployment or merge was made.
