# Kev on Ascend NPU

Kev's Qwen3.5 checkpoints serve on an Ascend 910B2 with `kev.serve --device npu:0`. The Gated DeltaNet mixer runs on
vllm-ascend's kernels (`kev/npu_qwen35.py`, `accelerate`), 3× faster than transformers' PyTorch fallback, and for serving
the same module rewrites the whole decoder layer with Ascend ops and replays each pass from a captured NPU graph (`fuse`,
on by default on an NPU): Kev-27B answers a `decision-v7` development record in 116 ms instead of 422 ms, with the same
answers. `kev/fused_qwen35.py` and `kev/cuda_graphs.py` stay CUDA-only.

## Requirements

Measured with this stack; other versions are untested:

| | |
|---|---|
| NPU | Ascend 910B2 (64 GB HBM, ~61 GiB usable by a process) |
| CANN | 9.0.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post2 |
| vllm-ascend | 0.23.0 (with triton-ascend 3.2.1); only its kernels are used, not vLLM's engine |
| transformers / peft | 5.5.4 / 0.21.0 |

Install Kev into the Python that has CANN's torch and torch_npu, and do not `uv sync` there: Kev's lock pins
`torch>=2.6,<2.9` (CUDA wheels on Linux), so a sync replaces the NPU's torch. Install the rest with pip (`peft`,
`accelerate`, and the `serve` extra's `fastapi`, `uvicorn`, `typesafe-sdk` for the server) and check torch is unchanged
afterwards. pyproject asks for
`transformers>=5.17`; 5.5.4 already has Qwen3.5 and is what every number here ran on.

Source CANN's environment first (`source /usr/local/Ascend/ascend-toolkit/set_env.sh`, or your install's path). It puts
the op compiler's Python packages (`tbe`) on `PYTHONPATH`, so extend that variable rather than replace it.

## Serving

```bash
scripts/serve_kev_npu.sh <card> [port] [run] [kev.serve flags...]
scripts/serve_kev_npu.sh 0                               # Kev-27B (v1 LoRA) on card 0, 127.0.0.1:8009
scripts/serve_kev_npu.sh 1 8010 jaredpalmer/kev-9b@v1    # Kev-9B on card 1
```

The script pins the process to one physical card with `ASCEND_RT_VISIBLE_DEVICES` (the process then sees it as
`npu:0`), refuses a card that already runs a process, and refuses a taken port before the load. Check a card with
`npu-smi info` first: a card can show a full HBM bar with an empty process table (memory held by a process in another
container), so read the HBM column as well. The script's header documents every `KEV_*` switch and the long-state
trade-off below.

The same in Python:

```python
import torch, torch_npu  # torch_npu before any .to("npu")
from kev.checkpoint import Checkpoint, LoadOptions
model = Checkpoint("jaredpalmer/kev-27b@v1-lora").load("npu:0", LoadOptions(dtype=torch.bfloat16, fused=True))
```

`fused=True` is opt-in for library callers, as on CUDA; `kev.serve` turns it on for an NPU. The DeltaNet kernels
(`LoadOptions.npu_kernels`, `KEV_NPU_KERNELS=0` to decline) are on for every NPU load, since they equal the reference to
bf16 rounding. `prewarm()` (vllm-ascend's custom ops) runs inside `load`, before the model touches the device; calling
the kernels after something else initialised the NPU makes `chunk_gated_delta_rule_fwd_h` fail to launch. Without
vllm-ascend, `load` warns and runs the reference layers.

Scoring a frozen suite works as on CUDA, with the DeltaNet kernels and without the serving-only fused layers:

```bash
ASCEND_RT_VISIBLE_DEVICES=<card> python -m kev.benchmark --run jaredpalmer/kev-9b --suite evals/v4/transfer-v4 \
    --device npu:0 --out runs/<name>
```

## Gated DeltaNet kernels

`flash-linear-attention` and `causal-conv1d` are CUDA-only, so on an NPU transformers runs the Gated DeltaNet with
`torch_chunk_gated_delta_rule`: fp32 with a 64-step Python loop per chunk for the WY (UT-transform) inverse. On a 910B2
that loop dominates a decision: 24 DeltaNet layers at ~17 ms each, a 165-token prefill spending 409 of its 478 ms there.

`kev/npu_qwen35.py` (`accelerate`, wired into `kev.checkpoint.Checkpoint.load` for any `npu` device, `KEV_NPU_KERNELS=0` to
decline) replaces each `Qwen3_5GatedDeltaNet.chunk_gated_delta_rule` with a driver over vllm-ascend's kernels: the
triton_ascend helpers in `vllm_ascend/ops/triton/fla/` (`chunk_local_cumsum`, `chunk_scaled_dot_kkt_fwd`, `solve_tril`,
`recompute_w_u_fwd`, `l2norm_fwd`) followed by the AscendC ops `torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h` and
`chunk_fwd_o`. Each of Kev's causal rows (state + one question branch) is packed as one variable-length segment of a
`cu_seqlens`, which is the contract those kernels already expect from vLLM's serving path; the wrapper
`vllm_ascend.ops...chunk.chunk_gated_delta_rule` itself cannot be reused because it reads vLLM's forward context and
prefill-context-parallel group, so this module drives the same kernels directly.

Two things had to be right for a single-node, prefill-only caller:
- `enable_custom_op()` must run before the backbone initialises the NPU device (vllm-ascend defers its custom-op
  extension so `ASCEND_RT_VISIBLE_DEVICES` can still change before `torch.npu.set_device()`). Called after the device is
  live, `chunk_gated_delta_rule_fwd_h` fails to launch (`AclNN_Parameter_Error: ADD_TO_LAUNCHER_LIST_AICORE failed`).
  `Checkpoint._load_torch` therefore calls `npu_qwen35.prewarm()` before building the model.
- `fwd_h` reads its state buffer even for a first chunk. Passing `initial_state=None` (which Kev does at every call site)
  left uninitialised memory and produced NaNs or garbage for some lengths; a zero initial state is passed instead, the
  same math. Inputs are also made contiguous first — the model hands `value` in as a non-contiguous view, and the
  triton_ascend helpers read by stride and return garbage otherwise.

The result equals the fp32 reference to bf16 rounding: on Kev's dims and lengths 40..864, batch 1..16, the core output
matches to max |dp| ~5e-4; end to end over decision-v7 development records, Kev-9B is max |dp| 0.012 with 0 argmax flips
against the reference DeltaNet, and the 100-record sample below is identical in accuracy and ECE.
`tests/test_npu.py::test_chunk_kernel_matches_the_reference_recurrence` checks the kernel on a card.

### Measured (one 910B2, `jaredpalmer/kev-9b@v1`, 100 decision-v7 development records, seed 0, 119 questions)

| | reference (`torch_chunk_gated_delta_rule`) | Ascend kernels |
|---|---|---|
| latency mean | 586 ms | 217 ms |
| latency median | 529 ms | 176 ms |
| latency p95 | 1057 ms | 185 ms |
| accuracy | 0.9328 | 0.9328 |
| ECE | 0.0613 | 0.0613 |
| NLL | 0.2497 | 0.2500 |

`runs/npu-kev-9b-sample/` is the reference path, `runs/npu-kev-9b-fla/` the kernel path (bf16, adapter unmerged, eager
attention, not fused). The per-record `max` (4.3 s) is one-time triton_ascend kernel compilation for a newly seen row
shape; the median and p95 are the steady state.

## Fused layers and captured graphs (Kev-27B)

With the mixer on those kernels, a Kev-27B pass is no longer waiting on the NPU: a 165-token decision was **6,539 kernel
launches, 166 ms of device time and 315 ms of wall time**, and the wall time equalled the time the host spent enqueueing
it. Two things caused that, and `fuse(lm)` in `kev/npu_qwen35.py` (`LoadOptions(fused=True)`, which also merges the
adapter; `KEV_FUSED=0` to decline) addresses both.

**The reference layer is dozens of small ops.** Every RMSNorm is six elementwise kernels, the rotary is ten, the MLP's
gate and up are separate GEMMs, and each mixer runs four projections. The rewrite is the Ascend counterpart of
`kev/fused_qwen35.py`: one projection GEMM per mixer (q/k/v, z, b, a concatenated), one for attention (q with its output
gate, k, v), one for the MLP (gate and up), `torch_npu.npu_rms_norm` for every norm (weight `1 + w` kept in fp32:
relative error against the fp32 reference 1.4e-3 instead of 5.5e-3 if it is rounded to bf16 first),
`npu_rotary_mul` for the rotary (partial: cos and sin cover the leading 64 of each head's 256 dimensions, so only that
slice is rotated), `npu_swiglu`, and `npu_fused_infer_attention_score` in place of the explicit scores. Weights are cast
to `FRACTAL_NZ`, the layout the cube unit reads, which is bit-identical and stops the runtime converting on every call
(13 ms of that pass). `torch_npu` only makes internal-format tensors while `torch.npu.config.allow_internal_format` is
on, and that flag also sends the mixer's depthwise conv down the legacy aclop path, which a graph cannot capture, so it
is turned on for the cast alone. That took the pass to 3,668 launches, 142 ms of device time, 205 ms of wall time.
`fuse` covers the dense Qwen3.5 MLP; it refuses mixture-of-experts layers.

**The pass is launch-bound, so it is replayed.** `torch.npu.NPUGraph` captures the whole text model
(`npu_qwen35.Graphs`, keyed by rows × padded length over a ladder up to 4,096 tokens, captured the first time a shape is
seen, 2-10 MiB each, `KEV_NPU_GRAPHS=0` to decline): 205 ms → **134 ms**, bit for bit the eager pass at the same padded
shape. Three things had to be true first:

- `chunk_local_cumsum` and `solve_tril` build their own chunk-index tables from `cu_seqlens` with
  `prepare_chunk_indices`, which reads the device tensor on the host (`.tolist()`) and uploads the result. That is two
  device syncs per DeltaNet layer, 96 in a Kev-27B pass, and a host copy cannot be captured at all. `_meta` now
  precomputes every table a helper might want (315 ms → 279 ms on its own).
- `npu_fusion_attention` refuses capture: it is the training op and draws a dropout seed on the host.
  `npu_fused_infer_attention_score` is the inference one and captures (they agree to 1e-3, bf16 rounding).
- Transformers' own `Qwen3_5TextModel.forward` builds a `[rows, 1, L, L]` float mask for every pass and decides whether
  the mixers need the padding mask with `torch.all(attention_mask == 1)`, a device read. `text_forward` replaces it:
  Kev's rows are causal and right-padded (`DecisionModel._pad_rows`), so a pad is never a key a real token reaches and
  the attention reads the 2,048 × 2,048 compressed causal mask that `sparse_mode=3` expects, at any length.

### Measured (one 910B2, `jaredpalmer/kev-27b@v1-lora`, the same 100 records, seed 0, 119 questions)

| | reference (`runs/npu-kev-27b-ref`) | fused + graphs (`runs/npu-kev-27b-fast`) |
|---|---|---|
| latency median | 422 ms | **116 ms** |
| latency mean | 454 ms | **142 ms** |
| latency p95 | 891 ms | **242 ms** |
| latency max | 930 ms | **303 ms** |
| accuracy | 0.9076 | 0.9076 |
| ECE | 0.0467 | 0.0462 |
| NLL | 0.2807 | 0.2800 |
| HBM | 52.6 GB | 52.6 GB |

The reference column is the path `runs/npu-kev-27b-sample` was measured on (adapter unmerged, eager attention, the
Ascend DeltaNet kernels), re-run on the fused code's commit. The fused column is the second pass over the sample, the
state a server runs in; the first pass, which captures a graph the first time it sees each of 13 shapes, averages
198 ms. Against the fused path the reference agrees to **max |dp| 0.017 with 0 argmax flips**, near the bar of the CUDA
fused path (`runs/fused-27b-h200`: 0.009, 0 flips). Which bucket a row is padded to moves the last bits, because the
GEMMs tile differently, so the ladder is part of the numbers: on a tiny Qwen3.5 model a replay padded to 128 tokens sits
0.4 % of the hidden states' scale from the eager pass at 50
(`tests/test_npu.py::test_fused_model_and_its_graphs_match_the_reference`).

The full-weight Kev-27B v2 (`jaredpalmer/kev-27b`) loads through the same path (`fuse` takes full weights as it takes a
merged adapter) but has not been measured on an NPU.

### Long states

Kev-27B v1, one question, a fresh process each (one-off measurements, not committed reports):

| state | fused | reference (SDPA) |
|---|---|---|
| 8.2k tokens | 2.31 s, 50.0 GiB | 3.1 s, 50.5 GiB |
| 16.4k tokens | 4.44 s, 52.2 GiB | 10.3 s, 53.0 GiB |
| 32.8k tokens | 9.40 s, 56.5 GiB | 18.5 s, 57.9 GiB |

Eager attention ran out of memory at 6,016 tokens trying to allocate the 3.24 GiB softmax; the fused attention never
builds the scores, so that ceiling is gone -- 6,516 tokens costs 1.68 s and 49.5 GiB.
32k is still the practical ceiling on a 64 GB card and it needs a fresh process: graphs captured for shorter shapes hold
their pool, and the same process ran out of memory at 32k after walking the ladder up from 200 tokens. A 64k row, which
the server accepts, sat at 65,514 of 65,536 MB and did not finish. Serve long states with `KEV_NPU_GRAPHS=0`, or keep
them under 16k.

## What is not ported

- The state-prefix cache. An NPU server recomputes each question row with its state: continuing a cached DeltaNet state
  goes through transformers' `recurrent_gated_delta_rule` (on 5.5.4), a decode kernel this module does not replace, so
  `DecisionModel` does not cache on an NPU and `kev.serve` builds its cache with size 0.
- The causal convolution stays on transformers' `F.silu(conv1d(...))` (one native op, cheap). CANN's
  `torch.ops.npu.npu_fused_causal_conv1d` and `npu_recurrent_gated_delta_rule` exist, but the recurrent op caps each
  sequence at 8 tokens (a decode kernel), so neither is used on Kev's prefill path.
- `default_device()` does not pick an NPU (pass `--device npu:0`), and `kev.train` and `kev.experiment` take cuda, mps
  and cpu only. The numbers above came from a timing harness around `Checkpoint.load` + `DecisionModel.probs`, before
  `kev.benchmark` took an NPU; `kev.benchmark` reads of the v2 checkpoints are to follow.
- Training, mixture-of-experts layers, and an fp32-on-NPU parity read.
