# System submission: Kev (Qwen3.5-4B / Qwen3.5-9B), self-hostable from Hub

Hi Florian — I'd like to submit the current Kev generation to JevBench. No public endpoint on my side, so I'm asking whether you can run it on your own hardware from a public repo + Hub weights, the way you ran `kev 4B` / `kev 8B` (commit `20fa626`) in v1.2.

## System

- **Name:** Kev (research preview)
- **Setup:** Qwen3.5-4B-Base / Qwen3.5-9B-Base + LoRA + a pointer readout head, prefill-only (no generation), block-causal over shared state with isolated question branches
- **Weights:** https://huggingface.co/jaredpalmer/kev-4b and https://huggingface.co/jaredpalmer/kev-9b (adapter + `head.pt` + pinned base revision)
- **Base:** `Qwen/Qwen3.5-4B-Base` @ `1001bb4d826a52d1f399e183466143f4da7b741b`, `Qwen/Qwen3.5-9B-Base` @ `68c46c4b3498877f3ef123c856ecfde50c39f404`
- **Repro source:** https://github.com/mobailabs/kev at commit `1bcd4b7` (fork of `jaredpalmer/kev`; the native MLX / inference-worker work is under PR jaredpalmer/kev#11)

## Reproduce

```
git clone https://github.com/mobailabs/kev && cd kev
git checkout 1bcd4b7
uv sync --extra serve
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-9b --port 8009
```

The server exposes a native TypeSafe-compatible `POST /v1/systemone`, so the existing `typesafe` adapter works unchanged (`--key-env ''`, no auth). Needs `transformers>=5.17`, `peft>=0.21`, `torch`. On CUDA the Qwen3.5 DeltaNet layers want `flash-linear-attention`; the generic PyTorch path also works (slower). On Apple Silicon it uses MLX by default.

Notes on config that you may want to name in the row:
- `KEV_DTYPE=bf16` for a GPU run (the published eval used bf16).
- Serving accepts up to 16,384 tokens for state and for state+one branch. Training used at most 384 state tokens, so long states are out of training distribution but are not truncated by the server.
- `KEV_TEMPERATURE=2.0` is the temperature fitted in-distribution for the Qwen3.5 family; default is 1.0 (uncalibrated). I'd report whichever you actually run.
- `KEV_DATE_FACTS=1` is an opt-in preprocessor that appends day-count sentences for absolute dates in the state. My hard-tier run above used it; without it temporal_numeric was 0.133 rather than 0.200. Say which you'd like measured.

## My own runs (public items, for reference only — please re-run)

I ran the public JevBench items through the local server with the official harness (`--adapter typesafe`, one request at a time). Raw `results.jsonl` / `manifest.json` / `summary.json` available on request.

| Split | n | acc |
|---|---:|---:|
| easy | 48 | 1.000 |
| original | 72 | 0.903 |
| hard (public) | 111 | 0.568 |

hard by family: adversarial 1.000, routing_hard 1.000, trap 0.875, multi_hop 0.778, ambiguous 0.571, judge_hard 0.529, tradeoff 0.500, probability 0.400, long_policy 0.368, temporal_numeric 0.200.

These were measured on an Apple M5 Max over MLX, so treat the latency as non-production and apply your usual adjustment (or ignore it — the point of the submission is Intelligence/Calibration, not my laptop's speed).

## Questions

1. Can you take this as a self-hosted submission run on your hardware, or would you rather I hand over raw results only?
2. There are already `kev 0.5B/0.6B/4B/8B` rows from commit `20fa626` (previous generation, Qwen2.5/Qwen3). Should the new Qwen3.5 checkpoints **replace** those rows or sit **alongside** them? They are different model generations.
3. Do you want me to open a PR adding a Kev adapter, or is the stock `typesafe` adapter enough?

Thanks — happy to answer anything or provide the raw artifacts.
