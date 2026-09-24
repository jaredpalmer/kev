---
language: en
license: apache-2.0
library_name: peft
base_model: jaredpalmer/qwen3.5-9b-base-awq
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - qwen3.5
  - awq
  - quantized
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
---

# Kev-9B-AWQ

<!-- TODO before publishing: this card is a template. Every number below is a placeholder pending a real
     scripts/quantize_awq.py run, a scripts/calibrate_checkpoint.py refit, and kev.benchmark / scripts/serving_bench.py
     reads on the quantized checkpoint (see PLAN.md's ship-gate discipline: no number ships without a report path). -->

Kev-9B-AWQ is [Kev-9B](https://huggingface.co/jaredpalmer/kev-9b) — the same LoRA adapter (r=16, 45.4M trainable
parameters) and pointer head, unchanged — served on an AWQ int4 quantization of its base, `Qwen/Qwen3.5-9B-Base`,
instead of bf16. The adapter is **not** folded into the quantized weights (merging a LoRA delta into packed int4
tensors isn't supported); it runs unmerged on top of the frozen quantized base at inference
(`kev.checkpoint.Meta.quantization`, which forces `LoadOptions.merge = False`).

Why this checkpoint exists: Kev-9B needs ~19 GB of GPU memory in bf16 (H100/H200-class hardware only,
`skills/kev-deploy/scripts/kev_serve.py: GPU_FOR`). AWQ int4 quantization of the base cuts that by roughly 3-4x,
which should put a cheap GPU tier (L4/L40S) back in reach for the largest Kev — the same tradeoff Kev-4B and
Kev-0.8B already serve well on.

- Hub: `jaredpalmer/kev-9b-awq` (TODO: not yet published). Base: `jaredpalmer/qwen3.5-9b-base-awq` (TODO: not yet
  published; produced by `scripts/quantize_awq.py`).
- Adapter and head are byte-identical to `jaredpalmer/kev-9b`; only `head.pt`'s `base` field and `temperature`
  differ (a refit is required — quantization noise can shift calibration even when it does not change the argmax).

## Results (TODO)

Quantizing a base after training (post-training quantization) usually costs a little out-of-domain accuracy and can
shift calibration; it must be measured, not assumed. Before publishing, fill in a table matching
`docs/model-cards/kev-9b.md`'s "Results" section, comparing this checkpoint against `jaredpalmer/kev-9b` on the same
frozen items (`evals/v4/transfer-v4` development, the locked test read once): in-distribution accuracy,
out-of-domain accuracy, Brier, ECE, confident-error rate, coverage at <=5% error, and the locked-test read. Use
`kev.benchmark --run jaredpalmer/kev-9b-awq --suite evals/v4/transfer-v4 --out runs/kev-9b-awq-transfer-v4` and
`kev.compare` against the existing `jaredpalmer/kev-9b` result dir for a paired read.

## How it was built (TODO once run)

- **Base**: `Qwen/Qwen3.5-9B-Base` quantized to AWQ int4 (`scripts/quantize_awq.py`, group size TBD, calibrated on a
  sample of `evals/v7/decision-v7`'s train partition — record the exact `--calib_n` and seed used).
- **Adapter and head**: unchanged from `jaredpalmer/kev-9b` (`decision-v7` recipe + the dates/unknowable delta;
  see `docs/model-cards/kev-9b.md` for the full recipe).
- **Temperature**: refit against this base with `scripts/calibrate_checkpoint.py` (do not reuse kev-9b's T=2.30
  as-is without checking it).

## Known limits

- Requires a CUDA GPU with AutoAWQ's kernels (`pip install autoawq` / `uv sync --extra awq`); no MPS or CPU path,
  unlike bf16 Kev-9B.
- `transformers >= 5.17` (the `qwen3_5` architecture), `peft >= 0.21`, `autoawq >= 0.2.9`.
- Same knowledge/date-arithmetic limits as `jaredpalmer/kev-9b` (set by the base and the adapter, not by
  quantization); see that card. This card's own accuracy/calibration table (above) is the quantization-specific
  read still needed.

## Use

```bash
uv run --extra serve --extra awq python -m kev.serve --run jaredpalmer/kev-9b-awq --port 8008
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
