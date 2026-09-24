---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - qwen3.8
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
model-index:
  - name: Kev-27B
    results:
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.896 }
          - { type: brier_score, value: 0.160 }
      - task: { type: text-classification, name: typed decision, out-of-domain, fresh panel }
        dataset: { type: mixed, name: "transfer-r6 test (1,260 questions, read once)" }
        metrics:
          - { type: accuracy, value: 0.863 }
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.870 }
---

# Kev-27B

Kev-27B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter plus a pointer head on `Qwen/Qwen3.8-27B` (revision `1d4bf0f2`), serving TypeSafe's public `/v1/systemone` contract, like every Kev.

**The most accurate and best-calibrated Kev.** On the locked out-of-domain test it scores **0.896** with served Brier **0.160**, against Kev-9B's 0.852 / 0.224; coverage at ≤ 5 % error, the share of decisions that can be automated at a 5 % error budget, is 0.835 against 0.645. It keeps questions buried in long states that the smaller Kevs lose (0.833 against 0.556 on a fresh panel) and answers MMLU-Pro at 0.665 against 0.515.

**Read this first.**
- **The base is post-trained, not a base model.** Every other Kev starts from a `-Base` checkpoint. `Qwen/Qwen3.8-27B` is Qwen's instruction-tuned release; what it was post-trained on (including any distillation from other models) is Qwen's and is not known to us. Comparisons with Jev or with the smaller Kevs are therefore not controlled comparisons of the method.
- **One registered gate was overridden.** Before training, the untrained base had to reach MMLU-Pro ≥ 0.65 on `transfer-v9`; it scored 0.635 and the project owner overrode the gate (recorded in `PLAN_27b.md`, A2, at git tag `research-archive-2026-09-24`). The trained model's own MMLU-Pro is 0.665.
- **It needs a data-centre GPU.** bf16 weights are 55 GB resident (about 66 GB with the serving buffers); one B200, H200 or H100 80 GB. There is no Mac path.

- Hub: `jaredpalmer/kev-27b` (trial `r6-27b-v2/01-trial-1`; registration and every read in `PLAN_27b.md`, "B1 v2", at git tag `research-archive-2026-09-24`). Numbers below: `runs/release/kev-27b-v2.json`.

## Results (as served: each checkpoint at its own fitted temperature)

| | **Kev-27B (T = 1.38)** | Kev-9B (T = 2.30) | Jev |
|---|---|---|---|
| **locked test**, out-of-domain accuracy / Brier (transfer-v4) | **0.896 / 0.160** | 0.852 / 0.224 | – |
| locked test, coverage at ≤ 5 % error | **0.835** | 0.645 | – |
| **fresh panel**, out-of-domain accuracy (transfer-r6 test, read once) | **0.863** | 0.842 | – |
| locked test, in-distribution accuracy (decision-v7) | 0.870 | 0.874 | – |
| out-of-domain accuracy / Brier (transfer-v4 dev) | 0.848 / 0.229 | 0.822 / 0.264 | 0.857 / 0.211 |
| buried questions in long states (longstate-v3, fresh) | **0.833** | 0.556 | – |
| MMLU-Pro (transfer-v9 dev, 10-way) | 0.665 | 0.515 | 0.840 |
| unknowable items answered at ≥ 0.9 (lower is better) | 0.00 | 0.00 | 0.09 |
| held-out policy structures, both siblings correct | 0.891 | 0.828 | 0.86 |
| real documents (documents-v1 dev, CFPB complaints; never trained on) | 0.862 | 0.833 | 0.868 |
| SemIf (144 authored decisions) | 0.972 | 0.910 | – |
| scienthoon (873 support tickets) | 0.796 | 0.755 | – |
| WANLI-v2 (1,002 NLI pairs) | 0.745 | 0.740 | – |
| TypeSafe (89 answered rows) | 0.865 | 0.820 | – |

Paired against Kev-9B (record-clustered bootstrap, 95 %): transfer-r6 test +2.1 pp [+0.3, +3.8]; longstate-v3 buried questions +27.7 [+23.1, +32.5]; SemIf +6.2 [+2.8, +10.4]; scienthoon +4.1 [+1.9, +6.3]; real documents +2.9 [+0.7, +5.3]; WANLI-v2 +0.5 [−1.6, +2.6].

How it was selected: two seeds were trained under a rule registered before any training (`PLAN_27b.md`, B1 v2, at tag `research-archive-2026-09-24`): development criteria (transfer-v4 ≥ 0.842, MMLU-Pro ≥ 0.65, unknowable share ≤ 0.05, held-out pairs ≥ 0.75, long states ≥ Kev-9B + 10 pp, pooled externals ≥ Kev-9B), then one read of two fresh panels against Kev-9B, then one locked read (≥ 0.862, Brier ≤ 0.237). Seed 1 missed MMLU-Pro (0.630); seed 2 passed every step and is this checkpoint. Three earlier 27B trials (round 6) had missed the development rule by less than a point; their record is in `PLAN.md` at tag `research-archive-2026-09-24` ("Round 6").

## Serving (bf16)

Served with the adapter folded into the bf16 weights (one rounding of W + delta), fused Qwen3.5 kernels, CUDA graphs and batching across concurrent requests (`kev.serve` on CUDA). Measured with `scripts/serving_bench.py` on 200 decision-v7 development records (280 questions): `runs/fused-27b-h200`, `runs/fused-27b-h200-iso`, `runs/fused-27b-b200`, and `runs/fused-27b-h100` / `runs/fused-27b-b300` / `runs/fused-27b-rtx6000` for the other GPUs tried. The release measurement, with the adapter unmerged and no batching, is `runs/serving-27b-h200`.

| | Kev-27B | Kev-9B (H100, reference) |
|---|---|---|
| served vs the evaluation path (bf16 backbone, fp32 adapter unmerged), max / mean \|Δp\| | 0.0087 / 0.0012, 0 answer flips (B200: 0.0163, 1 flip) | – |
| isolation: question alone vs the full request, max \|Δp\| | 0.0040, 0 flips | 0.005, 0 flips |
| isolation: question alone vs next to an unrelated probe question, max \|Δp\| | 0.0038, 0 flips | 0.023, 0 flips |
| model time, new state (2 questions short / 6 questions short / 5 questions on 2,200 tokens) | H200 39.6 / 65.5 / 267.9 ms; B200 31.8 / 46.5 / 178.0 ms | – |
| model time, cached state (same requests) | H200 22.3-71.9 ms; B200 17.4-52.1 ms | – |
| requests/s, decision-v7 development records at 1 / 8 / 32 / 64 concurrent clients | H200 21.6 / 31.7 / 36.4 / 39.7; B200 27.7 / 43.9 / 51.2 / 57.3 | – |
| GPU memory resident (weights + batching buffers) / load time | 65.5 GB / 19.2 s (H200, cached weights) | – |

Under load the model is compute-bound: a B200 serves 57.3 requests/s at 64 concurrent clients for about the H200's cost per request, with lower latency. An H100 80 GB serves 35.9 requests/s at 64 clients, also at about the same cost per request (`runs/fused-27b-h100`); an RTX PRO 6000 serves it slower and at a higher cost per request (`runs/fused-27b-rtx6000`). On the H100, B200, B300 and RTX PRO 6000 one answer in 280 changed against the evaluation path, within the release tolerance below.

Isolation (a question's answer must not depend on which other questions are asked with it) is exact in fp32 arithmetic for every Kev; in bf16 it holds to the precision band above. The release tolerance was registered before this measurement: max \|Δp\| ≤ 0.03 and at most one flip in 280 questions for both comparisons.

## How it was built

- **Base model**: `Qwen/Qwen3.8-27B` (revision `1d4bf0f2`, Apache-2.0), a hybrid of Gated DeltaNet and full-attention layers like the Qwen3.5 family, so questions run as separate causal rows continuing from the shared state (`kev/model.py`), and the frozen backbone is held in bf16 (`--weights_dtype bf16`; fp32 does not fit next to the optimiser on one GPU).
- **Recipe** (one epoch, lr 5e-5, LoRA r=16 on attention, MLP and DeltaNet projections, bf16, H200): `decision-v7` plus the dates / unknowable records of the 2026-09-21 delta, 1,400 long-state records (a real question buried among 1k-6k tokens of unrelated records) and soft targets on ambiguous records with MNLI kept hard (`evals/round6/b1v2/`, 15,401 records). No Jev outputs were used for training.
- **Calibration**: `head.pt` carries temperature 1.38, fitted on the trial's in-distribution development rows (out-of-fold ECE 0.039 → 0.022); `KEV_TEMPERATURE=1.0` gives the raw logits.

## Known limits

- Knowledge is still the gap to Jev: MMLU-Pro 0.665 against 0.840.
- The long-state gain is on synthetic buried states; on real long complaint narratives it scores 0.862 (Jev 0.868) without having trained on them.
- One seed of two passed the MMLU-Pro criterion (0.665 and 0.630 on 200 questions); that criterion is at the resolution limit of its 200 questions.
- In-distribution accuracy is not higher than Kev-9B's (0.870 against 0.874 on the locked test).

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b --port 8008      # CUDA, bf16 + CUDA graphs by default; ~55 GB
```

Or deploy your own endpoint with the `kev-deploy` skill (`KEV_MODEL=jaredpalmer/kev-27b modal deploy kev_serve.py`; B200, falling back to H200 and H100). Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.8-27B base is Apache-2.0; datasets carry their own licenses.
