---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-4B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - decision-model
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
  - name: Kev-4B (Qwen3)
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v4 development (1,204 records; ten trained public sources + programmatic policy pairs)" }
        metrics:
          - { type: accuracy, value: 0.854 }
          - { type: expected_calibration_error, value: 0.065, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.790 }
          - { type: brier_score, value: 0.328 }
---

# Kev-4B (Qwen3)

> **Previous generation (Qwen3).** This checkpoint is kept as the fast option on Apple Silicon (its attention-only backbone runs the packed forward at full speed on MPS). For accuracy and calibration use [Kev-4B (Qwen3.5)](kev-4b.md): on the locked test it scores 0.832 vs 0.806 out of domain against this model on the same items. Weights: `jaredpalmer/kev-4b@qwen3`.

Kev-4B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16) plus a pointer head on `Qwen/Qwen3-4B-Base`, serving TypeSafe's public `/v1/systemone` contract.

**The recommended Kev.** The best 4B checkpoint under a frozen, checksummed protocol after ~40 controlled 4B trials, and the first Kev within seven points of Jev out of domain on the same items. Same recipe run at three seeds: transfer 0.773 / **0.790** / 0.770; this checkpoint is the seed selected on the development partition (never on the locked test).

- Hub: `jaredpalmer/kev-4b`, revision tag `qwen3` (trial `v7-rc3/01-trial-1`); the repo's main revision now holds the Qwen3.5 checkpoint
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md` (full record at git tag `research-archive-2026-09-24`), `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-0.5B (prototype) | Kev-0.6B | **Kev-4B** | Jev |
|---|---|---|---|---|
| in-distribution accuracy (decision-v4 dev, 1,200 q) | 0.712 | 0.801 | **0.854** | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 560 q) | 0.561 | 0.620 | **0.790** | 0.857 |
| out-of-domain Brier | 0.50 | 0.536 | **0.328** | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | – | 10.8% | 8.2% | 3.7% |
| held-out policy structures, both siblings correct | – | 0.08 | 0.73 | 0.86 |
| option-order flip rate | 0.21 | 0.07 | 0.06 | 0.00 |

Per-source out-of-domain accuracy (Kev-4B / Jev): QNLI 0.89 / 0.93, SciQ 0.99 / 0.99, TweetEval-offensive 0.75 / 0.81, PAWS 0.72 / 0.79, MMLU 0.65 / 0.90, Emotion 0.66 / 0.59, deadline (3-level date arithmetic) 0.53 / 0.93, (A and B) or not C 0.97 / 0.97, if A then not B else C 0.88 / 0.78.

Seeds: three seeds on decision-v7: transfer 0.773 / **0.790** / 0.770, held-out rule pairs 0.62 / **0.73** / 0.67 (Jev 0.86); this checkpoint is seed 1, selected on development transfer accuracy. Trained on `decision-v7` (10k public records + 896 policy records over nine template families incl. four ordinal Score threshold families + 1,680 records from 60 random rule structures with negation anywhere); development/test items are byte-identical to v4, so every number here is comparable with earlier checkpoints.

**Locked test, read once** (`runs/locked/kev-4b-v7-preview-ungated/`): in-distribution **0.856** (Brier 0.211), out-of-domain **0.806** (Brier 0.294, confident errors 6.6%, held-out pairs 0.66). This partition will not be read again for this checkpoint.

## What we learned building it

- **Capacity dominates out of domain.** With public examples and synthetic budget held equal, 0.6B → 4B is +14–19 pp; 4B → 8B is +1–7 pp.
- **Fine-tuning erodes base capability, and the learning rate controls it.** The 4B base, zero-shot with a letter readout, scores 0.688 on the same MMLU items and 0.787 on PAWS; the default recipe (lr 2e-4) trained down to 0.60–0.66 / 0.56–0.71. Lowering lr to 5e-5 recovers most of it and is the single largest recipe improvement we found; fewer LoRA target modules and smaller ranks help less.
- **More public training data raises in-distribution accuracy and lowers transfer** at 4B (10k vs 3.4k records: −3 pp). Knowledge MCQ sources (ARC, OpenBookQA, CommonsenseQA) raise in-distribution accuracy to 0.86 without moving transfer.
- Programmatic contrastive policy pairs teach the trained rule structures (both-correct 0.85–1.0) but transfer to unseen structures only partially (0.5–0.6 at 4B, 0.03–0.11 at 0.6B).

## Known limits

- Held-out policy reasoning (unseen rule compositions, date arithmetic with grace periods) is far from Jev.
- Product-shaped questions with no training analogue are not guaranteed: on the TypeSafe docs example ("two charges on my card" → *Is there a billing problem?*) this checkpoint answers 0.48 (Kev-8B 0.95, Kev-0.6B 0.97) while picking the return reason correctly (wrong size 0.53; Kev-8B 0.84; Kev-0.6B prefers "none of the above" 0.58). Measure on your own inputs.
- Out-of-domain probabilities are usable but not calibrated (raw ECE 0.096); temperature fitted in-domain does not transfer.
- 4B fp32 needs ~16 GB; on a 32 GB Mac use `KEV_DTYPE=bf16`. Latency on an H100 is ~45 ms per packed request; on an M5 several hundred ms.

## Training

Frozen suite `evals/v4/decision-v4`: 10,000 public records (1,000 per source, ten sources) plus two programmatic policy arms of 448 records, two epochs, LoRA r=16 on attention and MLP projections, pointer head from scratch, cross-entropy on the option distribution, **lr 5e-5** (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing, one H100 (~40 min). Augmentation: option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate. Every number carries suite hash, code hashes, and git commit in `result.json`. See `PLAN.md` at git tag `research-archive-2026-09-24` ("Evidence and corrections") for the corrections we made to our own earlier claims.

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008      # KEV_DTYPE=bf16 on a 32 GB Mac
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; Qwen3 base is Apache-2.0; datasets carry their own licenses.
