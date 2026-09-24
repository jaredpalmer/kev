---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-8B-Base
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
  - allenai/ai2_arc
  - allenai/openbookqa
  - tau/commonsense_qa
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Kev-8B (Qwen3)
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v4/v6 development (1,204 records; trained public sources + programmatic policy pairs)" }
        metrics:
          - { type: accuracy, value: 0.869 }
          - { type: expected_calibration_error, value: 0.061, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.796 }
          - { type: brier_score, value: 0.337 }
---

# Kev-8B (Qwen3)

> **Previous generation (Qwen3).** This checkpoint is kept as the fast option on Apple Silicon (its attention-only backbone runs the packed forward at full speed on MPS). For accuracy and calibration use [Kev-9B](kev-9b.md): on the locked test it scores 0.837 vs 0.780 out of domain against this model on the same items. Weights: `jaredpalmer/kev-8b`.

Kev-8B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16) plus a pointer head on `Qwen/Qwen3-8B-Base` (revision `49e3418f`), serving TypeSafe's public `/v1/systemone` contract.

**The most accurate Kev.** The best checkpoint of any size under a frozen, checksummed protocol: best in-distribution accuracy, best out-of-domain accuracy (0.796 on transfer-v4 dev, six points from Jev), best held-out rule reasoning of any Kev at 8B. Same recipe at two seeds: 0.796 / 0.774; this checkpoint is the seed selected on the development partition.

- Hub: `jaredpalmer/kev-8b` (this repo; trial `v7-final/00-trial-0`)
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md` (full record at git tag `research-archive-2026-09-24`), `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-0.5B (prototype) | Kev-0.6B | Kev-4B | **Kev-8B** | Jev |
|---|---|---|---|---|
| in-distribution accuracy (decision-v4 dev, 1,200 q) | 0.712 | 0.801 | 0.854 | **0.863** | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 560 q) | 0.561 | 0.620 | 0.790 | **0.796** | 0.857 |
| out-of-domain Brier | 0.50 | 0.536 | 0.328 | **0.337** | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | – | 10.8% | 8.2% | 9.9% | 3.7% |
| held-out policy structures, both siblings correct | – | 0.08 | 0.73 | 0.69 | 0.86 |
| option-order flip rate | 0.21 | 0.02 | 0.00 | 0.00 | 0.00 |

Per-source out-of-domain accuracy (Kev-8B / Jev): QNLI 0.91 / 0.93, SciQ 1.00 / 0.99, TweetEval-offensive 0.79 / 0.81, PAWS 0.78 / 0.79, MMLU 0.70 / 0.90, Emotion 0.56 / 0.59, deadline (3-level date arithmetic) 0.60 / 0.93, (A and B) or not C 0.91 / 0.97, if A then not B else C 0.59 / 0.78.

Seeds: two seeds on decision-v7: transfer **0.796** / 0.774, held-out rule pairs 0.69 / 0.64 (Jev 0.86); this checkpoint is seed 0. Trained on `decision-v7` (10k public records + 896 policy records over nine template families incl. four ordinal Score threshold families + 1,680 records from 60 random rule structures with negation anywhere); development/test items are byte-identical to v4, so every number here is comparable with earlier checkpoints.

**Locked test, read once** (`runs/locked/kev-8b-v7-preview-ungated/`): in-distribution **0.870** (Brier 0.193), out-of-domain **0.780** (Brier 0.327, confident errors 7.6%, held-out pairs 0.62). This partition will not be read again for this checkpoint.

## What we learned building it

- **Capacity dominates out of domain.** With public examples and synthetic budget held equal, 0.6B → 4B is +14–19 pp; 4B → 8B is +1–7 pp.
- **Fine-tuning erodes base capability, and the learning rate controls it.** The 4B base, zero-shot with a letter readout, scores 0.688 on the same MMLU items and 0.787 on PAWS; the default recipe (lr 2e-4) trained down to 0.60–0.66 / 0.56–0.71. Lowering lr to 5e-5 recovers most of it and is the single largest recipe improvement we found; fewer LoRA target modules and smaller ranks help less.
- **More public training data raises in-distribution accuracy and lowers transfer** at 4B (10k vs 3.4k records: −3 pp). Knowledge MCQ sources (ARC, OpenBookQA, CommonsenseQA) raise in-distribution accuracy to 0.86 without moving transfer.
- Programmatic contrastive policy pairs teach the trained rule structures (both-correct 0.85–1.0) but transfer to unseen structures only partially (0.5–0.6 at 4B, 0.03–0.11 at 0.6B).

## Known limits

- Held-out policy reasoning (unseen rule compositions, date arithmetic with grace periods) is far from Jev.
- Product-shaped questions with no training analogue are not guaranteed; measure on your own inputs.
- Out-of-domain probabilities are usable but not calibrated (raw ECE 0.128); temperature fitted in-domain does not transfer.
- 8B fp32 needs ~33 GB and does not fit a 32 GB Mac; `KEV_DTYPE=bf16` (~17 GB) does. Training took ~70 min on one H100.

## Training

Frozen suite `evals/v6/decision-v6` (development/test bytes identical to v4): 13,000 public records (1,000 per source: the ten v4 sources plus ARC-Challenge, OpenBookQA, CommonsenseQA) plus two programmatic policy arms of 448 records, two epochs, LoRA r=16 on attention and MLP projections, pointer head from scratch, cross-entropy on the option distribution, **lr 5e-5** (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing, one H100 (~70 min). Augmentation: option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate. Every number carries suite hash, code hashes, and git commit in `result.json`. See `PLAN.md` at git tag `research-archive-2026-09-24` ("Evidence and corrections") for the corrections we made to our own earlier claims.

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-8b --port 8008      # KEV_DTYPE=bf16 on a 32 GB Mac
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; Qwen3 base is Apache-2.0; datasets carry their own licenses.
