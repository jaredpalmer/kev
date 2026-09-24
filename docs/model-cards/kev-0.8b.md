---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - qwen3.5
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
  - name: Kev-0.8B
    results:
      - task: { type: text-classification, name: typed decision, real documents, locked test }
        dataset: { type: mixed, name: "documents-v1 test (936 questions on CFPB complaint narratives; read once)" }
        metrics:
          - { type: accuracy, value: 0.821 }
          - { type: brier_score, value: 0.256 }
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,264 questions; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.833 }
          - { type: expected_calibration_error, value: 0.028, name: "ECE, as served" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (656 questions; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.657 }
          - { type: brier_score, value: 0.423 }
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.695 }
          - { type: brier_score, value: 0.396 }
---

# Kev-0.8B

Kev-0.8B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 11.3M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2`), serving TypeSafe's public `/v1/systemone` contract.

**This version (2026-09-24): real-document delta.** The `night2-du` Kev-0.8B (below) plus one epoch (lr 2e-5, seed 5) on `documents-v1` train, the set Kev-4B's round-8 delta used: 5,219 real US consumer-finance complaint narratives (CFPB, up to ~7k tokens) with 7,488 questions (which product, which main issue), mixed with 6,000 replayed `decision-v7` records. On complaint narratives it has never seen, accuracy goes from 0.608 to **0.821** on the locked test (+21.3 pp [+17.9, +24.5], 936 questions; Brier 0.528 → 0.256) and from 0.616 to **0.856** on a private held-out set (`documents-v2`, 953 questions; +24.0 pp [+21.0, +27.1]). On the development split it goes from 0.633 to 0.837, still below Jev's 0.868 (−3.2 pp [−5.5, −0.8]). The locked out-of-domain test moved 0.684 → 0.695 (+1.1 pp [−1.4, +3.4]) with served Brier 0.412 → 0.396.

**Read this before relying on the documents numbers.** The gain is measured **in distribution**, as on the [Kev-4B card](kev-4b.md): training and every documents suite share one source (CFPB complaints) and the same two question templates. It shows that a 0.8B Kev learns real long documents from a few thousand labelled examples. It does not show the same gain on other kinds of documents. Evaluation labels are AI-adjudicated (a unanimous three-model judge panel, or two agreeing adjudications) and human spot-checked (47/50 and 50/50). Out of domain it is still a sub-1B model (0.657 on transfer-v4 development; Jev 0.857).

**Why it took four rounds.** Rounds 7, 8 and 9 trained four 0.8B documents deltas. Each gained between 21 and 23 pp on documents, and each failed the short-state guard, among other guards, on the 656-question transfer-v4 development panel. That panel cannot tell a cost of about 1 pp from one of 2 pp. Before any training, round 11 registered a fresh-seed replication of round 9's recipe (replay 6,000, lr 2e-5). It judged both seeds on a pooled short-state panel: the 656 transfer-v4 development questions plus 1,150 transfer-r3 test questions. Both seeds passed every guard, with short-state accuracy −0.6 pp [−1.8, +0.6] for seed 4 and +0.4 [−0.9, +1.7] for seed 5 Neither seed shows a short-state cost that the pooled panel can detect. This checkpoint is seed 5, the larger documents estimate (+20.4 pp [+17.3, +23.5] against +20.2). The rule then required one read of the documents-v1 test (lower bound above zero) and one locked read (accuracy ≥ parent − 1 pp, served Brier ≤ parent + 0.005); both passed. The in-trial screening gate "held-out pairs ≥ 70 %" fails at this size, as it did for the released parent (0.484 vs 0.422), which is why the locked read is named `kev-08b-r11-ungated`.

- Hub: `jaredpalmer/kev-0.8b` (this repo; trial `r11-docs/03-trial-3`; the registration and every read are in `PLAN.md` rounds 9 and 11 on the `research/overnight-r6` branch). The previous version will be at tag `night2-du-release` (not yet created); the pre-delta v7 checkpoint at `v7-base`.
- Demo: [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev) runs Kev-4B and Kev-0.8B on ZeroGPU with the same encoder and API code as `kev.serve`.
- Code, suites, results, and the full research log: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md`, `runs/leaderboard.md`. The numbers below are in `runs/release/kev-08b-r11.json`.

## Results (as served: each checkpoint at its own fitted temperature)

| | **Kev-0.8B (this version, T = 2.52)** | `night2-du` Kev-0.8B (T = 2.41) | Jev |
|---|---|---|---|
| **real documents**, locked test (`documents-v1`, 936 questions) | **0.821** | 0.608 | – |
| real documents, private held-out (`documents-v2`, 953) | **0.856** | 0.616 | – |
| real documents, development (920) | **0.837** | 0.633 | 0.868 |
| real documents, Brier (locked test) | **0.256** | 0.528 | – |
| in-distribution accuracy (decision-v7 dev, 1,264 questions) | 0.833 | 0.825 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 656) | 0.657 | 0.652 | 0.857 |
| out-of-domain Brier / ECE | 0.423 / 0.039 | 0.430 / 0.054 | 0.211 / 0.049 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 0.0% | 0.3% | 3.7% |
| coverage at ≤ 5% error | 0.218 | 0.229 | 0.70 |
| held-out policy structures, both siblings correct | 0.484 | 0.422 | 0.86 |
| unknowable items answered at ≥ 0.9 (transfer-v9) | 0.00 | 0.00 | 0.09 |
| MMLU-Pro (transfer-v9 dev, 10-way) | 0.200 | 0.185 | 0.840 |
| **locked test**, out-of-domain accuracy / Brier | **0.695 / 0.396** | 0.684 / 0.412 | – |
| **locked test**, in-distribution accuracy | 0.832 | 0.834 | – |
| SemIf (144 authored decisions) | 0.722 | 0.701 | – |
| scienthoon (873 support tickets) | 0.535 | 0.520 | – |
| WANLI-v2 (1,002 NLI pairs) | 0.568 | 0.570 | – |
| TypeSafe (89 answered rows) | 0.618 | 0.629 | – |

Paired against the `night2-du` version (record-clustered bootstrap, 95 %): documents development +20.4 pp [+17.3, +23.5], locked test +21.3 [+17.9, +24.5], private held-out +24.0 [+21.0, +27.1]; SemIf +2.1 [−2.1, +6.2]; scienthoon +1.5 [−0.7, +3.6]; WANLI-v2 −0.2 [−1.2, +0.8]; TypeSafe −1.1 [−9.2, +7.1]; locked out-of-domain test +1.1 [−1.4, +3.4], served Brier −0.016 [−0.028, −0.005].

**Calibration.** The delta sharpened the raw logits (fitted temperature 2.41 → 2.52; raw out-of-domain Brier 0.489 on development, 0.441 on the locked test). As served, calibration is unchanged or slightly better. `KEV_TEMPERATURE=1.0` gives the raw values.

## Previous version: `night2-du` (2026-09-21), to be kept at tag `night2-du-release`

**The small member of the Kev family.** Same data and recipe as the 0.6B it replaces, on the Qwen3.5 base: in-distribution 0.825 (Kev-0.6B 0.801), out of domain 0.652 (0.620), and it is the first small Kev that learns any rule composition (held-out pairs 0.42 vs 0.08). Three seeds of the base recipe: transfer 0.622 / 0.634 / **0.643**; this checkpoint is seed 2 (selected on development accuracy) followed by a 9-minute **delta fine-tune** on 1,425 generated records (date-bearing policy cases with explicit day counts; evidence-free cases with uniform targets) mixed with 2,000 replayed training records — the same delta as Kev-4B and Kev-9B. Locked test against the pre-delta checkpoint: out of domain 0.668 → **0.684** (+2.2 pp [−0.8, +5.5]), Brier 0.473 → 0.460. Out of domain it is still a sub-1B model: use Kev-4B for accuracy; use this one where memory rules the 4B out, and measure on your own data.

- Trial `night2-08b-du2/00-trial-0`. The pre-delta checkpoint is at revision `v7-base`.

### Results (same frozen items for every row)

| | Kev-0.6B (Qwen3) | **Kev-0.8B** | Kev-4B | Kev-9B | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.801 | **0.825** | 0.872 | 0.872 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.620 | **0.652** | 0.797 | 0.822 | 0.857 |
| out-of-domain Brier | 0.536 | **0.499** | 0.299 | 0.286 | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 10.8% | 9.9% | 6.9% | 8.7% | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | – | 0.23 | 0.54 | 0.47 | 0.70 |
| held-out policy structures, both siblings correct | 0.08 | **0.42** | 0.78 | 0.83 | 0.86 |
| option-order flip rate | 0.07 | 0.08 | 0.08 | 0.03 | 0.00 |
| none-option present, accuracy | 0.80 | 0.83 | 0.92 | 0.90 | – |
| as served (built-in T = 2.41): Brier / ECE / confident errors | – | 0.430 / 0.054 / 0.3% | | | |

Per-source out-of-domain accuracy (Kev-0.8B / Jev): QNLI 0.85 / 0.93, SciQ 0.91 / 0.99, TweetEval-offensive 0.68 / 0.81, PAWS 0.55 / 0.79, MMLU 0.42 / 0.90, Emotion 0.54 / 0.59, authorization 0.97 / 1.00, deadline (3-level date arithmetic) 0.38 / 0.93, (A or B) and C 0.66 / 0.91, (A and B) or not C 0.56 / 0.97, if A then not B else C 0.59 / 0.78.

Paired against Kev-0.6B on the same items (record-clustered bootstrap), before the delta: +5.7 pp [+1.2, +10.0] out of domain; the delta adds +0.5 pp [−3.2, +3.8] on development and +2.2 pp on the locked test.

**Locked test, read once per checkpoint** (`runs/locked/kev-08b-night2-du-ungated/`; pre-delta `runs/locked/kev-08b-q35-ungated/`): in-distribution **0.834** (Brier 0.268, ECE 0.100), out-of-domain **0.684** (Brier 0.460, ECE 0.154, confident errors 8.7%, held-out pairs 0.45). Pre-delta: 0.827 / 0.668; Kev-0.6B on the same test items: 0.808 / 0.642.

### Known limits

- **Out of domain it is a sub-1B model.** Knowledge (MMLU 0.41) and paraphrase (PAWS 0.59) are near the untrained base; the same recipe reaches 0.79 at 4B and 0.81 at 9B on these items.
- **Slow on a Mac for its size.** The DeltaNet kernels have no MPS implementation; a five-question request takes ~0.33 s in bf16 on an M5 (Kev-0.6B: 0.12 s). On CUDA with `flash-linear-attention` it is fast.
- Requires `transformers >= 5.17` and `peft >= 0.21`.
- Ordinal hedging on date arithmetic (`deadline` 0.38): collapses to the middle level. `KEV_DATE_FACTS=1` (day counts appended to the state) helps the larger models more than this one.
- Confident-error rate out of domain is 9.9% for the raw logits; the built-in temperature (T = 2.41, fitted on the in-distribution development rows and stored in `head.pt`) brings it to 0.3% and ECE from 0.179 to 0.054 without changing any answer. `KEV_TEMPERATURE=1.0` gives the raw values. Probabilities are usable in-domain; treat them as advisory elsewhere.

### Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on attention, MLP and DeltaNet projections; pointer head from scratch; cross-entropy on the option distribution; lr 1e-4 (OneCycle), batch 8, bf16 autocast with fp32 master weights; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records; ~20 min on one H100. Then the delta: `--init_from jaredpalmer/kev-0.8b@v7-base --data evals/night2/dates_unknowable.jsonl --replay 2000 --lr 4e-5 --epochs 1`, 9 minutes. No Jev outputs were used for training.

### Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate. Every number carries suite hash, code hashes and git commit in `result.json`.

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
