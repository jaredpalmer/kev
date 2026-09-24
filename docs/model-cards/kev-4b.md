---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-4B-Base
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
  - name: Kev-4B
    results:
      - task: { type: text-classification, name: typed decision, skill records, locked test }
        dataset: { type: mixed, name: "hard-v1 test (1,088 questions; programmatic labels, held-out templates; read once)" }
        metrics:
          - { type: accuracy, value: 0.803 }
          - { type: brier_score, value: 0.278 }
      - task: { type: text-classification, name: typed decision, developer tooling, locked test }
        dataset: { type: mixed, name: "devtools-v1 test (1,071 questions; six public developer-tooling sources; read once)" }
        metrics:
          - { type: accuracy, value: 0.756 }
          - { type: brier_score, value: 0.342 }
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,264 questions; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.873 }
          - { type: expected_calibration_error, value: 0.013, name: "ECE, as served" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (656 questions; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.817 }
          - { type: brier_score, value: 0.243 }
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.838 }
          - { type: brier_score, value: 0.224 }
---

# Kev-4B

Kev-4B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 33.8M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-4B-Base` (revision `1001bb4d`), serving TypeSafe's public `/v1/systemone` contract.

**This version (2026-09-24, second update): skills delta.** The round-8 Kev-4B (below) plus one epoch (lr 2e-5) on 11,320 new training records mixed with 4,000 replayed `decision-v7` records. 6,000 come from `hard-v1`, our programmatically labelled suite of the skills Kev was worst at: long policy documents with exceptions and sublimits, trade-offs under stated priorities, probability and expected value, multi-hop over several facts, dates and arithmetic, judging a proposed answer, and abstaining when a fact is missing. 5,320 come from `devtools-v1`, developer-tooling decisions from four licence-checked public datasets (CodeReviewer, CommitPackFT, FlakeFlagger, Aegis). On the held-out test splits, read once, accuracy goes from 0.540 to **0.803** on `hard-v1` (+26.3 pp [+23.3, +29.5], 1,088 questions) and from 0.623 to **0.756** on `devtools-v1` (+13.4 pp [+10.1, +16.1], 1,071 questions). On the development splits it scores 0.786 on hard-v1 against Jev's 0.777 and 0.739 on devtools-v1 against Jev's 0.713. The locked out-of-domain test is unchanged within noise (0.835 → 0.838, +0.3 pp [−1.8, +2.3]; served Brier 0.233 → 0.224). On JevBench's public items, which no training or selection step saw, the hard tier goes from 0.450 to **0.541**.

**Read this before relying on the hard-v1 and devtools-v1 numbers.**

- **Both gains are measured in distribution.** `hard-v1` is generated: every label is computed by its family's solver, and the splits hold out *templates* (0-3 train, 4 development, 5 test) of the same seven generators. A held-out template is a new wording of a skill the model was trained on, not a new skill. JevBench's public hard tier is the out-of-distribution check, and there the gain is about a third as large (+9.0 pp, below).
- **devtools-v1 labels are the public datasets' own, not adjudicated for this suite.** Some are human (CodeReviewer: whether a reviewer commented on the hunk; Aegis: human safety labels), some heuristic or by construction (CommitPackFT: the commit type is the first verb of the subject; FlakeFlagger: the test both passed and failed over reruns; When2Call: built by NVIDIA's pipeline). Before this delta, every model we scored was near chance on two of the binary sources, Jev included: on development, CodeReviewer 0.473 to 0.553 and FlakeFlagger 0.500 to 0.520 across Kev-0.8B, Kev-4B, Kev-9B, Kev-27B and Jev. This version reaches 0.633 and 0.693 on them after training on the same sources, which may be the labelling proxy being learned rather than the decision. When2Call and the prompt-injection source are never trained on: When2Call rises 0.573 → 0.660, prompt injection stays at 0.753 (Jev 0.893).
- **Two reads went down.** The locked in-distribution test moved 0.875 → 0.865, and TypeSafe's 89 answered rows moved 0.843 → 0.798. That is 4 questions, but the paired interval (−4.5 pp [−9.3, −1.0]) excludes zero. TypeSafe is too small to gate on, so the rule only counts it inside the pooled external guard. Real documents are unchanged (development 0.895 → 0.891, −0.3 pp [−1.5, +0.9]).

- Hub: `jaredpalmer/kev-4b` (this repo; trial `r10-skills/00-trial-0`; the registration and every read are in `PLAN.md` round 10 at git tag `research-archive-2026-09-24`; spec `experiments/rounds/r10.json`). The previous (round-8) version is at tag `r8-documents-release`; the `night2-du` version at `night2-du-release`; the pre-delta v7 checkpoint at `v7-base`; the Qwen3 generation at `qwen3` ([its card](kev-4b-qwen3.md)).
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev). The numbers below are in `runs/release/kev-4b-r10.json`; JevBench in `runs/jevbench-public/kev-4b-r10/`.

## Results (as served: each checkpoint at its own fitted temperature)

| | **Kev-4B (this version, T = 2.41)** | round-8 Kev-4B (T = 2.96) | Jev |
|---|---|---|---|
| **hard-v1**, test (1,088 questions, read once) | **0.803** | 0.540 | – |
| hard-v1, development (1,083) | **0.786** | 0.503 | 0.777 |
| hard-v1 ECE, test / development | 0.084 / 0.095 | 0.112 / 0.137 | – / 0.035 |
| **devtools-v1**, test (1,071, read once) | **0.756** | 0.623 | – |
| devtools-v1, development (1,072) | **0.739** | 0.605 | 0.713 |
| real documents, development (`documents-v1`, 920) | 0.891 | 0.895 | 0.868 |
| in-distribution accuracy (decision-v7 dev, 1,264 questions) | 0.873 | 0.873 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 656) | 0.817 | 0.802 | 0.857 |
| out-of-domain Brier / ECE | 0.243 / 0.042 | 0.265 / 0.043 | 0.211 / 0.049 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 0.9% | 2.9% | 3.7% |
| coverage at ≤ 5% error | 0.620 | 0.552 | 0.70 |
| held-out policy structures, both siblings correct | 0.812 | 0.781 | 0.86 |
| unknowable items answered at ≥ 0.9 (transfer-v9) | 0.00 | 0.00 | 0.09 |
| MMLU-Pro (transfer-v9 dev, 10-way) | 0.565 | 0.515 | 0.840 |
| **locked test**, out-of-domain accuracy / Brier | **0.838 / 0.224** | 0.835 / 0.233 | – |
| **locked test**, in-distribution accuracy | 0.865 | 0.875 | – |
| SemIf (144 authored decisions) | 0.889 | 0.882 | – |
| scienthoon (873 support tickets) | 0.723 | 0.723 | – |
| WANLI-v2 (1,002 NLI pairs) | 0.693 | 0.691 | – |
| TypeSafe (89 answered rows) | 0.798 | 0.843 | – |
| JevBench public items, all 231 (report only) | 0.758 | 0.714 | – |
| JevBench public, hard tier (111): accuracy / ECE | 0.541 / 0.112 | 0.450 / 0.263 | – |

Jev's devtools-v1 figure is over all 1,074 development questions. Kev's rows drop the one CodeReviewer id that the builder reused for two different records (2 questions; `PLAN.md` round-10 amendment, at tag `research-archive-2026-09-24`), because paired bootstraps need unique ids.

Paired against the round-8 version (record-clustered bootstrap, 95 %): hard-v1 development +28.3 pp [+25.1, +31.4], test +26.3 [+23.3, +29.5]; devtools-v1 development +13.3 [+11.3, +15.4], test +13.4 [+10.1, +16.1]; the two tests pooled +19.9 [+17.8, +21.8]; documents development −0.3 [−1.5, +0.9]; SemIf +0.7 [−2.8, +4.2]; scienthoon 0.0 [−1.4, +1.5]; WANLI-v2 +0.2 [−1.8, +2.2]; TypeSafe −4.5 [−9.3, −1.0]; locked out-of-domain test +0.3 [−1.8, +2.3].

**How the release was decided.** By a rule registered before the training data was built (`PLAN.md` round 10, at git tag `research-archive-2026-09-24`). The primary criterion is the pooled hard-v1 + devtools-v1 development accuracy, with a lower bound above zero; this arm scored +20.8 pp [+18.8, +22.8]. Guards on short states, documents, WANLI-v2, scienthoon, the pooled externals and unknowable confidence are each sized to what the suite can resolve (short states +1.5 pp [−0.3, +3.5], pooled externals 0.0 [−1.2, +1.1]). Hard-set ECE may be no worse than the parent's plus 0.01. Then come one read of the two untouched test splits (pooled lower bound above zero) and one locked read. Three arms were trained: both sets, hard-v1 only, and devtools-v1 only. The two single-set arms failed external guards; this arm is the only one that passed.

**JevBench public items, report only.** `runs/jevbench-public/kev-4b-r10` holds JevBench's unchanged harness run against this checkpoint, served by the `kev-deploy` template. Across all 231 public items, accuracy goes from 0.714 (round-8 version) to 0.758. On the hard tier it goes from 0.450 to 0.541: paired over the 111 hard items that is +9.0 pp [+2.7, +15.3], with 12 items newly right and 2 newly wrong (exact McNemar p = 0.013). Hard-tier ECE falls 0.263 → 0.112. No JevBench item was used for training or selection. `evals/hard-v1/overlap.json` checks all 7,400 hard-v1 records against the 231 public items (8-gram Jaccard, threshold 0.2) and finds none above the threshold. JevBench's sealed half has not been read.

**Calibration.** This delta softened the raw logits (fitted temperature 2.96 → 2.41; raw out-of-domain Brier 0.269 on development, 0.242 on the locked test), the reverse of round 8. As served, out-of-domain Brier improved from 0.265 to 0.243 on development and confident errors from 2.9% to 0.9%. `KEV_TEMPERATURE=1.0` gives the raw values.

## Previous version: round-8 real-document delta (2026-09-24) at tag `r8-documents-release`

**Real-document delta.** The `night2-du` Kev-4B plus one epoch (lr 2e-5) on `documents-v1` train: 5,219 real US consumer-finance complaint narratives (CFPB, 2015-2024, up to ~7k tokens) with 7,488 questions (which product, which main issue), labels kept only where two open-weight teachers agreed with the consumer's own filing, mixed with 2,000 replayed `decision-v7` records. On complaint narratives it has never seen, accuracy goes from 0.804 to **0.904** on the locked test (+9.9 pp [+7.5, +12.4], 936 questions) and from 0.811 to **0.891** on a private held-out set (`documents-v2`, 953 questions); on the development split it scores 0.895 against Jev's 0.868. Everything else is unchanged within noise: locked out-of-domain test 0.835 (previous 0.837), served Brier 0.233 (0.232).

**Read this before relying on the documents numbers.** The gain is measured **in distribution**: training and every documents suite share one source (CFPB complaints) and the same two question templates. It shows Kev-4B learns real long documents from a few thousand labelled examples; it does not show the same gain on other kinds of documents. Evaluation labels are AI-adjudicated (a unanimous three-model judge panel, or two agreeing adjudications) and human spot-checked (47/50 and 50/50).

- Trial `r8-small/00-trial-0` (Hub revision `957b91e7`); the registration and every read are in `PLAN.md` round 8 at git tag `research-archive-2026-09-24`. The numbers below are in `runs/release/kev-4b-r8.json`.

### Results (as served: each checkpoint at its own fitted temperature)

| | **round-8 Kev-4B (T = 2.96)** | `night2-du` Kev-4B (T = 2.14) | Jev |
|---|---|---|---|
| **real documents**, locked test (`documents-v1`, 936 questions) | **0.904** | 0.804 | – |
| real documents, private held-out (`documents-v2`, 953) | **0.891** | 0.811 | – |
| real documents, development (920) | **0.895** | 0.811 | 0.868 |
| real documents, Brier (locked test) | **0.156** | 0.286 | – |
| in-distribution accuracy (decision-v7 dev, 1,264 questions) | 0.873 | 0.872 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev) | 0.802 | 0.797 | 0.857 |
| out-of-domain Brier / ECE | 0.265 / 0.043 | 0.264 / 0.040 | 0.211 / 0.049 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 2.9% | 2.6% | 3.7% |
| coverage at ≤ 5% error | 0.552 | 0.573 | 0.70 |
| held-out policy structures, both siblings correct | 0.781 | 0.781 | 0.86 |
| unknowable items answered at ≥ 0.9 (transfer-v9) | 0.00 | 0.00 | 0.09 |
| MMLU-Pro (transfer-v9 dev, 10-way) | 0.515 | 0.490 | 0.840 |
| **locked test**, out-of-domain accuracy / Brier | **0.835 / 0.233** | 0.837 / 0.232 | – |
| **locked test**, in-distribution accuracy | 0.875 | 0.871 | – |
| SemIf (144 authored decisions) | 0.882 | 0.889 | – |
| scienthoon (873 support tickets) | 0.723 | 0.696 | – |
| WANLI-v2 (1,002 NLI pairs) | 0.691 | 0.699 | – |
| TypeSafe (89 answered rows) | 0.843 | 0.843 | – |

Paired against the `night2-du` version (record-clustered bootstrap, 95 %): documents dev +8.4 pp [+6.0, +10.6], locked test +9.9 [+7.5, +12.4], private held-out +8.0 [+5.6, +10.3]; scienthoon +2.6 [+0.8, +4.4]; SemIf −0.7 [−2.8, +1.4]; WANLI-v2 −0.8 [−2.0, +0.4]; TypeSafe identical on all 89 rows. The release was decided by a rule registered before any read (`PLAN.md` round 8, at git tag `research-archive-2026-09-24`): documents development lower bound > 0, short-state and external guards sized to what each suite can resolve, then one read of the untouched documents test and one locked read. A first seed (round 7) gave the same documents gain and failed only per-suite lower bounds on the two smallest suites; it was not released.

**Calibration.** The delta sharpened the raw logits (fitted temperature 2.14 → 2.96; raw out-of-domain Brier 0.327, raw locked Brier 0.278). As served, calibration is unchanged. `KEV_TEMPERATURE=1.0` gives the raw values.

## Earlier version: `night2-du` (2026-09-21), kept at tag `night2-du-release`

**At its release, the recommended Kev.** The best accuracy per byte: out of domain 0.797 on the development partition and **0.837 on the locked test**, Brier 0.255 on the test, held-out rule pairs 0.77–0.78. This checkpoint is the `decision-v7` recipe (trial `q35-4b-s23/00-trial-0`, seed 2, selected on development accuracy) followed by a 9-minute **delta fine-tune** (`--init_from`, lr 2e-5, one epoch) on 1,425 additional records — date-bearing policy cases rendered with explicit day counts, and evidence-free cases with uniform targets — mixed with 2,000 replayed training records. Against the pre-delta checkpoint on the locked test: +1.0 pp [−0.1, +2.1], Brier 0.266 → 0.255, `deadline` 0.65 → 0.75.


| | Kev-4B (Qwen3) | Kev-4B before the delta (`v7-base`) | **Kev-4B, raw logits** | **Kev-4B as served (T = 2.14)** | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.854 | 0.877 | 0.872 | 0.872 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.790 | 0.794 | **0.797** | 0.797 | 0.857 |
| out-of-domain Brier | 0.328 | 0.316 | 0.299 | **0.264** | 0.211 |
| out-of-domain ECE | 0.102 | 0.130 | 0.122 | **0.040** | 0.049 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 8.2% | 8.2% | 6.9% | **2.6%** | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | 0.31 | 0.54 | 0.54 | 0.57 | 0.70 |
| held-out policy structures, both siblings correct | 0.73 | 0.78 | 0.78 | 0.78 | 0.86 |
| unknowable items answered at ≥ 0.9 (lower is better; transfer-v9) | 0.44 | 0.19 | **0.00** | 0.00 | 0.09 |
| **locked test**, out-of-domain accuracy / Brier | 0.806 / 0.294 | 0.832 / 0.266 | **0.837 / 0.255** | – | – |
| **locked test**, in-distribution accuracy | 0.856 | 0.870 | 0.871 | – | – |

Per-source out-of-domain accuracy (Kev-4B / Jev): QNLI 0.91 / 0.93, SciQ 0.97 / 0.99, TweetEval-offensive 0.74 / 0.81, PAWS 0.74 / 0.79, MMLU 0.70 / 0.90, Emotion 0.56 / 0.59, deadline (3-level date arithmetic) 0.60 / 0.93 — **0.85 with the `date_facts` preprocessor** (below), (A or B) and C 0.91 / 0.91, (A and B) or not C 0.88 / 0.97, if A then not B else C 1.00 / 0.78.

**Calibration is built in.** `head.pt` carries a temperature (T = 2.14) fitted on this checkpoint's in-distribution development rows by minimising negative log-likelihood ([`scripts/calibrate_checkpoint.py`](https://github.com/jaredpalmer/kev/blob/main/scripts/calibrate_checkpoint.py)); the pointer head divides its logits by it at inference. Every loader — `kev.serve`, `kev.benchmark`, the Space, anyone's harness — gets the calibrated probabilities by default. It never changes an answer: the argmax is identical, so accuracy is the same in both columns; confidences are re-ordered only slightly across questions with different option counts, which is why coverage moves by a point or two. `KEV_TEMPERATURE=1.0` restores the raw logits; the raw column is what the training produced. Per-(type, option-count) temperatures were tested and are worse out of domain. The fit uses no out-of-domain or test data.

**`date_facts` preprocessor.** Kev, like every Kev before it, cannot subtract dates reliably (the untrained base can; LoRA training erodes it). It can use a stated day count. `KEV_DATE_FACTS=1` appends one sentence per pair of absolute dates found in the state ("June 26, 2026 is 8 days before July 4, 2026"); this checkpoint was trained on such renderings, so with it `deadline` goes from 0.60 to 0.85 and overall out-of-domain accuracy from 0.797 to 0.820. It is preprocessing, reported separately, never folded into the model's own numbers.

**What the delta cost.** MMLU-Pro fell 0.500 → 0.490 and scienthoon's ECE rose 0.086 → 0.116; coverage at ≤ 5% error was unchanged (0.54 development, 0.67 → 0.68 locked test) and confident errors fell (8.2% → 6.9%). The pre-registered criteria for the delta (`PLAN.md` at tag `research-archive-2026-09-24`, "Round 2 autoresearch") were met for dates and for the unknowable-confidence behaviour; the coverage criterion asked for +5 pp and got 0; the locked read decided promotion.

**Newer evaluation columns** (`transfer-v9` development, Kev-4B / Jev): MMLU-Pro (10-way) 0.490 / 0.840; state buried among unrelated records 0.67 / 0.70; unknowable share at ≥ 0.9 confidence 0.00 / 0.09 (intact controls 0.94).

**External suites** (same items as their published Jev numbers): SemIf's authored 144 — 0.896 before the delta (live Jev 0.965; SemIf's untrained Qwen3.5-4B 0.813); scienthoon's 900 tickets — queue 0.918, angry 0.790, ECE 0.116 (Jev 0.897, 0.914, 0.105). On ekzhang's 1,000-question MMLU-Pro sample the shipped checkpoint scores 0.468 over all 1,000 questions (8 exceed the state limit and count as wrong; live Jev 0.835 on the same items, ekzhang reports 0.829). On SemIf's pinned third-party selections (`evals/external/{wanli,typesafe}-v1`): WANLI-256 accuracy 0.695 (live Jev 0.758); TypeSafe-102 equal-case agreement / total-variation distance 0.856 / 0.231 over the 89 rows within the 8,192-token serving context (13 rejected), 0.770 / 0.308 over all 102 with rejected rows scored as wrong (live Jev 0.891 / 0.125; published TypeSafe answers 0.883 / 0.127); plain accuracy on the answered rows 0.843, coverage at <= 5% error 0.02 (Jev 0.892, 0.84). The shipped temperature is fitted in distribution and does not transfer to every workload. On WANLI, a single temperature fitted on the workload's own labelled rows (`python -m kev.calibrate`, group-disjoint out-of-fold) lowers ECE from 0.166 as shipped to 0.052 (workload T 3.91 against the shipped 2.14). Accuracy is unchanged and coverage at <= 5% error does not improve. On TypeSafe the shipped temperature already fits and refitting does not help (ECE 0.158 as shipped, 0.175 out of fold).

### How it was built

- **Base model**: Qwen3.5-4B-Base, a hybrid of 24 Gated DeltaNet (linear attention) layers and 8 full-attention layers. Because the recurrent layers cannot honour a block-causal mask, questions run as separate causal rows that continue from the shared state (`kev/model.py: forward_rows_batch`); isolation is exact by construction (together vs alone within 1e-5) and on attention-only models this form is bit-identical to the packed one.
- **Recipe**: `decision-v7`, two epochs, LoRA r=16 (attention, MLP and DeltaNet projections), lr 5e-5 — the same data and settings as every other Kev, so the Qwen3 → Qwen3.5 difference is the base (`PLAN.md` at tag `research-archive-2026-09-24`, Qwen3.5 port §10: locked test +7.3 pp [+2.8, +11.7] over Kev-8B).
- **Delta**: `kev.train --init_from jaredpalmer/kev-4b@v7-base --data evals/night2/dates_unknowable.jsonl --replay 2000 --lr 2e-5 --epochs 1`. The 1,425 new records are generated (no public dataset): 900 date-bearing policy cases, a third rendered plainly, a third with a relational day-count sentence, a third with a `date_facts` field; 255 cases with the deciding sentence removed and a uniform soft target over the options, plus their 270 intact controls. Record hashes are in `evals/night2/manifest.json`; the source checkpoint's hashes are in `training_config.json`.
- Why a delta and not a retrain: it is a controlled change (one fixed checkpoint, one data addition, 9 minutes), and the results section shows exactly what it moved.

### Known limits

- Use [Kev-9B](kev-9b.md) when accuracy and calibration matter more than memory: 0.852 vs 0.837 out of domain on the locked test, Brier 0.237 vs 0.255.

- **Slower on a Mac than on a GPU.** The DeltaNet kernels have no MPS implementation, so on Apple Silicon `kev.serve` runs this checkpoint through MLX (`kev/mlx_model.py`, installed by `uv sync --extra serve`): five questions about a ~270-token text take 721 ms on an M5, or 136 ms when the text repeats. On CUDA with `flash-linear-attention` it answers in tens of milliseconds.
- Requires `transformers >= 5.17` (the `qwen3_5` architecture) and `peft >= 0.21`.
- Knowledge (MMLU 0.70 vs Jev 0.90; MMLU-Pro 0.490 vs 0.840), TweetEval (0.74 vs 0.81) and noisy-label Emotion (0.56 vs 0.59) are the remaining gap; knowledge is set by the base (the untrained Qwen3.5-4B scores the same).
- Date arithmetic without the preprocessor: `deadline` 0.60 (Jev 0.93). With `KEV_DATE_FACTS=1`: 0.85.
- The raw logits are over-confident out of domain; the built-in temperature (T = 2.14) fixes most of it without changing any answer. `KEV_TEMPERATURE=1.0` gives the raw values. Coverage at a 5% error budget is 0.54–0.68 against Jev's 0.70.
- 4B bf16 needs ~9 GB of GPU memory for its weights and ~14 GB with the server's batching buffers; training took 56 min on one H100 (peak 24.6 GB).

### Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on `q/k/v/o_proj`, `gate/up/down_proj`, `in_proj_qkv/z/a/b`, `out_proj`; pointer head from scratch; cross-entropy on the option distribution; lr 5e-5 (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. Then the delta described above (one epoch, lr 2e-5, 3,937 records seen, 9 minutes on one H100). No Jev outputs were used for training.

### Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate (`runs/locked/kev-4b-night2-du-ungated/`; the pre-delta read is `runs/locked/kev-4b-q35/`). Every number carries suite hash, code hashes and git commit in `result.json`. Untrained-base baselines use zero-shot letter logits on the same items (`scripts/base_mmlu_probe.py`).

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008      # KEV_DTYPE=bf16 on a Mac; slow on MPS, see limits
KEV_DATE_FACTS=1 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008   # + date preprocessing; KEV_TEMPERATURE=1.0 for raw logits
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
