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
          - { type: accuracy, value: 0.851 }
          - { type: brier_score, value: 0.244 }
      - task: { type: text-classification, name: typed decision, skill records, locked test }
        dataset: { type: mixed, name: "hard-v1 test (1,088 questions; programmatic labels, held-out templates; read once)" }
        metrics:
          - { type: accuracy, value: 0.665 }
          - { type: brier_score, value: 0.460 }
      - task: { type: text-classification, name: typed decision, developer tooling, locked test }
        dataset: { type: mixed, name: "devtools-v1 test (1,071 questions; six public developer-tooling sources; read once)" }
        metrics:
          - { type: accuracy, value: 0.637 }
          - { type: brier_score, value: 0.442 }
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,264 questions; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.827 }
          - { type: expected_calibration_error, value: 0.033, name: "ECE, as served" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (656 questions; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.648 }
          - { type: brier_score, value: 0.430 }
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.697 }
          - { type: brier_score, value: 0.397 }
---

# Kev-0.8B

Kev-0.8B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 11.3M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2`), serving TypeSafe's public `/v1/systemone` contract.

**This version (2026-09-24): documents and skills in one delta.** The `night2-du` Kev-0.8B (below) plus one epoch (lr 2e-5, seed 1) on 16,539 records trained together, mixed with 6,000 replayed `decision-v7` records. They combine three sets. `documents-v1` train holds 5,219 real US consumer-finance complaint narratives (CFPB, up to ~7k tokens) with 7,488 questions (which product, which main issue), the set Kev-4B's round-8 delta used. `hard-v1` train holds 6,000 programmatically labelled records in seven skill families: long policy documents, trade-offs, probability, multi-hop, dates and arithmetic, judging a proposed answer, and missing-fact abstention. `devtools-v1` train holds 5,320 developer-tooling decisions from four public datasets. Every read below was taken once, on held-out splits:

- **Real documents.** The locked test goes from 0.608 to **0.851** (+24.4 pp [+21.3, +27.6], 936 questions; Brier 0.528 → 0.244). A private held-out set (`documents-v2`, 953 questions) goes from 0.616 to **0.848**.
- **Skills.** The `hard-v1` test goes from 0.396 to **0.665** (+26.9 pp [+23.4, +30.4], 1,088 questions). The `devtools-v1` test goes from 0.472 to **0.637** (+16.4 pp [+13.1, +19.4], 1,071 questions).
- **Everything else.** The locked out-of-domain test moved 0.684 → 0.697 (+1.2 pp [−1.1, +3.7]), with served Brier 0.412 → 0.397. On JevBench's public items, which no training or selection step saw, accuracy over all items goes from 0.597 to 0.636. The hard tier moves only from 0.333 to 0.360, and that change is within noise.

It is still a sub-1B model. On the development splits it trails Jev everywhere it can be compared: documents 0.842 vs 0.868, hard-v1 0.594 vs 0.777, devtools-v1 0.602 vs 0.713, out of domain 0.648 vs 0.857.

**Read this before relying on these numbers.**

- **All three gains are measured in distribution.** Training and every documents suite share one source (CFPB complaints) and the same two question templates. Evaluation labels are AI-adjudicated (a unanimous three-model judge panel, or two agreeing adjudications) and human spot-checked (47/50 and 50/50). `hard-v1` is generated: its labels are computed by each family's solver, and the test split holds out *templates* (0-3 train, 4 development, 5 test) of the same seven generators, so a test item is a new wording of a trained skill. JevBench's public hard tier is the out-of-distribution check, and there the paired gain is +2.7 pp [−1.8, +7.2]: not distinguishable from zero.
- **devtools-v1 labels are the public datasets' own, not adjudicated for this suite.** Some are human (CodeReviewer: whether a reviewer commented on the hunk; Aegis: human safety labels), some heuristic or by construction (CommitPackFT: the commit type is the first verb of the subject; FlakeFlagger: the test both passed and failed over reruns; When2Call: built by NVIDIA's pipeline). Before training, every model was near chance on CodeReviewer and FlakeFlagger, Jev included (see the [Kev-4B card](kev-4b.md)). After training on the same sources, this version gets 0.500 → 0.667 on CodeReviewer development. On FlakeFlagger it moves 0.500 → 0.520 on development but 0.507 → 0.813 on test. The two splits disagree that widely on a 150-question source, so treat the FlakeFlagger number as unexplained, not as a skill.
- **One eval-only source got worse.** When2Call, which is never trained on and asks whether to call a tool, ask for a missing parameter or decline, fell from 0.260 to 0.167 on development and from 0.233 to 0.133 on test, below the one-in-four rate of guessing among its four options. Prompt injection, also eval-only, is flat (0.527 → 0.547; Jev 0.893). Do not use this checkpoint for tool-call routing.
- **Other reads that went down.** Out-of-domain development accuracy moved 0.652 → 0.648, and coverage at ≤ 5 % error 0.229 → 0.145. TypeSafe's 89 answered rows moved 0.629 → 0.596 (−3.4 pp [−13.5, +5.0], 3 questions).

**Why it took until round 15.** Six registered rounds of 0.8B deltas came before this one.

- **Rounds 7, 8 and 9: documents alone.** They trained four 0.8B documents deltas. Each gained +20.8 to +22.7 pp on documents, and each failed the short-state guard, among other guards, on the 656-question transfer-v4 development panel. That panel cannot tell a cost of about 1 pp from one of 2 pp.
- **Round 11: a bigger panel.** It judged fresh seeds on a pooled short-state panel: the 656 transfer-v4 development questions plus 1,150 transfer-r3 test questions. The documents delta passed there (round 11). Separately, a skills-only delta passed (round 12).
- **Round 13: stacking failed.** It trained the skills data on top of the round-11 documents checkpoint, the path Kev-4B took. The second delta eroded the first: seed 1's documents lower bound was −2.03 pp against a −2 pp floor, and seed 2's short-state accuracy was −1.6 pp [−3.2, 0.0].
- **Round 15: joint training passed.** Before any training it registered one epoch on all three sets together, from the released checkpoint. Both primaries had to hold (documents development, and hard-v1 + devtools-v1 development pooled), with round 12's guards on the pooled short-state panel.
  - This checkpoint, arm (a), scored documents +21.0 pp [+18.0, +24.0] and skills +18.0 [+15.8, +20.0], with short state −0.1 [−1.4, +1.3] and pooled externals +2.1 [+0.6, +3.5].
  - The second seed at the same settings, arm (c), also passed (documents +20.2 [+16.9, +23.5], skills +16.7 [+14.5, +18.9], short state −0.3 [−1.7, +1.1]).
  - A larger step, arm (b) at lr 4e-5, failed the short-state and scienthoon guards.
  - Confirmation then required the documents-v1 test and the pooled hard-v1 + devtools-v1 tests to have lower bounds above zero (skills tests pooled +21.7 pp [+19.5, +24.0]), and one locked read (accuracy ≥ parent − 1 pp, served Brier ≤ parent + 0.005). All passed.
  - The in-trial screening gate "held-out pairs ≥ 70 %" fails at this size, as it did for the released parent (0.422 for both), which is why the locked read is named `kev-08b-r15-ungated`.

- Hub: `jaredpalmer/kev-0.8b` (this repo; trial `r15-08b/00-trial-0`; the registration and every read are in `PLAN.md` rounds 9, 11, 12, 13 and 15 at git tag `research-archive-2026-09-24`; specs in `experiments/rounds/`). The previous version is at tag `night2-du-release`; the pre-delta v7 checkpoint at `v7-base`.
- Demo: [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev) runs Kev-4B and Kev-0.8B on ZeroGPU with the same encoder and API code as `kev.serve`.
- Code, suites, results, and the full research log: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md` (full record at git tag `research-archive-2026-09-24`), `runs/leaderboard.md`. The numbers below are in `runs/release/kev-08b-r15.json`; JevBench in `runs/jevbench-public/kev-08b-r15/`.

## Results (as served: each checkpoint at its own fitted temperature)

| | **Kev-0.8B (this version, T = 2.35)** | `night2-du` Kev-0.8B (T = 2.41) | Jev |
|---|---|---|---|
| **real documents**, locked test (`documents-v1`, 936 questions) | **0.851** | 0.608 | – |
| real documents, private held-out (`documents-v2`, 953) | **0.848** | 0.616 | – |
| real documents, development (920) | **0.842** | 0.633 | 0.868 |
| real documents, Brier (locked test) | **0.244** | 0.528 | – |
| **hard-v1**, test (1,088, read once) | **0.665** | 0.396 | – |
| hard-v1, development (1,083) | **0.594** | 0.350 | 0.777 |
| hard-v1 ECE, test / development | 0.125 / 0.112 | 0.138 / 0.140 | – / 0.035 |
| **devtools-v1**, test (1,071, read once) | **0.637** | 0.472 | – |
| devtools-v1, development (1,072) | **0.602** | 0.487 | 0.713 |
| in-distribution accuracy (decision-v7 dev, 1,264 questions) | 0.827 | 0.825 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 656) | 0.648 | 0.652 | 0.857 |
| out-of-domain Brier / ECE | 0.430 / 0.049 | 0.430 / 0.054 | 0.211 / 0.049 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 0.2% | 0.3% | 3.7% |
| coverage at ≤ 5% error | 0.145 | 0.229 | 0.70 |
| held-out policy structures, both siblings correct | 0.422 | 0.422 | 0.86 |
| unknowable items answered at ≥ 0.9 (transfer-v9) | 0.00 | 0.00 | 0.09 |
| MMLU-Pro (transfer-v9 dev, 10-way) | 0.230 | 0.185 | 0.840 |
| **locked test**, out-of-domain accuracy / Brier | **0.697 / 0.397** | 0.684 / 0.412 | – |
| **locked test**, in-distribution accuracy | 0.838 | 0.834 | – |
| SemIf (144 authored decisions) | 0.722 | 0.701 | – |
| scienthoon (873 support tickets) | 0.534 | 0.520 | – |
| WANLI-v2 (1,002 NLI pairs) | 0.602 | 0.570 | – |
| TypeSafe (89 answered rows) | 0.596 | 0.629 | – |
| JevBench public items, all 231 (report only) | 0.636 | 0.597 | – |
| JevBench public, hard tier (111): accuracy / ECE | 0.360 / 0.181 | 0.333 / 0.245 | – |

Jev's devtools-v1 figure is over all 1,074 development questions; Kev's rows drop the reused CodeReviewer id (2 questions), as on the Kev-4B card.

Paired against the `night2-du` version (record-clustered bootstrap, 95 %): documents development +21.0 pp [+18.0, +24.0], locked test +24.4 [+21.3, +27.6], private held-out +23.2 [+19.9, +26.4]; hard-v1 development +24.4 [+20.9, +28.0], test +26.9 [+23.4, +30.4]; devtools-v1 development +11.5 [+8.9, +14.0], test +16.4 [+13.1, +19.4]; SemIf +2.1 [−3.5, +7.6]; scienthoon +1.4 [−1.5, +4.1]; WANLI-v2 +3.2 [+1.7, +4.8]; TypeSafe −3.4 [−13.5, +5.0]; locked out-of-domain test +1.2 [−1.1, +3.7].

**JevBench public items, report only.** `runs/jevbench-public/kev-08b-r15` holds the unchanged harness run against this checkpoint. Over all 231 public items accuracy goes 0.597 → 0.636, most of it on the standard tier: 0.736 → 0.819, with 6 items newly right and none newly wrong. On the hard tier it goes 0.333 → 0.360: paired over the 111 hard items that is +2.7 pp [−1.8, +7.2], with 5 newly right and 2 newly wrong (exact McNemar p = 0.453). Hard-tier ECE falls 0.245 → 0.181. For comparison, Kev-4B's skills delta gained +9.0 pp on the same hard items. The skill data moved the 0.8B far less out of distribution than in distribution.

**Calibration.** The fitted temperature moved 2.41 → 2.35 (raw out-of-domain Brier 0.481 on development, 0.416 on the locked test). As served, out-of-domain calibration is unchanged. `KEV_TEMPERATURE=1.0` gives the raw values.

## Previous version: `night2-du` (2026-09-21) at tag `night2-du-release`

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
