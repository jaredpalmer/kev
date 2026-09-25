# Research plan

This file says where Kev stands, what we have learned, the rules every experiment follows, and what comes next. It is
short on purpose. The full research record (every registration, read, verdict and incident from the Qwen3 prototype
through night 3, rounds 4-18) is frozen at the git tag `research-archive-2026-09-24`; pointers below written as
`A:<path>` mean `git show research-archive-2026-09-24:<path>` (for example `A:PLAN.md`, the 1,322-line record, or
`A:PLAN_27b.md`, the Kev-27B plan). How to run an unattended research session is in
[`docs/autoresearch.md`](docs/autoresearch.md).

## Where we stand (2026-09-24, evening)

### The released family

All four are public in the [Kev collection](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd).
Kev-4B round 10 and Kev-0.8B round 15 were released and Kev-27B made public on 2026-09-24 (docs in PR #106).

| model | checkpoint | T | locked transfer-v4: acc / Brier | other confirmation reads (once each) | JevBench public: all / hard (hard ECE) |
|---|---|---|---|---|---|
| Kev-27B | `r6-27b-v2/01-trial-1` (B1 v2 seed 2), `Qwen/Qwen3.8-27B@1d4bf0f2` (post-trained), Hub `01b81998` | 1.38 | **0.896 / 0.160** | decision-v7 locked 0.870; transfer-r6 test 0.863 vs Kev-9B 0.842 (+2.1 pp [+0.35, +3.8]); longstate-v3 0.833 vs 0.556 | **0.866 / 0.721** (0.128) |
| Kev-9B | `night2-9b-du/00-trial-0` (v7 + dates/unknowable delta) | 2.30 | 0.852 / 0.237 | - | 0.762 / 0.568 (0.19) |
| Kev-4B | `r10-skills/00-trial-0` (night2 + documents-v1 (round 8) + hard-v1 & devtools-v1 (round 10)), Hub `139fdd94` | 2.41 | 0.838 / 0.224 | hard-v1 test 0.540 → 0.803; devtools-v1 test 0.623 → 0.756 | 0.758 / 0.541 (0.112) |
| Kev-0.8B | `r15-08b/00-trial-0` (night2 + documents-v1, hard-v1, devtools-v1 in one delta), Hub `9a45d25e` | 2.35 | 0.697 / 0.397 | documents-v1 test 0.608 → 0.851; hard-v1 test 0.396 → 0.665; devtools-v1 test 0.472 → 0.637; documents-v2 (private) 0.848 | 0.636 / 0.360 (0.181) |
| Jev (reference) | hosted | - | transfer-v4 dev 0.857 (never locked) | - | 0.866 / 0.741 (0.06; their board, tiers include held-out items) |

T is the temperature in `head.pt`, fitted on the trial's decision-v7 development rows; locked Brier is served at that T.
Evidence: `runs/release/kev-{27b-v2,4b-r10,08b-r15}.json`, `experiments/releases/*.json`, `runs/jevbench-public/`, the model
cards. Previous weights are Hub tags (`kev-4b@r8-documents-release`, `kev-4b@night2-du-release`, `kev-0.8b@night2-du-release`, `@v7-base`).

### Against Jev and outside models

- **Decision Index 0.2** (`multimodalart/jev-decision-index`, 40 benchmarks, chance-corrected): Jev 51.67 (ECE 0.065),
  AutoJev-27B 50.94 (ECE 0.023), Kev-9B 35.41 (ECE 0.16), Kev-4B 31.31 (ECE 0.20). The Kev entries are old checkpoints (the
  4B entry predates round 10); Kev-27B is not on it.
- **AutoJev-27B vs Kev-27B** (report only, protocol written before any AutoJev read; `A:runs/autojev-h2h/report.json`,
  `A:PLAN.md` "External head-to-head"). `denis-pplx/autojev-27b` is full-weight SFT of the same base revision on 73k
  curated synthetic decisions. AutoJev as served by its own server; Kev-27B at its shipped T 1.38 (the first version of the
  script used the in-trial fit 1.19; corrected the same day, accuracy unchanged). Paired on shared questions, development
  or public partitions only:

  | suite (questions) | AutoJev | Kev-27B | Jev | AutoJev − Kev, acc [95 %] | ECE AJ / Kev |
  |---|---|---|---|---|---|
  | transfer-v4 dev (656) | 0.863 | 0.848 | 0.857 | +1.5 [−0.6, +3.7] | 0.041 / 0.043 |
  | hard-v1 dev (1,083) | 0.782 | 0.733 | 0.777 | **+4.9 [+2.4, +7.3]** | 0.103 / 0.047 |
  | devtools-v1 dev (1,072) | 0.708 | 0.702 | 0.715 | +0.6 [−0.7, +1.8] | 0.105 / 0.096 |
  | documents-v1 dev (920) | 0.877 | 0.862 | 0.868 | +1.5 [−0.1, +3.3] | 0.015 / 0.088 |
  | SemIf (144) | 0.993 | 0.972 | 0.965 | +2.1 [0.0, +4.9] | 0.048 / 0.068 |
  | scienthoon (873) | 0.769 | 0.796 | 0.753 | **−2.7 [−4.9, −0.6]** | 0.071 / 0.044 |
  | WANLI-v2 (1,002) | 0.764 | 0.745 | - | **+2.0 [+0.3, +3.7]** | 0.082 / 0.070 |
  | TypeSafe (89) | 0.865 | 0.865 | - | +0.0 [−7.4, +8.1] | 0.082 / 0.057 |
  | transfer-v9 dev (1,046) | 0.812 | 0.822 | 0.854 | −1.1 [−3.0, +0.9] | 0.045 / 0.050 |

  Macro accuracy 0.826 vs 0.816; Brier better for AutoJev on seven of nine suites. JevBench public: AutoJev 0.870 (hard
  0.739, hard ECE 0.075) vs Kev-27B 0.866 (0.721, 0.128); paired on the 111 hard items +1.8 pp [−3.6, +7.2]. Kev-27B's
  unreleased round-10 skills arm (hard-v1 0.885, devtools-v1 0.787) would lead on those two suites; it failed our scienthoon guard.

### Running and pending

- **Round 17** (27B skills delta from Kev-27B with replay 10,000, study `r17-27b`, spec `experiments/rounds/r17.json`).
  Arm (a), lr 2e-5, is read out (`runs/r17-readout/round17.json` in the research checkout, not yet committed anywhere) and
  is **not a candidate**: primary +12.1 [+10.4, +13.9] (hard-v1 dev 0.733 → 0.895, +16.2 [+13.5, +19.0]; devtools-v1 dev
  0.702 → 0.783, +8.0 [+5.7, +10.4]), short state +0.3 [−0.9, +1.5], pooled externals −0.1 [−1.0, +0.7], hard-set ECE
  0.047 → 0.018, but documents −1.5 [−2.8, −0.4] fails the ≥ −2 pp lower-bound guard and scienthoon is below its bound.
  Arm (b), lr 1e-5, is still training or reading; **result pending**. Its reads land in the research checkout
  (`research/overnight-r6`); read it out there with `uv run python -m kev.rounds readout experiments/rounds/r17.json --root
  <research checkout>`. Main's harness refuses to launch reads for a recorded round, so a confirmation, if arm (b) passes,
  would be launched from that checkout.
- Decisions waiting on Jared: upload the documents-v1 and hard-v1 train partitions to `jaredpalmer/kev-suites` and bump
  `SUITES_REVISION` (see Next); submit Kev-27B (and the new 4B / 0.8B) to the Decision Index.
- Spend: Modal metered $1,377.01 at 2026-09-24T12:11Z plus round 17's admission bound ($100.24); night 3 used $292 of a
  $1,000 authorization. AI Gateway $0.07 (Jev reference reads only). Modal sponsors the project ($5,000 credits, more on request).

## What we have learned

Each finding names its evidence. Rates are percentage points, intervals are paired record-clustered 95 % bootstraps.

1. **Real-document and skill data are the largest levers measured.** One-epoch deltas gained +7 to +24 pp on held-out
   CFPB documents, +15 to +30 pp on hard-v1 and +8 to +17 pp on devtools-v1, at every size (rounds 7-18, `A:PLAN.md` "Night
   3"). These gains are in distribution by construction (same source or generators, held-out templates). The out-of-distribution
   check is JevBench: Kev-4B round 10 gained +9.0 pp [+2.7, +15.3] on its public hard tier (12 newly right, 2 newly wrong);
   Kev-0.8B round 15 +2.7 pp [−1.8, +7.2] (`runs/jevbench-public/`).
2. **The cost of such a delta falls on other suites and grows with size.** Free at 4B (rounds 8, 10). About 1 pp of
   short-state accuracy at 0.8B, which only the pooled 1,800-question short panel (transfer-v4 dev + transfer-r3 test) could
   bound (rounds 7-9 failed on the 656-question panel alone; rounds 11 and 15 passed on the pooled one). WANLI-v2,
   scienthoon or documents at 9B (rounds 7, 9, 11, 12, 16, 18). Scienthoon at 27B (round 10: −1.8 [−3.0, −0.7]), and
   scienthoon plus documents at 27B with more replay (round 17, arm (a)).
3. **At 0.8B, stacked deltas erode each other; the same data in one delta passes.** Skills on top of the documents
   candidate lost documents and short-state accuracy (round 13); documents + skills trained together passed everything
   (round 15). At 4B, stacking worked (round 8 → round 10), and more data from the same generators gave diminishing returns
   (round 14: +26 pp for the first 6,000 hard-v1 records, +5 for the next 12,000).
4. **Replay stops helping at 9B, and did not fix the 27B's external cost.** Replay 6,000 removed the external cost of the documents delta (round 9) but not of the
   skills delta (round 12: pooled externals −2.1 / −3.5). Replay 10,000 cut the external cost (round 16: −0.6) but then
   cost documents; training documents and skills together with replay 10,000 fixed documents (+7.0) and still failed
   WANLI-v2 and scienthoon (round 18). Every 9B documents arm of rounds 7, 9 and 11 gained +6.5 to +7.3; what moved between seeds was WANLI-v2.
   At 27B the skills delta failed scienthoon with replay 4,000 (round 10); with replay 10,000 (round 17, arm (a)) it still
   failed scienthoon and now also documents (−1.5 [−2.8, −0.4]), though pooled externals were flat (−0.1 [−1.0, +0.7]).
5. **A temperature fitted on easy in-distribution rows does not transfer to hard or distant workloads.** Served ECE on
   hard-v1 development: Kev-4B 0.137, Kev-9B 0.073, Kev-27B 0.047; one temperature refitted on hard-v1's own rows (group-disjoint,
   out of fold) gives 0.067 / 0.034 / 0.041 (`A:PLAN.md` "Target A, first measurement"). On WANLI a workload temperature
   took Kev-9B's ECE 0.131 → 0.037 (round 4.1, `runs/kev-*-wanli-v1/calibration.json`). Decision Index ECE: Kev-9B 0.16,
   Kev-4B 0.20, AutoJev 0.023. Training on hard data also moves it (Kev-4B round 10: JevBench hard ECE 0.263 → 0.112). A single
   global T does transfer from decision-v7 to transfer-v4 (night 2, #2a); per-(type, K) temperatures and a logistic
   reliability head made things worse (night 2 #2a, round 4.10).
6. **hard-v1 tracks JevBench's hard tier family by family**, with no shared items (screen counts in
   `evals/hard-v1/overlap.json`): Jev ahead on probability, dates and judging, Kev-27B level or ahead on long policies and
   ambiguity, on both (`A:PLAN.md` "Round 10", baselines paragraph, and "JevBench").
7. **Full-weight SFT on broad data edges out our LoRA recipe on the same base** (AutoJev table above): ahead or level on
   eight of nine suites, much better calibrated on JevBench's hard tier and on documents, behind on scienthoon. It is not a
   controlled comparison: AutoJev differs in method (full weights) and in data (73k broad synthetic decisions; our replay
   is decision-v7 only) at once.
8. **Knowledge is set by the base.** MMLU-Pro: untrained Qwen3.5-9B 0.540, Kev-9B 0.545 (0.515 after the night-2 delta),
   Kev on Qwen3.6-35B-A3B 0.550, untrained Qwen3.8-27B 0.635, Kev-27B 0.665, Jev 0.840 (night 2 #7/#8; `A:PLAN.md` "Qwen3.5
   port" Phase 0; A2). Solomon found the same at 27B.
9. **LoRA training on our format erodes a Base checkpoint's date arithmetic; stating the day count fixes the readout.**
   Qwen3.5-9B `deadline` 0.82 zero-shot → 0.72 trained; the adapted backbone read through the LM head scores the same as the
   pointer (the skill is lost in the representation, `A:PLAN.md` "Qwen3.5 port" §10). A post-trained 9B eroded more (0.70 → 0.47,
   round 4.8); question-side LoRA did not protect it at 4B / 9B (A1). With the day count stated, 0.65 → 1.00 (4B) on a fresh diagnostic
   (`runs/binding-diagnostic-v1`). The post-trained 27B kept 0.975, yet JevBench's temporal_numeric family is its weakest (0.07).
10. **Continue training with soft targets, not hard labels.** Ambiguity soft targets (open teacher disagrees with the
    public label at p ≥ 0.6) beat a matched hard-label delta on accuracy and Brier (round 4.9; release confirmation:
    Brier −0.011, accuracy +1.1 on the round-3 final panel), but alone were not a release. Softened MNLI targets caused
    the WANLI dip (round 6: −2.0 vs +0.3 with MNLI kept hard); threshold 0.8 was the best rule for long-state deltas.
11. **Buried synthetic states are not real documents.** Burying a state in 1-4k tokens of unrelated records cost 22-43 pp
    (round 4.12) and training closed +17 to +20 pp of it (rounds 5, 6); but on real CFPB narratives Kev-9B loses 2.6 pp
    from short to long, and the long-state training moved documents-v1 by ±1 pp (`A:PLAN_27b.md` "documents-v1 result").
    Judge long-document work on real documents.
12. **Guards must be sized to what a suite resolves.** On 656 questions the accuracy interval is about ±1.7 pp, so a −1 pp
    lower bound needs a point estimate near +0.7; TypeSafe's 89 rows swing ±6 pp; the 27B's MMLU-Pro gate on 200 questions
    split two seeds 0.630 / 0.665. Round 5 turned on three WANLI questions and round 6 on 0.05 pp. Seed variance at 9B is
    about ±1 pp on short states and ±2 pp on long panels; a one-seed lead of a point is noise.
13. **Negative results worth remembering** (each tried and recorded; do not repeat without a new reason):
    - calibration losses (label smoothing, CE + Brier, focal) against a matched CE control: no candidate (round 3);
    - question-side LoRA (A1): at 4B / 9B −3.4 to −6.0 pp transfer against the same-seed full-placement trial and no date
      arithmetic kept; at 27B trial C was −0.9 pp against trial A with `deadline` kept (0.97 vs 0.975), but lost MMLU
      (0.850 vs 0.863), held-out pairs (0.89 vs 0.92), coverage at ≤ 5 % error (0.645 vs 0.720) and the conditional
      rule task (−21.9 pp vs Kev-9B); placement stays `full`;
    - post-trained Qwen3.5-9B as base (round 4.8), Qwen3.6-35B-A3B (night 2 #8: +1.2 pp, worse calibration, 8× memory),
      DeltaNet-frozen LoRA (night 2 #5);
    - checkpoint averaging (4.5), reliability head (4.10), 9B → 0.8B self-distillation (4.11), `KEV_DATE_FACTS` as default
      (4.2: pushes TypeSafe documents past the context);
    - folding the night-2 delta data back into later deltas (round 6 `soft-du`), two epochs instead of one at 27B (round 6
      follow-up), more same-generator data on top of a skills delta (round 14);
    - anchoring toward the base's answers, WiSE-FT interpolation, option isolation, special embeddings, `head_dim`,
      `perm_kl`, `ord_w`, more public data at 4B; the from-scratch config space around lr 5e-5 is exhausted
      (overnight-1 and "Toward v0.2", `A:PLAN.md` History);
    - reinforcement learning has not been tried by us; the Laya review argued that REINFORCE against a proper score has
      the same optimum as the log loss we minimise (`A:PLAN.md` "Qwen3.5 port" §6).

## Standing rules for every round

These are the methods that held up. `docs/autoresearch.md` turns them into an operating procedure.

- **Register before training or reading.** The round's section in PLAN.md and its spec `experiments/rounds/r<N>.json`
  (arms, parents, reads, rule, confirmation stages) are committed before any training or read. The commit time is the
  registration time; do not write clock estimates into headings.
- **Development partitions select.** Selection sets are development partitions and panels already read.
- **Test and locked partitions are read once per candidate**, only for the candidate the committed rule selected, only
  after that rule passed. Never for a second candidate; never re-read. A read panel becomes a selection or regression set
  for everything after it. Locked reads are named `kev-<size>-r<N>` (`-ungated` when an in-trial screening gate failed).
- **Paired record-clustered bootstraps** decide: candidate minus parent on the same rows, `kev.rounds.paired` (2,000
  resamples, seed 0, micro). Criteria are on interval bounds; every number carries checkpoint, suite and partition, n and
  report path.
- **Guards are sized to what a suite can resolve.** Pool small suites (the pooled external guard, the pooled 1,800-question
  short-state panel); gate small suites only through the pool; a "not worse" guard has no point-estimate requirement.
- **Served against served.** Every side is served at the temperature fitted on its own decision-v7 development rows
  (`kev.metrics.served`); a release writes that T into `head.pt` (`scripts/calibrate_checkpoint.py`) and its reported numbers
  use the shipped T. A hard-set calibration guard (hard-v1 served ECE ≤ parent + 0.01) is part of every skills round since round 10.
- **One candidate per size**: the passing arm with the best registered rank; attribution arms are reported, never selected;
  say how many arms were tried ("one pass in five").
- **No Jev output in training, ever.** Jev is a reference read through the AI Gateway, budget-capped.
- **Open-weight teachers only for training labels** (DeepSeek, Qwen and similar) or programmatic solvers; closed frontier
  models may judge or filter evaluation labels only.
- **Frozen files never change.** New data is a new versioned directory with a manifest (sha256 of every partition and of the
  inputs); defects found later are documented, not fixed in place.
- **Budgets and state.** A spend authorization per session, checked before every launch against metered spend plus running
  admission bounds; a state file with every spawn id, bound, pull and read.
- **Report negative results as fully as positive ones**, in PLAN.md, with the failed criterion.
- **No publishing or Hub changes without Jared's explicit OK; code reaches main only through reviewed PRs.**

## Data policy for the SFT work (decided by Jared, 2026-09-24)

- The Kev-27B SFT corpus stays private. Its data lives in a private Hub dataset (planned `jaredpalmer/kev-private-train`);
  this public repo holds only each suite's `manifest.json`, with a `"mirror"` entry, as `evals/documents-v2` does.
- Its builders and generation prompts live in a private companion repo, not in this public repo. Manifests record the
  private repo's commit and the code's sha256.
- Training labels and generations come only from open-weight teachers (DeepSeek, Qwen and similar) or programmatic solvers.
  Closed frontier models may judge or filter evaluation labels only. Jev never.
- Model cards disclose each source's kind, size, licence, generation method and the contamination screens, not the text.

## Round 19 (registered)

### Round 19 - full-weight SFT of Kev-27B on a broad corpus (registered 2026-09-25T03:45Z, before any training or read)

**Why.** Three results point the same way. (1) On the same base weights, AutoJev-27B (full-weight SFT on 73k broad synthetic decisions) edges out Kev-27B (LoRA on Kev's narrower mix) on eight of nine of our suites and on the community Decision Index (50.94 vs Jev 51.67; Kev-9B 35.41). (2) On `breadth-v1` (14 held-out datasets in the Decision Index's five areas, never trained on) the development index is Jev 53.3, AutoJev 51.7, Kev-27B 50.2, Kev-4B 40.8, and nearly the whole gap is Retrieval & Classification (Kev-27B 62.9 vs 75.4 / 76.1). (3) Every LoRA delta at 9B and 27B that learned new skills paid on WANLI-v2 / scienthoon, and more replay did not fix it (rounds 10, 16, 17, 18): the missing ingredient is breadth of training data, not replay. This round tests whether full-weight SFT of the base on a broad corpus beats Kev-27B, and by how much it closes the gap to Jev and AutoJev, with calibration fitted on a broad held-out pool rather than on easy in-distribution rows.

**Data (private; policy in "Data policy for the SFT work").** `evals/sft-v1` (manifest only in this repo; partitions in the private dataset `jaredpalmer/kev-private-train` @ `119c1e7d`; manifest sha256 `6e0d0150`):
- public train splits: 93,798 train records from 24 licence-checked sources across the five areas (native labels; train splits only; the 15 breadth-v1 datasets, every source behind Kev's evaluation suites, WANLI, MMLU / MMLU-Pro and JevBench excluded; every record screened against all of them);
- Kev's existing training data: Kev-27B's own training set (b1v2: decision-v7 with soft targets + dates/unknowable + long states), documents-v1 train (minus 142 near-duplicates flagged against documents-v1/v2 evaluation narratives), hard-v1 train + a fresh-seed hard-v1 set, devtools-v1 train (minus 258 screen hits against its own dev/test);
- synthetic: 65,667 (train 61,094 + 6,259 soft-target records from training groups) kept records written and labelled only by open-weight models (GLM-5.3, DeepSeek-V4-Pro, Inkling; a fourth open model votes on disagreements), a question kept only when both blind labelers agree with the generator's intended answer; families: intent / dialogue state / out-of-scope routing, retrieval relevance, long documents, tool routing, rubric judging, abstention twins, numeric (code-computed answers); soft targets (labelers' vote distribution) only in judging / abstention; screened against JevBench and every Kev dev/test partition;
- partitions: train 198,691 (337,406 questions; 141.8M row tokens with the state shared per record) records; calibration 8,470 (14,960 questions); development 3,483 (6,146 questions) (public held-out phrasing + synthetic held-out items). Never trained on: calibration, development. Decision Index benchmarks that share a dataset with training data (in-distribution for any later Index read): ARC-Easy/Challenge, OpenBookQA, CommonsenseQA, GSM8K, WinoGrande, Amazon ESCI, BANKING77.

**Arms** (spec `experiments/rounds/r19.json`, plans `experiments/round19/`; 8×H200 per trial, FSDP2, fp32 masters, prefix-shared rows, balanced micro-batches, resume points hourly): fresh from `Qwen/Qwen3.8-27B` @ `1d4bf0f2` (not from Kev-27B), whole text backbone + pointer head, one epoch, 128 records per step (batch 8 × accum 2 × 8 GPUs), bf16 autocast, OneCycle (10 % warm-up), head lr 1e-4, `p_none_pair 0.25` (Kev-27B's recipe), max state 7,552 tokens:
- (a) `27b-lr2e6`: lr 2e-6 (AutoJev's);
- (b) `27b-lr5e6`: lr 5e-6;
- (c) `27b-olddata` (attribution, never the candidate): arm (b)'s settings on Kev-27B's own training set (`evals/round6/b1v2/train.jsonl` via `--data`; calibration and development from `evals/sft-v1`, like (a) and (b)), two epochs — full weights on the old data, so (b) vs (c) measures the data and (c) vs Kev-27B measures full weights vs LoRA.

**Calibration (registered approach).** Temperature is fitted on a broad held-out pool, not on decision-v7 development rows: every side of every comparison is served at the temperature fitted on its own development rows (for the SFT arms `evals/sft-v1` development: all public sources' held-out phrasing plus synthetic held-out items; Kev-27B keeps its shipped 1.38); a released SFT checkpoint gets its temperature from `scripts/calibrate_checkpoint.py` on the same rows, with the out-of-fold check. Soft targets only where open-weight labelers genuinely disagree; no label smoothing, focal or Brier terms (round 3: none improved ranking; smoothing hurt AURC). Reported, not gating: per-question-type temperatures (choice / noul / score) and by option count, chosen only if out-of-fold ECE improves with a separated interval; coverage at ≤ 5 % error and AURC on breadth-v1 and the Kev panel (issue #111); permutation flip rate (option order is reshuffled per record in training; test-time rotation averaging stays a serving option).

**Rule** (against Kev-27B, paired record-clustered bootstraps, 2,000 resamples):
1. primaries: breadth-v1 development accuracy lower bound > 0; pooled Kev development panel (transfer-v4 dev, hard-v1, devtools-v1, documents-v1) accuracy lower bound ≥ −1 pp;
2. guards: short state (transfer-v4 dev + transfer-r3 test) accuracy lower ≥ −2 pp, Brier upper ≤ +0.01, confident errors upper ≤ +1 pp; WANLI-v2 and scienthoon lower ≥ −2 pp each; pooled externals (SemIf, scienthoon, WANLI-v2, TypeSafe) lower ≥ −1.5 pp; unknowable share on transfer-v9 ≤ 0.05;
3. calibration: breadth-v1 ECE ≤ Kev-27B's + 0.01 and Kev-panel ECE ≤ Kev-27B's + 0.01;
4. candidate: the passing selectable arm with the largest breadth + Kev-panel accuracy gain.

Reported alongside (never gating): Jev and AutoJev on the same breadth-v1 development items (their committed reads) and the chance-corrected breadth index with intervals; JevBench public items through the unchanged harness.

**Confirmation** (candidate only, each read once, after the rule): breadth-v1 test (lower bound > 0 vs Kev-27B; Jev and AutoJev read once on the same test items, reported); hard-v1 + devtools-v1 + documents-v1 test pooled lower ≥ −1 pp; documents-v2 reported; locked transfer-v4 accuracy ≥ 0.886 (Kev-27B 0.896 − 1 pp) and served Brier ≤ 0.165; the bf16 serving check on main's path (max |Δp| ≤ 0.03, ≤ 1 flip in 280, isolation) before any release.

**Budget.** Modal: admission bounds $988 + $988 + $371 = $2,347 (expected spend ~$800-950: arms (a)/(b) ~9 h each with p_none_pair, so one automatic resume each; arm (c) ~1.5 h) (one epoch of the full mix measured at ~7.1 h / ~$293 on 8×H200; retries counted in each bound), reads ~$20 per arm; program cap $2,000 including reads and confirmation. AI Gateway: synthetic data $1,666 of the $2,460 key (spent before this registration, not training). Baseline: Modal metered $1599.26 at registration.

## RL pilot 1 (registered)

### RL pilot 1 - agentic RL after SFT on Kev-4B (registered 2026-09-25T23:40Z, before any training or read)

**Why.** Kev has never been trained with RL, and the Laya review's argument stands for single decisions: REINFORCE against a proper score has the log loss's optimum. What SFT cannot supervise is a sequence of decisions scored only at its end. `kev.rl` trains Kev on `kev.envs.Investigation` episodes: a hidden staff directory, a task that follows 1-2 manager hops to a city or team, and one typed choice per step (open a record / answer a value / escalate). 25 % of episodes redact a record on the chain, so no answer is knowable and escalating is right. Rewards are programmatic (+1 correct, -1 wrong, 0 escalate, -0.5 over budget, -0.02 per record opened); the solver scores 0.73 on average. Kev-4B is the testbed; larger checkpoints only if this passes.

**Calibration is the main risk.** Policy gradient sharpens toward whatever was rewarded. Guards: rollouts sample at the parent's served temperature (tau 2.41) and the loss uses the same tempered log-probabilities; an exact per-step KL(pi || pi_SFT); SFT replay (devtools-v1 train, the parent's own data, with kev.train's loss on raw logits). The RL checkpoint keeps the parent's temperature until it is refitted on its own development rows (`scripts/calibrate_checkpoint.py`), as every served comparison requires.

**Arms** (from `jaredpalmer/kev-4b` @ `139fdd94`, LoRA r16 continued, 1×H100, 150 iterations × 16 worlds × 8 rollouts, lr 1e-5, eval on 400 held-out-seed episodes in the held-out template every 25 iterations):
- (a) `rl-guarded`: kl_w 0.05, replay_w 0.5 (8 records per iteration) - the candidate;
- (b) `rl-noguard` (attribution, never the candidate): kl_w 0, no replay - shows what the guards cost and what they prevent.

**Rule** (candidate (a) vs parent Kev-4B, on held-out episodes and, for the panel, paired record-clustered bootstraps, 2,000 resamples):
1. env: greedy return gain lower bound > 0 (episode bootstrap), unknowable-escalation rate not lower, answer-step ECE not above the parent's;
2. guards, each checkpoint served at its own refitted temperature: pooled Kev development panel (transfer-v4 dev, hard-v1, devtools-v1, documents-v1) accuracy lower ≥ -1 pp, ECE ≤ parent + 0.01, Brier upper ≤ +0.01; WANLI-v2 and scienthoon lower ≥ -2 pp; fitted temperature reported next to the parent's 2.41;
3. passing all three means RL works at 4B and earns a registered follow-up: a matched SFT-on-solver-demonstrations control, harder environments, and a larger checkpoint. Failing means the result is reported and nothing scales.

**Budget.** ~2 × 1.5 h H100 ≈ $12 for training plus ~$10 for panel reads; cap $40.

**Result (negative, 2026-09-26T00:25Z).** Both arms collapsed to "always escalate" within 25 iterations and stayed there (stopped at iteration ~90, ~$6): held-out greedy return -0.41 -> 0.00, knowable-correct 0.27 -> 0.00, opens 1.6 -> 0, entropy 1.68 -> 0.006; per-step KL to the parent reached ~2.3 nats in both arms, so kl_w 0.05 did not hold the policy. The reward made escalating a flat 0 while the parent's answers were right a quarter of the time, so the policy found the one action that never loses. No checkpoint was saved and no panel read was made. Lessons: an abstention action needs a cost when the answer was knowable (otherwise RL learns blanket abstention, the calibration failure we were guarding against, expressed as behaviour), and the KL anchor has to be an order of magnitude stronger at this lr.

### RL pilot 1b - reward fixed, stronger anchor (registered 2026-09-26T00:30Z, before any training or read)

Changes from pilot 1: escalating scores +0.5 when the question is unknowable and -0.5 when it was knowable (always-escalate now averages -0.25; the solver 0.82; the parent -0.4). Arms (otherwise pilot 1's settings, both with replay 0.5): (a) `rl-kl0.2` kl_w 0.2 - the candidate; (b) `rl-kl0.05` kl_w 0.05 (attribution: the anchor's strength). Same rule as pilot 1, plus: an arm whose held-out greedy policy escalates > 90 % of knowable episodes, or whose step entropy falls below 0.05, fails as collapsed. Budget: ~$12 training + ~$10 reads; cap $40 including pilot 1's spend.

**Result (negative, 2026-09-26T00:50Z).** Both arms collapsed to always-escalate by iteration 25 again (greedy return -0.37 -> -0.26, knowable-correct 0.27 -> 0, entropy < 0.06, KL ~2.2 nats even at kl_w 0.2); stopped at iteration ~40 (~$3). The reward fix could not help: always-escalate (-0.25) still beats the parent (-0.37), so it stays the nearest improvement. The parent has to be able to win the task before RL can refine it.

### RL pilot 2 - SFT on solver demonstrations, then RL (registered 2026-09-26T00:55Z, before any training or read)

The standard recipe: `kev.rl --warmup_episodes 300 --warmup_epochs 2` behaviour-clones the solver on 300 train-namespace worlds (seeds from 10^6, apart from RL's), evaluated as phase `warmup`; RL then runs as in pilot 1b, and the KL now anchors to the policy as RL starts (post-warm-up), not the released parent. Arms: (a) `rl2-kl0.2` - candidate; (b) `rl2-kl0.05` - anchor attribution. **The RL question is read against the warm-up policy** (same run, same held-out 400 eval-seed episodes, paired by episode): the candidate needs a positive paired-bootstrap lower bound on greedy return vs warm-up, knowable-escalate not above warm-up + 0.05, answer-step ECE not above warm-up + 0.02, and must not trip pilot 1b's collapse rule. The panel gates (pilot 1's, at a temperature refit per checkpoint) are read for warm-up and final against the released parent only if the env gate passes. Spend cap $40 across pilots 1-2 (so far ~$10).

**Result (uninformative, 2026-09-26T01:10Z).** The warm-up alone solved the environment: 996 solver steps (2 epochs) took held-out greedy return from -0.37 to 0.822 (the solver's own return is 0.82), knowable-correct 0.993, unknowable-escalated 1.00, answer ECE 0.005, in both arms. With imitation already at the ceiling RL has no headroom, so no RL gain can be read (first 5-9 RL iterations moved the sampled return within noise, KL < 0.002). Both arms were stopped (~$3; ~$13 across pilots 1-2) and no panel read was made. Lesson: an environment with a cheap exact solver is a behaviour-cloning problem. RL only adds something where the best action depends on the model's own uncertainty, so no demonstration can supply it: noisy or conflicting records, where opening more costs and the right moment to stop depends on the belief. That redesign comes before any further RL spend or scale-up.

## Next

Goals and open questions, not registered rounds; each becomes a spec and a PLAN section before it runs.

1. **SFT program for Kev-27B: full-weight SFT on broad data, with calibration done right.** Findings 1, 5 and 7 point
   here. Questions to settle before registering:
   - Data: what broad decision corpus, built under the policy above (sources, sizes, open-weight teachers, contamination
     screens against JevBench public items and our frozen suites), and how much of it replaces or joins decision-v7 replay.
   - Method: full-weight SFT against the current LoRA recipe on the same data, so method and data are separated (the
     AutoJev comparison confounds them). The trainer exists (`kev.train --full_ft 1`, PR #122; checkpoints are a
     `save_pretrained` bf16 backbone of 51 GB plus `head.pt`, loaded by the same `kev.checkpoint` path, fused kernels and
     CUDA graphs included). The follow-up (PR #125) added the shared prefix in training (each state once, its questions
     from it; exact to the row form), micro-batches balanced by padded length, resume points (bit-identical continuation;
     full-weight trials are retried after a timeout and continue) and 24 h full-weight studies. Measured on Qwen3.8-27B,
     8 H200s, records shaped like the SFT corpus (`experiments/sft-v1-lengths.json`, `runs/sft-probe/sft2-*`): the whole
     mix (public 94k + components 38k + synthetic 60k) at 7.6 records/s, one epoch ≈ 7.1 h and $293, two ≈ 14.1 h and
     $582, before the in-trial reads; a resume point (~307 GB) blocks training ~23 s and writes in ~2.5 min behind it. The
     synthetic part alone runs 7.3 records/s shared against 2.7 in the row form (same balanced batching). Open: the 27B
     served path at the release isolation tolerance.
   - Calibration: fit the served temperature on a mixed development pool (decision-v7 + hard-v1 + devtools-v1) and gate on
     hard-set calibration, not only on easy rows (finding 5).
   - Evaluation: every existing short-state confirmation panel has been read at least once, so the round needs a new frozen
     panel; the AutoJev head-to-head suites are the comparison; JevBench's sealed half stays the external check.
   - The 27B LoRA skills path has now failed the external guards at replay 4,000 (round 10) and 10,000 (round 17, arm (a)),
     so B1 v2 (the released Kev-27B) remains the 27B baseline, and more replay is not the remedy; breadth of data, which
     is what this program adds, is. Round 17 arm (b) is reported when it lands.
2. **A 9B remedy other than replay** (finding 4): a KL term toward the released Kev-9B's own served answers on the replayed
   records (`kev.anchors` today targets the frozen base's zero-shot answers; this needs the released model's distributions as
   the target), or broader replay.
3. **Publish the documents-v1 and hard-v1 train partitions** (23 MB each, not in git, not yet in `kev-suites`): hard-v1 is
   programmatic and regenerates byte for byte; documents-v1 train is public-domain CFPB text with teacher-agreed labels.
   Decide, upload, bump `SUITES_REVISION`, so Kev-4B's and Kev-0.8B's training data is fetchable.
4. **Decision Index submission** for Kev-27B and the current Kev-4B / Kev-0.8B.
5. Smaller open items: rotation-averaged Choice met its round-4.4 gate (permutation flips 0.028 → 0.000 at 9B) and waits for
   a product decision (it multiplies latency); date arithmetic stays the weakest family at every size (finding 9).

## Record

One line per round or named study. `rN.json` is `experiments/rounds/rN.json` on main (the rule as data; `python -m
kev.rounds readout` reproduces rounds 5-18, see `tests/test_rounds.py`); `A:` is the archive tag. Verdicts are the registered
outcomes.

| round / study | date | what | verdict | where |
|---|---|---|---|---|
| Round 4 | 09-22 | twelve cheap levers before the 27B (4.1-4.12) | adopted: per-workload calibration report, paired-CI incumbent rule, full-partition MLX parity, OOF audit field; long states are a data problem (4.12); negative: 4.2, 4.5, 4.8, 4.10, 4.11; 4.9 missed its gate; 4.4 met its gate, serving mode pending | `A:PLAN.md` "Round 4", "Round 4 results" |
| Release confirmation | 09-22 | soft-target Kev-9B (`r4-soft`) on the round-3 final panel | not released (Brier bound, WANLI −1.2) | `A:PLAN.md` "Release confirmation"; `runs/rc-verdict` |
| Round 5 | 09-22 | long states + soft targets, all sizes | no release; 9B missed WANLI by 3 questions | `r5.json`; `runs/r5-verdict`; `A:PLAN.md` "Round 5" |
| Round 6 | 09-23 | overnight hill-climb: 22 delta trials (long states, soft-target variants) at 9B / 4B / 0.8B | no candidate; MNLI soft targets cause the WANLI dip | `r6.json`; `A:PLAN.md` "Round 6"; `A:runs/r6-readout` |
| A1 | 09-23 | question-side LoRA, from scratch, 4B × 2, 9B, 27B | negative: 4B / 9B lose 3-6 pp and the date arithmetic; 27B keeps both but loses MMLU, pairs and coverage; placement stays `full` | `A:PLAN.md` "Round 6" > A1 |
| A2 | 09-22 | Qwen3.8-27B zero-shot probe | 2 of 3 gates (MMLU-Pro 0.635 < 0.65); B1 authorized by Jared as a recorded override | `A:PLAN_27b.md` gating addendum; `runs/probes/qwen38-27b-*` |
| B1 | 09-23 | Kev-27B, v7 recipe, 1 epoch, 3 trials | trial A missed by 0.15 pp on the paired bound and 2 questions on one task | `A:PLAN.md` "Round 6" > B1 |
| Round 6 follow-up | 09-23 | Kev-27B, 2 epochs, seeds 1-2 | no candidate; two epochs bought nothing | `A:PLAN.md` "Round 6 follow-up" |
| B1 v2 | 09-23/24 | Kev-27B on v7 + dates/unknowable + long states + soft targets | seed 2 passed every criterion; bf16 serving check passed; released as Kev-27B | `A:PLAN_27b.md` "B1 v2", "bf16 serving check"; `runs/release/kev-27b-v2.json` |
| documents-v1 / v2 | 09-23 | real CFPB narratives; teacher-agreed train, judge-panel + double-adjudicated eval labels | frozen; spot checks 47/50 and 50/50 | `A:PLAN_27b.md` "documents-v1 result", "documents-v2"; `evals/documents-v{1,2}/manifest.json` |
| Round 7 | 09-23 | documents delta, all sizes | no candidate: 0.8B / 4B failed bounds on suites too small to resolve them, 9B / 27B paid on short states or externals | `r7.json`; `A:PLAN.md` "Round 7" |
| Round 8 | 09-24 | documents delta at 0.8B / 4B, guards sized to suites | **Kev-4B confirmed, released** (later superseded by round 10) | `r8.json`; `runs/r8-readout`; `runs/release/kev-4b-r8.json` |
| JevBench | 09-24 | public items, unchanged harness, released family | report; Kev-9B hard 0.568 vs Jev 0.741 | `A:PLAN.md` "JevBench"; `runs/jevbench-public` |
| Round 9 | 09-24 | documents at 9B / 0.8B with more replay / smaller step | no candidate in five arms | `r9.json`; `runs/r9-readout` |
| Round 10 | 09-24 | skills delta (hard-v1 + devtools-v1), 4B and 27B | **Kev-4B confirmed, released**; 27B failed scienthoon | `r10.json`; `runs/r10-readout`, `runs/r10-verdict` |
| Round 11 | 09-24 | round 9's recipe, fresh seeds, pooled short panel | 0.8B documents confirmed (superseded by 15); no 9B | `r11.json`; `A:runs/r11-verdict` |
| Round 12 | 09-24 | skills delta at 9B / 0.8B | 0.8B skills confirmed (superseded by 15); no 9B | `r12.json`; `A:runs/r12-verdict` |
| Round 13 | 09-24 | 0.8B skills on top of the documents candidate | no candidate: stacking erodes | `r13.json`; `runs/r13-readout` |
| Round 14 | 09-24 | 12,000 more hard-v1 records on the round-10 4B | no candidate: diminishing returns | `r14.json`; `A:runs/r14-readout` |
| Round 15 | 09-24 | 0.8B documents + skills in one delta | **Kev-0.8B confirmed, released** | `r15.json`; `runs/r15-readout`, `runs/r15-verdict` |
| Round 16 | 09-24 | 9B skills, replay 10,000 | no candidate (documents cost) | `r16.json`; `A:runs/r16-readout` |
| Round 17 | 09-24 | 27B skills, replay 10,000 | arm (a) lr 2e-5: no candidate (documents, scienthoon); arm (b) lr 1e-5: **pending** | `r17.json`; `A:PLAN.md` "Round 17" |
| Round 18 | 09-24 | 9B documents + skills, replay 10,000 | no candidate (WANLI-v2, scienthoon) | `r18.json`; `A:runs/r18-readout` |
| AutoJev head-to-head | 09-24 | AutoJev-27B vs Kev-27B, report only | see "Against Jev" above | `A:runs/autojev-h2h/report.json` |
| breadth-v1 | 09-24 | frozen eval-only panel over the Decision Index's five areas (14 held-out datasets, 150 records each, locked test unread); development baselines | report: chance-corrected index Jev 53.3, AutoJev-27B 51.7, Kev-27B 50.2, Kev-4B 40.8 (Kev-27B vs Jev −3.1 [−6.2, +0.1]); Kev-27B trails most on retrieval (SGD, CLINC150) | `evals/breadth-v1/manifest.json`; `runs/breadth-v1-report/report.md` |

Older milestones, all in `A:PLAN.md` (sections named in parentheses):

- **v3 protocol and matched data-vs-capacity study** (before 2026-09-19; "v3 protocol", "Status and deferred work"): capacity
  0.6B → 4B +17.5 to +19.0 pp transfer; compositional policy data +3.6 to +5.1 pp; immutable suites, grouped splits, minimal pairs.
- **Overnight-1** (branch `research/overnight-1`, PR #3; "Overnight autoresearch"): lr 5e-5 instead of 2e-4 is the lever at
  4B / 8B (+4.7 pp [+0.4, +9.6]); fine-tuning erodes base capability; the config space is exhausted; research previews.
- **Toward v0.2** (2026-09-19/20; "Toward v0.2"): `decision-v7` (random rule trees, ordinal Score families), the Qwen3 family
  release, anchoring not a lever; deadline stays the deciding family.
- **Qwen3.5 port** (2026-09-20; "Qwen3.5 port" §1-§10): row-batched forward for the hybrid backbone; Kev-9B locked 0.837 vs
  Kev-8B 0.780 (+7.3 pp [+2.8, +11.7]); the family moves to Qwen3.5; the deadline erosion is in the representation.
- **Night 2** (2026-09-20/21; "Round 2 autoresearch", "Results", "Decisions taken"): dates + unknowable delta promoted
  (Kev-9B locked 0.852), temperature built into `head.pt`, `date_facts` opt-in, 35B-A3B not shipped.
- **Round 3 calibration audit** (2026-09-21/22; "Round 3 autoresearch"): metric v2 (tie-aware coverage, AURC, full-statistic
  bootstrap); the matched loss screen selected nothing; the binding diagnostic showed the date gap is subtraction, not binding.
