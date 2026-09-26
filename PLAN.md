# Research plan

This file says where Kev stands, what we have learned, the rules every experiment follows, and what comes next. It is
short on purpose. The full research record (every registration, read, verdict and incident from the Qwen3 prototype
through night 3, rounds 4-18) is frozen at the git tag `research-archive-2026-09-24`; pointers below written as
`A:<path>` mean `git show research-archive-2026-09-24:<path>` (for example `A:PLAN.md`, the 1,322-line record, or
`A:PLAN_27b.md`, the Kev-27B plan). How to run an unattended research session is in
[`docs/autoresearch.md`](docs/autoresearch.md).

## Where we stand (2026-09-26)

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

- **Round 19** (full-weight SFT of Qwen3.8-27B on `sft-v1`) is read out: **no candidate**; every arm failed scienthoon, the
  pooled externals and the calibration criteria (see "Round 19 result"). **Round 20** (no training; round 19's checkpoints
  served at a temperature fitted on held-out datasets, plus WiSE-FT interpolation with the base) is read out: **no
  candidate**, 0 of 6 interpolations pass. The registered temperature fixed calibration: arm (a) breadth ECE 0.0085 vs
  Kev-27B 0.0118. Every arm still fails scienthoon and the pooled externals (see "Round 20 result"). A report-only analysis
  of the scienthoon failure finds that the whole cost is one borderline judgement (calm complaints read as "angry"), and
  that on this suite Kev-27B is the best of six LoRA checkpoints trained from the base on Kev's data. None of the other
  five, its own recipe's second seed included, would pass the guard against it (`runs/r20-scienthoon/analysis.md`).
  **Round 21** (retraining allowed, long context, extended data) is not registered yet (see Next).
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
- Spend: Modal metered $2,488.88 at 2026-09-26T00:02Z (round 20: +$54.87 over its registration baseline of $2,434.01,
  workspace-wide); $2,434.01 at 2026-09-25T22:53Z was +$834.75 over round 19's registration baseline of $1,599.26 (its
  training, re-scoring and reads, plus other apps of the workspace); earlier, $1,377.01 at 2026-09-24T12:11Z plus round 17's admission bound ($100.24), and night 3 used $292 of a
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
   reliability head made things worse (night 2 #2a, round 4.10). Held-out *items* of the training sources are still in
   distribution: round 19's SFT arms, served at T 0.955 fitted on `sft-v1` development rows, had breadth-v1 ECE 0.059 / 0.065
   against Kev-27B's 0.012; one temperature fitted on held-out *datasets* brought arm (a) to 0.017 (exploratory, chosen after
   seeing breadth-v1; "Round 19 result"). Round 20 registered such a pool (648 questions of eight held-out public sources
   plus MMLU-Pro, which no rule panel reads). It gave T 1.41 for arm (a): breadth ECE 0.0085 vs Kev-27B 0.0118, Kev-panel
   ECE 0.0193 vs 0.0216. Every final and every interpolation down to α 0.70 passed both calibration criteria ("Round 20
   result").
6. **hard-v1 tracks JevBench's hard tier family by family**, with no shared items (screen counts in
   `evals/hard-v1/overlap.json`): Jev ahead on probability, dates and judging, Kev-27B level or ahead on long policies and
   ambiguity, on both (`A:PLAN.md` "Round 10", baselines paragraph, and "JevBench").
7. **Full-weight SFT on broad data edges out our LoRA recipe on the same base** (AutoJev table above): ahead or level on
   eight of nine suites, much better calibrated on JevBench's hard tier and on documents, behind on scienthoon. It is not a
   controlled comparison: AutoJev differs in method (full weights) and in data (73k broad synthetic decisions; our replay
   is decision-v7 only) at once. Round 19 separated them on our own corpus: the broad data carries the gains (full weights
   on `sft-v1` against full weights on Kev-27B's own data: Kev panel +9.1 [+7.8, +10.5], short states +2.9 [+1.7, +4.3]),
   while full weights instead of LoRA on the same data cost short states −3.0 [−4.3, −1.9] and scienthoon −3.7 [−6.0, −1.5]
   and gained nothing ("Round 19 result"); the SFT arms kept that scienthoon cost (−2.9, −3.6).
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
    about ±1 pp on short states and ±2 pp on long panels; a one-seed lead of a point is noise. On scienthoon, six 27B LoRA
    checkpoints trained from the base on Kev's data score 0.740-0.796 (sd 2.1 pp), and Kev-27B is the top one. Most of the
    spread is one borderline Noul ("sounds angry") plus `priority`, whose label is not in the text. None of the other five
    passes the −2 pp scienthoon guard or the pooled-externals guard against Kev-27B (`runs/r20-scienthoon/analysis.md`).
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
    - WiSE-FT interpolation of the full-weight SFT checkpoints with the base at α 0.85 / 0.70 / 0.50 (round 20): the
      accuracy guards stayed failed, and scienthoon got worse toward the base for arm (a) and better for arm (b);
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
  Since round 20, a temperature that is served for a calibration criterion or shipped is fitted on a pool of held-out
  *datasets* (the spec's `temperature`), never on a training corpus's own calibration/development rows, which are in
  distribution (round 19: T 0.955, breadth-v1 ECE 0.059 vs 0.0085 on the held-out pool). `kev.rounds validate` and
  `scripts/calibrate_checkpoint.py` refuse a fit set that shares data with the checkpoint's training
  (`kev.rounds.pool_conflicts`); `docs/autoresearch.md` section 3 has the rule and its checks.
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

### Round 19 result

**No candidate.** All three arms were read in full: neither selectable arm passes the registered rule, and the attribution arm fails too. Read-out `runs/r19-readout/round19.json` (`python -m kev.rounds readout experiments/rounds/r19.json`; reproduced exactly by `tests/test_rounds.py::test_readout_reproduces_round_19`). Every side served at the temperature fitted on its own development rows: the SFT arms at T 0.955 ((a), (b); `sft-v1` development, 6,146 questions) and 1.0 ((c)); Kev-27B at 1.38 (its decision-v7 development rows, the shipped value). Paired record-clustered bootstraps against Kev-27B, 2,000 resamples, micro; accuracy and confident errors in pp, Brier absolute, ECE as served against its bar (Kev-27B's + 0.01).

| criterion (panel, n) | (a) `27b-lr2e6` | (b) `27b-lr5e6` | (c) `27b-olddata` (attribution) |
|---|---|---|---|
| 1 breadth-v1 dev acc, lower > 0 (3,075) | +1.5 [+0.4, +2.5] pass | +0.6 [−0.6, +1.7] **fail** | −0.1 [−1.0, +0.7] **fail** |
| 1 Kev panel acc, lower ≥ −1 (3,731) | +8.7 [+7.3, +10.0] pass | +8.3 [+6.9, +9.7] pass | −0.8 [−1.6, +0.1] **fail** |
| 2 short acc, lower ≥ −2 (1,806) | −1.0 [−2.16, +0.2] **fail** | −0.1 [−1.3, +1.2] pass | −3.0 [−4.3, −1.9] **fail** |
| 2 short Brier, upper ≤ +0.01 | +0.014 [+0.002, +0.024] **fail** | +0.005 [−0.007, +0.016] **fail** | +0.037 [+0.026, +0.049] **fail** |
| 2 short confident errors, upper ≤ +1 | +1.6 [+0.8, +2.4] **fail** | +0.6 [−0.3, +1.3] **fail** | +1.5 [+0.7, +2.2] **fail** |
| 2 WANLI-v2 acc, lower ≥ −2 (1,002) | +0.0 [−2.10, +2.0] **fail** | −0.1 [−2.30, +2.0] **fail** | +0.8 [−1.2, +2.7] pass |
| 2 scienthoon acc, lower ≥ −2 (873) | −2.9 [−4.5, −1.4] **fail** | −3.6 [−5.3, −1.7] **fail** | −3.7 [−6.0, −1.5] **fail** |
| 2 pooled externals acc, lower ≥ −1.5 (2,108) | −1.2 [−2.4, +0.0] **fail** | −1.6 [−2.9, −0.3] **fail** | −1.2 [−2.6, +0.1] **fail** |
| 2 unknowable share ≤ 0.05 (transfer-v9) | 0.000 pass | 0.000 pass | 0.000 pass |
| 3 breadth ECE ≤ 0.022 (Kev-27B 0.012) | 0.059 **fail** | 0.065 **fail** | 0.060 **fail** |
| 3 Kev-panel ECE ≤ 0.032 (Kev-27B 0.022) | 0.038 **fail** | 0.037 **fail** | 0.064 **fail** |

Accuracies (arm / Kev-27B): breadth 0.760 / 0.751 / 0.744 vs 0.745; Kev panel 0.863 / 0.860 / 0.768 vs 0.776; short 0.858 / 0.867 / 0.837 vs 0.868; scienthoon 0.767 / 0.761 / 0.759 vs 0.796. Arm (a) passes both primaries and fails six guards and both calibration criteria; arm (b) fails the breadth primary as well.

**Attribution** (report only; the registered design). (b) vs (c), the data (same full-weight recipe, `sft-v1` vs Kev-27B's own training set; `runs/r19-readout/b-vs-c.json`, (c) served at its own T 1.0): Kev panel +9.1 [+7.8, +10.5], short states +2.9 [+1.7, +4.3] (Brier −0.032 [−0.044, −0.021]), breadth +0.7 [−0.5, +1.8], scienthoon +0.1 [−1.6, +1.8], WANLI-v2 −0.9 [−2.8, +0.9], pooled externals −0.4 [−1.6, +0.8]; Kev-panel ECE −0.027. (c) vs Kev-27B, full weights vs LoRA on the same data (the table's last column): no accuracy gain anywhere (breadth −0.1, Kev panel −0.8), short states −3.0 [−4.3, −1.9], scienthoon −3.7 [−6.0, −1.5], ECE +0.043 to +0.048. So the broad data carries the gains, and the scienthoon and short-state costs come with full weights, not with the data.

**Breadth index** (report only; `scripts/breadth_report.py`, chance-corrected, index 0-100 per area and overall; `runs/r19-breadth-report/report.md`; rows as saved, which does not change accuracy):

| area | Kev-27B | AutoJev | SFT (a) | SFT (b) | (c) | Jev |
|---|---|---|---|---|---|---|
| Knowledge & Reasoning | 33.8 | 32.7 | 34.4 | 34.5 | 32.5 | 37.7 |
| Language Understanding | 75.6 | 77.9 | 76.0 | 74.8 | 72.1 | 77.4 |
| Retrieval & Classification | 62.9 | 76.1 | 70.9 | 70.8 | 60.5 | 75.4 |
| Tools & Automation | 64.2 | 62.4 | 64.6 | 62.5 | 66.1 | 63.6 |
| Arts & Human Taste | 14.7 | 9.3 | 19.3 | 16.0 | 14.7 | 12.7 |
| **Overall** | **50.2** | **51.7** | **53.0** | **51.7** | **49.2** | **53.3** |

The SFT arms close most of the Retrieval & Classification gap to Jev and AutoJev (CLINC150 0.873 → 0.953 / 0.960, SGD 0.647 → 0.727 / 0.720); (c) does not (60.5).

**Calibration finding.** The registered temperature was fitted on `sft-v1` development rows, which are held-out *items* of the training sources: for this purpose they are in distribution. It came out at T 0.955 (sharpening), and every calibration criterion failed. A temperature fitted on held-out *datasets* instead fixes breadth ECE on the same checkpoint. Two exploratory computations, both made after seeing breadth-v1 development, so neither is a result: Jared's, arm (a) with T fitted on transfer-v4 development + SemIf + scienthoon + WANLI-v2 + TypeSafe rows: T 1.59, breadth ECE 0.017 (Brier 0.311, vs 0.319 at 0.955), Kev-panel ECE 0.028; a reconstruction with SemIf + scienthoon + WANLI-v2 + TypeSafe + transfer-v9 + transfer-r3 test rows: the same T 1.59 and numbers for (a), T 1.45 / breadth ECE 0.018 / Kev-panel ECE 0.018 for (b), T 1.59 / 0.022 / 0.030 for (c). Both pools overlap rule panels (transfer-v4 development or transfer-r3 test, and the externals), so round 20 registers a different pool that no rule panel reads.

**Deviations.**
- (i) In-trial scoring hit `kev.experiment`'s 384-token default context. All three trials trained to the end and saved their checkpoints ((c) 17:11Z, (a) 17:17Z, (b) 20:57Z), then failed on the first `sft-v1` calibration record (`ContextOverflow: state exceeds 384 tokens: 2641`): `score_trial` built its predictor with `kev.suite.CONTEXT` instead of the suite's context (7,552). Main fixed it in #136 (9648d37). The three checkpoints were re-scored (calibration, development and transfer only; no retraining) with `modal_app.py::resume` from branch `rounds/r19-score`: the registration commit 059d3b1, #136 cherry-picked as e11e770, and 5ea55aa (`run_resume` honours `--timeout` and gets host memory for a full-weight 27B checkpoint; orchestration only). Main could not be used because #135 (f2bb629) changed `kev/model.py`, an evaluator file, after training, and `resume_trial` refuses a changed evaluator. Each trial's `provenance.json` records `resumed_git_commit` 5ea55aa and the changed non-evaluator files (`kev/experiment.py`, `modal_app.py`).
- (ii) The rule reads of (a) and (c) were launched at 18:28:40Z (by another session, `kev.rounds launch-reads` from the registration checkout), before their re-scores finished (18:52Z and 18:47Z); (b)'s reads were launched at 21:05Z while its re-score ran (finished 22:03Z). They are the registered commands on the final checkpoints (`/runs/<trial>/checkpoint`), which re-scoring does not touch; only the development rows the temperatures are fitted on came from the re-score.
- (iii) A workspace GPU cap serialised the arms: (b) trained from 11:49Z, (a) continued and (c) started only at 16:04Z. (a) and (b) each used one automatic retry after the 8 h timeout, continuing from their last resume point ((b) from step 1352; its retried losses matched attempt 1 to 0.001 at steps 1360-1420, close but not bit-identical in the logged value).

**Evidence** (committed; 30 MB in all): the read-out and `b-vs-c.json`; every arm read's `report.json` + `rows.json` (`runs/r19-27b-{lr2e6,lr5e6,olddata}-<tag>`); the three trials' `result.json`, `provenance.json`, in-trial transfer rows and `calibration/temperature.json`; Kev-27B's transfer-r3 test read (`runs/r19-P27-r3test`) and locked transfer-v4 rows (`runs/locked/kev-27b-v2-ungated/transfer/rows.json`, the parent side of the locked stage); the breadth report. The trials' development rows carry `sft-v1` record ids and option keys, so under the data policy they are in the private dataset `jaredpalmer/kev-private-train` @ `6cc50f5d` under `runs/r19/`, with sha256s in `runs/r19-readout/private-rows.json` (`scripts/private_rows.py restore` puts them in place for an account with access). Checkpoints (48 GB each) stay on the `kev-runs` volume. Spend: Modal metered $2,432.57 at 22:20Z, +$833.31 over the registration baseline, workspace-wide.

## Round 20 (registered)

### Round 20 - post-hoc remedies on round 19's checkpoints: a held-out-datasets temperature and WiSE-FT interpolation with the base (registered with this commit, written before any round-20 interpolation or read)

**Why.** Round 19's SFT arms failed on two counts. Calibration: the registered temperature, fitted on held-out items of the training sources, is in distribution (T 0.955) and left breadth ECE at 0.059 / 0.065 against a bar of 0.022. Accuracy drift from the base: scienthoon −2.9 / −3.6 pp, and short-state Brier and confident errors. Full weights on the old data show the same drift (arm (c): scienthoon −3.7, short states −3.0), so it comes with full weights, not with the new data. This round tests two post-hoc remedies on the same trained checkpoints, with no training: (1) serving at a temperature fitted on held-out datasets; (2) WiSE-FT, interpolating each final backbone with the base's (Wortsman et al., 2022), which pulls every weight back toward the base while keeping part of what SFT learned. WiSE-FT is on our negative list for LoRA (`lora_scale`, overnight-1 and "Toward v0.2"); the new reason is that full-weight SFT moved every weight, and the gains (Kev panel +8.7, breadth +1.5) may survive a partial step back while the base's lost skills return.

**Candidates** (spec `experiments/rounds/r20.json`; parent Kev-27B, `r6-27b-v2/01-trial-1`, reads as in round 19):

| arm | checkpoint | weight on the SFT backbone | selectable |
|---|---|---|---|
| `27b-a` | round 19 arm (a) final, `runs/r19-27b-lr2e6/00-trial-0` | 1 | no (reference) |
| `27b-b` | round 19 arm (b) final, `runs/r19-27b-lr5e6/00-trial-0` | 1 | no (reference) |
| `27b-a-w85`, `27b-a-w70`, `27b-a-w50` | `/runs/r20-wise/27b-a-w{85,70,50}/checkpoint` | 0.85 / 0.70 / 0.50 | yes |
| `27b-b-w85`, `27b-b-w70`, `27b-b-w50` | `/runs/r20-wise/27b-b-w{85,70,50}/checkpoint` | 0.85 / 0.70 / 0.50 | yes |

The finals already fail accuracy guards that no temperature can change (argmax is temperature-invariant): scienthoon, WANLI-v2 and the pooled externals for both, plus short-state accuracy for (a) and breadth for (b). They are read only to show the registered temperature's effect on the calibration criteria and as the α = 1 end of each interpolation path; the six interpolations are the only selectable candidates (`"select": false` on the finals).

**Interpolation.** `scripts/interpolate_checkpoint.py` (`modal_app.py::interpolate`, CPU container, `kev.budget` `INTERPOLATE_*`): text backbone = α · SFT + (1 − α) · base in fp32, rounded once to the checkpoint's bf16; the base `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` is built by the same `DecisionModel` path training uses, so tensor names match, and any name or shape mismatch is refused before anything is written; SFT pointer head and tokenizer files kept; `head.pt` records `interpolation: {alpha, sft: {path, weights_sha256}, base}`; each checkpoint is written to `.partial` and renamed when complete, with `interpolation.json` beside it. Tests on a random two-layer Qwen3.5: α = 1 and α = 0 reproduce the SFT and the base exactly, 0.5 the fp32 midpoint; the result loads through `kev.checkpoint` as a full-weight checkpoint.

**Temperature (the registered calibration method).** Every candidate is served at the temperature fitted (`kev.metrics.served`, the objective every served temperature uses) on its own rows of a pool of held-out datasets that no round-19 or round-20 rule panel reads: the `transfer-r3` **calibration** partition (read `r3cal`, 580 records of one question; the spec's `sources` allowlist keeps its eight held-out public sources (composition_holdout, emotion, legacy_holdout, mmlu, paws, qnli, sciq, tweet_offensive) and drops its 66 unknowable records and 66 intact controls, which are generated policy items of the kind Kev trains on: 448 questions) plus only the **MMLU-Pro** rows of `transfer-v9` development (200 questions; the spec's `sources` allowlist drops transfer-v9's buried states, which come from the same public sources as transfer-v4 items, and its unknowable records and controls): **648 questions per candidate**. Records whose id is in the candidate's transfer-v4 development rows are removed (`exclude_reads`; none expected: by state hash the pool shares nothing with transfer-v4 development or transfer-r3 development or test). `kev.rounds` refuses a pool read inside any panel a temperature-dependent criterion reads; each candidate's fit (rows, count, T) is written into the read-out. Kev-27B is served at its shipped 1.38, as in round 19. Choosing held-out datasets as the pool follows round 19's exploratory finding, but this pool has not been looked at for any candidate, and the verdict rests on untouched test partitions. A released candidate ships the temperature fitted on the same pool (`scripts/calibrate_checkpoint.py --rows <r3cal rows>:composition_holdout,emotion,legacy_holdout,mmlu,paws,qnli,sciq,tweet_offensive --rows <v9 rows>:mmlu_pro --exclude_rows <transfer4 rows>`).

**Reads.** Per interpolated candidate: round 19's ten rule reads (breadth, hard, devtools, docs, semif, scienthoon, wanli2, typesafe, v9, r3test) + `transfer4` (transfer-v4 development: an interpolation has no in-trial transfer read; it stands in for "transfer" in the Kev and short panels via `transfer_read`) + `r3cal`: 72 reads. The two finals reuse their round-19 reads (same checkpoints, same suites; `runs/r19-27b-{lr2e6,lr5e6}-<tag>` and their in-trial transfer reads); only `r3cal` is new for them: 2 reads. 74 reads, one H200 each, read timeout 3,600 s (round 19's 27B full-weight reads landed about 9 minutes after launch; round 19 registered 14,400 s).

**Rule** (round 19's, unchanged; against Kev-27B, paired record-clustered bootstraps, 2,000 resamples, seed 0, micro):
1. primaries: breadth-v1 development accuracy lower bound > 0; pooled Kev development panel (transfer-v4 dev, hard-v1, devtools-v1, documents-v1) accuracy lower ≥ −1 pp;
2. guards: short state (transfer-v4 dev + transfer-r3 test) accuracy lower ≥ −2 pp, Brier upper ≤ +0.01, confident errors upper ≤ +1 pp; WANLI-v2 and scienthoon lower ≥ −2 pp each; pooled externals (SemIf, scienthoon, WANLI-v2, TypeSafe) lower ≥ −1.5 pp; unknowable share on transfer-v9 ≤ 0.05;
3. calibration: breadth-v1 ECE ≤ Kev-27B's + 0.01 and Kev-panel ECE ≤ Kev-27B's + 0.01;
4. candidate: the passing selectable arm with the largest breadth + Kev-panel accuracy gain (`drop_ids` as in round 19). The read-out says how many of the six passed.

**Confirmation** (the candidate only, each read once, after the rule): `tests`: breadth-v1 test accuracy lower bound > 0 vs Kev-27B; pooled hard-v1 + devtools-v1 + documents-v1 test accuracy lower ≥ −1 pp; documents-v2 reported; `locked`: locked transfer-v4 accuracy ≥ 0.886 and served Brier ≤ 0.165 (absolute bars; round 19's spec encoded them relative to Kev-27B's served 0.896 / 0.160). Report-only steps outside the spec, as in round 19: Jev and AutoJev read once on the same breadth-v1 test items (`kev.jev`, AutoJev's own server; `scripts/breadth_report.py`), and before any release the bf16 serving check on main's path (`uv run modal run modal_app.py::serving --run <checkpoint> --gpu H200 --name serving-27b-r20 --flags=--isolation`: max |Δp| ≤ 0.03, ≤ 1 flip in 280).

**Run steps** (after this PR is merged): (1) `KEV_APP_NAME=kev-sft uv run modal run --detach modal_app.py::interpolate --sft /runs/r19-27b-lr2e6/00-trial-0/checkpoint --prefix 27b-a`, then the same with `r19-27b-lr5e6` and `--prefix 27b-b`, 60 s apart; (2) `uv run python -m kev.rounds launch-reads experiments/rounds/r20.json` once both have finished, then `readout`; (3) confirmation as `docs/autoresearch.md` says, and before the locked stage `KEV_GPU=H200 KEV_APP_NAME=kev-sft uv run modal deploy modal_app.py` from the merged main, because `run_locked_test` now reads a checkpoint without a trial (as `-ungated`).

**Budget.** Admission bounds: reads 74 × $6.27 (H200 at `kev.budget`'s trial resources × 1 h) = $463.62, interpolation 2 × $4.21 (8 CPU, 128 GiB, 3 h) = $8.41: **$472.04**; expected ~$110 (a read ~15 min at ~$5.66/h, an interpolation ~1 h of CPU). Confirmation, candidate only: tests 10 reads (candidate + parent) $62.65, locked $25.06, serving check $6.27: **~$94**; Jev on breadth-v1 test through the AI Gateway (~$0.07 at the development read's rate, cap $3). Baseline: Modal metered **$2,434.01 at 2026-09-25T22:53Z**.

### Round 20 result

**No candidate.** 0 of the 6 selectable interpolations pass the registered rule, and neither reference final passes. No confirmation read was made. Read-out: `runs/r20-readout/round20.json` (`python -m kev.rounds readout experiments/rounds/r20.json`; reproduced exactly by `tests/test_rounds.py::test_readout_reproduces_round_20`). Every arm was served at the temperature fitted on its own 648 pool questions: transfer-r3 calibration, eight sources, 448 questions, plus transfer-v9 MMLU-Pro, 200 questions; none was excluded as a transfer-v4 duplicate. Kev-27B was served at its shipped 1.38. Deltas are paired record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro): accuracy and confident errors in pp, Brier absolute, ECE as served against its bar.

| criterion (panel, n) | 27b-a (final) | 27b-a-w85 | 27b-a-w70 | 27b-a-w50 | 27b-b (final) | 27b-b-w85 | 27b-b-w70 | 27b-b-w50 |
|---|---|---|---|---|---|---|---|---|
| T (pool of 648) | 1.414 | 1.414 | 1.447 | 1.447 | 1.382 | 1.350 | 1.350 | 1.350 |
| 1 breadth acc, lower > 0 (3,075) | +1.5 [+0.4, +2.5] | +1.4 [+0.4, +2.5] | +1.7 [+0.7, +2.6] | +1.5 [+0.5, +2.5] | +0.6 [−0.6, +1.7] **fail** | +1.1 [−0.1, +2.3] **fail** | +1.4 [+0.3, +2.5] | +1.7 [+0.7, +2.8] |
| 1 Kev panel acc, lower ≥ −1 (3,731) | +8.7 [+7.3, +10.0] | +8.7 [+7.3, +10.1] | +8.8 [+7.5, +10.1] | +7.4 [+6.2, +8.5] | +8.3 [+6.9, +9.7] | +8.6 [+7.2, +10.0] | +8.8 [+7.4, +10.1] | +8.2 [+7.0, +9.5] |
| 2 short acc, lower ≥ −2 (1,806) | −1.0 [−2.2, +0.2] **fail** | −0.7 [−1.8, +0.4] | −0.6 [−1.7, +0.5] | −0.6 [−1.7, +0.5] | −0.1 [−1.3, +1.2] | −0.2 [−1.3, +1.0] | −0.3 [−1.4, +0.8] | −0.2 [−1.2, +0.9] |
| 2 short Brier, upper ≤ +0.01 | +0.005 [−0.006, +0.014] **fail** | +0.004 [−0.007, +0.013] **fail** | +0.000 [−0.010, +0.009] | −0.001 [−0.011, +0.009] | −0.001 [−0.012, +0.009] | −0.002 [−0.013, +0.008] | −0.004 [−0.015, +0.004] | −0.005 [−0.016, +0.004] |
| 2 short confident errors, upper ≤ +1 | −0.7 [−1.4, −0.1] | −0.7 [−1.4, −0.1] | −0.8 [−1.6, −0.2] | −0.7 [−1.4, −0.1] | −0.6 [−1.4, +0.1] | −0.7 [−1.4, +0.0] | −0.9 [−1.7, −0.2] | −0.8 [−1.6, −0.2] |
| 2 WANLI-v2, lower ≥ −2 (1,002) | +0.0 [−2.1, +2.0] **fail** | +0.0 [−2.1, +1.9] **fail** | +0.6 [−1.3, +2.4] | +0.0 [−1.9, +2.0] | −0.1 [−2.3, +2.0] **fail** | +0.1 [−2.1, +2.2] **fail** | +0.7 [−1.4, +2.8] | +1.1 [−1.0, +3.1] |
| 2 scienthoon, lower ≥ −2 (873) | −2.9 [−4.5, −1.4] **fail** | −3.2 [−4.8, −1.7] **fail** | −3.3 [−5.0, −1.7] **fail** | −3.9 [−5.6, −2.3] **fail** | −3.6 [−5.3, −1.7] **fail** | −3.8 [−5.7, −1.9] **fail** | −3.0 [−4.8, −1.3] **fail** | −1.8 [−3.4, −0.2] **fail** |
| 2 pooled externals, lower ≥ −1.5 (2,108) | −1.2 [−2.4, +0.0] **fail** | −1.3 [−2.6, −0.2] **fail** | −1.2 [−2.4, −0.1] **fail** | −1.9 [−3.1, −0.7] **fail** | −1.6 [−2.9, −0.3] **fail** | −1.7 [−3.0, −0.3] **fail** | −1.1 [−2.4, +0.1] **fail** | −0.5 [−1.7, +0.7] **fail** |
| 2 unknowable share ≤ 0.05 (transfer-v9) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 breadth ECE ≤ 0.0218 (Kev-27B 0.0118) | 0.0085 | 0.0087 | 0.0131 | 0.0237 **fail** | 0.0173 | 0.0171 | 0.0196 | 0.0244 **fail** |
| 3 Kev-panel ECE ≤ 0.0316 (Kev-27B 0.0216) | 0.0193 | 0.0213 | 0.0146 | 0.0249 | 0.0158 | 0.0148 | 0.0198 | 0.0148 |

Accuracies (arm, with Kev-27B's in brackets). Breadth: 0.760 / 0.759 / 0.762 / 0.760 / 0.751 / 0.756 / 0.759 / 0.762 (0.745). Kev panel: 0.863 / 0.863 / 0.864 / 0.850 / 0.860 / 0.862 / 0.864 / 0.858 (0.776). Scienthoon: 0.767 / 0.764 / 0.763 / 0.757 / 0.761 / 0.758 / 0.766 / 0.778 (0.796). Scienthoon and the pooled externals fail for every arm. `27b-a-w70` and `27b-b-w70` fail only those two. `27b-b-w50` also fails breadth ECE.

**Calibration: the registered method worked.** Fitted on the held-out-datasets pool, the temperatures came out at 1.35-1.45, where round 19 fitted 0.955 on `sft-v1` development rows. Every final and every interpolation down to α 0.70 passes both ECE criteria. Arm (a)'s final: breadth ECE 0.0085 vs Kev-27B 0.0118, Kev-panel 0.0193 vs 0.0216. At round 19's T 0.955 the same rows gave 0.059 / 0.038 (arm (b): 0.065 / 0.037; `runs/r20-readout/served.json`, report only). The short-state Brier guard also passes now for every arm except (a) and (a)-w85: (b)'s upper bound went from +0.016 to +0.009. At α 0.50 breadth ECE rises above the bar for both paths (0.0237, 0.0244).

**Interpolation: moving toward the base did not restore scienthoon.** For arm (a) it made scienthoon worse (−2.9 → −3.2 → −3.3 → −3.9 at α 1 → 0.85 → 0.70 → 0.50). For arm (b) it helped, but not enough: `27b-b-w50` came closest, with scienthoon −1.83 [−3.44, −0.23] (bar −2), pooled externals −0.47 [−1.68, +0.72] (bar −1.5) and breadth ECE 0.0244 > 0.0218. The Kev-panel gain survives the step back (+7.4 to +8.8 pp everywhere), and so does the breadth gain (+1.1 to +1.7). WANLI-v2 moves to +0.6 / +0.7 at α 0.70 and +1.1 for `27b-b-w50`. Short states stay within −0.7 to −0.2 pp for every interpolation.

**Breadth index** (report only; `scripts/breadth_report.py` on each arm's breadth rows served at its pool T, Kev-27B at 1.38; `runs/r20-breadth-report/report.md`): chance-corrected index, with the paired difference from Kev-27B.

| | index [95 %] | vs Kev-27B |
|---|---|---|
| 27b-b-w50 | 53.7 [51.0, 56.4] | +3.4 [+0.6, +6.2] |
| 27b-a-w70 | 53.5 [50.7, 56.4] | +3.3 [+0.7, +5.8] |
| 27b-a-w50 | 53.4 [50.4, 56.2] | +3.2 [+0.5, +5.7] |
| Jev | 53.3 [50.7, 56.3] | +3.1 [−0.1, +6.2] |
| 27b-b-w70 | 53.3 [50.5, 56.2] | +3.0 [−0.0, +6.0] |
| 27b-b-w85 | 53.1 [50.2, 56.1] | +2.8 [−0.1, +5.8] |
| 27b-a (final) | 53.0 [50.3, 55.9] | +2.8 [+0.2, +5.3] |
| 27b-a-w85 | 52.9 [50.0, 55.7] | +2.7 [+0.0, +5.2] |
| AutoJev | 51.7 [49.2, 54.7] | +1.5 [−1.2, +4.6] |
| 27b-b (final) | 51.7 [48.9, 54.6] | +1.5 [−1.4, +4.3] |
| Kev-27B | 50.2 [47.4, 53.3] | - |

Most of the interpolations' extra index is Retrieval & Classification: SGD 0.647 (Kev-27B) → 0.807 (a-w50) / 0.793 (b-w50), against Jev's 0.793.

**Scienthoon analysis** (report only, committed rows, no new reads; `runs/r20-scienthoon/analysis.md`, numbers in `drift.json`, `scripts/scienthoon_drift.py`):
- **The loss is one question type.** On `angry` ("The customer sounds angry.") the round-19/20 arms lose 5.8-13.1 pp. `queue` gains one question, and `priority` (whose label follows a rule absent from the text) moves −3.1 to +1.7. Without `angry` the arms sit at −1.4 to +1.0 pp.
- **What goes wrong.** The arms call a calm ticket about a real problem angry. Examples: "The box for order #8223 was crushed and the item inside is broken." and "17일 전에 반품했는데 환불이 안 됐어요." They make 28-54 such false positives; Kev-27B makes 9 and Jev 8. 53 of the 55 flipped questions have calm text and gold "not angry", so the gold labels are sound. The other 15 `angry` labels contradict their text (label noise every model misses), which puts the ceiling at 0.948.
- **Kev-27B is a favourable draw.** Six LoRA checkpoints were trained from the base on Kev's data: B1 v2 s1 and s2, B1 trials A and B, and the two 2-epoch seeds. They score 0.740-0.796 (mean 0.765, sd 0.021) with 9-59 false positives, and Kev-27B is the top one. Against Kev-27B, **none of the other five passes the scienthoon guard or the pooled-externals guard**. B1 v2 seed 1, Kev-27B's own recipe, is at −2.7 [−4.2, −1.4] / −1.5 [−2.6, −0.5]. The full-weight arms score 0.757-0.778, at or above the family mean.
- **The pooled externals' loss is the same loss.** Without the scienthoon `angry` questions, the pooled delta is −0.4 [−1.8, +0.8] for (a), 0.0 for (b), +0.7 for (c) and +0.5 [−0.8, +1.8] for b-w50. The short-state loss is a different, diffuse one: (a) is −18 questions spread over qnli, composition, emotion and tweet_offensive.
- **Not the data, and not full weights as such.** Arm (c), full weights on Kev-27B's own data, is the worst (54 false positives). AutoJev, full weights on other data, makes 0. No scienthoon read of the untrained base exists; the interpolations point both ways.
- **Recommendation for round 21: (d) plus a cheap (a).** Accept that this is not a regression of the recipe, and register the scienthoon and pooled-externals guards against something other than Kev-27B's single best draw. Two options: gate against the LoRA family (for example, point estimate ≥ its mean, and a lower bound ≥ −2 pp against its median checkpoint, B1 trial A), or score scienthoon without `priority` and report calm-text `angry` false positives as a diagnostic. This is Jared's call, made at registration, not retroactively. Add a small open-weight tone minimal-pair family to the extended data (the same problem written calm or angry, other domains than retail tickets, English and Korean; screened against scienthoon, breadth-v1 and transfer-r3's `emotion` / `tweet_offensive`; never a scienthoon paraphrase). Read snapshots as a report, never selected on scienthoon: (c) is free, since lr 2e-6 made fewer false positives than 5e-6 (28 vs 45). Do not build (b), a KL anchor toward Kev-27B's served answers, now. It is feasible: `--anchor` takes any `{record: {qid: {key: p}}}` file, so it needs a rows → targets converter and one Kev-27B read of the replay pool. But Kev-27B's own seed 1 shows its training data does not fix this boundary, so distilling on that data would not either.

**Deviations.**
- (i) The two interpolation jobs ran under the Modal app `kev-research`, not the registered `kev-sft`. The code, volume and command were the same, and the outputs are unaffected: each checkpoint's `weights_sha256` and inputs are in `runs/r20-wise/<arm>/interpolation.json`, 850 tensors each, 149-230 s per α.
- (ii) One read client (`27b-b-w50`) lost DNS at 23:56Z. Its detached Modal app kept running, and its 12 reads were pulled from `/bench` by hand at 23:57Z. They are the registered commands' outputs.
- Timeline: interpolations finished by 23:39Z, the 72 interpolation reads were launched at 23:39:46Z (the finals' two `r3cal` reads before them), all 74 had landed by 23:59:30Z, and the read-out was made at 00:02Z.

**Evidence** (committed): the read-out; `runs/r20-readout/served.json` (per-panel accuracy / ECE / Brier / NLL at each side's served T, and the finals at round 19's T; report only); every read's `report.json` + `rows.json` (`runs/r20-27b-<arm>-<tag>`, public-suite rows only, no `sft-v1` ids); `runs/r20-wise/*/interpolation.json`; the breadth report; the scienthoon analysis, plus the two scienthoon reads it uses that were not in git (`runs/jev-scienthoon-v1/rows.json`, converted by `scripts/freeze_scienthoon.py`, and round 17 arm (a)'s `runs/r17-27b-r10k-lr2e5-scienthoon`). The finals' development rows stay in the private dataset, as in round 19; the read-out's reproduction test restores them with `scripts/private_rows.py` and skips without access. The six interpolated checkpoints (51 GB each) stay on the `kev-runs` volume at `/runs/r20-wise/<arm>/checkpoint`. Spend: Modal metered $2,488.88 at 2026-09-26T00:02Z, +$54.87 over the registration baseline (workspace-wide), within the expected ~$110 and the $472.04 admission bound.

## Next

Goals and open questions, not registered rounds; each becomes a spec and a PLAN section before it runs.

0. **Round 21: retrain full weights, with long context and extended data.** Retraining is allowed (rounds 19-20 showed
   that post-hoc remedies do not move the accuracy guards). What is decided so far:
   - Context: 64k tokens is the target, 32k the fallback and 16k the last resort (a fit probe decides before registration).
   - Data: `sft-v1` extended into a new version.
   - Snapshots: kept (0.25 / 0.5 / 0.75 of the steps, #145), read as a report.
   - Calibration: round 20's held-out-datasets pool method.
   - The scienthoon remedy follows the round-20 analysis (`runs/r20-scienthoon/analysis.md`). Re-register the scienthoon
     and pooled-externals guards against something other than Kev-27B's single best draw (Jared's call, at registration):
     either against the LoRA family, or scienthoon scored without the text-unknowable `priority`. Add a small open-weight
     tone minimal-pair family to the data (calm vs angry wording of the same problem, other domains, English and Korean,
     no scienthoon paraphrase). Report calm-text `angry` false positives. The analysis argues against a KL anchor toward
     Kev-27B's answers, because its advantage is a seed draw that its training data does not fix.
   - Optional, about $6: one zero-shot read of the untrained base on scienthoon settles whether the base sits nearer
     Kev-27B or the arms.
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
kev.rounds readout` reproduces rounds 5-20, see `tests/test_rounds.py`); `A:` is the archive tag. Verdicts are the registered
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
| Round 19 | 09-25 | full-weight SFT of Qwen3.8-27B on `sft-v1` (lr 2e-6, 5e-6) + full weights on Kev-27B's own data (attribution) | no candidate: both SFT arms fail scienthoon, WANLI-v2, pooled externals, short-state Brier / confident errors and both ECE criteria ((b) also breadth); the data carries the gains, full weights the costs | `r19.json`; `runs/r19-readout`, `runs/r19-breadth-report`; "Round 19 result" |
| Round 20 | 09-25/26 | post-hoc on round 19's finals, no training: held-out-datasets temperature + WiSE-FT interpolation (α 0.85 / 0.70 / 0.50) | no candidate (0 of 6): every arm fails scienthoon and the pooled externals; the registered temperature passes both ECE criteria down to α 0.70 (arm (a) breadth ECE 0.0085 vs 0.0118); toward the base scienthoon worsens for (a); the scienthoon analysis traces the cost to calm complaints read as "angry" on a guard whose reference is the best of six LoRA draws | `r20.json`; `runs/r20-readout`, `runs/r20-breadth-report`, `runs/r20-scienthoon`; "Round 20 result" |
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
