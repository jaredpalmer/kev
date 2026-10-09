# Research plan

This file says where Kev stands, what we have learned, the rules every experiment follows, and what comes next. It is
short on purpose. The full research record (every registration, read, verdict and incident from the Qwen3 prototype
through night 3, rounds 4-18) is frozen at the git tag `research-archive-2026-09-24`; pointers below written as
`A:<path>` mean `git show research-archive-2026-09-24:<path>` (for example `A:PLAN.md`, the 1,322-line record, or
`A:PLAN_27b.md`, the Kev-27B plan). How to run an unattended research session is in
[`docs/autoresearch.md`](docs/autoresearch.md).

## Where we stand (2026-09-30)

### The released family

All four are public in the [Kev collection](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd).
Kev-4B round 10 and Kev-0.8B round 15 were released and Kev-27B made public on 2026-09-24 (docs in PR #106). Kev-27B v2
(round 23's `27b-k-w85`, full weights) replaced it on 2026-09-30; v1 (the LoRA adapter) stays at `jaredpalmer/kev-27b@v1-lora`
("Released: Kev-27B v2"). Kev-9B v2 (round 27's `9b-r18a`) replaced Kev-9B the same day; v1 stays at `jaredpalmer/kev-9b@v1`
("Released: Kev-9B v2"). These four were released together as Kev 1.0 on 2026-10-01 (below; "Released: Kev 1.0").

| model | checkpoint | T | locked transfer-v4: acc / Brier | other confirmation reads (once each) | JevBench public: all / hard (hard ECE) |
|---|---|---|---|---|---|
| Kev-27B v2 | round 23 `27b-k-w85` (0.85 round-22 full-weight SFT + 0.15 v1, SFT head), full bf16 weights `d27af6ab…`, `Qwen/Qwen3.8-27B@1d4bf0f2` (post-trained), Hub `28be62e9` (weights) | 1.32 (pool) | **0.889 / 0.154** | vs v1: breadth-v1 test +1.2 [+0.3, +2.2]; tasksource-heldout-v1 test +5.3 [+3.7, +6.8]; pooled hard / devtools / documents test +8.9 [+7.5, +10.3]; locked −0.8 [−2.0, +0.5]; CUAD test −1.6 [−3.0, −0.3], ECE 0.053 vs 0.007 | not read |
| Kev-27B v1 (tag `v1-lora`) | `r6-27b-v2/01-trial-1` (B1 v2 seed 2), LoRA, Hub `01b81998` | 1.38 | 0.896 / 0.160 | decision-v7 locked 0.870; transfer-r6 test 0.863 vs Kev-9B 0.842 (+2.1 pp [+0.35, +3.8]); longstate-v3 0.833 vs 0.556 | **0.866 / 0.721** (0.128) |
| Kev-9B v2 | round 27 `9b-r18a` (`r18-9b/00-trial-0`: v1 + one epoch on documents-v1 + hard-v1 + devtools-v1, replay 10,000), Hub `b5d8c18e` | 2.19 (pool) | **0.852 / 0.199** | vs v1: hard-v1 + devtools-v1 test +18.7 [+16.7, +20.8]; documents-v1 test +7.1 [+4.7, +9.2]; locked +0.0 [−1.7, +1.8] | not read |
| Kev-9B v1 (tag `v1`) | `night2-9b-du/00-trial-0` (v7 + dates/unknowable delta), Hub `2629c06a` | 2.30 | 0.852 / 0.224 | - | 0.762 / 0.568 (0.19) |
| Kev-4B | `r10-skills/00-trial-0` (night2 + documents-v1 (round 8) + hard-v1 & devtools-v1 (round 10)), Hub `139fdd94` | 2.41 | 0.838 / 0.224 | hard-v1 test 0.540 → 0.803; devtools-v1 test 0.623 → 0.756 | 0.758 / 0.541 (0.112) |
| Kev-0.8B | `r15-08b/00-trial-0` (night2 + documents-v1, hard-v1, devtools-v1 in one delta), Hub `9a45d25e` | 2.35 | 0.697 / 0.397 | documents-v1 test 0.608 → 0.851; hard-v1 test 0.396 → 0.665; devtools-v1 test 0.472 → 0.637; documents-v2 (private) 0.848 | 0.636 / 0.360 (0.181) |
| Jev (reference) | hosted | - | transfer-v4 dev 0.857 (never locked) | - | 0.866 / 0.741 (0.06; their board, tiers include held-out items) |

T is the temperature in `head.pt`, fitted on the trial's decision-v7 development rows (Kev-27B v2 and Kev-9B v2: their rounds'
registered pool of held-out datasets, 648 questions; round 28 tested the same pool for Kev-4B and Kev-0.8B and kept the shipped
T); locked Brier is served at that T.
Evidence: `runs/release/kev-{27b-r23,27b-v2,9b-r27,4b-r10,08b-r15}.json` (`kev-27b-v2` = v1's B1 v2 records), `experiments/releases/*.json`,
`runs/release/kev-27b-r23-published.json`, `runs/jevbench-public/`, the model cards. Previous weights are Hub tags (`kev-27b@v1-lora`,
`kev-4b@r8-documents-release`, `kev-4b@night2-du-release`, `kev-0.8b@night2-du-release`, `@v7-base`).

### Kev 1.0 baseline (released 2026-10-01)

**Kev 1.0 is the baseline Kev 2 is measured against.** A Kev 2 candidate at a given size is compared with the Kev 1.0
checkpoint of that size on identical items, at each side's shipped temperature, and the 1.0 reads below are its parent reads.
Kev 1.0 trains nothing: it versions the four released checkpoints with their formal cards (PR #206). Load a parent as
`jaredpalmer/kev-<size>@v1.0`; the GitHub release `kev-1.0` has the three adapter tarballs and Kev-27B is on the Hub only
("Released: Kev 1.0", `runs/release/kev-1.0.json`).

| size | checkpoint (trial) | Hub revision (weights) | T (fit) | locked transfer-v4 acc / Brier | breadth-v1 test index (all 14) | transfer-v4 dev | validated context |
|---|---|---|---|---|---|---|---|
| Kev-0.8B | `r15-08b/00-trial-0` | `9a45d25e` | 2.35 (in-distribution; kept by round 28) | 0.697 / 0.397 | 23.3 [21.2, 25.9] | 0.648 | 8,192 (16k lower bound −8.5 pp, `runs/r28-readout/context.json`) |
| Kev-4B | `r10-skills/00-trial-0` | `139fdd94` | 2.41 (in-distribution; kept by round 28) | 0.838 / 0.224 | 38.0 [35.5, 41.3] | 0.817 | 8,192 (16k lower bound −3.4 pp, `runs/r28-readout/context.json`) |
| Kev-9B v2 | `r18-9b/00-trial-0` (round 27 `9b-r18a`) | `b5d8c18e` | 2.19 (held-out pool) | 0.852 / 0.199 | 41.0 [38.8, 43.9] | 0.820 | 8,192 (16k lower bound −3.7 pp, `runs/r28-readout/context.json`) |
| Kev-27B v2 | round 23 `27b-k-w85` | `28be62e9` (`v1.0` = `af0e6d55`: card commits since) | 1.32 (held-out pool) | 0.889 / 0.154 | 52.3 [49.2, 55.4] | 0.851 | 65,536 (64k lower bound −2.4 pp, `runs/r28-readout/context.json`) |
| Jev (reference) | hosted | – | – | not read | 54.0 [51.2, 57.0] | 0.857 | – |

- Index: `runs/fam-breadth-test-report/report.json` (2026-09-30 family reads; Kev-27B v2's row is round 23's
  `r23c-27b-cand-breadthtest`) and, for Jev, `runs/r23-breadth-report/report.json`. Locked reads: `runs/locked/kev-{08b-r15,
  4b-r10,9b-r27,27b-r23}-ungated`. Everything else on each size's card (`docs/model-cards/`), traced in `docs/claims.json`.
- Held-out-dataset panels a Kev 2 rule should read against these parents: breadth-v1 dev / test (audited: without `routerbench`,
  `cfcolor`, `humicroedit`, `chessbench`), tasksource-heldout-v1 dev (without the seven private families; 4B 0.677, 0.8B 0.515
  at shipped T, `runs/r28-readout/round28.json`; 9B v2 0.706 at the pool T = its shipped T, `runs/r29-readout/round29.json`; 27B test 0.795), transfer-v4 dev, and the
  locked transfer-v4 test, read once per candidate. The breadth-v1 test and tasksource-heldout-v1 test partitions have been
  read for the 27B (round 23/24) and breadth-v1 test for every size (family reads): a Kev 2 confirmation on them is a second
  read and weaker evidence than a fresh panel.
- Validated context (report only, round 28's rule): the largest nominal bucket from 16k whose paired CUAD accuracy
  difference from the 8k bucket has a 95 % lower bound ≥ −3 pp (and every bucket below it too); 8,192 if 16k fails. Values
  from phase B's longdoc-v1 development reads (Kev-27B: round 23's `r23-27b-k-w85-longdoc`), `runs/r28-readout/context.json`:
  8,192 for Kev-0.8B, Kev-4B and Kev-9B v2 (each fails at 16k: lower bounds −8.5, −3.4, −3.7 pp; at 32k all three are
  measurably below 8k) and 65,536 for Kev-27B v2 (64k lower bound −2.4 pp).
- Known gaps a Kev 2 should be judged on: in-distribution gains (hard / devtools / documents splits are trained on), CUAD
  calibration at 27B (ECE 0.053 vs v1's 0.007), date arithmetic below 27B (`deadline` 0.35 / 0.65 / 0.725, Jev 0.95), Kev-0.8B's
  When2Call regression (test 0.133), Kev-4B / 0.8B temperatures fitted in distribution, and no Mac measurement of Kev-9B or
  Kev-27B.

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
  **Round 21** (full-weight SFT from the base on `sft-v2-r21`, 32k states, none pairs gated at 8k, two learning rates,
  snapshots read as candidates) **failed at startup**: both arms ran out of GPU memory at their 62nd step, before any
  snapshot, so nothing was read ("Round 21 result"). **Round 22** repeated its science and rule with a per-pass memory
  ceiling in the trainer and a training set sized to round 21's measured rate (`sft-v2-r22`, 145,840 records), one arm
  (lr 2e-6), and is read out: **no candidate**, 0 of 4 (snapshots at 0.25 / 0.5 / 0.75 and the final). The primaries pass
  and grow with training (final: breadth +1.3 [+0.3, +2.3], tasksource-heldout +3.8 [+2.3, +5.3], Kev panel +7.8), the
  out-of-domain suites are far better, and scienthoon (−5.5 [−7.8, −3.2]), the pooled externals (−2.2), short-state
  accuracy and CUAD calibration (ECE ~0.10 vs 0.06-0.07 at every length) fail ("Round 22 result"). Modal gave the trial
  2 of its 3 attempts; the final was finished by a manual continuation (PR #163 fixes the retries).
- **Round 24** (retrospective selection: the 2026-09-27 audit's suite verdicts as one rule over all 12 full-weight 27B
  checkpoints, no training) named `27b-r22-final` (round 22's final), which is **NOT CONFIRMED** ("Round 24 confirmation").
  It passed the tests stage on untouched test partitions: breadth-v1 +1.5 [+0.5, +2.5], tasksource-heldout-v1 +5.3
  [+3.7, +7.0], pooled hard/devtools/documents-v1 +8.6 [+7.1, +10.0], documents-v2 +3.8. Its breadth index on test is
  53.7 [50.5, 56.7], against Jev 54.0, Kev-27B 50.2 and AutoJev 50.0. It **failed the locked stage**: locked transfer-v4
  0.8841 (580 of 656) against the registered bar of 0.886 (582 needed; Kev-27B 0.8963). Served Brier 0.155 passed, and
  so did the bf16 serving check. On longdoc CUAD test it is worse: −1.8 [−3.2, −0.5], ECE 0.055 vs 0.007 (report only).
  The bar stands as registered and nothing is released.
- **Round 23** is re-registered (2026-09-28), not launched: no training, six blends of round 22's final toward Kev-27B's
  own weights (α 0.85 / 0.70 / 0.50, SFT head or blended heads), read under round 24's audited rule with round 24's
  confirmation ("Round 23 (registered)"). Its first registration (round 22's rule, scienthoon included) was never
  launched and is superseded.
- **Round 25** (continued full-weight SFT **from Kev-27B**, its LoRA merged into full weights, on `sft-v2-r25`: 45,515
  records with b1v2 replay at 30 %, no long-document families, states ≤ 16k; lr 1e-6 and 2e-6; 8 candidates) is read out:
  **no candidate** (0 of 8; "Round 25 result"). Both studies trained in one attempt (1.5 h each). Every candidate fails
  breadth-v1 ECE (0.023-0.035 against a bar of 0.0176). `27b-lr1e6` and `27b-lr1e6-s75` fail only that, passing 11 of 12
  criteria. The lr 2e-6 arms also fail short-state accuracy and the breadth primary. Replay held short states at lr 1e-6
  (−0.3 to −0.8 pp) and CUAD accuracy held on development, but CUAD ECE still rose (0.087-0.103 against 0.063). Against
  round 23's confirmed `27b-k-w85` (report only), no arm is ahead on any gated panel. Nothing was confirmed or released.
- **Round 26** (round 25's lr 1e-6 arm, same merged Kev-27B init and plan, with one change: tasksource-v1 doubled on
  `sft-v2-r26`, 58,515 records; 4 candidates) is read out: **no candidate** (0 of 4; "Round 26 result"). The study trained
  in one attempt (1.62 h). The final, `27b-lr1e6`, passes 11 of 12 criteria and fails only breadth-v1 ECE (0.0233 against a
  bar of 0.0176), like round 25's final (0.0235). The snapshots also fail the breadth primary (lower bounds −0.16 to −0.04
  pp), and s50 fails short-state accuracy. Doubling tasksource-v1 did not move breadth: against round 25's lr 1e-6 final
  (report only, same fraction of the run) breadth +0.2 [−0.3, +0.7], tasksource-heldout +0.5 [−0.5, +1.5], breadth ECE
  −0.000. Against `27b-k-w85` (report only) no arm is ahead on any gated panel. Nothing was confirmed or released.
- **Round 17** (27B skills delta from Kev-27B with replay 10,000, study `r17-27b`, spec `experiments/rounds/r17.json`).
  Arm (a), lr 2e-5, is read out (`runs/r17-readout/round17.json` in the research checkout, not yet committed anywhere) and
  is **not a candidate**: primary +12.1 [+10.4, +13.9] (hard-v1 dev 0.733 → 0.895, +16.2 [+13.5, +19.0]; devtools-v1 dev
  0.702 → 0.783, +8.0 [+5.7, +10.4]), short state +0.3 [−0.9, +1.5], pooled externals −0.1 [−1.0, +0.7], hard-set ECE
  0.047 → 0.018, but documents −1.5 [−2.8, −0.4] fails the ≥ −2 pp lower-bound guard and scienthoon is below its bound.
  Arm (b), lr 1e-5, is still training or reading; **result pending**. Its reads land in the research checkout
  (`research/overnight-r6`); read it out there with `uv run python -m kev.rounds readout experiments/rounds/r17.json --root
  <research checkout>`. Main's harness refuses to launch reads for a recorded round, so a confirmation, if arm (b) passes,
  would be launched from that checkout.
- **scienthoon removed** (2026-09-27): `evals/external/scienthoon-v1` is no longer a Kev eval; past verdicts stand, and from
  round 23 there is no scienthoon read or guard ("scienthoon removed" below). Round 24's audited rule, which round 23 now
  follows, gates no pooled externals: SemIf, WANLI-v2 and TypeSafe are reported.
- **Kev-27B v2 released** (2026-09-30, Jared's approval): round 23's `27b-k-w85`, T 1.32, is `jaredpalmer/kev-27b` main
  (weights commit `28be62e9`, card `0d7f9b49`); v1 is tag `v1-lora` ("Released: Kev-27B v2").
- **documents-v1 and hard-v1 train partitions published** (2026-09-30): both are in `jaredpalmer/kev-suites` at
  `cc4bac80` (`SUITES_REVISION`), so Kev-4B's and Kev-0.8B's training data is fetchable (Next, item 3).
- **Rounds 28 and 29** (Kev 1.0 prep, no training). Phase A of their sweep (68 short-state development reads, PR #203) is
  done. **Round 28 is read out: no candidate at either size** (`runs/r28-readout/round28.json`), so Kev-4B keeps T 2.41 and
  Kev-0.8B T 2.35. `4b-r10` (pool T 2.297 [2.047, 2.520]) fails both primary conditions: Brier −0.0001 [−0.0005, +0.0003], ECE
  0.0252 against 0.0240. `08b-r15` (pool T 2.520 [2.194, 2.828]) passes both (ECE 0.0375 against 0.0484, Brier −0.0020 [−0.0025,
  −0.0014]) and fails the hard-v1, devtools-v1 and documents-v1 ECE guards (+0.011, +0.007, +0.020 against a tolerance of 0.005).
  The long-state memory wall that blocked phase B is fixed (PR #202). Phase B's longdoc-v1 reads of the released checkpoints
  gave every size its validated context length (report only, `runs/r28-readout/context.json`): 8,192 for Kev-0.8B, 4B and
  9B v2, 65,536 for Kev-27B v2. **Round 29: no candidate; Kev-9B v2 stands** (`runs/r29-readout/round29.json`; every arm fails the tasksource-heldout-v1 primary, v2 also breadth-v1; ten arms' longdoc reads stopped for the budget, which cannot change the verdict). See "Round 28 (registered)" and "Round 29 (registered)".
- Decisions waiting on Jared: submit Kev-27B (and the new 4B / 0.8B) to the Decision Index.
- Spend: Modal metered $4,558.69 at 2026-09-29T13:01Z after round 26's last read (+$104.88 over its launch reading of
  $4,453.81; the 11:17Z reading was $4,650.37, +$196.56, and was revised down by 12:56Z; night ceiling ≈ $5,980, hard stop
  $5,700). $4,461.42 at 2026-09-29T06:00Z after round 25's last read (+$494.5 over its registration reading
  of $3,966.94; night ceiling ≈ $5,980, hard stop $5,600). Earlier: ~$3,790 (+~$80 of metering lag) at round 23's re-registration reading, 2026-09-28, after round
  24's confirmation (workspace-wide; night ceiling $5,000 metered). $3,571.88 at 2026-09-27T13:32Z (round 23's first
  registration reading; round 22 +$903.36 over its registration reading, workspace-wide). $2,668.52 at 2026-09-26T13:42Z (round 22's registration reading: round 21's failed trials and parent
  reads ~$135, round 22's ceiling probe ~$12; night ceiling $5,000 metered). $2,522.65 at 2026-09-26T06:49Z (round 21's registration baseline). Before that, $2,488.88 at 2026-09-26T00:02Z (round 20: +$54.87 over its registration baseline of $2,434.01,
  workspace-wide); $2,434.01 at 2026-09-25T22:53Z was +$834.75 over round 19's registration baseline of $1,599.26 (its
  training, re-scoring and reads, plus other apps of the workspace); earlier, $1,377.01 at 2026-09-24T12:11Z plus round 17's admission bound ($100.24), and night 3 used $292 of a
  $1,000 authorization. AI Gateway $0.13 (Jev reference reads only; $0.064 of it round 24's breadth-v1 test read). Modal sponsors the project ($5,000 credits, more on request).

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
   and gained nothing ("Round 19 result"); the SFT arms kept that scienthoon cost (−2.9, −3.6). Round 22 (full weights from
   the base on the extended `sft-v2-r22`, 32k states) grew the gains with training (final: breadth +1.3, held-out
   tasksource datasets +3.8, Kev panel +7.8, out-of-domain suites +1.5 to +3.9 with much lower ECE) and kept the costs on
   scienthoon (−5.5), the pooled externals (−2.2), short states (−1.2) and CUAD calibration. Its tone pairs removed the
   calm-called-angry errors (0 vs Kev-27B's 9) but it now misses angry tickets (28 vs 2) and loses `priority` (0.419 vs
   0.529) ("Round 22 result").
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
- **A suite found unsound as a gate is removed, not patched, and past verdicts stand.** Its directory and scripts leave the
  repo and it is listed in `kev.suite.REMOVED_SUITES` with the reason and the last round that read it. Rounds up to that one
  keep their registered rules and committed rows (`kev.rounds validate` lists the read as archived; read-outs reproduce from
  the rows). `load_split` refuses the suite, and `validate` / `launch` refuse any later round that names it.
  `evals/external/scienthoon-v1` was removed on 2026-09-27 (last read: round 22; "scienthoon removed" below). From round 23
  there is no scienthoon guard. There is no pooled-externals guard either: SemIf, WANLI-v2 and TypeSafe were reported, not
  gated. The 2026-09-27 audit found that panel unsound without scienthoon (81 % WANLI, split-half r 0.08; round 24),
  and round 23 was re-registered on round 24's rule. `evals/external/wanli-v2`, `wanli-v1` and `typesafe-v1` were removed on
  2026-09-30 (last read: round 26, round 5 for wanli-v1; "WANLI and TypeSafe removed" below). From round 27 SemIf is the only
  external read, report only.
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

**Scienthoon analysis** (report only, committed rows, no new reads; `runs/r20-scienthoon/analysis.md`, numbers in `drift.json`, `scripts/scienthoon_drift.py`, removed with the suite on 2026-09-27 and in git history at `9c41005`):
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

## Round 21 (registered)

### Round 21 - full-weight SFT of Qwen3.8-27B on sft-v2: long states and extended data (registered with this spec's commit, written before any round-21 training or read)

**Why.** Round 19's full-weight SFT on `sft-v1` gained where the broad data reached (Kev panel +8.7 pp, breadth-v1 +1.5 pp
for arm (a)) and round 20 fixed its calibration with a temperature fitted on held-out datasets (breadth ECE 0.0085 vs
Kev-27B 0.0118). What stayed failed were scienthoon and the pooled externals. Round 20's analysis traced that cost to one
borderline judgement (calm complaints read as "angry") on a guard whose reference, Kev-27B, is the best of six LoRA draws
of its recipe (`runs/r20-scienthoon/analysis.md`). Round 21 retrains from the base on an extended corpus: tone minimal
pairs for the "angry" boundary, a licence-filtered public multi-task component with its own held-out datasets, long
states up to the trained cap, out-of-domain, prompt-injection recognition and agent-trace, PII, grounding records. It re-bases the
scienthoon and pooled-externals guards on the round-20 evidence, decided here before any read. (Revised before any launch
after Jared's review of PR #152: 32k states accepted; none pairs kept at 0.25 but only on states of at most 8,192 tokens,
with their siblings counted in the micro-batch cost, a trainer knob in PR #153, which this round depends on; the per-source
cap on sft-v1's public sources tightened from 2,000 to 1,200; `sft-v2-r21`'s calibration and development limited to
states of at most 8,192 tokens. The earlier versions of `sft-v2-r21`, kev-private-train @ `97d545ff` and @ `ef38326e`,
were never used.)

**Data** (private; policy in "Data policy for the SFT work"; manifests only in this repo, partitions in
`jaredpalmer/kev-private-train` / `jaredpalmer/kev-private-evals`; built by kev-sft `assemble-v2` @ `1b5c7f6`,
`assemble/build_v2.py`).

`evals/sft-v2` (mirror `97d545ff`) is the union of these frozen components (each file hash-checked against its
component manifest; the manifest records every component's manifest path, commit, sha256 and mirror revision):

| component | kind | train | calibration | development |
|---|---|---|---|---|
| sft-v1 (`119c1e7d`) | round 19's corpus (public 24 sources, Kev components, open-weight synthetic) | 194,247 | 8,468 | 3,483 |
| tasksource-v1 (`f572d8f6`) | public multi-task collection, native labels (119 families, private list) | 58,191 | 3,481 | 2,088 |
| longify (`0bfa4c69`) | 8k-64k states built from sft-v1 train, exact labels | 8,000 | - | - |
| longdoc (`dce09c29`) | code-assembled long documents 8k-64k, code labels | 9,060 | 484 | 197 |
| ood (`3550e29b`) | open-weight out-of-domain | 7,312 | 388 | 156 |
| tone (`0a56df5d`) | open-weight tone minimal pairs (calm / frustrated / angry) | 7,544 | 372 | 159 |
| injection (`9160a875`) | indirect prompt injection recognition (defensive) | 2,761 | 133 | 55 |
| agents (`45ff503c`) | agent-session analytics over code-generated traces | 5,174 | 221 | 93 |
| guardrails-pii (`0ad2dbc6`) | PII classification (code-inserted fake PII) | 4,152 | 219 | 77 |
| guardrails-grounding (`0ad2dbc6`) | grounding / claim support | 5,662 | 258 | 162 |

Not merged: longify's two monitoring shards (built from held-out sft-v1 *train* records, which sft-v2 trains on, so
they are not held-out items of sft-v2), longdoc's `programmatic_split` and the `adjudicated` pools of ood, injection, agents, guardrails-pii and guardrails-grounding
(not in their components' default mixes), and agents' `ood_eval_soft` (soft-target evaluation questions: screened against,
not a suite, since `kev.benchmark` scores hard labels).

Selection: every record of every partition materialises, its soft targets name option keys and sum to one, and it is
admitted strictly in `kev.model.training_context(MAX_TRAIN_STATE)` (64k states) under the Kev-27B tokenizer. The whole
union was re-screened with the kev-sft screen rule (exact normalised state or content string, word 8-gram Jaccard > 0.2,
smaller-side containment ≥ 0.5) against every evaluation partition of every frozen Kev suite (102 partitions:
breadth-v1, longdoc-v1, documents-v2, tasksource-heldout-v1, transfer-r3 including its calibration partition, transfer-v9
and every older panel), JevBench public and the new eval-only partitions below (76,549 reference items). The screen
dropped 6,581 records: sft-v1 4,446, guardrails-pii 816, tasksource-v1 626, injection 466, agents 135, ood 88, guardrails-grounding 4. Most are template overlaps rather than item text. In sft-v1 (4,446): hard-v1's
own generators across its train / dev / test templates (2,794: 1,316 on question wording alone, 1,478 on scenario text with
new numbers and names), round-4 buried records and decision-v7 items that older suites (decision-v1 / v2 / v5) put in
calibration, and night-2 unknowable templates against transfer-r3 / transfer-v9's unknowable records. In the synthetic and
tasksource components, 1,956 are exact matches on question wording alone, a template shared with five or fewer of the
component's own held-out items, so the screen's template filter (a string in more than five reference items) does not
catch it: guardrails-pii 816 of its hits (15 % of its train), injection 466 (14 %), tasksource-v1 529, ood 88, agents 53,
grounding 4 (each hit checked by comparing the matched strings; counts only, no text). The rule is applied as written
anyway: it is the registered screen, and dropping them costs 2.0 % of the corpus. A later version could count a wording
as template text when more than five training records also carry it. None of the
hits is on a temperature-pool source: the only transfer-r3 / transfer-v9 matches are their unknowable records and intact
controls (night-2 and round-4 generators), which the pool's `sources` allowlist already drops. No state repeats across
components; 6,286 repeats inside sft-v1 train (as frozen) and 539 inside tone (its soft pool shares states with
its hard pool) are kept as their components froze them. Record changes: a `_meta.variant` other than clean (longdoc's
abstention twins, tone's calm / frustrated / angry versions, the view labels of injection, agents and guardrails; 31,313
records) moves to `_meta.twin`, because
`kev.benchmark` scores only clean rows; tasksource-v1 records carry source `tasksource` (the family moves to
`_meta.tasksource_source` in the private data), so no public file lists tasksource's families (Jared's decision).

| partition | records | questions | state tokens | row tokens (prefix shared) | soft-target questions |
|---|---|---|---|---|---|
| train | 302,103 | 586,690 | 609.0M | 647.8M | 19,947 |
| calibration | 14,024 | 28,018 | 20.4M | 22.2M | 297 |
| development | 6,470 | 12,655 | 8.7M | 9.5M | 179 |

State tokens of the train records (the `<state>` token plus the rendered state, Kev-27B tokenizer):

| ≤256 | 257-512 | 513-1k | 1k-2k | 2k-4k | 4k-8k | 8k-16k | 16k-32k | 32k-64k |
|---|---|---|---|---|---|---|---|---|
| 163,741 | 30,612 | 36,473 | 35,323 | 10,697 | 4,376 | 9,359 | 8,110 | 3,412 |

`trainable_sources`: 68 names (sft-v1's 59, plus `tasksource`, `longify`, `synthetic-v2/longdoc`, `synthetic-v2/ood`,
`synthetic-v2/tone`, `synthetic-v2-guardrails/injection`, `synthetic-v2-guardrails/grounding`, `synthetic-v2-guardrails/pii`, `synthetic-v2/agent_sessions`). Calibration and development are held-out items of the
training components, so they are in distribution: in-trial screening only, never a served or shipped temperature.

`evals/sft-v2-r21` (the training suite of this round; kev-private-train @ `c00d1c95`): sft-v2's records, same order,
train restricted to states of at most 32,768 tokens and at most 1,200 records of each sft-v1 public source (the records
with the smallest sha256 of the seed and the record digest), calibration and development to states of at most 8,192
tokens: train 233,265 (496,146 questions, 434.2M state tokens, 468.0M row tokens with the state shared), calibration
13,385 (25,362 questions; 554 longer records left), development 6,201 (11,569 questions; 231 left). The screening cap is
there for in-trial scoring cost: calibration and development are in-distribution screening partitions (held-out items of
the training components, read only for the trial's in-trial temperature and development score), no rule criterion reads
them (every candidate reads transfer-v4 through its own read, and the served temperature comes from the held-out-datasets
pool), and their 785 records over 8k tokens took about half of a trial's in-trial scoring time on one GPU. 3,412 train records over
the state cap leave (longify 1,600, longdoc 1,443, others 369), and 65,426 leave the 23 capped sources (multiwoz, massive,
helpsteer3 5,580 → 1,200; helpsteer2, snli, gsm8k, hotpotqa, nq, esci 4,650 and ledgar 4,640 → 1,200; openbookqa, medmcqa,
math_qa, siqa, cosmos_qa, winogrande, casehold, csqa, aqua_rat, qasc 3,720 → 1,200; arc 2,991, quartz 2,340, strategyqa
1,215 → 1,200). By component, train: sft-v1 128,821, tasksource-v1 58,190, longdoc 7,617, tone 7,544, ood 7,312, longify
6,400, guardrails-grounding 5,655, agents 4,875, guardrails-pii 4,152, injection 2,699. State tokens of its train records:
≤256 114,262 · 257-512 25,490 · 513-1k 32,383 · 1k-2k 29,841 · 2k-4k 9,563 · 4k-8k 4,257 · 8k-16k 9,359 · 16k-32k 8,110.
Why 32k and the caps: "Budget and memory" below.

Eval-only suites (private mirror `d6498d4c`, development partition only, report-only; each is its components' frozen
records with the same variant rule, so ood-v2 is byte for byte its component file; every sft-v2 record was screened
against them): `evals/ood-v2` (1,686 records, 4,988 questions: the ood component's held-out domains and Portuguese),
`evals/agents-ood-v1` (373 records, 2,084 questions: agents/ood_eval.jsonl), `evals/guardrails-ood-v1` (1,263 records, 4,949 questions: guardrails-pii/ood.jsonl + guardrails-grounding/ood.jsonl + injection/ood.jsonl). `evals/tasksource-heldout-v1` (development 2,386 records /
2,788 questions, locked test 2,395 / 2,833; 24 whole dataset families held out of tasksource-v1, kev-private-evals
@ `e4f2c71d`) is public as hashes and counts only; its family list is in the private kev-sft manifest (`tasksource-v1`
@ `3ffc81b`), and because rows carry each record's source, its rows and per-source reports stay in the private dataset
(like sft-v1's development rows, `scripts/private_rows.py`). `kev.suite.load_split` fetches and hash-checks it.

**Arms** (spec `experiments/rounds/r21.json`, plans `experiments/round21/`): full-weight SFT fresh from
`Qwen/Qwen3.8-27B` @ `1d4bf0f2`, 8×H200 per trial (FSDP2, fp32 masters, prefix-shared rows), one epoch of
`evals/sft-v2-r21`, batch 8 × accum 2 × 8 ranks = 128 records per step (`--length_sort 1`), bf16 autocast, OneCycle
(10 % warm-up), head lr 1e-4, `--max_state 32768`, `p_none_pair 0.25` with `none_pair_max_state 8192` (PR #153), seed 0;
questions per long record as the components cap them (longify at most 6 per record, 4 above 32k):
- (a) `27b-lr2e6`: lr 2e-6 (round 19's arm (a));
- (b) `27b-lr1e6`: lr 1e-6.

Each trial writes snapshots at 0.25 / 0.5 / 0.75 of its 1,823 optimizer steps (steps 456 / 912 / 1368;
`/runs/r21-27b-<lr>/00-trial-0/snapshots/step-000NNNN/checkpoint`, ~51 GB each, kept on the volume; no Hub mirror). The
candidates are every arm's three snapshots and its final checkpoint: 8, each selectable (`27b-<lr>-s25/s50/s75`). Every
candidate's "transfer" rows come from its own `transfer4` read (the spec's round-level `transfer_read`), finals included,
so the rule needs nothing from a trial's in-trial scoring: that scoring (calibration, development, in-trial transfer) is
screening only. Snapshots also protect against a run that exhausts its attempts: each snapshot is committed to the volume
when written and stays a candidate whatever happens to the run after it, and the final checkpoint is committed as soon
as it is complete (`modal_app.VolumeWatcher`), before in-trial scoring starts. The harness supports this without a
change: snapshot arms are checkpoint arms, launched with `launch-reads --arms`; a final arm whose trial ended without
`result.json` (attempts exhausted during scoring) is read the same way from its committed checkpoint; an arm with no
checkpoint at all is reported incomplete by the read-out and cannot be selected, while the others are ranked as usual.

**One departure from the planned recipe (32k states) and one trainer change (gated none pairs), decided before any
launch.** Both come from `assemble/epoch_estimate.py` (kev-sft), which deals the frozen train partition's token shapes
through kev.train's own micro-batch code (`microbatch_plan` / `balanced_runs` / `pass_tokens`, with PR #153's sibling
costs) and times each pass per GPU; a step lasts, per micro-batch slot, as long as the slowest rank's pass (FSDP2 waits
for every rank at every layer). Two pass-time models, because the short-state measurements disagree on what a none pair
costs: **A**, pass seconds = a + 0.478·P + 0.00282·n·S² (P padded tokens in thousands, S the longest state, n states; the
long-state probe `runs/sft-probe/lc-27b-8xh200`), a = 1.55 s fitted so the sft-v1 records alone reproduce round 19 arm
(a)'s measured 0.1475 s per record (none pairs 0.25); **B**, the same for passes over 8k tokens (a = 1.0, the probe), and
6.91 + 0.15·P for shorter passes, fitted to both round 19 (0.1475 s, pairs) and the no-pair corpus probe (0.13 s,
`runs/sft-probe/sft2-*`). Memory per GPU ≈ 59.3 + 1.17·P GiB (the same probe: 78 / 96 / 115 / 134 GiB at 16k / 32k / 48k
/ 64k); a pass over 64.7k padded tokens counts as over the limit.

| training suite (train records), none pairs | one epoch, model A | model B | peak pass | passes over the limit |
|---|---|---|---|---|
| sft-v2 at 64k (302,103), 0.25, today's trainer (as planned) | 168,100 s (46.7 h) | - | ~292 GiB | 2,529 |
| sft-v2 at 64k (302,103), none | 95,440 s (26.5 h) | - | ~137 GiB | 8 |
| sft-v2-r21 (233,265), 0.25, today's trainer | 107,482 s (29.9 h) | 101,592 s (28.2 h) | ~211 GiB | 1,416 |
| sft-v2 at 32k (298,691), 0.25 gated at 8k | 73,149 s (20.3 h) | 69,941 s (19.4 h) | ~126 GiB | 0 |
| public cap 2,000 (250,880), 0.25 gated at 8k | 68,230 s (19.0 h) | 65,203 s (18.1 h) | ~126 GiB | 0 |
| public cap 1,500 (239,880), 0.25 gated at 8k | 67,595 s (18.8 h) | 64,604 s (17.9 h) | ~130 GiB | 0 |
| **sft-v2-r21, cap 1,200 (233,265), 0.25 gated at 8k (registered)** | **67,224 s (18.7 h)** | **64,268 s (17.9 h)** | ~129 GiB | 0 |
| sft-v2-r21, 0.25 gated at 2k (not registered) | 63,415 s (17.6 h) | 60,764 s (16.9 h) | ~122 GiB | 0 |
| sft-v2-r21, no none pairs (not registered) | 55,491 s (15.4 h) | 53,691 s (14.9 h) | ~121 GiB | 0 |

- 64k states do not fit (Jared accepted the 32k fallback, PLAN "Next" item 0's own order): at 64k the long records set
  every step's length (a 64k record keeps its GPU busy ~44 s and the other ranks wait for it at every layer), and
  removing every sft-v1 public record still projects to 79,981 s (22.2 h) of training before in-trial scoring. The 64k
  records stay in `evals/sft-v2`. Reading at 64k is unaffected (longdoc-v1's 32k and 64k buckets are read and gated;
  Kev-27B, trained at 7.5k states, reads them without a resolvable drop, `runs/longdoc-v1-report/README.md`). What would
  make 64k trainable is a trainer change: length-bucketed steps, so all eight ranks run long passes together.
- None pairs stay (they teach "none of the above", which the unknowable guard and abstention calibration lean on), but
  today's trainer cannot run them at these lengths: a pair adds two copies of the whole record, state included, to the
  same pass, and the balancer does not see them, so any record over ~21k tokens with an eligible Choice becomes a pass
  over the GPU. PR #153 adds `--none_pair_max_state`: pairs only on states of at most N tokens (8,192 here, where every
  sft-v1 short record pairs as in round 19; 92.5 % of sft-v2-r21's train records are at most 8k), drawn from each record's
  own stream and counted in the micro-batch cost, so no pass outgrows its cost (0 passes over the limit, peak ~129 GiB).
- The per-source cap moves little: long records set the critical path, so each 500 fewer records per public source saves
  about 0.1-0.2 h. Jared's rule was to tighten 2,000 → 1,500 → 1,200 until the round fits ~21 h expected / 23 h worst
  per arm including scoring; at 1,200 it does not (below), and going further buys minutes (cap 700: −0.2 h). The
  registration stops at 1,200 and says so; the levers that would close the gap are listed under the budget.

**Budget and memory** (per arm, `evals/sft-v2-r21`, none pairs gated at 8k): training 67,224 s (A) / 64,268 s (B)
projected, × 1.03 for hourly resume points, plus ~10 min per attempt for the gate's state-token count (PR #153; ~6 min
on an M-series core) and ~10 min of model load: **19.9 h (A) / 19.1 h (B)** of training container time. In-trial scoring
(calibration + development + transfer-v4: 20,350 records, all of at most 8k tokens, at round 19's measured 0.292 s each)
adds **1.65 h**: 21.6 h (A) / 20.7 h (B) of container time. Two timeouts each lose up to an hour of training back to the
last resume point plus ~15 min of restart (expected 1.5 h in all, worst 2.5 h):

| per arm | training done (checkpoint + snapshots committed) | training + in-trial scoring | Jared's bar (incl. scoring) |
|---|---|---|---|
| expected | 21.4 h (A) / 20.6 h (B) | **23.1 h (A) / 22.2 h (B)** | ~21 h |
| worst | 22.4 h (A) / 21.6 h (B) | **24.1 h (A) / 23.2 h (B)** | ~23 h |

Training, every snapshot and the final checkpoint fit inside the three 8 h attempts (24 h) with 1.6-3.4 h to spare; with
the screening cap the in-trial scoring fits too in the expected case (0.9-1.8 h to spare) and in model B's worst case,
and runs 0.1 h past the last attempt in model A's worst case, where the third attempt would time out in the last minutes
of scoring. That would cost the round nothing: every candidate's rule reads are separate reads (above), and the final
checkpoint is committed before scoring. Against Jared's bar the expected case is still 1.2-2.1 h over and the worst case
0.2-1.1 h over; the remaining lever, not registered (Jared kept the 8k gate), is gating at 2,048 tokens (−1.0 h, B).
Finishing an interrupted in-trial scoring later with `modal_app.py::resume` would cost ~1.7 h × $41.17 ≈ $70 per arm,
outside the study bound, and is not planned. Peak memory projected ~129 GiB of 140 (round 19 measured 94.6 GB at 7.5k
states).

Timeout 28,800 s, `FULL_FT_RETRIES` 2: admission bound **$987.99 per study** (H200:8 at $41.17/h × 8 h × 3), $1,975.98 for
both; expected **~$914 (B) - $951 (A) per arm** (22.2 / 23.1 h × $41.17/h). If the
workspace GPU cap serialises the two trials, as in round 19, the round takes about twice as long. Reads (H200, per-suite
timeouts of `modal_app.READ_TIMEOUTS`, no size override): per candidate $72-75 of admission bound (longdoc-v1 10,800 s,
documents-v1 5,400 s, transfer-v9 3,600 s, the rest 1,800 s), 8 candidates **$595**; expected ~$30 each. Parent reads
Kev-27B still lacks (tsheld, ood, agentsood, guardood): $13 bound. Confirmation, candidate only: tests stage (candidate
+ parent, 14 reads) $100, locked $25, serving check ~$6.

Spend against the night's **$5,000 metered** ceiling, baseline **$2,522.65 at 2026-09-26T06:49Z**: at launch $2,522.65 +
$1,975.98 of study bounds = $4,498.63. Expected at the end of the rule stage ≈ $2,522.65 + $1,828-1,902 (studies) +
~$250 (candidate reads) + ~$10 (parent reads) ≈ **$4,610-4,685**; with the confirmation stage (~$60 expected, $131 bound)
≈ $4,670-4,745.
Under the spend rule (`docs/autoresearch.md` section 2) the reads cannot all be admitted at once once both studies have
spent their bounds ($4,499 + $595 > $5,000): launch one arm's four candidates, then the other's after the first batch
lands. The expected reserve is ~$255-330, so anything beyond the registered reads needs a fresh reading.

**Temperature (MUST, `docs/autoresearch.md` section 3).** Round 20's pool, unchanged: every candidate is served at the
temperature fitted (`kev.metrics.served`) on its own rows of transfer-r3 **calibration** (read `r3cal`, allowlist
composition_holdout, emotion, legacy_holdout, mmlu, paws, qnli, sciq, tweet_offensive: 448 questions) plus transfer-v9
**development MMLU-Pro** (read `v9`, allowlist mmlu_pro: 200 questions), minus any record in its transfer-v4 development
rows (`exclude_reads: transfer`). `kev.rounds validate` checks the pool against the arms' training (the study suite
`evals/sft-v2-r21`, which names `evals/sft-v2`, which names `evals/sft-v1` and its components; the snapshot arms name
`trained_on: [evals/sft-v2-r21]`): no pooled suite is training data, no pooled source is a trained source, and no training
corpus's calibration/development partition is pooled. That check is nominal (source names); the semantic side is the
screen above (no sft-v2 record matches a pool source's item) and tasksource-v1's catalog, which excludes MMLU (and so
MMLU-Pro), emotion / go_emotions, tweet_eval, QNLI and its SQuAD parent, PAWS, SciQ and SciTail by name before anything
else. Kev-27B is served at its shipped 1.38 (its trial's decision-v7 development rows, the same fit). A released
candidate ships the temperature `scripts/calibrate_checkpoint.py` fits on the same pool rows (`--rows <r3cal
rows>:<the eight sources> --rows <v9 rows>:mmlu_pro --exclude_rows <transfer rows>`), never with `--allow-in-distribution`.

**Reads per candidate** (tag: suite, partition): breadth (breadth-v1 dev), tsheld (tasksource-heldout-v1 dev), the Kev
panel's hard / devtools / docs (hard-v1, devtools-v1, documents-v1 dev) with transfer-v4 dev (every candidate's `transfer4`
read), semif, scienthoon, wanli2, typesafe, v9 (transfer-v9 dev: unknowable share and the pool's
MMLU-Pro), r3test (transfer-r3 test, the short panel), longdoc (longdoc-v1 dev), ood (ood-v2), agentsood (agents-ood-v1), guardood (guardrails-ood-v1), r3cal (the
pool). Kev-27B (`r6-27b-v2/01-trial-1`, Hub `01b81998`) has every one of these reads committed except tsheld, ood, agentsood and guardood, which `kev.rounds launch-reads experiments/rounds/r21.json --parents` makes (`runs/r21-P27-<tag>`); its
longdoc-v1 read is `runs/longdoc-v1-kev-27b` (the unpinned Hub id, served raw logits in its rows).

**Rule** (against Kev-27B, paired record-clustered bootstraps, `kev.rounds.paired`: 2,000 resamples, seed 0, micro; every
candidate at its pool temperature, Kev-27B at 1.38):
1. primaries: breadth-v1 dev accuracy lower bound > 0; tasksource-heldout-v1 dev accuracy lower bound > 0; Kev panel
   (transfer-v4 dev, hard-v1, devtools-v1, documents-v1 dev) accuracy lower bound ≥ −1 pp;
2. guards: short state (transfer-v4 dev + transfer-r3 test) accuracy lower ≥ −2 pp, Brier upper ≤ +0.01, confident errors
   upper ≤ +1 pp; WANLI-v2 lower ≥ −2 pp; **scienthoon lower ≥ −4 pp; pooled externals (SemIf, scienthoon, WANLI-v2,
   TypeSafe) lower ≥ −2.5 pp**; longdoc-v1 CUAD part: accuracy lower ≥ −2 pp over all lengths, and at 16k+ states
   (`acc_16k_plus`) candidate − parent ≥ −2 pp; unknowable share on transfer-v9 ≤ 0.05;
3. calibration (pool temperature): breadth ECE ≤ Kev-27B's + 0.01; Kev-panel ECE ≤ Kev-27B's + 0.01; tasksource-heldout
   ECE ≤ Kev-27B's + 0.01; longdoc-v1 CUAD ECE at 16k+ states (`ece_16k_plus`, `by_length`) ≤ Kev-27B's + 0.01;
4. candidate: the passing arm with the largest breadth + tasksource-heldout + Kev-panel accuracy gain (sum of the three
   paired deltas). The read-out says how many of the eight passed.

The scienthoon and pooled-externals bars are re-based, a pre-registered change made with this commit, before any
round-21 read, on round 20's evidence (`runs/r20-scienthoon/analysis.md`, `drift.json`): Kev-27B is the best of six LoRA
checkpoints trained from the base on Kev's data (scienthoon 0.740-0.796, mean 0.765, sd 2.1 pp), and none of its five
siblings would pass the old bars against it, its own recipe's second seed included (B1 v2 seed 1: scienthoon −2.7 [−4.2,
−1.4], pooled externals −1.5 [−2.6, −0.5]). The old −2 / −1.5 pp bars measured a seed draw of the reference, not a
regression. Against the new −4 / −2.5 pp bars (Jared's call, recorded here) one of the five siblings passes both (B1
trial A, the family's median checkpoint: −3.9 / −1.9), B1 v2 seed 1 misses both by 0.2 / 0.1 pp, the two-epoch seed 1
misses scienthoon by 0.01 pp, and the two worst draws (−7.7 / −3.7, −7.7 / −3.8) still fail; round 19's three finals
would still fail scienthoon (lower bounds −4.5, −5.3, −6.0 pp; `drift.json` `lora_siblings_under_the_guards`, "Round 19
result"). Reported with the rule, not gating: the `angry`
question's false positives per candidate (calm-text tickets called angry, scienthoon-v1, counted with
`scripts/scienthoon_drift.py`'s text-class rule (removed 2026-09-27); Kev-27B 9, Jev 8, round 19/20 arms 28-54), and the breadth index
(`scripts/breadth_report.py`).

`16k_plus` is `kev.metrics.calibration_by_length`'s tail of states of at least 16,384 tokens (Kev-27B tokenizer):
longdoc-v1's nominal 32k and 64k buckets; its nominal 16k bucket holds 13.1k-15.2k-token states and counts as 8k-16k. The
engine gives a by-length bucket a value but no paired interval, so the registered long-accuracy guard is the pair above:
the paired lower bound over the whole CUAD part (all five buckets) and the point difference at 16k+. The generated half of
longdoc-v1 (at ceiling for every system read so far), ood-v2, agents-ood-v1 and guardrails-ood-v1 are reported with accuracy and ECE, not gated.

**Confirmation** (the candidate only, each read once, after the rule; `docs/autoresearch.md` section 4):
- `tests`: breadth-v1 test accuracy lower bound > 0 vs Kev-27B; tasksource-heldout-v1 test accuracy lower bound > 0;
  pooled hard-v1 + devtools-v1 + documents-v1 test accuracy lower ≥ −1 pp; documents-v2 and longdoc-v1 test (CUAD and
  generated, by length) reported. Outside the spec, once each and reported: Jev and AutoJev on the same breadth-v1 test
  items (`kev.jev`, AutoJev's own server, `scripts/breadth_report.py`), and Jev on longdoc-v1 test (`kev.jev
  --count-refusals`; it refuses the 64k bucket).
- `locked`: locked transfer-v4 accuracy ≥ 0.886 (Kev-27B 0.896 − 1 pp) and served Brier ≤ 0.165, at the pool temperature
  (`runs/locked/kev-27b-r21-ungated`, the checkpoint read without a trial like round 20's).
- Before any release: the bf16 serving check on main's path at 8k, 32k and 64k states (`modal_app.py::serving
  --flags=--isolation` for short states, and a bf16 fused read of longdoc-v1 development against the fp32 read with
  `scripts/longdoc_report.py --parity`, per bucket): max |Δp| ≤ 0.03 and ≤ 1 flip in 280 questions. The release
  temperature is the pool fit above, via `scripts/calibrate_checkpoint.py` with `--rows` pool rows.

**Run steps** (after PR #153 and this PR are merged): (1) fetch `evals/sft-v2-r21` and the new eval suites into the
checkout (`load_split`), then `KEV_GPU=H200 KEV_APP_NAME=kev-sft uv run modal deploy modal_app.py` (the image copies
`evals/`; do not leave `evals/sft-v2/*.jsonl` in the checkout, ~3 GB the image does not need); (2) read the metered cost,
then `uv run python -m kev.rounds launch experiments/rounds/r21.json` and `watch`; in the first minutes check the log's
`none pairs: N of 233265 records` line and count optimizer steps per minute against the projection (1,823 steps, ~35-37 s
per step on average, longer on steps that carry long records); (3) as snapshots land, and for any final whose trial ends
without `result.json`, `launch-reads experiments/rounds/r21.json --arms <arms>` one arm's candidates at a time (spend
rule above); `launch-reads --parents` for Kev-27B's four new reads; (4) read-out, then confirmation as written. What may be
committed: public-suite rows and reports, as in rounds 19-20. The trials' calibration / development rows (sft-v2 record
ids) and every tasksource-heldout-v1 read's rows and report (its per-source breakdown names the private families) go to
the private dataset with `scripts/private_rows.py`.the private dataset with `scripts/private_rows.py`.

### Round 21 result

**Failed at startup, in both arms; no candidate, no read.** Both trials (`r21-27b-lr2e6`, `r21-27b-lr1e6`, launched
2026-09-26 ~11:05Z) ran out of GPU memory on rank 1 during the backward pass of their 62nd optimizer step (the logs' last
line is step 60; the crash came 75-86 s later): "Tried to allocate 3.05 GiB ... 137.69 GiB in use of 139.80 GiB", in the
checkpoint recompute of an MLP projection of the state pass (`kev/shared_prefix.py:129`, from `kev/train.py:567`). Same
seed and data order, so the same micro-batch in both arms. `failed.json` was written and the trials were not retried. No
snapshot was reached (the first was at step 456), so the rule was never read and stays uncontaminated. Spend: $2,522.65 at
06:49Z → $2,657.58 at 12:41Z, ~$135 (the two trials' ~1.5 h each and Kev-27B's four parent reads, which are kept:
`runs/r21-P27-{tsheld,ood,agentsood,guardood}`, agents-ood from the rerun `r21-P27-agentsood-b` copied into
`runs/r21-P27-agentsood`).

**Why it ran out of memory** (replayed offline: kev.train's epoch-0 shuffle, `none_pairs` and `microbatch_plan` for
`sft-v2-r21` with the registered arguments, then every micro-batch encoded as `encode_batch` builds it). The 62nd step's
first micro-batch on rank 1 held 8 records (a guardrails-PII record, two tool-routing, two synthetic long documents, a
grounding record, a hard-v1 long policy and a tasksource record) and the none-pair siblings of 4 of them: 16 states.
`--length_sort` cuts runs on *characters*, known before encoding, and the characters had costed this run like its
slot's seven others (208k character-padded cost each). But the PII record's state is 11,559 characters and 5,877
tokens (1.97 characters per token; over the corpus's states of more than 200 tokens the median is 4.1, p1 1.9, p99 5.5), so in tokens it set the padding of all 16
states to 5,877: 16 × 5,877 = 94,032 padded state tokens plus 31 branches × 150, a **98,682-token pass** where the slot's
other passes held 33-50k. The failed allocation is exactly one MLP intermediate of that state pass: 94,032 × 17,408 × 2
bytes = 3.05 GiB. Three things combine: (1) characters stand in for tokens with a 2.8× spread; (2) a shared-prefix pass
pads every state, siblings included, to its longest, so one dense state multiplies; (3) the balancer caps the *step*
(the cheapest cut into 16 runs), not a pass, so a step with 8 long records leaves the other 120 to be packed into 8 runs.
Over the whole epoch the characters plan holds 29,160 passes, 1,442 over 49,152 tokens, 209 over 64k, the largest 111k;
two earlier passes of 72.4k (step 16, 20 short states) and 69.2k (step 58, three 22k states) had survived.

**Why it was slow** (65 s per step against the registered 35 s; the rate measured from the logs: 10 → 60 steps in
3,245 s for lr 1e-6, 10 → 50 in 2,623 s for lr 2e-6, i.e. **1.96 records/s**, 128 records a step; the order is shuffled,
so the start is representative; one epoch at that rate ≈ 119,000 s ≈ 33 h). Two causes: the registered projection
(`assemble/epoch_estimate.py`) dealt token shapes, the trainer dealt characters, whose worse balance in tokens takes the
same pass-time model from 18.7 h to 22.6 h; and the measured steps run 1.47× that model (fitted on the five 10-step
intervals, residual ~5 %), which reproduces the measured rate (32.4 h per epoch). The probe below points at padded
multi-state passes with long states (an explicit state mask instead of the flash kernel's causal path): two unequal
states of ~19k tokens take 38.7 s per step where one 32k state takes 18.7.

Fixes for round 22: a per-pass memory ceiling in the trainer (PR #156, `--pass_tokens_max`), and a training set sized to
the measured rate (`evals/sft-v2-r22`).

## Round 22 (registered)

### Round 22 - round 21's science on a trainable recipe: full-weight SFT of Qwen3.8-27B on sft-v2-r22 with a per-pass memory ceiling, one arm (registered with this spec's commit, written before any round-22 training or read)

**Why.** Round 21 was registered and failed at startup with no read (above), so its question stands: does full-weight
SFT from the base on the extended corpus (long states, tone pairs, tasksource, out-of-domain, guardrails and agent
records) beat Kev-27B under round 21's re-based guards? Round 22 asks it again with the same rule, pool, reads and
parents, and with round 21's arm (a) (lr 2e-6) only; the recipe's memory plan, the training set's size and the read
timeout change, each because of what round 21 measured, and arm (b) is dropped for the budget ("Budget" below).

**Trainer change: a per-pass memory ceiling** (PR #156, which this round depends on). `--pass_tokens_max 40960`: the
plan cuts each step on exact token shapes (`kev.train.plan_shapes`: every variant the epoch trains, siblings included;
branches encoded without their state, the state counted once by `state_token_counts`), and a step whose costliest pass
is over 40,960 padded tokens (`pass_tokens`: states × the longest state + branches × the longest branch) gets one more
micro-batch per rank until none is, every rank the same count. Without the flag the plan is byte for byte today's. The
value comes from memory: the long-state probe's single records peak at 78 / 96 / 115 / 134 GiB at 16k / 32k / 48k / 64k
(`runs/sft-probe/lc-27b-8xh200`), and the new probe replays, on 8 H200s, the worst passes the round-22 plan allows
(`scripts/sft_probe.py --passes`, `runs/sft-probe/r22-ceiling-27b-8xh200`, one container, ~18 min, ~$12):

| pass (from the round-22 plan) | padded tokens | states × longest | branches × longest | peak GiB per GPU | s per step |
|---|---|---|---|---|---|
| most padded tokens with ≥ 8 states | 40,944 | 18 × 1,512 | 39 × 352 | 103.2 | 18.1 |
| most branches × state (one agents trace) | 30,951 | 1 × 29,631 | 10 × 132 | 103.7 | 18.5 |
| most states | 38,130 | 74 × 337 | 97 × 136 | 101.6 | 15.7 |
| most padded tokens with two long states | 40,958 | 2 × 18,859 | 9 × 360 | 108.2 | 38.7 |
| round 21's failing pass (replay) | 98,682 | 16 × 5,877 | 31 × 150 | out of memory at 137.4 | - |

Every pass the plan allows peaks at or below 108.2 GiB of 140 (the bar was ~125). The replay of round 21's pass fails as
round 21 did, to the byte: "Tried to allocate 3.05 GiB", 135.34 GiB allocated by PyTorch, in `kev/shared_prefix.py:129`. A 49,152 ceiling would add ~10 GiB
by the single-record slope and, by the pass-time model, save no time (fewer, longer passes: 20.0 h vs 19.5 h on one of the
candidate sets), so the lower value is registered. On sft-v2-r22 the ceiling gives 2,945 micro-batches per rank for 1,140 steps
(1.29 per accumulation slot instead of 1), the largest pass 40,954 tokens.

**Data** (private; manifests only in this repo). `evals/sft-v2-r22` (kev-private-train @ `f8d59fbb`; built by kev-sft
`assemble-v2-r22` @ `0ba98a3`, `assemble/build_v2.py --derived sft-v2-r22`, which rebuilt `sft-v2` byte for byte first,
so the validation and the screen are round 21's, against the same 102 evaluation partitions and 76,549 reference items):
sft-v2's records, same order, round 21's caps (states ≤ 32,768 tokens; calibration and development ≤ 8,192 tokens,
byte-identical to sft-v2-r21's), then, in the priority set for this round, downsampled until one epoch fits the measured rate:

| component | sft-v2-r21 train | sft-v2-r22 train | rule |
|---|---|---|---|
| tone, injection, guardrails-pii, guardrails-grounding, agents, ood, longdoc | 7,544 / 2,699 / 4,152 / 5,655 / 4,875 / 7,312 / 7,617 | all kept | - |
| sft-v1 Kev components (b1v2, documents-v1, hard-v1, devtools-v1) | 33,108 | all kept | - |
| longify | 6,400 | 3,200 | smallest sha256(seed:keep:longify: + digest) |
| tasksource-v1 | 58,190 (119 families) | 24,000 (all 119 families) | stratified by family, floors then largest remainders (family list private) |
| sft-v1 public sources | 28,362 (cap 1,200) | 12,000 (cap 500) | round 21's hash rule at 500 per source |
| sft-v1 synthetic (7 sources) | 67,351 | 33,678 | half of each source, same hash rule |
| **total** | **233,265** | **145,840** | |

Train 145,840 records, 337,130 questions, 332.0M state tokens (355.1M row tokens with the state shared); state tokens
≤256 67,187 · 257-512 16,368 · 513-1k 20,065 · 1k-2k 18,651 · 2k-4k 5,802 · 4k-8k 3,510 · 8k-16k 7,690 · 16k-32k 6,567.
Calibration 13,385 and development 6,201 records, as in round 21 (in-trial screening only). No public file names a
tasksource family.

Why these cuts: the time is in the long records. Per record, by the pass-time model, longdoc carries 26 % of round 21's
epoch, sft-v1 synthetic 22 % (its long documents alone 9.5 %), longify 22 %, agents 12 %; tasksource (2.6 %) and the public
sources (1.9 %) carry little, so cutting them alone saves minutes. Halving longify, as suggested for this round, keeps half of
the one component that repeats sft-v1 train records at length; tasksource keeps every family; the public cap halves
again; and sft-v1 synthetic, last in the priority, is halved because nothing else reaches the time.

**Epoch time, from the measured rate.** The pass-time model (per GPU: 1.55 s + 0.478 s per 1k padded tokens + 0.00282 s ×
states × (longest state in k)², the slowest rank per micro-batch slot, summed) scaled by the 1.47 that reproduces round 21's
measured steps, on the exact round-22 plan (the shuffle, pairs and ceiling of the registered arguments): **66,818 s =
18.6 h** for 1,140 steps (58.6 s per step). This is conservative for the passes the ceiling allows: the probe's short
multi-state passes ran at 0.6× the scaled model and single long states at the unscaled model, and only the two-long-state
passes ran slower (1.16×); anchored to the probe's times, the same plan takes **14.8 h**. Per arm, at 18.6 h: training ×
1.03 for hourly resume points = 19.1 h, three starts (load, the gate's and the ceiling's token counts) ~1.25 h, two
timeouts losing ~0.5 h each back to the last resume point (worst 1 h each): training done at **21.4 h expected / 22.4 h
worst**; in-trial scoring (calibration + development + transfer-v4, 20,350 records at 0.292 s each) 1.65 h: **23.0 h
expected / 24.0 h worst** of the 3 × 8 h. The final checkpoint is committed before scoring, and every candidate's rule
reads are its own reads, so a third attempt that times out in scoring costs the round nothing. Snapshots at 0.25 / 0.5 /
0.75 of 1,140 steps: 285 / 570 / 855.

**Arms** (spec `experiments/rounds/r22.json`, plans `experiments/round22/`): round 21's, fresh from `Qwen/Qwen3.8-27B` @
`1d4bf0f2`, 8×H200 per trial, one epoch of `evals/sft-v2-r22`, batch 8 × accum 2 × 8 ranks = 128 records per step
(`--length_sort 1`, `--pass_tokens_max 40960`), bf16 autocast, OneCycle (10 % warm-up), head lr 1e-4, `--max_state 32768`,
`p_none_pair 0.25` with `none_pair_max_state 8192`, seed 0, **one arm**:
- `27b-lr2e6`: lr 2e-6 (round 19's arm (a) and round 21's arm (a); in round 19 lr 2e-6 was the better of the two SFT
  learning rates: arm (b), 5e-6, also failed the breadth primary).

Round 21's arm (b), `27b-lr1e6` (lr 1e-6), is not registered (Jared's decision on the budget, below). Its point, a
smaller step from the base, is partly covered by the snapshots: s25 / s50 / s75 are the lr 2e-6 run after a quarter,
half and three quarters of the epoch, less-trained points of the same run (not the same thing as a smaller learning
rate under the full schedule, which this round does not test).

Candidates: the arm's snapshots at steps 285 / 570 / 855 (`27b-lr2e6-s25/s50/s75`,
`/runs/r22-27b-lr2e6/00-trial-0/snapshots/step-000NNNN/checkpoint`) and its final checkpoint, **4 in all**, each read with
its own `transfer4` read, exactly as round 21 registered.

**Temperature, reads, parents, rule and confirmation: round 21's, unchanged** ("Round 21 (registered)" above; the spec
differs from `r21.json` only in round number, one study and its four arms instead of two and eight, arm paths,
confirmation read paths and `read_timeout`). The
temperature pool (transfer-r3 calibration's eight held-out sources + transfer-v9 MMLU-Pro, minus transfer rows), the
reads per candidate, the primaries (breadth-v1, tasksource-heldout-v1 dev accuracy lower bound > 0; Kev panel ≥ −1 pp),
the guards (short state, WANLI-v2, scienthoon ≥ −4 pp, pooled externals ≥ −2.5 pp, longdoc CUAD all lengths and 16k+,
unknowable ≤ 0.05), the four calibration criteria, the rank and the tests / locked stages are as written there. Kev-27B's
reads are reused as they are: its committed reads plus round 21's four parent reads, `runs/r21-P27-tsheld`,
`runs/r21-P27-ood`, `runs/r21-P27-agentsood` (the rerun `r21-P27-agentsood-b`, 373 of 373 records, copied there) and
`runs/r21-P27-guardood`, which were made before any round-21 candidate existed.

**Read timeout.** A 27B read of agents-ood-v1 took 68 min (the default timeout is 30), guardrails-ood-v1 26 min and
longdoc-v1 2 h 46 min (its timeout is 3 h), so a report-only read could time out and leave a candidate incomplete. The
spec registers `read_timeout: {"27b": 14400}` (4 h for every 27B read). It raises the admission bound of a candidate's 17
reads to 17 × $25.06 = $426 (H200, 4 h each); the expected cost is unchanged (~$30 a candidate).

**Budget: one arm** (Jared's decision at registration). Two arms fitted the $5,000 metered ceiling only on paper:
both study bounds plus ~$250 of reads came to $4,894.51, which leaves none of the ~10 % reserve docs/autoresearch.md
section 2 keeps out of every plan (billing readings lag and are revised), and with the 4 h read timeout a candidate's read
batch has an admission bound of $426.03, so reading all 8 candidates would have waited on metered spend and, if the
studies spent their bounds, on a raised ceiling. With one study every candidate's reads are admitted under the rule, one
batch at a time, with the reserve intact:

| item (metered reading $2,668.52 at 2026-09-26T13:42Z, lagging; it includes most of the probe) | admission bound | expected |
|---|---|---|
| metered spend so far | $2,668.52 | $2,668.52 |
| study `r22-27b-lr2e6` (H200:8 at $41.17/h × 8 h × 3 attempts) | $987.99 | ~$947 (23.0 h at the scaled model; ~$750, 18.2 h, anchored to the probe) |
| reads of the 4 candidates (17 reads × 4 h × $6.27/h = $426.03 a candidate) | $1,704.13 in all, launched one candidate at a time: $426.03 at any moment | ~$120 (~$30 a candidate) |
| **projection at the rule stage** | **$4,082.54** at any moment (metered + study + one read batch) | **~$3,736** |
| reserve left of $5,000 | **$917.46 (18 %)** | **~$1,264 (25 %)** |
| confirmation, candidate only (tests stage 14 reads × $25.06, locked read at 4 h, serving check) | $350.85 + ~$25 + ~$6 | ~$60 |

The rule-stage figure counts the study's bound in full even while its spend is already metered, so it is conservative;
after the study ends, two candidates' read batches fit at once as well ($2,668.52 + ~$947 + 2 × $426.03 ≈ $4,468).
Confirmation runs after the rule's read-out, when the study and the reads are metered: ~$3,736 + ~$382 of bounds stays
inside the reserve.

**Run steps** (after PR #156 and this PR are merged): (1) fetch `evals/sft-v2-r22` into the checkout (`load_split`; not
`evals/sft-v2/*.jsonl`), copy round 21's four parent reads into `runs/`, then `KEV_GPU=H200 KEV_APP_NAME=kev-sft uv run modal
deploy modal_app.py`; (2) read the metered cost, then `uv run python -m kev.rounds launch experiments/rounds/r22.json` and
`watch`; in the first minutes check the log for `none pairs: N of 145840 records` and `plan: 2945 micro-batches per rank
for 1140 steps (--accum 2); the plan's largest pass ... of --pass_tokens_max 40960`, and count steps per minute against 58.6 s
per step (projected; 45-47 s anchored to the probe); (3) as snapshots land, `launch-reads experiments/rounds/r22.json --arms
<arm>` one candidate at a time (the budget table), reading the metered cost before each; (4) read-out, then confirmation as written. What may be committed: as round 21.

### Round 22 result

**No candidate.** All four candidates were read in full (17 of 17 reads each) and none passes the registered rule, so no
confirmation read was made. Read-out: `runs/r22-readout/round22.json` (`python -m kev.rounds readout
experiments/rounds/r22.json`; reproduced exactly by `tests/test_rounds.py::test_readout_reproduces_round_22` where the
private rows can be fetched). Every candidate was served at the temperature fitted on its own 648 pool questions
(transfer-r3 calibration, eight sources, 448, plus transfer-v9 MMLU-Pro, 200; none excluded as a transfer-v4 duplicate),
Kev-27B at its shipped 1.38. The final's pool fit, 1.382, is the same point of the fit's 121-point log grid as Kev-27B's
(its fit report names its own r3cal + v9 reads). Deltas are paired record-clustered bootstraps against Kev-27B (2,000
resamples, seed 0, micro): accuracy and confident errors in pp, Brier absolute, ECE as served against its bar.

| criterion (panel, n) | s25 (step 285) | s50 (step 570) | s75 (step 855) | final (step 1,140) |
|---|---|---|---|---|
| T (pool of 648) | 1.176 | 1.516 | 1.320 | 1.382 |
| 1 breadth-v1 acc, lower > 0 (3,075) | +0.9 [−0.1, +1.8] **fail** | +0.9 [−0.2, +2.0] **fail** | +1.2 [+0.2, +2.2] | +1.3 [+0.3, +2.3] |
| 1 tasksource-heldout acc, lower > 0 (2,788) | +3.0 [+1.5, +4.3] | +3.1 [+1.6, +4.6] | +3.3 [+1.9, +4.7] | +3.8 [+2.3, +5.3] |
| 1 Kev panel acc, lower ≥ −1 (3,731) | +6.0 [+4.7, +7.2] | +7.2 [+5.9, +8.4] | +7.0 [+5.8, +8.4] | +7.8 [+6.6, +9.1] |
| 2 short acc, lower ≥ −2 (1,806) | −0.7 [−1.8, +0.4] | −0.7 [−1.9, +0.6] | −1.7 [−2.9, −0.6] **fail** | −1.2 [−2.3, −0.2] **fail** |
| 2 short Brier, upper ≤ +0.01 | +0.009 [−0.002, +0.019] **fail** | +0.001 [−0.010, +0.011] **fail** | +0.009 [+0.000, +0.019] **fail** | +0.001 [−0.007, +0.010] |
| 2 short confident errors, upper ≤ +1 | −0.8 [−1.5, −0.1] | −1.1 [−1.8, −0.5] | −0.8 [−1.5, −0.2] | −1.2 [−1.9, −0.5] |
| 2 WANLI-v2, lower ≥ −2 (1,002) | +0.6 [−1.5, +2.7] | −0.5 [−2.5, +1.4] **fail** | +0.8 [−1.2, +2.7] | +0.5 [−1.5, +2.4] |
| 2 scienthoon, lower ≥ −4 (873) | −2.3 [−4.5, −0.2] **fail** | −6.2 [−8.7, −3.8] **fail** | −6.1 [−8.6, −3.7] **fail** | −5.5 [−7.8, −3.2] **fail** |
| 2 pooled externals, lower ≥ −2.5 (2,108) | −0.7 [−2.1, +0.6] | −3.1 [−4.6, −1.8] **fail** | −2.3 [−3.7, −1.0] **fail** | −2.2 [−3.6, −0.9] **fail** |
| 2 longdoc CUAD acc, lower ≥ −2 (2,254) | +0.5 [−0.8, +1.8] | −0.0 [−1.4, +1.3] | +0.5 [−0.5, +1.6] | +0.6 [−0.3, +1.6] |
| 2 CUAD acc at 16k+, cand − parent ≥ −2 (906) | 0.841 vs 0.837 | 0.831 vs 0.837 | 0.838 vs 0.837 | 0.839 vs 0.837 |
| 2 unknowable share ≤ 0.05 (transfer-v9) | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 breadth ECE ≤ 0.0218 (Kev-27B 0.0118) | 0.0181 | 0.0191 | 0.0172 | 0.0202 |
| 3 Kev-panel ECE ≤ 0.0316 (Kev-27B 0.0216) | 0.0132 | 0.0207 | 0.0220 | 0.0188 |
| 3 tasksource-heldout ECE ≤ 0.1037 (Kev-27B 0.0937) | 0.0652 | 0.0445 | 0.0518 | 0.0473 |
| 3 CUAD ECE at 16k+ ≤ 0.0808 (Kev-27B 0.0708) | 0.1102 **fail** | 0.1074 **fail** | 0.1042 **fail** | 0.1023 **fail** |

The s25 scienthoon lower bound is −4.47 (bar −4) and the s50 WANLI-v2 lower bound −2.50 (bar −2). Accuracies (s25 / s50 /
s75 / final, Kev-27B in brackets): breadth 0.754 / 0.754 / 0.757 / 0.758 (0.745); tasksource-heldout 0.707 / 0.708 /
0.711 / 0.716 (0.678); Kev panel 0.836 / 0.848 / 0.847 / 0.854 (0.776); short 0.861 / 0.861 / 0.850 / 0.855 (0.868);
scienthoon 0.773 / 0.734 / 0.735 / 0.741 (0.796); pooled externals 0.779 / 0.755 / 0.763 / 0.764 (0.787).

**The pattern.** The primaries pass and grow with training: at the final, breadth +1.30 [+0.33, +2.32], tasksource-heldout
+3.80 [+2.34, +5.29] (ECE 0.047 vs Kev-27B's 0.094) and the Kev panel +7.8 [+6.6, +9.1]. The report-only out-of-domain
suites are far better at every point (final vs Kev-27B, accuracy / ECE): ood-v2 0.959 / 0.022 vs 0.944 / 0.044 (4,988
questions), agents-ood-v1 0.988 / 0.031 vs 0.967 / 0.137 (2,084), guardrails-ood-v1 0.983 / 0.010 vs 0.944 / 0.079
(4,949); longdoc-v1's generated half is 1.000 for both. What fails is where round 19's SFT failed, and it grows after the
first quarter of the epoch: scienthoon (−2.3 at s25, −5.5 [−7.8, −3.2] at the final) and the pooled externals, which
carry scienthoon (−0.7, then −2.2 [−3.6, −0.9]); short-state accuracy at s75 and the final (lower bound −2.27 at the
final); and CUAD calibration. CUAD accuracy holds at every length, but its ECE is ~0.10 against Kev-27B's 0.06-0.07 in
**every** length bucket (final: under 8k 0.094 vs 0.059, 8k-16k 0.101 vs 0.063, 16k-32k 0.103 vs 0.070, 32k-64k 0.106 vs
0.072): the contract domain at the pool temperature, not state length. The other three calibration criteria pass at every
point.

**Scienthoon** (report only; `scripts/scienthoon_drift.py` with the four candidates added, `runs/r22-scienthoon/drift.json`; the
script was removed with the suite on 2026-09-27 and is in git history at `9c41005`).
The tone pairs removed the error round 20 traced: calm tickets called angry are **0** at every point, against 9 for
Kev-27B and 28-54 for the round-19/20 arms. The error changed sign instead: angry tickets called calm are 13 / 35 / 28 / 28
(Kev-27B 2), so `angry` is 0.852 at the final against 0.911. And `priority`, whose label follows a rule absent from the
text, fell from 0.529 to 0.464 / 0.423 / 0.402 / 0.419 (final −11.0 pp [−16.5, −6.2]). Without the `angry` question the
final is still −5.3 [−7.9, −2.9]; `queue` is level (0.952 vs 0.948).

**Breadth index** (report only; `scripts/breadth_report.py`, `runs/r22-breadth-report/report.md`, each candidate at its pool
T, Kev-27B at 1.38): final 52.3 [49.5, 55.2], +2.1 [−0.6, +4.7] vs Kev-27B 50.2; s25 51.7, s50 52.1, s75 52.0; AutoJev
51.7, Jev 53.3. The gain is Retrieval & Classification again (70.7 vs 62.9; CLINC150 0.953 vs 0.873, SGD 0.740 vs 0.647).

**Deviations.**
- (i) Modal gave the trial 2 of its 3 attempts. Attempt 1 timed out at 22:50:13Z at step ~560 (last resume point 510);
  attempt 2 resumed from 510, wrote snapshot 570 and timed out at 06:56:25Z at step ~1,040 (last resume point 998); that
  `FunctionTimeoutError` was final and no third container was scheduled. Root cause (reproduced on CPU by
  `scripts/modal_retry_probe.py` in PR #163): a timed-out attempt that does not answer Modal's cancellation within 30 s is
  killed, the kill spends a retry of its own, and the retry it triggers may even start next to a running attempt, so
  `Retries(2)` gave 2 attempts. PR #163 (open) turns Modal's retries off for trials and has the watcher continue a
  timed-out trial from its resume point, one call per attempt, counted against the same bound.
- (ii) The watcher marked the trial failed at 07:10Z and wrote a preliminary read-out without the final. The final was
  finished by a manual continuation, the third attempt the study's bound had counted: `modal_app.py::resume`
  (call fc-01M3GW90WZZ1Q4S60DFYCF4SGK, 07:29Z) from resume point 998, bit-exact (the losses at steps 1,010-1,040 equal
  attempt 2's). The final checkpoint was committed at 09:53:27Z. In-trial scoring (screening only) finished at 12:12Z:
  objective −0.410, `sft-v2-r22` development accuracy 0.909, in-trial transfer 0.849, in-trial T 0.933; the fp32
  `isolation_and_packing` gate failed at 0.00213, as for every bf16 27B trial.
- (iii) The trainer's plan was 2,950 micro-batches per rank for 1,140 steps, largest pass 40,947 tokens (the registration
  said 2,945 / 40,954; the same ceiling, noted, not a stop); none pairs on 22,921 of 145,840 records; ~50 s per step
  (projected 58.6, 45-47 anchored to the probe); peak 98.2 GiB per GPU. The snapshots at steps 285 / 570 / 855 were
  committed as registered.
- (iv) The reads' local clients lost the connection or DNS several times (s25: agents-ood and longdoc; s50: agents-ood; s75
  and the final: 12 jobs each). Every remote call finished, and their outputs were pulled from `/bench` by hand: they are
  the registered commands' outputs (17 of 17 per candidate; TypeSafe answered 89 of 102, as for the parent).
- Timeline: the trial started 2026-09-26 14:50Z; reads were launched for s25 ~21:12Z, s50 ~00:51Z, s75 ~04:49Z and the
  final ~10:25Z; the read-out was made at 13:26Z on 2026-09-27. Monitor logs: `runs/r22-readout/monitor.log`, `final.log`.

**Evidence** (committed): the read-out and its private-rows manifest; every public-suite read's `report.json` + `rows.json`
(`runs/r22-27b-lr2e6[-s25|-s50|-s75]-<tag>`); the ood-v2 / agents-ood-v1 / guardrails-ood-v1 reads' reports (aggregates)
and round 21's three such parent reports; the final trial's `provenance.json`, `result-public.json` (its `result.json`
without the per-task table), in-trial transfer rows, in-trial calibration temperature and the three `snapshot.json`
records; the breadth report; the scienthoon drift. Private, in `jaredpalmer/kev-private-train` @ `edec3920` under
`runs/r22/`, with sha256s in `runs/r22-readout/private-rows.json` (`scripts/private_rows.py restore`): every
tasksource-heldout-v1 read, rows and report, because both name the held-out families (the four candidates' and Kev-27B's
`r21-P27-tsheld`); the ood-v2 / agents-ood-v1 / guardrails-ood-v1 rows (held-out records of sft-v2's private components,
the candidates' and the parent's); the trial's development rows (sft-v2 record ids) and its full `result.json`, whose
per-task table names tasksource-v1's families. Checkpoints (~51 GB each) stay on the `kev-runs` volume. Spend: Modal
metered $3,571.88 at 2026-09-27T13:32Z (first read as $3,572.09, then revised), **+$903.36** over the registration reading
of $2,668.52, workspace-wide (the study's three attempts, 68 reads and PR #163's CPU probe), inside the $987.99 study
bound plus the reads' bounds; billing may still lag the final's reads.

## scienthoon removed (2026-09-27)

Jared removed `evals/external/scienthoon-v1` (the validation tickets of scienthoon/jev-ood-calibration) from Kev on
2026-09-27: its manifest, partitions, builder (`scripts/freeze_scienthoon.py`) and the round-20 analysis script
(`scripts/scienthoon_drift.py`) are gone from main; they remain in git history, e.g. at `9c41005`. It is unsound as a gate:
- It is 291 templated synthetic support tickets × 3 questions.
- `queue` (Choice) is saturated: every 27B scores 0.948-0.952.
- `priority` (Score) cannot be learned by construction: its own manifest says the label follows an org rule absent from the text.
- `angry` (Noul) has 15 of 291 gold labels that contradict the text, and it turns on ~12 stock closing phrases whose
  conventions are disputed. It is the question that round 20's analysis found behind the whole 27B cost
  (`runs/r20-scienthoon/analysis.md`).

What changes and what does not:
- **Past verdicts stand as registered.** Rounds 5-22 registered scienthoon reads, guards and the pooled externals with it;
  their outcomes, including every "failed scienthoon" in this file and the model cards, are not revisited. Round 22's reads,
  scienthoon included, were made before the removal, so its read-out applies its rule as written.
- **The record stays reproducible.** Committed rows under `runs/` stay (Jev's `runs/jev-scienthoon-v1`, the round reads,
  `runs/r20-scienthoon/`). `kev.suite.REMOVED_SUITES` names the suite and the reason, and `kev.rounds validate` lists a
  round ≤ 22's scienthoon read as archived instead of failing. Read-outs and verdicts are computed from rows, not the
  suite, so `tests/test_rounds.py` reproduces them unchanged. Model-card numbers still trace to committed reports through
  `scripts/verify_claims.py`. The README's external table dropped the row; its one README-only claim (0.911) went with it.
- **From round 23:** no scienthoon read, panel or guard. The pooled external guard is **SemIf + WANLI-v2 + TypeSafe**, and
  `validate` / `launch` refuse a round that still names the suite. The "Next" items on a scienthoon remedy are closed.
  (Superseded on 2026-09-28: round 24's audited rule gates no pooled externals, and round 23 was re-registered on it, so
  the three suites are reported, not gated; standing rules.)

## WANLI and TypeSafe removed (2026-09-30)

Jared removed `evals/external/wanli-v2`, `evals/external/wanli-v1` and `evals/external/typesafe-v1` from Kev on 2026-09-30,
with their builder (`scripts/freeze_semif_external.py`) and `scripts/compare_typesafe.py`; they remain in git history. The
2026-09-27 audit had already made them report-only (round 24's rule). The check that removed them:
- *WANLI (v2: 1,002 pairs, v1: SemIf's 256).* WANLI publishes each test pair's two crowd annotations
  (`anonymized_annotations.jsonl` at the pinned revision `61c95318`). The two annotators disagree on 271 of wanli-v2's 1,002
  pairs (27 %) and 63 of wanli-v1's 256 (25 %), and the published gold is always one of the two labels: on those pairs the
  gold is a coin flip between people. Every Kev scores far lower there (development rows, as served):

  | | agreed (731) | disputed (271) |
  |---|---|---|
  | Kev-27B v2 (`runs/r23-27b-k-w85-wanli2`) | 0.808 | 0.616 |
  | Kev-9B | 0.802 | 0.572 |
  | Kev-4B | 0.767 | 0.491 |
  | Kev-0.8B | 0.644 | 0.487 |

  Round 18's 9B arm (a), which failed the WANLI-v2 guard, is −0.6 pp on the agreed pairs and −3.0 pp on the disputed ones
  against the released Kev-9B. The audit had measured split-half r 0.04 across 23 checkpoints, all within 0.735-0.763, a
  half-width (1.95-2.10 pp) as wide as the 2 pp bar and ~11 % invalid labels.
- *TypeSafe (`typesafe-v1`, 102 questions over 20 cases, 89 answered at the 8k context they were scored under).* The gold is
  the argmax of TypeSafe's reference distribution, which evals.typesafe.ai describes as the average of two closed frontier
  models' answers (GPT-6 Astra and Claude Fable 5.1, high thinking). It measures agreement with those models, not correctness,
  and 13 of the 102 references put their answer below 0.75 (the two split). Across 20 Kev-27B checkpoints the split-half
  correlation is −0.01 (audit: −0.27), 70 of the 89 are right for all of them, and the rest does not rank checkpoints.
- *SemIf (`semif-v1`, 144 authored decisions) stays, report only.* Its labels hold up: of the 14 questions some Kev-27B
  checkpoint misses, the misses are genuine hard cases (neither candidate authorises the action, so the answer is
  insufficient), not label errors. But it is saturated (0.931-0.979 across 20 Kev-27B checkpoints, 130 of 144 right for all,
  split-half r 0.19), so it is a sanity check, not a ranking.

What changes and what does not: past verdicts stand as registered (rounds 5-26 read these suites; their read-outs and
confirmations reproduce from the committed rows, `kev.suite.REMOVED_SUITES` archives the reads). From round 27 no round
may read them. The README's external table keeps SemIf only; the model cards keep their WANLI and TypeSafe figures as the
record. Round 18's 9B arm (a) failed only WANLI-v2, scienthoon and the pooled externals, all now removed or ungated; whether
it is a Kev-9B candidate is a question for a new registered round on round 24's rule, not a re-reading of round 18.

## Round 24 (registered)

### Round 24 - retrospective selection: the audit's verdicts as one rule, applied to every full-weight 27B checkpoint (registered with this spec's commit, written before any computation under it)

**What this is, plainly.** This is **post hoc selection** among checkpoints that already exist and have already been read
on development data: round 19's two finals, round 20's six WiSE-FT blends and round 22's three snapshots and final (12
checkpoints; no training, no new development read). Their own rounds' verdicts stand as registered: rounds 19, 20 and 22
each selected no candidate, and nothing here revises them. The rule below is the 2026-09-27 evaluation-suite audit's
verdicts (report `audit/REPORT.md` in the audit worktree at `9c41005`; read-only, $6.68 of blind second-opinion spend)
applied uniformly: every suite the round rules read was scored against the same nine criteria (learnability, label noise,
seed noise, reliability, discrimination, power, saturation, contamination, redundancy), whether the SFT checkpoints pass
it or fail it, and each fix below is that audit's verdict for its suite. It is frozen in this commit before any
checkpoint's result under it is computed. It is not blind: the audit read these checkpoints' development rows (its
master table and its post-hoc what-ifs for round 22, e.g. round 22's final: short state without `emotion` −0.88 [−1.95,
+0.19], breadth learnable +1.54 [+0.57, +2.47], tasksource-heldout without the seven families +4.06 [+2.31, +5.77]), so
whoever reads this knows roughly where round 22's candidates fall. Selecting the best of 12 on development panels is
optimistic by construction. The guard against that optimism is the confirmation: untouched test partitions of breadth-v1,
tasksource-heldout-v1, hard-v1, devtools-v1, documents-v1, documents-v2 and longdoc-v1, and the locked transfer-v4 read,
each read once, for the named candidate only, under the same exclusions.

**Why the rule changes** (the audit's numbers; "P0" = the chance that a candidate exactly as good as Kev-27B fails the
registered bar, from round 22's half-widths):
- *scienthoon-v1*: removed (being removed from the repo separately). The blind reader agrees with gold on 1 of 45
  all-wrong items; round 22's final's −5.5 pp is 32 of 48 net questions on `priority`, whose label no text determines;
  Kev-27B sits +3.1 pp above its six LoRA siblings and 3 of the 5 others fail the −4 pp guard against it.
- *WANLI-v2*: report only. Split-half r 0.04 across 23 checkpoints (all within 0.735-0.763); half-width 1.95-2.10 pp
  against a 2 pp bar, P0 48-54 %; ~11 % invalid labels.
- *pooled externals*: removed. Without scienthoon it is 81 % WANLI (τ 0.75), split-half r 0.08. SemIf (saturated: best
  0.993, 90 % of items right for every checkpoint; 144 questions) and TypeSafe (gold = a reference argmax; 89 questions,
  split-half r −0.27) are reported.
- *`emotion`* (transfer-v4 dev and transfer-r3 test): dropped from the short-state and Kev panels. Keyword distant
  supervision: 26 % / 28 % of items wrong for every checkpoint, and the blind reader sides with gold on 2 of 21 / 1 of 25.
  It stays in the temperature pool: dropping it there alone moves T to 1.23 and worsens breadth ECE (0.020 → 0.028).
- *short-state Brier*: the +0.01 bar has P0 36-55 % (half-width 0.85-1.07 pp). The audit's recommendation is a margin the
  half-width can resolve, "e.g. Brier upper ≤ +0.02", and P(fail | Δ = 0) ≤ 10 %; at +0.02 that is ≈ 0.4-4.5 % from the
  same half-widths (bar − half-width over half-width / 1.96). Registered: upper ≤ +0.02. Confident errors keep +1 pp
  (P0 13-22 %), as specified.
- *breadth-v1*: `routerbench` dropped (asks which model answered correctly with no answers in the state: 44 % all-wrong,
  accuracy at the prior); `cfcolor`, `humicroedit` (chance for every system, Jev included) and `chessbench` (floor, 54 %
  all-wrong) moved to report. Split-half r 0.41 → 0.57.
- *tasksource-heldout-v1*: seven families dropped, coded T11 T12 T14 T15 T18 T22 T24 (all-wrong 21-55 %, blind reader with
  gold on ≤ 25 %; invalid labels, 2-way → 3-way mappings, lost span markup, shuffled-word NLI, preference without its
  criterion; 795 of 2,788 development questions). The names are private (the family list is), so the panel reads a private
  exclusion file registered by path and sha256 (below).
- *devtools-v1*: `flakeflagger` (labels from reruns not visible in the code; 3 development projects) and commitpackft
  `change_type` (task `commitpackft_type`: a verb heuristic on a commit message the state does not show; 24 % all-wrong,
  blind reader with gold on 1 of 25) dropped from the Kev panel.
- *longdoc CUAD*: accuracy tripwires kept (P0 1-18 %); ECE 16k+ report only (80 % of the final's 206 confident errors sit on
  wrong, non-unique or disputed gold; the gap is real but the question set needs a rebuild). The generated half (1.000 for
  every checkpoint) and ood-v2 / agents-ood-v1 / guardrails-ood-v1 (sound labels but near ceiling and the SFT components'
  own generators) are reported. hard-v1 (exact labels, but its train templates are in the SFT corpora and it carries +191 of
  round 22's final's +291 net Kev-panel questions) stays inside the Kev retention gate and is also reported alone and as
  the Kev panel without it.
- Not adopted from the audit's prospective recommendations (they need new reads or new suite versions): intervals and
  wider margins on the ECE criteria (their sampling sd is 0.006-0.010 against a 0.01 margin; the ECE deltas' paired
  intervals are reported), gating against the LoRA sibling family, and one sibling read per suite.

**Candidates** (spec `experiments/rounds/r24.json`; all selectable; each a checkpoint arm on the `kev-runs` volume with
`trained_on` its training suite, so the pool check covers it; parent Kev-27B, `r6-27b-v2/01-trial-1`, with the reads rounds
21-22 used):

| arm | checkpoint | trained on | its development reads |
|---|---|---|---|
| `27b-r19a`, `27b-r19b` | round 19 (a) lr 2e-6 / (b) lr 5e-6 finals, `/runs/r19-27b-{lr2e6,lr5e6}/00-trial-0/checkpoint` | `evals/sft-v1` | `runs/r19-27b-*-<tag>`; transfer-v4 dev = the trials' in-trial transfer read; `r3cal` from round 20; tasksource-heldout, longdoc and the three OOD suites from the 2026-09-27 sweep (`runs/sweep-r19-{a,b}-<tag>`) |
| `27b-r20{a,b}-w{85,70,50}` | round 20 blends, `/runs/r20-wise/27b-{a,b}-w{85,70,50}/checkpoint` | `evals/sft-v1` | `runs/r20-27b-*-<tag>` (incl. `transfer4`, `r3cal`); the sweep's `runs/sweep-r20-*-<tag>` |
| `27b-r22-s25`, `-s50`, `-s75`, `-final` | round 22 snapshots at steps 285 / 570 / 855 and the final, `/runs/r22-27b-lr2e6/00-trial-0/...` | `evals/sft-v2-r22` | round 22's 17 reads each (`runs/r22-27b-lr2e6[-sNN]-<tag>`) |

The sweep made the development reads the round-19/20 checkpoints lacked (tasksource-heldout, longdoc, ood-v2,
agents-ood-v1, guardrails-ood-v1; `/runs/<checkpoint>` through `modal_app.py::benchmarks`, raw logits), before this
registration and without a rule. A candidate missing any gating read is incomplete and is never selected; report-only
panels are marked `optional` and never make a candidate incomplete.

**Temperature.** Round 20-23's pool, unchanged: transfer-r3 calibration's eight held-out sources + transfer-v9 MMLU-Pro
(648 questions), minus transfer-v4 dev records; Kev-27B at its shipped 1.38. New, report only (the audit: the final's T
1.382 has a 90 % CI [1.20, 1.52] and breadth ECE moves 0.020 → 0.031 across it): each candidate's 90 % bootstrap
interval of its pooled T, 2,000 resamples of the pool's (source, record) clusters within each source, seed 0, the same
121-point grid and objective (`temperature.ci`).

**Rule** (against Kev-27B; paired record-clustered bootstraps, 2,000 resamples, seed 0, micro; `drop_ids` as before):
1. primaries: breadth-v1 dev accuracy lower > 0 without `routerbench`, `cfcolor`, `humicroedit`, `chessbench`;
   tasksource-heldout-v1 dev accuracy lower > 0 without the seven families; Kev panel (transfer-v4 dev without `emotion`,
   hard-v1, devtools-v1 without `flakeflagger` and `commitpackft_type`, documents-v1) accuracy lower ≥ −1 pp;
2. guards: short state (transfer-v4 dev + transfer-r3 test, both without `emotion`) accuracy lower ≥ −2 pp, Brier upper ≤
   +0.02, confident errors upper ≤ +1 pp; unknowable share on transfer-v9 ≤ 0.05; longdoc CUAD accuracy lower ≥ −2 pp and
   CUAD 16k+ accuracy candidate − Kev-27B ≥ −2 pp (as round 22);
3. calibration: breadth, Kev-panel and tasksource-heldout ECE ≤ Kev-27B's + 0.01 (same exclusions);
4. candidate: the passing checkpoint with the largest breadth + tasksource-heldout + Kev-panel accuracy gain.

Report only (no gate): SemIf, WANLI-v2, TypeSafe; CUAD ECE by length (16k+ included); longdoc generated; ood-v2,
agents-ood-v1, guardrails-ood-v1; hard-v1 alone; the Kev panel without hard-v1; breadth over all 14 sources (the
index's composition) and over the three moved sources; tasksource-heldout over all 24 families; the chance-corrected
breadth index (`scripts/breadth_report.py`). No pooled-externals guard; scienthoon is not read.

**The private exclusion list.** `runs/r24-private/tsheld-exclude.json` (gitignored): the seven family sources plus a
random salt (so the public sha256 cannot be matched against guessed lists), registered in the spec as `{path, sha256}`
(`a72030ab…`) and uploaded to `jaredpalmer/kev-private-train` under `runs/r24/` (manifest
`runs/r24-readout/private-exclude.json`; `scripts/private_rows.py restore`). `kev.rounds` refuses a file whose hash
differs and reports the panel missing without it.

**Harness** (this PR): panel filters `exclude_sources`, `exclude_tasks`, `exclude_file` and `source` as a list
(`kev.rounds.panel_filter`: the same rows leave both sides and any `versus` reference; the panel records `excluded`);
`optional` report-only panels; `temperature.ci`. Specs without these keys compute exactly as before
(`tests/test_rounds.py` reproduces rounds 5-20).

**Confirmation** (the named candidate only, each read once, as `docs/autoresearch.md` says; none if no candidate):
- `tests`: breadth-v1 test accuracy lower > 0 (same four sources out; all 14 reported), with Jev and AutoJev read once on
  the same test items (report; `kev.jev`, AutoJev's server, `scripts/breadth_report.py`); tasksource-heldout-v1 test lower
  > 0 (same seven families out); pooled hard-v1 + devtools-v1 (without `flakeflagger`, `commitpackft_type`) +
  documents-v1 test lower ≥ −1 pp; documents-v2 and longdoc-v1 test (CUAD by length, generated) reported. Kev-27B's side
  is read once too (`runs/r24c-27b-parent-<tag>`; no 27B test read of these suites exists).
- `locked`: locked transfer-v4 accuracy ≥ 0.886 and served Brier ≤ 0.165 (`kev-27b-r24-ungated`).
- Before any release: the bf16 serving check (`modal_app.py::serving --run <checkpoint> --gpu H200 --name serving-27b-r24
  --flags=--isolation`: max |Δp| ≤ 0.03, ≤ 1 flip in 280) and the long-state run at 8k / 32k / 64k (`--flags "--state_tokens
  8192,32768,65536 --reps 3"`, served vs benchmark agreement per length, same bars); release temperature = the pool fit
  written by `scripts/calibrate_checkpoint.py --rows <r3cal rows>:composition_holdout,emotion,legacy_holdout,mmlu,paws,qnli,sciq,tweet_offensive
  --rows <v9 rows>:mmlu_pro --exclude_rows <transfer4 rows>`.

**Budget.** The rule stage costs nothing (existing reads; the sweep's reads were made and paid for separately). Confirmation,
candidate only: tests 14 reads × $25.06 (H200, 4 h bound) = $350.85 bound (~$60 expected), locked ~$25, serving check ~$6,
Jev on breadth-v1 test through the AI Gateway (~$0.10, cap $3).

**Run steps.** (1) Put the reads in place: the round-22 and round-21 parent reads from the round-22 checkout (public rows
committed with the read-out, private ones restored from `jaredpalmer/kev-private-train`), the sweep's reads copied to
`runs/sweep-<checkpoint>-<tag>`; (2) `uv run python -m kev.rounds validate experiments/rounds/r24.json`; (3) `uv run python -m
kev.rounds readout experiments/rounds/r24.json`; (4) confirmation by Jared, if there is a candidate. Committed: the
read-out, the public rows it scores, and a private-rows manifest for the tasksource-heldout and OOD rows.

### Round 24 result

**Candidate: `27b-r22-final`** (round 22's final checkpoint, `/runs/r22-27b-lr2e6/00-trial-0/checkpoint`). 2 of 12
checkpoints pass the registered rule, round 22's final and round 20's `27b-r20a-w85`; the rank (breadth + tasksource-heldout
+ Kev-panel accuracy gain) puts the final first, +13.5 pp against +12.3. This is post hoc selection among checkpoints
already read on development data (above); no confirmation read has been made, and the verdict rests on the confirmation
stages. Read-out: `runs/r24-readout/round24.json` (`python -m kev.rounds readout experiments/rounds/r24.json`, run at
2026-09-27T23:57Z, after the registration commit `523e6ea` at 23:53:28Z; reproduced by
`tests/test_rounds.py::test_readout_reproduces_round_24` where the private rows can be fetched). Every checkpoint served
at its pooled T (648 questions, none excluded as a transfer-v4 duplicate), Kev-27B at 1.38; deltas are paired
record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro), accuracy and confident errors in pp, Brier
absolute, ECE as served against its bar. The exclusions removed 600 breadth questions, 795 tasksource-heldout, 380 Kev-panel
(`emotion` 80, `flakeflagger` 150, `commitpackft_type` 150) and 220 short-state (`emotion` 80 + 140) on both sides.

| criterion | r19 (a) | r19 (b) | r20 a-w85 | r20 a-w70 | r20 a-w50 | r20 b-w85 | r20 b-w70 | r20 b-w50 | r22 s25 | r22 s50 | r22 s75 | r22 final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T (pool, 648) [90 % CI] | 1.414 [1.29, 1.55] | 1.382 [1.23, 1.52] | 1.414 [1.29, 1.55] | 1.447 [1.29, 1.59] | 1.447 [1.32, 1.59] | 1.350 [1.23, 1.48] | 1.350 [1.20, 1.48] | 1.350 [1.23, 1.45] | 1.176 [1.05, 1.29] | 1.516 [1.38, 1.66] | 1.320 [1.18, 1.41] | 1.382 [1.23, 1.48] |
| 1 breadth acc, lower > 0 (2475) | +1.6 [+0.6, +2.5] | +0.8 [-0.4, +1.9] **fail** | +1.6 [+0.6, +2.5] | +1.7 [+0.7, +2.7] | +1.5 [+0.6, +2.3] | +1.2 [+0.1, +2.2] | +1.6 [+0.6, +2.6] | +2.1 [+1.1, +3.1] | +1.3 [+0.4, +2.1] | +1.0 [-0.0, +2.0] **fail** | +1.7 [+0.6, +2.6] | +1.5 [+0.6, +2.5] |
| 1 tasksource-heldout acc, lower > 0 (1993) | +1.4 [-0.3, +2.9] **fail** | -0.1 [-1.8, +1.7] **fail** | +1.7 [+0.2, +3.2] | +1.5 [-0.1, +3.1] **fail** | +1.0 [-0.5, +2.7] **fail** | +0.2 [-1.6, +1.8] **fail** | +1.4 [-0.2, +3.0] **fail** | +2.0 [+0.5, +3.6] | +2.4 [+0.8, +4.0] | +3.2 [+1.4, +4.9] | +3.7 [+2.0, +5.4] | +4.1 [+2.3, +5.8] |
| 1 Kev panel acc, lower ≥ −1 (3351) | +9.0 [+7.8, +10.1] | +8.2 [+7.0, +9.5] | +9.0 [+7.9, +10.2] | +8.9 [+7.8, +10.1] | +7.6 [+6.5, +8.8] | +8.5 [+7.3, +9.8] | +8.6 [+7.4, +9.8] | +8.3 [+7.1, +9.5] | +5.7 [+4.4, +7.0] | +7.3 [+6.1, +8.6] | +7.0 [+5.8, +8.3] | +7.9 [+6.7, +9.1] |
| 2 short acc, lower ≥ −2 (1586) | -0.9 [-2.1, +0.1] **fail** | -0.3 [-1.5, +0.9] | -0.7 [-1.8, +0.4] | -0.6 [-1.7, +0.4] | -0.7 [-1.8, +0.3] | -0.2 [-1.4, +1.0] | -0.4 [-1.5, +0.7] | -0.3 [-1.3, +0.8] | -0.8 [-2.0, +0.4] | -0.2 [-1.5, +1.1] | -1.4 [-2.7, -0.2] **fail** | -0.9 [-2.0, +0.2] |
| 2 short Brier, upper ≤ +0.02 | +0.003 [-0.008, +0.013] | -0.002 [-0.013, +0.009] | +0.002 [-0.009, +0.012] | -0.002 [-0.013, +0.008] | -0.001 [-0.013, +0.009] | -0.003 [-0.015, +0.007] | -0.005 [-0.017, +0.004] | -0.005 [-0.017, +0.005] | +0.008 [-0.004, +0.020] **fail** | -0.002 [-0.013, +0.009] | +0.010 [-0.001, +0.021] **fail** | -0.000 [-0.009, +0.009] |
| 2 short confident errors, upper ≤ +1 | -0.9 [-1.6, -0.3] | -0.6 [-1.3, +0.1] | -0.9 [-1.7, -0.3] | -1.1 [-1.8, -0.5] | -0.9 [-1.6, -0.3] | -0.6 [-1.4, +0.1] | -0.8 [-1.6, -0.2] | -0.8 [-1.5, -0.2] | -0.8 [-1.5, +0.0] | -1.2 [-2.0, -0.6] | -0.8 [-1.5, -0.1] | -1.1 [-1.9, -0.5] |
| 2 CUAD acc, lower ≥ −2 (2254) | +0.3 [-0.7, +1.3] | -0.4 [-1.8, +0.9] | +0.3 [-0.7, +1.2] | -0.1 [-1.1, +0.7] | -0.4 [-1.2, +0.5] | -0.5 [-1.7, +0.7] | -0.7 [-1.8, +0.4] | -0.8 [-2.0, +0.3] | +0.5 [-0.8, +1.8] | -0.0 [-1.4, +1.3] | +0.5 [-0.5, +1.6] | +0.6 [-0.3, +1.6] |
| 2 CUAD 16k+ acc, cand − parent ≥ −2 | 0.836 vs 0.837 | 0.831 vs 0.837 | 0.834 vs 0.837 | 0.831 vs 0.837 | 0.832 vs 0.837 | 0.832 vs 0.837 | 0.832 vs 0.837 | 0.830 vs 0.837 | 0.841 vs 0.837 | 0.831 vs 0.837 | 0.838 vs 0.837 | 0.839 vs 0.837 |
| 2 unknowable share ≤ 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 breadth ECE ≤ 0.0176 (Kev-27B 0.0076) | 0.0094 | 0.0149 | 0.0077 | 0.0141 | 0.0206 **fail** | 0.0167 | 0.0139 | 0.0255 **fail** | 0.0173 | 0.0192 **fail** | 0.0087 | 0.0149 |
| 3 Kev-panel ECE ≤ 0.0327 (Kev-27B 0.0227) | 0.0131 | 0.0099 | 0.0149 | 0.0169 | 0.0283 | 0.0094 | 0.0194 | 0.0210 | 0.0105 | 0.0275 | 0.0165 | 0.0155 |
| 3 tasksource-heldout ECE ≤ 0.0523 (Kev-27B 0.0423) | 0.0185 | 0.0168 | 0.0207 | 0.0126 | 0.0235 | 0.0139 | 0.0203 | 0.0295 | 0.0351 | 0.0474 | 0.0470 | 0.0482 |
| rank score (Δ breadth + Δ tsheld + Δ Kev, pp) | +11.9 | +9.0 | +12.3 | +12.0 | +10.1 | +9.8 | +11.6 | +12.4 | +9.4 | +11.5 | +12.4 | +13.5 |
| verdict | fail | fail | **PASS** | fail | fail | fail | fail | fail | fail | fail | fail | **PASS** |

What fails, and where: tasksource-heldout's primary is the common failure among round 19/20's checkpoints (6 of 8: every
one but `27b-r20a-w85` and `27b-r20b-w50`); breadth ECE (`27b-r20a-w50`, `27b-r20b-w50`, `27b-r22-s50`); short state
(accuracy: `27b-r19a`, `27b-r22-s75`; Brier upper bound: `27b-r22-s25` +0.0203 and `27b-r22-s75` +0.0208, against +0.02); breadth (`27b-r19b`,
`27b-r22-s50`, whose lower bound is −0.04 pp). The round-22 final passes short-state accuracy only because `emotion` is out
(−0.9 [−2.0, +0.2]; with it, round 22's registered read was −1.2 [−2.27, −0.22]), and its CUAD ECE at 16k+ is still 0.102
against Kev-27B's 0.071, now reported, not gated.

**Temperature uncertainty** (report only). 90 % bootstrap intervals of the pooled T are in the table (±0.12-0.15). Served at the
ends of its interval, the named candidate's calibration pass does not hold: breadth ECE 0.0234 at T 1.231 (bar 0.0176) and
tasksource-heldout ECE 0.0617 at T 1.481 (bar 0.0523); Kev-panel ECE stays inside (0.0096-0.0166). `27b-r20a-w85` stays
inside all three across its interval [1.289, 1.551] (breadth 0.0077-0.0171, Kev 0.0125-0.0269, tasksource-heldout
0.0207-0.0274). The rule serves at the point fit, so this changes no verdict; it is the audit's point that the ECE criteria
have no interval, and the released temperature is the same pool fit (`scripts/calibrate_checkpoint.py`).

**Report-only panels** (paired deltas against Kev-27B; ECE absolute):

| panel | r19 (a) | r19 (b) | r20 a-w85 | r20 a-w70 | r20 a-w50 | r20 b-w85 | r20 b-w70 | r20 b-w50 | r22 s25 | r22 s50 | r22 s75 | r22 final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breadth, all 14 sources, acc (3075) | +1.5 [+0.4, +2.5] | +0.6 [-0.6, +1.7] | +1.4 [+0.4, +2.5] | +1.7 [+0.7, +2.6] | +1.5 [+0.5, +2.5] | +1.1 [-0.1, +2.3] | +1.4 [+0.3, +2.5] | +1.7 [+0.7, +2.8] | +0.9 [-0.1, +1.8] | +0.9 [-0.2, +2.0] | +1.2 [+0.2, +2.2] | +1.3 [+0.3, +2.3] |
| breadth cfcolor+humicroedit+chessbench, acc (450) | +1.3 [-2.9, +5.1] | +0.2 [-4.0, +4.4] | +0.9 [-3.1, +4.9] | +1.8 [-2.2, +5.6] | +1.6 [-2.7, +5.3] | +1.3 [-2.9, +5.6] | +0.9 [-3.3, +5.1] | +0.2 [-4.0, +4.4] | -1.1 [-4.7, +2.4] | +0.7 [-3.6, +4.7] | -1.3 [-5.3, +2.7] | -0.7 [-4.7, +3.3] |
| tasksource-heldout all 24 families, acc (2788) | +1.0 [-0.3, +2.3] | +1.1 [-0.4, +2.6] | +1.1 [-0.1, +2.4] | +1.3 [+0.0, +2.6] | +0.9 [-0.5, +2.1] | +1.1 [-0.3, +2.5] | +2.0 [+0.7, +3.4] | +2.1 [+0.8, +3.4] | +3.0 [+1.5, +4.3] | +3.1 [+1.6, +4.6] | +3.3 [+1.9, +4.7] | +3.8 [+2.3, +5.3] |
| hard-v1 acc (1083) | +19.9 [+17.1, +22.7] | +18.8 [+16.1, +21.8] | +19.6 [+16.9, +22.4] | +19.0 [+16.3, +21.9] | +16.4 [+13.8, +19.3] | +19.3 [+16.6, +22.3] | +19.3 [+16.7, +22.2] | +18.3 [+15.6, +21.2] | +14.0 [+11.2, +17.0] | +16.3 [+13.6, +19.2] | +17.3 [+14.3, +20.2] | +17.6 [+14.9, +20.5] |
| Kev panel without hard-v1, acc (2268) | +3.7 [+2.6, +5.0] | +3.2 [+1.9, +4.4] | +4.0 [+2.8, +5.1] | +4.1 [+3.0, +5.1] | +3.4 [+2.3, +4.5] | +3.3 [+2.1, +4.6] | +3.5 [+2.3, +4.7] | +3.5 [+2.3, +4.6] | +1.7 [+0.4, +3.0] | +3.0 [+1.7, +4.3] | +2.1 [+0.9, +3.3] | +3.2 [+2.0, +4.4] |
| SemIf acc (144) | -0.7 [-3.5, +2.1] | -1.4 [-4.2, +1.4] | -0.7 [-3.5, +2.1] | -2.1 [-5.6, +1.4] | -2.8 [-6.2, +0.7] | -2.1 [-5.6, +0.7] | -2.8 [-6.2, +0.7] | -4.2 [-7.6, -0.7] | +0.7 [-1.4, +2.8] | -2.1 [-5.6, +1.4] | -1.4 [-4.9, +1.4] | -1.4 [-4.9, +2.1] |
| WANLI-v2 acc (1002) | +0.0 [-2.1, +2.0] | -0.1 [-2.3, +2.0] | +0.0 [-2.1, +1.9] | +0.6 [-1.3, +2.4] | +0.0 [-1.9, +2.0] | +0.1 [-2.1, +2.2] | +0.7 [-1.4, +2.8] | +1.1 [-1.0, +3.1] | +0.6 [-1.5, +2.7] | -0.5 [-2.5, +1.4] | +0.8 [-1.2, +2.7] | +0.5 [-1.5, +2.4] |
| TypeSafe acc (89) | +1.1 [-4.4, +5.4] | +0.0 [-6.2, +5.9] | +1.1 [-4.4, +5.4] | +0.0 [-5.0, +3.9] | -1.1 [-6.7, +3.5] | +0.0 [-6.2, +5.9] | -1.1 [-8.0, +5.4] | +1.1 [-5.0, +6.4] | -2.2 [-7.1, +2.2] | -4.5 [-9.3, +1.0] | -2.2 [-6.6, +2.0] | -2.2 [-6.6, +2.0] |
| ood-v2 acc (4988) | +1.3 [+0.7, +1.8] | +1.0 [+0.5, +1.6] | +1.2 [+0.7, +1.8] | +1.2 [+0.6, +1.7] | +1.0 [+0.5, +1.5] | +1.2 [+0.6, +1.7] | +1.4 [+0.8, +1.9] | +1.1 [+0.6, +1.6] | +0.8 [+0.3, +1.4] | +1.0 [+0.5, +1.5] | +1.4 [+0.9, +2.0] | +1.5 [+1.0, +2.1] |
| agents-ood-v1 acc (2084) | -0.1 [-0.9, +0.6] | -1.0 [-1.7, -0.3] | -0.0 [-0.8, +0.7] | -0.3 [-1.1, +0.5] | -1.1 [-2.0, -0.2] | -0.8 [-1.5, -0.0] | -0.6 [-1.3, +0.1] | -0.4 [-1.2, +0.3] | +1.7 [+0.9, +2.6] | +1.3 [+0.5, +2.1] | +2.4 [+1.6, +3.1] | +2.1 [+1.4, +2.8] |
| guardrails-ood-v1 acc (4949) | +0.0 [-0.6, +0.6] | +0.5 [-0.2, +1.1] | +0.1 [-0.5, +0.7] | -0.1 [-0.7, +0.5] | -1.1 [-1.8, -0.4] | +0.8 [+0.2, +1.4] | +1.1 [+0.4, +1.6] | +0.7 [+0.1, +1.3] | +3.3 [+2.7, +3.9] | +3.4 [+2.7, +4.1] | +3.7 [+3.1, +4.4] | +3.9 [+3.3, +4.6] |
| ood-v2 ECE | -0.017 [-0.023, -0.012] | -0.015 [-0.021, -0.009] | -0.017 [-0.023, -0.012] | -0.015 [-0.020, -0.009] | -0.009 [-0.014, -0.003] | -0.015 [-0.021, -0.010] | -0.013 [-0.019, -0.007] | -0.010 [-0.015, -0.005] | -0.017 [-0.022, -0.011] | -0.017 [-0.022, -0.011] | -0.023 [-0.029, -0.016] | -0.022 [-0.027, -0.016] |
| agents-ood-v1 ECE | -0.066 [-0.074, -0.057] | -0.066 [-0.074, -0.059] | -0.065 [-0.073, -0.057] | -0.063 [-0.072, -0.054] | -0.055 [-0.065, -0.045] | -0.068 [-0.076, -0.061] | -0.064 [-0.072, -0.056] | -0.055 [-0.063, -0.046] | -0.087 [-0.097, -0.078] | -0.106 [-0.114, -0.097] | -0.104 [-0.113, -0.096] | -0.106 [-0.114, -0.097] |
| guardrails-ood-v1 ECE | -0.020 [-0.027, -0.014] | -0.011 [-0.018, -0.004] | -0.020 [-0.026, -0.013] | -0.017 [-0.024, -0.011] | -0.017 [-0.023, -0.010] | -0.013 [-0.020, -0.007] | -0.012 [-0.019, -0.006] | -0.012 [-0.019, -0.005] | -0.069 [-0.074, -0.061] | -0.069 [-0.075, -0.060] | -0.070 [-0.076, -0.062] | -0.068 [-0.074, -0.061] |
| breadth ECE Δ (gated panel) | +0.002 [-0.011, +0.012] | +0.007 [-0.008, +0.017] | +0.000 [-0.011, +0.011] | +0.006 [-0.008, +0.017] | +0.013 [-0.003, +0.021] | +0.009 [-0.006, +0.019] | +0.006 [-0.008, +0.015] | +0.018 [-0.001, +0.026] | +0.010 [-0.008, +0.019] | +0.012 [-0.006, +0.020] | +0.001 [-0.010, +0.012] | +0.007 [-0.007, +0.016] |
| Kev ECE Δ | -0.010 [-0.020, +0.001] | -0.013 [-0.022, -0.001] | -0.008 [-0.019, +0.003] | -0.006 [-0.018, +0.004] | +0.006 [-0.009, +0.014] | -0.013 [-0.023, -0.001] | -0.003 [-0.016, +0.007] | -0.002 [-0.014, +0.007] | -0.012 [-0.022, +0.001] | +0.005 [-0.009, +0.014] | -0.006 [-0.019, +0.007] | -0.007 [-0.019, +0.005] |
| tsheld ECE Δ | -0.024 [-0.038, +0.001] | -0.025 [-0.040, -0.001] | -0.022 [-0.037, -0.001] | -0.030 [-0.039, -0.003] | -0.019 [-0.034, +0.004] | -0.028 [-0.042, -0.003] | -0.022 [-0.036, -0.000] | -0.013 [-0.028, +0.008] | -0.007 [-0.025, +0.009] | +0.005 [-0.016, +0.024] | +0.005 [-0.015, +0.024] | +0.006 [-0.017, +0.024] |
| CUAD ECE 16k+ (Kev-27B 0.071) | 0.084 | 0.060 | 0.081 | 0.077 | 0.072 | 0.066 | 0.070 | 0.068 | 0.110 | 0.107 | 0.104 | 0.102 |
| longdoc generated acc | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The breadth index (chance-corrected, all 14 sources; `scripts/breadth_report.py`, from rounds 20 and 22's reports at the
same temperatures, not recomputed): Kev-27B 50.2 [47.4, 53.3], AutoJev 51.7, Jev 53.3; round 19 (a) 53.0, (b) 51.7; round 20
a-w85 52.9, a-w70 53.5, a-w50 53.4, b-w85 53.1, b-w70 53.3, b-w50 53.7; round 22 s25 51.7, s50 52.1, s75 52.0, final 52.3
[49.5, 55.2] (+2.1 [−0.6, +4.7] vs Kev-27B; `27b-r20a-w85` +2.7 [+0.0, +5.2]). What the report-only columns show: the
round-22 checkpoints are the ones that learned the out-of-domain generators (agents-ood +1.3 to +2.4 and guardrails-ood +3.3 to
+3.9, against −1.1 to −0.0 and −1.1 to +1.1 for rounds 19/20) and the ones with the tasksource gain (+2.4 to +4.1 on the valid families, vs
−0.1 to +2.0), both in distribution for sft-v2-r22 (its tasksource-v1 families and its OOD components' generators); the
round-19/20 checkpoints keep the CUAD 16k+ ECE near Kev-27B (0.060-0.084 vs 0.071) where round 22's do not (0.102-0.110).
hard-v1 alone is +14.0 to +19.9 pp everywhere; without it the Kev-panel gain is +1.7 to +4.1.

**Checks.** With the exclusions removed, the panels reproduce the committed read-outs exactly: breadth over all 14 sources
(e.g. round 20 a-w85 +1.43, round 22 final +1.30), tasksource-heldout over all 24 families (final +3.80) and every pooled T
(round 20's and round 22's). Round 22's committed read-out reproduces bit for bit with this PR's `kev.rounds`, and
`tests/test_rounds.py` (with `KEV_ROUNDS_ROOT` at the research checkout) reproduces rounds 5-20. The round-22 final's
gated numbers match the audit's post-hoc what-ifs (breadth +1.54 [+0.57, +2.47], tasksource-heldout +4.06 [+2.31, +5.77],
short −0.88 [−1.95, +0.19]).

**Deviations.** (i) The private exclusion list was uploaded twice: the first `scripts/private_rows.py upload` pushed the file
(`kev-private-train@8d7d0835`) but failed to write its manifest into a missing directory; the rerun's identical commit was
skipped by the Hub and the manifest pins `8d7d0835`, the same sha256. (ii) The pooled T's interval resamples (source,
record) clusters within each source (`kev.metrics.cluster_resamples`, 2,000 resamples) where the audit drew 200
unstratified resamples; the final's interval is [1.231, 1.481] here, [1.20, 1.52] there. (iii) Round 19's finals' transfer-v4
development rows are their trials' in-trial transfer reads (as rounds 19 and 20 read them); every other checkpoint has a
`transfer4` read.

**Next (Jared).** Confirmation of `27b-r22-final` as registered (tests stage with Kev-27B's side, Jev and AutoJev on
breadth-v1 test, locked, the bf16 serving check at 8k / 32k / 64k), each read once. Given the temperature finding, a
passing confirmation should report the test-partition ECEs across the candidate's T interval as well.

**Evidence** (committed): the read-out and its table (`runs/r24-readout/readout.txt`); the public rows it scores: round
22's reads (`runs/r22-27b-lr2e6[-sNN]-<tag>`, report + rows; OOD reads: reports only), round 22's final trial's in-trial
transfer rows and provenance, round 21's parent OOD reports, and the 2026-09-27 sweep's reads (`runs/sweep-<checkpoint>-<tag>`:
longdoc report + rows, OOD reports). Private (`jaredpalmer/kev-private-train` under `runs/r24/`,
`scripts/private_rows.py restore`): the exclusion list (`runs/r24-readout/private-exclude.json`, @ `8d7d0835`) and 52 rows
files, every tasksource-heldout-v1 read and the OOD rows (`runs/r24-readout/private-rows.json`, @ `9beb213b`). Spend: $0
(no Modal job, no gateway call).

### Round 24 confirmation

**NOT CONFIRMED.** `27b-r22-final` passes the tests stage and fails the locked stage: locked transfer-v4 accuracy 0.8841
(580 of 656) against the registered bar of 0.886, which needs 582. It misses by 2 questions. Kev-27B scores 0.8963 (588)
on the same read. Nothing is released and Kev-27B stays the 27B release. Each confirmation read was made once, for the
named candidate only, as registered. Verdicts: `runs/r24-verdict/27b-tests.json`, `runs/r24-verdict/27b-locked.json`
(`python -m kev.rounds confirm experiments/rounds/r24.json --stage {tests,locked}`). Both sides are served at 1.382: the
candidate at its pool fit (648 questions, none excluded) and Kev-27B at its shipped T, which is the same grid point.
Deltas are paired record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro), in pp.

**The locked bar stands as registered.** The audit dropped `emotion` from the short-state and Kev panels of the
development rule. The locked criterion was registered on the whole locked transfer-v4 partition, `emotion` included, and
we do not adjust it after the result. Re-scoring the locked read without `emotion` once the candidate had missed by 2
questions is exactly the post hoc move the confirmation exists to rule out, so we did not compute it. The locked stage
says the candidate is no better than Kev-27B on short states: −1.2 [−2.6, +0.2]. That matches the development read,
where short-state accuracy passed only with `emotion` out (−0.9 [−2.0, +0.2]; "Round 24 result").

| stage / criterion (panel, n) | 27b-r22-final | Kev-27B | Δ [95 %] | verdict |
|---|---|---|---|---|
| tests: breadth-v1 test acc, lower > 0 (without `routerbench`, `cfcolor`, `humicroedit`, `chessbench`; 2,489, 600 out) | 0.835 | 0.820 | +1.5 [+0.5, +2.5] | pass |
| tests: tasksource-heldout-v1 test acc, lower > 0 (without the seven families; 2,024, 809 out) | 0.796 | 0.743 | +5.3 [+3.7, +7.0] | pass |
| tests: pooled hard-v1 + devtools-v1 (without `flakeflagger`, `commitpackft_type`) + documents-v1 test acc, lower ≥ −1 (2,795, 300 out) | 0.886 | 0.800 | +8.6 [+7.1, +10.0] | pass |
| **locked: transfer-v4 locked acc ≥ 0.886 (656)** | **0.8841 (580)** | 0.8963 (588) | −1.2 [−2.6, +0.2] | **fail** (needed 582) |
| locked: served Brier ≤ 0.165 (656) | 0.1549 | 0.1604 | −0.005 [−0.018, +0.006] | pass |

**Report-only test reads** (same exclusions; ECE as served):

| panel (n) | 27b-r22-final | Kev-27B | Δ [95 %] |
|---|---|---|---|
| hard-v1 test (1,088) | 0.916 | 0.749 | +16.7 [+14.0, +19.7] |
| devtools-v1 test, gated sources (771) | 0.815 | 0.789 | +2.6 [+0.1, +4.9] |
| documents-v1 test (936) | 0.908 | 0.869 | +4.0 [+2.0, +6.0] |
| documents-v2, private held-out test (953) | 0.919 | 0.881 | +3.8 [+1.7, +6.0] |
| breadth-v1 test, all 14 sources (3,089); ECE | 0.762; 0.017 | 0.748; 0.019 | +1.3 [+0.4, +2.3] |
| breadth-v1 test ECE, gated panel | 0.014 | 0.013 | - |
| tasksource-heldout-v1 test ECE, gated panel | 0.055 | 0.051 | - |
| longdoc-v1 test CUAD acc (2,194) | 0.872 | 0.890 | **−1.8 [−3.2, −0.5]** |
| longdoc-v1 test CUAD ECE | **0.055** | 0.007 | - |
| CUAD by length, acc / ECE: < 8k (867) | 0.873 / 0.058 | 0.900 / 0.025 | - |
| 8k-16k (443) | 0.880 / 0.056 | 0.892 / 0.028 | - |
| 16k-32k (442) | 0.871 / 0.062 | 0.882 / 0.019 | - |
| 32k-64k (442) | 0.862 / 0.066 | 0.876 / 0.014 | - |
| 16k+ (884) | 0.867 / 0.062 | 0.879 / 0.013 | - |
| longdoc-v1 test generated (2,400); ECE | 1.000; 0.001 | 1.000; 0.014 | - |

On CUAD the test read is worse than development. The candidate loses accuracy at every length (−1.1 to −2.7 pp), and its
ECE is 2-5× Kev-27B's in every bucket. The development rule's CUAD accuracy guard (lower ≥ −2 pp) would not hold on test
(−3.2); the 16k+ gap (−1.2 pp) would. Both are report only in this stage, so neither changes a verdict. It is the same
contract-domain miscalibration round 22 found on development (ECE ~0.10 against 0.06-0.07 in every bucket, "Round 22
result", PR #165); Kev-27B's test ECE (0.014-0.028 by bucket) is much lower than its development one.

**Breadth index on TEST** (chance-corrected, all 14 sources; `scripts/breadth_report.py`, `runs/r24-breadth-report/report.md`;
Jev through the AI Gateway (`kev.jev`), AutoJev by its own server, each read once on the same test items): candidate
**53.7 [50.5, 56.7]**, Jev **54.0** [51.2, 57.0], Kev-27B **50.2** [47.0, 53.2], AutoJev **50.0** [47.0, 53.3]. Against
Kev-27B: candidate +3.5 [+1.0, +6.0], Jev +3.9 [+0.9, +7.0], AutoJev −0.2 [−3.3, +3.0]. Pooled ECE: candidate 0.017, Kev-27B
0.019, Jev 0.052, AutoJev 0.040. The candidate's gain is Retrieval & Classification (69.7 vs 61.6; CLINC150 0.947 vs 0.847,
SGD 0.720 vs 0.607) and Arts & Human Taste (28.0 vs 18.7). It gives back Tools & Automation (62.6 vs 65.1; `routerbench`
0.300 vs 0.373). On this test panel it reaches Jev's index: −0.3 on the point estimates, no paired interval computed.

**Calibration at the ends of the temperature interval** (report only, as the round-24 result asked; the candidate served at
the ends of its pooled T's 90 % interval [1.231, 1.481] and at the point fit, Kev-27B at 1.382;
`runs/r24-verdict/27b-tests-t-interval.json`). ECE / Brier:

| test panel (n) | T 1.231 | T 1.382 (served) | T 1.481 | Kev-27B @ 1.382 |
|---|---|---|---|---|
| breadth-v1, gated (2,489) | 0.025 / 0.2346 | 0.014 / 0.2335 | 0.018 / 0.2335 | 0.013 / 0.2516 |
| breadth-v1, all 14 (3,089) | 0.030 / 0.3122 | 0.017 / 0.3111 | 0.019 / 0.3111 | 0.019 / 0.3267 |
| tasksource-heldout-v1, gated (2,024) | 0.035 / 0.2972 | 0.055 / 0.3009 | 0.068 / 0.3041 | 0.051 / 0.3682 |
| pooled hard + devtools + documents-v1 (2,795) | 0.022 / 0.1599 | 0.016 / 0.1603 | 0.021 / 0.1610 | 0.027 / 0.2720 |

The development finding holds on test. Across one interval, breadth ECE moves 0.014 → 0.025 at the low end, and
tasksource-heldout ECE moves 0.055 → 0.068 at the high end: they pull in opposite directions, so no single T in the
interval is best for both. Brier barely moves (≤ 0.004).

**Serving check** (bf16, H200; passed; `runs/serving-27b-r24/report.json`, `runs/serving-27b-r24-long/report.json`):
- Short states, 280 questions:
  - CUDA graphs vs fp32: max |Δp| 0.0235 with 1 argmax flip; eager bf16 vs fp32: 0.0267, 1 flip (bars ≤ 0.03, ≤ 1 flip).
  - Graphs vs eager: 0.022, 0 flips. Question isolation as served: 0.0078, 0 flips.
- Long states, served vs benchmark, 3 records / 15 questions per length:
  - 8k: max |Δp| 0.0083, 0 flips; new state 1.37 s, cached 584 ms; 70 GB peak.
  - 32k: 0.0079, 0 flips; 4.17 s / 612 ms; 79 GB.
  - 64k: 0.0024, 0 flips; 9.47 s / 725 ms; 87 GB.

**Release temperature** (had it passed): the pool fit, 1.3819, 90 % CI [1.231, 1.481] (648 questions; (source, group)
clusters resampled within each source, 2,000 resamples, seed 0).

**Deviation.** The tests-stage reads were launched at 00:09:43Z (`runs/r24-reads-27b-r22-final-tests.json`). That was
from the read-out computed at 23:57Z that named the candidate, about 2.5 min before the result commit `b9ceefe`
(00:12:16Z). The locked read was spawned at 00:22:56Z from `b9ceefe`. No read was repeated.

**Spend.** Modal metered ~$3,790 workspace-wide at round 23's re-registration reading, plus ~$80 of metering lag. Round
23's first registration read $3,571.88 (2026-09-27T13:32Z). The difference covers the 2026-09-27 audit sweep's reads, this
confirmation (14 test reads, the locked read, the two serving checks, AutoJev's breadth-v1 test read) and the workspace's
other apps; it is not split per job. AI Gateway: $0.064 (Jev on breadth-v1 test, 1,992 calls).

**Evidence** (committed): the verdicts and the T-interval table (`runs/r24-verdict/`); the launch records
(`runs/r24-reads-27b-r22-final-{tests,locked}.json`); the public test reads (`runs/r24c-27b-{cand,parent}-<tag>`: rows for
hard-v1, devtools-v1 and documents-v1, reports for the rest); the locked read (`runs/locked/kev-27b-r24-ungated/`); the
test breadth report (`runs/r24-breadth-report/`); Jev's and AutoJev's breadth-v1 test reports; the serving reports.
Private (`jaredpalmer/kev-private-train` under `runs/r24/`, `scripts/private_rows.py restore`): tasksource-heldout-v1
test rows and reports, breadth-v1 test rows of all four systems, documents-v2 rows (`runs/r24-verdict/private-rows.json`,
@ `79c69ff6`) and longdoc-v1 test rows (`runs/r24-verdict/private-rows-longdoc.json`, @ `79272e0f`).

**What follows.** The candidate's gains held on every untouched test partition. It failed on the one place Kev-27B is
strongest, short states, and it is worse on CUAD. Round 23 (PR #165) is re-registered on round 24's audited rule: round
22's final blended toward Kev-27B's own weights, to keep the broad gains and recover what Kev-27B does best.

## Round 23 (registered)

### Round 23 - post-hoc blends of round 22's final SFT checkpoint toward Kev-27B's own weights, under round 24's audited rule (re-registered with this spec's commit, 2026-09-28, written before any round-23 interpolation or read)

**Re-registration.** Round 23 was first registered on 2026-09-27 (PR #165, commit `9345b17`) with round 22's rule, which
included scienthoon and the pooled externals. That version was never launched: nothing was interpolated and nothing was
read. It is superseded by this one. The candidates, the interpolation tool and the temperature pool are unchanged. The
rule, the reads, the parent reads and the confirmation are now **round 24's, verbatim**. Scienthoon is gone
(`kev.suite.REMOVED_SUITES` refuses it from round 23; "scienthoon removed"). This section and `experiments/rounds/r23.json`
were written before any round-23 read. They were written after round 24's confirmation, and the design reacts to it (see
"Reuse of the test partitions" below).

**Why.** Round 24 named round 22's final checkpoint as its candidate and it was **not confirmed** ("Round 24
confirmation"). Its gains held on every untouched test partition: breadth-v1 +1.5 [+0.5, +2.5], tasksource-heldout-v1
+5.3 [+3.7, +7.0], pooled hard/devtools/documents-v1 +8.6 [+7.1, +10.0], documents-v2 +3.8, test breadth index 53.7
against Kev-27B's 50.2 and Jev's 54.0. It failed where Kev-27B is strongest. Locked transfer-v4 was 0.8841 against the
0.886 bar, 2 questions short (Kev-27B 0.8963). On longdoc CUAD test it was worse: −1.8 [−3.2, −0.5], ECE 0.055 against
0.007. Short states and contract documents are exactly what Kev-27B does well, so blending the SFT toward Kev-27B should
keep most of the broad gains and recover the short-state and CUAD behaviour.

Round 20 blended round 19's SFT checkpoints toward the **base**, and scienthoon did not recover (for arm (a) it got worse
toward the base). But scienthoon was the guard that failed there, and it has since been removed as unsound. The base is
also not the model that is good on short states and CUAD; Kev-27B is. Both checkpoints are fine-tunes of the same weights
(`Qwen/Qwen3.8-27B` @ `1d4bf0f2`): the SFT moved every weight (full weights, lr 2e-6, one epoch), and Kev-27B added a
rank-16 LoRA on every linear projection. Averaging fine-tunes of one initialisation is weight averaging in the sense of
"model soups" (Wortsman et al., 2022). It is also WiSE-FT with a fine-tuned endpoint in place of the zero-shot one. The
risk is a loss barrier between two fine-tunes that moved apart. That would show at α 0.50 first; α 0.85 and 0.70 stay
close to the SFT.

**Heads.** A blended backbone needs a pointer head. The SFT's head was trained on the SFT backbone and Kev-27B's on its
own, so neither matches a blend. Neither choice is clearly right, so both are registered: keep the SFT head (most of each
blend is the SFT), or blend the heads with the same α.

**Candidates** (spec `experiments/rounds/r23.json`; parent Kev-27B, `r6-27b-v2/01-trial-1`, Hub `01b81998`). Every
candidate is selectable and there are no reference arms: α = 1 is round 22's final (round 24's candidate), and α = 0 with
the blended head is Kev-27B.

| arm | checkpoint | weight on the SFT backbone | head |
|---|---|---|---|
| `27b-k-w85`, `27b-k-w70`, `27b-k-w50` | `/runs/r23-wise/27b-k-w{85,70,50}/checkpoint` | 0.85 / 0.70 / 0.50 | the SFT's |
| `27b-kh-w85`, `27b-kh-w70`, `27b-kh-w50` | `/runs/r23-wise/27b-kh-w{85,70,50}/checkpoint` | 0.85 / 0.70 / 0.50 | blended with the same α |

SFT endpoint: round 22's final checkpoint, `/runs/r22-27b-lr2e6/00-trial-0/checkpoint` (full weights, bf16). Other
endpoint: `jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195`, a LoRA adapter on the same base and revision.

**Interpolation (tool change with this PR; unchanged from the first registration).** `scripts/interpolate_checkpoint.py
--toward <checkpoint>` (`modal_app.py::interpolate --toward ... [--blend-head]`) takes another checkpoint of the SFT's base
and revision as the other endpoint, instead of the base.
- A full-weight checkpoint is streamed from its shards.
- A LoRA checkpoint is merged in fp32 as W + delta. W is the base built as its loader builds it (bf16 values upcast
  exactly). delta is peft's own `get_delta_weight` for every adapted layer: the value peft's merge adds, not rounded
  before the blend (the served Kev-27B, merged and fused, holds round(W + delta)).
- Each output tensor is α · SFT + (1 − α) · other in fp32, rounded once to bf16, as in round 20.
- Refused before anything is written: another base or revision; any tensor name or shape that differs between the two
  backbones; DoRA or other LoRA variants, LoRA biases, `modules_to_save` and trained token embeddings (they change weights
  outside W + delta).
- `--blend_head` (needs `--toward`) blends the pointer heads in fp32 with the same α. It refuses heads with different
  tensors, shapes, dtypes, `head_dim`, `option_isolation` or `special_embeddings`. Kev-27B's and the SFT's heads are both
  q/k 256 × 5120 with biases, fp32.
- `head.pt` keeps the SFT's meta and records `interpolation: {alpha, sft: {path, weights_sha256}, base, toward: {path,
  resolved, kind, weights_sha256, head_sha256, merge, adapted_tensors}, head: {kind: sft | blend, sft: {head_sha256,
  temperature}, toward: {head_sha256, temperature}}}`.
- Memory is round 20's (the base resident in bf16, the adapter small, fp32 temporaries per tensor), so `kev.budget`'s
  interpolation resources are unchanged.

Tests on the random two-layer Qwen3.5 (`tests/test_unit.py`):
- Toward a LoRA checkpoint, α = 1 is the SFT exactly.
- α = 0 with `--blend_head` is the LoRA model exactly: the backbone is its fp32 `merge_and_unload` rounded once to bf16,
  equal tensor for tensor to the checkpoint loaded merged, and the head is the LoRA's.
- α = 0.5 is the fp32 midpoint of the SFT and the unrounded merge, with the heads averaged; without `--blend_head` it is
  the same backbone with the SFT head.
- Toward a full checkpoint, α = 0 is its backbone and head exactly.
- Refusals: a base-revision mismatch, `--blend_head` without `--toward`, a head of another shape, a renamed tensor.

**Temperature (MUST, `docs/autoresearch.md` section 3): round 24's pool, with its interval.** Every candidate is served at
the temperature fitted on its own rows of transfer-r3 calibration (read `r3cal`, the eight held-out sources, `emotion`
included as in round 24: 448 questions) plus transfer-v9 development MMLU-Pro (read `v9`: 200 questions), minus its
transfer-v4 development records. Kev-27B is served at its shipped 1.38. Reported, not gating: each candidate's 90 %
bootstrap interval of its pooled T (`temperature.ci`: 2,000 resamples of (source, group) clusters within each source,
seed 0, the same grid and objective). A blend is trained on both endpoints' data, so each candidate names `trained_on:
[evals/sft-v2-r22, evals/v7/decision-v7, evals/round6/b1v2]`: round 22's training suite, plus Kev-27B's decision-v7 (whose
manifest its provenance hashes) and its `data` file's suite b1v2. `kev.rounds validate` checks the pool against all of
them and finds no pooled suite, source or training partition shared.

**Reads.** Round 24's 16 per candidate, same tags and suites: breadth, tsheld, hard, devtools, docs, transfer4, semif,
wanli2, typesafe, v9, r3test, r3cal, longdoc, ood, agentsood, guardood. There is no scienthoon read. `read_timeout` is
`{"27b": 14400}`, and each candidate's "transfer" rows are its own `transfer4` read (round-level `transfer_read`).
Kev-27B's reads are round 24's, the same files: its committed reads plus round 21's four parent reads
`runs/r21-P27-{tsheld,ood,agentsood,guardood}`. The tsheld read and the OOD rows are restored from the private dataset with
`scripts/private_rows.py restore --manifest runs/r22-readout/private-rows.json`.

**Rule: round 24's, verbatim.** The spec's `rule`, `reads`, `temperature` and `parents` equal `r24.json`'s. Everything is
against Kev-27B with paired record-clustered bootstraps (2,000 resamples, seed 0, micro), every candidate at its pool
temperature, Kev-27B at 1.38, and `drop_ids` as before. The exclusions leave both sides:
- breadth-v1 without `routerbench`, `cfcolor`, `humicroedit`, `chessbench`;
- tasksource-heldout-v1 without the seven families coded T11 T12 T14 T15 T18 T22 T24, read from the private exclusion
  file `runs/r24-private/tsheld-exclude.json` (registered by path and sha256 `a72030ab…`; restore with
  `scripts/private_rows.py restore --manifest runs/r24-readout/private-exclude.json`; `kev.rounds` refuses a file whose
  hash differs and reports the panel missing without it);
- the Kev panel and short state without `emotion`; devtools-v1 without `flakeflagger` and task `commitpackft_type`.

1. primaries: breadth-v1 dev accuracy lower > 0; tasksource-heldout-v1 dev accuracy lower > 0; Kev panel (transfer-v4
   dev, hard-v1, devtools-v1, documents-v1) accuracy lower ≥ −1 pp;
2. guards: short state (transfer-v4 dev + transfer-r3 test) accuracy lower ≥ −2 pp, Brier upper ≤ +0.02, confident
   errors upper ≤ +1 pp; unknowable share on transfer-v9 ≤ 0.05; longdoc CUAD accuracy lower ≥ −2 pp and CUAD 16k+
   accuracy candidate − Kev-27B ≥ −2 pp;
3. calibration: breadth, Kev-panel and tasksource-heldout ECE ≤ Kev-27B's + 0.01 (same exclusions);
4. candidate: the passing checkpoint with the largest breadth + tasksource-heldout + Kev-panel accuracy gain. The read-out
   says how many of the six passed.

Report only (`optional` panels, never gating, never make a candidate incomplete):
- SemIf, WANLI-v2 and TypeSafe. There is no pooled-externals guard and no scienthoon read.
- CUAD ECE by length, 16k+ included, and longdoc generated.
- ood-v2, agents-ood-v1 and guardrails-ood-v1.
- hard-v1 alone, and the Kev panel without hard-v1.
- breadth over all 14 sources and over the three moved sources; tasksource-heldout over all 24 families; the
  chance-corrected breadth index (`scripts/breadth_report.py`).

A note on #167's standing rule. #167 wrote that from round 23 the pooled external guard would be SemIf + WANLI-v2 + TypeSafe.
The audit found that panel unsound as a gate: without scienthoon it is 81 % WANLI, split-half r 0.08. Round 24 therefore
dropped it, and this round follows round 24, so the three suites are reported, not gated. The standing rule is updated
with this PR.

**Confirmation: round 24's, identical** (the named candidate only, each read once; none if there is no candidate):
- `tests`, each a paired lower bound against Kev-27B:
  - breadth-v1 test accuracy lower > 0 (same four sources out; all 14 reported);
  - tasksource-heldout-v1 test lower > 0 (same seven families out);
  - pooled hard-v1 + devtools-v1 (without `flakeflagger`, `commitpackft_type`) + documents-v1 test lower ≥ −1 pp;
  - reported: documents-v2, and longdoc-v1 test (CUAD by length, generated).
  - Candidate reads go to `runs/r23c-27b-cand-<tag>`. Kev-27B's side is **reused, not re-read**: its test reads from
    round 24's confirmation, `runs/r24c-27b-parent-<tag>` (the same checkpoint, suites and served T; private ones restored
    from `runs/r24-verdict/private-rows{,-longdoc}.json`).
  - The test breadth index compares with round 24's Jev and AutoJev reads (`runs/r24c-{jev,autojev}-breadthtest`),
    which are not read again.
- `locked`: locked transfer-v4 accuracy ≥ 0.886 and served Brier ≤ 0.165 (`runs/locked/kev-27b-r23-ungated`).
- Before any release: the bf16 serving check as round 24 ran it (`modal_app.py::serving --gpu H200 --flags=--isolation`:
  max |Δp| ≤ 0.03, ≤ 1 flip in 280; plus the long-state run at 8k / 32k / 64k with the same bars). The release
  temperature is the pool fit written by `scripts/calibrate_checkpoint.py` on the candidate's r3cal + v9 rows, with its
  interval reported, and the test-panel ECEs at the ends of that interval are reported as in round 24.

**Reuse of the test partitions (stated plainly).** Round 23's confirmation reads the same test partitions and the same
locked transfer-v4 that round 24's candidate and Kev-27B were read on. This round's design was chosen after seeing that
confirmation: blend toward Kev-27B because the candidate lost on locked short states and CUAD test. So these partitions
are no longer untouched for this question, and a pass would be weaker evidence than round 24's reads were. The bars are
round 24's, fixed here before any round-23 read. Only one candidate is confirmed, and each read is made once. A pass
should be reported with this caveat, and a release decision is Jared's.

**Budget.** Reference point: metered ~$3,790 at this registration, plus ~$80 of metering lag, so ~$3,870. The ceiling is
$5,000 metered, with the ~10 % reserve ($500) kept at every launch.

| item | admission bound | expected |
|---|---|---|
| spend so far (metered + lag) | ~$3,870 | ~$3,870 |
| two interpolations (CPU, 8 cores / 128 GiB / 3 h; `kev.budget.interpolation_bound`) | $8.41 | ~$3 (round 20: 150-230 s per α, plus loading) |
| reads, one candidate at a time (16 × $25.06, H200, 4 h bound each) | $400.96 at any moment; $2,405.76 over six | ~$30 a candidate incl. longdoc (~2.8 h), ~$180-250 over six |
| **projection at the rule stage** | **$4,279.37** at any moment | **~$4,055-4,125** |
| reserve left of $5,000 | $720.63 (14.4 %) | ~$875-945 (17.5-19 %) |
| confirmation, candidate only (tests 7 reads × $25.06, the locked read at 4 h, the serving check and long-state run) | $175.42 + $25.06 + $12.54 = $213.02 | ~$45-90 |

The six candidates' read bounds together do not fit under the ceiling, so reads launch **one candidate at a time**, in
the order `27b-k-w85`, `27b-kh-w85`, `27b-k-w70`, `27b-kh-w70`, `27b-k-w50`, `27b-kh-w50`. Each batch starts only after
the previous one has landed, the metered cost has been read again, and metered spend + lag is below $4,099.04 ($5,000 −
$500 − $400.96). Confirmation starts only while metered spend + lag + $213.02 ≤ $4,500. Expected at confirmation:
~$4,125 + $90 ≈ $4,215.

**Run steps** (after this PR is merged; nothing is launched before):
1. Restore the private inputs with `uv run python scripts/private_rows.py restore --manifest <m>`, for
   `runs/r22-readout/private-rows.json` (round 21's parent reads) and `runs/r24-readout/private-exclude.json` (the
   exclusion list). Then `uv run python -m kev.rounds validate experiments/rounds/r23.json` and read the metered cost.
2. Run `KEV_APP_NAME=kev-sft uv run modal run --detach modal_app.py::interpolate --sft
   /runs/r22-27b-lr2e6/00-trial-0/checkpoint --toward jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195
   --prefix 27b-k --study r23-wise`, and 60 s later the same with `--blend-head --prefix 27b-kh`.
3. When both have written `runs/r23-wise/<arm>/interpolation.json`, run `uv run python -m kev.rounds launch-reads
   experiments/rounds/r23.json --arms <arm>` one candidate at a time, in the order above, reading the metered cost before
   each.
4. `readout`, then confirmation as written. For the tests stage, restore `runs/r24-verdict/private-rows.json` and
   `private-rows-longdoc.json` first, so that Kev-27B's side is in place and is not relaunched.

What may be committed: as round 24. Public-suite rows and reports are committed. The tasksource-heldout-v1 reads, the
ood / agents-ood / guardrails-ood rows, and private test rows go to the private dataset with `scripts/private_rows.py`,
never to git, and the family names are never written publicly.

### Round 23 result

**Candidate: `27b-k-w85`** (0.85 · round 22's final + 0.15 · Kev-27B merged in fp32, the SFT's pointer head;
`/runs/r23-wise/27b-k-w85/checkpoint`). **5 of 6** candidates pass the registered rule; `27b-k-w50` fails breadth ECE
(0.0195 against the bar 0.0176). The rank (breadth + tasksource-heldout + Kev-panel accuracy gain) orders the passing five
`27b-k-w85` +13.8, `27b-kh-w85` +13.7, `27b-k-w70` +13.6, `27b-kh-w70` +13.5, `27b-kh-w50` +12.8. Read-out:
`runs/r23-readout/round23.json` (`python -m kev.rounds readout experiments/rounds/r23.json`, run 2026-09-28T20:31Z after
the last read landed), table `runs/r23-readout/readout.txt`, markdown `runs/r23-readout/tables.md`. Every candidate is served
at its pooled T (648 questions, none excluded as a transfer-v4 duplicate) and Kev-27B at 1.382. Deltas are paired
record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro): accuracy and confident errors in pp, Brier
absolute, ECE as served against its bar. The exclusions removed the same questions as round 24: 600 breadth, 795
tasksource-heldout, 380 Kev-panel and 220 short-state, on both sides.

**Interpolations** (`modal_app.py::interpolate`, 8-CPU containers, 304-398 s per α; `runs/r23-wise/<arm>/interpolation.json`).
SFT endpoint weights sha256 `3fa0182a…`, head `bfcf801e…` (T 1.0). Kev-27B endpoint: kind `lora`, merged in fp32 as base +
peft `get_delta_weight` (496 adapted tensors of 850), weights `41bf5af0…`, head `1322189d…` (T 1.382). Blended backbones:
α 0.85 `d27af6ab…`, α 0.70 `58332c7c…`, α 0.50 `7893a092…`; the `k` and `kh` arms of one α share the backbone
bit for bit. Checked locally on the pulled `head.pt` files: every `k` head equals the SFT head exactly (max |Δ| 0), every `kh`
head equals α · SFT + (1 − α) · Kev-27B in fp32 exactly, and `head.pt["interpolation"]` matches `interpolation.json` and both
endpoint head hashes.

| criterion | k-w85 | k-w70 | k-w50 | kh-w85 | kh-w70 | kh-w50 |
|---|---|---|---|---|---|---|
| T (pool, 648) [90 % CI] | 1.320 [1.203, 1.447] | 1.203 [1.097, 1.320] | 1.047 [0.955, 1.149] | 0.955 [0.871, 1.047] | 0.660 [0.602, 0.724] | 0.536 [0.489, 0.588] |
| 1 breadth acc, lower > 0 (2475) | +1.5 [+0.6, +2.4] | +1.5 [+0.6, +2.2] | +1.3 [+0.5, +2.0] | +1.6 [+0.7, +2.5] | +1.3 [+0.4, +2.1] | +1.7 [+0.9, +2.4] |
| 1 tasksource-heldout acc, lower > 0 (1993) | +4.2 [+2.5, +5.8] | +3.7 [+2.2, +5.3] | +3.3 [+1.9, +4.7] | +4.0 [+2.4, +5.6] | +3.8 [+2.3, +5.4] | +3.6 [+2.2, +4.9] |
| 1 Kev panel acc, lower ≥ −1 (3351) | +8.1 [+7.0, +9.4] | +8.4 [+7.4, +9.6] | +7.5 [+6.5, +8.6] | +8.1 [+7.0, +9.4] | +8.4 [+7.3, +9.6] | +7.6 [+6.6, +8.7] |
| 2 short acc, lower ≥ −2 (1586) | −0.9 [−1.95, +0.1] | −0.9 [−1.95, +0.0] | −0.8 [−1.7, +0.0] | −0.9 [−1.9, +0.1] | −0.8 [−1.8, +0.2] | −0.5 [−1.4, +0.4] |
| 2 short Brier, upper ≤ +0.02 | −0.000 [−0.009, +0.009] | +0.000 [−0.008, +0.008] | +0.000 [−0.007, +0.009] | −0.001 [−0.009, +0.008] | −0.001 [−0.009, +0.008] | −0.001 [−0.008, +0.007] |
| 2 short confident errors, upper ≤ +1 | −1.1 [−1.8, −0.4] | −0.5 [−1.1, +0.1] | −0.2 [−0.8, +0.3] | −0.9 [−1.7, −0.3] | −0.4 [−1.0, +0.2] | −0.1 [−0.7, +0.4] |
| 2 CUAD acc, lower ≥ −2 (2254) | +0.8 [−0.2, +1.8] | +0.7 [−0.1, +1.6] | +0.6 [+0.0, +1.4] | +0.8 [−0.2, +1.8] | +0.7 [−0.1, +1.6] | +0.6 [−0.1, +1.4] |
| 2 CUAD 16k+ acc, cand − parent ≥ −2 | 0.839 vs 0.837 | 0.840 vs 0.837 | 0.840 vs 0.837 | 0.839 vs 0.837 | 0.840 vs 0.837 | 0.841 vs 0.837 |
| 2 unknowable share ≤ 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 breadth ECE ≤ 0.0176 (Kev-27B 0.0076) | 0.0103 | 0.0156 | 0.0195 **fail** | 0.0117 | 0.0159 | 0.0136 |
| 3 Kev-panel ECE ≤ 0.0327 (Kev-27B 0.0227) | 0.0146 | 0.0151 | 0.0189 | 0.0115 | 0.0150 | 0.0258 |
| 3 tasksource-heldout ECE ≤ 0.0523 (Kev-27B 0.0423) | 0.0432 | 0.0315 | 0.0311 | 0.0401 | 0.0306 | 0.0358 |
| rank score (Δ breadth + Δ tsheld + Δ Kev, pp) | +13.8 | +13.6 | +12.1 | +13.7 | +13.5 | +12.8 |
| verdict | **PASS** | **PASS** | fail | **PASS** | **PASS** | **PASS** |

The closest calls: short-state accuracy lower bounds of −1.95 pp for `27b-k-w85` and `27b-k-w70` against −2 (the same
−1.95 as round 22's final in round 24), and `27b-k-w70`'s breadth ECE 0.0156.

**Report-only panels** (paired deltas against Kev-27B; ECE absolute where marked):

| panel | k-w85 | k-w70 | k-w50 | kh-w85 | kh-w70 | kh-w50 |
|---|---|---|---|---|---|---|
| breadth, all 14 sources, acc (3075) | +1.2 [+0.3, +2.2] | +1.1 [+0.3, +2.1] | +1.1 [+0.3, +1.8] | +1.2 [+0.3, +2.2] | +1.1 [+0.2, +2.0] | +1.5 [+0.7, +2.2] |
| breadth cfcolor+humicroedit+chessbench, acc (450) | −1.1 [−5.1, +2.9] | −0.7 [−4.2, +2.9] | +0.2 [−2.7, +3.1] | −1.1 [−5.1, +2.7] | −0.4 [−4.0, +2.9] | +0.2 [−2.4, +2.9] |
| tasksource-heldout all 24 families, acc (2788) | +3.9 [+2.6, +5.3] | +3.7 [+2.5, +5.0] | +3.4 [+2.2, +4.5] | +3.8 [+2.4, +5.1] | +3.7 [+2.5, +5.0] | +3.3 [+2.2, +4.4] |
| hard-v1 acc (1083) | +17.9 [+15.3, +20.8] | +18.1 [+15.5, +20.8] | +16.1 [+13.5, +18.9] | +17.8 [+15.2, +20.7] | +18.1 [+15.5, +20.9] | +16.0 [+13.4, +18.8] |
| Kev panel without hard-v1, acc (2268) | +3.5 [+2.3, +4.7] | +3.8 [+2.7, +4.9] | +3.5 [+2.5, +4.5] | +3.5 [+2.4, +4.7] | +3.7 [+2.7, +4.8] | +3.6 [+2.6, +4.6] |
| SemIf acc (144) | −0.7 [−3.5, +2.1] | −1.4 [−4.2, +1.4] | +0.0 [−2.8, +2.8] | −0.7 [−3.5, +2.1] | −1.4 [−4.2, +1.4] | +0.0 [−2.8, +2.8] |
| WANLI-v2 acc (1002) | +1.2 [−0.7, +3.0] | +1.7 [+0.0, +3.4] | +2.3 [+0.9, +3.7] | +1.2 [−0.7, +3.0] | +1.8 [+0.1, +3.5] | +2.3 [+0.9, +3.7] |
| TypeSafe acc (89) | −1.1 [−5.1, +2.5] | −1.1 [−5.1, +2.5] | +0.0 [−3.4, +3.2] | −1.1 [−5.1, +2.5] | −1.1 [−5.1, +2.5] | +0.0 [−3.4, +3.2] |
| ood-v2 acc (4988) | +1.2 [+0.7, +1.8] | +1.3 [+0.8, +1.8] | +1.1 [+0.6, +1.5] | +1.2 [+0.7, +1.7] | +1.3 [+0.8, +1.8] | +1.1 [+0.7, +1.5] |
| agents-ood-v1 acc (2084) | +2.1 [+1.4, +2.8] | +2.1 [+1.4, +2.8] | +1.6 [+1.0, +2.3] | +2.1 [+1.4, +2.8] | +2.1 [+1.4, +2.8] | +1.6 [+1.0, +2.2] |
| guardrails-ood-v1 acc (4949) | +4.0 [+3.4, +4.6] | +3.8 [+3.2, +4.5] | +3.7 [+3.1, +4.3] | +4.0 [+3.4, +4.7] | +3.9 [+3.3, +4.6] | +3.7 [+3.1, +4.3] |
| ood-v2 ECE Δ | −0.024 [−0.029, −0.018] | −0.023 [−0.028, −0.018] | −0.022 [−0.027, −0.017] | −0.025 [−0.030, −0.019] | −0.022 [−0.027, −0.017] | −0.016 [−0.020, −0.011] |
| agents-ood-v1 ECE Δ | −0.105 [−0.113, −0.097] | −0.101 [−0.109, −0.093] | −0.092 [−0.098, −0.084] | −0.105 [−0.113, −0.097] | −0.099 [−0.107, −0.091] | −0.080 [−0.087, −0.072] |
| guardrails-ood-v1 ECE Δ | −0.069 [−0.074, −0.061] | −0.070 [−0.075, −0.063] | −0.066 [−0.071, −0.058] | −0.069 [−0.074, −0.062] | −0.069 [−0.074, −0.061] | −0.060 [−0.065, −0.052] |
| breadth ECE Δ (gated panel) | +0.003 [−0.008, +0.013] | +0.008 [−0.007, +0.015] | +0.012 [−0.005, +0.017] | +0.004 [−0.008, +0.014] | +0.008 [−0.006, +0.017] | +0.006 [−0.008, +0.013] |
| Kev ECE Δ | −0.008 [−0.021, +0.004] | −0.008 [−0.019, +0.003] | −0.004 [−0.017, +0.007] | −0.011 [−0.022, +0.002] | −0.008 [−0.019, +0.003] | +0.003 [−0.009, +0.013] |
| tsheld ECE Δ | +0.001 [−0.019, +0.020] | −0.011 [−0.029, +0.007] | −0.011 [−0.026, +0.006] | −0.002 [−0.022, +0.016] | −0.012 [−0.029, +0.006] | −0.006 [−0.021, +0.012] |
| CUAD ECE (Kev-27B 0.063) | 0.097 | 0.095 | 0.096 | 0.097 | 0.097 | 0.089 |
| CUAD ECE 16k+ (Kev-27B 0.071) | 0.102 | 0.100 | 0.100 | 0.102 | 0.101 | 0.099 |
| longdoc generated acc / ECE | 1.000 / 0.001 | 1.000 / 0.001 | 1.000 / 0.002 | 1.000 / 0.001 | 1.000 / 0.002 | 1.000 / 0.004 |
| short-state acc, candidate / Kev-27B | 0.8947 / 0.9042 | 0.8947 / 0.9042 | 0.8960 / 0.9042 | 0.8953 / 0.9042 | 0.8966 / 0.9042 | 0.8991 / 0.9042 |

**What the blends did.** Very little, even at α 0.50. Every candidate reads almost like round 22's final (round 24: breadth
+1.5, tasksource-heldout +4.1, Kev +7.9, short −0.9 [−2.0, +0.2], CUAD ECE 16k+ 0.102). The two things the blend was meant to
recover barely move. Short-state accuracy goes from −0.9 to −0.5 pp at α 0.50 with the blended head. CUAD ECE goes from 0.097 (α 0.85)
to 0.089 (α 0.50, blended head), against Kev-27B's 0.063; at 16k+ it stays at 0.099-0.102 against 0.071. Half of Kev-27B's weights brings back almost
none of its short-state or contract-document behaviour. The accuracy gains mostly survive the blend (hard-v1 +16 to +18 pp,
agents-ood and guardrails-ood unchanged). The blended heads need much lower temperatures (0.955 → 0.536 as α falls, against
1.320 → 1.047 with the SFT head): averaging two heads trained on different backbones shrinks the pointer logits. Pooled T
recalibrates that, and the `kh` and `k` arms of one α score within 0.4 pp of each other on every gated accuracy panel.

**Deviations.** (i) App name: the spec's `"app": "kev-sft"` was not used. The interpolations ran with `KEV_APP_NAME=kev-r23`.
The reads ran through a wrapper that calls `kev.rounds.launch_arm_reads` on an in-memory copy of the spec with
`app = kev-r23`: the same `read_commands`, the same launch records `runs/r23-reads-<arm>.json` and the same per-arm lock.
The spec file is unchanged. (ii) The registration says one candidate at a time. From 12:40Z, on Jared's instruction to go
faster, two candidates' reads were in flight at once: `27b-kh-w85` was launched while `27b-k-w85`'s reads were still running,
and after that each new batch started when one landed. Every launch still read the metered cost first and was below the gate
(metered + $80 < $4,099.04). Launch times and readings are in the launch records and below. (iii) The tool counted 16 reads
per candidate, as the spec lists (the run request said 17).

**Spend.** Metered (Modal, workspace-wide): $3,787.06 before the interpolations, then $3,782.91 at the first read launch
(revised down). Later readings before each launch: $3,806.93, $3,842.55, $3,871.84, $3,887.04 and $3,922.27. $3,961.81
after the last read landed (2026-09-28T20:30Z). The rule stage cost about $175 metered so far, before metering lag.

**Evidence** (committed): the read-out, its tables and the launch records; each candidate's `interpolation.json`; the
public-suite reads `runs/r23-27b-<arm>-<tag>` (report + rows; OOD reads: reports only). Private (`jaredpalmer/kev-private-train`
under `runs/r23/`, `runs/r23-readout/private-rows.json` @ `4e8c117e`, `scripts/private_rows.py restore`): the six
tasksource-heldout-v1 reads (rows and reports) and the ood / agents-ood / guardrails-ood rows (30 files).

**Next.** The registered confirmation of `27b-k-w85`, each read once. Tests stage: candidate test reads to
`runs/r23c-27b-cand-<tag>`, with Kev-27B's side reused from round 24. Then the locked stage (`kev-27b-r23-ungated`), the bf16
serving check and the release temperature. The same caveat applies as for round 24: the candidate's short-state development
read is round 22's final's, which missed the locked bar by 2 questions.

### Round 23 confirmation

**CONFIRMED, with the registered caveat.** `27b-k-w85` passes both stages. It passes the tests stage on all three criteria
and the locked stage on both. Locked transfer-v4 accuracy is **0.8887 (583 of 656)** against the bar of 0.886, which needs
582, so it clears by **1 question**. Kev-27B scores 0.8963 (588) on the same read, and round 22's final scored 0.8841 (580)
in round 24. The test partitions and the locked read are the ones round 24 read, and this round's design reacted to that
confirmation ("Reuse of the test partitions" above). A pass here is weaker evidence than round 24's reads would have been,
and the release decision is Jared's. Nothing is published. Each confirmation read was made once, for the named candidate
only. Verdicts: `runs/r23-verdict/27b-tests.json` and `runs/r23-verdict/27b-locked.json` (`python -m kev.rounds confirm
experiments/rounds/r23.json --stage {tests,locked} --arm 27b-k-w85`). The candidate is served at its pool fit, 1.320 (648
questions), and Kev-27B at its shipped 1.382. Kev-27B's test side is round 24's reads (`runs/r24c-27b-parent-<tag>`), not
read again. Deltas are paired record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro), in pp.

| stage / criterion (panel, n) | 27b-k-w85 | Kev-27B | Δ [95 %] | verdict |
|---|---|---|---|---|
| tests: breadth-v1 test acc, lower > 0 (without `routerbench`, `cfcolor`, `humicroedit`, `chessbench`; 2,489, 600 out) | 0.832 | 0.820 | +1.2 [+0.3, +2.2] | pass |
| tests: tasksource-heldout-v1 test acc, lower > 0 (without the seven families; 2,024, 809 out) | 0.795 | 0.743 | +5.3 [+3.7, +6.8] | pass |
| tests: pooled hard-v1 + devtools-v1 (without `flakeflagger`, `commitpackft_type`) + documents-v1 test acc, lower ≥ −1 (2,795, 300 out) | 0.889 | 0.800 | +8.9 [+7.5, +10.3] | pass |
| **locked: transfer-v4 locked acc ≥ 0.886 (656)** | **0.8887 (583)** | 0.8963 (588) | −0.8 [−2.0, +0.5] | **pass** (needed 582) |
| locked: served Brier ≤ 0.165 (656) | 0.1537 | 0.1604 | −0.007 [−0.018, +0.004] | pass |

**Report-only test reads** (same exclusions; ECE as served):

| panel (n) | 27b-k-w85 | Kev-27B | Δ [95 %] |
|---|---|---|---|
| hard-v1 test (1,088) | 0.918 | 0.749 | +16.9 [+14.2, +19.8] |
| devtools-v1 test, gated sources (771) | 0.825 | 0.789 | +3.6 [+1.3, +5.9] |
| documents-v1 test (936) | 0.908 | 0.869 | +4.0 [+2.1, +5.9] |
| documents-v2, private held-out test (953) | 0.921 | 0.881 | +4.0 [+2.0, +6.1] |
| breadth-v1 test, all 14 sources (3,089); ECE | 0.757; 0.019 | 0.749; 0.019 | +0.8 [−0.1, +1.8] |
| breadth-v1 test ECE, gated panel | 0.015 | 0.013 | - |
| tasksource-heldout-v1 test ECE, gated panel | 0.049 | 0.051 | - |
| longdoc-v1 test CUAD acc (2,194) | 0.874 | 0.890 | **−1.6 [−3.0, −0.3]** |
| longdoc-v1 test CUAD ECE | **0.053** | 0.007 | - |
| CUAD by length, acc / ECE: < 8k (867) | 0.874 / 0.059 | 0.900 / 0.025 | - |
| 8k-16k (443) | 0.880 / 0.052 | 0.892 / 0.028 | - |
| 16k-32k (442) | 0.873 / 0.061 | 0.882 / 0.019 | - |
| 32k-64k (442) | 0.867 / 0.060 | 0.876 / 0.014 | - |
| 16k+ (884) | 0.870 / 0.059 | 0.879 / 0.013 | - |
| longdoc-v1 test generated (2,400); ECE | 1.000; 0.001 | 1.000; 0.014 | - |
| decision-v7 locked test acc (1,200; read by `locked_test`, not a criterion) | 0.866 | 0.870 | - |

Against round 22's final on the same partitions (round 24's confirmation), the blend is within noise everywhere. It is a
little lower on breadth-v1 test (+1.2 against +1.5 over Kev-27B) and on the breadth index. It is a little higher on the pooled
panel (+8.9 against +8.6) and on locked transfer-v4 (583 against 580). CUAD test is still worse than Kev-27B: −1.6 [−3.0,
−0.3] against round 22's −1.8, ECE 0.053 against 0.055. The development rule's CUAD guard (lower ≥ −2 pp) would not hold on
test (−3.0). That is report only in this stage and changes no verdict. It is the contract-domain miscalibration round 24
found, and the blend did not fix it.

**Breadth index on TEST** (chance-corrected, all 14 sources; `scripts/breadth_report.py --bootstrap 2000`,
`runs/r23-breadth-report/report.md`; the candidate's rows served at 1.320, Kev-27B's, Jev's and AutoJev's as round 24 read
them, none read again): candidate **52.3 [49.2, 55.4]**, Kev-27B 50.2 [47.0, 53.2], round 22's final 53.7 [50.5, 56.7] (served
at 1.382; this reproduces round 24's report exactly), Jev 54.0 [51.2, 57.0], AutoJev 50.0 [47.0, 53.3]. Against Kev-27B the
candidate is +2.1 [−0.4, +4.6]; round 22's final was +3.5 [+1.0, +6.0]. Pooled ECE: candidate 0.019, Kev-27B 0.019. The
candidate's gain is Retrieval & Classification (69.0 against 61.6) and Knowledge & Reasoning (33.8 against 31.7). It gives
back Tools & Automation (62.9 against 65.1).

**Calibration at the ends of the temperature interval** (report only, as in round 24: the candidate at the ends of its pooled
T's 90 % interval [1.203, 1.447] and at the point fit, Kev-27B at 1.382; `runs/r23-verdict/27b-tests-t-interval.json`).
ECE / Brier:

| test panel (n) | T 1.203 | T 1.320 (served) | T 1.447 | Kev-27B @ 1.382 |
|---|---|---|---|---|
| breadth-v1, gated (2,489) | 0.026 / 0.2358 | 0.015 / 0.2347 | 0.021 / 0.2346 | 0.013 / 0.2516 |
| breadth-v1, all 14 (3,089) | 0.029 / 0.3130 | 0.019 / 0.3120 | 0.020 / 0.3118 | 0.019 / 0.3267 |
| tasksource-heldout-v1, gated (2,024) | 0.034 / 0.2970 | 0.049 / 0.2998 | 0.067 / 0.3038 | 0.051 / 0.3682 |
| pooled hard + devtools + documents-v1 (2,795) | 0.019 / 0.1578 | 0.014 / 0.1584 | 0.024 / 0.1596 | 0.027 / 0.2720 |

Round 24's pattern holds. At the low end breadth ECE rises to 0.026, and at the high end tasksource-heldout ECE rises to
0.067. The two pull in opposite directions, and Brier moves by at most 0.004.

**Serving check** (bf16, H200; passed; `runs/serving-27b-r23/report.json`, `runs/serving-27b-r23-long/report.json`):
- Short states, 280 questions:
  - CUDA graphs vs fp32: max |Δp| 0.0223, 0 flips. Eager bf16 vs fp32: 0.0216, 0 flips. Bars: ≤ 0.03, ≤ 1 flip.
  - Graphs vs eager: 0.0270, 0 flips. Question isolation as served: 0.0039, 0 flips.
- Long states, served vs benchmark, 3 records / 15 questions per length:
  - 8k: max |Δp| 0.0064, 0 flips; new state 1.37 s, cached 599 ms; 70 GB peak.
  - 32k: 0.0095, 0 flips; 4.17 s / 630 ms; 79 GB.
  - 64k: 0.0017, 0 flips; 9.41 s / 733 ms; 87 GB.

**Release temperature** (on a copy of `head.pt`; the checkpoint on the volume is unchanged):
`scripts/calibrate_checkpoint.py --rows <r3cal rows>:composition_holdout,emotion,legacy_holdout,mmlu,paws,qnli,sciq,tweet_offensive
--rows <v9 rows>:mmlu_pro --exclude_rows <transfer4 rows>` gives **1.3195**, the read-out's pool fit. Its 90 % interval is
[1.203, 1.447] (648 questions; (source, group) clusters resampled within each source, 2,000 resamples, seed 0). On the fit rows,
ECE goes 0.048 → 0.034. The out-of-fold estimate (5 group-disjoint folds, fold T 1.26-1.35) is 0.038 [0.028, 0.068], not
separated from raw. transfer-v4 development (reported, not fitted): ECE 0.058 → 0.038. The fit checks the pool against the
training that SFT `head.pt` records (sft-v2-r22 and its components). Kev-27B's decision-v7 / b1v2 were checked by
`kev.rounds validate` (the arms' `trained_on`).

**Deviations.** (i) The locked read was spawned at 20:38Z, about 1 min after the tests-stage reads (20:37Z). Both were
launched after the result commit `cc5f346` (20:37Z), and the locked read finished (20:43Z) before the tests verdict. Round 24
did the same, and Jared had asked to go faster. Neither verdict depends on the other, and no read was repeated. (ii) Modal
apps: the locked read ran on a new deployment `kev-r23` (from `cc5f346`, `KEV_GPU=H200`), stopped once it was done. The test
reads and serving checks ran as ephemeral `kev-r23` apps. `kev-sft` was not touched. (iii) The tasksource-heldout test rows
were copied down by another client before this launcher pulled them, so the launcher's own pull refused to overwrite them.
The local rows are byte-identical to the volume copy (sha256 `871e05bd…`).

**Spend.** Modal metered, workspace-wide: $3,977.18 at 2026-09-28T23:14Z, after every round-23 job had ended. That is $190.12
above the $3,787.06 read before the interpolations, before metering lag, and inside the registered projections (rule stage
~$175; confirmation ~$15, against a $213.02 bound). AI Gateway: $0 (Jev's and AutoJev's breadth-v1 test reads are round 24's).

**Evidence** (committed): the verdicts and the T-interval table (`runs/r23-verdict/`); the launch records
(`runs/r23-reads-27b-k-w85-{tests,locked}.json`); the public test reads (`runs/r23c-27b-cand-<tag>`: rows for hard-v1,
devtools-v1 and documents-v1, reports for breadth-v1, documents-v2 and longdoc-v1); the locked read
(`runs/locked/kev-27b-r23-ungated/`); the test breadth report (`runs/r23-breadth-report/`); the serving reports. Private
(`jaredpalmer/kev-private-train` under `runs/r23/`, `scripts/private_rows.py restore`): tasksource-heldout-v1 test rows and
report, breadth-v1 test rows and documents-v2 rows (`runs/r23-verdict/private-rows.json`, @ `bab8219f`), and longdoc-v1 test
rows (`runs/r23-verdict/private-rows-longdoc.json`, @ `ebf2b333`).

**For Jared.** `27b-k-w85` (`/runs/r23-wise/27b-k-w85/checkpoint`, weights `d27af6ab…`) is the first 27B full-weight
checkpoint to pass a registered confirmation. It is not published. If it is released, it ships at T 1.3195, written with
`scripts/calibrate_checkpoint.py` into the released `head.pt`. Four points are worth weighing:
- it passed the locked bar by 1 question, 0.8 pp below Kev-27B;
- on CUAD test it is worse (−1.6 pp, ECE 0.053 against 0.007);
- the test partitions were not untouched for this question;
- it is still 0.85 round 22's final, and the blend recovered almost none of Kev-27B's short-state or CUAD behaviour
  ("What the blends did").

## Released: Kev-27B v2 (2026-09-30)

Jared approved the release on 2026-09-30. Kev-27B v2 (round 23's `27b-k-w85`, internal id `kev-27b-r23`) is now
`jaredpalmer/kev-27b` main, public; v1 (the B1 v2 LoRA adapter) is tag `v1-lora`. Record:
`runs/release/kev-27b-r23-published.json`.

- **v1 first.** Before any upload, the repo's main was checked to be `01b81998…` and tagged `v1-lora` (annotated tag object
  `512eeb74…`), which resolves to `01b81998` for an anonymous client.
- **Upload.** `modal_app.py::release_publish --public --confirm-public jaredpalmer/kev-27b --replace` (a CPU container, app
  `kev-release`; the private default stays and `--public` needs the repo named twice) ran `kev.publish` on
  `/runs/release/kev-27b-r23/checkpoint`: commit **`28be62e9`**. `kev.publish --replace` (new) deletes every file the upload
  does not carry in the same commit (`upload_folder(delete_patterns="*")`), so v1's `adapter_config.json`,
  `adapter_model.safetensors`, `result.json`, `provenance.json`, `train.log`, `training_config.json` and
  `training_metrics.json` went in the commit that added the shards; without `--replace`, a full-weight upload into a repo
  holding an adapter is now refused, because the loader rule would pick the adapter. The Hub's LFS hashes give weights
  `d27af6ab…` and `head.pt` `7968f17b…`, the release copy's. Card metadata: `base_model: Qwen/Qwen3.8-27B`,
  `base_model_relation: finetune`, `library_name: transformers`. The card went up without its verification line, which
  was added in a README-only commit after the check below (`0d7f9b49`, main since).
- **Anonymous verification.** `modal_app.py::release_verify` (new): an H200 with no Modal secret, no `kev-hf-cache` volume,
  a fresh `HF_HOME` and no token (`get_token()` is None in every job). `jaredpalmer/kev-27b` resolved to the full-weight
  layout at T 1.3195 with weights `d27af6ab…`. Served reads reproduced round 23's reads row for row: semif-v1 252 of 252
  and transfer-v4 development 764 of 764 rows have logits equal to the committed raw logits times fp32(1/T) bit for bit
  (`scripts/compare_release_rows.py`). A first decision-v7 development read of v2 (report only; the README's trained-sources
  cell) gives 0.865 (v1 0.866). `jaredpalmer/kev-27b@v1-lora` resolves to v1's adapter (`41bf5af0…`) at T 1.3819 and
  scores semif-v1 0.972 with argmax equal on 252 of 252 rows to the 2026-09-23 read; its logits are not bit-identical to
  that read (max |Δp| 0.032). The cause is causal-conv1d's CUDA kernel, which #125 added to the Modal image after that read;
  kev's code is not involved (`runs/drift-v1/REPORT.md`). Rows and reports:
  `runs/rel27-public/`.
- **Docs.** `docs/model-cards/kev-27b.md` is now v2's card (the candidate card, without the candidate wording, with a
  "Previous version" note for `@v1-lora`; every limitation and the post-hoc caveat kept); `kev-27b-v2.md` is gone. README
  (Models row, text, Serving H200 row from `runs/serving-27b-r23`), the family figures, kev-deploy's GPU notes and
  `docs/claims.json` follow. The collection already listed `jaredpalmer/kev-27b`. The Space serves 4B / 0.8B and is
  unchanged. GitHub release `kev-family`: `kev-27b.tar.gz` (v1's adapter) removed, `SHA256SUMS.txt` regenerated for the
  three remaining tarballs (each re-verified), notes updated (v2 is on the Hub only: 51 GB exceeds the 2 GB asset limit).
- **Spend.** Modal metered $4,574.06 at 2026-09-30T17:49Z before the upload and $4,575.82 at 18:23Z after it and the
  verification (metering lags; app `kev-release`: one CPU upload, one H200 container of about 40 minutes, plus a first attempt of about 10 minutes that failed on an existing output directory after loading).

### Release candidate (2026-09-28, private; the staging record)

Round 23's confirmed candidate `27b-k-w85` is staged as a **private** release candidate, **Kev-27B v2**, for Jared's review
(2026-09-28). Nothing is public: no public repository or revision, no public card, no README, Space or collection change,
no GitHub release. Kev-27B (`jaredpalmer/kev-27b@01b81998`) stays the released 27B model until Jared decides. The internal
release id is `kev-27b-r23`, because `kev-27b-v2` already names the released Kev-27B's records (B1 v2:
`experiments/releases/kev-27b-v2.json`, `runs/release/kev-27b-v2.json`, `/runs/release/kev-27b-v2` on the volume). Record:
`runs/release/kev-27b-r23-staging.json`.

- **Volume copy.** `modal_app.py::release_copy` (new; `scripts/release_checkpoint.py`, a CPU container, app `kev-release`)
  copied `/runs/r23-wise/27b-k-w85/checkpoint` to `/runs/release/kev-27b-r23/checkpoint`. It refuses an existing
  destination. The copy's weights sha256 equals the source's, `d27af6ab…` (the round's `interpolation.json`): 11 shards,
  17 files, 51.3 GB. The source is untouched; its `head.pt` is still `1a62fa3b…` at T 1.0.
- **Release temperature.** `scripts/calibrate_checkpoint.py` fitted T on the copy's `head.pt` (pulled, fitted, put back),
  with the round's registered pool rows: `r3cal` limited to its eight sources, `v9` limited to `mmlu_pro`, and the `transfer4`
  records excluded, 648 questions. It found no conflict with training. It gives **T 1.3195**, the round's pool fit exactly.
  Out-of-fold ECE (5 group-disjoint folds) goes from 0.048 to 0.038, and the intervals overlap. The 90 % interval of T is
  [1.20, 1.45]. The written `head.pt` (`7968f17b…`) is byte-identical to the monitor's copy.
- **Private Hub candidate.** `modal_app.py::release_publish` (new; `kev.publish --private` in a CPU container, so the weights
  never reach the laptop) uploaded the copy to `jaredpalmer/kev-27b-v2-candidate`, commit `0dd33bcc`. With `--private`,
  `kev.publish` now creates a missing repo private and refuses an existing public one (`kev.mirror.ensure_private`). It
  also links full-weight shards into its staging directory instead of copying 51 GB, and uploads `interpolation.json`.
  The repo is private. The Hub's LFS hashes give the same weights sha256 (`d27af6ab…`) and `head.pt` (`7968f17b…`).
  `base_model_relation: finetune`: every weight derives from fine-tunes of the one base. The Hub's `merge` relation is
  for merges of several listed base models, and Kev-27B is an adapter of the same base.
- **Verification.** An authenticated load from the Hub (`kev.checkpoint`, H200,
  `benchmarks --jobs jaredpalmer/kev-27b-v2-candidate@0dd33bcc@evals/external/semif-v1@rel27-hub-semif`) reproduced round 23's
  semif-v1 read row for row. All 252 rows are served at T 1.3195, and their logits equal the committed raw logits times
  fp32(1/T) bit for bit, so the backbone and head outputs are identical. Argmax is equal on 252 of 252.
- **Card and numbers.** `docs/model-cards/kev-27b-v2.md`. Every number traces to a committed report in `docs/claims.json`:
  `runs/r23-verdict/`, `runs/r23-readout/round23.json`, `runs/r23-breadth-report/`, the serving reports,
  `runs/release/kev-27b-r23.json` and the staging record. `runs/release/kev-27b-r23.json` comes from
  `scripts/release_numbers.py --release kev-27b-r23`; release specs may now give an arm a registered `pool` instead of a
  trial. The four recorded releases reproduce byte for byte from the research checkout's rows.
- **Long contracts.** The report-only longdoc-v1 test read landed while this was staged, and the card carries it
  (`runs/r23-verdict/27b-tests.json`, `longtest`). CUAD test: 0.874 against Kev-27B's 0.890, −1.6 [−3.0, −0.3]. ECE is
  0.053 against 0.007, and 2 to 4 times Kev-27B's in every length bucket. Round 24's unblended SFT was −1.8 with ECE
  0.055, so the blend recovered almost none of it. The generated bundles are at 1.000.
- **Caveats the card states.** Round 23 was designed after round 24's confirmation and confirmed on the same test and
  locked partitions. Short states are not better than Kev-27B's (locked −0.8 [−2.0, +0.5]; transfer-r3 test with
  `emotion` −2.1 [−3.5, −0.8]). Long contracts are worse and overconfident (above). There is no scienthoon read
  (removed), and the SFT parent was −5.5 there. hard-v1, devtools-v1 and documents-v1 are in distribution.
- **To release (Jared's call).**
  1. Tag Kev-27B's current weights on `jaredpalmer/kev-27b` (e.g. `b1v2-release`).
  2. Publish `/runs/release/kev-27b-r23/checkpoint` there from a container, with the card moved to `kev-27b.md`.
  3. Then README, the Space, the collection and kev-deploy's GPU list.
- **Spend.** Modal metered $3,971.71 before this work and $3,977.37 after the copy, upload and verification
  (workspace-wide, other apps included). App `kev-release`: CPU copy and upload, plus one H200 read of about 10 minutes.

## Round 25 (registered)

### Round 25 - continued full-weight SFT from Kev-27B: breadth with replay, no long-document families (registered with this spec's commit, written before any round-25 training or read)

**Why.** Every full-weight run so far started from the base and drifted away from what Kev-27B does well as training went
on. Round 22's final and round 23's `27b-k-w85` gained broadly (tasksource-heldout test +5.3, breadth +1.2 to +1.5) but
regressed on short states (locked transfer-v4 −0.8 to −1.2 pp against Kev-27B) and on long contracts (CUAD test −1.6 to
−1.8), and blending toward Kev-27B barely moved either ("Round 23 result"). Round 25 starts **from Kev-27B** (its LoRA
merged into full weights, with its head) and adds breadth with a replay of Kev-27B's own training data. Two hypotheses
shape the data: (i) replay holds the short-state skills; (ii) the long-document families (longify, longdoc) drive the
CUAD regression, and since Kev-27B already generalises to 64k states they are left out and states are capped at 16k.

**Init checkpoint.** `/runs/r25-init/kev-27b-merged/checkpoint` on the `kev-runs` volume: Kev-27B
(`jaredpalmer/kev-27b@01b81998`, `r6-27b-v2/01-trial-1`) with its rank-16 LoRA merged into the bf16 backbone, plus its
pointer head. It is produced separately and does not exist at registration. `kev.train --init_from` compares `base`,
`base_revision`, `lora` (0), `head_dim` (256), `option_isolation`, `special_embeddings` and `weights` ("full") before it
loads anything, and records the weights and head sha256s in each trial's provenance. The read-out compares each candidate
with Kev-27B as served (the parent reads below), not with the merged checkpoint.

**Data** (private; manifest only in this repo). `evals/sft-v2-r25` (kev-private-train @ `1c855ff9`, `sft-v2-r25/`; built by
kev-sft `assemble-r25` @ `c8644ef`, `assemble/derive_r25.py`, deterministic: a second run wrote the same bytes). Every
record is an `sft-v2-r22` record, unchanged, in sft-v2-r22's order; no new records or labels. Selection:
1. Components, from each record's source: keep tasksource-v1, tone, guardrails-pii, guardrails-grounding, injection, ood,
   agents, **b1v2** (Kev-27B's own training data, the replay) and sft-v1's other Kev components (hard-v1, devtools-v1,
   documents-v1). Drop longify and longdoc (hypothesis (ii)), and sft-v1's public and synthetic-v1 sources (not in this mix).
2. Screened again with sft-v2's rule (kev-sft `assemble/screen_v2.py`) against today's 105 evaluation partitions (76,207
   reference items: every Kev suite, the private evaluation mirror, JevBench public). **0 records** offend.
3. Train states of at most 16,384 tokens (Kev-27B tokenizer), each checked to fit `training_context(16384)`; 1,437
   records over the cap leave (1,161 agents, 188 grounding, 84 injection, 4 tasksource). Calibration and development are
   sft-v2-r22's (states ≤ 8,192) restricted to the kept components, screening only: kev.experiment's in-trial temperature
   and development score, which no criterion reads.
4. Train downsampled so the epoch fits one 4-hour attempt (see Budget); each by the smallest sha256(seed:keep:<component>:
   + record digest):

| component | available (after 1-3) | kept | rule | state tokens | questions |
|---|---|---|---|---|---|
| **b1v2 (replay)** | 13,763 | **13,763 (30.2 % of records)** | all | 2.03M | 16,788 |
| tasksource-v1 | 23,996 (round 22's 24,000, 119 families) | 13,000 (all 119 families) | stratified by family, floors then largest remainders (family list private) | 1.82M | 17,757 |
| tone | 7,544 | 3,772 | half of each source | 0.61M | 15,874 |
| guardrails-pii | 4,152 | 2,076 | half | 1.72M | 5,329 |
| guardrails-grounding | 5,467 | 2,734 | half | 5.50M | 12,587 |
| injection | 2,615 | 1,308 | half | 2.07M | 5,519 |
| ood | 7,312 | 3,656 | half | 2.94M | 11,019 |
| agents (≤ 16k) | 3,714 | 371 | a tenth | 3.32M | 1,719 |
| hard-v1 / devtools-v1 / documents-v1 | 19,345 | 4,835 | a quarter of each source | 2.70M | 6,966 |
| **total** | | **45,515** | | **22.70M** (28.53M row tokens with the state shared) | **93,558** (3,417 soft) |

State tokens: median 120, mean 499, p90 1,164, p99 7,130, max 16,365; ≤256 30,896 · 257-512 4,662 · 513-1k 4,390 ·
1k-2k 3,792 · 2k-4k 798 · 4k-8k 570 · 8k-16k 407. Calibration 4,904 and development 2,714 records. No public file names a
tasksource family.

Choices beyond the brief, each for a stated reason:
- **b1v2 is the screened copy** (13,763 of its 15,401 records): the records sft-v1 / sft-v2's screen had removed resemble
  evaluation items (most of b1v2's long v2 records among them), so replaying them would train on near-copies of reads.
- **A quarter of hard-v1 / devtools-v1 / documents-v1 is added.** They are not Kev-27B's training data. The rule's primary
  `1_kev_lower_at_least_minus_1pp` reads transfer-v4 + hard-v1 + devtools-v1 + documents-v1 development, and its paired
  interval is about ±1.2 pp wide, so a candidate that leaves those suites where Kev-27B is fails it. Round 22 trained on all
  of these components (Kev panel +7.8). Without them the round would likely fail that primary regardless of the
  hypotheses under test. Cost: 4,835 records, ~0.25 h.
- **Half of the new families and a tenth of agents** instead of all of them; **13,000 tasksource records** instead of
  round 22's 24,000. The budget and the 12-hour night fix a ~4-hour attempt (below), and agents traces (8.9k tokens on
  average at ≤ 16k), grounding and ood carry most of the time per record. The replay share (~30 %) was held, and the
  families were halved evenly.

**Epoch time, from the measured rate** (kev-sft `assemble/epoch_r25.py`, `assemble/manifests/sft-v2-r25.epoch_projection.json`).
Round 22's pass-time model (per GPU slot 1.55 s + 1.469 × (0.478 s per 1k padded tokens + 0.00282 s × states × longest²),
the slowest rank per slot, summed) on the exact plan kev.train deals (the `balanced_runs` + ceiling replica round 22
checked against `microbatch_plan`, 8 ranks, batch 8 × accum 2, `pass_tokens_max 40960`), over every record's exact epoch-0
token shapes (seed 0, none pairs at 0.25 gated at 8,192: 6,681 paired records):
**356 steps, 712 micro-batches per rank, largest pass 25,590 tokens, 2.22 h (22.5 s per step)**; 1.90 h anchored to round
22's measured ~50 s per step (against its projected 58.6). Snapshots at steps 89 / 178 / 267. One attempt: start ~0.4 h +
training × 1.03 (hourly resume points) + in-trial scoring (4,904 + 2,714 + 656 records at 0.292 s) 0.67 h = **3.36 h**
(3.02 h anchored) of the 4-hour timeout. A timeout is continued by `kev.rounds watch` from the last resume point, within
the ledger's three attempts.

**Arms** (spec `experiments/rounds/r25.json`, plans `experiments/round25/`, app `kev-r25`). Two studies in parallel, each
one trial on 8 × H200, `full_ft 1`, `init_from /runs/r25-init/kev-27b-merged/checkpoint`, one epoch of `evals/sft-v2-r25`,
128 records per step (`batch 8`, `accum 2`, `length_sort 1`, `pass_tokens_max 40960`), bf16, `max_state 16384`,
`p_none_pair 0.25` with `none_pair_max_state 8192`, seed 0, OneCycle with 10 % warm-up:
- `r25-27b-lr1e6`: lr 1e-6.
- `r25-27b-lr2e6`: lr 2e-6 (round 22's learning rate).

Head lr 1e-5 in both: round 22's 1e-4 was for a head trained from scratch; this head starts trained, and 1e-5 lets it
follow the moving backbone at 5-10× the backbone's rate.

Schedule: OneCycle with 10 % warm-up, the trainer's only schedule, unchanged. The continued-training literature does not
favour constant enough to justify a trainer change on the critical path. Re-warming then re-decaying the learning rate,
with replay, matches retraining on the union (Ibrahim et al., 2024, "Simple and Scalable Strategies to Continually
Pre-train LLMs"). Re-warming costs upstream loss mainly at high peak rates (Gupta et al., 2023, "Continual Pre-Training of
Large Language Models: How to (re)warm your model?"), and here the peak is 1-2e-6 from a start at 1/25 of it. Constant or
"infinite" schedules pay off when training will be extended later, which this one-epoch run is not. The decay also means
the final is annealed and the snapshots are its less-trained points, as in round 22.

Candidates: each arm's snapshots (steps 89 / 178 / 267, `/runs/r25-27b-lr{1e6,2e6}/00-trial-0/snapshots/step-00000NN/checkpoint`)
and its final checkpoint, **8 in all** (`27b-lr1e6-s25/s50/s75`, `27b-lr1e6`, and the same for lr2e6). Every arm names
`trained_on: [evals/sft-v2-r25, evals/round6/b1v2, evals/v7/decision-v7]`, so the pool check covers Kev-27B's training as
well as round 25's (`validate` follows sft-v2-r25 → sft-v2-r22 → sft-v2 → sft-v1 and its components).

**Rule, temperature pool, reads, parent reads, read timeout, exclusions and confirmation: round 24's audited rule,
verbatim** ("Round 24 (registered)"; as round 23 re-registered). The spec differs from `r24.json` only in round number,
app, the two studies and eight arms, and the confirmation's candidate read paths (`runs/r25c-…`,
`runs/locked/kev-{size}-r25-ungated/transfer`). The parent's test reads are round 24's (`runs/r24c-{size}-parent-{tag}`),
made for Kev-27B. In brief: pooled temperature (transfer-r3 calibration's eight held-out sources + transfer-v9 MMLU-Pro,
minus transfer rows); primaries breadth-v1 (4 sources excluded) and tasksource-heldout-v1 (the audit's 7 families
excluded) accuracy lower bound > 0 and the Kev panel ≥ −1 pp; guards short state (accuracy ≥ −2 pp, Brier ≤ +0.02,
confident errors ≤ +1 pp), longdoc CUAD all lengths and at 16k+ ≥ −2 pp, unknowable ≤ 0.05; three ECE criteria against
the parent + 0.01; rank by breadth, tasksource-heldout, Kev deltas; optional report panels as in round 24; then the
tests stage (breadth-v1 and tasksource-heldout-v1 test lower > 0, pooled hard/devtools/documents-v1 test ≥ −1 pp) and the
locked stage (locked transfer-v4 ≥ 0.886, served Brier ≤ 0.165).

**Honest caveat on the confirmation partitions.** The test partitions and the locked transfer-v4 set have been read in
rounds 22-24, for other models (round 22's final, round 24's candidate, round 23's blend); none has been read for a
round-25 model. Selection uses development reads only, and the confirmation reads each once. The bars were written before
those reads and are round 24's. Still, the confirmation is not untouched for the program; knowing how close earlier
candidates came to the locked bar (580 and 583 of 656 against 582) shaped this round's design.

**Budget** (Jared's authorization for the night: +$2,000 Modal, ceiling ≈ $5,980 metered; 12 h wall). Metered reading
**$3,966.94 at 2026-09-28T23:45Z**: headroom $2,013.06, reserve ~$200 (10 %), so at most **$1,813** of admission bounds
and spend at any launch. Every launch reads the metered cost first (docs/autoresearch.md section 2).

| item | admission bound | expected |
|---|---|---|
| study `r25-27b-lr1e6` (H200:8 at $41.17/h × 4 h × 3 attempts) | $494.00 | ~$138 (one attempt, 3.36 h; ~$124 anchored) |
| study `r25-27b-lr2e6` | $494.00 | ~$138 |
| reads of one candidate (16 reads × 4 h × $6.27/h) | $400.97 | ~$30 |
| reads of all 8 candidates | $400.97 each, spend-gated | ~$240 |
| confirmation, one candidate (tests stage 7 reads × $25.06; locked read at 4 h; serving check) | $175.43 + ~$25 + ~$6 | ~$25 |
| **peak at launch: both studies + one read batch** (+ ~$85 metered in the first hour) | **$1,389 + ~$85 ≈ $1,474** of $1,813 | |
| **projection at the end of the night** | | **~$4,510** (+$545; + whatever the init merge costs) |

Why the attempt is 4 hours and not 8: a study's bound counts three attempts, so 8-hour attempts (the 6-hour epoch the brief
allowed) would bound the two studies at $1,976, all of the night's headroom, and no read could start until both studies
ended. With 4-hour attempts, one candidate's read batch fits next to both studies from the first snapshot, with the reserve
intact. Two batches at once ($802) would pass $1,813 once the first hour's spend is metered. When the studies end, their
bounds are released and up to three batches run at once.

**Wall-clock plan** (t = 0 at launch; model times, anchored ones ~15 % sooner):
- t ≈ 0-0.4 h: both containers load the merged 27B and count tokens. In the first minutes check the log for
  `none pairs: N of 45515 records` and `plan: 712 micro-batches per rank for 356 steps` (projected; both arms have the
  same data and seed, so the same plan), and count steps per minute against 22.5 s per step.
- t ≈ 0.95 / 1.5 / 2.1 h: snapshots s25 / s50 / s75 of both arms; the final at ≈ 2.7 h, in-trial scoring done ≈ 3.4 h.
- Reads, spend-gated, in this order: lr1e6-s25 at ≈ 1 h; then one batch at a time while the studies run; from ≈ 3.4 h up to
  three batches at once. A batch takes ~3 h (longdoc-v1, 2 h 46 min, is the long pole; the rest finish within ~70 min),
  so all 8 candidates are read by **≈ 10-10.5 h**, then the read-out.
- Confirmation (deliberate, never automatic): the tests stage (~3 h, longdoc-v1 test the long pole) then the locked read
  (~1 h). It starts after the read-out, so it most likely finishes **after** the 12-hour window (≈ 14 h). Under the
  registered spend rule nothing else fits; the admission bounds overstate a read batch about 13×.

**Run steps** (after this PR is merged): (1) confirm `/runs/r25-init/kev-27b-merged/checkpoint` exists and its `head.pt`
meta says `weights: full`, `lora: 0`, `head_dim: 256`; fetch `evals/sft-v2-r25` (`load_split`) and restore the private
parent rows and the tasksource-heldout exclusion list (`scripts/private_rows.py restore --manifest runs/r24-readout/private-exclude.json`,
`runs/r24-readout/private-rows.json`, `runs/r22-readout/private-rows.json`); (2) `KEV_GPU=H200 KEV_APP_NAME=kev-r25 uv run
modal deploy modal_app.py`; (3) read the metered cost, `uv run python -m kev.rounds launch experiments/rounds/r25.json`, then
`watch`; (4) read-out, then confirmation as written.

### Round 25 result

**No candidate (0 of 8).** Every candidate fails `3_breadth_ece_at_most_parent_plus_0.01`: breadth-v1 ECE is 0.0227 to
0.0351 against a bar of 0.0176 (Kev-27B 0.0076). `27b-lr1e6-s75` and the `27b-lr1e6` final come closest. Each passes 11 of
12 criteria and fails only breadth ECE (0.0283 and 0.0235). The lr 2e-6 arms also fail the short-state accuracy guard
(lower bounds −2.08 to −2.71 pp) and the breadth primary. Nothing goes to confirmation, so no test partition or locked set
was read for round 25, and nothing is released. Read-out: `runs/r25-readout/round25.json`
(`python -m kev.rounds readout experiments/rounds/r25.json`, run 2026-09-29T05:51Z after the last read landed); table
`runs/r25-readout/readout.txt`; markdown `runs/r25-readout/tables.md`.

How it is served: every candidate at its pooled T (648 questions; none excluded as a transfer-v4 duplicate), Kev-27B at
1.382. Deltas are paired record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro). Accuracy and
confident errors are in pp, Brier is absolute, and ECE is shown as served against its bar. The exclusions removed the same
questions as in rounds 23 and 24, on both sides: 600 breadth, 795 tasksource-heldout, 380 Kev-panel and 220 short-state.

**Training** (both studies, one attempt each, app `kev-r25`, 8 × H200; `runs/r25-27b-lr{1e6,2e6}/00-trial-0`):
- **Setup:** the warm start loaded 850 full tensors plus the pointer head from `/runs/r25-init/kev-27b-merged/checkpoint`
  (`train.log`). The plan matched the projection: 712 micro-batches per rank, 356 steps, largest pass 25,590 of 40,960
  padded tokens. There were 6,688 none pairs (6,681 projected).
- **Speed and memory:** training took 5,635 s (lr 1e-6) and 5,282 s (lr 2e-6), i.e. 1.47-1.57 h against the projected
  2.22 h (1.90 h anchored). Whole trials took 8,539 s and 7,968 s, inside one 4-hour attempt. Peak memory was 80.4 GB per
  GPU. Each snapshot blocked training for 18-44 s.
- **Loss and gradients:** training loss went from 0.52 over the first 10 steps to 0.26-0.30 at the end. Mean gradient norm
  was 12.6 / 10.9 (max 106.6 / 79.5), and all 356 steps were clipped.
- **In-trial screening** (sft-v2-r25 development, in distribution; not read by the rule): 0.887 / 0.896, in-trial T 1.0 /
  0.966.

| criterion | lr1e6-s25 | lr1e6-s50 | lr1e6-s75 | lr1e6 | lr2e6-s25 | lr2e6-s50 | lr2e6-s75 | lr2e6 |
|---|---|---|---|---|---|---|---|---|
| T (pool, 648) [90 % CI] | 1.289 [1.149, 1.414] | 1.260 [1.122, 1.382] | 1.289 [1.149, 1.414] | 1.350 [1.203, 1.481] | 1.289 [1.176, 1.414] | 1.149 [1.023, 1.231] | 1.350 [1.231, 1.481] | 1.350 [1.203, 1.481] |
| 1 breadth acc, lower > 0 (2475) | +0.7 [+0.00, +1.4] **fail** | +0.6 [−0.04, +1.3] **fail** | +0.9 [+0.3, +1.6] | +0.8 [+0.2, +1.4] | +0.2 [−0.6, +1.0] **fail** | +0.6 [−0.1, +1.3] **fail** | +0.4 [−0.3, +1.1] **fail** | +0.4 [−0.3, +1.1] **fail** |
| 1 tasksource-heldout acc, lower > 0 (1993) | +0.8 [−0.5, +1.9] **fail** | +2.3 [+0.9, +3.7] | +2.9 [+1.6, +4.2] | +2.8 [+1.5, +4.1] | +1.6 [+0.1, +3.1] | +4.0 [+2.5, +5.5] | +4.5 [+3.1, +6.0] | +4.2 [+2.7, +5.7] |
| 1 Kev panel acc, lower ≥ −1 (3351) | +3.0 [+2.0, +4.0] | +5.2 [+4.2, +6.3] | +5.8 [+4.8, +7.0] | +6.2 [+5.2, +7.3] | +3.1 [+1.9, +4.2] | +5.6 [+4.4, +6.7] | +6.4 [+5.2, +7.6] | +6.4 [+5.3, +7.6] |
| 2 short acc, lower ≥ −2 (1586) | −0.50 [−1.20, +0.19] | −0.32 [−1.13, +0.50] | −0.76 [−1.51, +0.00] | −0.69 [−1.45, +0.06] | −1.64 [−2.71, −0.50] **fail** | −1.07 [−2.21, +0.13] **fail** | −0.88 [−2.08, +0.25] **fail** | −1.07 [−2.21, +0.13] **fail** |
| 2 short Brier, upper ≤ +0.02 | +0.006 [+0.001, +0.012] | +0.004 [−0.002, +0.010] | +0.001 [−0.005, +0.007] | +0.002 [−0.004, +0.008] | +0.017 [+0.005, +0.028] **fail** | +0.003 [−0.009, +0.013] | +0.001 [−0.011, +0.011] | +0.001 [−0.011, +0.012] |
| 2 short confident errors, upper ≤ +1 | +0.3 [−0.4, +0.8] | +0.3 [−0.4, +0.8] | +0.1 [−0.5, +0.7] | +0.2 [−0.4, +0.8] | +0.3 [−0.5, +1.0] **fail** | −0.1 [−0.9, +0.5] | −0.2 [−0.9, +0.4] | −0.2 [−0.9, +0.5] |
| 2 CUAD acc, lower ≥ −2 (2254) | +0.3 [−0.3, +0.9] | +0.4 [−0.4, +1.1] | +0.2 [−0.6, +0.9] | +0.1 [−0.7, +0.9] | −0.3 [−1.2, +0.5] | −0.0 [−0.8, +0.8] | −0.1 [−0.9, +0.7] | −0.2 [−1.0, +0.6] |
| 2 CUAD 16k+ acc, cand − parent ≥ −2 | 0.836 vs 0.837 | 0.838 vs 0.837 | 0.837 vs 0.837 | 0.837 vs 0.837 | 0.836 vs 0.837 | 0.839 vs 0.837 | 0.839 vs 0.837 | 0.837 vs 0.837 |
| 2 unknowable share ≤ 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 breadth ECE ≤ 0.0176 (Kev-27B 0.0076) | 0.0227 **fail** | 0.0236 **fail** | 0.0283 **fail** | 0.0235 **fail** | 0.0294 **fail** | 0.0301 **fail** | 0.0351 **fail** | 0.0306 **fail** |
| 3 Kev-panel ECE ≤ 0.0327 (Kev-27B 0.0227) | 0.0129 | 0.0166 | 0.0164 | 0.0185 | 0.0191 | 0.0121 | 0.0117 | 0.0136 |
| 3 tasksource-heldout ECE ≤ 0.0523 (Kev-27B 0.0423) | 0.0561 **fail** | 0.0474 | 0.0502 | 0.0458 | 0.0623 **fail** | 0.0516 | 0.0520 | 0.0505 |
| rank score (Δ breadth + Δ tsheld + Δ Kev, pp) | +4.4 | +8.1 | +9.7 | +9.8 | +4.8 | +10.1 | +11.2 | +11.0 |
| criteria passed | 8/12 | 10/12 | 11/12 | 11/12 | 6/12 | 9/12 | 9/12 | 9/12 |
| verdict | fail | fail | fail | fail | fail | fail | fail | fail |

Exact lower bounds of the breadth primary: s25 0.0000 (not > 0), s50 −0.0004, lr2e6 arms −0.0012 to −0.0061.

**Report-only panels** against Kev-27B (the full table, with every OOD and external panel, is `runs/r25-readout/tables.md`):

| panel | lr1e6-s25 | lr1e6-s50 | lr1e6-s75 | lr1e6 | lr2e6-s25 | lr2e6-s50 | lr2e6-s75 | lr2e6 |
|---|---|---|---|---|---|---|---|---|
| breadth, all 14 sources, acc (3075) | +0.8 [+0.1, +1.5] | +0.7 [+0.1, +1.4] | +1.0 [+0.3, +1.6] | +0.8 [+0.1, +1.4] | +0.4 [−0.5, +1.2] | +0.6 [−0.1, +1.3] | +0.4 [−0.3, +1.1] | +0.5 [−0.3, +1.2] |
| tasksource-heldout all 24 families, acc (2788) | +1.2 [+0.1, +2.3] | +2.2 [+1.0, +3.4] | +2.9 [+1.8, +4.1] | +2.8 [+1.7, +4.0] | +2.3 [+1.1, +3.6] | +4.6 [+3.3, +5.9] | +5.2 [+4.0, +6.5] | +4.9 [+3.7, +6.2] |
| hard-v1 acc (1083) | +6.3 [+3.8, +8.8] | +10.2 [+7.6, +12.8] | +11.8 [+9.3, +14.4] | +12.5 [+10.0, +15.0] | +7.3 [+4.6, +10.1] | +10.2 [+7.5, +13.0] | +12.5 [+9.7, +15.3] | +12.7 [+10.0, +15.6] |
| Kev panel without hard-v1, acc (2268) | +1.4 [+0.5, +2.3] | +2.8 [+1.8, +3.8] | +3.0 [+2.0, +4.0] | +3.2 [+2.2, +4.2] | +1.1 [+0.0, +2.1] | +3.4 [+2.2, +4.5] | +3.4 [+2.3, +4.6] | +3.4 [+2.2, +4.6] |
| transfer-v4 dev only, acc (576; the locked read's suite) | −0.9 [−1.9, +0.0] | −0.3 [−1.6, +0.9] | −0.9 [−2.1, +0.2] | −0.9 [−2.1, +0.2] | −0.9 [−2.4, +0.7] | +0.2 [−1.9, +2.3] | +0.7 [−1.2, +3.0] | +0.3 [−1.6, +2.6] |
| SemIf / WANLI-v2 / TypeSafe acc | +0.0 / −0.3 / +1.1 | +0.0 / +0.1 / +0.0 | +0.7 / +0.4 / −1.1 | +0.7 / +0.4 / −1.1 | +0.7 / +0.6 / −2.2 | −1.4 / −0.7 / +1.1 | −3.5 / −0.8 / +1.1 | −2.8 / −0.7 / +0.0 |
| ood-v2 / agents-ood / guardrails-ood acc | +0.5 / −0.2 / +3.2 | +0.7 / +0.3 / +3.7 | +0.9 / +0.2 / +3.9 | +0.8 / +0.3 / +4.0 | +0.3 / +0.5 / +3.9 | +1.0 / +0.2 / +3.9 | +1.0 / +0.5 / +4.2 | +1.0 / +0.7 / +4.2 |
| ood-v2 / agents-ood / guardrails-ood ECE Δ | −0.020 / −0.074 / −0.055 | −0.025 / −0.083 / −0.064 | −0.025 / −0.092 / −0.065 | −0.026 / −0.091 / −0.064 | −0.022 / −0.098 / −0.068 | −0.025 / −0.085 / −0.069 | −0.027 / −0.097 / −0.065 | −0.027 / −0.096 / −0.065 |
| CUAD ECE (Kev-27B 0.063) | 0.087 | 0.098 | 0.101 | 0.103 | 0.102 | 0.100 | 0.102 | 0.103 |
| CUAD ECE 16k+ (Kev-27B 0.071) | 0.089 | 0.100 | 0.105 | 0.112 | 0.105 | 0.112 | 0.110 | 0.109 |
| short-state acc, candidate (Kev-27B 0.9042) | 0.8991 | 0.9010 | 0.8966 | 0.8972 | 0.8878 | 0.8934 | 0.8953 | 0.8934 |

CUAD by length (accuracy / ECE; Kev-27B first): < 8k (896) 0.845 / 0.059, candidates 0.842-0.852 / 0.092-0.102;
8k-16k (452) 0.834 / 0.063, 0.827-0.838 / 0.089-0.112; 16k-32k (454) 0.841 / 0.070, 0.839-0.844 / 0.087-0.116; 32k-64k
(452) 0.832 / 0.072, 0.827-0.834 / 0.095-0.118 (per arm in `tables.md`). The generated longdoc items stay at 1.000 accuracy.

**What the round showed:**
- **Replay held short states at lr 1e-6 (hypothesis (i)):** short-state accuracy −0.3 to −0.8 pp, every lower bound ≥ −1.51.
  At lr 2e-6 the short-state loss is transfer-r3 test, not transfer-v4. On transfer-v4 development alone the lr 2e-6
  snapshots from s50 read +0.2 to +0.7, while the combined panel sits at −0.9 to −1.6.
- **Dropping the long-document families kept CUAD accuracy but not its calibration (hypothesis (ii), development only):**
  - Accuracy: CUAD −0.3 to +0.4 pp, and 16k+ equal to Kev-27B (0.836-0.839 vs 0.837). Round 22's CUAD loss showed on test
    (−1.8), and no round-25 test was read, so this is not yet evidence against it.
  - Calibration: CUAD ECE still rises to 0.087-0.103, and grows with training (lr 1e-6: s25 0.087 → final 0.103). Round 22's
    final, trained with the long-document data, read CUAD ECE 16k+ 0.102 (round 24's read), and the round-23 blends 0.097-0.102
    at all lengths. So the miscalibration comes with continued full-weight SFT served at a pooled short-state temperature, not
    with the long-document families (development reads, one round).
- **The gains are smaller than round 22 / 23's and grow with steps:** breadth +0.2 to +0.9 (27b-k-w85 +1.5), tasksource-heldout
  +0.8 to +4.5 (+4.2), Kev panel +3.0 to +6.4 (+8.1). guardrails-ood gains as much as w85 did (+3.2 to +4.2 against
  +4.0), agents-ood less (−0.2 to +0.7 against +2.1); OOD ECE improves throughout (agents-ood −0.07 to −0.10).
- **Breadth ECE and tasksource-heldout ECE want different temperatures** (report only, `runs/r25-readout/breadth-ece-by-t.json`):
  - The breadth panel's ECE-minimising T is 1.40-1.72, above every pooled fit.
  - At the upper end of each arm's 90 % T interval, breadth ECE would pass for most arms (0.0099-0.021; lr 1e-6 final 0.0135).
    At that T, tasksource-heldout ECE rises to 0.060-0.070 and would fail its bar of 0.0523.
  - At the lower end the reverse holds. This is round 24's pattern again. For no arm does any T in its 90 % interval pass
    both bars (41-point grid).

**Report only: against round 23's confirmed `27b-k-w85`.** No candidate was named, so every arm is compared.
- **Method:** each side at its own pooled T (w85 1.320). Paired bootstrap on the development reads, the same panels and
  exclusions, with `short` split into transfer-v4 development and transfer-r3 test.
- **Files:** `runs/r25-readout/vs-w85.md` and `vs-w85-<arm>.json`, from `runs/r25-readout/r25-vs-w85.py`.
- **Result: w85 is ahead or level on nearly every panel.**
  - Against the closest arm, the lr 1e-6 final: breadth −0.8 [−1.5, +0.0], tasksource-heldout −1.4 [−2.7, +0.1], Kev
    panel −1.9 [−2.9, −1.0] (hard-v1 −5.4, the rest −0.3), agents-ood −1.8 [−2.5, −1.2].
  - Short states are level: +0.3 [−0.7, +1.2]; transfer-v4 development −1.2 [−2.6, +0.2], transfer-r3 test +1.1 [+0.0, +2.4].
  - The candidate has more confident errors: short +1.3 [+0.7, +2.0] pp.
  - CUAD accuracy is level (−0.6 [−1.7, +0.3]), with CUAD ECE +0.006 [−0.009, +0.016]. By length, w85 reads 0.856 / 0.096
    (< 8k) and 0.839 / 0.102 (16k+), against 0.849 / 0.100 and 0.837 / 0.112.
- **Ranking:** no round-25 arm is ahead of w85 with a lower bound above 0 on any panel the rule gates. Two report-only
  panels are the exceptions: `27b-lr1e6-s25`'s CUAD ECE −0.011 [−0.022, −0.002], and `27b-lr2e6-s75` on tasksource-heldout's
  24 families +1.3 [+0.1, +2.4]. The lr 2e-6 s75 and final match w85 on tasksource-heldout (+0.4, +0.0) and on transfer-v4
  development (+0.3, +0.0), and trail it on breadth (−1.2, −1.1) and hard-v1 (−5.4, −5.2). Kev-27B and `27b-k-w85` remain
  Jared's release options ("Round 23 confirmation"); nothing here changes that choice.

**Deviations:**
- **(i) Four reads relaunched.**
  - *What failed:* four jobs of the first batch (`27b-lr1e6-s25`: agentsood, guardood, ood, tsheld) failed at load. The
    development partitions of the private suites agents-ood-v1, guardrails-ood-v1, ood-v2 and tasksource-heldout-v1 were
    not in this checkout, and the Modal container cannot read the private mirror.
  - *Fix:* the partitions were copied from the round-23 checkout, sha256-checked against each manifest, and the four jobs
    relaunched once under the same names. Nothing had been written under those names.
  - *Record:* `runs/r25-reads-27b-lr1e6-s25-retry.json`. Every later batch had the partitions and ran all 16 reads.
- **(ii) Spend accounting and read concurrency.**
  - *Registration:* one candidate's batch at a time while both studies ran ($400.97 each at 16 × 4 h × $6.27).
  - *What was done:* the gate was applied as docs/autoresearch.md §2 defines it, with the bounds of the calls still
    running. Each read job is its own call ($25.06), so a batch whose short reads had landed counted only for its
    remaining jobs, mostly the longdoc read (~2 h 40 min on a 27B).
  - *Result:* `27b-lr2e6-s25` launched at 01:07Z beside `27b-lr1e6-s25`'s four remaining jobs while both studies ran.
    Later batches launched as the gate allowed (`runs/r25-readout/r25-gate.py`, the local helper that computed it).
    Every launch read the metered cost first and was ≤ $1,813: $1,422.96 at 00:52Z (full-batch bounds), then, with
    running-call bounds, $1,498 and ~$1,518 for the two relaunches, and $1,585, $1,770, $1,762, $1,705 and $1,791 at
    01:07, 01:31, 02:09, 02:46 and 03:02Z. The watcher launched both finals' reads itself when the trials ended (02:36Z,
    02:45Z), without a gate of its own. By the same accounting that was about $1,650 at 02:36Z, with one study still
    running.
  - *Against the registration's accounting:* with full-batch bounds the 01:07Z launch would have counted
    95.81 + 988 + 802 = $1,886, over $1,813. Actual spend stayed far below the bounds.
- **(iii) Launch times:** s25 00:52Z / 01:07Z, lr1e6-s50 01:31Z, lr2e6-s50 02:09Z, finals 02:36Z / 02:45Z (watcher),
  lr1e6-s75 02:46Z, lr2e6-s75 03:02Z. The last read landed at 05:49Z. Both deploy and reads ran on app `kev-r25`, and
  the deployed app was stopped at 06:00Z. `kev-sft` was not touched.

**Spend** (Modal metered, workspace-wide):
- $3,966.94 at the registration reading (23:45Z).
- Readings before launches: $3,969.07 (00:22Z), $4,000.93 (00:51Z), $4,062.75 (01:06Z), $4,197.70 (01:30Z, revised to
  $4,141 by 01:45Z), $4,214.37 (02:09Z), $4,354.44 (03:02Z).
- $4,461.51 at 05:51Z after the last read, and $4,461.42 at 06:00Z.
- Total: **+$494.5** over the registration reading (two one-attempt studies ≈ $185 at list, 128 reads, 4 relaunches),
  before metering lag. That is inside the registered projection (~$4,510). The night's ceiling was ≈ $5,980, with a hard
  stop at $5,600.

**Evidence** (committed):
- The read-out, its tables and the report-only extras (`vs-w85*`, `breadth-ece-by-t.json`, `transfer4-vs-parent.json`,
  with the scripts that wrote them under `runs/r25-readout/`).
- The launch records `runs/r25-reads-*.json`.
- Every public-suite read's report and rows (`runs/r25-27b-<arm>-<tag>`); for the OOD reads, reports only.
- Per trial: `provenance.json`, `result-public.json` (without the per-task table), in-trial transfer rows and temperature,
  `training_metrics.json`, and the three `snapshot.json` records.

Private (`jaredpalmer/kev-private-train` @ `03a62890` under `runs/r25/`, with sha256s in `runs/r25-readout/private-rows.json`;
restore with `scripts/private_rows.py restore`):
- the eight tasksource-heldout-v1 reads (rows and reports);
- the ood / agents-ood / guardrails-ood rows;
- both trials' development rows and full `result.json`.

Checkpoints (~51 GB each: 6 snapshots + 2 finals) stay on the `kev-runs` volume.

**Next** (proposals, not registered; each needs its own spec):
- **Find what carries the breadth-ECE gap before spending more.** It is the only criterion `27b-lr1e6` and
  `27b-lr1e6-s75` fail, by 0.006-0.011. Blending back toward Kev-27B is not the obvious fix: in round 23, more Kev-27B
  weight raised breadth ECE (0.0103 at α 0.85 → 0.0195 at α 0.50, SFT head). A report-only breakdown of the gated breadth
  panel by source at the pooled T (no GPU) would show whether one or two datasets carry the gap.
- **The two calibration panels now pull the temperature apart.** Breadth-v1 wants T 1.40-1.72 and tasksource-heldout
  wants less. For every arm, no T inside its pooled 90 % interval passes both ECE bars (41-point grid, checked after the
  read-out). Round 22's final and `27b-k-w85` passed both at their pooled T, so continued SFT from Kev-27B widened the gap.

## Round 26 (registered)

### Round 26 - round 25's lr 1e-6 arm with twice the breadth data (registered with this spec's commit, written before any round-26 training or read)

**The single question.** Does doubling tasksource-v1, with everything else in round 25's lr 1e-6 arm held fixed, turn
that arm's small breadth gains into ones that pass round 24's audited rule? The fixed parts are the init, learning rate,
schedule, replay, other families, state cap, seed and rule. Round 26 changes one thing: 13,000 more tasksource-v1 records
(26,000 in all, stratified over the same 119 families).

**Why.** Round 25's read-out (`runs/r25-readout/round25.json`, PR #180) reads the lr 1e-6 arm at a safe learning rate:
- Short states held: −0.3 to −0.8 pp against Kev-27B, every lower bound ≥ −1.51.
- CUAD accuracy held on development.
- `27b-lr1e6-s75` and the final passed 11 of 12 criteria. Each failed only breadth-v1 ECE: 0.0283 and 0.0235 against a
  bar of 0.0176.
- The breadth gains were small: breadth-v1 +0.8 [+0.2, +1.4] and tasksource-heldout +2.8 at the final, against
  `27b-k-w85`'s +1.5 and +4.2.
- lr 2e-6 got more tasksource-heldout gain (+4.0 to +4.5) but drifted on short states (lower bounds −2.08 to −2.71 pp,
  failing the guard).

A higher learning rate is ruled out, so the remaining lever is more breadth data at the safe rate. The round-25 gains grew
with steps at both rates (breadth s25 +0.7 → final +0.8, tasksource-heldout +0.8 → +2.8). More breadth records give more
steps on the breadth distribution without raising the rate. The breadth ECE failure is the round's other question. Round
25's analysis (`runs/r25-readout/breadth-ece-by-t.json`) found breadth wanting a higher temperature (1.40-1.72) than the
pooled fit, with tasksource-heldout wanting a lower one. More breadth data could move the breadth panel toward the pooled
temperature, or it could not. Nothing about calibration changes here: the pool, the rule and the bars are round 24's.

**Init checkpoint.** Round 25's: `/runs/r25-init/kev-27b-merged/checkpoint` (Kev-27B `jaredpalmer/kev-27b@01b81998`,
its rank-16 LoRA merged into the bf16 backbone, plus its pointer head). It is on the `kev-runs` volume already. Round 25
loaded it (850 full tensors plus the head). `kev.train --init_from` checks compatibility again before loading.

**Data** (private; manifest only in this repo). `evals/sft-v2-r26` (kev-private-train @ `187ac5f0`, `sft-v2-r26/`;
built by kev-sft `assemble-r26` @ `a4d3526`, `assemble/derive_r26.py`). The build is deterministic: a second run from scratch, without
the measurement cache, wrote the same bytes (train sha256 `4578275c…`). Every record is an sft-v2-r21 record, unchanged,
with no new records or labels.
- **Source.** Round 25's source, sft-v2-r22, holds only 24,000 tasksource-v1 records (23,996 within the cap), so it cannot
  supply 26,000. sft-v2-r21 holds all 58,190 of sft-v2's. For every other kept component it holds exactly sft-v2-r22's
  records, in the same order, because sft-v2-r22 downsampled only longify, tasksource-v1 and sft-v1's public and synthetic
  sources.
- **Steps 1-3 are derive_r25.py's:** the same components, the same screen and the 16,384-token state cap.
  - The screen is kev-sft `assemble/screen_v2.py` against the same 105 evaluation partitions and 76,207 reference items.
    No evaluation suite has changed since round 25. **0 records** offend.
  - 1,443 train records are over the cap: 1,161 agents, 188 grounding, 84 injection and 10 tasksource.
- **Step 4 is the one change.**
  - Every component except tasksource-v1 is selected by round 25's rule and salt, so it keeps **exactly sft-v2-r25's
    records**. The build checks this.
  - tasksource-v1 keeps all 13,000 of sft-v2-r25's records, a superset, so round 26's data is round 25's plus additions.
    Each family is then filled up to its share of 26,000 (floors, then largest remainders over the 58,180 capped records)
    with its records not already in sft-v2-r25, ranked by the smallest sha256(seed:keep:tasksource-v1: + record digest). No
    family's round-25 records exceeded its new share. All 119 families are kept.
  - The build also checks that train is sft-v2-r25's train partition in order plus 13,000 tasksource-v1 records, and that
    calibration and development equal sft-v2-r25's byte for byte (4,904 and 2,714 records; same sha256).

| component | round 25 | **round 26** | state tokens | questions |
|---|---|---|---|---|
| **tasksource-v1** | 13,000 | **26,000 (+13,000; all 119 families)** | 3.68M | 35,548 |
| b1v2 (replay, kept whole) | 13,763 (30.2 %) | 13,763 (**23.5 %**) | 2.03M | 16,788 |
| tone / guardrails-pii / guardrails-grounding / injection / ood (half each) | 13,546 | 13,546 | 12.83M | 50,328 |
| agents (≤ 16k, a tenth) | 371 | 371 | 3.32M | 1,719 |
| hard-v1 / devtools-v1 / documents-v1 (a quarter) | 4,835 | 4,835 | 2.70M | 6,966 |
| **total** | 45,515 | **58,515** | **24.56M** (31.07M row tokens with the state shared) | **111,349** (4,645 soft) |

State tokens: median 102, mean 420, p90 1,015, p99 6,210, max 16,365. The histogram is ≤256 42,126 · 257-512 5,745 ·
513-1k 4,871 · 1k-2k 3,933 · 2k-4k 851 · 4k-8k 579 · 8k-16k 410. No public file names a tasksource family.

**What else changes, stated honestly.** Holding b1v2 whole while adding records lowers the replay share from 30.2 % to
23.5 %. At the same epoch count the model also takes 458 steps instead of 356 (OneCycle over the longer run, same peak
lr), so it spends more steps away from the init. Both follow from the one change. Neither was adjusted, because
compensating (more replay, or a shorter schedule) would be a second change. If short states drift, these are the first
suspects.

**Epoch time, from round 25's measured rate** (kev-sft `assemble/epoch_r26.py`,
`assemble/manifests/sft-v2-r26.epoch_projection.json`).
- **Method.** Round 22's pass-time model and the exact plan replica (as in round 25) run over every record's exact
  epoch-0 token shapes, calibrated to round 25's **measured** training time. `runs/r25-27b-lr1e6/00-trial-0` trained 356
  steps in 5,635 s (15.8 s per step) where the model projected 2.22 h, which gives an anchor of 0.704.
- **Projection.** **458 steps**, 915 micro-batches per rank, largest pass 25,590 of 40,960 padded tokens, 8,256 none-paired
  records.
- **Training time.** 2.51 h unanchored, **1.77 h anchored** (13.9 s per step), within the ~2.2 h target.
- **Snapshots.** Steps **115 / 229 / 344**.
- **One attempt.** Anchored training plus round 25's measured non-training time for the same trial (2,904 s: load, token
  counts, snapshots, final save, in-trial scoring of the same calibration / development and transfer-v4) = **2.57 h**. The
  unanchored model gives 3.32 h. Either fits the 4-hour timeout. A timeout is continued by `kev.rounds watch` from the last
  resume point, within the ledger's three attempts.

**Arm** (spec `experiments/rounds/r26.json`, plan `experiments/round26/lr1e6.json`, app `kev-r26`). One study,
`r26-27b-lr1e6`: one trial on 8 × H200 with a 14,400 s attempt.
- The plan is `experiments/round25/lr1e6.json` byte for byte: `full_ft 1` from the merged checkpoint, lr 1e-6, head lr
  1e-5, OneCycle with 10 % warm-up, one epoch, batch 8 × accum 2, `length_sort 1`, `pass_tokens_max 40960`, bf16,
  `max_state 16384`, `p_none_pair 0.25` with `none_pair_max_state 8192`, seed 0.
- Only the study's suite differs: `evals/sft-v2-r26`.

Candidates: the snapshots at steps 115 / 229 / 344 (`/runs/r26-27b-lr1e6/00-trial-0/snapshots/step-0000NNN/checkpoint`)
and the final, **4 in all** (`27b-lr1e6-s25/s50/s75`, `27b-lr1e6`).
- Every arm names `trained_on: [evals/sft-v2-r26, evals/round6/b1v2, evals/v7/decision-v7]`.
- The snapshot steps are the projection's. If the trainer's plan logs a different step count, the snapshot paths are
  corrected before any read, and nothing else changes.

**Rule, temperature pool, reads, parent reads, read timeout, exclusions and confirmation: round 24's audited rule,
verbatim, as in round 25.** The spec differs from `r25.json` only in:
- the round number, the app and the `registered` text;
- one study instead of two, with round 26's plan and suite;
- the four arms;
- the confirmation's candidate read paths (`runs/r26c-…`, `runs/locked/kev-{size}-r26-ungated/transfer`).

The parent is Kev-27B with the same reads, and its test reads are round 24's (`runs/r24c-{size}-parent-{tag}`). The same
caveat as round 25 applies to the confirmation partitions: they have been read for other models in rounds 22-24, and none
has been read for a round-25 or round-26 model.

**Budget** (Jared's authorization for the night: ceiling ≈ $5,980 metered). Metered reading **≈ $4,454** at registration:
headroom ≈ $1,526, reserve ≈ $153 (10 %). That leaves at most **≈ $1,373** of admission bounds plus spend at any launch.
Every launch reads the metered cost first (docs/autoresearch.md section 2).

| item | admission bound | expected |
|---|---|---|
| study `r26-27b-lr1e6` (H200:8 at $41.17/h × 4 h × 3 attempts) | $494.00 | ~$106 (one attempt, 2.57 h; ~$137 unanchored) |
| reads of one candidate (16 reads × 4 h × $6.27/h) | $400.97 | ~$30 |
| reads of all 4 candidates | $400.97 each, spend-gated | ~$120 |
| confirmation, one candidate (tests stage 7 reads × $25.06; locked read; serving check) | $175.43 + ~$25 + ~$6 | ~$25 |
| **peak while training: study + two read batches** | **$1,296** of $1,373 | |
| **peak after training: three read batches** | **$1,203** of $1,373 | |
| **projection at the end of the round** | | **≈ $4,680-4,710** (+$226-257; + ~$25-50 if a candidate goes to confirmation) |

A fourth concurrent batch ($1,604) does not fit, so reads run at most three at a time once the study's bound is released.
While the study runs, at most two batches run alongside it.

**Wall-clock plan** (t = 0 at launch; anchored times, unanchored ones ~40 % later):
- **t ≈ 0-0.4 h.** The container loads the merged 27B and counts tokens. In the first minutes, check the log for
  `none pairs: N of 58515 records` (8,256 projected; round 25's log read 6,688 against 6,681) and `plan: 915
  micro-batches per rank for 458 steps`. Then count steps per minute against 13.9 s per step.
- **Training.** Snapshots s25 / s50 / s75 at t ≈ 0.85 / 1.3 / 1.75 h. The final lands at ≈ 2.2 h and in-trial scoring is
  done at ≈ 2.6 h.
- **Reads, spend-gated.**
  - `27b-lr1e6-s25` at ≈ 0.85 h and `27b-lr1e6-s50` at ≈ 1.3 h, alongside the study (two batches).
  - `27b-lr1e6-s75` when the trial ends (≈ 2.6 h, three at once).
  - The final when the first batch ends (≈ 3.85 h).
  - A batch takes ~3 h: longdoc-v1, 2 h 46 min, is the long pole, and the rest finish within ~70 min. The snapshots are
    read by ≈ 5.6 h and the final by **≈ 6.9 h**, then the read-out.
- **Confirmation** (deliberate, never automatic) follows the read-out: the tests stage (~3 h) then the locked read
  (~1 h). It would finish after Jared's night.

**Run steps** (after this PR is merged):
1. Confirm `/runs/r25-init/kev-27b-merged/checkpoint` is still on the volume. Fetch `evals/sft-v2-r26` (`load_split`).
   Restore the private parent rows and the tasksource-heldout exclusion list as in round 25.
2. `KEV_GPU=H200 KEV_APP_NAME=kev-r26 uv run modal deploy modal_app.py`.
3. Read the metered cost, then run `uv run python -m kev.rounds launch experiments/rounds/r26.json`, then `watch`.
4. Read-out, then confirmation as written.

### Round 26 result

**No candidate (0 of 4).** Every candidate fails `3_breadth_ece_at_most_parent_plus_0.01`: breadth-v1 ECE is 0.0197 to
0.0287 against a bar of 0.0176 (Kev-27B 0.0076). The final, `27b-lr1e6`, passes the other 11 criteria, as round 25's final
did (breadth ECE 0.0233 here, 0.0235 there). The three snapshots also fail the breadth primary: their lower bounds are
−0.12, −0.16 and −0.04 pp. `27b-lr1e6-s50` also fails the short-state accuracy guard (lower bound −2.52 pp), and
`27b-lr1e6-s25` fails tasksource-heldout ECE (0.0627 against 0.0523). Nothing goes to confirmation, so no test partition or
locked set was read for round 26, and nothing is released. Read-out: `runs/r26-readout/round26.json`
(`python -m kev.rounds readout experiments/rounds/r26.json`, written by the watcher at 12:53Z once the last read landed and
re-run at 12:57Z); table `runs/r26-readout/readout.txt`; markdown `runs/r26-readout/tables.md`.

How it is served: every candidate at its pooled T (648 questions, none excluded as a transfer-v4 duplicate), Kev-27B at
1.382. Deltas are paired record-clustered bootstraps against Kev-27B (2,000 resamples, seed 0, micro). Accuracy and
confident errors are in pp, Brier is absolute, and ECE is shown as served against its bar. The exclusions removed the same
questions as in rounds 23-25, on both sides: 600 breadth, 795 tasksource-heldout, 380 Kev-panel and 220 short-state.

**Training** (one attempt, app `kev-r26`, 8 × H200; `runs/r26-27b-lr1e6/00-trial-0`):
- **Setup:** the warm start loaded 850 full tensors plus the pointer head from `/runs/r25-init/kev-27b-merged/checkpoint`
  (`train.log`: "delta: warm start from ..."; 58,515 training requests). The plan matched the projection exactly: 915
  micro-batches per rank, 458 steps, largest pass 25,590 of 40,960 padded tokens, snapshots after steps 115 / 229 / 344.
  So the snapshot paths in `r26.json` were right and nothing was corrected. There were 8,263 none pairs (8,256 projected).
- **Speed and memory:** training took 5,836 s (1.62 h, 12.7 s per step) against the projected 1.77 h anchored (2.51 h
  unanchored). The whole trial took 8,843 s, inside one 4-hour attempt. Peak memory was 80.4 GB per GPU. The snapshots
  blocked training for 19.8, 19.9 and 29.2 s.
- **Loss and gradients:** training loss went from 0.41 at step 10 to 0.27-0.31 over the last steps. Mean gradient norm was
  12.2 (max 61.2), and all 458 steps were clipped.
- **In-trial screening** (sft-v2-r26 development, in distribution; not read by the rule): 0.893, in-trial T 0.966.
  transfer-v4 development 0.855 raw (round 25's lr 1e-6 trial: 0.840).

| criterion | lr1e6-s25 | lr1e6-s50 | lr1e6-s75 | lr1e6 |
|---|---|---|---|---|
| T (pool, 648) [90 % CI] | 1.350 [1.203, 1.447] | 1.350 [1.203, 1.447] | 1.203 [1.097, 1.320] | 1.320 [1.176, 1.414] |
| 1 breadth acc, lower > 0 (2475) | +0.5 [−0.1, +1.2] **fail** | +0.6 [−0.2, +1.2] **fail** | +0.6 [−0.04, +1.3] **fail** | +1.0 [+0.3, +1.7] |
| 1 tasksource-heldout acc, lower > 0 (1993) | +2.4 [+1.1, +3.7] | +2.7 [+1.2, +4.2] | +3.5 [+2.0, +4.9] | +3.3 [+1.8, +4.8] |
| 1 Kev panel acc, lower ≥ −1 (3351) | +3.7 [+2.7, +4.7] | +5.0 [+3.9, +6.1] | +5.6 [+4.6, +6.7] | +5.6 [+4.5, +6.6] |
| 2 short acc, lower ≥ −2 (1586) | −0.57 [−1.26, +0.13] | −1.45 [−2.52, −0.44] **fail** | −0.63 [−1.58, +0.32] | −1.01 [−1.95, −0.06] |
| 2 short Brier, upper ≤ +0.02 | +0.005 [−0.001, +0.011] | +0.008 [−0.003, +0.018] | +0.000 [−0.011, +0.009] | +0.003 [−0.008, +0.011] |
| 2 short confident errors, upper ≤ +1 | −0.3 [−1.0, +0.4] | −0.2 [−1.0, +0.5] | −0.3 [−1.1, +0.4] | −0.2 [−0.9, +0.5] |
| 2 CUAD acc, lower ≥ −2 (2254) | +0.8 [+0.2, +1.5] | +0.1 [−0.5, +0.9] | +0.2 [−0.6, +1.0] | +0.2 [−0.6, +1.0] |
| 2 CUAD 16k+ acc, cand − parent ≥ −2 | 0.843 vs 0.837 | 0.834 vs 0.837 | 0.836 vs 0.837 | 0.834 vs 0.837 |
| 2 unknowable share ≤ 0.05 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 breadth ECE ≤ 0.0176 (Kev-27B 0.0076) | 0.0209 **fail** | 0.0197 **fail** | 0.0287 **fail** | 0.0233 **fail** |
| 3 Kev-panel ECE ≤ 0.0327 (Kev-27B 0.0227) | 0.0180 | 0.0190 | 0.0170 | 0.0197 |
| 3 tasksource-heldout ECE ≤ 0.0523 (Kev-27B 0.0423) | 0.0627 **fail** | 0.0416 | 0.0394 | 0.0419 |
| rank score (Δ breadth + Δ tsheld + Δ Kev, pp) | +6.6 | +8.2 | +9.7 | +9.8 |
| criteria passed | 9/12 | 9/12 | 10/12 | 11/12 |
| verdict | fail | fail | fail | fail |

Exact lower bounds of the breadth primary: s25 −0.0012, s50 −0.0016, s75 −0.0004, final +0.0028. The final's short-state
lower bound is −0.0195, 0.05 pp inside its guard.

**Report-only panels** against Kev-27B (the full table, with every OOD and external panel, is `runs/r26-readout/tables.md`):

| panel | lr1e6-s25 | lr1e6-s50 | lr1e6-s75 | lr1e6 |
|---|---|---|---|---|
| breadth, all 14 sources, acc (3075) | +0.6 [−0.1, +1.3] | +0.5 [−0.2, +1.2] | +0.6 [−0.1, +1.3] | +0.8 [+0.1, +1.6] |
| tasksource-heldout all 24 families, acc (2788) | +2.3 [+1.1, +3.4] | +2.9 [+1.7, +4.2] | +3.6 [+2.4, +4.8] | +3.4 [+2.2, +4.7] |
| hard-v1 acc (1083) | +9.3 [+6.8, +12.0] | +11.2 [+8.6, +13.8] | +12.7 [+10.1, +15.4] | +12.7 [+10.0, +15.3] |
| Kev panel without hard-v1, acc (2268) | +1.0 [+0.2, +1.9] | +2.1 [+1.1, +3.1] | +2.2 [+1.3, +3.2] | +2.2 [+1.1, +3.2] |
| SemIf / WANLI-v2 / TypeSafe acc | +0.0 / +0.4 / +1.1 | −0.7 / +0.4 / +0.0 | +0.7 / +0.3 / +0.0 | +0.7 / +0.1 / +0.0 |
| ood-v2 / agents-ood / guardrails-ood acc | +0.2 / −0.2 / +3.5 | +0.3 / +0.0 / +3.7 | +0.6 / +0.3 / +4.1 | +0.5 / +0.5 / +4.1 |
| ood-v2 / agents-ood / guardrails-ood ECE Δ | −0.024 / −0.071 / −0.056 | −0.022 / −0.072 / −0.058 | −0.029 / −0.092 / −0.066 | −0.029 / −0.088 / −0.065 |
| CUAD ECE (Kev-27B 0.063) | 0.078 | 0.086 | 0.099 | 0.096 |
| CUAD ECE 16k+ (Kev-27B 0.071) | 0.077 | 0.090 | 0.102 | 0.104 |
| short-state acc, candidate (Kev-27B 0.9042) | 0.8985 | 0.8897 | 0.8979 | 0.8941 |

CUAD by length (accuracy / ECE; Kev-27B first): < 8k (896) 0.845 / 0.059, candidates 0.848-0.853 / 0.083-0.096; 8k-16k
(452) 0.834 / 0.063, 0.832-0.847 / 0.081-0.106; 16k-32k (454) 0.841 / 0.070, 0.837-0.848 / 0.082-0.105; 32k-64k (452)
0.832 / 0.072, 0.830-0.838 / 0.083-0.108 (per arm in `tables.md`). The generated longdoc items stay at 1.000 accuracy.

**Report only: against round 25's lr 1e-6 arm at the same fraction of its run** (s25 / s50 / s75 / final against round 25's
s25 / s50 / s75 / final; each side at its own pooled T; the rule's panels plus short states split into transfer-v4
development and transfer-r3 test; `runs/r26-readout/vs-r25.md`, `vs-r25-<arm>.json`, from `r26-vs-ref.py --ref r25`):

| vs round 25 (panel, n) | s25 | s50 | s75 | final |
|---|---|---|---|---|
| T: round 26 / round 25 | 1.350 / 1.289 | 1.350 / 1.260 | 1.203 / 1.289 | 1.320 / 1.350 |
| breadth gated acc (2475) | −0.2 [−0.8, +0.4] | −0.0 [−0.6, +0.5] | −0.3 [−0.9, +0.3] | +0.2 [−0.3, +0.7] |
| breadth gated ECE Δ | −0.002 [−0.012, +0.010] | −0.004 [−0.013, +0.005] | +0.000 [−0.009, +0.010] | −0.000 [−0.008, +0.008] |
| tasksource-heldout gated acc (1993) | +1.7 [+0.7, +2.7] | +0.4 [−0.6, +1.5] | +0.6 [−0.5, +1.6] | +0.5 [−0.5, +1.5] |
| tasksource-heldout all 24 families (2788) | +1.1 [+0.2, +2.0] | +0.8 [−0.2, +1.7] | +0.6 [−0.3, +1.6] | +0.6 [−0.2, +1.6] |
| Kev panel acc (3351) | +0.7 [+0.0, +1.4] | −0.2 [−0.9, +0.5] | −0.2 [−1.0, +0.5] | −0.7 [−1.4, +0.0] |
| hard-v1 acc (1083) | +3.0 [+1.2, +4.9] | +1.0 [−0.6, +2.7] | +0.9 [−0.8, +2.6] | +0.2 [−1.4, +1.8] |
| short acc (1586) | −0.1 [−0.6, +0.4] | −1.1 [−2.1, −0.2] | +0.1 [−0.7, +0.9] | −0.3 [−1.1, +0.5] |
| transfer-v4 development acc (576) | +0.2 [−0.3, +0.9] | +0.2 [−1.2, +1.7] | +1.2 [+0.2, +2.6] | +1.2 [+0.2, +2.6] |
| transfer-r3 test acc (1010) | −0.2 [−1.1, +0.5] | −1.9 [−3.2, −0.8] | −0.5 [−1.6, +0.4] | −1.2 [−2.2, −0.3] |
| short confident errors | −0.5 [−1.0, −0.1] | −0.4 [−0.9, −0.1] | −0.4 [−0.9, +0.0] | −0.4 [−0.9, +0.0] |
| CUAD acc (2254) | +0.6 [+0.1, +1.1] | −0.2 [−0.5, +0.0] | +0.0 [−0.3, +0.3] | +0.0 [−0.2, +0.3] |
| CUAD ECE Δ | −0.008 [−0.017, +0.001] | −0.012 [−0.018, −0.005] | −0.002 [−0.008, +0.003] | −0.007 [−0.012, −0.001] |

**What the round showed** (development reads, one round):
- **Twice the tasksource-v1 records did not raise the breadth gains.** Against round 25's arm at the same fraction, breadth
  is level at every point (−0.3 to +0.2, every interval spanning 0), and the final's rank score equals round 25's (+9.8).
  Tasksource-heldout is higher only early (s25 +1.7 [+0.7, +2.7]); by the final the gain is +0.5 [−0.5, +1.5]. Against
  Kev-27B the breadth primary is +0.5 to +1.0 (round 25's lr 1e-6: +0.6 to +0.9).
- **Breadth ECE did not move either:** 0.0197-0.0287 against round 25's 0.0227-0.0283 at the same fractions; the paired
  deltas against round 25 are −0.004 to +0.000, all intervals spanning 0.
- **The registration's named suspects for short-state drift (replay share 23.5 % instead of 30.2 %, 458 steps instead of
  356) show a small effect, confined to transfer-r3 test:** the short-state panel reads −0.6 to −1.5 pp against Kev-27B
  (round 25's lr 1e-6: −0.3 to −0.8), and s50 fails its guard. Against round 25 the loss is on transfer-r3 test (s50
  −1.9 [−3.2, −0.8], final −1.2 [−2.2, −0.3]), while transfer-v4 development gains (+1.2 [+0.2, +2.6] at s75 and the final)
  and short-state confident errors fall (−0.4 to −0.5 pp).
- **CUAD calibration is a little better than round 25's, not fixed:** CUAD ECE 0.078-0.099 (round 25 0.087-0.103, Kev-27B
  0.063); final −0.007 [−0.012, −0.001] against round 25's final. It still grows with steps (s25 0.078 → final 0.096).
- **Temperature (report only, `runs/r26-readout/breadth-ece-by-t.json`, `both-bars-by-t.json`):** as in round 25, the
  gated breadth panel's ECE-minimising T is 1.44-1.56, above every pooled fit, and tasksource-heldout's is 0.92-1.14
  (`27b-k-w85`: 1.32 and 1.10). Unlike round 25, for s50 and the final some temperatures inside the pooled 90 % interval
  would meet both ECE bars together (41-point grid: s50 14 points, T 1.368-1.447; final 6 points, T 1.367-1.408; the
  final's closest point, T 1.372, reads breadth 0.0168 and tasksource-heldout 0.0473). For s25 and s75 none does. The rule
  serves the pool fit, so this changes no verdict.

**Report only: against round 23's confirmed `27b-k-w85`.** No candidate was named, so every arm is compared.
- **Method:** each side at its own pooled T (w85 1.320). Paired bootstrap on the development reads, the same panels and
  exclusions, with `short` split into transfer-v4 development and transfer-r3 test.
- **Files:** `runs/r26-readout/vs-w85.md` and `vs-w85-<arm>.json`, from `runs/r26-readout/r26-vs-ref.py --ref w85`
  (the w85 rows are read from the round-23 checkout).
- **Result: w85 is ahead or level on nearly every panel.**
  - Against the final: breadth −0.6 [−1.4, +0.2], tasksource-heldout −0.9 [−2.2, +0.4], Kev panel −2.6 [−3.6, −1.7]
    (hard-v1 −5.3, the rest −1.3), agents-ood −1.6 [−2.2, −1.0].
  - Short states are level: −0.1 [−1.0, +0.9]; transfer-v4 development +0.0 [−1.6, +1.7], transfer-r3 test −0.1
    [−1.2, +1.0]. The candidate has more confident errors: short +0.9 [+0.4, +1.5] pp.
  - CUAD accuracy is level (−0.6 [−1.7, +0.4]), with CUAD ECE −0.001 [−0.014, +0.009]. By length, w85 reads 0.856 / 0.096
    (< 8k) and 0.839 / 0.102 (16k+), against 0.849 / 0.093 and 0.834 / 0.104. Breadth ECE-by-T for w85: 0.0103 at its
    pooled T.
- **Ranking:** no round-26 arm is ahead of w85 with a lower bound above 0 on any panel the rule gates. The report-only
  exceptions are CUAD ECE at s25 (−0.019 [−0.032, −0.007]) and s50 (−0.012 [−0.024, −0.001]), and ood-v2 ECE at s75
  (−0.005 [−0.010, −0.001]). Kev-27B and `27b-k-w85` remain Jared's release options ("Round 23 confirmation"); nothing here
  changes that choice.

**Deviations:**
- **(i) The final's reads were launched through the registered spend gate, not by the watcher.** The watcher launches a
  finished trial's reads without a gate. At the trial's end the projected spend (~$160-200) plus two running batches
  ($802) plus the final's ($401) could have passed $1,373. So the watcher was stopped at 08:02Z, while the trial was
  still training, and replaced by a local helper (`runs/r26-readout/r26-finish.sh`). The helper polled the trial's call,
  ready to restart the watcher at once on a timeout. It waited for `27b-lr1e6-s75`'s launch, then for the gate, then
  restarted the watcher at 10:13Z. The watcher pulled the study, launched the final's reads once and wrote the read-out.
  The order, s75 at the trial's end and then the final, is the registration's wall-clock plan.
- **(ii) The gate counted whole batches, as registered** (`runs/r26-readout/r26-gate.py`): $494 while the trial's call
  ran, and $400.97 for each read batch until every one of its 16 jobs had its `report.json`. It read the metered cost
  before every launch. Launches (gate ≤ $1,373 over the $4,453.81 launch reading):
  - s25 at 07:37Z: $917.50;
  - s50 at 08:00Z: $1,346.60;
  - s75 at 09:32Z: $1,353.53. It held at 08:26Z, when the snapshot was complete ($1,782.50), and launched once the study's
    bound was released.
  - the final at 10:15Z: $1,369.81. It held from 09:36Z ($1,758.52) until s25's batch had landed.
  The helpers are `runs/r26-readout/r26-autolaunch.sh` and `r26-finish.sh`; copies of the running scripts, which lived
  in `runs/`.
- **(iii) A helper bug, no effect.** The snapshot launcher first looked for `snapshot.json` one directory above
  `checkpoint/`. It was fixed and restarted at 07:36Z, within a minute of the s25 snapshot's commit, and launched s25 at
  once.
- **(iv) No read failed.** The private suites' development partitions were in the checkout before the deploy
  (sha256-checked against their manifests), so all 64 reads ran first time. The last read landed at 12:53Z. The deploy,
  training and reads ran on app `kev-r26`, and the deployed app was stopped at 12:57Z. `kev-sft` was not touched.

**Spend** (Modal metered, workspace-wide):
- $4,453.81 at the launch reading (07:03Z), against ≈ $4,454 at registration.
- Readings before launches: $4,464.25 (07:16Z), $4,476.34 (07:36Z), $4,504.47 (08:00Z), $4,539.40 (08:26Z),
  $4,604.43 (09:32Z), $4,620.71 (10:13Z), $4,650.37 (11:17Z).
- $4,558.69 at 12:56Z and 13:01Z after the last read. That is below the 11:17Z reading, so the meter was revised down.
- Total: **+$104.88** over the launch reading on the latest reading (+$196.56 on the 11:17Z one), for one one-attempt
  study (~$100 at list) and 64 reads. Metering lag is still to come. The projection was ≈ $4,680-4,710. The night's
  ceiling was ≈ $5,980, with a hard stop at $5,700.

**Evidence** (committed):
- The read-out, its tables and the report-only extras (`vs-r25*`, `vs-w85*`, `breadth-ece-by-t.json`,
  `both-bars-by-t.json`), with the scripts that wrote them and the gate / launch helpers under `runs/r26-readout/`.
- The launch records `runs/r26-reads-*.json`.
- Every public-suite read's report and rows (`runs/r26-27b-<arm>-<tag>`); for the OOD reads, reports only.
- For the trial: `provenance.json`, `result-public.json` (without the per-task table), in-trial transfer rows and
  temperature, `training_metrics.json`, and the three `snapshot.json` records.

Private (`jaredpalmer/kev-private-train` @ `2f2c44d3` under `runs/r26/`, with sha256s in `runs/r26-readout/private-rows.json`;
restore with `scripts/private_rows.py restore`):
- the four tasksource-heldout-v1 reads (rows and reports);
- the ood / agents-ood / guardrails-ood rows;
- the trial's development rows and full `result.json`.

Checkpoints (~51 GB each: 3 snapshots + the final) stay on the `kev-runs` volume.

**Next** (proposals, not registered; each needs its own spec):
- **More breadth data at lr 1e-6 is not the lever.** Two rounds of the same recipe with 13,000 and 26,000 tasksource-v1
  records read the same breadth accuracy and the same breadth ECE. The gap to w85 on the Kev panel (hard-v1 −5.3) and
  on breadth (−0.6) is also unchanged.
- **The breadth-ECE failure now sits inside the temperature's sampling noise for the final.** Temperatures in its pooled
  90 % interval meet both ECE bars. A round that asks whether a larger or different held-out pool (more held-out public
  sources beside transfer-r3's eight and MMLU-Pro) fits a T that serves breadth and tasksource-heldout together would test
  that directly, without training. The pool change would have to be registered before any read under it.
- **Short-state drift tracks the replay share.** Transfer-r3 test fell against round 25 when replay went from 30.2 % to
  23.5 %. Any further continued-SFT round from Kev-27B should hold the replay share, not the replay count.

## Round 27 (registered)

**Question.** Is round 18's 9B documents-and-skills delta a Kev-9B candidate under the audited rule? Round 18 (2026-09-24)
selected no 9B candidate: arm (a) passed both primaries (skills +19.0, documents +7.0) and failed only WANLI-v2, scienthoon
and the pooled externals, all since removed or ungated (scienthoon removed, the audit, "WANLI and TypeSafe removed").
Round 18's verdict stands; this is a new, registered selection over its two finished trials, with no training.

**Arms** (checkpoints on the volume; their development reads from round 18, unchanged): `9b-r18a` =
`runs/r18-9b/00-trial-0` (lr 2e-5), `9b-r18b` = `runs/r18-9b/01-trial-1` (lr 1e-5); parent the released Kev-9B
(`jaredpalmer/kev-9b@2629c06a`, `night2-9b-du/00-trial-0`, its round-18 reads).

**Rule** (round 18's, with the 2026-09-27 audit's verdicts applied as in round 24): primaries hard-v1 + devtools-v1
development pooled accuracy lower > 0 and documents-v1 development lower > 0; guards: short state (transfer-v4 dev +
transfer-r3 test, without `emotion`) accuracy lower ≥ −2 pp, Brier upper ≤ +0.02, confident errors upper ≤ +1 pp;
unknowable share ≤ 0.05 (transfer-v9); hard-v1 ECE ≤ the parent's + 0.01. devtools-v1 drops `flakeflagger` and the
`commitpackft_type` task in every panel (the audit's Kev-panel exclusions). No WANLI, TypeSafe, scienthoon or pooled
external read; SemIf is reported (optional). Candidate: the passing arm with the larger sum of the two primary estimates.

**Temperature.** Every arm is served at one temperature fitted on round 24's held-out pool (transfer-r3 calibration's
eight sources + transfer-v9's `mmlu_pro`, 90 % bootstrap interval; standing rule: never an in-distribution partition),
not the in-distribution T each trial's head carries. `trained_on` lists decision-v7, documents-v1 and round10/skills
(round15/joint is their concatenation); it leaves out `evals/night2` (the released Kev-9B's delta), whose manifest lists
no sources, so the engine cannot check it: its records are four synthetic families (`night2_dates`, `night2_assertion`,
`night2_unknowable`, `night2_unknowable_control`), none in the pool. The parent keeps its shipped T 2.30.

**Confirmation** (round 18's, with the same exclusions): hard-v1 + devtools-v1 test pooled lower > 0 and documents-v1 test
lower > 0 (documents-v2 reported); locked transfer-v4 accuracy ≥ parent − 1 pp and served Brier ≤ parent + 0.005.

**Not blind.** The development reads were made on 2026-09-24 and their headline numbers are in "Round 18 result"; the
WANLI analysis that removed WANLI-v2 compared arm (a) with the parent. Selecting between two finished arms on reads already
seen is optimistic; the untouched test partitions and the locked read are the guard. Publishing a new Kev-9B still needs
Jared's explicit OK.

## Round 27 result (2026-09-30) — Kev-9B v2 confirmed

Both arms pass every criterion (`runs/r27-readout/round27.json`); the candidate is `9b-r18a` (lr 2e-5, the larger
primary sum). Served at the pool temperature (arm (a) T 2.194 [2.047, 2.406], arm (b) 2.000, 648 questions); against the
released Kev-9B at its shipped 2.30:

| arm | primary (hard + devtools dev) | hard-v1 dev | devtools-v1 dev | documents-v1 dev | short acc | short Brier | hard ECE |
|---|---|---|---|---|---|---|---|
| 9b-r18a | **+19.3 [+17.2, +21.6]** | +23.8 | +13.0 | **+7.0 [+4.8, +9.2]** | +0.1 [−1.1, +1.2] | −0.000 [−0.009, +0.009] | −0.023 |
| 9b-r18b | +16.5 [+14.4, +18.7] | +19.0 | +13.0 | +6.4 [+4.3, +8.6] | −0.4 [−1.5, +0.8] | +0.003 [−0.006, +0.011] | −0.014 |

**Confirmation of 9b-r18a** (`runs/r27-verdict/9b-{tests,locked}.json`, each read once): tests PASS — hard-v1 + devtools-v1
test pooled +18.7 [+16.7, +20.8] (hard-v1 +25.0, devtools-v1 +9.9 with the audit's exclusions), documents-v1 test +7.1
[+4.7, +9.2], documents-v2 (reported) +8.0 [+5.9, +10.2]; locked PASS — transfer-v4 test accuracy +0.0 [−1.7, +1.8]
(0.852 both, bar −1 pp), served Brier −0.025 [−0.047, −0.007] (0.199 vs 0.224). The parent's test reads are the 2026-09-30
family reads of `jaredpalmer/kev-9b@2629c06a` (`runs/r27c-9b-parent-*`). Infrastructure: the spec's `app` is `kev-r27`
(the registered `kev` named no deployed app, and the first locked launch failed before reading anything).

## Released: Kev-9B v2 (2026-09-30)

- Hub: v1 (`2629c06a`) tagged `v1` first; v2 uploaded from `/runs/release/kev-9b-r27/checkpoint` by
  `modal_app.py::release_publish --public --confirm-public jaredpalmer/kev-9b --replace` in one commit (`b5d8c18e`): adapter
  sha256 `2b2a70cf…`, `head.pt` `8e1dab2c…` (T 2.1936); the card is `docs/model-cards/kev-9b.md`. `--replace` dropped v1's
  `train.log`, which stays at the tag.
- The staged copy: `modal_app.py::release_copy` from `/runs/r18-9b/00-trial-0/checkpoint` (adapter hash equal); `head.pt` T
  set to the round's pool fit by `scripts/calibrate_checkpoint.py --temperature ... --reason ...` (it cannot list round15/joint's
  sources to refit; `kev.rounds validate` had checked the pool against the arm's `trained_on`).
- Verification (`modal_app.py::release_verify`, fresh HF cache, no token; `runs/rel9-public/`, `scripts/compare_release_rows.py`):
  semif-v1 252 of 252 rows and transfer-v4 development 764 of 764 with argmax equal to round 18's raw reads at T 2.19
  (max |Δp| 0.0014 and 0.0015; not bit-exact against reads made in-trial on another GPU); `@v1` loads v1 at T 2.30 and
  reproduces its served semif-v1 read (max |Δp| 0.0008, 0 flips).
- Numbers: `experiments/releases/kev-9b-r27.json` → `runs/release/kev-9b-r27.json`; README Models row, calibration and
  limits text, figures (`scripts/plot_family.py`, `scripts/plot_tweet.py`), AGENTS.md, claims (901 verified). GitHub release
  `kev-family`: `kev-9b.tar.gz` rebuilt from the Hub commit (+ `locked_test.json`), `SHA256SUMS.txt` regenerated.
- The kev-deploy / kev-finetune `KEV_REF` pins are unchanged: v2 is an adapter of the same shape, served by the same code.

## Released: Kev 1.0 (2026-10-01)

Jared approved the release of PR #206's package. Nothing was trained or re-read; the steps are `docs/releases/kev-1.0.md`'s
maintainer plan, and the record is `runs/release/kev-1.0.json`.

- **Hub.** Each 1.0 card was uploaded as `README.md` in a card-only commit (`parent_commit` = main before), then the
  annotated tag `v1.0` was put on that commit, so `@v1.0` shows the 1.0 card (plan step 4's alternative). Before each upload,
  main had the release weights: 0.8B, 4B and 9B main were the weights revisions, and 27B main (`ef78cc8a`) differed from
  `28be62e9` only in `README.md`. After each upload, every other file has the same blob and LFS sha256, `README.md` matches
  the repo card byte for byte, and `card_data` parses with the card's `base_model` / `base_model_relation`.

  | repo | weights | `v1.0` (card commit) | adapter / head |
  |---|---|---|---|
  | `kev-0.8b` | `9a45d25e` | `bf75a6a8` | `9b908623…` / `f400bd12…` |
  | `kev-4b` | `139fdd94` | `6cfce5c2` | `90e81735…` / `dd633435…` |
  | `kev-9b` | `b5d8c18e` | `db029f08` | `2b2a70cf…` / `8e1dab2c…` |
  | `kev-27b` | `28be62e9` | `af0e6d55` | 11 shards unchanged / `7968f17b…` |
- **GitHub.** The release `kev-1.0` (tag on PR #206's merge, `6b719c3`) is published and marked Latest. Its body is
  `docs/releases/kev-1.0.md` above the maintainer plan, which now includes an Assets section. Assets come from
  `scripts/build_release_assets.py`, and a second build gave the same `SHA256SUMS.txt`: `kev-0.8b.tar.gz` `0ae144c7…`,
  `kev-4b.tar.gz` `2e707e2e…`, `kev-9b.tar.gz` `acd13320…`, `SHA256SUMS.txt` `495d104f…`
  (`runs/release/kev-1.0-{SHA256SUMS.txt,manifest.json}`). Kev-27B is linked as `jaredpalmer/kev-27b@v1.0`. These are the
  replacement assets. The first upload (`943891a2…`, `6f87da10…`, `90518850…`, sums `bc5c520e…`) zeroed every file's mtime,
  so `kev.serve` reported an unpacked checkpoint's release date as 1969-12-31. PR #209 made the builder stamp
  2026-10-01 00:00 UTC (`SOURCE_DATE_EPOCH`) and made `Checkpoint.release_date` read UTC and skip pre-2000 mtimes. The
  tarballs were rebuilt and replaced the same day. Their members are byte-identical; only the mtimes differ
  (`runs/release/kev-1.0.json` `github.replaced_assets`).
- **kev-family** (plan step 7) is retired but not deleted. Its three tarballs and `SHA256SUMS.txt` were removed after checking
  that the adapters and heads inside were byte-identical to 1.0's. It is renamed "Kev family (superseded by Kev 1.0)", its
  body is a pointer to `kev-1.0` that lists the removed assets' hashes, and its tag is kept. Its old notes are in
  `runs/release/kev-family-notes-retired.md`.
- **Verification.** Each `jaredpalmer/kev-<size>@v1.0` was resolved through `kev.checkpoint` with no token and a fresh
  cache. It resolves to its card commit, the adapter, head and 27B shard hashes match, and the temperatures are 2.35 / 2.41 / 2.19 / 1.32.
  On a 2-record smoke benchmark (`evals/smoke-v1` development, MPS, fp32), 0.8B and 4B from the tag scored 2 of 2, and
  `kev-0.8b.tar.gz` from the release gave logits identical to the tag's (`runs/release/kev-1.0-verify/`). The extracted tarball
  served a System One request. The assets were downloaded again anonymously and `shasum -c` passed. The collection lists all
  four repos. The Space is RUNNING and `/decide` answers on Kev-4B and Kev-0.8B. It was not republished: its vendored
  `kev/checkpoint.py` lacks only #200's MLX full-weight path, which the Space does not use.
- `KEV_REF` pins (71d4829) are unchanged: the 1.0 checkpoints need no newer code.

## Next

Goals and open questions, not registered rounds; each becomes a spec and a PLAN section before it runs.

0. **Round 21: retrain full weights, with long context and extended data.** Registered: "Round 21 (registered)" and
   `experiments/rounds/r21.json`; it failed at startup (out of memory, no read) and was registered again as round 22 with
   a per-pass memory ceiling and a smaller training set; round 22 read out with no candidate ("Round 22 result"), and
   round 23 blends its final toward Kev-27B ("Round 23 (registered)"). Two trainer follow-ups round 21's failure
   points at, not in round 22: a padded multi-state pass with long states runs the state through an explicit mask (38.7 s
   per step for two unequal ~19k states, against 18.7 s for one 32k state), so packing unequal states varlen, or
   length-bucketed steps, would buy time; and each start spends ~10 min counting state tokens on every rank, which the
   ranks could split. The notes below are what round 21 started from; its registration records where it departs
   (32k states; none pairs gated at 8k tokens, PR #153; sft-v1's public sources capped at 1,200 train records;
   calibration / development limited to states of at most 8k tokens). Retraining is allowed (rounds 19-20
   showed that post-hoc remedies do not move the accuracy guards). What was decided before registration:
   - Context: 64k tokens is the target, 32k the fallback and 16k the last resort (a fit probe decides before registration).
   - Data: `sft-v1` extended into a new version.
   - Snapshots: kept (0.25 / 0.5 / 0.75 of the steps, #145), read as a report.
   - Calibration: round 20's held-out-datasets pool method.
   - *(Closed 2026-09-27: scienthoon was removed as an eval, see "scienthoon removed"; the two scienthoon items below are kept as written.)*
     The scienthoon remedy follows the round-20 analysis (`runs/r20-scienthoon/analysis.md`). Re-register the scienthoon
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
3. ~~**Publish the documents-v1 and hard-v1 train partitions**~~ Done 2026-09-30: both train partitions (23 MB each, not
   in git) are in `jaredpalmer/kev-suites` at `cc4bac803e73112689ec327ffa481c519cbc7a05`, now `SUITES_REVISION`, and
   `load_split` fetches and hash-checks them (hard-v1 `a08ca9c5…`, regenerated byte for byte by `scripts/build_hard_v1.py`
   before upload; documents-v1 `2d7e4c43…`, public-domain CFPB text with teacher-agreed labels). devtools-v1's train
   partition (8.9 MB) was already in git. Still not in the public mirror: the round-4/5 delta files (`evals/round4/*`,
   `evals/round5/*`, regenerated by their builders) and `evals/round6/b1v2/train.jsonl` (in the private kev-private-train).
4. **Decision Index submission** for Kev-27B and the current Kev-4B / Kev-0.8B.
5. Smaller open items: rotation-averaged Choice met its round-4.4 gate (permutation flips 0.028 → 0.000 at 9B) and waits for
   a product decision (it multiplies latency); date arithmetic stays the weakest family at every size (finding 9).

## Round 28 (registered)

### Round 28 - the small family's shipped temperatures refitted on held-out datasets, and a validated context length per size (post hoc, no training; registered with this spec's commit, before any round-28 read)

**Why.** Kev-4B (round 10, `r10-skills/00-trial-0`, Hub `139fdd94`, T 2.41) and Kev-0.8B (round 15, `r15-08b/00-trial-0`,
Hub `9a45d25e`, T 2.35) ship temperatures fitted on their trials' decision-v7 development rows. Those rows are held-out *items*
of a training corpus: round 19's failure mode, which `docs/autoresearch.md` §3.4 now forbids for any shipped temperature
(finding 5). `kev.rounds.temperature` on those rows reproduces what ships (4B 2.406, 0.8B 2.351). Kev-27B v2 (T 1.32) and
Kev-9B v2 (T 2.19, round 27's `9b-r18a`, `jaredpalmer/kev-9b` main since 2026-09-30, Hub `b5d8c18e`, PR #195) already ship the
held-out pool's fit, so Kev-9B is not an arm here. Its pool T is its shipped T, and its context length is measured on round
29's read of the same checkpoint. Argmax does not depend on T, so this round is about calibration only. Accuracy is identical
on every panel by construction, and it is reported anyway.

**Arms** (`experiments/rounds/r28.json`). `4b-r10` and `08b-r15` are the released checkpoints, each served at the temperature
fitted on the pool below. Each arm's parent is the same checkpoint at its shipped T. The engine serves a parent at its trial's
development-rows fit, which here is the shipped T. Both sides read the same rows, so every delta is a paired comparison of one
checkpoint's logits at two temperatures.

**Temperature pool** (rounds 20-27's, unchanged): the eight held-out sources of transfer-r3's calibration partition plus
transfer-v9's `mmlu_pro`, minus the transfer-v4 development records, with a 90 % bootstrap interval (`temperature.ci`).
`trained_on` lists what each checkpoint trained on:
- Kev-4B: the night-2 4B, then round 8's documents-v1 delta (2,000 decision-v7 replay), then round 10's hard-v1 + devtools-v1
  delta (`evals/round10/skills`, 4,000 replay).
- Kev-0.8B: the night-2 0.8B, then round 15's joint documents + skills delta (6,000 replay).
- Both arms name `evals/v7/decision-v7`, `evals/documents-v1`, `evals/round10/skills`, `evals/hard-v1` and `evals/devtools-v1`.
  `evals/round15/joint` is the concatenation of those, and its manifest lists no sources.
- `evals/night2` is left out, as in round 27. Its manifest lists no sources, so `validate` would refuse it as unlistable
  training. Its records are four synthetic families (`night2_dates`, `night2_assertion`, `night2_unknowable`,
  `night2_unknowable_control`), and none of them is in the pool.

`kev.rounds validate` finds **no pool conflict**: 24 training sources per arm, none of them pooled. As a sanity check, adding
`evals/round3/transfer-r3` to an arm's `trained_on` makes it refuse the round.

**Rule** (pool-T side minus shipped-T side, paired record-clustered bootstrap, 2,000 resamples, seed 0, micro; `drop_ids` as
in rounds 24 and 27):
1. Primary: the pool T must be strictly better calibrated on **breadth-v1 dev + tasksource-heldout-v1 dev pooled**, with the
   audited exclusions (breadth without `routerbench`, `cfcolor`, `humicroedit`, `chessbench`; tasksource-heldout without the
   seven families of round 24's private list, `runs/r24-private/tsheld-exclude.json`, sha256 `a72030ab…`). Two conditions:
   the Brier delta's upper bound < 0, **and** ECE (pool) < ECE (shipped).
   - Why Brier carries the interval: `kev.rounds` bootstraps both metrics. Brier is a proper score and additive per question,
     so its record-clustered bootstrap is the exact statistic, and it is also what the locked stage reads.
   - ECE is binned (10 bins) and not additive, so its resampled differences carry binning noise (the audit measured an ECE
     sampling sd of 0.006-0.010 per panel).
   - The ECE point condition stops a temperature that sharpens toward a better Brier but worse reliability from passing.
2. Guards: ECE (pool) ≤ ECE (shipped) + 0.005 on each of:
   - breadth-v1 dev (audited);
   - tasksource-heldout-v1 dev (audited);
   - transfer-v4 dev, the in-trial read, without `emotion`;
   - hard-v1 dev;
   - devtools-v1 dev, without `flakeflagger` and `commitpackft_type`;
   - documents-v1 dev.
3. Candidate: an arm that passes. There is one arm per size, so the arm is the size's candidate.

Report only (optional panels, never gating): accuracy (identical) and Brier on every panel; the unknowable share on transfer-v9;
breadth over all 14 sources; tasksource-heldout over all 24 families; ood-v2, agents-ood-v1 and guardrails-ood-v1 accuracy and
ECE; longdoc-v1 dev accuracy and ECE per length bucket at both temperatures. The by-length panels use edges 4,096 / 8,192 / 16,384
/ 32,768 (buckets `under_4k` … `32k_plus` = longdoc's nominal 4k / 8k / 16k / 32k / 64k). The small family's tokenizer counts
longdoc's states exactly as the suite's Qwen3.8-27B tokenizer does (ratio 1.0000 on 150 records). CUAD ECE is report only because
its labels are unreliable for calibration (the audit).

**Validated context length** (report only, registered here, gates nothing). The rule is computed by
`scripts/longdoc_report.py --context-margin -0.03` (`validated_context`, unit-tested) from each size's longdoc-v1
development read: Kev-4B and Kev-0.8B from this round, Kev-9B v2 from round 29's `9b-r18a` read of the same checkpoint (or,
if round 29 replaces v2, from that checkpoint's read).
- Statistic: for bucket b ∈ {16k, 32k, 64k}, the CUAD accuracy difference from the **8k bucket** (states of 6,553-7,618
  tokens: the 4-8k bucket the small family was trained at, `max_state` 7,552). It is paired on the same (target contract,
  repeat, question), about 445 questions per bucket, with a 95 % target-clustered bootstrap (2,000 resamples, seed 0).
  This is the `cuad_paired_vs_8k` statistic longdoc-v1 was built for (#150).
- b is within tolerance when its interval's **lower bound ≥ −3 pp** and every record of b and of 8k was answered.
- Validated context length = the nominal size of the largest bucket such that it and every bucket between it and 8k are within
  tolerance (16,384 / 32,768 / 65,536; 65,536 is the serving cap). If 16k fails, it is 8,192: the trained length, not extended.
  An unread or partly read bucket is not within tolerance.
- The generated half, the unpaired differences from the 4k control, the ECE per bucket and the served accuracy per bucket are
  reported next to it.
- Sizing. On Kev-27B's own read the paired intervals are 1.5-1.7 pp wide on each side (validated to 65,536 by this rule), so
  a model with no drop fails a bucket about 5 % of the time. The small models agree less with themselves across lengths, so
  their intervals will be wider. The rule then errs toward 8,192: it under-claims and never over-claims. An unpaired
  comparison with the 4-8k buckets pooled would have needed a margin of about 7 pp for the same error rate.

**Blocker: the long reads run out of memory at these sizes.** The 2026-09-30 family longdoc-v1 *test* reads failed with
`OutOfMemoryError` (`/bench/fam-*-longdoctest/failure.json` on the volume):
- Kev-4B, Kev-9B v1 and Kev-9B v2 at the first 32k record (720 of 1,200 records scored);
- Kev-0.8B at the first 64k record (960 scored), on an H100 and again on an H200.

Kev-27B's reads of the same suite completed, but they ran a bf16 backbone. The same wall should stop agents-ood-v1 (47 of 373
development states over 26k tokens, max 51,148) and guardrails-ood-v1 (12 of 1,263, max 39,072).

A plausible cause, not verified: these reads run in fp32 at a head size of 256. That rules out SDPA's flash kernel and maybe
the memory-efficient one too, which leaves the math kernel. Its fp32 L × L scores take about 58 GB for the 4B and 9B at the
32k bucket (16 heads × 30k² × 4 B) and about 115 GB for the 0.8B at 64k (8 heads). That fits where each size failed. Whatever
the fix is, it is an evaluator change: its own PR, `kev-verify`, before these reads.

Until then the longdoc, agents-ood and guardrails-ood reads are phase B (below) and stay unlaunched. Round 28's verdict does not
need them (its long and OOD panels are report only). Without phase B, no size has a validated context length.

**Confirmation** (per size whose arm passes, once; computed from rows that already exist, with no new read):
- `tests`: breadth-v1 test (without the four sources) ECE (pool) < ECE (shipped), Brier and the 14-source panel reported. The
  rows are the 2026-09-30 family reads `runs/fam-4b-breadthtest` and `runs/fam-08b-breadthtest` (the Hub checkpoints, raw
  logits).
- `locked`: transfer-v4 locked served Brier (pool) ≤ Brier (shipped) + 0.005, and accuracy identical (two criteria, ≥ 0 and ≤ 0).
  The rows are the release locked reads `runs/locked/kev-4b-r10-ungated` and `kev-08b-r15-ungated`.
- These partitions have been read once for these very checkpoints, and they are not read again. What has not been computed
  before this registration is their calibration at the pool T, and the pool fixes that T before either set is looked at. No
  committed record reports these checkpoints' breadth-v1 test ECE.
- Then `scripts/calibrate_checkpoint.py` in pool mode writes the pool T into a release copy's `head.pt` (`--rows <r3cal
  rows>:composition_holdout,emotion,legacy_holdout,mmlu,paws,qnli,sciq,tweet_offensive --rows <v9 rows>:mmlu_pro
  --exclude_rows <transfer rows>`, the same `pool_conflicts` check). Publishing it, and moving the model cards' served numbers,
  needs Jared's OK.

**Reads this round needs that do not exist yet** (the sweep after merge; H100). Each is one job of `modal_app.py::benchmarks`
at its suite's timeout. `launch-reads --dry-run` prints them, and phase-B jobs have to be dropped from its batch by hand.

| phase | reads | count |
|---|---|---|
| A (now) | `r28-{4b-r10,08b-r15}-{tsheld,r3cal,ood}` | 6 |
| B (after the memory fix) | `r28-{4b-r10,08b-r15}-{longdoc,agentsood,guardood}` | 6 |

Existing reads it uses: breadth-v1 dev `runs/fam-{4b,08b}-breadth`; hard-v1 / devtools-v1 / documents-v1 / transfer-v9
`runs/r10-4b-skills-*` and `runs/r15-08b-a-*`; the trials' in-trial transfer-v4 reads; for confirmation `runs/fam-{4b,08b}-breadthtest`
and the two locked reads. The fam-* rows are on the volume only (`modal volume get kev-runs /bench/fam-4b-breadth runs/`); the
rest are in the research checkout. The budget is shared with round 29, below.

## Round 28 result (2026-10-01) — no candidate at either size; validated context 8,192 (0.8B, 4B, 9B v2) and 65,536 (27B v2)

**Verdict** (`runs/r28-readout/round28.json`, re-read with every phase-B panel present; unchanged from phase A's read-out in
PR #203): no candidate. Kev-4B keeps T 2.41 and Kev-0.8B T 2.35.
- `4b-r10` (pool T 2.297 [2.047, 2.520]) fails both primary conditions: Brier −0.0001 [−0.0005, +0.0003], ECE 0.0252
  against 0.0240.
- `08b-r15` (pool T 2.520 [2.194, 2.828]) passes both (ECE 0.0375 against 0.0484, Brier −0.0020 [−0.0025, −0.0014]) and fails
  the hard-v1, devtools-v1 and documents-v1 ECE guards (+0.011, +0.007, +0.020 against a tolerance of 0.005).

**Report-only phase-B panels** (pool T minus shipped T, ECE; accuracy identical by construction):

| panel | `4b-r10` ECE pool / shipped, Δ [95 % CI] | `08b-r15` ECE pool / shipped, Δ [95 % CI] |
|---|---|---|
| longdoc-v1 CUAD (2,254 q) | 0.023 / 0.031, −0.008 [−0.012, +0.004] | 0.061 / 0.053, +0.008 [−0.006, +0.016] |
| longdoc-v1 generated (2,400 q) | 0.090 / 0.100, −0.010 [−0.011, −0.009] | 0.143 / 0.131, +0.012 [+0.011, +0.012] |
| agents-ood-v1 (2,084 q) | 0.193 / 0.203, −0.010 [−0.010, −0.010] | 0.103 / 0.097, +0.006 [+0.001, +0.013] |
| guardrails-ood-v1 (4,949 q) | 0.042 / 0.051, −0.009 [−0.013, −0.004] | 0.051 / 0.063, −0.011 [−0.012, −0.007] |

The pool T would have helped Kev-4B a little on every long and OOD panel and moved Kev-0.8B both ways; neither changes the
verdict, which the rule fixed on the gating panels. Kev-4B's agents-ood-v1 ECE (0.20) is the worst calibration of any
panel it was read on.

**Validated context length** (report only; `runs/r28-readout/context.{json,md}`, from `runs/r28-context/report.json`, the
registered `scripts/longdoc_report.py --context-margin -0.03` command plus Kev-27B v2's round-23 read, and
`runs/r28-context-served` for ECE at the shipped T). CUAD accuracy difference from the 8k bucket, paired on 445-447
questions, pp [95 % CI]:

| size | read | 16k | 32k | 64k | validated |
|---|---|---|---|---|---|
| Kev-0.8B | `r28-08b-r15-longdoc` | −5.2 [−8.5, −2.1] | −6.0 [−9.5, −2.5] | −7.9 [−11.8, −4.2] | **8,192** |
| Kev-4B | `r28-4b-r10-longdoc` | −1.1 [−3.4, +1.2] | −5.8 [−9.0, −2.8] | −5.2 [−8.2, −2.0] | **8,192** |
| Kev-9B v2 | `r29-9b-r18a-longdoc` | −1.4 [−3.7, +0.9] | −3.6 [−6.4, −0.9] | −5.2 [−7.9, −2.5] | **8,192** |
| Kev-27B v2 | `r23-27b-k-w85-longdoc` | +0.2 [−0.7, +1.2] | −0.2 [−1.2, +0.7] | −1.1 [−2.4, +0.0] | **65,536** |

- Kev-4B and Kev-9B v2 miss the 16k tolerance by 0.4 and 0.7 pp on the lower bound, with point estimates near −1 pp. The
  registration predicted wider intervals for the small models and a rule that errs toward 8,192, and that is what happened.
  At 32k all three small models are measurably below 8k (every upper bound < 0), so a longer claim would not have survived
  a looser margin either.
- Kev-0.8B already loses 6.8 pp from 4k to 8k (0.779 → 0.711; unpaired, different contracts), inside its trained length.
- The values are on the Kev 1.0 cards, README and release notes (PR #206, merged).

**Reads** (phase B, H100, `kev-sweepB`): `r28-{4b-r10,08b-r15}-{longdoc,agentsood,guardood}`. `r28-4b-r10-agentsood` was
lost in wave 1 (its command spawned two jobs, and a detached `modal run` keeps only the last one alive once the local client
goes; the volume kept a partial `predictions.jsonl`) and was re-read alone at a 10,800 s timeout as
`/bench/r28-4b-r10-agentsood-t10800` (373/373 records, 61 min), copied to `runs/r28-4b-r10-agentsood`. The wave-1 partial
stays on the volume. Rows of the agents-ood and guardrails-ood reads are private (held-out records of sft-v2's components);
their reports are here.

## Round 29 (registered)

### Round 29 - retrospective selection among every 9B delta of rounds 7, 9, 11, 16 and 18 under round 24's audited rule (post hoc, no training; registered with this spec's commit, before any round-29 read)

**What this is, plainly.** This is **post hoc selection among checkpoints that were already trained and already read on
development data**, the 9B analogue of round 24. The rounds that made them selected no 9B candidate (7, 9, 11, 16, 18); round 27
later selected and confirmed round 18's arm (a), which is now Kev-9B v2. Those verdicts stand, and nothing here revisits them.
The rule is round 24's audited rule, fixed in this commit before any computation under it. It is not blind: the documents,
hard-v1 and devtools-v1 development reads of several arms are in their rounds' read-outs, and round 27's table shows 18a and 18b.
Picking the best of 11 on development panels is optimistic by construction, and the guard against that is the confirmation,
on test partitions that no 9B checkpoint in this round except `9b-r18a` has read.

**Parent: the Kev-9B these arms were all trained from, not Kev-9B v2.** Every arm is one epoch from `jaredpalmer/kev-9b` as
it was before 2026-09-30: `night2-9b-du/00-trial-0`, Hub `2629c06a`, now the `v1` tag, T 2.30, which its development rows
reproduce (2.297). Kev-9B v2 is `9b-r18a` itself, and it is an arm. Two reasons for this choice:
- `kev.rounds` serves a parent at its trial's development-rows fit, which for `r18-9b/00-trial-0` is 2.297, not v2's shipped
  2.19. v2 cannot be a parent served at its shipped T without an engine change.
- As an arm served at the pool T, `9b-r18a` is v2 exactly as it ships: the pool fit on its reads is 2.1936, the value written
  into its `head.pt`. So every arm, v2 included, is compared with v1 on the same rows, and the rank puts each arm directly
  against v2.

The brief asked for "parent = the released Kev-9B at its shipped T". When it was written, the released Kev-9B was v1.

**Arms** (all on the `kev-runs` volume; none missing). Round 12's two 9B skills arms (`r12-skills/00-trial-0`, `01-trial-1`,
also on the volume) are outside this round's registered scope.

| arm | checkpoint | delta (one epoch from Kev-9B v1) | trained on |
|---|---|---|---|
| `9b-r7-s1`, `9b-r7-s2` | `r7-docs/00-trial-0`, `01-trial-1` | documents-v1, replay 2,000, lr 2e-5, seeds 1 / 2 | decision-v7, documents-v1 |
| `9b-r9-a`, `-b`, `-c` | `r9-docs/0{0,1,2}-trial-*` | documents-v1; (replay, lr) = (6,000, 2e-5), (2,000, 1e-5), (6,000, 1e-5); seed 3 | decision-v7, documents-v1 |
| `9b-r11-s4`, `9b-r11-s5` | `r11-docs/00-trial-0`, `01-trial-1` | documents-v1, replay 6,000, lr 2e-5, seeds 4 / 5 | decision-v7, documents-v1 |
| `9b-r16-lr1e5`, `9b-r16-lr2e5` | `r16-9b/00-trial-0`, `01-trial-1` | hard-v1 + devtools-v1 (`round10/skills`), replay 10,000 | decision-v7, round10/skills, hard-v1, devtools-v1 |
| `9b-r18a` (= Kev-9B v2), `9b-r18b` | `r18-9b/00-trial-0` (lr 2e-5), `01-trial-1` (lr 1e-5) | documents + skills (`round15/joint`), replay 10,000 | decision-v7, documents-v1, round10/skills, hard-v1, devtools-v1 |

`evals/night2` (v1's own delta) is left out of `trained_on` for the reason given in round 28. `validate` finds **no pool
conflict** for any arm.

**Temperature.** Every arm is served at round 28's pool: transfer-r3 calibration's eight sources plus transfer-v9 `mmlu_pro`,
minus the arm's transfer-v4 development records, with a 90 % interval. The parent keeps its shipped 2.30.

**Rule** (round 24's, verbatim except where noted; against Kev-9B v1; paired record-clustered bootstraps, 2,000 resamples,
seed 0, micro; `drop_ids`):
1. Primaries:
   - breadth-v1 dev accuracy lower > 0, without `routerbench`, `cfcolor`, `humicroedit`, `chessbench`;
   - tasksource-heldout-v1 dev accuracy lower > 0, without the seven families (the same private list);
   - Kev panel accuracy lower ≥ −1 pp: transfer-v4 dev without `emotion`, hard-v1, devtools-v1 without `flakeflagger` and
     `commitpackft_type`, and documents-v1.
2. Guards:
   - short state (transfer-v4 dev + transfer-r3 test, both without `emotion`): accuracy lower ≥ −2 pp, Brier upper ≤ +0.02,
     confident errors upper ≤ +1 pp;
   - unknowable share on transfer-v9 ≤ 0.05;
   - longdoc CUAD: accuracy lower ≥ −2 pp, and CUAD 16k+ accuracy (arm − parent) ≥ −2 pp.
3. Calibration: breadth, Kev-panel and tasksource-heldout ECE ≤ the parent's + 0.01, with the same exclusions.
4. Candidate: the passing arm with the largest breadth + tasksource-heldout + Kev-panel accuracy gain.

What round 24 had that this round drops: its WANLI-v2 and TypeSafe panels (both suites were removed on 2026-09-30), and any
pooled-externals or scienthoon read. SemIf is reported (optional). Also reported: breadth over all 14 sources and over the three
moved sources, tasksource-heldout over all 24 families, hard-v1 alone, the Kev panel without hard-v1, longdoc generated,
longdoc by nominal bucket, and ood-v2 / agents-ood-v1 / guardrails-ood-v1.

**What each outcome means** (written before any read):
- *No arm passes.* There is no candidate. Kev-9B v2's release stands, as round 27 registered it, and the result is recorded
  (including `9b-r18a`'s own failures, if any, as information).
- *`9b-r18a` is the candidate.* The audited rule agrees with round 27, and Kev-9B v2 stands. There is **no new confirmation**:
  its hard-v1, devtools-v1, documents-v1, documents-v2 and locked test reads were round 27's confirmation, and its breadth-v1 test
  was read in the 2026-09-30 family reads, so none of them can be read again as a confirmation.
- *Another arm X is the candidate.* X is confirmed against v1 by the stages below. Its accuracy on the same test items
  against Kev-9B v2 is reported next to each stage (`versus`: v2's round-27 test reads and its family breadth-v1 test read;
  accuracy only, which does not depend on T). Replacing v2 with X is Jared's decision, and only if X passes every stage.

**Confirmation** (X only, each read once; round 24's stages at 9B):
- `tests`:
  - breadth-v1 test accuracy lower > 0 (the same four sources out; all 14 reported);
  - tasksource-heldout-v1 test lower > 0 (the seven families out);
  - pooled hard-v1 + devtools-v1 (without `flakeflagger` and `commitpackft_type`) + documents-v1 test lower ≥ −1 pp;
  - documents-v2 and longdoc-v1 test (CUAD by length, generated) reported.
  - Parent test reads: `runs/fam-9b-breadthtest` and round 27's `runs/r27c-9b-parent-{hardtest,devtest,docs1test,docs2}`. New:
    `runs/r29c-9b-parent-{tshtest,longdoctest}`.
- `locked`: transfer-v4 locked accuracy ≥ v1's − 1 pp, and served Brier ≤ v1's + 0.005 (`kev-9b-r29-ungated`). This is round 24's
  construction: its absolute bars were Kev-27B's own −1 pp / +0.005.
- Before any release, the bf16 serving check: `modal_app.py::serving --run /runs/<X>/checkpoint --gpu H100 --name
  serving-9b-r29 --flags=--isolation`, max |Δp| ≤ 0.03 and ≤ 1 flip in 280. The release temperature is the pool fit, written
  by `scripts/calibrate_checkpoint.py` as in round 28.

**Blocker.** The CUAD guards gate, so round 29 cannot be read out until the longdoc-v1 reads can be made at 9B (round 28's
blocker: the family test reads ran out of memory at the first 32k record). The rule is not changed to get around that. Making
the long panel report-only at 9B would be a re-registration, and that is Jared's call, before any round-29 read.

**Reads this round needs that do not exist yet** (the sweep after merge; H100):

| phase | reads | count |
|---|---|---|
| A (now) | parent `r29-P9-{tsheld,ood}`; `breadth` for the 10 arms other than 18a; `tsheld` and `ood` for all 11; `hard` and `devtools` for r7 ×2, r9 ×3, r11 ×2; `r3test` for r7 ×2 and r9 ×3; `r3cal` for the 9 arms other than 18a / 18b | 2 + 10 + 22 + 14 + 5 + 9 = 62 |
| B (after the memory fix) | `longdoc`, `agentsood`, `guardood` for the parent and all 11 arms | 36 |
| confirmation (X ≠ 18a only) | `r29c-9b-cand-{breadthtest,tshtest,hardtest,devtest,docs1test,docs2}`, `r29c-9b-parent-tshtest`, the locked read, the serving check; phase B: `r29c-9b-{cand,parent}-longdoctest` | 7 + 1 + 2 = 10, + serving |

Existing reads it uses:
- parent: `runs/fam-9b-breadth`, `hv1-P9`, `dt1-P9`, `docs1-P9`, `n2-9b-du-v9`, `rc-parent-r3test`, `r5r-P9-semif`, the locked read;
- arms: their rounds' `docs` / `v9` / `semif` (and `hard` / `devtools` / `r3test` for r11, r16 and r18); round 27's `r27-9b-r18{a,b}-r3cal`;
  `runs/fam-9bnew-breadth` for 18a; every trial's in-trial transfer-v4 read.

They are in the research checkout, `/tmp/kev-r27` (fam-9b*, not yet committed) and the volume.

**Budget (rounds 28 + 29 together).** H100 at $5.675/h per container (`kev.budget.hourly_rate`). Expected costs are estimated
from past reads' latencies at 9B (breadth ~10 min, hard ~7, tasksource-heldout ~6, ood ~6.5, the short suites ~4-5, each
including the load); the 4B and 0.8B are cheaper.

| part | reads | expected | admission bound |
|---|---|---|---|
| phase A | 68 (6 + 62) | ≈ $40 | $193 (68 × $2.84 at 1,800 s) |
| phase B | 42 (6 + 36) | ≈ $115 | $318 (14 longdoc × $17.03 at 10,800 s + 28 × $2.84) |
| round 29 confirmation, if X ≠ 18a | 10 (8 now, 2 phase B) + serving | ≈ $30 | ≈ $77 |
| round 28 confirmation | 0 (existing rows) | $0 | $0 |
| **total** | 120 + serving | **≈ $185** | $588 if all in flight at once |

The cap is $300 with a 10 % reserve, so $270 is plannable against the metered reading at launch. At registration Modal read
$4,633.77 metered (2026-09-30T22:45Z); the last night ceiling was $5,980.

The expected total fits. The bounds do not fit all at once, so the sweep launches in waves, each while (metered − baseline) +
the bounds in flight < $270. Order: phase A; round 28's read-out; the memory-fix PR; phase B, longdoc first; round 29's
read-out; then confirmation. A 9B agents-ood read may need more than its 1,800 s default (Kev-27B's took about 66 min):
relaunch it alone with `--timeout 3600` rather than raising the spec's `read_timeout`, which would double every 9B bound.

## Round 29 result (2026-10-01) — no candidate; Kev-9B v2 stands (read out with ten arms' longdoc reads missing)

**Verdict** (`runs/r29-readout/round29.json`, `readout.txt`): **no arm passes**, so there is no candidate, no confirmation is
read, and Kev-9B v2's release stands as round 27 registered it. The read-out is not complete: phase B's longdoc-v1 reads of
the ten arms other than `9b-r18a` were stopped at 750-800 of 1,200 records for the budget (below), so the engine marks those
arms `incomplete`. That cannot change the verdict. Every arm already fails primary 1's tasksource-heldout-v1 condition
(accuracy lower bound > 0) on reads that are complete, and a missing read can only add failures: an arm passes only if every
criterion passes.

Against Kev-9B v1 (`jaredpalmer/kev-9b@2629c06a`, shipped T 2.30), each arm at the pool T; accuracy Δ in pp [95 % CI],
paired record-clustered bootstrap:

| arm | breadth-v1 dev (audited) | tasksource-heldout-v1 dev (audited) | Kev panel | short state | longdoc CUAD | failed criteria |
|---|---|---|---|---|---|---|
| `9b-r7-s1` | +0.9 [+0.2, +1.7] | −0.1 [−1.0, +0.9] | +1.2 [+0.4, +2.1] | −1.0 [−2.0, −0.2] | not read | `1_tasksource_heldout_lower_above_0` |
| `9b-r7-s2` | +0.7 [+0.0, +1.5] | −0.6 [−1.5, +0.4] | +1.9 [+1.1, +2.8] | −0.8 [−1.8, +0.2] | not read | `1_breadth`, `1_tasksource_heldout` |
| `9b-r9-a` | +1.0 [+0.3, +1.8] | −0.8 [−1.7, +0.2] | +1.7 [+0.8, +2.6] | −0.5 [−1.5, +0.4] | not read | `1_tasksource_heldout` |
| `9b-r9-b` | +0.5 [−0.1, +1.1] | −0.1 [−1.0, +0.8] | +1.5 [+0.7, +2.3] | −0.1 [−0.9, +0.6] | not read | `1_breadth`, `1_tasksource_heldout` |
| `9b-r9-c` | +0.6 [−0.0, +1.2] | −0.7 [−1.6, +0.2] | +1.6 [+0.8, +2.4] | −0.5 [−1.4, +0.3] | not read | `1_breadth`, `1_tasksource_heldout` |
| `9b-r11-s4` | +0.3 [−0.4, +1.0] | −0.5 [−1.4, +0.5] | +1.6 [+0.8, +2.4] | −0.7 [−1.5, +0.0] | not read | `1_breadth`, `1_tasksource_heldout`, `3_kev_ece` |
| `9b-r11-s5` | +0.4 [−0.2, +1.2] | −0.8 [−1.8, +0.2] | +1.2 [+0.3, +2.1] | −1.1 [−2.1, −0.1] | not read | `1_breadth`, `1_tasksource_heldout`, `2_short_acc`, `3_kev_ece` |
| `9b-r16-lr1e5` | +0.3 [−0.7, +1.2] | −0.9 [−2.3, +0.5] | +8.7 [+7.4, +10.0] | −0.3 [−1.5, +0.9] | not read | `1_breadth`, `1_tasksource_heldout`, `3_kev_ece` |
| `9b-r16-lr2e5` | +0.6 [−0.4, +1.5] | −0.7 [−2.1, +0.7] | +9.9 [+8.6, +11.2] | −0.1 [−1.3, +1.1] | not read | `1_breadth`, `1_tasksource_heldout`, `3_kev_ece` |
| `9b-r18a` (= v2) | +0.2 [−0.8, +1.2] | +0.6 [−0.8, +2.0] | +12.6 [+11.3, +14.1] | +0.1 [−1.1, +1.2] | +2.5 [+0.8, +4.3] | `1_breadth`, `1_tasksource_heldout` |
| `9b-r18b` | +0.2 [−0.8, +1.2] | −0.3 [−1.6, +1.1] | +10.8 [+9.4, +12.2] | −0.4 [−1.5, +0.8] | not read | `1_breadth`, `1_tasksource_heldout`, `3_kev_ece` |

(`1_breadth` = `1_breadth_lower_above_0`, `1_tasksource_heldout` = `1_tasksource_heldout_lower_above_0`, `2_short_acc` =
`2_short_acc_lower_at_least_minus_2pp`, `3_kev_ece` = `3_kev_ece_at_most_parent_plus_0.01`.)

- **What it says.** None of the eleven 9B deltas of rounds 7-18 beats Kev-9B v1 on held-out *datasets* by a resolvable
  margin: the documents-only deltas buy +0.5 to +1.0 pp on breadth-v1 (two of them with a lower bound above 0) and lose
  0.1-0.9 pp on tasksource-heldout-v1; the skills deltas (16, 18) gain 8.7-12.6 pp on the trained-family Kev panel and nothing
  measurable on either held-out panel. The audited rule would not have selected v2 either; v2 was selected by round 27's rule
  (trained-family panels with held-out guards), and it passes every guard and calibration criterion here.
- **Kev-9B v2 (`9b-r18a`) against v1**, complete: breadth-v1 dev 0.781 vs 0.779, tasksource-heldout-v1 dev 0.706 vs 0.700
  (ECE 0.032 vs 0.057), Kev panel 0.849 vs 0.723, short state 0.871 both, longdoc-v1 CUAD 0.837 vs 0.811 (+2.5 [+0.8, +4.3];
  16k+ 0.810 vs 0.792), ood-v2 0.889 vs 0.882. Its CUAD guards pass.
- Outcome under the registration's "What each outcome means": *No arm passes*. The result is recorded, including v2's
  own primary failures as information. Nothing is confirmed and no test partition was read.

**Reads and budget.** Phase B ran on H100 (`kev-sweepB`). Read and committed: `r29-P9-longdoc` and `r29-9b-r18a-longdoc`.
Stopped: the other ten arms' longdoc reads, launched 16:05Z with one job per detached `modal run`. They were stopped at
18:40Z at 750-800 of 1,200 records, because the workspace read $4,987.19 metered (September $4,682.78 + October) against the
session's $5,100 stop. Their own remaining ≈ 2.1 h × 10 × $5.675 ≈ $119 would have crossed it with no other spend, and other
sessions' jobs were adding about $40-90/h. Their partial `predictions.jsonl` stay on the volume. Not launched: agents-ood-v1
and guardrails-ood-v1 for the parent and all eleven arms (24 reads, ≈ $180 at the 9B rates measured here: agents-ood about
10 s per record at 4B). Both panels are report only (`optional`), so their absence leaves the comparisons complete. Finishing
the read-out formally needs the ten longdoc reads (≈ $270 from scratch at about 4.7 h each), and it would not change the
verdict.

## Round 30 - Gemma 4 31B IT under round 24's rule (registered 2026-10-05)

**Question.** Does Gemma 4 31B IT (`google/gemma-4-31B-it@842da379`, dense, attention-only with sliding-window layers) close
the gap to Kev-27B v2 when it gets v2's training rather than the one-epoch v7 LoRA recipe? The report-only reads above have
the one-epoch LoRA trials level with v1 and below v2 on hard-v1 (−16 pp), devtools (−7), docs (−4) and tasksource-heldout (−3).

**Arms** (`experiments/rounds/r30.json`, parent `31b` = Kev-27B v2, `runs/release/kev-27b-r23`, its round-23 reads
`runs/r23-27b-k-w85-*`, served as round 23 served it: on the round's pool over its own r3cal/v9 reads, `pooled: true`, a
new parent option in `kev.rounds` with a test):
- (a) `r30-g31-full`: round 22's full-weight recipe from the base (`experiments/round30/full-lr2e6.json`: `--full_ft 1`,
  bf16, lr 2e-6, head lr 1e-4, batch 8 × accum 2 on 8 ranks (4 × 4 after the fit check), `sft-v2-r26`), `H200:8`, 4 h timeout; candidates the final
  checkpoint and the 25/50/75% snapshots (`31b-full-s25/s50/s75`; their step paths are round 26's step counts as
  placeholders and are corrected from the trial's plan before any read). Two departures, both forced by the base:
  `--max_state 8192` (not 32,768) and no `--shared_prefix` / `--pass_tokens_max` (hybrid-only: Gemma trains on the packed
  mask). A fit check runs first on `H200:8` (`runs/sft-probe/r30-fit-g31-8xh200`, `--state_tokens 2048,4096,8192`, fallbacks
  batch 4 / 2); if 8192 does not fit at any batch, (a) is not launched and that is the result.
  Fit check result (2026-10-05): 2048 fits at batch 8 (98 GB/GPU), 4096 at batch 8 (123 GB), 8192 out of memory at
  batch 8 and fits at the registered fallback batch 4 × accum 4 (121 GB/GPU, same 128 records/step); (a) runs at
  batch 4 × accum 4. Report `runs/sft-probe/r30-fit-g31-8xh200`.
- (b) `r30-g31-lora2`: the one-epoch Gemma LoRA recipe (`experiments/gemma4-31b-it.json`) at two epochs, seeds 0 and 1
  (`experiments/round30/lora-2ep.json`, decision-v7), `H200`, 8 h timeout. Round 6's follow-up found two epochs no better
  at 27B on Qwen; this asks it again on a new base whose one-epoch trials are below v2.
  Seed 1 lost its worker at step 1440/3128 (Modal "worker disappeared", 2026-10-05 23:02 UTC; the retry refused the
  existing trial directory, and a LoRA trial has no resume point). It is rerun unchanged, before any read, as study
  `r30-g31-lora2b` (`experiments/round30/lora-2ep-s1.json`, bound $51); arm `31b-lora2-s1` points at it.
- Corrections before any read of the snapshot arms (2026-10-06): (1) the snapshot paths were copied from round 26 (steps
  115/229/344 of 458); Gemma's re-filtered sft-v2-r26 gave 454 steps and the trial wrote steps 114/227/341, so arms
  `31b-full-s25/s50/s75` point there (same 25/50/75% fractions). (2) Seed 0's agents-ood-v1, guardrails-ood-v1 and longdoc-v1
  reads ran out of memory before writing rows (Gemma 4 31B's 512-dim global heads fall back to SDPA's math kernel; fixed by
  `kev_sdpa`, queries chunked at inference) and were relaunched with the fix under the same names. The failed outputs on the
  volume were removed by mistake; their failure.json records are in `runs/r30-oom-failures/`. No rows had been read.

**Rule.** Round 24's audited rule and confirmation verbatim (breadth-v1 and the Kev panel primaries; tasksource-heldout
guard with `runs/r24-private/tsheld-exclude.json`; short-state and CUAD long-document guards; calibration criteria; no
pooled-externals guard; SemIf report only). The wanli-v2 / typesafe-v1 report-only reads are dropped (removed suites).
Confirmation stages are not run in this round without Jared.

**Budget.** $1,200 including reads (Jared, 2026-10-05); metered baseline $365.65 at 19:40 UTC. Admission bounds: fit check
$41.17, (a) $494.00 (H200:8 × 4 h × 3 attempts), (b) $100.24 (H200 × 8 h × 2 trials). Reads are spend-gated before each
batch: launch only while (metered − baseline) + running bounds < $1,080 (10% reserve).

**Readout (2026-10-06, `uv run python -m kev.rounds readout experiments/rounds/r30.json` -> `runs/r30-readout.txt`).** No
candidate. Every arm is below Kev-27B v2 on the Kev panel primary, so no guard can promote one. Paired against v2,
record-clustered, pp (95% intervals; corrected from the earlier 90% label, which applied only to temperature fits):

| arm | Kev panel | breadth-v1 | breadth index | tasksource-heldout | hard-v1 | Kev without hard | short-state acc | SemIf (report) |
|---|---|---|---|---|---|---|---|---|
| `31b-full` (final) | -1.6 [-2.6, -0.5] | +0.3 [-0.8, +1.4] | +0.7 [-0.5, +1.9] | -1.2 [-2.7, +0.3] | -4.7 [-6.7, -2.8] | -0.0 [-1.2, +1.2] | +0.5 [-0.7, +1.8] | -2.1 [-6.2, +1.4] |
| `31b-full-s75` | -1.9 [-3.0, -0.9] | +0.3 [-0.8, +1.4] | +0.7 [-0.5, +1.9] | -1.6 [-3.2, -0.1] | -4.7 [-6.6, -2.8] | -0.6 [-1.9, +0.7] | +0.3 [-0.9, +1.6] | -2.1 [-6.2, +1.4] |
| `31b-full-s50` | -2.9 [-4.0, -1.8] | -0.2 [-1.3, +0.9] | +0.0 [-1.2, +1.2] | -2.0 [-3.6, -0.4] | -6.3 [-8.5, -4.2] | -1.3 [-2.6, -0.1] | -0.1 [-1.4, +1.2] | -2.1 [-6.2, +1.4] |
| `31b-full-s25` | -6.6 [-7.8, -5.4] | -3.2 [-4.8, -1.7] | -3.0 [-4.5, -1.6] | -7.8 [-10.0, -5.8] | -12.8 [-15.2, -10.5] | -3.6 [-5.1, -2.2] | -2.5 [-4.1, -1.0] | -4.2 [-9.0, +0.7] |
| `31b-lora2-s0` | -8.1 [-9.3, -6.8] | -1.6 [-2.8, -0.4] | -0.9 [-2.2, +0.4] | -4.4 [-6.2, -2.5] | -16.7 [-19.3, -14.1] | -4.0 [-5.3, -2.7] | +0.8 [-0.4, +2.1] | -7.6 [-13.2, -2.8] |
| `31b-lora2-s1` | -8.1 [-9.4, -6.8] | -0.8 [-2.0, +0.4] | -0.5 [-1.8, +0.8] | -3.3 [-5.2, -1.5] | -16.3 [-19.0, -13.7] | -4.2 [-5.6, -2.9] | -0.5 [-1.8, +0.9] | -6.2 [-11.1, -2.1] |

Calibration: Kev panel ECE within +-0.01 of v2 for every arm; breadth ECE +0.009 to +0.024 (the full-weight arms' breadth index
ECE interval excludes 0); tasksource-heldout all-families ECE +0.055 for both LoRA seeds. guardrails-ood-v1 objective (report
only): full -0.021, s75 -0.023, s50 -0.026, s25 -0.073, LoRA s0 -0.196 / s1 -0.217, v2 -0.019.

What it says. v2's full-weight recipe closes most of the gap on Gemma: the final checkpoint ties v2 everywhere except hard-v1
(-4.7) and so the Kev panel (-1.6); the gain is monotone in steps (s25 < s50 < s75 < final), so a longer run might close it
further, but this round does not show that. Two-epoch LoRA is no better than the one-epoch trials on hard (-16) and
stays far below full weight; round 6's two-epoch negative holds on this base.

Not read. The CUAD/long-document guard (longdoc-v1) and agents-ood-v1 were stopped (Jared, 2026-10-06): the primary
already fixed the decision. Under `kev_sdpa`, long Gemma states are slow (agents-ood: about 60 of 373 records in 2.5 h on
one H200; longdoc: about 50 records per 15 min), so neither would have finished inside the 4 h read timeout. Their
partial `predictions.jsonl` stay on the volume, never pulled. Making long Gemma reads fast (a fused kernel for 512-dim
heads, or chunking keys too) is a prerequisite for any Gemma release.

Spend. Metered $863.67 at the readout, $498 over the $365.65 baseline, inside the $1,200 cap.

## Gemma 4 31B, report only (2026-10-05)

Development reads of the two `runs/gemma4-31b-it-2` trials (google/gemma-4-31B-it@842da37, v7 recipe, LoRA r16, one epoch, lr 5e-5; seeds 0 and 1), one `::benchmarks` call per trial, results in `runs/g31it-s{0,1}-<suite>` (transfer-v4 dev is the trial's own transfer read). Each Gemma trial is served at the temperature fitted on its own development rows (s0 1.782, s1 1.447); Kev-27B v1 (`runs/r6-27b-v2/01-trial-1`) at its fitted 1.382; Kev-27B v2 (`runs/release/kev-27b-r23*`, reads `runs/r23-27b-k-w85-*`) at round 23's registered pool temperature 1.32. Panel exclusions as round 24's rule (breadth: routerbench, cfcolor, humicroedit, chessbench; devtools: flakeflagger, commitpackft_type; transfer: emotion; the two codereviewer drops). Deltas are Gemma minus reference, accuracy in pp, `kev.metrics.paired_bootstrap` (micro, record-clustered, 2000 resamples, seed 0), on the records both sides scored. No test partition was read. Not a round: nothing here selects or ships a model.

| suite | n | Gemma s0 acc / Brier / ECE | Gemma s1 acc / Brier / ECE | v1 acc / Brier / ECE | v2 acc / Brier / ECE | s0 − v1 | s1 − v1 | s0 − v2 | s1 − v2 |
|---|---|---|---|---|---|---|---|---|---|
| transfer-v4 dev | 576 | 0.894 / 0.169 / 0.034 | 0.891 / 0.164 / 0.022 | 0.884 / 0.184 / 0.038 | 0.887 / 0.171 / 0.035 | +1.0 [-1.6, +4.0] | +0.7 [-1.6, +3.0] | +0.7 [-1.7, +3.3] | +0.3 [-1.9, +2.6] |
| transfer-v9 | 1046 | 0.824 / 0.256 / 0.041 | 0.826 / 0.252 / 0.031 | 0.822 / 0.265 / 0.050 | 0.820 / 0.252 / 0.036 | +0.2 [-1.8, +2.2] | +0.4 [-1.5, +2.2] | +0.4 [-1.8, +2.5] | +0.6 [-1.3, +2.5] |
| hard-v1 | 1083 | 0.752 / 0.328 / 0.017 | 0.762 / 0.323 / 0.017 | 0.733 / 0.338 / 0.047 | 0.912 / 0.120 / 0.033 | +1.8 [-0.8, +4.4] | +2.9 [+0.3, +5.3] | -16.1 [-18.7, -13.4] | -15.1 [-17.8, -12.4] |
| devtools-v1 | 772 | 0.750 / 0.343 / 0.077 | 0.741 / 0.332 / 0.077 | 0.781 / 0.293 / 0.031 | 0.816 / 0.260 / 0.039 | -3.1 [-4.9, -1.4] | -4.0 [-5.9, -2.3] | -6.6 [-8.9, -4.2] | -7.5 [-9.9, -5.1] |
| documents-v1 | 920 | 0.883 / 0.183 / 0.083 | 0.874 / 0.185 / 0.070 | 0.862 / 0.201 / 0.088 | 0.916 / 0.120 / 0.019 | +2.1 [+0.0, +4.0] | +1.2 [-1.0, +3.2] | -3.4 [-5.3, -1.5] | -4.2 [-6.3, -2.2] |
| breadth-v1 | 2475 | 0.825 / 0.242 / 0.017 | 0.829 / 0.239 / 0.016 | 0.820 / 0.245 / 0.008 | 0.836 / 0.223 / 0.010 | +0.5 [-0.5, +1.7] | +0.8 [-0.2, +2.0] | -1.0 [-2.1, +0.2] | -0.7 [-1.8, +0.5] |
| tasksource-heldout-v1 (round 24's seven families excluded) | 1993 | 0.758 / 0.347 / 0.038 | 0.755 / 0.347 / 0.035 | 0.748 / 0.353 / 0.042 | 0.790 / 0.302 / 0.043 | +1.0 [-0.8, +2.9] | +0.7 [-1.1, +2.5] | -3.2 [-5.0, -1.4] | -3.5 [-5.3, -1.8] |
| SemIf (report only) | 144 | 0.931 / 0.115 / 0.042 | 0.931 / 0.089 / 0.051 | 0.972 / 0.061 / 0.068 | 0.965 / 0.063 / 0.061 | -4.2 [-9.0, +0.0] | -4.2 [-8.3, -0.7] | -3.5 [-8.3, +0.7] | -3.5 [-8.3, +0.7] |

Chance-corrected breadth index (`scripts/breadth_report.py`, all 14 datasets, raw rows, paired record bootstrap, 2000 resamples; `runs/g31-breadth-index*`): v1 50.2 [47.4, 53.3], v2 52.0 [49.4, 54.9], Gemma s0 52.4 [49.5, 55.4], s1 52.5 [49.6, 55.4]. Gemma − v1: +2.1 [−0.6, +4.7] / +2.3 [−0.6, +5.1]; Gemma − v2: +0.3 [−2.5, +3.2] / +0.5 [−2.6, +3.6]. Its gain over v1 comes from the arts and routerbench datasets that round 24's breadth guard excludes; on the audited panel it is +0.5 / +0.8 pp over v1 and −1.0 / −0.7 pp under v2 (table).

Reading (two seeds agree on every sign). Against v1 a one-epoch LoRA on Gemma 4 31B IT is level or better on transfer-v4, v9, hard-v1, docs and breadth (hard s1 +2.9 [+0.3, +5.3]), and worse on devtools-v1 (−3.1 / −4.0, both intervals exclude zero). Against v2 it is level on transfer-v4, v9 and breadth and clearly worse where v2's full-weight SFT data taught the task: hard-v1 −16.1 / −15.1, devtools −6.6 / −7.5, docs −3.4 / −4.2, tasksource-heldout −3.2 / −3.5. Gemma's calibration is worse off its fitting distribution (devtools and docs ECE 0.07–0.08 against v2's 0.04 / 0.02). SemIf is report only: −4.2 / −4.2 vs v1, −3.5 / −3.5 vs v2 on 144 questions. tasksource-heldout is read with round 24's private exclusion list (`runs/r24-private/tsheld-exclude.json`, sha256 `a72030ab…`, restored from kev-private-train): level with v1 (+1.0 / +0.7) and below v2 (−3.2 / −3.5); v1's and v2's reads are `runs/r21-P27-tsheld` and `runs/r23-27b-k-w85-tsheld`.

So the base is competitive with v1 at LoRA strength, and the gap to v2 is the SFT data, not the base: the question for round 30 is whether v2's full-weight recipe closes it on Gemma.

## Round 31 - Gemma blend and another full-weight pass (registered 2026-10-09)

Jared approved the two-arm follow-up after round 30. Full-weight Gemma improved from 25% through the final checkpoint
but remained below Kev-27B v2 on the Kev panel and hard-v1. This round asks whether interpolation or another pass
closes that gap; it does not test new data, new architectures, or a Qwen continuation. The knowledge graph's
`topics/full-weight-sft.md` records the round-23 blend as the positive precedent and rounds 25/26 as negative continued
SFT precedents. Nothing is trained, constructed or read until this registration and its spec are committed.

**Two candidates, fixed before results** (`experiments/rounds/r31.json`):

- `31b-blend-w85`: the round-30 final full-weight backbone at `/runs/r30-g31-full/00-trial-0/checkpoint`, blended with
  the one-epoch Gemma IT seed-0 LoRA at `/runs/gemma4-31b-it-2/00-trial-0/checkpoint`. Alpha 0.85 on full weights, 0.15
  on the LoRA endpoint merged in fp32, rounded once to bf16; keep the full-weight pointer head. This is round 23's
  winning blend setting, not a sweep selected on Gemma reads. Existing `scripts/interpolate_checkpoint.py` checks
  base/revision, every tensor name/shape, and head compatibility; no new merging implementation. CPU construction:
  `KEV_HF_SECRET=huggingface-secret uv run modal run --detach modal_app.py::interpolate --sft /runs/r30-g31-full/00-trial-0/checkpoint --toward /runs/gemma4-31b-it-2/00-trial-0/checkpoint --base google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --alphas 0.85 --prefix 31b-k --study r31-g31-blend --timeout 10800`.
- `31b-cont`: warm-start the backbone and head from round 30's full-weight final and train one further pass over the
  same `sft-v2-r26` training partition, at peak backbone lr **1e-6**. Fresh AdamW and OneCycleLR, not an exact resume or
  the second half of a two-pass schedule; the completed trial deleted its optimizer resume point. Keep head lr 1e-4,
  state 8192, bf16, batch 4 x accumulation 4, checkpointing and length-sort; shuffle/augmentation seed 1 so the repeat
  is not the identical seed-0 order. Final only, no snapshot candidates. Plan `experiments/round31/full-cont-lr1e6.json`;
  new immutable study `r31-g31-cont`, H200:8, 10800 s per attempt, up to the existing three-attempt ledger. Round 30's
  measured 11,653 s wall time suggests a continuation attempt will be needed. The watcher must remain alive.

**Read and decision rule.** Parent remains the latest released Kev-27B v2, pinned at `28be62e9c5ae0471bda2b7c636a55224b4f4b887`,
served at the same registered round-24 temperature pool over its existing round-23 reads. Both new candidates get their
own pool and one batched `::benchmarks` call each for breadth-v1, excluded tasksource-heldout-v1, hard-v1, devtools-v1,
documents-v1, transfer-v4 development, transfer-v9 development, transfer-r3 calibration and the **already-spent**
transfer-r3 test short-state guard, plus report-only SemIf. `--allow-test` is used only for that registered short-state
guard, never advertised as fresh confirmation. No other test/locked partitions are read. Frozen eval bytes are unchanged.

Round 24's audited criteria and ranking are unchanged, but reads are staged: all non-long criteria are the short
screen; the selected survivor must also pass the two original CUAD criteria in `confirm.long`. The word "candidate"
in the screen's automatic output means **screen survivor only**, not promotion or release. No survivors means stop
without longdoc. If there is a survivor and its bounded read fits the remaining cap, launch its single longdoc-v1
**development** read once with 28800 s (explicit `::benchmarks` command; `launch-reads` otherwise uses the screen's
2700 s timeout). Existing parent longdoc rows are reused. If that read times out or cannot be admitted, its long guard
is **unread/incomplete**, never passed. Agents/guardrails OOD are omitted, not silently claimed as read. A winner would
still need separately registered confirmation and serving/long-input verification; this round publishes nothing.

Accuracy, Brier and ECE are recorded on both sides, paired by record with `kev.metrics.paired_bootstrap` through
`kev.rounds` (2,000 resamples, seed 0, 95% paired percentile intervals; the temperature-fit interval is separately 90%);
the chance-corrected all-family breadth index and hard/devtools/docs are also
reported. Same seven-family exclusion hash as round 30. No repeat of r30 candidate reads or new parent reads.

**Budget, hard cap $500 including reads and GPU checks.** `kev.budget` admission bounds: training $370.50 (three 3 h
attempts at $41.1664/h), CPU interpolation $4.21, twenty screen jobs at 2700 s on H200 $93.98, targeted GPU verification
at most 3600 s $6.27: **$474.95**, leaving $25.05 for incremental storage and reserve. Actual training is expected
to be much cheaper than all three full attempts. The selected survivor's eight-hour longdoc bound is $50.12; admit
only if completed actual costs + outstanding job bounds + $50.12 + $10 reserve fit $500. Otherwise stop incomplete.
Before launches use both the attempt/resource ledger (billing lag cannot authorize overspend) and the current Modal
metered summary. Baseline returned on 2026-10-09: **$816.49091164**, cycle 2026-10; this provider figure is lower than
the prior $863.67 report, so do not subtract the old report or claim the difference is spend refunded. Keep the new
baseline and actual job durations to attribute this round, and stop new admissions at $490.

No Hub writes, mirrors, published cards, README numbers, main commits or merge. Work stays on `base/gemma4` / PR #221.

**Execution and verification (2026-10-09).** The blend finished in 828 s: 832 tensors, alpha 0.85, fp32 arithmetic,
full-weight head retained. Its interpolation artifact pins both endpoints (full weights `98bd0ce6…`, merged LoRA
`5f94f57b…`), the base revision, head hashes, formula and output weights `306779c1…`. The continuation is spawned
with its attempt ledger and watcher intact, initially queued for H200:8.

The inference review found a latent unequal-query/key-length bug: the chunker used bottom-right implicit causality,
whereas Transformers' SDPA uses upper-left causality and trims excess keys. It now follows that reference and slices
position bias alongside queries. Twelve direct regression cases cover unequal lengths, explicit masks and bias.
Targeted H200 verification passes **15 tests**, including the two Gemma 12B implementations, Qwen rows, and those
regressions. Chunked versus original SDPA on identical Gemma inputs is held to **1e-5** with the math backend.
Packed/rows/prefix use a separate **1e-3** bound: deterministic eager and math-SDPA runs both showed the same
`0.000241607` probability drift (0.024 pp), so the former 1e-4 cross-shape assertion was not a chunking diagnostic.
Default fused SDPA still has larger shape-dependent drift (the earlier diagnostic reached 0.047); neither the test
backend nor the new tolerance changes the research read path, and this is not a serving-parity claim.
The complete requested unit suite has **571 passed, 21 archived-artifact skips, 2 warnings in 733.07 s** after the
pooled-parent fixes and integration of main's device-probing change (`fc4a17e`). Targeted conventions/pool checks
passed 27 tests before the privacy guard and 30 afterward, including the canonical delimiter rule. CPU weight-backed
checks pass eight tests;
the generated eight-record Qwen smoke fixture fails the strict merge-parity test identically on main and this branch
(max probability difference `2.6643e-5`, bound `1e-5`). This is recorded, not a green test or a relaxed bound.
H200 old/main-versus-branch verification gives byte-identical benchmark rows and identical scoring/report fields
except wall-clock `latency_ms`; converter output also matches. Fixed-seed GPU training is **not bit-identical**:
adapter/head maxima are `2.170235e-4` / `5.584210e-5`. A repeat of unchanged main differs by `2.164692e-4` /
`2.244860e-5`, establishing baseline nondeterminism, not exact training parity. The synthetic smoke merge fixture
also fails identically on both revisions on H200 (`2.8193e-5` against the unchanged `1e-5` bound). Its separate
server-OOM tests pass on both revisions in fresh processes; the first whole-suite run failed its allocation-pressure
precondition instead. The Modal image's installed fused Qwen3.5 kernels cannot score that hybrid checkpoint on CPU.

Structural review found that pooled-parent reads bypassed the canonical template resolver. `parent_side` now uses
`locations` before `temperature_pool`; a static/templated path regression covers this. Pooled parents require a registered
pool, and absent parent fit rows leave comparisons incomplete instead of raising. These cases have regression coverage.
The full unit rerun initially stalled in
two-rank snapshot-process shutdown after writing its checkpoint. That same test passes in isolation (98.33 s) with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; the complete requested suite passed with those thread limits and no test exclusions
(733.07 s). Historical readout replay used saved rows only: **134 passed, 1 skipped in 769.70 s**;
247 expected archive files remain unavailable after restoration, so incomplete archive coverage is not a green replay
or a reason to rerun historical panels.

**Thermonuclear structural review (2026-10-09).** Against main, the implementation review found and corrected the
templated pooled-parent path bypass (`kev/rounds.py:352`), the missing registered-pool invariant
(`kev/rounds.py:417`), and the missing-fit exception (`kev/rounds.py:860-862`). These reuse `locations`,
`temperature_pool`, and `Side` rather than introducing a second readout engine. In the model, the tokenizer-specific
delimiter contract and escaping stay in `kev.model.delimiters`/`user_tokens` (`kev/model.py:90,113`);
sliding/global masks and the inference-only SDPA adapter stay inside `DecisionModel`/the Transformers attention
interface (`kev/model.py:287,410-425`), not in serving or training call sites. Foreign-base admission reuses
`training_context`/`fits` (`kev/train.py:86-120`), leaving Qwen's admitted-base path intact. The implementation
does not push a source file across 1,000 lines; `kev/rounds.py` and the affected tests were already over that
threshold. No dramatically simpler implementation, new duplicated canonical rule, or unjustified special-case
layer was found. **Structural approval bar met for the current implementation**, not a claim of bit-exact training,
arbitrary release-parent provenance, serving parity under fused SDPA, or a complete round-31 verdict.

The review also found a **private-artifact boundary leak**: ten committed `tasksource-heldout-v1` reports contained
per-source breakdowns. These reports are removed from the branch's tracked tree and their directories ignored,
while aggregate scores, calibration and paired readouts remain committed; the full report copies remain local,
outside the Git index. `tests/test_conventions.py` now rejects any tracked report for this private suite with a
nonempty `tasks` breakdown, including reports identified by the suite-manifest hash. A cleanup commit does not
remove those files from existing branch history: a separate approved history rewrite would be needed to erase
that already-pushed exposure. Do not claim that historical privacy is repaired.

**App-scoped Modal spend, provisional.** Billing for the eleven identified round-31 apps (blend/read app, GPU and
CPU verification, and main training-repeat control) totals **$5.92757282 as observed 2026-10-09 22:05 UTC**. This
is a scoped billing-report snapshot, not the final round cost: the queued continuation has no GPU allocation yet,
billing may lag and new benchmark apps must be added if it later runs. The registered $500 hard cap and conditional
long-read admission still apply; workspace-wide metering alone must not be reported as this round's spend.

Pooled-parent provenance limitation: the engine checks the registered pool against every candidate's training
(`pool_training_problems`), but does not establish training-source separation for a release/blend parent lacking ordinary
trial provenance. Kev-27B v2 reuses round 23's registered parent reads and temperature pool; this is not a claim that
arbitrary pooled parents have a generic training-provenance audit.

**Blend short screen (2026-10-09), not the final round verdict.** All ten registered reads finished once. At pooled
T **1.0968** (90% temperature interval [0.9772, 1.2030], 648 questions), against the parent's own pool T **1.3195**:

| panel | accuracy delta, pp (paired 95% interval) | Brier delta | ECE candidate / parent |
|---|---|---|---|
| breadth, audited | +0.2 [−0.9, +1.3] | +0.007 | 0.0309 / 0.0103 |
| tasksource-heldout, excluded families | −1.0 [−2.6, +0.5] | +0.008 | 0.0362 / 0.0432 |
| Kev panel | −1.8 [−2.8, −0.7] | +0.019 | 0.0094 / 0.0146 |
| hard | −4.2 [−6.2, −2.3] | +0.056 | 0.0216 / 0.0326 |
| devtools, audited | −1.3 [−3.9, +1.4] | +0.003 | 0.0129 / 0.0386 |
| documents | −0.5 [−2.3, +1.2] | +0.005 | 0.0115 / 0.0191 |
| short state | +0.5 [−0.8, +1.8] | −0.002 | — |

The blend fails both positive-gain primaries, the Kev lower-bound guard, breadth ECE, and the short-state
confident-error upper bound (+0.0095 [0.0032, 0.0164], bar +0.01). Unknowable share is zero. No longdoc read is
admitted for this arm. `runs/r31-readout/round31.json` is explicitly incomplete for `31b-cont`, whose H200:8 call
remains queued with its continuation watcher alive; no continuation result or overall verdict is inferred.

**Continuation launch failure and unchanged recovery (2026-10-09, registered before retry).** At 22:44 UTC the
original call `fc-01M4H36GVNHJQQ9BGXFGTJ1MVZ` failed its source-hash guard before lease acquisition, model loading
or training. I redeployed `kev-r31` during its three-hour queue; the late container no longer had the sources
the launcher had hashed. No `/r31-g31-cont` directory exists on the runs volume, and there is no checkpoint, result,
provenance or continuation panel read. The failed call and its spawn/watch ledgers are preserved. This is a launch
failure, not negative model evidence; the watcher exited after recording the continuation as incomplete.

Recover the **same** `31b-cont` candidate under the new immutable study `r31-g31-cont2`. The plan
`experiments/round31/full-cont-lr1e6.json` is byte-for-byte unchanged: same round-30 initialization, training partition,
Gemma revision, H200:8, seed 1, fresh optimizer, LR, batch/accumulation, state limit, augmentations and final-only output.
The round still has exactly two candidates. Use the separate deployment `kev-r31-cont-frozen`; do not redeploy it
while the recovery is queued or running. Do not disable the source guard. The spec records this operational amendment
before launch; the failed original study is not an outstanding job. Add a **$10 failed-start reserve** to the original
$474.948 screen bound, for a conservative **$484.948** total inside the unchanged $500 hard cap. Billing for the
original app currently shows only $0.00169026 of CPU/memory costs and may lag, so it does not authorize overspending.
Continue to count actual failed-start costs and all outstanding bounds at every later admission. No blend or parent
read is repeated, and the continuation remains incomplete until the unchanged recovery produces evidence.

## Record

One line per round or named study. `rN.json` is `experiments/rounds/rN.json` on main (the rule as data; `python -m
kev.rounds readout` reproduces rounds 5-20, 22 and 24, see `tests/test_rounds.py`); `A:` is the archive tag. Verdicts are the registered
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
| Round 21 | 09-26 | full-weight SFT of Qwen3.8-27B on `sft-v2-r21` (32k states, extended data; lr 2e-6, 1e-6) | failed at startup: both arms out of GPU memory at step 62 (a 98.7k-token pass the characters plan costed like its slot's 33-50k-token passes), no snapshot, no read; ~$135 | `r21.json`; "Round 21 result" |
| Round 22 | 09-26/27 | round 21's science and rule on `sft-v2-r22` (145,840 records) with `--pass_tokens_max 40960`, one arm (lr 2e-6), 4 candidates (snapshots s25 / s50 / s75 + final) | no candidate (0 of 4): primaries pass and grow with training (final breadth +1.3, tasksource-heldout +3.8, Kev panel +7.8); every candidate fails scienthoon and CUAD ECE at 16k+, three the pooled externals, s75 and the final short-state accuracy; calm-called-angry errors gone, angry missed instead; Modal gave 2 of 3 attempts, final finished by a manual continuation | `r22.json`; `runs/r22-readout`, `runs/r22-breadth-report`, `runs/r22-scienthoon`; "Round 22 result" |
| scienthoon removed | 09-27 | `evals/external/scienthoon-v1` removed as unsound for a gate (saturated `queue`, text-unknowable `priority`, 15 of 291 `angry` labels contradicting the text) | past verdicts stand; pooled externals = SemIf + WANLI-v2 + TypeSafe from round 23 | "scienthoon removed"; `kev.suite.REMOVED_SUITES` |
| Round 24 | 09-27 | retrospective selection: the 2026-09-27 audit's suite verdicts as one rule on all 12 full-weight 27B checkpoints (rounds 19, 20, 22), no training or new read | candidate `27b-r22-final` (2 of 12 pass: it and `27b-r20a-w85`); **not confirmed**: tests stage passed (breadth-v1 test +1.5, tasksource-heldout-v1 test +5.3, pooled +8.6; test breadth index 53.7 vs Jev 54.0, Kev-27B 50.2), locked transfer-v4 0.8841 < 0.886 (580 of 656, 582 needed); CUAD test −1.8; not released | `r24.json`; `runs/r24-readout`, `runs/r24-verdict`, `runs/r24-breadth-report`; "Round 24 result", "Round 24 confirmation" |
| Round 23 | 09-27/28 | post-hoc, no training: round 22's final blended toward Kev-27B's own weights (LoRA merged in fp32), α 0.85 / 0.70 / 0.50 × {SFT head, blended head}; re-registered 09-28 on round 24's audited rule and confirmation (the first registration, on round 22's rule, never launched) | **`27b-k-w85` confirmed** (5 of 6 pass the rule; tests stage passed; locked transfer-v4 0.8887, 583 of 656, bar 582); released as Kev-27B v2 on 09-30 | `r23.json`; `runs/r23-readout`, `runs/r23-verdict`; "Round 23 result", "Round 23 confirmation", "Released: Kev-27B v2" |
| Release: Kev-27B v2 | 09-30 | round 23's `27b-k-w85` published to `jaredpalmer/kev-27b` main (full bf16 weights `d27af6ab…`, head `7968f17b…`, T 1.3195) in one commit that deleted v1's adapter files; v1 tagged `v1-lora` first | **released**; anonymous Hub load reproduced round 23's semif-v1 and transfer-v4 development reads row for row; `@v1-lora` loads v1 at T 1.38 | `runs/release/kev-27b-r23-published.json`; `runs/rel27-public/`; "Released: Kev-27B v2" |
| Round 25 | 09-28/29 | continued full-weight SFT from Kev-27B (LoRA merged) on `sft-v2-r25` (breadth + b1v2 replay, no long-document families, states ≤ 16k), lr 1e-6 / 2e-6, 8 candidates (snapshots + finals), round 24's rule | no candidate (0 of 8): every arm fails breadth ECE (0.023-0.035 vs 0.0176); lr 1e-6 s75 / final fail only that (11/12); lr 2e-6 also short-state accuracy and the breadth primary; CUAD accuracy held, CUAD ECE did not; none ahead of `27b-k-w85` (report only) | `r25.json`; `runs/r25-readout`; "Round 25 result" |
| Round 26 | 09-29 | round 25's lr 1e-6 arm with tasksource-v1 doubled (`sft-v2-r26`: sft-v2-r25 + 13,000 tasksource-v1 records, 58,515), one study, 4 candidates, round 24's rule | no candidate (0 of 4): every arm fails breadth ECE (0.020-0.029 vs 0.0176); the final fails only that (11/12), the snapshots also the breadth primary, s50 also short-state accuracy; against round 25's lr 1e-6 arm breadth and breadth ECE level (report only); none ahead of `27b-k-w85` (report only) | `r26.json`; `runs/r26-readout`; "Round 26 result" |
| WANLI and TypeSafe removed | 09-30 | `evals/external/{wanli-v2, wanli-v1, typesafe-v1}` removed as unsound for a gate (a quarter of WANLI's gold labels are one of two disagreeing annotators'; TypeSafe's gold is two closed frontier models' averaged answer, split-half r about 0) | past verdicts stand; SemIf is the only external read from round 27, report only | "WANLI and TypeSafe removed"; `kev.suite.REMOVED_SUITES` |
| Round 27 | 09-30 | post-hoc, no training: round 18's two 9B documents + skills deltas on the audited rule at a held-out-pool temperature | **`9b-r18a` confirmed** (both arms pass; tests: pooled +18.7, documents-v1 +7.1; locked transfer-v4 +0.0 [−1.7, +1.8], Brier −0.025); staged as Kev-9B v2, not published | `r27.json`; "Round 27 result"; `runs/r27-readout`, `runs/r27-verdict` |
| Release: Kev-9B v2 | 09-30 | round 27's `9b-r18a` published to `jaredpalmer/kev-9b` main (`b5d8c18e`, adapter `2b2a70cf…`, T 2.19); v1 tagged `v1` first | **released**; anonymous Hub load reproduces round 18's reads (argmax 252/252, 764/764); `@v1` loads v1 at T 2.30 | "Released: Kev-9B v2"; `runs/rel9-public/` |
| Round 28 | 09-30 | post-hoc, no training: Kev-4B / Kev-0.8B temperatures refitted on the held-out pool; validated context length per size (report only) | no candidate at either size: `4b-r10` fails both primaries (Brier −0.0001 [−0.0005, +0.0003], ECE 0.0252 vs 0.0240); `08b-r15` passes both and fails the hard / devtools / documents ECE guards; shipped T 2.41 / 2.35 stay; validated context (report only) 8,192 for 0.8B / 4B / 9B v2, 65,536 for 27B v2 (re-read 10-01 with every phase-B panel; verdict unchanged) | `r28.json`; `runs/r28-readout` (`context.{json,md}`); "Round 28 result" |
| Round 29 | 10-01 | post-hoc, no training: the eleven 9B deltas of rounds 7-18 against Kev-9B v1 under round 24's audited rule, pool T | **no candidate**: every arm fails the tasksource-heldout-v1 primary (lower bound > 0), and all but `9b-r7-s1` / `9b-r9-a` also breadth-v1; Kev-9B v2 stands. Ten arms' longdoc reads stopped for the budget (verdict-neutral); agents / guardrails OOD not read | `r29.json`; `runs/r29-readout`; "Round 29 result" |
| Kev 1.0 (release candidate) | 09-30 | the four released checkpoints versioned as one family: formal cards for all four, README family table, release notes and asset builder; no training, no new read | packaged (PR #206); validated context 8,192 (0.8B, 4B, 9B) / 65,536 (27B); the baseline Kev 2 is measured against | "Kev 1.0 baseline"; `docs/releases/kev-1.0.md`, `docs/releases/kev-1.0-assets.json` |
| Release: Kev 1.0 | 10-01 | the four checkpoints published as one versioned family: card-only Hub commits + tag `v1.0` on each (0.8B `bf75a6a8`, 4B `6cfce5c2`, 9B `db029f08`, 27B `af0e6d55`; weights unchanged), GitHub release `kev-1.0` with three deterministic adapter tarballs + `SHA256SUMS.txt`, `kev-family` retired to a pointer | **released**; anonymous `@v1.0` loads at T 2.35 / 2.41 / 2.19 / 1.32 with matching hashes, smoke reads and tarball logits identical to the tag's, assets re-downloaded and checked, Space `/decide` answers | "Released: Kev 1.0"; `runs/release/kev-1.0.json` |
| breadth-v1 | 09-24 | frozen eval-only panel over the Decision Index's five areas (14 held-out datasets, 150 records each, locked test unread); development baselines | report: chance-corrected index Jev 53.3, AutoJev-27B 51.7, Kev-27B 50.2, Kev-4B 40.8 (Kev-27B vs Jev −3.1 [−6.2, +0.1]); Kev-27B trails most on retrieval (SGD, CLINC150) | `evals/breadth-v1/manifest.json`; `runs/breadth-v1-report/report.md` |
| drift-v1 | 09-30 | why Kev-27B v1's semif-v1 logits differ between its 09-23 read and the 09-30 public read (max \|Δp\| 0.032, argmax equal) | report: #125's causal-conv1d CUDA kernel in the Modal image (replacing transformers' PyTorch conv in the 48 DeltaNet layers); no code drift (09-23 code and `main` bit-identical on the same kernels), no nondeterminism; breadth-v1 offset acc −0.1 pp, ECE +0.0008, 13/3,075 flips, below every round-25/26 margin; fp32 Qwen3.5 reads on CUDA carry fla's TF32 dots (≈ 0.003 in p) | `runs/drift-v1/REPORT.md`; AGENTS.md "What fp32-exact guarantees" |

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
