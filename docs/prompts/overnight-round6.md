# Overnight round 6: autoresearch on Kev

You are running an unattended overnight research session on the Kev repository at `/Users/jared/dev/kev`. Jared is asleep. You have from now until about 08:00 local time and **$1,000 of Modal spend**. Your job is to find changes that improve the released Kev checkpoints under this repo's pre-registered rules, confirm them the way the repo confirms things, and leave a report Jared can act on in the morning. You hill-climb and you confirm; you do not publish.

Work autonomously. Do not wait for a human. If an arm is blocked (authentication, a spend limit, a deploy you cannot fix in 30 minutes), write down what happened and move to the next arm.

## 1. Read first (30 minutes, no more)

- `AGENTS.md`, all of it. It is the operating manual: commands, conventions, what is frozen, what is canonical.
- `PLAN.md`: "Where we stand", "Round 4" and "Round 4 results", "Round 5" and "Round 5 result", "Release confirmation: soft-target Kev-9B", "Open questions". Everything you do tonight continues from round 5's "Next (to be registered, not read)" paragraph.
- `PLAN_27b.md`: the gating addendum and Phase A (A1, A2) and B1.
- `.agents/skills/kev-modal-study/SKILL.md` (how every GPU job is launched, monitored, pulled; its Gotchas section will save you hours) and `.agents/skills/kev-verify/SKILL.md` (how to prove a code change has no regression).
- `kev/autoresearch.py` (docstring, `challenge()`, `next_incumbent()`), `kev/experiment.py` (`validated_trial`, `GATES`, `EVALUATOR_FILES`), `experiments/rounds/r5.json` (`python -m kev.rounds readout`), `scripts/round4_deltas.py`, `scripts/build_long_states.py`, `scripts/build_soft_targets.py`.
- `runs/leaderboard.md`, `runs/autoresearch.jsonl`, `runs/r5-verdict/*.json`, `experiments/r5-combined.json`.

## 2. Hard rules

These override anything else in this prompt. When in doubt, stop that arm and write it down.

1. **Spend.** Read `uv run modal billing summary --json` at the start and record `metered_cost` as the baseline in `runs/r6-state.json`. Before every launch, read it again. Do not launch if `(metered_now - baseline) + sum(bound_usd of everything still running) >= 1000`. Every study's admission bound is printed at launch and saved in `runs/<name>.spawn.json`; benchmark passes get the bound `compute_bound(gpu, timeout, jobs)` from `modal_app.py`. Per-phase caps are in section 4; a phase may not borrow from the reserve. Billing readings lag and get revised; that is why the reserve exists.
2. **Selection on development partitions only.** Never pass `--allow-test`. The locked test (`modal_app.py::locked_test`) is read at most once per candidate, only after the candidate has passed the fresh-panel rule in section 3, and only for the candidate that rule selected. A fresh confirmatory panel is read once, for at most one candidate per size plus its parent, and only after the rule that decides was committed to `PLAN.md`. Never re-read a panel for a second candidate.
3. **No publishing, no Hub changes, no main.** Do not run `kev.publish`, `hf upload`, `hf repos tag`, `scripts/publish_space.sh`, and do not rewrite any released `head.pt`. Do not commit to `main`, do not force-push, do not open a PR. Work on branch `research/overnight-r6` and push it after every commit so nothing is lost if this machine sleeps.
4. **Frozen files stay frozen.** Never edit anything under `evals/` that exists. New data is a new directory with a `manifest.json` carrying sha256 per file and the inputs' hashes (copy the shape of `evals/round5/combined-v1/manifest.json`).
5. **No Jev in training, ever.** Teachers are open models only: the existing teacher rows in `runs/probes/qwen35-9b-semif-teacher2-decision-v7-train` and Kev's own outputs (`runs/r4-kev-9b-v7-train`). No `kev.jev` calls tonight; the AI Gateway budget is not authorized.
6. **The evaluator does not change.** No edits to the files in `kev/experiment.py: EVALUATOR_FILES`, the gates, `kev/metrics.py`, or any suite, with one exception: the A1 branch already changed `kev/model.py`, and section 5 Phase 0 says how it must be verified before anything runs on it. Scripts under `scripts/` and `modal_app.py` infrastructure constants may change.
7. **Every trial is a config-only plan** through `modal_app.py::study` with full provenance, and every read is `modal_app.py::benchmarks` or `::locked_test`. Never train locally; a 32 GB Mac cannot hold these models.
8. **`kev.autoresearch round/loop` are stale for the current family**: `SUITE` is decision-v4, `BASE_DEFAULTS` and `TRIAL_MINUTES` only know the Qwen3 bases, and `propose()` mutates from-scratch config knobs, which round 2 showed is an exhausted space. Do not run them. Write plans by hand, use `kev.autoresearch leaderboard` and `compare`, and `kev.metrics.paired_bootstrap` for decisions. You may extend `kev/autoresearch.py` if it is quick, but experiments matter more than tooling tonight.
9. **Modal hygiene** (all from the skill; the expensive ones repeated): deploy with `KEV_GPU=H200 uv run modal deploy modal_app.py` after any change to `kev/*.py` **or any new file under `evals/`** (the image copies `evals/`; the launcher only checks `kev/*.py` hashes, so a missing data file fails inside the container). Study names are immutable; a failed study needs a new name, and a study is never relaunched under the same name. Check `modal container list` before relaunching after a local error. Space detached `modal run` launches at least 30 seconds apart. Set every timeout with at least 50 percent headroom over the measured wall time of the closest previous trial; a timed-out container saves nothing. In the first five minutes of every study, count optimizer steps per minute in `modal container logs` and project the wall time against the timeout; cancel and relaunch with fewer records or epochs if it will not fit. Everything runs on H200 tonight; do not change `KEV_GPU` while trials are running.
10. **Write the decision rule before the result.** Every number you write into `PLAN.md` carries its checkpoint, suite and partition, n, and report path. Negative results are written as fully as positive ones. Use simple technical English; no slogans.
11. **Resilience.** Keep `runs/r6-state.json` current: phase, launched studies with spawn ids, what has been pulled, what is pending, spend readings. If this session is interrupted and restarted, read that file first and resume; detached Modal jobs keep running.

## 3. What "better" means tonight (register this in PLAN.md before launching anything)

Add a section `## Round 6 - overnight autoresearch (registered <UTC timestamp>, before any training or read)` to `PLAN.md` immediately after the round 5 result. It contains the budget baseline, the arms of every phase below, and these rules verbatim with the thresholds filled in. Commit it. Then launch.

**Parents.** The released checkpoints: `jaredpalmer/kev-9b` = `runs/night2-9b-du/00-trial-0` (transfer-v4 dev 0.822, locked 0.852, T 2.30); `jaredpalmer/kev-4b` = `runs/night2-4b-du/00-trial-0` (0.797, locked 0.837, T 2.14); `jaredpalmer/kev-0.8b` = `runs/night2-08b-du2/00-trial-0` (0.643, locked 0.684, T 2.41). Every arm is served at one temperature fitted on its own decision-v7 development rows (`kev.metrics.served`, `TEMPERATURE_FIT`); the parents' are their shipped ones. All intervals are paired record-clustered bootstraps, micro aggregation, 2,000 resamples, seed 0, candidate minus parent.

**Rule for delta candidates (Phase 2), per size:**

1. Long states (primary): accuracy on the buried questions of the long-state panel, lower 95 percent bound > 0 and point estimate >= +5 pp. Selection panel: `evals/round5/longstate-v2/development.jsonl` (591 buried questions; read by round 5, so it is a selection set now, not a confirmation set). Confirmation panel: `evals/round6/longstate-v3/development.jsonl`, built in Phase 0, never read before the confirmation.
2. Short states (guard): accuracy lower bound >= -1 pp, Brier upper bound <= +0.01, confident-error rate (wrong at p >= 0.9, over all questions) upper bound <= +1 pp. Selection: `transfer-v4` development, scored inside every trial. Confirmation: `evals/round6/transfer-r6/test.jsonl`, built in Phase 0.
3. External guards, each a paired interval on the suite's own questions: SemIf-144 (`evals/external/semif-v1`), scienthoon-900 (`scienthoon-v1`), WANLI (`wanli-v2` if Phase 0 builds it, else `wanli-v1`), TypeSafe answered rows (`typesafe-v1`). Accuracy lower bound >= -2 pp where n >= 500, >= -3 pp where n < 500. Pooled over all external questions: point estimate >= 0 and lower bound >= -1 pp. `transfer-v9` development unknowable share at p >= 0.9 <= 0.05. This replaces round 5's point-estimate gates, which turned on three WANLI questions.
4. Among candidates passing 1-3 on the selection sets, rank by primary point estimate, then short Brier, then pooled external accuracy. The top one per size is the confirmation candidate.
5. Confirmation: read the candidate and its parent once on `longstate-v3` and `transfer-r6`; rules 1 and 2 must hold there. Then one locked read (`locked_test --trial <study>/<trial> --name kev-<size>-r6 --decision evals/v7/decision-v7`): pass if locked transfer-v4 accuracy >= parent's - 1 pp (9B 0.842, 4B 0.827, 0.8B 0.674) and served Brier on the locked transfer items <= parent's + 0.005 (parents' locked rows are under `runs/locked/kev-*-night2-du-ungated`). Report the result either way; a size that fails anything is not a candidate, and nothing is re-read.

**Rule for A1 (Phase 1), from PLAN_27b:** question-side placement is adopted as the default for later runs if `contrastive_deadline` raw accuracy on `transfer-v4` development is >= 0.70 at 4B or >= 0.80 at 9B (today's full-placement trials: 0.55 / 0.72; the untrained bases: 0.68 / 0.82) with overall transfer accuracy within 1 pp of the same-seed full-placement trial (paired interval reported). Also report MMLU, held-out pairs, and coverage at <= 5 percent error. No locked read for A1.

**Rule for Kev-27B (Phase B1), from PLAN_27b B1 with the round-4 corrections:** a trial is a candidate if `transfer-v4` development accuracy >= 0.842 (Kev-9B + 2 pp) with a paired lower bound > 0 against `runs/night2-9b-du/00-trial-0`, MMLU-Pro on `transfer-v9` development >= 0.65, no `transfer-v4` task more than 3 pp below Kev-9B's, held-out pairs >= 0.75, unknowable share at p >= 0.9 <= 0.05. Then the same fresh short panel read (`transfer-r6`, accuracy lower bound > 0 against Kev-9B) and, if it passes, one locked read named `kev-27b-r6`: pass if locked transfer-v4 >= 0.862 and served Brier <= 0.237. Record in PLAN.md that the base is post-trained, and that Jared authorized these trials on 2026-09-22 although the A2 probe missed the MMLU-Pro gate (0.635 vs 0.65) while passing the other two (transfer-v4 0.812 zero-shot with the SemIf prompt, SemIf-144 0.944; `runs/probes/qwen38-27b-semif-*`). That is a human override of the gate, and the plan must say so.

## 4. Budget and time

Baseline: whatever `modal billing summary --json` says at the start (it read $543.79 metered when this prompt was written). Hard stop at $1,000 of tonight's spend by the rule in section 2.1. Caps are on admission bounds unless stated:

| phase | cap | wall-clock target |
|---|---|---|
| 0 setup, panels, data, registration | $10 | done by T+75 min |
| 1 A1 question-side LoRA (3 from-scratch trials) | $35 | launched by T+90 min |
| B1 Kev-27B (fit check + up to 3 trials + reads) | $250 | fit check by T+90 min, trials launched by T+120 min |
| 2 delta hill-climb (rounds 1-2, reads included) | $500 | round 1 launched by T+120 min, round 2 by T+5 h |
| 4 confirmations and locked reads | $80 | reads by T+8 h |
| reserve, never planned into | >= $125 | |

Cost facts: H200 is $4.54/h GPU; `compute_bound` adds CPU and memory and comes to about $6.27 per container-hour. Measured wall times on H200 for the round-5 recipe (`runs/r5-combined/*/result.json`): 9B combined delta 8,000 s, 4B 6,800 s, 0.8B 3,600 s. From-scratch v7 recipe on H100: 4B about 60 min, 9B 90-110 min. Long-state panel reads of a 9B in fp32 exceeded one hour on H100; run them on H200 with `--timeout 7200`. Set `run_trial`'s `max_containers` in `modal_app.py` from 8 to 24 before deploying, or the 27B, A1 and delta trials queue behind each other; if containers still sit pending, that is workspace GPU capacity, not a bug; do not relaunch.

`admit_study` caps one study at a $250 bound and a 14,400 s timeout. Split studies by size (`r6-9b`, `r6-4b`, `r6-08b`, `r6-a1`, `r6-27b`). Decision-v7's manifest pins only the Qwen3 bases, so every plan entry carries `base_revision`: 9B `68c46c4b3498877f3ef123c856ecfde50c39f404`, 4B `1001bb4d826a52d1f399e183466143f4da7b741b`, 0.8B `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, Qwen3.8-27B `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## 5. The program

### Phase 0: setup, verification, panels, data (target 75 minutes)

1. `git status`. The checkout is on `a1/question-side-lora` with one WIP commit (`4440076`) on top of `main` (`2ba8166`). Untracked probe results under `runs/probes/qwen38-27b-*` and `runs/r4-kev-9b-v7-train` belong in git where `.gitignore` allows (reports, not prediction dumps); add them.
2. **Verify the A1 WIP.** It adds `lora_placement` (`full` | `question`) to `DecisionModel`, `Meta`, `kev.train` and the trial allowlist. Requirements: the CI unit job passes (`uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py -q`); the weight-backed tests pass, including the new `test_question_side_lora_gates_the_state_exactly` (`uv run --extra serve python -m pytest tests/test_model.py -q`; downloads Qwen2.5-0.5B and Qwen3.5-0.8B-Base, a few minutes); and `full` placement is bit-identical to `main`, shown with the worktree parity harness in the `kev-verify` skill or, at minimum, by `test_model.py` being fully green plus the assertion in the new test that a full-placement model installs no hooks. If all of that is green within 45 minutes: `git checkout -b research/overnight-r6`, commit the WIP properly (message: what the gate is and why the recompute test exists). If not: `git checkout -b research/overnight-r6 main`, skip Phase 1 and 27B trial C, and write why in PLAN.md.
3. **Fresh panels** (frozen, sha256 in manifests, never read before Phase 4):
   - `evals/round6/transfer-r6`: `uv run python scripts/freeze_calibration_audit.py --out evals/round6 --tag r6 --panel_only --seed 2026092306`. Confirm the builder reports zero overlap with every state under `evals/` (that includes round 5's panel). Only `test.jsonl` is the panel.
   - `evals/round6/longstate-v3`: `scripts/build_long_states.py --out evals/round6/longstate-v3 --version longstate-v3 --seed round6-longstate-v3 --panel_partition calibration --dev_per_length 120` with the same four lengths as v2 (1k / 2k / 4k / 6k). The panel's primaries must be disjoint from `evals/round5/longstate-v2/development.jsonl` by `parent_id`; if the builder has no exclusion flag, add one (`--exclude_panel`). The same run builds this round's training variant: `--train_counts 300,300,1100,1100` (more 4k and 6k, same 2,800 total; the v2 mix 400/400/1000/1000 left 4k weak).
   - `evals/external/wanli-v2` (30-minute time box): about 1,000 WANLI test items from `alisawuffles/WANLI`, disjoint from `wanli-v1`'s 256 by premise and hypothesis text, rendered exactly as `scripts/freeze_semif_external.py` renders v1, dataset revision pinned in the manifest. If it does not build in 30 minutes, the WANLI guard uses v1 with the n < 500 margin and PLAN.md says so.
4. **Round-1 training data**, one script `scripts/build_round6_data.py`, outputs under `evals/round6/<variant>/` with manifests. Inputs: `evals/round5/longstate-v2/train.jsonl` (2,800 long records), `evals/round4/ambiguity-v1/{soft,hard}.jsonl` (1,738 records; 150 of the softened questions are MNLI, see its manifest), `evals/night2/dates_unknowable.jsonl`, the teacher rows above. Variants:
   - `soft-nomnli`: combined soft with the MNLI questions' hard labels restored (round 5's registered next question: is the WANLI drop the softened MNLI targets?).
   - `soft-lam03`: ambiguity targets at `--lam 0.3` (closer to the label), plus the long records.
   - `soft-thr08`: ambiguity targets at `--threshold 0.8` (only the most contested questions softened), plus the long records.
   - `soft-du`: combined soft plus the dates + unknowable records the released parents were trained on (a delta on a delta replays only decision-v7; this tests whether that forgets the night-2 delta).
   - `soft-longmix3`: `longstate-v3/train.jsonl` plus the ambiguity soft records.
   - `soft-half`: 1,400 long records (200/200/500/500, a fixed-seed subsample of v2 train) plus the ambiguity soft records; for 4B, where the long records cost scienthoon 1.6-3.9 pp.
   Verify each file loads through `kev.suite.read_jsonl` and that `validated_trial` accepts a plan pointing at it. Drop a variant that is not built by T+60 rather than delay the launch.
5. `modal_app.py`: `max_containers=24` on `run_trial`. Then `KEV_GPU=H200 uv run modal deploy modal_app.py`. Validate every plan file locally with the `load_plan` one-liner from the skill before launching.
6. Write the registration section (section 3) into `PLAN.md`, commit, push.

### Phase 1: A1, question-side LoRA (from scratch, v7 recipe)

Plan `experiments/round6/a1.json`, study `r6-a1`, `--suite evals/v7/decision-v7 --transfer evals/v4/transfer-v4 --timeout 10800`. Three trials with `lora_placement: "question"`, otherwise the released v7 recipe (`epochs 2, lr 5e-5, dtype bf16, checkpointing 1, p_none_pair 0.25, lora 16`): 4B seed 1 (`batch 4, accum 2`; reference `runs/q35-4b/01-trial-1`, transfer 0.800), 4B seed 2 (reference `runs/q35-4b-s23/00-trial-0`, 0.794), 9B seed 1 (`batch 2, accum 4`; reference `runs/q35-9b/01-trial-1`, 0.812). Read-out: the A1 rule, deadline raw per size, paired intervals against the same-seed reference from the saved `transfer/rows.json`. Also note in PLAN.md that a question-side checkpoint cannot be merged or served through MLX (the checkpoint loader refuses), so adoption has a Mac serving cost to state.

### Phase B1: Kev-27B (up to three trials)

1. Fit check first: `uv run modal run modal_app.py::smoke_base --base Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --gpu H200`. Record peak memory, which modules LoRA hits, and steady step time. Project the wall time of the v7 recipe (2 epochs over decision-v7's 12,576 records plus augmentation, effective batch 8) from the step time. If 2 epochs do not fit in 14,400 s with 50 percent headroom, run 1 epoch and say so. If 1 epoch does not fit either, raise the timeout cap in `admit_study` and the `run_trial` decorator to 28,800 s, keep the study bound under $250, redeploy, and say so.
2. Plan `experiments/round6/27b.json`, study `r6-27b`, all trials `base Qwen/Qwen3.8-27B`, `base_revision` above, `weights_dtype bf16, dtype bf16, checkpointing 1, batch 2, accum 4, p_none_pair 0.25, lora 16, lora_targets all, seed 0`: (A) `lr 5e-5`, (B) `lr 2e-5`, (C) `lr 5e-5, lora_placement question` (only if Phase 0 step 2 was green; this is the A1 ablation at the size where the base's date arithmetic is strongest, zero-shot deadline 0.95). If the fit check says three do not fit the $250 cap, drop (B).
3. When they finish: `pull`, then one `benchmarks` pass per candidate on `transfer-v9` (MMLU-Pro, unknowable) and `transfer-r6` if the dev rule passes, then the locked read if that passes. If the trials will not finish by T+8 h, leave them running and write the pull and read commands into the report as pending.

### Phase 2: delta hill-climb (the main event)

Unit: a one-epoch delta from the released parent, `init_from jaredpalmer/kev-<size>`, `replay 2000`, `max_state 7552`, `dtype bf16, checkpointing 1, p_none_pair 0.25, seed 1`, lr 2e-5 (0.8B: 4e-5), batch/accum as in `experiments/r5-combined.json`, `--suite evals/v7/decision-v7 --transfer evals/v4/transfer-v4`. Timeouts: 9B 12,600 s, 4B 10,800 s, 0.8B 7,200 s. Round-5 incumbents for reference (already on the volume and locally): C9 `r5-combined/00-trial-0`, C4 `03-trial-3`, C08 `05-trial-5`, and their reads under `runs/r5r-*`.

**Round 1 (one knob each, registered arms):**

- 9B, study `r6-9b` (7 trials): `soft-nomnli`; `soft-lam03`; `soft-thr08`; `soft-du`; `soft-longmix3`; combined soft with `replay 6000`; combined soft with `lr 1e-5`.
- 4B, study `r6-4b` (6 trials): `soft-half`; `soft-nomnli`; `soft-du`; `soft-longmix3`; combined soft with `replay 6000`; combined soft with `lr 1e-5`.
- 0.8B, study `r6-08b` (2 trials): combined soft with `lr 2e-5`; combined soft with `replay 6000`. (C08 failed on confident errors and scienthoon; these are the two drift levers.)

Launch order: `r6-a1`, `r6-27b`, `r6-9b`, `r6-4b`, `r6-08b`, 30 seconds apart, each redirected to `runs/<name>.log`. Then monitor per the skill.

**Reads per round.** When a study finishes, `pull` it and refresh the leaderboard. Every trial gets a long-panel read on `evals/round5/longstate-v2` (selection panel). Trials that pass the short-state guard on their in-trial `transfer-v4` result also get the four external suites and `transfer-v9`. Batch reads into `benchmarks` calls of up to ten jobs each, H200, `--timeout 7200`, names `r6-<size>-<arm>-<suite>`; a failed job leaves its directory on the volume, so a retry needs a new name. Write a `scripts/round6_readout.py` (start from `round5_confirm.py`; both since replaced by `experiments/rounds/r{5,6}.json` and `python -m kev.rounds readout`) that applies the section-3 rule to every trial from the saved rows and prints the ranked table with intervals; commit it before the first read-out.

**Round 2 (launched by T+5 h, up to 8 trials, mostly 9B and 4B):** combine the knobs that passed the guards and improved the primary or the externals in round 1 (two-knob combinations of round-1 winners), replicate the current best per size at seed 2, and drop anything that failed a guard with an interval clear of the threshold. If nothing in round 1 beats the round-5 incumbent on the ranking, round 2 tests the incumbent's recipe at seeds 2 and 3 for 4B and 0.8B (seed 2 already exists for 9B) and the two most promising round-1 knobs at the other size; say in PLAN.md that this is what happened. Do not launch a round 2 that cannot finish training and reads by T+7.5 h; a smaller round 2 beats an unread one.

### Phase 4: confirmation, write-up, hand-off

1. Pick at most one candidate per size by rule 4 of section 3 and write the choice, with its selection numbers, into PLAN.md before any confirmatory read. Copy `scripts/round5_confirm.py` to `scripts/round6_confirm.py` with the round-6 arms, panels and thresholds; commit (both since replaced by the `confirm` stages of `experiments/rounds/r6.json`, `python -m kev.rounds confirm`).
2. Read candidate and parent on `longstate-v3` and `transfer-r6` (and `wanli-v2` if built) in one `benchmarks` call per size. Apply the rule. Then the locked read for each passing size and for a passing 27B. One read each, no exceptions.
3. `uv run python -m kev.autoresearch leaderboard`. Write the results under the round-6 section in PLAN.md: spend (baseline, final metered reading, bound and actual per phase), one table per size (arm, primary and guards with intervals, verdict, report paths), the A1 table, the 27B table, the confirmatory and locked results with names, what is still pending on Modal with the exact commands to pull it, incidents (timeouts, relaunches, orphan directories on the volume to remove), and at most three next steps with the evidence for each. Add the A2 probe numbers to `PLAN_27b.md`'s gating addendum.
4. Commit and push. Do not open a PR. Print the PLAN.md round-6 section as your final message.

## 6. Things that have gone wrong before

- The image was deployed before a new `evals/` file existed and the trial failed on `FileNotFoundError` an hour in. Deploy after the data, not before.
- A `modal run ...::study` filtered through `grep` hid the `SystemExit` explaining why nothing launched. Redirect to a log file and read it.
- Three round-5 reads died at the one-hour benchmark timeout on H100 and their partial predictions had to be deleted unread. Long-state reads: H200, `--timeout 7200`.
- The spend-limit stop of the first overnight killed twelve containers mid-training. Check the reading before every launch; the dashboard limit cannot be raised from the CLI.
- `kev.data.none_pair` once trained zero mass on soft targets; fixed in #60. If you write a new data builder, check a handful of soft-target records by eye (`target` sums to 1, the label's mass is >= 0.5 unless the record is unknowable).
- A one-seed lead of 1 pp on 656 questions is noise (round 2). Every claim tonight is an interval on paired rows.
