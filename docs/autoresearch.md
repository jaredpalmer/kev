# Running an unattended research session

This is the operating program for a research session that runs without a human watching: an overnight session, or a
long hand-off during the day. It replaces the per-night prompts (the round-6 and night-3 programs, kept at the git tag
`research-archive-2026-09-24` under `docs/prompts/`) and folds in what went wrong on those nights. The research rules it
applies are in [`PLAN.md`](../PLAN.md), "Standing rules for every round"; the commands are in [`AGENTS.md`](../AGENTS.md).

A session hill-climbs and confirms under registered rules and leaves a record Jared can act on. It does not publish.

## 1. Start

Spend the first 30 minutes reading, not launching.

1. `AGENTS.md`, all of it (commands, frozen suites, canonical homes, Modal settings).
2. `PLAN.md`: where we stand, what we have learned, the standing rules, the data policy and Next. For a finding you want to
   build on, read its evidence in the archive: `git show research-archive-2026-09-24:PLAN.md`.
3. The skills in `.agents/skills/`: `kev-modal-study` (launching, watching and pulling GPU work; read its Gotchas),
   `kev-verify` (proving a code change has no regression), `kev-pr-description` (before any PR), `thermonuclear-code-review`.
4. `kev/rounds.py` (its docstring is the spec schema), the closest past spec in `experiments/rounds/`, and
   `kev/autoresearch.py` (`session`).
5. The hand-over itself: authorization (Modal dollars, AI Gateway dollars), what is in scope, what needs Jared.

Then set up:

- Work in a worktree on a research branch (`git worktree add -b research/<session> /tmp/kev-<session> origin/main`). Push
  after every commit so nothing is lost if the machine sleeps. Code meant for main goes through its own reviewed PR.
- Read `uv run modal billing summary --json` and record `metered_cost` as the baseline in the state file (section 6).
- If a previous session left a state file, read it first and resume from it; detached Modal jobs keep running without you.

## 2. Budgets and the spend rule

- The authorization is the session's total, counting everything still running. Before **every** launch, read the metered
  cost again and do not launch if `(metered_now - baseline) + sum(admission bounds of everything still running) >= authorization`.
- A study's admission bound is printed at launch and saved in `runs/<study>.spawn.json`; a benchmark call's bound is
  `compute_bound(gpu, timeout, trials)` (`kev/budget.py`). A spec's study `budget` must be at least its bound
  (`kev.rounds validate` checks it; `modal_app.admit_study` refuses a study over its budget before anything runs, and a
  study is capped at $250 and 28,800 s).
- Keep a reserve (about 10 % of the authorization) that no phase plans into: billing readings lag and get revised, and
  admission bounds overstate reads badly (a read batch carries the timeout of its slowest job).
- Record every reading with its UTC time in the state file. AI Gateway spend (Jev reference reads, label judges) has its own
  cap, enforced by the script that spends it, and is logged in `runs/<name>/usage.json`.
- The Modal workspace spend limit can only be raised from the dashboard; hitting it kills running containers mid-training.

## 3. Register a round

A round is a PLAN.md section plus a spec, committed together before any training or read.

1. Write the PLAN.md section: why (the measured gap and its evidence), the data (frozen first, with manifests), the arms, the
   rule (primary, guards with thresholds sized to each suite, rank), the confirmation stages, and the budget. Use the
   standing rules; do not invent a new statistic for one round.
2. Write `experiments/rounds/r<N>.json` by copying the closest past spec (r15 for a joint delta, r17 for a 27B, r10 for a
   skills round, r20 for post-hoc arms without training: a temperature pool, interpolated checkpoints). Leave out `"archive"`: that key marks the recorded rounds 5-18. Every plan file the spec names, and every
   parent read its rule needs, must exist in this checkout; if a parent lacks a read, `launch-reads <spec> --parents` makes it.
3. New data is a new directory under `evals/` with a `manifest.json` (sha256 per file, inputs' hashes). Under the SFT data
   policy (PLAN.md) private corpora keep only the manifest in git, with a `"mirror"` entry pointing at the private dataset.
4. `uv run python -m kev.rounds validate experiments/rounds/r<N>.json` (add `--partitions` to verify the partitions) until it
   prints `ok`. Commit the PLAN section and the spec in one commit, push. That commit time is the registration time.

## 4. Run it end to end

```bash
KEV_GPU=H200 uv run modal deploy modal_app.py                     # after any change to kev/*.py or any new file under evals/
uv run python -m kev.rounds launch experiments/rounds/r<N>.json   # one ::study per study, 60 s apart, logs in runs/<study>.log
caffeinate -i nohup uv run python -m kev.rounds watch experiments/rounds/r<N>.json > runs/r<N>.watch.log 2>&1 &
```

- In the first five minutes of every study, count optimizer steps per minute in `modal container logs <id>` and project
  the wall time against the timeout (`ep0 step N/M`: M is over all epochs). A timed-out container saves nothing; cancel
  (`FunctionCall.from_id(cid).cancel()`) and relaunch under a new study name with fewer records or a longer timeout.
- `watch` polls the spawned trials, pulls each finished study (one pull per study at a time), launches that arm's reads once
  (one batched `::benchmarks` call per arm, 60 s apart), waits for them and writes `runs/r<N>-readout/round<N>.json` and a
  table. It is restartable: state is in `runs/<study>.watch.json` and the launch intent in `runs/r<N>-reads-<arm>.json`.
  By hand: `launch-reads <spec> [--arms a,b] [--parents] [--dry-run]`, `readout <spec>`.
- Write the read-out into the PLAN section: every arm, every criterion with its interval, the verdict and what failed.
- **Confirmation is deliberate, never automatic.** For the candidate the read-out names, write the choice into PLAN.md and
  commit it, then per stage: `launch-reads <spec> --stage <stage> --arm <arm>`, then
  `confirm <spec> --stage <stage> --arm <arm>` (→ `runs/r<N>-verdict/<size>-<stage>.json`). Test panels before the locked read.
  One read each, no exceptions.
- Several registered rounds in sequence under a cap: `uv run python -m kev.autoresearch session experiments/rounds/r19.json
  [...] --spend-start <baseline> --spend-cap <authorization>`. It validates, launches and watches each round to its read-out,
  stops before a round whose budgets would pass the cap, appends to `runs/autoresearch-sessions.jsonl`, and prints the
  confirmation commands; it never runs them. `kev.autoresearch leaderboard` refreshes `runs/leaderboard.{jsonl,md}` (not
  committed), `compare` pairs trials against a reference on transfer accuracy, `release-check --study <name>` screens every config in that study (each config passes only if all its seeds pass their gates).

## 5. What a session may and may not touch

May: write specs, plans and PLAN.md sections; build new frozen data under new directories; launch studies and reads through
`modal_app.py`; change scripts and `modal_app.py` infrastructure constants; open PRs for code that belongs on main.

May not, without Jared's explicit OK:

- publish or change anything on the Hub (`kev.publish`, `hf upload`, `hf repos tag`, `scripts/publish_space.sh`, a released
  `head.pt`), make a private repo public, or deploy a public endpoint;
- commit to main, force-push, or merge a PR (code reaches main through reviewed, squash-merged PRs with green CI);
- edit anything that exists under `evals/` (frozen), or the evaluator: `kev/experiment.py: EVALUATOR_FILES`, the gates,
  `kev/metrics.py`, `kev/rounds.py`'s paired read. A needed evaluator change is its own PR, verified with `kev-verify` and
  `tests/test_rounds.py`, before any round depends on it;
- pass `--allow-test` or run `locked_test` outside a registered confirmation stage;
- put any Jev output, or any closed-model generation, into training data;
- train locally (a 32 GB Mac cannot hold these models) or run two training processes on one machine;
- delete a checkpoint or a snapshot from the runs volume (`modal volume rm`, `shutil.rmtree` in a container), or turn a
  full-weight trial's snapshots off (`"snapshot_fractions": "none"`) in a registered spec. Full-weight trials keep
  snapshots at 0.25, 0.5 and 0.75 of their steps (`kev.experiment.SNAPSHOT_FRACTIONS`) so a read can find the best point
  of a run after it ends: round 19 could not, because the only mid-run state was a resume point, deleted when the run
  finished, and AutoJev's best checkpoint was at 0.7 epoch. A 27B's snapshots are ~154 GB of volume per trial; the
  space is Jared's call, not the session's.

If an arm is blocked (authentication, a spend limit, a deploy that will not work in 30 minutes), write down what happened
and move to the next arm. Do not wait for a human.

## 6. Resilience

- **State file** `runs/<session>-state.json` (runs/ is gitignored; `git add -f` it on the research branch): baseline and
  authorization, spend readings with UTC times, every study with its spawn ids, bound and status, reads launched and
  pulled, candidates, PRs, pending decisions. Update it after every launch, pull and read, and commit it with the PLAN section.
- **Detached jobs.** Studies spawn on the deployed app and survive the local client; a local error after `study` may still
  have spawned trials, so run `modal container list` before relaunching, and never relaunch under the same study name.
  Probes and benchmarks run with `--detach`.
- **Watchers are local processes** and die with the machine or the network. Run them under `nohup` and `caffeinate`;
  restart `watch` after any interruption (it resumes from its state). It retries DNS and connection errors itself; a trial's
  own exception is a failure and is reported.
- **Network drops** kill local clients, not the remote work: a read whose client died has usually finished on Modal; pull
  its directory from the volume (`modal volume get kev-runs /<name> runs/<name>`) instead of relaunching it.
- A failed benchmark or probe leaves its directory on the volume; retry under a new name.

## 7. Reporting

At the end of the session (and in the state file as it goes):

- Each round's PLAN.md section carries its registration, read-out table, confirmation results and verdict, negative or not,
  with report paths.
- Update PLAN.md "Where we stand" (released and confirmed candidates, running jobs, spend) and "What we have learned" if a
  finding changed; add each round to the Record table.
- A session summary in PLAN.md: spend (baseline, final reading, running bounds), what is pending on Modal with the exact
  commands to finish it, incidents, and at most three next steps with their evidence.
- Every number carries checkpoint, suite and partition, n and report path; commit the read-outs and verdicts the numbers
  come from (`.gitignore` keeps reports, not prediction dumps; add a rule for new read-out directories).
- Clock stamps: registration and result times are commit times. Do not write a time into a heading before it happens;
  night 3's scratchpad did, and its stamps could not be used.

## 8. Known gotchas

- **Modal app-create rate limit.** More than about three detached `modal run`s within a minute fail with "App create rate
  limit exceeded" and nothing runs. `kev.rounds` staggers launches 60 s apart and batches an arm's reads into one call; do
  the same by hand.
- **`repo@sha` in benchmark jobs** used to shift every field of `run@suite@name@flags`; `modal_app.parse_jobs` now parses
  from the right, so pinned Hub revisions are safe. Suites and names must not contain `@` or `,`.
- **One pull per study.** Concurrent pulls of the same study deleted each other's trial directories; `pull_study` now holds
  a per-study lock. A pull while trials still run is safe and refreshes only unfinished trials.
- **Pulls leave full weights on the volume.** `::pull` (and `watch`) skips full-weight shards (`model*.safetensors`, ~51 GB
  per 27B checkpoint or snapshot) and resume points; everything else comes down (results, rows, `head.pt`, configs).
  Read a checkpoint or a snapshot on the volume: `::benchmarks --jobs "/runs/<study>/<trial>/snapshots/step-<N>/checkpoint@<suite>@<name>"`.
  `::pull --weights` copies the shards when something local really needs them.
- **Deploy after the data.** The image copies `evals/`; the launcher only checks `kev/*.py` hashes, so a trial whose data
  file was added after the deploy fails inside the container. `--gpu H200` on `study` needs an app deployed with `KEV_GPU=H200`.
- **27B.** H200 only (bf16 backbone, 55 GB resident); study timeouts up to 28,800 s (a 1-epoch skills delta at lr 2e-5 ran
  about 8.8 s per optimizer step); fp32 reads about three times a 9B's (spec `read_timeout: {"27b": 14400}`); the locked read
  needs `--timeout 14400 --memory-mb 131072` (spec `locked_args`) on an H200 (the GPU comes from the spec's `gpu` / the
  deployed app, or `--gpu H200` by hand). Every bf16-weights trial fails the in-trial
  `isolation_and_packing` gate (an fp32 check); read results from the rows and measure served isolation in bf16 separately.
- **`locked_test` naming.** When an in-trial screening gate failed, the tool requires the `-ungated` suffix
  (`kev-4b-r8-ungated`); the verdict still follows the registered rule.
- **Read timeouts per suite.** `modal_app.READ_TIMEOUTS` sets long-state panels 7,200 s, documents 5,400 s, transfer-v9
  3,600 s, else 1,800 s. One global `--timeout` inflates the admission bound of every job in the batch.
- **Budget admission.** A launch over its `--budget` exits before anything runs; relaunch with a budget at least the printed bound.
- **External servers are single-flight.** AutoJev's server answered one request at a time (HTTP 529 while busy); probe a
  foreign endpoint before a long `kev.benchmark --remote` and set `--remote-concurrency` to what it can take. Count the
  requests it rejects (for example a 422 past its context) as coverage, never drop them silently.
- **Temperature: shipped vs in-trial.** A trial's `result.json` and its locked summary are scored at the in-trial fit; a
  release ships the T that `scripts/calibrate_checkpoint.py` wrote into `head.pt`. The first AutoJev head-to-head served
  Kev-27B at the in-trial 1.19 instead of the shipped 1.38 and had to be corrected. Say which T every number uses.
- **Workspace capacity.** The workspace has run at most about ten GPU containers at once; pending containers are capacity,
  not a bug, so do not relaunch them.
- **Small suites.** A guard on 89 or 144 questions cannot resolve a 2-3 pp floor; gate them through a pooled panel.
- **Jev reads fail mid-run** on gateway 503s; rerun the whole read under a new name rather than stitching partial rows.
- **Soft-target data.** When writing a builder, check a handful of records by eye: `target` sums to 1 and the label's mass is
  at least 0.5 unless the record is unknowable (`kev.data.none_pair` once trained zero mass on soft targets, fixed in #60).
- Redirect `modal run ...::study` output to a log file; a filter can hide the `SystemExit` that explains why nothing launched.
