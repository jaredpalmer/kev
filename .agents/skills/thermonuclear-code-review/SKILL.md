---
name: thermonuclear-code-review
description: Extremely strict structural review of a Kev branch or PR (code-judo simplifications, spaghetti growth, files past 1k lines, boundaries, duplication of canonical helpers). Use when asked to review a PR, audit a diff, or run a "thermonuclear" or deep code quality review on this repo.
---

# Thermonuclear review, Kev edition

A review for implementation quality, not correctness: abstraction quality, maintainability, codebase health. Behaviour
is assumed to be checked elsewhere (`kev-verify`). Be ambitious: do not stop at local cleanups. Look for the "code
judo" move, a restructuring that keeps behaviour and makes the change dramatically smaller, more direct and more
obvious, so that whole branches, helpers, modes or layers disappear. Prefer the version that feels inevitable in
hindsight. Measure twice, cut once.

The standards below are adapted from cursor-team-kit's `thermo-nuclear-code-quality-review` (once vendored next to
this file); the second half is what a reviewer needs to apply them to this repository.

## Standards

1. **Be ambitious about structural simplification.** Ask of every meaningful change: can it be reframed so fewer
   concepts, branches or helper layers are needed? Prefer deleting complexity to rearranging it. A refactor that moves
   code around without reducing what a reader must hold in their head has not earned its diff.
2. **No file crosses 1,000 lines because of a PR** without a very strong reason. Ask whether the file should be
   decomposed first; extract modules or helpers instead of letting it sprawl. Waive only when the result is still
   clearly organised.
3. **No spaghetti growth.** New ad-hoc conditionals, scattered special cases, one-off booleans, nullable modes or
   "temporary" branches inserted into unrelated flows are design problems, not style nits. Push the logic behind a
   dedicated abstraction, typed model, dispatcher or module; reframe the state so the conditionals disappear rather
   than get centralised.
4. **Clean the design, do not just accept working code.** If behaviour can stay the same while the structure becomes
   meaningfully cleaner, ask for the cleaner version. Prefer removing moving pieces over spreading the same complexity.
5. **Direct, boring code over hacky or magical code.** Be skeptical of generic mechanisms hiding simple data-shape
   assumptions. Flag thin wrappers, identity abstractions and pass-through helpers that add indirection without clarity;
   the remedy is usually to delete the layer, not polish it.
6. **Type and boundary cleanliness.** Question casts, `Any`/`unknown`, optional parameters and silent fallbacks that
   paper over an unclear invariant. Prefer an explicit typed model or shared contract; make the boundary explicit so the
   control flow gets simpler.
7. **Logic in its canonical layer; reuse existing helpers.** Feature logic leaking into shared paths, implementation
   details leaking through APIs, and bespoke near-duplicates of an existing utility are all blockers. Move the code to
   the module that already owns the concept (table below).
8. **Orchestration smells.** Independent work serialised for no reason, and related updates that can leave state
   half-applied, are design smells when a cleaner atomic or parallel structure is obvious. Do not micro-optimise.

Findings go in this priority order: structural regressions; missed code-judo simplifications; branching complexity;
boundary / type-contract problems; file size and decomposition; modularity; legibility. Few high-conviction comments
beat many nits. Be direct and demanding without being rude; if the code makes the codebase messier, say so, and if it
missed a dramatic simplification, say that too. "Maybe rename this" is not feedback when the real issue is structural.

Useful shapes: "this pushes the file past 1k lines; can we decompose it first?", "this adds another special case to an
already busy flow; can it live behind its own abstraction?", "this looks like a bespoke helper for something we already
have; can we reuse the canonical one?", "there is a code-judo move here; can we reframe so these branches disappear?",
"this refactor moves complexity around but does not delete it; can the model itself be simpler?"

### Approval bar

Do not approve because behaviour seems correct. Approve when there is no clear structural regression, no visible path
to a dramatically simpler implementation left untaken, no unjustified file-size explosion, no spaghetti growth from
special-case branching, no hacky or magical abstraction, no wrapper / cast / optionality churn hiding the real design,
and no boundary leak or canonical-helper duplication. Each of those is a presumptive blocker until the author justifies
it. Otherwise leave explicit, actionable feedback and push for the cleaner decomposition. Say plainly when something is
fine.

## How to run it here

1. Get the diff (`git diff main...<branch>` or `gh pr diff <n>`) and read every changed file in full, not just hunks.
   The previous version of a file is `git show main:<path>`; for a reviewer without a shell, keep an `origin/main`
   worktree (e.g. `/tmp/kev-main`) and read the old file from there.
2. Read the callers of anything the diff touches (`grep` the symbol across `kev/`, `scripts/`, `space/`, `tests/`,
   `modal_app.py`, `playground/src`).
3. Check the canonical-helpers table below before accepting a new helper: a second copy of a rule that has a home is a
   blocker, not a nit. `tests/test_conventions.py` enforces several rows.
4. Ask for the parity evidence the `kev-verify` skill describes (bit-identical rows / weights against `main`) whenever the
   diff touches the model, loader, trainer, data converters or metrics. Green tests are not parity.
5. Report findings in the priority order above with `file:line` references, then an explicit verdict against the
   approval bar.

## Canonical helpers (reuse, do not re-derive)

| fact | home |
|---|---|
| load/resolve a checkpoint, read/write `head.pt` (`Meta`), warm-start LoRA+head, `LoadOptions` (+ `from_env` at CLI entry points only) | `kev/checkpoint.py` |
| whether a checkpoint is a LoRA adapter or full weights (the loader rule), its shards, the hash provenance pins | `kev.checkpoint.Checkpoint.full` / `shards` / `weights_sha256` |
| full-weight training state: fp32 masters and moments (host or device), FSDP2 sharding, each rank's share of an epoch, saving the backbone, resume points | `kev/full_ft.py` (`MasterAdamW`, `shard`, `rank_share`, `save_backbone`, `save_due` / `save_resume` / `load_resume`) |
| the training forward that runs each state once and its question branches from it (hybrid backbones) | `kev/shared_prefix.py` (`branch_hidden`, `Prefix`) |
| a trial container's price per hour and a study's admission bound (retries included), the study limits | `kev/budget.py` (`hourly_rate`, `compute_bound`, `MAX_TIMEOUT`, `MAX_BUDGET`, `FULL_FT_RETRIES`) |
| option keys for a question (choice/noul/score) | `kev.api.question_keys` |
| does a record fit the training context (`MAX_STATE/MAX_BRANCH/MAX_PACKED`); a lifted state limit (`training_context(max_state)`, ceiling `MAX_TRAIN_STATE`) | `kev.model.fits(rec, *tokenizers)`, `kev.model.training_context`; manifests write `kev.suite.CONTEXT` |
| default device / sync / empty_cache / allocated_bytes | `kev/device.py` |
| read/write JSON and JSONL as UTF-8 (`read_json`, `read_jsonl`, `write_json`, `write_jsonl`), read a manifest, sha256 a file, load a split, trainable/eval-only policy (`validate_training`), `semantic_hash`, `SYNTHETIC_SOURCES`, a state's normalised-text hash (`normalise_text`, `text_digest`), the git size limit for partitions (`GIT_LIMIT`), the Qwen3.5 tokenizer pin suite builders admit under (`ADMISSION_TOKENIZER`), an atomic state-file write (`write_json(..., atomic=True)`) and a local advisory lock (`file_lock`: pulls, an arm's read launches) | `kev/suite.py` |
| labelled request -> API request / internal record; augmentation that must skip soft-target questions (`augment`, `none_pair`) | `kev.data.api_request`, `kev.data.materialize`, `kev.data` |
| selective-prediction metrics, the rows metrics run on (`scored_rows`), a row at another temperature (`tempered_row`) and back to T=1 (`raw_row`), temperature fit (`fit_temperature`; the shipped grid is `TEMPERATURE_FIT`), group-disjoint folds and out-of-fold calibration (`grouped_folds`, `out_of_fold_rows`, `cross_validated_temperature`); per-workload report `kev.calibrate`; served-vs-served reads (`served` fits T on raw development rows and serves eval rows, `served_at`, `recorded`); the bootstrap resampling unit (`cluster_resamples`) and `paired_bootstrap` | `kev/metrics.py` |
| a registered round (spec schema, served temperature per trial `temperature`, `served_clean`, the paired read `paired`, rule evaluation, candidate ranking, read/launch commands, the benchmark job string `bench_job`, the watcher and its network-error test `transient`); a release's numbers read a release spec (`experiments/releases/`) | `kev/rounds.py` (+ `modal_app.parse_jobs` for the job string; the admission bound and trial resources are `kev/budget.py`) |
| predictors (local checkpoint, remote System One endpoint, Jev) | `kev/predictors.py` |
| rows from predictions, `summarize`, `evaluate_records` | `kev/benchmark.py` |
| research gates and thresholds | `kev.experiment.GATES`, `gate_report` |
| the unrelated sibling question for isolation checks (fp32 `mechanism_checks`, served `scripts/serving_bench.py --isolation`) | `kev.experiment.ISOLATION_PROBE` |
| published README/model-card numbers -> committed reports | `docs/claims.json` + `scripts/verify_claims.py` |

When a helper becomes canonical, add it here and add a rule to `tests/test_conventions.py`.

## Repo-specific things reviewers have caught

- A field dropped from an import while moving a function (`HELD_OUT_KEYS`), an env-var side channel into the loader,
  a temperature applied twice on the locked-test read, a `max_branch=960` admission literal disagreeing with the
  manifest it wrote. Look for exactly these shapes: silent partial moves, hidden channels, doubled application of a
  correction, and constants that exist twice.
- Version-named modules (`study_v3`, `transfer_v9`) are research scripts; core code (`train`, `experiment`, `serve`)
  must not import from them.
- Frozen suites under `evals/` and `runs/leaderboard.*` are artifacts: a refactor must not regenerate or commit them.
