---
name: thermonuclear-code-review
description: Strict structural code review of a Kev branch or PR (duplication of canonical helpers, spaghetti growth, boundaries, missed code-judo simplifications). Use when asked to review a PR, audit a diff, or run a "thermonuclear" review on this repo.
---

# Thermonuclear review, Kev edition

The standards are the installed `thermo-nuclear-code-quality-review` skill
(`.agents/skills/thermo-nuclear-code-quality-review/SKILL.md`, from cursor-team-kit, model invocation disabled): be
ambitious about structure, no spaghetti growth, no file crossing 1k lines, no thin wrappers or tri-state flags, logic in
its canonical layer, and its approval bar. Read that file first. This skill adds what a reviewer needs to apply those
standards to *this* repository.

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
5. Report findings in the standards' priority order with `file:line` references, then an explicit verdict against the
   approval bar. Few high-conviction comments beat many nits. Say plainly when something is fine.

## Canonical helpers (reuse, do not re-derive)

| fact | home |
|---|---|
| load/resolve a checkpoint, read/write `head.pt` (`Meta`), warm-start LoRA+head, `LoadOptions` (+ `from_env` at CLI entry points only) | `kev/checkpoint.py` |
| option keys for a question (choice/noul/score) | `kev.api.question_keys` |
| does a record fit the training context (`MAX_STATE/MAX_BRANCH/MAX_PACKED`) | `kev.model.fits(rec, *tokenizers)`; manifests write `kev.suite.CONTEXT` |
| default device / sync / empty_cache / allocated_bytes | `kev/device.py` |
| read/write JSON and JSONL as UTF-8 (`read_json`, `read_jsonl`, `write_json`, `write_jsonl`), read a manifest, sha256 a file, load a split, trainable/eval-only policy (`validate_training`), `semantic_hash`, `SYNTHETIC_SOURCES` | `kev/suite.py` |
| labelled request -> API request / internal record | `kev.data.api_request`, `kev.data.materialize` |
| selective-prediction metrics, the rows metrics run on (`scored_rows`), a row at another temperature (`tempered_row`), temperature fit (`fit_temperature`, `points=` for the grid), group-disjoint folds and out-of-fold calibration (`grouped_folds`, `cross_validated_temperature`), paired bootstrap | `kev/metrics.py` |
| predictors (local checkpoint, remote System One endpoint, Jev) | `kev/predictors.py` |
| rows from predictions, `summarize`, `evaluate_records` | `kev/benchmark.py` |
| research gates and thresholds | `kev.experiment.GATES`, `gate_report` |

When a helper becomes canonical, add it here and add a rule to `tests/test_conventions.py`.

## Repo-specific things reviewers have caught

- A field dropped from an import while moving a function (`HELD_OUT_KEYS`), an env-var side channel into the loader,
  a temperature applied twice on the locked-test read, a `max_branch=960` admission literal disagreeing with the
  manifest it wrote. Look for exactly these shapes: silent partial moves, hidden channels, doubled application of a
  correction, and constants that exist twice.
- Version-named modules (`study_v3`, `transfer_v9`) are research scripts; core code (`train`, `experiment`, `serve`)
  must not import from them.
- Frozen suites under `evals/` and `runs/leaderboard.*` are artifacts: a refactor must not regenerate or commit them.
