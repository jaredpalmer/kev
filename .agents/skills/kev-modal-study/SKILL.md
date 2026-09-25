---
name: kev-modal-study
description: Launch, monitor and pull Kev training studies, untrained-base probes, remote benchmarks and new-base smoke checks on Modal (modal_app.py). Use when running trials, delta fine-tunes, base probes, external evals or fit checks for the Kev repo.
---

# Kev on Modal — study workflow

All GPU work in this repo goes through `modal_app.py`. Never train large models locally (a 32 GB Mac swaps with an 8B in bf16 while Chrome is open).

## Studies (training trials): `modal_app.py`

1. **Plan file** in `experiments/<name>.json`: a list of trial dicts. Allowed keys: `kev/experiment.py::DEFAULTS`, `CHOICES`, plus `base`, `base_revision` (40-hex, required if the suite does not pin the base), `train_sources`, `anchor*`, `init_from` (Hub id[@rev] or `/runs/...` path), `data` (`evals/**/*.jsonl`), `replay` (int). Validate locally first:
   `uv run python -c "from pathlib import Path; from kev.experiment import load_plan; print(len(load_plan(Path('evals/v7/decision-v7'), Path('experiments/X.json'))))"`
2. **Deploy if `kev/*.py` changed** (the launcher refuses otherwise: "deployed app has different kev/*.py"): `uv run modal deploy modal_app.py`. Redeploying while trials run is safe — in-flight containers keep their image — but wait for trials that are seconds from finishing if you can.
3. **Launch** (each trial is spawned as its own call on the deployed app; survives disconnects):
   `uv run modal run modal_app.py::study --suite evals/v7/decision-v7 --plan experiments/X.json --name X --transfer evals/v4/transfer-v4 --budget 30 --timeout 5400`
   Names are immutable: a failed study needs a new name (`X2`). Timeout max 28800 (a 27B), budget max $250 per study (`modal_app.admit_study`). Bound cost is printed; H100 ≈ $3.95/h.
4. **Monitor**: `uv run modal app logs kev-research | grep -a -E "step .*/|evaluated|Error" | tail`. Per-trial status without logs:
   ```python
   import json, modal
   for name, cid in json.load(open("runs/X.spawn.json"))["calls"].items():
       fc = modal.FunctionCall.from_id(cid)
       try: print(name, fc.get(timeout=1)["clean_acc"])
       except TimeoutError: print(name, "running")
   ```
   A trial's own log: `uv run modal volume get kev-runs /X/00-trial-0/train.log /tmp/x.log --force`.
5. **Pull** when done (safe to repeat while trials are still finishing: a second pull keeps the trial directories that have a `result.json`, deletes and re-fetches the ones that do not (copies taken mid-run), prints which are still running and re-ranks): `uv run modal run modal_app.py::pull --name X` → `runs/X/<trial>/{result.json, provenance.json, checkpoint/, transfer/rows.json}`. Then `PYTHONPATH=. uv run python scripts/compare_q35.py` or a paired bootstrap (`kev.metrics.paired_bootstrap(rows_a, rows_b, metric="acc")`) against the released checkpoint's `transfer/rows.json`.
6. **Locked test** (once per candidate, selected on dev only): `uv run modal run --detach modal_app.py::locked_test --trial X/00-trial-0 --name <candidate> --decision evals/v7/decision-v7`; result at volume `/locked/<candidate>/summary.json`. Defaults are 3,600 s and 48 GB host memory; a 27B needs `--gpu H200 --timeout 14400 --memory-mb 131072` (bf16 weights are staged through host memory while loading).

Timing (H100, row-batched hybrid): 0.8B ≈ 20 min, 4B ≈ 60 min, 9B ≈ 90 min for the full v7 recipe; deltas (1 epoch over ~1k records + 2k replay) ≈ 10–20 min. Set `--timeout` with ≥ 50 % headroom; a timed-out container loses everything.

## Rounds (registered experiments): `kev.rounds`

A round is a spec, `experiments/rounds/r<N>.json` (copy the closest past round; the schema is `kev/rounds.py`'s docstring):
studies, arms (`<size>-<label>` = trial + parent), parents (trial + where each of its reads lives), read tags -> suites, the
rule (panels, criteria on paired bounds, rank) and confirmation stages. Commit it (and the PLAN registration) before training.
Leave out `archive`: that key marks a recorded round (rounds 5-18, evidence on the tag `research-archive-2026-09-24`), which the
harness reads out but never launches. The plan files and parent reads a new round names must be in this checkout.

1. `uv run python -m kev.rounds validate experiments/rounds/rN.json` (structure, suites, plans against their manifests, budget >= admission bound, parent reads present; `--partitions` also verifies the partitions).
2. `uv run modal deploy modal_app.py` if `kev/*.py` changed, then `uv run python -m kev.rounds launch experiments/rounds/rN.json` (one `::study` per study, 60 s apart, output in `runs/<study>.log`).
3. `uv run python -m kev.rounds watch experiments/rounds/rN.json` (run it under `nohup`/`caffeinate`; restartable): polls `runs/<study>.spawn.json`, retries DNS/connection errors, pulls a finished trial's study and launches that arm's reads once (one batched `::benchmarks` per arm, 60 s apart, `read_timeout` per size for a 27B), waits for the reads and writes `runs/rN-readout/roundN.json` + a table. By hand: `launch-reads <spec> [--arms a,b] [--parents] [--dry-run]`, `readout <spec>`.
4. Confirmation, once per candidate: `launch-reads <spec> --stage <stage> --arm <arm>` (test reads for the candidate and any missing parent read; the locked stage runs `::locked_test`), then `confirm <spec> --stage <stage> --arm <arm>` -> `runs/rN-verdict/<size>-<stage>.json`.

`uv run python -m kev.autoresearch session <spec> [...] --spend-start <metered $> --spend-cap <$>` chains launch + watch over several registered rounds and stops at the cap; it never confirms.

## Probes, remote benchmarks, fit checks (same file, ephemeral app: no deploy step)

These run attached (`modal run`, not `deploy`): the container mounts this checkout's `kev/`, `evals/` and `scripts/`,
results land on the `kev-runs` volume and are pulled automatically. Run with `--detach` for anything long and read the
log; all three skip names that already exist locally / on the volume.

- **Untrained-base probe** (zero-shot letter logits, same items as every README row; `scripts/base_mmlu_probe.py`):
  `KEV_GPU=H200 uv run modal run --detach modal_app.py::base_probe --bases Qwen/X-Base --revision <sha> [--suite evals/v9/transfer-v9] [--prompt semif] [--split test] [--adapter /runs/.../checkpoint --tag name]`
  Names are derived (`<base>-base[-semif][-<tag>]-<suite>[-<split>]`); output pulled to `runs/probes/<name>/report.json`. Use H200 for >= 30B bf16.
- **Benchmark any checkpoint on any suite or `--data` JSONL** (`run@suite@name[@flags]` entries, parsed from the right by `modal_app.parse_jobs`, so a pinned `repo@sha` run is safe; flags are extra `kev.benchmark` switches and start with `--`; every suite must exist in the checkout):
  `uv run modal run --detach modal_app.py::benchmarks --jobs "jaredpalmer/kev-9b@evals/external/semif-v1@kev-9b-semif,/runs/X/00-trial-0/checkpoint@evals/v9/transfer-v9@x-v9@--date_facts"`
  Output pulled to `runs/<name>/report.json`. This is how the external evals (SemIf, MMLU-Pro sample) and delta benches were scored.
  Each job gets its suite's timeout (`modal_app.READ_TIMEOUTS`: long-state panels 7,200 s, documents 5,400 s, transfer-v9 3,600 s, else 1,800 s); `--timeout N` sets one for every job (a 27B's fp32 reads run about three times longer than a 9B's).
- **Full-weight training probe** (`scripts/sft_probe.py`: records shaped like the SFT corpus, from the token shapes in
  `experiments/sft-v1-lengths.json` (`--flags "--mix all"` in the corpus's proportions, or one part: `public`, `components`,
  `synthetic`), `kev.train --full_ft 1` for `--max_steps` (at least 12: OneCycleLR's 10 % warm-up must be a step long),
  peak GPU / host memory, s/step, tokens/s, the hours and dollars of one and two epochs of that part of the corpus, the
  seconds each resume point took to write to the runs volume (`--train "... --save_every_steps N"`), optional
  bf16-vs-fp32 loader check on the checkpoint it writes to scratch disk; `--flags "--no_conv_kernel"` hides causal-conv1d
  for an A/B): `uv run modal run --detach modal_app.py::sft_probe --name <name> --gpu H200:8 --flags "--mix all" --train "--batch 8 --accum 2 --length_sort 1 --max_steps 14"`
  (`--gpu H200` runs one GPU with the masters in host memory and needs `--row_budget 8192`; without `--length_sort`
  eight ranks wait on whichever holds a long record). `--detach`: a probe outlives a dropped connection (its report is
  written to the volume either way). Report in `runs/sft-probe/<name>/report.json` (`modal volume get` it if the local
  client died).
  Full-weight studies: plans set `full_ft: 1, weights_dtype: bf16` (the trainer then shares each state across its
  questions, `shared_prefix`, and writes a resume point every `kev.experiment.RESUME_MINUTES`); `admit_study` asks for
  `kev.budget.trial_resources` (one GPU: 24 CPU, 360-400 GiB for the host-side masters; `--gpu H200:8`: 16 CPU,
  128-256 GiB), allows `--timeout` up to 86,400 s and a $1,000 budget, and gives each trial `FULL_FT_RETRIES` Modal
  retries: a timed-out trial is called again and continues from its last resume point (`kev.experiment.continue_trial`);
  an attempt that raised writes `failed.json` and is not continued. The bound counts every attempt, so an 8 x H200 day is
  `--timeout 28800` (three 8 h attempts, $939). `modal_app.py::resume --study <study> --suite <suite> --gpu H200:8` also continues unfinished
  full-weight trials by hand.
- **GPU-only tests** (they skip without CUDA): `uv run modal run modal_app.py::gpu_tests --tests "tests/test_model.py::test_shared_prefix_matches_rows" [--gpu H100]`.
  The image has causal-conv1d, and transformers then sends even CPU tensors to its CUDA kernel, so CPU variants skip there.
- **Does a new base fit?** (LoRA footprint, which modules it hits, peak GB, steady step time on two real records):
  `uv run modal run modal_app.py::smoke_base --base Qwen/X-Base --revision <sha> [--gpu H200]`
- Always give the entrypoint (`::base_probe`, `::benchmarks`, `::smoke_base`): the file has several.

## Gotchas
- "deployed app has different kev/*.py" **right after a deploy**: a warm `remote_source_hashes` container from the previous image answered the check. Stop the app's idle containers (`modal container list --json`, then `modal container stop -y <id>` for that app; they are the 1-CPU hash checks, not trials) and relaunch. A research deployment isolated with `KEV_APP_NAME=<name>` must use the same variable on `deploy` and `study`.
- Redirect `modal run ...::study` to a log file rather than filtering it through `rg`/`head`: a filter can hide the `SystemExit` that explains why nothing was spawned.
- If `study` dies locally with a transient error (e.g. `Authorization check failed`) the trials may already have been spawned on the deployed app: run `modal container list` before relaunching, and never relaunch under the same name (the trials refuse to overwrite `/runs/<name>/<trial>` and every copy fails). To kill a running trial use `FunctionCall.from_id(cid).cancel()` from the spawn.json; `modal container stop` only re-queues the input to a fresh container. Orphans without a spawn.json: `modal volume rm -r kev-runs /<name>` after they fail, then relaunch under a new name.
- Wall-clock check in the first 5 minutes: count optimizer steps/min from `modal container logs` and divide the printed denominator by it (`ep0 step N/M` — **M is the total optimizer steps over all epochs**, not per epoch); the printed `s/rec` is compute only and undercounts by 2-4× on MoE bases. Cancel and relaunch with fewer epochs if it will not fit the cap — a timed-out trial saves nothing.
- `--gpu H200` on `study` only works if the deployed app was deployed with `KEV_GPU=H200` (the GPU is fixed at deploy time); deploy H200, launch, then redeploy H100 for the small jobs.
- Symptom "config=... printed, then nothing, and `Modal Client → Modal Worker Heartbeat attempt failed`" = the container is thrashing host memory (checkpoint staging). Check `run_trial`'s `memory=` against the checkpoint size (bf16 bytes ≈ 2 × params); big bases need ≥ weights + 20 GB.
- Training progress is only visible via `modal container logs <ta-id>` (`modal container list` to find it); `modal app logs` shows the last ~50 lines across containers, and the volume's train.log is committed at the end.
- Modal rate-limits app creation: launching more than ~3 detached `modal run`s within a minute fails with "App create rate limit exceeded" (the log shows it; nothing runs). Space launches ≥ 30 s apart or batch jobs into one `benchmarks` call (`kev.rounds` does both: one call per arm, 60 s apart).
- Two pulls of the same study used to delete each other's trial directories; `pull_study` now takes a per-study lock (`runs/.pull-<study>.lock`), so a second pull waits and then refreshes.
- A failed `benchmarks`/`base_probe` job leaves its output directory on the volume; relaunch under a new name (`-2`, or `--tag`) or the next run fails with FileExistsError.
- `RuntimeError: aclose(): asynchronous generator is already running` at the end of a detached run is noise; the result line follows it.
- Report dicts must not gain top-level keys that collide with benchmark blocks (`unknowable`, `clean`, `tasks`).
- Modal's HF cache volume (`kev-hf-cache`) persists base weights; first pull of a new base adds minutes.
- Jev calls go through Vercel AI Gateway (`kev.jev`); the key is created with `vercel ai-gateway keys create` (no `--scope`) and kept in the environment only.
