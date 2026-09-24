# Kev — small Jev-style decision models

Causal LM + LoRA run prefill-only with a block-causal mask (shared state prefix, isolated question branches) and a
pointer readout over option boundary tokens, trained with log loss on public datasets plus generated policy and rule
records. No text generation. The released family is Qwen3.5 (0.8B / 4B / 9B) plus Kev-27B on the post-trained Qwen3.8-27B; `kev-0.5b` (Qwen2.5-0.5B, the prototype
this started as) and the Qwen3 generation are superseded but still published.
See README.md (deep dive), PLAN.md (where the research stands, what we have learned, the standing rules and what is next;
the full dated record of rounds 4-18 and everything before is frozen at git tag `research-archive-2026-09-24`),
docs/autoresearch.md (the operating program for an unattended research session) and docs/model-cards/ (one card per
checkpoint: recipe + metrics). README follows the Vercel Labs house style (tagline, for-the-badge badges, Highlights,
Title Case sections, API tables, Authors + License); model cards are formal.

## Commands
- Env: `uv sync` (add `--extra serve` for FastAPI + the TypeSafe SDK; on Apple Silicon it also pulls `mlx-lm` for the MLX backend, `--extra mlx` alone for library use). `kev` is a real package (setuptools, installed
  editable by `uv sync` since #14), Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`; 3.14 has no wheels for the pinned torch, `.python-version` selects 3.13,
  `uv sync --extra serve --python 3.13` overrides explicitly), `transformers>=5.17,<6`, `peft>=0.21`, `torch>=2.6,<2.9`. The dev
  group carries pytest, matplotlib and `modal==1.5.5`.
- Train: `uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-4B-Base --base_revision <sha> --epochs 2
  --lr 5e-5 --batch 4 --accum 2 --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --device cuda --out runs/kev-4b`
  (the released 4B/9B recipe; 0.8B uses `--lr 1e-4 --batch 8`). Records come from `--suite` (its training partition),
  `--data` (your own JSONL, optionally `+ --suite --replay N`), or built on the fly from the public sources
  (`--n_per_source`, the smoke path). `--help` lists everything; the knobs that matter:
  - losses: `--perm_kl/--perm_frac` (permutation KL), `--ord_w` (ranked probability score for Score),
    `--anchor/--anchor_w/--anchor_sources` (KL toward the frozen base's zero-shot answers, from `kev.anchors`),
    `--label_smoothing/--brier_w/--focal_gamma` (the round-3 calibration screen; all default 0 and none is in a release).
  - augmentation / mix: `--p_none`, `--p_none_distract`, `--p_distract`, `--p_none_pair`, `--synthetic_repeat`,
    `--public_frac`, `--train_sources`, `--holdout`.
  - architecture / precision: `--lora`, `--lora_targets all|dense|attn|qv` (`dense` freezes the DeltaNet projections on
    hybrid bases), `--head_dim`, `--option_isolation`, `--special_embeddings`, `--dtype` (autocast) vs `--weights_dtype`
    (frozen backbone; bf16 is required by the fused MoE experts of 35B-A3B).
  - Only one training process at a time: two on MPS slow each other ~10x.
- Smoke: `uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke` (~1 min).
- Benchmark (the eval path for everything current): `uv run python -m kev.benchmark --run <run dir | Hub id[@rev]>
  --suite evals/v4/transfer-v4 --out runs/<name>`; `--remote <url>` scores any System One endpoint (`--remote-concurrency N` keeps N requests in flight; rows are identical), `--data x.jsonl` your
  own labelled rows, `--date_facts` the opt-in preprocessing, `--allow-test` is the only way to read a locked test.
  Writes `rows.json` (per question, with logits) + `report.json` (accuracy, ECE/Brier/NLL, selective coverage and AURC,
  permutation, isolation). `kev.calibrate --rows <rows.json>` reports what one temperature fitted on those rows would do (raw / shipped / workload in-sample / workload group-disjoint OOF, paired bootstrap vs shipped; report only, writes `calibration.json` next to the rows); external-suite `rows.json` are committed for this. `kev.evaluate` is the legacy prototype eval (`runs/kev`, `eval.json`) and is not used for
  releases. `kev.compare --candidate <dir> --reference <dir>` pairs two result dirs (record-clustered bootstrap).
  `kev.jev --suite ... --out ...` scores Jev through Vercel AI Gateway (AI SDK `experimental_evaluate`, node worker in
  `playground/scripts/jev-evaluate.mjs`; needs `AI_GATEWAY_API_KEY` or `--provision-scope`; budget-capped).
- Studies: `kev.experiment --plan experiments/*.json --suite <suite> --out runs/<study>` runs config-only trials
  (allowlist + ranges in `experiment.py: DEFAULTS/CHOICES/validated_trial`, provenance, coverage/isolation gates,
  `results.jsonl` ledger, `--transfer <suite>` for an OOD read per trial, `--aggregate` to rank an existing directory,
  `--resume` for interrupted trials, `--wait-pid` to queue behind a training job).
- Rounds (every registered experiment since round 5): one spec per round, `experiments/rounds/r<N>.json`, committed before any
  training or read: studies (plan, GPU, timeout, budget), arms (`<size>-<label>`: trial + parent), parents (trial + where each
  of its reads lives), read tags -> suites (`--allow-test` for test panels, `locked_test` for the locked read), the rule
  (panels of reads, bootstrapped metrics, criteria on paired bounds, rank) and confirmation stages. `kev/rounds.py` is the
  one engine: `uv run python -m kev.rounds {validate,launch,watch,launch-reads,readout,confirm} <spec>`; `watch` polls the
  spawned trials (state in `runs/<study>.watch.json`, resumable; DNS/connection errors retried, a trial's own exception is a
  failure), pulls each finished trial's study (one pull per study at a time, `modal_app.pull_lock`) and launches its reads once (per-arm lock; the
  launch intent is written first to `runs/r<N>-reads-<arm>.json`, and an arm launched within its reads' timeout is not relaunched; a
  finished call that maps to no arm is logged and makes `watch` exit non-zero),
  60 s apart, then writes `runs/r<N>-readout/round<N>.json`; `confirm <spec> --stage <s>` writes `runs/r<N>-verdict/<size>-<s>.json`.
  Every side is served at the temperature fitted on its own development rows; deltas are `kev.rounds.paired` (2,000 resamples,
  seed 0, micro). Rounds 5-18 are recorded specs (`"archive": "research-archive-2026-09-24"`): their plans, reads, data builders
  and per-round scripts live on that git tag, not on main; `validate` lists what this checkout lacks instead of failing, and
  `launch`/`watch`/`launch-reads` refuse a recorded round (a new round is a new spec, without `archive`, and must have its plans
  and parents' reads). `tests/test_rounds.py` reproduces the committed read-outs of rounds 5-18 and verdicts of 8/10/11/12/15
  exactly; offline (CI) it runs round 5's read-out and round 15's locked verdict, whose rows are on main; every other case
  skips unless `KEV_ROUNDS_ROOT` points at a checkout with the archived rows and outputs (the research checkout, or a worktree
  of the tag plus its gitignored trial rows). `kev.autoresearch
  session <specs> --spend-start X --spend-cap Y` runs registered rounds to their read-outs under a spend cap and never confirms;
  it also keeps `leaderboard`, `release-check`, `compare`. An unattended session follows `docs/autoresearch.md` (start, spend rule,
  registration, run, what may not be touched, resilience, reporting, gotchas). Model-card numbers: `scripts/release_numbers.py --release <name>`
  reads `experiments/releases/<name>.json`.
- Frozen suites: `evals/<version>/<suite>/{manifest.json, *.jsonl}` with partitions `train/calibration/development/test`.
  Manifests pin dataset + base revisions and the sha256 of every partition. Current: `evals/v7/decision-v7` (the release
  recipe), `evals/v8/decision-v8`, `evals/v4/transfer-v4` and `evals/v9/transfer-v9` (MMLU-Pro, buried states,
  unknowable) for OOD, `evals/round3/{decision-r3,transfer-r3}` (calibration audit; the 1,260-record final panel was
  read by the 2026-09-22 release confirmation and has been a short-state guard since round 11), `evals/smoke-v1` for tests, plus `evals/external/` (semif-v1, scienthoon-v1, ekzhang-mmlupro-v1, and SemIf's pinned third-party selections wanli-v1 + typesafe-v1 via
  `scripts/freeze_semif_external.py`; `scripts/compare_typesafe.py` reports equal-case agreement/TVD against the reference and published answers, `--tokenizer` adds accuracy by state length; Kev-9B/4B scored 2026-09-22: WANLI 0.703/0.695 vs Jev 0.758, TypeSafe 0.809/0.856 agreement on 89 answered rows vs 0.891, `runs/kev-*-{wanli,typesafe}-v1`),
  `evals/night2/` (delta training data, `scripts/build_night2_data.py`), `evals/diagnostics/` (binding-v1), `evals/hard-v1`
  (programmatically labelled skill records in seven families: long policy documents, trade-offs, probability, multi-hop,
  temporal/numeric, judging a proposed answer, missing-fact abstention; `scripts/build_hard_v1.py` + `hard_v1_{common,policy,families,numeric}.py`,
  labels from each family's solver over `_meta.facts`, templates 0-3 train / 4 development / 5 test; long_policy states reach ~5k tokens, so train with
  `--max_state` >= 5120; its 23 MB train partition is not in git and not yet in the kev-suites mirror: the builder regenerates it byte for byte, ~1 min;
  `scripts/screen_overlap.py` checks it against JevBench's public items, counts only, in `overlap.json`), `evals/devtools-v1`
  (developer-tooling decisions from six licence-checked sources, human / heuristic / by-construction labels, no LLM labels; `scripts/build_devtools_v1.py --reproduce-v1`
  rebuilds it byte for byte from cached downloads; When2Call and prompt injection are eval-only. Known defects, frozen: CodeReviewer ids came from the dataset's
  non-unique `id` field, so `codereviewer/cls-test/13657` names two development records and `codereviewer/cls-test/19245` two test records
  (paired comparisons drop both; 66 more ids repeat inside train or across train and an eval partition, so check train/eval overlap by `text_sha256`),
  and its CodeReviewer `text_sha256` hashes the hunk without `lines_before_hunk`. Without `--reproduce-v1` the builder makes line-based unique ids, keys the
  whole state and admits commitpackft records after their message question is added; see its docstring), and the real-document suites
  `evals/documents-v1` (CFPB complaint narratives, product + issue Choice questions; its 23 MB train partition, Kev-4B's round-8 delta data, is not in git and not yet in the kev-suites mirror)
  and `evals/documents-v2` (held-out test only, private mirror, manifest only): `scripts/build_documents_v{1,2}.py` -> `label_documents_v1.py`
  (AI Gateway teachers/judges, spend ledger) -> `freeze_documents_v1.py` (no flag: report + adjudication queue; `--combine`, `--spot-check`, `--freeze ... --min-agreement 47`);
  label provenance in `runs/documents-v1-work/`.
  Partitions over ~10 MB are not in git; they are mirrored at the Hub dataset `jaredpalmer/kev-suites` (revision pinned
  in `kev/suite.py: SUITES_REVISION`) and `load_split` fetches + verifies them on first use. After freezing a new suite:
  `hf upload jaredpalmer/kev-suites evals . --type dataset --include "*.jsonl" --include "*.json"`, bump
  `SUITES_REVISION`, gitignore the large partitions. Never modify a frozen file; new data = new version.
  Held-out suites whose text must never be public (a private test set) keep only `manifest.json` in git and name their
  own mirror in it, `"mirror": {"dataset": "jaredpalmer/kev-private-evals", "revision": "<commit sha>"}` (`kev.suite.PRIVATE_DATASET`,
  a private dataset repo): freeze locally, `hf upload jaredpalmer/kev-private-evals evals . --type dataset --include "<suite path under evals/>/*.jsonl"` (e.g. `held/documents-v2/*.jsonl`),
  write the returned commit into the manifest, gitignore the partitions. `load_split` fetches and hash-checks them for accounts with
  access (`hf auth login` / `HF_TOKEN`; Modal images already carry a locally fetched copy under `evals/`) and raises a
  PermissionError naming the repo for everyone else; `tests/test_conventions.py` fails if such a suite tracks a partition.
- Modal (default for anything beyond smoke): `modal_app.py`; `uv run modal run modal_app.py::{smoke,study,pull,resume,
  locked_test,evaluate,base_probe,benchmarks,smoke_base,anchors}`; `uv run modal deploy modal_app.py` once so studies
  survive a disconnect. Image = `uv_sync` of pyproject/uv.lock (Linux torch wheel is CUDA) + `kev/` + `evals/`; Volumes
  `kev-hf-cache` (HF_HOME) and `kev-runs` (trial outputs, pulled to `runs/<study>` then ranked by
  `kev.experiment --aggregate`). `KEV_GPU` picks the GPU type (H100 default; T4 for the free tier), `KEV_APP_NAME`
  isolates a research deployment, `worker_environment` propagates app/GPU/secret settings (a dependency list that
  differs inside the container fails with "Function has N dependencies but container got M"). Legacy checkpoints: Hub id,
  or `modal volume put kev-runs runs/<run> /legacy/<run>` then `--existing /runs/legacy/<run>`. Eval on CUDA is
  fp32-exact (TF32 + fused SDPA off in `LocalPredictor`); training keeps TF32 and may use `--dtype bf16`. Probes,
  external benches and new-base fit checks lived in `modal_probe35.py` until 2026-09-21 (folded in at 90990a5); the
  how-to is the `kev-modal-study` skill.
- Figures: `uv run python scripts/plot_family.py` and `uv run python scripts/plot_tweet.py` regenerate docs/kev-family.png and docs/kev-benchmark{,-dark}.png from
  saved result files. Style lives in `scripts/chartstyle.py` (Geist type, Vercel color tokens, direct labels, no legends, one label/plot/value lane per bar set);
  new figures should import it rather than set their own rcParams. `kev.plot` (loss curves from train logs) is a debugging aid, not a README figure.
- Current family (2026-09-21, all Qwen3.5 + the dates/unknowable delta): `jaredpalmer/kev-9b` (`night2-9b-du/00-trial-0`), `jaredpalmer/kev-4b` (`r10-skills/00-trial-0` from the round-10 release: the round-8 checkpoint `r8-small/00-trial-0` (the night2 checkpoint + one epoch on `documents-v1` train) + one epoch on `hard-v1` + `devtools-v1` train; round-8 weights at tag `r8-documents-release`, night2 at `night2-du-release`;
  Qwen3 weights at tag `qwen3`), `jaredpalmer/kev-0.8b` (`r15-08b/00-trial-0` from the round-15 release: `night2-08b-du2/00-trial-0` + one epoch on `documents-v1` + `hard-v1` + `devtools-v1` train together, replay 6000; previous at tag `night2-du-release`), `jaredpalmer/kev-27b` (`r6-27b-v2/01-trial-1`, from the release study "B1 v2" in `PLAN_27b.md` at tag `research-archive-2026-09-24`; base `Qwen/Qwen3.8-27B` rev `1d4bf0f2`, post-trained, not `-Base`; bf16 backbone only (55 GB resident), so H200 or H100 80 GB, no Mac path; `KEV_GPU=H200` in the kev-deploy skill). Pre-delta v7 checkpoints at tag `v7-base` (`q35-9b/01-trial-1`, `q35-4b-s23/00-trial-0`, `q35-08b/02-trial-2`).
  Calibration is built into each checkpoint: `head.pt["temperature"]` (fitted by `scripts/calibrate_checkpoint.py` on the trial's development rows; 27B 1.38, 9B 2.30, 4B 2.41 (2.96 after the round-8 delta, 2.14 before it),
  0.8B 2.35 (2.41 before the round-15 delta)) is applied by `PointerHead` in eval mode; `KEV_TEMPERATURE=1.0` overrides to raw. The script also reports an out-of-fold grouped-CV ECE with bootstrap CIs alongside the in-sample fit (4B: 0.075 raw -> 0.020 OOF, separated) and stores it under `head.pt["temperature_fit"]["cross_validation"]`; re-run it after any new checkpoint before publishing. Opt-in: `KEV_DATE_FACTS=1` (day counts). Delta data: evals/night2/ (scripts/build_night2_data.py). Previous generation, kept for Mac latency: `kev-8b`, `kev-0.6b`, `kev-4b@qwen3` (cards `*-qwen3.md`). Qwen3.5 backbones are hybrid (Gated DeltaNet): `DecisionModel.hybrid`
  routes them through `forward_rows_batch` (one causal row per question, state repeated) and `_branch_rows_from_prefix` for serving; the packed
  block-causal mask is only valid on attention-only bases. Needs transformers>=5.17, peft>=0.21; CUDA wants `flash-linear-attention` + `triton>=3.7.1`
  (in the Modal image). MPS has no fast DeltaNet kernels, so on Apple Silicon `kev.serve` runs these checkpoints through `kev/mlx_model.py` (mlx-lm's Metal kernels; M5, 5 questions on a ~270-token state: Kev-4B 721 ms new state / 136 ms cached state vs 3302 / 847 ms for torch bf16; parity with fp32 torch on the full decision-v7 development partition: 4B max |dp| 0.025, 1 flip in 1,264 questions; 0.8B max 0.054, 4 flips (`runs/r4-mlx-parity-*`)). Plan and results: PLAN.md at tag `research-archive-2026-09-24`, History > "Qwen3.5 port".
- Delta fine-tuning: `kev.train --init_from <run dir | Hub id[@rev]>` warm-starts LoRA + head (compatibility checked before load; source hashes in
  provenance; allowlisted in `kev/experiment.py` so studies can run cheap delta trials from a released checkpoint). Use lr <= 2e-5 for deltas.
- Publish: `uv run python -m kev.publish --run runs/<run> --repo jaredpalmer/kev-<size> --card docs/model-cards/<name>.md` (needs `hf auth login`;
  `--private`, `--tag`, `--revision <branch>` for candidates). Repos are named by
  base model size (Kev-0.5B = Qwen2.5-0.5B); versions within a size are Hub tags (`hf repos tag create jaredpalmer/kev-0.5b vX.Y`).
  Collection: huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd. `--run` in serve/benchmark accepts a Hub id.
- HF Space (public demo, ZeroGPU): huggingface.co/spaces/jaredpalmer/kev. Source in `space/` (Gradio 6 `app.py`, `presets.py` mirrors the
  playground presets, `README.md` frontmatter `models:`/`datasets:` is what links the Space from the model and dataset pages). Publish with
  `scripts/publish_space.sh [repo] [message]`: it stages `space/` + `kev/{__init__,model,api,checkpoint}.py` into one `hf upload --type space` commit,
  so the vendored modules never drift from the repo (a stale vendored `model.py` is how the hugging-apps Space broke on Qwen3.5). Any
  change to `kev/model.py`, `kev/api.py` or `kev/checkpoint.py` that affects serving should be republished. ZeroGPU rules: `import spaces` first, load on CPU in
  fp32 (`PeftModel.from_pretrained(..., torch_device="cpu")`, otherwise peft picks the faked cuda device and crashes), merge, then
  `.to("cuda")` once at module scope; a restart reloads both models (~3 min). Check with `hf spaces logs jaredpalmer/kev` and the
  gradio_client `/decide` endpoint; the Space is also in the Kev collection and needs PRO to exist.
- Serve: `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009` (`--run` defaults to runs/kev and falls back to runs/smoke;
  the playground proxies :8009)
  - TypeSafe-compatible: `POST /v1/systemone`, `GET /v1/models` (model cards for `kev-latest` and `jev-latest`, plus device, dtype, temperature and prefix-cache stats), an `x-typesafe-request-id` header on every response, and bearer auth when `KEV_API_KEY` is set (unset = open server).
  - SDK: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")`
  - CUDA: bf16, fused kernels and CUDA graphs by default (`LoadOptions.fused` / `LoadOptions.cuda_graphs`, `KEV_FUSED=0` / `KEV_CUDA_GRAPHS=0` to decline). A server pass was
    kernel-launch bound (~60 ms on an H100 at any length). `kev/fused_qwen35.py` rewrites the merged Qwen3.5 layers with fla Triton kernels
    (it needs fla 0.5.2 exactly, pinned in the images, and refuses others: it patches fla's NB-keyed kernel launches; a pass continuing a
    cached DeltaNet state does not write it back). `kev/cuda_graphs.py` replays bucketed passes (state left-padded, rows right-padded, masked exactly; equal to eager up to bf16
    reassociation) and owns admission (`admits`), batches (`run`), capture policy (`capture_due`: idle, or a bucket that keeps recurring) and
    `stats()`; a failed capture leaves its bucket eager. `kev.serve.Server` runs every pass on one model thread that batches whatever is
    queued (`DecisionModel.probs_batch`, `CudaGraphs.run` over typed `Request`s; `probs_one` is the per-request rule every backend shares;
    `PrefixCache` keeps only the states that survive a batch); `/v1/systemone` is async, `Server.lock`
    excludes the model thread, `wait_idle()` waits for answers and captures, every response carries `server-timing`.
    Measure with `uv run modal run modal_app.py::serving --run <hub id> --gpu <GPU> --name <name>` (`scripts/serving_bench.py`: latency,
    parity vs fp32, throughput at 1/8/32/64 in-process clients, `--flags=--isolation` for question isolation on the served path; reports in `runs/serving-*` (graphs only) and `runs/fused-*`).
    `tests/test_model.py::test_cuda_graphs_match_eager` needs CUDA (run it on Modal). Over HTTP, Modal's `asgi_app` path caps a container at
    ~40-50 req/s; `modal.experimental.http_server` served ~99 req/s at 64 clients (Kev-4B, H100).
    Loading merges the fp32 adapter straight into bf16 weights (same bits as the old fp32 merge + cast), so Kev-9B needs ~17 GB, not 36 GB.
- Extra endpoints for the demo: `POST /v1/systemone/permute` (one Choice under n option orders), `POST /v1/systemone/separate`
  (each question alone; packed-vs-separate comparison). `/v1/systemone` also returns `latency_ms`.
- Web demo: `cd playground && npm run dev -- -p 3001` (:3000 is used by another project). Next 16 app router; `/kev/*` is
  rewritten to the FastAPI server (`KEV_API`, default http://127.0.0.1:8009); it uses only the `/v1/*` routes. Presets live in `playground/src/lib/kev.ts`.
  - `/chess` (`src/components/chess-game.tsx`, `src/lib/chess.ts`, chess.js): legal moves -> Choice options, board -> state, Score for eval;
    games in localStorage key `kev.chess.v1`.
  - React Compiler lint forbids sync setState in effects; schedule via setTimeout or move into handlers.
  - Next 16 dev only trusts `localhost`; other hostnames need `allowedDevOrigins` or the page SSRs but never hydrates
    (no console errors). `127.0.0.1` is allowed in `next.config.ts`. Verify hydration with `agent-browser` (CDP), not curl.
  - `playground/AGENTS.md` (regenerated by `next dev`) is the Next 16 rules file; read `node_modules/next/dist/docs/` before writing app code.
- Unit tests (no weights; the CI `python` job): `uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py tests/test_documents_tools.py tests/test_hard_v1.py tests/test_devtools_v1.py tests/test_rounds.py -q`.
  `tests/test_skill_scripts.py` covers the kev-finetune stdlib scripts (stdlib only, no network). Weight-backed parity tests
  (`runs/smoke-hl/00-trial-0/checkpoint` + a Qwen2.5-0.5B / Qwen3.5-0.8B-Base download, ~2.5 min, local only): `tests/test_model.py`; `tests/test_mlx.py` (Apple Silicon, ~30 s) for the MLX backend.
  `test_conventions.py` is a table of "one canonical home" rules (head.pt via `kev.checkpoint`, `KEV_*` via `LoadOptions.from_env`,
  option keys via `api.question_keys`, context via `model.fits`/`MAX_PACKED`, device via `kev.device`, ...); add a row when a new helper becomes canonical.
  It also asserts every published README/model-card number in `docs/claims.json` traces to a committed report (`scripts/verify_claims.py`).
- API tests (server must be up): `KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q`
- CI (.github/workflows/ci.yml) runs the unit job above and a `playground` job: `npm ci && npm run lint && npx next typegen && npx tsc --noEmit -p .`.

## Skills
Repo skills (`.agents/skills`, tracked in git):
- `kev-verify`: how to prove a change has no regression (unit suites, weight-backed parity, worktree parity harness against main) and ship it as a stacked, reviewed, squash-merged PR.
- `kev-pr-description`: how to write the PR title and body (the acdlite / sebmarkbage essay style, with a weak/strong pair from a real Kev PR). Read it before opening any PR.
- `kev-modal-study`: launching, monitoring and pulling Modal studies, base probes, remote benchmarks and new-base smoke checks.
- `thermonuclear-code-review`: the strict structural review (standards adapted from cursor-team-kit's thermo-nuclear review, the approval bar, and the table of canonical homes for shared rules).

Installed from other repos by `npx skills add` and pinned in `skills-lock.json` (`deslop`, `unslop` from cursor/plugins,
`grill-me` from mattpocock/skills); `.agents/skills/modal/` is gitignored and reinstalled with `uv run modal skills install`.

Published from this repo: `skills/kev-deploy` (`npx skills add jaredpalmer/kev@kev-deploy`), one self-contained Modal file (`scripts/kev_serve.py`, pinned `KEV_REF`, GPU list per released model from `runs/serving-*`: 0.8B L4, 4B L40S, 9B H100; CUDA graphs captured at start (`server.wait_idle()`); 64 concurrent inputs per container, autoscaling at 32; optional `KEV_REGION`; `KEV_FLASH=1` = Modal's experimental direct HTTP server, needs `KEV_REGION`, keeps one container up; fla pinned to 0.5.2; bearer key via a deploy-time Modal secret) that serves a released Kev checkpoint as a System One endpoint in one `modal deploy`; bump its `KEV_REF` after a serving change it should pick up. And `skills/kev-finetune` (`npx skills add jaredpalmer/kev@kev-finetune`): the user-facing fine-tuning skill (SKILL.md for agents,
README.md is the human cookbook, `references/` the long-form docs). agentskills.io format (validate with `uvx --from skills-ref agentskills validate skills/kev-finetune`).
Stdlib scripts: `extract_workload` (find Jev/TypeSafe call sites + labelled files, draft the spec), `convert_data` (CSV/JSONL -> records),
`generate_data` (OpenAI-compatible endpoint), `plan_size` (paired McNemar sizing; `--from-result` post hoc), `split_data` (`--holdout` keeps real rows out of train).
`scripts/kev_modal.py` is a self-contained Modal app (app `kev-finetune`, volumes `kev-finetune-runs` + `kev-hf-cache`) whose image clones this repo at `KEV_REF`
and pip-installs it, so it needs no local clone; bump `KEV_REF` after merging a kev/ change the skill depends on. Launch-time settings go into the image env via
`SETTINGS` (a Secret list that differs inside the container fails with "Function has N dependencies but container got M"). `train` derives every architecture flag
from the init checkpoint's `head.pt`; results are the skill's own `result.json` shape (not a research trial's); `publish` is private by default; `teardown` removes
runs / the endpoint / the volumes. Tests: `tests/test_skill_scripts.py`.

## Layout
- `kev/data.py`      dataset -> typed records (`Source`, `build`, `materialize`, `load_records`), trainable/eval-only source policy, permutation / none-of-the-above / distractor augmentation
- `kev/composition.py` generated rule structures (atoms, shapes, groups) behind the compositional training and eval records
- `kev/contrastive.py` programmatic contrastive pairs with code labels: one sentence changed flips the label
- `kev/suite.py`     frozen suites: digest/manifest/read_manifest/load_split (Hub mirror), CONTEXT + admission, `validate_training` (trainable/eval-only policy), `semantic_hash`, UTF-8 JSON helpers, freeze CLI
- `kev/study_v3.py`  builds the v3+ decision/transfer suites (public pool + generated families, grouped splits)
- `kev/transfer_v9.py` transfer-v9: transfer-v4 byte-for-byte plus MMLU-Pro, buried states and unknowable items
- `kev/model.py`     encode(), branch_mask(), `fits`/MAX_STATE/MAX_BRANCH/MAX_PACKED, PointerHead (carries the temperature), DecisionModel (packed mask, or `hybrid` row batches)
- `kev/train.py`     LoRA fine-tune: `training_requests` (suite / built / --data+--replay, context filter, policy checks, mix ablations),
                     `encode_batch` + `batch_loss` (CE, anchor KL, permutation KL, RPS, smoothing/Brier/focal), `main` orchestration. Grad accumulation over small padded batches.
- `kev/anchors.py`   frozen-base zero-shot distributions per training question, the target for `--anchor_w`
- `kev/checkpoint.py` Checkpoint (resolve run dir or Hub id, `head.pt` schema = `Meta`, load with `LoadOptions` -> torch `DecisionModel` or, with `backend="mlx"`/`"auto"`, `MLXDecisionModel`; `warm_start` for deltas)
- `kev/mlx_model.py` Apple Silicon backend: mlx-lm Qwen3.5 backbone, LoRA merged in fp32 on the CPU stream (`merge_lora`), Kev's encoder/rows and the torch PointerHead unchanged; state prefix = mlx-lm prompt cache, branches on a replicated copy
- `kev/device.py`    default_device / sync / empty_cache / allocated_bytes for cuda, mps, cpu
- `kev/metrics.py`   pure-numpy scoring of benchmark rows: ECE, Brier, NLL, selective prediction (tie-aware coverage@error, AURC), temperature fit, out-of-fold CV calibration report (`cross_validated_temperature`), paired bootstrap
- `kev/predictors.py` LocalPredictor (checkpoint), RemotePredictor (System One endpoint), JevPredictor (AI SDK worker)
- `kev/benchmark.py` rows from predictions, summarize(), evaluate_records(), CLI
- `kev/compare.py`   two saved result dirs -> paired bootstrap comparison + NLL sensitivity
- `kev/jev.py`       Jev through Vercel AI Gateway (key provisioning, budget cap) scored on a frozen suite
- `kev/experiment.py` config-only study runner: trial allowlist, gates, temperature fit, ledger, aggregate/resume
- `kev/rounds.py`    registered rounds as specs: validate, launch, watch (pull + reads), read-out, confirmation; the served-vs-served paired read
- `kev/autoresearch.py` spend-capped sessions over round specs (kev.rounds), plus the study leaderboard
- `kev/evaluate.py`  legacy prototype eval: acc/ECE, permutation stability, IIA shift, isolation probe, packed-vs-separate
- `kev/plot.py`      loss curve(s) from train logs + accuracy-vs-baselines bars from eval.json
- `kev/api.py`       TypeSafe request/response models; Noul/Choice/Score -> pointer options; `question_keys`; confidence formulas; `with_date_facts`
- `kev/serve.py`     FastAPI: /v1/systemone (+ permute, separate) and /v1/models; `Server` holds the checkpoint and the prefix cache
- `kev/publish.py`   run -> Hub repo: adapter, head.pt, tokenizer, trial result/provenance/log, the card as README.md; `--tag`, `--revision`, `--private`
- `modal_app.py`     every GPU entrypoint: trials, locked tests, probes, benches, anchors, smoke
- `scripts/`         one-off builders and read-outs: `calibrate_checkpoint.py`, `build_night2_data.py`, `build_binding_diagnostic.py`,
                     `freeze_{semif,scienthoon,calibration_audit}.py`, `{build,label,freeze}_documents_v*.py`, `calibration_audit.py`, `review_calibration_screen.py`,
                     `compare_{q35,night2}.py`, `temperature_groups.py`, `base_mmlu_probe.py`, `plot_*.py` + `chartstyle.py`, `publish_space.sh`
- `tests/test_api.py` conformance against the docs' example requests + official SDK

## Notes
- All JSON/JSONL is UTF-8 with LF endings regardless of platform locale: read/write through `kev.suite.read_json/read_jsonl/write_json/write_jsonl`
  (or pass `encoding=`), and `.gitattributes` pins `*.json`/`*.jsonl` to LF so sha256-checked partitions survive a Windows checkout (issue #12).
- `runs/` is gitignored except ledgers, reports, provenance and training logs (see .gitignore); never commit weights, prediction dumps or regenerated `runs/leaderboard.*`.
- Delimiters reuse existing Qwen special tokens (`<|fim_prefix|>` etc.) to avoid resizing embeddings;
  peft `trainable_token_indices` leaked memory on MPS.
- `output_hidden_states=True` on MPS blows memory; use the bare `.model` backbone's `last_hidden_state`.
- `PolyAI/banking77` uses a loading script (unsupported); use `legacy-datasets/banking77`.
- User text is tokenized via `model.user_tokens()`, which rewrites `<|name|>` -> `<¦name¦>` so callers cannot forge
  option/branch delimiter tokens (the fast tokenizer ignores `split_special_tokens`).
- Training data is built as TypeSafe-shaped requests and goes through `api.to_record()` (`data.materialize`),
  so train and serve text are identical.
- Serving path (`kev.checkpoint.Checkpoint.load` + `kev.serve`; `LoadOptions.from_env()` reads the `KEV_*` variables at CLI entry points only): backend `auto` (MLX for hybrid checkpoints on MPS when mlx-lm is installed and fp32 was not asked for; `KEV_BACKEND=torch|mlx` to force; library callers get torch unless they ask; `SCORING_INTERFACE` in kev/model.py names what both implementations expose), LoRA merged in fp32 then cast (`KEV_MERGE=0` to keep unmerged, torch only), bf16 by default on CUDA/MPS (`KEV_DTYPE=fp32` for the exact path; L4 numbers in README "Serving Performance"), `KEV_ATTN=sdpa` default on MPS,
  `KEV_SHAPE_BUCKET=64` on MPS, state-prefix LRU (`KEV_PREFIX_CACHE=4`; `KEV_PREFIX_MIN_TOKENS` defaults to the model's `prefix_min_tokens`: 384 for attention-only torch models, 0 for hybrid and MLX ones because their miss path would recompute the state per question). Checkpoints trained with
  `--weights_dtype bf16` always load bf16 with the adapter unmerged. Any change here must keep the parity
  tests in tests/test_model.py (merged vs unmerged, prefix vs full pass, bucket padding) and tests/test_mlx.py (MLX vs fp32 torch, prefix form vs row form, isolation; Apple Silicon only) passing; report numbers with the fp32 unmerged path.
  `scripts/mlx_parity.py --run <ckpt>` is the fuller read (60 records, latency of every path); MLX's fp32 GPU matmul is a reduced-precision fast path (~1e-3 relative on an M5), which is why the LoRA merge runs on `mx.cpu`.
- Serving context is 8,192 tokens for the state and 8,192 for a question branch (`kev.model.SERVE_MAX_*`); training used 384 / 1,024, so longer inputs are untested. No limit on questions per request: the row form runs `rows_per_pass` rows per forward pass (a 16,384-token budget counting the cached state per row), and an attention-only model switches from the packed mask to rows above `SERVE_MAX_PACKED` (`DecisionModel.rows_form`). Kev-4B on MLX, 64 questions on a 4.8k-token state: 4.3 s / 9.4 GB peak instead of 18.5 s / 24.7 GB in one pass. `n_perm` on `/permute` is 1..64.

## Calibration Research

- Metric version 2 accepts whole equal-confidence groups. `paired_bootstrap(..., aggregation="micro")` recomputes non-additive coverage/AURC for each paired record-group resample; default `macro` is equal task weight.
- `confident_error_rate` divides confident errors by all questions. `error_rate_at_0_9` divides by accepted questions. The empirical coverage-at-error envelope is not an unseen-data guarantee; freeze thresholds on a separate calibration set.
- Local benchmarks save logits and effective inference temperature. Research studies explicitly score raw logits, fit on the calibration partition for both parent and candidate, and report raw/recalibrated metrics separately. Temperature can reorder confidence across multiclass questions, even at fixed option count.
- Registered screen: `experiments/calibration-audit-protocol.json`; `scripts/review_calibration_screen.py --study <name> --out runs/<new-review>` verifies matched updates/tokens and gates against both unchanged parent and CE continuation. The screen ran and no candidate passed (round 3; PLAN.md Record, full text at tag `research-archive-2026-09-24`): the `--label_smoothing/--brier_w/--focal_gamma` losses exist but ship in nothing.
- `evals/round3/transfer-r3/test.jsonl` was frozen as a fresh final panel; the loss screen found no qualifying candidate, so its first read was the 2026-09-22 release confirmation of the soft-target Kev-9B, and since round 11 it is half of the pooled short-state guard. It is spent as confirmation. Select and record a candidate before evaluating it; do not reuse previously inspected test partitions as untouched confirmation.
- Isolate research deployments with `KEV_APP_NAME=kev-calibration-audit`. `worker_environment` propagates app/GPU/secret-name settings to prevent Modal dependency-count startup failures; secret values stay in Modal Secrets.
- Verify live spend/rates with `uv run modal billing summary --json` and `uv run modal billing rates --json`. Training admission uses the actual configured CPU and maximum host memory, not the old 2-CPU/48-GiB assumptions. Initial cancelled startup calls and successful jobs are recorded separately.

## Writing
- Use simple technical English. For README tone, use Jared's older Formik, TSDX, Razzle, and Backpack READMEs as references: explain the developer's problem, address the reader directly, and show code early. Avoid slogans, canned contrasts, and repeated claims. Keep detailed experiment history in PLAN.md and the model cards rather than repeating it in the README.
- README order is progressive: what Kev is, Highlights and the Models table (the first 30 seconds); Quick Start, Fine-Tune, Deploy and What to Expect
  (the first 5 minutes); then Playground, API, How It Works, Training, Benchmarks, Serving Performance, Limitations. The README describes the current family
  only. A release updates its Models row, the `MODELS`/`LOCKED` lists in `scripts/plot_tweet.py` and `scripts/plot_family.py` (then regenerate) and
  `docs/claims.json`; the dated release paragraph, the before/after numbers and the previous tag go in the model card and PLAN.md, not the README.
- Pull requests teach. Write the body the way Andrew Clark (`acdlite`) and Sebastian Markbåge (`sebmarkbage`) wrote theirs on facebook/react before 2023: the
  problem as it exists today first, then the mechanism as an argument with one concrete artifact per idea, then what is uncertain and what was left out. Define
  the one term the change hinges on (pointer head, temperature, flip rate, AURC, prefix cache) the first time it appears, so an engineer who is not an ML
  researcher can follow. Numbers carry their provenance (checkpoint, suite + partition, n, report path); the decision rule is written before the result. A list
  of files changed is the diff again, not a description. Length follows novelty: one sentence for a one-line fix, headings for a new loss or serving path.
  The title becomes the squash commit subject. The `kev-pr-description` skill has the reference PRs and a weak/strong pair.
