# Kev — prototype of a Jev-style decision model

Causal LM (Qwen2.5-0.5B + LoRA) run prefill-only with a block-causal mask (shared state prefix,
isolated question branches) and a pointer readout over option boundary tokens, trained with log loss
on converted public datasets (Banking77, BoolQ, AG News, MNLI, SST-5, Yelp). No text generation.
See README.md (deep dive) and docs/model-cards/ (one card per checkpoint: recipe + metrics; kev-0.5b.md is the superseded prototype). README follows the Vercel Labs house style (tagline, for-the-badge badges, Highlights, Title Case sections, API tables, Authors + License); MODEL_CARD.md is formal.

## Commands
- Env: `uv sync` (torch MPS, transformers, peft, datasets)
- Train: `uv run python -m kev.train --n_per_source 1500 --epochs 2 --out runs/kev` (~1h45m on M5 32GB)
  - `--holdout mnli,sst5` excludes sources (out-of-source eval); `--perm_kl/--perm_frac` permutation-consistency KL;
    `--ord_w` ordinal term for Score. Only one training process at a time: two on MPS slow each other ~10x.
  - `--state_mode frozen [--state_cache 1 --cache_dtype bf16 --cache_max_gb G --cache_device cpu|device|disk --cache_dir DIR]`: state through the base
    weights once (K/V kept across epochs/variants; ~230 KB fp32 per state token, ~61 GB for decision-v3 train, so the projected size is printed and
    refused above the default cap of a quarter of physical memory, or of the free disk space with `--cache_device disk`: one `.pt` per record under
    `--cache_dir`, default `<out>/state_cache`, kept after training and reusable by a later run of the same base and cache dtype), only branches
    adapted; state_mode and the cache dtype are recorded in head.pt (`Meta`) and honored by every loader: the adapter stays unmerged and the serving prefix cache runs the state through the base weights.
    `--branch_chunk N`: branches N questions at a time against the state K/V, backward per chunk (activation memory = state + one chunk; adapted
    mode: gradient-exact, the K/V gradients are pushed back through the state graph; frozen mode: against the cached base K/V, nothing to push
    back, so it is the one-pass frozen step in N-question pieces); not with `--perm_kl` or `--checkpointing`. `scripts/exp_equivalence.py` checks
    both mechanisms against the packed forward; timings land in training_metrics.json.
  - Ladder flags between adapted and frozen: `--lora_layers M` (LoRA in the top M layers only; peft `layers_to_transform`, so the checkpoint reloads
    from adapter_config.json as usual) and `--state_grad 0` (adapted mode, `--batch 1`, no `--branch_chunk`: the state runs through the adapter but
    every layer's state K/V are detached, so no gradient reaches the state; the forward is unchanged, so evaluate/benchmark need nothing). Both are
    recorded in head.pt and training_config.json.
- Eval:  `uv run python -m kev.evaluate --run runs/kev --n_per_source 150 --baseline --baseline_instruct Qwen/Qwen2.5-0.5B-Instruct`
  -> `runs/kev/eval.json` (acc/ECE/NLL per source, temperature scaling, permutation, IIA, isolation, packed-vs-separate, held-out sources)
- Smoke: `--n_per_source 40 --accum 4 --out runs/smoke` (~1 min)
- Research suite: `evals/decision-v1` (frozen, checksummed; train/calibration/development/test; manifest pins dataset + base
  revisions). `kev.suite` freezes; `kev.benchmark --run X --suite evals/decision-v1 --out runs/...` scores development;
  `--allow-test` is the only way to read the locked test. `kev.experiment --plan experiments/*.json` runs config-only trials
  (bounded allowlist, provenance, coverage/isolation gates, results.jsonl ledger, `--wait-pid` to queue behind a training job).
  `kev.jev` scores Jev via Vercel AI Gateway (AI SDK 7 `experimental_evaluate`, node worker in `playground/scripts/`;
  needs `AI_GATEWAY_API_KEY` or `--provision-scope`; budget-capped). `kev.compare` pairs two result dirs (record-clustered
  bootstrap). Historical checkpoints (`runs/kev`, `runs/kev2`) overlap the suite's training data: exploratory only.
  `--ord_w` is now the ranked probability score (proper); the old |E[level]-y| term was removed. `--perm_kl`/`--ord_w` default 0.
- Modal (default for anything beyond smoke): `modal_app.py`; `uv run modal run modal_app.py::{smoke,study,evaluate,base_probe,benchmarks,smoke_base}` (probes, external-eval benches and new-base fit checks lived in `modal_probe35.py` until 2026-09-21; see the `kev-modal-study` skill). Image = `uv_sync`
  of pyproject/uv.lock (Linux torch wheel is CUDA) + `kev/` + `evals/`; Volumes `kev-hf-cache` (HF_HOME) and `kev-runs` (trial outputs,
  pulled to runs/<study> then ranked by `kev.experiment --aggregate`). `KEV_GPU` picks the GPU type (H100 default). Legacy checkpoints:
  Hub id, or `modal volume put kev-runs runs/<run> /legacy/<run>` then `--existing /runs/legacy/<run>`. Eval on CUDA is fp32-exact
  (TF32 + fused SDPA off in `LocalPredictor`); training keeps TF32 and may use `--dtype bf16`. `--transfer <suite>` scores OOD per trial.
- Frozen suites: `evals/<version>/<suite>/{manifest.json, *.jsonl}`. Manifests pin dataset/base revisions and the sha256 of every
  partition. Partitions over ~10 MB are not in git; they are mirrored at the Hub dataset `jaredpalmer/kev-suites` (revision pinned in
  `kev/suite.py: SUITES_REVISION`) and `load_split` fetches + verifies them on first use. After freezing a new suite:
  `hf upload jaredpalmer/kev-suites evals . --type dataset --include "*.jsonl" --include "*.json"`, bump `SUITES_REVISION`, gitignore
  the large partitions. Never modify a frozen file; new data = new version.
- Figures: `uv run python scripts/plot_family.py` and `uv run python scripts/plot_tweet.py` regenerate docs/kev-family.png and docs/kev-benchmark.png from
  saved result files. Style lives in `scripts/chartstyle.py` (Geist type, Vercel color tokens, direct labels, no legends, one label/plot/value lane per bar set);
  new figures should import it rather than set their own rcParams. `kev.plot` (loss curves from train logs) is a debugging aid, not a README figure.
- Current family (2026-09-21, all Qwen3.5 + the dates/unknowable delta): `jaredpalmer/kev-9b` (`night2-9b-du/00-trial-0`), `jaredpalmer/kev-4b` (`night2-4b-du/00-trial-0`;
  Qwen3 weights at tag `qwen3`), `jaredpalmer/kev-0.8b` (`night2-08b-du2/00-trial-0`). Pre-delta v7 checkpoints at tag `v7-base` (`q35-9b/01-trial-1`, `q35-4b-s23/00-trial-0`, `q35-08b/02-trial-2`).
  Calibration is built into each checkpoint: `head.pt["temperature"]` (fitted by `scripts/calibrate_checkpoint.py` on the trial's development rows; 9B 2.30, 4B 2.14,
  0.8B 2.41) is applied by `PointerHead` in eval mode; `KEV_TEMPERATURE=1.0` overrides to raw. Re-run the script after any new checkpoint before publishing. Opt-in: `KEV_DATE_FACTS=1` (day counts). Delta data: evals/night2/ (scripts/build_night2_data.py). Previous generation, kept for Mac latency: `kev-8b`, `kev-0.6b`, `kev-4b@qwen3` (cards `*-qwen3.md`). Qwen3.5 backbones are hybrid (Gated DeltaNet): `DecisionModel.hybrid`
  routes them through `forward_rows_batch` (one causal row per question, state repeated) and `_branch_rows_from_prefix` for serving; the packed
  block-causal mask is only valid on attention-only bases. Needs transformers>=5.17, peft>=0.21; CUDA wants `flash-linear-attention` + `triton>=3.7.1`
  (in the Modal image). MPS has no fast DeltaNet kernels (Kev-4B 0.78 s vs 0.17 s for the Qwen3 one); MLX is the planned fix. Plan and results: PLAN.md, History > "Qwen3.5 port".
- Delta fine-tuning: `kev.train --init_from <run dir | Hub id[@rev]>` warm-starts LoRA + head (compatibility checked before load; source hashes in
  provenance; allowlisted in `kev/experiment.py` so studies can run cheap delta trials from a released checkpoint). Use lr <= 2e-5 for deltas.
- Publish: `uv run python -m kev.publish --run runs/<run> --repo jaredpalmer/kev-<size> --card docs/model-cards/<name>.md` (needs `hf auth login`). Repos are named by
  base model size (Kev-0.5B = Qwen2.5-0.5B); versions within a size are Hub tags (`hf repos tag create jaredpalmer/kev-0.5b vX.Y`).
  Collection: huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd. `--run` in serve/evaluate accepts a Hub id.
- HF Space (public demo, ZeroGPU): huggingface.co/spaces/jaredpalmer/kev. Source in `space/` (Gradio 6 `app.py`, `presets.py` mirrors the
  playground presets, `README.md` frontmatter `models:`/`datasets:` is what links the Space from the model and dataset pages). Publish with
  `scripts/publish_space.sh [repo] [message]`: it stages `space/` + `kev/{__init__,model,api,checkpoint}.py` into one `hf upload --type space` commit,
  so the vendored modules never drift from the repo (a stale vendored `model.py` is how the hugging-apps Space broke on Qwen3.5). Any
  change to `kev/model.py`, `kev/api.py` or `kev/checkpoint.py` that affects serving should be republished. ZeroGPU rules: `import spaces` first, load on CPU in
  fp32 (`PeftModel.from_pretrained(..., torch_device="cpu")`, otherwise peft picks the faked cuda device and crashes), merge, then
  `.to("cuda")` once at module scope; a restart reloads both models (~3 min). Check with `hf spaces logs jaredpalmer/kev` and the
  gradio_client `/decide` endpoint; the Space is also in the Kev collection and needs PRO to exist.
- Serve: `uv run --extra serve python -m kev.serve --run runs/kev --port 8008` (falls back to runs/smoke)
  - TypeSafe-compatible: `POST /v1/systemone`, `GET /v1/models` (no auth; also reports device, temperature and prefix-cache stats).
  - SDK: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`
- Extra endpoints for the demo: `POST /v1/systemone/permute` (one Choice under n option orders), `POST /v1/systemone/separate`
  (each question alone; packed-vs-separate comparison). `/v1/systemone` also returns `latency_ms`.
- Web demo: `cd playground && npm run dev -- -p 3001` (:3000 is used by another project). Next 16 app router; `/kev/*` is
  rewritten to the FastAPI server (`KEV_API`, default http://127.0.0.1:8009); it uses only the `/v1/*` routes. Presets live in `playground/src/lib/kev.ts`.
  - `/chess` (`src/components/chess-game.tsx`, `src/lib/chess.ts`, chess.js): legal moves -> Choice options, board -> state, Score for eval;
    games in localStorage key `kev.chess.v1`.
  - React Compiler lint forbids sync setState in effects; schedule via setTimeout or move into handlers.
  - Next 16 dev only trusts `localhost`; other hostnames need `allowedDevOrigins` or the page SSRs but never hydrates
    (no console errors). `127.0.0.1` is allowed in `next.config.ts`. Verify hydration with `agent-browser` (CDP), not curl.
- Unit tests (no weights, CI): `uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py -q`.
  Weight-backed parity tests (smoke checkpoint + Qwen2.5-0.5B download, ~2.5 min, local only): `tests/test_model.py`. `test_conventions.py` is a
  table of "one canonical home" rules (head.pt via `kev.checkpoint`, option keys via `api.question_keys`, context via `model.fits`/`MAX_PACKED`, ...); add a row when a new helper becomes canonical.
- API tests (server must be up): `KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q`

## Skills (.agents/skills)
- `kev-verify`: how to prove a change has no regression (unit suites, weight-backed parity, worktree parity harness against main) and ship it as a stacked, reviewed, squash-merged PR.
- `skills/kev-finetune` (published; `npx skills add jaredpalmer/kev@kev-finetune`): the user-facing fine-tuning skill (SKILL.md for agents, README.md is the
  human cookbook). agentskills.io format (validate with `uvx --from skills-ref agentskills validate skills/kev-finetune`). Stdlib scripts: `extract_workload`
  (find Jev/TypeSafe call sites + labelled files, draft the spec), `convert_data` (CSV/JSONL -> records), `generate_data` (OpenAI-compatible endpoint),
  `plan_size` (paired McNemar sizing; `--from-result` post hoc), `split_data` (`--holdout` keeps real rows out of train). `scripts/kev_modal.py` is a
  self-contained Modal app (app `kev-finetune`, volumes `kev-finetune-runs` + `kev-hf-cache`) whose image clones this repo at `KEV_REF` and pip-installs it, so
  it needs no local clone; bump `KEV_REF` after merging a kev/ change the skill depends on. Launch-time settings go into the image env via `SETTINGS` (a Secret
  list that differs inside the container fails with "Function has N dependencies but container got M"). `train` derives every architecture flag from the init
  checkpoint's `head.pt`; results are the skill's own `result.json` shape (not a research trial's); `publish` is private by default; `teardown` removes runs /
  the endpoint / the volumes. Tests for the stdlib scripts: `tests/test_skill_scripts.py`.
- `thermonuclear-code-review`: how to apply the installed `thermo-nuclear-code-quality-review` standards to this repo; its table lists the canonical home of each shared rule.
- `kev-modal-study`: launching and pulling Modal studies.

## Layout
- `kev/data.py`      dataset -> typed records, permutation / none-of-the-above / distractor augmentation
- `kev/suite.py`     frozen suites: digest/manifest/load_split (Hub mirror), CONTEXT + admission, `validate_training` (trainable/eval-only policy), `semantic_hash`, freeze CLI
- `kev/model.py`     encode(), branch_mask(), PointerHead, DecisionModel (`state_mode`: adapted = packed forward; frozen = base state K/V + adapted branches; the prefix cache honours it)
- `kev/cached_state.py` state K/V reuse: state_kv/branch_logits (frozen state mode), chunked_loss_backward / frozen_loss_backward (branch chunks: adapted, gradient-exact / frozen), StateCache
- `kev/train.py`     LoRA fine-tune: `training_requests` (suite / built / --data+--replay, context filter, policy checks, mix ablations),
                     `encode_batch` + `batch_loss` (CE, anchor KL, permutation KL), `main` orchestration. Grad accumulation over small padded batches.
- `kev/checkpoint.py` Checkpoint (resolve run dir or Hub id, `head.pt` schema = `Meta`, load with `LoadOptions`, `warm_start` for deltas)
- `kev/evaluate.py`  acc/ECE, permutation stability, IIA shift, isolation probe, packed-vs-separate (legacy prototype eval)
- `kev/device.py`    default_device / sync / empty_cache / allocated_bytes for cuda, mps, cpu
- `kev/metrics.py`   pure-numpy scoring of benchmark rows: ECE, Brier, NLL, selective prediction (coverage@error, AURC), temperature fit, paired bootstrap
- `kev/predictors.py` LocalPredictor (checkpoint), RemotePredictor (System One endpoint), JevPredictor (AI SDK worker)
- `kev/benchmark.py` rows from predictions, summarize(), evaluate_records(), CLI
- `kev/plot.py`      loss curve(s) from train logs + accuracy-vs-baselines bars from eval.json
- `kev/api.py`       TypeSafe request/response models; Noul/Choice/Score -> pointer options; confidence formulas
- `kev/serve.py`     FastAPI: /v1/systemone (+ permute, separate) and /v1/models; `Server` holds the checkpoint and the prefix cache
- `tests/test_api.py` conformance against the docs' example requests + official SDK

## Notes
- All JSON/JSONL is UTF-8 with LF endings regardless of platform locale: read/write through `kev.suite.read_json/read_jsonl/write_json/write_jsonl`
  (or pass `encoding=`), and `.gitattributes` pins `*.json`/`*.jsonl` to LF so sha256-checked partitions survive a Windows checkout (issue #12).
- Delimiters reuse existing Qwen special tokens (`<|fim_prefix|>` etc.) to avoid resizing embeddings;
  peft `trainable_token_indices` leaked memory on MPS.
- `output_hidden_states=True` on MPS blows memory; use the bare `.model` backbone's `last_hidden_state`.
- `PolyAI/banking77` uses a loading script (unsupported); use `legacy-datasets/banking77`.
- User text is tokenized via `model.user_tokens()`, which rewrites `<|name|>` -> `<¦name¦>` so callers cannot forge
  option/branch delimiter tokens (the fast tokenizer ignores `split_special_tokens`).
- Training data is built as TypeSafe-shaped requests and goes through `api.to_record()` (`data.materialize`),
  so train and serve text are identical.
- Serving path (`kev.checkpoint.Checkpoint.load` + `kev.serve`; `LoadOptions.from_env()` reads the `KEV_*` variables at CLI entry points only): LoRA merged in fp32 then cast (`KEV_MERGE=0` to keep unmerged), `KEV_ATTN=sdpa` default on MPS,
  `KEV_SHAPE_BUCKET=64` on MPS, state-prefix KV LRU (`KEV_PREFIX_CACHE=4`, `KEV_PREFIX_MIN_TOKENS=384`). Any change here must keep the parity
  tests in tests/test_model.py (merged vs unmerged, prefix vs full pass, bucket padding) passing; report numbers with the fp32 unmerged path.

## Calibration Research

- Metric version 2 accepts whole equal-confidence groups. `paired_bootstrap(..., aggregation="micro")` recomputes non-additive coverage/AURC for each paired record-group resample; default `macro` is equal task weight.
- `confident_error_rate` divides confident errors by all questions. `error_rate_at_0_9` divides by accepted questions. The empirical coverage-at-error envelope is not an unseen-data guarantee; freeze thresholds on a separate calibration set.
- Local benchmarks save logits and effective inference temperature. Research studies explicitly score raw logits, fit on the calibration partition for both parent and candidate, and report raw/recalibrated metrics separately. Temperature can reorder confidence across multiclass questions, even at fixed option count.
- Registered screen: `experiments/calibration-audit-protocol.json`; `scripts/review_calibration_screen.py --study <name> --out runs/<new-review>` verifies matched updates/tokens and gates against both unchanged parent and CE continuation.
- `evals/round3/transfer-r3/test.jsonl` is a fresh final panel, not a search set. The first loss screen found no qualifying candidate, so it remains unscored. Select and record a candidate before evaluating it; do not reuse previously inspected test partitions as untouched confirmation.
- Isolate research deployments with `KEV_APP_NAME=kev-calibration-audit`. `worker_environment` propagates app/GPU/secret-name settings to prevent Modal dependency-count startup failures; secret values stay in Modal Secrets.
- Verify live spend/rates with `uv run modal billing summary --json` and `uv run modal billing rates --json`. Training admission uses the actual configured CPU and maximum host memory, not the old 2-CPU/48-GiB assumptions. Initial cancelled startup calls and successful jobs are recorded separately.

## Writing
- Use simple technical English. For README tone, use Jared's older Formik, TSDX, Razzle, and Backpack READMEs as references: explain the developer's problem, address the reader directly, and show code early. Avoid slogans, canned contrasts, and repeated claims. Keep detailed experiment history in PLAN.md and the model cards rather than repeating it in the README.
