---
name: kev-verify
description: Verify a Kev change has no regression and ship it as a reviewed PR. Use when refactoring, editing kev/*.py, scripts, the Space or the playground, and when opening or merging Kev pull requests (stacked branches, squash merges).
---

# Verify and ship a Kev change

Behaviour is defined by numbers: probabilities, saved weights, frozen-suite bytes. A refactor is done when the numbers
are bit-identical to `main`, not when the tests are green. Work bottom-up: unit suites, then weight-backed parity, then
the harness below for anything that touches the model, the loader, the trainer, the data converters or the metrics.

## 1. Fast suites (also CI)

```bash
uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py -q
```

`test_conventions.py` fails when a rule that has a canonical home is re-derived elsewhere (head.pt access, KEV_* env
reads, option keys, the context literal, device selection). Do not add an allowlist entry to make it pass; call the
helper. Add a row when a new helper becomes canonical.

## 2. Weight-backed parity (local, ~3 min)

```bash
uv run --extra serve python -m pytest tests/test_model.py -q     # needs runs/smoke-hl/00-trial-0/checkpoint
```

Merged vs unmerged LoRA, prefix cache vs full pass, shape-bucket padding, row form vs packed mask, hybrid isolation on
Qwen3.5-0.8B-Base, `--init_from` end to end. The Qwen2.5 checkpoint in `runs/smoke-hl` is only there for the packed mask,
which the hybrid Qwen3.5 bases never use. Run it for any change under `kev/model.py`, `kev/checkpoint.py`, `kev/serve.py`, `kev/train.py`.

## 3. Serving

```bash
uv run --extra serve python -m kev.serve --run runs/smoke-hl/00-trial-0/checkpoint --port 8009 &
KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q
```

Space changes: `python3 -m py_compile space/app.py`; the Space vendors `kev/{model,api,checkpoint}.py` via
`scripts/publish_space.sh`, so any change to those needs a republish. Playground: `cd playground && npm run lint && npx tsc --noEmit -p .`.

### Browser end-to-end checks

- If there is no local checkpoint, `--run jaredpalmer/kev-0.6b` provides a small public CPU fallback.
  This verifies serving integration, not parity with the released Qwen3.5 family.
- Start the playground with `cd playground && npm run dev -- -p 3001`; it proxies `/kev/*` to :8009.
  Wait for the server's Uvicorn ready log before loading the page, since model metadata is fetched once on mount.
- At `/`, click **Support triage**, then **Run**: expect six answer cards covering Choice, Noul, and Score.
  The header displays the base and checkpoint run, not the API model alias. Inspect `/v1/models` separately
  when testing the model-card contract.
- The Questions textarea is `#questions`; it accepts JSON directly, so API edge cases can be tested through
  the real UI without changing TypeScript types or mocking requests. Capture the POST response as well as pixels.
- At `/chess`, use **Model vs model**, **New game**, then **Step** for a bounded one-move test.
  Expect a legal move, populated move/evaluation panels, and Black to move. Avoid **Play** for a one-request test.
- `KEV_API_KEY` is read at server startup. Restart with a throwaway local key to verify rejection without a bearer
  header and acceptance with the correct header. The playground has no key input and will show 401 in this mode.
  Restore the open server afterward. `/openapi.json` remains accessible without a key.
- If desktop tools cannot connect to a display, use real headless Chromium via an isolated Playwright environment
  when approved; save full-page screenshots and network responses. Do not substitute mocked frontend responses.
  A full-page capture can include a sticky footer over a card; also capture a scrolled viewport when needed.

#### Devin Secrets Needed

None for local testing with public checkpoints and a throwaway local API key. Private checkpoints require
`HF_TOKEN`; hosted protected endpoints require their configured API key rather than the local test value.

## 4. Parity harness against main

Run the *old* code from a worktree and the new code from the checkout on the same inputs, then compare bytes. Both halves
run on **Qwen3.5**, the architecture the family ships: the benchmark scores the released Kev-0.8B (hybrid Gated DeltaNet,
row form, prefix cache) and the trainer fine-tunes Qwen3.5-0.8B-Base. CPU with fixed seeds is deterministic on these too
(checked 2026-09-22: max |Δ| = 0.0 on the benchmark rows and on all 372 adapter tensors).

```bash
git worktree add /tmp/kev-main origin/main
OLD="env PYTHONPATH=/tmp/kev-main $PWD/.venv/bin/python"          # `import kev` resolves to the worktree; kev is not installed in the venv
SUITE="$PWD/evals/smoke-v1"                                        # absolute: the worktree process must read this checkout's files
# benchmark rows/report (model, loader, metrics, api, data), ~15 s per tree on an M-series CPU
(cd /tmp/kev-main && $OLD -m kev.benchmark --run jaredpalmer/kev-0.8b --suite "$SUITE" --out /tmp/bench-main)
uv run python -m kev.benchmark --run jaredpalmer/kev-0.8b --suite "$SUITE" --out /tmp/bench-new
# -> rows.json must be identical; report.json identical on every numeric field
# training (trainer, losses, augmentation), ~5 min per tree (reference DeltaNet kernels on CPU): same args, then compare
# head.pt["head"] tensors and adapter_model.safetensors
ARGS="--n_per_source 4 --epochs 1 --accum 2 --batch 2 --device cpu --base Qwen/Qwen3.5-0.8B-Base --lr 1e-4 --perm_kl 0.2 --perm_frac 1 --p_none_pair 0.5 --ord_w 0.3"
(cd /tmp/kev-main && OMP_NUM_THREADS=4 $OLD -m kev.train $ARGS --out /tmp/train-main); OMP_NUM_THREADS=4 uv run python -m kev.train $ARGS --out /tmp/train-new
# data converters: json.dumps(build(3, "test", 0, only=[...])) from both trees must be equal
```

Everything the worktree process opens must be an absolute path into this checkout. "max |Δ| = 0.0" is the bar; a nonzero
difference is a behaviour change to explain in the PR or fix. The packed block-causal mask exists only on attention-only
bases, so a change to that path also needs the Qwen2.5 checkpoint in `runs/smoke-hl` (`tests/test_model.py` covers it);
nothing else should be verified on Qwen2.5. Anything that depends on CUDA kernels, bf16 or the 4B / 9B sizes is verified on
Modal with the real base (`kev-modal-study`), not here.

## 5. Ship

- One branch per concern, stacked on the previous branch while it is under review. Write the body with the
  `kev-pr-description` skill (problem, mechanism, uncertainty, scope); the parity evidence from this skill goes in its
  `Test plan` as re-runnable commands and numbers, not "tests pass".
- Review every PR with the `thermonuclear-code-review` skill (a read-only subagent works well) and apply the findings
  before merging; the reviewer has caught real bugs (a dropped import, a double-applied temperature).
- Merge with `gh pr merge <n> --squash`. Because of the squash, rebase the next stacked branch with
  `git rebase --onto origin/main <merged-branch> <next-branch>` (a plain rebase replays the already-merged commits and conflicts).
- Never commit regenerated `runs/leaderboard.*` or frozen `evals/` files as part of a refactor.
