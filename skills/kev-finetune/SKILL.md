---
name: kev-finetune
description: Fine-tune a Kev decision model (open Jev-style System One model) on a user's own workload and serve it with calibrated probabilities. Interviews the user, finds existing Jev/TypeSafe questions or labelled data in their codebase, generates enough synthetic training data to measure a gain, trains and calibrates on Modal, scores against the released checkpoint, deploys a TypeSafe-compatible endpoint, and tears everything down. Use when someone wants noul/choice/score questions answered on their own domain (routing, triage, moderation, classification, scoring), wants calibrated confidence for automation thresholds, mentions Jev, TypeSafe, System One or Kev, or asks to fine-tune, evaluate, deploy or clean up Kev.
license: Apache-2.0
compatibility: Requires Python 3.10+, uv, and a Modal account (`uvx modal setup`). Training uses one H100 (about $1-3 per run). Optional - an OpenAI-compatible chat endpoint for data generation, a Hugging Face token for private publishing.
metadata:
  author: jaredpalmer
  version: "1.1"
  repository: https://github.com/jaredpalmer/kev
---

# Fine-tune Kev on the user's workload

Jev (TypeSafe's System One model) answers typed questions about a text without generating tokens, but it is a fixed
hosted model: on the user's data it is out of distribution and its probabilities cannot be recalibrated. Kev is the open
reconstruction; because it can be trained, you can fine-tune it on a few hundred to a few thousand labelled examples of
the user's exact questions and fit its temperature on a held-out slice, so the probabilities it serves are calibrated for
*their* data. This skill takes a user from "I have this decision to automate" to a deployed, measured endpoint, then
cleans up. Nothing needs a local GPU or a clone of the Kev repo.

Scripts (`scripts/`) are short, standard-library Python plus one Modal app. If the user's case does not fit them,
read the script and change it; they are meant to be edited, not worked around.

| Script | Purpose |
| --- | --- |
| `extract_workload.py` | scan a codebase for Jev / TypeSafe / System One calls and labelled data; draft `workload.json` |
| `convert_data.py` | CSV / JSONL of existing labelled examples -> Kev records (column mapping) |
| `generate_data.py` | workload spec -> balanced labelled records via any OpenAI-compatible model; `--dry-run` prints the prompt |
| `plan_size.py` | how many records make the fine-tune vs baseline comparison statistically meaningful |
| `split_data.py` | validate records, split by state into train / calibration / development |
| `kev_modal.py` | Modal: `validate`, `train`, `evaluate`, `compare`, `pull`, `publish`, `teardown`, and the `Serve` endpoint |

Run `modal` as `uvx modal ...` if it is not installed. Commands below run from the skill directory.

## Phase 0: interview

Do not generate anything before you can answer these. Ask what you cannot infer; keep it to one round if possible.

1. **The decision.** What is being decided, about what input, and what happens with the answer (route, escalate, score,
   block)? One sentence each. If they mention automating a share of the volume, note the error rate they can tolerate:
   that becomes the coverage-at-error target.
2. **Where it lives today.** Run `python3 scripts/extract_workload.py <their repo> --out workload.json`. It lists
   files that call Jev / TypeSafe / `/v1/systemone` (Python SDK `Noul/Choice/Score`, AI SDK `experimental_evaluate`,
   raw request bodies), extracts question literals with `_source` file:line, and lists CSV/JSON/JSONL files that look
   labelled. Read the sources it points at and fix every `_todo`. If there is no such code, write the questions with the
   user in the System One shape (`assets/workload.example.json`, rules in `references/data-format.md`).
3. **Labelled data they already have.** Tickets with their routing, logs with outcomes, a spreadsheet of judgments.
   Even 50 real rows matter: they become the development set or the style reference for generation. If yes, plan to
   use `convert_data.py`.
4. **Data source for the rest.** Offer, in this order: (a) convert existing labels; (b) an LLM writes records from the
   spec (`generate_data.py`; ask which endpoint and key they want to use: OpenAI, Vercel AI Gateway, Ollama, or any
   OpenAI-compatible URL); (c) you write records yourself from the `--dry-run` prompt in batches of 20 (fine for the
   first 100, slow beyond).
5. **Base size.** Recommend `jaredpalmer/kev-4b` (~12 min and ~$1 per run for 400 records, ~15 min for 1000).
   `kev-0.8b` for fast loops, `kev-9b` for the final model. Same recipe on all three; a run transfers unchanged.
6. **Deployment and money.** Modal account ready (`modal setup`)? Budget: each `train` prints a cost bound before it
   starts. Where will the model be called from (so you can wire the endpoint in at the end)? Should the checkpoint stay
   on the Modal volume (default), be pulled locally, or be published to a *private* Hub repo?

Record the answers in `workload.json` (`domain`, `state`, `state_example` if inputs are objects, `questions`,
`guidance` with the labelling rules, `variety`). The questions must be the exact instructions and option names
production will send: the fine-tune binds those strings.

## Phase 1: data

```bash
python3 scripts/plan_size.py workload.json --baseline-acc 0.75 --min-gain 0.05        # how many records
python3 scripts/convert_data.py workload.json their.csv --state body --label team=dept --out data/x.real.jsonl   # if they have labels
export KEV_GEN_API_KEY=...   # or OPENAI_API_KEY / AI_GATEWAY_API_KEY; KEV_GEN_BASE_URL for non-OpenAI endpoints
python3 scripts/generate_data.py workload.json --n <plan total> --out data/x.jsonl --model gpt-4.1-mini --examples data/x.real.jsonl
python3 scripts/split_data.py data/x.jsonl --out data/x
```

`plan_size.py` turns "detect a +5 point accuracy gain at 80% power" into a record count (typically ~1000 for three
questions per record; ~$0.25 of gpt-4.1-mini). Do not settle for 300 records unless the user only wants a smoke test:
on the example workload 400 records gave +0.6 points with a ±6 point CI (nothing), 1050 records gave +5.9 points with a
CI of [+2.3, +9.7]. If the baseline accuracy is unknown, measure it first (`evaluate --run jaredpalmer/kev-4b` on a
small split) and re-plan.

Read the `split_data.py` report. Fix the spec and regenerate when a label is under 5% or missing, states are flagged
long, or samples read alike. Real labelled rows: keep them as the development/calibration side when possible
(split the real file separately and use its `development.jsonl`), since the number that matters is performance on
real inputs. `references/data-generation.md` covers all four sources, object-shaped states, soft labels, and what to
change in the scripts for unusual shapes.

Optional CPU pre-flight: `modal run scripts/kev_modal.py::validate --data data/x --init-from jaredpalmer/kev-4b`.

## Phase 2: train, calibrate, score

```bash
modal run scripts/kev_modal.py::train --data data/x --name x-v1 --init-from jaredpalmer/kev-4b
```

One container: warm-start the released LoRA and pointer head, one epoch on `train.jsonl` mixed with 2000 public replay
records (keeps general skill), fit a temperature on `calibration.jsonl` and write it into the checkpoint, score
`development.jsonl` for the fine-tuned model *and* the baseline (also temperature-fitted on the user's calibration
slice), paired bootstrap, forgetting check on public data. Reports land in `runs/x-v1/`: `result.json`,
`errors.jsonl` (wrong answers with the state text, most confident first), `train.log`.

Names are immutable: a retry needs `x-v2`. Long runs: `modal run --detach ... ::train`, later
`modal run scripts/kev_modal.py::pull --name x-v1`. `--gpu` / `KEV_GPU` picks the GPU, `--timeout` bounds the cost.

## Phase 3: read, decide, iterate

Show the printed table as is (baseline vs fine-tuned, raw vs calibrated). Then use `references/hill-climbing.md`:

- `accuracy`, `brier` are the headline; the bootstrap CI says whether the gain is real on this development set.
  `python3 scripts/plan_size.py --from-result runs/x-v1/result.json` says how much more data would make it so.
- `ece` and `confident errors` are the calibration story: calibrated must beat raw for both models, and the fine-tuned
  model's confident errors must not exceed the baseline's. Otherwise do not deploy it.
- `coverage at 5% error` (and `selective` cutoffs in `result.json`) is the business number: the share of decisions that
  can be automated at that error budget, and the confidence threshold to use.
- Regression on public records within ~2 points; more means the delta forgot (lower `--lr`, keep `--replay`).

The lever is data: read `errors.jsonl`, fix `guidance`, generate targeted records, re-split, retrain as `x-v2`,
`modal run scripts/kev_modal.py::compare --a x-v2 --b x-v1`. Go to `kev-9b` only when the data stops moving the 4B.

## Phase 4: deploy and wire in

```bash
modal secret create kev-serve-key KEV_API_KEY=$(openssl rand -hex 24)      # recommended
KEV_SERVE_SECRET=kev-serve-key KEV_SERVE_RUN=x-v1 modal deploy scripts/kev_modal.py
```

Prints the URL of a TypeSafe System One endpoint (`POST /v1/systemone`, `GET /v1/models`) serving calibrated
probabilities in bf16 on an L4 (0.8B/4B; `KEV_SERVE_GPU=A100-80GB` for 9B), scaling to zero after 5 idle minutes.
Then change the user's existing client: TypeSafe SDK -> `base_url=URL, api_key=<key>, model="kev-latest"`; AI SDK or
raw HTTP -> post the same body to `URL/v1/systemone` with `Authorization: Bearer <key>`. Instructions and option names
must match training. Confirm with `evaluate --remote URL` (needs `KEV_REMOTE_API_KEY`). Details, local serving
(`pull --checkpoint` + `kev.serve`), thresholds and optional private Hub publishing: `references/deploy.md`.

## Phase 5: tear down

```bash
modal run scripts/kev_modal.py::teardown --run x-v1 --yes            # one run's weights, reports and data
modal run scripts/kev_modal.py::teardown --endpoint                  # stop the deployed endpoint, keep the runs
modal run scripts/kev_modal.py::teardown --everything --yes [--cache] # stop the app, delete the runs volume (and the base-weight cache)
```

Tell the user what stays: Modal secrets they created (`modal secret delete <name>`), any Hub repo they published,
the local `runs/` and `data/` directories. An idle deployed endpoint costs nothing; a stopped one is recreated by
`modal deploy`.

## Gotchas

- `--init-from` must be a Kev checkpoint (Hub id or a run name on the volume); base, LoRA rank and head size are read
  from it. Do not pass `--base`.
- Labels: option *name* for choice, `true`/`false` for noul, level *index* from 0 for score. `split_data.py` names the
  offending line. `convert_data.py --score-offset 1` for 1..N ratings, `--map` for renamed categories.
- Never fit the temperature on `development.jsonl`, and stop tuning against it after a few rounds. For a decision
  that matters, hold back an untouched file (ideally real data) for one final `evaluate`.
- The baseline's carried temperature was fitted on public data; the skill refits it on the user's calibration slice so
  "baseline calibrated" is the fair zero-shot number. Calibration and thresholds are per checkpoint: re-read them after
  every retrain.
- A `modal run` that dies with a network error may have started the container: `modal app list` before relaunching,
  and relaunch under a new name.
- Qwen3.5 backbones need the DeltaNet kernels (in the image). On a Mac they are slow; serve from Modal or a CUDA box.
- The image clones Kev at `KEV_REF` (pinned commit). Override it to test a branch; `result.json["kev_ref"]` records it.
