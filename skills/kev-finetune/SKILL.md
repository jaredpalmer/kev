---
name: kev-finetune
description: Fine-tune a Kev decision model (open Jev-style System One model) on one workload with LLM-generated synthetic data, calibrate it, score it against the released checkpoint, and serve it as a TypeSafe-compatible endpoint on Modal. Use when someone wants Jev/TypeSafe-style noul, choice or score questions answered on their own domain, wants calibrated probabilities for a classifier or triage/routing task, or asks to fine-tune, evaluate, deploy or publish Kev.
license: Apache-2.0
compatibility: Requires Python 3.10+, uv, and a Modal account (`uvx modal setup`). Training uses one H100 (about $1.50 per Kev-4B run). Optional - an OpenAI-compatible chat endpoint for data generation and a Hugging Face token for publishing.
metadata:
  author: jaredpalmer
  version: "1.0"
  repository: https://github.com/jaredpalmer/kev
---

# Fine-tune Kev on your workload

Jev (TypeSafe's System One model) answers typed questions about a text without generating tokens, but it is a fixed
hosted model: on your data it is out of distribution and its probabilities are uncalibrated. Kev is the open
reconstruction. Because you can train it, you can (1) fine-tune it on a few hundred labelled examples of your exact
questions and (2) fit its temperature on a held-out slice so the probabilities it serves mean what they say. This skill
does both on Modal, measures the gain against the released checkpoint, and deploys the result.

Everything runs through three scripts; nothing here needs a local GPU or a local clone of the Kev repo.

| Script | Runs where | What it does |
| --- | --- | --- |
| `scripts/generate_data.py` | your machine | workload spec -> labelled JSONL via any OpenAI-compatible chat model (stdlib only) |
| `scripts/split_data.py` | your machine | validates records, splits by state into train / calibration / development (stdlib only) |
| `scripts/kev_modal.py` | Modal | `validate`, `train` (delta fine-tune + calibrate + score vs baseline), `evaluate`, `compare`, `pull`, `publish`, and the `Serve` endpoint |

Run `modal` as `uvx modal ...` if it is not installed; the commands below assume the skill directory is the working directory.

## Workflow

### 1. Pin down the workload

Ask for (or infer from the codebase) the questions the deployed model will be asked. Write them as a workload spec,
`workload.json`, in the exact System One shape: a `domain` sentence, a `state` description (what one input looks like),
and `questions` (`noul` yes/no, `choice` with named options, `score` with ordered levels). Copy
`assets/workload.example.json` and edit it; `references/data-format.md` lists the rules. Keep states under ~1400
characters (384 tokens); Kev's training context drops longer records.

Then ask two things before spending money:

- **Which base?** Recommend `jaredpalmer/kev-4b` to start (about 20 minutes and $1.50 per run on an H100; the
  balance of quality and iteration speed). `jaredpalmer/kev-0.8b` is for fast loops (~10 min), `jaredpalmer/kev-9b`
  for the final model (~40 min). All three carry the same recipe, so a run transfers to a larger base unchanged.
- **How to get the labelled data?** Three options, in order of preference:
  1. *Existing labels* (tickets with their routing, logs with outcomes): convert them to the JSONL format yourself.
  2. *An LLM writes them*: `scripts/generate_data.py` against OpenAI, Vercel AI Gateway, Ollama or any compatible
     endpoint (`KEV_GEN_API_KEY`, `KEV_GEN_BASE_URL`, `--model`). Balanced per option, deduplicated, resumable.
  3. *You write them*: run `scripts/generate_data.py workload.json --dry-run` to get the batch prompt, answer it in
     batches of 20, append the records to a JSONL file. Slower; fine for a first 100.

Aim for 300-600 records for a first run, 1000+ for a final model. Real labelled examples, even 50, are worth more than
synthetic ones: pass them with `--examples real.jsonl` so the generator matches their style.

### 2. Generate and split

```bash
export KEV_GEN_API_KEY=...                         # or OPENAI_API_KEY / AI_GATEWAY_API_KEY
python3 scripts/generate_data.py workload.json --n 400 --out data/support.jsonl --model gpt-4.1-mini
python3 scripts/split_data.py data/support.jsonl --out data/support
```

Read the split output. Fix the spec and regenerate if a label is under 5% or missing, if many states are flagged as
long, or if the sampled records read alike. `split_data.py` writes `train.jsonl` (70%), `calibration.jsonl` (15%, the
temperature fit) and `development.jsonl` (15%, scoring). Records sharing a state never cross partitions.

Optional pre-flight on CPU (no GPU cost): `modal run scripts/kev_modal.py::validate --data data/support --init-from jaredpalmer/kev-4b`.

### 3. Train, calibrate, score

```bash
modal run scripts/kev_modal.py::train --data data/support --name support-v1 --init-from jaredpalmer/kev-4b
```

One container does, in order: warm-start the released LoRA and pointer head (`kev.train --init_from`), train one epoch
on your `train.jsonl` mixed with 2000 replay records from the public recipe (so the model keeps its general skill),
fit a temperature on `calibration.jsonl` and write it into the checkpoint (`head.pt["temperature"]`, applied by every
loader), score `development.jsonl`, score the *baseline* (the checkpoint you started from, also temperature-fitted on
your calibration slice) on the same records, run a paired bootstrap, and check both models on a sample of the public
`decision-v7` development partition for forgetting. Reports land in `runs/<name>/` locally:

- `result.json`: raw and calibrated metrics for both models, per question, bootstrap CIs, regression check.
- `errors.jsonl`: every wrong development answer, most confident first, with the state text.
- `train.log`, `development/report.json`, `baseline/development/report.json`.

Names are immutable (a failed or repeated run needs `support-v2`). For runs longer than your shell tolerates, use
`modal run --detach ... ::train` and later `modal run scripts/kev_modal.py::pull --name support-v1`. The GPU is
`--gpu` / `KEV_GPU` (default H100); `--timeout` bounds the cost, which is printed before anything starts.

### 4. Read the numbers and hill-climb

Report the table exactly as printed: baseline vs fine-tuned, raw vs calibrated. Then decide using
`references/hill-climbing.md`. The short version:

- `accuracy` and `brier` are the headline; the bootstrap CI says whether the gain is real on this development set.
- `ece` and `confident errors (p>=0.9, wrong)` are the calibration story. Calibrated ECE should be well under raw ECE
  for *both* models; if the fine-tuned model's confident errors are not lower than the baseline's, do not deploy it.
- `coverage at 5% error` is how much of the workload can be automated at a 5% error budget: the business number.
- Regression: fine-tuned accuracy on the public records should be within ~2 points of the baseline; a larger drop
  means the delta forgot (lower `--lr`, keep `--replay 2000`, fewer epochs).

The main lever is data. Open `errors.jsonl`, find the pattern (one option under-labelled, an ambiguous rule, states
too short to decide), fix the spec's `guidance`, generate targeted records, re-split, retrain as `-v2`, and run
`modal run scripts/kev_modal.py::compare --a support-v2 --b support-v1`. Move to `kev-9b` only when the data stops
improving the 4B.

### 5. Deploy

```bash
KEV_SERVE_RUN=support-v1 modal deploy scripts/kev_modal.py
```

Prints a URL. It is a TypeSafe System One endpoint (`POST /v1/systemone`, `GET /v1/models`) serving calibrated
probabilities in bf16 on an L4 (0.8B and 4B; set `KEV_SERVE_GPU=A100-80GB` for 9B) that scales to zero after five
idle minutes. Point the TypeSafe SDK at it (`TypeSafeClient(api_key=..., base_url=URL, model="kev-latest")`) or curl
it. For bearer auth, create a Modal secret holding `KEV_SERVE_API_KEY` and deploy with `KEV_SERVE_SECRET=<secret name>`.
`references/deploy.md` has the request shape, local serving on a Mac (`::pull --checkpoint`), and Hub publishing
(`::publish`, needs `KEV_HF_SECRET`).

## Gotchas

- `--init-from` must be a Kev checkpoint (Hub id or a run name on the volume); the trainer derives base, LoRA rank and
  head size from it. Do not pass `--base`.
- Labels are the option *name* for choice, `true`/`false` for noul, the level *index* (from 0) for score. A record with
  an unlabelled question is rejected; `split_data.py` tells you the line.
- Never fit the temperature on `development.jsonl` and never tune on it by hand more than a few rounds: it stops being
  a held-out set. Hold back a final untouched file for the last check if the decision matters.
- Calibration is per checkpoint. The baseline's carried temperature was fitted on public data; the skill refits it
  on your calibration slice for the comparison, so "baseline calibrated" is the fair zero-shot number.
- A `modal run` that dies with a network error may already have started the container: `modal app list` /
  `modal app logs kev-finetune` before relaunching, and relaunch under a new name.
- Qwen3.5 backbones need the DeltaNet kernels (`flash-linear-attention`, in the image). Serving them on a Mac is slow
  (no MPS kernels); use the Modal endpoint or pull to a CUDA box.
- The image clones the Kev repo at the commit in `KEV_REF`. Override it to test a branch; the training and scoring
  code, and the volume outputs, are versioned by that commit (`result.json["kev_ref"]`).
