# Kev

Small Jev-like decision models you can train and run yourself.

<p>
  <a href="https://github.com/jaredpalmer/kev/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/jaredpalmer/kev/ci.yml?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd"><img alt="Weights: Kev-0.8B · 4B · 9B" src="https://img.shields.io/badge/WEIGHTS-0.8B%20%C2%B7%204B%20%C2%B7%209B-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/spaces/jaredpalmer/kev"><img alt="Demo on Hugging Face Spaces" src="https://img.shields.io/badge/DEMO-HF%20Spaces-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/datasets/jaredpalmer/kev-suites"><img alt="Frozen eval suites" src="https://img.shields.io/badge/EVAL%20SUITES-frozen-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="PLAN.md"><img alt="Research log" src="https://img.shields.io/badge/RESEARCH%20LOG-PLAN.md-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

Kev is a family of small decision models built on Qwen3.5 and based on the architecture described in [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked). You can use the pretrained weights or train your own. The API matches TypeSafe's [System One](https://docs.typesafe.ai/api), so you can point their Python SDK at your local server.

## Highlights

- 0.8B, 4B, and 9B models, with training code and evaluation data.
- Yes/no (`noul`), multiple-choice (`choice`), and rating (`score`) questions in the same request.
- Questions share the input text but can't read each other.
- Runs on CUDA, ROCm, and Apple Silicon (MLX). The 4B and 9B models fit a 32 GB Mac; see [Serving Performance](#serving-performance) for what to expect.
- A web playground for trying your own inputs and checking how option order affects the answers. Or try Kev-4B and Kev-0.8B in the browser at [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev), no install needed.

![Kev playground](docs/playground.png)

## Quick Start

You'll need Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). The repo's `.python-version` makes `uv sync` use 3.13; torch has no wheels for 3.14 yet.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

This starts Kev-4B locally in bf16 (`KEV_DTYPE=fp32` for the exact path the evaluations use). The first run downloads the adapter and base model. `--run` also accepts a local checkpoint directory or a Hub revision, such as `jaredpalmer/kev-4b@qwen3` for the previous generation.

In another terminal, send it a ticket:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

Example response from Kev-4B, running in bf16 on an Apple M5:

```json
{
  "model": "kev-latest",
  "answers": {
    "department":  { "type": "choice", "choice": "returns", "confidence": 0.21,
                     "probabilities": { "returns": 0.47, "shipping": 0.28, "billing": 0.25 } },
    "escalate":    { "type": "noul", "noul": 0.93 },
    "frustration": { "type": "score", "score": 1.44, "confidence": 0.78,
                     "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                     "probabilities": { "0": 0.00, "1": 0.56, "2": 0.44 } }
  },
  "usage": { "input_tokens": 101, "output_tokens": 161 },
  "latency_ms": 495
}
```

The ticket mentions a return, a late delivery, and a billing problem, and the department probabilities say so. That is the point of getting probabilities back instead of a single label.

### Host It on Modal

To get your own HTTPS endpoint instead of a local server, you don't need this repo:

```bash
pip install modal && modal setup
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py
KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py
```

That serves Kev-4B on an L40S at `https://<your-workspace>--kev-api.modal.run`, with the same API as above behind `Authorization: Bearer <key>`, in tens of milliseconds of model time per request (see [Serving Performance](#serving-performance)). It scales to zero when idle. `KEV_MODEL=jaredpalmer/kev-9b` picks another model, and the GPU follows. With a coding agent, `npx skills add jaredpalmer/kev@kev-deploy` does the same and wires the URL into your code; see [skills/kev-deploy](skills/kev-deploy/).

### Python

The TypeSafe SDK is included in `uv sync --extra serve`:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8009",
    model="kev-latest",
)
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    },
)
print(response.nouls["billing"].noul)
print(response.choices["tone"].choice)
print(response.scores["urgency"].score)
```

### Playground

With the server still running, open another terminal. You'll need Node 20.9+:

```bash
cd playground
npm install
npm run dev -- -p 3001
```

Open [localhost:3001](http://localhost:3001), load a preset, and edit the text and questions. Press `⌘↵` to run it. "Packed vs separate" compares asking all questions at once with asking them one at a time. "Permute" runs a Choice question with six option orders. There are also presets for testing question isolation and fake delimiter tokens.

There's a [chess demo](http://localhost:3001/chess), too. The board is the input, legal moves are Choice options, and a Score question rates the position. You can play against Kev or let it play itself. Games are saved in `localStorage`.

![Kev chess](docs/chess.png)

## Models

Start with Kev-4B. Use Kev-9B when accuracy and calibration matter more than memory. Use Kev-0.8B if you need the smallest model. All three are built on Qwen3.5 bases with the same base recipe; Kev-4B and Kev-0.8B also carry the later training passes described below.

| Model | Base | Accuracy: Trained Sources | Accuracy: New Sources | Brier: New Sources | Model Card |
|---|---|---|---|---|---|
| [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) | Qwen3.5-0.8B-Base | 0.833 / 0.832 | 0.657 / 0.695 | 0.489 / 0.441 | [Details](docs/model-cards/kev-0.8b.md) |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Qwen3.5-4B-Base | 0.873 / 0.865 | 0.817 / 0.838 | 0.269 / 0.242 | [Details](docs/model-cards/kev-4b.md) |
| [Kev-9B](https://huggingface.co/jaredpalmer/kev-9b) | Qwen3.5-9B-Base | 0.872 / 0.874 | **0.822 / 0.852** | **0.286 / 0.237** | [Details](docs/model-cards/kev-9b.md) |
| Jev | Hosted | 0.845 / – | 0.857 / – | 0.211 / – | – |

Each cell is **development / test**. "Trained sources" means held-out examples from the datasets used to train Kev. "New sources" means datasets and policy rule types Kev wasn't trained on. Every model was evaluated on the same development sets (`decision-v7`, `transfer-v4`) and the same test sets, which were read once per released checkpoint, after model selection. Lower Brier is better.

Kev-9B trails Jev by 3.5 points on the new-source development set (0.822 vs 0.857) and scores 0.852 on the test set, which Jev hasn't been run on. We don't know which datasets Jev was trained on, so this isn't a controlled comparison of the two architectures.

All three models were updated on 2026-09-21 with a short second training pass on generated examples: policy cases with explicit day counts, and cases whose deciding evidence was removed, trained toward a uniform answer. On the test set this moved Kev-9B from 0.837 to 0.852 (95% CI +0.8 to +2.9 points), Kev-4B from 0.832 to 0.837, and Kev-0.8B from 0.668 to 0.684. The previous weights are at revision `v7-base`. Details and costs are in the model cards and [PLAN.md](PLAN.md).

Kev-4B was updated again on 2026-09-24 with one epoch on real documents: 5,219 US consumer-finance complaint narratives (CFPB) with product and main-issue questions. On complaints it has never seen, accuracy rose from 0.804 to 0.904 on the locked test and from 0.811 to 0.891 on a private held-out set, and on the development split it scores 0.895 against Jev's 0.868. That gain is in distribution (the training data and every documents suite share one source and the same two question templates). Everything else moved within noise (as served, its new-source Brier was 0.265 / 0.233, the same as before). The weights from before that update are at revision `night2-du-release`.

Later the same day Kev-4B got a second one-epoch update, on skill data: 6,000 generated records covering long policy documents, trade-offs, probability, multi-hop reasoning, dates, judging a proposed answer and missing facts (`hard-v1`, labels computed by solvers), plus 5,320 developer-tooling decisions from four public datasets (`devtools-v1`). On the held-out test splits accuracy rose from 0.540 to 0.803 on `hard-v1` and from 0.623 to 0.756 on `devtools-v1`, and on JevBench's public hard tier, which it never saw, from 0.450 to 0.541. The `hard-v1` gain is in distribution (the test split holds out templates of the same generators), which is why the JevBench gain is the one to generalise from. The new-source test moved 0.835 → 0.838; the in-distribution test fell 0.875 → 0.865. The round-8 weights will be at revision `r8-documents-release`.

Kev-0.8B got the same real-document update (with 6,000 replayed records instead of 2,000). On complaints it has never seen, accuracy rose from 0.608 to 0.821 on the locked test and from 0.616 to 0.856 on the private held-out set; on the development split it scores 0.837, below Jev's 0.868. Its new-source test moved 0.684 → 0.695. Three earlier rounds of this update gained as much on documents but could not rule out a short-state cost of about 1 point; this one was checked with two seeds on a pooled short-state panel of 656 + 1,150 questions instead of 656, and neither seed showed a cost. The previous weights will be at revision `night2-du-release`.

Probabilities are calibrated by default. Each checkpoint stores a temperature (Kev-9B 2.30, Kev-4B 2.41, Kev-0.8B 2.52) fitted on its in-distribution development set, and the pointer head applies it when the model is loaded. It never changes an answer: on new sources Kev-9B's calibration error goes from 0.106 to 0.042 and its confident errors (wrong answers with probability ≥ 0.9) from 8.7% to 4.0%, about Jev's 3.7%, with accuracy identical. Set `KEV_TEMPERATURE=1.0` for the raw logits. `scripts/calibrate_checkpoint.py` also reports the out-of-fold (group-disjoint 5-fold) calibration error with bootstrap intervals, so the in-sample fit can be checked against held-out records. On the `night2-du` Kev-4B's development rows the calibration error is 0.075 raw and 0.020 out of fold (95% interval 0.014 to 0.041), and the interval on the difference excludes zero. The accuracy numbers in the table are the same either way; the Brier numbers are for the raw logits.

One optional setting: `KEV_DATE_FACTS=1` appends the number of days between any two absolute dates found in the state ("June 26, 2026 is 8 days before July 4, 2026"). Kev can't subtract dates reliably but it can use a stated day count: on the deadline policy questions Kev-9B goes from 0.80 to 0.90 (Jev 0.93). The table above doesn't use it.

![Accuracy by source for Kev and Jev](docs/kev-family.png)

All weights are in the [Kev collection](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd) and the [GitHub release](https://github.com/jaredpalmer/kev/releases/tag/kev-family), which includes tarballs and SHA-256 checksums.

<details>
<summary>Previous generation (Qwen3) and the prototype</summary>

The first Kev family used Qwen3 bases with the same data and settings. Those weights stay published and are the faster choice on a Mac (see [Serving Performance](#serving-performance)), but they are no longer developed.

| Model | Base | Accuracy: Trained Sources | Accuracy: New Sources | Brier: New Sources | Model Card |
|---|---|---|---|---|---|
| Kev-0.6B (Qwen3) — `jaredpalmer/kev-0.6b` | Qwen3-0.6B-Base | 0.801 / 0.808 | 0.620 / 0.642 | 0.536 / 0.483 | [Details](docs/model-cards/kev-0.6b-qwen3.md) |
| Kev-4B (Qwen3) — `jaredpalmer/kev-4b@qwen3` | Qwen3-4B-Base | 0.854 / 0.856 | 0.790 / 0.806 | 0.328 / 0.294 | [Details](docs/model-cards/kev-4b-qwen3.md) |
| Kev-8B (Qwen3) — `jaredpalmer/kev-8b` | Qwen3-8B-Base | 0.863 / 0.870 | 0.796 / 0.780 | 0.337 / 0.327 | [Details](docs/model-cards/kev-8b-qwen3.md) |

Because only the base changed, the two generations are a controlled comparison. On the development set the accuracy gain is within noise; on the test set Kev-9B is 7.3 points ahead of Kev-8B (95% CI +2.8 to +11.7) with a Brier score 0.08 lower, Kev-4B is 2.9 points ahead of its predecessor (−0.9 to +6.4), and Kev-0.8B is 4.8 points ahead of Kev-0.6B (+0.2 to +9.3). [The research log](PLAN.md#qwen35-port-2026-09-20) has the full experiment, including the criteria we set in advance and how the results measured against them.

The original [Kev-0.5B](https://huggingface.co/jaredpalmer/kev-0.5b) used Qwen2.5-0.5B and is kept for reference; see its [model card](docs/model-cards/kev-0.5b.md).

</details>

## API

### `POST /v1/systemone`

`state` is the text to evaluate. Each question has instructions and, where needed, a set of answers to choose from.

```jsonc
{
  "state": "…",                          // string | object | array — the content to evaluate
  "model": "kev-latest",
  "questions": {
    "<id>": {                            // you choose the id; the model never sees it
      "type": "noul" | "choice" | "score",
      "instructions": "…",               // string | object | array, optional
      "criteria": …                      // noul: {true?, false?}  choice: {option: description|null}  score: [level, …]
    }
  }
}
```

| Type | Criteria | Answer |
|---|---|---|
| `noul` | Optional descriptions for `true` and `false` | `noul`: probability of yes |
| `choice` | 1–255 option names, each with a description or `null` | `choice`: most likely option; `probabilities` and `confidence` |
| `score` | 1–255 descriptions, ordered from lowest to highest | `score`: mean level index, starting at 0; `legend`, `probabilities`, and `confidence` |

For Choice with `K > 1` options, confidence is `(p_max − 1/K) / (1 − 1/K)`. A single option has confidence 1. Score confidence measures how close the distribution is to its most likely level. It's an approximation of TypeSafe's formula, which isn't public. Neither field is a measured accuracy rate.

Objects and arrays are converted to labeled text. Delimiter-like strings in user input are escaped before tokenization. Invalid requests return `422`. `usage.output_tokens` counts tokens in the serialized answers, not generated tokens.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | Model cards (`name`, `description`, `release_date`) plus the loaded checkpoint's details |
| `POST` | `/v1/systemone/permute` | Run one Choice question with different option orders (`n_perm` 1 to 64, default 6) |
| `POST` | `/v1/systemone/separate` | Run each question in its own forward pass |

A request may carry any number of questions. The server runs them a token budget at a time (one maximal row of 16,384 tokens per forward pass, counting the cached document once per question in that pass), so memory does not grow with the question count and the answers do not depend on the split. Every response carries an `x-typesafe-request-id` header. The server binds to `127.0.0.1` and is open by default; set `KEV_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`, as the TypeSafe clients always send it.

## How It Works

Each checkpoint is a rank-16 LoRA adapter and a small pointer head on a Qwen base model. On an attention-only base (Qwen3), the state and questions go into one token sequence:

```text
<state> …state…
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>
<q> instructions <opt> option 1 </opt> <opt> option 2 </opt> … <decide>
```

The attention mask lets a token read the state and its own question, but not other questions or future tokens. Each question's position IDs restart just after the state. This lets the model process the state once and answer each question independently.

Qwen3.5 mixes attention layers with Gated DeltaNet layers, which are recurrent and ignore attention masks. For those models, each question runs as its own row: the state followed by that question, with the same positions as above. The rows are independent, so isolation is exact, and the server computes the state once and reuses its cache for every row. On attention-only models the two forms give identical probabilities (`tests/test_model.py`).

The pointer head scores each option's `</opt>` hidden state against the question's `<decide>` hidden state. A softmax turns those scores into probabilities. Because `<decide>` comes last, it can attend to the full option list.

Training uses cross-entropy on the correct answer. The adapter and head are trained together; the rest of the base weights stay fixed. Training examples and API requests use the same text format. No Jev outputs were used for training.

Asking questions together or separately produces probabilities within 4e-6 in the fp32 tests. This does **not** mean option order is irrelevant: options within a question can still affect one another. See [the model code](kev/model.py) and [parity tests](tests/test_model.py).

## Serving Performance

On CUDA, install `flash-linear-attention` for the Qwen3.5 models (the Modal images do this). The server runs them in bf16 and replays CUDA graphs. Model time per request (the `latency_ms` the API returns), median of 20 requests, as new state / repeated state. A new state is the normal API call, since every ticket is a new state. A repeated state is served from the prefix cache.

| Model | GPU ($/h) | 2 questions | 6 questions | 5 questions, 370-token state | 5 questions, 2,200-token state | 6 questions without graphs |
|---|---|---|---|---|---|---|
| Kev-0.8B | L4 (0.80) | 21 / 12 ms | 37 / 28 ms | 46 / 24 ms | 156 / 32 ms | 138 / 74 ms |
| Kev-4B | L40S (1.95) | 36 / 20 ms | 50 / 34 ms | 59 / 32 ms | 179 / 39 ms | 102 / 54 ms |
| Kev-4B | H100 (3.95) | 21 / 12 ms | 30 / 20 ms | 34 / 19 ms | 87 / 22 ms | 113 / 71 ms |
| Kev-4B | A100 80 GB (2.50) | 31 / 18 ms | 55 / 42 ms | 67 / 40 ms | 182 / 46 ms | 163 / 85 ms |
| Kev-4B | L4 (0.80) | 90 / 49 ms | 157 / 116 ms | 200 / 108 ms | 642 / 130 ms | 205 / 118 ms |
| Kev-9B | H100 (3.95) | 26 / 14 ms | 37 / 24 ms | 42 / 23 ms | 116 / 27 ms | 112 / 60 ms |
| Kev-9B | L40S (1.95) | 60 / 32 ms | 80 / 53 ms | 91 / 46 ms | 270 / 54 ms | 136 / 76 ms |
| Kev-9B | A100 80 GB (2.50) | 42 / 24 ms | 73 / 56 ms | 95 / 54 ms | 267 / 60 ms | 207 / 110 ms |

Without graphs, a forward pass took about 60 ms on an H100 whatever its length. A pass launches about 2,000 kernels, and Python could not issue them faster than the GPU ran them. A new state costs two passes (the state, then the question rows on its cache) and a repeated one costs one. That is why the last column depends more on the host CPU than on the model or the GPU. [`kev/cuda_graphs.py`](kev/cuda_graphs.py) pads each pass to a shape bucket (at most 1/8 larger), captures it once as a CUDA graph, and replays it with one call. The padding is masked exactly, so the answers match the eager pass up to bf16 rounding. On 200 clean decision-v7 development records (280 questions), the graphed answers differ from the fp32 evaluation path by at most 0.016 to 0.028 across these eight setups; eager bf16 differs by 0.012 to 0.039. Either way at most one of the 280 highest-probability answers changes.

A new bucket's first request runs eagerly, about as fast as before. Its graphs are then captured on a background thread, one at a time, about 0.4 s each on an H100. States over 1,024 tokens run the state pass eagerly because it is compute-bound there. `KEV_CUDA_GRAPHS=0` turns graphs off, and `/v1/models` reports how many are captured. `scripts/serving_bench.py` (`uv run modal run modal_app.py::serving --run jaredpalmer/kev-4b --gpu L40S --name <name>`) reproduces a row, and the reports behind this table are in `runs/serving-*/report.json`.

Pick the GPU by the model. An L4 is enough for Kev-0.8B. It runs out of compute on Kev-4B, where 6 questions take about the same time with graphs as without, so an L40S is the cheapest GPU that answers Kev-4B in tens of milliseconds. An H100 is fastest for both larger models. The A100 costs more than the L40S and is slower here. Loading merges the fp32 adapter into the bf16 weights with fp32 math, which gives the same bits as merging an fp32 copy and casting. Kev-9B therefore needs about 17 GB of GPU memory to load instead of 36 GB, and fits an L40S.

The server runs in bf16 by default on CUDA and Apple GPUs. That choice was measured on Kev-4B on an L4 with three questions and 20 requests per row, before CUDA graphs. Every precision returned the same probabilities to two decimals.

| Precision | 101-token request | 330-token request |
|---|---|---|
| fp32 | 209 ms | 850 ms |
| fp32 with TF32 matmuls | 113 ms | 354 ms |
| bf16 (default) | 118 ms | 189 ms |

Transformers warns at load that `causal_conv1d` is missing and falls back to PyTorch. Before graphs, installing it made no difference (114 vs 118 ms). In a graphed pass the fallback convolution is about 7% of the GPU time on an H100. The prebuilt `causal-conv1d` 1.7.0 wheel does not load against this torch build, so the images still skip it. `KEV_DTYPE=fp32` serves the exact path the evaluations use; the benchmark and research tools always score in fp32 regardless of this default.

On Apple Silicon there are no PyTorch kernels for the DeltaNet layers, so the server runs the Qwen3.5 models through [MLX](https://github.com/ml-explore/mlx-lm) instead (`uv sync --extra serve` installs it on Macs). Only the backbone changes. Kev's encoder, the pointer head and the calibration are the same code, and the probabilities match the fp32 PyTorch path to bf16 rounding. On all 1,024 clean decision-v7 development records (1,264 questions), Kev-4B's largest difference is 0.025 and the mean 0.0016, and the highest-probability answer changes on one question (none through the prefix cache); Kev-0.8B's largest is 0.054 and the mean 0.0023, with four changed answers (0.3%). Median time on an M5 (32 GB) for five questions with three options each on a ~270-token state, through the model directly:

| Model | New state | Repeated state (prefix cache) | PyTorch bf16 on MPS, new / repeated |
|---|---|---|---|
| Kev-0.8B | 149 ms | 28 ms | 1062 / 276 ms |
| Kev-4B | 721 ms | 136 ms | 3302 / 847 ms |

The server caches every state prefix for these models, so a repeated document pays only for its questions. `KEV_BACKEND=torch` restores the PyTorch path, as does `KEV_DTYPE=fp32` (asking for the exact path always means PyTorch); `/v1/models` reports which backend and dtype are serving. The previous-generation Qwen3 models (`jaredpalmer/kev-4b@qwen3`, `kev-8b`, `kev-0.6b`) still run on plain PyTorch MPS and remain a fine choice on a Mac.

`scripts/mlx_parity.py --run jaredpalmer/kev-4b` reproduces the parity and latency numbers on your machine.

For the attention-only models the server merges the LoRA weights (fp32 math, one rounding to bf16), uses SDPA attention on Apple GPUs, pads MPS inputs to 64-token buckets, and caches the state prefix for repeated requests (four states of at least 384 tokens by default). With a repeated 772-token state, Kev-4B (Qwen3) answers in 242 ms instead of 861 ms.

You can disable these with `KEV_MERGE=0`, `KEV_ATTN=eager`, `KEV_SHAPE_BUCKET=1`, and `KEV_PREFIX_CACHE=0`. On 24 new-source records, bf16 probabilities differed from fp32 by at most 0.017, with no change in the highest-probability answer. That is a small check, not a guarantee for every input.

## Training

The released models use `decision-v7`: 10,000 examples from ten public datasets, 896 generated policy examples, and 1,680 examples from 60 generated rule structures. All train for two epochs with LoRA rank 16 and cross-entropy. The learning rate is `1e-4` for 0.8B and `5e-5` for 4B/9B. For Qwen3.5 bases the adapter also covers the DeltaNet projections; `kev.train` picks the right targets from the model config.

```bash
# sanity run, ~1 minute
uv run python -m kev.train --n_per_source 40 --accum 4 --out runs/smoke

# Kev-0.8B (~20 min on one H100; the Mac path works but is slow for Qwen3.5 bases)
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
    --epochs 2 --lr 1e-4 --batch 8 --dtype bf16 --p_none_pair 0.25 --device cuda --out runs/kev-0.8b

# the Kev-4B recipe (one H100 via Modal, ~1 h; see below). Swap in Qwen/Qwen3-4B-Base for the previous generation.
uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-4B-Base --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
    --epochs 2 --lr 5e-5 --batch 4 --accum 2 --dtype bf16 --checkpointing 1 --p_none_pair 0.25 --device cuda --out runs/kev-4b
```

### Fine-tuning on your own data

The released models were trained on public datasets and generated policy examples. If your questions look different — your own routing categories, your own escalation rules, another language — a short fine-tune on a few hundred labelled examples usually helps more than any prompt change.

Put your examples in a JSONL file, one request per line. It's the same shape as an API request, plus a `label` on every question:

```jsonl
{"state": {"subject": "Charged twice", "body": "I see two charges for order #4411. Please refund one."},
 "questions": {
   "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems", "access": "Login and account access"}, "label": "billing"},
   "angry":    {"type": "noul",   "instructions": "Is the customer angry?", "label": false},
   "priority": {"type": "score",  "instructions": "How urgent is this ticket?", "criteria": ["low", "normal", "high"], "label": 1}}}
```

For `choice` the label is the option name, for `noul` it's `true` or `false`, and for `score` it's the level's position starting at 0. Keep 10–20% of the file aside for evaluation.

Then start from a released checkpoint with `--init_from`:

```bash
uv run python -m kev.train --data train.jsonl --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
    --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --out runs/mine

uv run python -m kev.benchmark --run runs/mine --data heldout.jsonl --out runs/mine-eval
uv run --extra serve python -m kev.serve --run runs/mine --port 8009
```

`--init_from` loads the adapter and pointer head from the released model before training, so you keep what Kev already knows and add your domain on top. Starting from the base model instead throws that away: in one user's test on 836 support-tool decisions, a fine-tune from the base scored 0.33 on Kev's own evaluation set, against 0.84 for the released model; the same data with `--init_from` kept 0.83 there and reached 0.88 on the new domain. Use a smaller learning rate than the from-scratch recipe (`2e-5` is a good start), and pick `--base` to match the checkpoint you start from; the trainer checks that the base, revision, LoRA rank, and head size agree before it loads anything.

`--batch 1 --accum 8` in bf16 fits the 0.8B model on a 4 GB GPU. The benchmark reports accuracy, Brier score, and calibration per question type, so you can see which of your questions the fine-tune helped. The checkpoint you started from is recorded in `runs/mine/training_config.json`.

Use `uv run python -m kev.train --help` for all training options. The released models don't use the optional `--perm_kl` or `--ord_w` losses. The [model cards](docs/model-cards/) have the training settings and dataset lists; [PLAN.md](PLAN.md) records what was tried and what helped.

On a Mac, run one training job at a time. Two jobs on the same Apple GPU are much slower. Use Modal for longer runs.

If you work with a coding agent, the [`kev-finetune` skill](skills/kev-finetune/) does all of this on Modal without a local GPU: it interviews you, finds the questions your code already asks Jev or TypeSafe, converts labels you have or generates enough with any LLM to measure a gain, fine-tunes from a released checkpoint, fits the temperature on a held-out slice, scores the result against the baseline, deploys a System One endpoint, and tears it all down. Install it with `npx skills add jaredpalmer/kev@kev-finetune`; the [README](skills/kev-finetune/README.md) is the same recipe for humans.

### Modal

Each trial gets its own H100. The study keeps running if you disconnect, and you can download the results when it finishes:

```bash
uv run modal token new                                    # once; opens the browser
KEV_GPU=T4 uv run modal run modal_app.py::smoke           # end-to-end check, ~1 minute of GPU

uv run modal deploy modal_app.py                          # once; studies run on the deployed app and survive disconnects
uv run modal run modal_app.py::study \
    --suite evals/v7/decision-v7 --plan experiments/v7-final.json \
    --name my-study --transfer evals/v4/transfer-v4 --budget 30 --timeout 7200
uv run modal run modal_app.py::pull --name my-study       # results -> runs/my-study, ranked
```

[Study plans](experiments/v7-final.json) list training settings. Each trial saves the settings, code hashes, dataset hashes, and results. Choose models using the development results, not the locked test. After choosing a final candidate, you can read its test results once:

```bash
uv run modal run modal_app.py::locked_test --trial my-study/00-trial-0 --name my-candidate   # one read, ever
```

## Evaluation

The evaluation data under `evals/` is frozen: dataset versions and file checksums are recorded in each manifest. Large training files are downloaded from [the Hub mirror](https://huggingface.co/datasets/jaredpalmer/kev-suites) and checked against those hashes.

```bash
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v4/transfer-v4 --out runs/my-eval      # out of domain
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v9/transfer-v9 --out runs/my-eval-v9   # + MMLU-Pro, buried states, unknowable items
uv run python -m kev.benchmark --run jaredpalmer/kev-4b --suite evals/v7/decision-v7 --out runs/my-eval-id   # in distribution
uv run python -m kev.benchmark --remote http://127.0.0.1:8009 --suite evals/v4/transfer-v4 --out runs/my-remote   # any System One endpoint
```

These commands use development data. Test data requires `--allow-test`. The benchmark reports accuracy, Brier score, calibration error, the share of decisions you could automate at a 5% error budget, option-order changes, and question isolation. `transfer-v9` adds 10-way MMLU-Pro, records buried among unrelated text, and "unknowable" records whose deciding evidence was removed; for those it reports how often the model still answers with at least 0.9 confidence (Kev-9B 0%, Jev 9%, Kev-8B 26%). Published accuracy numbers use fp32 evaluation, not the bf16 serving path. The published numbers in this README and the model cards are checked against the committed reports they come from (`docs/claims.json`, `uv run python scripts/verify_claims.py`, run in CI).

`evals/external/` holds two other projects' test sets converted to this format, with their published live Jev results: [SemIf](https://github.com/TheoLeeCJ/SemIf)'s 144 authored decisions (Kev-9B 0.917, Jev 0.965) and [scienthoon](https://github.com/scienthoon/jev-ood-calibration)'s 900 support tickets (Kev-9B 0.952 on routing and 0.911 on tone, Jev 0.897 and 0.914). It also holds the two third-party selections SemIf uses for its own external comparison, rebuilt from the same hash-verified sources with `scripts/freeze_semif_external.py`. `wanli-v1` is 256 WANLI test pairs asked as a choice between supported, insufficient and contradicted. Live Jev scores 0.758; the classes are near-balanced, so balanced accuracy is the same to three digits. SemIf reports 0.637 balanced accuracy for untrained Qwen3.5-4B. `typesafe-v1` is the 102 public evals.typesafe.ai questions over 20 cases, scored the way SemIf scores them with `scripts/compare_typesafe.py`: agreement with the reference answer and total-variation distance to the reference distribution, averaged within each case and then over cases. Live Jev scores 0.891 / 0.125, the published TypeSafe answers score 0.883 / 0.127 on the same rows, and SemIf reports 0.845 / 0.177. TypeSafe's documents are long. 17 of the 102 fit the training context and 89 fit the 8,192-token serving context the suite is scored under, so Kev's numbers cover the rows it answers, with the rejected count stated and the all-102 figure (rejected rows count as wrong) alongside. Kev trails Jev on both suites. WANLI accuracy is 0.703 for Kev-9B and 0.695 for Kev-4B against 0.758. TypeSafe agreement / distance is 0.809 / 0.226 for Kev-9B and 0.856 / 0.231 for Kev-4B on the 89 answered rows, and 0.728 / 0.304 and 0.770 / 0.308 over all 102. Document length explains part of the gap. Kev-9B answers 0.92 of the 26 documents inside its 384-token training context and 0.75 to 0.79 of the longer ones (Kev-4B 0.88, then 0.81 to 0.88), and 13 documents exceed even the serving context. A longer training context is the planned fix ([#48](https://github.com/jaredpalmer/kev/issues/48)).

`kev.jev` runs the same questions against Jev through Vercel AI Gateway. `kev.compare` compares two saved runs with paired bootstrap confidence intervals. For the full experiment history, see [PLAN.md](PLAN.md) and [the leaderboard](runs/leaderboard.md).

## Limitations

- Calibration is by a single temperature fitted in distribution. As served, Kev-9B assigns at least 0.9 probability to a wrong answer on 4.0% of new-source questions (Jev 3.7%); a fixed temperature can't reorder confidences, so the share of decisions you can automate at a 5% error budget (0.45–0.57) is still below Jev's 0.70. Test it on your own data before choosing a probability threshold.
- Fine-tuning can make the base model worse at individual tasks. Date arithmetic is the clearest case: the untrained Qwen3.5-9B base gets 0.82 on the `deadline` policy questions and the first Kev-9B got 0.72 ([issue #8](https://github.com/jaredpalmer/kev/issues/8)). Training on examples that state the day count, plus `KEV_DATE_FACTS=1`, recovers it (0.90). Knowledge questions (MMLU 0.74 vs Jev 0.90; MMLU-Pro 0.52 vs 0.84) are the remaining large gap, and they're set by the base model: a Kev trained on the 35B-A3B mixture-of-experts base didn't move them ([PLAN.md](PLAN.md)).
- The current models are slow on Apple Silicon (see Serving Performance) and need `transformers >= 5.17`.
- Changing option order can change an answer. Question isolation doesn't prevent this.
- Training uses at most 384 state tokens and 1,024 tokens for the state plus one question. Serving allows 8,192 tokens for the state plus one question; longer context wasn't covered by training.
- The server handles one request at a time. It caches repeated state text, but doesn't batch requests from different callers.

## Development

```bash
uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py -q  # no weights, no server; runs in CI
KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q   # against a running server
cd playground && npm run lint && npx next typegen && npx tsc --noEmit -p .
```

The API tests run TypeSafe's example requests and the official SDK against your local server.

<details>
<summary>Troubleshooting</summary>

- If MPS runs out of memory during training, check that you're running only one job. Don't enable `output_hidden_states` or add tokens with peft's `trainable_token_indices`; both have caused memory problems here.
- If the playground loads but buttons don't work, use `localhost:3001`. Next.js checks development hostnames. Other hosts need an entry in `allowedDevOrigins` in `playground/next.config.ts`.
- If dataset loading reports `Dataset scripts are no longer supported`, use `legacy-datasets/banking77`. This repo already uses it.

</details>

## Authors

- Jared Palmer ([@jaredpalmer](https://github.com/jaredpalmer))

Built with [Devin](https://devin.ai). Thanks to [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) for the architecture write-up, [TypeSafe](https://docs.typesafe.ai/api) for the API design, [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B-Base) for the base models, [3x3xX3N0N](https://github.com/jaredpalmer/kev/issues/8) for showing where the date-arithmetic failure really is, and [Radexito](https://github.com/Radexito) for `--init_from`.

Related work: [Hydragen](https://arxiv.org/abs/2402.05099), [DeFT](https://arxiv.org/abs/2404.00242), [FIRST](https://arxiv.org/abs/2406.15657).

## License

[Apache-2.0](LICENSE). The Qwen3 and Qwen3.5 base models are also Apache-2.0. Training datasets have their own licenses; see the [model cards](docs/model-cards/).
