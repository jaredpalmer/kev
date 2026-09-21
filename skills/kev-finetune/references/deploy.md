# Serving and publishing

## Modal endpoint

```bash
KEV_SERVE_RUN=support-v1 modal deploy scripts/kev_modal.py
```

`KEV_SERVE_RUN` is a run name on the `kev-finetune-runs` volume or a Hub id (`jaredpalmer/kev-4b`, `you/kev-4b-support`,
`repo@tag`). The deploy prints the URL, `https://<workspace>--kev-finetune-api.modal.run`. The container loads the
checkpoint once (LoRA merged in fp32, cast to bf16; the fitted temperature is applied automatically), serves up to 8
concurrent requests, and scales to zero after 5 idle minutes; the first request after that pays a cold start of
about a minute for the 4B.

| Base | `KEV_SERVE_GPU` |
| --- | --- |
| kev-0.8b, kev-4b | `L4` (default) |
| kev-9b | `A100-80GB` or `H100` (fp32 merge needs 36 GB) |

Redeploying with a different `KEV_SERVE_RUN` replaces the endpoint. Several models at once: deploy under different app
names (`KEV_APP_NAME=kev-support modal deploy ...`), which also changes the URL label.

### Auth

Without configuration the endpoint is public (the URL is the secret). To require a bearer token:

```bash
modal secret create kev-serve-key KEV_SERVE_API_KEY=$(openssl rand -hex 24)
KEV_SERVE_SECRET=kev-serve-key KEV_SERVE_RUN=support-v1 modal deploy scripts/kev_modal.py
```

Requests then need `Authorization: Bearer <KEV_SERVE_API_KEY>`, which is what the TypeSafe SDK sends as `api_key`.

### Request

```bash
curl -s $URL/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 5521 arrived two weeks late and I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this ticket first?",
                    "criteria": {"returns": "...", "shipping": "...", "billing": "...", "account": "..."}},
    "escalate":    {"type": "noul", "instructions": "Does this ticket need urgent attention from a human within the hour?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm or neutral", "Annoyed", "Angry or threatening to leave"]}}}'
```

Send the *same* instructions and option names you trained on. The response has per-question `probabilities`
(calibrated), `choice` / `noul` / `score` (expected level), `confidence`, `usage` and `latency_ms`. `GET /v1/models`
reports the run, base, temperature and prefix-cache statistics.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
client = TypeSafeClient(api_key="<KEV_SERVE_API_KEY or any string>", base_url=URL, model="kev-latest")
r = client.system_one(state=ticket, questions={"department": Choice(instructions=..., criteria={...}),
                                               "escalate": Noul(instructions=...), "frustration": Score(instructions=..., criteria=[...])})
r.choices["department"].probabilities, r.nouls["escalate"].noul, r.scores["frustration"].score
```

Confirm the served numbers match the offline score: `modal run scripts/kev_modal.py::evaluate --data data/support --name support-v1-served --remote $URL`.

### Using the probabilities

The calibrated `confidence` of a choice answer is the thing to threshold. `result.json["development"]["calibrated"]["coverage_at_5pct_error"]`
tells you what share of traffic clears a 5% error budget; `selective` gives the confidence cutoffs at 50% and 80%
coverage. Route everything below the cutoff to a human. Re-check the cutoff whenever you retrain: the temperature and
the cutoffs are per checkpoint.

## Run it locally

```bash
modal run scripts/kev_modal.py::pull --name support-v1 --checkpoint
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run ../runs/support-v1/checkpoint --port 8009
```

Same API on `127.0.0.1:8009`. Qwen3.5 bases are slow on Apple Silicon (no DeltaNet kernels for MPS): the 4B answers
in ~0.8 s; a CUDA machine is fast. The playground in the repo (`cd playground && npm run dev -- -p 3001`) works against
this server.

## Publish to the Hugging Face Hub

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...            # a write token
KEV_HF_SECRET=huggingface-secret modal run scripts/kev_modal.py::publish --name support-v1 --repo you/kev-4b-support [--private]
```

Uploads the adapter, `head.pt` (with the temperature), tokenizer, `result.json`, `training_config.json` and `train.log`
with a generated model card (baseline vs fine-tuned table; `--card your.md` to supply your own). The repo id then works
anywhere a Kev checkpoint does: `KEV_SERVE_RUN=you/kev-4b-support`, `kev.serve --run you/kev-4b-support`,
`--init-from you/kev-4b-support` for the next delta.
