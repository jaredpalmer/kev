# Serving, wiring in, publishing, tearing down

## Modal endpoint

```bash
modal secret create kev-serve-key KEV_API_KEY=$(openssl rand -hex 24)
KEV_SERVE_SECRET=kev-serve-key KEV_SERVE_RUN=x-v1 modal deploy scripts/kev_modal.py
```

`KEV_SERVE_RUN` is a run name on the `kev-finetune-runs` volume or a Hub id (`jaredpalmer/kev-4b`, `you/kev-4b-x`,
`repo@tag`). The deploy prints the URL, `https://<workspace>--kev-finetune-api.modal.run`. The container loads the
checkpoint once (LoRA folded into the bf16 weights with the delta computed in fp32, the fitted temperature applied automatically), warms up the
fused kernels and captures CUDA graphs (a 6-question Kev-4B request: ~17 ms on an H100, ~130 ms without), serves up
to 8 concurrent requests, and scales to zero after 5 idle minutes. Cold start after idle is 1-2 minutes for the 4B;
`KEV_SERVE_MIN_CONTAINERS=1` at deploy time keeps one warm (~$0.80/h on an L4).

| Base | `KEV_SERVE_GPU` |
| --- | --- |
| kev-0.8b, kev-4b | `L4` (default; the cheapest, but it runs out of compute on the 4B under load: `L40S` there) |
| kev-9b | `H100` or `L40S` (about 17 GB of GPU memory in bf16) |

Without `KEV_SERVE_SECRET` the endpoint is public (the URL is the only secret); with it, requests need
`Authorization: Bearer <KEV_API_KEY>` and everything else gets 401. Redeploying with another `KEV_SERVE_RUN`
replaces the model behind the same URL. Several models at once: `KEV_APP_NAME=kev-support modal deploy ...` (new app,
new URL label).

## Wiring it into the user's code

The endpoint speaks TypeSafe's System One protocol, so the change is the base URL and key, not the request shape.
Instructions and option names must be the ones the model trained on.

TypeSafe Python SDK:

```python
client = TypeSafeClient(api_key=KEV_KEY, base_url=KEV_URL, model="kev-latest")   # was: TypeSafeClient(api_key=TYPESAFE_KEY)
```

AI SDK / raw HTTP (the AI SDK's `experimental_evaluate` targets the gateway, so switch those calls to a fetch):

```ts
const r = await fetch(`${KEV_URL}/v1/systemone`, { method: "POST",
  headers: { "content-type": "application/json", authorization: `Bearer ${KEV_KEY}` },
  body: JSON.stringify({ state, model: "kev-latest", questions }) });   // questions use type "noul", not "boolean"
```

curl:

```bash
curl -s $KEV_URL/v1/systemone -H "authorization: Bearer $KEV_KEY" -H 'content-type: application/json' -d '{
  "state": "...", "model": "kev-latest",
  "questions": {"department": {"type": "choice", "instructions": "...", "criteria": {"returns": "...", "shipping": "..."}},
                "escalate":   {"type": "noul",   "instructions": "..."}}}'
```

The response has per-question `probabilities` (calibrated), `choice` / `noul` / `score` (expected level), `confidence`,
`usage`, `latency_ms`. `GET /v1/models` reports the served run, base and temperature.

Confirm the served numbers match the offline score before switching traffic:
`KEV_REMOTE_API_KEY=$KEV_KEY modal run scripts/kev_modal.py::evaluate --data data/x --name x-v1-served --remote $KEV_URL`.

### Thresholds

The number to threshold is the calibrated `confidence` (choice) or `noul` probability. `result.json` ->
`development.calibrated.coverage_at_5pct_error` is the share of traffic that clears a 5% error budget, and
`development.calibrated.selective["0.5"|"0.8"].confidence_cutoff` are the cutoffs at 50% / 80% coverage. Route below
the cutoff to a human. Thresholds and temperature are per checkpoint: re-read them after every retrain.

## Run it locally instead

```bash
modal run scripts/kev_modal.py::pull --name x-v1 --checkpoint
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run ../runs/x-v1/checkpoint --port 8009
```

Same API on `127.0.0.1:8009`. Qwen3.5 bases are slow on Apple Silicon (no DeltaNet kernels for MPS, ~0.8 s per request
for the 4B); a CUDA machine is fast. The repo's playground works against this server.

## Publishing (optional, private by default)

Only if the user wants the weights outside Modal. The Hub repo is private unless `--public`.

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...            # a write token
KEV_HF_SECRET=huggingface-secret modal run scripts/kev_modal.py::publish --name x-v1 --repo you/kev-4b-x
```

Uploads the adapter, `head.pt` (with the temperature), tokenizer, `result.json`, `training_config.json`, `train.log`
and a generated card (`--card your.md` to replace it). The repo id then works anywhere a Kev checkpoint does:
`KEV_SERVE_RUN=you/kev-4b-x`, `kev.serve --run you/kev-4b-x`, `--init-from you/kev-4b-x` for the next delta.
Remove it with `hf repo delete you/kev-4b-x`.

## Tear down

| Command | Removes |
| --- | --- |
| `modal run scripts/kev_modal.py::teardown --run x-v1 --yes` | one run: weights, reports, uploaded data (`--run a,b` for several) |
| `modal run scripts/kev_modal.py::teardown --endpoint` | the deployed app: the URL stops answering; runs stay |
| `modal run scripts/kev_modal.py::teardown --everything --yes` | the app and the `kev-finetune-runs` volume |
| `... --everything --cache --yes` | also the shared `kev-hf-cache` (base weights; re-downloaded on the next run) |

Manual equivalents: `modal app stop kev-finetune`, `modal volume delete kev-finetune-runs --yes`,
`modal secret delete kev-serve-key`. Secrets are never deleted by the script. Locally: `rm -rf runs/ data/`.
An idle deployed endpoint costs nothing; the volume costs Modal's storage rate for the checkpoints on it
(~0.3 GB per 4B delta).
