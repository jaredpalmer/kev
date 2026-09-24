---
name: kev-deploy
description: Deploy a Kev decision model (the open Jev-style System One model) as the user's own TypeSafe-compatible HTTPS endpoint on Modal with one command, wire it into their code, and take it down. Use when someone wants to host Kev, get a Kev API URL, replace Jev / TypeSafe calls with a self-hosted model, pick a Kev size or GPU, add an API key, keep an endpoint warm, or stop and remove a Kev deployment.
license: Apache-2.0
compatibility: Requires Python 3.10+ and a Modal account (`pip install modal && modal setup`, free tier works for the small models). No GPU, no clone of the Kev repo, no Hugging Face account for the public checkpoints.
metadata:
  author: jaredpalmer
  version: "1.1"
  repository: https://github.com/jaredpalmer/kev
---

# Deploy Kev on Modal

`scripts/kev_serve.py` is a single self-contained Modal app. `modal deploy` on it builds an image with the kev package at
a pinned commit, loads a Kev checkpoint from the Hugging Face Hub on a GPU sized for it, and serves TypeSafe's System One
protocol (`POST /v1/systemone`, `GET /v1/models`) at `https://<workspace>--kev-api.modal.run`. It scales to zero when idle.
The file is short on purpose: if the user's case does not fit, read it and edit it.

## 1. Check the prerequisites

```bash
python3 -c "import modal" 2>/dev/null || pip install modal
modal profile current 2>/dev/null || modal setup      # opens a browser to sign in; the user must do this step
modal skills install -y                               # optional: Modal's own agent skill + docs (into .agents/, or -g for home)
```

If `modal setup` is needed, tell the user and wait; do not try to authenticate for them. `modal skills install` gives you
Modal's official skill and documentation, which helps with anything beyond this file: GPUs, secrets, volumes, logs, billing.

## 2. Pick the model

| `KEV_MODEL` | GPU (automatic; fallbacks in parentheses) | $/h while up | Model time, 6 questions (new / repeated state) | Cold start (cached weights) | When |
| --- | --- | --- | --- | --- | --- |
| `jaredpalmer/kev-0.8b` | L4 (L40S) | 0.80 | 37 / 28 ms | ~40 s | cheapest, prototyping |
| `jaredpalmer/kev-4b` (default) | L40S (H100) | 1.95 | 42 / 28 ms (H100: 18 / 13 ms) | ~35 s | the default: best quality per dollar |
| `jaredpalmer/kev-9b` | H100 (H200, L40S) | 3.95 | 24 / 17 ms | ~55 s | best released accuracy |

Model time is the `latency_ms` the API returns (median of 20 requests, measured in the Kev repo: `runs/fused-*/report.json`
for Kev-4B and Kev-9B, `runs/serving-*/report.json` for Kev-0.8B). A new state is the normal call, since every ticket is a
new state; a repeated state is served from a prefix cache. The very first cold start of an account also downloads the
weights and compiles kernels (1-2 minutes); both are cached on the `kev-hf-cache` volume afterwards. Other GPUs work with
`KEV_GPU` but are worse picks: an L4 runs out of compute on Kev-4B, and an A100 is slower than an L40S here and costs more.

A warm container costs the GPU's hourly rate only while it is up; after five idle minutes it scales to zero.
`KEV_MIN_CONTAINERS=1` keeps one warm (no cold starts, pays the hourly rate all the time). `@revision` pins a checkpoint
revision (`jaredpalmer/kev-4b@v7-base`).

## 3. Deploy

Always set an API key unless the user explicitly wants a public URL; without one, anyone with the URL can spend their
GPU time.

```bash
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py   # or use the skill's copy
export KEV_API_KEY=$(openssl rand -hex 24)                # save it: it is the endpoint's bearer token
KEV_MODEL=jaredpalmer/kev-4b modal deploy kev_serve.py    # prints the URL
```

Settings are read at deploy time; redeploying with other values replaces the model behind the same URL.
`KEV_APP_NAME=kev-9b` gives a second, independent endpoint (`https://<workspace>--kev-9b-api.modal.run`).
`KEV_GPU=H100` overrides the GPU list (comma-separated). `KEV_REGION=us` (or `us-east`, `eu`, ...) pins where the container
runs: without it Modal takes the first region with a free GPU, which can be another continent (an unpinned Kev-4B landed in
Frankfurt and added ~150 ms to every round trip from the US). A pinned region costs 1.15-1.75x on Modal; pin it near the
callers for latency-sensitive use.

### Throughput

A container answers concurrent requests in batches: its model thread takes everything waiting and runs it through shared
passes. In-process, Kev-4B on an H100 serves about 95 six-question requests/s (120 on mixed short records); an L40S about
45. Over HTTP the front door matters more than the GPU (Kev-4B, H100, a client in the same region, measured 2026-09-24):

| Front door | One request, round trip | 8 / 32 concurrent clients | Scaling |
| --- | --- | --- | --- |
| web endpoint (default) | ~77 ms | 77 / 103 req/s | Modal adds containers past 32 concurrent requests each |
| `KEV_FLASH=1` (experimental) | ~46 ms | 70 / 112 req/s per container | one container stays up; past ~32 concurrent per container requests queue at the proxy (p99 ~4 s at 64) |

For steady high traffic, keep containers warm (`KEV_MIN_CONTAINERS=2` or more) so bursts do not wait for a cold start, and
size it at about 32 concurrent requests per container. `KEV_FLASH=1` needs `KEV_REGION` (its proxy is regional) and its
URL is printed as `https://<workspace>--<app>-kev.<region>.modal.direct`.

## 4. Verify

The first request after a deploy or an idle period waits for the cold start. Modal answers a request that waits longer
than 150 s with an HTTP 303 to a result URL, so warm the endpoint with a redirect-following call first:

```bash
curl -sL --max-time 900 $KEV_URL/v1/models -H "authorization: Bearer $KEV_API_KEY"   # returns once the model is loaded
```

```bash
curl -s $KEV_URL/v1/systemone -H "authorization: Bearer $KEV_API_KEY" -H 'content-type: application/json' -d '{
  "state": "Order 4411 arrived late and the box was crushed. Two charges appear on my card.", "model": "kev-latest",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}},
                "urgent": {"type": "noul", "instructions": "Does this need urgent human attention?"}}}'
curl -s $KEV_URL/v1/models -H "authorization: Bearer $KEV_API_KEY"       # served checkpoint, base, temperature
```

Expect per-question `probabilities` (calibrated by the checkpoint's own temperature), `choice` / `noul` / `score`, and
`latency_ms`, the model time: tens of milliseconds warm (table above). The round trip adds the network and Modal's proxy,
about 80-100 ms from a client in the US to a us-east container over a kept-alive connection, more with a new TLS
connection per request, so reuse one HTTP client. The first request of a new shape (question set, state length) runs
without a CUDA graph, about 2-4x slower; the server captures one in the background and later requests use it.
`/v1/models` reports the served checkpoint, its temperature and the number of captured graphs. A request without the key
must return 401.

## 5. Wire it in

The protocol is TypeSafe's, so only the base URL and key change:

```python
client = TypeSafeClient(api_key=KEV_API_KEY, base_url=KEV_URL, model="kev-latest")   # was: TypeSafeClient(api_key=TYPESAFE_KEY)
```

```ts
const r = await fetch(`${KEV_URL}/v1/systemone`, { method: "POST",
  headers: { "content-type": "application/json", authorization: `Bearer ${KEV_API_KEY}` },
  body: JSON.stringify({ state, model: "kev-latest", questions }) });   // question type "noul", not "boolean"
```

Kev answers typed questions about a state (Choice over named options, Noul yes/no, Score over ordered levels) without
generating text. It was trained on public classification, policy and rule data; on the user's own domain, measure it on a
few hundred labelled examples before relying on it, and if it falls short, fine-tune it with the `kev-finetune` skill.

## 6. Take it down

```bash
modal app stop kev            # or the KEV_APP_NAME used; the URL stops working immediately
modal volume delete kev-hf-cache   # optional: the cached weights (shared with kev-finetune; next deploy re-downloads)
```

## Troubleshooting

- **First request returns nothing or a 303**: the cold start is still running (weights download on the very first start,
  or Modal is still finding a GPU; `modal app logs kev` says "waiting to be scheduled"). Follow redirects (`curl -L`), use a
  longer client timeout, or deploy with `KEV_MIN_CONTAINERS=1`.
- **CUDA out of memory on start**: the GPU is too small for that checkpoint (Kev-9B needs ~18 GB, Kev-4B ~10 GB); use the
  table above or `KEV_GPU=H100`.
- **Slow round trips with fast `latency_ms`**: the container is far from the caller or every request opens a new
  connection; set `KEV_REGION` and reuse the HTTP client.
- **401 with the right key**: the key is fixed at deploy time; redeploy with the same `KEV_API_KEY` exported.
- **Logs**: `modal app logs kev` shows the load line (`serving <model> on <GPU> ... ready in Ns`) and every request.
