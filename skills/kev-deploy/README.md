# Deploy Kev on Modal

Your own Kev endpoint, speaking TypeSafe's System One protocol, in three commands. It scales to zero when idle, so an
unused endpoint costs nothing.

```bash
pip install modal && modal setup                  # once: sign in to Modal in the browser
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py
KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py
```

The deploy prints `https://<your-workspace>--kev-api.modal.run`. Keep the key: requests need
`Authorization: Bearer <key>`. Point any TypeSafe client at the URL:

```python
client = TypeSafeClient(api_key=KEV_API_KEY, base_url="https://<your-workspace>--kev-api.modal.run", model="kev-latest")
```

| Model | Set | GPU ($/h while up) | Warm model time, 6 questions (new / repeated state) | First request after idle |
| --- | --- | --- | --- | --- |
| Kev-0.8B | `KEV_MODEL=jaredpalmer/kev-0.8b` | L4 (0.80) | 37 / 28 ms | ~40 s |
| Kev-4B (default) | nothing | L40S (1.95) | 42 / 28 ms | ~35 s |
| Kev-9B | `KEV_MODEL=jaredpalmer/kev-9b` | H100 (3.95) | 24 / 17 ms | ~55 s |
| Kev-27B | `KEV_MODEL=jaredpalmer/kev-27b` | B200 (6.25) | 47 / 32 ms | ~50 s |

Concurrent requests are batched in each container (about 100 requests/s per H100 container for Kev-4B over HTTP), and
Modal adds containers past 32 concurrent requests each. `KEV_FLASH=1` (with `KEV_REGION`) uses Modal's experimental direct
HTTP server: a ~46 ms round trip instead of ~77 ms, one container kept up.

The round trip adds the network and Modal's proxy (about 80-100 ms from the US to a us-east container with a kept-alive
connection); `KEV_REGION=us` keeps the container near US callers. The very first deploy also downloads the weights and
compiles kernels (1-2 minutes); later cold starts reuse the cache.

`KEV_MIN_CONTAINERS=1` keeps it warm; `modal app stop kev` takes it down. With an agent, `npx skills add
jaredpalmer/kev@kev-deploy` and ask it to deploy Kev; it follows [SKILL.md](SKILL.md). `modal skills install` adds
Modal's own agent skill and docs alongside it, for anything beyond this file (GPUs, secrets, logs, billing).
