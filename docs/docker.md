# Deploying Kev with Docker (Linux / WSL2)

Kev runs as a single prefill-only HTTP service (`kev.serve`): no text generation, one
forward pass plus a pointer head. The image exposes the TypeSafe-compatible
`POST /v1/systemone` endpoint.

## Prerequisites

- Docker 23+ with [BuildKit](https://docs.docker.com/build/buildkit/) (default).
- For the default `gpu` variant: an NVIDIA driver plus the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Verify the container can see the GPU:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

**WSL2:** use either Docker Desktop with WSL2 integration enabled, or Docker CE plus the
container toolkit installed inside the WSL distro. Install the NVIDIA Windows driver
(WSL2-ready); you do not install a separate Linux driver inside WSL. Run the `nvidia-smi`
check above from the WSL shell.

## Build

| `KEV_DEVICE` | When | Contents |
|---|---|---|
| `gpu` (default) | NVIDIA GPU | fla Triton kernels, `triton>=3.7.1`, causal-conv1d (fused fast path) |
| `cpu` | no GPU / disable acceleration | no NVIDIA-only packages; eager PyTorch kernels (slow) |

```bash
docker build -t kev .                                # GPU (default)
docker build --build-arg KEV_DEVICE=cpu -t kev .     # CPU / disable GPU
```

The build cannot see the host's devices (`--gpus` applies only to `docker run`), so the
GPU is not probed at build time — disable it explicitly when needed.

The `gpu` variant is x86_64: `causal-conv1d` is installed from a prebuilt wheel. On an
arm64 host, build `cpu` (or install `causal-conv1d` from source yourself).

The Linux `torch` wheel bundles CUDA 12, so the image does not start from a CUDA base.

## Run

The image follows a mount convention: your checkpoint goes to `/models/mine` and its base
model to `/models/base`, so with nothing mounted the service cannot start — mount the two
directories and it runs with no arguments (`--host`/`--port` are already set in the image
entrypoint).

GPU (default):

```bash
docker run --rm -d --name kev --gpus all --shm-size=2g \
  -p 8000:8000 \
  -v /data/Qwen3.5-4B-Base:/models/base:ro \
  -v /data/kev-mine:/models/mine:ro \
  -v kev-triton:/triton-cache \
  kev
```

CPU (image built with `KEV_DEVICE=cpu`; drop `--gpus` and the triton volume):

```bash
docker run --rm -d --name kev -p 8000:8000 \
  -v /data/Qwen3.5-4B-Base:/models/base:ro \
  -v /data/kev-mine:/models/mine:ro \
  kev
```

The checkpoint's `head.pt` must point at the in-container base path; see
[Offline / local base model](#offline--local-base-model) below. Override the default
`--run` for a checkpoint at another path or a released Hub id (`jaredpalmer/kev-0.8b`,
`jaredpalmer/kev-4b`, `jaredpalmer/kev-9b`, …):

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v kev-models:/models -v kev-triton:/triton-cache \
  kev --run jaredpalmer/kev-4b
```

A Hub id is downloaded into `/models` (`HF_HOME`) on first start; the `kev-models` volume
keeps it across restarts.

## Persistent volumes

| Path | Purpose |
|---|---|
| `/models/mine` | fine-tuned checkpoint served by the default CMD (`--run /models/mine`) |
| `/models/base` | its base model when fully offline (repointed `head.pt`, see below) |
| `/models` (`HF_HOME`) | downloaded weights for a Hub id (`--run <hub-id>`) |
| `/triton-cache` (`TRITON_CACHE_DIR`) | compiled fla kernels; avoids recompiling on restart |

The first GPU request compiles the fla kernels (a few minutes). The cache volume makes
this a one-time cost.

## Verify

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool

curl -s http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":"Evaluating a backend for a small team.",
       "questions":{"q1":{"type":"choice",
         "instructions":"Which architecture?",
         "criteria":{"monolith":"one unit","microservices":"many units"}}}}'
```

An `answers.q1.choice` in the response means the service is up.

## Offline / local base model

To serve with no network access, point the checkpoint at the `/models/base` convention
instead of a Hub id. kev's loader reads only `head.pt`'s `base` field — editing
`adapter_config.json` alone has no effect.

```bash
python scripts/repoint_checkpoint.py --run /path/to/checkpoint --base /models/base
```

The script rewrites only the base reference, preserves the temperature and all other
metadata, is idempotent, and does not require the target to exist yet. The path given to
`--base` is the path *inside the container*, so running this on the host writes
`/models/base` literally into `head.pt`. Then mount the directories and start with no
arguments — the default CMD already serves `/models/mine`:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /data/Qwen3.5-4B-Base:/models/base:ro \
  -v /data/kev-mine:/models/mine:ro \
  -v kev-triton:/triton-cache \
  kev
```
