"""Persistent serving endpoints for kev.serve on Modal (scale-to-zero).

Two checkpoints, one URL each, so every machine (laptop, staging, production) points at
the same brain instead of loading its own copy:

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal deploy modal_serve.py
    -> https://davey--kev-serve-serve-factory.us-east.modal.run   (factory-cloud-v8: factory organizer engine)
    -> https://davey--kev-serve-serve-unified.us-east.modal.run   (unified-v1: agent-memory-hub + RTV2)

The image is the kev-research research image (single source of truth: imported from
modal_app.py, which this file also mounts so the import works inside the container) with one
serving overlay: triton pinned to the exact version the training image pairs with fla 0.5.2
(a fresh builder context can resolve 3.8.x, whose autotuner signature fla rejects).

Checkpoints come from the shared kev-runs volume (148 MB LoRA adapters; push updates with
`MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal volume put kev-runs runs/<name> /<name>`);
base weights and the triton compile cache come from the shared kev-hf-cache volume, so a
shape's kernel compile is paid once ever, not per container.

Auth: bearer key required (kev.serve KEV_API_KEY). Create once:
`uv run modal secret create kev-serve-api KEV_API_KEY=<key>`.

Cost: T4 scale-to-zero bills only while a request is in flight plus `scaledown_window` after
the last one; cold starts reload the base from the HF cache volume (on billing, minutes).
KEV_GPU_SERVE overrides the GPU ("T4" default; "A10"/"L4" for lower latency).
"""
import os
from dataclasses import replace

import modal

from modal_app import HF_MOUNT, RUNS_MOUNT, hf_cache, image as research_image, runs_volume

_ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("KEV_GPU_SERVE", "T4")

# Serving overlay on the research image: pin triton to the exact pairing modal_app.py trains
# with, and mount modal_app.py so the import above resolves inside the container too.
image = (
    research_image.uv_pip_install("triton==3.7.1")
    .add_local_file(os.path.join(_ROOT, "modal_app.py"), "/root/modal_app.py")
)

hf_cache = hf_cache
runs_volume = runs_volume

app = modal.App("kev-serve")

# Common function options: T4 scale-to-zero; 5 min warm tail after the last request.
DEF = dict(image=image, gpu=GPU, cpu=4, memory=(16384, 49152),
           volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache},
           secrets=[modal.Secret.from_name("kev-serve-api")],
           scaledown_window=1200, timeout=1800)

CHECKPOINTS = {
    "serve-factory": "factory-cloud-v8",   # Software Factory organizer engine (com.factory.kev-serve today)
    "serve-unified": "unified-v1",         # agent-memory-hub decisions (com.agent-memory-hub.kev-serve today)
}


def _asgi(checkpoint: str):
    """Return kev.serve's FastAPI app with a loaded checkpoint, mirroring kev.serve.main()'s
    cuda serving defaults (bf16, CUDA graphs, fused kernels). Runs once per cold container."""
    import torch

    from kev.checkpoint import Checkpoint, LoadOptions, fused_available
    from kev.device import select
    from kev.serve import Server, app as fastapi_app

    run_path = f"{RUNS_MOUNT}/{checkpoint}/checkpoint"
    if not os.path.exists(f"{run_path}/head.pt"):
        raise FileNotFoundError(f"checkpoint not found on volume: {run_path} — push it with modal volume put")

    opts = LoadOptions.from_env()
    if opts.backend is None:
        opts = replace(opts, backend="auto")
    dev = select("cuda", dtype=torch.bfloat16)
    if opts.dtype is None:
        opts = replace(opts, dtype=torch.bfloat16)
    if opts.cuda_graphs is None:
        opts = replace(opts, cuda_graphs=True)
    if opts.fused is None:
        opts = replace(opts, fused=fused_available())

    ck = Checkpoint(run_path)
    tok, model = ck.load(dev, opts)
    fastapi_app.state.server = Server(ck, tok, model, dev)
    print(f"serving {ck.requested} ({run_path}) on {dev}", flush=True)
    return fastapi_app


@app.function(**DEF)
@modal.asgi_app()
def serve_factory():
    return _asgi(CHECKPOINTS["serve-factory"])


@app.function(**DEF)
@modal.asgi_app()
def serve_unified():
    return _asgi(CHECKPOINTS["serve-unified"])
