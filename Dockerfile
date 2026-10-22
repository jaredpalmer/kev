# Kev serving image for Linux / WSL2: kev.serve, a TypeSafe-compatible
# POST /v1/systemone endpoint (prefill-only, no text generation).
#
# KEV_DEVICE=gpu (default) installs the NVIDIA fused path; cpu skips those packages and
# runs the eager PyTorch reference kernels. A build cannot see the host's GPU (--gpus
# applies to `docker run`), so choose it:
#   docker build -t kev .                                # NVIDIA GPU
#   docker build --build-arg KEV_DEVICE=cpu -t kev .     # no GPU / disable acceleration

FROM astral/uv:python3.13-bookworm-slim

ARG KEV_DEVICE=gpu

ENV UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_CACHE_DIR=/app/.cache/uv \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models \
    TRITON_CACHE_DIR=/triton-cache

WORKDIR /app

# Validate the ARG before the expensive dependency layers. gcc/libc6-dev: Triton JITs the
# fla kernels on the first request. Referencing $KEV_DEVICE in a RUN puts its value in the
# layer cache key, so gpu and cpu builds keep separate caches from this layer down.
RUN set -eu; \
    case "$KEV_DEVICE" in \
      gpu|cpu) ;; \
      *) echo "KEV_DEVICE must be gpu or cpu (got '$KEV_DEVICE')" >&2; exit 1 ;; \
    esac; \
    if [ "$KEV_DEVICE" = gpu ]; then \
      apt-get update; \
      apt-get install -y --no-install-recommends gcc libc6-dev; \
    fi

# Common dependencies. The Linux torch wheel bundles CUDA 12, so no CUDA base image.
COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --frozen --extra serve --no-dev --no-install-project

# fla and causal-conv1d, the Gated DeltaNet kernels. fla's plain dependency set is
# einops + fla-core only (torch/triton live behind extras we do not request). causal-conv1d
# goes in --no-deps: its metadata asks for bare torch + packaging + ninja, torch is already
# installed by the sync above, and this keeps the build-only ninja package out of the image.
RUN --mount=type=cache,target=/app/.cache/uv \
    set -eu; \
    if [ "$KEV_DEVICE" = gpu ]; then \
      uv pip install "flash-linear-attention==0.5.2"; \
      uv pip install --no-deps \
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"; \
    fi

# Source last, so editing kev/ rebuilds only the layers below. --inexact keeps the
# packages above, which are not in the lockfile: it only protects packages the lock does
# not mention at all, so a package that *is* in the lock keeps its locked version.
COPY kev/ kev/
RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --frozen --extra serve --no-dev --inexact

# triton last, after the sync above. torch 2.8 pins triton 3.4, which fla's Hopper chunked
# backward gets wrong (fla#640) — but triton *is* in the lockfile, so the sync above would
# put 3.4.0 back over any earlier upgrade. Installing it here is the only order that holds.
RUN --mount=type=cache,target=/app/.cache/uv \
    set -eu; \
    if [ "$KEV_DEVICE" = gpu ]; then \
      uv pip install "triton>=3.7.1"; \
    fi

EXPOSE 8000
# host/port are fixed in the entrypoint. The default CMD serves a checkpoint bind-mounted
# at /models/mine (its base at /models/base; repoint head.pt first, see docs/docker.md):
#   docker run --gpus all -v ./base:/models/base:ro -v ./mine:/models/mine:ro kev
# Override CMD for any other path or a released Hub id:  docker run kev --run jaredpalmer/kev-4b
ENTRYPOINT ["/app/.venv/bin/python", "-m", "kev.serve", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--run", "/models/mine"]
