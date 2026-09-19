# CPU-only image for the kev /v1/systemone API.
# The PyPI default torch wheel bundles CUDA; pull the CPU build from the PyTorch index
# first so the project install finds the version range already satisfied.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY kev/ kev/

RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<2.9" \
 && uv pip install --system .[serve]

# Checkpoint + base model cache; mount a volume to persist across restarts.
ENV KEV_HOST=0.0.0.0 HF_HOME=/hf
VOLUME /hf
EXPOSE 8008

# Weights aren't in git; pull the published checkpoint from the Hub.
CMD ["python", "-m", "kev.serve", "--run", "jaredpalmer/kev-0.5b", "--port", "8008"]
