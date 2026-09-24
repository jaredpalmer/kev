# Runs the TypeSafe-compatible server (kev.serve). The PyPI torch wheel for Linux carries CUDA, so the
# same image serves on CPU or, with --gpus all and the NVIDIA Container Toolkit, on a GPU host.
#
#   docker build -t kev .
#   docker run --rm -p 8008:8008 -v kev-models:/root/.cache/huggingface kev                          # Kev-4B, CPU
#   docker run --rm --gpus all -p 8008:8008 -v kev-models:/root/.cache/huggingface kev               # Kev-4B, GPU
#   docker run --rm -p 8008:8008 kev --run jaredpalmer/kev-9b --port 8008                            # another model
#
# The named volume keeps the multi-GB base-model download out of the container layer. KEV_API_KEY,
# KEV_DTYPE, KEV_BACKEND and the other KEV_* settings pass through with -e, as documented in README.md.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# setuptools needs pyproject.toml, the lockfile, README.md, LICENSE and the package itself; nothing else
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY kev ./kev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra serve --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8008

# 0.0.0.0 so the port is reachable outside the container; the local default stays 127.0.0.1
ENTRYPOINT ["python", "-m", "kev.serve", "--host", "0.0.0.0"]
CMD ["--run", "jaredpalmer/kev-4b", "--port", "8008"]
