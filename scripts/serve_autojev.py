"""Serve denis-pplx/autojev-27b with its own unmodified server on Modal, for the report-only head-to-head in PLAN.md
("External head-to-head: AutoJev-27B vs Kev-27B"). The repo and the weights are pinned; nothing of Kev's code is used.

    AUTOJEV_API_KEY=<key> uv run modal deploy scripts/serve_autojev.py
    KEV_REMOTE_API_KEY=<key> uv run python -m kev.benchmark --remote https://<workspace>--autojev-api.modal.run \
        --remote-model jev-latest --remote-concurrency 4 --suite evals/v4/transfer-v4 --out runs/autojev-transfer-v4
    uv run modal app stop autojev
"""
import os, subprocess

import modal

REPO, COMMIT = "https://github.com/denis-pplx/autojev.git", "ee63c1515980491a742f0bd0685c8dc5ca1f00c3"
WEIGHTS, REVISION = "denis-pplx/autojev-27b", "6f5b557e037f5edb25c7dc92dbc6553e5a19c015"
BASE, BASE_REVISION = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"   # its processor and config are read from the base

image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .apt_install("git", "curl")
         .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh",
                       f"git clone {REPO} /opt/autojev && cd /opt/autojev && git checkout {COMMIT}",
                       "cd /opt/autojev && /root/.local/bin/uv sync --frozen --python 3.12")
         .env({"HF_HOME": "/hf", "AUTOJEV_HOST": "0.0.0.0", "PORT": "8000", "PATH": "/opt/autojev/.venv/bin:/root/.local/bin:$PATH"}))
hf = modal.Volume.from_name("kev-hf-cache")
app = modal.App("autojev")
secrets = [modal.Secret.from_dict({"AUTOJEV_API_KEY": os.environ.get("AUTOJEV_API_KEY", "")})]


@app.function(image=image, gpu="H200", volumes={"/hf": hf}, secrets=secrets, timeout=6 * 3600, scaledown_window=1200, max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.web_server(8000, startup_timeout=3600)
def api():
    from huggingface_hub import snapshot_download
    checkpoint = snapshot_download(WEIGHTS, revision=REVISION)
    snapshot_download(BASE, revision=BASE_REVISION, allow_patterns=["*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
    hf.commit()
    subprocess.Popen(["/opt/autojev/.venv/bin/autojev-serve"], cwd="/opt/autojev", env={**os.environ, "AUTOJEV_CHECKPOINT": checkpoint})
