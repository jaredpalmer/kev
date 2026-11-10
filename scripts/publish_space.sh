#!/usr/bin/env bash
# Publish the Hugging Face Space (space/ + the kev modules it imports) as one commit.
#   scripts/publish_space.sh [repo_id] [commit message]
# Needs `hf auth login`. First time: hf repos create jaredpalmer/kev --type space --sdk gradio --flavor zero-a10g --public
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${1:-jaredpalmer/kev}"
MSG="${2:-Update Space from $(git rev-parse --short HEAD)}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp space/app.py space/presets.py space/requirements.txt space/README.md "$STAGE/"
mkdir -p "$STAGE/kev"
cp kev/__init__.py kev/model.py kev/api.py kev/checkpoint.py "$STAGE/kev/"
python3 -m py_compile "$STAGE/app.py" "$STAGE/presets.py"
hf upload "$REPO" "$STAGE" . --type space --commit-message "$MSG" --exclude "**/__pycache__/**"
echo "https://huggingface.co/spaces/$REPO"
