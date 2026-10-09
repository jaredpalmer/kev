#!/usr/bin/env bash
# Isolated CUDA serving with Kev's pinned fused kernels and explicit cache paths.
set -euo pipefail

usage() {
  echo 'Usage: local_cuda.sh setup|check|serve --env DIR --cache DIR [--python PATH] [-- serve arguments]' >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
action="$1"
shift
env_dir=''
cache_dir=''
python='3.13'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|--cache|--python)
      [[ $# -ge 2 ]] || usage
      case "$1" in
        --env) env_dir="$2" ;;
        --cache) cache_dir="$2" ;;
        --python) python="$2" ;;
      esac
      shift 2 ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
case "$action" in setup|check|serve) ;; *) usage ;; esac
[[ -n "$env_dir" && -n "$cache_dir" ]] || usage
[[ "$action" == serve || $# == 0 ]] || usage
mkdir -p "$cache_dir" "$cache_dir/tmp"
cache_dir="$(cd "$cache_dir" && pwd -P)"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

export UV_CACHE_DIR="$cache_dir/uv"
export UV_PYTHON_INSTALL_DIR="$cache_dir/python"
export HF_HOME="${HF_HOME:-$cache_dir/huggingface}"
export TRITON_CACHE_DIR="$cache_dir/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_dir/inductor"
export TORCH_HOME="$cache_dir/torch"
export CUDA_CACHE_PATH="$cache_dir/cuda"
export XDG_CACHE_HOME="$cache_dir/xdg"
export TMPDIR="$cache_dir/tmp"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_LINK_MODE=copy

if [[ "$action" == setup ]]; then
  [[ ! -e "$env_dir/pyvenv.cfg" ]] || { echo 'Refusing to modify an existing environment; select a new --env directory.' >&2; exit 2; }
  uv venv --python "$python" "$env_dir"
  uv pip install --python "$env_dir/bin/python" 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$env_dir/bin/python" -e "$root[serve]" 'torch==2.8.0' socksio einops --index-url https://pypi.org/simple
  # Torch pins Triton 3.4; Kev's Hopper fused path explicitly overrides that pin.
  uv pip install --python "$env_dir/bin/python" --no-deps \
    'flash-linear-attention==0.5.2' 'fla-core==0.5.2' 'triton==3.7.1' --index-url https://pypi.org/simple
fi

[[ -x "$env_dir/bin/python" ]] || { echo "Environment Python missing: $env_dir/bin/python" >&2; exit 2; }
env_dir="$(cd "$env_dir" && pwd -P)"
"$env_dir/bin/python" "$root/scripts/cuda_preflight.py"
if [[ "$action" == serve ]]; then
  export KEV_FUSED=1
  export KEV_CUDA_GRAPHS=1
  export KEV_DTYPE=bf16
  export KEV_BACKEND=torch
  export PYTHONUNBUFFERED=1
  cd "$root"
  exec "$env_dir/bin/python" -m kev.serve "$@"
fi
