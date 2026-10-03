#!/usr/bin/env bash
# Serve a Qwen3.5 Kev checkpoint as a System One endpoint on one Ascend NPU (docs/ascend-npu.md).
#
#   scripts/serve_kev_npu.sh <card> [port] [run] [kev.serve flags...]
#   scripts/serve_kev_npu.sh 0                                  # card 0, port 8009, jaredpalmer/kev-27b@v1-lora
#   scripts/serve_kev_npu.sh 1 8010 jaredpalmer/kev-9b@v1       # Kev-9B on card 1
#   scripts/serve_kev_npu.sh 0 8009 jaredpalmer/kev-27b@v1-lora --host 0.0.0.0
#
# The default run is the Kev-27B the docs' numbers were measured on (the v1 LoRA). The current `jaredpalmer/kev-27b`
# (v2, full weights) and `jaredpalmer/kev-9b` take the same path, since fuse covers full weights as well as a merged
# adapter, but have not been measured on an NPU.
#
# The card number is the physical one `npu-smi info` prints. It goes into ASCEND_RT_VISIBLE_DEVICES, so the process
# sees that card as npu:0 and cannot touch any other; the script refuses to start if something already runs on it.
# Kev-27B holds 47.7 GiB of bf16 weights and a 910B2 gives a process about 61 GiB, so one card serves one 27B.
#
# Run it in an environment that has CANN's set_env.sh sourced and torch_npu, vllm-ascend and kev's serve dependencies
# installed into the system torch (do not `uv sync` there: it would replace torch_npu's torch). PYTHON picks the
# interpreter (default python3). HF_HOME, HF_HUB_OFFLINE and any proxy settings are inherited, not set here.
#
# The server listens on 127.0.0.1. Pass --host 0.0.0.0 to expose it, and set KEV_API_KEY when you do: with it set
# every request needs `authorization: Bearer <key>`; unset, the server is open to whoever can reach it.
#
# The first request of each new shape (rows x padded length) is slow -- about 1.9 s, then 1.0 s, then 150 ms for a
# two-question 55-token state on Kev-27B -- because that shape's kernels compile and its graph is captured. That is
# per shape, once, for the life of the process.
#
# Check it is up:
#   curl -s localhost:8009/v1/models | head
#   curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
#     "state": "The parcel arrived late and the shoes are the wrong size.",
#     "questions": {"route": {"type": "choice", "instructions": "Which team should handle this?",
#                             "criteria": {"returns": "a return or exchange", "shipping": "a delivery problem"}}}}'
#
# --- what the environment variables do ---
#
# KEV_FUSED=1 (default, set 0 to decline)
#   Rewrites the merged Qwen3.5 decoder layers with the Ascend fused ops and replays passes from captured graphs
#   (kev/npu_qwen35.py): Kev-27B 422 ms -> 116 ms median on decision-v7 development records. A LoRA checkpoint is
#   merged into the bf16 weights first, as on CUDA. Off gives the reference layers (docs/ascend-npu.md).
#
# KEV_NPU_GRAPHS=1 (default, set 0 for long states -- see below)
#   Whether the fused model replays a pass from a captured Ascend graph instead of enqueueing every kernel.
#
# KEV_NPU_KERNELS=1 (default, set 0 to decline)
#   The Gated DeltaNet mixer on vllm-ascend's kernels instead of the pure-PyTorch fallback. Off is ~3x slower again.
#
# The state-prefix cache (KEV_PREFIX_CACHE) is always off on an NPU: reusing a cached state would continue the DeltaNet
# recurrence through a kernel kev.npu_qwen35 does not replace, so every request runs its state.
#
# KEV_DTYPE, KEV_ATTN, KEV_MERGE, KEV_TEMPERATURE
#   bf16 / eager / merged / the checkpoint's own temperature are what the measurements used. KEV_DTYPE=fp32 is the
#   exact reference path and does not fit a 27B on a 64 GB card. KEV_TEMPERATURE=1.0 serves raw logits.
#
# --- graphs and long states: why the one you turn OFF buys you context ---
#
#   1. KEV_FUSED=1 RAISES the ceiling, and you want it on. The stock transformers attention materialises a
#      [rows, 1, L, L] float mask and an [L, L] score matrix per head; that ran a 6k-token Kev-27B state out of memory
#      on a 64 GB card. The fused attention reads a 2,048 x 2,048 compressed causal mask instead and never builds the
#      scores, so memory grows with the state, not with its square.
#
#   2. KEV_NPU_GRAPHS=1 LOWERS it. A captured graph keeps its intermediate buffers in a memory pool that is never
#      handed back to the allocator. Each is only 2-10 MiB, but they accumulate, and a 32k-token Kev-27B pass needs
#      ~8.6 GiB of activations at once out of the ~13 GiB left after the weights: a fresh process ran 32k, the same
#      process after capturing graphs for short requests ran out of memory on it. Graphs are only captured up to 4,096
#      tokens, because past that a pass is bound by the NPU rather than by launch overhead, so a long-state workload
#      gives up nothing by disabling them.
#
# So: default (graphs on) for ordinary decision requests; KEV_NPU_GRAPHS=0 when states above ~16k tokens have to work.
# A 64k-token Kev-27B state does not fit a 64 GB card either way.
set -euo pipefail

CARD="${1:?usage: $0 <card> [port] [run] [kev.serve flags...]   (card is the physical NPU npu-smi prints)}"
PORT="${2:-8009}"
RUN="${3:-jaredpalmer/kev-27b@v1-lora}"
PYTHON="${PYTHON:-python3}"

cd "$(dirname "$0")/.."

if ! npu-smi info | sed -n '/Process id/,$p' | grep -q "No running processes found in NPU ${CARD}\b"; then
    echo "NPU ${CARD} already has a process (or does not exist); pick a free card:" >&2
    npu-smi info | sed -n '/Process id/,$p' >&2
    exit 1
fi

# before the model load, not after it: uvicorn binds last, so a taken port otherwise costs a full load
"${PYTHON}" - "${PORT}" <<'PY'
import socket, sys
with socket.socket() as s:
    try:
        s.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        sys.exit(f"port {sys.argv[1]} is already in use (another kev.serve?); pass a different port")
PY

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"          # the process sees this card, and only this card, as npu:0
export KEV_FUSED="${KEV_FUSED:-1}"
export KEV_NPU_GRAPHS="${KEV_NPU_GRAPHS:-1}"
export KEV_NPU_KERNELS="${KEV_NPU_KERNELS:-1}"

echo "serving ${RUN} on physical NPU ${CARD}, port ${PORT}" \
     "(fused=${KEV_FUSED} graphs=${KEV_NPU_GRAPHS} deltanet_kernels=${KEV_NPU_KERNELS})"
exec "${PYTHON}" -m kev.serve --run "${RUN}" --device npu:0 --host 127.0.0.1 --port "${PORT}" "${@:4}"
