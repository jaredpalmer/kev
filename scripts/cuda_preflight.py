"""Compile and execute the kernels needed for local fused CUDA serving before loading weights."""
import json
import sys


def check():
    import torch
    import triton
    import fla
    from kev.checkpoint import fused_available
    from kev.device import sync
    from fla.modules.conv import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    if not fused_available():
        raise RuntimeError("Kev's pinned fused kernels are not available in this environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check CUDA_VISIBLE_DEVICES and the host driver")
    torch.manual_seed(0)
    device = "cuda"
    with torch.no_grad():
        x = torch.randn(1, 128, 64, device=device, dtype=torch.bfloat16)
        weight = torch.randn(64, 4, device=device, dtype=torch.bfloat16)
        conv, _ = causal_conv1d(x, weight, activation="silu")
        q = torch.randn(1, 128, 4, 64, device=device, dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        g = torch.full((1, 128, 4), -0.1, device=device, dtype=torch.float32)
        beta = torch.full((1, 128, 4), 0.5, device=device, dtype=torch.bfloat16)
        delta, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        sync(device)
        if not torch.isfinite(conv).all() or not torch.isfinite(delta).all():
            raise RuntimeError("Fused kernel smoke test returned nonfinite values")
    return {
        "status": "ok", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__, "fla": fla.__version__,
        "checks": ["fused_import", "causal_conv1d", "chunk_gated_delta_rule"],
    }


if __name__ == "__main__":
    try:
        print(json.dumps(check()))
    except Exception as error:
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        raise SystemExit(1)
