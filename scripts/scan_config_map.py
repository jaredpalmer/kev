"""Which launch config should fla's DeltaNet state-scan kernel use for each pass shape Kev runs?

fla 0.5.2 autotunes chunk_gated_delta_rule_fwd_kernel_h_blockdim64 on the key ['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST']
plus argument dtypes: no sequence length T and no row count N, so one config, picked on the kernel's first call, serves
every later pass. In Kev that first call is CudaGraphs' 1 x 16-token probe pass at model load (config S on an H100:
BV 32, 4 warps, 4 stages); tuned on a batch of question rows it picks another (R: BV 64, 4 warps, 2 stages; both picks
in runs/scan-width-ab-4b-h100-1). Before changing anything we need the whole map: for each shape Kev runs, how long each
candidate config takes.

Shapes, for the DeltaNet head counts of every Kev base (H = 16 key heads, K = V = 128; HV = 16 / 32 / 48 value heads for
Kev-0.8B / 4B and 9B / 27B):
- row passes: N question rows continuing a cached state (initial state read, no final state), short T
- state passes: N new states (no initial state, final state written), T up to the graph bank width, and the long single
  states that run eagerly
Inputs are random but go through fla's own preprocessing (L2 norm, beta sigmoid, gate cumsum, the intra-chunk WY step),
so the scan sees tensors shaped and scaled as in serving. Each config is timed alone (the autotuner given that one
config), and its outputs are compared bit for bit with the first config's.

    uv run modal run modal_app.py::script --script scan_config_map.py --name scan-config-map-h100-1 --gpu H100
"""
import argparse, platform, statistics
from pathlib import Path

import torch
import triton

from fla.modules.l2norm import l2norm_fwd
from fla.ops.common import chunk_delta_h
from fla.ops.common.gate import fused_beta_sigmoid
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.gate import gdn_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2

from kev.device import empty_cache, sync
from kev.suite import write_json

H, K, V = 16, 128, 128
VALUE_HEADS = {16: "Kev-0.8B", 32: "Kev-4B / Kev-9B", 48: "Kev-27B"}
ROW_SHAPES = [(n, t) for n in (1, 6, 16, 32) for t in (16, 64, 256)]                     # (rows, tokens per row)
STATE_SHAPES = [(n, t) for n in (1, 4, 16) for t in (256, 1024, 4096)] + [(1, 16384), (1, 65536)]   # (states, tokens)
S_CONFIG = {"BV": 32, "num_warps": 4, "num_stages": 4}   # what Kev's probe pass picks on Kev-4B
R_CONFIG = {"BV": 64, "num_warps": 4, "num_stages": 2}   # what a row pass picks


def autotuner(kernel):
    """The Triton Autotuner inside a decorated kernel: @triton.heuristics wraps it, so look through .fn."""
    while not hasattr(kernel, "configs"): kernel = kernel.fn
    return kernel


def describe(cfg):
    return {"BV": cfg.kwargs["BV"], "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}


def name(cfg):
    d = describe(cfg)
    return f"BV{d['BV']}/w{d['num_warps']}/s{d['num_stages']}"


def inputs(n, t, hv, seed=0):
    """The scan's inputs for n sequences of t tokens, through fla's preprocessing as fused_qwen35 calls it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    def randn(*shape): return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    k, _ = l2norm_fwd(randn(n, t, H, K))
    v = randn(n, t, hv, V)
    beta = fused_beta_sigmoid(randn(n, t, hv), scale=1.0)
    a_log = torch.log(torch.empty(hv, device="cuda").uniform_(1, 16, generator=g))   # Qwen3.5's A_log init range
    gate = gdn_gate_chunk_cumsum(g=randn(n, t, hv), A_log=a_log, chunk_size=64, scale=RCP_LN2, dt_bias=torch.zeros(hv, device="cuda"))
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=gate, beta=beta, chunk_size=64)
    return k, w, u, gate


def scan(k, w, u, gate, h0, final):
    return chunk_delta_h.chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=gate, initial_state=h0, output_final_state=final, chunk_size=64)


def time_ms(fn, warmup, reps):
    for _ in range(warmup): fn()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def measure(kind, n, t, hv, configs, warmup, reps):
    """One shape: median ms per config, whether each config's outputs equal the first's bit for bit, autotune's own pick."""
    k, w, u, gate = inputs(n, t, hv)
    h0 = torch.randn((n, hv, K, V), device="cuda", dtype=torch.float32) if kind == "row" else None
    final = kind == "state"
    tuner = autotuner(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    times, same, reference = {}, {}, None
    try:
        for cfg in configs:
            tuner.configs = [cfg]   # one config: the autotuner launches it without benchmarking
            times[name(cfg)] = round(time_ms(lambda: scan(k, w, u, gate, h0, final), warmup, reps), 4)
            out = [x for x in scan(k, w, u, gate, h0, final) if x is not None]
            if reference is None: reference = out
            same[name(cfg)] = all(torch.equal(a, b) for a, b in zip(out, reference))
        tuner.configs, tuner.cache = list(configs), {}   # all configs, nothing cached: autotune on this shape alone
        scan(k, w, u, gate, h0, final); sync("cuda")
        pick = name(tuner.best_config)
    finally:
        tuner.configs, tuner.cache = list(configs), {}
    best = min(times, key=times.get)
    programs = {name(c): triton.cdiv(V, c.kwargs["BV"]) * n * hv for c in configs}
    return {"kind": kind, "n": n, "t": t, "hv": hv, "model": VALUE_HEADS[hv], "ms": times, "bit_identical": same, "best": best,
            "best_ms": times[best], "autotune_pick": pick, "programs": programs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--heads", default="16,32,48", help="value-head counts to measure")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    configs = list(autotuner(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64).configs)   # fla's candidates for this GPU
    p = torch.cuda.get_device_properties(0)
    report = {"environment": {"gpu": p.name, "sms": p.multi_processor_count, "torch": torch.__version__, "triton": triton.__version__,
                              "cuda": torch.version.cuda, "python": platform.python_version()},
              "configs": [describe(c) for c in configs], "S": S_CONFIG, "R": R_CONFIG, "shapes": []}
    print("configs:", [name(c) for c in configs], flush=True)
    s, r = (next(name(c) for c in configs if describe(c) == x) for x in (S_CONFIG, R_CONFIG))
    for hv in map(int, a.heads.split(",")):
        for kind, shapes in (("row", ROW_SHAPES), ("state", STATE_SHAPES)):
            for n, t in shapes:
                m = measure(kind, n, t, hv, configs, a.warmup, a.reps)
                m["S_over_best"], m["R_over_best"] = round(m["ms"][s] / m["best_ms"], 3), round(m["ms"][r] / m["best_ms"], 3)
                report["shapes"].append(m)
                write_json(out / "report.json", report)
                print(f"{kind:>5} HV={hv:<2} N={n:<2} T={t:<5}  best {m['best']:>11} {m['best_ms']:8.3f} ms  S x{m['S_over_best']:<5}  "
                      f"R x{m['R_over_best']:<5} autotune {m['autotune_pick']:>11}  bits {'same' if all(m['bit_identical'].values()) else 'DIFFER'}", flush=True)
                empty_cache("cuda")
    report["all_bit_identical"] = all(all(m["bit_identical"].values()) for m in report["shapes"])
    write_json(out / "report.json", report)
    print("all configs bit-identical" if report["all_bit_identical"] else "SOME CONFIGS CHANGE BITS", flush=True)


if __name__ == "__main__":
    main()
