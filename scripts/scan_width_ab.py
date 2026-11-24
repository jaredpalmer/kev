"""Does autotuning fla's DeltaNet state scan separately for wide passes (kev.fused_qwen35._TunedByWidth) make served
passes faster than main, where one config, picked on CudaGraphs' 1 x 16 probe at load, runs every pass?

Two arms on one loaded model (bf16, fused kernels, CUDA graphs), in blocks interleaved main, width, main, width, ...
(--rounds), ending on main. Each block starts from nothing, as a load does: empty autotune caches and new CudaGraphs, whose probe pass
tunes.
  main   every pass tuned together: fla's own behaviour
  width  passes of WIDE_SCAN or more sequences tuned apart from the rest
Workloads, each through DecisionModel.probs_batch (what kev.serve's model thread calls):
  rows         16 cached short states x 6 questions: question-row passes only
  states       the same 16 requests with their states new: the graphed state pass, then the rows
  long-<n>     one new state of about n tokens, 5 questions: an eager state pass, then graphed rows; each rep a new state
Per workload and block: GPU time per batch (CUDA events, median of --reps) after 3 warm-up batches and their captures;
the first warm-up batch's wall time (it pays any autotune); probabilities compared with the first block's bit for bit;
the config each autotune cache picked (True = wide).

Decision rule, written before the first run: a workload is faster with the width split when, in every round, the width block's
median is below both neighbouring main blocks' and the mean gain over the main blocks is more than twice the largest
difference between two main blocks' medians (the drift).

    uv run modal run modal_app.py::script --script scan_width_ab.py --name scan-width-ab-4b-h100-1 --gpu H100
"""
import argparse, gc, statistics, time
from pathlib import Path

import torch
from fla.ops.common import chunk_delta_h

import kev.fused_qwen35 as fused
from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.cuda_graphs import CudaGraphs
from kev.device import empty_cache, sync
from kev.model import admit, rows_of
from kev.predictors import kernel_environment
from kev.suite import write_json
from scripts.serving_bench import FIVE, PARAGRAPH, request

RUN = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"   # Kev-4B as released when this was measured
WIDE_SCAN = fused.WIDE_SCAN


def scan():
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    if not isinstance(kernel, fused._TunedByWidth): raise RuntimeError("fuse() did not wrap fla's state scan")
    return kernel


def config_name(cfg):
    return f"BV{cfg.kwargs['BV']}/w{cfg.num_warps}/s{cfg.num_stages}"


def start_block(model, arm):
    """Empty autotune caches and new graphs (whose probe pass tunes first), passes split as the arm says."""
    fused.WIDE_SCAN = WIDE_SCAN if arm == "width" else float("inf")   # main: no pass is wide
    scan().caches = {}
    model.graphs = None
    gc.collect(); empty_cache("cuda")
    model.graphs = CudaGraphs(model.lm, model.pad_id)


def picks():
    return {str(c): {str(k): config_name(cfg) for k, cfg in cache.items()} for c, cache in scan().caches.items()}


def gpu_ms(fn):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(); out = fn(); end.record(); end.synchronize()
    return begin.elapsed_time(end), out


def flat(probs):
    return [p.float().cpu() for ps in probs for p in ps]


def measure(model, batches, reps):
    """batches(i) -> a probs_batch call for rep i. 3 warm-up calls (the first timed by wall clock), each followed by a
    capture of what ran eagerly, then `reps` timed calls."""
    first = None
    for i in range(3):
        start = time.perf_counter()
        batches(i)()
        sync("cuda")
        if first is None: first = (time.perf_counter() - start) * 1000
        model.graphs.capture_pending()
    times, probs = [], []
    for i in range(reps):
        ms, (p, _) = gpu_ms(batches(3 + i))
        times.append(ms); probs += flat(p)
    return {"median_gpu_ms": round(statistics.median(times), 3), "gpu_ms": [round(x, 3) for x in times], "first_wall_ms": round(first, 1)}, probs


def verdict(blocks, workload):
    """The decision rule above, for one workload over the blocks in run order."""
    seq = [(b["arm"], b["workloads"][workload]["median_gpu_ms"]) for b in blocks if "median_gpu_ms" in b["workloads"].get(workload, {})]
    main = [ms for arm, ms in seq if arm == "main"]
    wins, gains = [], []
    for i, (arm, ms) in enumerate(seq):
        if arm != "width": continue
        around = [m for a, m in seq[max(0, i - 1):i + 2] if a == "main"]
        if not around: continue
        wins.append(all(ms < m for m in around)); gains.append(statistics.mean(around) - ms)
    drift = max(main) - min(main) if len(main) > 1 else None
    gain = statistics.mean(gains) if gains else None
    return {"main_ms": main, "width_ms": [ms for arm, ms in seq if arm == "width"], "mean_gain_ms": None if gain is None else round(gain, 3),
            "drift_ms": None if drift is None else round(drift, 3), "every_round": bool(wins) and all(wins),
            "faster": bool(wins) and all(wins) and drift is not None and gain > 2 * drift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--reps", type=int, default=30, help="timed batches per short workload and block")
    ap.add_argument("--long", default="16384,61440", help="approximate state tokens of the long workloads ('' for none)")
    ap.add_argument("--long-reps", type=int, default=3, help="timed new long states per length and block")
    ap.add_argument("--rounds", type=int, default=3, help="main + width block pairs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok, model = Checkpoint(a.run).load("cuda", LoadOptions(dtype=torch.bfloat16, fused=True, cuda_graphs=True))
    scan()
    n = a.requests
    short = [admit(model, tok, to_record(request("6 questions, short state", 1 + i))[0]) for i in range(n)]   # distinct states
    lengths = [int(x) for x in a.long.split(",") if x]
    paragraph = len(tok(PARAGRAPH, add_special_tokens=False).input_ids)
    long = {t: [admit(model, tok, to_record(SystemOneRequest.model_validate({"state": f"Ticket {i}. " + PARAGRAPH * (t // paragraph), "questions": FIVE}))[0])
                for i in range(3 + a.long_reps)] for t in lengths}   # 3 warm-up states, then a new one per timed rep
    report = {"run": a.run, "environment": kernel_environment(model, "cuda"), "cuda": torch.version.cuda, "requests": n,
              "long_state_tokens": {f"long-{t}": len(rows_of(long[t][0])[0]) for t in lengths},
              "blocks": [], "reference_questions": {}}
    reference = {}

    def record(block, name, result, probs):
        if name not in reference: reference[name] = probs; report["reference_questions"][name] = len(probs)
        ref = reference[name]
        result["vs_first_block"] = {"bit_identical": len(probs) == len(ref) and all(torch.equal(p, q) for p, q in zip(probs, ref)),
                                    "max_dp": max(float((p - q).abs().max()) for p, q in zip(probs, ref)),
                                    "flips": sum(int(p.argmax() != q.argmax()) for p, q in zip(probs, ref))}
        block["workloads"][name] = result

    for r, arm in enumerate(["main", "width"] * a.rounds + ["main"]):   # ends on main: every width block has a main block on each side
        r //= 2
        start_block(model, arm)
        block = {"round": r, "arm": arm, "workloads": {}}
        report["blocks"].append(block)
        try:
            _, prefixes = model.probs_batch(short, [None] * n, [True] * n)
            result, probs = measure(model, lambda i: lambda: model.probs_batch(short, prefixes, [False] * n), a.reps)
            record(block, "rows", result, probs)
            result, probs = measure(model, lambda i: lambda: model.probs_batch(short, [None] * n, [False] * n), a.reps)
            record(block, "states", result, probs)
            del prefixes
            for t in lengths:
                try:
                    result, probs = measure(model, lambda i: lambda: model.probs_batch([long[t][i]], [None], [False]), a.long_reps)
                    record(block, f"long-{t}", result, probs)
                except torch.OutOfMemoryError as e:
                    block["workloads"][f"long-{t}"] = {"error": f"OutOfMemoryError: {str(e)[:300]}"}
                    gc.collect(); empty_cache("cuda")
        finally:
            block["picks"], block["graphs"] = picks(), model.graphs.stats()
            write_json(out / "report.json", report)
        print(f"round {r} {arm:>5}: " + ", ".join(f"{w} {v.get('median_gpu_ms', v.get('error'))}" for w, v in block["workloads"].items())
              + f"  picks {block['picks']}", flush=True)
    report["verdict"] = {w: verdict(report["blocks"], w) for w in ["rows", "states", *(f"long-{t}" for t in lengths)]}
    report["all_bit_identical"] = all(v["vs_first_block"]["bit_identical"] for b in report["blocks"] for v in b["workloads"].values() if "vs_first_block" in v)
    fused.WIDE_SCAN = WIDE_SCAN
    write_json(out / "report.json", report)
    for w, v in report["verdict"].items(): print(w, v, flush=True)
    print("all blocks bit-identical" if report["all_bit_identical"] else "SOME BLOCKS DIFFER", flush=True)


if __name__ == "__main__":
    main()
