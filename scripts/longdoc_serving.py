"""Long-document serving cost on CUDA, for one checkpoint on evals/longdoc-v1 development records.

    uv run modal run --detach modal_app.py::script --script longdoc_serving.py --name longdoc-serving-27b-h200 --gpu H200 \
        --args "--run jaredpalmer/kev-27b"

K records of each part per bucket (--per-bucket) through kev.serve.Server as `kev.serve` loads a checkpoint on CUDA (bf16,
fused kernels, CUDA graphs) and at its default request limits (kev.model.SERVE_MAX_STATE: 64k-token states). Each request is
a new state (the prefix cache is cleared before it, so its peak memory is its own); latency is the server's model time
(latency_ms) and the wall time of Server.probs; peak memory is kev.device.allocated_bytes (CUDA: the peak) during the
request, and resident memory the weights and graph buffers before it. One repeat of the last record per bucket measures a
cached state. Writes <out>/report.json. (Scoring parity, exact path vs the long-row path, is read from the benchmark rows:
scripts/longdoc_report.py --parity.)
"""
import argparse, json, statistics, time
from pathlib import Path

import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import allocated_bytes, empty_cache
from kev.serve import Server
from kev.suite import load_split, write_json

GB = 1e9


def pick(records, bucket, part, k):
    return [r for r in records if r["_meta"]["bucket"] == bucket and r["_meta"]["part"] == part][:k]


def serving(run, records, k):
    ck = Checkpoint(run)
    t = time.time()
    tok, m = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
    report = {"load_seconds": round(time.time() - t, 1), "dtype": m.dtype, "temperature": m.head.temperature}
    server = Server(ck, tok, m, "cuda", release_date="-")
    try:
        warm = pick(records, 4096, "synthetic", 1)[0]
        for _ in range(3): server.probs(materialize(warm))   # first-call autotuning and graph captures happen here
        server.wait_idle()
        report["resident_gb"] = round(torch.cuda.memory_allocated() / GB, 2)
        buckets = {}
        for bucket in (4096, 8192, 16384, 32768, 65536):
            rows = []
            for part in ("cuad", "synthetic"):
                for r in pick(records, bucket, part, k):
                    rec = materialize(r)
                    with server.lock: server.prefix_cache.clear()
                    empty_cache("cuda"); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
                    t = time.perf_counter(); _, stats = server.probs(rec); wall = 1000 * (time.perf_counter() - t)
                    rows.append({"id": r["_meta"]["id"], "state_tokens": stats["state_tokens"], "questions": len(r["questions"]),
                                 "latency_ms": stats["latency_ms"], "wall_ms": round(wall, 1),
                                 "peak_gb": round(allocated_bytes("cuda") / GB, 2), "peak_over_resident_gb": round((allocated_bytes("cuda") - base) / GB, 2)})
                    print(bucket, rows[-1], flush=True)
            _, hit = server.probs(rec)   # the last record again: its state is cached now
            lat = [x["latency_ms"] for x in rows]
            buckets[f"{bucket // 1024}k"] = {"requests": len(rows), "state_tokens_median": statistics.median(x["state_tokens"] for x in rows),
                                             "latency_ms_median": statistics.median(lat), "latency_ms_max": max(lat),
                                             "ms_per_1k_state_tokens": round(statistics.median(x["latency_ms"] / x["state_tokens"] * 1000 for x in rows), 2),
                                             "peak_gb_max": max(x["peak_gb"] for x in rows), "peak_over_resident_gb_max": max(x["peak_over_resident_gb"] for x in rows),
                                             "cached_state_latency_ms": hit["latency_ms"], "cached_state_hit": hit["prefix_cache_hit"], "requests_detail": rows}
        report["buckets"] = buckets
        report["cuda_graphs"] = m.graphs.stats() if m.graphs is not None else None
        report["gpu"] = torch.cuda.get_device_name(0)
        report["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / GB, 1)
    finally:
        server.close()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-27b")
    ap.add_argument("--suite", default="evals/longdoc-v1")
    ap.add_argument("--per-bucket", type=int, default=3, help="served requests per part and bucket")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    records = load_split(a.suite, "development")
    report = {"run": a.run, "suite": a.suite, "split": "development", "serving": serving(a.run, records, a.per_bucket)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out) / "report.json", report)
    print(json.dumps({k: v for k, v in report["serving"].items() if k != "buckets"}, indent=1))


if __name__ == "__main__":
    main()
