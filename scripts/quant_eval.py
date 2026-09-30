"""One precision of a checkpoint as the KEV_* variables load it (bf16, or KEV_QUANT's int8 / nf4, optionally from a KEV_BASE
saved quantized) with one load: kev.benchmark's scoring on the development split of each suite, then the served path
(kev.serve.Server; CUDA graphs unless KEV_CUDA_GRAPHS=0, checked against eager) on serving_bench's request shapes.
Compare the precisions with scripts/quant_compare.py.

    KEV_QUANT=nf4 python scripts/quant_eval.py --run jaredpalmer/kev-27b --out runs/quant/nf4
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from serving_bench import CASES, latency, request   # noqa: E402

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.device import allocated_bytes
from kev.predictors import LocalPredictor
from kev.serve import Server
from kev.suite import CONTEXT, load_split, read_manifest, write_json


def floats(x):
    if isinstance(x, dict): return [f for k in sorted(x) for f in floats(x[k])]
    if isinstance(x, list): return [f for v in x for f in floats(v)]
    return [x] if isinstance(x, float) else []


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--suites", default="evals/v4/transfer-v4,evals/v7/decision-v7")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)

opts = LoadOptions.from_env()
if opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)
start = time.time()
p = LocalPredictor(a.run, "cuda", opts)
report = {"quant": opts.quant or "bf16", "base": opts.base, "cuda_graphs": opts.cuda_graphs, "gpu": torch.cuda.get_device_name(),
          "load_s": round(time.time() - start), "loaded_gib": torch.cuda.memory_allocated() / 2**30}
print(json.dumps(report), flush=True)
torch.cuda.reset_peak_memory_stats()
for suite in a.suites.split(","):
    manifest = read_manifest(suite)
    p.context = manifest.get("context", CONTEXT)
    r, _ = evaluate_records(load_split(suite, "development"), p, out / Path(suite).name, heldout_sources=tuple(manifest["holdout_sources"]))
    report[Path(suite).name] = {"clean": r["clean"], "latency_ms": r["latency_ms"]}
    print(json.dumps(report), flush=True)
report["eval_peak_gib"] = allocated_bytes("cuda") / 2**30
write_json(out / "report.json", report)

# LocalPredictor turned these off for fp32-exact scoring; the served path runs with them
torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True)
server = Server(p.checkpoint, p.tok, p.model, "cuda")
graphs, p.model.graphs = p.model.graphs, None
eager = {case: floats(server.answer(request(case, 0))["answers"]) for case in CASES}
p.model.graphs = graphs
report["served_latency"] = latency(server, reps=5)
if graphs is not None:
    report["graphs_vs_eager_max_dp"] = max(max(abs(x - y) for x, y in zip(eager[c], floats(server.answer(request(c, 0))["answers"]))) for c in CASES)
report["peak_gib"] = allocated_bytes("cuda") / 2**30
write_json(out / "report.json", report)
print("QUANT_EVAL_DONE", json.dumps(report), flush=True)
