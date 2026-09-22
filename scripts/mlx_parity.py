"""MLX backend parity and latency against the fp32 torch path, on this Mac.

    uv run --extra mlx python scripts/mlx_parity.py --run jaredpalmer/kev-0.8b --n 60
    uv run --extra mlx python scripts/mlx_parity.py --run jaredpalmer/kev-4b --n 40 --out runs/mlx-parity-4b/report.json

For n clean development records of decision-v7: probabilities from MLX (full pass, prefix pass, each question alone) vs
the torch fp32 path (max |dp|, argmax flips, calibrated), and median latency of each path. Writes a JSON report
(runs/<name>/report.json is kept by .gitignore; runs/mlx-parity-{4b,0.8b} hold the numbers the README quotes).
"""
import argparse, gc, json, statistics, time

import torch
from pathlib import Path

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import empty_cache
from kev.model import load_tokenizer
from kev.suite import load_split, write_json


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1)


def reference(ck, tok, recs):
    """fp32 torch probabilities for every record plus its latency, with the model released on return: a 4B in fp32 and its
    MLX twin do not fit a 32 GB Mac at once."""
    _, ref = ck.load("mps", LoadOptions(backend="torch"))
    targets = [ref.probs(ref.encode(tok, rec)) for rec in recs]
    enc = ref.encode(tok, recs[0]); _, prefix = ref.probs_and_prefix(enc)
    ms = {"torch_fp32_full": timed(lambda: ref.probs(enc), 5), "torch_fp32_prefix_hit": timed(lambda: ref.probs_with_prefix(enc, prefix), 5)}
    return targets, ms, ref.dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--suite", default="evals/v7/decision-v7"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    recs = [materialize(r) for r in load_split(a.suite, "development") if r["_meta"]["variant"] == "clean"][: a.n]
    ck = Checkpoint(a.run)
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    targets, torch_ms, torch_dtype = reference(ck, tok, recs)
    gc.collect(); empty_cache("mps")
    _, mlx = ck.load("mps", LoadOptions(backend="mlx"))
    report = {"run": a.run, "records": len(recs), "questions": 0, "torch_dtype": torch_dtype, "mlx_dtype": mlx.dtype}
    dp = {"rows": [], "prefix": [], "alone": []}; flips = {"rows": 0, "prefix": 0, "alone": 0}   # rows = the torch row form; prefix = what serving runs
    for rec, target in zip(recs, targets):
        enc = mlx.encode(tok, rec)
        rows = [torch.softmax(z, -1) for z in mlx.forward_rows(enc)]; via_prefix = mlx.probs(enc)
        alone = [mlx.probs(mlx.encode(tok, {"state": rec["state"], "questions": [q]}))[0] for q in rec["questions"]]
        for name, got in (("rows", rows), ("prefix", via_prefix), ("alone", alone)):
            for p, t in zip(got, target):
                dp[name].append(float((p - t).abs().max())); flips[name] += int(p.argmax() != t.argmax())
        report["questions"] += len(target)
    for name in dp:
        report[name] = {"max_dp": max(dp[name]), "mean_dp": statistics.mean(dp[name]), "argmax_flips": flips[name], "over_0.02": sum(d > 0.02 for d in dp[name])}
    enc = mlx.encode(tok, recs[0]); _, prefix = mlx.probs_and_prefix(enc)
    report["latency_ms"] = {"tokens": len(enc["ids"]), "questions_in_record": len(enc["decide_idx"]),
                            "mlx_rows": timed(lambda: mlx.forward_rows(enc), 10), "mlx_miss": timed(lambda: mlx.probs(enc), 10),
                            "mlx_prefix_hit": timed(lambda: mlx.probs_with_prefix(enc, prefix), 10), **torch_ms}
    print(json.dumps(report, indent=1))
    if a.out: write_json(Path(a.out), report)


if __name__ == "__main__":
    main()
