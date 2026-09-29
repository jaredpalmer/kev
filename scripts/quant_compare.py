"""Per-question parity of quantized runs against the bf16 run (scripts/quant_eval.py outputs), plus each run's metrics.

    python scripts/quant_compare.py runs/quant      # expects runs/quant/{bf16,int8,nf4}/<suite>/predictions.jsonl
"""
import sys
from pathlib import Path

from kev.suite import read_json, read_jsonl


def probs(path):
    out = {}
    for rec in read_jsonl(path):
        for qid, dist in rec["prediction"]["probabilities"].items():
            out[rec["id"], qid] = [dist[k] for k in sorted(dist)]
    return out


def parity(ref_file, got_file):
    """-> (max |dp|, mean |dp|, argmax flips, questions) of got against ref, per question its largest |dp|."""
    ref, got = probs(ref_file), probs(got_file)
    assert ref.keys() == got.keys(), f"{got_file}: question sets differ from {ref_file}"
    dps = [max(abs(x - y) for x, y in zip(ref[k], got[k])) for k in ref]
    flips = sum(max(range(len(ref[k])), key=ref[k].__getitem__) != max(range(len(got[k])), key=got[k].__getitem__) for k in ref)
    return max(dps), sum(dps) / len(dps), flips, len(dps)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    ref_dir = root / "bf16"
    for q in ("bf16", "int8", "nf4"):
        if not (root / q / "report.json").exists(): print(f"{q}: no report.json"); continue
        rep = read_json(root / q / "report.json")
        print(f"{q}: load {rep['load_s']} s, weights {rep['loaded_gib']:.1f} GiB, eval peak {rep['eval_peak_gib']:.1f} GiB, "
              f"served graphs-vs-eager max|dp| {rep.get('graphs_vs_eager_max_dp', 'n/a')}")
        for suite in sorted(p.name for p in ref_dir.iterdir() if (p / "predictions.jsonl").exists()):
            c = rep[suite]["clean"]
            line = f"  {suite}: acc {c['acc']:.3f} brier {c['brier']:.3f} ece {c['ece']:.3f} n {c['n']} median {rep[suite]['latency_ms']['median']:.0f} ms"
            if q != "bf16":
                mx, mean, flips, n = parity(ref_dir / suite / "predictions.jsonl", root / q / suite / "predictions.jsonl")
                line += f" | vs bf16: max|dp| {mx:.3f} mean|dp| {mean:.4f} flips {flips}/{n}"
            print(line)
        print("  served ms (new state):", {case: round(v["new_ms"]) for case, v in rep.get("served_latency", {}).items()})
