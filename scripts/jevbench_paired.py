"""JevBench public items, candidate against reference, from the harness's committed per-item outcomes (report only; no
JevBench item is ever used for selection). Writes the one JSON a model card's JevBench numbers point at.

    uv run python scripts/jevbench_paired.py --candidate runs/jevbench-public/kev-4b-r10 \
        --reference runs/jevbench-public/kev-4b-r8 --out runs/jevbench-public/kev-4b-r10/paired-vs-kev-4b-r8.json
    uv run python scripts/jevbench_paired.py --candidate runs/jevbench-public/kev-08b-r15 \
        --reference runs/jevbench-public/kev-08b --out runs/jevbench-public/kev-08b-r15/paired-vs-kev-08b.json

Per tier (easy / original = "standard" / hard): accuracy and ECE as the harness's summary-<tier>.json reports them.
"all": pooled accuracy over every public item (sum of n_correct over sum of n_scorable). Paired, per tier, on items both
runs scored: accuracy delta with an item bootstrap (every JevBench item is its own group; 10,000 resamples, seed 0,
percentile 95 % interval: with a handful of discordant items the percentile sits on a boundary between two attainable
deltas, and 2,000 resamples flip the Kev-0.8B pair's upper bound between +7.2 and +8.1 with the seed; 10,000 do not), discordant counts and the exact two-sided McNemar p.
"""
import argparse, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import read_json, read_jsonl, write_json  # noqa: E402

TIERS = ("easy", "original", "hard")
SAMPLES, SEED = 10000, 0


def outcomes(run, tier):
    rows = read_jsonl(Path(run) / f"{tier}.jsonl")
    out = {r["task_id"]: bool(r["correct"]) for r in rows if r.get("valid")}
    if len(out) != sum(1 for r in rows if r.get("valid")):
        raise SystemExit(f"{run}/{tier}.jsonl: duplicate task_id")
    return out


def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(cand, ref):
    keys = sorted(cand.keys() & ref.keys())
    a, b = np.array([cand[k] for k in keys], float), np.array([ref[k] for k in keys], float)
    d = a - b
    idx = np.random.default_rng(SEED).integers(0, len(d), (SAMPLES, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    newly_right, newly_wrong = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"n": len(keys), "delta": float(d.mean()), "ci95": [float(lo), float(hi)], "newly_right": newly_right,
            "newly_wrong": newly_wrong, "mcnemar_exact_p": mcnemar_exact(newly_right, newly_wrong)}


def side(run):
    tiers = {}
    for t in TIERS:
        s = read_json(Path(run) / f"summary-{t}.json")
        tiers[t] = {"n": s["n_scorable"], "n_correct": s["n_correct"], "accuracy": s["accuracy"], "ece": s["ece"]["ece"], "brier": s["brier_mean"]}
    n, k = sum(v["n"] for v in tiers.values()), sum(v["n_correct"] for v in tiers.values())
    return {"run": str(run), "tiers": tiers, "all": {"n": n, "n_correct": k, "accuracy": k / n}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True); ap.add_argument("--reference", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {"candidate": side(a.candidate), "reference": side(a.reference), "bootstrap": {"samples": SAMPLES, "seed": SEED, "unit": "item"},
           "paired": {t: paired(outcomes(a.candidate, t), outcomes(a.reference, t)) for t in TIERS}}
    write_json(a.out, rep)
    c, r, h = rep["candidate"], rep["reference"], rep["paired"]["hard"]
    print(f"all {r['all']['accuracy']:.3f} -> {c['all']['accuracy']:.3f} | hard {r['tiers']['hard']['accuracy']:.3f} -> {c['tiers']['hard']['accuracy']:.3f} "
          f"{100 * h['delta']:+.1f} [{100 * h['ci95'][0]:+.1f}, {100 * h['ci95'][1]:+.1f}] {h['newly_right']}/{h['newly_wrong']} p={h['mcnemar_exact_p']:.3f} | "
          f"hard ECE {r['tiers']['hard']['ece']:.3f} -> {c['tiers']['hard']['ece']:.3f}")


if __name__ == "__main__":
    main()
