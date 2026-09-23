"""Two-checkpoint averaging (PLAN.md round 4, item 4.5; round-3 C6): mean probabilities of several seeds of one recipe,
read from saved rows, against the released seed. Each arm gets a temperature fitted on its own development rows (the
ensemble's development rows are averaged the same way), so the comparison is served-vs-served.

    uv run python scripts/checkpoint_average.py --reference runs/q35-9b/01-trial-1 --others runs/q35-9b/00-trial-0 --out runs/r4-average-9b
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, paired_bootstrap, scored_rows, served  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

KEYS = ("n", "acc", "brier", "nll", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")


def averaged(runs):
    """Rows of the probability-averaged ensemble. Logits are log(mean p), so a temperature applies to the ensemble as a
    whole; every run must hold the same clean questions with the same labels and option order."""
    indexed = [{(r["id"], r["question"]): r for r in scored_rows(rows)} for rows in runs]
    if any(ix.keys() != indexed[0].keys() for ix in indexed):
        raise ValueError("runs do not cover the same questions")
    out = []
    for key, row in indexed[0].items():
        if any(ix[key]["keys"] != row["keys"] or ix[key]["label"] != row["label"] for ix in indexed):
            raise ValueError(f"labels or option order differ at {key}")
        p = np.mean([ix[key]["p"] for ix in indexed], axis=0)
        out.append({**row, "p": p.tolist(), "logits": np.log(np.maximum(p, 1e-12)).tolist(), "inference_temperature": 1.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="trial dir of the released seed")
    ap.add_argument("--others", required=True, help="comma-separated trial dirs of the other seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trials = [a.reference, *a.others.split(",")]
    load = lambda t, part: scored_rows(read_json(Path(t) / part / "rows.json"))
    t_ref, ref = served(load(a.reference, "development"), load(a.reference, "transfer"))
    t_ens, ens = served(averaged([load(t, "development") for t in trials]), averaged([load(t, "transfer") for t in trials]))
    report = {"trials": trials, "temperature": {"reference": t_ref, "ensemble": t_ens},
              "reference": {k: metrics(ref)[k] for k in KEYS}, "ensemble": {k: metrics(ens)[k] for k in KEYS},
              "ensemble_minus_reference": {m: paired_bootstrap(ens, ref, metric=m, aggregation="micro") for m in ("acc", "brier", "coverage_at_5pct_error", "aurc")}}
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "report.json", report)
    for arm in ("reference", "ensemble"):
        print(f"{arm:10} T={report['temperature'][arm]:.2f} " + " ".join(f"{k} {report[arm][k]:.3f}" for k in KEYS if k != "n"))
    for m, b in report["ensemble_minus_reference"].items():
        print(f"  delta {m:24} {b[f'micro_{m}_delta']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
