"""Fit one temperature on a checkpoint's in-distribution development rows and write it into head.pt, so every loader
(kev.serve, kev.benchmark, the Space, third-party harnesses) serves calibrated probabilities by default. It also reports
an out-of-fold, group-disjoint cross-validated calibration estimate with bootstrap intervals, so the in-sample fit can
be checked against held-out records; the value written is always the full-development fit.

    uv run python scripts/calibrate_checkpoint.py --run runs/night2-9b-du/00-trial-0/checkpoint --rows runs/night2-9b-du/00-trial-0/development/rows.json [--transfer runs/.../transfer/rows.json]

Fitting on the development partition only (never on transfer or test); the optional --transfer rows are reported, not fitted.
Argmax never changes; accuracy is identical before and after. KEV_TEMPERATURE=1.0 restores raw logits at load time.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import cross_validated_temperature, metrics  # noqa: E402
from kev.checkpoint import read_meta, write_meta  # noqa: E402
from kev.suite import read_json, write_json

GRID = np.exp(np.linspace(np.log(0.25), np.log(4), 121))


def fit(rows):
    return float(GRID[int(np.argmin([metrics(rows, float(t))["nll"] for t in GRID]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--rows", required=True, help="in-distribution development rows.json (fit set)")
    ap.add_argument("--transfer", help="out-of-domain rows.json, reported before/after (never fitted)")
    ap.add_argument("--temperature", type=float, help="skip fitting and write this value")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = [r for r in read_json(a.rows) if r["variant"] == "clean"]
    T = a.temperature or fit(dev)
    for name, rows in (("development", dev), *((("transfer", [r for r in read_json(a.transfer) if r["variant"] == "clean"]),) if a.transfer else ())):
        raw, cal = metrics(rows), metrics(rows, T)
        print(f"{name:12} T={T:.2f}  acc {raw['acc']:.3f} -> {cal['acc']:.3f} | brier {raw['brier']:.3f} -> {cal['brier']:.3f} | ece {raw['ece']:.3f} -> {cal['ece']:.3f} | conf-err {raw['confident_error_rate']:.3f} -> {cal['confident_error_rate']:.3f} | cov@5% {raw['coverage_at_5pct_error']:.2f} -> {cal['coverage_at_5pct_error']:.2f}")
    cv = cross_validated_temperature(dev, fit=fit, folds=a.folds, seed=a.seed)
    ci = cv["ece_ci95"]
    fold_map = cv.pop("fold_of")
    temperatures = ", ".join(f"{t:.2f}" for t in cv["temperatures"])
    print(f"development  OOF T=[{temperatures}] ece raw {cv['raw']['ece']:.3f} [{ci['raw'][0]:.3f}, {ci['raw'][1]:.3f}]"
          f" -> oof {cv['out_of_fold']['ece']:.3f} [{ci['out_of_fold'][0]:.3f}, {ci['out_of_fold'][1]:.3f}]"
          f"  delta [{ci['delta'][0]:.3f}, {ci['delta'][1]:.3f}] separated={cv['separated']}")
    write_json(Path(a.run) / "calibration_folds.json", fold_map)
    meta = read_meta(a.run)
    meta.temperature = T; meta.extra["temperature_fit"] = {"rows": a.rows, "n": len(dev), "method": "min NLL over a 121-point log grid 0.25..4", "cross_validation": cv}
    write_meta(a.run, meta); print(f"wrote temperature {T:.2f} to {a.run}/head.pt")


if __name__ == "__main__":
    main()
