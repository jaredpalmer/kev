"""Fit one temperature on a checkpoint's in-distribution development rows and write it into head.pt, so every loader
(kev.serve, kev.benchmark, the Space, third-party harnesses) serves calibrated probabilities by default. It also reports
an out-of-fold, group-disjoint cross-validated calibration estimate with bootstrap intervals, so the in-sample fit can
be checked against held-out records; the value written is always the full-development fit.

    uv run python scripts/calibrate_checkpoint.py --run runs/night2-9b-du/00-trial-0/checkpoint --rows runs/night2-9b-du/00-trial-0/development/rows.json [--transfer runs/.../transfer/rows.json]

Fitting on the development partition only (never on transfer or test); the optional --transfer rows are reported, not fitted.
Argmax never changes; accuracy is identical before and after. KEV_TEMPERATURE=1.0 restores raw logits at load time.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import TEMPERATURE_FIT as FIT, TEMPERATURE_FIT_METHOD, cross_validated_temperature, fit_temperature, metrics  # noqa: E402
from kev.checkpoint import read_meta, write_meta  # noqa: E402
from kev.suite import read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--rows", required=True, help="in-distribution development rows.json (fit set)")
    ap.add_argument("--transfer", help="out-of-domain rows.json, reported before/after (never fitted)")
    ap.add_argument("--temperature", type=float, help="skip fitting (and cross-validation) and write this value")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = [r for r in read_json(a.rows) if r["variant"] == "clean"]
    T = a.temperature or fit_temperature(dev, **FIT)
    for name, rows in (("development", dev), *((("transfer", [r for r in read_json(a.transfer) if r["variant"] == "clean"]),) if a.transfer else ())):
        raw, cal = metrics(rows), metrics(rows, T)
        print(f"{name:12} T={T:.2f}  acc {raw['acc']:.3f} -> {cal['acc']:.3f} | brier {raw['brier']:.3f} -> {cal['brier']:.3f} | ece {raw['ece']:.3f} -> {cal['ece']:.3f} | conf-err {raw['confident_error_rate']:.3f} -> {cal['confident_error_rate']:.3f} | cov@5% {raw['coverage_at_5pct_error']:.2f} -> {cal['coverage_at_5pct_error']:.2f}")
    meta = read_meta(a.run)
    meta.temperature = T
    if a.temperature is not None:
        meta.extra["temperature_fit"] = {"method": "manual"}
        write_meta(a.run, meta); print(f"wrote temperature {T:.2f} to {a.run}/head.pt")
        return
    cv = cross_validated_temperature(dev, folds=a.folds, seed=a.seed, **FIT)
    ci = cv["ece_ci95"]
    temperatures = ", ".join(f"{t:.2f}" for t in cv["temperatures"])
    print(f"development  OOF T=[{temperatures}] ece raw {cv['raw']['ece']:.3f} [{ci['raw'][0]:.3f}, {ci['raw'][1]:.3f}]"
          f" -> oof {cv['out_of_fold']['ece']:.3f} [{ci['out_of_fold'][0]:.3f}, {ci['out_of_fold'][1]:.3f}]"
          f"  delta [{ci['delta'][0]:.3f}, {ci['delta'][1]:.3f}] separated={cv['separated']}")
    meta.extra["temperature_fit"] = {"rows": a.rows, "n": len(dev), "method": TEMPERATURE_FIT_METHOD, "cross_validation": cv}
    write_meta(a.run, meta); print(f"wrote temperature {T:.2f} to {a.run}/head.pt")


if __name__ == "__main__":
    main()
