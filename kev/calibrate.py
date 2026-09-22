"""Per-workload calibration report from a saved rows.json: what one temperature fitted on *these* rows would do.

    uv run python -m kev.calibrate --rows runs/kev-9b-wanli-v1/rows.json [--out runs/kev-9b-wanli-v1/calibration.json]

A checkpoint ships one temperature fitted on its in-distribution development rows (scripts/calibrate_checkpoint.py). It
transfers to our out-of-domain suites, not necessarily to a deployer's workload: Kev-9B on WANLI-256 is served at
mean confidence 0.82 against accuracy 0.70. This module reports, on one rows file, four arms of the same predictions:

    raw           the checkpoint's logits at T=1 (restored from the recorded inference_temperature)
    shipped       as served (the checkpoint's temperature)
    workload      one temperature fitted on all these rows (in-sample; the number a deployer would write down)
    workload_oof  the same fit, group-disjoint out-of-fold: what fitting on your labelled rows buys on rows you did not fit

and a record-clustered paired bootstrap of workload_oof against shipped on ECE, Brier, coverage at <= 5% error and
AURC. Argmax and accuracy are identical in every arm by construction. Report only: nothing is written to a checkpoint.
"""
import argparse
from pathlib import Path

from kev.metrics import TEMPERATURE_FIT, fit_temperature, metrics, out_of_fold_rows, paired_bootstrap, raw_row, scored_rows, tempered_row
from kev.suite import read_json, write_json

KEYS = ("n", "acc", "ece", "brier", "nll", "mean_conf", "confident_error_rate", "coverage_at_5pct_error", "aurc")
COMPARED = ("ece", "brier", "coverage_at_5pct_error", "aurc")


def workload_report(rows, folds=5, seed=0, samples=1000):
    served = scored_rows(rows)
    if not served:
        raise ValueError("no clean, knowable rows to calibrate")
    raw = [raw_row(row) for row in served]
    temperature = fit_temperature(raw, **TEMPERATURE_FIT)
    oof, fold_temperatures = out_of_fold_rows(raw, folds, seed, **TEMPERATURE_FIT)
    arms = {"raw": raw, "shipped": served, "workload": [tempered_row(row, temperature) for row in raw], "workload_oof": oof}
    return {"n": len(served), "shipped_temperature": sorted({row.get("inference_temperature", 1.0) for row in served}),
            "workload_temperature": temperature, "fold_temperatures": fold_temperatures,
            "arms": {name: {k: metrics(arm)[k] for k in KEYS} for name, arm in arms.items()},
            "oof_vs_shipped": {metric: paired_bootstrap(oof, served, samples=samples, seed=seed, metric=metric, aggregation="micro") for metric in COMPARED},
            "fit": TEMPERATURE_FIT, "folds": folds, "seed": seed,
            "note": "argmax and accuracy are temperature-invariant; workload is in-sample, workload_oof is the held-out estimate"}


def format_report(report):
    lines = [f"n={report['n']}  shipped T={', '.join(f'{t:.2f}' for t in report['shipped_temperature'])}  workload T={report['workload_temperature']:.2f}"
             f"  folds T=[{', '.join(f'{t:.2f}' for t in report['fold_temperatures'])}]",
             f"{'arm':13}{'acc':>7}{'ece':>7}{'brier':>7}{'nll':>7}{'conf':>7}{'c-err':>7}{'cov@5%':>8}{'aurc':>7}"]
    for name, m in report["arms"].items():
        lines.append(f"{name:13}{m['acc']:7.3f}{m['ece']:7.3f}{m['brier']:7.3f}{m['nll']:7.3f}{m['mean_conf']:7.3f}{m['confident_error_rate']:7.3f}{m['coverage_at_5pct_error']:8.2f}{m['aurc']:7.3f}")
    for metric, b in report["oof_vs_shipped"].items():
        lo, hi = b["ci95"]
        lines.append(f"workload_oof - shipped  {metric:24}{b[f'micro_{metric}_delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True, help="rows.json written by kev.benchmark")
    ap.add_argument("--out", help="where to write the report (default: calibration.json next to the rows)")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args()
    report = workload_report(read_json(a.rows), folds=a.folds, seed=a.seed, samples=a.samples)
    report["rows"] = a.rows
    out = Path(a.out) if a.out else Path(a.rows).with_name("calibration.json")
    write_json(out, report)
    print(format_report(report)); print(f"wrote {out}")


if __name__ == "__main__":
    main()
