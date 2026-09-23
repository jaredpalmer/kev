"""Apply the registered round-5 release rule (PLAN.md, "Round 5", registered 2026-09-22 before any training or read) to
the reads pulled under runs/r5r-*. Every arm is served at the temperature fitted on its own decision-v7 development rows.

    uv run python scripts/round5_confirm.py --out runs/r5-verdict

Reads runs/r5r-<arm>-<suite>/rows.json for suite in {long, r5test, v9, semif, scienthoon, wanli, typesafe}; writes one report
per size with every criterion and its outcome, plus the attribution arms' numbers (reported, never gating).
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, paired_bootstrap, raw_row, recorded, served, tempered_row, unknowable_report  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

SAMPLES = 2000   # the registered resample count
TRIALS = {"C9": "runs/r5-combined/00-trial-0", "C9-s2": "runs/r5-combined/01-trial-1", "H9": "runs/r5-combined/02-trial-2",
          "C4": "runs/r5-combined/03-trial-3", "H4": "runs/r5-combined/04-trial-4", "C08": "runs/r5-combined/05-trial-5",
          "P9": "runs/night2-9b-du/00-trial-0", "P4": "runs/night2-4b-du/00-trial-0", "P08": "runs/night2-08b-du2/00-trial-0",
          "S9": "runs/r4-soft/00-trial-0"}
SIZES = {"9b": ("C9", "P9", ("C9-s2", "H9", "S9")), "4b": ("C4", "P4", ("H4",)), "0.8b": ("C08", "P08", ())}
EXTERNALS = ("semif", "scienthoon", "wanli", "typesafe")
KEYS = ("n", "acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")


def temperature(arm):
    return served(read_json(Path(TRIALS[arm]) / "development/rows.json"), [])[0]


def read(arm, suite, t):
    """Every clean row of one arm on one suite, served at its temperature (unknowable rows included for unknowable_report)."""
    return [tempered_row(raw_row(recorded(r)), t) for r in read_json(f"runs/r5r-{arm}-{suite}/rows.json") if r["variant"] == "clean"]


def knowable(rows, source=None):
    return [r for r in rows if r["source"] != "unknowable" and (source is None or r["source"] == source)]


def boot(a, b, metric):
    x = paired_bootstrap(a, b, samples=SAMPLES, seed=0, metric=metric, aggregation="micro")
    return {"delta": x[f"micro_{metric}_delta"], "ci95": x["ci95"]}


def size_report(candidate, parent, others):
    t = {arm: temperature(arm) for arm in (candidate, parent, *others)}
    long = {arm: knowable(read(arm, "long", t[arm]), "longstate") for arm in t}
    short = {arm: knowable(read(arm, "r5test", t[arm])) for arm in t}
    rep = {"temperature": t,
           "long": {arm: {k: metrics(rows)[k] for k in ("n", "acc", "brier")} for arm, rows in long.items()},
           "short": {arm: {k: metrics(rows)[k] for k in KEYS} for arm, rows in short.items()},
           "long_minus_parent": {arm: boot(long[arm], long[parent], "acc") for arm in t if arm != parent},
           "short_minus_parent": {arm: {m: boot(short[arm], short[parent], m) for m in ("acc", "brier", "confident_error_rate")} for arm in t if arm != parent}}
    lp, sp = rep["long_minus_parent"][candidate], rep["short_minus_parent"][candidate]
    rep["v9_unknowable_share"] = {arm: unknowable_report(read(arm, "v9", t[arm]))["share_at_0_9"] for arm in (candidate, parent)}
    rep["externals"] = {e: {arm: metrics(knowable(read(arm, e, t[arm])))["acc"] for arm in (candidate, parent)} for e in EXTERNALS}
    rep["criteria"] = {
        "1_long_lower_above_0": lp["ci95"][0] > 0, "1_long_point_at_least_5pp": lp["delta"] >= 0.05,
        "2_short_accuracy_lower_at_least_minus_1pp": sp["acc"]["ci95"][0] >= -0.01, "2_short_brier_upper_at_most_0.01": sp["brier"]["ci95"][1] <= 0.01,
        "2_short_confident_errors_upper_at_most_1pp": sp["confident_error_rate"]["ci95"][1] <= 0.01,
        "3_unknowable_share_at_most_0.05": rep["v9_unknowable_share"][candidate] <= 0.05,
        **{f"3_{e}_accuracy_within_1pp": v[candidate] - v[parent] >= -0.01 for e, v in rep["externals"].items()}}
    rep["passed"] = all(rep["criteria"].values())
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--sizes", default=",".join(SIZES))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for size in a.sizes.split(","):
        candidate, parent, others = SIZES[size]
        rep = {"size": size, "candidate": candidate, "parent": parent, **size_report(candidate, parent, others)}
        write_json(out / f"{size}.json", rep)
        print(f"== {size}: {candidate} vs {parent}  T={ {k: round(v, 2) for k, v in rep['temperature'].items()} }")
        for arm, d in rep["long_minus_parent"].items():
            print(f"  long  {arm:6} acc {rep['long'][arm]['acc']:.3f} (parent {rep['long'][parent]['acc']:.3f})  delta {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
        for arm, d in rep["short_minus_parent"].items():
            print(f"  short {arm:6} " + "  ".join(f"{m} {v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for m, v in d.items()))
        print("  v9 unknowable", rep["v9_unknowable_share"], " externals", {e: f"{v[candidate]:.3f} vs {v[parent]:.3f}" for e, v in rep["externals"].items()})
        print("  criteria", rep["criteria"], "-> PASSED" if rep["passed"] else "-> not released")


if __name__ == "__main__":
    main()
