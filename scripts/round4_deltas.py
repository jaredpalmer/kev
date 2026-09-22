"""Round-4 delta read-out (PLAN.md 4.9, 4.11, 4.12): each candidate against the released checkpoint and against its
matched continuation control, on transfer-v4 development, with served probabilities (each arm's temperature refitted on
its own development rows, never on transfer or test, as registered).

    uv run python scripts/round4_deltas.py --candidate runs/r4-soft/00-trial-0 --control runs/r4-deltas/01-trial-1 \
        --released runs/night2-9b-du/00-trial-0 --out runs/r4-readout/ambiguity-9b

Paired, record-clustered bootstraps (kev.metrics.paired_bootstrap, micro aggregation, the headline accuracy) on accuracy,
coverage at <= 5% error, AURC and Brier.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, paired_bootstrap, served  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

KEYS = ("n", "acc", "brier", "ece", "coverage_at_5pct_error", "aurc", "confident_error_rate")
COMPARED = ("acc", "coverage_at_5pct_error", "aurc", "brier")


def served_trial(trial):
    """(temperature fitted on the trial's development rows, its transfer rows served at that temperature)."""
    return served(read_json(Path(trial) / "development/rows.json"), read_json(Path(trial) / "transfer/rows.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True); ap.add_argument("--control", required=True); ap.add_argument("--released", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {name: served_trial(path) for name, path in (("candidate", a.candidate), ("control", a.control), ("released", a.released))}
    report = {"trials": {"candidate": a.candidate, "control": a.control, "released": a.released},
              "temperature": {name: t for name, (t, _) in arms.items()},
              "transfer": {name: {k: metrics(rows)[k] for k in KEYS} for name, (_, rows) in arms.items()},
              "candidate_minus": {ref: {m: paired_bootstrap(arms["candidate"][1], arms[ref][1], metric=m, aggregation="micro") for m in COMPARED} for ref in ("released", "control")}}
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "report.json", report)
    for name, m in report["transfer"].items():
        print(f"{name:9} T={report['temperature'][name]:.2f} " + " ".join(f"{k} {m[k]:.3f}" for k in KEYS if k != "n"))
    for ref, deltas in report["candidate_minus"].items():
        print("  vs " + ref + ": " + "  ".join(f"{m} {b[f'micro_{m}_delta']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]" for m, b in deltas.items()))


if __name__ == "__main__":
    main()
