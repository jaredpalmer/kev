"""Apply the registered release-confirmation rule for the soft-target Kev-9B (PLAN.md, "Release confirmation", registered
2026-09-22 before any read). Every arm is served at the temperature fitted on its own decision-v7 development rows.

    uv run python scripts/release_confirm.py --out runs/rc-verdict

Reads the benchmark rows pulled to runs/rc-{cand,ctrl,parent}-{r3test,v9}, runs/rc-cand-<external> and the parent's
external reads runs/r4-kev-9b-<external>-raw, and writes one report with every criterion and its outcome.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, paired_bootstrap, unknowable_report  # noqa: E402
from kev.rounds import SAMPLES, served_clean, temperature  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

ARMS = {"candidate": ("runs/r4-soft/00-trial-0", "cand"), "control": ("runs/r4-deltas/01-trial-1", "ctrl"), "parent": ("runs/night2-9b-du/00-trial-0", "parent")}
EXTERNALS = ("semif", "scienthoon", "wanli", "typesafe")
KEYS = ("n", "acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error", "aurc")


def arm(tag, suite, t):
    """(knowable rows, every clean row incl. unknowable) of one arm on one suite, served at its temperature. The knowable
    subset is exactly what kev.metrics.served_at scores; the unknowable rows are kept for unknowable_report."""
    everything = served_clean(read_json(f"runs/rc-{tag}-{suite}/rows.json"), t)
    return [r for r in everything if r["source"] != "unknowable"], everything


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"registered": "PLAN.md, Release confirmation: soft-target Kev-9B", "temperature": {}, "final_panel": {}, "transfer_v9": {}, "externals": {}}
    panel, v9 = {}, {}
    for name, (trial, tag) in ARMS.items():
        t = report["temperature"][name] = temperature(trial, ".")   # fitted on the arm's own development rows
        knowable, everything = arm(tag, "r3test", t); panel[name] = knowable
        report["final_panel"][name] = {**{k: metrics(knowable)[k] for k in KEYS}, "unknowable": unknowable_report(everything)}
        v9[name] = unknowable_report(arm(tag, "v9", t)[1])
    boot = lambda ref, m: paired_bootstrap(panel["candidate"], panel[ref], samples=SAMPLES, seed=0, metric=m, aggregation="micro")
    deltas = {ref: {m: boot(ref, m) for m in ("brier", "confident_error_rate", "acc", "coverage_at_5pct_error", "aurc")} for ref in ("parent", "control")}
    report["final_panel"]["candidate_minus"] = deltas
    p = deltas["parent"]
    primary = {"P1_brier_upper_below_0": p["brier"]["ci95"][1] < 0, "P2_confident_errors_upper_below_0": p["confident_error_rate"]["ci95"][1] < 0,
               "P3_accuracy_lower_at_least_minus_1pp": p["acc"]["ci95"][0] >= -0.01}
    report["transfer_v9"] = v9
    secondary = {"unknowable_share_at_most_0.05": v9["candidate"]["share_at_0_9"] <= 0.05}
    for ext in EXTERNALS:
        cand, par = read_json(f"runs/rc-cand-{ext}/report.json")["clean"]["acc"], read_json(f"runs/r4-kev-9b-{ext}-raw/report.json")["clean"]["acc"]
        report["externals"][ext] = {"candidate": cand, "parent": par, "delta": cand - par}
        secondary[f"{ext}_accuracy_within_1pp"] = cand - par >= -0.01
    report["criteria"] = {"primary": primary, "secondary": secondary, "passed": all(primary.values()) and all(secondary.values())}
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "report.json", report)
    print("temperatures:", {k: round(v, 2) for k, v in report["temperature"].items()})
    for name in ARMS:
        m = report["final_panel"][name]; print(f"panel {name:9} " + " ".join(f"{k} {m[k]:.4f}" if k != "n" else f"n {m[k]}" for k in KEYS))
    for ref, d in deltas.items():
        print(f"  candidate - {ref}: " + "  ".join(f"{m} {b[f'micro_{m}_delta']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]" for m, b in d.items()))
    print("transfer-v9 unknowable share >= 0.9:", {k: round(v["share_at_0_9"], 3) for k, v in v9.items()})
    print("externals:", {k: f"{v['candidate']:.3f} vs {v['parent']:.3f}" for k, v in report["externals"].items()})
    print("criteria:", report["criteria"])


if __name__ == "__main__":
    main()
