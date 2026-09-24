"""Round 8 rule 4 (PLAN.md): confirmation of a passing arm against its released parent, read once. Committed before any
confirmation read.

    uv run python scripts/round8_confirm.py docs --size 4b --out runs/r8-verdict     # documents-v1 test (gates the locked read)
    uv run python scripts/round8_confirm.py locked --size 4b --out runs/r8-verdict   # locked transfer-v4 after locked_test
    uv run python scripts/round8_confirm.py docs2 --size 4b --out runs/r8-verdict    # documents-v2 (private), reported only

Reads: runs/r8c-<size>-{cand,parent}-{docs1test,docs2}; locked: runs/locked/kev-<size>-r8 vs the parent's locked read.
Every arm is served at the temperature fitted on its own development rows (as the selection reads were).
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import boot, knowable, serve  # noqa: E402

PARENTS = {"9b": ("runs/night2-9b-du/00-trial-0", "runs/locked/kev-9b-night2-du-ungated"), "4b": ("runs/night2-4b-du/00-trial-0", "runs/locked/kev-4b-night2-du-ungated"),
           "08b": ("runs/night2-08b-du2/00-trial-0", "runs/locked/kev-08b-night2-du-ungated")}
CANDIDATES = {8: {"4b": "runs/r8-small/00-trial-0", "08b": "runs/r8-small/01-trial-1"}}   # round 9+: --candidate names the trial chosen by the read-out
f = lambda b: f"{100 * b['delta']:+.2f} [{100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["docs", "locked", "docs2"]); ap.add_argument("--size", required=True, choices=list(PARENTS)); ap.add_argument("--out", required=True)
    ap.add_argument("--round", type=int, default=8); ap.add_argument("--candidate", help="trial dir (required from round 9)")
    a = ap.parse_args()
    cand = a.candidate or CANDIDATES[a.round][a.size]
    (parent, parent_locked), tag = PARENTS[a.size], f"r{a.round}"
    t = {k: served(read_json(Path(x) / "development/rows.json"), [])[0] for k, x in (("cand", cand), ("parent", parent))}
    trial = {"cand": cand, "parent": parent}
    if a.stage == "locked":
        R = {"cand": f"runs/locked/kev-{a.size}-{tag}-ungated/transfer/rows.json", "parent": f"{parent_locked}/transfer/rows.json"}
        rows = {k: knowable(serve(trial[k], R[k], t[k])[0]) for k in trial}
        m = {k: metrics(rows[k]) for k in trial}
        rep = {"stage": "locked", "temperature": t, "locked": {k: {"acc": m[k]["acc"], "brier": m[k]["brier"], "n": len(rows[k])} for k in trial},
               "acc_delta": boot(rows["cand"], rows["parent"], "acc"), "brier_delta": boot(rows["cand"], rows["parent"], "brier")}
        rep["criteria"] = {"locked_acc_at_least_parent_minus_1pp": m["cand"]["acc"] >= m["parent"]["acc"] - 0.01, "served_brier_at_most_parent_plus_0.005": m["cand"]["brier"] <= m["parent"]["brier"] + 0.005}
        print(f"locked transfer-v4: acc {m['parent']['acc']:.4f} -> {m['cand']['acc']:.4f} ({f(rep['acc_delta'])}); Brier {m['parent']['brier']:.4f} -> {m['cand']['brier']:.4f}")
    else:
        suite = {"docs": "docs1test", "docs2": "docs2"}[a.stage]
        rows = {k: knowable(serve(trial[k], f"runs/{tag}c-{a.size}-{k}-{suite}/rows.json", t[k])[0]) for k in trial}
        rep = {"stage": a.stage, "temperature": t, "acc": {k: metrics(rows[k])["acc"] for k in trial}, "n": len(rows["cand"]),
               "acc_delta": boot(rows["cand"], rows["parent"], "acc"), "brier_delta": boot(rows["cand"], rows["parent"], "brier")}
        rep["criteria"] = {"docs_test_lower_above_0": rep["acc_delta"]["ci95"][0] > 0} if a.stage == "docs" else {}
        print(f"{suite}: acc {rep['acc']['parent']:.4f} -> {rep['acc']['cand']:.4f} {f(rep['acc_delta'])} (n {rep['n']}); Brier delta {rep['brier_delta']['delta']:+.4f} {[round(x, 4) for x in rep['brier_delta']['ci95']]}")
    rep["passed"] = all(rep["criteria"].values()) if rep["criteria"] else None
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / f"{a.size}-{a.stage}.json", rep)
    print("criteria", rep["criteria"], "->", {True: "PASSED", False: "failed", None: "reported"}[rep["passed"]])


if __name__ == "__main__":
    main()
