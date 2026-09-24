"""Round 10 rule 4 (PLAN.md): confirmation of the candidate against its parent, read once. Committed before any
confirmation read.

    uv run python scripts/round10_confirm.py tests --size 4b --candidate runs/r10-skills/00-trial-0 --out runs/r10-verdict
    uv run python scripts/round10_confirm.py locked --size 4b --candidate runs/r10-skills/00-trial-0 --out runs/r10-verdict

tests: hard-v1 test + devtools-v1 test pooled (reads runs/r10c-<size>-{cand,parent}-{hardtest,devtest}), paired lower bound
> 0 (the duplicated devtools id dropped on both sides, PLAN amendment). locked: runs/locked/kev-<size>-r10-ungated against
the parent's locked read; accuracy ≥ parent − 1 pp and served Brier ≤ parent + 0.005.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import boot, knowable, serve  # noqa: E402
from round10_readout import DUPLICATE_IDS, PARENTS  # noqa: E402

PARENT_LOCKED = {"4b": "runs/locked/kev-4b-r8-ungated", "27b": "runs/locked/kev-27b-v2-ungated"}
f = lambda b: f"{100 * b['delta']:+.2f} [{100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["tests", "locked"]); ap.add_argument("--size", required=True, choices=list(PARENTS))
    ap.add_argument("--candidate", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    trial = {"cand": a.candidate, "parent": PARENTS[a.size][0]}
    t = {k: served(read_json(Path(x) / "development/rows.json"), [])[0] for k, x in trial.items()}
    rows = lambda k, path: [r for r in knowable(serve(trial[k], path, t[k])[0]) if r["id"] not in DUPLICATE_IDS]
    if a.stage == "tests":
        R = {k: {s: rows(k, f"runs/r10c-{a.size}-{k}-{s}/rows.json") for s in ("hardtest", "devtest")} for k in trial}
        pooled = {k: R[k]["hardtest"] + R[k]["devtest"] for k in trial}
        rep = {"stage": "tests", "temperature": t, "pooled": boot(pooled["cand"], pooled["parent"], "acc"),
               **{s: {"cand": metrics(R["cand"][s])["acc"], "parent": metrics(R["parent"][s])["acc"], "delta": boot(R["cand"][s], R["parent"][s], "acc"), "n": len(R["cand"][s])} for s in ("hardtest", "devtest")},
               "hard_ece": {k: metrics(R[k]["hardtest"])["ece"] for k in trial}}
        rep["criteria"] = {"pooled_lower_above_0": rep["pooled"]["ci95"][0] > 0}
        print(f"hard-v1 test {rep['hardtest']['parent']:.3f} -> {rep['hardtest']['cand']:.3f} {f(rep['hardtest']['delta'])} | devtools-v1 test {rep['devtest']['parent']:.3f} -> {rep['devtest']['cand']:.3f} {f(rep['devtest']['delta'])} | pooled {f(rep['pooled'])} | hard ECE {rep['hard_ece']['parent']:.3f} -> {rep['hard_ece']['cand']:.3f}")
    else:
        R = {"cand": rows("cand", f"runs/locked/kev-{a.size}-r10-ungated/transfer/rows.json"), "parent": rows("parent", f"{PARENT_LOCKED[a.size]}/transfer/rows.json")}
        m = {k: metrics(R[k]) for k in trial}
        rep = {"stage": "locked", "temperature": t, "locked": {k: {"acc": m[k]["acc"], "brier": m[k]["brier"]} for k in trial}, "acc_delta": boot(R["cand"], R["parent"], "acc")}
        rep["criteria"] = {"locked_acc_at_least_parent_minus_1pp": m["cand"]["acc"] >= m["parent"]["acc"] - 0.01, "served_brier_at_most_parent_plus_0.005": m["cand"]["brier"] <= m["parent"]["brier"] + 0.005}
        print(f"locked transfer-v4 acc {m['parent']['acc']:.4f} -> {m['cand']['acc']:.4f} ({f(rep['acc_delta'])}); Brier {m['parent']['brier']:.4f} -> {m['cand']['brier']:.4f}")
    rep["passed"] = all(rep["criteria"].values())
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / f"{a.size}-{a.stage}.json", rep)
    print("criteria", rep["criteria"], "->", "PASSED" if rep["passed"] else "failed")


if __name__ == "__main__":
    main()
