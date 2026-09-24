"""Round 15 confirmation (PLAN.md): the joint 0.8B candidate against the released Kev-0.8B, read once. Committed before
any confirmation read.

    uv run python scripts/round15_confirm.py tests  --candidate runs/r15-08b/00-trial-0 --out runs/r15-verdict
    uv run python scripts/round15_confirm.py locked --candidate runs/r15-08b/00-trial-0 --out runs/r15-verdict

tests: documents-v1 test lower bound > 0, hard-v1 + devtools-v1 test pooled lower bound > 0, documents-v2 reported (reads
runs/r15c-08b-cand-{docs1test,hardtest,devtest,docs2}; the parent's reads of the same test partitions from rounds 11-12).
locked: runs/locked/kev-08b-r15-ungated against the parent's locked read: accuracy >= parent - 1 pp, served Brier <= +0.005.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import boot, knowable, serve  # noqa: E402
from round10_readout import DUPLICATE_IDS  # noqa: E402

PARENT = "runs/night2-08b-du2/00-trial-0"
PARENT_READS = {"docs1test": "runs/r11c-08b-parent-docs1test", "hardtest": "runs/r12c-08b-parent-hardtest", "devtest": "runs/r12c-08b-parent-devtest",
                "docs2": "runs/r11c-08b-parent-docs2"}
PARENT_LOCKED = "runs/locked/kev-08b-night2-du-ungated"
f = lambda b: f"{100 * b['delta']:+.2f} [{100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["tests", "locked"]); ap.add_argument("--candidate", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trial = {"cand": a.candidate, "parent": PARENT}
    t = {k: served(read_json(Path(x) / "development/rows.json"), [])[0] for k, x in trial.items()}
    rows = lambda k, path: [r for r in knowable(serve(trial[k], path, t[k])[0]) if r["id"] not in DUPLICATE_IDS]
    if a.stage == "tests":
        R = {s: {"cand": rows("cand", f"runs/r15c-08b-cand-{s}/rows.json"), "parent": rows("parent", f"{PARENT_READS[s]}/rows.json")} for s in PARENT_READS}
        d = {s: boot(R[s]["cand"], R[s]["parent"], "acc") for s in R}
        pooled = boot(R["hardtest"]["cand"] + R["devtest"]["cand"], R["hardtest"]["parent"] + R["devtest"]["parent"], "acc")
        rep = {"stage": "tests", "temperature": t, "deltas": d, "pooled_skills": pooled, "acc": {s: {k: metrics(R[s][k])["acc"] for k in trial} for s in R}}
        rep["criteria"] = {"docs1test_lower_above_0": d["docs1test"]["ci95"][0] > 0, "skills_tests_pooled_lower_above_0": pooled["ci95"][0] > 0}
        for s in R: print(f"{s:10} {rep['acc'][s]['parent']:.3f} -> {rep['acc'][s]['cand']:.3f} {f(d[s])}" + ("  (reported)" if s == "docs2" else ""))
        print(f"skills tests pooled {f(pooled)}")
    else:
        R = {"cand": rows("cand", "runs/locked/kev-08b-r15-ungated/transfer/rows.json"), "parent": rows("parent", f"{PARENT_LOCKED}/transfer/rows.json")}
        m = {k: metrics(R[k]) for k in trial}
        rep = {"stage": "locked", "temperature": t, "locked": {k: {"acc": m[k]["acc"], "brier": m[k]["brier"]} for k in trial}, "acc_delta": boot(R["cand"], R["parent"], "acc")}
        rep["criteria"] = {"locked_acc_at_least_parent_minus_1pp": m["cand"]["acc"] >= m["parent"]["acc"] - 0.01, "served_brier_at_most_parent_plus_0.005": m["cand"]["brier"] <= m["parent"]["brier"] + 0.005}
        print(f"locked transfer-v4 acc {m['parent']['acc']:.4f} -> {m['cand']['acc']:.4f} ({f(rep['acc_delta'])}); Brier {m['parent']['brier']:.4f} -> {m['cand']['brier']:.4f}")
    rep["passed"] = all(rep["criteria"].values())
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / f"08b-{a.stage}.json", rep)
    print("criteria", rep["criteria"], "->", "PASSED" if rep["passed"] else "failed")


if __name__ == "__main__":
    main()
