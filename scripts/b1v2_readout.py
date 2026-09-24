"""B1 v2 (PLAN_27b): apply the registered Kev-27B rule to the two `r6-27b-v2` seeds, from saved rows. Committed before
any of those seeds' reads.

    uv run python scripts/b1v2_readout.py select  --out runs/r6-verdict     # selection on development sets -> candidate
    uv run python scripts/b1v2_readout.py confirm --candidate runs/r6-27b-v2/00-trial-0 --out runs/r6-verdict

Selection (per seed; reads runs/r6-27bv2-s<seed>-{v9,long,semif,scienthoon,wanli2,typesafe}): transfer-v4 development
accuracy >= 0.842; MMLU-Pro on transfer-v9 development >= 0.65; unknowable share at p >= 0.9 <= 0.05; held-out pairs
>= 0.75; buried questions of longstate-v2 development >= Kev-9B's (0.572) + 10 pp with a paired lower bound > 0;
pooled external accuracy point estimate >= Kev-9B's. The passing seed with the higher transfer-v4 accuracy is the
candidate. Confirmation (reads runs/r6c-27b-{cand,parent}-{r6test,long3}, read once): transfer-r6 test accuracy paired
lower bound > 0 against Kev-9B and no task whose paired interval lies entirely below -3 pp; longstate-v3 buried accuracy
lower bound > 0 and point >= +10 pp. Every arm is served at the temperature fitted on its own development rows.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import metrics, served_at, unknowable_report  # noqa: E402
from kev.rounds import paired as boot, served_clean, temperature  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

EXTERNALS = ("semif", "scienthoon", "wanli2", "typesafe")

P9 = "runs/night2-9b-du/00-trial-0"
SEEDS = {1: "runs/r6-27b-v2/00-trial-0", 2: "runs/r6-27b-v2/01-trial-1"}
P9_READS = {"long": "runs/r5r-P9-long", "semif": "runs/r5r-P9-semif", "scienthoon": "runs/r5r-P9-scienthoon", "wanli2": "runs/r6-P9-wanli2", "typesafe": "runs/r5r-P9-typesafe"}


def parent():
    t = temperature(P9, ".")
    return t, {k: served_at(read_json(f"{d}/rows.json"), t) for k, d in P9_READS.items()}


def select(out):
    pt, prow = parent()
    plong = [r for r in prow["long"] if r["source"] == "longstate"]
    p_ext = [r for s in EXTERNALS for r in prow[s]]
    report = {"parent": P9, "parent_temperature": pt, "parent_long_acc": metrics(plong)["acc"], "parent_pooled_external_acc": metrics(p_ext)["acc"], "seeds": {}}
    for seed, trial in SEEDS.items():
        res, tag = read_json(Path(trial) / "result.json"), f"runs/r6-27bv2-s{seed}"
        t = temperature(trial, ".")
        v9 = served_clean(read_json(f"{tag}-v9/rows.json"), t)
        long = [r for r in served_at(read_json(f"{tag}-long/rows.json"), t) if r["source"] == "longstate"]
        ext = [r for s in EXTERNALS for r in served_at(read_json(f"{tag}-{s}/rows.json"), t)]
        s = {"trial": trial, "temperature": t, "transfer_acc": res["transfer"]["clean"]["acc"], "mmlu_pro": read_json(f"{tag}-v9/report.json")["tasks"]["mmlu_pro"]["acc"],
             "unknowable_share": unknowable_report(v9)["share_at_0_9"], "pairs": res["transfer"]["paired_flip"]["both_correct_rate"],
             "long_acc": metrics(long)["acc"], "long_delta": boot(long, plong, "acc"), "pooled_external_acc": metrics(ext)["acc"], "pooled_external_delta": boot(ext, p_ext, "acc")}
        s["criteria"] = {"transfer_at_least_0.842": s["transfer_acc"] >= 0.842, "mmlu_pro_at_least_0.65": s["mmlu_pro"] >= 0.65,
                         "unknowable_at_most_0.05": s["unknowable_share"] <= 0.05, "pairs_at_least_0.75": s["pairs"] >= 0.75,
                         "long_at_least_parent_plus_10pp": s["long_acc"] >= report["parent_long_acc"] + 0.10, "long_lower_above_0": s["long_delta"]["ci95"][0] > 0,
                         "pooled_external_point_at_least_parent": s["pooled_external_acc"] >= report["parent_pooled_external_acc"]}
        s["passed"] = all(s["criteria"].values())
        report["seeds"][seed] = s
    passing = [s for s in report["seeds"].values() if s["passed"]]
    report["candidate"] = max(passing, key=lambda s: s["transfer_acc"])["trial"] if passing else None
    Path(out).mkdir(parents=True, exist_ok=True); write_json(Path(out) / "27b-v2-selection.json", report)
    for seed, s in report["seeds"].items():
        print(f"seed {seed}: transfer {s['transfer_acc']:.3f} mmlu_pro {s['mmlu_pro']:.3f} unk {s['unknowable_share']:.3f} pairs {s['pairs']:.2f} "
              f"long {s['long_acc']:.3f} (parent {report['parent_long_acc']:.3f}; {s['long_delta']['delta']:+.3f} [{s['long_delta']['ci95'][0]:+.3f}]) "
              f"ext {s['pooled_external_acc']:.3f} (parent {report['parent_pooled_external_acc']:.3f})  failed: {[k for k, v in s['criteria'].items() if not v] or 'none'}")
    print("candidate:", report["candidate"])


def confirm(candidate, out):
    t = {"cand": temperature(candidate, "."), "parent": temperature(P9, ".")}
    trial = {"cand": candidate, "parent": P9}
    rows = {(k, s): served_at(read_json(f"runs/r6c-27b-{k}-{s}/rows.json"), t[k]) for k in trial for s in ("r6test", "long3")}
    long = {k: [r for r in rows[k, "long3"] if r["source"] == "longstate"] for k in trial}
    acc, ld = boot(rows["cand", "r6test"], rows["parent", "r6test"], "acc"), boot(long["cand"], long["parent"], "acc")
    tasks = {}
    for task in sorted({r["task"] for r in rows["cand", "r6test"]}):
        c, p = [r for r in rows["cand", "r6test"] if r["task"] == task], [r for r in rows["parent", "r6test"] if r["task"] == task]
        tasks[task] = {"n": len(c), **boot(c, p, "acc")}
    below = [k for k, v in tasks.items() if v["ci95"][1] < -0.03]
    rep = {"candidate": candidate, "parent": P9, "temperature": t, "r6test_acc_delta": acc, "r6test_tasks": tasks, "long3_delta": ld,
           "r6test": {k: metrics(rows[k, "r6test"])["acc"] for k in trial}, "long3": {k: metrics(long[k])["acc"] for k in trial},
           "criteria": {"r6test_lower_above_0": acc["ci95"][0] > 0, "no_task_entirely_below_minus_3pp": not below, "long3_lower_above_0": ld["ci95"][0] > 0, "long3_point_at_least_10pp": ld["delta"] >= 0.10}}
    rep["passed"] = all(rep["criteria"].values())
    Path(out).mkdir(parents=True, exist_ok=True); write_json(Path(out) / "27b-v2-confirm.json", rep)
    print(f"transfer-r6 {rep['r6test']['parent']:.3f} -> {rep['r6test']['cand']:.3f} {acc['delta']:+.4f} [{acc['ci95'][0]:+.4f}, {acc['ci95'][1]:+.4f}]; tasks below: {below or 'none'}")
    print(f"longstate-v3 {rep['long3']['parent']:.3f} -> {rep['long3']['cand']:.3f} {ld['delta']:+.4f} [{ld['ci95'][0]:+.4f}, {ld['ci95'][1]:+.4f}]")
    print("criteria", rep["criteria"], "-> PASSED (locked read allowed)" if rep["passed"] else "-> failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["select", "confirm"]); ap.add_argument("--candidate"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    select(a.out) if a.mode == "select" else confirm(a.candidate, a.out)
