"""Round 10 (PLAN.md, skills delta on hard-v1 + devtools-v1): rules 1-3 per arm against its own parent. Committed before
any round-10 read.

    uv run python scripts/round10_readout.py --out runs/r10-readout

Arms runs/r10-skills/{00,01,02}-trial-* (4B on skills / hard / devtools) and runs/r10-skills-27b/00-trial-0, read as
runs/r10-<arm>-{hard,devtools,docs,semif,scienthoon,wanli2,typesafe,v9}; short = the trial's transfer/rows.json. Parents:
the released Kev-4B (round 8, reads r8-4b-s2-*, hv1-P4r8, dt1-P4r8) and Kev-27B (reads r6-27bv2-s2-*, hv1-27b, dt1-27b).
Every arm is served at the temperature fitted on its own decision-v7 development rows.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served, unknowable_report  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import EXTERNALS, boot, knowable, serve  # noqa: E402

SUITES = ("hard", "devtools", "docs", *EXTERNALS)
PARENTS = {"4b": ("runs/r8-small/00-trial-0", {"hard": "runs/hv1-P4r8", "devtools": "runs/dt1-P4r8", "docs": "runs/r8-4b-s2-docs", "v9": "runs/r8-4b-s2-v9",
                                             **{s: f"runs/r8-4b-s2-{s}" for s in EXTERNALS}}),
           "27b": ("runs/release/kev-27b-v2", {"hard": "runs/hv1-27b", "devtools": "runs/dt1-27b", "docs": "runs/r6-27bv2-s2-docs", "v9": "runs/r6-27bv2-s2-v9",
                                               **{s: f"runs/r6-27bv2-s2-{s}" for s in EXTERNALS}})}
ARMS = {"4b-skills": ("runs/r10-skills/00-trial-0", "4b"), "4b-hard": ("runs/r10-skills/01-trial-1", "4b"), "4b-devtools": ("runs/r10-skills/02-trial-2", "4b"),
        "27b-skills": ("runs/r10-skills-27b/00-trial-0", "27b")}


def arm_rows(trial, reads, t):
    return {"short": knowable(serve(trial, Path(trial) / "transfer/rows.json", t)[0]), **{s: knowable(serve(trial, f"{reads[s]}/rows.json", t)[0]) for s in SUITES}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    report = {"arms": {}}
    for arm, (trial, size) in ARMS.items():
        reads = {s: f"runs/r10-{arm}-{s}" for s in (*SUITES, "v9")}
        if not (Path(trial) / "transfer/rows.json").exists() or not all(Path(f"{d}/rows.json").exists() for d in reads.values()):
            report["arms"][arm] = "not read yet"; continue
        ptrial, preads = PARENTS[size]
        pt, t = (served(read_json(Path(x) / "development/rows.json"), [])[0] for x in (ptrial, trial))
        P, C = arm_rows(ptrial, preads, pt), arm_rows(trial, reads, t)
        pooled = lambda R: R["hard"] + R["devtools"]
        rep = {"trial": trial, "parent": ptrial, "temperature": t, "parent_temperature": pt,
               "primary": boot(pooled(C), pooled(P), "acc"), **{f"{s}_acc": (metrics(C[s])["acc"], metrics(P[s])["acc"]) for s in ("hard", "devtools", "docs")},
               "hard_delta": boot(C["hard"], P["hard"], "acc"), "devtools_delta": boot(C["devtools"], P["devtools"], "acc"), "docs_delta": boot(C["docs"], P["docs"], "acc"),
               "short": {m: boot(C["short"], P["short"], m) for m in ("acc", "brier", "confident_error_rate")},
               "externals": {s: boot(C[s], P[s], "acc") for s in EXTERNALS},
               "pooled_external": boot([r for s in EXTERNALS for r in C[s]], [r for s in EXTERNALS for r in P[s]], "acc"),
               "hard_ece": {"candidate": metrics(C["hard"])["ece"], "parent": metrics(P["hard"])["ece"], "delta": boot(C["hard"], P["hard"], "ece")},
               "unknowable_share": unknowable_report(serve(trial, f"{reads['v9']}/rows.json", t)[0])["share_at_0_9"]}
        s = rep["short"]
        rep["criteria"] = {"1_primary_lower_above_0": rep["primary"]["ci95"][0] > 0,
                           "2_short_acc_lower_at_least_minus_2pp": s["acc"]["ci95"][0] >= -0.02, "2_short_brier_upper_at_most_0.01": s["brier"]["ci95"][1] <= 0.01,
                           "2_short_confident_errors_upper_at_most_1pp": s["confident_error_rate"]["ci95"][1] <= 0.01,
                           "2_docs_lower_at_least_minus_2pp": rep["docs_delta"]["ci95"][0] >= -0.02,
                           **{f"2_{k}_lower_at_least_minus_2pp": rep["externals"][k]["ci95"][0] >= -0.02 for k in ("wanli2", "scienthoon")},
                           "2_pooled_lower_at_least_minus_1.5pp": rep["pooled_external"]["ci95"][0] >= -0.015, "2_unknowable_at_most_0.05": rep["unknowable_share"] <= 0.05,
                           "3_hard_ece_at_most_parent_plus_0.01": rep["hard_ece"]["candidate"] <= rep["hard_ece"]["parent"] + 0.01}
        rep["passed"] = all(rep["criteria"].values())
        report["arms"][arm] = rep
    for size in ("4b", "27b"):
        passing = [(k, v) for k, v in report["arms"].items() if k.startswith(size + "-") and isinstance(v, dict) and v["passed"]]
        report[f"candidate_{size}"] = max(passing, key=lambda kv: kv[1]["primary"]["delta"])[0] if passing else None
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "round10.json", report)
    f = lambda b: f"{100 * b['delta']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]"
    for arm, r in report["arms"].items():
        if not isinstance(r, dict): print(f"{arm:12} {r}"); continue
        print(f"{arm:12} primary {f(r['primary'])} | hard {r['hard_acc'][1]:.3f}->{r['hard_acc'][0]:.3f} {f(r['hard_delta'])} | devtools {r['devtools_acc'][1]:.3f}->{r['devtools_acc'][0]:.3f} {f(r['devtools_delta'])} | "
              f"docs {f(r['docs_delta'])} | short {f(r['short']['acc'])} | pooled ext {f(r['pooled_external'])} | hard ECE {r['hard_ece']['parent']:.3f}->{r['hard_ece']['candidate']:.3f} | unk {r['unknowable_share']:.3f} "
              f"-> {'PASS' if r['passed'] else 'fail: ' + ', '.join(k for k, v in r['criteria'].items() if not v)}")
    print({k: v for k, v in report.items() if k.startswith("candidate_")})


if __name__ == "__main__":
    main()
