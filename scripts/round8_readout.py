"""Round 8 (PLAN.md): round 7's documents delta at 0.8B / 4B, seed 2, judged with guards sized to what each suite can
resolve. Committed before any round-8 read.

    uv run python scripts/round8_readout.py --out runs/r8-readout
    uv run python scripts/round8_readout.py --round 9 --out runs/r9-readout     # round 9 (PLAN.md): same rule, its own arms

Arms runs/r8-small/{00-trial-0 (4B), 01-trial-1 (0.8B)}, read as runs/r8-<arm>-{docs,semif,scienthoon,wanli2,typesafe,v9};
parents, rows and temperatures exactly as scripts/round7_readout.py.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served, unknowable_report  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import EXTERNALS, boot, knowable, serve  # noqa: E402
from round7_readout import JEV, PARENTS, rows_at  # noqa: E402

ROUNDS = {8: {"4b-s2": ("runs/r8-small/00-trial-0", "P4"), "08b-s2": ("runs/r8-small/01-trial-1", "P08")},
          9: {"9b-a": ("runs/r9-docs/00-trial-0", "P9"), "9b-b": ("runs/r9-docs/01-trial-1", "P9"), "9b-c": ("runs/r9-docs/02-trial-2", "P9"),
              "08b-d": ("runs/r9-docs/03-trial-3", "P08"), "08b-e": ("runs/r9-docs/04-trial-4", "P08")},
          11: {"9b-s4": ("runs/r11-docs/00-trial-0", "P9"), "9b-s5": ("runs/r11-docs/01-trial-1", "P9"),
               "08b-s4": ("runs/r11-docs/02-trial-2", "P08"), "08b-s5": ("runs/r11-docs/03-trial-3", "P08")}}
# round 11 (PLAN.md): the short-state guard pools transfer-v4 dev with transfer-r3 test (1,260 records)
SHORT_EXTRA = {11: {"cand": "runs/r11-{arm}-r3test", "P9": "runs/rc-parent-r3test", "P08": "runs/r11-P08-r3test"}}
GATING_SUITES = ("wanli2", "scienthoon")   # large enough to resolve a -2 pp floor; SemIf-144 and TypeSafe-89 only enter the pooled guard


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--round", type=int, default=8, choices=list(ROUNDS)); a = ap.parse_args()
    ARMS, tag = ROUNDS[a.round], f"r{a.round}"
    jev = [r for r in read_json(JEV) if r["variant"] == "clean"]
    report = {"jev_docs_acc": metrics(jev)["acc"], "arms": {}}
    for arm, (trial, ptag) in ARMS.items():
        needed = [f"runs/{tag}-{arm}-docs/rows.json"] + ([f"{SHORT_EXTRA[a.round]['cand'].format(arm=arm)}/rows.json"] if a.round in SHORT_EXTRA else [])
        if not (Path(trial) / "result.json").exists() or not all(Path(n).exists() for n in needed):
            report["arms"][arm] = "not read yet"; continue
        ptrial, preads = PARENTS[ptag]
        pt, t = (served(read_json(Path(x) / "development/rows.json"), [])[0] for x in (ptrial, trial))
        P, C = rows_at(ptrial, preads, pt), rows_at(trial, {s: f"runs/{tag}-{arm}-{s}" for s in ("docs", *EXTERNALS)}, t)
        if a.round in SHORT_EXTRA:
            extra = SHORT_EXTRA[a.round]
            P["short"] = P["short"] + knowable(serve(ptrial, f"{extra[ptag]}/rows.json", pt)[0])
            C["short"] = C["short"] + knowable(serve(trial, f"{extra['cand'].format(arm=arm)}/rows.json", t)[0])
        rep = {"trial": trial, "parent": ptrial, "temperature": t, "docs": boot(C["docs"], P["docs"], "acc"), "docs_acc": metrics(C["docs"])["acc"],
               "parent_docs_acc": metrics(P["docs"])["acc"], "docs_vs_jev": boot(C["docs"], jev, "acc"),
               "short": {m: boot(C["short"], P["short"], m) for m in ("acc", "brier", "confident_error_rate")},
               "externals": {s: {**boot(C[s], P[s], "acc"), "n": len(C[s]), "gating": s in GATING_SUITES} for s in EXTERNALS},
               "pooled_external": boot([r for s in EXTERNALS for r in C[s]], [r for s in EXTERNALS for r in P[s]], "acc"),
               "unknowable_share": unknowable_report(serve(trial, f"runs/{tag}-{arm}-v9/rows.json", t)[0])["share_at_0_9"]}
        s = rep["short"]
        rep["criteria"] = {"1_docs_lower_above_0": rep["docs"]["ci95"][0] > 0,
                           "2_short_acc_lower_at_least_minus_2pp": s["acc"]["ci95"][0] >= -0.02, "2_short_brier_upper_at_most_0.01": s["brier"]["ci95"][1] <= 0.01,
                           "2_short_confident_errors_upper_at_most_1pp": s["confident_error_rate"]["ci95"][1] <= 0.01,
                           **{f"3_{k}_lower_at_least_minus_2pp": rep["externals"][k]["ci95"][0] >= -0.02 for k in GATING_SUITES},
                           "3_pooled_lower_at_least_minus_1.5pp": rep["pooled_external"]["ci95"][0] >= -0.015, "3_unknowable_at_most_0.05": rep["unknowable_share"] <= 0.05}
        rep["passed"] = all(rep["criteria"].values())
        report["arms"][arm] = rep
    for size in ("9b", "4b", "08b"):
        passing = [(k, v) for k, v in report["arms"].items() if k.startswith(size + "-") and isinstance(v, dict) and v["passed"]]
        report[f"candidate_{size}"] = max(passing, key=lambda kv: kv[1]["docs"]["delta"])[0] if passing else None
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / f"round{a.round}.json", report)
    f = lambda b: f"{100 * b['delta']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]"
    print(f"Jev documents-v1 dev acc {report['jev_docs_acc']:.3f}")
    for arm, r in report["arms"].items():
        if not isinstance(r, dict): print(f"{arm:7} {r}"); continue
        print(f"{arm:7} docs {r['parent_docs_acc']:.3f} -> {r['docs_acc']:.3f} {f(r['docs'])} vs Jev {f(r['docs_vs_jev'])} | short acc {f(r['short']['acc'])} | "
              + " ".join(f"{k}{'' if v['gating'] else '(report)'}:{f(v)}" for k, v in r["externals"].items()) + f" | pooled {f(r['pooled_external'])} | unk {r['unknowable_share']:.3f} "
              f"-> {'PASS' if r['passed'] else 'fail: ' + ', '.join(k for k, v in r['criteria'].items() if not v)}")
    print({k: v for k, v in report.items() if k.startswith("candidate_")})


if __name__ == "__main__":
    main()
