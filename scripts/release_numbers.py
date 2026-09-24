"""Every number a model card prints for a release candidate and its parent, from committed rows, each served at the
temperature fitted on its own decision-v7 development rows (the one written into head.pt). One JSON per release, the
`source` that docs/claims.json points at.

    uv run python scripts/release_numbers.py --release kev-4b-r8 --out runs/release/kev-4b-r8.json
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics, served, unknowable_report  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import EXTERNALS, boot, knowable, serve  # noqa: E402

KEYS = ("n", "acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error")
READS = ("docs1_dev", "docs1_test", "docs2", "long2", "r6test", "long3", "hard_dev", "devtools_dev", "hard_test", "devtools_test", *EXTERNALS)   # optional per release
DUPLICATE_IDS = {"codereviewer/cls-test/13657", "codereviewer/cls-test/19245"}   # devtools-v1 ids used twice (PLAN.md round 10 amendment)
RELEASES = {   # arm: trial + where each read lives
    "kev-27b-v2": {   # PLAN_27b B1 v2; the comparison column is the released Kev-9B (the rule's reference), not a parent
        "candidate": {"trial": "runs/release/kev-27b-v2", "docs1_dev": "runs/r6-27bv2-s2-docs", "v9": "runs/r6-27bv2-s2-v9", "locked": "runs/locked/kev-27b-v2-ungated",
                      "long2": "runs/r6-27bv2-s2-long", "r6test": "runs/r6c-27b-cand-r6test", "long3": "runs/r6c-27b-cand-long3", **{s: f"runs/r6-27bv2-s2-{s}" for s in EXTERNALS}},
        "parent": {"trial": "runs/night2-9b-du/00-trial-0", "docs1_dev": "runs/docs1-P9", "v9": "runs/n2-9b-du-v9", "locked": "runs/locked/kev-9b-night2-du-ungated",
                   "long2": "runs/r5r-P9-long", "r6test": "runs/r6c-27b-parent-r6test", "long3": "runs/r6c-27b-parent-long3",
                   "semif": "runs/r5r-P9-semif", "scienthoon": "runs/r5r-P9-scienthoon", "wanli2": "runs/r6-P9-wanli2", "typesafe": "runs/r5r-P9-typesafe"},
    },
    "kev-4b-r10": {   # PLAN.md round 10: the round-8 Kev-4B + one epoch on hard-v1 + devtools-v1
        "candidate": {"trial": "runs/release/kev-4b-r10", "docs1_dev": "runs/r10-4b-skills-docs", "v9": "runs/r10-4b-skills-v9", "locked": "runs/locked/kev-4b-r10-ungated",
                      "hard_dev": "runs/r10-4b-skills-hard", "devtools_dev": "runs/r10-4b-skills-devtools", "hard_test": "runs/r10c-4b-cand-hardtest", "devtools_test": "runs/r10c-4b-cand-devtest",
                      **{s: f"runs/r10-4b-skills-{s}" for s in EXTERNALS}},
        "parent": {"trial": "runs/r8-small/00-trial-0", "docs1_dev": "runs/r8-4b-s2-docs", "v9": "runs/r8-4b-s2-v9", "locked": "runs/locked/kev-4b-r8-ungated",
                   "hard_dev": "runs/hv1-P4r8", "devtools_dev": "runs/dt1-P4r8", "hard_test": "runs/r10c-4b-parent-hardtest", "devtools_test": "runs/r10c-4b-parent-devtest",
                   **{s: f"runs/r8-4b-s2-{s}" for s in EXTERNALS}},
    },
    "kev-08b-r15": {   # PLAN.md round 15: the released Kev-0.8B + one epoch on documents-v1 train + hard-v1 + devtools-v1, replay 6,000, lr 2e-5
        "candidate": {"trial": "runs/release/kev-08b-r15", "docs1_dev": "runs/r15-08b-a-docs", "docs1_test": "runs/r15c-08b-cand-docs1test", "docs2": "runs/r15c-08b-cand-docs2",
                      "hard_dev": "runs/r15-08b-a-hard", "devtools_dev": "runs/r15-08b-a-devtools", "hard_test": "runs/r15c-08b-cand-hardtest", "devtools_test": "runs/r15c-08b-cand-devtest",
                      "v9": "runs/r15-08b-a-v9", "locked": "runs/locked/kev-08b-r15-ungated", **{s: f"runs/r15-08b-a-{s}" for s in EXTERNALS}},
        "parent": {"trial": "runs/night2-08b-du2/00-trial-0", "docs1_dev": "runs/docs1-P08", "docs1_test": "runs/r11c-08b-parent-docs1test", "docs2": "runs/r11c-08b-parent-docs2",
                   "hard_dev": "runs/hv1-P08", "devtools_dev": "runs/dt1-P08", "hard_test": "runs/r12c-08b-parent-hardtest", "devtools_test": "runs/r12c-08b-parent-devtest",
                   "v9": "runs/r5r-P08-v9", "locked": "runs/locked/kev-08b-night2-du-ungated", "semif": "runs/r5r-P08-semif", "scienthoon": "runs/r5r-P08-scienthoon",
                   "wanli2": "runs/r6-P08-wanli2", "typesafe": "runs/r5r-P08-typesafe"},
    },
    "kev-08b-r11": {   # PLAN.md round 11: the released Kev-0.8B + one epoch on documents-v1 train, replay 6,000, lr 2e-5 (seed 5)
        "candidate": {"trial": "runs/release/kev-08b-r11", "docs1_dev": "runs/r11-08b-s5-docs", "docs1_test": "runs/r11c-08b-cand-docs1test", "docs2": "runs/r11c-08b-cand-docs2",
                      "v9": "runs/r11-08b-s5-v9", "locked": "runs/locked/kev-08b-r11-ungated", **{s: f"runs/r11-08b-s5-{s}" for s in EXTERNALS}},
        "parent": {"trial": "runs/night2-08b-du2/00-trial-0", "docs1_dev": "runs/docs1-P08", "docs1_test": "runs/r11c-08b-parent-docs1test", "docs2": "runs/r11c-08b-parent-docs2",
                   "v9": "runs/r5r-P08-v9", "locked": "runs/locked/kev-08b-night2-du-ungated", "semif": "runs/r5r-P08-semif", "scienthoon": "runs/r5r-P08-scienthoon",
                   "wanli2": "runs/r6-P08-wanli2", "typesafe": "runs/r5r-P08-typesafe"},
    },
    "kev-4b-r8": {
        "candidate": {"trial": "runs/r8-small/00-trial-0", "docs1_dev": "runs/r8-4b-s2-docs", "docs1_test": "runs/r8c-4b-cand-docs1test", "docs2": "runs/r8c-4b-cand-docs2",
                      "v9": "runs/r8-4b-s2-v9", "locked": "runs/locked/kev-4b-r8-ungated", **{s: f"runs/r8-4b-s2-{s}" for s in EXTERNALS}},
        "parent": {"trial": "runs/night2-4b-du/00-trial-0", "docs1_dev": "runs/docs1-P4", "docs1_test": "runs/r8c-4b-parent-docs1test", "docs2": "runs/r8c-4b-parent-docs2",
                   "v9": "runs/n2-4b-du-v9", "locked": "runs/locked/kev-4b-night2-du-ungated", "semif": "runs/r5r-P4-semif", "scienthoon": "runs/r5r-P4-scienthoon",
                   "wanli2": "runs/r6-P4-wanli2", "typesafe": "runs/r5r-P4-typesafe"},
    },
}


def summary(rows):
    m = metrics(rows)
    return {k: m[k] for k in KEYS if k in m}


def arm(spec):
    trial = spec["trial"]
    t = served(read_json(Path(trial) / "development/rows.json"), [])[0]
    rows = lambda path: [r for r in knowable(serve(trial, path, t)[0]) if r["id"] not in DUPLICATE_IDS]
    read = {k: rows(f"{spec[k]}/rows.json") for k in READS if k in spec}
    out = {"trial": trial, "temperature": t,
           "decision_dev": summary(rows(Path(trial) / "development/rows.json")),
           "transfer_dev": summary(rows(Path(trial) / "transfer/rows.json")),
           "heldout_pairs_both_correct": read_json(Path(trial) / "result.json")["transfer"]["paired_flip"]["both_correct_rate"],
           "unknowable_share_at_0_9": unknowable_report(serve(trial, f"{spec['v9']}/rows.json", t)[0])["share_at_0_9"],
           "mmlu_pro": read_json(f"{spec['v9']}/report.json")["tasks"]["mmlu_pro"]["acc"],
           "locked_transfer": summary(rows(f"{spec['locked']}/transfer/rows.json")),
           "locked_decision": summary(rows(f"{spec['locked']}/decision/rows.json")),
           **{k: summary(read[k]) for k in read}}
    long = {k: [r for r in read[k] if r["source"] == "longstate"] for k in ("long2", "long3") if k in read}   # the buried questions the rule scores
    out.update({f"{k}_buried": summary(v) for k, v in long.items()})
    return out, {**read, **{f"{k}_buried": v for k, v in long.items()}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True, choices=list(RELEASES)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (cand, crows), (parent, prows) = (arm(RELEASES[a.release][k]) for k in ("candidate", "parent"))
    report = {"release": a.release, "candidate": cand, "parent": parent,
              "paired_acc_delta": {k: boot(crows[k], prows[k], "acc") for k in crows},
              "jev": {"docs1_dev": metrics([r for r in read_json("runs/jev-documents-v1/rows.json") if r["variant"] == "clean"])["acc"]}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)
    for k in [k for k in cand if isinstance(cand[k], dict) and "acc" in cand[k] and k in parent]:
        print(f"{k:16} parent {parent[k]['acc']:.3f} / {parent[k]['brier']:.3f}  ->  release {cand[k]['acc']:.3f} / {cand[k]['brier']:.3f}")
    print("T", round(parent["temperature"], 2), "->", round(cand["temperature"], 2), "| pairs", round(parent["heldout_pairs_both_correct"], 3), "->", round(cand["heldout_pairs_both_correct"], 3),
          "| MMLU-Pro", parent["mmlu_pro"], "->", cand["mmlu_pro"], "| unknowable", parent["unknowable_share_at_0_9"], "->", cand["unknowable_share_at_0_9"])


if __name__ == "__main__":
    main()
