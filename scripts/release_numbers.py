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
RELEASES = {   # arm: trial + where each read lives
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
    rows = lambda path: knowable(serve(trial, path, t)[0])
    out = {"trial": trial, "temperature": t,
           "decision_dev": summary(rows(Path(trial) / "development/rows.json")),
           "transfer_dev": summary(rows(Path(trial) / "transfer/rows.json")),
           "heldout_pairs_both_correct": read_json(Path(trial) / "result.json")["transfer"]["paired_flip"]["both_correct_rate"],
           "unknowable_share_at_0_9": unknowable_report(serve(trial, f"{spec['v9']}/rows.json", t)[0])["share_at_0_9"],
           "mmlu_pro": read_json(f"{spec['v9']}/report.json")["tasks"]["mmlu_pro"]["acc"],
           "locked_transfer": summary(rows(f"{spec['locked']}/transfer/rows.json")),
           "locked_decision": summary(rows(f"{spec['locked']}/decision/rows.json")),
           **{k: summary(rows(f"{spec[k]}/rows.json")) for k in ("docs1_dev", "docs1_test", "docs2", *EXTERNALS)}}
    return out, {k: rows(f"{spec[k]}/rows.json") for k in ("docs1_dev", "docs1_test", "docs2", *EXTERNALS)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True, choices=list(RELEASES)); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    (cand, crows), (parent, prows) = (arm(RELEASES[a.release][k]) for k in ("candidate", "parent"))
    report = {"release": a.release, "candidate": cand, "parent": parent,
              "paired_acc_delta": {k: boot(crows[k], prows[k], "acc") for k in crows},
              "jev": {"docs1_dev": metrics([r for r in read_json("runs/jev-documents-v1/rows.json") if r["variant"] == "clean"])["acc"]}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)
    for k in ("decision_dev", "transfer_dev", "docs1_dev", "docs1_test", "docs2", "locked_transfer", "locked_decision", *EXTERNALS):
        print(f"{k:16} parent {parent[k]['acc']:.3f} / {parent[k]['brier']:.3f}  ->  release {cand[k]['acc']:.3f} / {cand[k]['brier']:.3f}")
    print("T", round(parent["temperature"], 2), "->", round(cand["temperature"], 2), "| pairs", round(parent["heldout_pairs_both_correct"], 3), "->", round(cand["heldout_pairs_both_correct"], 3),
          "| MMLU-Pro", parent["mmlu_pro"], "->", cand["mmlu_pro"], "| unknowable", parent["unknowable_share_at_0_9"], "->", cand["unknowable_share_at_0_9"])


if __name__ == "__main__":
    main()
