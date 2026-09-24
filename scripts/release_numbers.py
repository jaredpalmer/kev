"""Every number a model card prints for a release candidate and its parent, from committed rows, each served at the
temperature fitted on its own decision-v7 development rows (the one written into head.pt). One JSON per release, the
`source` that docs/claims.json points at. The release spec (experiments/releases/<release>.json) names each arm's trial
and where each of its reads lives; comparisons use kev.rounds' registered paired read.

    uv run python scripts/release_numbers.py --release kev-4b-r8 --out runs/release/kev-4b-r8.json
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.metrics import metrics, served_at, unknowable_report  # noqa: E402
from kev.rounds import paired, served_clean, temperature  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

KEYS = ("n", "acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error")
READS = ("docs1_dev", "docs1_test", "docs2", "long2", "r6test", "long3", "hard_dev", "devtools_dev", "hard_test", "devtools_test",
         "semif", "scienthoon", "wanli2", "typesafe")   # optional per release; the report keeps this order


def summary(rows):
    m = metrics(rows)
    return {k: m[k] for k in KEYS if k in m}


def arm(spec, drop):
    trial = spec["trial"]
    t = temperature(trial, ".")
    rows = lambda path: [r for r in served_at(read_json(path), t) if r["id"] not in drop]
    read = {k: rows(f"{spec[k]}/rows.json") for k in READS if k in spec}
    out = {"trial": trial, "temperature": t,
           "decision_dev": summary(rows(Path(trial) / "development/rows.json")),
           "transfer_dev": summary(rows(Path(trial) / "transfer/rows.json")),
           "heldout_pairs_both_correct": read_json(Path(trial) / "result.json")["transfer"]["paired_flip"]["both_correct_rate"],
           "unknowable_share_at_0_9": unknowable_report(served_clean(read_json(f"{spec['v9']}/rows.json"), t))["share_at_0_9"],
           "mmlu_pro": read_json(f"{spec['v9']}/report.json")["tasks"]["mmlu_pro"]["acc"],
           "locked_transfer": summary(rows(f"{spec['locked']}/transfer/rows.json")),
           "locked_decision": summary(rows(f"{spec['locked']}/decision/rows.json")),
           **{k: summary(read[k]) for k in read}}
    long = {k: [r for r in read[k] if r["source"] == "longstate"] for k in ("long2", "long3") if k in read}   # the buried questions the rule scores
    out.update({f"{k}_buried": summary(v) for k, v in long.items()})
    return out, {**read, **{f"{k}_buried": v for k, v in long.items()}}


def main():
    releases = sorted(p.stem for p in (ROOT / "experiments/releases").glob("*.json"))
    ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True, choices=releases); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    release = read_json(ROOT / f"experiments/releases/{a.release}.json")
    (cand, crows), (parent, prows) = (arm(release[k], set(release["drop_ids"])) for k in ("candidate", "parent"))
    report = {"release": a.release, "candidate": cand, "parent": parent,
              "paired_acc_delta": {k: paired(crows[k], prows[k], "acc") for k in crows},
              "jev": {"docs1_dev": metrics([r for r in read_json("runs/jev-documents-v1/rows.json") if r["variant"] == "clean"])["acc"]}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)
    for k in [k for k in cand if isinstance(cand[k], dict) and "acc" in cand[k] and k in parent]:
        print(f"{k:16} parent {parent[k]['acc']:.3f} / {parent[k]['brier']:.3f}  ->  release {cand[k]['acc']:.3f} / {cand[k]['brier']:.3f}")
    print("T", round(parent["temperature"], 2), "->", round(cand["temperature"], 2), "| pairs", round(parent["heldout_pairs_both_correct"], 3), "->", round(cand["heldout_pairs_both_correct"], 3),
          "| MMLU-Pro", parent["mmlu_pro"], "->", cand["mmlu_pro"], "| unknowable", parent["unknowable_share_at_0_9"], "->", cand["unknowable_share_at_0_9"])


if __name__ == "__main__":
    main()
