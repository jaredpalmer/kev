"""Accuracy by state length on evals/round4/longstate-v1 (PLAN.md round 4, item 4.12; PLAN_27b A3), paired against the
same primaries unburied (longstate_control), from a kev.benchmark rows.json.

    uv run python scripts/longstate_report.py runs/r4-kev-4b-longstate-2 [runs/<other> ...]
"""
import re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.metrics import cluster_resamples, scored_rows  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

LENGTH = re.compile(r"^longstate_(\d+)_")   # build_long_states tags each buried question's task with its length


def by_length(rows, samples=2000, seed=0):
    """{length: accuracy buried, accuracy of the same questions unburied, n, and the paired 95% CI of the difference}.
    The bootstrap resamples record groups (kev.metrics.cluster_resamples), so sibling questions of one primary move
    together, as in every other bootstrap in the repo."""
    ok = lambda r: int(np.argmax(r["p"]) == r["label"])
    clean = scored_rows(rows)
    control = {(r["group"], r["question"]): ok(r) for r in clean if r["source"] == "longstate_control"}
    pairs = defaultdict(list)
    for r in clean:
        m = LENGTH.match(r["task"])
        if r["source"] == "longstate" and m and (r["group"], r["question"]) in control:
            pairs[int(m.group(1))].append({"source": r["source"], "group": r["group"], "buried": ok(r), "unburied": control[(r["group"], r["question"])]})
    out = {}
    for length, p in sorted(pairs.items()):
        d = np.asarray([x["buried"] - x["unburied"] for x in p], dtype=float)
        boots = [d[idx].mean() for idx in cluster_resamples(p, samples, seed)]
        out[str(length)] = {"n": len(p), "buried": float(np.mean([x["buried"] for x in p])), "unburied": float(np.mean([x["unburied"] for x in p])),
                            "delta": float(d.mean()), "ci95": np.quantile(boots, [0.025, 0.975]).tolist(), "unit": "record group"}
    return out


def main():
    for run in sys.argv[1:]:
        report = by_length(read_json(Path(run) / "rows.json"))
        write_json(Path(run) / "longstate.json", report)
        print(run)
        for length, r in report.items():
            print(f"  {length:>5} tokens  n={r['n']:3}  buried {r['buried']:.3f}  unburied {r['unburied']:.3f}  delta {r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
