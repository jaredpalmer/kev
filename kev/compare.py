import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev.metrics import paired_bootstrap
from kev.suite import read_json, write_json


def nll_sensitivity(rows, floor):
    tasks = defaultdict(list)
    for row in rows:
        if row["variant"] == "clean":
            tasks[row["task"]].append(-math.log(max(row["p"][row["label"]], floor)))
    return float(np.mean([np.mean(v) for v in tasks.values()]))


def none_diagnostics(rows):
    out = {}
    for variant in ("none_present", "none_absent"):
        subset = [r for r in rows if r["variant"] == variant]
        values = [r["p"][r["keys"].index("none_of_these")] for r in subset]
        # No observations means an undefined diagnostic, not a measured zero.
        out[variant] = {"n": len(values), "mean_p_none": float(np.mean(values)) if values else None,
                        "p_none_above_half": float(np.mean(np.array(values) > .5)) if values else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    candidate, reference = Path(a.candidate), Path(a.reference)
    left, right = [read_json(p / "report.json") for p in (candidate, reference)]
    if not left.get("suite_sha256") or left["suite_sha256"] != right.get("suite_sha256"):
        raise ValueError("comparison requires matching frozen-suite hashes")
    lr, rr = [read_json(p / "rows.json") for p in (candidate, reference)]
    result = {"candidate": str(candidate), "reference": str(reference), "suite_sha256": left["suite_sha256"],
              "clean": {"candidate": left["clean"], "reference": right["clean"]},
              "paired": {metric: paired_bootstrap(lr, rr, metric=metric) for metric in ("nll", "acc", "brier")},
              "nll_floor_sensitivity": {str(floor): {"candidate_macro_nll": nll_sensitivity(lr, floor),
                                                   "reference_macro_nll": nll_sensitivity(rr, floor)} for floor in (1e-3, 1e-6, 1e-9)},
              "none_of_the_above": {"candidate": none_diagnostics(lr), "reference": none_diagnostics(rr)},
              "caveats": ["Uncalibrated returned distributions; probabilities normalized for scoring; raw responses preserved.",
                          "Historical kev checkpoint trained on all six sources; its heldout-source status does not transfer to this study.",
                          "Jev uses a hosted alias and rounds some probabilities to zero; NLL depends on the declared floor.",
                          "CPU local latency and hosted network latency are not comparable hardware benchmarks.",
                          "This is a development-suite comparison, not the locked final test."]}
    write_json(a.out, result)
    print("task                 kev acc    Jev acc    kev NLL    Jev NLL")
    for task in left["tasks"]:
        l, r = left["tasks"][task], right["tasks"][task]
        print(f"{task:20} {l['acc']:8.3f} {r['acc']:10.3f} {l['nll']:10.3f} {r['nll']:10.3f}")
    print(json.dumps({"paired": result["paired"], "none_of_the_above": result["none_of_the_above"]}, indent=2))


if __name__ == "__main__":
    main()
