"""Equal-case modal agreement and total-variation distance on evals/external/typesafe-v1, SemIf's protocol for the public
TypeSafe cases: per row, agreement = argmax(p) == reference argmax and TVD = 1/2 * sum |p - reference|; rows are averaged
within each case (group) and cases are averaged with equal weight. Published answers carried in the suite (TypeSafe's
Jev, plus the other models the eval page publishes) are scored the same way, so every number is on identical items.

    uv run python scripts/compare_typesafe.py --suite evals/external/typesafe-v1 --run runs/n2-9b-typesafe-v1 --run runs/jev-typesafe-v1

A run's rejected records (over the serving context, see rejected.json) have no prediction: `evaluated` covers the rows the
model answered and is the headline, reported with the answered/total count; `all_rows` counts each missing row as
agreement 0 and TVD 1.
"""
import argparse
from pathlib import Path

from kev.suite import load_split, read_json, write_json


def case_means(scores, records):
    """scores: {record id: (agreement, tvd)} -> equal-case means over the records given (missing ids score 0 / 1)."""
    cases = {}
    for rec in records:
        cases.setdefault(rec["_meta"]["group_id"], []).append(scores.get(rec["_meta"]["id"], (0.0, 1.0)))
    mean = lambda xs: sum(xs) / len(xs)
    return {"rows": len(records), "cases": len(cases),
            "equal_case_modal_agreement": mean([mean([a for a, _ in rows]) for rows in cases.values()]),
            "equal_case_total_variation": mean([mean([t for _, t in rows]) for rows in cases.values()])}


def score(p, target):
    """p, target: {option id: probability} over the same ids; the reference argmax is unique by construction of the suite."""
    modal = max(p, key=p.get)
    return float(modal == max(target, key=target.get)), sum(abs(p[k] - target[k]) for k in target) / 2


def score_run(directory, records):
    targets = {r["_meta"]["id"]: r["_meta"]["target"] for r in records}
    scores = {}
    for row in read_json(Path(directory) / "rows.json"):
        scores[row["id"]] = score(dict(zip(row["keys"], row["p"])), targets[row["id"]])
    answered = [r for r in records if r["_meta"]["id"] in scores]
    return {"evaluated": case_means(scores, answered), "all_rows": case_means(scores, records)}


def score_published(records):
    out = {}
    for rec in records:
        for name, pub in rec["_meta"]["published"].items():
            out.setdefault(name, {"model": pub["model"], "scores": {}})["scores"][rec["_meta"]["id"]] = score(pub["p"], rec["_meta"]["target"])
    return {name: {"model": v["model"], "evaluated": case_means(v["scores"], [r for r in records if r["_meta"]["id"] in v["scores"]]),
                   "all_rows": case_means(v["scores"], records)} for name, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="evals/external/typesafe-v1")
    ap.add_argument("--run", action="append", default=[], help="kev.benchmark output directory (repeatable)")
    ap.add_argument("--out", help="write the comparison as JSON")
    a = ap.parse_args()
    records = load_split(a.suite, "development")
    result = {"suite": a.suite, "published": score_published(records), "runs": {run: score_run(run, records) for run in a.run}}
    for name, r in list(result["published"].items()) + list(result["runs"].items()):
        head, every = r["evaluated"], r["all_rows"]
        print(f"{name:40s} agreement {head['equal_case_modal_agreement']:.3f}  tvd {head['equal_case_total_variation']:.3f}"
              f"  ({head['rows']}/{every['rows']} rows answered; all rows {every['equal_case_modal_agreement']:.3f} / {every['equal_case_total_variation']:.3f})")
    if a.out:
        write_json(a.out, result)


if __name__ == "__main__":
    main()
