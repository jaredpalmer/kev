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

from kev.data import materialize
from kev.model import MAX_STATE, load_tokenizer, user_tokens
from kev.suite import SERVING_CONTEXT_8K, load_split, read_json, write_json

LENGTH_BUCKETS = ((0, MAX_STATE), (MAX_STATE, 2048), (2048, SERVING_CONTEXT_8K["max_state"]))   # inside the training context / longer / much longer (typesafe-v1's admission)


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


def accuracy_by_state_length(rows, records, tok):
    """Plain accuracy of the answered rows bucketed by the state's token count: where Kev's documents are longer than
    anything in its training context, this is what separates the length effect from the task."""
    tokens = {r["_meta"]["id"]: len(user_tokens(tok, materialize(r)["state"])) for r in records}
    out = {}
    for lo, hi in LENGTH_BUCKETS:
        hits = [int(max(range(len(row["p"])), key=row["p"].__getitem__) == row["label"]) for row in rows if lo <= tokens[row["id"]] < hi]
        out[f"{lo}-{hi}"] = {"rows": len(hits), "acc": sum(hits) / len(hits) if hits else None}
    return out


def score_run(directory, records, tok=None):
    targets = {r["_meta"]["id"]: r["_meta"]["target"] for r in records}
    rows = read_json(Path(directory) / "rows.json")
    scores = {row["id"]: score(dict(zip(row["keys"], row["p"])), targets[row["id"]]) for row in rows}
    answered = [r for r in records if r["_meta"]["id"] in scores]
    out = {"evaluated": case_means(scores, answered), "all_rows": case_means(scores, records)}
    if tok is not None: out["accuracy_by_state_tokens"] = accuracy_by_state_length(rows, records, tok)
    return out


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
    ap.add_argument("--tokenizer", help="base tokenizer (e.g. Qwen/Qwen3.5-4B-Base): also report accuracy by state length")
    a = ap.parse_args()
    records = load_split(a.suite, "development")
    tok = load_tokenizer(a.tokenizer) if a.tokenizer else None
    result = {"suite": a.suite, "published": score_published(records), "runs": {run: score_run(run, records, tok) for run in a.run}}
    for name, r in list(result["published"].items()) + list(result["runs"].items()):
        head, every = r["evaluated"], r["all_rows"]
        print(f"{name:40s} agreement {head['equal_case_modal_agreement']:.3f}  tvd {head['equal_case_total_variation']:.3f}"
              f"  ({head['rows']}/{every['rows']} rows answered; all rows {every['equal_case_modal_agreement']:.3f} / {every['equal_case_total_variation']:.3f})")
        for bucket, b in r.get("accuracy_by_state_tokens", {}).items():
            print(f"{'':40s}   state {bucket} tokens: {b['rows']} rows, acc {b['acc']:.2f}" if b["rows"] else f"{'':40s}   state {bucket} tokens: 0 rows")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(a.out, result)


if __name__ == "__main__":
    main()
