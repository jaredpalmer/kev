"""Finish a Jev read of evals/longdoc-v1 whose kev.jev run stopped on its longest states, and record what Jev refused.

    AI_GATEWAY_API_KEY=... uv run python scripts/longdoc_jev_finish.py --suite evals/longdoc-v1 \
        --predictions runs/longdoc-v1-jev-r3/predictions.jsonl --out runs/longdoc-v1-jev

Why: past its context the AI Gateway answers a Jev request with HTTP 400 or 503 (GatewayInternalServerError either way),
so kev.jev --count-refusals counts the 400s but keeps retrying a 503 as a transient outage and stops the read. This takes
the rows kev.jev already wrote (predictions.jsonl: one line per answered record, the same rows evaluate_records keeps),
sends every remaining development record through the same JevPredictor (count_refusals, `--attempts` tries) and lists each
record Jev did not answer in rejected.json with its error. rows.json / report.json (kev.benchmark.summarize over the
answered rows, coverage counting the rest as rejected) / usage.json as kev.jev writes them.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.benchmark import prediction_rows, summarize  # noqa: E402
from kev.predictors import PRICE_PER_MILLION, JevPredictor  # noqa: E402
from kev.suite import ENCODING, digest, load_split, read_jsonl, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="evals/longdoc-v1")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--budget", type=float, default=2.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    records = load_split(a.suite, "development")
    done = {p["id"]: p for p in read_jsonl(a.predictions)}
    prior = {"records": len(done), "input_tokens": sum(p["prediction"]["usage"]["inputTokens"] for p in done.values()),
             "output_tokens": sum(p["prediction"]["usage"].get("outputTokens") or 0 for p in done.values()), "predictions": a.predictions,
             "predictions_sha256": digest(a.predictions)}
    predictor = JevPredictor(os.environ["AI_GATEWAY_API_KEY"], a.budget, 5000, count_refusals=True, attempts=a.attempts)
    rows, rejected = [], []
    try:
        with (out / "predictions.jsonl").open("w", encoding=ENCODING) as f:
            for r in records:
                rid = r["_meta"]["id"]
                if rid in done:
                    rows += done[rid]["rows"]; f.write(json.dumps(done[rid]) + "\n"); continue
                try:
                    pred = predictor(r)
                except Exception as e:   # JevRefused (400/413/422) or a 5xx on every attempt: unanswered, recorded
                    rejected.append({"id": rid, "error_type": type(e).__name__, "error": str(e), "state_tokens": r["_meta"]["state_tokens"]}); continue
                new = prediction_rows(r, pred); rows += new
                f.write(json.dumps({"id": rid, "prediction": pred, "rows": new}) + "\n")
    finally:
        predictor.close()
    write_json(out / "rows.json", rows)
    write_json(out / "rejected.json", rejected)
    report = summarize(rows)
    answered = {row["id"] for row in rows}
    report.update(coverage={"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                            "evaluated_records": len(answered), "evaluated_questions": len(rows), "rejected_records": len(rejected)},
                  suite_sha256=digest(Path(a.suite) / "manifest.json"), split="development", finished_by="scripts/longdoc_jev_finish.py")
    usage = predictor.accounting()
    usage["prior_kev_jev_run"] = prior
    usage["estimated_usd_total"] = (usage["input_tokens"] + prior["input_tokens"]) * PRICE_PER_MILLION / 1e6
    report["provider"] = usage
    write_json(out / "report.json", report)
    write_json(out / "usage.json", usage)
    print(json.dumps({"answered": len(answered), "rejected": len(rejected), "usage": usage}, indent=1))


if __name__ == "__main__":
    main()
