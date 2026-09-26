"""Score a predictor on a frozen suite partition (or your own labelled JSONL).

    uv run python -m kev.benchmark --run runs/<run>/checkpoint --suite evals/<v>/decision-<v> --out runs/<name>
    uv run python -m kev.benchmark --remote http://127.0.0.1:8008 --suite ... --out ...      # any System One endpoint

Every prediction becomes one row per question (prediction_rows); kev.metrics scores rows; evaluate_records writes
predictions.jsonl, rows.json and report.json. Predictors live in kev.predictors.
"""
import argparse
import functools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kev.api import question_keys, with_date_facts
from kev.checkpoint import LoadOptions
from kev.contrastive import paired_flip
from kev.data import api_request, load_records
from kev.device import default_device
from kev.metrics import EPSILON, grouped_metrics, metrics, unknowable_report
from kev.model import ROW_PASS_TOKENS, ContextOverflow
from kev.predictors import LocalPredictor, RemotePredictor, RotationAveraged
from kev.suite import CONTEXT, ENCODING, digest, load_split, read_manifest, record_digest, write_json


def labels(q):
    """(option keys, label index) of a labelled request question."""
    keys = question_keys(q["type"], q.get("criteria"))
    return keys, keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])


def validate_distribution(raw, keys):
    if set(raw) != set(keys):
        raise ValueError("probability keys do not match requested options")
    p = np.array([raw[k] for k in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("non-finite or out-of-range probabilities")
    total = float(p.sum())
    if total <= 0 or abs(total - 1) > max(1e-5, len(keys) * 0.005 + 1e-8):
        raise ValueError(f"invalid probability sum: {total}")
    return p / total, total


def prediction_rows(record, prediction):
    if set(prediction["probabilities"]) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    meta = record["_meta"]
    rows = []
    for qid, q in record["questions"].items():
        keys, y = labels(q)
        p, total = validate_distribution(prediction["probabilities"][qid], keys)
        row = {"id": meta["id"], "group": meta["group_id"], "question": qid,
               "source": meta["source"], "task": q["src"], "type": q["type"],
               "variant": meta["variant"], "keys": keys, "label": y, "control_id": meta.get("control_id"),
               "pair_id": meta.get("pair_id"), "sibling": meta.get("sibling"), # suites frozen before parent_id existed stored the parent's id in group_id for variants
               "parent": meta.get("parent_id") or (meta["id"] if meta["variant"] == "clean" else meta["group_id"]),
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        if "logits" in prediction:
            raw_logits = prediction["logits"][qid]
            if set(raw_logits) != set(keys) or not all(math.isfinite(raw_logits[k]) for k in keys):
                raise ValueError("logit keys or values do not match the requested options")
            row["logits"] = [float(raw_logits[k]) for k in keys]
            row["inference_temperature"] = prediction["inference_temperature"]
        if "kernels" in prediction: row["kernels"] = prediction["kernels"]   # a long row off the fp32-exact kernels (LocalPredictor)
        rows.append(row)
    return rows


def summarize(rows, temperature=1.0, heldout_sources=()):
    """Report over benchmark rows. heldout_sources: sources the scored model never trained on; their tasks are also
    reported as a separate block."""
    clean = [r for r in rows if r["variant"] == "clean"]
    tasks = grouped_metrics(clean, "task")
    variants = grouped_metrics(rows, "variant")
    lookup = {(r["id"], r["question"]): r for r in clean}
    diffs, flips = [], []
    for row in rows:
        if row["variant"] == "permuted" and row["type"] == "choice":
            original = lookup[(row["parent"], row["question"])]
            aligned = [row["p"][row["keys"].index(k)] for k in original["keys"]]
            diffs.append(float(np.max(np.abs(np.array(aligned) - original["p"]))))
            flips.append(int(np.argmax(aligned) != np.argmax(original["p"])))
    knowable = [r for r in clean if r["source"] != "unknowable"]     # unknowable records are scored on confidence, never on accuracy
    return {"objective": -float(np.mean([v["nll"] for k, v in tasks.items() if not k.startswith("unknowable_") or k.startswith("unknowable_control")])),
            "paired_flip": paired_flip(clean), "unknowable": unknowable_report(clean),
            "clean": metrics(knowable), "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics([r for r in clean if r["source"] in heldout_sources], "task") if any(r["source"] in heldout_sources for r in clean) else {},
            "permutation": {"n": len(diffs), "mean_max_delta": float(np.mean(diffs)) if diffs else None,
                            "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "calibrated_clean": metrics(knowable, temperature),
            "metric_policy": {"version": 2, "selective_ties": "whole_confidence_groups",
                              "coverage_at_error": "in-sample maximum over confidence thresholds; not a deployed error guarantee",
                              "aurc": "right-step integral over whole confidence groups",
                              "confident_error_rate": "high-confidence errors divided by all questions",
                              "error_rate_at_0_9": "errors divided by questions accepted at p_max >= 0.9",
                              "nll": "exact from logits when recorded; otherwise from floored probabilities",
                              "nll_floor": EPSILON, "renormalize_returned_probabilities": True,
                              "raw_sums_outside_1e_5": sum(abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows),
                              "returned_zeros": sum(r["zero_count"] for r in rows)}}


def predictions(records, predictor):
    """Yield, in record order, a zero-argument callable that returns predictor(record) or raises what it raised. A
    predictor with `concurrency` > 1 (RemotePredictor: one independent HTTP request per call) keeps that many calls in
    flight on a thread pool; in-process predictors (one GPU) have no `concurrency` and run one record at a time. Either
    way the caller sees each outcome in the same order as the plain sequential loop."""
    workers = getattr(predictor, "concurrency", 1)
    if workers <= 1 or len(records) <= 1:
        for record in records:
            yield functools.partial(predictor, record)
        return
    executor = ThreadPoolExecutor(max_workers=min(workers, len(records)))
    try:
        for future in [executor.submit(predictor, record) for record in records]:
            yield future.result
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def evaluate_records(records, predictor, directory, temperature=1.0, heldout_sources=(), skip_overlong=False):
    """skip_overlong: for external data that was not admitted to a context (--data, or an eval-only suite frozen as published),
    records the predictor cannot encode are counted in coverage["rejected_records"] and listed in rejected.json instead of
    aborting. Admitted suites never trigger it; reports must state how rejected records enter any headline number."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    coverage = {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                "evaluated_records": 0, "evaluated_questions": 0, "rejected_records": 0, "truncated_records": 0}
    rows, latencies, rejected = [], [], []
    with (directory / "predictions.jsonl").open("w", encoding=ENCODING) as output:
        preds = predictions(records, predictor)
        for record in records:
            try:
                pred = next(preds)()
                new_rows = prediction_rows(record, pred)
            except ContextOverflow as error:
                if skip_overlong:
                    coverage["rejected_records"] += 1; rejected.append({"id": record["_meta"]["id"], "error": str(error)}); continue
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            except Exception as error:
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            output.write(json.dumps({"request_sha256": record_digest(api_request(record)), "id": record["_meta"]["id"],
                                     "prediction": pred, "rows": new_rows}, allow_nan=False) + "\n")
            output.flush()
            rows.extend(new_rows)
            latencies.append(pred["latency_ms"])
            coverage["evaluated_records"] += 1
            coverage["evaluated_questions"] += len(new_rows)
            if coverage["evaluated_records"] % 50 == 0:
                print(f"evaluated {coverage['evaluated_records']}/{len(records)}", flush=True)
    write_json(directory / "rows.json", rows)
    if rejected: write_json(directory / "rejected.json", rejected)
    report = summarize(rows, temperature, heldout_sources)
    report.update(coverage=coverage, latency_ms={"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, .95))},
                  calibration={"inference_temperature": getattr(predictor, "temperature", None),
                               "additional_temperature": temperature, "logits_recorded": all("logits" in r for r in rows)})
    long = [r for r in rows if "kernels" in r]
    if long:   # absent when every row ran the exact kernels, so such reports are unchanged
        report["long_rows"] = {"count": len(long), "records": len({r["id"] for r in long}), "kernels": sorted({r["kernels"] for r in long}), "threshold": ROW_PASS_TOKENS}
    write_json(directory / "report.json", report)
    return report, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="checkpoint dir or Hub id (local scoring)")
    ap.add_argument("--remote", help="base URL of a System One-compatible endpoint to score instead of a local checkpoint")
    ap.add_argument("--remote-model", default="kev-latest")
    ap.add_argument("--remote-concurrency", type=int, default=1, help="requests kept in flight against --remote (1 = sequential); rows and their order do not depend on it")
    ap.add_argument("--suite", help="frozen suite directory (scores its development partition)")
    ap.add_argument("--data", help="your own labelled requests, one JSON object per line (kev.data.load_records); an alternative to --suite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=default_device())
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--split", choices=["development", "calibration", "train"], default="development",
                    help="suite partition to score (train: teacher predictions for distillation; --allow-test reads the locked test instead)")
    ap.add_argument("--date_facts", action="store_true", help="apply kev.api.with_date_facts to every state before scoring (the opt-in serving preprocessor); reported in report.json")
    ap.add_argument("--rotations", type=int, default=1, help="average every Choice question over this many cyclic option rotations (kev.predictors.RotationAveraged); 1 = one order")
    a = ap.parse_args()
    if bool(a.run) == bool(a.remote): ap.error("give exactly one of --run or --remote")
    if a.rotations < 1: ap.error("--rotations must be >= 1")
    if a.remote_concurrency < 1: ap.error("--remote-concurrency must be >= 1")
    if bool(a.suite) == bool(a.data): ap.error("give exactly one of --suite or --data")
    if a.data:
        records, heldout, split, source_hash = load_records(a.data), [], "custom", digest(Path(a.data))
        context, skip_overlong = CONTEXT, True
    else:
        split = "test" if a.allow_test else a.split
        records = load_split(a.suite, split, allow_test=a.allow_test)
        manifest = read_manifest(a.suite)
        heldout = manifest["holdout_sources"]; source_hash = digest(Path(a.suite) / "manifest.json")
        context, skip_overlong = manifest.get("context", CONTEXT), bool(manifest.get("eval_only"))
    if a.date_facts:
        records = [{**r, "state": with_date_facts(r["state"])} for r in records]
    predictor = RemotePredictor(a.remote, a.remote_model, os.environ.get("KEV_REMOTE_API_KEY", "local"), concurrency=a.remote_concurrency) if a.remote else LocalPredictor(a.run, a.device, LoadOptions.from_env(), context=context)
    scorer = RotationAveraged(predictor, a.rotations) if a.rotations > 1 else predictor
    report, _ = evaluate_records(records, scorer, a.out, heldout_sources=tuple(heldout), skip_overlong=skip_overlong)
    report.update(suite_sha256=source_hash, data=a.data, date_facts=a.date_facts, rotations=a.rotations, run=a.run or a.remote, split=split,
                  calibration_applied=predictor.temperature != 1.0 if not a.remote else None,
                  remote={"base_url": a.remote, "requested_model": a.remote_model, "served_model": predictor.served_model, "concurrency": a.remote_concurrency} if a.remote else None)
    write_json(Path(a.out) / "report.json", report)
    print(json.dumps({"objective": report["objective"], "clean": report["clean"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
