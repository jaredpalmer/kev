"""Score a System One-compatible endpoint on the frozen typed-decisions suite (zero-shot, no training).

    uv run python scripts/score_typed_decisions.py \
        --suite evals/external/typed-decisions-v1 --base-url http://127.0.0.1:8009 \
        --out runs/typed-decisions-kev4b

Columns match the dataset card so the row can sit next to Uniform/Prior/MiniLM/ModernBERT/Jev 1.13.0: accuracy,
soft accuracy (argmax of gold), macro F1, KL(gold||pred), total variation, Brier, ECE, Score MAE and within-one-level
(Score only), and ms per case. Metrics are reported overall, per workflow and per question.

Gold is a teacher distribution, so every distribution-shaped metric is computed against it, not against a hard label.
Records whose state exceeds the 384-token serving cap are counted as wrong and listed; they are never truncated.
"""
import argparse
import json
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from kev.suite import load_split, write_json

EPS = 1e-9
# state cap used by training; the suite flags rows over this many characters (~4 chars/token)
MAX_STATE_CHARS = 384 * 4


def request_body(record):
    return {"state": record["state"], "questions": record["questions"], "model": "kev-latest"}


def predict(base_url, record, timeout=180, retries=3):
    body = json.dumps(request_body(record)).encode()
    request = urllib.request.Request(f"{base_url.rstrip('/')}/v1/systemone", data=body, method="POST",
                                     headers={"content-type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            return payload, (time.perf_counter() - start) * 1000
        except Exception as error:  # transient 5xx / timeouts retry with backoff
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"endpoint failed after {retries} attempts: {last}")


def align(record, answers, qid):
    """-> (keys, predicted distribution, gold distribution) in a shared key order, or None for an overlong record."""
    meta = record["_meta"]["gold"][qid]
    gold = meta["probabilities"]
    keys = list(gold.keys())
    answer = answers[qid]
    if meta["type"] == "noul":
        p = [float(answer["noul"]) if k == "true" else 1.0 - float(answer["noul"]) for k in keys]
    elif meta["type"] == "choice":
        p = [float(answer["probabilities"][k]) for k in keys]
    else:
        p = [float(answer["probabilities"][k]) for k in keys]
    return keys, np.asarray(p, float), np.asarray([gold[k] for k in keys], float)


def one_row(record, answers, latency_ms):
    row = {"id": record["_meta"]["id"], "workflow": record["_meta"]["workflow"], "overlong": record["_meta"]["overlong"],
           "latency_ms": latency_ms, "questions": {}}
    for qid in record["questions"]:
        gold_meta = record["_meta"]["gold"][qid]
        if record["_meta"]["overlong"]:
            row["questions"][qid] = {"type": gold_meta["type"], "overlong": True, "label": gold_meta["label"]}
            continue
        keys, p, g = align(record, answers, qid)
        entry = {"type": gold_meta["type"], "keys": keys, "p": (p / p.sum()).tolist(), "g": (g / g.sum()).tolist(),
                 "label": str(gold_meta["label"])}
        if gold_meta["type"] == "score":
            entry["gold_score"] = float(gold_meta.get("score") or (g * np.arange(len(g))).sum())
        row["questions"][qid] = entry
    return row


def metrics(rows, qtype=None):
    """Pooled metrics over question rows; accuracy/baked metrics use the gold argmax, distribution metrics use gold."""
    nll, acc, conf, brier, kl, tv, mae, within = [], [], [], [], [], [], [], []
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for row in rows:
        for qid, e in row["questions"].items():
            if e.get("overlong"):
                # overlong: counted wrong, no distribution
                acc.append(0); conf.append(0.0)
                continue
            if qtype and e["type"] != qtype:
                continue
            p, g = np.asarray(e["p"]), np.asarray(e["g"])
            keys = e["keys"]
            y = keys.index(e["label"])
            nll.append(-math.log(max(float(p[y]), EPS)))
            pred = int(p.argmax())
            acc.append(int(pred == y))
            conf.append(float(p.max()))
            target = np.eye(len(p))[y]
            brier.append(float(((p - target) ** 2).sum()))
            kl.append(float((g * (np.log(np.maximum(g, EPS)) - np.log(np.maximum(p, EPS)))).sum()))
            tv.append(float(0.5 * np.abs(p - g).sum()))
            per_class[y]["tp" if pred == y else "fn"] += 1
            if pred != y:
                per_class[pred]["fp"] += 1
            if e["type"] == "score":
                levels = np.arange(len(g))
                mae.append(abs(float(p @ levels) - float(g @ levels)))
                within.append(int(abs(int(p.argmax()) - int(g.argmax())) <= 1))
    f1 = [2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"]) for c in per_class.values() if (c["tp"] + c["fp"] + c["fn"])]
    result = {"n": len(acc), "acc": float(np.mean(acc)), "nll": float(np.mean(nll)) if nll else None,
              "macro_f1": float(np.mean(f1)) if f1 else None, "brier": float(np.mean(brier)) if brier else None,
              "kl": float(np.mean(kl)) if kl else None, "tv": float(np.mean(tv)) if tv else None,
              "ece": ece(conf, acc) if nll else None}
    if mae:
        result.update(score_mae=float(np.mean(mae)), within_one_level=float(np.mean(within)))
    return result


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="evals/external/typed-decisions-v1")
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    records = load_split(a.suite, "development")
    if a.limit:
        records = records[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    with (out / "predictions.jsonl").open("w") as stream:
        for i, record in enumerate(records, 1):
            payload, latency = predict(a.base_url, record)
            row = one_row(record, payload["answers"], latency)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if i % 25 == 0:
                print(f"scored {i}/{len(records)}", flush=True)
    by_workflow = defaultdict(list)
    for row in rows:
        by_workflow[row["workflow"]].append(row)
    qids = sorted({q for row in rows for q in row["questions"]})
    report = {
        "candidate": "Kev, specialist? no - generalist, zero-shot, no training",
        "endpoint": {"base_url": a.base_url, "model": "kev-latest"},
        "suite": a.suite, "records": len(rows),
        "overall": metrics(rows),
        "by_workflow": {wf: metrics(g) for wf, g in sorted(by_workflow.items())},
        "by_question": {q: metrics_q(rows, q) for q in qids},
        "overlong": {"count": sum(r["overlong"] for r in rows), "ids": [r["id"] for r in rows if r["overlong"]],
                     "policy": "counted as wrong, never truncated"},
        "latency_ms": {"p50": float(np.median([r["latency_ms"] for r in rows])),
                       "p95": float(np.quantile([r["latency_ms"] for r in rows], .95)),
                       "unit": "per case (five questions)"},
        "reference_points": {"uniform": 0.308, "prior": 0.470, "minilm_l6": 0.587, "modernbert_base": 0.646,
                             "factor_ceiling": 0.704, "jev_1_13_0_generalist": 0.727, "teacher_self_agreement": 0.735}}
    write_json(out / "report.json", report)
    print(json.dumps({"overall": report["overall"], "by_workflow": report["by_workflow"],
                      "overlong": report["overlong"]["count"], "latency_ms": report["latency_ms"]}, indent=2))


def metrics_q(rows, qid):
    selected = [{"questions": {qid: row["questions"][qid]}} for row in rows if qid in row["questions"]]
    return metrics(selected)


if __name__ == "__main__":
    main()
