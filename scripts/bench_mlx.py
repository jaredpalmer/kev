"""Benchmark the running MLX Kev service.

This deliberately measures only the active HTTP backend. It does not load a
second model or compare against the historical PyTorch implementation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


BASE_STATE = (
    "Customer order 4411 was placed on Monday morning through the mobile app. "
    "The package was expected to arrive on Thursday afternoon, but the carrier delivered it on Saturday evening. "
    "The tracking page showed several unexplained scans between the regional depot and the local delivery office. "
    "When the customer opened the box, the outer cardboard was crushed on one corner and the protective seal was broken. "
    "The main product still powers on, although one accessory is missing and another accessory has visible scratches. "
    "The customer contacted support twice and received two different delivery estimates from two different agents. "
    "A payment statement now contains two charges with the same order number, one pending and one completed. "
    "The customer says that only one order was placed and supplied the receipt, the tracking number, and photos of the damaged box. "
    "The first support agent suggested waiting for the pending charge to disappear. "
    "The second agent suggested opening a shipping investigation and sending the damaged item to returns. "
    "The customer needs the missing accessory before a scheduled event next week and asks whether a replacement can be sent immediately. "
    "The account history shows no earlier complaint for this customer, and the delivery address has not changed. "
    "The order is still inside the seller's replacement period, but the carrier claim window closes soon."
)
QUESTIONS = [
    {"instr": "Is there a billing issue?", "options": ["yes", "no"], "label": 0},
    {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 2},
    {"instr": "Was the delivery late?", "options": ["yes", "no"], "label": 0},
    {"instr": "Was the parcel damaged?", "options": ["yes", "no"], "label": 0},
    {"instr": "Which issue is most urgent?", "options": ["billing", "delivery", "damage", "none"], "label": 1},
    {"instr": "Should the card charge be reviewed?", "options": ["yes", "no"], "label": 0},
    {"instr": "Is this a return request?", "options": ["yes", "no"], "label": 1},
    {"instr": "Does the customer need follow-up?", "options": ["yes", "no"], "label": 0},
]


def post(url: str, payload: dict) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    started = time.perf_counter()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.load(response)
    return result, (time.perf_counter() - started) * 1000


def request_for(n_questions: int, state: str = BASE_STATE) -> dict:
    return {"state": state, "questions": QUESTIONS[:n_questions]}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * p / 100
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(rows: list[dict]) -> dict:
    wall = [r["wall_ms"] for r in rows]
    model = [r["model_ms"] for r in rows]
    return {
        "n": len(rows),
        "wall_ms": {"p50": round(percentile(wall, 50), 2), "p95": round(percentile(wall, 95), 2), "max": round(max(wall), 2), "mean": round(statistics.mean(wall), 2)},
        "model_ms": {"p50": round(percentile(model, 50), 2), "p95": round(percentile(model, 95), 2), "max": round(max(model), 2), "mean": round(statistics.mean(model), 2)},
        "cache_hits": sum(int(r["cache_hit"]) for r in rows),
        "cache_misses": sum(int(not r["cache_hit"]) for r in rows),
        "state_tokens": sorted(set(r["state_tokens"] for r in rows)),
        "input_tokens": sorted(set(r["input_tokens"] for r in rows)),
    }


def run_serial(url: str, payloads: list[dict], warmup: int) -> dict:
    for payload in payloads[:warmup]:
        post(url, payload)
    rows = []
    for payload in payloads[warmup:]:
        result, wall = post(url, payload)
        rows.append({"wall_ms": wall, "model_ms": result["latency_ms"], "cache_hit": result["prefix_cache_hit"], "state_tokens": result["state_tokens"], "input_tokens": result["tokens"]})
    return summarize(rows)


def run_parallel(url: str, payload: dict, n: int, workers: int) -> dict:
    def one(_):
        result, wall = post(url, payload)
        return {"wall_ms": wall, "model_ms": result["latency_ms"], "cache_hit": result["prefix_cache_hit"], "state_tokens": result["state_tokens"], "input_tokens": result["tokens"]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, range(n)))
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--concurrency", default="2,4")
    parser.add_argument("--out", default="runs/mlx-benchmark.json")
    args = parser.parse_args()
    if args.requests <= args.warmup:
        parser.error("--requests must be greater than --warmup")
    url = args.base_url.rstrip("/") + "/api/predict"
    info_url = args.base_url.rstrip("/") + "/api/info"
    with urllib.request.urlopen(info_url, timeout=10) as response:
        info = json.load(response)

    results = {}
    for n_questions in (1, 3, 5, 8):
        unique = [request_for(n_questions, BASE_STATE + f" Ticket {i}.") for i in range(args.requests)]
        repeated = [request_for(n_questions) for _ in range(args.requests)]
        results[f"unique_state_q{n_questions}"] = run_serial(url, unique, args.warmup)
        results[f"repeated_state_q{n_questions}"] = run_serial(url, repeated, args.warmup)

    for workers in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        results[f"concurrent_{workers}_repeated_q5"] = run_parallel(url, request_for(5), args.requests, workers)

    report = {
        "backend": info,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requests_per_case": args.requests,
        "warmup": args.warmup,
        "results": results,
    }
    out = __import__("pathlib").Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
