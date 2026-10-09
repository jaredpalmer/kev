"""Measure a running System One server using the canonical serving benchmark's requests.

Reports client wall time (including queueing) separately from server model time.
New-state rounds have unique ticket IDs, so warmup cannot turn them into cache hits.
"""
import argparse
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from kev.suite import write_json
from scripts.serving_bench import CASES, request


def summary(rows, elapsed=None):
    wall = sorted(r["wall_ms"] for r in rows)
    result = {
        "n": len(rows), "wall_p50_ms": round(statistics.median(wall), 2),
        "wall_p95_ms": round(wall[math.ceil(len(wall) * 0.95) - 1], 2),
        "model_p50_ms": round(statistics.median(r["model_ms"] for r in rows), 2),
    }
    if elapsed is not None:
        result["requests_per_s"] = round(len(rows) / elapsed, 2)
    return result


def run(base_url, reps, clients, case_names, api_key):
    report = {"base_url": base_url, "reps": reps, "latency": {}, "throughput": {}}
    serial = 0
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=120, trust_env=False,
                      headers={"Authorization": f"Bearer {api_key}"},
                      limits=httpx.Limits(max_connections=max(clients) + 1)) as client:
        report["before"] = client.get("/v1/models").raise_for_status().json()

        def new_id():
            nonlocal serial
            serial += 1
            return time.time_ns() + serial

        def one(case, ticket_id):
            payload = request(case, ticket_id).model_dump(mode="json", exclude_none=True)
            started = time.perf_counter()
            response = client.post("/v1/systemone", json=payload).raise_for_status()
            wall_ms = (time.perf_counter() - started) * 1000
            body = response.json()
            return {"wall_ms": wall_ms, "model_ms": body["latency_ms"], "input_tokens": body["usage"]["input_tokens"]}

        for case in case_names:
            first = one(case, new_id())
            # Give the server's idle graph capture a chance to finish.
            for _ in range(3):
                one(case, new_id())
                time.sleep(0.2)
            new = [one(case, new_id()) for _ in range(reps)]
            cached_id = new_id()
            one(case, cached_id)
            cached = [one(case, cached_id) for _ in range(reps)]
            report["latency"][case] = {"first": first, "new_state": summary(new), "cached_state": summary(cached)}
            for concurrency in clients:
                with ThreadPoolExecutor(concurrency) as pool:
                    list(pool.map(lambda i: one(case, i), [new_id() for _ in range(concurrency * 2)]))
                    time.sleep(0.5)
                    ids = [new_id() for _ in range(max(reps, concurrency * 4))]
                    started = time.perf_counter()
                    rows = list(pool.map(lambda i: one(case, i), ids))
                    report["throughput"][f"{case} @ {concurrency} clients"] = summary(rows, time.perf_counter() - started)
            print(case, report["latency"][case], flush=True)
        report["after"] = client.get("/v1/models").raise_for_status().json()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--clients", default="1,4,8,16")
    ap.add_argument("--case", choices=list(CASES), action="append")
    ap.add_argument("--api-key", default="local")
    args = ap.parse_args()
    try:
        clients = [int(c) for c in args.clients.split(",")]
    except ValueError:
        ap.error("--clients must be comma-separated positive integers")
    if args.reps < 1 or not clients or min(clients) < 1:
        ap.error("--reps and --clients must be positive")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, run(args.url, args.reps, clients, args.case or list(CASES), args.api_key))


if __name__ == "__main__":
    main()
