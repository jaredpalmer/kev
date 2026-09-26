import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from kev.benchmark import evaluate_records
from kev.predictors import JevPredictor
from kev.suite import digest, load_split, read_manifest, write_json


def provision_key(scope):
    result = subprocess.run(["vercel", "ai-gateway", "api-keys", "create", "--scope", scope,
                             "--name", "kev-jev-evaluation", "--limit", "1", "--refresh-period", "none",
                             "--expiration", "7d", "--no-color", "--non-interactive"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Vercel key creation failed; inspect account permissions in the Vercel dashboard")
    text = result.stdout + result.stderr
    match = re.search(r"\bvck_[A-Za-z0-9_-]+", text)
    if not match:
        raise RuntimeError("Key was created but its output format was not recognized; no key output was logged")
    return match.group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provision-scope", help="Explicitly authorize a $1 non-renewing, seven-day key on this team")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--max-calls", type=int, default=700)
    ap.add_argument("--count-refusals", action="store_true",
                    help="count requests Jev refuses (HTTP 400/413/422, e.g. past its context) as rejected records (rejected.json, "
                         "report coverage) and continue; other errors still stop the read")
    ap.add_argument("--attempts", type=int, default=4, help="tries per request on gateway 5xx errors (backoff 1, 2, 4, ... s, at most 30 s)")
    a = ap.parse_args()
    # a provisioned key carries a $1 limit of its own; a caller's key may run a larger panel (breadth-v1 dev: ~2,000 records)
    if not 0 < a.budget <= 5 or not 1 <= a.max_calls <= 5000 or not 1 <= a.attempts <= 12:
        ap.error("budget must be in (0, 5], max-calls in [1, 5000] and attempts in [1, 12]")
    if Path(a.out).exists():
        ap.error("output directory already exists")
    records = load_split(a.suite, "development")
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key and a.provision_scope:
        key = provision_key(a.provision_scope)
    if not key:
        ap.error("Set AI_GATEWAY_API_KEY or explicitly select --provision-scope")
    predictor = JevPredictor(key, a.budget, a.max_calls, count_refusals=a.count_refusals, attempts=a.attempts)
    try:
        heldout = read_manifest(a.suite)["holdout_sources"]
        report, _ = evaluate_records(records, predictor, a.out, heldout_sources=tuple(heldout), skip_overlong=a.count_refusals)
        report.update(suite_sha256=digest(Path(a.suite) / "manifest.json"), split="development", provider=predictor.accounting())
        write_json(Path(a.out) / "report.json", report)
        print(json.dumps({"clean": report["clean"], "variants": report["variants"], "provider": report["provider"]}, indent=2))
    finally:
        predictor.close()
        if Path(a.out).exists():
            write_json(Path(a.out) / "usage.json", predictor.accounting())


if __name__ == "__main__":
    main()
