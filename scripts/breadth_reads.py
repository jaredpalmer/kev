"""The two remote baseline reads of evals/breadth-v1 (development only), as kev.benchmark result directories.

    AI_GATEWAY_API_KEY=... uv run python scripts/breadth_reads.py jev --out runs/breadth-v1-jev
    KEV_REMOTE_API_KEY=... uv run python scripts/breadth_reads.py autojev --url https://<workspace>--autojev-breadth-api.modal.run --out runs/breadth-v1-autojev

Both use kev.benchmark.evaluate_records with skip_overlong, so a record a server will not answer is counted in
coverage["rejected_records"], listed in rejected.json and scored wrong by scripts/breadth_report.py (the Decision Index
rule: unanswered = wrong), instead of aborting the read.

jev: kev.predictors.JevPredictor (the kev.jev path) with a longer retry: Vercel AI Gateway answered some long ContractNLI
  documents with HTTP 503 several times in a row, and kev.jev's four attempts over 7 s stopped two full reads (at 829 and
  575 records). Here a 5xx is retried ATTEMPTS times with exponential backoff capped at 60 s; a 4xx still surfaces.
autojev: denis-pplx/autojev-27b behind its own server (research-archive-2026-09-24:scripts/serve_autojev.py, app renamed
  autojev-breadth); it answers one request at a time (529 while busy: waited out) and refuses requests over its 8,192-token
  limit with 422 (rejected record). Adapted from the archive's scripts/autojev_read.py.
"""
import argparse, json, os, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.benchmark import evaluate_records  # noqa: E402
from kev.data import api_request  # noqa: E402
from kev.model import ContextOverflow  # noqa: E402
from kev.predictors import PRICE_PER_MILLION, JevPredictor, RemotePredictor  # noqa: E402
from kev.suite import digest, load_split, read_manifest, write_json  # noqa: E402

SUITE = "evals/breadth-v1"
ATTEMPTS = 8


class PatientJev(JevPredictor):
    """JevPredictor.__call__ with ATTEMPTS tries and backoff min(60, 2**attempt) s on 5xx or transport errors; a record
    that still fails becomes a rejected record (ContextOverflow is how evaluate_records' skip_overlong counts it)."""

    def __call__(self, record):
        if self.calls >= self.max_calls or (self.input_tokens + 65536) * PRICE_PER_MILLION / 1e6 > self.budget:
            raise RuntimeError("Jev evaluation reached the request/token cost cap")
        request = json.dumps(api_request(record)) + "\n"
        for attempt in range(ATTEMPTS):
            self.process.stdin.write(request); self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line: raise RuntimeError("Jev SDK worker exited without a response")
            result = json.loads(line); self.calls += 1
            if "error" not in result: break
            status = result["error"]["status"]
            if status is not None and status < 500: raise RuntimeError(f"Jev request failed: {result['error']['name']} (HTTP {status})")
            if attempt == ATTEMPTS - 1: raise ContextOverflow(f"Jev failed {ATTEMPTS} times: {result['error']['name']} (HTTP {status})")
            self.retries += 1
            time.sleep(min(60, 2 ** attempt))
        usage = result["usage"]
        if usage.get("inputTokens") is None: raise RuntimeError("Jev returned no input token usage; cannot account for cost")
        self.input_tokens += usage["inputTokens"]; self.output_tokens += usage.get("outputTokens") or 0
        probabilities = {}
        for qid, q in record["questions"].items():
            answer = result["answers"][qid]
            probabilities[qid] = {"false": 1 - answer["probability"], "true": answer["probability"]} if q["type"] == "noul" else answer["probabilities"]
        return {**result, "probabilities": probabilities}


class AutoJev(RemotePredictor):
    def __call__(self, record):
        for _ in range(1200):
            try:
                self.retries = 1
                return super().__call__(record)
            except RuntimeError as error:
                cause = str(error)
                if "529" in cause or any(c in cause for c in ("502", "503", "504", "timed out")): time.sleep(1.0); continue   # busy or a transient gateway error
                if "422" in cause: raise ContextOverflow(f"AutoJev refused the request (HTTP 422: its 8,192-token limit): {cause[:200]}") from error
                raise
        raise RuntimeError("AutoJev stayed busy for 20 minutes")


def wait_ready(url, key):
    for i in range(240):   # the first request starts the container and loads 55 GB of weights
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{url}/v1/models", headers={"authorization": f"Bearer {key}"}), timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f"waiting for {url} ({i}): {type(e).__name__}", flush=True); time.sleep(15)
    raise SystemExit("server never became ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["jev", "autojev"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", help="AutoJev server base URL")
    ap.add_argument("--budget", type=float, default=3.0, help="Jev spend cap in USD (list price x input tokens)")
    a = ap.parse_args()
    if Path(a.out).exists(): ap.error("output directory already exists")
    try:
        records, manifest = load_split(SUITE, "development"), read_manifest(SUITE)
    except PermissionError as error:
        raise SystemExit(f"cannot read without the suite's partitions: {error}") from None
    extra = {}
    if a.system == "jev":
        predictor = PatientJev(os.environ["AI_GATEWAY_API_KEY"], a.budget, 5000)
    else:
        if not a.url: ap.error("autojev needs --url")
        key = os.environ["KEV_REMOTE_API_KEY"]
        extra["models"] = wait_ready(a.url.rstrip("/"), key)
        predictor = AutoJev(a.url, "jev-latest", key, timeout=300)
    try:
        report, _ = evaluate_records(records, predictor, a.out, heldout_sources=tuple(manifest["holdout_sources"]), skip_overlong=True)
        report.update(suite_sha256=digest(Path(SUITE) / "manifest.json"), split="development", run=a.system)
        if a.system == "jev":
            report["provider"] = predictor.accounting()
        else:
            report["remote"] = {"base_url": a.url, "requested_model": "jev-latest", "served_model": predictor.served_model, "concurrency": 1,
                                "weights": "denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015", "server": "github.com/denis-pplx/autojev@ee63c1515980491a742f0bd0685c8dc5ca1f00c3", **extra}
        write_json(Path(a.out) / "report.json", report)
        print(json.dumps({"acc": report["clean"]["acc"], "coverage": report["coverage"]}))
    finally:
        if a.system == "jev":
            predictor.close()
            if Path(a.out).exists(): write_json(Path(a.out) / "usage.json", predictor.accounting())


if __name__ == "__main__":
    main()
