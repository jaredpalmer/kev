"""Report-only reads of AutoJev-27B for the head-to-head in PLAN.md: kev.benchmark's own evaluate_records over the
remote predictor, with AutoJev's server behaviour made explicit instead of fatal. Its server answers one request at a
time and returns 529 while busy (waited out here), and refuses a question whose state + question exceeds 8,192 tokens
with a 422 (counted as a rejected record in coverage and listed in rejected.json, never silently dropped).

    KEV_REMOTE_API_KEY=<key> uv run python scripts/autojev_read.py --suite evals/external/typesafe-v1 --out runs/autojev-typesafe
"""
import argparse, os, sys, time, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.benchmark import evaluate_records  # noqa: E402
from kev.model import ContextOverflow  # noqa: E402
from kev.predictors import RemotePredictor  # noqa: E402
from kev.suite import digest, load_split, read_manifest  # noqa: E402

URL = "https://jp-1083--autojev-api.modal.run"


class AutoJev(RemotePredictor):
    def __call__(self, record):
        for _ in range(600):
            try:
                self.retries = 1
                return super().__call__(record)
            except RuntimeError as error:
                cause = str(error)
                if "529" in cause or any(c in cause for c in ("502", "503", "504")): time.sleep(1.0); continue   # busy, or a transient gateway error
                if "422" in cause: raise ContextOverflow(f"AutoJev refused the request (HTTP 422: its 8,192-token question-branch limit): {cause[:200]}") from error
                raise
        raise RuntimeError("AutoJev stayed busy for 10 minutes")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--suite", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    records, manifest = load_split(a.suite, "development"), read_manifest(a.suite)
    predictor = AutoJev(URL, "jev-latest", os.environ["KEV_REMOTE_API_KEY"])
    report, _ = evaluate_records(records, predictor, a.out, heldout_sources=tuple(manifest["holdout_sources"]), skip_overlong=True)
    print(a.suite, "acc", round(report["clean"]["acc"], 3), "coverage", report["coverage"], "manifest", digest(Path(a.suite) / "manifest.json")[:12])


if __name__ == "__main__":
    main()
