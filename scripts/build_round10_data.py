"""Round 10 (PLAN.md) training files: hard-v1 train, devtools-v1 train (trainable sources only), and both, each with a
manifest (sha256, counts per source). Records are copied unchanged from the frozen suites, verified through load_split.

    uv run python scripts/build_round10_data.py --out evals/round10
"""
import argparse, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import digest, load_split, read_manifest, write_json, write_jsonl  # noqa: E402

SUITES = {"hard": "evals/hard-v1", "devtools": "evals/devtools-v1"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    parts = {}
    for name, suite in SUITES.items():
        allowed = set(read_manifest(suite).get("trainable_sources") or [])
        recs = [r for r in load_split(suite, "train") if not allowed or r["_meta"]["source"] in allowed]
        parts[name] = (recs, {"suite": suite, "suite_manifest_sha256": digest(Path(suite) / "manifest.json")})
    parts["skills"] = (parts["hard"][0] + parts["devtools"][0], {"concat_of": ["hard", "devtools"]})
    for name, (recs, meta) in parts.items():
        out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "train.jsonl", recs)
        write_json(out / "manifest.json", {**meta, "records": len(recs), "questions": sum(len(r["questions"]) for r in recs),
                                           "sources": dict(Counter(r["_meta"]["source"] for r in recs)), "sha256": digest(out / "train.jsonl")})
        print(name, len(recs), "records", digest(out / "train.jsonl")[:12])


if __name__ == "__main__":
    main()
