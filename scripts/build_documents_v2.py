"""documents-v2 candidates: a held-out test set only (PLAN.md round 7, confirmation 5), drawn exactly as documents-v1
(scripts/build_documents_v1.py: same source snapshot, products, issues, length buckets, per-cell test count and record
shape) from the narratives v1 never drew: every v1 candidate (all three splits, dropped questions included) is excluded
by text hash and complaint id. Its partitions go only to the private mirror (kev.suite.PRIVATE_DATASET); never commit them.

    uv run python scripts/build_documents_v2.py --out runs/documents-v2-work/candidates
"""
import argparse, hashlib, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_documents_v1 import BUCKETS, REPO, REVISION, SPLITS, prepare, record, rows  # noqa: E402
from kev.suite import read_jsonl, write_json, write_jsonl  # noqa: E402

PER_CELL = SPLITS["test"]   # documents per (product, bucket) cell, as documents-v1 test
RIGHTS = "consumer narratives published by the CFPB with consent; the CFPB considers them public domain for FOIA purposes"


def select(source_rows, v1, seed):
    """Up to PER_CELL records per (product, bucket) cell from the rows whose text hash and complaint id no v1 candidate
    has (and no earlier row had), in a seeded order. Returns (records, stats)."""
    seen, v1_ids = {r["_meta"]["text_sha256"] for r in v1}, {r["_meta"]["id"] for r in v1}
    cells, stats = defaultdict(list), Counter()
    for r in map(prepare, source_rows):
        if r is None or r["bucket"] is None: continue
        if r["text_sha256"] in seen or f"cfpb/{r['complaint_id']}" in v1_ids: stats["excluded_or_duplicate"] += 1; continue
        seen.add(r["text_sha256"]); cells[r["key"], r["bucket"]].append(r)
    rng, recs = random.Random(seed), []
    for (key, bucket), pool in sorted(cells.items()):
        rng.shuffle(pool)
        recs += [record(r, RIGHTS) for r in pool[:PER_CELL]]
    rng.shuffle(recs)
    return recs, stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", default="documents-v2")
    ap.add_argument("--v1", default="runs/documents-v1-work/candidates", help="documents-v1 candidates directory (every split is excluded)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    v1 = [r for split in ("train", "development", "test") for r in read_jsonl(Path(a.v1) / f"{split}.jsonl")]
    recs, stats = select(rows(), v1, a.seed)
    out.mkdir(parents=True); write_jsonl(out / "test.jsonl", recs)
    write_json(out / "build.json", {"repo": REPO, "revision": REVISION, "seed": a.seed, "per_cell": PER_CELL, "excluded_v1_candidates": len(v1), "stats": dict(stats),
                                    "per_split": {"test": len(recs)}, "questions": {"test": sum(len(r["questions"]) for r in recs)}, "buckets": BUCKETS,
                                    "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(len(recs), "records,", sum(len(r["questions"]) for r in recs), "questions", dict(stats))


if __name__ == "__main__":
    main()
