"""Contamination screen for a generated suite against an external benchmark we must not tune toward (JevBench).

    uv run python scripts/screen_overlap.py --suite evals/hard-v1 --external /tmp/jevbench/datasets/public --out evals/hard-v1/overlap.json

For every record in every partition of --suite (train, development, test), compares
  - the normalised state with every external item's normalised "state" (exact match),
  - every question's normalised instructions with every external item's normalised "question" (exact match),
  - the word 8-grams of the record (state plus question instructions) with those of each external item (state plus
    question): the maximum Jaccard similarity over external items, per record.
Normalisation: casefold, words only (\\w+), single spaces. Writes counts and distributions only, never external text, and
exits non-zero if any record has an exact match or a maximum 8-gram Jaccard above --max-jaccard (default 0.2).
"""
import argparse, hashlib, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.api import render  # noqa: E402
from kev.suite import digest, read_jsonl, write_json  # noqa: E402

N = 8
PARTITIONS = ("train", "development", "test")


def words(value):
    return re.findall(r"\w+", (value if isinstance(value, str) else render(value)).casefold())


def norm(value):
    return " ".join(words(value))


def grams(tokens):
    return {" ".join(tokens[i:i + N]) for i in range(len(tokens) - N + 1)}


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def external_items(directory):
    """(states, questions, gram sets, file report) of the external benchmark; only hashes and gram sets are kept."""
    states, questions, sets, files = set(), set(), [], {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        rows = read_jsonl(path)
        files[path.name] = {"items": len(rows), "sha256": digest(path)}
        for row in rows:
            s, q = row.get("state"), row.get("question")
            if s is not None: states.add(key(norm(s)))
            if q is not None: questions.add(key(norm(q)))
            sets.append(grams(words(s or "") + words(q or "")))
    if not sets: raise SystemExit(f"no external items found under {directory}")
    return states, questions, sets, files


def quantiles(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"min": round(xs[0], 4), "median": round(q(0.5), 4), "p99": round(q(0.99), 4), "max": round(xs[-1], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--external", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-jaccard", type=float, default=0.2)
    a = ap.parse_args()
    ext_states, ext_questions, ext_sets, files = external_items(a.external)
    index = defaultdict(list)
    for i, s in enumerate(ext_sets):
        for g in s: index[g].append(i)
    per_split, offenders, all_j = {}, Counter(), []
    buckets = (0.0, 0.01, 0.05, 0.1, 0.2, 1.0)
    for split in PARTITIONS:
        path = Path(a.suite) / f"{split}.jsonl"
        if not path.exists(): continue
        stats = defaultdict(lambda: {"records": 0, "exact_state": 0, "exact_question": 0, "any_shared_8gram": 0, "over_threshold": 0, "max_jaccard": []})
        for r in read_jsonl(path):
            fam = r["_meta"].get("family", r["_meta"]["source"])
            s = stats[fam]; s["records"] += 1
            instr = [q.get("instructions") for q in r["questions"].values()]
            exact_s = key(norm(r["state"])) in ext_states
            exact_q = any(key(norm(x)) in ext_questions for x in instr if x is not None)
            mine = grams(words(r["state"]) + [w for x in instr if x is not None for w in words(x)])
            shared = Counter(i for g in mine for i in index.get(g, ()))
            best = max((c / (len(mine) + len(ext_sets[i]) - c) for i, c in shared.items()), default=0.0)
            s["exact_state"] += exact_s; s["exact_question"] += exact_q; s["any_shared_8gram"] += bool(shared)
            s["over_threshold"] += best > a.max_jaccard; s["max_jaccard"].append(best); all_j.append(best)
            if exact_s or exact_q or best > a.max_jaccard: offenders[r["_meta"]["id"]] += 1
        per_split[split] = {fam: {**{k: v for k, v in s.items() if k != "max_jaccard"}, "max_jaccard": quantiles(s["max_jaccard"]),
                                  "max_jaccard_histogram": {f"<={hi}": sum(lo < x <= hi or (lo == 0.0 and x == 0.0) for x in s["max_jaccard"]) for lo, hi in zip(buckets, buckets[1:])}}
                            for fam, s in sorted(stats.items())}
    report = {"suite": a.suite, "external": {"files": files, "items": len(ext_sets)}, "n": N, "normalisation": "casefold, \\w+ tokens, single spaces",
              "threshold": a.max_jaccard, "records": len(all_j), "offending_records": len(offenders),
              "max_jaccard_overall": quantiles(all_j) if all_j else None, "partitions": per_split,
              "note": "counts only; no external text is stored. A record offends on an exact normalised state or question match, or a maximum word 8-gram Jaccard above the threshold."}
    write_json(Path(a.out), report)
    print(f"{len(all_j)} records screened against {len(ext_sets)} external items; offending records: {len(offenders)}; "
          f"max 8-gram Jaccard {report['max_jaccard_overall']['max'] if all_j else None}")
    if offenders: raise SystemExit(1)


if __name__ == "__main__":
    main()
