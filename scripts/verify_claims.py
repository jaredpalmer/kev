#!/usr/bin/env python3
"""Verify that published numbers trace to committed evidence.

docs/claims.json lists claim records: {"printed": exact string in the docs, "in": text files where it must literally
appear, "source": JSON/JSONL under the repo, "select": key/value filter for JSONL rows (must match exactly one row),
"path": dotted key path (or list of paths with "derive": "macro_mean"; "derive": "over_requested" rescales an accuracy over
evaluated questions to all requested questions, counting rejected ones wrong), optional "scale" multiplier}.

A claim passes when the printed string literally occurs in every `in` file and the source value, scaled and formatted
to the printed precision, equals the printed digits. Run: uv run python scripts/verify_claims.py [--claims PATH].
"""
import argparse
from pathlib import Path

from kev.suite import read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _dig(obj, dotted):
    parts = dotted.split(".")  # keys can contain dots (models.Kev-0.8B.temperature): match the longest prefix
    while parts:
        for n in range(len(parts), 0, -1):
            key = ".".join(parts[:n])
            if key in obj:
                obj = obj[key]
                parts = parts[n:]
                break
        else:
            raise KeyError(".".join(parts))
    return obj


def _value(claim, source):
    derive = claim.get("derive")
    if derive == "macro_mean":
        values = [_dig(source, p) for p in claim["path"]]
        return sum(values) / len(values)
    value = _dig(source, claim["path"])
    if derive == "over_requested":
        return value * source["coverage"]["evaluated_questions"] / source["coverage"]["requested_questions"]
    assert derive is None, claim
    return value


def verify(root, claims=None):
    failures = []
    texts, sources = {}, {}
    for claim in read_json(claims or root / "docs/claims.json"):
        label = f"{claim['printed']} <- {claim['source']}:{claim['path']}"
        for name in claim["in"]:
            if name not in texts:
                texts[name] = (root / name).read_text(encoding="utf-8")
            if claim["printed"] not in texts[name]:
                failures.append(f"{label}: '{claim['printed']}' not in {name}")
        if claim["source"] not in sources:
            sources[claim["source"]] = read_jsonl(root / claim["source"]) if claim["source"].endswith(".jsonl") \
                else read_json(root / claim["source"])
        source = sources[claim["source"]]
        if "select" in claim:
            rows = [r for r in source if all(r.get(k) == v for k, v in claim["select"].items())]
            if len(rows) != 1:
                failures.append(f"{label}: select matched {len(rows)} rows, want 1")
                continue
            source = rows[0]
        value = _value(claim, source) * claim.get("scale", 1)
        wanted = claim["printed"].replace(",", "").rstrip("%")
        decimals = len(wanted.split(".")[1]) if "." in wanted else 0
        if f"{value:.{decimals}f}" != wanted:
            failures.append(f"{label}: source value {value} formats to {value:.{decimals}f}, printed {wanted}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, default=None, help="claims file (default docs/claims.json)")
    a = ap.parse_args()
    failures = verify(ROOT, a.claims)
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)
    print(f"verified {len(read_json(a.claims or ROOT / 'docs/claims.json'))} claims")


if __name__ == "__main__":
    main()
