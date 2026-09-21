#!/usr/bin/env python3
"""Check a labelled Kev JSONL file and split it into train / calibration / development partitions.

    python3 scripts/split_data.py data/support.jsonl --out data/support
    python3 scripts/split_data.py data/support.jsonl            # check only, no files written

One record per line, in the System One request shape plus a `label` on every question:

    {"state": "...", "questions": {"team": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "other": null}, "label": "billing"},
                                   "urgent": {"type": "noul", "instructions": "...", "label": true},
                                   "priority": {"type": "score", "instructions": "...", "criteria": ["low", "medium", "high"], "label": 2}}}

Records with the same state always land in the same partition, so a calibration or development score is never inflated
by a state the model trained on. Exact duplicates are dropped; the same state with different labels for the same
question is reported as a conflict and the whole record is dropped. Standard library only; the tokenizer-level check
(Kev's 384-token state / 2048-token request limits) happens on Modal (kev_modal.py::validate) or at the start of training.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TYPES = ("noul", "choice", "score")
MAX_OPTIONS = 255
STATE_CHARS_WARN = 1400   # ~384 tokens of English; longer states are dropped by the trainer


def render_length(value):
    """Rough character length of a state or option after Kev renders it (kev.api.render flattens objects and lists)."""
    if value is None: return 0
    if isinstance(value, (str, int, float, bool)): return len(str(value))
    return len(json.dumps(value, ensure_ascii=False))


def normalized_state(state):
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return " ".join(text.casefold().split())


def state_key(state):
    return hashlib.sha256(normalized_state(state).encode()).hexdigest()


def label_key(q):
    """The label as the key Kev reports probabilities under (option name / "true"|"false" / level index as a string)."""
    if q["type"] == "noul": return str(bool(q["label"])).lower()
    return str(q["label"])


def check_question(qid, q):
    """Return a list of problems with one labelled question (empty when it is valid)."""
    problems = []
    if not isinstance(q, dict): return [f"question {qid!r} is not an object"]
    t = q.get("type")
    if t not in TYPES: return [f"question {qid!r}: type must be one of {TYPES}, got {t!r}"]
    if "instructions" not in q or q["instructions"] in (None, ""): problems.append(f"question {qid!r}: instructions are missing")
    if "label" not in q: return problems + [f"question {qid!r}: no label"]
    label, criteria = q["label"], q.get("criteria")
    if t == "noul":
        if not isinstance(label, bool): problems.append(f"question {qid!r}: noul label must be true or false, got {label!r}")
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
            problems.append(f"question {qid!r}: noul criteria may only describe \"true\" and \"false\"")
    elif t == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            problems.append(f"question {qid!r}: choice criteria must be an object with 1..{MAX_OPTIONS} options")
        elif label not in criteria:
            problems.append(f"question {qid!r}: label {label!r} is not one of the criteria {list(criteria)}")
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_OPTIONS:
            problems.append(f"question {qid!r}: score criteria must be a list of 2..{MAX_OPTIONS} ordered levels")
        elif isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(criteria):
            problems.append(f"question {qid!r}: score label must be a level index 0..{len(criteria) - 1}, got {label!r}")
    target = q.get("target")
    if target is not None:
        keys = ["false", "true"] if t == "noul" else (list(criteria) if t == "choice" else [str(i) for i in range(len(criteria or []))])
        if not isinstance(target, dict) or set(target) - set(keys) or not all(isinstance(v, (int, float)) and v >= 0 for v in target.values()) or sum(target.values()) <= 0:
            problems.append(f"question {qid!r}: target must map option keys to nonnegative weights with positive total")
    return problems


def check_record(record):
    problems = []
    if not isinstance(record, dict) or "state" not in record: return ["record needs a state"]
    if record["state"] in (None, ""): problems.append("state is empty")
    qs = record.get("questions")
    if not isinstance(qs, dict) or not qs: return problems + ["record needs a non-empty questions object"]
    for qid, q in qs.items():
        problems += check_question(qid, q)
    return problems


def read_records(path):
    """(valid records, problems) from a JSONL file; problems are strings prefixed with the line number."""
    records, problems = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"line {n}: not JSON ({error.msg})"); continue
        found = check_record(record)
        if found: problems += [f"line {n}: {p}" for p in found]
        else: records.append(record)
    return records, problems


def dedupe(records):
    """Drop exact duplicates and records whose state carries conflicting labels for the same question."""
    seen, by_state, conflicts = set(), defaultdict(dict), set()
    kept = []
    for r in records:
        fingerprint = json.dumps({"state": r["state"], "questions": r["questions"]}, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen: continue
        seen.add(fingerprint)
        key = state_key(r["state"])
        for qid, q in r["questions"].items():
            previous = by_state[key].setdefault((qid, json.dumps(q.get("instructions"), sort_keys=True)), label_key(q))
            if previous != label_key(q): conflicts.add(key)
        kept.append(r)
    dropped_dupes = len(records) - len(kept)
    kept = [r for r in kept if state_key(r["state"]) not in conflicts]
    return kept, dropped_dupes, len(conflicts)


def label_table(records):
    """{qid: Counter(label key)} over all records."""
    table = defaultdict(Counter)
    for r in records:
        for qid, q in r["questions"].items():
            table[qid][label_key(q)] += 1
    return table


def split(records, fractions, seed):
    """Deterministic split by state: every record with the same state goes to one partition."""
    groups = defaultdict(list)
    for r in records: groups[state_key(r["state"])].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_cal, n_dev = round(n * fractions["calibration"]), round(n * fractions["development"])
    parts = {"development": keys[:n_dev], "calibration": keys[n_dev:n_dev + n_cal], "train": keys[n_dev + n_cal:]}
    return {name: [r for k in ks for r in groups[k]] for name, ks in parts.items()}


def warnings_for(records):
    out = []
    long_states = sum(render_length(r["state"]) > STATE_CHARS_WARN for r in records)
    if long_states: out.append(f"{long_states} states are longer than {STATE_CHARS_WARN} characters; the trainer drops requests over 384 state tokens")
    for qid, counts in sorted(label_table(records).items()):
        total = sum(counts.values())
        rare = [k for k, c in counts.items() if c / total < 0.05]
        if rare: out.append(f"question {qid!r}: labels {rare} are under 5% of {total} records; add more or the model will rarely predict them")
        expected = None
        sample = next(q for r in records for q2, q in r["questions"].items() if q2 == qid)
        if sample["type"] == "choice": expected = set(sample["criteria"])
        elif sample["type"] == "noul": expected = {"true", "false"}
        else: expected = {str(i) for i in range(len(sample["criteria"]))}
        missing = expected - set(counts)
        if missing: out.append(f"question {qid!r}: no record is labelled {sorted(missing)}; the model cannot learn options it never sees as correct")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Exit status 1 when no valid record remains or a partition would be empty.")
    ap.add_argument("data", help="labelled JSONL (one record per line)")
    ap.add_argument("--out", help="directory for train.jsonl, calibration.jsonl, development.jsonl and summary.json (omit to check only)")
    ap.add_argument("--calibration", type=float, default=0.15, help="fraction of states for the calibration partition (temperature fit); default 0.15")
    ap.add_argument("--development", type=float, default=0.15, help="fraction of states for the development partition (scoring); default 0.15")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not 0 < a.calibration + a.development < 1: ap.error("calibration + development fractions must be between 0 and 1")

    records, problems = read_records(a.data)
    for p in problems[:30]: print(f"invalid: {p}", file=sys.stderr)
    if len(problems) > 30: print(f"... {len(problems) - 30} more invalid lines", file=sys.stderr)
    records, dupes, conflicts = dedupe(records)
    if not records:
        print("no valid records", file=sys.stderr); return 1
    print(f"{len(records)} valid records ({len(problems)} invalid lines, {dupes} exact duplicates dropped, {conflicts} states with conflicting labels dropped)")
    types = Counter(q["type"] for r in records for q in r["questions"].values())
    print(f"questions by type: {dict(types)}; states: {len({state_key(r['state']) for r in records})} distinct")
    for qid, counts in sorted(label_table(records).items()):
        print(f"  {qid}: " + ", ".join(f"{k}={c}" for k, c in counts.most_common()))
    for w in warnings_for(records): print(f"warning: {w}")

    if not a.out: return 0
    parts = split(records, {"calibration": a.calibration, "development": a.development}, a.seed)
    if any(not v for v in parts.values()):
        print("a partition would be empty; you need more records (aim for 300+)", file=sys.stderr); return 1
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(a.data), "seed": a.seed, "records": len(records), "invalid_lines": len(problems), "duplicates_dropped": dupes,
               "conflicting_states_dropped": conflicts, "partitions": {}}
    for name, rows in parts.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["partitions"][name] = {"records": len(rows), "labels": {qid: dict(c) for qid, c in label_table(rows).items()}}
        print(f"wrote {out / f'{name}.jsonl'}: {len(rows)} records")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
