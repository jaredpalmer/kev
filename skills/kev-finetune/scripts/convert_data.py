#!/usr/bin/env python3
"""Turn a CSV / TSV / JSONL export of already-labelled examples into Kev training records.

    python3 scripts/convert_data.py workload.json tickets.csv --state body --label department=team --label escalate=urgent --out data/support.jsonl
    python3 scripts/convert_data.py workload.json tickets.jsonl --state subject,body --label priority=prio --score-offset 1 --map department="Billing Ops:billing,Ship:shipping"

The workload spec supplies the questions (type, instructions, criteria); this script supplies the state and labels from
your columns. Labels are matched to the question's options case-insensitively (`--map` handles renames); yes/no/1/0/
true/false become noul labels; score labels may be level indices, 1-based ratings (`--score-offset 1`) or level texts.
Rows whose label cannot be mapped are counted by reason and skipped. Standard library only. Next: split_data.py.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import check_question  # noqa: E402

TRUE, FALSE = {"true", "t", "yes", "y", "1", "on"}, {"false", "f", "no", "n", "0", "off"}


def rows_from(path):
    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip(): yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else data.get("rows") or data.get("data") or [])
    else:
        with path.open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")


def get(row, column):
    """row[column], with dotted paths for nested JSON (`meta.team`)."""
    value = row
    for part in column.split("."):
        if not isinstance(value, dict) or part not in value: return None
        value = value[part]
    return value


def map_label(q, value, mapping, score_offset):
    """-> (label, None) or (None, reason); the reason names the kind of problem, not the value, so it can be counted."""
    if value is None or (isinstance(value, str) and not value.strip()): return None, "empty label"
    text = str(value).strip()
    if text in mapping: text = mapping[text]
    if q["type"] == "noul":
        if isinstance(value, bool): return value, None
        if text.lower() in TRUE: return True, None
        if text.lower() in FALSE: return False, None
        return None, "not a yes/no value"
    if q["type"] == "choice":
        if text in q["criteria"]: return text, None
        hits = [k for k in q["criteria"] if k.casefold() == text.casefold()] or [k for k, v in q["criteria"].items() if isinstance(v, str) and v.casefold() == text.casefold()]
        return (hits[0], None) if len(hits) == 1 else (None, "not an option (use --map)")
    levels = q["criteria"]
    try:
        index = int(float(text)) - score_offset
        return (index, None) if 0 <= index < len(levels) else (None, f"level outside 0..{len(levels) - 1} after --score-offset {score_offset}")
    except ValueError:
        hits = [i for i, lvl in enumerate(levels) if str(lvl).casefold() == text.casefold() or str(lvl).casefold().startswith(text.casefold() + ":")]
        return (hits[0], None) if len(hits) == 1 else (None, "not a level (index or level text)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="workload spec with the questions (assets/workload.example.json shape)")
    ap.add_argument("data", help="CSV, TSV, JSON array or JSONL with one labelled example per row")
    ap.add_argument("--state", required=True, help="column(s) holding the input text; several columns (comma-separated) become an object state")
    ap.add_argument("--label", action="append", default=[], metavar="QID=COLUMN", help="which column labels which question (repeatable; default: a column named like the question id)")
    ap.add_argument("--map", action="append", default=[], metavar="QID=FROM:TO,FROM:TO", help="rename label values before matching (repeatable)")
    ap.add_argument("--score-offset", type=int, default=0, help="subtract this from numeric score labels (1 for 1..N ratings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many rows (0 = all)")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    questions = spec["questions"]
    columns = {qid: qid for qid in questions}
    for item in a.label:
        qid, _, col = item.partition("=")
        if qid not in questions: ap.error(f"--label {qid}: not a question in the spec ({list(questions)})")
        columns[qid] = col
    mappings = {qid: {} for qid in questions}
    for item in a.map:
        qid, _, pairs = item.partition("=")
        if qid not in questions: ap.error(f"--map {qid}: not a question in the spec")
        for pair in pairs.split(","):
            src, _, dst = pair.partition(":"); mappings[qid][src.strip()] = dst.strip()
    state_cols = [c.strip() for c in a.state.split(",") if c.strip()]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept, skipped, examples, labels = 0, Counter(), {}, {qid: Counter() for qid in questions}
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for n, row in enumerate(rows_from(a.data), 1):
            if a.limit and n > a.limit: break
            values = {c: get(row, c) for c in state_cols}
            if all(v in (None, "") for v in values.values()): skipped["empty state"] += 1; continue
            state = str(values[state_cols[0]]) if len(state_cols) == 1 else {c: v for c, v in values.items() if v not in (None, "")}
            record = {"state": state, "questions": {}}
            reason = None
            for qid, q in questions.items():
                raw = get(row, columns[qid])
                label, reason = map_label(q, raw, mappings[qid], a.score_offset)
                if reason: reason = f"{qid}: {reason}"; examples.setdefault(reason, str(raw)[:40]); break
                record["questions"][qid] = {**{k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}, "label": label}
                problems = check_question(qid, record["questions"][qid])
                if problems: reason = problems[0]; break
            if reason: skipped[reason] += 1; continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n"); kept += 1
            for qid, q in record["questions"].items(): labels[qid][str(q["label"]).lower() if q["type"] == "noul" else str(q["label"])] += 1
    print(f"{kept} records written to {out}; {sum(skipped.values())} rows skipped")
    for reason, count in skipped.most_common(8): print(f"  skipped {count}: {reason}" + (f" (e.g. {examples[reason]!r})" if reason in examples else ""))
    for qid, c in labels.items(): print(f"  {qid}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    if kept == 0: print("nothing converted: check --state / --label column names against the file's header", file=sys.stderr); return 1
    print(f"next: python3 scripts/split_data.py {out} --out {out.with_suffix('')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
