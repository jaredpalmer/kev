#!/usr/bin/env python3
"""Find where a codebase already asks Jev / TypeSafe System One questions, and draft a workload spec from them.

    python3 scripts/extract_workload.py path/to/repo                    # report: call sites, question literals, data files
    python3 scripts/extract_workload.py path/to/repo --out workload.json   # also write a draft spec to fill in

Looks for the TypeSafe Python SDK (`client.system_one(...)`, `Noul(...)`, `Choice(...)`, `Score(...)`), the AI SDK
(`experimental_evaluate({ model: "typesafe-ai/jev", questions: {...} })`), raw `/v1/systemone` request bodies in any
language, and labelled data files (CSV / JSON / JSONL) that could become training records. Question literals are parsed
best-effort: every extracted question carries `_source` (file:line) so you can check it, and anything that could not be
parsed is left as a `_todo` string. Standard library only.
"""
import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "target", ".cache", "site-packages", ".turbo", "coverage"}
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".rb", ".go", ".java", ".kt", ".cs", ".php", ".rs", ".swift", ".yaml", ".yml", ".toml", ".md", ".json"}
DATA_EXT = {".csv", ".jsonl", ".json", ".tsv", ".ndjson"}
MARKERS = [  # (regex, weight, what it means)
    (re.compile(r"system_one\(|/v1/systemone|systemone", re.I), 5, "System One call"),
    (re.compile(r"TypeSafeClient|typesafe[_-]sdk|from typesafe|@typesafe", re.I), 4, "TypeSafe SDK"),
    (re.compile(r"typesafe-ai/jev|experimental_evaluate|\bjev-latest\b"), 5, "AI SDK / Jev"),
    (re.compile(r"\b(Noul|Choice|Score)\(\s*instructions"), 4, "SDK question"),
    (re.compile(r"[\"']?type[\"']?\s*[:=]\s*[\"'](noul|boolean|choice|score)[\"']"), 3, "typed question literal"),
    (re.compile(r"\bjev\b", re.I), 1, "mentions Jev"),
]
TYPE_RE = re.compile(r"[\"']?type[\"']?\s*[:=]\s*[\"'](noul|boolean|choice|score)[\"']")
INSTR_RE = re.compile(r"instructions[\"']?\s*[:=]\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`[^`]*`)")
SDK_RE = re.compile(r"[\"']?([\w-]+)[\"']?\s*:\s*(Noul|Choice|Score)\s*\(")
CRIT_RE = re.compile(r"criteria[\"']?\s*[:=]\s*")


def files_under(root):
    for path in sorted(Path(root).rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts) or not path.is_file(): continue
        if path.suffix.lower() in CODE_EXT | DATA_EXT and path.stat().st_size < 5_000_000:
            yield path


def read(path):
    try: return path.read_text(encoding="utf-8", errors="replace")
    except OSError: return ""


def balanced(text, start):
    """The bracketed literal ({...} or [...]) starting at text[start], or None."""
    if start >= len(text) or text[start] not in "{[": return None
    depth, i, quote = 0, start, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\": i += 1
            elif c == quote: quote = None
        elif c in "\"'`": quote = c
        elif c in "{[": depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0: return text[start:i + 1]
        i += 1
    return None


def literal(text):
    """Parse a Python / JSON / loose-JS object or array literal; None when it cannot be parsed."""
    if text is None: return None
    for attempt in (text, re.sub(r"(?m)^\s*//.*$", "", text)):
        for parser in (json.loads, ast.literal_eval):
            try: return parser(attempt)
            except Exception: pass
        # JS-style: unquoted keys and trailing commas; single-quoted strings only as a last resort (apostrophes inside double quotes)
        loose = re.sub(r"([{,]\s*)([A-Za-z_$][\w$]*)\s*:", r'\1"\2":', attempt)
        loose = re.sub(r",\s*([}\]])", r"\1", loose)
        for candidate in (loose, loose.replace("'", '"')):
            try: return json.loads(candidate)
            except Exception: pass
    return None


def unquote(s):
    try: return ast.literal_eval(s) if s[0] in "\"'" else s[1:-1]
    except Exception: return s[1:-1]


def enclosing_block(text, pos):
    """The innermost {...} containing pos (best effort: scan back to an unmatched '{')."""
    depth = 0
    for i in range(pos, -1, -1):
        if text[i] == "}": depth += 1
        elif text[i] == "{":
            if depth == 0: return i, balanced(text, i)
            depth -= 1
    return None, None


def question_id_before(text, block_start):
    m = re.search(r"[\"']?([\w-]+)[\"']?\s*:\s*$", text[max(0, block_start - 80):block_start])
    return m.group(1) if m else None


def extract_questions(text, path):
    """Typed question literals in one file -> {id: question} drafts."""
    found = {}
    for m in TYPE_RE.finditer(text):
        start, block = enclosing_block(text, m.start())
        if not block: continue
        line = text.count("\n", 0, m.start()) + 1
        qtype = {"boolean": "noul"}.get(m.group(1), m.group(1))
        instr = INSTR_RE.search(block)
        if not instr: continue   # a type declaration or a response shape, not a question
        crit = CRIT_RE.search(block)
        raw = balanced(block, crit.end()) if crit else None
        criteria = literal(raw)
        qid = question_id_before(text, start) or f"question_{len(found) + 1}"
        q = {"type": qtype, "instructions": unquote(instr.group(1)), "_source": f"{path}:{line}"}
        if qtype != "noul" or crit: q["criteria"] = criteria if criteria is not None else (f"_todo: parse `{(raw or block[crit.end():crit.end() + 80]).strip()[:120]}`" if crit else "_todo: criteria")
        found[qid] = q
    for m in SDK_RE.finditer(text):   # Python SDK: "id": Choice(instructions=..., criteria={...})
        qid, cls = m.group(1), m.group(2)
        call = balanced(text.replace("(", "[").replace(")", "]"), m.end() - 1) or ""
        call = text[m.end() - 1:m.end() - 1 + len(call)]
        line = text.count("\n", 0, m.start()) + 1
        instr = INSTR_RE.search(call); crit = CRIT_RE.search(call)
        q = {"type": {"Noul": "noul", "Choice": "choice", "Score": "score"}[cls], "instructions": unquote(instr.group(1)) if instr else "_todo: instructions", "_source": f"{path}:{line}"}
        if cls != "Noul":
            criteria = literal(balanced(call, crit.end())) if crit else None
            q["criteria"] = criteria if criteria is not None else "_todo: criteria (could not parse)"
        found.setdefault(qid, q)
    return found


def data_candidates(paths):
    out = []
    for p in paths:
        if p.suffix.lower() not in DATA_EXT or p.name in ("package.json", "package-lock.json", "tsconfig.json", "uv.lock"): continue
        text = read(p)
        lines = text.count("\n")
        if p.suffix.lower() == ".json":
            data = literal(text[:2_000_000])
            if not isinstance(data, list) or len(data) < 20: continue
            head, rows = json.dumps(data[0], ensure_ascii=False)[:200], len(data)
        else:
            if lines < 20: continue
            head, rows = text.split("\n", 1)[0][:200], lines
        out.append((p, rows, head))
    return sorted(out, key=lambda t: -t[1])[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="directory to scan (default: current)")
    ap.add_argument("--out", help="write a draft workload spec here (fill in the _todo fields, then delete the _ keys)")
    ap.add_argument("--max-lines", type=int, default=6, help="matching lines to show per file")
    a = ap.parse_args()
    paths = list(files_under(a.root))
    hits, questions = defaultdict(list), {}
    for p in paths:
        if p.suffix.lower() not in CODE_EXT: continue
        text = read(p)
        score = 0
        for n, line in enumerate(text.splitlines(), 1):
            for rx, w, what in MARKERS:
                if rx.search(line):
                    score += w; hits[p].append((n, what, line.strip()[:140])); break
        if score >= 3:
            for qid, q in extract_questions(text, p.relative_to(a.root) if a.root != "." else p).items():
                questions.setdefault(qid, q)
        hits[p] = (score, hits[p]) if score else None
    ranked = sorted(((p, s, ls) for p, v in hits.items() if v for s, ls in [v]), key=lambda t: -t[1])
    if not ranked: print("no Jev / TypeSafe / System One usage found; describe the workload by hand from assets/workload.example.json")
    else:
        print(f"{len(ranked)} file(s) mention Jev / TypeSafe / System One (score = weighted marker hits):")
        for p, s, ls in ranked[:15]:
            print(f"\n  {p}  [score {s}]")
            for n, what, line in ls[:a.max_lines]: print(f"    {n:>5}: {line}   <- {what}")
    if questions:
        print(f"\n{len(questions)} question literal(s) extracted (check each _source; _todo marks what could not be parsed):")
        for qid, q in questions.items():
            print(f"  {qid} ({q['type']}): {str(q.get('instructions'))[:90]}  [{q['_source']}]" + (f"  criteria: {json.dumps(q['criteria'], ensure_ascii=False)[:100]}" if "criteria" in q else ""))
    data = data_candidates(paths)
    if data:
        print("\nlabelled-looking data files (candidates for scripts/convert_data.py):")
        for p, rows, head in data: print(f"  {p}  ({rows} rows)  first line: {head}")
    if a.out:
        spec = {"name": Path(a.root).resolve().name, "domain": "_todo: one or two sentences: what the inputs are and who writes them",
                "state": "_todo: what one input looks like (length, voice, fields). Add state_example if states are objects.",
                "questions": {qid: {k: v for k, v in q.items()} for qid, q in questions.items()} or {"_todo": "no question literals found; write them in the System One shape (see assets/workload.example.json)"},
                "guidance": "_todo: the labelling rules a careful human would follow; resolve the ambiguous cases here",
                "variety": ["_todo: axes to vary (tone, length, language, edge cases)"],
                "_sources": [str(p) for p, _, _ in ranked[:15]], "_data_files": [str(p) for p, _, _ in data]}
        Path(a.out).write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\ndraft spec written to {a.out}; fill in every _todo, remove the _source/_sources/_data_files keys, then: python3 scripts/generate_data.py {a.out} --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
