"""longdoc-v1: a frozen, eval-only, report-only probe of how Kev degrades on long documents, past the state length it was
trained on (7,552 tokens).

    uv run python scripts/build_longdoc_v1.py --out evals/longdoc-v1                  # ~10 min; CUAD from the Hub (pinned)
    uv run python scripts/screen_longdoc_v1.py --suite evals/longdoc-v1               # overlap.json, counts only
    uv run python scripts/longdoc_report.py --suite evals/longdoc-v1 --result NAME=runs/<read> ...

Five length buckets by state tokens under the Qwen3.8-27B tokenizer (Kev-27B's base, TOKENIZER): 4k (inside the trained
range, the control), 8k, 16k, 32k and 64k. A bucket's states hold 84-93 % of its nominal size and every question row (state +
one question) fits the nominal size, so a server with that context admits the whole bucket. Two parts, the same number of
records in every bucket:

  cuad       real contracts with expert labels (CUAD v1, CC BY 4.0; CUAD below). Each record is one target contract padded
             with other CUAD contracts of the same partition (whole, the last one cut at a paragraph and marked as an
             excerpt) to the bucket's length, the target placed at 10 %, 50 % or 90 % of the state. Questions name the target
             by its filing title: two clause-presence questions (Noul: one clause the annotators found, one they did not),
             one clause-category choice (one found, three not found) and, when the annotators' governing-law answer is one of
             the common jurisdictions, a governing-law choice. Paired by design: the 8k-64k buckets use the same targets with
             the same questions, only the padding differs; the 4k bucket uses the targets short enough for it (<= 3,100
             tokens), each more than once with different padding. Labels are CUAD's (`is_impossible` in CUAD_v1.json, the
             Governing Law answer in master_clauses.csv).
  synthetic  generated bundles of service agreements (scripts/longdoc_v1_synthetic.py): locate one agreement by a unique
             fact, a detail buried at 10 / 50 / 90 % depth, a two-hop credit lookup across sections far apart, and a term
             that is stated or absent (answer "not stated"). Labels computed from `_meta.facts` by its solver.

Partitions: development and test (locked: read once per candidate with --allow-test, never for this probe). CUAD contracts
are dealt to the two partitions by company (the filing name's first field) in a seeded hash order, so no company spans
both. Every state is deduplicated by normalised text across both partitions. The partitions are ~100 MB each: a private
mirror (manifest only in git, `mirror` names kev.suite.PRIVATE_DATASET). Deterministic: the same inputs give the same bytes.
"""
import argparse, csv, hashlib, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.data import materialize  # noqa: E402
from kev.model import encode, load_tokenizer, long_eval_context, rows_of, user_tokens  # noqa: E402
from kev.suite import PRIVATE_DATASET, digest, read_jsonl, text_digest, write_json, write_jsonl  # noqa: E402
from scripts import longdoc_v1_synthetic as SYN  # noqa: E402

VERSION = SEED = "longdoc-v1"
PARTITIONS = ("development", "test")
BUCKETS = (4096, 8192, 16384, 32768, 65536)
WINDOW = (0.80, 0.93)                     # a bucket's states: this share of its nominal size (rows must also fit the nominal size)
RECORDS = 120                             # per bucket, part and partition
DEPTHS = SYN.DEPTHS
TOKENIZER = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")   # Kev-27B's base (modal_app.KEV_27B_BASE)
CUAD = {"repo": "theatticusproject/cuad", "revision": "a3c393f5d103fd0c516374e4fdff676c8176dcb1",
        "files": {"json": "CUAD_v1/CUAD_v1.json", "clauses": "CUAD_v1/master_clauses.csv", "readme": "CUAD_v1/CUAD v1 ReadMe _ Datasheet/CUAD_v1_README.txt"},
        "licence": "CC-BY-4.0", "licence_evidence": ["https://www.atticusprojectai.org/cuad (dataset page: 'CC BY 4.0')",
                                                     "CUAD_v1_README.txt: 'CUAD is licensed under the Creative Commons Attribution 4.0 (CC BY 4.0) license and free to the public for commercial and non-commercial use.'",
                                                     "Hub card theatticusproject/cuad: license cc-by-4.0"],
        "attribution": "Hendrycks, Burns, Chen and Ball, CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review (NeurIPS 2021 Datasets and Benchmarks); The Atticus Project, Inc.",
        "changes": "contract text whitespace normalised (trailing spaces, runs of blank lines and of spaces collapsed); contracts concatenated into review files; questions written from CUAD's category descriptions"}
CUAD_TARGET_MAX = 6500                    # target contracts for 8k-64k (tokens); every one fits the 8k bucket with padding room
CUAD_TARGET_MAX_4K = 3100                 # ... and for the 4k bucket (room for at least one other contract)
NOT_YES_NO = {"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
              "Notice Period To Terminate Renewal", "Governing Law", "Competitive Restriction Exception", "Warranty Duration"}
CODE = ("scripts/build_longdoc_v1.py", "scripts/longdoc_v1_synthetic.py", "kev/api.py", "kev/data.py", "kev/model.py")


class Tokens:
    """State token counts as kev.model.encode counts them (user text + the state delimiter) under TOKENIZER."""

    def __init__(self):
        self.tok = load_tokenizer(*TOKENIZER)

    def __call__(self, text):
        return len(user_tokens(self.tok, text)) + 1


def window(T):
    return int(WINDOW[0] * T), int(WINDOW[1] * T)


# ------------------------------------------------------------------------------------------------------------- CUAD
def clean(text):
    text = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def company(title):
    """The filer: the filing name's first field ("LIMEENERGYCO" of "LIMEENERGYCO_09_09_1999-EX-10-..."; the name before " - "
    for the few named "Company - Agreement"), letters and digits only."""
    return re.sub(r"[^A-Z0-9]", "", re.split(r"_| - ", title)[0].upper())


def title_key(name):
    """Letters and digits of a contract name without ".pdf": CUAD_v1.json titles and master_clauses.csv file names differ in
    punctuation for 11 contracts ('&' vs '_', a trailing space or '-', a stray quote)."""
    return re.sub(r"[^A-Z0-9]", "", re.sub(r"\.pdf'?$", "", name.strip(), flags=re.I).upper())


def load_cuad(raw_files, count):
    """CUAD contracts: title -> {text, tokens, company, present (yes/no categories found), absent, law, questions text}."""
    data = json.loads(Path(raw_files["json"]).read_text(encoding="utf-8"))["data"]
    laws = {}
    with Path(raw_files["clauses"]).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            laws[title_key(row["Filename"])] = row["Governing Law-Answer"].strip()
    contracts, details = {}, {}
    for doc in data:
        (para,) = doc["paragraphs"]
        present, absent = [], []
        for q in para["qas"]:
            cat = q["id"].split("__")[-1]
            if cat in NOT_YES_NO: continue
            details[cat] = q["question"].split("Details:")[-1].strip().replace("  ", " ")
            (absent if q["is_impossible"] else present).append(cat)
        text = clean(para["context"])
        if title_key(doc["title"]) not in laws: raise ValueError(f"no master_clauses row for {doc['title']}")
        contracts[doc["title"]] = {"text": text, "tokens": count(text), "company": company(doc["title"]), "present": sorted(present),
                                   "absent": sorted(absent), "law": laws[title_key(doc["title"])]}
    return contracts, details


def deal(contracts):
    """Companies to development / test by a seeded hash (half each, by contract count as it falls)."""
    out = {}
    for title, c in contracts.items():
        h = int(hashlib.sha256(f"{SEED}:{c['company']}".encode()).hexdigest(), 16)
        out[title] = PARTITIONS[h % 2]
    return out


def cuad_questions(title, c, details, prevalence, laws_common):
    """The fixed question set of one target (the same in every bucket): specs with native labels."""
    rng = random.Random(f"{SEED}:cuad:questions:{title}")
    plausible = [k for k in c["absent"] if prevalence[k] >= 0.10] or c["absent"]
    yes = rng.choice(c["present"]); no = rng.choice(plausible)
    cat_yes = rng.choice([k for k in c["present"] if k != yes] or [yes])
    pool_no = [k for k in plausible if k != no]
    if len(pool_no) < 3: pool_no = [k for k in c["absent"] if k != no]
    cat_no = rng.sample(pool_no, 3) if len(pool_no) >= 3 else []
    specs = [{"qid": "presence_found", "kind": "presence", "category": yes, "label": True},
             {"qid": "presence_not_found", "kind": "presence", "category": no, "label": False}]
    if len(cat_no) == 3:
        opts = [cat_yes] + cat_no; rng.shuffle(opts)
        specs.append({"qid": "category", "kind": "category", "options": opts, "label": cat_yes})
    if c["law"] in laws_common:
        others = rng.sample([x for x in laws_common if x != c["law"]], 3)
        opts = [c["law"]] + others; rng.shuffle(opts)
        specs.append({"qid": "governing_law", "kind": "governing_law", "options": opts, "label": c["law"]})
    return specs


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def cuad_request(title, specs, details):
    ref = f'Consider only the contract titled "{title}" in this file.'
    qs = {}
    for s in specs:
        if s["kind"] == "presence":
            qs[s["qid"]] = {"type": "noul", "instructions": f"{ref} {details[s['category']]}", "label": s["label"], "src": None}
        elif s["kind"] == "category":
            qs[s["qid"]] = {"type": "choice", "instructions": f"{ref} Which one of these clause types does it contain?",
                            "criteria": {slug(k): k for k in s["options"]}, "label": slug(s["label"]), "src": None}
        else:
            qs[s["qid"]] = {"type": "choice", "instructions": f"{ref} Which jurisdiction's law governs it?",
                            "criteria": {slug(k): k for k in s["options"]}, "label": slug(s["label"]), "src": None}
    return qs


def cut(text, budget_tokens, count):
    """The longest prefix of `text` ending at a paragraph break that holds at most budget_tokens tokens, or None."""
    paras = text.split("\n\n")
    lo, hi, best = 1, len(paras), None
    while lo <= hi:
        mid = (lo + hi) // 2
        piece = "\n\n".join(paras[:mid])
        if count(piece) <= budget_tokens: best, lo = piece, mid + 1
        else: hi = mid - 1
    return best


def cuad_state(blocks):
    n = len(blocks)
    parts, starts, pos = [f"Contract review file: {n} commercial contracts filed with the SEC, in no particular order.\n\n"], [], None
    pos = len(parts[0])
    for k, (title, text, excerpt) in enumerate(blocks, start=1):
        head = f"===== Contract {k} of {n}: {title}{' (excerpt; the rest of this contract is not included)' if excerpt else ''} =====\n"
        starts.append((pos + len(head), len(text)))
        block = head + text + f"\n===== End of contract {k} =====\n\n"
        parts.append(block); pos += len(block)
    return "".join(parts).rstrip() + "\n", starts


def cuad_record(title, contracts, pool, T, depth, rng, count):
    """One review file around `title` at bucket T with the target's centre nearest `depth`; None if the draw misses."""
    lo, hi = window(T)
    target = contracts[title]
    fill = rng.uniform(0.86 * T, WINDOW[1] * T - 64)
    others = [t for t in pool if t != title and contracts[t]["company"] != target["company"]]
    rng.shuffle(others)
    chosen, used = [], target["tokens"] + 40
    for t in others:
        n = contracts[t]["tokens"] + 30
        if used + n <= fill:
            chosen.append((t, contracts[t]["text"], False)); used += n
        elif fill - used >= 300:
            piece = cut(contracts[t]["text"], int(fill - used - 30), count)
            if piece: chosen.append((t, piece, True)); used += count(piece) + 30
            break
        if fill - used < 300: break
    if not chosen: return None                               # every file holds at least one contract besides the target
    best = None
    for slot in range(len(chosen) + 1):
        blocks = chosen[:slot] + [(title, target["text"], False)] + chosen[slot:]
        state, starts = cuad_state(blocks)
        start, length = starts[slot]
        frac = (start + length / 2) / len(state)
        if best is None or abs(frac - depth) < abs(best[0] - depth): best = (frac, blocks, state, starts[slot])
    _, blocks, state, (start, length) = best
    total = count(state)
    if not lo <= total <= hi: return None
    centre = round(count(state[:start + length // 2]) / total, 4)
    return {"state": state, "state_tokens": total, "blocks": [(t, e) for t, _, e in blocks], "target_depth": centre,
            "target_start": round(count(state[:start]) / total, 4)}


def build_cuad(split, contracts, titles, details, count, report, seen, n=RECORDS):
    prevalence = {k: sum(k in c["present"] for c in contracts.values()) / len(contracts) for k in details}
    law_counts = Counter(c["law"] for c in contracts.values())
    laws_common = sorted(k for k, v in law_counts.items() if v >= 3 and k and not re.search(r"[;/]", k))
    pool = sorted((t for t in titles if contracts[t]["present"] and contracts[t]["absent"]), key=lambda t: hashlib.sha256(f"{SEED}:{t}".encode()).hexdigest())
    targets = [t for t in pool if contracts[t]["tokens"] <= CUAD_TARGET_MAX]
    short = [t for t in targets if contracts[t]["tokens"] <= CUAD_TARGET_MAX_4K]
    report[split] = {"contracts": len(titles), "eligible_targets": len(pool), "targets_8k_64k_pool": len(targets), "targets_4k_pool": len(short)}
    records = defaultdict(list)
    for T in BUCKETS:
        chosen = short if T == BUCKETS[0] else targets
        stats = Counter()
        i = 0
        while len(records[T]) < n:
            if i >= 40 * n: raise RuntimeError(f"cuad {split} {T}: only {len(records[T])} records ({dict(stats)})")
            title = chosen[i % len(chosen)]; rep = i // len(chosen)
            depth = DEPTHS[len(records[T]) % len(DEPTHS)]
            rng = random.Random(f"{SEED}:cuad:{split}:{T}:{title}:{rep}:{i}")
            i += 1
            got = cuad_record(title, contracts, titles, T, depth, rng, count)
            if got is None: stats["length_rejected"] += 1; continue
            if text_digest(got["state"]) in seen: stats["duplicate_state"] += 1; continue
            seen.add(text_digest(got["state"]))
            specs = cuad_questions(title, contracts[title], details, prevalence, laws_common)
            qs = cuad_request(title, specs, details)
            b = T // 1024
            for s in specs:
                qs[s["qid"]]["src"] = f"cuad.{s['kind']}.{b}k"
                others = [t for t, _ in got["blocks"] if t != title]
                if s["kind"] == "presence": s["distractors_with_clause"] = sum(s["category"] in contracts[t]["present"] for t in others)
                if s["kind"] == "governing_law": s["distractors_with_law"] = sum(contracts[t]["law"] == s["label"] for t in others)
            rid = f"longdoc-v1/cuad/{split}/{b}k/{len(records[T]):04d}"
            records[T].append({"state": got["state"], "questions": qs, "_meta": {
                "id": rid, "source": "longdoc_cuad", "group_id": f"cuad:{title}", "variant": "clean", "split": split, "part": "cuad",
                "bucket": T, "state_tokens": got["state_tokens"], "depth_target": depth, "target": title, "target_tokens": contracts[title]["tokens"],
                "target_depth": got["target_depth"], "target_start": got["target_start"], "repeat": rep,
                "contracts": [{"title": t, "excerpt": e} for t, e in got["blocks"]], "questions": specs}})
            stats["accepted"] += 1
        report[f"{split}:{T}"] = dict(stats)
    return records


# ------------------------------------------------------------------------------------------------------------- synthetic
def build_synthetic(split, count, report, n=RECORDS):
    records = defaultdict(list)
    for T in BUCKETS:
        rng = random.Random(f"{SEED}:synthetic:{split}:{T}")
        stats, b = Counter(), T // 1024
        while len(records[T]) < n:
            if stats["attempts"] >= 60 * n: raise RuntimeError(f"synthetic {split} {T}: only {len(records[T])} ({dict(stats)})")
            stats["attempts"] += 1
            depth = DEPTHS[len(records[T]) % len(DEPTHS)]
            got = SYN.generate(rng, T, depth, count)
            if got is None: stats["draw_rejected"] += 1; continue
            facts = json.loads(json.dumps(got["facts"]))   # the stored facts alone decide every label
            qs = {}
            for qid, q in got["questions"].items():
                qs[qid] = {**q, "label": SYN.solve(facts, got["specs"][qid]), "src": f"synthetic.{got['specs'][qid]['kind']}.{b}k"}
            rid = f"longdoc-v1/synthetic/{split}/{b}k/{len(records[T]):04d}"
            records[T].append({"state": got["state"], "questions": qs, "_meta": {
                "id": rid, "source": "longdoc_synthetic", "group_id": rid, "variant": "clean", "split": split, "part": "synthetic",
                "bucket": T, "state_tokens": got["state_tokens"], "depth_target": depth, "agreements": got["agreements"],
                "questions": got["specs"], "facts": facts}})
        stats["accepted"] = len(records[T])
        report[f"{split}:{T}"] = dict(stats)
    return records


# ------------------------------------------------------------------------------------------------------------- checks
def admit(record, tok):
    """Encodes strictly in the long-document context, the state inside its bucket's window and every row (state + one
    question) inside the bucket's nominal size. -> the longest row in tokens."""
    T = record["_meta"]["bucket"]
    ctx = long_eval_context()
    enc = encode(tok, materialize(record), max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True)
    state, _, rows = rows_of(enc)
    lo, hi = window(T)
    if not lo <= len(state) <= hi: raise ValueError(f"{record['_meta']['id']}: state {len(state)} outside [{lo}, {hi}]")
    if len(state) != record["_meta"]["state_tokens"]: raise ValueError(f"{record['_meta']['id']}: token count drifted")
    longest = len(state) + max(len(r["ids"]) for r in rows)
    if longest > T: raise ValueError(f"{record['_meta']['id']}: a row of {longest} tokens exceeds {T}")
    return longest


def quantiles(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"min": xs[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": xs[-1], "mean": round(sum(xs) / len(xs), 1)}


def summary(records):
    out = {}
    for part in ("cuad", "synthetic"):
        for T in BUCKETS:
            rs = [r for r in records if r["_meta"]["part"] == part and r["_meta"]["bucket"] == T]
            kinds, balance = Counter(), Counter()
            for r in rs:
                for q in r["questions"].values():
                    kinds[q["src"].split(".")[1]] += 1
                    if q["type"] == "noul": balance[f"noul_{str(q['label']).lower()}"] += 1
                    else: balance[f"position {list(q['criteria']).index(q['label'])}/{len(q['criteria'])}"] += 1
            depth_key = "target_depth" if part == "cuad" else "depth_target"
            out[f"{part}/{T // 1024}k"] = {"records": len(rs), "questions": sum(kinds.values()), "kinds": dict(sorted(kinds.items())),
                                           "labels": dict(sorted(balance.items())), "state_tokens": quantiles([r["_meta"]["state_tokens"] for r in rs]),
                                           "depth": dict(sorted(Counter(round(r["_meta"][depth_key], 1) for r in rs).items()))}
            if part == "synthetic":
                out[f"{part}/{T // 1024}k"]["absent_not_stated"] = sum(not r["_meta"]["questions"]["absent"]["stated"] for r in rs)
                out[f"{part}/{T // 1024}k"]["agreements"] = quantiles([r["_meta"]["agreements"] for r in rs])
            else:
                out[f"{part}/{T // 1024}k"]["distinct_targets"] = len({r["_meta"]["target"] for r in rs})
    return out


def fetch_cuad():
    from huggingface_hub import hf_hub_download
    out = {}
    for key, name in CUAD["files"].items():
        out[key] = hf_hub_download(CUAD["repo"], name, repo_type="dataset", revision=CUAD["revision"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=RECORDS, help="records per bucket, part and partition (tests use fewer)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    count = Tokens()
    files = fetch_cuad()
    contracts, details = load_cuad(files, count)
    split_of = deal(contracts)
    parts, report, seen = {}, {"cuad": {}, "synthetic": {}}, set()
    for split in ("test", "development"):   # test first, so a state seen in test is never regenerated for development
        titles = sorted(t for t, s in split_of.items() if s == split)
        cu = build_cuad(split, contracts, titles, details, count, report["cuad"], set(seen), a.records)
        sy = build_synthetic(split, count, report["synthetic"], a.records)
        records = []
        for T in BUCKETS:
            records += cu[T] + sy[T]
        for r in records:
            key = text_digest(r["state"])
            if key in seen: raise ValueError(f"duplicate state: {r['_meta']['id']}")
            seen.add(key); r["_meta"]["text_sha256"] = key
            r["_meta"]["longest_row_tokens"] = admit(r, count.tok)
        parts[split] = records
    out.mkdir(parents=True)
    manifest_files = {}
    for split in PARTITIONS:
        path = out / f"{split}.jsonl"
        write_jsonl(path, parts[split])
        if read_jsonl(path) != parts[split]: raise AssertionError(f"{path} does not round-trip")
        manifest_files[path.name] = {"sha256": digest(path), "records": len(parts[split]), "questions": sum(len(r["questions"]) for r in parts[split]),
                                     "bytes": path.stat().st_size, "in_git": False, "by_bucket": summary(parts[split])}
    write_json(out / "manifest.json", {
        "version": VERSION, "seed": SEED, "eval_only": True, "report_only": True, "partitions": list(PARTITIONS), "locked": ["test"],
        "files": manifest_files, "trainable_sources": [], "eval_only_sources": ["longdoc_cuad", "longdoc_synthetic"], "holdout_sources": [],
        "mirror": {"dataset": PRIVATE_DATASET, "revision": None},
        "mirror_note": "partitions are ~100 MB each: kept out of git (GIT_LIMIT) in the private eval mirror, fetched and hash-checked by kev.suite.load_split",
        "buckets": {f"{T // 1024}k": {"nominal_tokens": T, "state_tokens": list(window(T)), "row_limit": T} for T in BUCKETS},
        "bucket_rule": f"state tokens (kev.model.encode under the tokenizer below, state delimiter included) in [{WINDOW[0]}, {WINDOW[1]}] x nominal; every row (state + one question) <= nominal",
        "records_per_bucket_part": a.records, "depths": list(DEPTHS),
        "parts": {"cuad": {"source": CUAD, "questions": "presence_found / presence_not_found (Noul, CUAD is_impossible), category (choice: one found, three not found), "
                                                        "governing_law (choice, master_clauses Governing Law-Answer, only when it is one of the jurisdictions answered >= 3 times)",
                           "design": f"target contracts <= {CUAD_TARGET_MAX} tokens for 8k-64k (same targets and questions in each bucket), <= {CUAD_TARGET_MAX_4K} for 4k (reused with new padding); "
                                     "padding: other contracts of the same partition from other companies, whole, the last cut at a paragraph and marked as an excerpt; target centre at 10/50/90 % of the state",
                           "report": report["cuad"]},
                  "synthetic": {"generator": "scripts/longdoc_v1_synthetic.py", "kinds": ["locate", "detail", "multihop", "absent"],
                                "labels": "programmatic: longdoc_v1_synthetic.solve over _meta.facts after a JSON round trip; no model or human judgement",
                                "templates": "written for this suite; no text, pool or template shared with hard-v1", "report": report["synthetic"]}},
        "tokenizer": {"model": TOKENIZER[0], "revision": TOKENIZER[1]}, "base_revisions": {TOKENIZER[0]: TOKENIZER[1]},
        "context": {**long_eval_context(), "truncate": False,
                    "note": "kev.model.long_eval_context: scored past the serving context; `long_document` makes kev.predictors.LocalPredictor run the state once with question rows from its cache under the fused attention kernels (its precision policy)"},
        "selection": "normalised-text state dedupe across both partitions (test first); CUAD companies dealt to partitions by seeded hash; every record admitted by kev.model.encode (strict) in the context above and its bucket rule",
        "label_protocol": "no LLM labels: CUAD's expert annotations (cuad part) or code (synthetic part)",
        "overlap_screen": "overlap.json (scripts/screen_longdoc_v1.py): JevBench public items, every Kev development/test partition, LEDGAR (in the SFT corpus) and ContractNLI (breadth-v1); counts only",
        "code_sha256": {name: digest(ROOT / name) for name in CODE},
    })
    print(json.dumps({s: {k: v for k, v in f.items() if k != "by_bucket"} for s, f in manifest_files.items()}, indent=1))


if __name__ == "__main__":
    main()
