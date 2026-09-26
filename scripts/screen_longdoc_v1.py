"""Overlap screen for evals/longdoc-v1 (counts only, never reference text): writes <suite>/overlap.json.

    uv run python scripts/screen_longdoc_v1.py --suite evals/longdoc-v1 --jevbench /tmp/jevbench/datasets/public

References, each item = its state plus its question instructions:
  jevbench      JevBench's public items (the external benchmark we must not tune toward)
  kev_eval      every Kev development and test partition under evals/ (private mirrors fetched with the caller's access)
  ledgar        LEDGAR provisions (coastalcph/lex_glue at the SFT corpus's pinned revision): the SFT corpus trains on LEDGAR,
                and LEDGAR and CUAD are both clauses from contracts filed with the SEC
Per longdoc record (state + instructions) and reference item, on word 8-grams (casefold, \\w+, as scripts/screen_overlap.py):
the item's containment in the record (shared grams / the item's grams; items with fewer than MIN_GRAMS ignored, grams shared
by more than COMMON reference items of a collection ignored as boilerplate) and exact normalised state equality. A record
is counted as overlapping a collection when some item of it is contained at >= CONTAINMENT. ContractNLI (a breadth-v1
dataset, SEC-filed NDAs) is reported on its own from the breadth-v1 partitions.
"""
import argparse, hashlib, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.api import render  # noqa: E402
from kev.suite import digest, load_split, read_jsonl, read_manifest, write_json  # noqa: E402

N, MIN_GRAMS, COMMON, CONTAINMENT = 8, 10, 50, 0.5
LEDGAR = ("coastalcph/lex_glue", "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c", [f"ledgar/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")])


def words(value):
    return re.findall(r"\w+", (value if isinstance(value, str) else render(value)).casefold())


def grams(tokens):
    return {hash(" ".join(tokens[i:i + N])) for i in range(len(tokens) - N + 1)}


def norm_key(value):
    return hashlib.sha256(" ".join(words(value)).encode()).hexdigest()


def item_text(r):
    """(state, instructions) of a Kev record or an external item."""
    if "questions" in r and isinstance(r["questions"], dict):
        return r.get("state", ""), [q.get("instructions") or "" for q in r["questions"].values()]
    return r.get("state") or r.get("text") or "", [r.get("question") or ""]


class Collection:
    def __init__(self, name):
        self.name, self.sets, self.states, self.files = name, [], set(), {}

    def add(self, state, instructions):
        s = grams(words(state) + [w for x in instructions for w in words(x)])
        if len(s) >= MIN_GRAMS: self.sets.append(s)
        if state: self.states.add(norm_key(state))

    def index(self):
        df = Counter(g for s in self.sets for g in s)
        self.common = {g for g, c in df.items() if c > COMMON}
        self.sizes = [len(s - self.common) for s in self.sets]
        self.inv = defaultdict(list)
        for i, s in enumerate(self.sets):
            for g in s - self.common: self.inv[g].append(i)

    def screen(self, mine):
        shared = Counter(i for g in mine for i in self.inv.get(g, ()))
        return max((c / self.sizes[i] for i, c in shared.items() if self.sizes[i] >= MIN_GRAMS), default=0.0)


def kev_eval(exclude):
    c, contractnli = Collection("kev_eval"), Collection("contractnli")
    for manifest in sorted((ROOT / "evals").rglob("manifest.json")):
        d = manifest.parent
        if d.resolve() == Path(exclude).resolve(): continue
        files = read_manifest(d).get("files", {})
        for split in ("development", "test"):
            if f"{split}.jsonl" not in files: continue
            try: rows = load_split(d, split, allow_test=True)   # read for screening only (counts); never scored here
            except Exception as e: c.files[f"{d.relative_to(ROOT)}/{split}.jsonl"] = f"unreadable: {type(e).__name__}"; continue
            c.files[f"{d.relative_to(ROOT)}/{split}.jsonl"] = {"records": len(rows), "sha256": files[f"{split}.jsonl"]["sha256"]}
            for r in rows:
                state, instr = item_text(r); c.add(state, instr)
                if r.get("_meta", {}).get("source") == "contractnli": contractnli.add(state, instr)
    return c, contractnli


def jevbench(directory):
    c = Collection("jevbench")
    for path in sorted(Path(directory).glob("*.jsonl")):
        rows = read_jsonl(path); c.files[path.name] = {"items": len(rows), "sha256": digest(path)}
        for r in rows: c.add(*item_text(r))
    return c


def ledgar():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    c = Collection("ledgar")
    for name in LEDGAR[2]:
        path = hf_hub_download(LEDGAR[0], name, repo_type="dataset", revision=LEDGAR[1])
        rows = pq.read_table(path).to_pylist(); c.files[name] = {"items": len(rows), "sha256": digest(path)}
        for r in rows: c.add(r["text"], [])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--jevbench", default="/tmp/jevbench/datasets/public")
    a = ap.parse_args()
    suite = Path(a.suite)
    kev, cnli = kev_eval(suite)
    collections = [jevbench(a.jevbench), kev, cnli, ledgar()]
    for c in collections: c.index(); print(f"{c.name}: {len(c.sets)} items", flush=True)
    out = {"suite": str(suite), "n": N, "normalisation": "casefold, \\w+ tokens", "min_grams": MIN_GRAMS, "common_grams_over_items": COMMON,
           "containment_threshold": CONTAINMENT, "references": {c.name: {"items": len(c.sets), "files": c.files} for c in collections},
           "partitions": {}, "note": "counts only; no reference text is stored. A record overlaps a collection when one of its items has >= "
                                     f"{CONTAINMENT} of its word 8-grams inside the record (after dropping grams shared by > {COMMON} of that collection's items)."}
    for split in ("development", "test"):
        stats = defaultdict(lambda: defaultdict(lambda: {"records": 0, "exact_state": 0, "overlapping": 0, "max_containment": 0.0}))
        for r in read_jsonl(suite / f"{split}.jsonl"):
            m = r["_meta"]
            mine = grams(words(r["state"]) + [w for q in r["questions"].values() for w in words(q["instructions"])])
            key = norm_key(r["state"])
            for c in collections:
                s = stats[c.name][f"{m['part']}/{m['bucket'] // 1024}k"]
                best = c.screen(mine)
                s["records"] += 1; s["exact_state"] += key in c.states; s["overlapping"] += best >= CONTAINMENT
                s["max_containment"] = round(max(s["max_containment"], best), 4)
            print(split, m["id"], flush=True) if m["id"].endswith("0000") else None
        out["partitions"][split] = {c: dict(v) for c, v in stats.items()}
    write_json(suite / "overlap.json", out)
    for split, cs in out["partitions"].items():
        for c, groups in cs.items():
            print(split, c, {g: (v["overlapping"], v["records"], v["max_containment"]) for g, v in groups.items()})


if __name__ == "__main__":
    main()
