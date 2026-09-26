"""hard-v1: a frozen suite of programmatically labelled decision records aimed at the skills where Kev is weak.

    uv run python scripts/build_hard_v1.py --out evals/hard-v1

Seven families (scripts/hard_v1_policy.py, scripts/hard_v1_families.py), each its own generator with its own seeded RNG
per split and six surface templates:

  long_policy       a generated insurance / benefits / device-protection policy (800-5,000 tokens) and a claim: settlement
                    outcome and amount payable, from the generator's own rule engine
  tradeoff          options with attributes, hard requirements, then a lexicographic, weighted or total-cost rule
  probability       expected value, base rates (Bayes), independent events, which-is-more-likely, conditional shares
  multi_hop         approval chains, service dependency graphs, ownership and control, on-call paging
  temporal_numeric  business-day deadlines, date differences, time zones, recurring schedules, pro-rata refunds, shift pay
  judge             a question and a proposed answer that is right or carries one planted error: is it correct? (+ the value)
  ambiguous         the deciding fact present or absent, with an explicit insufficient-information option; twins share a group

Every label is computed: the builder takes it from the family's solver applied to the record's `_meta.facts` (after a JSON
round trip, so the stored facts alone determine it), never from the generator. Templates 0-3 are training templates,
template 4 is development only and template 5 test only (TEMPLATE_SPLITS), so development and test measure transfer to an
unseen phrasing and layout; seeds differ per (family, split). States are deduplicated by normalised rendered text across
all partitions. Every record is validated through kev.data.materialize and must encode, untruncated, in the long-state
training context (kev.model.training_context(MAX_TRAIN_STATE)) and in the serving context under the Qwen3.5 tokenizer.
Deterministic: the same arguments give the same bytes.
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.api import render  # noqa: E402
from kev.data import materialize  # noqa: E402
from kev.model import MAX_TRAIN_STATE_8K as MAX_TRAIN_STATE, fits, load_tokenizer, training_context  # noqa: E402
from kev.suite import ADMISSION_TOKENIZER as TOKENIZER, GIT_LIMIT, SERVING_CONTEXT_8K as SERVING_CONTEXT, digest, text_digest, write_json, write_jsonl  # noqa: E402
from scripts.hard_v1_common import Ctx  # noqa: E402
from scripts.hard_v1_families import ABSTAIN_KEYS, FAMILIES, labels  # noqa: E402

SEED = "hard-v1-20260923"
TEMPLATE_SPLITS = {"train": (0, 1, 2, 3), "development": (4,), "test": (5,)}
SIZES = {"train": 6000, "development": 700, "test": 700}
LONG_POLICY_TOKENS = (800, 5000)
CODE = ("scripts/build_hard_v1.py", "scripts/hard_v1_common.py", "scripts/hard_v1_policy.py", "scripts/hard_v1_families.py",
        "scripts/hard_v1_numeric.py", "kev/api.py", "kev/data.py", "kev/model.py")


def family_counts(total):
    """Records per family: equal shares, the remainder to `ambiguous` first (its records come in twins, so its count must be
    even) and then in family order. Needs one record per family and a twin pair for `ambiguous`: below that a family
    would get none (or, after the even adjustment, a negative count)."""
    names = list(FAMILIES)
    if total < len(names) + 1: raise ValueError(f"a partition needs at least {len(names) + 1} records (one per family, two for ambiguous), got {total}")
    base, extra = divmod(total, len(names))
    counts = {f: base for f in names}
    order = ["ambiguous"] + [f for f in names if f != "ambiguous"]
    for f in order[:extra]: counts[f] += 1
    if counts["ambiguous"] % 2:
        counts["ambiguous"] += 1; counts[order[-1]] -= 1
    return counts


def normalised(state):
    """The deduplication key of a state: text_digest of the text the model reads (kev.api.render)."""
    return text_digest(render(state))


class Checker:
    """Context admission under the pinned tokenizer: the long-state training context and the serving context."""

    def __init__(self):
        self.tok = load_tokenizer(*TOKENIZER)
        self.train_ctx = training_context(MAX_TRAIN_STATE)
        self.serve_ctx = {k: v for k, v in SERVING_CONTEXT.items() if k != "truncate"}

    def state_tokens(self, state):
        return len(self.tok.encode(render(state), add_special_tokens=False))

    def admit(self, record):
        rec = materialize(record)
        return fits(rec, self.tok, **self.train_ctx) and fits(rec, self.tok, **self.serve_ctx)


def item_records(family, t, item, seen, checker):
    """(records, None) for one generated item (an ambiguous twin pair is two records), every label computed from the stored
    facts; or (None, reason) when the item is rejected: a state already used, a long_policy state outside
    LONG_POLICY_TOKENS, or a record that does not fit the contexts (only checked with a checker)."""
    recs = []
    for it in item if isinstance(item, list) else [item]:
        facts = json.loads(json.dumps(it["facts"]))   # the stored facts alone must determine every label
        qs = {qid: dict(q) for qid, q in it["questions"].items()}
        for qid, lab in labels(family, facts, qs).items(): qs[qid]["label"] = lab
        key = normalised(it["state"])
        if key in seen or any(key == r["_meta"]["text_sha256"] for r in recs): return None, "duplicate_state"
        rec = {"state": it["state"], "questions": qs, "_meta": {"template": f"{family}/t{t}", "family": family, **it["meta"], "facts": facts, "text_sha256": key}}
        materialize(rec)   # request validation: a failure here is a generator bug, so let it raise
        if checker:
            n = checker.state_tokens(rec["state"])
            rec["_meta"]["state_tokens"] = n
            if family == "long_policy" and not LONG_POLICY_TOKENS[0] <= n <= LONG_POLICY_TOKENS[1]: return None, "length_rejected"
            if not checker.admit(rec): return None, "context_rejected"
        recs.append(rec)
    return recs, None


def build_split(split, n_total, seen, checker=None, report=None, seed=SEED):
    """One partition: every family's share, generated from its own RNG (seeded `{seed}:{family}:{split}`) with this split's
    templates. `seen` holds the normalised states already used and gains this partition's."""
    records = []
    report = report if report is not None else {}
    for family, count in family_counts(n_total).items():
        gen = FAMILIES[family][0]
        ctx = Ctx(f"{seed}:{family}:{split}")
        templates = TEMPLATE_SPLITS[split]
        made, attempts, stats = [], 0, Counter()
        while len(made) < count:
            attempts += 1
            if attempts > 60 * count + 200: raise RuntimeError(f"{family}/{split}: only {len(made)}/{count} after {attempts} draws ({dict(stats)})")
            t = templates[len(made) % len(templates)] if family != "ambiguous" else templates[(len(made) // 2) % len(templates)]
            try:
                item = gen(ctx, t)
            except ValueError:   # a draw with too few distinct options or a tie the solver refuses; draw again
                item = None
            if item is None: stats["draw_rejected"] += 1; continue
            if len(made) + (len(item) if isinstance(item, list) else 1) > count: stats["overflow"] += 1; continue
            recs, rejected = item_records(family, t, item, seen, checker)
            if rejected: stats[rejected] += 1; continue
            gid = f"hard-v1/{family}/{split}/g{len(made):05d}"
            for rec in recs:
                rid = f"hard-v1/{family}/{split}/{len(made):05d}"
                m = rec["_meta"]
                rec["_meta"] = {"id": rid, "source": f"hard_{family}", "group_id": gid if len(recs) > 1 else rid, "variant": "clean", "split": split, **m}
                seen.add(m["text_sha256"]); made.append(rec)
        stats["attempts"] = attempts
        report[family] = dict(stats)
        records.extend(made)
    random.Random(f"{seed}:{split}:order").shuffle(records)
    return records


def build(sizes=SIZES, checker=None):
    """{split: records} and the per-family generation report; test first, then development, then train, so a state seen in
    an evaluation partition is never regenerated for training."""
    seen, parts, report = set(), {}, {}
    for split in ("test", "development", "train"):
        report[split] = {}
        parts[split] = build_split(split, sizes[split], seen, checker, report[split])
    return parts, report


def quantiles(xs):
    xs = sorted(xs)
    if not xs: return None
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"min": xs[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": xs[-1], "mean": round(sum(xs) / len(xs), 1)}


def summary(records):
    by = defaultdict(lambda: {"records": 0, "questions": 0, "types": Counter(), "positions": Counter(), "noul_true": 0, "noul": 0,
                              "labels": Counter(), "subtypes": Counter(), "templates": Counter()})
    for r in records:
        s = by[r["_meta"]["family"]]
        s["records"] += 1; s["questions"] += len(r["questions"]); s["subtypes"][r["_meta"].get("subtype")] += 1; s["templates"][r["_meta"]["template"]] += 1
        for q in r["questions"].values():
            s["types"][q["type"]] += 1
            if q["type"] == "choice":
                keys = list(q["criteria"]); s["positions"][f"{keys.index(q['label'])}/{len(keys)}"] += 1
                if q["label"] in ABSTAIN_KEYS: s["labels"]["abstain_or_none"] += 1
            elif q["type"] == "noul":
                s["noul"] += 1; s["noul_true"] += bool(q["label"])
            else:
                s["labels"][f"level_{q['label']}"] += 1
    out = {}
    for fam, s in sorted(by.items()):
        out[fam] = {"records": s["records"], "questions": s["questions"], "question_types": dict(s["types"]),
                    "label_position": dict(sorted(s["positions"].items())), "noul_true_rate": round(s["noul_true"] / s["noul"], 3) if s["noul"] else None,
                    "other_labels": dict(s["labels"]), "subtypes": dict(sorted(s["subtypes"].items(), key=lambda kv: str(kv[0]))), "templates": dict(sorted(s["templates"].items()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=SIZES["train"])
    ap.add_argument("--development", type=int, default=SIZES["development"])
    ap.add_argument("--test", type=int, default=SIZES["test"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checker = Checker()
    parts, report = build({"train": a.train, "development": a.development, "test": a.test}, checker)
    files, tokens = {}, {}
    for split in ("train", "development", "test"):
        path = out / f"{split}.jsonl"
        write_jsonl(path, parts[split])
        size = path.stat().st_size
        files[path.name] = {"sha256": digest(path), "records": len(parts[split]), "questions": sum(len(r["questions"]) for r in parts[split]),
                            "bytes": size, "in_git": size <= GIT_LIMIT, "by_family": summary(parts[split])}
        tokens[split] = {fam: quantiles([r["_meta"]["state_tokens"] for r in parts[split] if r["_meta"]["family"] == fam]) for fam in FAMILIES}
        tokens[split]["all"] = quantiles([r["_meta"]["state_tokens"] for r in parts[split]])
    manifest = {
        "version": "hard-v1", "partitions": ["train", "development", "test"], "locked": ["test"], "seed": SEED,
        "seeds": {fam: {split: f"{SEED}:{fam}:{split}" for split in ("train", "development", "test")} for fam in FAMILIES},
        "families": list(FAMILIES), "templates": {split: list(t) for split, t in TEMPLATE_SPLITS.items()},
        "template_policy": "six surface templates per family; templates 0-3 train only, template 4 development only, template 5 test only",
        "trainable_sources": [f"hard_{fam}" for fam in FAMILIES], "eval_only_sources": [], "holdout_sources": [], "eval_only": False,
        "labels": "programmatic: every label is the family solver applied to _meta.facts (JSON round trip); no model or human judgement anywhere",
        "selection": "normalised exact-state deduplication across all partitions (test, then development, then train); long_policy states kept only at "
                     f"{LONG_POLICY_TOKENS[0]}-{LONG_POLICY_TOKENS[1]} tokens; every record admitted in both contexts below",
        "context": {**checker.train_ctx, "truncate": False}, "serving_context": SERVING_CONTEXT,
        "tokenizer": {"model": TOKENIZER[0], "revision": TOKENIZER[1]}, "base_revisions": {TOKENIZER[0]: TOKENIZER[1]},
        "state_tokens": tokens, "generation_report": report, "files": files,
        "code_sha256": {name: digest(ROOT / name) for name in CODE},
        "large_partitions": [n for n, f in files.items() if not f["in_git"]],
        "large_partitions_note": "partitions over 10 MB are gitignored; regenerate with this script (same bytes) or fetch from the kev-suites Hub mirror once uploaded",
        "overlap_screen": "overlap.json (scripts/screen_overlap.py): exact and word 8-gram overlap against the public JevBench items, counts only",
    }
    write_json(out / "manifest.json", manifest)
    for name, f in files.items():
        print(f"{name}: {f['records']} records, {f['questions']} questions, {f['bytes'] / 1e6:.1f} MB, sha256 {f['sha256'][:16]}")


if __name__ == "__main__":
    main()
