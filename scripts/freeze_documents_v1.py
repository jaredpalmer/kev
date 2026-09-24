"""documents-v1: apply the B2 label rules to the candidates and the LLM answers (scripts/label_documents_v1.py), write the
adjudication queue and, once adjudications exist, the frozen suite.

    uv run python scripts/freeze_documents_v1.py                           # agreement tables, writes the adjudication queue
    uv run python scripts/freeze_documents_v1.py --combine                 # two independent adjudications -> adjudications.jsonl
    uv run python scripts/freeze_documents_v1.py --spot-check              # the 50-question human sample from the final test split
    uv run python scripts/freeze_documents_v1.py --freeze evals/documents-v1 --min-agreement 47   # needs adjudications + spot-check reviews

Rules (PLAN_27b, B2 revised; at git tag research-archive-2026-09-24). Train: a question keeps its native label only if
both teachers chose it; otherwise the question is dropped (a record with no question left is dropped). Development and test: a question is verified if all three
judges chose the native label; every other question goes to the adjudication queue and is adjudicated twice,
independently (runs/documents-v1-work/adjudication/out and out2); it is decided only where both agree, otherwise dropped.
Unparsed judge answers count as disagreement.

--freeze runs every gate (adjudications complete, the spot-check reviews cover exactly the sample drawn from the frozen
test split, reviewer agreement >= --min-agreement) before it writes anything, and it never writes into --work: a failed
freeze leaves nothing on disk (a failure after the first partition is written, such as a failed private upload, removes
the suite directory again). documents-v1 and documents-v2 registered 47/50 as the bar.
"""
import argparse, json, random, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.suite import PRIVATE_DATASET, SERVING_CONTEXT, digest, read_json, read_jsonl, read_manifest, write_json, write_jsonl  # noqa: E402

WORK = Path("runs/documents-v1-work")
TEACHERS = ("deepseek/deepseek-v3.2", "alibaba/qwen3-235b-a22b-thinking")
JUDGES = ("anthropic/claude-opus-4.5", "openai/gpt-5", "google/gemini-3-flash")
SPOT_CHECK_SIZE, SPOT_CHECK_SEED = 50, "documents-v1-spot-check"   # documents-v2 reuses the seed


def answers(work, split, models):
    """Each model's cached answers by record id; failed calls (an `error` result, retried by the labeller) are skipped,
    which votes exactly as before: a failed call has no answers, so it never matched a label."""
    out = {}
    for m in models:
        path = work / "labels" / split / (m.replace("/", "__") + ".jsonl")
        out[m] = {r["id"]: r for r in read_jsonl(path) if "error" not in r} if path.exists() else {}
    return out


def vote(ans, model, rid, qid):
    r = ans[model].get(rid)
    a = (r or {}).get("answers") or {}
    return (a.get(qid) or {}).get("label"), (a.get(qid) or {}).get("reason", "")


def train_split(work):
    """Records whose questions both teachers labelled with the native label; None for a held-out-only suite (documents-v2)."""
    path = work / "candidates" / "train.jsonl"
    if not path.exists(): return None, {}
    recs, ans, kept, c = read_jsonl(path), answers(work, "train", TEACHERS), [], Counter()
    for r in recs:
        qs = {}
        for qid, q in r["questions"].items():
            votes = [vote(ans, m, r["_meta"]["id"], qid)[0] for m in TEACHERS]
            c["questions"] += 1; c["unlabelled"] += any(v is None for v in votes)
            if all(v == q["label"] for v in votes): qs[qid] = dict(q); c["kept"] += 1   # kept whole: kev.data.materialize needs "src"
        if qs: kept.append({**r, "questions": qs})
    return kept, dict(c)


def eval_split(work, split, adjudications):
    path = work / "candidates" / f"{split}.jsonl"
    if not path.exists(): return None, [], {}
    recs, ans, kept, queue, c = read_jsonl(path), answers(work, split, JUDGES), [], [], Counter()
    for r in recs:
        qs, rid = {}, r["_meta"]["id"]
        for qid, q in r["questions"].items():
            votes = {m: vote(ans, m, rid, qid) for m in JUDGES}
            c["questions"] += 1
            if all(v[0] == q["label"] for v in votes.values()):
                qs[qid] = dict(q); c["verified_unanimous"] += 1; continue
            item = f"{rid}#{qid}"
            queue.append({"id": item, "split": split, "document": r["state"], "source": "cfpb", "question": {"type": "choice", "instructions": q["instructions"], "options": q["criteria"]},
                          "proposed_label": q["label"], "label_origin": "native", "judges": [{"model": m, "label": v[0], "rationale": v[1]} for m, v in votes.items()], "adjudication": None})
            adj = adjudications.get(item)
            if adj is None: c["awaiting_adjudication"] += 1; continue
            if adj["verdict"] == "drop": c["dropped_agreed" if adj.get("agreed") else "dropped_disagreement"] += 1; continue
            label = q["label"] if adj["verdict"] == "accept" else adj["label"]
            if label not in q["criteria"]: raise ValueError(f"{item}: adjudicated label {label!r} is not an option of the question")
            qs[qid] = {**q, "label": label}; c["kept_native" if adj["verdict"] == "accept" else "relabelled"] += 1
        if qs: kept.append({**r, "questions": qs})
    return kept, queue, dict(c)


def read_verdicts(directory):
    """Adjudicator output shards: one JSON verdict per line, among whatever else the adjudicator wrote (prose, broken
    lines); the last well-formed verdict for an id wins."""
    out = {}
    for path in sorted(Path(directory).glob("shard-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("{"): continue
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            if r.get("verdict") in ("accept", "relabel", "drop"): out[r["id"]] = r
    return out


def combine(work):
    """Agreement of two independent adjudications decides an item; any disagreement (or a missing verdict) drops it."""
    queue = [r["id"] for r in read_jsonl(work / "adjudication_queue.jsonl")]
    a, b = read_verdicts(work / "adjudication" / "out"), read_verdicts(work / "adjudication" / "out2")
    rows, c = [], Counter()
    key = lambda v: (v["verdict"], v.get("label") if v["verdict"] == "relabel" else None)
    for item in queue:
        x, y = a.get(item), b.get(item)
        if x and y and key(x) == key(y):
            rows.append({"id": item, "verdict": x["verdict"], "label": x.get("label") if x["verdict"] == "relabel" else None, "reasons": [x.get("reason", ""), y.get("reason", "")], "agreed": True}); c["agreed_" + x["verdict"]] += 1
        else:
            rows.append({"id": item, "verdict": "drop", "label": None, "reasons": [(x or {}).get("reason", "missing"), (y or {}).get("reason", "missing")], "agreed": False,
                         "verdicts": [(x or {}).get("verdict"), (y or {}).get("verdict")]}); c["disagreed_dropped" if x and y else "missing_dropped"] += 1
    write_jsonl(work / "adjudications.jsonl", rows)
    decided = sum(v for k, v in c.items() if k.startswith("agreed"))
    print(dict(c), f"adjudicator agreement {decided}/{len(queue)} = {decided / len(queue):.3f}")
    return dict(c)


def spot_check_sample(test):
    """The protocol's human sample: SPOT_CHECK_SIZE test questions drawn with a fixed seed, in the tools/review input format."""
    items = [(r, qid) for r in test for qid in r["questions"]]
    spot = random.Random(SPOT_CHECK_SEED).sample(items, SPOT_CHECK_SIZE)
    return [{"id": f"{r['_meta']['id']}#{qid}", "document": r["state"], "source": f"cfpb · {r['_meta']['length_bucket']} · {r['_meta']['chars']} chars",
             "question": {"type": "choice", "instructions": r["questions"][qid]["instructions"], "options": r["questions"][qid]["criteria"]},
             "proposed_label": r["questions"][qid]["label"], "label_origin": "frozen", "judges": [], "adjudication": None} for r, qid in spot]


def spot_check_gate(work, test, min_agreement):
    """The reviews must cover exactly the sample drawn from this test split, with the same proposed labels, and accept at
    least min_agreement of them. Returns the manifest's spot_check entry; raises SystemExit otherwise."""
    reviews = read_jsonl(work / "spot_check_reviews.jsonl")
    sample = {r["id"]: r["proposed_label"] for r in spot_check_sample(test)}
    if {r["id"] for r in reviews} != set(sample) or any(r["proposed_label"] != sample[r["id"]] for r in reviews):
        raise SystemExit("spot-check reviews do not match the sample; nothing frozen")
    agreed = sum(r["verdict"] == "accept" for r in reviews)
    if agreed < min_agreement: raise SystemExit(f"spot check {agreed}/{len(sample)} is below the registered {min_agreement}/{len(sample)}; nothing frozen")
    return {"reviewer": "Jared Palmer", "tool": "tools/review", "sample": f"{SPOT_CHECK_SIZE} test questions, seed {SPOT_CHECK_SEED}", "agreement": f"{agreed}/{len(sample)}",
            "disagreements": [{"id": r["id"], "frozen": r["proposed_label"], "reviewer": r["label"], "verdict": r["verdict"]} for r in reviews if r["verdict"] != "accept"],
            "reviews_sha256": digest(work / "spot_check_reviews.jsonl")}


def evals_relative(out):
    """The suite's path under its evals/ root, which is also its path in the private mirror."""
    root = next((p for p in out.resolve().parents if p.name == "evals"), None)
    if root is None: raise SystemExit(f"--freeze {out} is not under an evals/ directory; the private mirror stores suites by their path under evals/")
    return out.resolve().relative_to(root)


def upload_private(out, names):
    """Push the frozen partitions to the private mirror at the suite's path under evals/; returns the commit to pin."""
    from huggingface_hub import CommitOperationAdd, HfApi
    rel = evals_relative(out)
    info = HfApi().create_commit(PRIVATE_DATASET, repo_type="dataset", commit_message=f"{rel}: frozen partitions",
                                 operations=[CommitOperationAdd(f"{rel}/{n}", str(out / n)) for n in names])
    return info.oid


def freeze(work, out, splits, report, *, version, min_agreement, private):
    """Every gate first, then the partitions, the optional private upload and the manifest; any failure after the suite
    directory exists removes it again."""
    if "test" not in splits: raise SystemExit(f"no test candidates in {work / 'candidates'}; nothing frozen")
    if any(r.get("awaiting_adjudication") for r in report.values()): raise SystemExit("adjudications missing; nothing frozen")
    if out.exists(): raise FileExistsError(out)
    if private: evals_relative(out)
    human = spot_check_gate(work, splits["test"], min_agreement)
    build = read_json(work / "candidates" / "build.json")
    out.mkdir(parents=True)
    try:
        write_suite(out, splits, report, human, build, version=version, private=private)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(f"frozen {out}")


def write_suite(out, splits, report, human, build, *, version, private):
    """The partitions, the optional private upload (the manifest pins its commit) and the manifest. A failure after the
    upload leaves that commit in the mirror, unreferenced by any manifest."""
    files = {}
    for split, recs in splits.items():
        write_jsonl(out / f"{split}.jsonl", recs)
        files[f"{split}.jsonl"] = {"sha256": digest(out / f"{split}.jsonl"), "records": len(recs), "questions": sum(len(r["questions"]) for r in recs),
                                   "by_length": dict(Counter(r["_meta"]["length_bucket"] for r in recs))}
    mirror = {"mirror": {"dataset": PRIVATE_DATASET, "revision": upload_private(out, list(files))}} if private else {}
    # A held-out-only suite (no train split, documents-v2) declares nothing trainable. documents-v2's manifest was frozen
    # before this rule with ["cfpb"]; frozen files never change, and it has no train partition to validate either way.
    trainable = ["cfpb"] if "train" in splits else []
    write_json(out / "manifest.json", {"version": version, "partitions": list(splits), "locked": ["test"], "files": files, **mirror,
                                       "source": build["repo"] + "@" + build["revision"], "rights": "consumer narratives published by the US CFPB with consent; the CFPB considers them public domain for FOIA purposes" if private else "US CFPB consumer complaint database, US government work (public domain)",
                                       "label_protocol": "PLAN_27b B2 revised: train = native label kept where both open-weight teachers agree; development/test = native label verified by a unanimous three-judge panel or adjudicated",
                                       "teachers": TEACHERS, "judges": JUDGES, "adjudicators": "two independent Devin subagents per item (Claude family); decided only on agreement", "label_report": report, "spot_check": human,
                                       "description": f"AI-adjudicated, human spot-checked ({human['agreement']} agreement)",
                                       "trainable_sources": trainable, "holdout_sources": [], "context": SERVING_CONTEXT,
                                       "base_revisions": read_manifest(ROOT / "evals/v7/decision-v7")["base_revisions"], "candidates_build": build})


def main():
    ap = argparse.ArgumentParser(description="without a mode flag: print the label report and write the adjudication queue")
    ap.add_argument("--freeze", default=""); ap.add_argument("--combine", action="store_true")
    ap.add_argument("--spot-check", action="store_true", help="write the 50-item human sample from the (final) test split")
    ap.add_argument("--min-agreement", type=int, help="required with --freeze: spot-check accepts needed out of 50 (documents-v1 and v2 registered 47)")
    ap.add_argument("--work", default=str(WORK)); ap.add_argument("--version", default="documents-v1")
    ap.add_argument("--private", action="store_true", help="partitions go only to kev.suite.PRIVATE_DATASET; the manifest pins that commit")
    a = ap.parse_args()
    if a.freeze and a.min_agreement is None: ap.error("--freeze needs --min-agreement (the registered spot-check bar)")
    work = Path(a.work)
    if a.combine: return combine(work)
    adj_path = work / "adjudications.jsonl"
    adjudications = {r["id"]: r for r in read_jsonl(adj_path)} if adj_path.exists() else {}
    train, tc = train_split(work)
    splits, report, queue = {"train": train}, {"train": tc}, []
    for split in ("development", "test"):
        splits[split], q, report[split] = eval_split(work, split, adjudications); queue += q
    splits = {k: v for k, v in splits.items() if v is not None}; report = {k: v for k, v in report.items() if k in splits}
    print(json.dumps(report, indent=1))
    if a.spot_check:
        if "test" not in splits: raise SystemExit(f"no test candidates in {work / 'candidates'}")
        if report["test"].get("awaiting_adjudication"): raise SystemExit("test adjudications missing")
        write_jsonl(work / "spot_check.jsonl", spot_check_sample(splits["test"]))
        return print(f"spot-check sample: {SPOT_CHECK_SIZE} of {sum(len(r['questions']) for r in splits['test'])} test questions -> {work / 'spot_check.jsonl'}")
    if a.freeze: return freeze(work, Path(a.freeze), splits, report, version=a.version, min_agreement=a.min_agreement, private=a.private)
    write_jsonl(work / "adjudication_queue.jsonl", queue)
    print(f"adjudication queue: {len(queue)} items -> {work / 'adjudication_queue.jsonl'}")


if __name__ == "__main__":
    main()
