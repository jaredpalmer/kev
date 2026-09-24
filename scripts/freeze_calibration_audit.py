import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kev.composition import DEV_SHAPES, check_group, generate as compose
from kev.contrastive import generate as contrastive
from kev.data import ALL_REPOS, ALL_SOURCES, build, materialize
from kev.model import fits, load_tokenizer
from kev.suite import digest, load_split, read_jsonl, read_manifest, record_digest, semantic_hash, SPLITS, text_digest, validate_training, write_json, write_jsonl
from kev.transfer_v9 import QWEN35, unknowable

PUBLIC = ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq")


def state_fingerprint(record):
    state = record["state"]
    if isinstance(state, dict) and "policy" in state and "case" in state:
        return semantic_hash(record)

    def strings(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [s for v in value.values() for s in strings(v)]
        if isinstance(value, list):
            return [s for v in value for s in strings(v)]
        return []

    leaves = strings(state)
    text = max(leaves, key=len) if leaves else json.dumps(state, sort_keys=True)
    return text_digest(text)


def reserve_existing(evals):
    states, origins, files = set(), set(), {}
    for path in sorted(evals.rglob("*.jsonl")):
        files[str(path.relative_to(ROOT))] = digest(path)
        for r in read_jsonl(path):
            if "state" not in r or "questions" not in r:
                continue
            states.add(state_fingerprint(r))
            m = r.get("_meta", {})
            if m.get("repo") and m.get("row") is not None:
                origins.add((m["repo"], m.get("split"), str(m["row"])))
    return states, origins, files


def new_public(source, count, seed, revisions, states, origins, tokenizers):
    rows = build(3000, "test", seed, sources={source: ALL_SOURCES[source]}, repos={source: ALL_REPOS[source]}, revisions=revisions)
    kept, admission = [], Counter()
    for r in rows:
        admission["considered"] += 1
        m = r["_meta"]
        fingerprint = state_fingerprint(r)
        origin = (m["repo"], m["split"], str(m["row"]))
        if fingerprint in states or origin in origins:
            admission["reserved_overlap"] += 1
            continue
        if not fits(materialize(r), *tokenizers):
            admission["context_rejected"] += 1
            continue
        r["_meta"].update(variant="clean", group_id=f"{source}/{fingerprint}")
        kept.append(r)
        states.add(fingerprint)
        origins.add(origin)
        if len(kept) == count:
            return kept, dict(admission)
    raise ValueError(f"not enough fresh {source} records: {len(kept)}/{count}; {dict(admission)}")


def fresh_generated(seed, states, tokenizers, composition_groups, policy_pairs, unknown_pairs):
    candidates = compose(composition_groups * 3, seed + ":rules", shapes=DEV_SHAPES, styles=(1,), source="composition_holdout")
    grouped = defaultdict(list)
    for r in candidates:
        grouped[r["_meta"]["group_id"]].append(r)
    selected, counts = [], Counter()
    for group in grouped.values():
        family = group[0]["_meta"]["family"]
        check_group(group)
        keys = {state_fingerprint(r) for r in group}
        if counts[family] >= composition_groups or keys & states or not all(fits(materialize(r), *tokenizers) for r in group):
            continue
        selected.extend(group)
        states.update(keys)
        counts[family] += 1
    if any(counts[f] != composition_groups for f in DEV_SHAPES):
        raise ValueError("not enough fresh compositional groups")
    records, _ = contrastive(policy_pairs * 3, seed + ":policy", families=("deadline", "authorization"))
    counts = Counter()
    for a, b in zip(records[::2], records[1::2]):
        family = a["_meta"]["family"]
        keys = {state_fingerprint(a), state_fingerprint(b)}
        if counts[family] >= policy_pairs or keys & states or not all(fits(materialize(r), *tokenizers) for r in (a, b)):
            continue
        for r in (a, b):
            r["_meta"].update(source="legacy_holdout", variant="clean", group_id=r["_meta"]["pair_id"])
        selected.extend((a, b))
        states.update(keys)
        counts[family] += 1
    if any(counts[f] != policy_pairs for f in ("deadline", "authorization")):
        raise ValueError("not enough fresh policy pairs")
    records = unknowable(unknown_pairs * 4, seed + ":unknown")
    groups = defaultdict(list)
    for r in records:
        key = r["_meta"]["group_id"].split("/", 1)[1]
        groups[key].append(r)
    counts = Counter()
    for group in groups.values():
        family = group[0]["_meta"]["family"]
        keys = {state_fingerprint(r) for r in group}
        if counts[family] >= unknown_pairs or keys & states or not all(fits(materialize(r), *tokenizers) for r in group):
            continue
        selected.extend(group)
        states.update(keys)
        counts[family] += 1
    if len(counts) != 11 or any(n != unknown_pairs for n in counts.values()):
        raise ValueError("not enough fresh unknown/control groups")
    return selected


def freeze_suite(out, partitions, metadata):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {**metadata, "files": {}}
    for split in SPLITS:
        rows = partitions.get(split, [])
        path = out / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(rows), "questions": sum(len(r["questions"]) for r in rows)}
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/round3")
    ap.add_argument("--seed", type=int, default=2026092103)
    ap.add_argument("--tag", default="r3", help="suite names: decision-<tag>, transfer-<tag>")
    ap.add_argument("--panel_only", action="store_true", help="freeze only transfer-<tag> (a fresh evaluation panel); skip the decision-<tag> training suite")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise FileExistsError(out)
    parents = ROOT / "evals/v7/decision-v7", ROOT / "evals/v4/transfer-v4"
    pm = read_manifest(parents[0])
    revisions = read_manifest(ROOT / "evals/transfer-v2")["dataset_revisions"]
    tokenizers = [load_tokenizer(base, revision=revision) for base, revision in QWEN35.items() if "0.8" not in base]
    states, origins, reserved_files = reserve_existing(ROOT / "evals")
    reservation_hash = hashlib.sha256("\n".join(sorted(states)).encode()).hexdigest()
    transfer = {"train": [], "development": load_split(parents[1], "development")}
    admission = {}
    public = {}
    for source in PUBLIC:
        public[source], admission[source] = new_public(source, 200, a.seed, revisions, states, origins, tokenizers)
        print(source, "fresh records", len(public[source]), "admission", admission[source], flush=True)
    for split, start, stop, cg, pp, up in (("calibration", 0, 60, 4, 10, 3), ("test", 60, 200, 10, 20, 5)):
        transfer[split] = [r for source in PUBLIC for r in public[source][start:stop]]
        transfer[split].extend(fresh_generated(f"audit-{a.seed}-{split}", states, tokenizers, cg, pp, up))
        random.Random(f"{a.seed}:{split}").shuffle(transfer[split])
    delta = read_jsonl(ROOT / "evals/night2/dates_unknowable.jsonl")
    replay = random.Random(a.seed).sample(load_split(parents[0], "train"), 2000)
    training = delta + replay
    if not all(fits(materialize(r), *tokenizers) for r in training):
        raise ValueError("training corpus contains overlong records")
    for r in training:
        r["_meta"].setdefault("group_id", r["_meta"]["id"])
        r["_meta"].setdefault("variant", "clean")
    decision = {"train": training, "calibration": load_split(parents[0], "calibration"), "development": load_split(parents[0], "development"), "test": []}
    allowed = sorted({r["_meta"]["source"] for r in training})
    validate_training(training, {**pm, "trainable_sources": allowed})
    common = {"version": 4, "seed": a.seed, "base_revisions": QWEN35,
              "context": pm["context"], "code_sha256": digest(Path(__file__)),
              "evaluation_policy": "Old development items for search/regression only. Fresh test predictions remain sealed until a candidate is selected in writing.",
              "pretraining_overlap": "unknown; disjointness is from recorded local training/evaluation states, not foundation-model pretraining"}
    train_manifest = {"files": "not frozen (--panel_only)"} if a.panel_only else freeze_suite(out / f"decision-{a.tag}", decision, {**common, "trainable_sources": allowed,
                                "eval_only_sources": pm["eval_only_sources"], "holdout_sources": [],
                                "dataset_revisions": pm["dataset_revisions"], "training_parent_sha256": digest(parents[0] / "manifest.json"),
                                "delta_sha256": digest(ROOT / "evals/night2/dates_unknowable.jsonl"),
                                "replay_ids": [r["_meta"]["id"] for r in replay]})
    test_manifest = freeze_suite(out / f"transfer-{a.tag}", transfer, {**common, "trainable_sources": [], "eval_only": True,
                               "eval_only_sources": list(PUBLIC) + ["legacy_holdout", "composition_holdout", "unknowable", "unknowable_control"],
                               "holdout_sources": list(PUBLIC) + ["legacy_holdout", "composition_holdout"],
                               "dataset_revisions": revisions, "reservation_sha256": reservation_hash,
                               "reserved_files": reserved_files, "admission": admission,
                               "calibration_role": "Select a deployable confidence threshold after temperature fitting on decision-r3 calibration. Never train model weights here.",
                               "test_role": "Fresh final audit; no candidate selection or hyperparameter fitting from its outcomes."})
    test_states = {state_fingerprint(r) for r in transfer["test"]}
    other_states = {state_fingerprint(r) for split in ("train", "calibration", "development") for r in decision[split] + transfer.get(split, [])}
    if test_states & other_states:
        raise ValueError("final audit overlaps training, calibration or development")
    print(json.dumps({"decision": train_manifest["files"], "transfer": test_manifest["files"], "final_test_overlap": 0}, indent=2))


if __name__ == "__main__":
    main()
