import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from kev.composition import DEV_SHAPES, HELD_OUT_KEYS, TEST_SHAPES, TRAIN_SHAPES
from kev.contrastive import FAMILIES, generate
from kev.data import ALL_REPOS, ALL_SOURCES, EVAL_ONLY, REPOS, SOURCES, TRAINABLE, TRANSFER_REPOS, TRANSFER_SOURCES, build, dataset_ref, materialize, source_seed
from kev.model import MAX_BRANCH, SERVE_MAX_BRANCH, SERVE_MAX_PACKED, SERVE_MAX_STATE, fits, load_tokenizer, training_context

SPLITS = ("train", "calibration", "development", "test")
BASES = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base")
# the encoder limits every frozen record satisfies, as written into manifests ("context")
CONTEXT = {**training_context(), "truncate": False}
# what a manifest records for an eval-only suite frozen as published rather than admitted to the training context
SERVING_CONTEXT = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH, "max_packed": SERVE_MAX_PACKED, "truncate": False}
# Clean records are admitted with this many branch tokens to spare, so the variants that add an option (contrast_cases'
# none-of-these, training-time none/distractor augmentation) still encode under MAX_BRANCH.
ADMISSION_BRANCH_HEADROOM = 64
# Frozen suites are mirrored on the Hub. Manifests (with the sha256 of every partition) and the development/test
# partitions live in git; large training partitions are fetched from this dataset on first use and verified against
# the manifest, so the suite hash and every provenance record stay unchanged. A suite whose partitions must never be
# public (a held-out test set) names its own mirror in the manifest, {"mirror": {"dataset": ..., "revision": ...}},
# usually the private PRIVATE_DATASET; only its manifest is in git, which publishes the hashes but not the text.
SUITES_DATASET = "jaredpalmer/kev-suites"
PRIVATE_DATASET = "jaredpalmer/kev-private-evals"
SUITES_REVISION = "a88f56db5341397299137cb68775c2ea6e3f68cb"
# partitions larger than this stay out of git (gitignored; the manifest's sha256 still pins them)
GIT_LIMIT = 10 * 1024 * 1024
# the pinned tokenizer suites built for the Qwen3.5 family are admitted and length-counted under (hard-v1, devtools-v1, long states)
ADMISSION_TOKENIZER = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
# programmatic policy sources (kev.study_v3 / kev.contrastive); the trainer's mix ablations treat them as one group
SYNTHETIC_SOURCES = ("legacy_policy", "compositional", "contrastive")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record_digest(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def normalise_text(text):
    """Casefolded, whitespace runs collapsed to one space: the text two states are compared on for exact deduplication."""
    return " ".join(text.casefold().split())


def text_digest(text):
    """sha256 of normalise_text(text): the `text_sha256` of suite builders (kev.data computes the same inline; it cannot
    import this module). scripts/screen_overlap.py tokenises differently on purpose (words only, for n-gram overlap)."""
    return hashlib.sha256(normalise_text(text).encode()).hexdigest()


# Every JSON/JSONL file this repo writes is UTF-8 with LF line endings, whatever the platform's locale says (issue #12:
# frozen partitions are sha256-checked byte for byte, and they contain non-ASCII text). Read them the same way.
ENCODING = "utf-8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding=ENCODING))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding=ENCODING)


def read_jsonl(path):
    # split on "\n" only: str.splitlines() also breaks on U+2028, U+2029 and U+0085, which write_jsonl leaves unescaped
    return [json.loads(line) for line in Path(path).read_text(encoding=ENCODING).split("\n") if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding=ENCODING)


def semantic_hash(r):
    """State hash that ignores sentence order for policy/case records (contrastive, compositional), so the same case
    told in a different order counts as the same state; record_digest of the state otherwise."""
    state = r["state"]
    if isinstance(state, dict) and "policy" in state and "case" in state:
        state = {"policy": state["policy"], "sentences": sorted(s.rstrip(".") for s in state["case"].split(". "))}
    return record_digest(state)


def validate_training(records, manifest):
    """Every training record must come from a source the manifest declares trainable, never from an eval-only or held-out
    one, and compositional records must not use a held-out rule structure."""
    allowed = set(manifest.get("trainable_sources", []))
    forbidden = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) | set(manifest.get("holdout_sources", []))
    for r in records:
        m = r["_meta"]
        if m["source"] in forbidden or (allowed and m["source"] not in allowed):
            raise ValueError(f"eval-only or undeclared training source: {m['source']}")
        if m["source"] == "compositional":
            held_shape = m["family"] in DEV_SHAPES + TEST_SHAPES
            held_structure = m.get("structure") in HELD_OUT_KEYS
            if held_shape or held_structure or (m["family"] not in TRAIN_SHAPES and not m["family"].startswith("rand:")):
                raise ValueError("held-out compositional structure in training")
    if not records:
        raise ValueError("empty training partition")


def read_manifest(directory):
    return read_json(Path(directory) / "manifest.json")


def load_split(directory, split, allow_test=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test; never use it for search")
    directory = Path(directory)
    manifest = read_manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists():
        fetch_partition(directory, path.name)
    if digest(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = read_jsonl(path)
    if len(records) != manifest["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records


def fetch_partition(directory, filename):
    """Download one partition of a frozen suite from its Hub mirror into place: the manifest's own "mirror" if it names
    one, else SUITES_DATASET@SUITES_REVISION. The caller verifies the sha256."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    directory = Path(directory).resolve()
    evals_root = next((p for p in directory.parents if p.name == "evals"), None)
    if evals_root is None:
        raise FileNotFoundError(f"{directory / filename} is missing and is not under an evals/ tree")
    relative = directory.relative_to(evals_root) / filename
    mirror = read_manifest(directory).get("mirror")
    repo, revision = (mirror["dataset"], mirror["revision"]) if mirror else (SUITES_DATASET, SUITES_REVISION)   # a named mirror pins its own revision
    try:
        cached = hf_hub_download(repo, str(relative), repo_type="dataset", revision=revision)
    except (RepositoryNotFoundError, GatedRepoError) as e:   # a private mirror answers "not found" to anyone without access
        raise PermissionError(f"{relative} is only in {repo}, which is missing or private to this account; `hf auth login` "
                              "(or HF_TOKEN) with access to it, or ask for it") from e
    shutil.copyfile(cached, directory / filename)
    print(f"fetched {relative} from {repo}@{revision[:10]}", flush=True)


def case_copy(record, variant):
    result = copy.deepcopy(record)
    result["_meta"]["group_id"] = record["_meta"].get("group_id", record["_meta"]["id"])   # bootstrap unit (pair for contrastive)
    result["_meta"]["parent_id"] = record["_meta"]["id"]                                   # the clean record this variant perturbs
    result["_meta"]["id"] += "/" + variant
    result["_meta"]["variant"] = variant
    return result


def contrast_cases(record, seed=0):
    candidates = [(qid, q) for qid, q in record["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not candidates:
        return []
    qid, q = candidates[0]
    nk = "none_of_these"
    if nk in q["criteria"]:
        raise ValueError("reserved contrast option collision")
    out = []
    for variant in ("none_present", "none_absent"):
        r = case_copy(record, variant)
        r["questions"] = {qid: copy.deepcopy(q)}
        rq = r["questions"][qid]
        rq["criteria"][nk] = "None of these options describes the answer"
        if variant == "none_absent":
            rq["criteria"].pop(rq["label"])
            rq["label"] = nk
        keys = list(rq["criteria"])
        random.Random(source_seed(seed, record["_meta"]["id"])).shuffle(keys)
        rq["criteria"] = {k: rq["criteria"][k] for k in keys}
        r["_meta"]["none_key"] = nk
        out.append(r)
    r = case_copy(record, "permuted")
    rng = random.Random(source_seed(seed, record["_meta"]["id"]))
    for q in r["questions"].values():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            rng.shuffle(keys)
            q["criteria"] = {k: q["criteria"][k] for k in keys}
    out.append(r)
    return out


def select_unique(records, count, seen, tokenizers, report):
    selected = []
    for record in sorted(records, key=lambda r: r["_meta"]["row_sha256"]):
        report["considered"] += 1
        key = record["_meta"]["text_sha256"]
        if key in seen:
            report["duplicate_state"] += 1
            continue
        rec = materialize(record)   # a record that fails request validation is a converter bug: let it abort the freeze
        if not fits(rec, *tokenizers, max_branch=MAX_BRANCH - ADMISSION_BRANCH_HEADROOM):
            report["context_rejected"] += 1
            continue
        record["_meta"].update(group_id=record["_meta"]["id"], variant="clean")
        seen.add(key)
        selected.append(record)
        report["accepted"] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)}/{count} records fit the common context policy")


def training_state_hashes(suite_dir):
    """Normalized state hashes of a frozen suite's training + calibration partitions, for exact-match contamination checks."""
    hashes = set()
    for split in ("train", "calibration"):
        for record in load_split(suite_dir, split):
            hashes.add(record["_meta"]["text_sha256"])
    return hashes


def freeze(directory, train=300, calibration=40, development=80, test=80, seed=20260918, holdout=("mnli", "sst5"),
           sources=None, repos=None, exclude_states_from=None, contrastive_pairs=0, contrastive_holdout_families=(), contrastive_eval_pairs=40):
    from huggingface_hub import HfApi

    sources = SOURCES if sources is None else sources
    repos = REPOS if repos is None else repos
    eval_only = set(holdout) >= set(sources)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite frozen suite {directory}")
    hub = HfApi()
    revisions = {repo: hub.dataset_info(*dataset_ref(repo)[:1], revision=dataset_ref(repo)[1]).sha for repo in set(repos.values())}
    base_revisions = {base: hub.model_info(base).sha for base in BASES}
    tokenizers = [load_tokenizer(base, revision=revision) for base, revision in base_revisions.items()]
    trainable_here = [x for x in sources if x not in holdout]
    if set(trainable_here) & set(EVAL_ONLY):
        raise ValueError(f"eval-only sources cannot be trainable: {sorted(set(trainable_here) & set(EVAL_ONLY))}")
    manifest = {
        "version": 2, "seed": seed, "holdout_sources": list(holdout), "eval_only": eval_only,
        "trainable_sources": trainable_here, "eval_only_sources": [x for x in sources if x in holdout],
        "policy": {"trainable": list(TRAINABLE), "eval_only": list(EVAL_ONLY)},
        "dataset_revisions": revisions, "base_revisions": base_revisions,
        "context": {**CONTEXT, "admission_branch_headroom": ADMISSION_BRANCH_HEADROOM},
        "selection": "Normalized exact-state deduplication across partitions; common tokenizer context admission; no fuzzy decontamination or pretraining-contamination claim.",
        "legacy_checkpoints": "Training/calibration overlap for pre-manifest checkpoints is unknown; exploratory only.",
        "objective": "Negative macro-average clean development NLL, equal weight per task; raw probabilities.",
        "excluded_training_states_from": str(exclude_states_from) if exclude_states_from else None,
        "files": {}, "admission": {},
    }
    partitions = {split: [] for split in SPLITS}
    seen = set(training_state_hashes(exclude_states_from)) if exclude_states_from else set()
    manifest["excluded_training_state_hashes"] = len(seen)
    for source in sources:
        report = Counter()
        test_pool = build(max(3 * (development + test), 600), "test", seed, only=[source], revisions=revisions, sources=sources, repos=repos)
        chosen = select_unique(test_pool, development + test, seen, tokenizers, report)
        partitions["development"].extend(chosen[:development])
        partitions["test"].extend(chosen[development:])
        if source not in holdout:
            train_pool = build(max(3 * (train + calibration), 800), "train", seed, only=[source], revisions=revisions, sources=sources, repos=repos)
            chosen = select_unique(train_pool, train + calibration, seen, tokenizers, report)
            partitions["calibration"].extend(chosen[:calibration])
            partitions["train"].extend(chosen[calibration:])
        manifest["admission"][source] = dict(report)
        print(f"froze {source}: {dict(report)}", flush=True)
    if contrastive_pairs:
        families = list(FAMILIES)
        held = list(contrastive_holdout_families)
        unknown = set(held) - set(families)
        if unknown: raise ValueError(f"unknown contrastive families: {sorted(unknown)}")
        trainable_fams = [f for f in families if f not in held]
        # training/calibration pairs and development/test pairs come from disjoint seeds within trainable families;
        # held-out families appear only in development/test (never trained anywhere)
        train_recs, rep_train = generate(contrastive_pairs, seed=f"{seed}-train", families=trainable_fams) if trainable_fams and not eval_only else ([], {})
        # eval-only suites carry the held-out families only; trainable suites carry their trainable families only
        eval_families = (held or families) if eval_only else trainable_fams
        eval_recs, rep_eval = generate(contrastive_eval_pairs, seed=f"{seed}-eval", families=eval_families)
        for r in train_recs + eval_recs:
            r["_meta"].update(group_id=r["_meta"]["family_id"], variant="clean")   # siblings are one bootstrap unit
            if r["_meta"]["text_sha256"] in seen: raise ValueError("contrastive state collides with an existing state")
        seen.update(r["_meta"]["text_sha256"] for r in train_recs + eval_recs)
        n_cal = 2 * max(len(train_recs) // 20, 1) if train_recs else 0
        partitions["calibration"].extend(train_recs[:n_cal]); partitions["train"].extend(train_recs[n_cal:])
        # stratified by family: alternate pairs (siblings adjacent) between development and test
        for i in range(0, len(eval_recs), 2):
            partitions["development" if (i // 2) % 2 == 0 else "test"].extend(eval_recs[i : i + 2])
        manifest["contrastive"] = {"pairs_per_family_train": contrastive_pairs, "pairs_per_family_eval": contrastive_eval_pairs,
                                   "eval_families": eval_families, "trainable_families": trainable_fams, "eval_only_families": held,
                                   "train_report": rep_train, "eval_report": rep_eval,
                                   "note": "labels from rule evaluators; ablation + invariance checks passed for every kept pair; no LLM"}
        if trainable_fams and not eval_only: manifest["trainable_sources"].append("contrastive")
        else: manifest["eval_only_sources"].append("contrastive")
        print(f"contrastive: {len(train_recs)} train/cal records, {len(eval_recs)} dev/test records; held-out families {held}", flush=True)
    for split in ("development", "test"):
        extras = []
        per_source = Counter()
        for record in partitions[split]:
            source = record["_meta"]["source"]
            if per_source[source] < 12:
                variants = contrast_cases(record, seed)
                for variant in variants:
                    if not fits(materialize(variant), *tokenizers):
                        raise ValueError(f"variant of {record['_meta']['id']} exceeds the training context")
                extras.extend(variants)
                per_source[source] += bool(variants)
        partitions[split].extend(extras)
    directory.mkdir(parents=True)
    for split, records in partitions.items():
        path = directory / f"{split}.jsonl"
        write_jsonl(path, records)
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records),
                                        "questions": sum(len(r["questions"]) for r in records)}
    manifest["code_hashes"] = {name: digest(Path(__file__).parent / name) for name in ("data.py", "api.py", "model.py", "suite.py")}
    write_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest["files"], indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--calibration", type=int, default=40)
    ap.add_argument("--development", type=int, default=80)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--transfer", action="store_true", help="eval-only suite from the eight transfer-v1 sources (no train/calibration partitions)")
    ap.add_argument("--sources", help="comma-separated source names (any of ALL_SOURCES); trainable ones get train/calibration partitions")
    ap.add_argument("--holdout", help="comma-separated sources kept eval-only within --sources (eval-only policy sources are always held out)")
    ap.add_argument("--exclude-states-from", help="frozen suite whose train+calibration states must not appear here")
    ap.add_argument("--contrastive-pairs", type=int, default=0, help="programmatic contrastive pairs per trainable family for training (0 = none)")
    ap.add_argument("--contrastive-holdout", default="", help="comma-separated contrastive families kept eval-only")
    a = ap.parse_args()
    cfam = tuple(x for x in a.contrastive_holdout.split(",") if x)
    if min(a.development, a.test) < 1:
        ap.error("split sizes must be positive")
    if a.transfer:
        freeze(a.out, 0, 0, a.development, a.test, a.seed, holdout=tuple(TRANSFER_SOURCES), sources=TRANSFER_SOURCES,
               repos=TRANSFER_REPOS, exclude_states_from=a.exclude_states_from, contrastive_pairs=a.contrastive_pairs, contrastive_holdout_families=cfam)
    elif a.sources:
        names = [x for x in a.sources.split(",") if x]
        unknown = set(names) - set(ALL_SOURCES)
        if unknown: ap.error(f"unknown sources: {sorted(unknown)}")
        holdout = tuple(x for x in names if x in EVAL_ONLY or x in (a.holdout or "").split(","))
        if any(x not in holdout for x in names) and min(a.train, a.calibration) < 1:
            ap.error("train and calibration sizes must be positive when a trainable source is included")
        freeze(a.out, a.train, a.calibration, a.development, a.test, a.seed, holdout=holdout,
               sources={k: ALL_SOURCES[k] for k in names}, repos={k: ALL_REPOS[k] for k in names}, exclude_states_from=a.exclude_states_from,
               contrastive_pairs=a.contrastive_pairs, contrastive_holdout_families=cfam)
    else:
        if min(a.train, a.calibration) < 1: ap.error("split sizes must be positive")
        freeze(a.out, a.train, a.calibration, a.development, a.test, a.seed, exclude_states_from=a.exclude_states_from)


if __name__ == "__main__":
    main()
