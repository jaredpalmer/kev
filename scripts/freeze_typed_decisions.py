"""Freeze LocalLLaMA/typed-decisions as an eval-only Kev suite for zero-shot comparison.

    uv run python scripts/freeze_typed_decisions.py --revision <sha> --out evals/external/typed-decisions-v1

Source: huggingface.co/datasets/LocalLLaMA/typed-decisions (Apache-2.0). Each row is one shared state with five
typed questions (Noul/Choice/Score); the row's ``state`` and ``questions`` JSON are exactly a
``POST /v1/systemone`` body, so a Kev record can carry them verbatim. Gold is the mean of three teacher samples, so
every question has a full distribution; we keep the distribution, the argmax label and the expected score in
``_meta`` and use the label only for the accuracy-shaped metrics.

Only the public ``test`` split is frozen here (400 cases, 2000 decisions). This suite is eval-only: it is never
trained on, so a score measures zero-shot agreement with the teacher, not correctness.

Reference points from the dataset card (all on this test split): Uniform 0.308, Prior 0.470, MiniLM-L6 0.587,
ModernBERT-base 0.646, factor ceiling 0.704, Jev 1.13.0 generalist 0.727, teacher self-agreement 0.735.
"""
import argparse
import json
from pathlib import Path

from kev.suite import digest, record_digest, write_json

REPO = "LocalLLaMA/typed-decisions"
SPLIT_FILE = "all/test-00000-of-00001.parquet"
# Kev's training cap; states above this cannot be served exactly and are counted, not truncated (see score script).
MAX_STATE_CHARS = 384 * 4


def load_rows(revision):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, SPLIT_FILE, repo_type="dataset", revision=revision)
    return pd.read_parquet(path)


def convert(row):
    """-> one Kev eval record carrying the SystemOne state/questions verbatim plus full per-question gold."""
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    state_json = row["state"]
    state = json.loads(state_json)
    qmeta = {}
    for qid, q in questions.items():
        g = gold[qid]
        entry = {"type": q["type"], "probabilities": g["probabilities"], "label": str(g["label"])}
        if q["type"] == "noul":
            entry["probability_true"] = g.get("noul")
        elif q["type"] == "score":
            entry["score"] = g.get("score")
        qmeta[qid] = entry
    return {"state": state, "questions": questions,
            "_meta": {"id": row["id"], "source": f"typed_{row['workflow']}", "workflow": row["workflow"],
                      "variant": "clean", "split": row["split"], "repo": REPO, "split_file": SPLIT_FILE,
                      "state_chars": len(state_json), "overlong": len(state_json) > MAX_STATE_CHARS,
                      "gold": qmeta, "row_sha256": record_digest({"state": state, "questions": questions})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True, help="pinned dataset revision (commit sha)")
    ap.add_argument("--out", default="evals/external/typed-decisions-v1")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    frame = load_rows(a.revision)
    records = [convert(row) for _, row in frame.iterrows()]
    records.sort(key=lambda r: r["_meta"]["id"])
    overlong = [r["_meta"]["id"] for r in records if r["_meta"]["overlong"]]
    by_workflow = {}
    for r in records:
        by_workflow.setdefault(r["_meta"]["workflow"], 0)
        by_workflow[r["_meta"]["workflow"]] += 1
    out.mkdir(parents=True)
    files = {}
    for name, rows in (("development.jsonl", records), ("train.jsonl", []), ("calibration.jsonl", []), ("test.jsonl", [])):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[name] = {"sha256": digest(out / name), "records": len(rows)}
    write_json(out / "manifest.json", {
        "version": 1,
        "external": {"repo": f"https://huggingface.co/datasets/{REPO}", "revision": a.revision, "license": "apache-2.0",
                     "file": SPLIT_FILE, "file_sha256": digest_repo_file(a.revision)},
        "base_revisions": {}, "dataset_revisions": {}, "holdout_sources": [],
        "trainable_sources": [], "eval_only_sources": sorted(by_workflow), "files": files, "eval_only": True,
        "context": {"max_state": 384, "max_branch": 1024, "max_packed": 2048, "truncate": False},
        "protocol": {
            "note": "Zero-shot only: this suite is never trained on. Gold is the mean of three teacher-sample "
                    "distributions; a score measures agreement with that teacher, not correctness.",
            "state": "raw JSON from the dataset, passed through verbatim (SystemOne shape)",
            "overlong_records": overlong,
            "overlong_policy": "records whose state exceeds the 384-token serving cap are counted as wrong, never truncated",
            "reference_points": {"uniform": 0.308, "prior": 0.470, "minilm_l6": 0.587, "modernbert_base": 0.646,
                                 "factor_ceiling": 0.704, "jev_1_13_0_generalist": 0.727, "teacher_self_agreement": 0.735}}})
    print(json.dumps({"records": len(records), "by_workflow": by_workflow, "overlong": overlong}, indent=2))


def digest_repo_file(revision):
    from huggingface_hub import hf_hub_download
    return digest(hf_hub_download(REPO, SPLIT_FILE, repo_type="dataset", revision=revision))


if __name__ == "__main__":
    main()
