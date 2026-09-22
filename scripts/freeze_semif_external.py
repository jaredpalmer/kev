"""Freeze SemIf's pinned third-party selections (WANLI-256, TypeSafe public 102) as eval-only Kev suites.

    python benchmarks/build_wanli.py    --source SRC/wanli-test.jsonl --selection benchmarks/manifests/source-selection.jsonl --output OUT/wanli256.jsonl
    python benchmarks/build_typesafe.py --source-dir SRC            --selection benchmarks/manifests/source-selection.jsonl --output OUT/typesafe102.jsonl
    uv run python scripts/freeze_semif_external.py --semif /tmp/semif --rows OUT/wanli256.jsonl    --out evals/external/wanli-v1
    uv run python scripts/freeze_semif_external.py --semif /tmp/semif --rows OUT/typesafe102.jsonl --out evals/external/typesafe-v1

The rows are SemIf's build_*.py outputs (github.com/TheoLeeCJ/SemIf, MIT): the same 256 WANLI test pairs (CC-BY-4.0, HF
revision pinned in each row) and the same 102 evals.typesafe.ai cases (hash-verified `*-cases.js` snapshots) that SemIf
scored, so Kev, live Jev and SemIf's published numbers are on identical items. WANLI is a 3-way choice over
supported / insufficient / contradicted in SemIf's per-row option order. TypeSafe rows keep their primitive (noul or
choice) and carry the reference distribution and the published TypeSafe/Jev answer in `_meta` so `scripts/compare_typesafe.py`
can report equal-case modal agreement and total-variation distance the way SemIf does. TypeSafe documents are long:
17 of 102 fit the 384-token training context and 89 the 8,192-token serving context; the manifest records the latter.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from kev.serve import INFER_MAX_BRANCH, INFER_MAX_STATE
from kev.suite import CONTEXT, digest, read_jsonl, record_digest, write_json, write_jsonl

SOURCES = {
    "wanli": {"source": "wanli", "rows": 256, "context": CONTEXT,
              "protocol": "SemIf reports balanced accuracy on the 256 (direct Qwen3.5-4B logits 0.637); we report accuracy plus everything kev.benchmark reports"},
    "typesafe": {"source": "typesafe", "rows": 102, "context": {**CONTEXT, "max_state": INFER_MAX_STATE, "max_branch": INFER_MAX_BRANCH, "max_packed": INFER_MAX_STATE + INFER_MAX_BRANCH},
                 "protocol": "SemIf reports equal-case modal agreement with the reference argmax (direct 0.845, published Jev 0.883) and total-variation distance to the "
                             "reference distribution (0.177, 0.127) over the 20 cases; scripts/compare_typesafe.py computes both from a kev.benchmark output. "
                             "Records over the serving context are rejected by kev.benchmark; the headline is over answered rows with the rejected count stated, and the all-rows figure (rejected = wrong) alongside"},
}


def question(row, kind):
    ids = [o["id"] for o in row["options"]]
    desc = {o["id"]: o["description"].removeprefix(o["id"] + ": ") for o in row["options"]}
    src = "wanli_nli" if kind == "wanli" else f"typesafe_{row['provenance']['workflow']}"
    q = {"type": "choice", "instructions": row["question"], "criteria": desc, "label": ids[row["label"]], "src": src}
    if kind == "typesafe" and row["primitive"] == "noul":
        assert ids == ["true", "false"], row["id"]
        q.update(type="noul", label=ids[row["label"]] == "true")
    return q


def convert(row, kind):
    source = SOURCES[kind]["source"]
    state = json.loads(row["state"]) if kind == "typesafe" else row["state"]
    rec = {"state": state, "questions": {"decision": question(row, kind)},
           "_meta": {"row": row["id"], "source": source, "repo": "TheoLeeCJ/SemIf", "split": row["split"], "id": f"{source}/{row['id']}",
                     "group_id": f"{source}/{row['group_id']}", "variant": "clean", "family": row["family"], "provenance": row["provenance"],
                     "text_sha256": hashlib.sha256(json.dumps(row["state"], sort_keys=True, ensure_ascii=False).casefold().encode()).hexdigest()}}
    if kind == "typesafe":
        ids = [o["id"] for o in row["options"]]
        rec["_meta"]["target"] = dict(zip(ids, row["target_distribution"]))
        rec["_meta"]["published"] = {name: {"model": m["model"], "p": dict(zip(ids, m["distribution"]))} for name, m in row["published_models"].items()}
    rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
    return rec


def upstream(rows):
    """Distinct pinned upstream sources across the rows (WANLI revision + rights; TypeSafe workflow snapshot hashes)."""
    keep = ("source", "source_revision", "rights", "workflow", "snapshot_sha256")
    seen = {json.dumps(p, sort_keys=True): p for p in ({k: v for k, v in r["provenance"].items() if k in keep} for r in rows)}
    return [seen[k] for k in sorted(seen)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semif", required=True, help="SemIf checkout (pins the selection manifest commit)")
    ap.add_argument("--rows", required=True, help="SemIf build_wanli.py / build_typesafe.py output")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, out = read_jsonl(a.rows), Path(a.out)
    if out.exists(): raise FileExistsError(out)
    kind = "typesafe" if "primitive" in rows[0] else "wanli"
    spec = SOURCES[kind]
    if len(rows) != spec["rows"] or len({r["id"] for r in rows}) != spec["rows"]:
        raise ValueError(f"expected {spec['rows']} unique {kind} rows, got {len(rows)}")
    commit = subprocess.run(["git", "-C", a.semif, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    records = [convert(r, kind) for r in rows]
    tasks = sorted({r["questions"]["decision"]["src"] for r in records})
    out.mkdir(parents=True)
    files = {}
    for name, part in (("development.jsonl", records), ("train.jsonl", []), ("calibration.jsonl", []), ("test.jsonl", [])):
        write_jsonl(out / name, part); files[name] = {"sha256": digest(out / name), "records": len(part)}
    write_json(out / "manifest.json", {"version": 1,
                                       "external": {"repo": "https://github.com/TheoLeeCJ/SemIf", "commit": commit, "license": "MIT",
                                                    "selection": "benchmarks/manifests/source-selection.jsonl",
                                                    "files": {"source-selection.jsonl": digest(Path(a.semif) / "benchmarks/manifests/source-selection.jsonl"), Path(a.rows).name: digest(a.rows)},
                                                    "upstream": upstream(rows)},
                                       "base_revisions": {}, "dataset_revisions": {}, "holdout_sources": [], "trainable_sources": [], "eval_only_sources": [spec["source"]],
                                       "context": spec["context"], "files": files, "eval_only": True, "tasks": tasks, "protocol": {"note": spec["protocol"]}})
    print({t: sum(r["questions"]["decision"]["src"] == t for r in records) for t in tasks})


if __name__ == "__main__":
    main()
