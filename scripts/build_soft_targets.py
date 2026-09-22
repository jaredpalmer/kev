"""Ambiguity soft targets (PLAN.md round 4, item 4.9; round-3 C5): where an open teacher confidently disagrees with a
public training label, train toward a mixture of the label and the teacher instead of the hard label.

    uv run python scripts/build_soft_targets.py --teacher runs/probes/qwen35-9b-semif-teacher-decision-v7-train --out evals/round4/soft-v1

Reads the teacher's rows.json (scripts/base_mmlu_probe.py --all_questions over decision-v7/train) and the same training
records. A question is ambiguous when the teacher's top option is not the label and holds >= --threshold of the mass;
its target becomes (1 - lam) * one_hot(label) + lam * teacher. Writes two files with the same records:

    soft.jsonl   ambiguous questions carry `target` (kev.data.materialize trains soft-target cross-entropy on it)
    hard.jsonl   the identical records with hard labels: the matched continuation control

Other questions of a selected record keep their hard label in both files. Teacher outputs are open-weight (never Jev's).
"""
import argparse, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import digest, load_split, read_json, read_manifest, write_json, write_jsonl  # noqa: E402

SUITE = "evals/v7/decision-v7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="probe output dir with rows.json and report.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--lam", type=float, default=0.5)
    a = ap.parse_args()
    if not 0.5 <= a.threshold < 1 or not 0 < a.lam < 1:
        raise SystemExit("threshold must be in [0.5, 1) and lam in (0, 1)")
    teacher = {(r["id"], r["question"]): r for r in read_json(Path(a.teacher) / "rows.json")}
    records = {r["_meta"]["id"]: r for r in load_split(SUITE, "train")}
    targets, per_source, seen = {}, Counter(), Counter()
    for (rid, qid), row in teacher.items():
        seen[row["source"]] += 1
        top = max(range(len(row["p"])), key=row["p"].__getitem__)
        if top == row["label"] or row["p"][top] < a.threshold:
            continue
        mix = [a.lam * p + (1 - a.lam) * (i == row["label"]) for i, p in enumerate(row["p"])]
        targets[(rid, qid)] = dict(zip(row["keys"], mix)); per_source[row["source"]] += 1
    soft, hard = [], []
    for rid in sorted({rid for rid, _ in targets}):
        rec = records[rid]
        hard.append(rec)
        soft.append({**rec, "questions": {qid: ({**q, "target": targets[(rid, qid)]} if (rid, qid) in targets else q) for qid, q in rec["questions"].items()}})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "soft.jsonl", soft); write_jsonl(out / "hard.jsonl", hard)
    teacher_report = read_json(Path(a.teacher) / "report.json")
    manifest = {"suite": SUITE, "suite_manifest_sha256": digest(Path(SUITE) / "manifest.json"), "partition": "train",
                "teacher": {"base": teacher_report.get("base"), "revision": teacher_report.get("revision"), "readout": teacher_report.get("readout"),
                            "rows_sha256": digest(Path(a.teacher) / "rows.json")},
                "rule": {"threshold": a.threshold, "lam": a.lam, "ambiguous": "teacher top option != label and teacher top probability >= threshold",
                         "target": "(1 - lam) * one_hot(label) + lam * teacher"},
                "questions_scored": dict(seen), "questions_softened": dict(per_source), "records": len(soft),
                "files": {name: {"sha256": digest(out / name), "records": len(rows)} for name, rows in (("soft.jsonl", soft), ("hard.jsonl", hard))},
                "trainable_sources": read_manifest(SUITE)["trainable_sources"]}
    write_json(out / "manifest.json", manifest)
    print(f"{len(targets)} of {sum(seen.values())} questions softened in {len(soft)} records:", dict(per_source))


if __name__ == "__main__":
    main()
