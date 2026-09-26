"""Accuracy, ECE and Brier by length bucket and part on evals/longdoc-v1, for one or more benchmark results.

    uv run python scripts/longdoc_report.py --suite evals/longdoc-v1 --out runs/longdoc-v1-report \
        --result "Kev-27B (T=1.38 shipped)=runs/longdoc-v1-kev-27b" \
        --result "r19 SFT (raw T=1)=runs/longdoc-v1-r19-sft@1.0" --result "r19 SFT (T=1.59, exploratory)=runs/longdoc-v1-r19-sft@1.59" \
        --result "Jev=runs/longdoc-v1-jev"

A result is NAME=DIR[@T]: DIR holds kev.benchmark / kev.jev rows.json; @T serves the rows at temperature T from their raw logits
(kev.metrics.served_at; the rows' recorded inference temperature is undone first), otherwise they are scored as returned.
Records a system did not answer (rejected.json: Jev refusals) are listed per bucket and left out of that system's metrics;
`coverage` says how many. Per bucket: accuracy / ECE (10 bins) / Brier over the questions of both parts and of each part and
kind, the difference in accuracy from the 4k control with a 95 % record-clustered bootstrap interval (2,000 resamples, seed 0;
the buckets hold different records, resampled independently), and `falls` when that interval lies below zero. For CUAD,
whose 8k-64k buckets ask the same questions about the same target contracts, the paired difference from 8k as well. For
the synthetic detail questions, accuracy by depth (10 / 50 / 90 %). For CUAD also `cuad.no_sft_ledgar`: the questions whose
target contract contains no LEDGAR provision of the SFT corpus (overlap.json, cuad_targets.sft_v1_ledgar), the sensitivity
read for that contamination. Writes report.json and report.md.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.metrics import metrics, served_at  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

BUCKETS = ("4k", "8k", "16k", "32k", "64k")
SAMPLES, SEED = 2000, 0


def parse(spec):
    name, _, path = spec.rpartition("=")   # the name may itself contain "=" ("T=1.38"); the path never does
    path, _, t = path.partition("@")
    return name, Path(path), float(t) if t else None


def bucket_of(row):
    return row["task"].split(".")[-1]


def summary(rows):
    if not rows: return None
    m = metrics(rows)
    return {"questions": len(rows), "acc": round(m["acc"], 4), "ece": round(m["ece"], 4), "brier": round(m["brier"], 4)}


def correct_by_record(rows):
    out = defaultdict(list)
    for r in rows: out[r["id"]].append(int(np.argmax(r["p"]) == r["label"]))
    return out


def diff_ci(a_rows, b_rows):
    """(mean acc of b - a, 95 % bootstrap interval), records resampled independently within each side."""
    rng = np.random.default_rng(SEED)
    a, b = list(correct_by_record(a_rows).values()), list(correct_by_record(b_rows).values())
    if not a or not b: return None
    def mean(groups, idx): return sum(sum(groups[i]) for i in idx) / sum(len(groups[i]) for i in idx)
    point = mean(b, range(len(b))) - mean(a, range(len(a)))
    d = [mean(b, rng.integers(0, len(b), len(b))) - mean(a, rng.integers(0, len(a), len(a))) for _ in range(SAMPLES)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"delta": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "falls": bool(hi < 0)}


def paired_ci(a_rows, b_rows, key):
    """Paired accuracy difference b - a over questions matched by key(row), clustered by target."""
    rng = np.random.default_rng(SEED)
    a = {key(r): r for r in a_rows}; pairs = defaultdict(list)
    for r in b_rows:
        k = key(r)
        if k in a: pairs[k[0]].append(int(np.argmax(r["p"]) == r["label"]) - int(np.argmax(a[k]["p"]) == a[k]["label"]))
    groups = list(pairs.values())
    if not groups: return None
    n = sum(len(g) for g in groups)
    point = sum(sum(g) for g in groups) / n
    d = []
    for _ in range(SAMPLES):
        idx = rng.integers(0, len(groups), len(groups))
        d.append(sum(sum(groups[i]) for i in idx) / max(1, sum(len(groups[i]) for i in idx)))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"questions": n, "delta": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "falls": bool(hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="evals/longdoc-v1")
    ap.add_argument("--result", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from kev.suite import load_split
    meta = {r["_meta"]["id"]: r["_meta"] for r in load_split(a.suite, "development")}
    overlap = Path(a.suite) / "overlap.json"
    seen_ledgar = set(read_json(overlap)["cuad_targets"]["sft_v1_ledgar"]["titles_with_contained_item"]) if overlap.exists() else set()
    out = {"suite": a.suite, "split": "development", "systems": {}}
    for spec in a.result:
        name, path, t = parse(spec)
        rows = read_json(path / "rows.json")
        rows = served_at(rows, t) if t is not None else rows
        rejected = read_json(path / "rejected.json") if (path / "rejected.json").exists() else []
        answered = {r["id"] for r in rows}
        sys_out = {"dir": str(path), "temperature": t if t is not None else "as returned", "buckets": {}}
        by = defaultdict(list)
        for r in rows: by[bucket_of(r)].append(r)
        for b in BUCKETS:
            rs = by.get(b, [])
            ids = [i for i, m in meta.items() if f"{m['bucket'] // 1024}k" == b]
            entry = {"all": summary(rs), "coverage": {"records": len(ids), "answered": sum(i in answered for i in ids),
                                                      "rejected": sum(1 for x in rejected if f"/{b}/" in x["id"])}}
            for part in ("cuad", "synthetic"):
                prs = [r for r in rs if r["task"].startswith(part)]
                entry[part] = summary(prs)
                for kind in sorted({r["task"].split(".")[1] for r in prs}):
                    entry[f"{part}.{kind}"] = summary([r for r in prs if r["task"].split(".")[1] == kind])
            entry["cuad.no_sft_ledgar"] = summary([r for r in rs if r["task"].startswith("cuad") and meta[r["id"]]["target"] not in seen_ledgar])
            if b != "4k":
                entry["vs_4k"] = {"all": diff_ci(by.get("4k", []), rs),
                                  **{part: diff_ci([r for r in by.get("4k", []) if r["task"].startswith(part)], [r for r in rs if r["task"].startswith(part)]) for part in ("cuad", "synthetic")}}
            if b not in ("4k", "8k"):
                key = lambda r: (meta[r["id"]]["target"], meta[r["id"]]["repeat"], r["question"])
                entry["cuad_paired_vs_8k"] = paired_ci([r for r in by.get("8k", []) if r["task"].startswith("cuad")], [r for r in rs if r["task"].startswith("cuad")], key)
            detail = [r for r in rs if r["task"].startswith("synthetic.detail")]
            entry["synthetic.detail_by_depth"] = {str(d): summary([r for r in detail if meta[r["id"]]["depth_target"] == d]) for d in (0.1, 0.5, 0.9)}
            sys_out["buckets"][b] = entry
        falls = [b for b in BUCKETS[1:] if (sys_out["buckets"][b].get("vs_4k") or {}).get("all") and sys_out["buckets"][b]["vs_4k"]["all"]["falls"]]
        sys_out["first_bucket_below_4k"] = falls[0] if falls else None
        out["systems"][name] = sys_out
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out) / "report.json", out)
    lines = ["| system | part | " + " | ".join(BUCKETS) + " |", "|---|---|" + "---|" * len(BUCKETS)]
    for name, s in out["systems"].items():
        for part in ("all", "cuad", "synthetic"):
            cells = []
            for b in BUCKETS:
                e = s["buckets"][b][part]
                cells.append("-" if e is None else f"{e['acc']:.3f} / {e['ece']:.3f} / {e['brier']:.3f}")
            lines.append(f"| {name} | {part} | " + " | ".join(cells) + " |")
    lines += ["", "accuracy / ECE / Brier per cell; first bucket whose accuracy is below the 4k control (95 % CI): " +
              "; ".join(f"{n}: {s['first_bucket_below_4k']}" for n, s in out["systems"].items())]
    (Path(a.out) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
