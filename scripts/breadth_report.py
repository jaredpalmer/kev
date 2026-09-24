"""Score kev.benchmark result directories on evals/breadth-v1 the way the community Decision Index scores its panel.

    uv run python scripts/breadth_report.py --suite evals/breadth-v1 \
        --result Kev-27B=runs/breadth-v1-kev-27b --result Jev=runs/breadth-v1-jev --out runs/breadth-v1-report

Per dataset: the raw score (question accuracy, or case-exact accuracy where the manifest says so), coverage-adjusted
(a question or record with no row, e.g. refused as too long, counts as wrong against the full partition), the chance
level of uniform random answers on the same questions, and the chance-corrected skill clip((score - chance) / (1 -
chance)). An area is the plain mean of its datasets; the index is 100 x the mean of the five areas (the Decision Index
0.2 formula, data/methodology.json in multimodalart/jev-decision-index). Calibration (ECE, Brier, NLL; kev.metrics) is
computed on the returned probabilities of answered questions, per dataset, per area (pooled rows) and overall (pooled,
plus the mean of the dataset ECEs). Writes report.json and report.md into --out and prints the markdown.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.benchmark import labels  # noqa: E402
from kev.metrics import metrics  # noqa: E402
from kev.suite import digest, load_split, read_json, read_manifest, write_json  # noqa: E402

CALIBRATION = ("ece", "brier", "nll", "mean_conf")


def clip(x):
    return min(1.0, max(0.0, x))


def skill(score, chance):
    """(score - chance) / (1 - chance), clipped to [0, 1]: 0 is random guessing, 1 perfect."""
    return clip((score - chance) / (1 - chance)) if chance < 1 else 0.0


def option_count(q):
    return 2 if q["type"] == "noul" else len(q["criteria"])


def dataset_score(records, rows, metric):
    """Coverage-adjusted score of one dataset. records: its suite records; rows: {(record id, question id): row} of one
    system (any subset). Returns score, chance, skill, answered share, accuracy on answered units and unit count."""
    units = []   # (chance, answered, correct)
    for r in records:
        rid = r["_meta"]["id"]
        per_q = []
        for qid, q in r["questions"].items():
            row = rows.get((rid, qid))
            if row is not None and row["label"] != labels(q)[1]:
                raise ValueError(f"{rid}/{qid}: row label {row['label']} differs from the suite's; wrong suite or partition?")
            per_q.append((1 / option_count(q), row is not None, row is not None and int(np.argmax(row["p"])) == row["label"]))
        if metric == "case_exact":
            units.append((float(np.prod([c for c, _, _ in per_q])), all(a for _, a, _ in per_q), all(ok for _, _, ok in per_q)))
        elif metric == "accuracy":
            units += per_q
        else:
            raise ValueError(f"unknown metric {metric!r}")
    n = len(units)
    answered = sum(a for _, a, _ in units)
    correct = sum(ok for _, _, ok in units)
    score, chance = correct / n, float(np.mean([c for c, _, _ in units]))
    return {"units": n, "score": score, "chance": chance, "skill": skill(score, chance), "answered": answered / n,
            "accuracy_answered": correct / answered if answered else None}


def record_outcomes(records, rows, metric):
    """Per record (correct units, units, summed chance): the cluster the bootstrap resamples. Accuracy counts questions,
    case_exact counts the record as one unit."""
    out = []
    for r in records:
        rid = r["_meta"]["id"]
        oks = [(rid, qid) in rows and int(np.argmax(rows[(rid, qid)]["p"])) == rows[(rid, qid)]["label"] for qid in r["questions"]]
        chances = [1 / option_count(q) for q in r["questions"].values()]
        out.append((float(all(oks)), 1.0, float(np.prod(chances))) if metric == "case_exact" else (float(sum(oks)), float(len(oks)), float(sum(chances))))
    return np.array(out)


def bootstrap_index(records, systems_rows, manifest, samples=2000, seed=0):
    """Paired record-clustered bootstrap of the index: each resample draws records with replacement within every dataset
    (the same draw for every system) and recomputes dataset skill, area means and the index. Returns per system the 95 %
    interval of its index, and of its difference from the first system."""
    rng = np.random.default_rng(seed)
    names = list(systems_rows)
    keyed = {n: {(r["id"], r["question"]): r for r in rows if r.get("variant", "clean") == "clean"} for n, rows in systems_rows.items()}
    by_dataset = {}
    for r in records: by_dataset.setdefault(r["_meta"]["source"], []).append(r)
    outcomes = {d: {n: record_outcomes(by_dataset[d], keyed[n], spec["metric"]) for n in names} for d, spec in manifest["datasets"].items()}
    index = {n: np.zeros(samples) for n in names}
    for area, spec in manifest["areas"].items():
        area_skill = {n: np.zeros(samples) for n in names}
        for d in spec["datasets"]:
            k = len(by_dataset[d])
            draw = rng.integers(0, k, size=(samples, k))
            for n in names:
                o = outcomes[d][n][draw]                      # samples x k x 3
                score, chance = o[..., 0].sum(1) / o[..., 1].sum(1), o[..., 2].sum(1) / o[..., 1].sum(1)
                area_skill[n] += np.clip((score - chance) / (1 - chance), 0, 1) / len(spec["datasets"])
        for n in names: index[n] += 100 * area_skill[n] / len(manifest["areas"])
    ci = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
    return {"samples": samples, "seed": seed, "unit": "record, resampled within each dataset, paired across systems",
            "index": {n: ci(index[n]) for n in names},
            "difference_from": names[0], "difference": {n: ci(index[n] - index[names[0]]) for n in names[1:]}}


def calibration(rows):
    if not rows: return {k: None for k in CALIBRATION} | {"n": 0}
    m = metrics(rows)
    return {"n": m["n"], **{k: m[k] for k in CALIBRATION}}


def score_system(records, rows, manifest):
    """The full report for one system: per dataset, per area and overall."""
    by_key = {(r["id"], r["question"]): r for r in rows if r.get("variant", "clean") == "clean"}
    by_dataset = {}
    for r in records: by_dataset.setdefault(r["_meta"]["source"], []).append(r)
    datasets = {}
    for name, spec in manifest["datasets"].items():
        recs = by_dataset.get(name, [])
        if not recs: raise ValueError(f"no {name} records in this partition")
        answered_rows = [by_key[(r["_meta"]["id"], qid)] for r in recs for qid in r["questions"] if (r["_meta"]["id"], qid) in by_key]
        datasets[name] = {"area": spec["area"], "metric": spec["metric"], **dataset_score(recs, by_key, spec["metric"]), "calibration": calibration(answered_rows),
                          "_rows": answered_rows}
    areas = {}
    for area, spec in manifest["areas"].items():
        ds = [datasets[d] for d in spec["datasets"]]
        areas[area] = {"label": spec["label"], "datasets": spec["datasets"], "skill": float(np.mean([d["skill"] for d in ds])),
                       "score": float(np.mean([d["score"] for d in ds])), "calibration": calibration([row for d in ds for row in d["_rows"]])}
    all_rows = [row for d in datasets.values() for row in d["_rows"]]
    eces = [d["calibration"]["ece"] for d in datasets.values() if d["calibration"]["ece"] is not None]
    overall = {"index": 100 * float(np.mean([a["skill"] for a in areas.values()])), "raw_index": 100 * float(np.mean([a["score"] for a in areas.values()])),
               "question_accuracy_answered": float(np.mean([int(np.argmax(r["p"])) == r["label"] for r in all_rows])) if all_rows else None,
               "answered_questions": len(all_rows), "questions": sum(len(r["questions"]) for r in records),
               "calibration": calibration(all_rows), "mean_dataset_ece": float(np.mean(eces)) if eces else None}
    for d in datasets.values(): del d["_rows"]
    return {"datasets": datasets, "areas": areas, "overall": overall}


def markdown(systems, manifest):
    names = list(systems)
    head = "| | " + " | ".join(f"{n} acc / index / ECE" for n in names) + " |\n|---|" + "---|" * len(names) + "\n"
    fmt = lambda a, s, e: f"{a:.3f} / {100 * s:.1f} / {e:.3f}" if e is not None else f"{a:.3f} / {100 * s:.1f} / -"
    lines = [f"| {spec['label']} | " + " | ".join(fmt(systems[n]["areas"][area]["score"], systems[n]["areas"][area]["skill"], systems[n]["areas"][area]["calibration"]["ece"]) for n in names) + " |"
             for area, spec in manifest["areas"].items()]
    lines.append("| **Overall** | " + " | ".join(f"{systems[n]['overall']['raw_index'] / 100:.3f} / **{systems[n]['overall']['index']:.1f}** / {systems[n]['overall']['calibration']['ece']:.3f} ({systems[n]['overall']['mean_dataset_ece']:.3f})" for n in names) + " |")
    table = head + "\n".join(lines)
    detail = "| dataset (area, metric, chance) | " + " | ".join(f"{n} score / skill / answered / ECE" for n in names) + " |\n|---|" + "---|" * len(names) + "\n"
    for d, spec in manifest["datasets"].items():
        first = systems[names[0]]["datasets"][d]
        cells = []
        for n in names:
            x = systems[n]["datasets"][d]
            ece = x["calibration"]["ece"]
            cells.append(f"{x['score']:.3f} / {100 * x['skill']:.1f} / {x['answered']:.2f} / " + (f"{ece:.3f}" if ece is not None else "-"))
        detail += f"| {d} ({spec['area']}, {spec['metric']}, {first['chance']:.3f}) | " + " | ".join(cells) + " |\n"
    return ("Areas: raw score (coverage-adjusted) / chance-corrected index (0-100) / ECE of the area's pooled rows. Overall: raw index / Decision-Index-style index / pooled ECE (mean of the 14 dataset ECEs).\n\n"
            + table + "\n\n" + detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--result", action="append", required=True, help="NAME=DIR of a kev.benchmark result (rows.json [+ report.json])")
    ap.add_argument("--split", choices=["development", "test"], default="development")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--out", help="directory for report.json and report.md")
    ap.add_argument("--bootstrap", type=int, default=0, help="paired record-clustered bootstrap resamples for index intervals (0 = none)")
    a = ap.parse_args()
    manifest = read_manifest(a.suite)
    try:
        records = load_split(a.suite, a.split, allow_test=a.allow_test)
    except PermissionError as error:   # breadth-v1's partitions live only in a private mirror (manifest "mirror")
        raise SystemExit(f"cannot score without the suite's partitions: {error}") from None
    systems, sources = {}, {}
    for spec in a.result:
        name, _, directory = spec.partition("=")
        if not directory: ap.error(f"--result {spec!r} is not NAME=DIR")
        systems[name] = score_system(records, read_json(Path(directory) / "rows.json"), manifest)
        report = Path(directory) / "report.json"
        sources[name] = {"dir": directory, "rows_sha256": digest(Path(directory) / "rows.json"),
                         "run": read_json(report).get("run") if report.exists() else None}
    text = markdown(systems, manifest)
    uncertainty = None
    if a.bootstrap:
        uncertainty = bootstrap_index(records, {name: read_json(Path(sources[name]["dir"]) / "rows.json") for name in systems}, manifest, a.bootstrap)
        first = uncertainty["difference_from"]
        text += "\nIndex, 95 % paired record bootstrap (" + str(a.bootstrap) + " resamples): " + "; ".join(
            f"{n} {systems[n]['overall']['index']:.1f} [{lo:.1f}, {hi:.1f}]" for n, (lo, hi) in uncertainty["index"].items()) + ". Difference from " + first + ": " + "; ".join(
            f"{n} {systems[n]['overall']['index'] - systems[first]['overall']['index']:+.1f} [{lo:+.1f}, {hi:+.1f}]" for n, (lo, hi) in uncertainty["difference"].items()) + ".\n"
    print(text)
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        write_json(out / "report.json", {"suite": a.suite, "suite_manifest_sha256": digest(Path(a.suite) / "manifest.json"), "split": a.split,
                                         "formula": manifest["scoring"], "sources": sources, "systems": systems, "uncertainty": uncertainty})
        (out / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
