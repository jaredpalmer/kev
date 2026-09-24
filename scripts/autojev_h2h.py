"""Report-only head-to-head (PLAN.md, "External head-to-head: AutoJev-27B vs Kev-27B"): AutoJev as served (its own
fitted temperature) against Kev-27B at its fitted temperature, paired on shared (id, question) rows, plus Jev where a
read of the same suite exists. AutoJev's refused records (its 8,192-token question-branch limit) are reported and also
scored as wrong in a coverage-adjusted accuracy, the Decision Index convention.

    uv run python scripts/autojev_h2h.py --out runs/autojev-h2h
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kev.metrics import metrics  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402
from round6_readout import boot, knowable, serve  # noqa: E402

KEV = "runs/release/kev-27b-v2"
SUITES = {   # tag: (AutoJev read, Kev-27B read, Jev read or None)
    "transfer-v4": ("runs/autojev-transfer-v4", f"{KEV}/transfer", "runs/jev-transfer-v4"),
    "hard-v1": ("runs/autojev-hard", "runs/hv1-27b", "runs/jev-hard-v1-r2"),
    "devtools-v1": ("runs/autojev-devtools", "runs/dt1-27b", "runs/jev-devtools-v1"),
    "documents-v1": ("runs/autojev-docs", "runs/r6-27bv2-s2-docs", "runs/jev-documents-v1"),
    "semif": ("runs/autojev-semif", "runs/r6-27bv2-s2-semif", "runs/jev-semif-v1"),
    "scienthoon": ("runs/autojev-scienthoon", "runs/r6-27bv2-s2-scienthoon", "runs/jev-scienthoon-v1"),
    "wanli2": ("runs/autojev-wanli2", "runs/r6-27bv2-s2-wanli2", None),
    "typesafe": ("runs/autojev-typesafe", "runs/r6-27bv2-s2-typesafe", "runs/jev-typesafe-v1"),
    "transfer-v9": ("runs/autojev-v9", "runs/r6-27bv2-s2-v9", "runs/jev-transfer-v9"),
}
DUPLICATE_IDS = {"codereviewer/cls-test/13657", "codereviewer/cls-test/19245"}   # devtools-v1 (PLAN.md round 10 amendment)


def clean(rows):
    return [r for r in knowable(rows) if r["id"] not in DUPLICATE_IDS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    t = read_json(f"{KEV}/result.json")["calibration_fit"]["temperature"]
    report = {"kev_temperature": t, "suites": {}}
    for tag, (aj, kev, jev) in SUITES.items():
        if not Path(f"{aj}/report.json").exists(): report["suites"][tag] = "not read yet"; continue
        A, K = clean(read_json(f"{aj}/rows.json")), clean(serve(KEV, f"{kev}/rows.json", t)[0])
        keys = {(r["id"], r["question"]) for r in A} & {(r["id"], r["question"]) for r in K}
        As, Ks = [r for r in A if (r["id"], r["question"]) in keys], [r for r in K if (r["id"], r["question"]) in keys]
        cov = read_json(f"{aj}/report.json")["coverage"]
        refused_q = sum(1 for r in K if (r["id"], r["question"]) not in keys)
        ma, mk = metrics(As), metrics(Ks)
        row = {"shared_questions": len(keys), "autojev_refused_records": cov["rejected_records"], "kev_questions_not_answered_by_autojev": refused_q,
               "autojev": {k: ma[k] for k in ("acc", "brier", "ece", "coverage_at_5pct_error")}, "kev27b": {k: mk[k] for k in ("acc", "brier", "ece", "coverage_at_5pct_error")},
               "autojev_minus_kev": {m: boot(As, Ks, m) for m in ("acc", "brier")},
               "autojev_coverage_adjusted_acc": ma["acc"] * len(keys) / (len(keys) + refused_q), "kev27b_acc_all": metrics(K)["acc"]}
        if jev and Path(f"{jev}/rows.json").exists():
            J = [r for r in clean([x for x in read_json(f"{jev}/rows.json") if x.get("variant", "clean") == "clean"]) if (r["id"], r["question"]) in keys]
            if J: row["jev"] = {"acc": metrics(J)["acc"], "brier": metrics(J)["brier"], "ece": metrics(J)["ece"], "n": len(J)}
        report["suites"][tag] = row
    done = {k: v for k, v in report["suites"].items() if isinstance(v, dict)}
    if done:
        pooled_a = [r for tag in done for r in clean(read_json(f"{SUITES[tag][0]}/rows.json"))]
        report["macro_acc"] = {"autojev": sum(v["autojev"]["acc"] for v in done.values()) / len(done), "kev27b": sum(v["kev27b"]["acc"] for v in done.values()) / len(done), "suites": sorted(done)}
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "report.json", report)
    f = lambda b: f"{100 * b['delta']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]"
    print(f"{'suite':13} {'n':>5} {'AutoJev':>8} {'Kev-27B':>8} {'Jev':>6}   AutoJev - Kev (acc)      Brier AJ / Kev    ECE AJ / Kev   refused")
    for tag, r in report["suites"].items():
        if not isinstance(r, dict): print(f"{tag:13} {r}"); continue
        j = f"{r['jev']['acc']:.3f}" if "jev" in r else "  -  "
        print(f"{tag:13} {r['shared_questions']:>5} {r['autojev']['acc']:>8.3f} {r['kev27b']['acc']:>8.3f} {j:>6}   {f(r['autojev_minus_kev']['acc']):22}   {r['autojev']['brier']:.3f} / {r['kev27b']['brier']:.3f}    {r['autojev']['ece']:.3f} / {r['kev27b']['ece']:.3f}   {r['autojev_refused_records']}")
    if "macro_acc" in report: print("macro acc:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["macro_acc"].items()})


if __name__ == "__main__":
    main()
