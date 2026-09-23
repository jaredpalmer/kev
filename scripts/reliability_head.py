"""Reliability head (PLAN.md round 4, item 4.10): re-rank confidences with a logistic model of P(correct) on features
of the served distribution, fitted on the decision-v7 calibration partition (never used for training or selection),
evaluated on out-of-distribution rows it never saw. A temperature cannot reorder confidences within a question type;
this can, so it is the one post-hoc lever on coverage at a fixed error budget and AURC.

    uv run python scripts/reliability_head.py --trial runs/night2-9b-du/00-trial-0 --external runs/kev-9b-wanli-v1/rows.json --out runs/r4-reliability-9b

Features: p_max, margin (top-1 minus top-2), normalised entropy, log K, question type. Trial rows were saved raw and
without logits, so they are tempered to the checkpoint's shipped temperature from floored probabilities (approximate,
like scripts/calibration_audit.py); external rows are used as served. Accuracy is unchanged by construction; the
comparison is AURC and coverage at <= 5% error of the head's confidence against p_max, paired over the same questions
with the record-clustered bootstrap unit of kev.metrics.
"""
import argparse, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import read_meta  # noqa: E402
from kev.metrics import area_under_risk_coverage, cluster_resamples, coverage_at_error, scored_rows, served_at  # noqa: E402
from kev.suite import read_json, write_json  # noqa: E402

TYPES = ("choice", "noul", "score")


def features(rows):
    out = []
    for r in rows:
        p = np.sort(np.asarray(r["p"], dtype=float))[::-1]; k = len(p)
        entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-12)))) / math.log(k) if k > 1 else 0.0
        out.append([p[0], p[0] - (p[1] if k > 1 else 0.0), entropy, math.log(k), *(float(r["type"] == t) for t in TYPES)])
    return np.asarray(out)


def correct(rows):
    return np.asarray([int(np.argmax(r["p"]) == r["label"]) for r in rows], dtype=float)


def fit_logistic(x, y, l2=1.0, steps=50):
    """L2-regularised logistic regression by Newton's method on standardised features (intercept unpenalised)."""
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = np.c_[np.ones(len(x)), (x - mu) / sd]; w = np.zeros(z.shape[1]); penalty = np.r_[0.0, np.full(z.shape[1] - 1, l2)]
    for _ in range(steps):
        p = 1 / (1 + np.exp(-z @ w))
        grad = z.T @ (p - y) + penalty * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(penalty)
        w -= np.linalg.solve(hess, grad)
    return lambda x_new: 1 / (1 + np.exp(-np.c_[np.ones(len(x_new)), (x_new - mu) / sd] @ w))


def compare(rows, head, samples=1000, seed=0):
    ok = correct(rows).astype(bool); base = np.asarray([max(r["p"]) for r in rows]); new = head(features(rows))
    stats = lambda conf, idx: (area_under_risk_coverage(conf[idx], ok[idx]), coverage_at_error(conf[idx], ok[idx], 0.05))
    deltas = []
    for idx in cluster_resamples(rows, samples, seed):
        (a0, c0), (a1, c1) = stats(base, idx), stats(new, idx); deltas.append((a1 - a0, c1 - c0))
    (a0, c0), (a1, c1) = stats(base, np.arange(len(rows))), stats(new, np.arange(len(rows))); deltas = np.asarray(deltas)
    ci = lambda col: np.quantile(deltas[:, col], [0.025, 0.975]).tolist()
    return {"n": len(rows), "acc": float(ok.mean()), "p_max": {"aurc": a0, "coverage_at_5pct_error": c0}, "head": {"aurc": a1, "coverage_at_5pct_error": c1},
            "delta": {"aurc": a1 - a0, "aurc_ci95": ci(0), "coverage_at_5pct_error": c1 - c0, "coverage_ci95": ci(1)},
            "improved": bool(ci(0)[1] < 0 or ci(1)[0] > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="trial dir with calibration/ and transfer/ rows.json and checkpoint/head.pt")
    ap.add_argument("--external", action="append", default=[], help="served rows.json of an external suite (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    temperature = read_meta(Path(a.trial) / "checkpoint").temperature
    fit_rows = served_at(read_json(Path(a.trial) / "calibration/rows.json"), temperature)
    head = fit_logistic(features(fit_rows), correct(fit_rows))
    report = {"trial": a.trial, "temperature": temperature, "fit": {"partition": "decision-v7/calibration", "n": len(fit_rows)},
              "features": ["p_max", "margin", "normalised_entropy", "log_k", *[f"type_{t}" for t in TYPES]],
              "evaluations": {"transfer-v4/development": compare(served_at(read_json(Path(a.trial) / "transfer/rows.json"), temperature), head)}}
    for path in a.external:
        report["evaluations"][path] = compare(scored_rows(read_json(path)), head)
    Path(a.out).mkdir(parents=True, exist_ok=True); write_json(Path(a.out) / "report.json", report)
    for name, e in report["evaluations"].items():
        d = e["delta"]
        print(f"{name:40} n={e['n']:4} aurc {e['p_max']['aurc']:.4f} -> {e['head']['aurc']:.4f} [{d['aurc_ci95'][0]:+.4f},{d['aurc_ci95'][1]:+.4f}]"
              f"  cov@5% {e['p_max']['coverage_at_5pct_error']:.3f} -> {e['head']['coverage_at_5pct_error']:.3f} [{d['coverage_ci95'][0]:+.3f},{d['coverage_ci95'][1]:+.3f}]  improved={e['improved']}")


if __name__ == "__main__":
    main()
