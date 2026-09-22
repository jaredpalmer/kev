"""Scoring of benchmark rows: accuracy, calibration (ECE, Brier, NLL), selective prediction (coverage at an error budget,
risk-coverage curve, AURC, thresholds), temperature fitting and the record-clustered paired bootstrap.

Pure numpy over the row dicts that kev.benchmark.prediction_rows produces ({"p", "label", "type", "keys", "task", "source",
"variant", ..., optionally "logits"}). No torch, no model: usable on saved rows.json files.
"""
import math
from collections import defaultdict

import numpy as np

EPSILON = 1e-9


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def _check_temperature(temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def _tempered_logits(row, temperature):
    """(z - max z) / T from the recorded logits, or from the floored probabilities when the row has none."""
    _check_temperature(temperature)
    p = np.asarray(row["p"], dtype=float)
    z = np.asarray(row["logits"], dtype=float) if "logits" in row else np.log(np.maximum(p, EPSILON))
    if z.shape != p.shape or not np.isfinite(z).all():
        raise ValueError("logits must be finite and match the option count")
    return (z - z.max()) / temperature


def probabilities_at_temperature(row, temperature=1.0):
    """The row's returned probabilities at T=1; otherwise the softmax of its tempered logits."""
    _check_temperature(temperature)
    if temperature == 1:
        return np.asarray(row["p"], dtype=float)
    p = np.exp(_tempered_logits(row, temperature))
    return p / p.sum()


def nll_at_temperature(row, temperature=1.0):
    """Exact from logits when recorded; otherwise from the floored probabilities."""
    if "logits" in row:
        z = _tempered_logits(row, temperature)
        return float(np.log(np.exp(z).sum()) - z[row["label"]])
    return -math.log(max(float(probabilities_at_temperature(row, temperature)[row["label"]]), EPSILON))


def _row_scores(row):
    """The additive per-question quantities whose means metrics() reports (used by the paired bootstrap)."""
    p = np.asarray(row["p"], dtype=float); y = row["label"]
    conf, ok = float(p.max()), bool(p.argmax() == y)
    return {"acc": float(ok), "nll": nll_at_temperature(row), "brier": float(((p - np.eye(len(p))[y]) ** 2).sum()), "mean_conf": conf,
            "confident_error_rate": float(conf >= 0.9 and not ok), "coverage_at_0_9": float(conf >= 0.9), "confidence_bias": conf - float(ok)}


def metrics(rows, temperature=1.0):
    if not rows:
        raise ValueError("cannot score an empty population")
    nll, acc, conf, brier, mae, rps = [], [], [], [], [], []
    for row in rows:
        p = probabilities_at_temperature(row, temperature)
        y = row["label"]
        target = np.eye(len(p))[y]
        nll.append(nll_at_temperature(row, temperature))
        acc.append(int(p.argmax() == y)); conf.append(float(p.max()))
        brier.append(float(((p - target) ** 2).sum()))
        if row["type"] == "score":
            mae.append(abs(float(p @ np.arange(len(p))) - y))
            rps.append(float(((p.cumsum()[:-1] - target.cumsum()[:-1]) ** 2).mean()))
    result = {"n": len(rows), "nll": float(np.mean(nll)), "acc": float(np.mean(acc)),
              "ece": ece(conf, acc), "brier": float(np.mean(brier)), "mean_conf": float(np.mean(conf))}
    confidence, correct = np.asarray(conf), np.asarray(acc, dtype=bool)
    high = confidence >= 0.9
    result.update(confident_error_rate=float(np.mean(high & ~correct)), coverage_at_0_9=float(high.mean()),
                  accuracy_at_0_9=float(correct[high].mean()) if high.any() else None,
                  coverage_at_5pct_error=coverage_at_error(confidence, correct, 0.05), coverage_at_1pct_error=coverage_at_error(confidence, correct, 0.01),
                  aurc=area_under_risk_coverage(confidence, correct),
                  error_rate_at_0_9=float((~correct[high]).mean()) if high.any() else None,
                  # signed over-confidence (mean top probability minus accuracy) and errors within the top confidence bins;
                  # the sign is diagnostic: untrained readouts run positive, outcome-trained ones near zero or negative
                  confidence_bias=float(confidence.mean() - correct.mean()),
                  top_bins={str(t): {"n": int((confidence >= t).sum()), "errors": int(((confidence >= t) & ~correct).sum()),
                                     "error_rate": float((~correct[confidence >= t]).mean()) if (confidence >= t).any() else None} for t in (0.9, 0.95, 0.99)})
    result["selective"] = {}
    for fraction in (0.5, 0.8):
        cutoff = np.sort(confidence)[-max(1, math.ceil(len(rows) * fraction))]
        selected = confidence >= cutoff
        result["selective"][str(fraction)] = {"coverage": float(selected.mean()), "accuracy": float(correct[selected].mean()),
                                            "confidence_cutoff": float(cutoff)}
    if mae:
        result.update(score_mae=float(np.mean(mae)), ranked_probability_score=float(np.mean(rps)))
    return result


def coverage_at_error(confidence, correct, budget):
    """Selective automation: the largest share of decisions that can be accepted, in descending confidence order, while
    the empirical error among the accepted stays <= budget (jev-benchmarks' "coverage at a fixed error budget"). A model
    whose probabilities are honest gets high coverage; one that is confidently wrong gets little, whatever its accuracy."""
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("error budget must be in [0, 1]")
    _, accepted, errors = _risk_curve_arrays(confidence, correct)
    ok = np.flatnonzero(errors <= budget * accepted)
    return float(accepted[ok[-1]] / accepted[-1]) if len(ok) else 0.0


def _selective_inputs(confidence, correct):
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct)
    if confidence.ndim != 1 or correct.shape != confidence.shape:
        raise ValueError("confidence and correctness must be equal-length vectors")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.isin(correct, [False, True]).all():
        raise ValueError("correctness must be boolean")
    return confidence, correct.astype(bool)


def _risk_curve_arrays(confidence, correct):
    confidence, correct = _selective_inputs(confidence, correct)
    if not len(confidence):
        return confidence, np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-confidence, kind="stable")
    confidence, errors = confidence[order], np.cumsum(~correct[order])
    ends = np.r_[np.flatnonzero(confidence[1:] != confidence[:-1]), len(order) - 1]
    return confidence[ends], ends + 1, errors[ends]


def risk_coverage_curve(confidence, correct):
    thresholds, accepted, errors = _risk_curve_arrays(confidence, correct)
    return [{"threshold": float(t), "accepted": int(n), "errors": int(e),
             "coverage": float(n / accepted[-1]), "risk": float(e / n)}
            for t, n, e in zip(thresholds, accepted, errors)]


def area_under_risk_coverage(confidence, correct):
    _, accepted, errors = _risk_curve_arrays(confidence, correct)
    if not len(accepted):
        return 0.0
    return float(np.sum(np.diff(np.r_[0, accepted]) * errors / accepted) / accepted[-1])


def select_threshold(confidence, correct, budget, min_accepted=1):
    if not math.isfinite(budget) or not 0 <= budget <= 1 or min_accepted < 1:
        raise ValueError("invalid error budget or minimum accepted count")
    thresholds, accepted, errors = _risk_curve_arrays(confidence, correct)
    ok = np.flatnonzero((errors <= budget * accepted) & (accepted >= min_accepted))
    return float(thresholds[ok[-1]]) if len(ok) else None


def evaluate_threshold(confidence, correct, threshold):
    confidence, correct = _selective_inputs(confidence, correct)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1] or None for abstain-all")
    accepted = confidence >= threshold if threshold is not None else np.zeros(len(confidence), dtype=bool)
    n, errors = int(accepted.sum()), int((accepted & ~correct).sum())
    return {"threshold": threshold, "n": len(confidence), "accepted": n, "errors": errors,
            "coverage": n / len(confidence) if len(confidence) else 0.0, "risk": errors / n if n else None}


def unknowable_report(rows):
    """Confidence on records whose deciding evidence was removed (source 'unknowable') against their intact controls.
    Accuracy on the unknowable records is meaningless by construction; what is scored is whether the model knows it
    cannot know: mean max-probability and the share of records answered at >= 0.9."""
    unk = [r for r in rows if r["source"] == "unknowable"]; ctl = [r for r in rows if r["source"] == "unknowable_control"]
    if not unk: return None
    conf = lambda rs: [float(max(r["p"])) for r in rs]
    by_id = {r["id"]: r for r in ctl}
    paired = [(max(r["p"]), max(by_id[r["control_id"]]["p"])) for r in unk if r.get("control_id") in by_id]
    return {"n": len(unk), "mean_max_p": float(np.mean(conf(unk))), "share_at_0_9": float(np.mean([c >= 0.9 for c in conf(unk)])),
            "control_mean_max_p": float(np.mean(conf(ctl))) if ctl else None, "control_share_at_0_9": float(np.mean([c >= 0.9 for c in conf(ctl)])) if ctl else None,
            "control_acc": float(np.mean([int(np.argmax(r["p"]) == r["label"]) for r in ctl])) if ctl else None,
            "paired_confidence_drop": float(np.mean([c - u for u, c in paired])) if paired else None,
            "share_less_confident_than_control": float(np.mean([u < c for u, c in paired])) if paired else None}


def grouped_metrics(rows, key, temperature=1.0):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: metrics(group, temperature) for name, group in sorted(groups.items())}


def tempered_row(row, temperature):
    """The row as a calibrated predictor would have returned it: probabilities and logits at `temperature`, so metrics()
    at T=1 scores it (per-row temperatures, e.g. out-of-fold, cannot go through metrics(rows, T))."""
    return {**row, "p": probabilities_at_temperature(row, temperature).tolist(), "logits": _tempered_logits(row, temperature).tolist(),
            "inference_temperature": temperature}


def scored_rows(rows):
    """The rows metrics are computed on: clean variants of knowable records."""
    return [row for row in rows if row["variant"] == "clean" and row["source"] != "unknowable"]


def fit_temperature(rows, aggregation="macro", points=81):
    """Temperature minimizing the (micro or per-task macro) mean NLL over a `points`-point log grid on 0.25..4.
    scripts/calibrate_checkpoint.py fits the released checkpoints with points=121."""
    if aggregation not in ("micro", "macro"):
        raise ValueError("invalid calibration aggregation")
    clean = scored_rows(rows)
    if not clean:
        raise ValueError("cannot fit temperature without labelled calibration rows")
    if any(row.get("inference_temperature", 1.0) != 1.0 for row in clean):
        raise ValueError("fit temperature on raw logits, not previously calibrated outputs")
    candidates = np.exp(np.linspace(np.log(0.25), np.log(4), points))
    weights = np.ones(len(clean))
    if aggregation == "macro":
        counts = defaultdict(int)
        for row in clean:
            counts[row["task"]] += 1
        weights = np.asarray([1.0 / counts[row["task"]] for row in clean])
    losses = [np.average([nll_at_temperature(r, float(t)) for r in clean], weights=weights) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def _source_groups(rows):
    """dict[source, list[np.ndarray of row indices]]: one array per (source, group) cluster, in first-seen order."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row["source"], row["group"])].append(i)
    sources = defaultdict(list)
    for (source, _), indices in groups.items():
        sources[source].append(np.asarray(indices, dtype=int))
    return sources


def grouped_folds(rows, folds, seed):
    """Fold id per row, disjoint in (source, group) so sibling questions and variants of one record never straddle
    train/test, assigned round-robin within each source so every fold sees every source."""
    units = sorted({(row["source"], row["group"]) for row in rows})
    if len(units) < folds:
        raise ValueError("fewer distinct (source, group) units than folds")
    rng = np.random.default_rng(seed)
    fold_of, offset = {}, 0
    for source in sorted({source for source, _ in units}):
        own = [unit for unit in units if unit[0] == source]
        for i, j in enumerate(rng.permutation(len(own))):
            fold_of[own[j]] = (offset + i) % folds
        offset += len(own)
    return np.asarray([fold_of[(row["source"], row["group"])] for row in rows])


def cross_validated_temperature(rows, folds=5, seed=0, samples=1000, **fit_kwargs):
    """Out-of-fold calibration report: each fold's temperature is fit on the other folds (fit_temperature(**fit_kwargs))
    and applied to its held-out rows, which are then scored like any calibrated prediction. The bootstrap resamples
    source-stratified groups (the unit paired_bootstrap uses) and reports raw vs out-of-fold ECE with a 95% interval on
    the paired delta; `separated` is True when that interval excludes zero."""
    if folds < 2 or samples < 1:
        raise ValueError("folds must be >= 2 and samples >= 1")
    clean = scored_rows(rows)
    fold_id = grouped_folds(clean, folds, seed)
    temperatures, oof = [], list(clean)
    for fold in range(folds):
        held = fold_id == fold
        temperature = fit_temperature([row for row, out in zip(clean, held) if not out], **fit_kwargs)
        temperatures.append(temperature)
        for i in np.flatnonzero(held):
            oof[i] = tempered_row(clean[i], temperature)
    keys = ("n", "ece", "brier", "nll", "confident_error_rate", "coverage_at_5pct_error")
    raw, out_of_fold = metrics(clean), metrics(oof)
    correct = np.asarray([np.argmax(row["p"]) == row["label"] for row in clean])   # argmax is temperature-invariant
    conf = np.asarray([[max(row["p"]) for row in rows] for rows in (clean, oof)])
    sources = _source_groups(clean)
    rng = np.random.default_rng(seed)
    values = np.empty((samples, 2))
    for s in range(samples):
        drawn = np.concatenate([grouped[i] for grouped in sources.values() for i in rng.integers(0, len(grouped), size=len(grouped))])
        values[s] = [ece(conf[0][drawn], correct[drawn]), ece(conf[1][drawn], correct[drawn])]
    ci = {name: np.quantile(v, [0.025, 0.975]).tolist() for name, v in (("raw", values[:, 0]), ("out_of_fold", values[:, 1]), ("delta", values[:, 1] - values[:, 0]))}
    return {"raw": {k: raw[k] for k in keys}, "out_of_fold": {k: out_of_fold[k] for k in keys}, "ece_ci95": ci,
            "separated": bool(ci["delta"][1] < 0 or ci["delta"][0] > 0), "temperatures": temperatures,
            "folds": folds, "seed": seed, "samples": samples, "groups": sum(len(v) for v in sources.values()),
            "unit": "source-stratified (source, group); sibling questions and variants stay together"}


def paired_bootstrap(candidate, reference, samples=1000, seed=0, metric="nll", aggregation="macro"):
    nonlinear = {"coverage_at_5pct_error", "coverage_at_1pct_error", "aurc", "ece"}   # recomputed on every resample
    additive = metric not in nonlinear                                                  # else a mean of _row_scores
    if (additive and metric not in _row_scores({"p": [1.0], "label": 0})) or aggregation not in ("micro", "macro") or samples < 1:
        raise ValueError("unsupported bootstrap metric, aggregation, or sample count")

    def index(rows):
        out = {}
        for r in scored_rows(rows):
            key = r["id"], r["question"]
            if key in out:
                raise ValueError("duplicate paired example")
            out[key] = r
        return out

    a, b = index(candidate), index(reference)
    if not a or a.keys() != b.keys():
        raise ValueError("paired comparison requires identical complete clean examples")
    keys = sorted(a)
    for key in keys:
        row, other = a[key], b[key]
        if row["keys"] != other["keys"] or row["label"] != other["label"]:
            raise ValueError("paired comparison labels or option order differ")
        if any(row[field] != other[field] for field in ("source", "group", "task", "type")):
            raise ValueError("paired comparison group or task metadata differ")
    sources = _source_groups([a[key] for key in keys])
    task_names = np.asarray([a[key]["task"] for key in keys])
    statistics = []
    for indexed in (a, b):
        rows = [indexed[key] for key in keys]
        conf = np.asarray([max(r["p"]) for r in rows])
        correct = np.asarray([np.argmax(r["p"]) == r["label"] for r in rows])
        values = np.asarray([_row_scores(r)[metric] for r in rows]) if additive else None
        statistics.append((conf, correct, values))

    def statistic(indices, data):
        conf, correct, values = data
        if values is not None:
            return float(values[indices].mean())
        conf, correct = conf[indices], correct[indices]
        if metric == "aurc":
            return area_under_risk_coverage(conf, correct)
        if metric == "ece":
            return ece(conf, correct)
        return coverage_at_error(conf, correct, 0.05 if metric == "coverage_at_5pct_error" else 0.01)

    def delta(indices):
        parts = [indices] if aggregation == "micro" else [indices[task_names[indices] == t] for t in np.unique(task_names[indices])]
        return float(np.mean([statistic(part, statistics[0]) - statistic(part, statistics[1]) for part in parts]))

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        drawn = [units[i] for units in sources.values() for i in rng.integers(0, len(units), size=len(units))]
        values.append(delta(np.concatenate(drawn)))
    return {f"{aggregation}_{metric}_delta": delta(np.arange(len(keys))), "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "samples": samples, "groups": sum(len(v) for v in sources.values()), "aggregation": aggregation,
            "unit": "source-stratified original record; sibling questions stay together",
            "method": "paired cluster percentile bootstrap; full statistic recomputed in each resample"}
