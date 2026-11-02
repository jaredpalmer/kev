"""One-shot, supervised pointer-channel reweighting; no backbone updates or backprop.

Round 31: fit bounded channel gains on training records, fold them into head.q,
then read development-only panels at the unchanged shipped temperature.
"""
import argparse
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.data import materialize
from kev.experiment import mechanism_checks, source_hashes
from kev.metrics import paired_bootstrap
from kev.predictors import LocalPredictor
from kev.suite import (digest, load_split, read_manifest, record_digest, semantic_hash,
                       text_digest, validate_training, write_json)
from scripts.weight_space_pilot import CHECKPOINT, SUITES, parameter_hashes, tensor_hash

FIT_SUITES = ("development", "hard", "documents")
FIT_RECORDS = 512
RIDGE = 0.1
MAX_GAIN_CHANGE = 0.25
ARMS = ("baseline", "learned", "half", "shuffled")


def identities(record):
    meta, state = record["_meta"], record["state"]
    state_hash = text_digest(state) if isinstance(state, str) else semantic_hash(record)
    return {("id", meta["id"]), ("group", meta["group_id"]), ("state", state_hash)}


def fit_sample(records, excluded, count=FIT_RECORDS):
    """Label-independent, source-balanced sample; at most one record per original group."""
    buckets = defaultdict(list)
    for record in sorted(records, key=lambda r: record_digest(r["_meta"]["id"])):
        if record["_meta"]["variant"] == "clean":
            buckets[record["_meta"]["source"]].append(record)
    seen, selected = set(excluded), []
    for row in zip_longest(*(buckets[k] for k in sorted(buckets))):
        for record in row:
            if record is None or identities(record) & seen:
                continue
            selected.append(record)
            seen.update(identities(record))
            if len(selected) == count:
                return selected
    raise ValueError(f"only {len(selected)}/{count} disjoint training records")


@torch.no_grad()
def capture_features(predictor, record):
    """The actual head inputs from LocalPredictor, including its long-row kernel policy."""
    features = []

    def capture(head, inputs):
        decide, options = inputs
        q, k = head.q(decide).double(), head.k(options).double()
        features.append((k * q * head.scale / head.temperature).cpu())

    hook = predictor.model.head.register_forward_pre_hook(capture)
    try:
        prediction = predictor(record)
    finally:
        hook.remove()
    questions = materialize(record)["questions"]
    if len(features) != len(questions):
        raise ValueError("head feature capture missed a question")
    result = []
    for x, question in zip(features, questions, strict=True):
        z = torch.tensor([prediction["logits"][question["qid"]][k] for k in question["keys"]], dtype=torch.float64)
        if not torch.allclose(x.sum(-1), z, atol=2e-5, rtol=2e-5):
            raise ValueError("captured features disagree with actual head logits")
        target = torch.tensor(question["target"], dtype=torch.float64) if "target" in question else torch.eye(len(z), dtype=torch.float64)[question["label"]]
        result.append((x, z, target))
    return result


def curvature(features, logits, target):
    """Exact gradient/curvature of softmax log loss with respect to multiplicative gains."""
    if (features.ndim != 2 or logits.shape != (len(features),) or target.shape != logits.shape
            or not all(torch.isfinite(t).all() for t in (features, logits, target))
            or (target < 0).any() or not torch.isclose(target.sum(), target.new_tensor(1.0))):
        raise ValueError("invalid head-fit features or targets")
    p = logits.softmax(-1)
    centered = features - p @ features
    return features.T @ (target - p), centered.T @ (p[:, None] * centered)


def solve_gains(rhs, hessian, ridge=RIDGE, bound=MAX_GAIN_CHANGE):
    """A damped Newton solve, restricted to mean-zero changes (not scalar temperature)."""
    d = len(rhs)
    if (d < 2 or hessian.shape != (d, d) or not torch.isfinite(rhs).all()
            or not torch.isfinite(hessian).all() or not np.isfinite(ridge) or ridge <= 0
            or not np.isfinite(bound) or not 0 < bound < 1):
        raise ValueError("invalid bounded head solve")
    damping = ridge * hessian.trace() / d
    if damping <= 0:
        raise ValueError("head features have no curvature")
    a = hessian + damping * torch.eye(d, dtype=hessian.dtype, device=hessian.device)
    ones = torch.ones_like(rhs)
    solved = torch.linalg.solve(a, torch.stack((rhs, ones), dim=1))
    delta = solved[:, 0] - solved[:, 1] * solved[:, 0].sum() / solved[:, 1].sum()
    delta *= min(1.0, bound / max(float(delta.abs().max()), 1e-30))
    if not torch.isfinite(delta).all() or abs(float(delta.mean())) > 1e-10:
        raise ValueError("invalid mean-zero solution")
    return delta, float(damping)


@torch.no_grad()
def apply_gains(head, originals, delta):
    if (delta.shape != (head.q.out_features,) or not torch.isfinite(delta).all()
            or float(delta.abs().max()) > MAX_GAIN_CHANGE + 1e-8):
        raise ValueError("invalid pointer gains")
    gain = (1 + delta).to(head.q.weight)
    head.q.weight.copy_(originals["weight"] * gain[:, None])
    head.q.bias.copy_(originals["bias"] * gain)


def fit_head(predictor, samples, out):
    d = predictor.model.head.q.out_features
    rhs, hessian = torch.zeros(d, dtype=torch.float64), torch.zeros((d, d), dtype=torch.float64)
    cache, selection = [], []
    for tag, records in samples.items():
        predictor.context = read_manifest(SUITES[tag])["context"]
        for i, record in enumerate(records):
            entries = capture_features(predictor, record)
            weight = 1 / (len(samples) * len(records) * len(entries))
            for x, z, target in entries:
                r, h = curvature(x, z, target)
                rhs += weight * r
                hessian += weight * h
                cache.append((x, z, target, weight))
            selection.append({"suite": SUITES[tag], "id": record["_meta"]["id"], "group": record["_meta"]["group_id"],
                              "record_sha256": record_digest(record), "source": record["_meta"]["source"]})
            if (i + 1) % 50 == 0:
                print(f"fit/{tag}: {i+1}/{len(records)}", flush=True)
    delta, damping = solve_gains(rhs, hessian)
    fit_loss = {arm: sum(w * float(-(target * (z + x @ (delta * scale)).log_softmax(-1)).sum())
                        for x, z, target, w in cache) for arm, scale in (("baseline", 0), ("learned", 1), ("half", 0.5))}
    save_file({"delta": delta, "rhs": rhs, "hessian": hessian}, str(out / "fit.safetensors"))
    write_json(out / "fit-selection.json", selection)
    return delta, {"records": len(selection), "questions": len(cache), "fitted_parameters": d - 1,
                   "ridge": RIDGE, "damping": damping, "max_gain_change": MAX_GAIN_CHANGE,
                   "delta": delta.tolist(), "weighted_in_sample_nll": fit_loss,
                   "selection_sha256": digest(out / "fit-selection.json"),
                   "statistics_sha256": digest(out / "fit.safetensors")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    suites = {tag: (load_split(path, "development"), read_manifest(path)) for tag, path in SUITES.items()}
    excluded = {key for records, _ in suites.values() for record in records for key in identities(record)}
    samples = {}
    for tag in FIT_SUITES:
        training = load_split(SUITES[tag], "train")
        validate_training(training, read_manifest(SUITES[tag]))
        samples[tag] = fit_sample(training, excluded)
        excluded.update(key for record in samples[tag] for key in identities(record))
    predictor = LocalPredictor(CHECKPOINT, "cuda", LoadOptions(merge=False))
    model = predictor.model
    if not isinstance(model, torch.nn.Module):
        raise ValueError("head edit requires the torch backend")
    initial = parameter_hashes(model)
    head = model.head
    originals = {"weight": head.q.weight.detach().clone(), "bias": head.q.bias.detach().clone()}
    report = {"checkpoint": CHECKPOINT, "environment": predictor.environment, "temperature": predictor.temperature,
              "temperature_policy": "shipped temperature unchanged; no temperature fitting in the primary",
              "script_sha256": digest(Path(__file__)), "kev_source_hashes": source_hashes(),
              "weight_audit_source_sha256": digest(Path(__file__).with_name("weight_space_pilot.py")),
              "fit_inputs": {tag: {"suite": SUITES[tag], "partition": "train", "manifest_sha256": digest(Path(SUITES[tag]) / "manifest.json"),
                                   "train_sha256": digest(Path(SUITES[tag]) / "train.jsonl")} for tag in FIT_SUITES},
              "scope": "supervised one-shot head fitting, not data-free; no backbone fine-tuning, locked tests or publishing",
              "arms": {}, "comparisons": {}}
    try:
        delta, report["fit"] = fit_head(predictor, samples, out)
        if parameter_hashes(model) != initial:
            raise ValueError("feature capture changed model parameters")
        gains = {"baseline": torch.zeros_like(delta), "learned": delta, "half": delta / 2,
                 "shuffled": delta[torch.from_numpy(np.random.default_rng(0).permutation(len(delta)))].clone()}
        save_file(gains, str(out / "gains.safetensors"))
        write_json(out / "fit.json", report)
        all_rows = {}
        for arm, edit in gains.items():
            apply_gains(head, originals, edit)
            current = parameter_hashes(model)
            changed = {name for name in current if current[name] != initial[name]}
            if changed - {"head.q.weight", "head.q.bias"} or (arm == "baseline" and changed):
                raise ValueError(f"unexpected changed weights: {changed}")
            report["arms"][arm] = {"changed_parameters": sorted(changed), "unchanged_parameters": len(current) - len(changed),
                "weights": {name: {"before_sha256": initial[f"head.q.{name}"], "after_sha256": tensor_hash(value),
                                   "relative_delta_norm": float((value - originals[name]).norm() / originals[name].norm())}
                            for name, value in (("weight", head.q.weight), ("bias", head.q.bias))}, "reads": {}}
            all_rows[arm] = {}
            for tag, (records, manifest) in suites.items():
                predictor.context = manifest["context"]
                directory = out / arm / tag
                print(f"{arm}/{tag}: {len(records)} records", flush=True)
                result, rows = evaluate_records(records, predictor, directory, heldout_sources=manifest["holdout_sources"])
                result.update(run=CHECKPOINT, head_edit=arm, suite=SUITES[tag], split="development",
                              suite_sha256=digest(Path(SUITES[tag]) / "manifest.json"), environment=predictor.environment)
                if tag == "transfer":
                    result["mechanism_checks"] = mechanism_checks(records, predictor)
                    if not result["mechanism_checks"]["passed"]:
                        raise ValueError("edited head violates isolation")
                write_json(directory / "report.json", result)
                report["arms"][arm]["reads"][tag] = result["clean"]
                all_rows[arm][tag] = rows
                write_json(out / "progress.json", report)
            write_json(out / arm / "provenance.json", {"parent": CHECKPOINT, "fit_inputs": report["fit_inputs"],
                       "suite_sha256": digest(Path(SUITES["development"]) / "manifest.json"),
                       "trained": arm != "baseline", "training_method": "one-shot supervised head-channel solve", "backbone_trained": False})
        for arm in ARMS[1:]:
            report["comparisons"][arm] = {tag: {metric: paired_bootstrap(all_rows[arm][tag], all_rows["baseline"][tag],
                samples=2000, seed=0, metric=metric, aggregation="micro") for metric in ("acc", "brier", "ece", "nll")}
                for tag in SUITES if tag != "development"}
        report["learned_vs_shuffled_hard"] = paired_bootstrap(all_rows["learned"]["hard"], all_rows["shuffled"]["hard"],
            samples=2000, seed=0, metric="acc", aggregation="micro")
    finally:
        apply_gains(head, originals, torch.zeros(head.q.out_features, dtype=torch.float64))
        if parameter_hashes(model) != initial:
            raise ValueError("original model not restored exactly")
    report["restored_exactly"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "report.json", report)
    print(f"Finished in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
