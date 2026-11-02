"""Transfer Experiment 8's sparse weight delta to Kev, without training or publishing.

Run through modal_app.py::script; registration: experiments/rounds/r30.json.
The donor is instruction-tuned Qwen, NOT Kev's base: add its verified edit delta,
never replace Kev's columns with the donor's weights. This is a transfer pilot,
not a reproduction of the paper's math experiment or a Kev release selection.
"""
import argparse
import hashlib
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.experiment import mechanism_checks, source_hashes
from kev.metrics import paired_bootstrap
from kev.predictors import LocalPredictor
from kev.suite import CONTEXT, digest, load_split, read_json, read_manifest, write_json

CHECKPOINT = "jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
DONOR = "Qwen/Qwen3.5-0.8B"
DONOR_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LOG_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
EDIT_URL = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{LOG_REVISION}/exp8_math/runs/p50c/edit"
FILES = {"edit.npz": "e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7",
         "manifest.json": "83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225"}
SUITES = {"development": "evals/v7/decision-v7", "transfer": "evals/v4/transfer-v4",
          "hard": "evals/hard-v1", "documents": "evals/documents-v1"}
LAYERS = (15, 19)
ARMS = {"baseline": 0.0, "paper": 1.0, "half": 0.5}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sparse_delta(base, spec, indices, values):
    """Verify the published bf16 edit bit for bit before returning its fp32 delta."""
    if base.dtype != torch.bfloat16 or list(base.shape) != spec["shape"] or tensor_hash(base) != spec["base_sha256"]:
        raise ValueError("donor tensor does not match the published base")
    if (indices.ndim != 1 or values.ndim != 1 or indices.dtype.kind not in "iu" or values.dtype != np.uint16
            or len(indices) != len(values) or len(np.unique(indices)) != len(indices)
            or len(indices) != spec["changed_entries"] or (indices < 0).any() or (indices >= base.numel()).any()):
        raise ValueError("invalid sparse edit indices/values")
    edited = base.clone()
    edited.view(torch.int16).reshape(-1)[torch.from_numpy(indices.astype(np.int64))] = torch.from_numpy(values.view(np.int16).copy())
    if tensor_hash(edited) != spec["edited_sha256"] or not torch.isfinite(edited).all():
        raise ValueError("rebuilt donor does not match the published edit")
    return edited.float() - base.float()


def load_deltas(out):
    for name, expected in FILES.items():
        with urllib.request.urlopen(f"{EDIT_URL}/{name}", timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"published artifact hash mismatch: {name}")
        (out / name).write_bytes(data)
    manifest = read_json(out / "manifest.json")
    index = read_json(hf_hub_download(DONOR, "model.safetensors.index.json", revision=DONOR_REVISION))["weight_map"]
    deltas, donor_audit = {}, {}
    with np.load(io.BytesIO((out / "edit.npz").read_bytes()), allow_pickle=False) as archive:
        for i, (layer, spec) in enumerate(zip(LAYERS, manifest["tensors"], strict=True)):
            name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            if spec["name"] != name or spec["form"] != "sparse" or spec["shape"] != [1024, 3584]:
                raise ValueError("unexpected published edit layout")
            path = hf_hub_download(DONOR, index[name], revision=DONOR_REVISION)
            with safe_open(path, framework="pt") as weights:
                base = weights.get_tensor(name)
            delta = sparse_delta(base, spec, archive[f"t{i}_idx"], archive[f"t{i}_val"])
            columns = (delta != 0).any(dim=0).nonzero().flatten().tolist()
            if len(columns) != 64:
                raise ValueError("expected 64 edited columns per layer")
            deltas[layer] = delta
            donor_audit[str(layer)] = {**spec, "columns": columns, "delta_sha256": tensor_hash(delta)}
    save_file({str(k): v for k, v in deltas.items()}, str(out / "delta.safetensors"))
    return deltas, donor_audit


def target_weights(model):
    targets = {}
    for layer in LAYERS:
        matches = [module for name, module in model.lm.named_modules() if name.endswith(f"layers.{layer}.mlp.down_proj")]
        if len(matches) != 1:
            raise ValueError(f"expected one down_proj for layer {layer}")
        module = matches[0]
        targets[layer] = getattr(module, "base_layer", module).weight
    return targets


@torch.no_grad()
def set_edit(targets, originals, deltas, scale):
    if not np.isfinite(scale) or not 0 <= scale <= 1 or targets.keys() != originals.keys() or targets.keys() != deltas.keys():
        raise ValueError("invalid weight edit")
    proposed = {}
    for layer, weight in targets.items():
        original, delta = originals[layer], deltas[layer]
        if weight.shape != delta.shape or weight.shape != original.shape or weight.dtype != torch.float32:
            raise ValueError("expected shape-matched fp32 Kev weights")
        proposed[layer] = original.to(weight.device) + scale * delta.to(weight.device)
        if not torch.isfinite(proposed[layer]).all():
            raise ValueError("nonfinite weight edit")
    for layer, weight in targets.items():
        weight.copy_(proposed[layer])


def parameter_hashes(model):
    return {name: tensor_hash(value) for name, value in model.named_parameters()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deltas, donor_audit = load_deltas(out)
    predictor = LocalPredictor(CHECKPOINT, "cuda", LoadOptions(merge=False))
    meta = predictor.checkpoint.meta
    if meta.base != "Qwen/Qwen3.5-0.8B-Base" or meta.base_revision != "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68":
        raise ValueError("unexpected Kev base")
    model = predictor.model
    targets = target_weights(model)
    originals = {layer: weight.detach().clone() for layer, weight in targets.items()}
    initial_hashes = parameter_hashes(model)
    target_names = {name for name, value in model.named_parameters() if any(value is weight for weight in targets.values())}
    if len(target_names) != 2:
        raise ValueError("expected exactly two target parameters")
    suites = {tag: (load_split(path, "development"), read_manifest(path)) for tag, path in SUITES.items()}
    report = {"checkpoint": CHECKPOINT, "base": meta.base, "base_revision": meta.base_revision,
              "donor": DONOR, "donor_revision": DONOR_REVISION, "logs_revision": LOG_REVISION,
              "artifact_sha256": FILES, "donor_tensors": donor_audit, "environment": predictor.environment,
              "temperature": predictor.temperature, "temperature_policy": "unchanged shipped T, no fitting",
              "script_sha256": digest(Path(__file__)), "kev_source_hashes": source_hashes(), "arms": {}, "comparisons": {},
              "scope": "development-only transfer pilot; no training, test read, release selection or publishing"}
    all_rows = {}
    for arm, scale in ARMS.items():
        set_edit(targets, originals, deltas, scale)
        current = parameter_hashes(model)
        changed = {name for name in current if current[name] != initial_hashes[name]}
        if changed != (target_names if scale else set()):
            raise ValueError(f"unexpected changed parameters: {changed}")
        audit = {str(layer): {"before_sha256": tensor_hash(originals[layer]), "after_sha256": tensor_hash(weight),
                             "changed_entries": int((weight != originals[layer]).sum()),
                             "relative_delta_norm": float((weight - originals[layer]).norm() / originals[layer].norm())}
                 for layer, weight in targets.items()}
        report["arms"][arm] = {"scale": scale, "weights": audit, "unchanged_other_parameters": len(current) - len(changed), "reads": {}}
        all_rows[arm] = {}
        for tag, (records, manifest) in suites.items():
            predictor.context = manifest.get("context", CONTEXT)
            print(f"{arm}/{tag}: {len(records)} records", flush=True)
            directory = out / arm / tag
            result, rows = evaluate_records(records, predictor, directory, heldout_sources=manifest["holdout_sources"])
            result.update(run=CHECKPOINT, weight_edit_scale=scale, suite=SUITES[tag], split="development",
                          suite_sha256=digest(Path(SUITES[tag]) / "manifest.json"), environment=predictor.environment)
            if tag == "transfer":
                result["mechanism_checks"] = mechanism_checks(records, predictor)
                if not result["mechanism_checks"]["passed"]:
                    raise ValueError("isolation check failed")
            write_json(directory / "report.json", result)
            report["arms"][arm]["reads"][tag] = result["clean"]
            all_rows[arm][tag] = rows
            write_json(out / "progress.json", report)
        write_json(out / arm / "provenance.json", {"suite_sha256": digest(Path(SUITES["development"]) / "manifest.json"),
                                                   "parent": CHECKPOINT, "weight_edit_scale": scale, "trained": False})
    set_edit(targets, originals, deltas, 0)
    if parameter_hashes(model) != initial_hashes:
        raise ValueError("original weights were not restored exactly")
    report["restored_exactly"] = True
    for arm in ("paper", "half"):
        report["comparisons"][arm] = {}
        for tag in SUITES:
            if tag == "development":
                continue
            report["comparisons"][arm][tag] = {
                metric: paired_bootstrap(all_rows[arm][tag], all_rows["baseline"][tag], samples=2000,
                                         seed=0, metric=metric, aggregation="micro")
                for metric in ("acc", "brier", "ece", "nll")}
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "report.json", report)
    print(f"Finished in {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
