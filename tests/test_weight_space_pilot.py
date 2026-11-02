"""CPU-only, synthetic tensors: no downloaded weights or evaluation partitions."""
from types import SimpleNamespace
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from kev.rounds import validate
from kev.suite import read_json
from scripts.weight_space_pilot import LAYERS, set_edit, sparse_delta, target_weights, tensor_hash


def fixture_edit():
    base = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    edited = base.clone()
    edited[0, 1] = -2
    spec = {"shape": [2, 2], "base_sha256": tensor_hash(base), "edited_sha256": tensor_hash(edited), "changed_entries": 1}
    return base, edited, spec, np.array([1], dtype=np.int32), np.array([int(edited.view(torch.int16)[0, 1].item()) & 0xffff], dtype=np.uint16)


def test_sparse_edit_rebuilds_bf16_bits():
    base, edited, spec, indices, values = fixture_edit()
    delta = sparse_delta(base, spec, indices, values)
    assert torch.equal(base.float() + delta, edited.float())
    assert torch.count_nonzero(delta) == 1


@pytest.mark.parametrize("bad", ["base", "edited", "shape", "negative", "out_of_bounds", "duplicate", "dtype", "count"])
def test_sparse_edit_rejects_invalid_artifacts(bad):
    base, _, spec, indices, values = fixture_edit()
    if bad == "base": spec["base_sha256"] = "wrong"
    if bad == "edited": spec["edited_sha256"] = "wrong"
    if bad == "shape": spec["shape"] = [4]
    if bad == "negative": indices[0] = -1
    if bad == "out_of_bounds": indices[0] = 4
    if bad == "duplicate": indices = np.array([1, 1]); values = np.repeat(values, 2)
    if bad == "dtype": values = values.astype(np.float32)
    if bad == "count": spec["changed_entries"] = 2
    with pytest.raises(ValueError):
        sparse_delta(base, spec, indices, values)


def test_edit_is_delta_transfer_not_donor_replacement_and_never_accumulates():
    base, _, spec, indices, values = fixture_edit()
    delta = sparse_delta(base, spec, indices, values)
    target = torch.nn.Parameter(base.float() + 10)
    original = target.detach().clone()
    args = ({15: target}, {15: original}, {15: delta})
    set_edit(*args, 1)
    assert torch.equal(target, original + delta)
    set_edit(*args, 0.5)
    assert torch.equal(target, original + delta / 2)
    set_edit(*args, 0)
    assert torch.equal(target, original)


def test_bad_edit_fails_before_any_parameter_changes():
    targets = {layer: torch.nn.Parameter(torch.ones(2, 2)) for layer in LAYERS}
    originals = {layer: weight.detach().clone() for layer, weight in targets.items()}
    deltas = {15: torch.ones(2, 2), 19: torch.full((2, 2), float("nan"))}
    with pytest.raises(ValueError, match="nonfinite"):
        set_edit(targets, originals, deltas, 1)
    assert all(torch.equal(targets[k], v) for k, v in originals.items())
    with pytest.raises(ValueError, match="invalid"):
        set_edit(targets, originals, deltas, float("nan"))


def test_targets_are_base_layers_not_lora_parameters():
    layers = torch.nn.ModuleDict()
    bases = {i: torch.nn.Linear(2, 2, bias=False) for i in LAYERS}
    for i in LAYERS:
        projection = torch.nn.ModuleDict({"base_layer": bases[i], "lora_A": torch.nn.Linear(2, 1, bias=False)})
        layers[str(i)] = torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({"down_proj": projection})})
    lm = torch.nn.Module()
    lm.add_module("layers", layers)
    targets = target_weights(SimpleNamespace(lm=lm))
    assert all(targets[i] is bases[i].weight for i in LAYERS)


def test_pilot_registration_is_report_only_and_development_only():
    spec = read_json("experiments/rounds/r30.json")
    assert not validate(spec, rows=False).problems
    assert all(not arm["select"] for arm in spec["arms"].values())
    assert not spec.get("confirm") and not spec["rule"]["criteria"]
    assert all("--allow-test" not in read.get("flags", "") for read in spec["reads"].values())


def test_donor_download_uses_index_and_verifies_sparse_artifacts(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from scripts import weight_space_pilot as pilot
    base = torch.zeros(1024, 3584, dtype=torch.bfloat16)
    edited = base.clone()
    edited[:, :64] = 1
    indices = (edited != base).flatten().nonzero().flatten().numpy().astype(np.int32)
    values = edited.view(torch.int16).flatten()[indices.astype(np.int64)].numpy().view(np.uint16)
    names = [f"model.language_model.layers.{layer}.mlp.down_proj.weight" for layer in LAYERS]
    manifest = {"tensors": [{"name": name, "form": "sparse", "shape": list(base.shape),
                             "base_sha256": tensor_hash(base), "edited_sha256": tensor_hash(edited),
                             "changed_entries": len(indices)} for name in names]}
    buffer = io.BytesIO()
    np.savez(buffer, t0_idx=indices, t0_val=values, t1_idx=indices, t1_val=values)
    artifacts = {"edit.npz": buffer.getvalue(), "manifest.json": json.dumps(manifest).encode()}
    monkeypatch.setattr(pilot, "FILES", {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()})
    monkeypatch.setattr(pilot.urllib.request, "urlopen", lambda url, **kwargs: io.BytesIO(artifacts[url.rsplit("/", 1)[1]]))
    shard = "model.safetensors-00001-of-00001.safetensors"
    save_file({name: base.clone() for name in names}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(names, shard)}), encoding="utf-8")
    calls = []
    def download(repo, name, revision):
        assert repo == pilot.DONOR and revision == pilot.DONOR_REVISION
        calls.append(name)
        return str(tmp_path / name)
    monkeypatch.setattr(pilot, "hf_hub_download", download)
    deltas, audit = pilot.load_deltas(tmp_path)
    assert calls == ["model.safetensors.index.json", shard, shard]
    assert all(torch.equal(delta, edited.float() - base.float()) for delta in deltas.values())
    assert all(len(entry["columns"]) == 64 for entry in audit.values())
