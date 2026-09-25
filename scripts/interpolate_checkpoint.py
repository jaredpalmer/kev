"""WiSE-FT for full-weight checkpoints: a text backbone alpha * sft + (1 - alpha) * base, with the SFT pointer head.

A full-weight SFT checkpoint (kev.train --full_ft 1) is a save_pretrained backbone plus head.pt. Interpolating its
weights with the base it was trained from trades what fine-tuning learned against what the base knew, without training
(Wortsman et al., "Robust fine-tuning of zero-shot models", 2022). Round 20 reads six such checkpoints (PLAN.md).

    uv run python scripts/interpolate_checkpoint.py --sft runs/r19-27b-lr2e6/00-trial-0/checkpoint \\
        --alphas 0.85,0.70,0.50 --out runs/r20-wise/27b-a-w{w}      # -> runs/r20-wise/27b-a-w85/checkpoint, ...
    uv run modal run modal_app.py::interpolate --sft /runs/r19-27b-lr2e6/00-trial-0/checkpoint --prefix 27b-a   # on Modal

The base is built exactly as training builds it (kev.model.DecisionModel on meta.base @ meta.base_revision in the run's
weights dtype), so its state_dict carries the names save_backbone wrote; every SFT tensor must match one base tensor in
name and shape, and every base tensor must be covered, or nothing is written. The base stays resident; the SFT side
streams tensor by tensor from its safetensors shards, and each output shard is written as soon as it is complete, with
the SFT shard's file name, tensor names, dtype and metadata, and the SFT index copied, so the result loads through
kev.checkpoint's full-weight rule unchanged. Arithmetic is fp32 (both sides upcast exactly from bf16), rounded once to
the SFT tensor's dtype. head.pt is the SFT's (same pointer head and temperature) with `interpolation` added to its meta:
{alpha, sft: {path, weights_sha256}, base: "<repo>@<revision>"}. Config and tokenizer files are copied; the SFT run's
training_config.json / training_metrics.json are not (they describe a training this checkpoint did not have). Each
checkpoint is written to <out>/checkpoint.partial and renamed to <out>/checkpoint when complete, with a report in
<out>/interpolation.json.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, read_meta, write_meta  # noqa: E402
from kev.model import DecisionModel, load_tokenizer  # noqa: E402
from kev.suite import write_json  # noqa: E402

CHUNK = 1 << 24                                                      # elements per fp32 step: bounds the temporaries (~200 MB)
NOT_COPIED = ("head.pt", "training_config.json", "training_metrics.json")   # head.pt is rewritten; the rest describe the SFT run
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def mix(sft, base, alpha):
    """alpha * sft + (1 - alpha) * base computed in fp32 and rounded once to sft's dtype."""
    if sft.shape != base.shape: raise ValueError(f"shape {tuple(sft.shape)} vs base {tuple(base.shape)}")
    out = torch.empty_like(sft)
    s, b, o = sft.reshape(-1), base.reshape(-1), out.view(-1)
    for i in range(0, s.numel(), CHUNK):
        o[i:i + CHUNK] = (alpha * s[i:i + CHUNK].float() + (1 - alpha) * b[i:i + CHUNK].float()).to(sft.dtype)
    return out


def base_backbone(meta):
    """{name: tensor} of the base backbone as kev.train builds it before training (DecisionModel, the run's weights dtype)."""
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    model = DecisionModel(meta.base, tok, "cpu", revision=meta.base_revision, head_dim=meta.head_dim, dtype=DTYPES[meta.weights_dtype])
    return model.lm.state_dict()


def check_layout(ck, base):
    """{name: shard file} of the SFT checkpoint after checking that its tensors are the base's, name for name and shape for
    shape (read from the safetensors headers; nothing is loaded)."""
    layout, shapes = {}, {}
    for shard in ck.shards():
        with safe_open(shard, "pt") as f:
            for name in f.keys():
                if name in layout: raise ValueError(f"{ck.path}: {name} is in two shards ({layout[name]}, {shard.name})")
                layout[name], shapes[name] = shard.name, tuple(f.get_slice(name).get_shape())
    only_sft, only_base = sorted(set(layout) - set(base)), sorted(set(base) - set(layout))
    wrong = sorted(n for n in set(layout) & set(base) if shapes[n] != tuple(base[n].shape))
    if only_sft or only_base or wrong:
        raise ValueError(f"{ck.path} does not match its base: {len(only_sft)} tensors only in the SFT save (e.g. {only_sft[:2]}), "
                         f"{len(only_base)} only in the base (e.g. {only_base[:2]}), {len(wrong)} with another shape (e.g. {wrong[:2]})")
    return layout


def write_checkpoint(ck, base, alpha, out, provenance):
    """One interpolated checkpoint at out (via out.partial). -> tensor count."""
    partial = out.with_name(out.name + ".partial")
    if partial.exists(): shutil.rmtree(partial)   # an earlier attempt that died before its rename: never a checkpoint
    partial.mkdir(parents=True)
    count = 0
    for shard in ck.shards():
        with safe_open(shard, "pt") as f:
            tensors = {name: mix(f.get_tensor(name), base[name], alpha) for name in f.keys()}
            save_file(tensors, partial / shard.name, metadata=f.metadata())
        count += len(tensors)
        del tensors
    for p in Path(ck.path).iterdir():
        if p.is_file() and p.name not in NOT_COPIED and not (p.name.startswith("model") and p.name.endswith(".safetensors")):
            shutil.copy2(p, partial / p.name)
    meta = read_meta(ck.path)
    meta.extra["interpolation"] = {"alpha": alpha, **provenance}
    write_meta(partial, meta)
    partial.rename(out)
    return count


def interpolate(sft, alphas, outs, base=None, revision=None, on_done=None, log=print):
    """Write one checkpoint per alpha (weight on the SFT backbone) to outs[i] (a .../checkpoint directory; its parent gets
    interpolation.json). base / revision, when given, must be the SFT run's own. on_done(report) runs after each
    checkpoint (modal_app.py commits the runs volume there). -> the reports."""
    if len(alphas) != len(outs): raise ValueError("one output per alpha")
    if any(not 0 <= a <= 1 for a in alphas): raise ValueError(f"alphas must be in [0, 1]: {alphas}")
    outs = [Path(o) for o in outs]
    if taken := [str(o) for o in outs if o.exists()]: raise FileExistsError(f"refusing to overwrite {taken}")
    ck = Checkpoint(sft)
    if not ck.full: raise ValueError(f"{sft} is a LoRA adapter; interpolate a full-weight checkpoint (kev.train --full_ft 1)")
    meta = ck.meta
    if (base or meta.base) != meta.base or (revision or meta.base_revision) != meta.base_revision:
        raise ValueError(f"{sft} was trained from {meta.base}@{meta.base_revision}, not {base}@{revision}")
    t0 = time.time()
    log(f"hashing {sft}"); provenance = {"sft": {"path": str(sft), "weights_sha256": ck.weights_sha256()}, "base": f"{meta.base}@{meta.base_revision}"}
    log(f"loading {provenance['base']} as training builds it"); backbone = base_backbone(meta)
    check_layout(ck, backbone)
    reports = []
    for alpha, out in zip(alphas, outs):
        started = time.time()
        tensors = write_checkpoint(ck, backbone, alpha, out, provenance)
        report = {"alpha": alpha, **provenance, "checkpoint": str(out), "tensors": tensors, "weights_sha256": Checkpoint(out).weights_sha256(),
                  "seconds": round(time.time() - started, 1), "formula": "fp32(alpha) * fp32(sft) + fp32(1 - alpha) * fp32(base), rounded once to the SFT dtype"}
        write_json(out.parent / "interpolation.json", report)
        log(f"alpha {alpha}: {tensors} tensors -> {out} ({report['seconds']} s)")
        reports.append(report)
        if on_done: on_done(report)
    log(f"done in {time.time() - t0:.0f} s")
    return reports


def weight_label(alpha):
    """0.85 -> "85": the {w} of an output template (round 20 names checkpoints 27b-a-w85)."""
    return f"{round(alpha * 100):02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sft", required=True, help="full-weight checkpoint directory (config.json + model*.safetensors + head.pt)")
    ap.add_argument("--alphas", required=True, help="comma-separated weights on the SFT backbone, e.g. 0.85,0.70,0.50")
    ap.add_argument("--out", required=True, help="output directory template with {w} (alpha x 100); the checkpoint goes to <out>/checkpoint")
    ap.add_argument("--base", help="refuse unless the SFT run's base is this (default: head.pt's)")
    ap.add_argument("--revision", help="refuse unless the SFT run's base revision is this (default: head.pt's)")
    a = ap.parse_args()
    if "{w}" not in a.out: ap.error("--out needs {w}")
    alphas = [float(x) for x in a.alphas.split(",")]
    interpolate(a.sft, alphas, [Path(a.out.format(w=weight_label(x))) / "checkpoint" for x in alphas], a.base, a.revision)


if __name__ == "__main__":
    main()
