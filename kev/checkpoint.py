"""Trained checkpoints: a run directory or a Hub repo holding a LoRA adapter, `head.pt` and the tokenizer.

This is the one place that knows the layout of `head.pt` and how a checkpoint becomes a `DecisionModel`:
`kev.serve`, `kev.benchmark`, `kev.train --init_from`, `kev.publish`, the scripts and the Hugging Face Space all go
through it. The Space vendors this file next to `model.py` and `api.py` (scripts/publish_space.sh), so it must not
import the data or suite modules at import time.

    ck = Checkpoint("jaredpalmer/kev-4b")          # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, default_attn, is_hybrid, load_tokenizer, pad_id

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")


def is_hub_id(run):
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """Local run directory as given, or a Hub repo id like jaredpalmer/kev-4b, optionally pinned to a revision or tag
    with `@` (jaredpalmer/kev-4b@qwen3), downloaded to the HF cache. Returns a str path."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


@dataclass
class Meta:
    """Contents of `head.pt`. Every reader gets the same defaults for fields older checkpoints did not write.
    `extra` keeps the rest of the file (training args, suite hash, init provenance, temperature fit) so a
    read-modify-write round trip loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "weights_dtype", "temperature", "holdout")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.KNOWN if k in d}, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}   # known fields win over a stray key in extra


def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu"))


def write_meta(run, meta):
    torch.save(meta.to_dict(), f"{run}/head.pt")


@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint is turned into a model. Defaults are the exact path every reported number uses; the fields
    are the same knobs the KEV_* environment variables expose to the command-line tools (see from_env).

    dtype        None = fp32, the exact path every reported number uses (bf16 when the checkpoint was trained with a bf16
                 backbone). kev.serve defaults to bf16 on CUDA and MPS instead: half the memory, 2-4.5x lower latency on an
                 L4 (Kev-4B: 209 -> 118 ms at 101 tokens, 850 -> 189 ms at 330 tokens), probabilities within ~0.01 and
                 the same argmax on the checks run so far. KEV_DTYPE=fp32 restores the exact path when serving.
    merge        fold the LoRA into the base weights in fp32 before any cast. Exact in fp32; in bf16 it is faster (~15%)
                 and closer to the fp32 numbers than the unmerged adapter (kev-4b, 24 dev records: max |dp| 0.017 vs
                 0.029, 0 vs 1 argmax flips). Ignored for adapters that carry trained token embeddings.
    merge_device where the fp32 backbone is loaded and merged; None = the serving device. "cpu" keeps the fp32 copy in host
                 memory and moves only the cast model, so the GPU never holds more than the serving dtype: Kev-4B in bf16
                 fits a 16 GB card this way, while the fp32 merge on the card runs out of memory. Same weights, slower
                 load. Ignored when nothing is merged.
    attn         attention backend; None = the model default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS measured
                 parity with eager and is a few percent faster.
    lora_scale   WiSE-FT-style interpolation between base (0) and fine-tuned weights (1), at inference.
    temperature  None = the temperature the checkpoint carries (fitted by scripts/calibrate_checkpoint.py); 1.0 = raw logits.
    backend      None = torch, the path every reported number uses. "mlx" = kev.mlx_model (Metal kernels for the hybrid
                 Qwen3.5 backbones through mlx-lm; the pointer head and encoder are shared; refused for attention-only
                 bases, which MPS already runs well). "auto" = mlx when the device is mps, the base is hybrid, mlx-lm is
                 installed and fp32 was not asked for (an explicit dtype=float32 means "the exact path"), else torch;
                 kev.serve uses auto. The MLX path always merges the adapter and ignores `attn` and `dtype` (the backbone
                 runs as stored, bf16).
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    merge_device: str | None = None
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    backend: str | None = None

    BACKENDS = (None, "torch", "mlx", "auto")

    @classmethod
    def from_env(cls, env=os.environ):
        """KEV_DTYPE=bf16|fp16|fp32, KEV_MERGE=0, KEV_MERGE_DEVICE=cpu, KEV_ATTN=sdpa|eager, KEV_LORA_SCALE, KEV_TEMPERATURE, KEV_BACKEND=torch|mlx|auto.
        For command-line entry points only; library code passes an explicit LoadOptions. Explicit values that equal a
        library default are kept (fp32 as torch.float32, "torch" as a string) so a caller with its own default, like
        kev.serve, can tell "asked for it" from "did not say"."""
        backend = env.get("KEV_BACKEND") or None
        if backend not in cls.BACKENDS: raise ValueError(f"KEV_BACKEND must be one of torch, mlx, auto; got {backend!r}")
        return cls(dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(env.get("KEV_DTYPE", "")),
                   merge=env.get("KEV_MERGE", "1") != "0", merge_device=env.get("KEV_MERGE_DEVICE") or None,
                   attn=env.get("KEV_ATTN") or None,
                   lora_scale=float(env.get("KEV_LORA_SCALE", "1")),
                   temperature=float(env["KEV_TEMPERATURE"]) if env.get("KEV_TEMPERATURE") else None, backend=backend)


def mlx_available():
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # what the caller asked for (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.meta = read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    def release_date(self):
        """ISO date for the TypeSafe model card: the Hub commit date for a Hub checkpoint (falls back to the cached file's
        date offline), the time head.pt was written for a local run."""
        if is_hub_id(self.requested):
            from huggingface_hub import HfApi
            repo, _, revision = self.requested.partition("@")
            try:
                return HfApi().model_info(repo, revision=revision or None).last_modified.date().isoformat()
            except Exception:
                pass
        return datetime.date.fromtimestamp(self.file("head.pt").stat().st_mtime).isoformat()

    def hybrid_base(self):
        """Whether the base has Gated DeltaNet layers (Qwen3.5), read from its config without loading weights."""
        from transformers import AutoConfig
        return is_hybrid(AutoConfig.from_pretrained(self.meta.base, revision=self.meta.base_revision).get_text_config())

    def backend(self, device, opts=LoadOptions()):
        """The backend `load` will use: LoadOptions.backend resolved ("auto" -> mlx only where it pays and is installed)."""
        if opts.backend not in LoadOptions.BACKENDS: raise ValueError(f"unknown backend {opts.backend!r}")
        if opts.backend != "auto": return opts.backend or "torch"
        exact = opts.dtype is torch.float32   # KEV_DTYPE=fp32: the caller wants the reported-numbers path, not a faster one
        return "mlx" if str(device) == "mps" and not exact and mlx_available() and self.hybrid_base() else "torch"

    def load(self, device, opts=LoadOptions()):
        """-> (tokenizer, model) in eval mode with the LoRA applied and the pointer head loaded. The model is a
        DecisionModel (torch) or an MLXDecisionModel (backend mlx); both expose the same scoring interface."""
        meta = self.meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        m = self._load_mlx(tok, opts) if self.backend(device, opts) == "mlx" else self._load_torch(tok, device, opts)
        m.head.load_state_dict(meta.head); m.eval()
        m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, m

    def _load_mlx(self, tok, opts):
        from .mlx_model import MLXDecisionModel, merge_lora
        if not opts.merge: raise ValueError("the MLX backend always merges the adapter (KEV_MERGE=0 needs backend=torch)")
        if self.meta.option_isolation: raise ValueError("option_isolation needs the packed mask; not available on the MLX backend")
        if not self.hybrid_base(): raise ValueError(f"the MLX backend is for the hybrid (Qwen3.5) bases; {self.meta.base} is attention-only and runs on MPS with backend=torch")
        base_dir = resolve_run(f"{self.meta.base}@{self.meta.base_revision or ''}")   # the base snapshot the torch path already cached
        m = MLXDecisionModel(base_dir, pad_id(tok), head_dim=self.meta.head_dim)
        merge_lora(m.lm, self.path, opts.lora_scale)
        return m

    def _load_torch(self, tok, device, opts):
        from peft import PeftModel
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # trained with a bf16 backbone (--weights_dtype bf16, e.g. the 35B-A3B MoE whose fused experts need bf16): load it
            # the same way and keep the fp32 adapter unmerged rather than folding it into bf16 weights.
            dtype, merge = torch.bfloat16, False
        merge = merge and not self.adapter_config().get("trainable_token_indices")   # token-trained adapters stay unmerged
        stage = opts.merge_device if merge and opts.merge_device else device   # where the fp32 weights live until the cast
        m = DecisionModel(meta.base, tok, stage, lora=None, revision=meta.base_revision, head_dim=meta.head_dim,
                          option_isolation=meta.option_isolation, dtype=torch.float32 if merge else dtype,
                          attn=opts.attn or default_attn(device))
        m.lm = PeftModel.from_pretrained(m.lm, self.path, torch_device=str(stage)).to(stage)   # trainable token embeddings, if any, live in the adapter
        if opts.lora_scale != 1:
            for module in m.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for k in module.scaling: module.scaling[k] *= opts.lora_scale
            m.lora_scale = opts.lora_scale
        if merge: m.lm = m.lm.merge_and_unload()     # in fp32: exact
        if dtype != torch.float32: m.lm = m.lm.to(dtype)
        if str(stage) != str(device): m.to(device); m.device = device
        return m

    COMPAT_FIELDS = ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings")

    def warm_start(self, model, ours):
        """Delta training: load this checkpoint's adapter and pointer head into `model` (a fresh DecisionModel built with
        LoRA). `ours` is the Meta the new run will save; every architecture field is compared BEFORE loading, because peft
        loads matching keys silently and a half-loaded adapter still trains and still reports a loss. Returns provenance."""
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from .suite import digest   # lazy: the Space vendors this module without kev/suite.py
        for name in self.COMPAT_FIELDS:
            theirs, mine = getattr(self.meta, name), getattr(ours, name)
            if theirs != mine and not (name == "base_revision" and None in (theirs, mine)):
                raise ValueError(f"--init_from {self.path}: {name} is {theirs!r} there and {mine!r} here")
        weights = load_peft_weights(self.path, device="cpu")
        have = set(get_peft_model_state_dict(model.lm))
        unexpected, missing = sorted(set(weights) - have), sorted(have - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {self.path} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(model.lm, weights)
        model.head.load_state_dict(self.meta.head)
        return {"init_from": self.requested, "resolved": self.path, "adapter_sha256": digest(self.file("adapter_model.safetensors")),
                "head_sha256": digest(self.file("head.pt")), "adapter_tensors": len(weights)}


def load(run, device, opts=LoadOptions()):
    """Convenience: Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
