"""Apple Silicon backend for the Qwen3.5 checkpoints: mlx-lm's Metal implementation of the hybrid backbone under Kev's
own encoder and pointer head.

MPS has no Gated DeltaNet kernels, so the PyTorch path runs reference code there (Kev-4B ~0.8 s per request). This module
runs the same computation on Metal through mlx-lm and keeps everything Kev-specific unchanged: `kev.model.encode` builds the
tokens, `rows_of` splits them into one causal row per question (the hybrid form the torch path uses too), and the readout is
the very same `PointerHead` (fp32, with the checkpoint's temperature) applied to the branch hidden states.

Same contract as `DecisionModel` for serving and scoring: encode / forward / probs / probs_and_prefix / probs_with_prefix /
head / dtype. Selected by `LoadOptions(backend="mlx")` (or "auto" on Apple Silicon) in `kev.checkpoint`; never by the
benchmark, whose reported numbers stay on the fp32 torch path. Parity against that path is measured in
tests/test_mlx.py (max |dp| and argmax flips on development records, prefix vs full pass, one question vs several).

Two ways in, both through kev.checkpoint: a LoRA checkpoint is its base's mlx-lm model (`load_base`) with the adapter
folded in (`merge_lora`); a full-weight checkpoint (kev.train --full_ft, scripts/merge_lora_checkpoint.py) is built from its
own config.json and shards (`load_full`), nothing merged, so the only resident copy is the weights themselves.
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from .model import PointerHead, encode, probs_one, row_groups, rows_of

CACHE_LIMIT = 1 << 30   # MLX's buffer cache keeps a buffer per new request shape; Kev-4B on an M5, 50 requests: 1.0 GB cached vs 3.7 GB unbounded, same latency
MLX_DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}   # config.json's dtype names (kev.checkpoint.Checkpoint.saved_dtype) -> MLX
FULL_MODEL_TYPE = "qwen3_5_text"   # what save_pretrained of transformers' Qwen3_5TextModel writes (Qwen3.5 and Qwen3.8 alike)
FULL_PREFIX = "language_model.model."   # the text backbone inside mlx-lm's qwen3_5.Model; a full checkpoint's names carry no prefix
PREFILL_CHUNK = 1024    # state tokens per prefix pass (MLXDecisionModel.prefix); on an M5 1,024 / 2,048 / 4,096 / one pass take the same time, the peak grows with the chunk


def load_base(base_dir):
    """mlx-lm's model of a Hub base snapshot, weights as stored (bf16 for the Qwen3.5 bases); the LoRA path merges into it."""
    mx.set_cache_limit(CACHE_LIMIT)
    return load_model(Path(base_dir))[0]


def load_full(ckpt_dir, shards, dtype):
    """mlx-lm's Qwen3.5 model holding a full-weight checkpoint's backbone: the architecture from the checkpoint's own
    config.json (the text model's config, model_type qwen3_5_text, which mlx-lm's qwen3_5.ModelArgs takes as its
    text_config), every tensor from `shards` in `dtype` (config.json's name: "bfloat16" for every kev.train --full_ft run).
    Nothing is merged or cast: the model holds exactly the saved values after mlx-lm's own layout rules for a
    transformers export (qwen3_5 sanitize: the DeltaNet conv kernel moved to MLX's axis order and the zero-centred RMSNorm
    weights stored as 1 + w, as for the Hub bases the LoRA path loads). Strict: every name and shape must match the model
    exactly and every tensor must be `dtype`, else ValueError. The LM head is removed before any weight is materialized:
    Kev reads hidden states, and the checkpoint carries none (vision tower and MTP were never loaded either)."""
    ckpt_dir = Path(ckpt_dir)
    config = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != FULL_MODEL_TYPE or "text_config" in config:
        raise ValueError(f"{ckpt_dir}: the MLX backend loads full-weight checkpoints of transformers' Qwen3_5TextModel "
                         f"(model_type {FULL_MODEL_TYPE!r}); config.json has model_type {config.get('model_type')!r}")
    if dtype not in MLX_DTYPES: raise ValueError(f"{ckpt_dir}: no MLX dtype for weights stored as {dtype!r}")
    want = MLX_DTYPES[dtype]
    args = qwen3_5.ModelArgs.from_dict(config)
    text = qwen3_5.TextModelArgs.from_dict(args.text_config)
    pattern = ["full_attention" if (i + 1) % text.full_attention_interval == 0 else "linear_attention" for i in range(text.num_hidden_layers)]
    if config.get("layer_types", pattern) != pattern:   # mlx-lm places attention layers by full_attention_interval alone
        raise ValueError(f"{ckpt_dir}: config.json's layer_types are not every {text.full_attention_interval}th layer full attention, "
                         f"the only layout mlx-lm's qwen3_5 builds")
    mx.set_cache_limit(CACHE_LIMIT)
    lm = qwen3_5.Model(args)    # parameters are lazy: nothing is allocated until evaluated
    if "lm_head" in lm.language_model: del lm.language_model["lm_head"]   # untied bases (Qwen3.8-27B): 2.5 GB never read
    weights = {}
    for shard in shards:
        for name, value in mx.load(str(shard)).items():
            if FULL_PREFIX + name in weights: raise ValueError(f"{ckpt_dir}: tensor {name} is in more than one shard")
            weights[FULL_PREFIX + name] = value
    conv = [k for k in weights if k.endswith("linear_attn.conv1d.weight")]
    if not conv or any(weights[k].ndim != 3 or weights[k].shape[1] != 1 for k in conv):
        # sanitize infers "transformers layout, shift the norms" from the conv kernels; anything else would load wrong norms silently
        raise ValueError(f"{ckpt_dir}: expected transformers' DeltaNet conv kernels [channels, 1, width]; this is not a kev.train --full_ft export")
    weights = lm.language_model.sanitize(weights)
    expected = {k: v.shape for k, v in tree_flatten(lm.parameters())}
    missing, unexpected = sorted(set(expected) - set(weights)), sorted(set(weights) - set(expected))
    wrong = sorted(k for k in set(expected) & set(weights) if tuple(weights[k].shape) != tuple(expected[k]))
    cast = sorted(k for k, v in weights.items() if v.dtype != want)
    if missing or unexpected or wrong or cast:
        strip = lambda names: [n.removeprefix(FULL_PREFIX) for n in names[:2]]
        raise ValueError(f"{ckpt_dir} does not match mlx-lm's Qwen3.5 text model: {len(missing)} tensors missing (e.g. {strip(missing)}), "
                         f"{len(unexpected)} unexpected (e.g. {strip(unexpected)}), {len(wrong)} with another shape (e.g. {strip(wrong)}), "
                         f"{len(cast)} not {dtype} (e.g. {strip(cast)})")
    lm.load_weights(list(weights.items()), strict=True)
    del weights
    lm.eval()
    mx.eval(lm.parameters())
    mx.clear_cache()
    return lm


def merge_lora(lm, adapter_dir, scale=1.0):
    """Fold a PEFT adapter into the mlx-lm model's weights the way the torch path does: W + (B @ A) * alpha / r in fp32,
    rounded once to the backbone dtype. `scale` is LoadOptions.lora_scale (WiSE-FT interpolation). Returns the tensor count."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("trainable_token_indices"):
        raise ValueError("the MLX backend does not carry trained token embeddings (special_embeddings checkpoints); use backend=torch")
    alpha = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    weights = mx.load(str(adapter_dir / "adapter_model.safetensors"))
    params = dict(tree_flatten(lm.parameters()))
    merged = {}
    with mx.stream(mx.cpu):   # the GPU's fp32 matmul is a reduced-precision fast path (~1e-3 relative on an M5); the merge is one-time and must be exact
        for name, a in weights.items():
            if not name.endswith(".lora_A.weight"):
                continue
            stem = name[: -len(".lora_A.weight")]
            # peft names the wrapped text model `base_model.model.<layers...>`; mlx-lm nests it as `language_model.model.<layers...>`
            target = stem.replace("base_model.model.", "language_model.model.", 1) + ".weight"
            if target not in params:
                raise ValueError(f"adapter tensor {stem} has no weight in the mlx-lm model (looked for {target})")
            base = params[target]
            delta = (weights[stem + ".lora_B.weight"].astype(mx.float32) @ a.astype(mx.float32)) * (alpha * scale)
            merged[target] = (base.astype(mx.float32) + delta).astype(base.dtype)
            mx.eval(merged[target])   # one tensor at a time: one graph over every target peaks at ~2.9x the base weights and leaves ~2x of them in MLX's buffer cache (25.8 GB RSS for the 4B)
    lm.load_weights(list(merged.items()), strict=False)
    mx.eval(lm.parameters())
    del params, weights   # drop the pre-merge weights and the adapter before clearing the cache, or ~7 GB stay cached
    mx.clear_cache()   # hand the merge transients back to the OS; MLX keeps freed buffers otherwise
    return len(merged)


class MLXDecisionModel:
    """Prefill-only scorer: hidden states from mlx-lm, logits from the shared torch PointerHead."""
    backend, device, hybrid, option_isolation = "mlx", "mlx", True, False
    prefix_min_tokens = 0   # kev.serve caches the state prefix for every request: on Metal the branch-only pass is always the cheaper one

    def __init__(self, lm, pad_id, head_dim=256):
        self.lm = lm                                                  # mlx-lm qwen3_5.Model from load_base (+ merge_lora) or load_full
        self.text = self.lm.language_model.model                      # Qwen3_5TextModel: embeddings -> layers -> final norm = `.model.last_hidden_state`
        self.pad_id = pad_id
        self.head = PointerHead(self.text.embed_tokens.weight.shape[1], dp=head_dim).eval()

    @property
    def dtype(self):
        return str(self.text.embed_tokens.weight.dtype).removeprefix("mlx.core.")

    def eval(self):
        self.head.eval(); return self

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=False, **kw)

    def _hidden(self, rows, cache=None):
        """[N, L, d] hidden states of right-padded token rows. Pads sit after every real token and both layer kinds are
        causal (attention: causal mask; DeltaNet: a left-to-right recurrence), so no real token sees a pad."""
        L = max(len(r) for r in rows)
        ids = mx.array([r + [self.pad_id] * (L - len(r)) for r in rows], dtype=mx.int32)
        h = self.text(ids, cache=cache)
        mx.eval(h)
        return h

    def _logits(self, h, decide, opts):
        """One question's logits through the fp32 pointer head (temperature included, eval mode)."""
        idx = mx.array([decide, *opts], dtype=mx.int32)
        picked = torch.from_numpy(np.asarray(h[idx].astype(mx.float32)))
        with torch.no_grad():
            return self.head(picked[0], picked[1:])

    def forward_rows(self, enc):
        """Row form, as the torch path computes it: every question is one causal row of state + branch tokens, the state
        recomputed per row. The reference the prefix form is checked against (tests/test_mlx.py); serving uses `forward`."""
        S, _, rows = rows_of(enc)
        out = [None] * len(rows)
        for idx in row_groups([len(S) + len(r["ids"]) for r in rows]):
            part = [rows[i] for i in idx]
            h = self._hidden([S + r["ids"] for r in part])
            for j, i in enumerate(idx):
                r = part[j]
                out[i] = self._logits(h[j], len(S) + r["decide"], [len(S) + o for o in r["opts"]])
        return out

    # --- state prefix: the state runs once into an mlx-lm prompt cache (KV for the attention layers, conv + recurrent state
    # for the DeltaNet layers); the branches run as one batch on a replicated copy, so the prefix stays pristine and can be
    # reused by the next request with the same state. On Metal this is also the cheapest way to answer a single request
    # (state once instead of once per question), so it is the only path `forward` / `probs` take.

    def prefix(self, enc):
        """The state into a fresh prompt cache, PREFILL_CHUNK tokens per pass, each pass's cache evaluated before the next
        (mlx-lm's own prefill does the same; the hidden states a pass returns are never read). One pass over the whole
        state kept every layer's activations of every token alive at once and left the DeltaNet conv states lazy (each
        holding its layer's whole [Ls, conv_dim] input): Kev-4B, 8,192 tokens, 4.6 GB above the weights instead of 1.2 GB,
        the same 6.5 s. Exact: a pass continues the cached keys/values (rotary offset = cache offset), conv window and
        fp32 recurrent state, so chunked and one-pass differ by float reassociation only (fp32 on the CPU: max |dp| 1e-6;
        bf16 on Metal: up to ~0.01, like any change of kernel shapes; tests/test_mlx.py, runs/mlx-long-states/prefill-ab-*)."""
        Ls = enc["seg"].count(0)
        cache, ids = make_prompt_cache(self.lm), enc["ids"][:Ls]
        for start in range(0, Ls, PREFILL_CHUNK):
            self.text(mx.array([ids[start:start + PREFILL_CHUNK]], dtype=mx.int32), cache=cache)
            mx.eval([c.state for c in cache])
        return Ls, cache

    def _branch_logits(self, enc, cache):
        """Branches as rows on a replicated copy of the state cache, one row_groups pass (and its cache copies) at a time."""
        _, _, rows = rows_of(enc)
        out = [None] * len(rows)
        for idx in row_groups([len(r["ids"]) for r in rows], enc["seg"].count(0)):
            part = [rows[i] for i in idx]
            batch = [type(c).merge([c] * len(part)) for c in cache]      # merge copies the arrays: `cache` is not mutated
            h = self._hidden([r["ids"] for r in part], batch)
            for j, i in enumerate(idx): out[i] = self._logits(h[j], part[j]["decide"], part[j]["opts"])
        return out

    def _branch_probs(self, enc, cache):
        return [F.softmax(z, -1) for z in self._branch_logits(enc, cache)]

    def forward(self, enc):
        """List of logits tensors, one per question."""
        return self._branch_logits(enc, self.prefix(enc)[1])

    def probs(self, enc):
        return self._branch_probs(enc, self.prefix(enc)[1])

    def probs_and_prefix(self, enc):
        prefix = self.prefix(enc)
        return self._branch_probs(enc, prefix[1]), prefix

    def probs_with_prefix(self, enc, prefix):
        Ls, cache = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        return self._branch_probs(enc, cache)

    def probs_batch(self, encs, prefixes, keep):
        """kev.serve's batch call: one request at a time on Metal."""
        out = [probs_one(self, e, p, k) for e, p, k in zip(encs, prefixes, keep)]
        return [o[0] for o in out], [o[1] for o in out]
