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
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from .model import PointerHead, encode, probs_one, rows_of, rows_per_pass


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
        mx.eval(list(merged.values()))
    lm.load_weights(list(merged.items()), strict=False)
    mx.eval(lm.parameters())
    return len(merged)


class MLXDecisionModel:
    """Prefill-only scorer: hidden states from mlx-lm, logits from the shared torch PointerHead."""
    backend, device, hybrid, option_isolation = "mlx", "mlx", True, False
    prefix_min_tokens = 0   # kev.serve caches the state prefix for every request: on Metal the branch-only pass is always the cheaper one

    def __init__(self, base_dir, pad_id, head_dim=256):
        self.lm, _ = load_model(Path(base_dir))                       # weights as stored (bf16 for the Qwen3.5 bases)
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
        chunk, out = rows_per_pass([S + r["ids"] for r in rows]), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            h = self._hidden([S + r["ids"] for r in part])
            out += [self._logits(h[i], len(S) + r["decide"], [len(S) + o for o in r["opts"]]) for i, r in enumerate(part)]
        return out

    # --- state prefix: the state runs once into an mlx-lm prompt cache (KV for the attention layers, conv + recurrent state
    # for the DeltaNet layers); the branches run as one batch on a replicated copy, so the prefix stays pristine and can be
    # reused by the next request with the same state. On Metal this is also the cheapest way to answer a single request
    # (state once instead of once per question), so it is the only path `forward` / `probs` take.

    def prefix(self, enc):
        Ls = enc["seg"].count(0)
        cache = make_prompt_cache(self.lm)
        self._hidden([enc["ids"][:Ls]], cache)
        return Ls, cache

    def _branch_logits(self, enc, cache):
        """Branches as rows on a replicated copy of the state cache, rows_per_pass rows (and cache copies) at a time."""
        _, _, rows = rows_of(enc)
        chunk, out = rows_per_pass([r["ids"] for r in rows], enc["seg"].count(0)), []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            batch = [type(c).merge([c] * len(part)) for c in cache]      # merge copies the arrays: `cache` is not mutated
            h = self._hidden([r["ids"] for r in part], batch)
            out += [self._logits(h[i], r["decide"], r["opts"]) for i, r in enumerate(part)]
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
