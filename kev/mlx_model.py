"""Native MLX scorer for Qwen3.5 Kev checkpoints.

This path is intentionally prefill-only.  It uses MLX-LM's Metal Gated
DeltaNet implementation, then applies Kev's pointer head directly to hidden
states instead of running the vocabulary head or a generation loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .encoding import rows_of


class MLXDecisionModel:
    def __init__(self, model, tok, head, model_path: Path):
        import mlx.core as mx

        self.model = model
        self.tok = tok
        self.device = "mlx"
        self.hybrid = True
        self.pad_id = getattr(tok, "pad_token_id", None) or 0
        self.model_path = str(model_path)
        self._mx = mx
        self._head = {k: mx.array(v.detach().cpu().numpy()) for k, v in head.items()}
        self._scale = 1.0 / np.sqrt(self._head["q.weight"].shape[0])

    def encode(self, tok, rec, **kw):
        # The shared encoder is tokenizer-backend independent.  `tok` is the
        # HuggingFace tokenizer used by Kev, while MLX receives only token IDs.
        from .encoding import encode

        return encode(tok, rec, **kw)

    def _hidden(self, ids, cache=None):
        import mlx.core as mx

        x = mx.array(ids, dtype=mx.int32)
        h = self.model.model(x, cache=cache)
        mx.eval(h)
        return h

    @staticmethod
    def _copy_cache(cache):
        return [type(c).from_state(c.state) for c in cache]

    def _replicate_cache(self, cache, n):
        # KVCache.merge produces a batched cache; ArraysCache.merge keeps the
        # recurrent state batch dimension. Both are native MLX-LM operations.
        copies = [self._copy_cache(cache) for _ in range(n)]
        return [type(copies[0][i]).merge([c[i] for c in copies]) for i in range(len(copies[0]))]

    def _readout(self, h, decide, opts):
        q_w, q_b = self._head["q.weight"], self._head["q.bias"]
        k_w, k_b = self._head["k.weight"], self._head["k.bias"]
        q = h[decide].astype(q_w.dtype) @ q_w.T + q_b
        k = h[self._mx.array(opts, dtype=self._mx.int32)].astype(k_w.dtype) @ k_w.T + k_b
        return (k @ q) * self._scale

    def _probs_from_hidden(self, h, rows):
        import mlx.core as mx

        out = []
        for i, r in enumerate(rows):
            z = self._readout(h[i], r["decide"], r["opts"])
            out.append(mx.softmax(z))
        mx.eval(out)
        return [np.asarray(x).astype(np.float32) for x in out]

    def probs(self, enc):
        _, _, rows = rows_of(enc)
        state = enc["ids"][: enc["seg"].count(0)]
        inputs = [state + r["ids"] for r in rows]
        width = max(len(x) for x in inputs)
        padded = [x + [self.pad_id] * (width - len(x)) for x in inputs]
        h = self._hidden(padded)
        state_len = len(state)
        local_rows = [{**r, "decide": state_len + r["decide"], "opts": [state_len + x for x in r["opts"]]} for r in rows]
        return self._probs_from_hidden(h, local_rows)

    def prefix(self, enc):
        from mlx_lm.models.cache import make_prompt_cache

        state_len = enc["seg"].count(0)
        cache = make_prompt_cache(self.model)
        state = enc["ids"][:state_len]
        h = self._hidden([state], cache=cache)
        return state_len, cache

    def _branches(self, enc, prefix):
        import mlx.core as mx

        state_len, cache = prefix
        _, _, rows = rows_of(enc)
        width = max(len(r["ids"]) for r in rows)
        lengths = [len(r["ids"]) for r in rows]
        padded = [r["ids"] + [self.pad_id] * (width - len(r["ids"])) for r in rows]
        batch_cache = self._replicate_cache(cache, len(rows))
        for c in batch_cache:
            if c.__class__.__name__ == "ArraysCache":
                c.prepare(lengths=lengths)
            elif hasattr(c, "prepare"):
                c.prepare(right_padding=[width - n for n in lengths])
        try:
            h = self._hidden(padded, cache=batch_cache)
        finally:
            for c in batch_cache:
                if hasattr(c, "finalize"):
                    c.finalize()
        return self._probs_from_hidden(h, rows)

    def probs_and_prefix(self, enc):
        prefix = self.prefix(enc)
        return self._branches(enc, prefix), prefix

    def probs_with_prefix(self, enc, prefix):
        if enc["seg"].count(0) != prefix[0]:
            raise ValueError("prefix does not match this record's state")
        return self._branches(enc, prefix)


def _load_base(path: Path):
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    model, _ = load_model(path, lazy=False)
    tok = AutoTokenizer.from_pretrained(str(path))
    return model, tok


def _merge_lora(model, adapter_path: Path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    if cfg.get("trainable_token_indices"):
        raise ValueError("native MLX backend does not implement trainable token embeddings")
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    weights = mx.load(str(adapter_path / "adapter_model.safetensors"))
    params = dict(tree_flatten(model.parameters()))
    merged = {}
    for name in list(weights):
        if not name.endswith(".lora_A.weight"):
            continue
        prefix = name[: -len(".lora_A.weight")]
        b_name = prefix + ".lora_B.weight"
        target = prefix.replace("base_model.model.", "language_model.model.") + ".weight"
        if b_name not in weights or target not in params:
            raise ValueError(f"MLX LoRA mapping missing {target}")
        delta = mx.matmul(weights[b_name], weights[name]) * scale
        # Merge in fp32, then round once to the backbone dtype. Rounding the
        # LoRA delta before adding it can move logits near a decision boundary.
        base = params[target]
        merged[target] = (base.astype(mx.float32) + delta.astype(mx.float32)).astype(base.dtype)
    model.load_weights(list(merged.items()), strict=False)
    mx.eval(model.parameters())


def load_mlx(run: str):
    from huggingface_hub import snapshot_download
    from .evaluate import resolve_run

    adapter_path = Path(resolve_run(run))
    meta = __import__("torch").load(adapter_path / "head.pt", map_location="cpu", weights_only=True)
    if meta.get("option_isolation", False):
        raise ValueError("native MLX backend does not implement option_isolation")
    base = snapshot_download(meta["base"], revision=meta.get("base_revision"), allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    model, tok = _load_base(Path(base))
    _merge_lora(model, adapter_path)
    return tok, MLXDecisionModel(model, tok, meta["head"], Path(base))
