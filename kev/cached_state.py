"""Shared-state K/V reuse: run the state tokens once, then the question branches against their cached keys/values.

Two mechanisms on top of the packed forward in kev.model:
  frozen state (state_mode="frozen"): the base weights build the state K/V once (adapter off, no grad) and the
    LoRA-adapted branches attend to it. The state is never adapted, so its K/V can be kept across epochs, augmentations
    and adapters (StateCache). This is a different model function from the packed forward.
  chunked branches (branch_chunk=N): adapted state pass with grad, its K/V detached into leaves, the questions N at a
    time against those leaves, then the accumulated K/V gradients pushed back through the state graph. Same gradients as
    the packed forward; activation memory is bounded by the state plus one chunk of branches.
  both (frozen_loss_backward): the questions N at a time against the frozen state's K/V; the state carries no gradient,
    so there is nothing to push back and the gradients are those of one branch pass over all questions.
Branch tokens keep their packed position ids (they restart after the state) and attend to every state token plus the
causal part of their own branch, exactly what branch_mask_batch allows them in the packed sequence.
Requires full-attention layers (no sliding window) and no gradient checkpointing in training mode (it drops the cache)."""
import contextlib, hashlib, time
from collections import OrderedDict
from pathlib import Path
import torch
from .device import sync   # re-exported: scripts and tests import it from here
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer
from .model import OPT_DECIDE


def state_len(enc):
    """Number of leading state tokens (seg == 0) of an encoded record."""
    return enc["seg"].count(0)


INT8 = "int8"   # cache dtype name: keys/values stored as int8 with one fp16 scale per (layer, head, token) row over the head dim


def size_bytes(kv):
    """Bytes of a cache entry: per-layer (keys, values), or int8 entries' (keys, key scales, values, value scales)."""
    return sum(t.numel() * t.element_size() for pair in kv for t in pair)


def kv_bytes_per_token(model, dtype=None):
    """Bytes of per-layer keys+values for one state token: 2 x layers x kv_heads x head_dim x itemsize (dtype = model's);
    for "int8", 2 x layers x kv_heads x (head_dim + one fp16 scale)."""
    cfg = model.lm.config
    if dtype == INT8:
        return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.head_dim + 2)
    itemsize = torch.empty((), dtype=dtype or next(model.lm.parameters()).dtype).element_size()
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * itemsize


def quantize_int8(t):
    """Symmetric int8 quantization of t along its last dim with one scale per row (per layer, head and token for K/V of
    shape [1, Hkv, S, dh]). Returns (int8 codes, fp16 scales [..., 1]); the codes are computed against the fp16-rounded
    scale, so dequantize_int8 with the stored scale reproduces exactly what the quantizer saw."""
    t = t.float()
    scale = (t.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-6).to(torch.float16)
    return (t / scale.float()).round().clamp_(-127, 127).to(torch.int8), scale


def dequantize_int8(q, scale, dtype=torch.float32):
    return (q.float() * scale.float()).to(dtype)


def quantize_kv(kv):
    """Per-layer (keys, values) -> per-layer (key codes, key scales, value codes, value scales)."""
    return [(*quantize_int8(k), *quantize_int8(v)) for k, v in kv]


def dequantize_kv(qkv, dtype=torch.float32, device=None):
    """Inverse of quantize_kv: per-layer (keys, values) in dtype (on device), the rounded values the branches attend to."""
    return [(dequantize_int8(qk, sk, dtype).to(device=device), dequantize_int8(qv, sv, dtype).to(device=device)) for qk, sk, qv, sv in qkv]


def kv_dtype(kv, dtype, device=None):
    """Per-layer (keys, values) as a cache of `dtype` hands them out: cast to a torch dtype (and device; tensors already there
    are returned as-is, not copied), or for "int8" rounded through quantize_kv/dequantize_kv (returned in fp32)."""
    if dtype == INT8:
        return dequantize_kv(quantize_kv(kv), torch.float32, device)
    return [(k.to(dtype=dtype, device=device), v.to(dtype=dtype, device=device)) for k, v in kv]


def _check(model):
    cfg = model.lm.config
    layer_types = getattr(cfg, "layer_types", None) or []
    if getattr(cfg, "use_sliding_window", False) or "sliding_attention" in layer_types:
        raise NotImplementedError("cached state needs full-attention layers; this config uses sliding-window attention")
    if "linear_attention" in layer_types:
        raise NotImplementedError("cached state needs per-layer attention K/V; hybrid (Gated DeltaNet) backbones keep recurrent state instead")
    if model.training and getattr(model.lm, "is_gradient_checkpointing", False):
        raise RuntimeError("gradient checkpointing drops past_key_values in training mode; disable it for cached-state passes")


def state_forward(model, state_ids, adapter=True, grad=False):
    """Run the state tokens alone (positions 0..S-1, plain causal attention, use_cache=True).
    Returns (hidden [S, d] fp32, kv): kv is the model's own per-layer (keys, values) [1, Hkv, S, dh] after rotary.
    adapter=False runs the base weights (LoRA disabled); grad=False runs without autograd."""
    _check(model)
    ids = torch.tensor(list(state_ids), device=model.device)[None]
    pos = torch.arange(ids.shape[1], device=model.device)[None]
    no_adapter = model.lm.disable_adapter() if not adapter and hasattr(model.lm, "disable_adapter") else contextlib.nullcontext()
    with (contextlib.nullcontext() if grad else torch.no_grad()), no_adapter:
        out = model.lm(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=True)
    return out.last_hidden_state[0].float(), [(layer.keys, layer.values) for layer in out.past_key_values.layers]


def state_kv(model, state_ids, adapter=True, grad=False):
    """Per-layer (keys, values) of the state tokens; see state_forward."""
    return state_forward(model, state_ids, adapter, grad)[1]


def compute_dtype(model):
    """The dtype the backbone's matmuls run in: the autocast dtype while autocast is active on the model's device type
    (train.py --dtype bf16 keeps fp32 master weights), otherwise the parameter dtype. Used for the cache handed to the
    model, so a bf16 run does not keep a fp32 copy of every cached state K/V per branch pass for SDPA to cast down again."""
    kind = str(model.device).split(":")[0]
    if kind in ("cuda", "cpu") and torch.is_autocast_enabled(kind):
        return torch.get_autocast_dtype(kind)
    return next(model.lm.parameters()).dtype


def cache_from_kv(model, kv):
    """A DynamicCache whose layers reference kv's tensors (cast to the compute dtype/device only when they differ; no copy
    otherwise, so gradients reach kv's tensors). The model appends branch K/V to a fresh cat, never into kv."""
    _check(model)
    if len(kv) != model.lm.config.num_hidden_layers:
        raise ValueError(f"expected {model.lm.config.num_hidden_layers} layers of K/V, got {len(kv)}")
    cache = DynamicCache()
    for k, v in kv_dtype(kv, compute_dtype(model), model.device):
        layer = DynamicLayer(); layer.lazy_initialization(k, v); layer.keys, layer.values = k, v
        cache.layers.append(layer)
    return cache


def branch_inputs(enc, question_ids=None, device="cpu", dtype=torch.float32):
    """Inputs for one pass of the selected questions' tokens over a cache of the S state tokens:
    (ids [1, Lb], position_ids [1, Lb], additive mask [1, 1, Lb, S+Lb], decide index per question, option-end indices
    per question), indices local to the branch sequence (in question_ids order; default all questions).
    Mask rule: a branch token sees every state token and the tokens at or before it in its own question (option
    isolation: an option token sees only its own span, <decide> the whole question); the diagonal is always kept."""
    S = state_len(enc); seg = enc["seg"]
    qs = list(range(len(enc["decide_idx"]))) if question_ids is None else list(question_ids)
    spans = [(S if q == 0 else enc["decide_idx"][q - 1] + 1, enc["decide_idx"][q] + 1) for q in qs]   # <decide> ends each branch
    sel = [i for s, e in spans for i in range(s, e)]
    local = {i: n for n, i in enumerate(sel)}
    Lb = len(sel)
    ids = torch.tensor([enc["ids"][i] for i in sel], device=device)[None]
    pos = torch.tensor([enc["pos"][i] for i in sel], device=device)[None]
    s = torch.tensor([seg[i] for i in sel], device=device)
    block = torch.tril(torch.ones(Lb, Lb, dtype=torch.bool, device=device)) & (s[None, :] == s[:, None])
    if enc.get("option_isolation"):
        o = torch.tensor([enc["opt"][i] for i in sel], device=device)
        block &= (o[None, :] < 0) | (o[:, None] == OPT_DECIDE) | (o[None, :] == o[:, None])
    allow = torch.ones(Lb, S + Lb, dtype=torch.bool, device=device)
    allow[:, S:] = block | torch.eye(Lb, dtype=torch.bool, device=device)
    mask = torch.zeros(Lb, S + Lb, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[None, None]
    decide = [local[enc["decide_idx"][q]] for q in qs]
    opts = [[local[i] for i in enc["opt_idx"][q]] for q in qs]
    return ids, pos, mask, decide, opts


def branch_hidden(model, enc, kv, question_ids=None):
    """Adapter-enabled pass of the selected questions over a fresh cache built on kv (kv itself is never mutated).
    Returns (hidden [Lb, d] fp32, decide indices, option-end indices), local to the branch sequence."""
    ids, pos, mask, decide, opts = branch_inputs(enc, question_ids, model.device, next(model.lm.parameters()).dtype)
    out = model.lm(input_ids=ids, position_ids=pos, attention_mask=mask, past_key_values=cache_from_kv(model, kv), use_cache=True)
    return out.last_hidden_state[0].float(), decide, opts


def branch_logits(model, enc, kv, question_ids=None):
    """One logits tensor per selected question: the pointer readout at its <decide> over its option ends."""
    h, decide, opts = branch_hidden(model, enc, kv, question_ids)
    return [model.head(h[d], h[torch.tensor(oi, device=model.device)]) for d, oi in zip(decide, opts)]


def _branch_chunks_backward(model, enc, kv, question_loss_fn, chunk, scale, ctx):
    """The questions `chunk` at a time against kv, each chunk's loss times `scale` backpropagated right away (at most one
    chunk's activations alive); ctx wraps the forward and loss only. Returns the summed unscaled loss as a float."""
    total, Q = 0.0, len(enc["decide_idx"])
    for start in range(0, Q, chunk):
        qs = range(start, min(start + chunk, Q))
        with ctx:
            zs = branch_logits(model, enc, kv, qs)
            loss = sum(question_loss_fn(z, q) for z, q in zip(zs, qs))
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        (loss * scale).backward(); total += loss.item()
        del zs, loss
    return total


def chunked_loss_backward(model, enc, question_loss_fn, chunk, scale=1.0, times=None, forward_ctx=None):
    """Gradient-exact alternative to the packed forward+backward of one record: adapted state pass with grad, then the
    questions `chunk` at a time against its detached K/V (leaf tensors), each chunk's loss times `scale` backpropagated
    right away, and finally the accumulated K/V gradients pushed through the state graph.
    question_loss_fn(logits, q) -> scalar loss of question q (the caller folds in any per-record normalization).
    Returns the summed unscaled loss as a float. times (a dict/Counter) accumulates seconds under "state" and "branch";
    forward_ctx (e.g. torch.autocast) wraps the forward passes only."""
    _check(model)
    ctx = forward_ctx or contextlib.nullcontext()
    def tick(key, t0):
        if times is not None: sync(model.device); times[key] += time.perf_counter() - t0
    t = time.perf_counter()
    with ctx:
        kv = state_kv(model, enc["ids"][:state_len(enc)], adapter=True, grad=True)
    leaves = [(k.detach().requires_grad_(True), v.detach().requires_grad_(True)) for k, v in kv]
    tick("state", t)
    t = time.perf_counter()
    total = _branch_chunks_backward(model, enc, leaves, question_loss_fn, chunk, scale, ctx)
    tick("branch", t)
    t = time.perf_counter()
    # push the branches' gradients w.r.t. the state K/V back through the state's graph (half the LoRA gradient flows this way)
    pairs = [(src, leaf.grad) for (k, v), (lk, lv) in zip(kv, leaves) for src, leaf in ((k, lk), (v, lv)) if src.requires_grad and leaf.grad is not None]
    if pairs:
        torch.autograd.backward([src for src, _ in pairs], grad_tensors=[g for _, g in pairs])
    tick("state", t)
    return total


def frozen_loss_backward(model, enc, kv, question_loss_fn, chunk, scale=1.0, times=None, forward_ctx=None):
    """Frozen-state counterpart of chunked_loss_backward: the questions `chunk` at a time against kv, the record's base state
    K/V (a StateCache entry or state_kv(..., adapter=False, grad=False)), each chunk's loss times `scale` backpropagated right
    away. The state carries no gradient, so nothing is pushed back: the gradients are those of one branch pass over all the
    questions, with at most one chunk's activations alive. Same loss function, return value and forward_ctx as
    chunked_loss_backward; times accumulates seconds under "branch" (the caller times the state pass, if any)."""
    _check(model)
    t = time.perf_counter()
    total = _branch_chunks_backward(model, enc, kv, question_loss_fn, chunk, scale, forward_ctx or contextlib.nullcontext())
    if times is not None: sync(model.device); times["branch"] += time.perf_counter() - t
    return total


class StateCache:
    """Store of per-record state K/V keyed by the caller (record id + state hash), kept in `dtype` on `device`: a torch device
    (tensors in memory) or "disk" (one torch.save file of CPU tensors per entry under `cache_dir`, named by the sha1 of the
    key; only the index is in memory, bytes are counted on disk, and a file left by an earlier run over the same directory is
    adopted on first lookup, so the directory can be reused by a later run of the same base and dtype).
    dtype is a torch dtype or "int8" (INT8): int8 codes plus one fp16 scale per (layer, head, token) row, stored that way and
    dequantized to fp32 by get() and by put()'s return value, so a record's branches see the same rounded K/V on the miss
    that stores it as on every later hit.
    max_bytes > 0 evicts the oldest entries first (an entry larger than the cap is returned but not kept), so a cap below
    one epoch's worth of records only serves reuse within a batch (none-pair siblings, permuted copies). Entries are
    handed back as stored (cast them with kv_dtype for the model; cache_from_kv does this itself) and must not be mutated."""

    def __init__(self, dtype=torch.float32, device="cpu", max_bytes=0, cache_dir=None):
        self.dtype, self.device, self.max_bytes = dtype, device, max_bytes
        if device == "disk" and cache_dir is None: raise ValueError("device='disk' needs cache_dir")
        self.dir = Path(cache_dir) if device == "disk" else None
        if self.dir is not None: self.dir.mkdir(parents=True, exist_ok=True)
        self.entries = OrderedDict(); self.bytes = 0; self.hits = self.misses = 0   # key -> kv, or (path, bytes) on disk

    def path(self, key):
        return self.dir / (hashlib.sha1(key.encode()).hexdigest() + ".pt")

    def get(self, key):
        e = self.entries.get(key)
        if e is None and self.dir is not None and self.path(key).exists():   # written by an earlier run: adopt it
            e = self.entries[key] = (self.path(key), self.path(key).stat().st_size); self.bytes += e[1]
        if e is None: self.misses += 1; return None
        self.hits += 1
        e = torch.load(e[0], map_location="cpu") if self.dir is not None else e
        return dequantize_kv(e) if self.dtype == INT8 else e

    def _drop(self, key):
        e = self.entries.pop(key)
        if self.dir is not None: e[0].unlink(missing_ok=True); self.bytes -= e[1]
        else: self.bytes -= size_bytes(e)

    def put(self, key, kv):
        """Store kv under key; returns the entry as the model should see it (cast, or for int8 the quantize-dequantized fp32 K/V)."""
        store_dev = "cpu" if self.dir is not None else self.device
        if self.dtype == INT8:
            q = quantize_kv([(k.detach(), v.detach()) for k, v in kv])   # quantized where the K/V live (the training device)
            seen = dequantize_kv(q)
            kv = [tuple(t.to(store_dev) for t in layer) for layer in q]
        else:
            kv = seen = kv_dtype([(k.detach(), v.detach()) for k, v in kv], self.dtype, store_dev)
        if key in self.entries: self._drop(key)
        if self.dir is not None:   # written to a sibling temp file and renamed, so a killed run never leaves a truncated entry
            path = self.path(key); tmp = path.with_suffix(".tmp"); torch.save(kv, tmp); tmp.replace(path); entry = (path, path.stat().st_size); n = entry[1]
        else:
            entry = kv; n = size_bytes(kv)
        if self.max_bytes and n > self.max_bytes:   # larger than the cap: handed back, store untouched
            if self.dir is not None: path.unlink()
            return seen
        while self.max_bytes and self.entries and self.bytes + n > self.max_bytes:
            self._drop(next(iter(self.entries)))   # oldest first
        self.entries[key] = entry; self.bytes += n
        return seen

    def __len__(self):
        return len(self.entries)

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "bytes": self.bytes, "entries": len(self.entries), "tier": str(self.device),
                "dir": str(self.dir) if self.dir is not None else None, "dtype": str(self.dtype).replace("torch.", ""), "max_bytes": self.max_bytes}
