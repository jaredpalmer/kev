"""Ascend NPU kernels for the Qwen3.5 decoder layers: the Gated DeltaNet mixer (`accelerate`) and, for serving a merged
checkpoint, the whole layer (`fuse`).

Transformers ships a pure-PyTorch fallback for the gated delta rule (`torch_chunk_gated_delta_rule`) whenever
flash-linear-attention and causal-conv1d are absent, which is the case on Ascend: fla's Triton kernels are written for
CUDA. That fallback runs the chunked recurrence in fp32 with a 64-step Python loop for the WY (UT-transform) inverse per
chunk, and on a 910B2 it dominates a Kev forward -- 24 DeltaNet layers at ~17 ms each, ~85% of a decision (a 165-token
prefill: 409 ms of 478 ms).

vllm-ascend already carries the same math as AscendC kernels plus triton_ascend helpers (`vllm_ascend.ops.triton.fla`,
`torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h` / `chunk_fwd_o`). They are built for vLLM's paged, variable-length
serving path and read vLLM's forward context and prefill-context-parallel group, so the wrapper
`vllm_ascend.ops...chunk.chunk_gated_delta_rule` cannot be called outside a vLLM worker. This module drives the same
kernels directly for Kev's single-node, prefill-only case: each of the model's causal rows (state + one question branch)
becomes one variable-length segment of a packed `cu_seqlens`, exactly the contract the kernels expect.

`accelerate(lm)` replaces `Qwen3_5GatedDeltaNet.chunk_gated_delta_rule` (a per-instance attribute the transformers layer
calls) with `chunk_gated_delta_rule` below; the projections, conv, gate and gated norm around it are untouched, so the
patch is independent of whether the LoRA is merged. The result equals the fp32 reference to bf16 rounding (max |dp| ~1e-3
on core output, ~8e-3 on the final recurrent state over Kev's dims and lengths 40..864, batch 1..16).

On NPU, hybrid rows are recomputed (`DecisionModel.probs_and_prefix`) rather than continued from a cached state.
Transformers 5.5.4 continues that state with `recurrent_gated_delta_rule`, a decode kernel this patch does not replace,
so the chunk is called with `initial_state=None`. The AscendC `fwd_h` kernel reads its state buffer even for a first
chunk, and an absent (`None`) state left uninitialised memory that produced NaNs or garbage for some lengths; a zero
initial state is passed instead, which is the same math.

With the mixer on those kernels the pass is bound by how fast the host can enqueue it, not by the NPU: a 165-token
Kev-27B decision was 6,539 kernel launches, 166 ms of device time and 315 ms of wall time, the wall time being the
enqueueing. `fuse(lm)` is the Ascend answer to that, the counterpart of `kev.fused_qwen35` on CUDA. It rewrites the
merged layers so the same math runs in far fewer, larger ops -- one projection GEMM per mixer (q/k/v, z, b, a
concatenated), one for attention (q with its output gate, k, v) and one for the MLP (gate and up), their weights cast
to the FRACTAL_NZ layout the cube unit reads; `npu_rms_norm` for every RMSNorm (weight 1 + w kept in fp32) and for the
mixer's gated norm; `npu_rotary_mul`; `npu_swiglu`; and `npu_fused_infer_attention_score` in place of the reference's
explicit scores. It also replaces the text model's own forward, which otherwise builds a [rows, 1, L, L] float mask for
every pass -- that mask is what ran a 6k-token state out of memory on a 64 GB card. Kev's rows are causal and
right-padded (`DecisionModel._pad_rows`), so a pad is never a key any real query reaches and the attention reads the
2,048 x 2,048 compressed causal mask its `sparse_mode=3` expects, at any length. 205 ms, 3,668 launches, 142 ms of
device time.

That leaves a pass still waiting on its own launches, so `Graphs` replays it from a captured Ascend graph: 134 ms, at a
given shape bit for bit what the eager pass returns. Equal to the reference layers to bf16 rounding (over 119
decision-v7 development questions on Kev-27B: max |dp| 0.017, 0 argmax flips), and only for inference -- the
projections replace the originals (the merged LoRA is inside them), so there is no backward, no unmerged adapter and no
cache.
"""
import types
import warnings

import torch
import torch.nn.functional as F

from .device import sync

_HELPERS = None      # (chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd, l2norm_fwd, prepare_chunk_indices)
_META = {}           # (B, T, H, chunk_size, device) -> Meta: identical across a forward's DeltaNet layers


def _load_helpers():
    """Import the triton_ascend helpers and register the AscendC ops once. Raises if vllm-ascend is unavailable."""
    global _HELPERS
    if _HELPERS is not None:
        return _HELPERS
    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend custom ops are unavailable (enable_custom_op() returned False)")
    init_device_properties_triton()   # the triton_ascend helpers read the device's vector-core count
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd
    from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
    _HELPERS = (chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd, l2norm_fwd, prepare_chunk_indices)
    return _HELPERS


CUMSUM_BUFFER = 2 ** 18   # the unified-buffer budget chunk_local_cumsum sizes its block from (vllm_ascend.ops.triton.fla.cumsum)
SOLVE_BLOCK = 608 * 2     # solve_tril's large block (vllm_ascend.ops.triton.fla.solve_tril)


def _meta(B, T, H, chunk_size, device):
    """Packed-segment metadata for a [B, T] batch of H-head rows: cu_seqlens delimiting each equal-length row and every
    chunk index table the helpers below need. Cached because every DeltaNet layer in one forward sees the same shape.

    The tables matter beyond the lookup: given only `cu_seqlens`, `chunk_local_cumsum` and `solve_tril` build their own
    with `prepare_chunk_indices`, which reads the device tensor on the host (`.tolist()`) and uploads the result. Those
    two copies per layer are a device sync each -- 96 stalls in a Kev-27B pass -- and a host copy cannot be captured in
    an Ascend graph at all. Passed in, both helpers stay on the device."""
    key = (B, T, H, chunk_size, str(device))
    got = _META.get(key)
    if got is None:
        import triton
        _, _, _, _, _, prepare_chunk_indices = _load_helpers()
        cu = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int64)
        host = cu.cpu()
        table = lambda size: prepare_chunk_indices(host, size).to(device)
        ci = table(chunk_size)
        got = (cu, ci, tuple(ci.to(torch.int64).reshape(-1).tolist()), tuple(cu.tolist()),
               table(triton.next_power_of_2(CUMSUM_BUFFER // (H * chunk_size))), table(SOLVE_BLOCK))
        _META[key] = got
    return got


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False):
    """Drop-in for transformers' `torch_chunk_gated_delta_rule` on Ascend. query/key/value are [B, T, H, D] (H, D equal
    across q/k/v after the model's key-head repeat), g/beta are [B, T, H]. Returns (core_attn_out [B, T, H, Dv],
    last_recurrent_state [B, H, Dk, Dv] or None). Each of the B rows is packed as one cu_seqlens segment."""
    chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd, l2norm_fwd, _ = _load_helpers()
    B, T, H, Dk = query.shape
    Dv = value.shape[-1]
    out_dtype = query.dtype
    # The model hands these in as views (value is non-contiguous after the head reshape, query/key after the key-head
    # repeat); the triton_ascend helpers read them by stride and return garbage on a non-contiguous input, so make the
    # per-row layout explicit before packing the batch into one variable-length sequence.
    q = query.to(torch.bfloat16).contiguous(); k = key.to(torch.bfloat16).contiguous(); v = value.to(torch.bfloat16).contiguous()
    bet = beta.to(torch.bfloat16).contiguous(); gg = g.to(torch.float32).contiguous()
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q); k = l2norm_fwd(k)
    q = q.reshape(1, B * T, H, Dk); k = k.reshape(1, B * T, H, Dk); v = v.reshape(1, B * T, H, Dv)
    bet = bet.reshape(1, B * T, H); gg = gg.reshape(1, B * T, H)
    cu, ci, ci_host, cu_host, ci_cumsum, ci_solve = _meta(B, T, H, chunk_size, query.device)

    gc = chunk_local_cumsum(gg, chunk_size=chunk_size, cu_seqlens=cu, block_indices=ci_cumsum)
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=bet, g_cumsum=gc, cu_seqlens=cu, chunk_indices=ci, output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=cu, chunk_indices_bt=ci, chunk_indices_large_block=ci_solve, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=bet, A=A, g_cumsum=gc, cu_seqlens=cu, chunk_indices=ci)

    k_a = k.transpose(1, 2).contiguous(); w_a = w.transpose(1, 2).contiguous(); u_a = u.transpose(1, 2).contiguous()
    g_a = gc.transpose(1, 2).contiguous(); q_a = q.transpose(1, 2).contiguous()
    state = torch.zeros(B, H, Dk, Dv, device=query.device, dtype=torch.bfloat16) if initial_state is None else initial_state.to(torch.bfloat16).contiguous()
    h, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k_a, w_a, u_a, g=g_a, gk=None, initial_state=state, output_final_state=True, chunk_size=chunk_size,
        save_new_value=True, cu_seqlens=cu_host, chunk_indices=ci_host, use_exp2=False, transpose_state_layout=False)
    o = torch.ops._C_ascend.chunk_fwd_o(q_a, k_a, v_new, h, Dk ** -0.5, g=g_a, g_gamma=None,
        cu_seqlens=cu_host, chunk_indices=ci_host, chunk_size=chunk_size, transpose_state_layout=False)
    o = o.transpose(1, 2).contiguous().reshape(B, T, H, Dv).to(out_dtype)
    return o, (final_state if output_final_state else None)


def available():
    """Whether the NPU DeltaNet kernels can be loaded (vllm-ascend present and its custom ops register)."""
    try:
        _load_helpers()
        return True
    except Exception:
        return False


def prewarm():
    """Register the AscendC ops and read the device's core counts. Must run before the backbone initialises the NPU
    device: vllm-ascend defers its custom-op extension so `ASCEND_RT_VISIBLE_DEVICES` can still change before
    `torch.npu.set_device()`, and calling it after the device is live makes `chunk_gated_delta_rule_fwd_h` fail to launch
    (AclNN_Parameter_Error)."""
    _load_helpers()


def accelerate(lm):
    """Route every Qwen3.5 Gated DeltaNet layer's chunked recurrence through the Ascend kernels, in place. Patches the
    per-instance `chunk_gated_delta_rule` attribute the transformers layer calls, so it is independent of the LoRA merge.
    Returns the number of layers patched."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    _load_helpers()
    n = 0
    for module in lm.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            module.chunk_gated_delta_rule = chunk_gated_delta_rule
            n += 1
    return n


# --- fused serving layers (the Ascend counterpart of kev.fused_qwen35) ---

CAUSAL_BLOCK = 2048   # the compressed causal mask sparse_mode 3 reads, whatever the row length
_CAUSAL = {}


def _causal(device):
    """The compressed causal mask, one per device. Equal to a full [L, L] mask bit for bit on Kev's shapes."""
    mask = _CAUSAL.get(str(device))
    if mask is None:
        mask = _CAUSAL[str(device)] = torch.triu(torch.ones(CAUSAL_BLOCK, CAUSAL_BLOCK, device=device, dtype=torch.bool), 1)
    return mask


def _concat(*linears):
    """One NZ weight [sum(out), in] for several bias-free projections of the same input. FRACTAL_NZ is the layout the
    cube unit reads: the same bits in the order it wants them, so the result is bit for bit the ND one and the runtime
    stops converting on every call (13 ms of a 165-token Kev-27B pass)."""
    if any(l.bias is not None for l in linears): raise ValueError("fused projections assume bias-free Linear layers")
    return _nz(torch.cat([l.weight for l in linears], 0).contiguous())


def _nz(weight):
    """Cast one weight to FRACTAL_NZ. torch_npu only makes internal-format tensors while `allow_internal_format` is on,
    and that flag also sends the mixer's depthwise conv down the legacy aclop path, which an Ascend graph cannot
    capture, so it is turned on for the cast alone and off again. The cast tensors stay NZ."""
    import torch_npu
    torch.npu.config.allow_internal_format = True
    try:
        return torch_npu.npu_format_cast(weight, 29)   # ACL_FORMAT_FRACTAL_NZ
    finally:
        torch.npu.config.allow_internal_format = False


def rmsnorm_forward(self, x):
    """Qwen3_5RMSNorm.forward as one kernel. `gamma` is the reference's 1 + weight, kept in fp32: rounding it to bf16
    first would cost more than the norm itself does (relative error against the fp32 reference 1.4e-3 against 5.5e-3)."""
    import torch_npu
    return torch_npu.npu_rms_norm(x, self.gamma, self.eps)[0]


def gated_norm(out, z, gamma, eps):
    """The DeltaNet mixer's gated RMSNorm: norm(out) * weight, then the swish gate, over [tokens, head_v_dim] rows."""
    import torch_npu
    return (torch_npu.npu_rms_norm(out, gamma, eps)[0].float() * F.silu(z.float())).to(out.dtype)


def deltanet_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Qwen3_5GatedDeltaNet.forward with one projection GEMM and fused norms. No cache: on Ascend a pass never continues
    a cached DeltaNet state (kev.model.DecisionModel.probs_and_prefix recomputes the rows instead)."""
    if cache_params is not None:
        raise NotImplementedError("the fused Ascend layers do not keep a DeltaNet cache; load with fused=False (KEV_FUSED=0)")
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape
    mixed, z, b, a = F.linear(hidden_states, self.in_proj).split(self.splits, -1)
    mixed = F.silu(self.conv1d(mixed.transpose(1, 2))[:, :, :T]).transpose(1, 2)
    q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], -1)
    q = q.reshape(B, T, -1, self.head_k_dim)
    k = k.reshape(B, T, -1, self.head_k_dim)
    v = v.reshape(B, T, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    repeat = self.num_v_heads // self.num_k_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, 2)
        k = k.repeat_interleave(repeat, 2)
    out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
                                    use_qk_l2norm_in_kernel=True)
    out = gated_norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim), self.norm.gamma,
                     self.norm.variance_epsilon)
    return self.out_proj(out.reshape(B, T, -1))


def attention_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
    """Qwen3_5Attention.forward with one projection GEMM, fused q/k norms, fused rotary and the Ascend attention. Rows
    stay in [batch, tokens, heads, dim] from the projection to the output, which is the layout both fused ops take."""
    import torch_npu
    if past_key_values is not None:
        raise NotImplementedError("the fused Ascend layers do not keep an attention cache; load with fused=False (KEV_FUSED=0)")
    shape = hidden_states.shape[:-1]
    q_gate, k, v = F.linear(hidden_states, self.qkv).split(self.splits, -1)
    q, gate = q_gate.reshape(*shape, -1, 2 * self.head_dim).chunk(2, -1)
    cos, sin = (t.unsqueeze(2) for t in position_embeddings)
    q = rotary(torch_npu.npu_rms_norm(q.contiguous(), self.q_norm.gamma, self.q_norm.eps)[0], cos, sin)
    k = rotary(torch_npu.npu_rms_norm(k.reshape(*shape, -1, self.head_dim), self.k_norm.gamma, self.k_norm.eps)[0], cos, sin)
    v = v.reshape(*shape, -1, self.head_dim)
    # the inference attention, not npu_fusion_attention: the training op draws a dropout seed on the host, which an
    # Ascend graph capture refuses (the two agree to 1e-3, bf16 rounding)
    out = torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=self.config.num_attention_heads, num_key_value_heads=self.config.num_key_value_heads,
        input_layout="BSND", atten_mask=_causal(q.device), scale=self.scaling, sparse_mode=3, next_tokens=0)[0]
    return self.o_proj(out.reshape(*shape, -1) * torch.sigmoid(gate.reshape(*shape, -1))), None


def rotary(x, cos, sin):
    """apply_rotary_pos_emb for one [batch, tokens, heads, dim] tensor. Qwen3.5's rotary is partial: cos and sin cover
    the leading `cos.shape[-1]` dimensions of each head (64 of 256 here) and the rest passes through."""
    import torch_npu
    rot = cos.shape[-1]
    if rot == x.shape[-1]:
        return torch_npu.npu_rotary_mul(x, cos, sin)
    return torch.cat([torch_npu.npu_rotary_mul(x[..., :rot].contiguous(), cos, sin), x[..., rot:]], -1)


def mlp_forward(self, x):
    import torch_npu
    return self.down_proj(torch_npu.npu_swiglu(F.linear(x, self.gate_up)))


def text_pass(lm, inputs_embeds, position_ids, mixer_mask):
    """Embedding to final norm over one padded batch of causal rows: the body a graph captures and the eager pass runs.
    `position_ids` is [rows, tokens] or the reference's 3 or 4 mrope planes."""
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    text_position_ids, position_ids = (position_ids[0], position_ids[1:]) if position_ids.shape[0] == 4 else (None, position_ids)
    hidden_states = inputs_embeds
    position_embeddings = lm.rotary_emb(hidden_states, position_ids)
    for layer in lm.layers[: lm.config.num_hidden_layers]:
        hidden_states = layer(hidden_states, position_embeddings=position_embeddings,
                              attention_mask=mixer_mask if _layer_type(layer) == "linear_attention" else None,
                              position_ids=text_position_ids)
    return lm.norm(hidden_states)


def text_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                 inputs_embeds=None, use_cache=None, **kwargs):
    """Qwen3_5TextModel.forward for Kev's causal rows: no [rows, 1, L, L] float mask (the fused attention reads the
    compressed causal one instead), replayed from a captured graph when one fits the shape."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast
    if past_key_values is not None or use_cache:
        raise NotImplementedError("the fused Ascend layers keep no cache; load with fused=False (KEV_FUSED=0)")
    if attention_mask is not None and attention_mask.ndim != 2:
        raise NotImplementedError(f"the fused Ascend layers take causal rows with a [rows, tokens] mask, not a "
                                  f"{attention_mask.ndim}D one; load with fused=False (KEV_FUSED=0)")
    # the mixers zero padded tokens; the reference skips that when nothing is padded, and reads the mask to find out
    mixer_mask = None if attention_mask is None or bool(torch.all(attention_mask == 1)) else attention_mask
    tokens = input_ids if inputs_embeds is None else inputs_embeds   # ids, or embeddings the caller built itself
    if position_ids is None:
        position_ids = torch.arange(tokens.shape[1], device=tokens.device)[None].expand(tokens.shape[0], -1)
    if self.graphs is not None:
        hidden = self.graphs.run(tokens, position_ids, mixer_mask)
        if hidden is not None:
            return Qwen3_5ModelOutputWithPast(last_hidden_state=hidden, past_key_values=None)
    if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
    return Qwen3_5ModelOutputWithPast(last_hidden_state=text_pass(self, inputs_embeds, position_ids, mixer_mask),
                                      past_key_values=None)


# Padded row lengths a graph is captured for. Fine at the short end, where the pass is launch-bound and the padding is
# what it costs: a 609-token row rounded to 1024 instead of 640 ran 304 ms instead of ~200. Past the ladder a pass is
# device-bound and replay has nothing left to save, so it stays eager and any length works.
GRAPH_LENGTHS = (128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096)
MAX_GRAPHS = 16                # a graph holds only its own intermediates: 2-10 MiB at Kev's shapes
GRAPH_MIN_FREE = 6 * 2 ** 30   # HBM a capture leaves free (a 27B pass already holds 48 GiB of weights)


def _token_axis(t):
    """Which dimension of a pass input counts tokens: [rows, tokens] ids, positions or mask, [rows, tokens, hidden]
    embeddings (floating), [planes, rows, tokens] mrope positions."""
    return 1 if t.ndim == 2 or t.is_floating_point() else t.ndim - 1


def _slice(t, axis, start, stop=None):
    index = [slice(None)] * t.ndim
    index[axis] = slice(start, stop)
    return tuple(index)


class Graphs:
    """Captured passes of the fused text model, one per input shape (rows, padded length, ids or embeddings, ...).

    A Kev-27B pass is thousands of kernel launches and the host cannot enqueue them as fast as a 910B2 runs them: a
    165-token decision is 142 ms of device time and 205 ms of wall time, and the mixer's triton_ascend helpers cost
    ~170 us each to launch against 10-150 us of work. Replaying the pass as one graph launch gives that back (205 ->
    134 ms). The rows a shape is padded out to are causal right padding, which no real token attends to
    (`DecisionModel._pad_rows`), so a replay equals the eager pass up to bf16 reassociation: the padded length tiles the
    GEMMs differently (~0.4 % of the hidden states' scale on a 910B2), and replays of one graph are bit-identical. The
    output keeps the padded length; callers slice each row to its own.

    A shape is captured the first time it is seen, which costs that pass twice (capture records a run, it does not
    perform one, so the shape has to be warmed first) and every later pass of that shape a third of its time. Graphs
    share one memory pool and hold their own intermediates (2-10 MiB each at Kev's shapes).

    A caller may hand the text model embeddings and mrope position planes instead of ids and a [rows, tokens] position
    matrix; that is a shape like any other here."""

    def __init__(self, lm, pad_id):
        self.lm, self.pad_id = lm, pad_id
        self.pool = torch.npu.graph_pool_handle()
        self.entries, self.refused = {}, set()

    def fills(self, tokens):
        """What the padded tail of each input holds: pad tokens (or zero embeddings), position 0, masked out."""
        return (self.pad_id if not tokens.is_floating_point() else 0, 0, 0)

    def bucket(self, tokens):
        """The padded length this pass is captured at, or None for a pass past the ladder."""
        return next((n for n in GRAPH_LENGTHS if n >= tokens.shape[_token_axis(tokens)]), None)

    def key(self, inputs, length):
        """The padded shape of every input: what one captured graph can serve."""
        def shape(t):
            if t is None: return None
            padded = list(t.shape)
            padded[_token_axis(t)] = length
            return (*padded, t.dtype)
        return tuple(shape(t) for t in inputs)

    def room(self):
        free, _total = torch.npu.mem_get_info()
        return free > GRAPH_MIN_FREE

    def run(self, tokens, pos, mask):
        """-> hidden states for this pass from a replayed graph, or None to run it eagerly. `tokens` is the row ids, or
        their embeddings when the caller built them itself."""
        length = self.bucket(tokens)
        if length is None: return None
        inputs = (tokens, pos, mask)
        key = self.key(inputs, length)
        if key in self.refused: return None
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= MAX_GRAPHS or not self.room(): return None
            entry = self.capture(key, inputs, length)
            if entry is None: return None
        graph, buffers, out = entry
        for buffer, value, fill in zip(buffers, inputs, self.fills(tokens)):
            if buffer is None: continue
            axis = _token_axis(buffer)
            buffer[_slice(buffer, axis, 0, value.shape[axis])] = value
            buffer[_slice(buffer, axis, value.shape[axis])] = fill   # the tail is padding, not what the last pass left
        graph.replay()
        return out.clone()   # the graph writes this buffer again on the next replay

    def capture(self, key, inputs, length):
        """Buffers this shape's inputs padded out to its bucket length, then records one pass over them."""
        buffers = []
        for value, fill in zip(inputs, self.fills(inputs[0])):
            if value is None:
                buffers.append(None)
                continue
            axis = _token_axis(value)
            padded = list(value.shape)
            padded[axis] = length
            buffer = torch.full(padded, fill, device=value.device, dtype=value.dtype)
            buffer[_slice(buffer, axis, 0, value.shape[axis])] = value
            buffers.append(buffer)
        tokens, pos, mask = buffers
        embed = (lambda: tokens) if tokens.is_floating_point() else (lambda: self.lm.embed_tokens(tokens))
        graph = torch.npu.NPUGraph()
        device = tokens.device
        try:
            text_pass(self.lm, embed(), pos, mask)   # warm this shape: capture records a pass, it does not run one
            sync(device)
            with torch.npu.graph(graph, pool=self.pool):
                out = text_pass(self.lm, embed(), pos, mask)
            sync(device)
        except Exception as e:
            sync(device)
            self.refused.add(key)
            warnings.warn(f"kev.npu_qwen35: could not capture a graph for {tuple(inputs[0].shape)} padded to "
                          f"{length} tokens ({e}); running that shape eagerly")
            return None
        self.entries[key] = (graph, buffers, out)
        return self.entries[key]


@torch.no_grad()
def fuse(lm, pad_id=None, graphs=True):
    """Rewrite a merged Qwen3.5 text backbone in place for serving on Ascend (see the module docstring). The layers keep
    no cache and have no backward; the concatenated projections replace the originals, so the model holds no more weights
    than before. With `pad_id` the rewritten model also replays its passes from captured graphs (`Graphs`; `graphs=False`,
    which is what KEV_NPU_GRAPHS=0 reaches through `LoadOptions.npu_graphs`, to decline)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    if not all(hasattr(layer.mlp, "gate_proj") for layer in lm.layers):
        raise NotImplementedError("kev's fused Ascend layers cover the dense Qwen3.5 MLP, not mixture-of-experts layers; load with fused=False (KEV_FUSED=0)")
    for module in lm.modules():
        if isinstance(module, Qwen3_5RMSNorm):
            module.gamma = 1.0 + module.weight.float()
            module.forward = types.MethodType(rmsnorm_forward, module)
    for layer in lm.layers:
        if _layer_type(layer) == "linear_attention":
            m = layer.linear_attn
            m.in_proj = _concat(m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a)
            m.splits = [m.conv_dim, m.value_dim, m.num_v_heads, m.num_v_heads]
            m.norm.gamma = m.norm.weight.float()   # the mixer's gated norm is a plain weight, not the layers' 1 + w
            m.out_proj.weight.data = _nz(m.out_proj.weight.data)
            del m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a
            m.forward = types.MethodType(deltanet_forward, m)
        else:
            m = layer.self_attn
            m.qkv = _concat(m.q_proj, m.k_proj, m.v_proj)
            m.splits = [m.q_proj.out_features, m.k_proj.out_features, m.v_proj.out_features]
            m.o_proj.weight.data = _nz(m.o_proj.weight.data)
            del m.q_proj, m.k_proj, m.v_proj
            m.forward = types.MethodType(attention_forward, m)
        layer.mlp.gate_up = _concat(layer.mlp.gate_proj, layer.mlp.up_proj)
        layer.mlp.down_proj.weight.data = _nz(layer.mlp.down_proj.weight.data)
        del layer.mlp.gate_proj, layer.mlp.up_proj
        layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)
    lm.graphs = Graphs(lm, pad_id) if graphs and pad_id is not None else None
    lm.forward = types.MethodType(text_forward, lm)
    return lm


def _layer_type(layer):
    """Which mixer this decoder layer holds. transformers 5.5 calls it `layer_type`, later versions `block_type`."""
    return getattr(layer, "layer_type", None) or layer.block_type
