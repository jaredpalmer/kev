"""Fused CUDA kernels for serving the Qwen3.5 decoder layers (flash-linear-attention's Triton ops).

With CUDA graphs and batching (kev.cuda_graphs) a served pass is limited by the GPU work itself, and transformers'
reference Qwen3.5 layers spend over a third of it outside the matrix multiplies: the causal convolution falls back to a
PyTorch depthwise conv behind a concatenation of the cached conv state, the gating, the query/key head repeat and the
gated norm run as a dozen separate fp32 elementwise kernels per DeltaNet layer, and every RMSNorm and SwiGLU is several
more. fuse() rewrites those layers in place for inference:
- DeltaNet mixer: one projection GEMM for q/k/v, z, b and a (their weights concatenated); fla's Triton causal conv
  that starts from the cached conv state (no concatenation); fla's chunked gated delta rule with the gate
  (-exp(A_log) * softplus(a + dt_bias)), the beta sigmoid and the q/k L2 norm computed inside the kernel and the key
  heads shared by their value heads (no repeat); fla's fused gated RMSNorm.
- MLP: one GEMM for gate and up (weights concatenated) and a fused SwiGLU.
- the decoder layer's two zero-centred RMSNorms as fla's fused RMSNorm with weight 1 + w in fp32, the second one also
  adding the residual.
- attention: one GEMM for q (with its output gate), k and v, the q/k RMSNorms fused, the sigmoid output gate fused.
The math is the reference's; the rounding is not (the fused kernels keep fp32
where the reference rounds to bf16 in between), so fused and reference answers agree to bf16 noise, like the graphs.
Only for serving: there is no backward, the fused projections replace the originals (the merged LoRA is inside), and a
pass that continues a cached DeltaNet state does not advance it (see deltanet_forward).
"""
import types

import torch
import torch.nn.functional as F
from fla.modules.activations import sigmoidglu, swiglu
from fla.modules.conv import causal_conv1d
from fla.modules.fused_norm_gate import rms_norm_gated
from fla.modules.layernorm import rms_norm
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states, apply_rotary_pos_emb, eager_attention_forward


class _FixedNB:
    """A Triton kernel launched with NB=1. fla passes these kernels NB, a token-count class (cdiv of the rows by a few
    thousand), as a constexpr and an autotuning key, but their bodies never read it: every new batch shape compiled and
    autotuned them again, stalling a busy server for about a second each time."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        return lambda *args, **kwargs: launch(*args, **{**kwargs, "NB": 1})


def _fix_nb():
    import fla.modules.conv.triton.ops as conv, fla.modules.fused_norm_gate as gated, fla.modules.l2norm as l2, fla.modules.layernorm as norm
    for module, name in ((conv, "causal_conv1d_fwd_kernel"), (gated, "layer_norm_gated_fwd_kernel"), (l2, "l2norm_fwd_kernel"), (norm, "layer_norm_fwd_kernel")):
        if not isinstance(getattr(module, name), _FixedNB): setattr(module, name, _FixedNB(getattr(module, name)))


FLA_VERSION = "0.5.2"   # the flash-linear-attention these kernels and _fix_nb were written and measured against


def _concat(*linears):
    """One weight [sum(out), in] for several bias-free projections of the same input."""
    if any(l.bias is not None for l in linears): raise ValueError("fused projections assume bias-free Linear layers")
    return torch.cat([l.weight for l in linears], 0).contiguous()


def deltanet_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Qwen3_5GatedDeltaNet.forward with fused kernels. Cache contract: a pass on an empty cache (a state) fills it, as the
    reference does; a pass continuing a cached state (question rows) reads it and leaves it as it was. Kev's question rows
    never continue from each other, and writing back every row's final states cost as much as reading them."""
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    B, T, _ = hidden_states.shape
    mixed, z, b, a = F.linear(hidden_states, self.in_proj).split(self.splits, -1)
    layer = cache_params.layers[self.layer_idx] if cache_params is not None else None
    previous = layer is not None and layer.has_previous_state[0]
    fill = layer is not None and not previous
    mixed, conv_state = causal_conv1d(mixed, self.conv_weight, None, initial_state=layer.conv_states[0] if previous else None,
                                      output_final_state=fill, activation="silu")
    q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], -1)
    out, recurrent = chunk_gated_delta_rule(
        q.reshape(B, T, -1, self.head_k_dim), k.reshape(B, T, -1, self.head_k_dim), v.reshape(B, T, -1, self.head_v_dim),
        g=a, beta=b, initial_state=layer.recurrent_states[0] if previous else None, output_final_state=fill,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=self.A_log, dt_bias=self.dt_bias, use_beta_sigmoid_in_kernel=True)
    if fill:
        if not layer.is_conv_states_initialized[0]: layer.lazy_initialization(conv_states=conv_state, conv_kernel_size=conv_state.shape[-1])
        layer.conv_states[0].copy_(conv_state)   # in place: graph buffers keep their address
        layer.has_previous_state[0] = True
        cache_params.update_recurrent_state(recurrent, self.layer_idx)
    out = rms_norm_gated(out, z.reshape(B, T, -1, self.head_v_dim), self.norm.weight, None, activation="swish", eps=self.norm.variance_epsilon)
    return self.out_proj(out.reshape(B, T, -1))


def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    """Qwen3_5Attention.forward with one projection GEMM, fused q/k RMSNorms and a fused sigmoid output gate."""
    shape = hidden_states.shape[:-1]
    q_gate, k, v = F.linear(hidden_states, self.qkv).split(self.splits, -1)
    q, gate = q_gate.reshape(*shape, -1, 2 * self.head_dim).chunk(2, -1)
    q = rms_norm(q, self.q_norm_weight, None, eps=self.q_norm.eps).transpose(1, 2)
    k = rms_norm(k.reshape(*shape, -1, self.head_dim), self.k_norm_weight, None, eps=self.k_norm.eps).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    v = v.reshape(*shape, -1, self.head_dim).transpose(1, 2)
    if past_key_values is not None: k, v = past_key_values.update(k, v, self.layer_idx)
    attend = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
    out, weights = attend(self, q, k, v, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs)   # [B, T, heads, dim]: transposed back already
    return self.o_proj(sigmoidglu(gate.reshape(*shape, -1), out.reshape(*shape, -1))), weights


def mlp_forward(self, x):
    gate, up = F.linear(x, self.gate_up).chunk(2, -1)
    return self.down_proj(swiglu(gate, up))


def decoder_forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
    """Qwen3_5DecoderLayer.forward with fused norms: the post-mixer norm also adds the residual."""
    residual = hidden_states
    h = rms_norm(hidden_states, self.input_norm_weight, None, eps=self.input_layernorm.eps)
    if self.block_type == "linear_attention":
        h = self.linear_attn(hidden_states=h, cache_params=past_key_values, attention_mask=attention_mask, **kwargs)
    else:
        h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
                              position_embeddings=position_embeddings, **kwargs)
    h, residual = rms_norm(h, self.post_norm_weight, None, residual=residual, eps=self.post_attention_layernorm.eps, prenorm=True)
    return residual + self.mlp(h)


@torch.no_grad()
def fuse(lm):
    """Rewrite a Qwen3.5 text backbone (DecisionModel.lm, merged) in place for fused-kernel serving. Contract changes
    against the reference: no backward, and a pass that continues a cached DeltaNet state leaves that state as it was
    (Kev's question rows never continue from each other). Needs flash-linear-attention FLA_VERSION exactly: it patches
    fla's kernel launches (_fix_nb)."""
    import fla
    if fla.__version__ != FLA_VERSION:
        raise RuntimeError(f"kev's fused kernels need flash-linear-attention=={FLA_VERSION}, found {fla.__version__}; "
                           "install that version or load with LoadOptions(fused=False) (KEV_FUSED=0 for kev.serve)")
    _fix_nb()
    for layer in lm.layers:
        if layer.block_type == "linear_attention":
            m = layer.linear_attn
            m.in_proj = _concat(m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a)
            m.splits = [m.conv_dim, m.value_dim, m.num_v_heads, m.num_v_heads]
            m.conv_weight = m.conv1d.weight.squeeze(1).contiguous()
            del m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a
            m.forward = types.MethodType(deltanet_forward, m)
        else:
            m = layer.self_attn
            m.qkv = _concat(m.q_proj, m.k_proj, m.v_proj)
            m.splits = [m.q_proj.out_features, m.k_proj.out_features, m.v_proj.out_features]
            m.q_norm_weight, m.k_norm_weight = 1.0 + m.q_norm.weight.float(), 1.0 + m.k_norm.weight.float()
            del m.q_proj, m.k_proj, m.v_proj
            m.forward = types.MethodType(attention_forward, m)
        layer.mlp.gate_up = _concat(layer.mlp.gate_proj, layer.mlp.up_proj)
        del layer.mlp.gate_proj, layer.mlp.up_proj
        layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)
        layer.input_norm_weight = 1.0 + layer.input_layernorm.weight.float()
        layer.post_norm_weight = 1.0 + layer.post_attention_layernorm.weight.float()
        layer.forward = types.MethodType(decoder_forward, layer)
    return lm
