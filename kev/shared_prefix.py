"""Training forward through a shared state prefix, for the hybrid Qwen3.5 backbones (kev.train --shared_prefix).

The row form (kev.model.forward_rows_batch) trains a record with q questions as q causal rows, state + one branch each,
so the state runs q times. Here each record's state runs once and its branches continue from it, which costs
state + the branches instead of q x (state + branch). The two are the same computation: a branch token attends to the
same tokens at the same positions, and it reaches the state through what each layer leaves behind at the end of the
state, as the layer's own cache path reads it for inference (kev.model._branch_rows_from_prefix): the attention layers'
keys and values, and the Gated DeltaNet layers' short-convolution window (the conv's left context) and recurrent state
(the chunked delta rule's `initial_state`, which fla differentiates). Two things differ from serving:
- `Prefix` holds that per-layer state functionally (transformers' cache layers copy into static buffers in place, which
  autograd cannot see through), so every branch's gradient flows back into the state and accumulates there;
- each decoder layer runs its state pass and its branch pass inside one step, and gradient checkpointing (when the
  backbone has it on) recomputes the step as a whole, so no cache outlives its layer or is written twice on replay.
States of a batch are left-padded (the padded positions are zeroed and stay zero: a zero input leaves the DeltaNet state,
the conv window and the residual stream unchanged), branches right-padded.
Exactness: fp32 with transformers' reference kernels matches the row form to ~1e-5 (logits and every gradient; tiny
random Qwen3.5 in tests/test_unit.py, Qwen3.5-0.8B-Base in tests/test_model.py). On CUDA fla's Triton kernels round their
fp32 dots like TF32 and the shared prefix left-pads states, so it rounds differently from the row form: measured against the
exact kernels, it is as far from the exact answer as the row form's own batchings are, with no systematic shift.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method
from torch.utils.checkpoint import checkpoint


class Prefix:
    """One decoder layer's view of the shared state, duck-typing the cache calls a Qwen3.5 layer makes. The state pass
    stores what the layer leaves (keys and values, or the conv window and the recurrent state); `branch` gathers it once
    per branch row; the branch pass reads it. Nothing is written in place."""
    record_past = True   # keeps a one-token pass off the in-place decode kernel

    def __init__(self):
        self.keys = self.values = self.conv = None
        self.recurrent_states = {}

    @property
    def layers(self):   # layer code reads `cache.layers[layer_idx].recurrent_states[0]`: every index is this prefix
        return self

    def __getitem__(self, layer_idx):
        return self

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return self.conv is not None

    def update_conv_state(self, conv_input, layer_idx, conv_kernel_size, **kwargs):
        if self.conv is not None: return torch.cat([self.conv, conv_input], -1)   # the branch sees the state's last inputs
        self.conv = F.pad(conv_input, (max(0, conv_kernel_size - conv_input.shape[-1]), 0))[..., -conv_kernel_size:]
        return conv_input

    def update_recurrent_state(self, recurrent_state, layer_idx, **kwargs):
        self.recurrent_states.setdefault(0, recurrent_state)   # the state's final state; a branch's own is discarded
        return recurrent_state

    def update(self, keys, values, layer_idx, *args, **kwargs):
        if self.keys is None:
            self.keys, self.values = keys, values
            return keys, values
        return torch.cat([self.keys, keys], -2), torch.cat([self.values, values], -2)

    def branch(self, owner):
        """Gather the stored state per branch row (owner[i] = the record of branch i)."""
        pick = lambda t: None if t is None else t.index_select(0, owner)
        self.keys, self.values, self.conv = pick(self.keys), pick(self.values), pick(self.conv)
        self.recurrent_states = {k: pick(v) for k, v in self.recurrent_states.items()}


def _masks(allow, dtype, attn):
    """Additive float mask for eager attention, boolean for SDPA (which also takes a float mask, but only in the query's
    dtype, and autocast changes that)."""
    allow = allow[:, None]
    return allow if attn == "sdpa" else torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)


def branch_hidden(lm, splits, pad_id, device):
    """splits[b] = kev.model.rows_of(record b's encoding). -> per record, per question, the branch's final hidden states
    [branch length, d] in fp32 (what the pointer head reads; the state's own hidden states are never needed)."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    owner = [b for b, (_, _, rows) in enumerate(splits) for _ in rows]
    branches = [r for _, _, rows in splits for r in rows]
    Ls, Lb = max(len(s) for s, _, _ in splits), max(len(r["ids"]) for r in branches)
    s_ids = torch.full((len(splits), Ls), pad_id, device=device); s_pos = torch.zeros_like(s_ids); s_real = torch.zeros_like(s_ids, dtype=torch.bool)
    for b, (ids, pos, _) in enumerate(splits):   # left-padded
        s_ids[b, Ls - len(ids):] = torch.tensor(ids, device=device); s_pos[b, Ls - len(ids):] = torch.tensor(pos, device=device); s_real[b, Ls - len(ids):] = True
    b_ids = torch.full((len(branches), Lb), pad_id, device=device); b_pos = torch.zeros_like(b_ids); b_real = torch.zeros_like(b_ids, dtype=torch.bool)
    for i, r in enumerate(branches):   # right-padded; their positions already continue the state
        b_ids[i, :len(r["ids"])] = torch.tensor(r["ids"], device=device); b_pos[i, :len(r["pos"])] = torch.tensor(r["pos"], device=device); b_real[i, :len(r["ids"])] = True
    run = _method(base, "kev_shared_prefix", _forward)   # on an FSDP2 root: embeddings and final norm gathered as for base(...)
    h_b = run(s_ids, s_pos, s_real, b_ids, b_pos, b_real, torch.tensor(owner, device=device))
    out = [[] for _ in splits]
    for i, b in enumerate(owner): out[b].append(h_b[i, :len(branches[i]["ids"])])
    return out


def _forward(base, s_ids, s_pos, s_real, b_ids, b_pos, b_real, owner):
    """The decoder stack over left-padded states and right-padded branches (owner[i] = the state of branch i)."""
    device, (Ls, Lb) = s_ids.device, (s_ids.shape[1], b_ids.shape[1])
    h_s = base.embed_tokens(s_ids) * s_real[..., None]; h_b = base.embed_tokens(b_ids) * b_real[..., None]
    dtype, attn = h_s.dtype, base.config._attn_implementation
    # attention: a state token sees the real state tokens up to itself; a branch token its record's real state and its
    # own branch so far; a padded query keeps its diagonal so no softmax row is empty (its output is zeroed or unused)
    causal = lambda n: torch.ones(n, n, dtype=torch.bool, device=device).tril()
    eye = lambda n: torch.eye(n, dtype=torch.bool, device=device)
    allow_s = (causal(Ls)[None] & s_real[:, None, :]) | eye(Ls)[None]
    allow_b = torch.cat([s_real[owner][:, None, :].expand(-1, Lb, -1), (causal(Lb)[None] & b_real[:, None, :]) | eye(Lb)[None]], -1)
    masks_s = {"full_attention": _masks(allow_s, dtype, attn), "linear_attention": s_real.to(dtype)}
    masks_b = {"full_attention": _masks(allow_b, dtype, attn), "linear_attention": b_real.to(dtype)}
    rope_s, rope_b = base.rotary_emb(h_s, s_pos[None].expand(3, -1, -1)), base.rotary_emb(h_b, b_pos[None].expand(3, -1, -1))

    recompute = base.gradient_checkpointing and base.training and torch.is_grad_enabled()
    for layer in base.layers:
        step = _method(layer, "kev_shared_prefix_step", _step)
        args = (h_s, h_b, (rope_s, rope_b), (masks_s[layer.block_type], masks_b[layer.block_type]), (s_pos, b_pos), s_real, owner)
        h_s, h_b = checkpoint(step, *args, use_reentrant=False) if recompute else step(*args)
    return base.norm(h_b).float()


def _step(layer, h_s, h_b, ropes, masks, positions, s_real, owner):
    """One decoder layer: the state pass, then the branch pass reading what it left (`Prefix`). A single call on the layer
    (both passes call its plain `forward`), so an FSDP2 unit is gathered once for both and checkpointing replays both."""
    prefix = Prefix()
    h_s = layer.forward(h_s, position_embeddings=ropes[0], attention_mask=masks[0], position_ids=positions[0], past_key_values=prefix)
    prefix.branch(owner)
    h_b = layer.forward(h_b, position_embeddings=ropes[1], attention_mask=masks[1], position_ids=positions[1], past_key_values=prefix)
    return h_s * s_real[..., None], h_b


def _method(module, name, fn):
    """`fn` bound to `module` as `name`; on an FSDP2 unit also registered as a forward method, so the unit's parameters
    are gathered around the call and resharded after it, and backward re-gathers them from its outputs' hooks."""
    if not hasattr(module, name):
        setattr(module, name, types.MethodType(fn, module))
        if isinstance(module, FSDPModule): register_fsdp_forward_method(module, name)
    return getattr(module, name)
