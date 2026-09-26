"""Decision model: causal LM backbone + block-causal branch mask + pointer readout."""
import copy, math, os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import LinearAttentionCacheLayerMixin

# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt, /opt, decide) so no
# embedding rows need to be added/trained; LoRA adapts their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
# training context: state tokens, tokens per question branch, and the whole packed record. Frozen suites are admitted with
# this rule (kev.suite) and training applies it to records built on the fly, so train and eval see the same population.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
# serving context (kev.serve): a state of up to 64k tokens (twice Jev's 32k; the Qwen3.5 / Qwen3.8 bases' window is 262k)
# and a question row (state + its branch, the encoder's max_branch) of up to 8k tokens more. Until 64k states these were
# 8,192 / 8,192 (SERVE_MAX_*_8K below).
SERVE_MAX_STATE = 65536
SERVE_MAX_BRANCH = SERVE_MAX_STATE + 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH
# tokens one inference pass holds (rows_per_pass), and the longest request an attention-only backbone runs packed
# (rows_form: its block-causal mask is L x L); longer ones run as rows
ROW_PASS_TOKENS = 16384
# the longest state a checkpoint may be trained on (kev.train --max_state): the longest one served, which still leaves
# every question its training branch budget (training_context(MAX_TRAIN_STATE)["max_branch"] <= SERVE_MAX_BRANCH)
MAX_TRAIN_STATE = SERVE_MAX_STATE
# the limits before 64k states: the suites frozen until then were admitted under these and record them (kev.suite.SERVING_CONTEXT_8K)
SERVE_MAX_STATE_8K, SERVE_MAX_BRANCH_8K, MAX_TRAIN_STATE_8K = 8192, 8192, 7552


def training_context(max_state=MAX_STATE):
    """The encoder limits for training with the state limit lifted to `max_state`: the row (state + one branch) and
    packed limits grow by the same amount, so every question keeps its token budget. training_context() is the default
    training context (kev.suite.CONTEXT without `truncate`)."""
    if not MAX_STATE <= max_state <= MAX_TRAIN_STATE:
        raise ValueError(f"max_state must be in [{MAX_STATE}, {MAX_TRAIN_STATE}]")
    extra = max_state - MAX_STATE
    return {"max_state": max_state, "max_branch": MAX_BRANCH + extra, "max_packed": MAX_PACKED + extra}


def rows_per_pass(rows, prefix_len=0, budget=ROW_PASS_TOKENS):
    """How many causal rows one inference forward pass takes: as many as fit `budget` tokens counting the cached state
    each row carries (prefix_len) plus its own tokens, at least one. Memory per pass is bounded by one maximal row however
    many questions a request has, and the rows are independent, so the answers do not depend on the split. Kev-4B on MLX,
    a 4.8k-token state with 64 questions: 24.7 GB peak in one pass, 8.9 GB one row at a time (and faster: the cost was
    64 copies of the state cache)."""
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


class ContextOverflow(ValueError):
    """A record does not encode within its context (state, branch or packed limit). Serving turns it into a 422; the
    benchmark counts it as a rejected record for suites scored as published (skip_overlong)."""


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def pad_id(tok):
    """The id used to right-pad token rows (never attended to); Qwen tokenizers define one, others fall back to 0."""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def is_hybrid(config):
    """Whether a (text) config has Gated DeltaNet layers (Qwen3.5). Such backbones cannot honour the block-causal mask and
    run the row form; on Apple Silicon they are what the MLX backend is for."""
    return "linear_attention" in set(getattr(config, "layer_types", None) or [])


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: instruction/state tokens, and the <decide> token


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (index of </opt> token for each option), opt (per-token option index within its
    question: OPT_NONE for state/instruction, 0..K-1 for option spans, OPT_DECIDE for <decide>).

    option_isolation=True: every option span is its own sub-branch (it sees state + instruction + itself only), all
    option spans share the same position ids, and <decide> sits at one fixed position after the longest span. Then the
    per-option representations and <decide>'s attention over them are permutation-invariant by construction.
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_tokens) + 1}")
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(f"branch too long: {len(br)} tokens with a {len(S)}-token state (row limit {max_branch})")
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0 + len(instr))) + [p0 + len(instr) + i for sp in spans for i in range(len(sp))] + [p0 + len(instr) + longest]
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    return {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation, "decide_idx": decide_idx, "opt_idx": opt_idx,
            "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + 1 > max_state}


def fits(rec, *tokenizers, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    """True when the internal record encodes strictly (no truncation) within the training context under every tokenizer
    given (frozen suites are admitted against the tokenizers of all their pinned bases)."""
    try:
        return all(len(encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)["ids"]) <= max_packed for tok in tokenizers)
    except ValueError:
        return False


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence.

    Padded key positions are masked for every query; padded query rows keep the diagonal so no row is fully
    masked (finfo.min, not -inf, so softmax stays finite either way). Real tokens never see pads because pads sit
    after them (causal) and belong to no segment (-1).

    opts (option isolation): within a question, an option-span token may attend to state, the instruction, and its own
    span only; <decide> attends to everything in its question. Instruction tokens never see option spans (causal)."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), L), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide", "opts"}: the branch tokens of question
    k with their (already state-continuing) positions, and the readout offsets *within the branch*. Feeding
    state + rows[k] as one causal row is equivalent to the packed block-causal form for that question, on any
    architecture: the row contains exactly the tokens question k may attend to, in the same positions."""
    seg = enc["seg"]; Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1                                    # <decide> is the last token of its branch
        if seg[start] != k or seg[end - 1] != k: raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        """dp = pointer dimension (head capacity knob)."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        # calibration: logits are divided by this at inference (eval mode) only. 1.0 = raw. A checkpoint carries the value fitted on
        # its in-distribution development rows (scripts/calibrate_checkpoint.py -> head.pt["temperature"]); training always sees T=1 so
        # a fitted value stays meaningful, and the argmax is unchanged by construction.
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature

    def many(self, h_decide, h_opts, owner):  # [Q,d], [sum K,d], question of each option [sum K] -> logits [sum K]
        """forward() for many questions at once (serving batches): one pass instead of one per question."""
        z = (self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


# What a loaded model exposes to kev.serve, kev.predictors and the Space: the scoring interface both DecisionModel (torch)
# and kev.mlx_model.MLXDecisionModel implement. tests/test_mlx.py checks the MLX class against this list.
SCORING_INTERFACE = ("encode", "forward", "probs", "probs_and_prefix", "probs_with_prefix", "probs_batch", "eval",
                     "head", "backend", "dtype", "device", "hybrid", "option_isolation", "prefix_min_tokens")


EAGER_STATES = 4   # long new states (past the graphed state pass) whose eager prefixes one batched run holds at once: each
                   # holds its state's keys, values and DeltaNet states (~0.3 GB for 2,200 tokens on Kev-27B)


def probs_one(model, enc, prefix, keep):
    """-> (probs, prefix to keep) for one request, on any backend: the question rows on the cached prefix on a hit, one
    pass that also returns the prefix when it is to be kept, the plain pass otherwise."""
    if prefix is not None: return model.probs_with_prefix(enc, prefix), prefix
    return model.probs_and_prefix(enc) if keep else (model.probs(enc), None)


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False, special_embeddings=False, lora_targets="all", dtype=torch.float32,
                 weights=None, direct_load=False):
        """weights: a full-weight checkpoint directory whose saved backbone replaces the base's (kev.checkpoint's loader rule).
        direct_load: load the backbone straight onto `device` (transformers device_map) instead of staging it in host memory;
        full-weight training on several GPUs in one container needs it (N processes x a 51 GB checkpoint otherwise). Off by
        default: it changes where the rotary buffers are computed, so every other path keeps its bits."""
        super().__init__()
        # backbone only (no vocab head): we never generate text.
        # eager on MPS/CPU (known-good with our float 4D mask); SDPA on CUDA (accepts arbitrary additive masks).
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        # dtype: fp32 for training and exact evaluation; bf16 is a serving option for large backbones (8B on a 32 GB Mac)
        load = {"dtype": dtype, "attn_implementation": attn}
        if direct_load: load["device_map"] = {"": torch.cuda.current_device() if device == "cuda" else device}   # "cuda": under torchrun, this rank's GPU
        self.lm = AutoModel.from_pretrained(weights, **load) if weights else AutoModelForCausalLM.from_pretrained(name, revision=revision, **load).model
        self.pad_id = pad_id(tok)
        # hybrid backbones (Qwen3.5: Gated DeltaNet layers, recurrent) cannot honour the block-causal mask, so every
        # question runs as its own causal row continuing from the state (rows_of). Attention-only backbones keep the
        # packed form; the two agree to fp32 noise (tests/test_model.py::test_rows_match_packed).
        self.hybrid = is_hybrid(self.lm.config)
        if self.hybrid and option_isolation: raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
        self.option_isolation = option_isolation
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = {"trainable_token_indices": {"embed_tokens": [tok.convert_tokens_to_ids(t) for t in SPECIAL]}} if special_embeddings else {}
            targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # "all" minus the DeltaNet projections on hybrids (retention ablation)
                       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}[lora_targets]
            if self.hybrid and lora_targets in ("all", "attn"):
                # Gated DeltaNet projections (transformers 5 names, verified on Qwen3_5TextModel); the mixer's out_proj too
                targets = targets + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets, **extra)
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    backend = "torch"           # kev.mlx_model.MLXDecisionModel is the other implementation of this scoring interface
    graphs = None               # kev.cuda_graphs.CudaGraphs for the serving passes of a hybrid backbone on CUDA (LoadOptions.cuda_graphs)

    @property
    def prefix_min_tokens(self):
        """kev.serve caches the state prefix from this many state tokens. Attention-only backbones: 384, below which the
        branch-only pass is not faster than one packed pass on MPS (per-op overhead). Hybrid backbones: always, because
        their miss path otherwise recomputes the state once per question (Kev-0.8B bf16 on MPS, 5 questions: 1011 -> 413 ms)."""
        return 0 if self.hybrid else 384

    @property
    def dtype(self):
        return str(next(self.lm.parameters()).dtype).removeprefix("torch.")

    def encode(self, tok, rec, **kw):
        """encode() with this model's option-isolation setting; use this from serving/eval code."""
        return encode(tok, rec, option_isolation=self.option_isolation, **kw)

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    SHAPE_BUCKET = int(os.environ.get("KEV_SHAPE_BUCKET", "64"))   # MPS: pad the sequence to a multiple of this (per-shape kernel warm-up); 1 disables

    def _pad_rows(self, rows):
        """Right-pad (ids, pos) token rows into [N, L] id / position tensors and a [N, L] attention mask (1 = real token).
        Pads sit after every real token and are masked keys, so they never change a real token's hidden state (parity
        measured exact). On MPS in eval mode L is rounded up to a SHAPE_BUCKET multiple so kernels are warmed per bucket."""
        L = max(len(ids) for ids, _ in rows)
        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device); pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device); att[i, : len(rid)] = 1
        return ids, pos, att

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records under the packed block-causal mask."""
        ids, pos, _ = self._pad_rows([(e["ids"], e["pos"]) for e in encs])
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        lm_dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=lm_dtype, opts=[e["opt"] for e in encs] if isolate else None, length=ids.shape[1])
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()   # head stays fp32

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def rows_form(self, encs):
        """Whether these records run as causal rows: hybrid backbones always (the recurrent layers cannot honour the packed
        mask), attention-only ones when the packed sequence would exceed ROW_PASS_TOKENS (its L x L mask grows without
        bound with the number of questions). The two forms agree (tests/test_model.py::test_rows_match_packed)."""
        return self.hybrid or any(len(e["ids"]) > ROW_PASS_TOKENS for e in encs)

    def _rows_hidden(self, rows, cache=None, prefix_len=0):
        """Hidden states of causal token rows, one [L_i, d] tensor per row. In eval mode the rows go through the backbone
        rows_per_pass at a time; training keeps one batch (its batches are small and autograd needs the whole graph anyway).
        With `cache`, the rows are branches continuing the cached state: in eager mode the cache is replicated once per
        chunk, leaving the caller's prefix pristine, and the cached tokens are marked real in the attention mask."""
        chunk = len(rows) if self.training else rows_per_pass([ids for ids, _ in rows], prefix_len)
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad_rows(part)
            past = {}
            if cache is not None:
                replica = copy.copy(cache)
                replica.layers = [copy.copy(layer) for layer in cache.layers]
                for source, target in zip(cache.layers, replica.layers):
                    if isinstance(source, LinearAttentionCacheLayerMixin):
                        target.conv_states = source.conv_states.copy()
                        target.recurrent_states = source.recurrent_states.copy()
                        target.is_conv_states_initialized = source.is_conv_states_initialized.copy()
                        target.is_recurrent_states_initialized = source.is_recurrent_states_initialized.copy()
                        target.has_previous_state = source.has_previous_state.copy()
                        target.conv_kernel_size = source.conv_kernel_size.copy()
                replica.reorder_cache(torch.zeros(len(part), dtype=torch.long, device=self.device))
                att = torch.cat([torch.ones((len(part), prefix_len), dtype=torch.long, device=self.device), att], 1)
                past = {"past_key_values": replica, "use_cache": True}
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, **past).last_hidden_state.float()
            out += [h[i, : len(row_ids)] for i, (row_ids, _) in enumerate(part)]
        return out

    def forward_rows_batch(self, encs, shared_prefix=False):
        """Row form: every question of every record is one causal row = state tokens + its branch tokens. Returns the same
        nested logits as forward_batch. Exact isolation by construction (rows are independent); the state is recomputed
        per row (Q x state tokens), which training accepts; serving and probs() use the prefix cache instead. shared_prefix
        (training, hybrid Qwen3.5): the same rows computed with each state run once and its branches continuing from it,
        gradients included (kev.shared_prefix)."""
        if shared_prefix:
            from .shared_prefix import branch_hidden   # here, not at the top: the HF Space vendors model.py alone
            splits = [rows_of(e) for e in encs]
            return [[self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]) for h, r in zip(hs, rows)]
                    for hs, (_, _, rows) in zip(branch_hidden(self.lm, splits, self.pad_id, self.device), splits)]
        rows, readouts = [], []   # one causal row per question; readouts[i] = (record, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"])); readouts.append((b, len(S) + r["decide"], [len(S) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """Returns list of logits tensors, one per question."""
        return self.forward_batch([enc])[0]

    def forward_batch(self, encs, shared_prefix=False):
        """List (per record) of lists (per question) of logits. Row form or packed block-causal mask, see rows_form
        (the packed mask already runs each state once; shared_prefix does that for the row form of a hybrid backbone)."""
        if self.rows_form(encs): return self.forward_rows_batch(encs, shared_prefix and self.hybrid)
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        """Probabilities per question. A record that runs as rows (every record on a hybrid backbone) takes the serving miss
        path: its state once, then the question rows from its cache. forward() keeps the row form, which re-runs the state
        per question; the two agree to fp32 rounding (#77)."""
        if self.rows_form([enc]): return self.probs_and_prefix(enc)[0]
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    # --- state-prefix reuse (serving): the state is encoded once, question branches attend to its cached keys/values.
    # Exact by construction: branch tokens never attend to each other across questions (block-causal mask) and the state
    # never sees the branches (causal), so the state's hidden states and KV are identical with or without the branches.

    def _branch_rows_from_prefix(self, enc, cache):
        """Row-form serving: the branches run as causal rows continuing the cached state (exactly the forward_rows_batch
        layout, minus the recomputed state)."""
        S, _, rows = rows_of(enc)
        hs = self._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=len(S))
        ps = [F.softmax(self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]), -1) for h, r in zip(hs, rows)]
        return list(torch.cat(ps).cpu().split([len(p) for p in ps]))   # one device sync for the request, not one per question

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only. Returns (n_state_tokens, kv cache, state hidden states [Ls, d] fp32 or None). Only the
        packed pass (probs_with_prefix) reads the hidden states; a hybrid backbone always runs rows, so it keeps None, as the
        graphed prefixes do (a cached fp32 [Ls, d] copy is 160 MiB for 8k tokens on Kev-27B). An attention-only backbone
        keeps them even when this record runs as rows: the same state with fewer questions may be packed."""
        Ls = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device); pos = torch.tensor([enc["pos"][:Ls]], device=self.device)
        # the cache must know the layer types (hybrid backbones keep recurrent + conv states per DeltaNet layer)
        out = self.lm(input_ids=ids, position_ids=pos, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        return Ls, out.past_key_values, None if self.hybrid else out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One full pass that also returns the state prefix (KV cropped to the state, state hidden states or None, see prefix): a cache miss
        costs a single forward pass, not two."""
        Ls = enc["seg"].count(0)
        if self.rows_form([enc]):
            # recurrent layers cannot be cropped back to the state (and an over-long packed pass is what the row form avoids),
            # so a miss here is a state pass (kept as the prefix) plus the branch rows
            Ls, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, cache), (Ls, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device); pos = torch.tensor([enc["pos"]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)
        out = self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, past_key_values=DynamicCache(config=self.lm.config), use_cache=True)
        h = out.last_hidden_state[0].float()
        out.past_key_values.crop(-(len(enc["ids"]) - Ls))     # keep the state only (negative = drop that many trailing tokens; positive form deprecated in transformers 5)
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (Ls, out.past_key_values, h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """probs() for a record whose state tokens equal the cached prefix's; only the branches run. The cache is cropped
        back to the state afterwards so it can be reused."""
        Ls, cache, h_state = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        if self.rows_form([enc]):
            return self._branch_rows_from_prefix(enc, cache)
        ids = torch.tensor([enc["ids"][Ls:]], device=self.device); pos = torch.tensor([enc["pos"][Ls:]], device=self.device)
        dt = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([enc["seg"]], self.device, dtype=dt, opts=[enc["opt"]] if enc.get("option_isolation") else None)[:, :, Ls:, :]
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - Ls))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]

    @torch.no_grad()
    def probs_batch(self, encs, prefixes, keep):
        """kev.serve's model thread: several requests at once. prefixes[i] is request i's cached state prefix or None;
        keep[i] whether to return its new prefix. -> (probs per request, prefix per request: the cached one, the new one if
        kept, else None). With CUDA graphs the requests the graphed passes admit run together (kev.cuda_graphs: shared
        state and row passes; a state too long for the graphed state pass gets its own eager pass first); the rest, and
        every other backend, one at a time. Rows are independent, so a request's answers do not depend on the batch."""
        splits = [rows_of(e) for e in encs]
        def fits(i, cached):
            S, _, rows = splits[i]
            return self.graphs is not None and self.graphs.admits(len(S), [len(r["ids"]) for r in rows], cached)
        long = {i for i in range(len(encs)) if prefixes[i] is None and not fits(i, False) and fits(i, True)}
        batched = [i for i in range(len(encs)) if i in long or fits(i, prefixes[i] is not None)]
        out = {i: probs_one(self, encs[i], prefixes[i], keep[i]) for i in sorted(set(range(len(encs))) - set(batched))}
        runs, held = [[]], 0   # graphed runs, each holding at most EAGER_STATES long states' eager prefixes at once
        for i in batched:
            if i in long and held == EAGER_STATES: runs.append([]); held = 0
            runs[-1].append(i); held += i in long
        from .cuda_graphs import Request   # here, not at the top: the HF Space vendors model.py without cuda_graphs.py
        for run in filter(None, runs):
            pre = {i: self.prefix(encs[i]) if i in long else prefixes[i] for i in run}
            X, caches = self.graphs.run([Request(splits[i][0], splits[i][1], [(r["ids"], r["pos"]) for r in splits[i][2]],
                                                 None if pre[i] is None else pre[i][1], keep[i], [[r["decide"], *r["opts"]] for r in splits[i][2]])
                                         for i in run])   # per question its picks: the <decide> position, then its options
            ps = iter(self._readout_many(X, [len(r["opts"]) for i in run for r in splits[i][2]]))
            for i, cache in zip(run, caches):
                made = pre[i] if i in long else None if cache is None else (len(splits[i][0]), cache, None)
                out[i] = [next(ps) for _ in splits[i][2]], prefixes[i] or (made if keep[i] else None)
        return [out[i][0] for i in range(len(encs))], [out[i][1] for i in range(len(encs))]

    def _readout_many(self, X, ks):
        """Probabilities of many questions from their picked hidden states X, laid out per question as its <decide> then
        its ks[q] options, in a fixed number of kernels and one device sync. -> one tensor per question."""
        owner = [q for q, k in enumerate(ks) for _ in range(k)]
        slot = [j for k in ks for j in range(k)]
        starts = [0]
        for k in ks: starts.append(starts[-1] + 1 + k)
        dec = torch.tensor(starts[:-1]).to(self.device, non_blocking=True)
        opt, own, sl = torch.tensor([[starts[q] + 1 + j for q, j in zip(owner, slot)], owner, slot]).to(self.device, non_blocking=True)
        z = self.head.many(X[dec], X[opt], own)
        Z = torch.full((len(ks), max(ks)), float("-inf"), device=self.device).index_put_((own, sl), z)
        return torch.softmax(Z, -1)[own, sl].cpu().split(ks)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
