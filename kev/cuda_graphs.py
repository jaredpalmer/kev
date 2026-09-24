"""CUDA graphs for serving the hybrid (Qwen3.5) backbones on CUDA, batched across requests.

A served request is two kinds of forward pass: the state pass (the document, on a prefix-cache miss) and the row pass (one
causal row per question, continuing the cached state). At a few hundred tokens the GPU finishes a pass long before Python
finishes issuing it: a pass launches ~2,000 kernels, and the flash-linear-attention wrapper around every DeltaNet layer
costs ~1 ms of CPU. Replaying a captured graph issues the same kernels in one call. And because one request leaves most of
the GPU idle, kev.serve hands this module every request that is waiting when the GPU frees up: one state pass computes all
their new states, one row pass all their questions.

Two sets of buffers, shared by every graph:
- the state bank: per layer the attention keys/values [GRAPH_STATES, heads, BANK_WIDTH, dim], each state right-aligned at
  the end, and the DeltaNet conv/recurrent states. The state pass writes new states into it; cached states are copied in.
  One fixed layout, so a row pass reads any state the same way whatever state pass wrote it.
- the row buffers: [rows, heads, Sr + Lb, dim] per attention layer and the DeltaNet states per question row. The row pass
  starts by gathering each row's state from the bank (one index_select per tensor, inside the graph), then writes the
  row's own keys after it.
Both are flat buffers viewed at the graph's shape, so memory is fixed however many shapes appear. Graph keys are kept
coarse because a batch's shape changes with the traffic: state passes by (states, state bucket), row passes by (rows,
row bucket, Sr = the longest state rounded up to a power of two, which only lengthens attention).

A graph has fixed shapes, so passes are padded to buckets: token counts to bucket() (under a quarter more), row and state counts
to count_bucket() (empty entries are masked). The padding is masked exactly:
- the state pass is LEFT-padded. The DeltaNet layers see zeroed inputs at the pads (their padding mask), so keys, values
  and queries are zero and the recurrent state is still zero when the real tokens start; the causal convolution sees
  the same zeros its own padding supplies. The attention layers mask the pads as keys, and a pad query attends to itself
  so no row is fully masked (a NaN there would survive the zeroing: NaN * 0 = NaN). Real tokens keep their position ids.
- question rows are right-padded (after every real token, as the eager path already does); a state's keys are
  right-aligned in the bucket and the slots before them are masked.
So the results equal the eager passes up to floating-point reassociation (another chunking of the DeltaNet scan, other
GEMM shapes), not bit for bit, and a request's answers do not depend on what else shares its batch.

Capturing a graph costs ~0.4 s (an H100, Kev-4B), so the first pass of a new bucket runs the same code eagerly from the
same buffers and the bucket joins `pending`; capture_pending() captures them later (kev.serve does it when no request is
waiting). Replays run one at a time (kev.serve has one model thread), each refills what it reads, and its results are
copied out before the next replay; that is what makes the shared buffers and the shared memory pool safe.
"""
from collections import OrderedDict
from typing import NamedTuple

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

# Limits of the graphed passes (not the model's context: that is kev.model.MAX_STATE / SERVE_MAX_STATE)
GRAPH_TOKENS = 32768   # rows x positions one row pass may hold in the attention buffers (about 1 GB on Kev-4B and 9B)
GRAPH_STATE = 1024     # longest state bucket the state pass graphs. A longer state pass is compute-bound, and the padding plus
                       # the explicit mask (no causal flash attention) made a 2,200-token one slower as a graph (L40S, Kev-4B:
                       # 223 vs 208 ms per request), so it runs eagerly
GRAPH_ROW = 1024       # longest question-row bucket the row pass graphs
GRAPH_ROWS = 32        # question rows per graphed pass; more run as several replays
GRAPH_STATES = 16      # states per state pass: the bank's entries
BANK_WIDTH = 4096      # positions per bank entry (states are right-aligned in it; about 2 GB for 16 entries on Kev-4B and 9B)
GRAPHS_KEPT = 256      # captured graphs kept, least recently used evicted (a busy server met ~160 on mixed traffic)
PAD_SPLIT = 4096       # padded tokens under which a batched pass is never split by length (see length_groups)
HOT_BUCKET = 3         # eager passes after which a busy server captures a bucket's graph anyway (capture_due)


def bucket(n, steps=8, floor=16):
    """n rounded up to one of `steps` steps per power of two, in steps of at least `floor`: the padding adds less than
    2/steps of n (just past a power of two, one step is almost that), or less than `floor` tokens for short n."""
    step = max(floor, (1 << (n - 1).bit_length()) // steps)
    return -(-n // step) * step


def pow2(n):
    return 1 << (n - 1).bit_length()


def count_bucket(n):
    """A row or state count rounded up to 1, 2, 3, 4, 6, 8, 12, 16, 24, 32: the empty entries padding a pass add less than
    half, and a traffic mix meets few distinct counts."""
    return bucket(n, steps=4, floor=1)


def length_groups(lengths, cap):
    """Indices grouped into padded passes of at most `cap` items, in order of length. A group is split only where padding
    to its longest item would more than double its tokens past PAD_SPLIT: a small pass costs about the same padded or not,
    while a batch padded to one long outlier wastes most of a large one."""
    groups, cur = [], []
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        padded = (len(cur) + 1) * bucket(lengths[i])
        if cur and (len(cur) == cap or (padded > PAD_SPLIT and padded > 2 * (sum(lengths[j] for j in cur) + lengths[i]))):
            groups.append(cur); cur = []
        cur.append(i)
    return groups + [cur]


class Request(NamedTuple):
    """One request for CudaGraphs.run."""
    state: list        # the state's token ids
    positions: list    # and their position ids
    rows: list         # its question rows, [(token ids, position ids)]
    cached: object     # the cached state (a DynamicCache) it continues, or None for a new state
    keep: bool         # return a new state's cache (to keep in the prefix cache)
    picks: list        # per row, the positions whose hidden states are wanted


class _Row(NamedTuple):
    """One question row of a batch: its tokens, its state's bank entry and length, and where its picks go in the output."""
    ids: list
    pos: list
    entry: int
    state_len: int
    picks: list
    at: int


class BufferKV(DynamicLayer):
    """An attention cache layer over preallocated [N, heads, T, dim] views: the first `filled` positions hold the cached
    state, a pass writes its keys/values after them instead of concatenating (no allocation, no Python state change,
    so one layer object serves the warm-up, the capture and every replay)."""

    def __init__(self, keys, values, filled):
        super().__init__()
        self.buffers, self.filled, self.is_initialized = (keys, values), filled, True
        self.keys, self.values = keys[..., :filled, :], values[..., :filled, :]

    def update(self, key_states, value_states, *args, **kwargs):
        end = self.filled + key_states.shape[-2]
        for buf, new in zip(self.buffers, (key_states, value_states)): buf[..., self.filled:end, :].copy_(new)
        return tuple(buf[..., :end, :] for buf in self.buffers)


def set_linear(layer, conv, recurrent, previous):
    """Point a DeltaNet cache layer at these conv/recurrent state tensors (transformers' LinearAttentionLayer fields)."""
    layer.conv_states[0], layer.recurrent_states[0] = conv, recurrent
    layer.is_conv_states_initialized[0] = layer.is_recurrent_states_initialized[0] = True
    layer.conv_kernel_size[0], layer.has_previous_state[0] = conv.shape[-1], previous
    layer.device, layer.dtype = conv.device, conv.dtype


def is_attention(layer):
    return isinstance(layer, DynamicLayer)


def layer_tensors(layer):
    """The two tensors a prefix-cache layer holds: attention keys/values, or DeltaNet conv/recurrent states."""
    return (layer.keys, layer.values) if is_attention(layer) else (layer.conv_states[0], layer.recurrent_states[0])


class Buffers:
    """Per layer two flat buffers and the per-entry shape they hold: attention [heads, T, dim] (T set per view, `tokens`
    entries x positions in total), DeltaNet states (`entries` entries)."""

    def __init__(self, probe, tokens, entries, device):
        self.slots = []
        for layer in probe.layers:
            ts, attention = layer_tensors(layer), is_attention(layer)
            size = [tokens * t.shape[1] * t.shape[3] if attention else entries * t[0].numel() for t in ts]
            self.slots.append((attention, [torch.zeros(n, dtype=t.dtype, device=device) for n, t in zip(size, ts)], [t.shape[1:] for t in ts]))

    def views(self, n, length):
        """Per layer, its two buffers viewed for n entries: attention [n, heads, length, dim], DeltaNet [n, *state]."""
        def view(flat, shape, attention):
            shape = (n, shape[0], length, shape[2]) if attention else (n, *shape)
            return flat[:torch.Size(shape).numel()].view(shape)
        return [tuple(view(f, s, attention) for f, s in zip(flats, shapes)) for attention, flats, shapes in self.slots]


class CudaGraphs:
    def __init__(self, lm, pad_id):
        self.lm, self.pad_id = lm, pad_id
        self.device, self.dtype = next(lm.parameters()).device, next(lm.parameters()).dtype
        self.pool, self.stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()   # the graphs' shared memory pool, the capture stream
        self.graphs = OrderedDict()   # key -> (graph, input buffer)
        self.pending = {}             # key -> (pass body, input buffer): buckets that ran eagerly, not captured yet
        self.eager_runs = {}          # key -> how many passes of a pending bucket ran eagerly
        self.failed = {}              # key -> (why its capture failed, input buffer); the bucket keeps running eagerly
        self.captures = 0
        probe = DynamicCache(config=lm.config)   # learn the cache layout (layer kinds, state shapes, dtypes) from one eager pass
        with torch.no_grad():
            lm(input_ids=torch.full((1, 16), pad_id, device=self.device), past_key_values=probe, use_cache=True)
        # BufferKV and set_linear rely on this layout: plain attention layers and single-state DeltaNet layers that update
        # their states in place. Anything else (another cache layer type, record_past, which assigns instead of copying)
        # would leave the buffers stale and move probabilities silently, so refuse it here.
        if not all(type(l) is DynamicLayer or (type(l) is LinearAttentionLayer and l.number_of_states == 1 and not l.record_past) for l in probe.layers):
            raise ValueError(f"CUDA graphs support attention and single-state DeltaNet cache layers only; got {sorted({type(l).__name__ for l in probe.layers})}")
        self.bank = Buffers(probe, GRAPH_STATES * BANK_WIDTH, GRAPH_STATES, self.device)
        # the bank has one layout for every pass, so the state and row graphs agree on it: attention [GRAPH_STATES, heads,
        # BANK_WIDTH, dim] with states right-aligned at the end, DeltaNet [GRAPH_STATES, *state]
        self.bank_views = self.bank.views(GRAPH_STATES, BANK_WIDTH)
        self.rowbuf = Buffers(probe, GRAPH_TOKENS, GRAPH_ROWS, self.device)
        self.hidden = torch.zeros(GRAPH_ROWS * GRAPH_ROW * lm.config.hidden_size, dtype=self.dtype, device=self.device)

    def _cache(self, views, filled, previous):
        """A DynamicCache over buffer views: attention layers hold `filled` cached positions, DeltaNet layers the states."""
        cache = DynamicCache(config=self.lm.config)
        for i, (layer, v) in enumerate(zip(cache.layers, views)):
            if is_attention(layer): cache.layers[i] = BufferKV(*v, filled)
            else: set_linear(layer, *v, previous)
        return cache

    def _mask(self, allow):
        """Additive attention mask [B, 1, Lq, Lk] from a boolean one [B, Lq, Lk], in the backbone's dtype."""
        return torch.zeros(allow.shape, dtype=self.dtype, device=self.device).masked_fill(~allow, torch.finfo(self.dtype).min)[:, None]

    def _forward(self, ids, pos, full_mask, linear_mask, cache):
        return self.lm(input_ids=ids, position_ids=pos, attention_mask={"full_attention": full_mask, "linear_attention": linear_mask},
                       past_key_values=cache, use_cache=True).last_hidden_state

    def _replay(self, key, body, rows):
        """One pass for `key` on `rows` (token ids, positions, lengths, indices; int64): the graph's replay, or body() run
        eagerly when the bucket has no graph yet (it then waits in `pending`). body(buf) runs the pass reading the uploaded
        rows from buf. Every body for a key reads and writes the same buffer views, so the one kept in `pending` stands for
        all of them."""
        if key in self.graphs:
            self.graphs.move_to_end(key)
            graph, buf = self.graphs[key]
        elif key in self.failed:
            graph, buf = None, self.failed[key][1]
        else:
            graph, buf = None, self.pending.setdefault(key, (body, torch.zeros((len(rows), len(rows[0])), dtype=torch.long, device=self.device)))[1]
            self.eager_runs[key] = self.eager_runs.get(key, 0) + 1
        buf.copy_(torch.tensor(rows, dtype=torch.long), non_blocking=True)
        if graph is not None: graph.replay()
        else: body(buf)

    @torch.no_grad()
    def capture_pending(self, limit=None):
        """Capture graphs for up to `limit` (None = all) buckets that have run eagerly, the most-run first. The caller must
        keep other passes out (kev.serve captures on its model thread). Each capture overwrites the shared buffers, which is
        fine between passes: a pass refills them. On a side stream, a warm-up pass first (Triton autotuning and cuBLAS
        setup must not happen inside a capture), then the capture; not torch.cuda.graph, whose synchronize, gc.collect and
        empty_cache on entry would stall a busy server for each capture. thread_local: CUDA calls of other threads (a
        request tokenizing, a model card read) cannot invalidate the capture. A capture that fails leaves its bucket running
        eagerly, for good."""
        for _ in range(len(self.pending) if limit is None else min(limit, len(self.pending))):
            key = max(self.pending, key=self.eager_runs.get)
            (body, buf), _ = self.pending.pop(key), self.eager_runs.pop(key)
            current, graph = torch.cuda.current_stream(), torch.cuda.CUDAGraph()
            self.stream.wait_stream(current)
            try:
                with torch.cuda.stream(self.stream):
                    body(buf)                                       # warm-up: autotuning and lazy setup happen outside the capture
                    graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                    try: body(buf)
                    finally: graph.capture_end()
            except Exception as e:   # e.g. out of memory for a new shape: serve it eagerly rather than stop serving
                # a failed capture_end leaves the allocator routing this thread's allocations into the graph pool, which
                # would fail every later capture and let eager passes allocate graph memory: end that explicitly
                try: getattr(torch._C, "_cuda_endAllocateToPool", lambda *a: None)(self.device.index, self.pool)   # private API: best effort
                except Exception: pass
                self.failed[key] = (f"{type(e).__name__}: {e}", buf)
                print(f"kev.cuda_graphs: capturing {key} failed, it runs eagerly: {self.failed[key][0]}", flush=True)
                continue
            finally:
                current.wait_stream(self.stream)
            self.graphs[key] = (graph, buf); self.captures += 1
            while len(self.graphs) > GRAPHS_KEPT: self.graphs.popitem(last=False)

    def capture_due(self, idle):
        """Capture one pending graph now? Always when idle; under load once a bucket keeps running eagerly (HOT_BUCKET
        eager passes), because a request arriving meanwhile waits for the capture (~0.4 s)."""
        runs = list(self.eager_runs.values())   # a snapshot: wait_idle and /v1/models call this off the model thread
        return bool(runs) and (idle or max(runs) >= HOT_BUCKET)

    def stats(self):
        return {"captured": self.captures, "kept": len(self.graphs), "pending": len(self.pending), "failed": len(self.failed)}

    # Serving: which requests the graphed passes take, and one batch of them

    @staticmethod
    def admits(state_len, row_lens, cached):
        """Whether a request's passes fit the graphed passes: its rows, and its state (a new one runs the graphed state
        pass; a cached one only has to fit the bank)."""
        return bucket(max(row_lens)) <= GRAPH_ROW and (state_len <= BANK_WIDTH if cached else bucket(state_len) <= GRAPH_STATE)

    @torch.no_grad()
    def run(self, requests):
        """Admitted requests (Request) -> (their picked hidden states, float32 [sum of picks, d] in request, row, pick
        order; per request its state's cache: the cached one, a new one if `keep`, else None). States of a similar length
        share a state pass (identical states once); each request takes one bank entry, so at most GRAPH_STATES per group.
        Every row pass writes its picks straight to their place in the output."""
        at = [0]
        for r in requests: at.append(at[-1] + sum(map(len, r.picks)))
        out = torch.empty((at[-1], self.lm.config.hidden_size), dtype=torch.float32, device=self.device)
        caches = [None] * len(requests)
        for group in length_groups([len(r.state) for r in requests], GRAPH_STATES):
            new = {}                                                 # distinct new states of the group -> (ids, positions, keep)
            for i in group:
                r = requests[i]
                if r.cached is None: new[tuple(r.state)] = (r.state, r.positions, r.keep or new.get(tuple(r.state), (0, 0, False))[2])
            made = dict(zip(new, self.states(list(new.values()), bucket(max(map(len, new)))))) if new else {}
            entry = {S: j for j, S in enumerate(new)}                # cached states go into the entries after the new ones
            rows, free = [], len(new)
            for i in group:                                          # after the state pass: its padded entries are written
                r = requests[i]
                if r.cached is None: e, caches[i] = entry[tuple(r.state)], made[tuple(r.state)]
                else: e, free, caches[i] = free, free + 1, r.cached; self.load_state(e, r.cached, len(r.state))
                offset = at[i]
                for (ids, pos), picks in zip(r.rows, r.picks):
                    rows.append(_Row(ids, pos, e, len(r.state), picks, offset)); offset += len(picks)
            self.rows(rows, out)
        return out, caches

    @torch.no_grad()
    def states(self, items, Sb):
        """State pass for up to GRAPH_STATES states, items = [(ids, pos, keep)] each at most Sb <= GRAPH_STATE tokens.
        Leaves state i in bank entry i and returns, for each kept state, a DynamicCache equal to the eager prefix pass's
        (None for the others: copying every state of a batch cost more than the passes)."""
        Nb = count_bucket(len(items))

        def body(buf):   # buf rows = [ids | positions | real length], left-padded
            pad = Sb - buf[:, 2 * Sb:]
            i = torch.arange(Sb, device=self.device)
            q, k = i[None, :, None], i[None, None, :]
            allow = ((k <= q) & (k >= pad[:, :, None])) | (k == q)
            views = [(a[:Nb, :, BANK_WIDTH - Sb:], b[:Nb, :, BANK_WIDTH - Sb:]) if attention else (a[:Nb], b[:Nb])
                     for (attention, *_), (a, b) in zip(self.bank.slots, self.bank_views)]
            self._forward(buf[:, :Sb], buf[:, Sb:2 * Sb], self._mask(allow), (i[None] >= pad).long(), self._cache(views, 0, False))

        rows = [[self.pad_id] * (Sb - len(ids)) + list(ids) + [0] * (Sb - len(ids)) + list(pos) + [len(ids)] for ids, pos, _ in items]
        self._replay(("states", Nb, Sb), body, rows + [[self.pad_id] * Sb + [0] * Sb + [0]] * (Nb - len(items)))
        out = []   # a cache for each kept state: its own copy of the bank entry (the bank belongs to the next pass)
        for j, (ids, _, keep) in enumerate(items):
            if not keep: out.append(None); continue
            cache, S = DynamicCache(config=self.lm.config), len(ids)
            for layer, (a, b) in zip(cache.layers, self.bank_views):
                if is_attention(layer):
                    layer.keys, layer.values = a[j:j + 1, :, BANK_WIDTH - S:].clone(), b[j:j + 1, :, BANK_WIDTH - S:].clone()
                    layer.dtype, layer.device, layer.is_initialized = self.dtype, self.device, True
                else:
                    set_linear(layer, a[j:j + 1].clone(), b[j:j + 1].clone(), True)
            out.append(cache)
        return out

    @torch.no_grad()
    def load_state(self, entry, cache, S):
        """Copy a cached state (a DynamicCache of S <= BANK_WIDTH tokens, from either path) into bank entry `entry`."""
        for layer, v in zip(cache.layers, self.bank_views):
            for buf, t in zip(v, layer_tensors(layer)):
                if is_attention(layer): buf[entry, :, BANK_WIDTH - S:].copy_(t[0, :, t.shape[-2] - S:])
                else: buf[entry].copy_(t[0])

    def rows(self, rows, out):
        """Row passes for question rows (_Row), each continuing its state in the bank; each row's picked hidden states land
        in out[row.at:]. The rows see the last Sr positions of the bank, Sr the longest state rounded up to a power of two
        (it only lengthens attention). Rows of a similar length share a pass; as few passes as the buffers allow."""
        for idx in length_groups([len(r.ids) for r in rows], GRAPH_ROWS):
            Lb, Sr = bucket(max(len(rows[i].ids) for i in idx)), pow2(max(16, max(rows[i].state_len for i in idx)))
            group = min(GRAPH_ROWS, GRAPH_TOKENS // (Sr + Lb))
            group = max(n for n in range(1, group + 1) if count_bucket(n) <= group)   # the most rows whose padded count fits
            for start in range(0, len(idx), group):
                part = [rows[i] for i in idx[start:start + group]]
                h = self._row_pass(part, Sr, Lb)
                src, dst = zip(*[(n * Lb + p, r.at + j) for n, r in enumerate(part) for j, p in enumerate(r.picks)])
                src, dst = torch.tensor([src, dst]).to(self.device, non_blocking=True)
                out.index_copy_(0, dst, h.flatten(0, 1).index_select(0, src).float())   # only the wanted positions leave the pass buffer

    def _row_pass(self, rows, Sr, Lb):
        """One row pass; -> the pass buffer [rows, Lb, d] (valid until the next pass)."""
        Nb, T = count_bucket(len(rows)), Sr + Lb
        hidden = self.hidden[:Nb * Lb * self.lm.config.hidden_size].view(Nb, Lb, -1)

        def body(buf):   # buf rows = [ids | positions | row length | state length | bank entry]
            rowlen, plen, entry = buf[:, 2 * Lb:2 * Lb + 1], buf[:, 2 * Lb + 1:2 * Lb + 2], buf[:, 2 * Lb + 2]
            views = self.rowbuf.views(Nb, T)
            for (attention, *_), dst, src in zip(self.bank.slots, views, self.bank_views):   # each row's state, from the bank
                for d, s in zip(dst, src):
                    if attention: d[:, :, :Sr].copy_(s[:, :, BANK_WIDTH - Sr:].index_select(0, entry))
                    else: d.copy_(s.index_select(0, entry))
            q = torch.arange(Lb, device=self.device)[None, :, None]
            k = torch.arange(T, device=self.device)[None, None, :]
            allow = ((k < Sr) & (k >= Sr - plen[:, :, None])) | ((k >= Sr) & (k - Sr <= q) & (k - Sr < rowlen[:, :, None])) | (k - Sr == q)
            linear = (torch.arange(Lb, device=self.device)[None] < rowlen).long()
            hidden.copy_(self._forward(buf[:, :Lb], buf[:, Lb:2 * Lb], self._mask(allow), linear, self._cache(views, Sr, True)))

        uploads = [list(r.ids) + [self.pad_id] * (Lb - len(r.ids)) + list(r.pos) + [0] * (Lb - len(r.pos)) + [len(r.ids), r.state_len, r.entry] for r in rows]
        self._replay(("rows", Nb, Lb, Sr), body, uploads + [[self.pad_id] * Lb + [0] * Lb + [0, 0, 0]] * (Nb - len(rows)))
        return hidden[:len(rows)]
