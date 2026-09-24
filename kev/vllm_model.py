"""CUDA serving backend: vLLM runs the backbone, Kev keeps its encoder and pointer head.

The torch path answers one request at a time under kev.serve's lock, so concurrent clients queue behind each other and
the GPU runs one small prefill per step. Here every question row (state + branch, `rows_of`, the same row form the torch
path uses for the hybrid Qwen3.5 backbones) is one vLLM pooling request: the engine batches rows across concurrent
requests (continuous batching over a paged KV cache, fused Gated DeltaNet kernels, CUDA graphs) and returns each row's
final hidden states ("token_embed" with ALL pooling, no activation). The readout is the very same fp32 `PointerHead` (with
the checkpoint's temperature) the torch and MLX paths apply, on the `<decide>` and `</opt>` positions.

State sharing (on unless KEV_VLLM_SHARE_STATE=0): sent as is, every row recomputes the state in front of it, so a request
costs (questions x state) tokens. Instead the state runs once as its own request, cut to its last PREFIX_UNIT boundary.
vLLM's prefix cache in "align" mode, with a prefix_match_unit finer than the hybrid block (784 tokens on Qwen3.8-27B),
keeps that request's DeltaNet state and attention KV at exactly that boundary, and the rows, sent after it finishes, start
there. The engine returns hidden states only for the tokens it computed, so the readout offsets by each row's cache hit.

vLLM cannot load a PEFT adapter onto every projection Kev trains (the DeltaNet in_proj_* are packed differently), so the
adapter is folded into the base in fp32 by the torch loader (the exact merge LoadOptions.merge describes), rounded once to
the serving dtype and exported as a plain `*ForCausalLM` checkpoint. The export is cached under $HF_HOME/kev-vllm, keyed
by everything that changes its weights, so only the first cold start pays for it.

vLLM is not a kev dependency (it pins its own torch); it lives in the Modal serving image (modal_serve.py). Selected by
`LoadOptions(backend="vllm")` (KEV_BACKEND=vllm) in `kev.checkpoint`; parity against the fp32 torch path and the load test
against torch bf16 are `modal run modal_serve.py::parity` / `::loadtest`.
"""
import asyncio, hashlib, json, os, shutil, threading, uuid
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm import PoolingParams, TokensPrompt
from vllm.config import PoolerConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from .model import SERVE_MAX_PACKED, PointerHead, encode, rows_of

EXPORT_VERSION = 1   # bump when the export layout changes, so cached exports are rebuilt
GPU_MEMORY_UTILIZATION = float(os.environ.get("KEV_VLLM_GPU_MEMORY", "0.85"))   # share of the GPU vLLM may take (weights + KV / state cache)
MAX_NUM_SEQS = int(os.environ.get("KEV_VLLM_MAX_SEQS", "256"))                   # rows the engine schedules at once
SHARE_STATE = os.environ.get("KEV_VLLM_SHARE_STATE", "1") != "0"                 # rows start from the state in vLLM's prefix cache
PREFIX_UNIT = 16   # vLLM prefix_match_unit: cache hits land on multiples of it; it must divide the hybrid block, which vLLM rounds to a multiple of 16
# The state request is one more engine round trip before the rows can start (+30-45 ms on Kev-4B and Qwen3.8-27B). For a
# 34-token state it cost more than the tokens it saved; from a 370-token state on it won latency and throughput
# (runs/serve/vllm-*-pair). Lengths in between are unmeasured.
SHARE_MIN_STATE = 256


def export_dir(checkpoint, dtype, lora_scale):
    """Where the merged export of this checkpoint lives: a hash of the adapter, the base and every merge setting."""
    h = hashlib.sha256(checkpoint.file("adapter_model.safetensors").read_bytes())
    h.update(json.dumps([EXPORT_VERSION, checkpoint.meta.base, checkpoint.meta.base_revision, str(dtype), lora_scale]).encode())
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "kev-vllm"
    return root / h.hexdigest()[:16]


def export_merged(lm, tok, out, dtype):
    """Write a merged backbone (the text model DecisionModel.lm) as a checkpoint vLLM loads as `*ForCausalLM`. Kev never
    reads the vocab head, so the export ties it to the embeddings instead of storing one. Atomic: written beside `out`,
    then renamed, so a crashed export is never mistaken for a finished one."""
    tmp = out.with_name(out.name + f".tmp-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    cfg = lm.config
    cfg.architectures = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[cfg.model_type]]
    cfg.tie_word_embeddings = True
    cfg.dtype = dtype
    cfg.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    save_file({f"model.{k}": v.to(dtype).contiguous() for k, v in lm.state_dict().items()}, str(tmp / "model.safetensors"))
    try:
        tmp.rename(out)
    except OSError:          # another container finished the same export first
        shutil.rmtree(tmp)
    return out


class VLLMDecisionModel:
    """Prefill-only scorer: hidden states from a vLLM engine, logits from the shared torch PointerHead."""
    backend, device, hybrid, option_isolation = "vllm", "vllm", True, False
    concurrent = True        # kev.serve skips its lock and its own state-prefix cache: the engine batches rows and keeps states itself
    prefix_min_tokens = None

    def __init__(self, model_dir, dtype, head_dim=256, share_state=SHARE_STATE):
        cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        self._dtype = str(dtype).removeprefix("torch.")
        self.head = PointerHead(cfg["hidden_size"], dp=head_dim).eval()
        self.share_state = share_state
        self.tokens = Counter()   # prompt tokens, tokens the engine computed, engine requests, re-run rows (the load test reports them)
        # vLLM does not enable prefix caching or chunked prefill for hybrid pooling models by default; align mode needs both
        cache = (dict(enable_prefix_caching=True, mamba_cache_mode="align", prefix_match_unit=PREFIX_UNIT, enable_chunked_prefill=True)
                 if share_state else dict(enable_prefix_caching=False))
        args = AsyncEngineArgs(model=str(model_dir), runner="pooling", dtype=self._dtype, max_model_len=SERVE_MAX_PACKED,
                               pooler_config=PoolerConfig(seq_pooling_type="LAST", tok_pooling_type="ALL"),
                               gpu_memory_utilization=GPU_MEMORY_UTILIZATION, max_num_seqs=MAX_NUM_SEQS, **cache)
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="kev-vllm", daemon=True).start()
        self.engine = self._run(self._start(args))

    async def _start(self, args):
        return AsyncLLM.from_engine_args(args)   # inside the loop: its output handler is a task on this loop

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    @property
    def dtype(self):
        return self._dtype

    def eval(self):
        self.head.eval(); return self

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=False, **kw)

    async def _pool(self, ids, task, salt=None):
        """One engine request -> (final hidden states of the tokens it computed, how many leading tokens came from the cache)."""
        prompt = TokensPrompt(prompt_token_ids=ids, **({"cache_salt": salt} if salt else {}))
        params = PoolingParams(task=task, use_activation=False, skip_reading_prefix_cache=False)   # token_embed skips the cache by default
        async for out in self.engine.encode(prompt, params, uuid.uuid4().hex):
            last = out
        self.tokens.update(prompt=len(ids), computed=len(ids) - last.num_cached_tokens, requests=1)
        return last.outputs.data, last.num_cached_tokens

    async def _rows(self, S, rows, salt=None):
        """(hidden states, cache hit) of each causal row state + branch. With state sharing, more than one row and a state of
        at least SHARE_MIN_STATE tokens, the state runs first on its own ("embed": one pooled vector comes back, not [L, d]),
        so the rows find it in the cache."""
        shared = len(S) // PREFIX_UNIT * PREFIX_UNIT
        if self.share_state and len(rows) > 1 and shared >= SHARE_MIN_STATE:
            await self._pool(S[:shared], "embed", salt)
        # With prefix caching vLLM stops a prompt's prefill at its last unit boundary to cache it there, an extra engine step
        # per row for an entry Kev never reads. Padding the row past <decide> to the unit avoids the stop; a causal model's
        # hidden states at the readout do not see what follows.
        pad = (lambda ids: ids + ids[-1:] * (-len(ids) % PREFIX_UNIT)) if self.share_state else (lambda ids: ids)
        return list(await asyncio.gather(*(self._pool(pad(S + r["ids"]), "token_embed", salt) for r in rows)))

    async def _logits(self, enc):
        S, _, rows = rows_of(enc)
        outs = await self._rows(S, rows)
        # A hit that reaches a row's own readout (the same question on the same state, asked before) leaves nothing to read.
        # Those rows run again under a fresh cache salt: they still share this state among themselves, but no earlier entry.
        redo = [i for i, ((_, hit), r) in enumerate(zip(outs, rows)) if hit > len(S) + min(r["decide"], *r["opts"])]
        if redo:
            for i, o in zip(redo, await self._rows(S, [rows[i] for i in redo], salt=uuid.uuid4().hex)): outs[i] = o
            self.tokens.update(redone=len(redo))
        out = []
        with torch.no_grad():
            for (h, hit), r in zip(outs, rows):
                picked = h[torch.tensor([len(S) + r["decide"], *(len(S) + o for o in r["opts"])]) - hit].float().cpu()
                out.append(self.head(picked[0], picked[1:]))
        return out

    def forward(self, enc):
        """List of logits tensors, one per question. Thread-safe: callers on many threads share the engine."""
        return self._run(self._logits(enc))

    def close(self):
        """Stop the engine and free its GPU memory (modal_serve.py's bench runs two engines, one after the other)."""
        self.engine.shutdown()
        self.loop.call_soon_threadsafe(self.loop.stop)

    def probs(self, enc):
        return [F.softmax(z, -1) for z in self.forward(enc)]

    def probs_and_prefix(self, enc):
        return self.probs(enc), None

    def probs_with_prefix(self, enc, prefix):
        return self.probs(enc)
