"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth
when KEV_API_KEY is set (unset = open server, the local default). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison). KEV_PREFIX_CACHE / KEV_PREFIX_MIN_TOKENS / KEV_PREFIX_MAX_TOKENS size the state-prefix cache; KEV_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts). A state over kev.model.SERVE_MAX_STATE tokens is refused with a 422 (kev.model.admit);
KEV_TRUNCATE_STATES=1 reads its first SERVE_MAX_STATE tokens instead, and then every response says whether it did. Backend
and precision follow LoadOptions (KEV_BACKEND, KEV_DTYPE, ...): on Apple Silicon a hybrid checkpoint runs on MLX
by default, without a torch MPS probe, elsewhere on torch in bf16.
"""
import argparse, asyncio, atexit, hmac, os, queue, random, sys, threading, time, traceback, uuid
from concurrent.futures import Future
import torch
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, fused_available, is_hub_id, mlx_available
from .device import DEVICE_HELP, DEVICES, empty_cache, out_of_memory, select, sync
from .model import SERVE_MAX_STATE, ContextOverflow, admit

PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + DeltaNet states; attention-only backbones also the state's hidden states); 0 disables
PREFIX_MIN_TOKENS = os.environ.get("KEV_PREFIX_MIN_TOKENS")               # states shorter than this are not cached; default = the model's prefix_min_tokens (0 for hybrid backbones and MLX, 384 for attention-only torch models)
PREFIX_MAX_TOKENS = int(os.environ.get("KEV_PREFIX_MAX_TOKENS", "65536"))  # state tokens the cache holds in all (least recently used evicted first); a longer state is not cached.
                                                                         # One 64k state (Kev-27B: ~1.3 GB of keys, values and DeltaNet states), or four 16k ones, not four 64k ones
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
TRUNCATE_STATES = os.environ.get("KEV_TRUNCATE_STATES", "0") == "1"    # unset = a state over SERVE_MAX_STATE tokens gets a 422; 1 = read its first SERVE_MAX_STATE tokens, and every response says whether it did (truncated, usage.state_tokens / state_tokens_used)
API_KEY = os.environ.get("KEV_API_KEY")                                  # unset = open server; set = require Authorization: Bearer <key>, as the TypeSafe clients always send
MAX_BATCH = 64                                                           # requests the model thread takes at once (kev.cuda_graphs splits them to fit its buffers)
MODEL_NAMES = ("kev-latest", "jev-latest")                               # both names serve this checkpoint; jev-latest is the TypeSafe SDK default model, so an unconfigured client works


@dataclass
class PrefixCache:
    """State prefixes kept across requests, least recently used first: (state token ids, option_isolation) -> prefix.
    At most `size` states and `max_tokens` state tokens in all; states shorter than min_tokens or longer than max_tokens
    are not cached. A batch keeps (copies) only the new states that will still be here after it, its last distinct ones
    within both bounds: the rest would be evicted by the batch itself."""
    size: int
    min_tokens: int
    max_tokens: int = PREFIX_MAX_TOKENS
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    oom_retries: int = 0   # batches that ran out of device memory with states cached, dropped them and ran again (Server._run)

    def plan(self, encs):
        """-> (key per request, None when its state is not cached; its cached prefix or None; whether to keep a new one)."""
        lengths = [enc["seg"].count(0) for enc in encs]
        keys = [(tuple(enc["ids"][:n]), bool(enc.get("option_isolation"))) if self.size and self.min_tokens <= n <= self.max_tokens else None
                for enc, n in zip(encs, lengths)]
        survivors, tokens = set(), 0
        for key in dict.fromkeys(k for k in reversed(keys) if k is not None):   # most recent first, as store() keeps them
            if len(survivors) == self.size or tokens + len(key[0]) > self.max_tokens: break
            survivors.add(key); tokens += len(key[0])
        return keys, [self.entries.get(k) if k is not None else None for k in keys], [k in survivors for k in keys]

    def over(self, keys):
        """Whether these keys (states) exceed either bound: more than `size` states or `max_tokens` state tokens."""
        return len(keys) > self.size or sum(len(k[0]) for k in keys) > self.max_tokens

    def make_room(self, keys, cached, keep):
        """Drop now, before the batch runs, the entries this batch's store() will evict anyway, so an old state does not
        stay resident through the passes of the new one that replaces it. store() (re)inserts every hit and every kept new
        state as most recent and evicts the least recently used while over a bound; evicting the same keys first leaves
        the same cache (tests/test_unit.py). A hit dropped here is still held by `cached` for this batch. Kev-4B on MLX:
        a 64k state's cache is 2.2 GB, so a new 64k state after another one peaked with both (runs/mlx-long-states)."""
        order = dict.fromkeys(self.entries)                     # least recently used first, as store() evicts
        for key, old, k in zip(keys, cached, keep):
            if key is not None and (old is not None or k):
                order.pop(key, None); order[key] = None
        while self.over(order):
            key = next(iter(order)); del order[key]; self.entries.pop(key, None)

    def store(self, keys, cached, prefixes):
        """Record hits and misses, and (re)insert the batch's prefixes in order: most recently used last."""
        for key, old, new in zip(keys, cached, prefixes):
            if key is None: continue
            self.hits += old is not None; self.misses += old is None
            if new is None: continue
            self.entries.pop(key, None); self.entries[key] = new
            while self.over(self.entries): self.entries.pop(next(iter(self.entries)))

    def clear(self):
        self.entries.clear()


@dataclass
class Server:
    """The loaded checkpoint, the state-prefix cache (PrefixCache), and the one model thread that runs every forward pass.

    Request threads encode their record and queue it; the model thread takes everything queued when it becomes free and
    runs it as one batch (model.probs_batch: with CUDA graphs, shared state and row passes; otherwise one request at a
    time), then answers each request. It also captures pending CUDA graphs when the graphs say so (capture_due). `lock` is
    held around each batch and capture: hold it to use the model directly."""
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    batches: int = 0
    batched_requests: int = 0
    release_date: str = field(default="")   # for the TypeSafe model card; resolved once (may ask the Hub)
    truncate_states: bool = field(default_factory=lambda: TRUNCATE_STATES)   # KEV_TRUNCATE_STATES; off: an over-length state is refused (kev.model.admit)

    def __post_init__(self):
        self.release_date = self.release_date or self.checkpoint.release_date()
        self.prefix_cache = PrefixCache(PREFIX_CACHE_SIZE, int(PREFIX_MIN_TOKENS) if PREFIX_MIN_TOKENS else self.model.prefix_min_tokens)
        self.queue, self.stopping = queue.Queue(), threading.Event()
        # the model thread gives up the GIL at every CUDA sync and waits to get it back while the event loop parses and
        # answers requests; at Python's default 5 ms switch interval those waits stretched a batch's model time ~2x.
        # Process-wide, so close() puts it back.
        self.switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        self.thread = threading.Thread(target=self._work, name="kev-model", daemon=True)
        self.thread.start()
        atexit.register(self.close)   # a daemon thread killed inside a CUDA call at interpreter exit aborts the process

    def close(self):
        """Stop the model thread after its current batch; requests still queued fail, and so do later ones (submit)."""
        if self.stopping.is_set(): return
        self.stopping.set(); self.thread.join()
        while not self.queue.empty():
            self.queue.get_nowait()[1].set_exception(RuntimeError("the server stopped")); self.queue.task_done()
        sys.setswitchinterval(self.switch_interval)

    def submit(self, rec):
        """Queue one record for the model thread. -> a Future of (probabilities, stats). The state prefix (tokens up to the
        first question) is cached across requests, so a repeated state only pays for its question rows. latency_ms is the
        model time of the batch the request ran in (not its wait in the queue). A record kev.model.admit refuses (a state
        over SERVE_MAX_STATE tokens unless truncate_states, a question row over SERVE_MAX_BRANCH) is a 422 here, before
        it reaches the model."""
        if self.stopping.is_set(): raise HTTPException(503, "the server is stopping")
        try: enc = admit(self.model, self.tok, rec, truncate=self.truncate_states)
        except ValueError as e:   # ContextOverflow, or a record the encoder cannot take
            state = isinstance(e, ContextOverflow) and e.max_state is not None
            raise HTTPException(422, str(e) + (f"; or start the server with KEV_TRUNCATE_STATES=1 to read only its first {e.max_state:,} tokens (responses then say truncated: true)" if state else ""))
        done = Future()
        self.queue.put((enc, done))
        return done

    def probs(self, rec):
        return self.submit(rec).result()

    def _work(self):
        graphs = getattr(self.model, "graphs", None)
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty:
                if graphs is not None and graphs.capture_due(idle=True):
                    with self.lock: graphs.capture_pending(limit=1)
                continue
            while len(batch) < MAX_BATCH:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            try:
                with self.lock: results = self._run([enc for enc, _ in batch])
            except Exception as e:   # every request of the batch gets the error; the thread lives on
                traceback.clear_frames(e.__traceback__)   # the batch's futures keep the exception, and its frames held the failed pass's tensors (142 MiB, a 3,578-token state on Kev-0.8B); the traceback keeps its lines
                results = [e] * len(batch)
            for (_, done), result in zip(batch, results):
                (done.set_exception if isinstance(result, Exception) else done.set_result)(result)
                self.queue.task_done()
            if graphs is not None and graphs.capture_due(idle=False):
                with self.lock: graphs.capture_pending(limit=1)

    def _run(self, encs):
        """One batch through model.probs_batch, with the prefix cache. -> per request (probs, stats). A pass out of device
        memory while states are cached drops the cache and runs once more: the cache only saves time, and kept resident
        it failed every later batch of that size (#75, @vtxyer)."""
        sync(self.device); t = time.time()
        for retry in (False, True):
            keys, cached, keep = self.prefix_cache.plan(encs)
            self.prefix_cache.make_room(keys, cached, keep)
            try: ps, prefixes = self.model.probs_batch(encs, cached, keep); break
            except Exception as e:   # states held: the cache's, and this batch's hits make_room already dropped from it
                held = self.prefix_cache.entries or any(c is not None for c in cached)
                if retry or not held or not out_of_memory(e): raise
            cached = None; self.prefix_cache.clear(); self.prefix_cache.oom_retries += 1   # after the except: its traceback holds the failed pass's tensors
            empty_cache(self.device)
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.prefix_cache.store(keys, cached, prefixes)
        self.batches += 1; self.batched_requests += len(encs)
        return [([q.tolist() for q in p], {"tokens": len(enc["ids"]), "state_tokens": enc["state_tokens"], "state_tokens_used": enc["seg"].count(0),
                                           "latency_ms": dt, "prefix_cache_hit": c is not None})   # state_tokens: the request's state; _used: what the model read (less only when truncated)
                for enc, p, c in zip(encs, ps, cached)]

    def wait_idle(self):
        """Block until every submitted request is answered and no CUDA graph waits to be captured (benchmarks, warm-up)."""
        self.queue.join()
        graphs = getattr(self.model, "graphs", None)
        while graphs is not None and graphs.capture_due(idle=True): time.sleep(0.01)
        with self.lock: pass                                   # a capture in progress finishes

    def answer(self, req):
        """The /v1/systemone response body for one request."""
        rec, meta = to_record(prepare(req))
        return self._body(req, meta, *self.probs(rec))

    async def answer_async(self, req):
        """answer() for the event loop: a request waiting on the model thread holds no worker thread, so a container takes
        as many concurrent requests as its batches can absorb (FastAPI runs sync endpoints on a 40-thread pool)."""
        rec, meta = to_record(prepare(req))
        return self._body(req, meta, *await asyncio.wrap_future(self.submit(rec)))

    def _body(self, req, meta, ps, m):
        """The TypeSafe body. A server that may truncate (truncate_states) also says, on every response, whether it did:
        `truncated`, and usage.state_tokens (the request's state) / state_tokens_used (what the model read), both counting
        the <state> token. The TypeSafe SDKs ignore fields they do not model, so clients keep parsing."""
        answers = to_answers(ps, meta)
        body = {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)}, "latency_ms": m["latency_ms"]}
        if self.truncate_states:
            body["usage"].update(state_tokens=m["state_tokens"], state_tokens_used=m["state_tokens_used"])
            body["truncated"] = m["state_tokens"] > m["state_tokens_used"]
        return body


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    return req.model_copy(update={"state": with_date_facts(req.state)}) if DATE_FACTS else req


app = FastAPI(title="kev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["x-typesafe-request-id", "server-timing"])


@app.middleware("http")
async def typesafe(request, call_next):
    """Bearer auth (when API_KEY is set) and the request id every TypeSafe client reads off the response."""
    started = time.perf_counter()
    if API_KEY and request.url.path.startswith("/v1") and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"}, 401, {"www-authenticate": "Bearer"})
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"   # time inside this process, for telling it from the network
    return resp


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    return await server().answer_async(req)


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)   # each order is a forward pass; 0 divided by nothing, unbounded counts ran forever (#30)
    seed: int = 0


@app.post("/v1/systemone/permute")
def systemone_permute(r: PermuteSystemOne):
    """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
    q = r.request.questions.get(r.question)
    if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
    rng = random.Random(r.seed); keys = list(q.criteria); runs = []
    for i in range(r.n_perm):
        order = list(keys)
        if i > 0: rng.shuffle(order)
        one = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        resp = server().answer(one); a = resp["answers"][r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": resp["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return truncation_marks({"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}, resp)


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    parts = [server().answer(req.model_copy(update={"questions": {qid: q}})) for qid, q in req.questions.items()]
    answers = {qid: a for p in parts for qid, a in p["answers"].items()}
    body = {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts), "output_tokens": output_tokens(server().tok, answers)},
            "latency_ms": round(sum(p["latency_ms"] for p in parts), 1)}
    return truncation_marks(body, parts[0])


def truncation_marks(body, part):
    """A demo endpoint built from several /v1/systemone answers on one state carries a truncating server's marks
    (Server._body: `truncated`, usage.state_tokens / state_tokens_used) from any one of them: each read the same tokens."""
    if "truncated" in part:
        body["truncated"] = part["truncated"]
        body.setdefault("usage", {}).update({k: part["usage"][k] for k in ("state_tokens", "state_tokens_used")})
    return body


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus the Kev serving details
    a client may ignore: the run, the base, the device, the backend and precision, the temperature, prefix-cache stats."""
    s = server()
    ck, meta = s.checkpoint, s.checkpoint.meta
    card = {"description": f"Kev pointer head on {meta.base}, serving {ck.requested} at temperature {s.model.head.temperature:.2f}",
            "release_date": s.release_date,
            "run": ck.requested, "base": meta.base, "lora": meta.lora, "device": s.device, "backend": s.model.backend, "dtype": s.model.dtype,
            "temperature": s.model.head.temperature, "max_state_tokens": SERVE_MAX_STATE, "truncate_states": s.truncate_states,
            "cuda_graphs": graphs.stats() if (graphs := getattr(s.model, "graphs", None)) else None,
            "prefix_cache": {"size": s.prefix_cache.size, "min_state_tokens": s.prefix_cache.min_tokens, "max_tokens": s.prefix_cache.max_tokens, "hits": s.prefix_cache.hits,
                             "misses": s.prefix_cache.misses, "cached_states": len(s.prefix_cache.entries), "oom_retries": s.prefix_cache.oom_retries},
            "batches": {"count": s.batches, "requests": s.batched_requests, "queued": s.queue.qsize()}}
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def _serving_device(requested, opts, run):
    """Device and checkpoint for this process. A hybrid checkpoint on Apple Silicon with the MLX backend does not consult
    the torch MPS probe. Everywhere else the probe runs before the checkpoint is opened, so a driver crash stays in the child."""
    probe_dtype = opts.dtype or torch.bfloat16
    mlx_candidate = (sys.platform == "darwin" and requested in ("auto", "mps") and opts.backend in ("auto", "mlx")
                     and opts.dtype is not torch.float32)
    if mlx_candidate:
        ck = Checkpoint(run)
        if mlx_available() and ck.hybrid_base():
            return "mps", ck
        return select(requested, dtype=probe_dtype), ck
    return select(requested, dtype=probe_dtype), Checkpoint(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve beyond this machine (a container, a VM behind a proxy)")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", choices=("auto", *DEVICES), default="auto", help=DEVICE_HELP)
    a = ap.parse_args()
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    opts = LoadOptions.from_env()
    if opts.backend is None: opts = replace(opts, backend="auto")   # serving default: MLX for the hybrid Qwen3.5 checkpoints on Apple Silicon (LoadOptions.backend); KEV_BACKEND=torch to decline
    dev, ck = _serving_device(a.device, opts, run)
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default: 2-4.5x faster than fp32 on an L4, same answers (LoadOptions.dtype); KEV_DTYPE=fp32 for the exact path
    cuda_optimizations = dev == "cuda" and torch.version.hip is None   # these serving paths are measured on NVIDIA
    if cuda_optimizations and opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)   # serving default: a pass is ~2,000 kernel launches, so replaying graphs cuts warm latency several-fold (kev.cuda_graphs); KEV_CUDA_GRAPHS=0 to decline
    fused_default = cuda_optimizations and opts.fused is None
    if fused_default: opts = replace(opts, fused=fused_available())   # serving default: fused Qwen3.5 kernels, ~1/3 less GPU time per batch (kev.fused_qwen35), when fla is installed; KEV_FUSED=0 to decline, KEV_FUSED=1 to insist
    tok, model = ck.load(dev, opts)
    if fused_default and not opts.fused and model.hybrid: print("fused Qwen3.5 kernels off: install the flash-linear-attention version kev/fused_qwen35.py pins (FLA_VERSION) to turn them on")
    app.state.server = Server(ck, tok, model, dev)
    print(f"serving {ck.requested} ({ck.path}) on {dev} via {model.backend} ({model.dtype}) {a.host}:{a.port}; states over {SERVE_MAX_STATE:,} tokens "
          f"{'truncated (KEV_TRUNCATE_STATES=1)' if app.state.server.truncate_states else 'refused (422)'}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
