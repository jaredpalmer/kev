"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth
when KEV_API_KEY is set (unset = open server, the local default). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison). KEV_PREFIX_CACHE / KEV_PREFIX_MIN_TOKENS size the state-prefix cache; KEV_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts). Backend and precision follow LoadOptions (KEV_BACKEND, KEV_DTYPE, ...): on Apple
Silicon the hybrid Qwen3.5 checkpoints run on MLX by default, elsewhere on torch in bf16.
"""
import argparse, hmac, os, random, threading, time, uuid
import torch
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, is_hub_id
from .device import default_device, empty_cache, sync
from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PREFIX_MIN_TOKENS = os.environ.get("KEV_PREFIX_MIN_TOKENS")               # states shorter than this are not cached; default = the model's prefix_min_tokens (0 for hybrid backbones and MLX, 384 for attention-only torch models)
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"
API_KEY = os.environ.get("KEV_API_KEY")                                  # unset = open server; set = require Authorization: Bearer <key>, as the TypeSafe clients always send
MODEL_NAMES = ("kev-latest", "jev-latest")                               # both names serve this checkpoint; jev-latest is the TypeSafe SDK default model, so an unconfigured client works


@dataclass
class Server:
    """The loaded checkpoint and the state-prefix cache shared by every request (one model, one lock)."""
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    prefix_cache: dict = field(default_factory=dict)   # (state token ids, option_isolation) -> prefix, in LRU order
    prefix_hits: int = 0
    prefix_misses: int = 0
    oom_retries: int = 0
    release_date: str = field(default="")   # for the TypeSafe model card; resolved once (may ask the Hub)

    def __post_init__(self):
        self.release_date = self.release_date or self.checkpoint.release_date()

    @property
    def prefix_min_tokens(self):
        return int(PREFIX_MIN_TOKENS) if PREFIX_MIN_TOKENS else self.model.prefix_min_tokens

    def probs(self, rec):
        """One forward pass. The state prefix (tokens up to the first question) is cached across requests, so a repeated
        state only pays for its question branches. Exact: the state's activations do not depend on the branches."""
        try: enc = self.model.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        except ValueError as e: raise HTTPException(422, str(e))
        Ls = enc["seg"].count(0); key = (tuple(enc["ids"][:Ls]), bool(enc.get("option_isolation")))
        with self.lock:
            sync(self.device); t = time.time()
            try:
                ps, hit = self._pass(enc, key, Ls)
            except torch.OutOfMemoryError:
                # The cache is holding a share of this process's budget that nothing else can reach, and the pass that
                # just failed for want of memory is exactly the one that cannot free it. Drop all of it and take one
                # more pass: without this the process stays wedged, because every later request meets the same
                # resident prefix and fails the same way.
                self.prefix_cache.clear()
                empty_cache(self.device)
                self.oom_retries += 1
                ps, hit = self._pass(enc, key, Ls)
            sync(self.device); dt = time.time() - t
        return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": hit}

    def _pass(self, enc, key, state_tokens):
        """One forward pass and whether the state prefix was already cached. The caller holds the lock."""
        cache = self.prefix_cache
        eligible = PREFIX_CACHE_SIZE and state_tokens >= self.prefix_min_tokens
        if eligible and key in cache:
            prefix = cache.pop(key)                            # pop + reinsert = LRU order
            ps = self.model.probs_with_prefix(enc, prefix); cache[key] = prefix
            self.prefix_hits += 1
            return ps, True
        if eligible:
            # Free what this prefix displaces before the pass that builds it, not after. A prefix for a long state is
            # gigabytes of KV, so holding the outgoing one across the incoming pass asks the card for both at once --
            # and on a card with a per-process ceiling that is the request that fails, leaving the outgoing prefix
            # cached because this eviction never runs.
            while len(cache) >= PREFIX_CACHE_SIZE: cache.pop(next(iter(cache)))
            ps, prefix = self.model.probs_and_prefix(enc)      # one pass, and the state prefix is kept for next time
            cache[key] = prefix
            self.prefix_misses += 1
            return ps, False
        return self.model.probs(enc), False

    def answer(self, req):
        """The /v1/systemone response body for one request."""
        rec, meta = to_record(prepare(req))
        ps, m = self.probs(rec)
        answers = to_answers(ps, meta)
        return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)}, "latency_ms": m["latency_ms"]}


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    return req.model_copy(update={"state": with_date_facts(req.state)}) if DATE_FACTS else req


app = FastAPI(title="kev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["x-typesafe-request-id"])


@app.middleware("http")
async def typesafe(request, call_next):
    """Bearer auth (when API_KEY is set) and the request id every TypeSafe client reads off the response."""
    if API_KEY and request.url.path.startswith("/v1") and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"}, 401, {"www-authenticate": "Bearer"})
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    return resp


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    return server().answer(req)


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
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    parts = [server().answer(req.model_copy(update={"questions": {qid: q}})) for qid, q in req.questions.items()]
    answers = {qid: a for p in parts for qid, a in p["answers"].items()}
    return {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts), "output_tokens": output_tokens(server().tok, answers)},
            "latency_ms": round(sum(p["latency_ms"] for p in parts), 1)}


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus the Kev serving details
    a client may ignore: the run, the base, the device, the backend and precision, the temperature, prefix-cache stats."""
    s = server()
    ck, meta = s.checkpoint, s.checkpoint.meta
    card = {"description": f"Kev pointer head on {meta.base}, serving {ck.requested} at temperature {s.model.head.temperature:.2f}",
            "release_date": s.release_date,
            "run": ck.requested, "base": meta.base, "lora": meta.lora, "device": s.device, "backend": s.model.backend, "dtype": s.model.dtype,
            "temperature": s.model.head.temperature,
            "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": s.prefix_min_tokens, "hits": s.prefix_hits,
                             "misses": s.prefix_misses, "cached_states": len(s.prefix_cache), "oom_retries": s.oom_retries}}
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    dev = default_device()
    opts = LoadOptions.from_env()
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default: 2-4.5x faster than fp32 on an L4, same answers (LoadOptions.dtype); KEV_DTYPE=fp32 for the exact path
    if opts.backend is None: opts = replace(opts, backend="auto")   # serving default: MLX for the hybrid Qwen3.5 checkpoints on Apple Silicon (LoadOptions.backend); KEV_BACKEND=torch to decline
    ck = Checkpoint(run)
    tok, model = ck.load(dev, opts)
    app.state.server = Server(ck, tok, model, dev)
    print(f"serving {ck.requested} ({ck.path}) on {dev} via {model.backend} ({model.dtype}) :{a.port}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
