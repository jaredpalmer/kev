"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models (no auth). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison). KEV_PREFIX_CACHE / KEV_PREFIX_MIN_TOKENS size the state-prefix KV cache; KEV_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts).
"""
import argparse, os, random, threading, time
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import Checkpoint, LoadOptions, is_hub_id
from .device import default_device, sync

# inference limits (training used 384/1024); per-branch cap mirrors Jev's ~32k, bounded by the base model window
INFER_MAX_STATE, INFER_MAX_BRANCH = 8192, 8192
INFER_MAX_PACKED = 8192  # bound total inference memory across all question branches
PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PREFIX_MIN_TOKENS = int(os.environ.get("KEV_PREFIX_MIN_TOKENS", "384"))   # below this the branch-only pass is not faster on MPS (per-op overhead dominates)
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"


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

    def probs(self, rec):
        """One forward pass. The state prefix (tokens up to the first question) is cached across requests, so a repeated
        state only pays for its question branches. Exact: the state's activations do not depend on the branches."""
        try: enc = self.model.encode(self.tok, rec, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
        except ValueError as e: raise HTTPException(422, str(e))
        if len(enc["ids"]) > INFER_MAX_PACKED:
            raise HTTPException(422, f"request exceeds {INFER_MAX_PACKED} packed tokens: {len(enc['ids'])}")
        Ls = enc["seg"].count(0); key = (tuple(enc["ids"][:Ls]), bool(enc.get("option_isolation")))
        cache, hit = self.prefix_cache, False
        with self.lock:
            sync(self.device); t = time.time()
            eligible = PREFIX_CACHE_SIZE and Ls >= PREFIX_MIN_TOKENS
            if eligible and key in cache:
                prefix = cache.pop(key)                            # pop + reinsert = LRU order
                ps = self.model.probs_with_prefix(enc, prefix); cache[key] = prefix
                self.prefix_hits += 1; hit = True
            elif eligible:
                ps, prefix = self.model.probs_and_prefix(enc)      # one pass, and the state prefix is kept for next time
                cache[key] = prefix
                while len(cache) > PREFIX_CACHE_SIZE: cache.pop(next(iter(cache)))
                self.prefix_misses += 1
            else:
                ps = self.model.probs(enc)
            sync(self.device); dt = time.time() - t
        return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": hit}

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def server() -> Server:
    return app.state.server


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    return server().answer(req)


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)
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
    s = server()
    return {"models": [{"id": "kev-latest", "aliases": ["jev-latest"], "run": s.checkpoint.requested, "base": s.checkpoint.meta.base,
                        "lora": s.checkpoint.meta.lora, "device": s.device, "temperature": s.model.head.temperature,
                        "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": PREFIX_MIN_TOKENS, "hits": s.prefix_hits,
                                         "misses": s.prefix_misses, "cached_states": len(s.prefix_cache)}}]}


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
    ck = Checkpoint(run)
    tok, model = ck.load(dev, opts)
    app.state.server = Server(ck, tok, model, dev)
    print(f"serving {ck.requested} ({ck.path}) on {dev} :{a.port}")   # /v1/models reports the run as given, not the resolved cache path
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
