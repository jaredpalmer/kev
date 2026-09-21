"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008
"""
import argparse, json, os, random, re, time
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .data import DISTRACTORS, NONE
from .encoding import rows_of
from .evaluate import load
from .inference import InferenceBusy, InferenceCancelled, InferenceTimeout, InferenceUnavailable, InferenceWorker

# inference limits (training used 384/640); per-branch cap mirrors Jev's ~32k, bounded by the base model window
INFER_MAX_STATE, INFER_MAX_BRANCH = 8192, 8192

@asynccontextmanager
async def _lifespan(_app):
    yield
    worker = STATE.get("worker")
    if worker is not None:
        worker.stop(timeout_s=5.0)


app = FastAPI(title="kev", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {"run": None, "tok": None, "model": None, "dev": None, "worker": None, "prefix_cache": {}, "prefix_hits": 0, "prefix_misses": 0, "prefix_coalesced": 0}
PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PREFIX_MIN_TOKENS = int(os.environ.get("KEV_PREFIX_MIN_TOKENS", "384"))
INFER_QUEUE_SIZE = int(os.environ.get("KEV_INFER_QUEUE", "64"))
INFER_TIMEOUT_S = float(os.environ.get("KEV_INFER_TIMEOUT_S", "120"))
INFER_BATCH_ROWS = int(os.environ.get("KEV_INFER_BATCH_ROWS", "8"))
INFER_BATCH_WAIT_MS = float(os.environ.get("KEV_INFER_BATCH_WAIT_MS", "1"))
INFER_BATCH_MAX_ROWS = int(os.environ.get("KEV_INFER_BATCH_MAX_ROWS", "32"))
INFER_BATCH_MAX_TOKENS = int(os.environ.get("KEV_INFER_BATCH_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("KEV_TEMPERATURE", "1.0"))               # opt-in: probabilities ^ (1/T), renormalised; 2.0 is the value fitted in-distribution for the Qwen3.5 family (scripts/temperature_groups.py)
DATE_FACTS = os.environ.get("KEV_DATE_FACTS", "0") == "1"                  # opt-in: append day counts between absolute dates in the state (api.with_date_facts)   # below this the branch-only pass is not faster on MPS (per-op overhead dominates)


class Question(BaseModel):
    instr: str
    options: list[str]


class Record(BaseModel):
    state: str
    questions: list[Question]


class PermuteReq(BaseModel):
    state: str
    question: Question
    n_perm: int = 6
    seed: int = 0


def _rec(r: Record):
    return {"state": r.state, "questions": [{"instr": q.instr, "options": q.options, "label": 0} for q in r.questions]}


def _sync(dev):
    if dev == "mps": torch.mps.synchronize()
    elif dev == "cuda": torch.cuda.synchronize()


def _encode(rec):
    tok, model = STATE["tok"], STATE["model"]
    try:
        return model.encode(tok, rec, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH, strict=True)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _batch_key(rec):
    # Exact text is a conservative key: equal text always produces the same
    # encoded state, while different text is never merged accidentally.
    return (rec.get("state"), bool(rec.get("option_isolation")))


def _probs_core_many(records, encs=None):
    """One forward pass; the state prefix (tokens up to the first question) is cached across requests, so a repeated state
    only pays for its question branches. Exactness: the state's activations do not depend on the branches."""
    model, dev = STATE["model"], STATE["dev"]
    encs = [_encode(rec) for rec in records] if encs is None else encs
    Ls = encs[0]["seg"].count(0); key = (tuple(encs[0]["ids"][:Ls]), bool(encs[0].get("option_isolation")))
    row_sets = [rows_of(enc)[2] for enc in encs]
    batch_rows = sum(len(rows) for rows in row_sets)
    batch_tokens = sum(len(rows) * (Ls + max((len(row["ids"]) for row in rows), default=0)) for rows in row_sets)
    if batch_rows > INFER_BATCH_MAX_ROWS or batch_tokens > INFER_BATCH_MAX_TOKENS:
        raise HTTPException(413, "request or batch exceeds inference memory limits")
    cache = STATE["prefix_cache"]
    _sync(dev); t = time.monotonic()
    # MLX's recurrent prefix pass is substantially cheaper than repeating
    # the state once per question, even for short states. The old MPS
    # threshold remains for the PyTorch fallback where cache setup can
    # dominate short requests.
    eligible = PREFIX_CACHE_SIZE and (Ls >= PREFIX_MIN_TOKENS or dev == "mlx")
    if eligible and key in cache:
        prefix = cache.pop(key)                       # pop + reinsert = LRU order
        if len(encs) > 1 and hasattr(model, "probs_with_prefix_batch"):
            pss = model.probs_with_prefix_batch(encs, prefix)
        else:
            pss = [model.probs_with_prefix(enc, prefix) for enc in encs]
        cache[key] = prefix
        STATE["prefix_hits"] += len(encs); hits = [True] * len(encs)
    elif eligible:
        if len(encs) > 1 and hasattr(model, "probs_and_prefix_batch"):
            pss, prefix = model.probs_and_prefix_batch(encs)
        else:
            prefix = None
            pss = []
            for enc in encs:
                ps, prefix = model.probs_and_prefix(enc)
                pss.append(ps)
        cache[key] = prefix
        while len(cache) > PREFIX_CACHE_SIZE: cache.pop(next(iter(cache)))
        STATE["prefix_misses"] += 1
        STATE["prefix_coalesced"] = STATE.get("prefix_coalesced", 0) + max(0, len(encs) - 1)
        hits = [False] * len(encs)
    else:
        pss = [model.probs(enc) for enc in encs]; hits = [False] * len(encs)
    _sync(dev); dt = time.monotonic() - t
    out = []
    for enc, ps, hit in zip(encs, pss, hits):
        if TEMPERATURE != 1.0:                  # opt-in calibration: pointer logit scaling (argmax unchanged)
            ps = [(lambda q: q / q.sum())(torch.as_tensor(p).clamp_min(1e-9) ** (1.0 / TEMPERATURE)) for p in ps]
        out.append(([p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": hit, "batch_requests": len(encs), "batch_rows": batch_rows, "batch_tokens": batch_tokens}))
    return out


def _probs_core(rec):
    return _probs_core_many([rec])[0]


def _batch_jobs(jobs):
    from .inference import BatchItem

    encs, valid = [], []
    outcomes = [None] * len(jobs)
    for i, job in enumerate(jobs):
        try:
            encs.append(_encode(job.payload))
            valid.append(i)
        except Exception as exc:
            outcomes[i] = BatchItem(error=exc, fatal=isinstance(exc, (MemoryError, SystemError, RuntimeError)))
    if not valid:
        return outcomes
    chunks, chunk = [], []
    rows_used = tokens_used = 0
    for i, enc in zip(valid, encs):
        rows = rows_of(enc)[2]
        row_count = len(rows)
        state_len = enc["seg"].count(0)
        token_count = row_count * (state_len + max((len(row["ids"]) for row in rows), default=0))
        if row_count > INFER_BATCH_MAX_ROWS or token_count > INFER_BATCH_MAX_TOKENS:
            outcomes[i] = BatchItem(error=HTTPException(413, "request exceeds inference batch memory limits"))
            continue
        if chunk and (rows_used + row_count > INFER_BATCH_MAX_ROWS or tokens_used + token_count > INFER_BATCH_MAX_TOKENS):
            chunks.append(chunk)
            chunk, rows_used, tokens_used = [], 0, 0
        chunk.append((i, enc))
        rows_used += row_count
        tokens_used += token_count
    if chunk:
        chunks.append(chunk)
    for chunk_index, chunk in enumerate(chunks):
        indices = [i for i, _ in chunk]
        chunk_encs = [enc for _, enc in chunk]
        try:
            values = _probs_core_many([jobs[i].payload for i in indices], chunk_encs)
        except Exception as exc:
            fatal = isinstance(exc, (MemoryError, SystemError, RuntimeError))
            for i in indices:
                outcomes[i] = BatchItem(error=exc, fatal=fatal)
            if fatal:
                for remaining in chunks[chunk_index + 1:]:
                    for i, _ in remaining:
                        outcomes[i] = BatchItem(error=exc, fatal=True)
                break
            continue
        for i, value in zip(indices, values):
            outcomes[i] = BatchItem(value=value)
    return outcomes


async def _probs_async(rec, request: Request):
    worker = STATE.get("worker")
    if worker is None:
        raise HTTPException(503, "inference worker is not ready", headers={"Retry-After": "1"})
    try:
        return await worker.run_async(lambda: _probs_core(rec), timeout_s=INFER_TIMEOUT_S,
                                      batch_key=_batch_key(rec), payload=rec,
                                      is_disconnected=request.is_disconnected)
    except InferenceBusy as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    except InferenceCancelled:
        raise HTTPException(499, "client disconnected")
    except InferenceTimeout as exc:
        raise HTTPException(504, str(exc)) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest, request: Request):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    if DATE_FACTS: req = req.model_copy(update={"state": with_date_facts(req.state)})
    rec, meta = to_record(req)
    ps, m = await _probs_async(rec, request)
    answers = to_answers(ps, meta)
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(STATE["tok"], answers)}, "latency_ms": m["latency_ms"]}


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = 6
    seed: int = 0


@app.post("/v1/systemone/permute")
async def systemone_permute(r: PermuteSystemOne, request: Request):
    """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
    q = r.request.questions.get(r.question)
    if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
    rng = random.Random(r.seed); keys = list(q.criteria); runs = []
    for i in range(r.n_perm):
        order = list(keys)
        if i > 0: rng.shuffle(order)
        req = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        if DATE_FACTS: req = req.model_copy(update={"state": with_date_facts(req.state)})
        rec, meta = to_record(req); ps, m = await _probs_async(rec, request)
        a = to_answers(ps, meta)[r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": m["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
async def systemone_separate(req: SystemOneRequest, request: Request):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    answers, tokens, ms = {}, 0, 0.0
    for qid, q in req.questions.items():
        rec, meta = to_record(req.model_copy(update={"questions": {qid: q}, **({"state": with_date_facts(req.state)} if DATE_FACTS else {})})); ps, m = await _probs_async(rec, request)
        answers.update(to_answers(ps, meta)); tokens += m["tokens"]; ms += m["latency_ms"]
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": output_tokens(STATE["tok"], answers)}, "latency_ms": round(ms, 1)}


@app.get("/v1/models")
def models():
    return {"models": [{"id": "kev-latest", "aliases": ["jev-latest"], "run": STATE["run"], "base": STATE["base"]}]}


@app.get("/api/info")
def info():
    ev = f"{STATE['run']}/eval.json"
    effective_min = 0 if STATE["dev"] == "mlx" else PREFIX_MIN_TOKENS
    worker = STATE.get("worker")
    return {"run": STATE["run"], "device": STATE["dev"], "base": STATE["base"], "lora": STATE["lora"],
            "none_option": NONE, "distractors": DISTRACTORS, "has_eval": os.path.exists(ev),
            "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": effective_min, "hits": STATE["prefix_hits"], "misses": STATE["prefix_misses"], "coalesced_misses": STATE["prefix_coalesced"], "cached_states": len(STATE["prefix_cache"]), "worker_state": worker.state if worker else "STOPPED", "queue_depth": worker.queue_depth if worker else 0, "queue_capacity": INFER_QUEUE_SIZE, "batch_requests": INFER_BATCH_ROWS, "batch_wait_ms": INFER_BATCH_WAIT_MS, "batch_max_rows": INFER_BATCH_MAX_ROWS, "batch_max_tokens": INFER_BATCH_MAX_TOKENS}}


@app.get("/api/eval")
def eval_json():
    p = f"{STATE['run']}/eval.json"
    if not os.path.exists(p): raise HTTPException(404, "no eval.json for this run")
    return json.load(open(p))


@app.post("/api/predict")
async def predict(r: Record, request: Request):
    """All questions in one block-causal pass (shared state prefix)."""
    ps, meta = await _probs_async(_rec(r), request)
    return {"probs": ps, **meta}


@app.post("/api/predict_separate")
async def predict_separate(r: Record, request: Request):
    """Each question alone against the same state (N passes). For packed-vs-separate comparison."""
    rec = _rec(r); out, tokens, ms = [], 0, 0.0
    for q in rec["questions"]:
        ps, meta = await _probs_async({"state": rec["state"], "questions": [q]}, request)
        out.append(ps[0]); tokens += meta["tokens"]; ms += meta["latency_ms"]
    return {"probs": out, "tokens": tokens, "latency_ms": round(ms, 1)}


@app.post("/api/permute")
async def permute(r: PermuteReq, request: Request):
    """Shuffle the option order n_perm times; return each ordering's probs mapped back to original indices."""
    rng = random.Random(r.seed); K = len(r.question.options); runs = []
    for i in range(r.n_perm):
        perm = list(range(K))
        if i > 0: rng.shuffle(perm)
        q = {"instr": r.question.instr, "options": [r.question.options[j] for j in perm], "label": 0}
        ps, meta = await _probs_async({"state": r.state, "questions": [q]}, request)
        orig = [0.0] * K
        for pos, j in enumerate(perm): orig[j] = ps[0][pos]
        runs.append({"perm": perm, "probs": orig, "argmax": perm[max(range(K), key=lambda i: ps[0][i])], "latency_ms": meta["latency_ms"]})
    argmaxes = {x["argmax"] for x in runs}
    spread = [max(x["probs"][j] for x in runs) - min(x["probs"][j] for x in runs) for j in range(K)]
    return {"runs": runs, "argmax_stable": len(argmaxes) == 1, "n_distinct_argmax": len(argmaxes), "spread": spread}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()
    from .evaluate import resolve_run
    is_hub_id = re.fullmatch(r"[\w.-]+/[\w.-]+", a.run) and not os.path.isdir(a.run)
    run = a.run if is_hub_id or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    label = run                       # what /v1/models reports: the Hub id or run path as given, not the resolved cache path
    run = resolve_run(run)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    meta = torch.load(f"{run}/head.pt", map_location="cpu")
    if dev == "mps" and not os.environ.get("KEV_ATTN"): os.environ["KEV_ATTN"] = "sdpa"   # serving default on Apple GPUs (parity measured)
    default_backend = "mlx" if "qwen3.5" in str(meta["base"]).lower() else "auto"
    backend = os.environ.get("KEV_BACKEND", default_backend)
    if dev == "mps" and backend in ("auto", "mlx"):
        try:
            from .mlx_model import load_mlx
            tok, model = load_mlx(run)
            dev = "mlx"
            print("using native MLX Qwen3.5 backend")
        except Exception as e:
            if backend == "mlx":
                raise
            print(f"MLX backend unavailable ({type(e).__name__}: {e}); using PyTorch")
            tok, model = load(run, dev)
    else:
        tok, model = load(run, dev)
    STATE.update(run=label, tok=tok, model=model, dev=dev, base=meta["base"], lora=meta["lora"],
                 worker=InferenceWorker(max_queue=INFER_QUEUE_SIZE, request_timeout_s=INFER_TIMEOUT_S,
                                        batch_fn=_batch_jobs if dev == "mlx" else None,
                                        max_batch=INFER_BATCH_ROWS, batch_wait_s=INFER_BATCH_WAIT_MS / 1000.0))
    print(f"serving {label} ({run}) on {dev} :{a.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
