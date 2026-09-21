"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008
"""
import argparse, json, os, random, re, threading, time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .data import DISTRACTORS, NONE
from .evaluate import load
from .model import encode

# inference limits (training used 384/640); per-branch cap mirrors Jev's ~32k, bounded by the base model window
INFER_MAX_STATE, INFER_MAX_BRANCH = 8192, 8192

app = FastAPI(title="kev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {"run": None, "tok": None, "model": None, "dev": None, "lock": threading.Lock(), "prefix_cache": {}, "prefix_hits": 0, "prefix_misses": 0}
PREFIX_CACHE_SIZE = int(os.environ.get("KEV_PREFIX_CACHE", "4"))          # states kept (KV + hidden); 0 disables
PREFIX_MIN_TOKENS = int(os.environ.get("KEV_PREFIX_MIN_TOKENS", "384"))
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


def _probs(rec):
    """One forward pass; the state prefix (tokens up to the first question) is cached across requests, so a repeated state
    only pays for its question branches. Exactness: the state's activations do not depend on the branches."""
    tok, model, dev = STATE["tok"], STATE["model"], STATE["dev"]
    try: enc = model.encode(tok, rec, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH)
    except ValueError as e: raise HTTPException(422, str(e))
    Ls = enc["seg"].count(0); key = (tuple(enc["ids"][:Ls]), bool(enc.get("option_isolation")))
    cache = STATE["prefix_cache"]
    with STATE["lock"]:
        _sync(dev); t = time.time()
        # MLX's recurrent prefix pass is substantially cheaper than repeating
        # the state once per question, even for short states. The old MPS
        # threshold remains for the PyTorch fallback where cache setup can
        # dominate short requests.
        eligible = PREFIX_CACHE_SIZE and (Ls >= PREFIX_MIN_TOKENS or dev == "mlx")
        if eligible and key in cache:
            prefix = cache.pop(key)                       # pop + reinsert = LRU order
            ps = model.probs_with_prefix(enc, prefix); cache[key] = prefix
            STATE["prefix_hits"] += 1; hit = True
        elif eligible:
            ps, prefix = model.probs_and_prefix(enc)      # one pass, and the state prefix is kept for next time
            cache[key] = prefix
            while len(cache) > PREFIX_CACHE_SIZE: cache.pop(next(iter(cache)))
            STATE["prefix_misses"] += 1; hit = False
        else:
            ps = model.probs(enc); hit = False
        _sync(dev); dt = time.time() - t
    if TEMPERATURE != 1.0:                      # opt-in calibration: same as scaling the pointer logits by 1/T (argmax unchanged)
        ps = [(lambda q: q / q.sum())(torch.as_tensor(p).clamp_min(1e-9) ** (1.0 / TEMPERATURE)) for p in ps]
    return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": hit}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    if DATE_FACTS: req = req.model_copy(update={"state": with_date_facts(req.state)})
    rec, meta = to_record(req)
    ps, m = _probs(rec)
    answers = to_answers(ps, meta)
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(STATE["tok"], answers)}, "latency_ms": m["latency_ms"]}


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = 6
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
        req = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        if DATE_FACTS: req = req.model_copy(update={"state": with_date_facts(req.state)})
        rec, meta = to_record(req); ps, m = _probs(rec)
        a = to_answers(ps, meta)[r.question]
        runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": m["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    answers, tokens, ms = {}, 0, 0.0
    for qid, q in req.questions.items():
        rec, meta = to_record(req.model_copy(update={"questions": {qid: q}, **({"state": with_date_facts(req.state)} if DATE_FACTS else {})})); ps, m = _probs(rec)
        answers.update(to_answers(ps, meta)); tokens += m["tokens"]; ms += m["latency_ms"]
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": output_tokens(STATE["tok"], answers)}, "latency_ms": round(ms, 1)}


@app.get("/v1/models")
def models():
    return {"models": [{"id": "kev-latest", "aliases": ["jev-latest"], "run": STATE["run"], "base": STATE["base"]}]}


@app.get("/api/info")
def info():
    ev = f"{STATE['run']}/eval.json"
    effective_min = 0 if STATE["dev"] == "mlx" else PREFIX_MIN_TOKENS
    return {"run": STATE["run"], "device": STATE["dev"], "base": STATE["base"], "lora": STATE["lora"],
            "none_option": NONE, "distractors": DISTRACTORS, "has_eval": os.path.exists(ev),
            "prefix_cache": {"size": PREFIX_CACHE_SIZE, "min_state_tokens": effective_min, "hits": STATE["prefix_hits"], "misses": STATE["prefix_misses"], "cached_states": len(STATE["prefix_cache"])}}


@app.get("/api/eval")
def eval_json():
    p = f"{STATE['run']}/eval.json"
    if not os.path.exists(p): raise HTTPException(404, "no eval.json for this run")
    return json.load(open(p))


@app.post("/api/predict")
def predict(r: Record):
    """All questions in one block-causal pass (shared state prefix)."""
    ps, meta = _probs(_rec(r))
    return {"probs": ps, **meta}


@app.post("/api/predict_separate")
def predict_separate(r: Record):
    """Each question alone against the same state (N passes). For packed-vs-separate comparison."""
    rec = _rec(r); out, tokens, ms = [], 0, 0.0
    for q in rec["questions"]:
        ps, meta = _probs({"state": rec["state"], "questions": [q]})
        out.append(ps[0]); tokens += meta["tokens"]; ms += meta["latency_ms"]
    return {"probs": out, "tokens": tokens, "latency_ms": round(ms, 1)}


@app.post("/api/permute")
def permute(r: PermuteReq):
    """Shuffle the option order n_perm times; return each ordering's probs mapped back to original indices."""
    rng = random.Random(r.seed); K = len(r.question.options); runs = []
    for i in range(r.n_perm):
        perm = list(range(K))
        if i > 0: rng.shuffle(perm)
        q = {"instr": r.question.instr, "options": [r.question.options[j] for j in perm], "label": 0}
        ps, meta = _probs({"state": r.state, "questions": [q]})
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
    STATE.update(run=label, tok=tok, model=model, dev=dev, base=meta["base"], lora=meta["lora"])
    print(f"serving {label} ({run}) on {dev} :{a.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
