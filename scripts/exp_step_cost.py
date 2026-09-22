"""Cost of one training record (forward + backward, no optimizer step) as a function of state length S and question count Q,
on Jev-shaped synthetic inputs (kev's own suite is short-state, so the long-state story has to be shown synthetically).

Modes: packed = DecisionModel.forward_batch + backward (state adapted); chunked = kev.cached_state.chunked_loss_backward with
--chunk questions per chunk (state adapted, gradient-exact); frozen = base state K/V (no grad) then branch_logits + backward
(cache miss); frozen_cached = branch_logits + backward over precomputed state K/V (cache hit); frozen_chunked / frozen_cached_chunked
= the same two with the branches --chunk questions at a time, backward per chunk (frozen_loss_backward). Per cell: median wall seconds of
--repeats after one warmup, peak device memory (MPS: driver/current allocated deltas; CUDA: max_memory_allocated; CPU: 0),
packed length, dense mask bytes and forward token counts. Out-of-memory cells are recorded (oom: true) and skipped upward.

Run: cd <repo> && PYTHONPATH=. uv run python scripts/exp_step_cost.py --device mps --states 128,512,1024,2048,4096 --questions 1,4,16,64 --out runs/step_cost.json
"""
import argparse, gc, json, os, time
import torch
import torch.nn.functional as F
from kev.device import allocated_bytes, empty_cache
from kev.model import DecisionModel, load_tokenizer, encode, user_tokens
from kev.cached_state import branch_logits, chunked_loss_backward, frozen_loss_backward, size_bytes, state_kv, state_len, sync

PARAGRAPH = ("The quarterly review covered the rollout of the new billing pipeline across the three regional data centers. "
             "Migration of the customer ledger finished two days ahead of schedule, although the reconciliation job still "
             "reports a small number of duplicate invoices each night that the finance team clears by hand. Support volume "
             "rose by about twelve percent during the first week, mostly password resets and questions about the new "
             "statement layout, and returned to baseline afterwards. The infrastructure group asked for a larger connection "
             "pool on the reporting replica and for an alert on queue depth; both requests were approved. Next quarter the "
             "team plans to retire the legacy exporter, move the nightly batch to an hourly cadence, and publish a customer "
             "facing status page for the pipeline. ")
QUESTIONS = ["Which of the following best describes the current status of the migration described above?",
             "Based on the report, what should the team prioritize in the coming quarter to reduce manual work?",
             "Which risk is most likely to affect customers if the reconciliation job is left unchanged?",
             "According to the review, which request from the infrastructure group was approved?",
             "What is the most plausible reason support volume rose during the first week of the rollout?"]
OPTIONS = ["finished ahead of schedule", "delayed by regional outages", "paused pending a security audit", "still in the planning stage",
           "retire the legacy exporter", "double the support headcount", "roll back the billing pipeline", "rewrite the customer ledger",
           "duplicate invoices reaching customers", "loss of historical statements", "slower password resets", "missing status page updates",
           "a larger connection pool", "a second reporting replica", "a new finance dashboard", "an hourly batch cadence"]


def text_of_tokens(tok, text, n):
    """The first n tokens of text (repeated as needed), decoded back to text."""
    ids = user_tokens(tok, text)
    while len(ids) < n: ids = ids + user_tokens(tok, " " + text)
    return tok.decode(ids[:n])


def synthetic_record(tok, S, Q, branch_tokens, K):
    """A record whose state encodes to exactly S tokens (<state> + S-1 text tokens) with Q questions of K options, each branch
    about branch_tokens tokens: <q> instr <opt> o </opt> x K <decide> = 2 + |instr| + K * (2 + |opt|)."""
    instr_n = max(1, (branch_tokens - 2 - 2 * K) // 2)
    opt_n = max(1, (branch_tokens - 2 - 2 * K - instr_n) // K)
    qs = []
    for q in range(Q):
        instr = text_of_tokens(tok, QUESTIONS[q % len(QUESTIONS)], instr_n)
        opts = [text_of_tokens(tok, OPTIONS[(q * K + j) % len(OPTIONS)], opt_n) for j in range(K)]
        qs.append({"instr": instr, "options": opts, "label": 0})
    n = S - 1
    for _ in range(8):   # decode/re-tokenize may not round-trip exactly: nudge the token count until the state is S long
        rec = {"state": text_of_tokens(tok, PARAGRAPH, n), "questions": qs}
        enc = encode(tok, rec, max_state=S + 8, max_branch=10 ** 7, strict=False)
        got = state_len(enc)
        if got == S: return rec, enc
        n += S - got
    raise ValueError(f"could not build a state of exactly {S} tokens (got {got})")


def mem_reset(dev):
    """Free cached memory and return the baseline counters."""
    gc.collect(); empty_cache(dev)
    if dev == "mps":
        return {"current": allocated_bytes(dev), "driver": torch.mps.driver_allocated_memory()}
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats(); return {"current": torch.cuda.memory_allocated()}
    return {}


def mem_delta(dev, base):
    """Memory above the baseline, in MB. MPS: the driver delta is the high-water mark of the caching allocator since the reset
    (peak proxy), the current delta is what is still allocated; CUDA: peak allocated; CPU: 0."""
    if dev == "mps":   # kev.device.allocated_bytes is the current allocation on MPS and the peak on CUDA
        return {"peak_mb": (torch.mps.driver_allocated_memory() - base["driver"]) / 2 ** 20,
                "current_mb": (allocated_bytes(dev) - base["current"]) / 2 ** 20}
    if dev == "cuda":
        return {"peak_mb": (allocated_bytes(dev) - base["current"]) / 2 ** 20,
                "current_mb": (torch.cuda.memory_allocated() - base["current"]) / 2 ** 20}
    return {"peak_mb": 0.0, "current_mb": 0.0}


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or any(s in str(e).lower() for s in ("out of memory", "failed to allocate"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--states", default="128,512,1024,2048,4096")
    ap.add_argument("--questions", default="1,4,16,64")
    ap.add_argument("--branch_tokens", type=int, default=40)
    ap.add_argument("--options", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", default="packed,frozen,frozen_cached,chunked,frozen_chunked,frozen_cached_chunked")
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--out", default="runs/step_cost.json")
    a = ap.parse_args()
    dev = a.device; modes = a.modes.split(",")
    unknown = set(modes) - {"packed", "frozen", "frozen_cached", "chunked", "frozen_chunked", "frozen_cached_chunked"}
    if unknown: ap.error(f"unknown --modes: {sorted(unknown)}")
    states = [int(s) for s in a.states.split(",")]; questions = [int(q) for q in a.questions.split(",")]
    torch.manual_seed(0)
    tok = load_tokenizer(a.base)
    t0 = time.perf_counter(); model = DecisionModel(a.base, tok, dev, lora=a.lora); model.train()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.02)   # PEFT starts lora_B at 0 (identity adapter, zero lora_A grads)
    res = {"env": {"device": dev, "base": a.base, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                   "attn": model.lm.config._attn_implementation, "load_s": time.perf_counter() - t0, "threads": torch.get_num_threads()},
           "args": vars(a), "cells": []}

    def q_loss(z, q): return F.cross_entropy(z.float()[None], torch.tensor([0], device=dev))

    def step_fn(mode, enc, S, Q, kv_hit):
        """The compute of one training step in `mode`: a closure whose call runs forward + backward for the record."""
        def loss_of(zs): return sum(q_loss(z, q) for q, z in enumerate(zs)) / Q
        if mode == "packed":
            return lambda: loss_of(model.forward_batch([enc])[0]).backward()
        if mode == "chunked":
            return lambda: chunked_loss_backward(model, enc, lambda z, q: q_loss(z, q) / Q, a.chunk)
        if mode == "frozen":
            return lambda: loss_of(branch_logits(model, enc, state_kv(model, enc["ids"][:S], adapter=False, grad=False))).backward()
        if mode == "frozen_cached":
            return lambda: loss_of(branch_logits(model, enc, kv_hit)).backward()
        if mode == "frozen_chunked":
            return lambda: frozen_loss_backward(model, enc, state_kv(model, enc["ids"][:S], adapter=False, grad=False), lambda z, q: q_loss(z, q) / Q, a.chunk)
        if mode == "frozen_cached_chunked":
            return lambda: frozen_loss_backward(model, enc, kv_hit, lambda z, q: q_loss(z, q) / Q, a.chunk)
        raise ValueError(f"unknown mode {mode!r}")

    oom_at = {m: [] for m in modes}   # (S, Q) cells that ran out of memory: larger cells in both dimensions are skipped
    rows = []
    for S in states:
        for Q in questions:
            rec, enc = synthetic_record(tok, S, Q, a.branch_tokens, a.options)
            L = len(enc["ids"]); Lb = L - S
            cell = {"S": S, "Q": Q, "L": L, "branch_tokens": Lb, "branch_tokens_per_question": Lb / Q, "mask_bytes": L * L * 4,
                    "state_kv_bytes_fp32": S * 2 * model.lm.config.num_hidden_layers * model.lm.config.num_key_value_heads * model.lm.config.head_dim * 4,
                    "modes": {}}
            chunks = [min(a.chunk, Q - s) for s in range(0, Q, a.chunk)]
            per_q = Lb / Q; chunk_mask = max(int(c * per_q) * (S + int(c * per_q)) * 4 for c in chunks)
            tokens = {"packed": {"forward_tokens": L, "mask_bytes": L * L * 4},
                      "chunked": {"forward_tokens": L, "mask_bytes": chunk_mask, "chunks": len(chunks)},
                      "frozen": {"forward_tokens": L, "mask_bytes": Lb * (S + Lb) * 4},
                      "frozen_cached": {"forward_tokens": Lb, "mask_bytes": Lb * (S + Lb) * 4},
                      "frozen_chunked": {"forward_tokens": L, "mask_bytes": chunk_mask, "chunks": len(chunks)},
                      "frozen_cached_chunked": {"forward_tokens": Lb, "mask_bytes": chunk_mask, "chunks": len(chunks)}}
            for mode in modes:
                r = dict(tokens[mode])
                if any(S >= s0 and Q >= q0 for s0, q0 in oom_at[mode]):
                    r.update(oom=True, skipped=True); cell["modes"][mode] = r; continue
                kv_hit = None
                try:
                    if mode in ("frozen_cached", "frozen_cached_chunked"):
                        kv_hit = state_kv(model, enc["ids"][:S], adapter=False, grad=False); r["state_kv_bytes"] = size_bytes(kv_hit)
                    fn = step_fn(mode, enc, S, Q, kv_hit)
                    model.zero_grad(set_to_none=True); fn(); sync(dev)   # warmup
                    ts, mems = [], []
                    for _ in range(a.repeats):
                        model.zero_grad(set_to_none=True); base = mem_reset(dev)
                        sync(dev); t = time.perf_counter(); fn(); sync(dev); ts.append(time.perf_counter() - t)
                        mems.append(mem_delta(dev, base))
                    r.update(seconds=sorted(ts)[len(ts) // 2], seconds_min=min(ts), seconds_all=ts,
                             peak_mb=max(m["peak_mb"] for m in mems), current_mb=max(m["current_mb"] for m in mems), oom=False)
                except RuntimeError as e:
                    if not is_oom(e): raise
                    r.update(oom=True, error=str(e)[:200]); oom_at[mode].append((S, Q))
                finally:
                    model.zero_grad(set_to_none=True); del kv_hit; mem_reset(dev)
                cell["modes"][mode] = r
                print(f"S={S:5d} Q={Q:3d} L={L:6d} {mode:21s} " + ("OOM" if r["oom"] else f"{r['seconds']:8.3f}s {r['peak_mb']:9.1f}MB"), flush=True)
            res["cells"].append(cell); rows.append(cell)

    print(f"\n{'S':>5} {'Q':>3} {'L':>6} {'mask MB':>8} " + " ".join(f"{m:>21s}" for m in modes) + "   (seconds / peak MB)")
    for c in rows:
        line = f"{c['S']:5d} {c['Q']:3d} {c['L']:6d} {c['mask_bytes'] / 2 ** 20:8.1f} "
        for m in modes:
            r = c["modes"][m]
            line += " " + (f"{'OOM':>21s}" if r["oom"] else f"{r['seconds']:9.3f}s {r['peak_mb']:8.0f}MB")
        print(line)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f: json.dump(res, f, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
