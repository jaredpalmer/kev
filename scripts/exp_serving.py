"""Inference-side effects of a frozen-state decision model: latency per (state tokens S, questions Q) cell, median ms after
one warmup with the device synchronized, and whether each measurement ran out of memory.

  full       one packed prefill of state + Q questions (model.probs, adapted mode: what kev serves today)
  separate   Q packed prefills of state + 1 question (a client that does not pack)
  state_pass the base-weights state pass alone (state_kv adapter=False): the one-time cost of a reusable state cache
  requery    Q new questions against an already cached state (branch_logits only): a follow-up question on a stored state
  shared     one base state cache serving two adapters: branch_logits under adapter A and B over the same cache vs a
             from-scratch frozen-mode computation per adapter (max-abs logits diff; the latency saved is the state pass)

--run: checkpoint dir (kev.checkpoint.load, adapter unmerged). Without head.pt (or omitted) an untrained DecisionModel with lora=16 is built and its
lora_B perturbed so the adapter is not the identity. Adapter B: --run2 (second checkpoint on the same base) or, otherwise,
adapter A with lora_B perturbed by a different seed.
Run: cd <repo> && PYTHONPATH=. uv run python scripts/exp_serving.py --device cpu --run runs/kev --out runs/serving.json
"""
import argparse, gc, json, math, os, time
import torch
from kev.model import DecisionModel, load_tokenizer, encode
from kev.checkpoint import LoadOptions, load, read_meta
from kev.device import empty_cache
from kev.cached_state import branch_logits, size_bytes, state_kv, state_len, sync

PARAGRAPH = ("Incident report INC-4471. At 02:13 UTC the checkout service began returning HTTP 503 for roughly 40% of requests "
             "in the eu-west region. The on-call engineer observed that the primary Postgres instance had hit its connection limit "
             "after a deploy of the order-history worker opened one pool per process instead of one pool per host. Latency on the "
             "load balancer stayed normal and the cache hit rate was unchanged. The worker was rolled back at 02:41 UTC and "
             "connections drained within six minutes. No data was lost. Customers who retried succeeded; about 1,900 orders were "
             "delayed. A follow-up task was filed to add a connection-count alert and to cap the pool size in the worker's configuration. ")
OPTIONS = ["database", "load balancer", "cache", "network"]


def question(q):
    return {"instr": f"Question {q + 1}: which component does this part of the report point to?", "options": OPTIONS, "label": 0}


def record(tok, S, Q):
    """A record whose encoding has exactly S state tokens (text repeated past S, truncated by encode) and Q questions."""
    n = len(tok(PARAGRAPH, add_special_tokens=False).input_ids)
    return {"state": PARAGRAPH * (math.ceil(S / n) + 1), "questions": [question(q) for q in range(Q)]}


def enc_of(tok, rec, S):
    return encode(tok, rec, max_state=S, max_branch=S + 4096)


def is_oom(exc):
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower() or "can't allocate memory" in str(exc).lower()


def timed(fn, dev, n):
    """{"ms": median, "min_ms": best, "oom": bool} over n runs after one warmup (device synchronized), and the last result."""
    try:
        r = fn(); ts = []
        for _ in range(n):
            sync(dev); t = time.perf_counter(); r = fn(); sync(dev); ts.append(1e3 * (time.perf_counter() - t))
        return {"ms": sorted(ts)[n // 2], "min_ms": min(ts), "oom": False}, r
    except (RuntimeError, MemoryError) as exc:
        if not is_oom(exc): raise
    free(dev)   # after the handler: the exception's traceback held the failed forward's tensors until then
    return {"ms": None, "min_ms": None, "oom": True}, None


def free(dev):
    gc.collect()
    empty_cache(dev)


def perturb_lora_b(model, seed, std=0.02):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.add_(torch.randn(p.shape, generator=g).to(p) * std)


class Adapters:
    """Two adapters on one model: A = the loaded/constructed one; B = --run2 (adapter + head) or A with perturbed lora_B."""

    def __init__(self, model, run2, dev):
        self.model, self.names = model, ["A", "B"]
        lora_b = {n: p for n, p in model.named_parameters() if "lora_B" in n}
        if run2:
            meta = read_meta(run2)
            model.lm.load_adapter(run2, adapter_name="second"); model.lm.to(dev)
            self.heads = [{k: v.clone() for k, v in model.head.state_dict().items()}, meta.head]
            self.use = self._use_loaded; self.kind = f"loaded from {run2}"
        else:
            a = {n: p.detach().clone() for n, p in lora_b.items()}
            g = torch.Generator().manual_seed(1)
            b = {n: t + torch.randn(t.shape, generator=g).to(t) * 0.02 for n, t in a.items()}
            self.weights, self.params = [a, b], lora_b
            self.use = self._use_emulated; self.kind = "A with lora_B + N(0, 0.02) noise (seed 1)"

    def _use_loaded(self, i):
        self.model.lm.set_adapter(["default", "second"][i]); self.model.head.load_state_dict(self.heads[i])

    def _use_emulated(self, i):
        with torch.no_grad():
            for n, p in self.params.items(): p.copy_(self.weights[i][n])


def maxabs(zs, ws):
    return max((z.float() - w.float()).abs().max().item() for z, w in zip(zs, ws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--run", default="", help="checkpoint dir; without head.pt an untrained lora=16 model is measured")
    ap.add_argument("--run2", default="", help="second checkpoint (same base) for the shared-cache check")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base", help="base for the untrained model")
    ap.add_argument("--states", default="128,512,1024,2048,4096,8192")
    ap.add_argument("--questions", default="1,8,32")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="runs/serving.json")
    a = ap.parse_args()
    dev = a.device; torch.manual_seed(0)
    states = [int(s) for s in a.states.split(",")]; questions = [int(q) for q in a.questions.split(",")]

    t0 = time.perf_counter()
    if a.run and os.path.exists(f"{a.run}/head.pt"):
        tok, model = load(a.run, dev, LoadOptions(merge=False)); source = a.run   # unmerged: the adapters are switched below
    else:
        tok = load_tokenizer(a.base); model = DecisionModel(a.base, tok, dev, lora=16); model.eval()
        perturb_lora_b(model, 0); source = f"untrained {a.base} lora=16, lora_B ~ N(0, 0.02) (seed 0)"
    trained_mode = model.state_mode
    adapters = Adapters(model, a.run2, dev); adapters.use(0)
    lm_dtype = str(next(model.lm.parameters()).dtype).replace("torch.", "")
    res = {"env": {"device": dev, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                   "attn": model.lm.config._attn_implementation, "dtype": lm_dtype, "load_s": time.perf_counter() - t0},
           "model": {"source": source, "base": model.lm.config._name_or_path, "trained_state_mode": trained_mode, "adapter_b": adapters.kind},
           "config": vars(a), "states": {}, "cells": []}
    print(f"model: {source}; adapter B: {adapters.kind}; device {dev} {lm_dtype} {model.lm.config._attn_implementation}")

    for S in states:
        ids = enc_of(tok, record(tok, S, 1), S)["ids"][:S]
        with torch.no_grad():
            t_state, kv = timed(lambda: state_kv(model, ids, adapter=False), dev, a.repeats)
        cache_bytes = size_bytes(kv) if kv is not None else None
        res["states"][S] = {"S": len(ids), "state_pass": t_state, "cache_bytes": cache_bytes,
                            "cache_bytes_bf16": cache_bytes // 2 if cache_bytes else None}
        for Q in questions:
            rec = record(tok, S, Q); enc = enc_of(tok, rec, S)
            singles = [enc_of(tok, {"state": rec["state"], "questions": [q]}, S) for q in rec["questions"]]
            assert state_len(enc) == len(ids) and all(e["ids"][:S] == ids for e in singles)
            cell = {"S": S, "Q": Q, "tokens": len(enc["ids"]), "branch_tokens": len(enc["ids"]) - S, "cache_bytes": cache_bytes,
                    "state_pass": t_state}
            with torch.no_grad():
                model.state_mode = "adapted"
                cell["full"], _ = timed(lambda: model.probs(enc), dev, a.repeats)
                cell["separate"], _ = timed(lambda: [model.probs(e) for e in singles], dev, a.repeats)
                if kv is not None:
                    cell["requery"], z_a = timed(lambda: branch_logits(model, enc, kv), dev, a.repeats)
                    model.state_mode = "frozen"
                    cell["scratch_a"], z_a_scratch = timed(lambda: model(enc), dev, a.repeats)
                    adapters.use(1)
                    cell["shared_b"], z_b = timed(lambda: branch_logits(model, enc, kv), dev, a.repeats)
                    cell["scratch_b"], z_b_scratch = timed(lambda: model(enc), dev, a.repeats)
                    adapters.use(0)
                    if None not in (z_a, z_a_scratch, z_b, z_b_scratch):
                        cell["shared"] = {"maxabs_a": maxabs(z_a, z_a_scratch), "maxabs_b": maxabs(z_b, z_b_scratch),
                                          "maxabs_a_vs_b": maxabs(z_a, z_b),   # the two adapters really are different functions
                                          "saved_ms": cell["scratch_b"]["ms"] - cell["shared_b"]["ms"]}
                    del z_a, z_a_scratch, z_b, z_b_scratch
                else:
                    for k in ("requery", "scratch_a", "shared_b", "scratch_b"): cell[k] = {"ms": None, "min_ms": None, "oom": True}
            res["cells"].append(cell); free(dev)
            print(fmt_row(cell), flush=True)
        del kv; free(dev)

    model.state_mode = trained_mode
    print(); print(header()); [print(fmt_row(c)) for c in res["cells"]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f: json.dump(res, f, indent=1, default=str)
    print("wrote", a.out)


COLS = [("S", 6), ("Q", 3), ("tokens", 7), ("cacheMB", 8), ("full", 9), ("separate", 9), ("state_pass", 10), ("requery", 9),
        ("shared_b", 9), ("scratch_b", 9), ("saved", 8), ("diff_a", 8), ("diff_b", 8), ("a_vs_b", 8)]


def header():
    return "  ".join(f"{n:>{w}}" for n, w in COLS) + "   (ms, median; OOM marked)"


def fmt_row(c):
    def ms(t): return "OOM" if t is None or t["oom"] else f"{t['ms']:.1f}"
    def g(k): return f"{c['shared'][k]:.2g}" if "shared" in c else "-"
    vals = [c["S"], c["Q"], c["tokens"], f"{c['cache_bytes'] / 2**20:.1f}" if c["cache_bytes"] else "OOM", ms(c["full"]), ms(c["separate"]),
            ms(c["state_pass"]), ms(c["requery"]), ms(c["shared_b"]), ms(c["scratch_b"]),
            f"{c['shared']['saved_ms']:.1f}" if "shared" in c else "-", g("maxabs_a"), g("maxabs_b"), g("maxabs_a_vs_b")]
    return "  ".join(f"{str(v):>{w}}" for v, (_, w) in zip(vals, COLS))


if __name__ == "__main__":
    main()
