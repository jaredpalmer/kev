"""Serving latency and parity on CUDA: the kev.serve path with and without CUDA graphs (kev.cuda_graphs).

    uv run modal run modal_app.py::serving --run jaredpalmer/kev-4b --gpu A100-80GB --name serving-4b-a100
    python scripts/serving_bench.py --run jaredpalmer/kev-4b --out runs/serving-4b      # on a CUDA machine
    uv run modal run modal_app.py::serving --run jaredpalmer/kev-27b --gpu H200 --name lc-serve-27b-h200-3 --flags "--state_tokens 8192,16384,32768,49152,65536 --reps 3"             # long states only (long_states)

Parity: for n clean development records of decision-v7, the probabilities served in bf16 (eager, and through CUDA graphs:
the state-pass miss path and the cached-state hit path) against the fp32 eager path every reported number uses (max and
mean |dp|, argmax flips). Latency: model time (what /v1/systemone reports as latency_ms) for four request shapes, each
with a new state per request (the usual API call: every ticket is a new state) and with a repeated state (prefix cache
hit), eager and with graphs. Isolation (--isolation): on the same records through the served bf16 path, each question
alone against (a) the full request and (b) the question plus an unrelated sibling (kev.experiment.ISOLATION_PROBE); the
fp32 mechanism check (kev.experiment.mechanism_checks, 8 records, tolerance 1e-3) is exact arithmetic, this is the
precision the API serves. Writes report.json.
"""
import argparse, gc, json, statistics, time
from pathlib import Path

import torch

from kev.api import SystemOneRequest
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import allocated_bytes, empty_cache
from kev.experiment import ISOLATION_PROBE
from kev.serve import Server
from kev.suite import load_split, write_json

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}},
    "return_reason": {"type": "choice", "instructions": "If the customer wants to return something, why?",
                      "criteria": {"wrong_size": "The item doesn't fit", "wrong_item": "A different product was delivered", "damaged": "The item arrived broken or faulty",
                                   "changed_mind": "The item is fine, the customer no longer wants it", "other": "A return reason that fits none of the above"}},
    "requested_resolution": {"type": "choice", "instructions": "What does the customer want to happen?",
                             "criteria": {"exchange": "Swap the item for a different one", "refund": "Money back", "replacement": "The same item sent again", "information": "Just an answer, no action needed"}},
    "tone": {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}},
    "escalate": {"type": "noul", "instructions": "Does this message require urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]},
}
TICKET = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. What are you going to do about this?"
PARAGRAPH = ("I ordered a pair of running shoes on the first of the month and paid with my credit card. The confirmation email said "
             "delivery in three to five business days, but the tracking page did not update for over a week, and when the package "
             "finally arrived the box was crushed on one side. The shoes inside were a size ten instead of the size nine I ordered. ")
FIVE = {k: QUESTIONS[k] for k in ("department", "return_reason", "requested_resolution", "escalate", "frustration")}
# name -> (state text, questions); the playground's Support triage preset is "triage"
CASES = {"2 questions, short state": (TICKET, {k: QUESTIONS[k] for k in ("department", "escalate")}),
         "6 questions, short state": (TICKET, QUESTIONS),
         "5 questions, 370-token state": (PARAGRAPH * 5, FIVE),
         "5 questions, 2,200-token state": (PARAGRAPH * 30, FIVE)}


def request(case, i):
    """i = 0 repeats the case's state; i > 0 prefixes a ticket number, a new state of (almost) the same length."""
    state, questions = CASES[case]
    return SystemOneRequest.model_validate({"state": state if i == 0 else f"Ticket {i}. {state}", "questions": questions})


def latency(server, reps):
    """Median model time per case, new and repeated state. first_ms is the case's first request: with graphs its new
    buckets run eagerly and are captured once the server is idle; it also pays any first-call kernel autotuning."""
    out = {}
    for case in CASES:
        first = server.answer(request(case, 0))
        row = {"tokens": first["usage"]["input_tokens"], "first_ms": first["latency_ms"]}
        for mode in ("new", "cached"):
            ms = [server.answer(request(case, i if mode == "new" else 0))["latency_ms"] for i in range(1, reps + 3)][2:]
            row[f"{mode}_ms"] = statistics.median(ms)
        out[case] = row
    return out


def throughput(server, suite, levels=(1, 8, 32, 64)):
    """Concurrent clients calling Server.probs (in-process, no HTTP): p50 / p99 per request and requests/s, on 256
    decision-v7 development records (short states, 1-6 questions: API-like traffic) and on 64 requests of 5 questions over
    a 2,200-token state. Every level runs twice: a first pass over all levels meets their batch shapes (they run eagerly
    and get their CUDA graphs captured, reported as `first`), then the server settles and the second pass is the steady
    state."""
    import random
    from concurrent.futures import ThreadPoolExecutor
    from kev.api import to_record
    samples = {"6 questions, new short state": [to_record(request("6 questions, short state", 1000 + i))[0] for i in range(256)],
               "decision-v7 development": random.Random(0).choices([materialize(r) for r in load_split(suite, "development")], k=256),
               "5 questions, 2,200-token state": [to_record(request("5 questions, 2,200-token state", 1000 + i))[0] for i in range(64)]}
    def run(recs, c):
        def one(rec):
            t = time.perf_counter(); server.probs(rec); return time.perf_counter() - t
        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter(); lat = sorted(pool.map(one, recs)); wall = time.perf_counter() - start
        return {"p50_ms": round(1000 * statistics.median(lat), 1), "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1), "requests_per_s": round(len(lat) / wall, 1)}

    out = {}
    for name, recs in samples.items():
        first = {c: run(recs, c) for c in levels}   # meets every level's batch shapes first, so the timed runs replay graphs
        server.wait_idle()
        for c in levels:
            out[f"{name} @ {c} clients"] = {**run(recs, c), "first": first[c]}
            print(name, c, out[f"{name} @ {c} clients"], flush=True)
    return out


def agreement(pairs):
    """[(p, q), ...] probability vectors of the same question -> max and mean of max |p - q|, and argmax flips."""
    dp = [float((p - q).abs().max()) for p, q in pairs]
    return {"max_dp": max(dp), "mean_dp": statistics.mean(dp), "argmax_flips": sum(int(p.argmax() != q.argmax()) for p, q in pairs)}


def isolation(m, tok, raw):
    """Each question alone vs in the full request ("packed") and vs after ISOLATION_PROBE ("sibling"), through the served
    path (probs_batch: bf16, fused kernels and graphs if loaded). raw = labelled records, before materialize."""
    def serve(record):
        return m.probs_batch([m.encode(tok, materialize(record))], [None], [False])[0][0]
    out = {"packed": [], "sibling": []}
    for r in raw:
        full = serve(r)
        for i, (qid, q) in enumerate(r["questions"].items()):
            alone = serve({**r, "questions": {qid: q}})[0]
            out["packed"].append((alone, full[i]))
            out["sibling"].append((alone, serve({**r, "questions": {"isolated_probe": ISOLATION_PROBE, qid: q}})[1]))
    return {k: {"questions": len(v), **agreement(v)} for k, v in out.items()}


def long_states(ck, lengths, reps, out):
    """--state_tokens: per state length, a request of 5 decision-v7 questions on a state of exactly that many tokens of
    decision-v7 text (scripts/sft_probe.build_long), through kev.benchmark's path (kev.predictors.LocalPredictor, bf16 as a
    bf16-trained checkpoint loads, the admission context lifted to the longest length; then evaluate_records on one record
    per length) and through the served one (bf16, fused kernels, CUDA graphs; a state past the graphed state pass runs its
    state eagerly). The two load one after the other (both would not fit with a 27B). Model time for a new state (median
    of `reps`, each a different record) and a cached one, peak GPU memory (the allocator's) per path, and the agreement of
    the served answers with the benchmark's."""
    from kev.benchmark import evaluate_records
    from kev.model import training_context
    from kev.predictors import LocalPredictor
    from scripts.sft_probe import build_long
    report, save = {"run": str(ck.path), "gpu": torch.cuda.get_device_name(0), "lengths": {}}, lambda: write_json(Path(out) / "report.json", report)
    context = {**training_context(max(lengths)), "truncate": False}   # what a 64k-state suite's manifest would admit under
    predictor = LocalPredictor(ck.path, "cuda", context=context)
    records = {n: [{**r, "_meta": {**r["_meta"], "group_id": r["_meta"]["id"], "variant": "clean"}} for r in build_long(predictor.tok, reps + 1, n, 5, 0)[0]] for n in lengths}
    reference = {}
    for n, recs in records.items():
        try:
            torch.cuda.reset_peak_memory_stats()
            pred = predictor(recs[0])
            reference[n] = [torch.tensor(list(p.values())) for p in pred["probabilities"].values()]
            row = {"latency_ms": round(pred["latency_ms"], 1), "input_tokens": pred["input_tokens"], "peak_allocated_gb": round(allocated_bytes("cuda") / 1e9, 1)}
        except Exception as error:
            row = {"error": f"{type(error).__name__}: {str(error)[:500]}"}
        report["lengths"][n] = {"benchmark": row}; print(n, row, flush=True); save()
    try:   # kev.benchmark's scoring loop end to end on one record per length
        bench, _ = evaluate_records([recs[0] for recs in records.values()], predictor, Path(out) / "benchmark")
        report["evaluate_records"] = {"coverage": bench["coverage"], "latency_ms": bench["latency_ms"]}
    except Exception as error:
        report["evaluate_records"] = {"error": f"{type(error).__name__}: {str(error)[:500]}"}
    del predictor; gc.collect(); empty_cache("cuda")
    torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True)   # LocalPredictor turned them off; serving keeps them
    tok, m = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
    report["resident_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    server = Server(ck, tok, m, "cuda", release_date="-")
    server.answer(request("5 questions, 2,200-token state", 1)); server.wait_idle()   # kernels compiled, short buckets captured
    for n, recs in records.items():
        try:
            torch.cuda.reset_peak_memory_stats()
            new = [server.probs(materialize(r)) for r in recs]
            cached = [server.probs(materialize(recs[0])) for _ in range(reps)]
            row = {"tokens": new[0][1]["tokens"], "state_tokens": new[0][1]["state_tokens"], "first_ms": new[0][1]["latency_ms"],
                   "new_ms": statistics.median(s["latency_ms"] for _, s in new[1:]), "cached_ms": statistics.median(s["latency_ms"] for _, s in cached),
                   "cache_hits": sum(s["prefix_cache_hit"] for _, s in cached), "peak_allocated_gb": round(allocated_bytes("cuda") / 1e9, 1)}
            if n in reference: row["served_vs_benchmark"] = agreement([(torch.tensor(p), q) for p, q in zip(new[0][0], reference[n])])
        except Exception as error:
            row = {"error": f"{type(error).__name__}: {str(error)[:500]}"}
        report["lengths"][n]["served"] = row; print(n, row, flush=True); save()
    report["graphs"] = m.graphs.stats()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--suite", default="evals/v7/decision-v7"); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reference", default="fp32", choices=["fp32", "none"]); ap.add_argument("--out", required=True)
    ap.add_argument("--isolation", action="store_true", help="also measure question isolation in the served precision")
    ap.add_argument("--state_tokens", default="", help="comma-separated state lengths: only the long-state latency and memory read (long_states), --reps records each")
    a = ap.parse_args()
    if a.state_tokens:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        report = long_states(Checkpoint(a.run), [int(x) for x in a.state_tokens.split(",")], a.reps, a.out)
        print(json.dumps(report, indent=1)); write_json(Path(a.out) / "report.json", report)
        return
    raw = [r for r in load_split(a.suite, "development") if r["_meta"]["variant"] == "clean"][: a.n]
    recs = [materialize(r) for r in raw]
    ck = Checkpoint(a.run)
    report = {"run": a.run, "gpu": torch.cuda.get_device_name(0), "records": len(recs)}
    targets = None
    if a.reference == "fp32":   # fp32 as the evaluation path runs (kev.predictors.LocalPredictor): no TF32, no fused SDPA; probs() is the prefix form, within fp32 rounding of its rows
        tok, ref = ck.load("cuda")
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        targets = [ref.probs(ref.encode(tok, r)) for r in recs]
        torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True)   # serving keeps them
        del ref; gc.collect(); empty_cache("cuda")
    t = time.time()
    tok, m = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
    report["load_seconds"] = round(time.time() - t, 1)
    report["resident_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)   # weights + graph buffers
    graphs = m.graphs
    server = Server(ck, tok, m, "cuda", release_date="-")
    report["latency_graphs"] = latency(server, a.reps)   # first, so first_ms includes the captures
    served = {"eager": [], "graphs_miss": [], "graphs_hit": []}
    with server.lock:   # the model directly: keep the server's model thread (and its idle captures) out
        for r in recs:
            enc = m.encode(tok, r)
            m.graphs = None; served["eager"].append(m.probs_with_prefix(enc, m.prefix(enc)))
            m.graphs = graphs; m.probs_batch([enc], [None], [False]); graphs.capture_pending()   # so both reads below replay graphs
            (miss,), (prefix,) = m.probs_batch([enc], [None], [True])                         # the served path, a new state
            (hit,), _ = m.probs_batch([enc], [prefix], [False])                               # and a cached one
            served["graphs_miss"].append(miss); served["graphs_hit"].append(hit)
    report["questions"] = sum(len(p) for p in served["eager"])
    pairs = {f"{k}_vs_eager_bf16": (v, served["eager"]) for k, v in served.items() if k != "eager"}
    if targets: pairs.update({f"{k}_vs_fp32": (v, targets) for k, v in served.items()})
    for name, (got, ref) in pairs.items():
        report[name] = agreement([(p, q) for ps, qs in zip(got, ref) for p, q in zip(ps, qs)])
    if a.isolation:
        with server.lock: report["isolation_served"] = isolation(m, tok, raw)
    report["throughput"] = throughput(server, a.suite)
    report["graphs"] = graphs.stats()
    with server.lock: m.graphs = None; server.prefix_cache.clear()
    report["latency_eager"] = latency(server, a.reps)
    print(json.dumps(report, indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out) / "report.json", report)


if __name__ == "__main__":
    main()
