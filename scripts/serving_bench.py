"""Serving latency and parity on CUDA: the kev.serve path with and without CUDA graphs (kev.cuda_graphs).

    uv run modal run modal_app.py::serving --run jaredpalmer/kev-4b --gpu A100-80GB --name serving-4b-a100
    python scripts/serving_bench.py --run jaredpalmer/kev-4b --out runs/serving-4b      # on a CUDA machine

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
from kev.device import empty_cache
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
    buckets run eagerly and are captured in the background afterwards; it also pays any first-call kernel autotuning."""
    out = {}
    for case in CASES:
        first = server.answer(request(case, 0))
        row = {"tokens": first["usage"]["input_tokens"], "first_ms": first["latency_ms"]}
        for mode in ("new", "cached"):
            ms = [server.answer(request(case, i if mode == "new" else 0))["latency_ms"] for i in range(1, reps + 3)][2:]
            row[f"{mode}_ms"] = statistics.median(ms)
        out[case] = row
    return out


def agreement(pairs):
    """[(p, q), ...] probability vectors of the same question -> max and mean of max |p - q|, and argmax flips."""
    dp = [float((p - q).abs().max()) for p, q in pairs]
    return {"max_dp": max(dp), "mean_dp": statistics.mean(dp), "argmax_flips": sum(int(p.argmax() != q.argmax()) for p, q in pairs)}


def isolation(m, tok, raw):
    """Each question alone vs in the full request ("packed") and vs after ISOLATION_PROBE ("sibling"), served as
    configured (bf16, graphs if loaded). raw = labelled records, before materialize."""
    def serve(record):
        return m.probs_and_prefix(m.encode(tok, materialize(record)))[0]
    out = {"packed": [], "sibling": []}
    for r in raw:
        full = serve(r)
        for i, (qid, q) in enumerate(r["questions"].items()):
            alone = serve({**r, "questions": {qid: q}})[0]
            out["packed"].append((alone, full[i]))
            out["sibling"].append((alone, serve({**r, "questions": {"isolated_probe": ISOLATION_PROBE, qid: q}})[1]))
    return {k: {"questions": len(v), **agreement(v)} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-4b"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--suite", default="evals/v7/decision-v7"); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reference", default="fp32", choices=["fp32", "none"]); ap.add_argument("--out", required=True)
    ap.add_argument("--isolation", action="store_true", help="also measure question isolation in the served precision")
    a = ap.parse_args()
    raw = [r for r in load_split(a.suite, "development") if r["_meta"]["variant"] == "clean"][: a.n]
    recs = [materialize(r) for r in raw]
    ck = Checkpoint(a.run)
    report = {"run": a.run, "gpu": torch.cuda.get_device_name(0), "records": len(recs)}
    targets = None
    if a.reference == "fp32":   # the exact evaluation path (kev.predictors.LocalPredictor): no TF32, no fused SDPA
        tok, ref = ck.load("cuda")
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        targets = [ref.probs(ref.encode(tok, r)) for r in recs]
        torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True)   # serving keeps them
        del ref; gc.collect(); empty_cache("cuda")
    t = time.time()
    tok, m = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True))
    report["load_seconds"] = round(time.time() - t, 1)
    report["resident_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)   # weights + graph buffers
    graphs = m.graphs
    server = Server(ck, tok, m, "cuda", release_date="-")
    report["latency_graphs"] = latency(server, a.reps)   # first, so first_ms includes the captures
    served = {"eager": [], "graphs_miss": [], "graphs_hit": []}
    for r in recs:
        enc = m.encode(tok, r)
        m.graphs = None; served["eager"].append(m.probs_with_prefix(enc, m.prefix(enc)))
        m.graphs = graphs; m.probs_and_prefix(enc); graphs.capture_pending()   # so both reads below replay graphs
        p, prefix = m.probs_and_prefix(enc); served["graphs_miss"].append(p); served["graphs_hit"].append(m.probs_with_prefix(enc, prefix))
    report["questions"] = sum(len(p) for p in served["eager"])
    pairs = {f"{k}_vs_eager_bf16": (v, served["eager"]) for k, v in served.items() if k != "eager"}
    if targets: pairs.update({f"{k}_vs_fp32": (v, targets) for k, v in served.items()})
    for name, (got, ref) in pairs.items():
        report[name] = agreement([(p, q) for ps, qs in zip(got, ref) for p, q in zip(ps, qs)])
    if a.isolation:
        report["isolation_served"] = isolation(m, tok, raw)
    report["graphs_captured"] = graphs.captures
    m.graphs = None; server.prefix_cache.clear()
    report["latency_eager"] = latency(server, a.reps)
    print(json.dumps(report, indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out) / "report.json", report)


if __name__ == "__main__":
    main()
