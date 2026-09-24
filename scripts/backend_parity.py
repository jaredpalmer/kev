"""Serving-backend parity and latency against the fp32 torch path, on this machine.

    uv run --extra mlx python scripts/backend_parity.py --run jaredpalmer/kev-0.8b --n 60
    uv run --extra mlx python scripts/backend_parity.py --run jaredpalmer/kev-4b --n 40 --out runs/mlx-parity-4b/report.json
    # ExecuTorch (its own environment, AGENTS.md): a program exported by pytorch/executorch examples/kev
    python scripts/backend_parity.py --backend executorch --program kev-cpu/model.pte --native_tokenizer --n 0 \
        --reference /tmp/reference-0.8b.pt --out runs/et-parity-0.8b-xnnpack-fp32/report.json

For n clean development records of a suite (decision-v7 by default; 0 = all of them), encoded within the suite's
admission context as kev.benchmark does: probabilities from the backend (the prefix path serving runs, each question
alone, and on MLX the row form) vs the torch fp32 path (max |dp|, argmax flips, calibrated), and the median latency of
each path. Records longer than an ExecuTorch program's exported shapes are counted as rejected, not scored.
`--native_tokenizer` (executorch) also scores every record with the C++ tokenizer the example's native runner links, and
counts the questions whose tokens differ from the Hugging Face tokenizer's. `--reference` caches the fp32 probabilities
(computing them for a full partition takes a while) so several programs can be compared against one reference. Writes a JSON report (runs/<name>/report.json is kept by .gitignore;
runs/r4-mlx-parity-{4b,0.8b} and runs/et-parity-* hold the numbers the README and PLAN.md quote).
"""
import argparse, gc, json, os, statistics, time
from pathlib import Path
from types import SimpleNamespace

import torch

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import empty_cache
from kev.model import ContextOverflow, load_tokenizer, rows_of
from kev.suite import CONTEXT, load_split, read_manifest, write_json

# the request pytorch/executorch examples/kev/main.cpp bundles: a 19-token state, three questions asked in two calls
SUPPORT = {"state": "I was charged twice for invoice 4411. Please refund the duplicate charge.", "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this ticket?",
                   "criteria": {"billing": "Payments, invoices, and refunds", "technical": "Bugs, outages, and integrations", "sales": "Pricing, upgrades, and new accounts"}},
    "refund_requested": {"type": "choice", "instructions": "Does the customer explicitly ask for a refund?", "criteria": {"no": None, "yes": None}},
    "duplicate_charge": {"type": "choice", "instructions": "Was the customer charged more than once?", "criteria": {"no": None, "yes": None}}}}


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1)


def support_calls():
    """The bundled request as the two calls main.cpp makes on one prefix: two questions, then one."""
    names = list(SUPPORT["questions"])
    return [to_record(SystemOneRequest.model_validate({"state": SUPPORT["state"], "questions": {k: SUPPORT["questions"][k] for k in part}}))[0]
            for part in (names[:2], names[2:])]


def document_calls(tok, recs):
    """One call of five three-option questions on the first development state of 256+ tokens (the README's MLX shape)."""
    state = next(r["state"] for r in recs if len(tok(r["state"]).input_ids) >= 256)
    questions = [{"instr": f"{q} Answer from the text.", "options": ["yes", "no", "the text does not say"], "label": 0}
                 for q in ("Is a person named?", "Is a date given?", "Is a place named?", "Is a number given?", "Is the tone neutral?")]
    return [{"state": state, "questions": questions}]


def request_latency(model, tok, calls, context, reps=5):
    """Median ms over `reps` runs (after two warm-ups) of: prefill, the calls on the cached state, and the whole request."""
    encs = [model.encode(tok, rec, **context) for rec in calls]
    runs = []
    for i in range(2 + reps):
        t0 = time.perf_counter(); prefix = model.prefix(encs[0])
        t1 = time.perf_counter()
        for enc in encs: model.probs_with_prefix(enc, prefix)
        t2 = time.perf_counter()
        if i >= 2: runs.append(((t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000))
    return {"state_tokens": encs[0]["seg"].count(0), "questions": sum(len(e["decide_idx"]) for e in encs),
            **{name: round(statistics.median(r[j] for r in runs), 1) for j, name in enumerate(("prefill", "cached_evaluation", "total"))}}


def reference(ck, tok, recs, context, cache):
    """fp32 torch probabilities for every record plus its latency, with the model released on return: a 4B in fp32 and its
    MLX twin do not fit a 32 GB Mac at once. With `cache`, read from / written to that file."""
    if cache and os.path.exists(cache):
        saved = torch.load(cache)
        if saved["run"] == ck.requested and saved.get("context", context) == context and len(saved["targets"]) >= len(recs):
            return saved["targets"][: len(recs)], saved["latency_ms"], saved["dtype"]
    _, ref = ck.load("mps", LoadOptions(backend="torch"))
    targets = [ref.probs(ref.encode(tok, rec, strict=True, **context)) for rec in recs]
    enc = ref.encode(tok, recs[0], strict=True, **context); _, prefix = ref.probs_and_prefix(enc)
    ms = {"torch_fp32_full": timed(lambda: ref.probs(enc), 5), "torch_fp32_prefix_hit": timed(lambda: ref.probs_with_prefix(enc, prefix), 5)}
    if cache: torch.save({"run": ck.requested, "context": context, "targets": targets, "latency_ms": ms, "dtype": ref.dtype}, cache)
    return targets, ms, ref.dtype


class NativeTokenizer:
    """The C++ HFTokenizer (pytorch-tokenizers, what the ExecuTorch runner links) behind the two calls kev.model.encode
    makes, so a record encodes exactly as kev.cpp encodes it. Delimiter ids come from the reference tokenizer."""

    def __init__(self, tokenizer_json, tok):
        from pytorch_tokenizers import CppHFTokenizer
        self.cpp, self.tok = CppHFTokenizer(), tok
        self.cpp.load(str(tokenizer_json))

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(self.cpp.encode(text, 0, 0)))

    def convert_tokens_to_ids(self, token):
        return self.tok.convert_tokens_to_ids(token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--suite", default="evals/v7/decision-v7"); ap.add_argument("--out", default="")
    ap.add_argument("--backend", choices=("mlx", "executorch"), default="mlx")
    ap.add_argument("--program", default="", help="executorch: the model.pte exported from --run")
    ap.add_argument("--native_tokenizer", action="store_true", help="executorch: also score with the C++ tokenizer (tokenizer.json next to --program)")
    ap.add_argument("--reference", default="", help="cache file for the fp32 torch probabilities")
    a = ap.parse_args()
    clean = [materialize(r) for r in load_split(a.suite, "development") if r["_meta"]["variant"] == "clean"]
    recs = clean[: a.n] if a.n else clean
    ctx = read_manifest(a.suite).get("context", CONTEXT)   # the suite's admission context, as kev.benchmark encodes it
    context = {"max_state": ctx["max_state"], "max_branch": ctx["max_branch"]}
    ck = Checkpoint(a.run)
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    targets, torch_ms, torch_dtype = reference(ck, tok, recs, context, a.reference)
    gc.collect(); empty_cache("mps")
    _, model = ck.load("mps", LoadOptions(backend=a.backend, program=a.program or None))
    report = {"run": a.run, "suite": a.suite, "context": context, "backend": a.backend, "program": a.program or None,
              "records": len(recs), "questions": 0, "rejected_records": 0, "torch_dtype": torch_dtype, f"{a.backend}_dtype": model.dtype}
    paths = (["rows"] if hasattr(model, "forward_rows") else []) + ["prefix", "alone"]   # rows = the torch row form (MLX); prefix = what serving runs
    native = NativeTokenizer(Path(a.program).parent / "tokenizer.json", tok) if a.native_tokenizer else None
    if native: paths.append("native_tokenizer"); report["native_token_mismatch_questions"] = 0
    dp, flips = {p: [] for p in paths}, {p: 0 for p in paths}
    for rec, target in zip(recs, targets):
        try: enc = model.encode(tok, rec, **context)
        except ContextOverflow:   # longer than the program's exported shapes: refused, as kev.serve would
            report["rejected_records"] += 1; continue
        got = {"prefix": model.probs(enc),
               "alone": [model.probs(model.encode(tok, {"state": rec["state"], "questions": [q]}, **context))[0] for q in rec["questions"]]}
        if "rows" in paths: got["rows"] = [torch.softmax(z, -1) for z in model.forward_rows(enc)]
        if native:
            nenc = model.encode(native, rec, **context); got["native_tokenizer"] = model.probs(nenc)
            (S, _, rows), (nS, _, nrows) = rows_of(enc), rows_of(nenc)
            report["native_token_mismatch_questions"] += sum(S != nS or r["ids"] != n["ids"] for r, n in zip(rows, nrows))
        for name in paths:
            for p, t in zip(got[name], target):
                dp[name].append(float((p - t).abs().max())); flips[name] += int(p.argmax() != t.argmax())
        report["questions"] += len(target)
    for name in paths:
        report[name] = {"max_dp": max(dp[name]), "mean_dp": statistics.mean(dp[name]), "argmax_flips": flips[name], "over_0.02": sum(d > 0.02 for d in dp[name])}
    enc = model.encode(tok, recs[0], **context); _, prefix = model.probs_and_prefix(enc)
    ms = {"miss": timed(lambda: model.probs(enc), 10), "prefix_hit": timed(lambda: model.probs_with_prefix(enc, prefix), 10)}
    if "rows" in paths: ms["rows"] = timed(lambda: model.forward_rows(enc), 10)
    report["latency_ms"] = {"tokens": len(enc["ids"]), "questions_in_record": len(enc["decide_idx"]),
                            **{f"{a.backend}_{k}": v for k, v in ms.items()}, **torch_ms,
                            "support_request": request_latency(model, tok, support_calls(), context),
                            "document_request": request_latency(model, tok, document_calls(tok, clean), context)}
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)


if __name__ == "__main__":
    main()
