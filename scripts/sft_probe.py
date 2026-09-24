"""Full-weight training probe: peak memory, seconds per optimizer step and throughput of `kev.train --full_ft 1` on
realistic record lengths, the wall time and cost that implies for 50k / 100k / 200k records, and a loader check of the
checkpoint it writes (bf16 and fp32 eval memory, their parity).

    KEV_GPU=H200 uv run modal run modal_app.py::sft_probe --name sft-probe-27b-h200 --gpu H200 --train "--batch 1 --accum 32 ..."
    python scripts/sft_probe.py --base Qwen/Qwen3.8-27B --revision <sha> --gpu H200 --out runs/x --train "..."   # on a GPU box

Data: decision-v7 training records whose states are lengthened, by appending other training records' states, to a
length drawn per record (60 % as they are, 30 % 1.2-1.8k tokens, 10 % 3k tokens up to kev.model.MAX_TRAIN_STATE), so a
record averages about 1,000 tokens with a tail at the longest state Kev trains on. Record tokens are the packed encoding
(the state once); forward tokens are what the row form runs (the state once per question, kev.model.rows_of).
Throughput is measured after the first `--warmup` steps (Triton autotuning, allocator growth). Writes report.json.
"""
import argparse, copy, os, random, shlex, statistics, subprocess, sys, threading, time
from pathlib import Path

import torch

from kev.api import render
from kev.budget import GPU_HOURLY, compute_bound, gpu_count
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import allocated_bytes, empty_cache
from kev.model import MAX_TRAIN_STATE, encode, load_tokenizer, rows_of, training_context
from kev.suite import load_split, read_json, write_json, write_jsonl

SUITE = Path(__file__).resolve().parents[1] / "evals/v7/decision-v7"
CORPUS = (50_000, 100_000, 200_000)   # records of ~1,000 tokens each: the SFT corpus sizes being planned


def build(tok, n, seed):
    """-> (labelled requests, per-record (record tokens, forward tokens, questions))."""
    pool, rng, context = load_split(SUITE, "train"), random.Random(seed), training_context(MAX_TRAIN_STATE)
    count = lambda text: len(tok(text, add_special_tokens=False).input_ids)
    out, stats = [], []
    for i in rng.sample(range(len(pool)), n):
        r = copy.deepcopy(pool[i]); u = rng.random()
        target = 0 if u < 0.6 else rng.randint(1200, 1800) if u < 0.9 else rng.randint(3000, MAX_TRAIN_STATE - 300)
        if target:
            state = render(r["state"])
            while count(state) < target: state += "\n\n" + render(pool[rng.randrange(len(pool))]["state"])
            r["state"] = tok.decode(tok(state, add_special_tokens=False).input_ids[:target])
        try: enc = encode(tok, materialize(r), max_state=context["max_state"], max_branch=context["max_branch"], strict=True)
        except ValueError: continue
        S, _, rows = rows_of(enc)
        out.append(r); stats.append((len(enc["ids"]), sum(len(S) + len(row["ids"]) for row in rows), len(rows)))
    return out, stats


class GpuMemory(threading.Thread):
    """Peak GPU memory in use as nvidia-smi reports it (every process, allocator cache and CUDA contexts included)."""
    def __init__(self):
        super().__init__(daemon=True); self.peak, self.running = 0, True
    def run(self):
        while self.running:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            self.peak = max([self.peak, *(int(x) for x in out.split())]); time.sleep(2)


def load_check(run, recs, n):
    """Eval memory and parity of the saved checkpoint through kev.checkpoint: bf16 (how full weights load) against fp32."""
    tok, probs, report = None, {}, {}
    for name, dtype in (("bf16", None), ("fp32", torch.float32)):
        torch.cuda.reset_peak_memory_stats(); started = time.time()
        tok, model = Checkpoint(run).load("cuda", LoadOptions(dtype=dtype, temperature=1.0))
        load_seconds = time.time() - started
        with torch.no_grad():
            probs[name] = [p for r in recs[:n] for p in model.probs(model.encode(tok, materialize(r), max_state=MAX_TRAIN_STATE, max_branch=training_context(MAX_TRAIN_STATE)["max_branch"]))]
        report[name] = {"dtype": model.dtype, "load_seconds": round(load_seconds, 1), "peak_allocated_gb": round(allocated_bytes("cuda") / 1e9, 1)}
        del model; empty_cache("cuda")
    delta = [float((a - b).abs().max()) for a, b in zip(probs["bf16"], probs["fp32"])]
    flips = sum(int(a.argmax() != b.argmax()) for a, b in zip(probs["bf16"], probs["fp32"]))
    return {**report, "questions": len(delta), "max_abs_dp": max(delta), "mean_abs_dp": statistics.mean(delta), "argmax_flips": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", required=True, help="the Modal GPU spec this runs on (H200, H200:8): prices the projections")
    ap.add_argument("--records", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default="", help="extra kev.train arguments (batch, accum, max_steps, lr, row_budget, ...)")
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--check_load", type=int, default=0, help="questions for the bf16/fp32 loader check (0 = skip)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True)
    tok = load_tokenizer(a.base, revision=a.revision)
    recs, stats = build(tok, a.records, a.seed)
    write_jsonl(out / "probe.jsonl", recs)
    record_tokens, forward_tokens = statistics.mean(s[0] for s in stats), statistics.mean(s[1] for s in stats)
    data = {"records": len(recs), "record_tokens_mean": record_tokens, "record_tokens_p50": statistics.median(s[0] for s in stats),
            "record_tokens_max": max(s[0] for s in stats), "forward_tokens_mean": forward_tokens, "questions_mean": statistics.mean(s[2] for s in stats)}
    print(data, flush=True)

    gpus = gpu_count(a.gpu)
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else [sys.executable]
    cmd = [*launcher, "-m", "kev.train", "--base", a.base, "--base_revision", a.revision, "--data", str(out / "probe.jsonl"),
           "--full_ft", "1", "--weights_dtype", "bf16", "--dtype", "bf16", "--checkpointing", "1", "--device", "cuda",
           "--max_state", str(MAX_TRAIN_STATE), "--out", str(out / "checkpoint"), *shlex.split(a.train)]
    memory = GpuMemory(); memory.start(); started = time.time()
    with (out / "train.log").open("w", encoding="utf-8") as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:   # the whole log to train.log, progress to the container log
            log.write(line)
            if line.startswith(("ep", "device", "saved")) or "Error" in line: print(line.rstrip(), flush=True)
    code = proc.returncode
    memory.running = False
    if code: raise SystemExit(f"kev.train failed ({code}); see {out / 'train.log'}")
    metrics = read_json(out / "checkpoint/training_metrics.json")
    report = {"gpu": a.gpu, "base": a.base, "revision": a.revision, "train_args": a.train, "data": data, "warmup": a.warmup,
              "fixed_overhead_seconds": round((time.time() - started) - sum(metrics["step_seconds"])),   # load, optimizer setup, save
              "peak_nvidia_smi_gb": round(memory.peak / 1024, 1), "training_metrics": metrics}
    report.update(throughput(report))
    if a.check_load: report["load_check"] = load_check(str(out / "checkpoint"), recs, a.check_load)
    write_json(out / "report.json", report)
    print({k: v for k, v in report.items() if k != "training_metrics"}, flush=True)


def throughput(report):
    """The derived numbers of a report: the mean steady step (after `warmup` steps; the mean, not the median, because the
    slow steps are the long records a real run also meets), rates, memory, and the projections: n records of 1,000 record
    tokens at the measured record-token rate (so at the probe's questions per record) plus the fixed overhead."""
    metrics, data, gpu = report["training_metrics"], report["data"], report["gpu"]
    per_step = statistics.mean(metrics["step_seconds"][report["warmup"]:] or metrics["step_seconds"])
    records_per_step = metrics["records_seen"] / metrics["optimizer_steps"]
    rate, hourly = records_per_step / per_step, compute_bound(gpu, 3600, 1, full_ft=True)
    seconds = lambda n: n * 1000 / (rate * data["record_tokens_mean"]) + report["fixed_overhead_seconds"]
    return {"seconds_per_step": per_step, "records_per_step": records_per_step, "records_per_second": rate,
            "record_tokens_per_second": rate * data["record_tokens_mean"], "forward_tokens_per_second": rate * data["forward_tokens_mean"],
            "optimizer_seconds_per_step": metrics["optimizer_seconds"] / metrics["optimizer_steps"],
            "peak_allocated_gb_rank0": round(metrics["peak_device_bytes"] / 1e9, 1), "peak_host_rss_gb_rank0": round(metrics["peak_rss_bytes"] / 1e9, 1),
            "usd_per_hour": round(hourly, 2), "gpu_usd_per_hour": GPU_HOURLY[gpu.partition(":")[0]] * gpu_count(gpu),
            "projections_1000_token_records": {n: {"hours": round(seconds(n) / 3600, 2), "usd": round(seconds(n) / 3600 * hourly, 2)} for n in CORPUS}}


if __name__ == "__main__":
    main()
