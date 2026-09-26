"""Full-weight training probe: peak memory, seconds per optimizer step and throughput of `kev.train --full_ft 1` on
records shaped like the SFT corpus, the wall time and cost that implies for one and two epochs of it, the time a resume
point takes to write, and a loader check of the checkpoint it writes (bf16 and fp32 eval memory, their parity).

    uv run modal run modal_app.py::sft_probe --name sft-probe-27b-8xh200-all --gpu H200:8 --mix all --train "--batch 4 --accum 4 --length_sort 1 ..."
    python scripts/sft_probe.py --base Qwen/Qwen3.8-27B --revision <sha> --gpu H200 --out runs/x --mix synthetic --train "..."   # on a GPU box
    uv run modal run modal_app.py::sft_probe --name lc-27b-8xh200 --gpu H200:8 --records 40 --train "--batch 1 --accum 1 --length_sort 1 --max_steps 5" --flags "--state_tokens 16384,32768,49152,65536 --questions 3 --warmup 2"   # long states: one run per length (long_states)

Data: `experiments/sft-v1-lengths.json` holds token shapes (state tokens, branch tokens of each question) sampled from
the three parts of the private SFT corpus, counts only. A probe record draws a shape (from one part with `--mix <part>`,
or from all three in the corpus's proportions with `--mix all`), fills its state with decision-v7 training text to that
length and takes, for each question, the decision-v7 question whose branch is nearest in length. Record tokens are the
packed encoding, the state once, which is also what a shared prefix runs (kev.shared_prefix); row tokens are what the
row form runs (the state once per question). Throughput is the mean step after `--warmup` steps.
`--no_conv_kernel` hides causal-conv1d from the trainer (transformers then runs its PyTorch convolution), for an A/B.
Writes report.json.
"""
import argparse, bisect, os, random, shlex, shutil, statistics, subprocess, sys, threading, time
from pathlib import Path

import torch

from kev.api import render
from kev.budget import GPU_HOURLY, gpu_count, hourly_rate
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.device import allocated_bytes, empty_cache
from kev.model import MAX_TRAIN_STATE, encode, load_tokenizer, rows_of, training_context, user_tokens
from kev.suite import ADMISSION_BRANCH_HEADROOM, load_split, read_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SUITE, PROFILE = ROOT / "evals/v7/decision-v7", ROOT / "experiments/sft-v1-lengths.json"


def build(tok, n, seed, mix):
    """-> (labelled requests, per record (record tokens, row tokens, questions), records in the corpus part(s)), shaped like `mix`."""
    profile, rng, context = read_json(PROFILE), random.Random(seed), training_context(MAX_TRAIN_STATE)
    parts = list(profile["parts"]) if mix == "all" else [mix]
    pool = load_split(SUITE, "train")
    shape_of = lambda r: rows_of(encode(tok, materialize(r), max_state=context["max_state"], max_branch=context["max_branch"]))
    questions = sorted(((len(shape_of({"state": "", "questions": {"q": q}})[2][0]["ids"]), i, j), q) for i, r in enumerate(pool) for j, q in enumerate(r["questions"].values()))
    lengths = [key[0] for key, _ in questions]

    def nearest(b):   # a question of the available length closest to b, at random among those of that length
        at = bisect.bisect_left(lengths, b)
        length = min(lengths[max(at - 1, 0):at + 1], key=lambda n: abs(n - b))
        return questions[rng.randrange(bisect.bisect_left(lengths, length), bisect.bisect_right(lengths, length))][1]
    text = "\n\n".join(render(r["state"]) for r in rng.sample(pool, 400))
    words = tok(text, add_special_tokens=False).input_ids
    out, stats = [], []
    for i in range(n):
        part = rng.choices(parts, weights=[profile["records"][p] for p in parts])[0]
        state, branches = rng.choice(profile["parts"][part]["shapes"])
        start = rng.randrange(len(words) - state)
        picks = [nearest(b) for b in branches]
        r = {"state": tok.decode(words[start:start + max(state - 1, 1)]), "questions": {f"q{j}": q for j, q in enumerate(picks)},
             "_meta": {"source": "probe", "id": f"probe/{i}", "part": part}}
        S, _, rows = shape_of(r)
        out.append(r); stats.append((len(S) + sum(len(x["ids"]) for x in rows), sum(len(S) + len(x["ids"]) for x in rows), len(rows)))
    return out, stats, sum(profile["records"][p] for p in parts)


def build_long(tok, n, state_tokens, questions, seed):
    """--state_tokens: n records whose state is exactly `state_tokens` tokens (the <state> delimiter included) of
    decision-v7 training text, each with `questions` decision-v7 questions drawn at random. -> (requests, stats as build's)."""
    rng, context = random.Random(f"{seed}:{state_tokens}"), training_context(state_tokens)
    pool = load_split(SUITE, "train")
    shape_of = lambda r: rows_of(encode(tok, materialize(r), max_state=context["max_state"], max_branch=context["max_branch"], strict=True))
    order = rng.sample(pool, len(pool))
    words, used = [], 0
    while len(words) < 2 * state_tokens:   # a text twice the state, so records start at different places
        words += tok("\n\n".join(render(r["state"]) for r in order[used:used + 500]) + "\n\n", add_special_tokens=False).input_ids; used += 500
    out, stats = [], []
    for i in range(n):
        start, take = rng.randrange(len(words) - state_tokens), state_tokens - 1
        for _ in range(8):   # decoding a token slice and encoding it again can merge or split a few tokens at the seams
            text = tok.decode(words[start:start + take])
            have = len(user_tokens(tok, text))
            if have == state_tokens - 1: break
            take += state_tokens - 1 - have
        else:
            raise ValueError(f"could not cut a state of exactly {state_tokens} tokens")
        while True:   # questions are drawn again until every branch fits the training context with the admission headroom
            picks = [rng.choice(list(r["questions"].values())) for r in rng.sample(pool, questions)]   # (augmentation may add an option)
            r = {"state": text, "questions": {f"q{j}": q for j, q in enumerate(picks)}, "_meta": {"source": "probe", "id": f"probe/{state_tokens}/{i}"}}
            try: S, _, rows = shape_of(r)
            except ValueError: continue
            if len(S) + max(len(x["ids"]) for x in rows) <= context["max_branch"] - ADMISSION_BRANCH_HEADROOM: break
        assert len(S) == state_tokens
        out.append(r); stats.append((len(S) + sum(len(x["ids"]) for x in rows), sum(len(S) + len(x["ids"]) for x in rows), len(rows)))
    return out, stats


def host_memory_bytes():
    """Memory the container uses (cgroup v2, then v1; the machine's used memory where neither exists)."""
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        if os.path.exists(path): return int(Path(path).read_text(encoding="utf-8"))
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines())
    return (int(info["MemTotal"].split()[0]) - int(info["MemAvailable"].split()[0])) * 1024


class GpuMemory(threading.Thread):
    """Peak GPU memory in use as nvidia-smi reports it (every process, allocator cache and CUDA contexts included), overall
    and per GPU, and the container's peak host memory."""
    def __init__(self):
        super().__init__(daemon=True); self.peak, self.per_gpu, self.host, self.running = 0, [], 0, True
    def run(self):
        while self.running:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            used = [int(x) for x in out.split()]
            self.per_gpu = [max(a, b) for a, b in zip(self.per_gpu, used)] if self.per_gpu else used
            self.peak = max([self.peak, *used]); self.host = max(self.host, host_memory_bytes()); time.sleep(2)


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


def train_command(a, out, data, max_state, extra):
    """The kev.train command line of one probe run (torchrun over every GPU of a multi-GPU spec)."""
    gpus = gpu_count(a.gpu)
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else [sys.executable]
    return [*launcher, "-m", "kev.train", "--base", a.base, "--base_revision", a.revision, "--data", str(data),
            "--full_ft", "1", "--weights_dtype", "bf16", "--dtype", "bf16", "--checkpointing", "1", "--device", "cuda",
            "--max_state", str(max_state), "--out", str(out), *shlex.split(extra)]


def run_training(cmd, log_path, env):
    """Run kev.train, the whole log to log_path and progress to the container log. -> (return code, GpuMemory, seconds)."""
    memory = GpuMemory(); memory.start(); started = time.time()
    with log_path.open("w", encoding="utf-8") as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as proc:
        for line in proc.stdout:
            log.write(line); log.flush()
            if line.startswith(("ep", "device", "saved")) or "Error" in line: print(line.rstrip(), flush=True)
    memory.running = False; memory.join()
    return proc.returncode, memory, time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--gpu", required=True, help="the Modal GPU spec this runs on (H200, H200:8): prices the projections")
    ap.add_argument("--mix", default="all", help="a part of experiments/sft-v1-lengths.json, or all (the corpus's proportions)")
    ap.add_argument("--records", type=int, default=2000, help="probe records (with --state_tokens: per length)"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default="", help="extra kev.train arguments (batch, accum, max_steps, lr, row_budget, shared_prefix, save_every_steps, ...)")
    ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--check_load", type=int, default=0, help="questions for the bf16/fp32 loader check (0 = skip)")
    ap.add_argument("--resume_dir", default="", help="write the resume points here (e.g. on the runs volume, to time them) instead of the scratch checkpoint")
    ap.add_argument("--no_conv_kernel", action="store_true")
    ap.add_argument("--state_tokens", default="", help="comma-separated state lengths in tokens: instead of the corpus mix, one run per length on records "
                                                       "of exactly that state (--questions each), --max_state set to it; a run that fails is reported, not fatal")
    ap.add_argument("--questions", type=int, default=3, help="questions per record with --state_tokens")
    ap.add_argument("--fallbacks", default="", help="with --state_tokens: ';'-separated extra kev.train arguments tried in order at a length whose run failed")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True)
    tok = load_tokenizer(a.base, revision=a.revision)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if a.no_conv_kernel:   # an importable stub that raises makes transformers fall back, as without the package
        (out / "stub/causal_conv1d").mkdir(parents=True); (out / "stub/causal_conv1d/__init__.py").write_text("raise ImportError('hidden by sft_probe')\n", encoding="utf-8")
        env["PYTHONPATH"] = f"{out / 'stub'}:{env.get('PYTHONPATH', '')}"
    if a.state_tokens: return long_states(a, out, tok, env)
    recs, stats, corpus = build(tok, a.records, a.seed, a.mix)
    write_jsonl(out / "probe.jsonl", recs)
    mean = lambda i: statistics.mean(s[i] for s in stats)
    data = {"mix": a.mix, "records": len(recs), "record_tokens_mean": mean(0), "record_tokens_max": max(s[0] for s in stats), "row_tokens_mean": mean(1),
            "questions_mean": mean(2), "corpus_records": corpus}
    print(data, flush=True)
    if a.resume_dir:   # the trainer writes <out>/resume; point it at the requested directory
        (out / "checkpoint").mkdir(); (out / "checkpoint/resume").symlink_to(a.resume_dir, target_is_directory=True)
    cmd = train_command(a, out / "checkpoint", out / "probe.jsonl", MAX_TRAIN_STATE, ("--resume 1 " if a.resume_dir else "") + a.train)
    code, memory, seconds = run_training(cmd, out / "train.log", env)
    if code: raise SystemExit(f"kev.train failed ({code}); see {out / 'train.log'}")
    metrics = read_json(out / "checkpoint/training_metrics.json")
    report = {"gpu": a.gpu, "base": a.base, "revision": a.revision, "train_args": a.train, "conv_kernel": not a.no_conv_kernel, "data": data, "warmup": a.warmup,
              "fixed_overhead_seconds": round(seconds - sum(metrics["step_seconds"]) - sum(metrics["resume_seconds"])),   # load, optimizer setup, save
              "peak_nvidia_smi_gb": round(memory.peak / 1024, 1), "training_metrics": metrics}
    report.update(throughput(report))
    if a.check_load: report["load_check"] = load_check(str(out / "checkpoint"), recs, a.check_load)
    write_json(out / "report.json", report)
    print({k: v for k, v in report.items() if k != "training_metrics"}, flush=True)


def long_states(a, out, tok, env):
    """--state_tokens: per length, a short run of `--records` records with states of exactly that length (resume points are
    not written: --resume_dir is ignored). A run that fails (out of memory, say) is retried with each of --fallbacks in
    turn. Per length: the attempts, and for the one that ran, peak GPU memory (nvidia-smi, per GPU; the allocator's peak
    on rank 0), the container's peak host memory, the mean steady step, seconds per record and tokens per second."""
    report = {"gpu": a.gpu, "base": a.base, "revision": a.revision, "train_args": a.train, "fallbacks": a.fallbacks, "questions": a.questions,
              "warmup": a.warmup, "usd_per_hour": round(hourly_rate(a.gpu, full_ft=True), 2), "lengths": []}
    for length in (int(x) for x in a.state_tokens.split(",")):
        recs, stats = build_long(tok, a.records, length, a.questions, a.seed)
        write_jsonl(out / f"probe-{length}.jsonl", recs)
        data = {"state_tokens": length, "records": len(recs), "record_tokens_mean": statistics.mean(s[0] for s in stats),
                "row_tokens_mean": statistics.mean(s[1] for s in stats), "questions_mean": statistics.mean(s[2] for s in stats)}
        print(data, flush=True)
        entry = {**data, "status": "failed", "attempts": []}
        for i, extra in enumerate(["", *(f for f in a.fallbacks.split(";") if f.strip())]):
            ckpt, log = out / f"checkpoint-{length}-{i}", out / f"train-{length}-{i}.log"
            code, memory, seconds = run_training(train_command(a, ckpt, out / f"probe-{length}.jsonl", length, f"{a.train} {extra}"), log, env)
            text = log.read_text(encoding="utf-8")
            attempt = {"extra": extra.strip(), "return_code": code, "seconds": round(seconds), "peak_nvidia_smi_gb": round(memory.peak / 1024, 1),
                       "peak_nvidia_smi_gb_per_gpu": [round(m / 1024, 1) for m in memory.per_gpu], "peak_host_gb": round(memory.host / 1e9, 1),
                       "out_of_memory": "OutOfMemoryError" in text or "out of memory" in text}
            if code:
                attempt["log_tail"] = text[-3000:]
            else:
                metrics = read_json(ckpt / "training_metrics.json")
                steady = metrics["step_seconds"][a.warmup:] or metrics["step_seconds"]
                per_step, per_step_records = statistics.mean(steady), metrics["records_seen"] / metrics["optimizer_steps"]
                attempt.update({"training_metrics": metrics, "fixed_overhead_seconds": round(seconds - sum(metrics["step_seconds"])),
                                "seconds_per_step": per_step, "records_per_step": per_step_records, "seconds_per_record": per_step / per_step_records,
                                "record_tokens_per_second": per_step_records * data["record_tokens_mean"] / per_step,
                                "row_tokens_per_second": per_step_records * data["row_tokens_mean"] / per_step,
                                "peak_allocated_gb_rank0": round(metrics["peak_device_bytes"] / 1e9, 1), "peak_host_rss_gb_rank0": round(metrics["peak_rss_bytes"] / 1e9, 1)})
            shutil.rmtree(ckpt, ignore_errors=True)   # a 27B's checkpoint is 51 GB of scratch disk
            entry["attempts"].append(attempt)
            print({k: v for k, v in attempt.items() if k not in ("training_metrics", "log_tail")}, flush=True)
            if not code: entry["status"] = "ok" if i == 0 else f"ok with {extra.strip()}"; break
        report["lengths"].append(entry)
        write_json(out / "report.json", report)   # after every length, so a later failure keeps the earlier ones


def throughput(report):
    """The derived numbers of a report: the mean steady step (after `warmup` steps; the mean, not the median, because the
    slow steps are the long records a real run also meets), rates, memory, and the projections for one and two epochs of
    the corpus part(s) the mix draws from (`corpus_records` records at the measured records/s, plus the fixed overhead)."""
    metrics, data, gpu = report["training_metrics"], report["data"], report["gpu"]
    per_step = statistics.mean(metrics["step_seconds"][report["warmup"]:] or metrics["step_seconds"])
    records_per_step = metrics["records_seen"] / metrics["optimizer_steps"]
    rate, hourly = records_per_step / per_step, hourly_rate(gpu, full_ft=True)
    hours = lambda epochs: (epochs * data["corpus_records"] / rate + report["fixed_overhead_seconds"]) / 3600
    return {"seconds_per_step": per_step, "records_per_step": records_per_step, "records_per_second": rate,
            "record_tokens_per_second": rate * data["record_tokens_mean"], "row_tokens_per_second": rate * data["row_tokens_mean"],
            "optimizer_seconds_per_step": metrics["optimizer_seconds"] / metrics["optimizer_steps"], "resume_blocking_seconds": metrics["resume_seconds"], "resume_write_seconds": metrics.get("resume_write_seconds", []),
            "peak_allocated_gb_rank0": round(metrics["peak_device_bytes"] / 1e9, 1), "peak_host_rss_gb_rank0": round(metrics["peak_rss_bytes"] / 1e9, 1),
            "usd_per_hour": round(hourly, 2), "gpu_usd_per_hour": GPU_HOURLY[gpu.partition(":")[0]] * gpu_count(gpu),
            "projections": {f"{e} epoch{'s' * (e > 1)}": {"records": e * data["corpus_records"], "hours": round(hours(e), 2), "usd": round(hours(e) * hourly, 2)} for e in (1, 2)}}


if __name__ == "__main__":
    main()
