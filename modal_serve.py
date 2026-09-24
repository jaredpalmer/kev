"""Serve Kev on Modal through vLLM (kev.vllm_model), and measure it against the torch serving path on the same GPU.

    uv run modal deploy modal_serve.py                                             # System One endpoint, vLLM backend
    uv run modal run modal_serve.py::parity --run jaredpalmer/kev-4b               # kev.benchmark: fp32 torch vs bf16 torch vs vLLM
    uv run modal run modal_serve.py::loadtest --run jaredpalmer/kev-4b             # p50/p99 + throughput at 1/8/32 concurrent clients
    KEV_GPU=H100 uv run modal run modal_serve.py::bench --run jaredpalmer/kev-4b --name vllm-4b-h100      # state sharing on vs off
    KEV_GPU=H100 uv run modal run modal_serve.py::bench --base Qwen/Qwen3.8-27B --revision <sha> --name vllm-27b-h100 --gpu-memory 0.92

The two backends need two images: the torch path runs the locked kev environment (torch 2.8 + flash-linear-attention, as
skills/kev-deploy and modal_app.py do), vLLM brings its own torch, so its image installs vllm and kev's other
dependencies next to the local `kev/` source. Both mount the `kev-hf-cache` volume, which also keeps the merged vLLM
export (kev.vllm_model.export_dir) so only the first cold start merges. parity writes kev.benchmark result directories to
the `kev-runs` volume under /serve/<name> and pulls them to runs/serve/<name>; compare them with kev.compare.
Settings: KEV_MODEL (endpoint checkpoint, default jaredpalmer/kev-4b), KEV_GPU (default L40S), KEV_API_KEY (bearer auth,
uploaded as a Modal secret), KEV_APP_NAME (default kev-vllm).
"""
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
SETTINGS = {"KEV_MODEL": "jaredpalmer/kev-4b", "KEV_APP_NAME": "kev-vllm", "KEV_GPU": "L40S"}
SETTINGS = {k: os.environ.get(k, v) for k, v in SETTINGS.items()}
GPU = SETTINGS["KEV_GPU"]
SECRET = {k: os.environ[k] for k in ("KEV_API_KEY", "HF_TOKEN") if os.environ.get(k)}
ENV = {"HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
       "TRITON_CACHE_DIR": "/hf/triton-cache", **SETTINGS}

app = modal.App(SETTINGS["KEV_APP_NAME"])
torch_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(ROOT), groups=[], extras=["serve"])
    .uv_pip_install("flash-linear-attention", "triton>=3.7.1")   # Gated DeltaNet kernels (fla needs triton >= 3.7.1 on Hopper)
    .env({**ENV, "KEV_BACKEND": "torch"})
    .add_local_python_source("kev")
    .add_local_dir(ROOT / "evals", "/root/evals")
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("vllm==0.29.0", "transformers>=5.17,<6", "peft>=0.21", "accelerate>=1.15.0", "datasets>=3.0",
                    "scikit-learn>=1.9.1", "fastapi>=0.115", "typesafe-sdk>=0.6.0", "uvicorn>=0.30",
                    "flash-linear-attention")   # fla: the torch fp32 reference in parity runs in this image too
    .env({**ENV, "KEV_BACKEND": "vllm"})
    .add_local_python_source("kev")
    .add_local_dir(ROOT / "evals", "/root/evals")
    .add_local_dir(ROOT / "scripts", "/root/scripts")   # bench reuses scripts/serving_bench.py's request shapes
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_dict(SECRET)] if SECRET else []
VOLUMES = {"/hf": hf_cache, "/runs": runs_volume}
WARMUP = {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "model": "kev-latest",
          "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                       "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}},
                        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"}}}


def load_server(run):
    """kev.serve's Server for `run`, loaded the way kev.serve.main does on CUDA (bf16, CUDA graphs for torch, KEV_BACKEND from the image)."""
    import torch
    from kev.api import SystemOneRequest
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.serve import Server
    ck = Checkpoint(run)
    tok, model = ck.load("cuda", replace(LoadOptions.from_env(), dtype=torch.bfloat16, cuda_graphs=True))
    server = Server(ck, tok, model, "cuda")
    server.answer(SystemOneRequest.model_validate(WARMUP))   # compile kernels / capture graphs before the first real request
    hf_cache.commit()
    return server


@app.cls(image=vllm_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes={"/hf": hf_cache}, secrets=secrets,
         scaledown_window=300, timeout=600, startup_timeout=1800)
@modal.concurrent(max_inputs=64)   # the engine batches concurrent requests; kev.serve no longer serializes them
class Kev:
    @modal.enter()
    def load(self):
        from kev.serve import app as api
        started = time.time()
        api.state.server = load_server(SETTINGS["KEV_MODEL"])
        print(f"serving {SETTINGS['KEV_MODEL']} via vllm, ready in {time.time() - started:.0f}s", flush=True)
        self.api = api

    @modal.asgi_app(label=f"{SETTINGS['KEV_APP_NAME']}-api")
    def web(self):
        return self.api


def benchmark(run, suite, name, dtype, env=()):
    """kev.benchmark on the suite's development partition with this image's backend at KEV_DTYPE=dtype, to /runs/serve/<name>."""
    out = Path("/runs/serve") / name
    try:
        subprocess.run([sys.executable, "-m", "kev.benchmark", "--run", run, "--suite", f"/root/{suite}", "--out", str(out), "--device", "cuda"],
                       check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root", "KEV_DTYPE": dtype, **dict(env)})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return (out / "report.json").read_text(encoding="utf-8")


@app.function(image=torch_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def benchmark_torch(run, suite, name, dtype):
    return benchmark(run, suite, name, dtype)


@app.function(image=vllm_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def benchmark_vllm(run, suite, name):
    # kev.benchmark touches CUDA before the engine starts, and a forked engine process cannot re-initialize it
    return benchmark(run, suite, name, "bf16", env={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})


def load_test(run, suite, levels, requests):
    """Server.probs from `c` client threads at once for each concurrency level c: per-request latency p50/p99 and
    requests/s over `requests` development records (the same sample for every level and backend)."""
    import random
    import statistics
    from concurrent.futures import ThreadPoolExecutor
    from kev.data import materialize
    from kev.suite import load_split
    server = load_server(run)
    records = [materialize(r) for r in load_split(f"/root/{suite}", "development")]
    sample = random.Random(0).choices(records, k=requests)

    def one(rec):
        t = time.perf_counter(); server.probs(rec); return time.perf_counter() - t

    for rec in sample: one(rec)   # warm every shape before timing: torch captures each new CUDA-graph bucket in the background
    while server.capture_lock.locked(): time.sleep(0.1)
    report = {"backend": server.model.backend, "gpu": GPU, "run": run, "levels": {}}
    for c in levels:
        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter()
            lat = sorted(pool.map(one, sample))
            wall = time.perf_counter() - start
        report["levels"][c] = {"p50_ms": round(1000 * statistics.median(lat), 1), "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1),
                               "requests_per_s": round(len(lat) / wall, 2), "questions_per_s": round(sum(len(r["questions"]) for r in sample) / wall, 2)}
        print(report["levels"][c], flush=True)
    return report


@app.function(image=torch_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def load_test_torch(run, suite, levels, requests):
    return load_test(run, suite, levels, requests)


@app.function(image=vllm_image, gpu=GPU, cpu=4, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=7200)
def load_test_vllm(run, suite, levels, requests):
    return load_test(run, suite, levels, requests)


def base_export_dir(base, revision):
    return Path(ENV["HF_HOME"]) / "kev-vllm" / f"base-{base.replace('/', '--')}-{revision[:12]}"


@app.function(image=vllm_image, cpu=8, memory=(131072, 262144), volumes={"/hf": hf_cache}, secrets=secrets, timeout=7200)
def export_base(base, revision):
    """The untrained base as a vLLM checkpoint (kev.vllm_model.export_merged with no adapter), for benchmarks of a size that has
    no mergeable Kev checkpoint yet. CPU only; safetensors serializes a copy, so a 27B needs about twice its 54 GB of RAM."""
    import torch
    from kev.model import DecisionModel, load_tokenizer
    from kev.vllm_model import export_merged
    out = base_export_dir(base, revision)
    if not (out / "model.safetensors").exists():
        tok = load_tokenizer(base, revision=revision)
        export_merged(DecisionModel(base, tok, "cpu", lora=None, revision=revision, dtype=torch.bfloat16).lm, tok, out, torch.bfloat16)
        hf_cache.commit()
    return str(out)


def engine_dir(run, base, revision):
    """(vLLM checkpoint directory, base, revision) the bench serves: a Kev run's merged export or an untrained base's."""
    import torch
    from kev.checkpoint import Checkpoint
    from kev.vllm_model import export_dir
    if not run:
        return base_export_dir(base, revision), base, revision
    ck = Checkpoint(run)
    out = export_dir(ck, torch.bfloat16, 1.0)
    if not (out / "model.safetensors").exists(): raise SystemExit(f"{run} has no vLLM export yet; `parity` or a deploy builds it")
    return out, ck.meta.base, ck.meta.base_revision


def serving_bench(model_dir, tok, share_state, reps=20, levels=(1, 8, 32, 64)):
    """scripts/serving_bench.py's measurements for the vLLM backend: model time per request for its four request shapes (new
    and repeated state), then requests/s at `levels` concurrent callers on its three samples (256 new 6-question short
    states, 256 decision-v7 development records, 64 new 2,200-token states), each level run twice with the second kept, as
    that script does for the torch path. Adds the engine's token count per request: prompt tokens (state x questions when
    every row carries its state) and computed tokens (what is left after the prefix cache). The pointer head is random:
    the timing does not depend on it, and parity is `parity`'s job."""
    import random
    import statistics
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    import torch
    sys.path.insert(0, "/root/scripts")
    from serving_bench import CASES, request
    from kev.api import to_record
    from kev.data import materialize
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from kev.suite import load_split
    from kev.vllm_model import VLLMDecisionModel
    started = time.time()
    model = VLLMDecisionModel(model_dir, torch.bfloat16, share_state=share_state).eval()
    report = {"share_state": share_state, "load_seconds": round(time.time() - started, 1)}

    def one(rec):
        enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        t = time.perf_counter(); model.probs(enc); return time.perf_counter() - t

    def per_request(before, n):
        d = model.tokens - before
        return {"prompt_tokens": round(d["prompt"] / n), "computed_tokens": round(d["computed"] / n), "engine_requests": round(d["requests"] / n, 2), "redone_rows": d["redone"]}

    latency = {}
    for k, case in enumerate(CASES):   # new states numbered per case: two cases share a ticket text, and a repeated question is not a new one
        row = {"first_ms": round(1000 * one(to_record(request(case, 0))[0]), 1)}
        for mode in ("new", "cached"):
            recs = [to_record(request(case, 100 * (k + 1) + i if mode == "new" else 0))[0] for i in range(1, reps + 3)]
            one(recs[0]); one(recs[1]); before = Counter(model.tokens)
            row[f"{mode}_ms"] = round(1000 * statistics.median(one(r) for r in recs[2:]), 1)
            row[f"{mode}_tokens"] = per_request(before, reps)
        latency[case] = row
        print(case, row, flush=True)
    report["latency"] = latency

    samples = {"6 questions, new short state": [to_record(request("6 questions, short state", 1000 + i))[0] for i in range(256)],
               "decision-v7 development": random.Random(0).choices([materialize(r) for r in load_split("/root/evals/v7/decision-v7", "development")], k=256),
               "5 questions, 2,200-token state": [to_record(request("5 questions, 2,200-token state", 1000 + i))[0] for i in range(64)]}

    def run_level(recs, c):
        model._run(model.engine.reset_prefix_cache())   # every level replays the same sample: without this, a repeat is served from the last level's entries
        before = Counter(model.tokens)
        with ThreadPoolExecutor(c) as pool:
            start = time.perf_counter(); lat = sorted(pool.map(one, recs)); wall = time.perf_counter() - start
        return {"p50_ms": round(1000 * statistics.median(lat), 1), "p99_ms": round(1000 * lat[int(0.99 * (len(lat) - 1))], 1),
                "requests_per_s": round(len(lat) / wall, 1), **per_request(before, len(recs))}

    report["throughput"] = {}
    for name, recs in samples.items():
        for c in levels: run_level(recs, c)   # the first pass warms every shape; the second is kept (serving_bench does the same)
        for c in levels:
            report["throughput"][f"{name} @ {c} clients"] = res = run_level(recs, c)
            print(name, c, res, flush=True)
    report["engine_tokens"] = dict(model.tokens)
    model.close()
    return report


@app.function(image=vllm_image, gpu=GPU, cpu=8, memory=(32768, 131072), volumes=VOLUMES, secrets=secrets, timeout=10800)
def bench_vllm(run, base, revision, name, gpu):
    """serving_bench without state sharing (every row carries its state), then with it, one engine after the other on this
    GPU: separate containers of the same GPU type differed by ~25% on identical work. -> /runs/serve/<name>/report.json"""
    import json
    from kev.model import load_tokenizer
    model_dir, base, revision = engine_dir(run, base, revision)
    tok = load_tokenizer(base, revision=revision)
    report = {"run": run or f"{base}@{revision}", "gpu": gpu}
    for mode, share in (("rows", False), ("share", True)):
        report[mode] = serving_bench(model_dir, tok, share)
        out = Path("/runs/serve") / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")   # after each mode: a timeout keeps the first
        runs_volume.commit()
    return report


@app.local_entrypoint()
def bench(name: str, run: str = "", base: str = "", revision: str = "", gpu: str = GPU, gpu_memory: str = ""):
    """bench_vllm on one `gpu`. `run` = a Kev checkpoint whose vLLM export exists; `base` + `revision` = an untrained base
    (exported first). Pulled to runs/serve/<name>."""
    import json
    if bool(run) == bool(base): raise SystemExit("give exactly one of --run or --base (with --revision)")
    if base: print("export:", export_base.remote(base, revision))
    env = [modal.Secret.from_dict({"KEV_VLLM_GPU_MEMORY": gpu_memory})] if gpu_memory else []
    print(json.dumps(bench_vllm.with_options(gpu=gpu, secrets=[*secrets, *env]).remote(run, base, revision, name, gpu), indent=1))
    Path("runs/serve").mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "kev-runs", f"/serve/{name}", "runs/serve/"], check=True)


@app.local_entrypoint()
def parity(run: str = SETTINGS["KEV_MODEL"], suite: str = "evals/v7/decision-v7", tag: str = "", kinds: str = "torch-fp32,torch-bf16,vllm-bf16"):
    """Three kev.benchmark runs of `run` on the same GPU type: torch fp32 (the reported-numbers path), torch bf16 (the
    current serving path) and vLLM bf16 (`kinds` picks a subset). Pulled to runs/serve/; `kev.compare --candidate runs/serve/<x> --reference ...`."""
    name = tag or run.split("/")[-1].replace("@", "-")
    spawn = {"torch-fp32": lambda n: benchmark_torch.spawn(run, suite, n, "fp32"), "torch-bf16": lambda n: benchmark_torch.spawn(run, suite, n, "bf16"),
             "vllm-bf16": lambda n: benchmark_vllm.spawn(run, suite, n)}
    kinds = kinds.split(",")
    for job in [spawn[k](f"{name}-{k}") for k in kinds]: job.get()
    Path("runs/serve").mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        subprocess.run(["modal", "volume", "get", "--force", "kev-runs", f"/serve/{name}-{kind}", "runs/serve/"], check=True)
    print(f"pulled runs/serve/{name}-{{{','.join(kinds)}}}")


@app.local_entrypoint()
def loadtest(run: str = SETTINGS["KEV_MODEL"], suite: str = "evals/v7/decision-v7", levels: str = "1,8,32", requests: int = 256):
    """Same records, same GPU type, both backends; prints one JSON report per backend."""
    import json
    cs = [int(c) for c in levels.split(",")]
    for report in (load_test_torch.remote(run, suite, cs, requests), load_test_vllm.remote(run, suite, cs, requests)):
        print(json.dumps(report, indent=1))
