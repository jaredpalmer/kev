"""Run kev studies on Modal: one GPU container per trial, results pulled back into runs/.

    KEV_GPU=T4 uv run modal run modal_app.py::smoke                        # ~2 min end to end on a T4 (free tier)
    uv run modal run modal_app.py::study --suite evals/decision-v1 \\
        --plan experiments/mbp-comparison.json --name mbp-comparison-v1     # N trials in parallel on H100s
    uv run modal run modal_app.py::evaluate --run jaredpalmer/kev-0.5b \\
        --suite evals/transfer-v1 --name transfer-kev-v01-h100             # score a Hub checkpoint as a research trial
    uv run modal run modal_app.py::base_probe --bases Qwen/Qwen3.5-9B-Base  # untrained-base rows (zero-shot letter logits)
    uv run modal run modal_app.py::benchmarks --jobs run@suite-or-jsonl@name # kev.benchmark on suites or external .jsonl files
    uv run modal run modal_app.py::smoke_base --base Qwen/X --revision sha   # does a new base fit? LoRA footprint, peak GB, step time

The same `kev.experiment.execute_trial` runs here and on the MBP; only the device differs. Every trial records
the local git commit (KEV_GIT_COMMIT), the suite hash, and the hashes of the kev/*.py files that were shipped, and
`kev.experiment --aggregate` ranks the study locally afterwards so the ledger is produced by one code path.

Volumes: kev-hf-cache (base weights, downloaded once), kev-runs (trial outputs). Secrets: set KEV_HF_SECRET=<modal secret
name> to attach a Secret carrying HF_TOKEN for gated bases; run_mirror (the private Hub copy of snapshots and final
checkpoints, kev.mirror) always mounts that Secret, `huggingface-secret` unless KEV_HF_SECRET names another.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import modal

from kev.budget import (FULL_FT_RETRIES, INTERPOLATE_CPU, INTERPOLATE_MEMORY, INTERPOLATE_TIMEOUT, MAX_BUDGET, MAX_TIMEOUT, TRIAL_CPU, TRIAL_MEMORY,   # run_trial's resources and
                        compute_bound, hourly_rate, interpolation_bound, trial_disk, trial_resources)                                                    # the admission bounds

APP_NAME = os.environ.get("KEV_APP_NAME", "kev-research")

ROOT = Path(__file__).resolve().parent
RUNS_MOUNT, HF_MOUNT = "/runs", "/hf"
GPU = os.environ.get("KEV_GPU", "H100")   # H100 needs a payment method on the workspace; KEV_GPU=T4 for the free tier

def worker_environment(app_name, gpu, secret_name=None):
    env = {"HF_HOME": HF_MOUNT, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
           "TRITON_CACHE_DIR": f"{HF_MOUNT}/triton-cache",   # compiled DeltaNet kernels and their autotuning results survive the container
           "KEV_APP_NAME": app_name, "KEV_GPU": gpu}
    if secret_name:
        env["KEV_HF_SECRET"] = secret_name
    return env


app = modal.App(APP_NAME)
CAUSAL_CONV1D = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(ROOT), groups=[], extras=["serve"])   # exact locked deps (serve: kev.serve for serving_bench); Linux torch wheels are the CUDA build
    # Gated DeltaNet kernels for the Qwen3.5 hybrid backbones (transformers falls back to slow reference code without them)
    # fla refuses its gated chunk backward on Hopper with Triton 3.4-3.7.0 (incorrect results, fla#640); torch 2.8 pins 3.4
    .uv_pip_install("flash-linear-attention==0.5.2", "triton>=3.7.1")   # pinned: kev.fused_qwen35 patches fla kernel launches
    # the DeltaNet short convolution: transformers uses causal-conv1d's CUDA kernel (forward and backward) when it is
    # importable and a PyTorch conv otherwise; the prebuilt wheel matches the image's torch 2.8 / CUDA 12 / Python 3.13.
    # --no-deps: resolving its torch requirement would put back torch's pinned triton 3.4, which fla refuses on Hopper
    .uv_pip_install(CAUSAL_CONV1D, extra_options="--no-deps")
    .uv_pip_install("pytest")   # gpu_tests
    .env(worker_environment(APP_NAME, GPU, os.environ.get("KEV_HF_SECRET")))
    .add_local_python_source("kev")
    .add_local_file(ROOT / "uv.lock", "/root/uv.lock")
    .add_local_file(ROOT / "pyproject.toml", "/root/pyproject.toml")
    .add_local_dir(ROOT / "evals", "/root/evals")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
    .add_local_dir(ROOT / "tests", "/root/tests")
    .add_local_file(ROOT / "experiments/sft-v1-lengths.json", "/root/experiments/sft-v1-lengths.json")   # scripts/sft_probe.py
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_name(os.environ["KEV_HF_SECRET"])] if os.environ.get("KEV_HF_SECRET") else []
MIRROR_SECRET = os.environ.get("KEV_HF_SECRET") or "huggingface-secret"   # run_mirror's HF_TOKEN (the same name in the container: KEV_HF_SECRET is in the image env)


def local_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def local_source_hashes():
    sys.path.insert(0, str(ROOT))
    from kev.experiment import source_hashes
    return source_hashes()


@app.function(image=image, cpu=1, memory=1024, timeout=120)
def remote_source_hashes():
    """Hashes of kev/*.py inside the deployed image: the launcher compares them with the checkout before spawning."""
    from kev.experiment import source_hashes
    return source_hashes()


@app.function(image=image, gpu=GPU, cpu=TRIAL_CPU, memory=TRIAL_MEMORY, max_containers=24, retries=0, timeout=28800,   # a 35B-A3B bf16 checkpoint (70 GB) is staged through host memory while loading; the old 48 GB cap stalled the container
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None):
    """One trial in one container (trial below)."""
    return trial(study, index, label, config, suite, expected_sources, git_commit, existing, transfer)


@app.function(image=image, gpu=GPU, cpu=TRIAL_CPU, memory=TRIAL_MEMORY, max_containers=24, retries=0, timeout=86400, ephemeral_disk=trial_disk(True),
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_full_trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None):
    """A full-weight trial: run_trial with the disk its resume points need (kev.budget.trial_disk; with_options cannot set it)."""
    return trial(study, index, label, config, suite, expected_sources, git_commit, existing, transfer)


def trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None):
    """One trial in one container. `existing` is a checkpoint path on the runs volume or a Hub id (legacy scoring). A
    full-weight trial whose directory exists already is the same call again (Modal retried it after a timeout, or `resume`
    spawned it): it continues from its last resume point, unless an earlier attempt failed with an error (failed.json)."""
    import torch
    from kev.experiment import continue_trial, execute_trial, source_hashes
    from kev.suite import write_json

    os.environ["KEV_GIT_COMMIT"] = git_commit
    if source_hashes() != expected_sources:
        raise RuntimeError("container received different kev/*.py than the launcher hashed")
    out = Path(RUNS_MOUNT) / study / f"{index:02d}-{label}"
    runs_volume.reload()
    again = out.exists()
    if again and config.get("full_ft") and (out / "failed.json").exists():
        return failed_trial(label, out)   # a retry after an attempt that failed with an error: report it, do not run again
    if again and (not config.get("full_ft") or (out / "result.json").exists()):
        raise FileExistsError(f"refusing to overwrite remote trial: {out}")
    print(f"[{label}] {torch.cuda.get_device_name(0)} torch {torch.__version__} {'continuing' if again else 'config='+json.dumps(config)}", flush=True)
    transfer = Path("/root") / transfer if transfer else None
    stop = threading.Event()
    repo = config.get("snapshot_hub_repo")   # a private Hub mirror of each committed snapshot and the final checkpoint (off unless set)
    committer = threading.Thread(target=commit_resume_points, args=(out / "checkpoint" / "resume", stop, out / "snapshots"),
                                 kwargs={"final_dir": out / "checkpoint", "mirror": (lambda path: spawn_mirror([path], repo)) if repo else None}, daemon=True)
    if config.get("full_ft"): committer.start()
    try:
        report, _ = (continue_trial(Path("/root") / suite, out, expected_sources, "cuda", transfer) if again
                     else execute_trial(config or {}, Path("/root") / suite, out, expected_sources, "cuda", existing, transfer))
    except Exception as error:   # a timeout kills the container before it gets here
        if out.exists(): write_json(out / "failed.json", {"error": f"{type(error).__name__}: {str(error)[:2000]}"})
        if config.get("full_ft"): return failed_trial(label, out)   # returned, not raised: Modal retries only what raises
        raise
    finally:
        stop.set()
        if committer.is_alive(): committer.join()
        runs_volume.commit()
        hf_cache.commit()
    return {"label": label, "objective": report["objective"], "clean_acc": report["clean"]["acc"],
            "wall_seconds": report["wall_seconds"], "gates": report["gates"]["checks"]}


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 49152), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_locked_test(trial_path, name, suites, git_commit, redo_interrupted=False):
    """Read the locked test partitions ONCE for a promoted trial. Writes /runs/locked/<name>/... ; refuses to rerun."""
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import digest, load_split, read_json, write_json
    os.environ["KEV_GIT_COMMIT"] = git_commit
    trial = Path(RUNS_MOUNT) / trial_path
    out = Path(RUNS_MOUNT) / "locked" / name
    runs_volume.reload()
    summary = None
    if out.exists():
        # an interrupted read may finish the suites it never touched; a suite that was read is never read again
        prior = read_json(out / "summary.json") if (out / "summary.json").exists() else {"suites": {}}
        if all(label in prior["suites"] for label in suites):
            raise FileExistsError(f"locked test already read for {name}; a second read is not allowed")
        interrupted = []
        for label in list(suites):
            if label in prior["suites"]:
                suites.pop(label)
            elif (out / label).exists():
                # a read that crashed before any aggregate was produced: no number was ever observed, so completing it does
                # not enable selection on the test; it must be requested explicitly and is recorded
                if not redo_interrupted: raise RuntimeError(f"{label} partition was touched but not summarised; pass redo_interrupted to complete it")
                shutil.rmtree(out / label); interrupted.append(label)
        summary = {**prior, "resumed_for": sorted(suites), "interrupted_reads_redone": interrupted}
    # a checkpoint made without a trial (round 20's interpolations: /runs/<study>/<name>/checkpoint) ran no in-trial gates
    # and fitted no temperature: its read is '-ungated' and saved raw (kev.rounds serves every read at its registered T)
    result = read_json(trial / "result.json") if (trial / "result.json").exists() else None
    if not (result or {}).get("gates", {}).get("passed") and not name.endswith("-ungated"):
        raise RuntimeError(f"{'trial did not pass its gates' if result else 'no trial result.json (a checkpoint without in-trial gates)'}; "
                           "name the read '<name>-ungated' to record an exploratory read")
    out.mkdir(parents=True, exist_ok=True)
    temperature = result.get("temperature", 1.0) if result else 1.0
    predictor = LocalPredictor(str(trial / "checkpoint"), "cuda", LoadOptions(temperature=1.0))   # raw logits; the trial's fitted temperature is applied by evaluate_records below
    summary = summary or {"trial": trial_path, "trial_result_sha256": digest(trial / "result.json") if result else None, "temperature": temperature, "git_commit": git_commit, "suites": {}}
    try:
        for label, suite in suites.items():
            records = load_split(Path("/root") / suite, "test", allow_test=True)
            report, _ = evaluate_records(records, predictor, out / label, temperature, heldout_sources=tuple(r["_meta"]["source"] for r in records))
            summary["suites"][label] = {"suite": suite, "suite_sha256": digest(Path("/root") / suite / "manifest.json"), "clean": report["clean"], "tasks": report["tasks"],
                                        "paired_flip": report["paired_flip"], "variants": report["variants"], "permutation": report["permutation"], "coverage": report["coverage"]}
            print(f"[{name}] {label} test: acc {report['clean']['acc']:.3f} brier {report['clean']['brier']:.3f}", flush=True)
    finally:
        write_json(out / "summary.json", summary)
        runs_volume.commit()
    return summary


def run_tool(cmd, out, block="clean"):
    """Run a repo script/module inside the container against the mounted checkout, refusing to overwrite `out` on the
    volume; returns the report's `block` (None = the whole report). Shared by the probe, bench and serving functions."""
    import subprocess as sp
    if out.exists():
        raise FileExistsError(f"{out} exists on the volume")
    try:
        sp.run([str(c) for c in cmd], check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        runs_volume.commit(); hf_cache.commit()
    from kev.suite import read_json
    report = read_json(out / "report.json")
    return report if block is None else report[block]


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_base_probe(base, suite, name, tasks="all", prompt="plain", split="development", revision=None, adapter=None, all_questions=False):
    """Untrained baseline: the base model's zero-shot letter-logit readout on a frozen suite partition
    (scripts/base_mmlu_probe.py; --adapter measures a Kev adapter through the same readout). Writes benchmark-compatible
    rows/report under /runs/probes/<name>."""
    out = Path(RUNS_MOUNT) / "probes" / name
    cmd = [sys.executable, "/root/scripts/base_mmlu_probe.py", "--base", base, "--suite", f"/root/{suite}", "--tasks", tasks, "--device", "cuda", "--out", out, "--prompt", prompt, "--split", split]
    if revision: cmd += ["--revision", revision]
    if adapter: cmd += ["--adapter", adapter]
    if all_questions: cmd += ["--all_questions"]
    return run_tool(cmd, out)


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_bench(run, suite, name, flags=""):
    """kev.benchmark for a checkpoint (Hub id or /runs path) on a suite's development partition or a --data .jsonl
    (external evals), written to /runs/bench/<name>. flags: extra benchmark switches, e.g. "--date_facts"."""
    out = Path(RUNS_MOUNT) / "bench" / name
    source = ["--data", f"/root/{suite}"] if suite.endswith(".jsonl") else ["--suite", f"/root/{suite}"]
    return run_tool([sys.executable, "-m", "kev.benchmark", "--run", run, *source, "--out", out, "--device", "cuda", *flags.split()], out)


MIRROR_TIMEOUT = 4 * 3600   # a 27B checkpoint is ~51 GB; several per call when mirror_snapshots uploads a whole study


@app.function(image=image, cpu=4, memory=(16384, 65536), retries=0, timeout=MIRROR_TIMEOUT,
              volumes={RUNS_MOUNT: runs_volume}, secrets=[modal.Secret.from_name(MIRROR_SECRET)])
def run_mirror(paths, repo, force=False):
    """Upload complete checkpoint directories on the volume (/runs/..., snapshots or final checkpoints) to a PRIVATE Hub
    model repo (kev.mirror: refuses a public repo, creates a missing one private, retries once, never raises for an upload)
    and commit the records it writes next to them. CPU only; spawned by a full-weight trial whose plan sets
    snapshot_hub_repo, or by mirror_snapshots. -> {path: record or None}."""
    from kev.mirror import mirror
    runs_volume.reload()
    try:
        return {str(p): mirror(p, repo, force=force) for p in paths}
    finally:
        runs_volume.commit()


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_serving(run, name, flags=""):
    """scripts/serving_bench.py (served latency with and without CUDA graphs, parity against fp32) -> /runs/serving/<name>."""
    out = Path(RUNS_MOUNT) / "serving" / name
    return run_tool([sys.executable, "/root/scripts/serving_bench.py", "--run", run, "--suite", "/root/evals/v7/decision-v7", "--out", out, *flags.split()], out, block=None)


@app.local_entrypoint()
def serving(run: str, name: str, gpu: str = GPU, flags: str = ""):
    report = run_serving.with_options(gpu=gpu).remote(run, name, flags)
    print(json.dumps(report, indent=1))
    pull_volume(f"/serving/{name}", ROOT / "runs")


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 131072), retries=0, timeout=2400,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_smoke_base(base, revision):
    """Does a base fit? Load it through DecisionModel with the Kev LoRA config, report the adapter size and which modules it
    hit, run one training step on real records with gradient checkpointing, and report peak memory and steady step time."""
    import time
    import torch
    from kev.data import materialize
    from kev.device import allocated_bytes, sync
    from kev.model import DecisionModel, load_tokenizer
    from kev.suite import load_split
    t0 = time.time(); tok = load_tokenizer(base, revision=revision)
    m = DecisionModel(base, tok, "cuda", dtype=torch.bfloat16, lora=16, revision=revision)
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.lm.config.use_cache = False; m.train()
    trainable = [(n, p.numel()) for n, p in m.lm.named_parameters() if p.requires_grad]
    recs = [materialize(r) for r in load_split("/root/evals/v7/decision-v7", "development")[:2]]
    encs = [m.encode(tok, r, strict=True) for r in recs]

    def step():
        m.lm.zero_grad(set_to_none=True); m.head.zero_grad(set_to_none=True); ts = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = m.forward_batch(encs)
        loss = sum(torch.nn.functional.cross_entropy(z.float()[None], torch.tensor([q["label"]], device="cuda")) for zs, r in zip(logits, recs) for z, q in zip(zs, r["questions"]))
        loss.backward(); sync("cuda")
        return round(time.time() - ts, 2), loss.item()
    torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    first, loss = step()                       # the first step pays Triton compilation
    steady = [step()[0] for _ in range(3)]
    return {"base": base, "hybrid": m.hybrid, "load_seconds": round(t1 - t0), "first_step_seconds": first, "steady_step_seconds_2_records": steady,
            "questions_per_record": [len(r["questions"]) for r in recs], "trainable_params_M": round(sum(k for _, k in trainable) / 1e6, 1),
            "lora_module_names": sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in trainable}), "routed_expert_lora_params": sum(k for n, k in trainable if ".experts." in n),
            "peak_gb": round(allocated_bytes("cuda") / 1e9, 1), "weights_gb": round(sum(p.numel() * p.element_size() for p in m.lm.parameters()) / 1e9, 1), "loss": round(loss, 3)}


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 65536), retries=0, timeout=3600, ephemeral_disk=trial_disk(True),
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_sft_probe(name, base, revision, gpu, train, records, check_load, flags=""):
    """scripts/sft_probe.py (full-weight memory, s/step, throughput, projections, loader check) in the container's scratch
    disk: the checkpoint it writes (51 GB for a 27B) stays there; report.json, train.log and training_metrics.json land in
    /runs/sft-probe/<name>. Resume points (`--save_every_steps` in `train`) are written to the volume, where a trial writes
    them, and deleted after the probe (a 27B's are ~307 GB)."""
    out, scratch = Path(RUNS_MOUNT) / "sft-probe" / name, Path("/tmp/sft-probe")
    if out.exists():
        raise FileExistsError(f"{out} exists on the volume")
    (out / "resume").mkdir(parents=True)
    try:
        subprocess.run([sys.executable, "/root/scripts/sft_probe.py", "--base", base, "--revision", revision, "--gpu", gpu, "--out", str(scratch),
                        "--records", str(records), "--train", train, "--check_load", str(check_load), "--resume_dir", str(out / "resume"), *flags.split()],
                       check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        shutil.rmtree(out / "resume", ignore_errors=True)
        for f in ("report.json", "train.log", "checkpoint/training_metrics.json"):
            if (scratch / f).exists(): shutil.copy(scratch / f, out / Path(f).name)
        runs_volume.commit(); hf_cache.commit()
    from kev.suite import read_json
    return read_json(out / "report.json")


@app.function(image=image, gpu=GPU, cpu=4, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={HF_MOUNT: hf_cache}, secrets=secrets)
def run_gpu_tests(tests):
    """pytest on a GPU for the tests that need CUDA (they skip locally), e.g. tests/test_model.py::test_cuda_graphs_match_eager."""
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", *tests.split()], cwd="/root", env={**os.environ, "PYTHONPATH": "/root"}, capture_output=True, text=True)
    hf_cache.commit()
    return done.returncode, done.stdout[-20000:] + done.stderr[-5000:]


@app.local_entrypoint()
def gpu_tests(tests: str, gpu: str = "H100"):
    """uv run modal run modal_app.py::gpu_tests --tests "tests/test_model.py::test_shared_prefix_matches_rows" [--gpu H100]"""
    code, output = run_gpu_tests.with_options(gpu=gpu).remote(tests)
    print(output)
    if code: raise SystemExit(code)


@app.function(image=image, gpu=GPU, cpu=4, memory=(65536, 131072), retries=0, timeout=14400,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_rl(init_from, name, flags=""):
    """kev.rl (agentic RL after SFT) from a checkpoint (Hub id or /runs path) -> /runs/rl/<name>."""
    out = Path(RUNS_MOUNT) / "rl" / name
    return run_tool([sys.executable, "-m", "kev.rl", "--init_from", init_from, "--out", out, "--device", "cuda", *flags.split()], out)


@app.local_entrypoint()
def rl(init_from: str, name: str, gpu: str = GPU, flags: str = ""):
    """uv run modal run modal_app.py::rl --init-from jaredpalmer/kev-4b --name kev-4b-inv-v1 --flags "--iters 100 --replay /root/evals/devtools-v1/train.jsonl"
    then `modal volume get kev-runs rl/<name> runs/rl/`; score the checkpoint (/runs/rl/<name>) with `benchmarks` against its parent."""
    print(json.dumps(run_rl.with_options(gpu=gpu).remote(init_from, name, flags), indent=1))


KEV_27B_BASE = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")   # Kev-27B's base (post-trained), what full-weight SFT targets


@app.local_entrypoint()
def sft_probe(name: str, gpu: str = "H200", base: str = KEV_27B_BASE[0], revision: str = KEV_27B_BASE[1], train: str = "", records: int = 2000,
              check_load: int = 0, timeout: int = 3600, flags: str = ""):
    """Full-weight training probe on one container: `--gpu H200` (masters in host memory) or `--gpu H200:8` (FSDP2). `--train`
    passes kev.train arguments (batch, accum, max_steps, lr, row_budget, shared_prefix, save_every_steps); `--flags` more
    sft_probe.py switches (--mix synthetic, --no_conv_kernel). Pulled to runs/sft-probe/<name>."""
    cpu, memory = trial_resources(gpu, full_ft=True)
    print(f"admission bound ${hourly_rate(gpu, full_ft=True) * timeout / 3600:.2f} ({gpu}, {cpu} CPU, {memory[0] // 1024}-{memory[1] // 1024} GiB, {timeout} s)", flush=True)
    call = run_sft_probe.with_options(gpu=gpu, cpu=cpu, memory=memory, timeout=timeout).spawn(name, base, revision, gpu, train, records, check_load, flags)
    print(f"spawned sft probe {name}: call {call.object_id}", flush=True)
    report = call.get()
    pull_volume(f"/sft-probe/{name}", ROOT / "runs/sft-probe")
    print(json.dumps({k: v for k, v in report.items() if k != "training_metrics"}, indent=1))


@app.function(image=image, cpu=INTERPOLATE_CPU, memory=INTERPOLATE_MEMORY, retries=0, timeout=INTERPOLATE_TIMEOUT,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_interpolate(sft, alphas, names, base, revision, study):
    """scripts/interpolate_checkpoint.py on a CPU container: one checkpoint per alpha at /runs/<study>/<name>/checkpoint,
    the runs volume committed after each."""
    sys.path.insert(0, "/root")
    from scripts.interpolate_checkpoint import interpolate as write_interpolations
    runs_volume.reload()
    try:
        return write_interpolations(sft, alphas, [Path(RUNS_MOUNT) / study / n / "checkpoint" for n in names], base, revision,
                                    on_done=lambda report: runs_volume.commit(), log=lambda m: print(m, flush=True))
    finally:
        runs_volume.commit(); hf_cache.commit()


@app.local_entrypoint()
def interpolate(sft: str, prefix: str, alphas: str = "0.85,0.70,0.50", base: str = KEV_27B_BASE[0], revision: str = KEV_27B_BASE[1],
                study: str = "r20-wise", timeout: int = INTERPOLATE_TIMEOUT):
    """WiSE-FT checkpoints of a full-weight SFT checkpoint on the runs volume (--sft /runs/<trial>/checkpoint) with its base:
    /runs/<study>/<prefix>-w<alpha x 100>/checkpoint for each alpha (weight on the SFT backbone). Refuses unless the SFT
    run's base is --base @ --revision (Kev-27B's by default). Reports land in runs/<study>/<name>/interpolation.json."""
    sys.path.insert(0, str(ROOT))
    from scripts.interpolate_checkpoint import weight_label
    if not sft.startswith(f"{RUNS_MOUNT}/"): raise SystemExit(f"--sft is a checkpoint on the runs volume ({RUNS_MOUNT}/...), not {sft}")
    values = [float(a) for a in alphas.split(",")]
    names = [f"{prefix}-w{weight_label(a)}" for a in values]
    written = volume_names(f"/{study}")[0] if study in volume_names("/")[0] else set()
    taken = [n for n in names if n in written and "checkpoint" in volume_names(f"/{study}/{n}")[0]]
    if taken: raise SystemExit(f"/{study}/{taken} hold checkpoints already; interpolations are written once")
    print(f"admission bound ${interpolation_bound(timeout):.2f} ({INTERPOLATE_CPU} CPU, {INTERPOLATE_MEMORY[1] // 1024} GiB, {timeout} s, no GPU)", flush=True)
    call = run_interpolate.with_options(timeout=timeout).spawn(sft, values, names, base, revision, study)
    print(f"spawned interpolation of {sft} -> /{study}/{names}: call {call.object_id}", flush=True)
    for report in call.get():
        name = Path(report["checkpoint"]).parent.name
        (ROOT / "runs" / study / name).mkdir(parents=True, exist_ok=True)
        pull_volume(f"/{study}/{name}/interpolation.json", ROOT / "runs" / study / name)
        print(f"{name}: alpha {report['alpha']}, {report['tensors']} tensors, weights {report['weights_sha256'][:12]}, {report['seconds']} s", flush=True)


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 65536), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_anchors(base, suite, name, revision=None):
    """Frozen-base zero-shot targets for a suite's training partition -> /runs/anchors/<name>.json (kev.anchors)."""
    from kev.anchors import build
    out = Path(RUNS_MOUNT) / "anchors" / f"{name}.json"
    if out.exists():
        raise FileExistsError(f"anchors {name} exist")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = build(base, Path("/root") / suite, out, device="cuda", revision=revision)
    finally:
        runs_volume.commit(); hf_cache.commit()
    return meta


@app.local_entrypoint()
def anchors(base: str, suite: str, name: str, revision: str = "", gpu: str = GPU):
    call = run_anchors.with_options(gpu=gpu).spawn(base, suite, name, revision or None)
    print(f"spawned anchors {name}: call {call.object_id}; result lands at /runs/anchors/{name}.json on the volume")


def pulled(path, weights=False):
    """Whether a pull copies this runs-volume file. With `weights` everything. By default not:
    - backbone shards: a `model*.safetensors` file directly in a directory named `checkpoint` (a trial's final checkpoint,
      <trial>/checkpoint, and every snapshot, <trial>/snapshots/step-<N>/checkpoint): ~51 GB per 27B checkpoint;
    - resume points: anything under a `resume/` directory (fp32 optimizer state, ~307 GB for a 27B).
    Everything else is pulled: LoRA adapters (adapter_model.safetensors), head.pt, configs, tokenizers, snapshot.json,
    results, rows. Benchmarks and reads run on the volume paths, so nothing local needs the shards."""
    parts = PurePosixPath(path).parts
    shard = len(parts) > 1 and parts[-2] == "checkpoint" and parts[-1].startswith("model") and parts[-1].endswith(".safetensors")
    return weights or not (shard or "resume" in parts[:-1])


def pull_volume(remote, local_parent, weights=True):
    """Copy `remote` (a runs-volume path) to local_parent/<its last component>. weights=False leaves the files `pulled`
    skips on the volume (and says how much); weights=True is `modal volume get` of everything."""
    if weights:
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", remote, str(local_parent)], check=True)
        return
    from concurrent.futures import ThreadPoolExecutor
    from modal.volume import FileEntryType
    base = PurePosixPath("/" + remote.strip("/")).parent
    files = [e for e in runs_volume.listdir(remote, recursive=True) if e.type == FileEntryType.FILE]
    keep = [e for e in files if pulled(e.path)]

    def fetch(entry):
        target = Path(local_parent) / PurePosixPath("/" + entry.path.lstrip("/")).relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out: runs_volume.read_file_into_fileobj(entry.path, out)

    (Path(local_parent) / PurePosixPath(remote).name).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool: list(pool.map(fetch, keep))
    left = [e for e in files if not pulled(e.path)]
    print(f"pulled {len(keep)} file(s) of {remote} ({sum(e.size for e in keep) / 1e9:.2f} GB); left {len(left)} weight/resume file(s) "
          f"({sum(e.size for e in left) / 1e9:.1f} GB) on the volume (pull --weights to copy them)", flush=True)


@app.local_entrypoint()
def base_probe(bases: str, suite: str = "evals/v4/transfer-v4", tasks: str = "all", prompt: str = "plain", split: str = "development", revision: str = "", adapter: str = "", tag: str = "", gpu: str = GPU, all_questions: bool = False):
    """Untrained-base rows (the same items as every README row). Names are derived (<base>-base[-semif][-<tag>]-<suite>[-<split>]);
    results are pulled to runs/probes/<name>. e.g. KEV_GPU=H200 ... --bases Qwen/Qwen3.5-35B-A3B-Base --revision <sha>"""
    jobs = []
    for base in bases.split(","):
        name = base.split("/")[-1].lower().replace(".", "") + ("-semif" if prompt == "semif" else "-base") + (f"-{tag}" if tag else "") + "-" + suite.split("/")[-1] + ("" if split == "development" else f"-{split}")
        if (ROOT / "runs/probes" / name).exists(): print(f"skip {name}: exists locally"); continue
        jobs.append((base, suite, name, tasks, prompt, split, revision or None, adapter or None, all_questions))
    for (base, _, name, *_), result in zip(jobs, run_base_probe.with_options(gpu=gpu).starmap(jobs, return_exceptions=True)):
        if isinstance(result, Exception): print(f"{name}: FAILED {type(result).__name__}: {str(result)[:200]}"); continue
        pull_volume(f"/probes/{name}", ROOT / "runs/probes")
        print(f"{name}: acc {result['acc']:.3f} brier {result['brier']:.3f} conf-err {result['confident_error_rate']:.3f}")


# Per-read timeouts by suite (fp32 evaluation; a 9B on H100/H200). Long-state panels take over an hour for ~900 records of
# 6k-token rows; one timeout for a mixed batch made every job carry the slowest one's admission bound (round 6).
READ_TIMEOUTS = (("longstate", 7200), ("documents", 5400), ("transfer-v9", 3600))
DEFAULT_READ_TIMEOUT = 1800


def read_timeout(suite):
    return next((t for key, t in READ_TIMEOUTS if key in suite), DEFAULT_READ_TIMEOUT)


class BenchJob(NamedTuple):
    run: str      # Hub id[@revision] or a /runs path
    suite: str    # suite directory or .jsonl under the checkout
    name: str     # output: /runs/bench/<name>, pulled to runs/<name>
    flags: str    # extra kev.benchmark switches, each starting with --


def parse_jobs(jobs):
    """run@suite@name[@flags] entries, comma-separated. Parsed from the right: flags (when present) start with '--', then
    the name and the suite, and everything before them is the run, so a pinned Hub revision (repo@sha) stays in the run
    (split from the left, it shifted every field and round 10's parent test reads failed before scoring anything)."""
    out = []
    for job in jobs.split(","):
        parts = job.split("@")
        flags = parts.pop() if parts[-1].startswith("--") else ""
        if len(parts) not in (3, 4) or not all(parts):
            raise ValueError(f"benchmark job {job!r} is not run@suite@name[@flags] (flags start with --)")
        *run, suite, name = parts
        out.append(BenchJob("@".join(run), suite, name, flags))
    return out


@app.local_entrypoint()
def benchmarks(jobs: str, gpu: str = GPU, timeout: int = 0):
    """Score checkpoints on suites or --data .jsonl files: comma-separated run@suite@name[@flags] entries (parse_jobs), e.g.
    "jaredpalmer/kev-9b@evals/external/semif-v1@kev-9b-semif,/runs/X/00-trial-0/checkpoint@evals/v9/transfer-v9@x-v9@--date_facts".
    A full-weight trial's snapshot is read the same way: /runs/X/00-trial-0/snapshots/step-<N>/checkpoint@<suite>@<name>
    (its head.pt carries the raw temperature, 1.0, as a trial's final checkpoint does). Results are pulled to runs/<name>.
    Each job gets its suite's timeout (READ_TIMEOUTS); --timeout N sets one for all of them (raise it for a 27B, whose
    fp32 reads run about three times longer than a 9B's)."""
    entries = parse_jobs(jobs)
    missing = sorted({e.suite for e in entries if not (ROOT / e.suite).exists()})
    if missing: raise SystemExit(f"no such suite or data file in this checkout: {missing}")
    calls = [run_bench.with_options(gpu=gpu, timeout=timeout or read_timeout(e.suite)).spawn(*e) for e in entries]
    for (run, suite, name, _), call in zip(entries, calls):
        try: result = call.get()
        except Exception as e: print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}"); continue
        pull_volume(f"/bench/{name}", ROOT / "runs")
        print(f"{name}: acc {result['acc']:.3f} brier {result['brier']:.3f}")


def mirror_targets(study):
    """The complete checkpoint directories of a study on the volume: each trial's final checkpoint (checkpoint/head.pt)
    and its complete snapshots (snapshots/step-*/checkpoint/snapshot.json), as /runs paths."""
    from kev.full_ft import SNAPSHOT_INFO
    targets = []
    for trial in sorted(volume_names(f"/{study}")[0]):
        dirs, _ = volume_names(f"/{study}/{trial}")
        if "snapshots" in dirs:
            for step in sorted(volume_names(f"/{study}/{trial}/snapshots")[0], key=lambda s: int(s.removeprefix("step-"))):
                if SNAPSHOT_INFO in volume_names(f"/{study}/{trial}/snapshots/{step}/checkpoint")[1]: targets.append(f"{RUNS_MOUNT}/{study}/{trial}/snapshots/{step}/checkpoint")
        if "checkpoint" in dirs and "head.pt" in volume_names(f"/{study}/{trial}/checkpoint")[1]: targets.append(f"{RUNS_MOUNT}/{study}/{trial}/checkpoint")
    return targets


@app.local_entrypoint()
def mirror_snapshots(study: str = "", paths: str = "", repo: str = "jaredpalmer/kev-snapshots", force: bool = False, dry_run: bool = False):
    """(Re)upload complete snapshots and final checkpoints from the runs volume to a PRIVATE Hub repo (kev.mirror; the
    volume copy stays primary): every one of a study's (--study X) and/or explicit checkpoint directories (--paths
    /runs/a/checkpoint,...). Prints what it would upload with sizes; --dry-run stops there. A directory already
    recorded for this repo is skipped unless --force. A 27B checkpoint is ~51 GB: run with --detach."""
    from kev.mirror import destination
    targets = (mirror_targets(study) if study else []) + [p for p in paths.split(",") if p]
    if not targets: raise SystemExit("nothing to mirror: give --study and/or --paths (complete checkpoint directories under /runs)")
    for t in targets:
        size = sum(e.size for e in runs_volume.listdir(t.removeprefix(RUNS_MOUNT), recursive=True))
        print(f"{t} -> {repo}/{destination(t, RUNS_MOUNT)} ({size / 1e9:.1f} GB)", flush=True)
    if dry_run: return
    call = run_mirror.spawn(targets, repo, force)
    print(f"mirroring {len(targets)} checkpoint(s) to private {repo}: call {call.object_id}", flush=True)
    for path, entry in call.get().items():
        print(f"{path}: {'commit ' + entry['commit'] if entry else 'NOT uploaded (see the call log)'}", flush=True)


@app.local_entrypoint()
def smoke_base(base: str, revision: str, gpu: str = "H200"):
    """Memory and step-time check for a base that has not been trained yet (LoRA footprint, which modules it hits, peak GB)."""
    print(json.dumps(run_smoke_base.with_options(gpu=gpu).remote(base, revision), indent=1))


class Job(NamedTuple):
    """The arguments of one run_trial call (spawned or starmapped as *job)."""
    study: str
    index: int
    label: str
    config: dict
    suite: str
    expected_sources: dict
    git_commit: str
    existing: str | None
    transfer: str | None


def failed_trial(label, out):
    """The result of a full-weight trial that failed with an error (failed.json). Returned rather than raised, because
    Modal retries a raised call, and a full-weight trial's retries are for timeouts (they continue from a resume point);
    kev.rounds.poll_modal and launch() read "failed" as the trial's failure."""
    return {"label": label, "failed": json.loads((out / "failed.json").read_text(encoding="utf-8"))["error"]}


RESUME_COMMIT_POLL = 15   # seconds between looks at a training trial's latest.json


def commit_resume_points(resume_dir, stop, snapshot_dir=None, final_dir=None, mirror=None):
    """While a full-weight trial trains, commit the runs volume each time the trainer completes a resume point (its
    latest.json changes), a snapshot (kev.full_ft.completed_snapshots under `snapshot_dir` gains a step) or the final
    checkpoint (`final_dir`/training_metrics.json, written last, appears). A timeout kills the container without running
    trial()'s `finally`, and the retry can only continue from a committed point and keep committed snapshots. A failed
    commit is reported loudly and tried again on the next look; after `stop` it looks once more. `mirror(path)`, when
    given, is called for each newly committed snapshot and final checkpoint (spawn_mirror: the private Hub copy)."""
    from kev.full_ft import completed_snapshot_dirs
    latest = resume_dir / "latest.json"
    read = lambda: latest.read_text(encoding="utf-8") if latest.exists() else None
    snaps = lambda: completed_snapshot_dirs(snapshot_dir) if snapshot_dir else {}
    final = lambda: bool(final_dir) and (final_dir / "training_metrics.json").exists()
    committed, committed_snaps, committed_final = read(), set(snaps()), final()   # what a new container finds on the volume is committed already
    while True:
        stopping = stop.wait(RESUME_COMMIT_POLL)
        marker, found = read(), snaps()
        new_snaps = sorted(set(found) - committed_snaps)
        new_point, new_final = marker is not None and marker != committed, final() and not committed_final
        if new_point or new_snaps or new_final:
            point = json.loads(marker) if new_point else None
            what = lambda detail: " and ".join(([f"resume point {point['step']}" + (f" ({point['dir']})" if detail else "")] if point else [])
                                               + ([f"snapshot(s) at step(s) {new_snaps}"] if new_snaps else []) + (["the final checkpoint"] if new_final else []))
            started = time.time()
            try:
                runs_volume.commit()
            except Exception as error:   # noqa: BLE001 - loud, and retried: the point stays on disk until it is committed
                print(f"!!! {what(False)} NOT committed to the runs volume ({type(error).__name__}: {str(error)[:300]}); "
                      f"retrying in {RESUME_COMMIT_POLL} s; a timeout before then continues from the previous point", flush=True)
                if stopping: break
                continue
            committed, committed_snaps, committed_final = marker, committed_snaps | set(new_snaps), committed_final or new_final
            print(f"[volume] committed {what(True)} to the runs volume in {time.time() - started:.0f} s", flush=True)
            if mirror:
                for s in new_snaps: mirror(found[s])
                if new_final: mirror(final_dir)
        if stopping: break


def spawn_mirror(paths, repo):
    """Start run_mirror for committed checkpoint directories on the volume; a failure to start it is logged, never raised
    (the volume copy is primary; modal_app.py::mirror_snapshots uploads them later)."""
    try:
        call = run_mirror.spawn([str(p) for p in paths], repo)
        print(f"[mirror] {len(paths)} checkpoint(s) -> private {repo}: call {call.object_id}", flush=True)
    except Exception as error:   # noqa: BLE001
        print(f"!!! [mirror] could not start the upload of {[str(p) for p in paths]} to {repo} ({type(error).__name__}: {str(error)[:300]}); "
              "the volume copy is unaffected (retry: modal_app.py::mirror_snapshots)", flush=True)


def admit_study(suite, plan_path, name, gpu, existing, transfer, budget, timeout):
    """Validate a study locally before anything is spawned (name, budget bound against the timeout, plan, uncommitted
    changes) and build the run_trial jobs. Returns (jobs, bound_usd, options: GPU, timeout, retries and the resources a
    full-weight study needs, kev.budget.trial_resources, plus "function": run_full_trial for a full-weight study, whose
    containers need the disk with_options cannot give)."""
    from kev.experiment import load_plan
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("study name must be a simple unique identifier")
    if (ROOT / "runs" / name).exists():
        raise FileExistsError("choose a new study name; existing results are immutable")
    trials = load_plan(ROOT / suite, ROOT / plan_path) if plan_path else []
    full_ft = any(t.get("full_ft") for t in trials)
    if not 60 <= timeout <= MAX_TIMEOUT[full_ft] or not 0 < budget <= MAX_BUDGET[full_ft]:   # kev.budget: 8 h / $250, full-weight 24 h / $1,000
        raise ValueError(f"timeout must be 60..{MAX_TIMEOUT[full_ft]} seconds and study budget <= ${MAX_BUDGET[full_ft]}")
    upper = compute_bound(gpu, timeout, len(trials) + len(existing), full_ft)
    if upper > budget:
        raise ValueError(f"timeout-based compute bound ${upper:.2f} exceeds budget ${budget:.2f}")
    print(f"Compute admission bound ${upper:.2f}; excludes image build, startup, and storage; "
          + (f"counts {FULL_FT_RETRIES} retries per trial (a timed-out full-weight trial continues from its resume point)." if full_ft else "no automatic retries."), flush=True)
    commit, sources = local_git_commit(), local_source_hashes()
    if subprocess.run(["git", "status", "--porcelain", "kev", "evals"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("warning: kev/ or evals/ has uncommitted changes; provenance records the last commit, not the working tree", flush=True)
    entries = [(None, p) for p in existing] + [(t, None) for t in trials]
    cpu, memory = trial_resources(gpu, full_ft)
    retries = modal.Retries(max_retries=FULL_FT_RETRIES, initial_delay=30.0, backoff_coefficient=1.0) if full_ft else 0   # a retry continues the trial
    options = {"gpu": gpu, "timeout": timeout, "retries": retries, "cpu": cpu, "memory": memory, "function": "run_full_trial" if full_ft else "run_trial"}
    return [Job(name, i, Path(ex).name if ex else f"trial-{i}", cfg or {}, suite, sources, commit, ex, transfer) for i, (cfg, ex) in enumerate(entries)], upper, options


def deployed_run_trial(sources, function="run_trial"):
    """run_trial (or run_full_trial) on the *deployed* app (modal deploy modal_app.py), after checking it ships this
    checkout's kev/*.py. Spawns on the ephemeral app die with the local client; the deployed app has no parent to lose."""
    try:
        target = modal.Function.from_name(APP_NAME, function); target.hydrate()
        deployed_sources = modal.Function.from_name(APP_NAME, "remote_source_hashes").remote()
    except Exception as error:
        raise SystemExit(f"deployed app not usable ({type(error).__name__}: {str(error)[:120]}); run `uv run modal deploy modal_app.py` first")
    if deployed_sources != sources:
        changed = sorted(k for k in set(deployed_sources) | set(sources) if deployed_sources.get(k) != sources.get(k))
        raise SystemExit(f"deployed app has different kev/*.py than this checkout ({', '.join(changed)}); run `uv run modal deploy modal_app.py` first")
    return target


def launch_detached(suite, plan_path, name, gpu, existing=(), transfer=None, budget=20.0, timeout=1800):
    """Validate locally, spawn every trial as its own call on the deployed app, record the call ids and return. Results
    land on the volume; `pull --name` collects and ranks them."""
    from kev.suite import write_json
    jobs, upper, options = admit_study(suite, plan_path, name, gpu, existing, transfer, budget, timeout)
    fn = deployed_run_trial(local_source_hashes(), options.pop("function")).with_options(**options)
    calls = [fn.spawn(*job) for job in jobs]
    (ROOT / "runs").mkdir(exist_ok=True)
    write_json(ROOT / "runs" / f"{name}.spawn.json", {"name": name, "calls": {j.label: c.object_id for j, c in zip(jobs, calls)}, "bound_usd": round(upper, 2), "timeout": timeout})
    print(f"spawned study {name}: {len(jobs)} independent trial(s) on {gpu}, bound ${upper:.2f}. Pull later: modal run modal_app.py::pull --name {name}", flush=True)


def launch(suite, plan_path, name, gpu, existing=(), transfer=None, budget=20.0, timeout=1800):
    """Attached variant: run the trials on this app, wait, then pull and rank. Dies with the local client."""
    jobs, _, options = admit_study(suite, plan_path, name, gpu, existing, transfer, budget, timeout)
    fn = {"run_trial": run_trial, "run_full_trial": run_full_trial}[options.pop("function")].with_options(**options, max_containers=24)
    print(f"launching {len(jobs)} trial(s) on {gpu} for study {name}", flush=True)
    results = list(fn.starmap(jobs, return_exceptions=True))
    for job, result in zip(jobs, results):
        print(job.label, result if isinstance(result, Exception) else json.dumps(result), flush=True)
    failures = [r for r in results if isinstance(r, Exception) or "failed" in r]   # a full-weight trial returns its failure (failed_trial)
    if len(failures) == len(results):
        raise SystemExit(f"all {len(results)} trial(s) failed; nothing to pull")
    target = pull_study(name)
    print(f"study pulled to {target}; {len(failures)} failure(s)", flush=True)
    if failures:
        raise SystemExit(1)


def pull_lock(study):
    """One pull of a study at a time: two concurrent pulls (a watcher launching reads for two trials that finished together)
    deleted and re-fetched each other's trial directories. A second pull waits for the first, then refreshes."""
    from kev.suite import file_lock
    return file_lock(ROOT / "runs" / f".pull-{study}.lock")


def pull_study(study, weights=False):
    """Download a study directory from the runs volume into runs/<study> and rank it. A study pulled before all its trials
    finished is refreshed: finished trial directories (with result.json) are kept, unfinished ones are fetched again.
    Full-weight shards and resume points stay on the volume unless `weights` (see `pulled`)."""
    with pull_lock(study):
        return _pull_study(study, weights)


def _pull_study(study, weights=False):
    target = ROOT / "runs" / study
    if not target.exists():
        pull_volume(f"/{study}", target.parent, weights=weights)   # recreates runs/<study>/... locally (gitignored)
    else:
        # a local trial dir without result.json is a copy taken while the trial was still running: replace it
        for p in target.glob("*-trial-*"):
            if p.is_dir() and not (p / "result.json").exists(): shutil.rmtree(p)
        missing = sorted(d for d in volume_names(f"/{study}")[0] if not (target / d).exists())
        for d in missing: pull_volume(f"/{study}/{d}", target, weights=weights)
        (target / "results.jsonl").unlink(missing_ok=True)   # derived from the trials' result.json; aggregate rebuilds it
        running = sorted(p.name for p in target.glob("*-trial-*") if p.is_dir() and not (p / "result.json").exists())
        print(f"{study}: fetched {len(missing)} trial dir(s) {missing or ''}" + (f"; still running (or failed, no result.json): {running}" if running else ""))
    subprocess.run([sys.executable, "-m", "kev.experiment", "--aggregate", "--out", str(target)], check=True, cwd=ROOT)
    return target


def volume_names(path):
    """Names of the entries directly under `path` on the runs volume, by kind: (directories, files)."""
    from modal.volume import FileEntryType
    entries = runs_volume.listdir(path)
    return ({Path(e.path).name for e in entries if e.type == FileEntryType.DIRECTORY}, {Path(e.path).name for e in entries if e.type == FileEntryType.FILE})


@app.local_entrypoint()
def study(suite: str, plan: str, name: str, gpu: str = GPU, existing: str = "", transfer: str = "", budget: float = 20.0, timeout: int = 1800, detached: bool = True):
    """detached (default): every trial is spawned on the deployed app and the command returns; `pull --name` afterwards.
    detached=False runs attached (pulls automatically, but dies with the local client)."""
    if detached: launch_detached(suite, plan, name, gpu, [e for e in existing.split(",") if e], transfer or None, budget, timeout)
    else: launch(suite, plan, name, gpu, [e for e in existing.split(",") if e], transfer or None, budget, timeout)


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 49152), retries=0, timeout=7200,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_resume(study, trial, suite, transfer, expected_sources, git_commit):
    """Finish calibration/development/transfer scoring for an interrupted trial whose checkpoint is complete."""
    from kev.experiment import resume_trial
    os.environ["KEV_GIT_COMMIT"] = git_commit
    out = Path(RUNS_MOUNT) / study / trial
    runs_volume.reload()
    try:
        report, _ = resume_trial(Path("/root") / suite, out, expected_sources, "cuda", Path("/root") / transfer if transfer else None)
    finally:
        runs_volume.commit()
    return {"trial": trial, "objective": report["objective"], "transfer_acc": (report.get("transfer") or {}).get("clean", {}).get("acc")}


@app.local_entrypoint()
def resume(study: str, suite: str, transfer: str = "evals/v4/transfer-v4", gpu: str = GPU, timeout: int = 28800):
    """Spawn evaluation for every trial in a study that has checkpoint/head.pt but no result.json, and continue every
    unfinished full-weight trial (no head.pt, no failed.json) from its last resume point on the deployed run_trial."""
    fn = modal.Function.from_name(APP_NAME, "run_resume").with_options(gpu=gpu)
    sources, commit = local_source_hashes(), local_git_commit()
    for t in sorted(volume_names(f"/{study}")[0]):
        dirs, files = volume_names(f"/{study}/{t}")
        finished = "checkpoint" in dirs and "head.pt" in volume_names(f"/{study}/{t}/checkpoint")[1]
        if finished and "result.json" not in files:
            c = fn.spawn(study, t, suite, transfer, sources, commit); print(f"resuming {study}/{t}: call {c.object_id}")
        elif not finished and not {"result.json", "failed.json"} & files and "provenance.json" in files:
            config = json.loads(b"".join(runs_volume.read_file(f"/{study}/{t}/provenance.json")))["config"]
            if not config.get("full_ft"): print(f"skip {study}/{t}: unfinished, not full-weight"); continue
            index, label = t.split("-", 1)
            cpu, memory = trial_resources(gpu, True)
            c = deployed_run_trial(sources, "run_full_trial").with_options(gpu=gpu, cpu=cpu, memory=memory, timeout=timeout).spawn(study, int(index), label, config, suite, sources, commit, None, transfer or None)
            print(f"continuing {study}/{t} from its last resume point: call {c.object_id}")
        else:
            print(f"skip {study}/{t}: {'has result' if 'result.json' in files else 'failed' if 'failed.json' in files else 'no finished checkpoint'}")


@app.local_entrypoint()
def pull(name: str, weights: bool = False):
    """Pull a finished (or partially finished) study from the volume and rank the trials that have a result.json. Without
    --weights, full-weight backbone shards (checkpoint/ and snapshots/) and resume points stay on the volume; reads and
    benchmarks run on the volume paths (/runs/<study>/<trial>/checkpoint, .../snapshots/step-<N>/checkpoint)."""
    target = pull_study(name, weights)
    print(f"pulled {target}", flush=True)


@app.local_entrypoint()
def locked_test(trial: str, name: str, decision: str = "evals/v4/decision-v4", transfer: str = "evals/v4/transfer-v4", gpu: str = GPU, redo_interrupted: bool = False,
                timeout: int = 3600, memory_mb: int = 49152):
    """One locked-test read for a promoted trial (path under the runs volume, e.g. v4-4b-baseline/01-trial-1). A 27B needs
    --gpu H200 --timeout 14400 --memory-mb 131072 (its bf16 weights are staged through host memory while loading)."""
    from kev.suite import read_json
    target = ROOT / "runs/locked" / name
    if (target / "summary.json").exists() and all(k in read_json(target / "summary.json")["suites"] for k in ("decision", "transfer")):
        raise FileExistsError(f"{target} is complete; the locked test is read once per candidate")
    fn = modal.Function.from_name(APP_NAME, "run_locked_test").with_options(gpu=gpu, timeout=timeout, memory=(32768, max(32768, memory_mb)))
    summary = fn.remote(trial, name, {"decision": decision, "transfer": transfer}, local_git_commit(), redo_interrupted)
    if target.exists(): shutil.rmtree(target)   # local copy only; the volume is the record
    target.parent.mkdir(parents=True, exist_ok=True)
    pull_volume(f"/locked/{name}", target.parent)
    print(json.dumps({k: {"acc": v["clean"]["acc"], "brier": v["clean"]["brier"]} for k, v in summary["suites"].items()}, indent=1))


@app.local_entrypoint()
def smoke(gpu: str = GPU):
    launch("evals/smoke-v1", "experiments/smoke.json", "smoke", gpu)


@app.local_entrypoint()
def evaluate(run: str, suite: str, name: str, gpu: str = GPU, transfer: str = ""):
    """Score an existing checkpoint (Hub id, or a path under the runs volume) on a suite's development partition."""
    launch(suite, None, name, gpu, [run], transfer or None)
