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
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
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
# full-weight attempt leases (TrialLease), apart from the runs volume: a heartbeat commits every minute, and a commit of the
# runs volume would carry the half-written shards of a resume point in progress with it
leases_volume = modal.Volume.from_name("kev-leases", create_if_missing=True)
LEASES_MOUNT = "/leases"
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
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache, LEASES_MOUNT: leases_volume}, secrets=secrets)
def run_full_trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None, attempt=None):
    """A full-weight trial: run_trial with the disk its resume points need (kev.budget.trial_disk; with_options cannot set it)
    and the attempt lease (TrialLease; `attempt` = {"nonce", "number"} from the ledger entry that spawned it)."""
    return trial(study, index, label, config, suite, expected_sources, git_commit, existing, transfer, attempt)


def trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None, attempt=None):
    """One trial in one container. `existing` is a checkpoint path on the runs volume or a Hub id (legacy scoring). A
    full-weight trial whose directory exists already is its next attempt (continue_full_trial spawned it after a timeout,
    for `kev.rounds watch` or `resume`): it continues from its last resume point, unless an earlier attempt failed with an
    error (failed.json). A full-weight attempt first takes the trial's lease (TrialLease) and returns {"refused": ...}
    without touching the trial when another attempt's container may still be running."""
    from kev.experiment import source_hashes

    os.environ["KEV_GIT_COMMIT"] = git_commit
    if source_hashes() != expected_sources:
        raise RuntimeError("container received different kev/*.py than the launcher hashed")
    lease, beating = None, threading.Event()
    if config.get("full_ft"):
        lease = TrialLease(Path(LEASES_MOUNT) / study / f"{index:02d}-{label}" / "attempt.json", leases_volume, attempt, modal.current_function_call_id())
        refusal = lease.acquire()
        if refusal:
            print(f"!!! [{label}] refused: {refusal}", flush=True)
            return {"label": label, "refused": refusal}   # no failed.json: the trial is the other attempt's
        threading.Thread(target=heartbeat, args=(lease, beating), daemon=True).start()
    try:
        return run_attempt(study, index, label, config, suite, expected_sources, existing, transfer)
    finally:
        if lease:   # last: while the runs volume is still being committed above, the heartbeat keeps the lease fresh
            beating.set(); lease.end()


def run_attempt(study, index, label, config, suite, expected_sources, existing, transfer):
    """trial()'s work once the attempt may run: train or continue, score, commit the volumes."""
    import torch
    from kev.experiment import continue_trial, execute_trial
    from kev.suite import write_json
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
    committer = None
    if config.get("full_ft"):   # snapshot_hub_repo: a private Hub mirror of each committed snapshot and the final checkpoint (off unless set)
        watcher = VolumeWatcher(out / "checkpoint" / "resume", out / "snapshots", out / "checkpoint", mirror_to(config.get("snapshot_hub_repo")))
        committer = threading.Thread(target=commit_resume_points, args=(watcher, stop), daemon=True); committer.start()
    try:
        report, _ = (continue_trial(Path("/root") / suite, out, expected_sources, "cuda", transfer) if again
                     else execute_trial(config or {}, Path("/root") / suite, out, expected_sources, "cuda", existing, transfer))
    except Exception as error:   # a timeout kills the container before it gets here
        if out.exists(): write_json(out / "failed.json", {"error": f"{type(error).__name__}: {str(error)[:2000]}"})
        if config.get("full_ft"): return failed_trial(label, out)   # returned, not raised: an error is final, never continued
        raise
    finally:
        stop.set()
        if committer and committer.is_alive(): committer.join()
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


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_script(script, name, args=""):
    """scripts/<script> --out /runs/scripts/<name> <args>: a one-off GPU measurement that writes report.json and has no
    entrypoint of its own (scripts/longdoc_serving.py)."""
    out = Path(RUNS_MOUNT) / "scripts" / name
    return run_tool([sys.executable, f"/root/scripts/{script}", "--out", out, *args.split()], out, block=None)


@app.local_entrypoint()
def script(script: str, name: str, gpu: str = GPU, args: str = "", timeout: int = 3600):
    report = run_script.with_options(gpu=gpu, timeout=timeout).remote(script, name, args)
    print(json.dumps(report, indent=1))
    pull_volume(f"/scripts/{name}", ROOT / "runs")


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

    def copy_out():
        for f in ("report.json", "train.log", "checkpoint/training_metrics.json", *(p.name for p in scratch.glob("train-*.log"))):   # train-<length>-<attempt>.log: --state_tokens
            if (scratch / f).exists(): shutil.copy(scratch / f, out / Path(f).name)
        runs_volume.commit()

    def mirror(stop):   # a --state_tokens probe writes its report after every length; a timeout would lose what is only on scratch
        while not stop.wait(120): copy_out()
    stop = threading.Event(); threading.Thread(target=mirror, args=(stop,), daemon=True).start()
    try:
        subprocess.run([sys.executable, "/root/scripts/sft_probe.py", "--base", base, "--revision", revision, "--gpu", gpu, "--out", str(scratch),
                        "--records", str(records), "--train", train, "--check_load", str(check_load), "--resume_dir", str(out / "resume"), *shlex.split(flags)],
                       check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        stop.set()
        shutil.rmtree(out / "resume", ignore_errors=True)
        copy_out(); hf_cache.commit()
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
def run_interpolate(sft, alphas, names, base, revision, study, toward=None, blend_head=False):
    """scripts/interpolate_checkpoint.py on a CPU container: one checkpoint per alpha at /runs/<study>/<name>/checkpoint,
    the runs volume committed after each."""
    sys.path.insert(0, "/root")
    from scripts.interpolate_checkpoint import interpolate as write_interpolations
    runs_volume.reload()
    try:
        return write_interpolations(sft, alphas, [Path(RUNS_MOUNT) / study / n / "checkpoint" for n in names], base, revision,
                                    on_done=lambda report: runs_volume.commit(), log=lambda m: print(m, flush=True),
                                    toward=toward or None, blend_head=blend_head)
    finally:
        runs_volume.commit(); hf_cache.commit()


@app.local_entrypoint()
def interpolate(sft: str, prefix: str, alphas: str = "0.85,0.70,0.50", base: str = KEV_27B_BASE[0], revision: str = KEV_27B_BASE[1],
                study: str = "r20-wise", timeout: int = INTERPOLATE_TIMEOUT, toward: str = "", blend_head: bool = False):
    """WiSE-FT checkpoints of a full-weight SFT checkpoint on the runs volume (--sft /runs/<trial>/checkpoint) with its base,
    or with --toward another checkpoint of the same base and revision (a /runs/... checkpoint or a Hub id@rev; full weights,
    or a LoRA adapter merged in fp32; --blend-head blends the pointer heads too, else the SFT head is kept):
    /runs/<study>/<prefix>-w<alpha x 100>/checkpoint for each alpha (weight on the SFT backbone). Refuses unless the SFT
    run's base is --base @ --revision (Kev-27B's by default). Reports land in runs/<study>/<name>/interpolation.json."""
    sys.path.insert(0, str(ROOT))
    from kev.checkpoint import is_hub_id
    from scripts.interpolate_checkpoint import weight_label
    if not sft.startswith(f"{RUNS_MOUNT}/"): raise SystemExit(f"--sft is a checkpoint on the runs volume ({RUNS_MOUNT}/...), not {sft}")
    if toward and not (toward.startswith(f"{RUNS_MOUNT}/") or (is_hub_id(toward) and "@" in toward)):
        raise SystemExit(f"--toward is a checkpoint on the runs volume ({RUNS_MOUNT}/...) or a pinned Hub id (repo@revision), not {toward}")
    if blend_head and not toward: raise SystemExit("--blend-head needs --toward (the base has no pointer head)")
    values = [float(a) for a in alphas.split(",")]
    names = [f"{prefix}-w{weight_label(a)}" for a in values]
    written = volume_names(f"/{study}")[0] if study in volume_names("/")[0] else set()
    taken = [n for n in names if n in written and "checkpoint" in volume_names(f"/{study}/{n}")[0]]
    if taken: raise SystemExit(f"/{study}/{taken} hold checkpoints already; interpolations are written once")
    print(f"admission bound ${interpolation_bound(timeout):.2f} ({INTERPOLATE_CPU} CPU, {INTERPOLATE_MEMORY[1] // 1024} GiB, {timeout} s, no GPU)", flush=True)
    call = run_interpolate.with_options(timeout=timeout).spawn(sft, values, names, base, revision, study, toward or None, blend_head)
    print(f"spawned interpolation of {sft}{f' toward {toward}' if toward else ''}{' (heads blended)' if blend_head else ''} -> /{study}/{names}: call {call.object_id}", flush=True)
    for report in call.get():
        name = Path(report["checkpoint"]).parent.name
        (ROOT / "runs" / study / name).mkdir(parents=True, exist_ok=True)
        pull_volume(f"/{study}/{name}/interpolation.json", ROOT / "runs" / study / name)
        print(f"{name}: alpha {report['alpha']}, {report['tensors']} tensors, weights {report['weights_sha256'][:12]}, {report['seconds']} s", flush=True)


MERGE_TIMEOUT = 7200   # round 25's Kev-27B merge: base load + merge + a 51 GB write + hash, well under an hour expected


@app.function(image=image, cpu=INTERPOLATE_CPU, memory=INTERPOLATE_MEMORY, retries=0, timeout=MERGE_TIMEOUT,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_merge_adapter(lora, out, like=None):
    """scripts/merge_lora_checkpoint.py on a CPU container: <out>/checkpoint on the runs volume, then one commit (timed)."""
    sys.path.insert(0, "/root")
    from scripts.merge_lora_checkpoint import merge
    runs_volume.reload()
    started = time.time()
    try:
        report = merge(lora, out, like, log=lambda m: print(f"[{time.time() - started:7.1f} s] {m}", flush=True))
    finally:
        commit = time.time(); runs_volume.commit()
        print(f"runs volume commit: {time.time() - commit:.1f} s", flush=True)
    return {**report, "commit_seconds": round(time.time() - commit, 1), "container_seconds": round(time.time() - started, 1)}


@app.local_entrypoint()
def merge_adapter(lora: str, out: str, like: str = ""):
    """A LoRA checkpoint (a pinned Hub id repo@revision or a /runs/... directory) merged into its base as a full-weight
    checkpoint at <out>/checkpoint on the runs volume (scripts/merge_lora_checkpoint.py), e.g. round 25's initialization
    --lora jaredpalmer/kev-27b@<sha> --out /runs/r25-init/kev-27b-merged [--like /runs/<full-weight trial>/checkpoint].
    The report lands in runs/<out without /runs>/merge.json."""
    sys.path.insert(0, str(ROOT))
    from kev.checkpoint import is_hub_id
    if not out.startswith(f"{RUNS_MOUNT}/"): raise SystemExit(f"--out is a directory on the runs volume ({RUNS_MOUNT}/...), not {out}")
    if not (lora.startswith(f"{RUNS_MOUNT}/") or (is_hub_id(lora) and "@" in lora)):
        raise SystemExit(f"--lora is a checkpoint on the runs volume ({RUNS_MOUNT}/...) or a pinned Hub id (repo@revision), not {lora}")
    rel = out.removeprefix(RUNS_MOUNT).rstrip("/")
    try: existing = volume_names(rel)[0]
    except Exception: existing = set()   # nothing there yet (listdir of a missing path raises)
    if "checkpoint" in existing: raise SystemExit(f"{rel}/checkpoint exists; merges are written once")
    print(f"admission bound ${interpolation_bound(MERGE_TIMEOUT):.2f} ({INTERPOLATE_CPU} CPU, {INTERPOLATE_MEMORY[1] // 1024} GiB, {MERGE_TIMEOUT} s, no GPU)", flush=True)
    call = run_merge_adapter.spawn(lora, out, like or None)
    print(f"spawned merge of {lora} -> {out}/checkpoint: call {call.object_id}", flush=True)
    report = call.get()
    (ROOT / "runs" / rel.lstrip("/")).mkdir(parents=True, exist_ok=True)
    pull_volume(f"{rel}/merge.json", ROOT / "runs" / rel.lstrip("/"))
    print(json.dumps(report, indent=1), flush=True)


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
READ_TIMEOUTS = (("longstate", 7200), ("documents", 5400), ("transfer-v9", 3600), ("longdoc", 10800))
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


RELEASE_CPU, RELEASE_MEMORY, RELEASE_TIMEOUT = 8, (8192, 32768), 4 * 3600   # copy + hash + upload of a ~51 GB checkpoint, no GPU


@app.function(image=image, cpu=RELEASE_CPU, memory=RELEASE_MEMORY, retries=0, timeout=RELEASE_TIMEOUT, volumes={RUNS_MOUNT: runs_volume})
def run_release_copy(src, dst, expect=None):
    """scripts/release_checkpoint.py: copy a checkpoint on the runs volume to a new release directory (never over one) and
    check the weights hash of the copy against the source's (and `expect`). CPU only."""
    sys.path.insert(0, "/root")
    from scripts.release_checkpoint import copy_checkpoint
    runs_volume.reload()
    report = copy_checkpoint(src, dst, expect, log=lambda m: print(m, flush=True))
    runs_volume.commit()
    return report


@app.local_entrypoint()
def release_copy(src: str, dst: str, expect: str = ""):
    """Stage a release checkpoint: --src /runs/<...>/checkpoint --dst /runs/release/<name>/checkpoint (refused if it exists),
    weights sha256 of the copy checked against the source's and --expect. The source is only read."""
    for p in (src, dst):
        if not p.startswith(f"{RUNS_MOUNT}/") or not p.endswith("/checkpoint"): raise SystemExit(f"{p} is not a /runs/.../checkpoint directory")
    print(json.dumps(run_release_copy.remote(src, dst, expect or None), indent=1), flush=True)


@app.function(image=image, cpu=RELEASE_CPU, memory=RELEASE_MEMORY, retries=0, timeout=RELEASE_TIMEOUT, volumes={RUNS_MOUNT: runs_volume},
              secrets=[modal.Secret.from_name(MIRROR_SECRET)])
def run_release_publish(run, repo, card, message, public=False, replace=False):
    """kev.publish of a release checkpoint on the runs volume (`card`: the model card's text). Private by default: kev.publish
    --private creates a missing repo private and refuses one that is not. `public` (the approved release) drops --private;
    `replace` passes --replace (the target revision then holds exactly this upload). CPU only; HF_TOKEN from MIRROR_SECRET."""
    import subprocess as sp
    import tempfile
    runs_volume.reload()
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f: f.write(card)
    cmd = [sys.executable, "-m", "kev.publish", "--run", run, "--repo", repo, "--card", f.name, "--message", message,
           *([] if public else ["--private"]), *(["--replace"] if replace else [])]
    out = sp.run(cmd, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"}, capture_output=True, text=True)
    print(out.stdout[-4000:], out.stderr[-4000:], flush=True)
    if out.returncode: raise RuntimeError(f"kev.publish exited {out.returncode}")
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo, repo_type="model", files_metadata=True)
    return {"repo": repo, "private": info.private, "commit": info.sha, "files": sorted(s.rfilename for s in info.siblings),
            "lfs_sha256": {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs}}


@app.local_entrypoint()
def release_publish(run: str, repo: str, card: str, message: str, public: bool = False, confirm_public: str = "", replace: bool = False):
    """Upload a staged release checkpoint (--run /runs/release/<name>/checkpoint) to a PRIVATE Hub repo with --card as its
    README, from a CPU container (the weights never leave Modal and Hugging Face). An approved public release passes
    --public --confirm-public <repo> (the repo named twice), usually with --replace after tagging the previous version."""
    if not run.startswith(f"{RUNS_MOUNT}/release/"): raise SystemExit(f"--run is a staged release checkpoint under {RUNS_MOUNT}/release/, not {run}")
    if public and confirm_public != repo: raise SystemExit(f"--public uploads to a public repo; repeat it as --confirm-public {repo}")
    print(json.dumps(run_release_publish.remote(run, repo, Path(card).read_text(encoding="utf-8"), message, public, replace), indent=1), flush=True)


RELEASE_VERIFY_DISK = 524288   # MiB (Modal's minimum explicit request): a fresh HF cache for a 27B release (51 GB) and, for a LoRA revision, its base (~55 GB)


@app.function(image=image, gpu=GPU, cpu=4, memory=(32768, 131072), retries=0, timeout=3 * 3600, ephemeral_disk=RELEASE_VERIFY_DISK)
def run_release_verify(jobs):
    """What an anonymous user gets from a public release: no Modal secret, no kev-hf-cache volume, a fresh HF_HOME on the
    scratch disk and no Hugging Face token. For each "<hub id[@rev]>@<suite>" job: kev.checkpoint's resolution of the id
    (layout by the loader rule, head.pt temperature, weights and head sha256), then kev.benchmark on the suite's
    development partition as a separate process. -> {"hf_home", "jobs": [{checkpoint (with the token it saw: None), report, rows}]}."""
    import tempfile
    home = tempfile.mkdtemp(prefix="hf-anon-", dir="/tmp")
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_") or k == "HF_HUB_DISABLE_PROGRESS_BARS"}
    env.update(HF_HOME=home, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONPATH="/root", TRITON_CACHE_DIR=f"{home}/triton-cache")
    resolve = ("import json, sys; from pathlib import Path; from huggingface_hub import get_token; from kev.checkpoint import Checkpoint; "
               "from kev.suite import digest; ck = Checkpoint(sys.argv[1]); print(json.dumps({'requested': sys.argv[1], 'token': get_token(), "
               "'path': ck.path, 'layout': 'full' if ck.full else 'lora', 'temperature': ck.meta.temperature, 'weights_sha256': ck.weights_sha256(), "
               "'head_sha256': digest(ck.file('head.pt')), 'files': sorted(p.name for p in Path(ck.path).iterdir())}))")
    from kev.suite import read_json
    out = {"hf_home": home, "jobs": []}
    for job in jobs:   # every step in a fresh process with the anonymous environment (nothing imported here saw the image's HF_HOME)
        run, suite = job.rsplit("@", 1)
        info = json.loads(subprocess.run([sys.executable, "-c", resolve, run], check=True, cwd="/root", env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1])
        print(json.dumps(info), flush=True)
        dest = Path(tempfile.mkdtemp(prefix="bench-", dir="/tmp")) / "out"   # kev.benchmark refuses an existing --out
        subprocess.run([sys.executable, "-m", "kev.benchmark", "--run", run, "--suite", f"/root/{suite}", "--out", str(dest), "--device", "cuda"],
                       check=True, cwd="/root", env=env)
        out["jobs"].append({"job": job, "checkpoint": info, "report": read_json(dest / "report.json"), "rows": read_json(dest / "rows.json")})
    return out


@app.local_entrypoint()
def release_verify(jobs: str, out: str, gpu: str = "H200"):
    """Anonymous verification of a public release (run_release_verify): --jobs "jaredpalmer/kev-27b@evals/external/semif-v1,..."
    (comma-separated; `@rev` pins a revision: jaredpalmer/kev-27b@v1-lora@evals/...), each job's report.json + rows.json
    written to <out>/<n>-<repo>/ and the resolutions to <out>/verify.json."""
    from kev.suite import write_json
    result = run_release_verify.with_options(gpu=gpu).remote([j.strip() for j in jobs.split(",") if j.strip()])
    root = Path(out)
    for i, job in enumerate(result["jobs"]):
        d = root / f"{i}-{job['checkpoint']['requested'].replace('/', '_').replace('@', '_')}"
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "report.json", job.pop("report")); write_json(d / "rows.json", job.pop("rows"))
        job["dir"] = str(d)
    write_json(root / "verify.json", result)
    print(json.dumps(result, indent=1), flush=True)


@app.local_entrypoint()
def smoke_base(base: str, revision: str, gpu: str = "H200"):
    """Memory and step-time check for a base that has not been trained yet (LoRA footprint, which modules it hits, peak GB)."""
    print(json.dumps(run_smoke_base.with_options(gpu=gpu).remote(base, revision), indent=1))


def inkling_weight_smoke(report_path: str | Path | None = None) -> dict:
    """Direct Python wrapper for the isolated Inkling Sandbox smoke; do not invoke through Modal CLI."""
    from scripts.inkling_weight_smoke_launch import launch

    if report_path is None:
        return launch(app_name=APP_NAME)
    return launch(Path(report_path), app_name=APP_NAME)


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
    """The result of a full-weight trial that failed with an error (failed.json). Returned rather than raised, so that it
    cannot pass for an interruption: only a timeout is continued (continue_full_trial, from a resume point), and a retried
    call would find failed.json and report it again; kev.rounds.poll_modal and launch() read "failed" as the trial's failure."""
    return {"label": label, "failed": json.loads((out / "failed.json").read_text(encoding="utf-8"))["error"]}


RESUME_COMMIT_POLL = 15   # seconds between looks at a training trial's latest.json


class VolumeWatcher:
    """What a full-weight trial has committed to the runs volume, and one look for more (poll). A timeout kills the
    container without running trial()'s `finally`, and the next attempt can only continue from a committed point and keep
    committed snapshots, so the runs volume is committed each time the trainer completes a resume point (its latest.json
    changes), a snapshot (kev.full_ft.completed_snapshots under `snapshot_dir` gains a step) or the final checkpoint
    (`final_dir`/training_metrics.json, written last, appears). What is on disk when the watcher is built (what a new
    container finds on the volume) counts as committed already, so build it before training starts. `mirror(path)`, when
    given, is called once for each newly committed snapshot and final checkpoint, after the commit that includes it
    (mirror_to: the private Hub copy)."""

    def __init__(self, resume_dir, snapshot_dir=None, final_dir=None, mirror=None):
        self.latest, self.snapshot_dir, self.final_dir, self.mirror = resume_dir / "latest.json", snapshot_dir, final_dir, mirror
        self.committed, self.committed_snaps, self.committed_final = self._marker(), set(self._snaps()), self._final()

    def _marker(self):
        return self.latest.read_text(encoding="utf-8") if self.latest.exists() else None

    def _snaps(self):
        from kev.full_ft import completed_snapshot_dirs
        return completed_snapshot_dirs(self.snapshot_dir) if self.snapshot_dir else {}

    def _final(self):
        return bool(self.final_dir) and (self.final_dir / "training_metrics.json").exists()

    def poll(self):
        """One look: commit the runs volume if anything new is complete, then mirror what that commit included. A failed
        commit is reported loudly and leaves everything uncommitted, so the next look tries again."""
        marker, found = self._marker(), self._snaps()
        new_snaps = sorted(set(found) - self.committed_snaps)
        new_point, new_final = marker is not None and marker != self.committed, self._final() and not self.committed_final
        if not (new_point or new_snaps or new_final): return
        point = json.loads(marker) if new_point else None
        what = lambda detail: " and ".join(([f"resume point {point['step']}" + (f" ({point['dir']})" if detail else "")] if point else [])
                                           + ([f"snapshot(s) at step(s) {new_snaps}"] if new_snaps else []) + (["the final checkpoint"] if new_final else []))
        started = time.time()
        try:
            runs_volume.commit()
        except Exception as error:   # noqa: BLE001 - loud, and retried: the point stays on disk until it is committed
            print(f"!!! {what(False)} NOT committed to the runs volume ({type(error).__name__}: {str(error)[:300]}); "
                  f"retrying in {RESUME_COMMIT_POLL} s; a timeout before then continues from the previous point", flush=True)
            return
        self.committed, self.committed_snaps, self.committed_final = marker, self.committed_snaps | set(new_snaps), self.committed_final or new_final
        print(f"[volume] committed {what(True)} to the runs volume in {time.time() - started:.0f} s", flush=True)
        if self.mirror:
            for s in new_snaps: self.mirror(found[s])
            if new_final: self.mirror(self.final_dir)


def commit_resume_points(watcher, stop):
    """While a full-weight trial trains (in a thread of trial()), look every RESUME_COMMIT_POLL seconds (VolumeWatcher.poll);
    after `stop` it looks once more, so the final checkpoint written right before the trial stops is committed too."""
    while True:
        stopping = stop.wait(RESUME_COMMIT_POLL)
        watcher.poll()
        if stopping: break


LEASE_SETTLE = 10   # seconds between writing a lease and reading it back (two attempts that claimed it at once: one loses)


class TrialLease:
    """A full-weight attempt's claim on its trial: <leases>/<study>/<trial>/attempt.json on the kev-leases volume, written
    by the attempt itself: {"nonce", "attempt", "call_id", "started", "heartbeat", "ended"}. kev.budget.lease_state reads
    it: a lease that is not ended and has a heartbeat younger than kev.budget.LEASE_STALE means that attempt's container may
    still be writing the trial (training, or committing the runs volume in the 30 s after its timeout). acquire() refuses
    then; continue_full_trial waits for such a lease to end or go stale before it spawns. `volume` needs reload/commit
    (leases_volume, or a fake); `clock`/`sleep` are injectable for tests."""

    def __init__(self, path, volume, attempt, call_id, clock=time.time, sleep=time.sleep):
        attempt = attempt or {}
        self.path, self.volume, self.clock, self.sleep = Path(path), volume, clock, sleep
        self.nonce, self.number, self.call_id = attempt.get("nonce") or uuid.uuid4().hex, attempt.get("number"), call_id
        self.record, self.lock = None, threading.Lock()   # the heartbeat thread and end() both write

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def _write(self):
        from kev.suite import write_json
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, self.record, atomic=True)
        self.volume.commit()

    def acquire(self):
        """None when this attempt holds the lease (written and committed, and still ours after LEASE_SETTLE), or why not."""
        from kev.budget import LEASE_STALE, lease_state
        self.volume.reload()
        other = self.read()
        if lease_state(other, self.clock()) == "fresh" and other.get("nonce") != self.nonce:
            return (f"attempt {other.get('attempt')} (call {other.get('call_id')}) holds the trial: its heartbeat is {self.clock() - other['heartbeat']:.0f} s old, "
                    f"under the {LEASE_STALE} s after which its container counts as gone")
        now = self.clock()
        self.record = {"nonce": self.nonce, "attempt": self.number, "call_id": self.call_id, "started": now, "heartbeat": now, "ended": None,
                       "previous": {k: other.get(k) for k in ("nonce", "attempt", "call_id", "heartbeat", "ended")} if other else None}
        self._write()
        self.sleep(LEASE_SETTLE)
        self.volume.reload()
        mine = self.read()
        if not mine or mine.get("nonce") != self.nonce:
            return f"lost the lease to attempt {(mine or {}).get('attempt')} (call {(mine or {}).get('call_id')}), which claimed it at the same time"
        return None

    def beat(self):
        """Refresh the heartbeat (and commit it); a lease another attempt took, or this one ended, is left alone."""
        with self.lock:
            if self.record["ended"]: return False
            self.volume.reload()
            current = self.read()
            if current and current.get("nonce") != self.nonce:
                print(f"!!! [lease] {self.path} now belongs to attempt {current.get('attempt')} (call {current.get('call_id')}); not refreshing it", flush=True)
                return False
            self.record["heartbeat"] = self.clock(); self._write()
            return True

    def end(self):
        """Mark the lease ended (a clean exit: the next attempt need not wait for it to go stale)."""
        try:
            with self.lock:
                self.volume.reload()
                current = self.read()
                if current and current.get("nonce") != self.nonce: return
                self.record["ended"] = self.clock(); self._write()
        except Exception as error:   # noqa: BLE001 - the lease then goes stale on its own
            print(f"!!! [lease] could not mark {self.path} ended ({type(error).__name__}: {str(error)[:200]}); it goes stale in kev.budget.LEASE_STALE", flush=True)


def heartbeat(lease, stop):
    """TrialLease.beat every kev.budget.LEASE_HEARTBEAT seconds until `stop` (a thread of trial(), apart from the runs
    volume's commits); a failed beat is reported and tried again at the next one."""
    from kev.budget import LEASE_HEARTBEAT
    while not stop.wait(LEASE_HEARTBEAT):
        try:
            lease.beat()
        except Exception as error:   # noqa: BLE001
            print(f"!!! [lease] heartbeat not committed ({type(error).__name__}: {str(error)[:200]}); next try in {LEASE_HEARTBEAT} s", flush=True)


def mirror_to(repo):
    """The VolumeWatcher `mirror` of a trial whose plan sets snapshot_hub_repo: one run_mirror spawn per committed checkpoint."""
    return (lambda path: spawn_mirror([path], repo)) if repo else None


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
          + (f"counts {1 + FULL_FT_RETRIES} attempts per trial (a timed-out full-weight trial is continued from its resume point by "
             "`kev.rounds watch` or `modal_app.py::resume --trial`, each continuation a new call)." if full_ft else "no automatic retries."), flush=True)
    commit, sources = local_git_commit(), local_source_hashes()
    if subprocess.run(["git", "status", "--porcelain", "kev", "evals"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("warning: kev/ or evals/ has uncommitted changes; provenance records the last commit, not the working tree", flush=True)
    entries = [(None, p) for p in existing] + [(t, None) for t in trials]
    cpu, memory = trial_resources(gpu, full_ft)
    # Modal's retries stay off even for a full-weight trial: one call is one attempt, and continuations are counted calls
    # (kev.budget.FULL_FT_RETRIES says why Modal's retries gave round 22 two attempts instead of three)
    options = {"gpu": gpu, "timeout": timeout, "retries": 0, "cpu": cpu, "memory": memory, "function": "run_full_trial" if full_ft else "run_trial", "full_ft": full_ft}
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
    full_ft = options.pop("full_ft")
    fn = deployed_run_trial(local_source_hashes(), options.pop("function")).with_options(**options)
    (ROOT / "runs").mkdir(exist_ok=True)
    # the attempt ledger: `calls` is each trial's current call (the one kev.rounds watch polls), `attempts` every attempt it
    # started, continuations included (kev.budget.trial_attempts counts them against the bound; gpu and timeout are what
    # the bound was admitted for, so a continuation uses them). Each attempt is written pending (a nonce, no call) before its
    # spawn and gets its call id after: a crash in between leaves a pending entry, never an unrecorded attempt.
    record = {"name": name, "calls": {j.label: None for j in jobs}, "attempts": {j.label: [pending_attempt()] for j in jobs}, "bound_usd": round(upper, 2),
              "gpu": gpu, "timeout": timeout, "full_ft": full_ft, "modal_retries": 0}
    write_json(spawn_record(name), record, atomic=True)
    for job in jobs:
        entry = record["attempts"][job.label][0]
        call = fn.spawn(*job, attempt={"nonce": entry["nonce"], "number": 1}) if full_ft else fn.spawn(*job)
        entry["call"] = record["calls"][job.label] = call.object_id
        write_json(spawn_record(name), record, atomic=True)
    print(f"spawned study {name}: {len(jobs)} independent trial(s) on {gpu}, bound ${upper:.2f}. Pull later: modal run modal_app.py::pull --name {name}", flush=True)


def launch(suite, plan_path, name, gpu, existing=(), transfer=None, budget=20.0, timeout=1800):
    """Attached variant: run the trials on this app, wait, then pull and rank. Dies with the local client."""
    jobs, _, options = admit_study(suite, plan_path, name, gpu, existing, transfer, budget, timeout)
    if options.pop("full_ft"): print("attached: a full-weight trial that times out is not continued (run detached, or `resume --trial ... --beyond-bound`)", flush=True)
    fn = {"run_trial": run_trial, "run_full_trial": run_full_trial}[options.pop("function")].with_options(**options, max_containers=24)
    print(f"launching {len(jobs)} trial(s) on {gpu} for study {name}", flush=True)
    results = list(fn.starmap(jobs, return_exceptions=True))
    for job, result in zip(jobs, results):
        print(job.label, result if isinstance(result, Exception) else json.dumps(result), flush=True)
    failures = [r for r in results if isinstance(r, Exception) or "failed" in r or "refused" in r]   # a full-weight trial returns its failure (failed_trial) or refusal (TrialLease)
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


def spawn_record(study):
    return ROOT / "runs" / f"{study}.spawn.json"


def pending_attempt(now=time.time):
    """A ledger entry written before its spawn: the nonce the attempt's lease will carry, no call yet."""
    return {"nonce": uuid.uuid4().hex, "call": None, "at": now()}


def trial_lease(study, trial):
    """The trial's lease (TrialLease) as the kev-leases volume has it now, or None."""
    try:
        return json.loads(b"".join(leases_volume.read_file(f"/{study}/{trial}/attempt.json")))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


LEASE_POLL = 30   # seconds between looks at a fresh lease while a continuation waits for it


def continue_full_trial(study, trial, config, suite, transfer, sources, commit, beyond_bound=None, now=time.time, sleep=time.sleep, wait=None):
    """Spawn the next attempt of a full-weight trial (<NN>-<label> under /runs/<study>): run_full_trial again, which
    continues from the trial's last resume point, or only scores it when its checkpoint is complete
    (kev.experiment.continue_trial). Under a per-study lock, against the study's attempt ledger (runs/<study>.spawn.json):
    1. a pending ledger entry (written before a spawn whose call id was never recorded) is resolved first: adopted when the
       trial's lease carries its nonce (its container started; the lease names the call), refused while younger than
       kev.budget.LEASE_STALE (it may still start), then marked abandoned (still counted: should it start late, its lease
       claim finds the next attempt's fresh lease and it refuses);
    2. refused once kev.budget.trial_attempts has no attempt left, and unless the current call ended by a timeout (or was
       refused by a lease), since a running call means a second attempt would share the directory;
    3. waits up to `wait` (default LEASE_STALE + a minute) for the trial's lease to be ended or stale, so the old
       container is demonstrably gone (its 30 s cancellation grace, its last commits), else refuses;
    4. records a pending entry, spawns with the ledger's GPU and timeout, Modal's retries off and that entry's nonce, then
       records the call id as the trial's current call.
    `beyond_bound` = (gpu, timeout) spawns for a study without a ledger (an attached launch, or a record written before
    the ledger), outside any admission bound, and says so (the lease wait still applies). Returns the call id."""
    from kev.budget import LEASE_STALE, lease_state, trial_attempts
    from kev.rounds import poll_modal
    from kev.suite import file_lock, read_json, write_json
    index, label = trial.split("-", 1)
    wait = LEASE_STALE + 60 if wait is None else wait
    # kev.suite.file_lock serialises ledger changes on this machine only (its docstring predates this third use; kev/suite.py
    # is an evaluator file, so it is not edited here); across machines and containers the trial's lease keeps attempts apart
    with file_lock(ROOT / "runs" / f".continue-{study}.lock"):
        record = read_json(spawn_record(study)) if spawn_record(study).exists() else None
        ledger = record is not None and record.get("modal_retries") == 0 and label in record["calls"]
        if not ledger and not beyond_bound:
            raise SystemExit(f"{study}/{trial}: no attempt ledger for this trial in {spawn_record(study).relative_to(ROOT)} (spawned before the ledger, attached, or "
                             "from another checkout), so a continuation cannot be counted against the study's admission bound; pass --beyond-bound to spawn one anyway")
        save = lambda: write_json(spawn_record(study), record, atomic=True)
        if ledger:
            entries = record["attempts"][label]
            lease = trial_lease(study, trial)
            for entry in entries:   # an attempt whose call id was never recorded, but whose container took the lease
                if entry["call"] is None and lease and lease.get("nonce") == entry["nonce"]:
                    entry["call"] = lease["call_id"]; entry.pop("abandoned", None)
                    if entry is entries[-1] or lease_state(lease, now()) == "fresh": record["calls"][label] = lease["call_id"]
                    save(); print(f"{study}/{trial}: adopted the unrecorded call {lease['call_id']} of a pending attempt from its lease", flush=True)
            last = entries[-1]
            if last["call"] is None and not last.get("abandoned"):
                if now() - last["at"] < LEASE_STALE:
                    raise SystemExit(f"{study}/{trial}: an attempt was spawned {now() - last['at']:.0f} s ago without a recorded call (pending nonce {last['nonce']}); "
                                     f"it may still start, so nothing is spawned until its lease shows it or {LEASE_STALE} s pass")
                last["abandoned"] = now(); save()
                print(f"!!! {study}/{trial}: pending attempt {last['nonce']} never took the lease in {LEASE_STALE} s; counted as abandoned", flush=True)
            used, allowed = trial_attempts(record, label)
            if used >= allowed:
                raise SystemExit(f"{study}/{trial}: {used} of {allowed} attempts used; the admission bound (${record['bound_usd']}) counts no more")
            if last["call"] is not None:
                status = poll_status(last["call"], poll_modal)
                if status not in ("timeout", "refused"):
                    raise SystemExit(f"{study}/{trial}: its current call {last['call']} is {status}; only a call that ended by a timeout (or a lease refusal) is continued")
            gpu, timeout = record["gpu"], record["timeout"]
        else:
            gpu, timeout = beyond_bound
            print(f"!!! {study}/{trial}: continuing outside the admission bound (no attempt ledger), {gpu} for up to {timeout} s", flush=True)
        deadline = now() + wait
        while lease_state(lease := trial_lease(study, trial), now()) == "fresh":
            if now() >= deadline:
                raise SystemExit(f"{study}/{trial}: attempt {lease.get('attempt')}'s lease (call {lease.get('call_id')}) is still fresh, heartbeat {now() - lease['heartbeat']:.0f} s ago; "
                                 f"its container may still be running, so no continuation (it goes stale {LEASE_STALE} s after the last heartbeat)")
            print(f"{study}/{trial}: waiting for attempt {lease.get('attempt')}'s lease (heartbeat {now() - lease['heartbeat']:.0f} s ago) to end or go stale", flush=True)
            sleep(LEASE_POLL)
        entry = pending_attempt(now)
        if ledger: entries.append(entry); save()
        cpu, memory = trial_resources(gpu, True)
        call = deployed_run_trial(sources, "run_full_trial").with_options(gpu=gpu, cpu=cpu, memory=memory, timeout=timeout, retries=0).spawn(
            study, int(index), label, config, suite, sources, commit, None, transfer or None, attempt={"nonce": entry["nonce"], "number": used + 1 if ledger else None})
        if ledger:
            entry["call"] = record["calls"][label] = call.object_id; save()
            print(f"continuing {study}/{trial} from its last resume point: call {call.object_id} (attempt {used + 1} of {allowed})", flush=True)
        else:
            print(f"continuing {study}/{trial} from its last resume point: call {call.object_id} (not in any ledger)", flush=True)
        return call.object_id


def poll_status(call_id, poll):
    """kev.rounds.poll_modal's status of a call; a raised error is "failed (...)", or "unreachable (...)" when it is this
    machine's network (kev.rounds.transient)."""
    from kev.rounds import transient
    try:
        return poll(call_id)
    except Exception as error:   # noqa: BLE001 - the trial's own error, or the network: either way not a timeout
        return f"{'unreachable' if transient(error) else 'failed'} ({type(error).__name__})"


@app.local_entrypoint()
def resume(study: str, suite: str, transfer: str = "evals/v4/transfer-v4", gpu: str = GPU, timeout: int = 28800, trial: str = "", beyond_bound: bool = False):
    """Spawn evaluation for every trial in a study that has checkpoint/head.pt but no result.json, and continue every
    unfinished full-weight trial (no result.json, no failed.json) with the next attempt (continue_full_trial: counted in
    the study's attempt ledger, with the ledger's GPU and timeout). `trial` (<NN>-<label> or <label>) limits it to one
    trial: `kev.rounds watch` continues a timed-out trial this way. --gpu/--timeout serve run_resume and, with
    --beyond-bound, a continuation of a trial that has no ledger."""
    fn = modal.Function.from_name(APP_NAME, "run_resume").with_options(gpu=gpu)
    sources, commit = local_source_hashes(), local_git_commit()
    trials = sorted(t for t in volume_names(f"/{study}")[0] if not trial or trial in (t, t.split("-", 1)[-1]))
    if trial and not trials: raise SystemExit(f"no trial {trial!r} under /{study}")
    for t in trials:
        dirs, files = volume_names(f"/{study}/{t}")
        finished = "checkpoint" in dirs and "head.pt" in volume_names(f"/{study}/{t}/checkpoint")[1]
        if {"result.json", "failed.json"} & files or "provenance.json" not in files:
            print(f"skip {study}/{t}: {'has result' if 'result.json' in files else 'failed' if 'failed.json' in files else 'no provenance.json'}"); continue
        config = json.loads(b"".join(runs_volume.read_file(f"/{study}/{t}/provenance.json")))["config"]
        record = json.loads(spawn_record(study).read_text(encoding="utf-8")) if spawn_record(study).exists() else {}
        if config.get("full_ft") and (not finished or record.get("modal_retries") == 0):   # a ledgered trial is scored by its own next attempt
            try:
                continue_full_trial(study, t, config, suite, transfer, sources, commit, (gpu, timeout) if beyond_bound else None)
            except SystemExit as refusal:
                if trial: raise   # the one trial asked for (the watcher reads the exit status)
                print(f"skip {refusal}")
        elif finished:
            c = fn.spawn(study, t, suite, transfer, sources, commit); print(f"resuming {study}/{t}: call {c.object_id}")
        else:
            print(f"skip {study}/{t}: unfinished, not full-weight")


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
