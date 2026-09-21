"""Fine-tune, calibrate, evaluate, serve and publish a Kev decision model on Modal from your own labelled JSONL.

    modal run scripts/kev_modal.py::validate --data data/support --init-from jaredpalmer/kev-4b     # token limits, CPU only
    modal run scripts/kev_modal.py::train --data data/support --name support-v1                         # delta fine-tune + calibrate + score
    modal run scripts/kev_modal.py::evaluate --run jaredpalmer/kev-4b --data data/support --name support-base
    modal run scripts/kev_modal.py::compare --a support-v1 --b support-v2                               # paired bootstrap on development
    modal run scripts/kev_modal.py::pull --name support-v1 [--checkpoint]                               # reports (and weights) to runs/<name>
    KEV_SERVE_RUN=support-v1 modal deploy scripts/kev_modal.py                                          # System One endpoint on a GPU
    KEV_HF_SECRET=huggingface-secret modal run scripts/kev_modal.py::publish --name support-v1 --repo you/kev-4b-support

The image clones github.com/jaredpalmer/kev at KEV_REF and installs it; every container runs the same kev.train /
kev.benchmark / kev.serve code the released checkpoints were built and measured with. Trial outputs live on the Modal
volume `kev-finetune-runs` under /runs/<name> (names are immutable: a new attempt needs a new name); base weights are
cached on `kev-hf-cache`. Environment (read at launch time): KEV_GPU (training GPU, default H100), KEV_SERVE_GPU (default L4;
Kev-9B needs A100-80GB or H100), KEV_SERVE_RUN (run name on the volume or a Hub id), KEV_SERVE_SECRET (Modal secret holding
KEV_SERVE_API_KEY for bearer auth), KEV_HF_SECRET (Modal secret holding HF_TOKEN, needed by publish), KEV_APP_NAME, KEV_REF.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = os.environ.get("KEV_APP_NAME", "kev-finetune")
KEV_REPO = "https://github.com/jaredpalmer/kev.git"
KEV_REF = os.environ.get("KEV_REF", "f0ae8fb9af5762254974dcdba8c82cbf48587b2e")
KEV_ROOT = "/kev"
SUITE = f"{KEV_ROOT}/evals/v7/decision-v7"        # the public recipe the released checkpoints trained on: replay source and regression check
RUNS, HF = "/runs", "/hf"
GPU = os.environ.get("KEV_GPU", "H100")
SERVE_GPU = os.environ.get("KEV_SERVE_GPU", "L4")
SERVE_RUN = os.environ.get("KEV_SERVE_RUN", "jaredpalmer/kev-4b")
DEFAULT_INIT = "jaredpalmer/kev-4b"
GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "A100-80GB": 2.50, "A100": 2.10, "L40S": 1.95, "A10G": 1.10, "L4": 0.80, "T4": 0.59}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
PARTITIONS = ("train", "calibration", "development")

# Launch-time settings travel into the image env (names only, never secret values): the module is re-evaluated inside the
# container, and a Secret/Volume list that differs there fails the container with "Function has N dependencies but got M".
SETTINGS = {k: os.environ[k] for k in ("KEV_APP_NAME", "KEV_REF", "KEV_SERVE_RUN", "KEV_HF_SECRET", "KEV_SERVE_SECRET") if os.environ.get(k)}
SETTINGS.setdefault("KEV_SERVE_RUN", SERVE_RUN); SETTINGS.setdefault("KEV_REF", KEV_REF)

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .run_commands(f"git clone {KEV_REPO} {KEV_ROOT} && git -C {KEV_ROOT} checkout --quiet {KEV_REF}")
    .uv_pip_install(f"kev[serve] @ file://{KEV_ROOT}")
    # Gated DeltaNet kernels for the Qwen3.5 hybrid backbones; torch 2.8 pins triton 3.4, fla needs >= 3.7.1 on Hopper
    .uv_pip_install("flash-linear-attention", "triton>=3.7.1")
    .env({"HF_HOME": HF, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
          "TRITON_CACHE_DIR": f"{HF}/triton-cache"})   # compiled DeltaNet kernels persist on the cache volume: saves ~1 min per cold start
    .env(SETTINGS)
)
runs = modal.Volume.from_name("kev-finetune-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
VOLUMES = {RUNS: runs, HF: hf_cache}
hf_secret = [modal.Secret.from_name(os.environ["KEV_HF_SECRET"])] if os.environ.get("KEV_HF_SECRET") else []
serve_secret = [modal.Secret.from_name(os.environ["KEV_SERVE_SECRET"])] if os.environ.get("KEV_SERVE_SECRET") else []


# --- shared helpers (run inside containers) ---------------------------------------------------------------------------

def size_billions(base):
    m = re.search(r"(\d+(?:\.\d+)?)B", base)
    return float(m.group(1)) if m else 4.0


def recipe(size):
    """Training knobs by backbone size: what the released deltas used (experiments/night2-*-du.json)."""
    if size < 2: return {"lr": 4e-5, "batch": 8, "accum": 1, "checkpointing": 0}
    if size < 6: return {"lr": 2e-5, "batch": 4, "accum": 2, "checkpointing": 1}
    return {"lr": 2e-5, "batch": 2, "accum": 4, "checkpointing": 1}


def resolve_checkpoint(run):
    """A run name on the volume -> its checkpoint directory; anything else (local path, Hub id) is passed through."""
    runs.reload()
    candidate = Path(RUNS) / run / "checkpoint"
    return str(candidate) if (candidate / "head.pt").exists() else run


def check_partitions(directory, tok, names=PARTITIONS):
    """Load each partition through the training path and count the records Kev's context cannot hold."""
    from kev.api import render
    from kev.data import load_records, materialize
    from kev.model import MAX_STATE, fits
    report = {}
    for part in names:
        path = Path(directory) / f"{part}.jsonl"
        if not path.exists():
            continue
        records = load_records(path)
        over = [r["_meta"]["id"] for r in records if not fits(materialize(r), tok)]
        state_tokens = sorted(len(tok(render(r["state"]), add_special_tokens=False).input_ids) for r in records)
        report[part] = {"records": len(records), "over_limit": len(over), "over_limit_ids": over[:20],
                        "state_tokens": {"median": state_tokens[len(state_tokens) // 2], "max": state_tokens[-1], "limit": MAX_STATE},
                        "questions_per_record": round(sum(len(r["questions"]) for r in records) / len(records), 2)}
    return report


def tempered(rows, temperature):
    from kev.metrics import probabilities_at_temperature
    return [{**r, "p": probabilities_at_temperature(r, temperature).tolist()} for r in rows]


def error_rows(rows, records, temperature, limit=60):
    """Wrong development answers, most confident first, with the state text so the misses can be read and fixed with data."""
    from kev.metrics import probabilities_at_temperature
    by_id = {r["_meta"]["id"]: r for r in records}
    out = []
    for r in rows:
        p = probabilities_at_temperature(r, temperature); i = int(p.argmax())
        if i == r["label"]: continue
        rec = by_id[r["id"]]
        out.append({"id": r["id"], "question": r["question"], "confidence": round(float(p[i]), 3), "predicted": r["keys"][i], "label": r["keys"][r["label"]],
                    "probabilities": {k: round(float(v), 3) for k, v in zip(r["keys"], p)},
                    "instructions": rec["questions"][r["question"]]["instructions"], "state": rec["state"]})
    return sorted(out, key=lambda e: -e["confidence"])[:limit]


def score(run, development, out, calibration=None, temperature=None):
    """Raw-logit predictions of one checkpoint on the development records; the temperature is fitted on the calibration
    records when given (min NLL), else `temperature` (default: raw). Returns (report, rows, temperature)."""
    import gc
    import torch
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.metrics import fit_temperature
    from kev.predictors import LocalPredictor
    predictor = LocalPredictor(run, "cuda", LoadOptions(temperature=1.0))
    T = temperature or 1.0
    if calibration:
        _, cal_rows = evaluate_records(calibration, predictor, Path(out) / "calibration")
        T = fit_temperature(cal_rows, aggregation="micro")
    report, rows = evaluate_records(development, predictor, Path(out) / "development", T)
    del predictor; gc.collect(); torch.cuda.empty_cache()
    return report, rows, T


def summary_block(report, rows, temperature):
    """Raw and calibrated metrics on the development rows plus per-question calibrated metrics."""
    from kev.metrics import grouped_metrics
    clean = [r for r in rows if r["variant"] == "clean"]
    return {"temperature": temperature, "raw": report["clean"], "calibrated": report["calibrated_clean"],
            "per_question": grouped_metrics(clean, "question", temperature), "n_questions": len(clean)}


KEYS = (("accuracy", "acc", "{:.3f}"), ("brier (lower is better)", "brier", "{:.3f}"), ("nll (lower is better)", "nll", "{:.3f}"),
        ("ece (lower is better)", "ece", "{:.3f}"), ("confident errors (p>=0.9, wrong)", "confident_error_rate", "{:.3f}"),
        ("coverage at 5% error", "coverage_at_5pct_error", "{:.2f}"), ("mean confidence", "mean_conf", "{:.3f}"))


def render_table(columns):
    """columns: [(title, metrics dict)] -> aligned text table over KEYS."""
    width = max(len(k[0]) for k in KEYS) + 2
    col = max(12, *(len(t) + 2 for t, _ in columns))
    lines = [" " * width + "".join(f"{t:>{col}}" for t, _ in columns)]
    for label, key, fmt in KEYS:
        cells = "".join(f"{(fmt.format(m[key]) if m.get(key) is not None else '-'):>{col}}" for _, m in columns)
        lines.append(f"{label:<{width}}{cells}")
    return "\n".join(lines)


def print_result(result):
    dev, base = result["development"], result.get("baseline")
    cols = []
    if base:
        cols += [(f"{result['init_from']} raw", base["development"]["raw"]), (f"{result['init_from']} calibrated", base["development"]["calibrated"])]
    cols += [(f"{result['name']} raw", dev["raw"]), (f"{result['name']} calibrated", dev["calibrated"])]
    print(f"\n== development partition ({dev['n_questions']} questions), fitted temperature {dev['temperature']:.2f}" + (f" (baseline {base['development']['temperature']:.2f})" if base else ""))
    print(render_table(cols))
    if base:
        print("\nper question (calibrated accuracy / brier):")
        for qid, m in dev["per_question"].items():
            b = base["development"]["per_question"].get(qid, {})
            print(f"  {qid:<24} baseline {b.get('acc', float('nan')):.3f} / {b.get('brier', float('nan')):.3f}   finetuned {m['acc']:.3f} / {m['brier']:.3f}   n={m['n']}")
        for metric, boot in result.get("bootstrap", {}).items():
            lo, hi = boot["ci95"]
            print(f"  {metric} delta (finetuned - baseline): {boot[f'micro_{metric}_delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]" + ("  significant" if lo > 0 or hi < 0 else ""))
    reg = result.get("regression")
    if reg:
        before = f"baseline acc {reg['baseline']['acc']:.3f} brier {reg['baseline']['brier']:.3f} -> " if reg.get("baseline") else ""
        print(f"\n== regression check on {reg['n']} public decision-v7 development records (raw logits): "
              f"{before}finetuned acc {reg['finetuned']['acc']:.3f} brier {reg['finetuned']['brier']:.3f}")
    if result.get("errors"):
        print(f"\n{result['errors']} wrong development answers listed in errors.jsonl (most confident first)")


def stage(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- remote functions ---------------------------------------------------------------------------------------------------

@app.function(image=image, cpu=2, memory=8192, timeout=1200, volumes=VOLUMES, secrets=hf_secret)
def run_validate(name, init_from):
    from kev.checkpoint import Checkpoint
    from kev.model import load_tokenizer
    runs.reload()
    ck = Checkpoint(init_from)
    tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
    return {"base": ck.meta.base, "partitions": check_partitions(Path(RUNS) / name / "data", tok)}


@app.function(image=image, gpu=GPU, cpu=4, memory=(32768, 131072), timeout=4 * 3600, retries=0, volumes=VOLUMES, secrets=hf_secret)
def run_train(name, config, baseline=True, regression=300):
    """Delta fine-tune from `config['init_from']` on /runs/<name>/data/train.jsonl, fit a temperature on calibration.jsonl,
    score development.jsonl (and the init checkpoint on the same records), check the public recipe for forgetting."""
    import random
    import torch
    from kev.checkpoint import Checkpoint, read_meta, write_meta
    from kev.data import load_records
    from kev.metrics import metrics, paired_bootstrap
    from kev.model import load_tokenizer
    from kev.suite import load_split, write_json, write_jsonl

    runs.reload()
    out = Path(RUNS) / name
    if (out / "checkpoint").exists() or (out / "result.json").exists():
        raise FileExistsError(f"/runs/{name} already holds a run; choose a new name")
    started = time.time()
    init = Checkpoint(config["init_from"])
    meta, args = init.meta, init.meta.extra.get("args", {})
    size = size_billions(meta.base)
    cfg = {**recipe(size), **{k: config[k] for k in ("lr", "batch", "accum") if config.get(k)},   # 0 = the recipe's value
           **{k: config[k] for k in ("epochs", "seed", "replay", "p_none_pair")}}
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    checks = check_partitions(out / "data", tok)
    write_json(out / "config.json", {"name": name, "init_from": config["init_from"], "init_resolved": init.path, "base": meta.base, "base_revision": meta.base_revision,
                                     "config": cfg, "data": checks, "kev_ref": KEV_REF, "gpu": torch.cuda.get_device_name(0)})
    stage(f"{meta.base} ({size:g}B) from {config['init_from']}; config {json.dumps(cfg)}; data {json.dumps({k: (v['records'], v['over_limit']) for k, v in checks.items()})}")
    for part, c in checks.items():
        if c["over_limit"]: print(f"warning: {c['over_limit']} {part} records exceed Kev's training context and will be dropped (state limit {c['state_tokens']['limit']} tokens)", flush=True)

    run = str(out / "checkpoint")
    cmd = [sys.executable, "-m", "kev.train", "--data", str(out / "data/train.jsonl"), "--init_from", config["init_from"], "--out", run, "--device", "cuda", "--dtype", "bf16",
           "--base", meta.base, "--lora", meta.lora, "--head_dim", meta.head_dim, "--lora_targets", args.get("lora_targets", "all"),
           "--option_isolation", int(meta.option_isolation), "--special_embeddings", int(meta.special_embeddings), "--weights_dtype", meta.weights_dtype,
           "--epochs", cfg["epochs"], "--lr", cfg["lr"], "--batch", cfg["batch"], "--accum", cfg["accum"], "--checkpointing", cfg["checkpointing"],
           "--seed", cfg["seed"], "--p_none_pair", cfg["p_none_pair"]]
    if meta.base_revision: cmd += ["--base_revision", meta.base_revision]
    if cfg["replay"]: cmd += ["--suite", SUITE, "--replay", cfg["replay"]]
    cmd = [str(c) for c in cmd]
    stage("training: " + " ".join(cmd[2:]))
    try:
        with (out / "train.log").open("w", encoding="utf-8") as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=KEV_ROOT) as proc:
            for line in proc.stdout:
                log.write(line); log.flush()
                if line.startswith(("ep", "saved", "device", "delta", "replay", "dropped")) or "Error" in line or "Traceback" in line: print(line.rstrip(), flush=True)
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd, "kev.train failed; see train.log")
    finally:
        runs.commit(); hf_cache.commit()

    calibration, development = load_records(out / "data/calibration.jsonl"), load_records(out / "data/development.jsonl")
    stage(f"scoring {name} on {len(calibration)} calibration + {len(development)} development records")
    report, rows, T = score(run, development, out, calibration)
    m = read_meta(run); m.temperature = T
    m.extra["temperature_fit"] = {"rows": "calibration.jsonl", "n": len(calibration), "method": "min NLL, micro, kev.metrics.fit_temperature", "value": T}
    write_meta(run, m)   # the checkpoint now serves calibrated probabilities by default
    result = {"name": name, "init_from": config["init_from"], "base": meta.base, "config": cfg, "temperature": T, "data": checks, "kev_ref": KEV_REF,
              "development": summary_block(report, rows, T)}
    errors = error_rows(rows, development, T)
    write_jsonl(out / "errors.jsonl", errors); result["errors"] = len(errors)
    write_json(out / "result.json", result); runs.commit()

    if baseline:
        stage(f"scoring the baseline {config['init_from']} on the same records")
        b_report, b_rows, b_T = score(config["init_from"], development, out / "baseline", calibration)
        result["baseline"] = {"run": config["init_from"], "development": summary_block(b_report, b_rows, b_T)}
        result["bootstrap"] = {metric: paired_bootstrap(tempered(rows, T), tempered(b_rows, b_T), metric=metric, aggregation="micro") for metric in ("acc", "brier", "ece")}
        write_json(out / "result.json", result); runs.commit()
    if regression:
        groups = {}   # whole groups: a permuted variant is scored against its parent and a contrastive pair against its sibling
        for r in load_split(SUITE, "development"): groups.setdefault(r["_meta"]["group_id"], []).append(r)
        chosen = random.Random(cfg["seed"]).sample(sorted(groups), min(regression, len(groups)))
        sample = [r for g in chosen for r in groups[g]]
        stage(f"regression check: {len(sample)} public decision-v7 development records")
        _, r_rows, _ = score(run, sample, out / "regression/finetuned")
        result["regression"] = {"n": len(sample), "suite": "evals/v7/decision-v7", "finetuned": metrics([r for r in r_rows if r["variant"] == "clean"])}
        if baseline:
            _, rb_rows, _ = score(config["init_from"], sample, out / "regression/baseline")
            result["regression"]["baseline"] = metrics([r for r in rb_rows if r["variant"] == "clean"])
    result["wall_seconds"] = round(time.time() - started); result["gpu"] = torch.cuda.get_device_name(0)
    write_json(out / "result.json", result); runs.commit(); hf_cache.commit()
    return result


def score_job(name, predictor_factory, data_dir, calibrate):
    """Score any predictor on /runs/<name>/data/development.jsonl (+ calibration.jsonl when calibrate). Used by the GPU and
    CPU evaluate functions."""
    from kev.benchmark import evaluate_records
    from kev.data import load_records
    from kev.metrics import fit_temperature
    from kev.suite import write_json, write_jsonl
    runs.reload()
    out = Path(RUNS) / name
    if (out / "development").exists(): raise FileExistsError(f"/runs/{name}/development exists; choose a new name")
    development = load_records(data_dir / "development.jsonl")
    predictor = predictor_factory()
    T = 1.0
    if calibrate and (data_dir / "calibration.jsonl").exists():
        _, cal_rows = evaluate_records(load_records(data_dir / "calibration.jsonl"), predictor, out / "calibration")
        T = fit_temperature(cal_rows, aggregation="micro")
    report, rows = evaluate_records(development, predictor, out / "development", T)
    result = {"name": name, "temperature": T, "development": summary_block(report, rows, T), "kev_ref": KEV_REF}
    errors = error_rows(rows, development, T); write_jsonl(out / "errors.jsonl", errors); result["errors"] = len(errors)
    write_json(out / "result.json", result); runs.commit()
    return result


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 65536), timeout=3600, retries=0, volumes=VOLUMES, secrets=hf_secret)
def run_evaluate(name, run):
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    target = resolve_checkpoint(run)
    result = score_job(name, lambda: LocalPredictor(target, "cuda", LoadOptions(temperature=1.0)), Path(RUNS) / name / "data", calibrate=True)
    result["run"] = run
    return result


@app.function(image=image, cpu=2, memory=4096, timeout=3600, retries=0, volumes=VOLUMES)
def run_evaluate_remote(name, base_url, model, api_key):
    """Score any System One-compatible endpoint (a deployed Kev, or Jev with a TypeSafe key) on the same development file.
    Remote probabilities are already what the service returns, so no temperature is fitted."""
    from kev.predictors import RemotePredictor
    result = score_job(name, lambda: RemotePredictor(base_url, model, api_key), Path(RUNS) / name / "data", calibrate=False)
    result["remote"] = {"base_url": base_url, "model": model}
    return result


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes=VOLUMES)
def run_compare(a, b, metrics=("acc", "brier", "ece")):
    """Paired bootstrap between two runs scored on the same development file (rows.json + fitted temperatures)."""
    from kev.metrics import paired_bootstrap
    from kev.suite import read_json
    runs.reload()
    sides = []
    for name in (a, b):
        result = read_json(Path(RUNS) / name / "result.json")
        rows = read_json(Path(RUNS) / name / "development/rows.json")
        sides.append((tempered(rows, result["temperature"]), result["development"]["calibrated"]))
    out = {"a": a, "b": b, "a_metrics": sides[0][1], "b_metrics": sides[1][1],
           "bootstrap": {m: paired_bootstrap(sides[0][0], sides[1][0], metric=m, aggregation="micro") for m in metrics}}
    return out


CARD = """---
base_model: {base}
base_model_relation: adapter
library_name: peft
tags: [kev, decision-model, typesafe, system-one, calibration]
---

# {repo}

A [Kev](https://github.com/jaredpalmer/kev) decision model fine-tuned from `{init_from}` on {n_train} labelled records
for one workload, with a temperature ({temperature:.2f}) fitted on {n_cal} held-out records so the served probabilities
are calibrated by default.

| development ({n_dev} questions) | {init_from} (calibrated) | this model (calibrated) |
| --- | --- | --- |
| accuracy | {b_acc:.3f} | {acc:.3f} |
| brier | {b_brier:.3f} | {brier:.3f} |
| ece | {b_ece:.3f} | {ece:.3f} |
| coverage at 5% error | {b_cov:.2f} | {cov:.2f} |

Serve it with the Kev repo (`uv run --extra serve python -m kev.serve --run {repo}`) or the `kev-finetune` skill's Modal
endpoint. Trained with `kev.train --init_from` (LoRA delta + pointer head) at kev commit `{kev_ref}`; `result.json`,
`training_config.json` and `train.log` are in this repo.
"""


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes=VOLUMES, secrets=hf_secret)
def run_publish(name, repo, private, message, card):
    from kev.suite import read_json
    if not os.environ.get("HF_TOKEN"): raise RuntimeError("no HF_TOKEN in the container: create a Modal secret with HF_TOKEN and pass its name as KEV_HF_SECRET")
    runs.reload()
    out = Path(RUNS) / name
    result = read_json(out / "result.json")
    dev, base = result["development"], (result.get("baseline") or {}).get("development", {})
    b = base.get("calibrated", {})
    text = card or CARD.format(base=result["base"], repo=repo, init_from=result["init_from"], n_train=result["data"]["train"]["records"], temperature=result["temperature"],
                               n_cal=result["data"]["calibration"]["records"], n_dev=dev["n_questions"], acc=dev["calibrated"]["acc"], brier=dev["calibrated"]["brier"],
                               ece=dev["calibrated"]["ece"], cov=dev["calibrated"]["coverage_at_5pct_error"], b_acc=b.get("acc", float("nan")), b_brier=b.get("brier", float("nan")),
                               b_ece=b.get("ece", float("nan")), b_cov=b.get("coverage_at_5pct_error", float("nan")), kev_ref=KEV_REF[:12])
    card_path = out / "model-card.md"; card_path.write_text(text, encoding="utf-8")
    message = message or f"Upload {name} (delta from {result['init_from']}; development acc {dev['calibrated']['acc']:.3f}, brier {dev['calibrated']['brier']:.3f}, T {result['temperature']:.2f})"
    cmd = [sys.executable, "-m", "kev.publish", "--run", str(out / "checkpoint"), "--repo", repo, "--card", str(card_path), "--message", message]
    if private: cmd.append("--private")
    subprocess.run(cmd, check=True, cwd=KEV_ROOT)
    runs.commit()
    return f"https://huggingface.co/{repo}"


WARMUP = {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.", "model": "kev-latest",
          "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}},
                        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
                        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}}


@app.cls(image=image, gpu=SERVE_GPU, cpu=2, memory=(16384, 65536), volumes=VOLUMES, secrets=hf_secret + serve_secret,
         min_containers=int(os.environ.get("KEV_SERVE_MIN_CONTAINERS", "0")), scaledown_window=300, timeout=600, startup_timeout=900)
@modal.concurrent(max_inputs=8)
class Serve:
    """TypeSafe System One-compatible endpoint (POST /v1/systemone, GET /v1/models) for KEV_SERVE_RUN, in bf16.
    Deploy: KEV_SERVE_RUN=<run name | Hub id> modal deploy scripts/kev_modal.py"""

    @modal.enter()
    def load(self):
        import torch
        from fastapi import Request
        from fastapi.responses import JSONResponse
        from kev.api import SystemOneRequest
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server, app as api
        run = SERVE_RUN   # from the image env, fixed at deploy time
        ck = Checkpoint(resolve_checkpoint(run))
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16))
        api.state.server = Server(ck, tok, model, "cuda")
        key = os.environ.get("KEV_SERVE_API_KEY")
        if key:
            @api.middleware("http")
            async def bearer(request: Request, call_next):
                if request.headers.get("authorization") != f"Bearer {key}":
                    return JSONResponse({"error": "unauthorized; send Authorization: Bearer <KEV_SERVE_API_KEY>"}, status_code=401)
                return await call_next(request)
        started = time.time()
        api.state.server.answer(SystemOneRequest.model_validate(WARMUP))   # compiles the DeltaNet kernels now, not on the first user request (~1 min uncached)
        hf_cache.commit()                                                    # keep the compiled kernels for the next cold start
        print(f"serving {run} ({ck.path}) temperature {model.head.temperature:.2f} on {torch.cuda.get_device_name(0)}; warm-up {time.time() - started:.0f}s", flush=True)
        self.api = api

    @modal.asgi_app(label=f"{APP_NAME}-api")
    def web(self):
        return self.api


# --- local entrypoints ------------------------------------------------------------------------------------------------

def check_name(name):
    if not NAME.fullmatch(name): raise SystemExit("--name must be letters, digits, - or _ (max 80 characters)")
    try:
        entries = {Path(e.path).name for e in runs.listdir("/")}
    except Exception:
        entries = set()
    if name in entries: raise SystemExit(f"/runs/{name} already exists on the volume; names are immutable, choose a new one (e.g. {name}-2)")


def upload_data(name, source, required, optional=()):
    """Copy the partitions of a local split directory (or one JSONL as development.jsonl) to /runs/<name>/data/."""
    source = Path(source)
    files = {p: source / f"{p}.jsonl" for p in (*required, *optional)} if source.is_dir() else {"development": source}
    missing = [str(f) for p, f in files.items() if not f.exists() and p in required]
    if missing: raise SystemExit(f"missing data files: {missing}; run scripts/split_data.py first")
    files = {p: f for p, f in files.items() if f.exists()}
    with runs.batch_upload() as batch:
        for part, path in files.items():
            batch.put_file(str(path), f"/{name}/data/{part}.jsonl")
    return {p: sum(1 for l in f.read_text(encoding="utf-8").splitlines() if l.strip()) for p, f in files.items()}


def download_run(name, checkpoint=False, into="runs"):
    """Reports, logs and errors of a run into runs/<name>/ (weights only with checkpoint=True; prediction dumps are skipped)."""
    from modal.volume import FileEntryType
    target = Path(into) / name
    for e in runs.listdir(f"/{name}", recursive=True):
        if e.type != FileEntryType.FILE: continue
        rel = Path(e.path.lstrip("/")).relative_to(name)
        if (rel.parts[0] == "checkpoint" and not checkpoint) or rel.parts[0] == "data" or rel.name == "predictions.jsonl": continue
        dest = target / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in runs.read_file(e.path): f.write(chunk)
    return target


def bound(gpu, timeout):
    return GPU_HOURLY.get(gpu, 4.0) * timeout / 3600


@app.local_entrypoint()
def validate(data: str, name: str = "validate", init_from: str = DEFAULT_INIT):
    """Tokenizer-level check of a split directory against Kev's training context; CPU only, no GPU cost."""
    scratch = f"{name}-{int(time.time())}"
    counts = upload_data(scratch, data, ("train",), ("calibration", "development"))
    try:
        report = run_validate.remote(scratch, init_from)
    finally:
        runs.remove_file(f"/{scratch}", recursive=True)
    print(json.dumps(report, indent=1))
    for part, c in report["partitions"].items():
        if c["over_limit"]: print(f"warning: {c['over_limit']} of {c['records']} {part} records exceed the context (state tokens > {c['state_tokens']['limit']} or the packed request > 2048); shorten them or accept the drop")
    print(f"ok: {counts} records checked against {report['base']}'s tokenizer")


@app.local_entrypoint()
def train(data: str, name: str, init_from: str = DEFAULT_INIT, epochs: int = 1, lr: float = 0.0, replay: int = 2000, batch: int = 0, accum: int = 0,
          seed: int = 0, p_none_pair: float = 0.25, regression: int = 300, baseline: bool = True, gpu: str = GPU, timeout: int = 10800):
    """Delta fine-tune `init_from` on data/train.jsonl, calibrate on calibration.jsonl, score development.jsonl against the
    baseline and pull the reports to runs/<name>/. lr/batch/accum default to the released recipe for the backbone size."""
    check_name(name)
    counts = upload_data(name, data, PARTITIONS)
    config = {"init_from": init_from, "epochs": epochs, "lr": lr, "replay": replay, "batch": batch, "accum": accum, "seed": seed, "p_none_pair": p_none_pair}
    print(f"uploaded {counts} to /runs/{name}/data; training on {gpu}, cost bound ${bound(gpu, timeout):.2f} for the {timeout}s timeout (typical for 400 records: 0.8B ~8 min, 4B ~15 min, 9B ~30 min)", flush=True)
    result = run_train.with_options(gpu=gpu, timeout=timeout).remote(name, config, baseline, regression)
    print_result(result)
    target, script = download_run(name), os.path.relpath(__file__)
    print(f"\nreports in {target}/ (result.json, errors.jsonl, train.log). Serve it: KEV_SERVE_RUN={name} modal deploy {script}; weights: modal run {script}::pull --name {name} --checkpoint")


@app.local_entrypoint()
def evaluate(data: str, name: str, run: str = "", remote: str = "", remote_model: str = "kev-latest", gpu: str = GPU):
    """Score a checkpoint (run name on the volume, Hub id) or a System One endpoint (--remote URL, key from KEV_REMOTE_API_KEY)
    on data/development.jsonl; a checkpoint is also calibrated on data/calibration.jsonl when present."""
    if bool(run) == bool(remote): raise SystemExit("give exactly one of --run or --remote")
    check_name(name)
    counts = upload_data(name, data, ("development",), ("calibration",))
    print(f"uploaded {counts}", flush=True)
    if remote:
        result = run_evaluate_remote.remote(name, remote, remote_model, os.environ.get("KEV_REMOTE_API_KEY", "local"))
    else:
        result = run_evaluate.with_options(gpu=gpu).remote(name, run)
    dev = result["development"]
    print(f"\n== {name}: {run or remote} on {dev['n_questions']} development questions, temperature {dev['temperature']:.2f}")
    print(render_table([(f"{name} raw", dev["raw"]), (f"{name} calibrated", dev["calibrated"])]))
    for qid, m in dev["per_question"].items(): print(f"  {qid:<24} acc {m['acc']:.3f} brier {m['brier']:.3f} n={m['n']}")
    target = download_run(name)
    print(f"\n{result['errors']} wrong answers in {target}/errors.jsonl; full report in {target}/development/report.json")


@app.local_entrypoint()
def compare(a: str, b: str):
    """Paired bootstrap (calibrated probabilities) between two runs scored on the same development file."""
    out = run_compare.remote(a, b)
    print(render_table([(a, out["a_metrics"]), (b, out["b_metrics"])]))
    for metric, boot in out["bootstrap"].items():
        lo, hi = boot["ci95"]
        print(f"{metric} delta ({a} - {b}): {boot[f'micro_{metric}_delta']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]" + ("  significant" if lo > 0 or hi < 0 else ""))


@app.local_entrypoint()
def pull(name: str, checkpoint: bool = False):
    """Download a run's reports (and with --checkpoint its weights, for local serving with kev.serve) into runs/<name>/."""
    target = download_run(name, checkpoint)
    if (target / "result.json").exists(): print_result(json.loads((target / "result.json").read_text(encoding="utf-8")))
    print(f"pulled to {target}")


@app.local_entrypoint()
def publish(name: str, repo: str, private: bool = False, message: str = "", card: str = ""):
    """Upload /runs/<name>/checkpoint to the Hugging Face Hub as `repo` with a generated model card (--card to supply your own)."""
    if not hf_secret: raise SystemExit("set KEV_HF_SECRET=<modal secret name holding HF_TOKEN> (create one with: modal secret create huggingface-secret HF_TOKEN=hf_...)")
    text = Path(card).read_text(encoding="utf-8") if card else ""
    print(run_publish.remote(name, repo, private, message, text))
