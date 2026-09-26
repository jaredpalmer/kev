"""Configuration-only experiment runner.

A study is a JSON plan: a list of 1..8 trials, each a dict of allowlisted training parameters plus a base model
whose revision is pinned in the suite manifest. Every trial trains on the suite's training partition, fits a
temperature on the calibration partition, and is scored on the development partition. The locked test is never
read here.

Two execution modes share one code path:
  local  : trials run sequentially on this machine (one GPU job at a time); `--wait-pid` queues behind a run.
  modal  : `modal_app.py` runs `execute_trial` once per container and `--aggregate` ranks the results afterwards.
"""
import argparse
import copy
import fcntl
import gc
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.device import default_device, empty_cache
from kev.full_ft import snapshot_fractions, too_many_snapshots
from kev.metrics import fit_temperature, paired_bootstrap
from kev.model import MAX_STATE, MAX_TRAIN_STATE
from kev.predictors import LocalPredictor
from kev.suite import CONTEXT, ENCODING, digest, load_split, read_json, read_manifest, record_digest, validate_training, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"epochs": 1, "seed": 0, "lr": 0.0002, "lora": 16, "accum": 8, "batch": 1,
            "perm_kl": 0.0, "perm_frac": 0.3, "ord_w": 0.0,
            "p_none": 0.1, "p_none_distract": 0.12, "p_distract": 0.15, "p_none_pair": 0.0, "synthetic_repeat": 1, "public_frac": 1.0, "head_lr": 0.0, "weight_decay": 0.01, "anchor_w": 0.0,
            "label_smoothing": 0.0, "brier_w": 0.0, "focal_gamma": 0.0}
RANGES = {"epochs": (1, 5), "seed": (0, 10000), "lr": (1e-6, 0.001), "lora": (1, 64), "accum": (1, 256), "batch": (1, 64),
          "perm_kl": (0, 2), "perm_frac": (0, 1), "ord_w": (0, 2),
          "label_smoothing": (0, 0.2), "brier_w": (0, 2), "focal_gamma": (0, 4),
          "p_none": (0, 0.4), "p_none_distract": (0, 0.4), "p_distract": (0, 0.4), "p_none_pair": (0, 1), "synthetic_repeat": (1, 6), "public_frac": (0.05, 1.0), "head_lr": (0, 0.01), "weight_decay": (0, 0.3), "anchor_w": (0, 5)}
CHOICES = {"dtype": ("fp32", "bf16"), "checkpointing": (0, 1), "option_isolation": (0, 1), "special_embeddings": (0, 1), "head_dim": (128, 256, 512, 1024),
           "lora_targets": ("all", "dense", "attn", "qv"), "weights_dtype": ("fp32", "bf16"), "full_ft": (0, 1), "length_sort": (0, 1), "shared_prefix": (0, 1)}
CHOICE_DEFAULTS = {"dtype": "fp32", "checkpointing": 0, "option_isolation": 0, "special_embeddings": 0, "head_dim": 256, "lora_targets": "all", "weights_dtype": "fp32", "full_ft": 0, "length_sort": 0, "shared_prefix": None}   # kev.train's defaults for the categorical knobs (shared_prefix: on with full_ft)
# optional integer knobs, passed to kev.train only when a trial sets them (so existing plans keep their config hashes)
OPTIONAL_INTS = {"max_state": (MAX_STATE, MAX_TRAIN_STATE), "row_budget": (0, 65536), "max_steps": (0, 100000), "save_every_steps": (0, 100000),
                 "snapshot_every_steps": (0, 100000), "none_pair_max_state": (1, MAX_TRAIN_STATE), "pass_tokens_max": (1, 1 << 20)}
# full-weight trials write snapshots (kev.full_ft.SnapshotWriter: loadable bf16 checkpoints, kept on the runs volume) into
# <trial>/snapshots/step-<N>/checkpoint after these fractions of their optimizer steps, unless the trial sets
# snapshot_fractions ("none" for none). Applied by train_checkpoint like RESUME_MINUTES, so plans keep their config hashes.
SNAPSHOT_FRACTIONS = "0.25,0.5,0.75"
# trial keys the container acts on, not kev.train: snapshot_hub_repo (a private Hub model repo) makes modal_app mirror
# each committed snapshot and the final checkpoint there (modal_app.run_mirror); off unless a trial sets it
TRIAL_ONLY = ("snapshot_hub_repo",)


def validated_trial(value, manifest):
    if not isinstance(value, dict) or set(value) - (DEFAULTS.keys() | CHOICES.keys() | OPTIONAL_INTS.keys() | {"base", "train_sources", "base_revision", "anchor", "anchor_sources", "init_from", "data", "replay", "snapshot_fractions", *TRIAL_ONLY}):
        raise ValueError("trial may change only the allowlisted training parameters and base")
    result = {**DEFAULTS, **value}
    if result.get("full_ft") and result.get("weights_dtype") != "bf16":
        raise ValueError("full_ft trains bf16 weights: set weights_dtype bf16")
    if result.get("save_every_steps") and not result.get("full_ft"):
        raise ValueError("save_every_steps writes resume points, which are for full_ft trials")
    if ("snapshot_fractions" in result or result.get("snapshot_every_steps") or "snapshot_hub_repo" in result) and not result.get("full_ft"):
        raise ValueError("snapshots (and their Hub mirror) are for full_ft trials")
    if "snapshot_fractions" in result and not isinstance(result["snapshot_fractions"], str):
        raise ValueError('snapshot_fractions is a string: "0.25,0.5" or "none"')
    if "snapshot_hub_repo" in result and not re.fullmatch(r"[\w.-]+/[\w.-]+", str(result["snapshot_hub_repo"])):
        raise ValueError("snapshot_hub_repo is a Hub model repo id, owner/name (it must be private; the mirror refuses a public one)")
    if "data" in result and not re.fullmatch(r"evals/[\w./-]+\.jsonl", str(result["data"])):
        raise ValueError("data must be a .jsonl under evals/ (shipped with the image, hashed in provenance)")
    if "replay" in result and (not isinstance(result["replay"], int) or not 0 <= result["replay"] <= 20000 or "data" not in result):
        raise ValueError("replay is an int <= 20000 and needs data")
    for key, (lo, hi) in OPTIONAL_INTS.items():
        if key in result and (isinstance(result[key], bool) or not isinstance(result[key], int) or not lo <= result[key] <= hi):
            raise ValueError(f"{key} is an int in [{lo}, {hi}] (optional; kev.train's default when absent)")
    if "none_pair_max_state" in result and not result["p_none_pair"] > 0:
        raise ValueError("none_pair_max_state gates none pairs: it needs p_none_pair > 0")
    if "pass_tokens_max" in result and (not result.get("length_sort") or result.get("row_budget") or result["perm_kl"] > 0):
        raise ValueError("pass_tokens_max caps the passes length_sort 1 plans: it needs length_sort 1, and neither row_budget nor perm_kl")
    if result.get("full_ft"):   # kev.budget.MAX_SNAPSHOTS, over max_steps (an every-N plan needs it); kev.train checks the real count again
        fractions = snapshot_fractions(result.get("snapshot_fractions", SNAPSHOT_FRACTIONS))   # raises ValueError on a bad list
        if problem := too_many_snapshots(fractions, result.get("snapshot_every_steps", 0), result.get("max_steps") or None):
            raise ValueError(problem)
    if "init_from" in result and not re.fullmatch(r"(/runs/[\w./-]+|[\w-]+/[\w.-]+(@[\w.-]+)?)", str(result["init_from"])):
        raise ValueError("init_from must be a checkpoint path on the runs volume or a Hub id (optionally @revision); the trainer records its weights and head hashes in provenance")
    if result.get("base") not in manifest["base_revisions"]:
        # a base the frozen suite did not pin may still be used if the trial pins its own full commit sha (recorded in provenance)
        if not re.fullmatch(r"[0-9a-f]{40}", str(result.get("base_revision", ""))):
            raise ValueError("base must have a revision pinned in the suite, or the trial must pin a 40-hex base_revision")
    elif "base_revision" in result and result["base_revision"] != manifest["base_revisions"][result["base"]]:
        raise ValueError("trial base_revision conflicts with the suite's pinned revision")
    if ("anchor" in result) != (result.get("anchor_w", 0) > 0):
        raise ValueError("anchor (a targets file) and anchor_w > 0 must be given together")
    if "anchor" in result and not re.fullmatch(r"[\w./-]+\.json", str(result["anchor"])):
        raise ValueError("anchor must be a .json path")
    if "anchor_sources" in result and (not isinstance(result["anchor_sources"], str) or set(result["anchor_sources"].split(",")) - set(manifest.get("trainable_sources", []))):
        raise ValueError("anchor_sources must be trainable sources of the suite")
    if "train_sources" in result:
        names = result["train_sources"].split(",") if isinstance(result["train_sources"], str) else None
        if not names or set(names) - set(manifest.get("trainable_sources", [])):
            raise ValueError("train_sources must be a comma-separated subset of the suite's trainable sources")
    for key, (lo, hi) in RANGES.items():
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            raise ValueError(f"invalid {key}")
        if isinstance(DEFAULTS[key], int) and not isinstance(v, int):
            raise ValueError(f"{key} must be an integer")
    for key, allowed in CHOICES.items():
        if key in result and result[key] not in allowed:
            raise ValueError(f"invalid {key}")
    if sum(result[k] for k in ("p_none", "p_none_distract", "p_distract")) > 1:
        raise ValueError("augmentation probabilities sum to more than one")
    if sum(result[k] > 0 for k in ("label_smoothing", "brier_w", "focal_gamma")) > 1:
        raise ValueError("a trial may change only one loss modifier")
    return result


RESUME_MINUTES = 60   # full-weight trials: minutes between resume points (kev.train --save_every_minutes). A 27B's on 8 H200s
# (~307 GB) blocked training 23 s and wrote in ~2.5 min behind it, slowing those steps ~50 %: ~2.6 % of an hour (runs/sft-probe/sft2-27b-8xh200-all-balanced)

# files whose change would alter what a score means; trainer/runner files may differ when resuming an interrupted evaluation
EVALUATOR_FILES = {f"kev/{n}" for n in ("model.py", "benchmark.py", "data.py", "api.py", "suite.py", "evaluate.py", "contrastive.py", "composition.py", "study_v3.py")}


def source_hashes():
    paths = list((ROOT / "kev").glob("*.py")) + [ROOT / name for name in ("uv.lock", "pyproject.toml", "modal_app.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths) if path.exists()}


def git_commit():
    """Commit hash from the environment inside containers (no .git there), else from the working tree."""
    if os.environ.get("KEV_GIT_COMMIT"):
        return os.environ["KEV_GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@contextmanager
def study_lock():
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs/.research.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another research runner owns the GPU queue") from None
        yield


# research screening thresholds; gate_report's policy string is generated from them. The gate *names* ("..._over_5pp",
# "..._at_least_70pct") are ledger keys and must be renamed by hand if a threshold changes.
GATES = {"isolation_tolerance": 0.001, "task_accuracy_regression": 0.05, "variant_accuracy_regression": 0.05, "permutation_flip_increase": 0.05,
         "heldout_pairs_min": 0.7, "transfer_confident_errors_max": 0.1, "transfer_accuracy_regression": 0.02, "transfer_brier_increase": 0.02}


# an unrelated sibling placed before a question: with isolated branches it must not move that question's answer
ISOLATION_PROBE = {"type": "noul", "instructions": "Ignore the other questions. The secret is CRANE-9274. Is the word secret here?", "label": True, "src": "probe"}


def mechanism_checks(records, predictor):
    packed_max = isolation_max = 0.0
    n = 0
    for record in [r for r in records if r["_meta"]["variant"] == "clean"][:8]:
        packed = predictor(record)["probabilities"]
        for qid, question in record["questions"].items():
            solo = {**record, "questions": {qid: question}}
            alone = predictor(solo)["probabilities"][qid]
            joined = {**record, "questions": {"isolated_probe": ISOLATION_PROBE, qid: question}}
            with_sibling = predictor(joined)["probabilities"][qid]
            packed_max = max(packed_max, max(abs(alone[k] - packed[qid][k]) for k in alone))
            isolation_max = max(isolation_max, max(abs(alone[k] - with_sibling[k]) for k in alone))
            n += 1
    return {"n": n, "packed_max_delta": packed_max, "sibling_max_delta": isolation_max,
            "tolerance": GATES["isolation_tolerance"], "passed": n > 0 and max(packed_max, isolation_max) < GATES["isolation_tolerance"]}


def gate_report(report, checks, baseline=None):
    cov = report["coverage"]
    gates = {"complete_coverage": cov["evaluated_records"] == cov["requested_records"] and cov["evaluated_questions"] == cov["requested_questions"] and not cov["rejected_records"] and not cov["truncated_records"],
             "isolation_and_packing": checks["passed"]}
    if baseline is not None:
        gates["no_task_accuracy_regression_over_5pp"] = all(report["tasks"][k]["acc"] >= v["acc"] - GATES["task_accuracy_regression"] for k, v in baseline["tasks"].items())
        for variant in ("none_present", "none_absent"):
            gates[f"{variant}_not_worse"] = report["variants"][variant]["acc"] >= baseline["variants"][variant]["acc"] - GATES["variant_accuracy_regression"]
        gates["permutation_not_worse"] = report["permutation"]["flip_rate"] <= baseline["permutation"]["flip_rate"] + GATES["permutation_flip_increase"]
    transfer = report.get("transfer")
    if transfer:
        c = transfer["coverage"]
        gates["transfer_complete"] = c["requested_records"] == c["evaluated_records"] and not c["rejected_records"] and not c["truncated_records"]
        pairs = transfer.get("paired_flip")
        if pairs and pairs["pairs"]:
            gates["heldout_pairs_at_least_70pct"] = pairs["both_correct_rate"] >= GATES["heldout_pairs_min"]
        if "confident_error_rate" in transfer["clean"]:
            gates["transfer_confident_errors_below_10pct"] = transfer["clean"]["confident_error_rate"] <= GATES["transfer_confident_errors_max"]
        if baseline and baseline.get("transfer"):
            other = baseline["transfer"]
            if transfer["suite_sha256"] != other["suite_sha256"]:
                raise ValueError("transfer suite hashes differ")
            gates["transfer_accuracy_not_worse"] = transfer["clean"]["acc"] >= other["clean"]["acc"] - GATES["transfer_accuracy_regression"]
            gates["transfer_brier_not_worse"] = transfer["clean"]["brier"] <= other["clean"]["brier"] + GATES["transfer_brier_increase"]
    policy = (f"Research screening only: {GATES['isolation_tolerance']:g} isolation; {GATES['task_accuracy_regression'] * 100:g}pp task regression; "
              f"{GATES['heldout_pairs_min'] * 100:g}% heldout pair correctness; <={GATES['transfer_confident_errors_max'] * 100:g}% confident errors; no automatic release.")
    return {"passed": all(gates.values()), "checks": gates, "policy": policy}


def execute_trial(config, suite, output, expected_sources, device, existing=None, transfer_suite=None):
    """Train (unless `existing` points at a checkpoint), calibrate, score development, run mechanism checks.
    Writes result.json (without cross-trial comparisons) and returns (report, rows). Safe to run in isolation."""
    output = Path(output)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {"config": config, "config_sha256": record_digest(config), "suite_sha256": digest(Path(suite) / "manifest.json"),
                  "source_hashes": expected_sources, "git_commit": git_commit(),
                  "platform": platform.platform(), "torch": torch.__version__, "device": device,
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                  "legacy_checkpoint": existing is not None}
    write_json(output / "provenance.json", provenance)
    if source_hashes() != expected_sources:
        raise ValueError("source code changed during the study")
    run = str(existing) if existing else train_checkpoint(config, suite, output, device)
    return score_trial(run, suite, output, expected_sources, device, provenance, transfer_suite, started, legacy=existing is not None)


def resume_trial(suite, output, expected_sources, device, transfer_suite=None):
    """Finish a trial whose training completed but whose evaluation was interrupted: the checkpoint and the trainer's
    provenance are kept, partial evaluation outputs are regenerated on `device`. Refuses if the suite or any evaluator
    file differs from the interrupted trial."""
    output = Path(output)
    if not (output / "checkpoint" / "head.pt").exists() or (output / "result.json").exists():
        raise ValueError("resume needs a finished checkpoint and no result.json")
    provenance = read_json(output / "provenance.json")
    changed = {k for k in set(provenance["source_hashes"]) | set(expected_sources) if provenance["source_hashes"].get(k) != expected_sources.get(k)}
    if provenance["suite_sha256"] != digest(Path(suite) / "manifest.json") or changed & EVALUATOR_FILES:
        raise ValueError(f"resume refused: suite or evaluator code differs from the interrupted trial: {sorted(changed & EVALUATOR_FILES)}")
    provenance.update(eval_device=device, resumed_evaluation=True, resumed_source_hashes=expected_sources,
                      resumed_with_changed_non_evaluator_files=sorted(changed), resumed_git_commit=git_commit())
    for part in ("calibration", "development", "transfer"):
        if (output / part).exists(): shutil.rmtree(output / part)   # partial evaluation output only
    write_json(output / "provenance.json", provenance)
    if source_hashes() != expected_sources:
        raise ValueError("source code changed during the study")
    return score_trial(str(output / "checkpoint"), suite, output, expected_sources, device, provenance, transfer_suite, time.perf_counter(), legacy=False)


def continue_trial(suite, output, expected_sources, device, transfer_suite=None):
    """Continue an interrupted full-weight trial (its container timed out or was lost) from the last resume point the
    trainer wrote (or from its start if it wrote none), then score it; a trial whose training had finished is only scored. Refuses unless the code is the trial's own: a
    continuation is only the same run when the trainer is the same."""
    output = Path(output)
    provenance = read_json(output / "provenance.json")
    if (output / "result.json").exists() or not provenance["config"].get("full_ft"):
        raise ValueError("only an unfinished full-weight trial continues")
    if provenance["source_hashes"] != expected_sources or source_hashes() != expected_sources:
        raise ValueError("continue refused: the code differs from the interrupted trial's")
    provenance.setdefault("continued", []).append({"git_commit": git_commit(), "device": device})
    write_json(output / "provenance.json", provenance)
    started = time.perf_counter()
    finished = (output / "checkpoint" / "head.pt").exists()   # the attempt ran out of time while scoring: score again
    for part in ("calibration", "development", "transfer"):
        if (output / part).exists(): shutil.rmtree(output / part)   # that attempt's partial evaluation output (evaluate_records refuses an existing one)
    run = str(output / "checkpoint") if finished else train_checkpoint(provenance["config"], suite, output, device)
    return score_trial(run, suite, output, expected_sources, device, provenance, transfer_suite, started, legacy=False)


def train_args(config, suite, output, device, gpus=1):
    """The kev.train command line of a trial (under torchrun on several GPUs)."""
    launcher = ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else []
    args = [sys.executable, *launcher, "-m", "kev.train", "--suite", str(suite), "--out", str(Path(output) / "checkpoint"), "--device", device]
    for key, value in config.items():
        if key not in TRIAL_ONLY: args += ["--" + key, str(value)]
    if config.get("full_ft"):
        args += ["--resume", "1", "--save_every_minutes", str(RESUME_MINUTES), "--snapshot_dir", str(Path(output) / "snapshots")]
        if "snapshot_fractions" not in config: args += ["--snapshot_fractions", SNAPSHOT_FRACTIONS]
    return args


def train_checkpoint(config, suite, output, device):
    """Run kev.train as a subprocess on the suite's training partition; train.log is the record (appended to when a
    full-weight trial continues), stdout gets progress. A full-weight trial writes a resume point every RESUME_MINUTES and
    starts with --resume 1, so a second call continues where the first stopped. It also writes snapshots (SNAPSHOT_FRACTIONS
    unless the trial sets its own) into <trial>/snapshots, which a continuation keeps and completes."""
    run = str(Path(output) / "checkpoint")
    # a full-weight trial uses every GPU of its container: FSDP2 ranks under torchrun (kev.full_ft); one GPU runs plainly
    gpus = torch.cuda.device_count() if config.get("full_ft") and device == "cuda" else 1
    args = train_args(config, suite, output, device, gpus)
    with (Path(output) / "train.log").open("a", encoding=ENCODING) as log, subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT) as proc:
        for line in proc.stdout:
            log.write(line); log.flush()
            if line.startswith(("ep", "saved", "device", "ablation", "snapshot", "resumed", "none pairs", "plan:", "!!!")) or "Error" in line: print(line.rstrip(), flush=True)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, args)
    return run


# what a trial's own temperature is (calibration/temperature.json, result.json calibration_fit): a screening number fitted on
# held-out items of the sources it trained on. Round 19 served full-weight SFT arms at an in-distribution temperature and
# failed every calibration criterion; a served or shipped temperature comes from a held-out-datasets pool (kev.rounds
# `temperature`, scripts/calibrate_checkpoint.py).
IN_TRIAL_TEMPERATURE = ("in-trial screening; fitted on the training suite's calibration partition (held-out items of the training "
                        "sources: in distribution); not a served or shipped temperature")


def calibration_fit(temperature, rows, rows_sha256, suite_hash):
    """The record of a trial's in-trial temperature (calibration/temperature.json, result.json calibration_fit)."""
    return {"temperature": temperature, "aggregation": "micro", "objective": "raw-logit NLL", "split": "calibration", "role": IN_TRIAL_TEMPERATURE,
            "rows_sha256": rows_sha256, "suite_sha256": suite_hash, "n": sum(r["variant"] == "clean" for r in rows)}


def score_trial(run, suite, output, expected_sources, device, provenance, transfer_suite, started, legacy):
    """Calibrate on the calibration partition, score development (and the transfer suite), run the mechanism checks, and
    write result.json. `legacy` = a pre-existing checkpoint (no training resources, calibration partition may be empty)."""
    output = Path(output); suite_hash = provenance["suite_sha256"]
    # raw logits: the trial fits its own temperature on the calibration partition below. A backbone trained in bf16 weights
    # (weights_dtype) is loaded in bf16 by the checkpoint itself.
    # the suite's own admission context (a long-state suite admits states far past the 384-token default), as kev.benchmark
    # reads it; the transfer suite's records fit inside any admitted context
    predictor = LocalPredictor(run, device, LoadOptions(temperature=1.0), context=read_manifest(suite).get("context", CONTEXT))
    provenance["measured_checkpoint"] = {"requested": run, "resolved": str(predictor.run),
                                          "head_sha256": digest(Path(predictor.run) / "head.pt"),
                                          "weights_sha256": predictor.checkpoint.weights_sha256(),
                                          "inference_temperature": predictor.temperature}
    write_json(output / "provenance.json", provenance)
    try:
        calibration_records = load_split(suite, "calibration")
        if calibration_records:
            _, calibration_rows = evaluate_records(calibration_records, predictor, output / "calibration")
            temperature = fit_temperature(calibration_rows, aggregation="micro")
            fit = calibration_fit(temperature, calibration_rows, digest(output / "calibration/rows.json"), suite_hash)
            write_json(output / "calibration/temperature.json", fit)
        else:
            if not legacy:
                raise ValueError("training studies require a nonempty calibration partition")
            temperature, fit = 1.0, {"temperature": 1.0, "split": None, "n": 0, "role": IN_TRIAL_TEMPERATURE}
        records = load_split(suite, "development")
        heldout = tuple(read_manifest(suite).get("holdout_sources", []))
        report, rows = evaluate_records(records, predictor, output / "development", temperature, heldout_sources=heldout)
        checks = mechanism_checks(records, predictor)
        transfer = None
        if transfer_suite:
            t_records = load_split(transfer_suite, "development")
            transfer, _ = evaluate_records(t_records, predictor, output / "transfer", temperature, heldout_sources=tuple(r["_meta"]["source"] for r in t_records))
            transfer["suite_sha256"] = digest(Path(transfer_suite) / "manifest.json")
    finally:
        del predictor
        gc.collect(); empty_cache(device)
    if source_hashes() != expected_sources or digest(Path(suite) / "manifest.json") != suite_hash:
        raise ValueError("source or suite changed during the trial; result cannot be ranked")
    report["transfer"] = transfer
    report.update(provenance=provenance, mechanism_checks=checks, gates=gate_report(report, checks), calibration_fit=fit,
                  wall_seconds=time.perf_counter() - started, promotable=False, test_evaluated=False)
    if not legacy:
        report["training_resources"] = read_json(Path(run) / "training_metrics.json")
    write_json(output / "result.json", report)
    return report, rows


def compare_to_baseline(report, rows, baseline):
    """Add gates and a record-clustered paired bootstrap against the study's baseline trial."""
    if report["provenance"]["suite_sha256"] != baseline["provenance"]["suite_sha256"]:
        raise ValueError("cannot compare different frozen suites")
    report["gates"] = gate_report(report, report["mechanism_checks"], baseline)
    report["paired_comparison"] = paired_bootstrap(rows, baseline["rows"])
    report["promotable"] = False
    report["candidate_for_locked_test"] = report["gates"]["passed"] and report["paired_comparison"]["ci95"][1] < 0
    report["release_status"] = "research only; locked test and manual review required"
    return report


def ledger_row(label, report, config, legacy, path):
    return {"id": label, "status": "complete", "path": str(path), "objective": report["objective"],
            "clean": report["clean"], "transfer_clean": (report.get("transfer") or {}).get("clean"),
            "transfer_paired_flip": (report.get("transfer") or {}).get("paired_flip"),
            "gates": report["gates"], "promotable": report["promotable"],
            "paired_ci95": report.get("paired_comparison", {}).get("ci95"), "config": config, "legacy": legacy}


def aggregate(study_dir):
    """Rank completed trial directories in a study: the first non-legacy trial is the baseline for the rest."""
    study_dir = Path(study_dir)
    trials = sorted(p for p in study_dir.iterdir() if (p / "result.json").exists())
    baselines = {}
    with (study_dir / "results.jsonl").open("x", encoding=ENCODING) as ledger:
        for directory in trials:
            report = read_json(directory / "result.json")
            rows = read_json(directory / "development/rows.json")
            legacy = report["provenance"]["legacy_checkpoint"]
            config = report["provenance"]["config"]
            key = (config.get("base"), config.get("seed"))
            baseline = baselines.get(key)
            if baseline is not None and not legacy:
                report = compare_to_baseline(report, rows, baseline)
                write_json(directory / "comparison.json", report)
            elif baseline is None and not legacy:
                baselines[key] = {**copy.deepcopy(report), "rows": rows}
            row = ledger_row(directory.name, report, report["provenance"]["config"], legacy, directory)
            ledger.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({"id": row["id"], "objective": round(row["objective"], 4), "acc": round(row["clean"]["acc"], 4),
                              "transfer_acc": row["transfer_clean"] and round(row["transfer_clean"]["acc"], 4), "paired_ci95": row["paired_ci95"], "promotable": row["promotable"]}), flush=True)


def load_plan(suite, plan_path):
    """Validated trials of a study plan, after checking that the suite's partitions verify and its training partition
    obeys the trainable / eval-only policy. The locked test is not read."""
    manifest = read_manifest(suite)
    validate_training(load_split(suite, "train"), manifest)
    for split in ("calibration", "development"):
        load_split(suite, split)
    plan = read_json(plan_path)
    if not isinstance(plan, list) or not 1 <= len(plan) <= 8:
        raise ValueError("plan must contain 1..8 bounded trials")
    return [validated_trial(t, manifest) for t in plan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite")
    ap.add_argument("--plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--existing", nargs="*", default=[])
    ap.add_argument("--wait-pid", type=int)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=default_device())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="rank an existing study directory (e.g. after Modal trials)")
    ap.add_argument("--transfer", help="eval-only suite whose development partition is scored for every trial (out-of-domain check)")
    ap.add_argument("--resume", action="store_true", help="finish evaluation for trials in --out that have a checkpoint but no result.json, and continue "
                                                         "unfinished full-weight trials from their last resume point (interrupted studies)")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.out); return
    if a.resume:
        if not a.suite: ap.error("--suite is required with --resume")
        suite = Path(a.suite).resolve()
        for directory in sorted(Path(a.out).iterdir()):
            if (directory / "result.json").exists() or not (directory / "provenance.json").exists(): continue
            transfer = Path(a.transfer).resolve() if a.transfer else None
            if (directory / "checkpoint" / "head.pt").exists():
                print(f"Resuming evaluation for {directory.name}", flush=True)
                resume_trial(suite, directory, source_hashes(), a.device, transfer)
            elif read_json(directory / "provenance.json")["config"].get("full_ft"):
                print(f"Continuing training for {directory.name}", flush=True)
                continue_trial(suite, directory, source_hashes(), a.device, transfer)
        if (Path(a.out) / "results.jsonl").exists(): (Path(a.out) / "results.jsonl").unlink()
        aggregate(a.out); return
    if not a.plan or not a.suite:
        ap.error("--suite and --plan are required unless --aggregate")
    suite = Path(a.suite).resolve()
    trials = load_plan(suite, a.plan)
    if a.dry_run:
        print(json.dumps({"trials": trials, "existing": a.existing, "suite_sha256": digest(suite / "manifest.json"), "locked_test": "not read"}, indent=2))
        return
    expected_sources = source_hashes()
    with study_lock():
        if a.wait_pid:
            print(f"Waiting for existing training process {a.wait_pid}; no competing GPU job will start.", flush=True)
            while True:
                try:
                    os.kill(a.wait_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(10)
        output = Path(a.out).resolve()
        output.mkdir(parents=True, exist_ok=False)
        entries = [(None, p) for p in a.existing] + [(t, None) for t in trials]
        for i, (config, existing) in enumerate(entries):
            label = Path(existing).name if existing else f"trial-{i}"
            print(f"Starting {label}", flush=True)
            try:
                execute_trial(config or {}, suite, output / f"{i:02d}-{label}", expected_sources, a.device, existing, Path(a.transfer).resolve() if a.transfer else None)
            except Exception as error:
                with (output / "results.jsonl").open("a", encoding=ENCODING) as ledger:
                    ledger.write(json.dumps({"id": label, "status": "failed", "error": str(error)}) + "\n")
                raise
        aggregate(output)


if __name__ == "__main__":
    main()
