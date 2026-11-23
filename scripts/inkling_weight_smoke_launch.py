"""Isolated Sandbox launcher called by modal_app.inkling_weight_smoke via Python."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

from scripts import inkling_weight_smoke as smoke

APP_NAME = "kev-research"
STUDY_TAG = "inkling2-mini-plumbing"
ROOT = Path(__file__).resolve().parents[1]
MARKER_PATH = Path.home() / "inkling-feasibility" / "inkling2-mini-h200-smoke.marker.json"
DEFAULT_REPORT_PATH = Path.home() / "inkling-feasibility" / "inkling2-mini-h200-smoke.json"
HUB_BASE_URL = "https://huggingface.co"
HUB_METADATA_TIMEOUT_SECONDS = 10
REQUIRED_HUB_FILES = ("config.json", "tokenizer_config.json", "tokenizer.json")
SMOKE_ENV = {
    "HF_HOME": "/tmp/inkling-hf",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_TOKEN": "",
    "HF_DATASETS_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}


class HubMetadataError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def terminate(sandbox, name: str) -> dict:
    try:
        target = sandbox if sandbox is not None else modal.Sandbox.from_name(APP_NAME, name)
        return {"exit_code": target.terminate(wait=True)}
    except Exception as exc:
        return {"error_type": type(exc).__name__}


def result_from(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None


def _has_active_smoke_sandbox() -> bool:
    return next(iter(modal.Sandbox.list(tags={"study": STUDY_TAG})), None) is not None


def _hub_request(path: str, method: str) -> bytes:
    url = f"{HUB_BASE_URL}{path}"
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Accept": "application/json", "User-Agent": "kev-inkling-metadata-preflight/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HUB_METADATA_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise HubMetadataError(f"Hub {method} {path} returned HTTP {response.status}")
            if method == "HEAD":
                return b""
            body = response.read(65537)
            if len(body) > 65536:
                raise HubMetadataError(f"Hub metadata response exceeded 64 KiB: {path}")
            return body
    except urllib.error.HTTPError as exc:
        detail = exc.read(300).decode("utf-8", "replace").replace("\n", " ")
        raise HubMetadataError(f"Hub {method} {path} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        raise HubMetadataError(f"Hub {method} {path} failed: {type(reason).__name__}") from exc


def verify_pinned_hub_metadata() -> dict:
    revision = smoke.MODEL_REVISION
    model_id = smoke.MODEL_ID
    metadata_path = f"/api/models/{model_id}/revision/{revision}"
    try:
        metadata = json.loads(_hub_request(metadata_path, "GET"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HubMetadataError(f"Hub returned invalid metadata for {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise HubMetadataError(f"Hub returned invalid metadata for {metadata_path}")
    if metadata.get("id") != model_id or metadata.get("sha") != revision:
        raise HubMetadataError(f"Hub did not resolve the exact pinned revision: {metadata_path}")
    if metadata.get("private") or metadata.get("gated"):
        raise HubMetadataError(f"Pinned model is not anonymously public: {metadata_path}")

    file_statuses = {}
    for filename in REQUIRED_HUB_FILES:
        path = f"/{model_id}/resolve/{revision}/{filename}"
        _hub_request(path, "HEAD")
        file_statuses[filename] = 200
    return {
        "model_id": model_id,
        "revision": revision,
        "resolved_sha": metadata["sha"],
        "required_files": file_statuses,
    }


def launch(report_path: Path = DEFAULT_REPORT_PATH) -> dict:
    report_path = Path(report_path)
    if MARKER_PATH.resolve().is_relative_to(ROOT) or report_path.resolve().is_relative_to(ROOT):
        raise ValueError("smoke marker and report must remain outside the repository")
    if report_path.resolve() == MARKER_PATH.resolve():
        raise ValueError("report path must not replace the durable launch marker")
    if not report_path.parent.is_dir():
        raise FileNotFoundError(report_path.parent)
    if report_path.exists():
        raise FileExistsError(report_path)
    if MARKER_PATH.exists():
        raise FileExistsError(MARKER_PATH)
    if smoke.cost_bound() > smoke.MAX_COST_USD:
        raise RuntimeError("configured smoke cost bound exceeds $20")
    hub_metadata = verify_pinned_hub_metadata()

    app = modal.App.lookup(APP_NAME)
    if _has_active_smoke_sandbox():
        raise RuntimeError("an Inkling smoke sandbox exists; inspect and terminate it, never relaunch")

    run_id = uuid.uuid4().hex
    name = f"inkling2-mini-{run_id}"
    tags = {"study": STUDY_TAG, "run_id": run_id}
    marker = {
        "run_id": run_id,
        "app_name": APP_NAME,
        "sandbox_name": name,
        "sandbox_tags": tags,
        "hub_metadata": hub_metadata,
        "status": "creating",
        "cost_bound_usd": smoke.cost_bound(),
    }
    smoke.claim_marker(MARKER_PATH, marker)
    started = time.monotonic()
    deadline = started + smoke.END_TO_END_WATCHDOG_SECONDS
    times = {"create_requested_at": timestamp()}
    sandbox = None
    sandbox_id = None
    create_attempted = False
    remote = None
    failure = None
    exec_started = None
    exec_ended = None

    watchdog = threading.Timer(smoke.END_TO_END_WATCHDOG_SECONDS, lambda: terminate(sandbox, name))
    watchdog.daemon = True
    watchdog.start()
    try:
        create_attempted = True
        sandbox = modal.Sandbox.create(
            app=app,
            name=name,
            tags=tags,
            image=modal.Image.debian_slim(python_version="3.13"),
            gpu="H200",
            cpu=(smoke.SANDBOX_CPU_CORES, smoke.SANDBOX_CPU_CORES),
            memory=(smoke.SANDBOX_RAM_GIB * 1024, smoke.SANDBOX_RAM_GIB * 1024),
            timeout=smoke.SANDBOX_SECONDS,
            idle_timeout=30,
            secrets=[],
            env=SMOKE_ENV,
        )
        sandbox_id = sandbox.object_id
        times["sandbox_created_at"] = timestamp()
        if time.monotonic() >= deadline:
            raise TimeoutError("end-to-end watchdog expired during Sandbox creation")

        for source, destination in (
            (ROOT / "kev/model.py", "/tmp/kev_model.py"),
            (ROOT / "kev/device.py", "/tmp/kev_device.py"),
            (ROOT / "scripts/inkling_weight_smoke.py", "/tmp/inkling_weight_smoke.py"),
        ):
            sandbox.filesystem.write_bytes(source.read_bytes(), destination)
        remaining = min(smoke.SANDBOX_SECONDS, math.floor(deadline - time.monotonic()))
        if remaining <= 0:
            raise TimeoutError("end-to-end watchdog expired before smoke execution")

        times["sandbox_exec_started_at"] = timestamp()
        exec_started = time.monotonic()
        process = sandbox.exec(
            "python",
            "-u",
            "/tmp/inkling_weight_smoke.py",
            timeout=remaining,
            env=SMOKE_ENV,
            secrets=[],
        )
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        returncode = process.wait()
        exec_ended = time.monotonic()
        times["sandbox_exec_ended_at"] = timestamp()
        remote = result_from(stdout)
        if remote is None:
            raise RuntimeError(f"smoke returned no JSON result (exit={returncode}; stderr={stderr[-1000:]})")
        if returncode or remote.get("status") != "passed":
            failure = {"error_type": "RemoteSmokeFailure", "stage": remote.get("stage"), "details": remote}
    except BaseException as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
    finally:
        watchdog.cancel()
        times["termination_started_at"] = timestamp()
        shutdown = terminate(sandbox, name) if create_attempted else {"not_created": True}
        ended = time.monotonic()
        times["sandbox_terminated_at"] = timestamp() if shutdown.get("exit_code") is not None else None

    stopped = shutdown.get("exit_code") is not None
    if not stopped and failure is None:
        failure = {"error_type": "ShutdownUnobserved"}
    elapsed = max(0, math.ceil(ended - started))
    measured_seconds = min(smoke.SANDBOX_SECONDS, elapsed) if stopped else smoke.SANDBOX_SECONDS
    usage_cost = smoke.cost_breakdown(measured_seconds)
    status = "passed" if stopped and failure is None and remote and remote.get("status") == "passed" else "failed"
    if not stopped:
        status = "failed_shutdown"
    report = {
        "run_id": run_id,
        "app_name": APP_NAME,
        "sandbox_name": name,
        "sandbox_id": sandbox_id,
        "sandbox_tags": tags,
        "status": status,
        "interpretation": "PLUMBING SMOKE ONLY; random untrained pointer head; no quality verdict",
        "remote_result": remote,
        "hub_metadata": hub_metadata,
        "failure": failure,
        "timestamps": times,
        "sandbox_terminated": stopped,
        "shutdown": shutdown,
        "elapsed_since_create_request_seconds": round(ended - started, 3),
        "elapsed_exec_seconds": round(exec_ended - exec_started, 3) if exec_ended is not None else None,
        "usage_estimate": {
            "label": "conservative local wall-clock estimate; Modal billing is authoritative",
            "sandbox_seconds_estimate": measured_seconds,
            "h200_seconds_estimate": measured_seconds,
            "cpu_core_seconds_estimate": measured_seconds * smoke.SANDBOX_CPU_CORES,
            "ram_gib_seconds_estimate": measured_seconds * smoke.SANDBOX_RAM_GIB,
            "image_build_allowance_seconds": smoke.IMAGE_BUILD_ALLOWANCE_SECONDS,
            "egress_allowance_gib_not_enforced": smoke.MAX_EGRESS_GIB,
            "cost_breakdown_usd": usage_cost,
            "estimated_total_cost_usd": usage_cost["total"],
        },
        "cost_bound_usd": smoke.cost_bound(),
        "cost_bound_breakdown_usd": smoke.cost_breakdown(),
        "cost_bound_basis": "Modal rates with 300-second build and 1 MiB egress allowances; allowances are modeled, not service-enforced caps",
        "cost_bound_is_hard_service_cap": False,
        "sandbox_timeout_seconds": smoke.SANDBOX_SECONDS,
        "end_to_end_watchdog_seconds": smoke.END_TO_END_WATCHDOG_SECONDS,
    }
    smoke.write_report(report_path, report)
    smoke.update_marker(
        MARKER_PATH,
        {
            "status": status,
            "report": str(report_path),
            "sandbox_id": sandbox_id,
            "sandbox_terminated": stopped,
            "estimated_total_cost_usd": usage_cost["total"],
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    report = launch(parser.parse_args().report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
