"""Offline research gates for Inkling-2-Loop; this launcher never loads model weights.

Prerequisite checkout:
    git clone https://github.com/huggingface/transformers.git upstream
    git -C upstream fetch origin 6173a2f679ff4ed393afc499aefc569da7e68bc1
    git -C upstream switch --detach 6173a2f679ff4ed393afc499aefc569da7e68bc1

Example: .venv/bin/python scripts/inkling_compat.py --upstream /path/to/pinned-transformers
         --report $HOME/inkling-feasibility/inkling-compat-report.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PINNED_REVISION = "6173a2f679ff4ed393afc499aefc569da7e68bc1"
REQUIRED_GATES = ("source_pin", "tiny_architecture", "tokenizer_embedding", "canonical_text_loader", "row_execution", "converter_numeric", "selective_token_parity", "peft_modules_to_save")


def verify_upstream(path: Path) -> dict:
    root = path.resolve()
    if not (root / "src/transformers/__init__.py").is_file():
        raise ValueError(f"{root} is not a Transformers source checkout")
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"git {' '.join(args)} failed")
        return result.stdout.strip()
    top = Path(git("rev-parse", "--show-toplevel")).resolve()
    revision = git("rev-parse", "HEAD")
    if top != root:
        raise ValueError(f"--upstream must be the checkout root: {root}")
    if revision != PINNED_REVISION:
        raise ValueError(f"upstream revision mismatch: expected {PINNED_REVISION}, got {revision}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"upstream checkout is dirty or has untracked files: {status}")
    return {"root": str(root), "revision": revision, "clean_worktree": True}


def exit_code(report: dict) -> int:
    gates = {gate["name"]: gate for gate in report.get("gates", ())}
    return int(any(name not in gates or gates[name].get("status") != "pass" for name in REQUIRED_GATES))


def loading_info_json(info: dict) -> dict:
    mismatched = []
    for item in sorted(info.get("mismatched_keys", ()), key=str):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise TypeError("unexpected Transformers mismatched_keys entry")
        key, checkpoint_shape, model_shape = item
        mismatched.append({"key": key, "checkpoint_shape": list(checkpoint_shape), "model_shape": list(model_shape)})
    return {"missing_keys": sorted(info.get("missing_keys", ())),
            "unexpected_keys": sorted(info.get("unexpected_keys", ())),
            "mismatched_keys": mismatched, "error_msgs": list(info.get("error_msgs", ()))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="clean Git checkout at the pinned Transformers revision")
    parser.add_argument("--report", required=True, type=Path, help="machine-readable JSON report path")
    args = parser.parse_args(argv)
    try:
        source = verify_upstream(args.upstream)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    root = Path(source["root"])
    sys.path[:0] = [str(root / "src"), str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        import transformers
        actual = Path(transformers.__file__).resolve()
        expected = (root / "src/transformers").resolve()
        if not actual.is_relative_to(expected):
            raise RuntimeError(f"Transformers imported from {actual}, expected source under {expected}")
        import inkling_compat_probe
    except Exception as exc:
        print(f"ERROR: pinned Transformers imports failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    source["transformers_file"] = str(actual)
    return inkling_compat_probe.run(args.report, source)


if __name__ == "__main__":
    raise SystemExit(main())
