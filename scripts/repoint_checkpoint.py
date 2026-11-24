"""Relocate a checkpoint's base-model reference.

Rewrites ``head.pt``'s ``base`` -- the one field kev's loader reads -- to a local path
and clears ``base_revision``, leaving every other field (pointer weights, temperature,
provenance) untouched. When ``adapter_config.json`` is present, its own base reference
is updated to match, only so peft does not warn.

Use this to serve a checkpoint from a local base directory without network access --
an offline host, or a container with the weights mounted:

    python scripts/repoint_checkpoint.py --run runs/kev-mine --base /models/base

The target path is not checked for existence: it often lives only inside the deployment
target. The operation is idempotent.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from kev.checkpoint import read_meta, write_meta

DEFAULT_BASE = "/models/base"


def repoint(run, base=DEFAULT_BASE):
    """Rewrite head.pt's ``base`` to ``base``; sync adapter_config.json if present.
    Returns (old_base, new_base, changed)."""
    meta = read_meta(run)
    old = meta.base
    if old == base and meta.base_revision is None:
        return old, base, False

    meta.extra.update(
        base_repointed_from=old,
        base_repointed_to=base,
        base_repointed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    meta.base, meta.base_revision = base, None
    write_meta(run, meta)

    adapter = Path(run, "adapter_config.json")
    if adapter.is_file():
        cfg = json.loads(adapter.read_text(encoding="utf-8"))
        if cfg.get("base_model_name_or_path") != base:
            cfg["base_model_name_or_path"] = base
            cfg["revision"] = None
            text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
            adapter.write_text(text, encoding="utf-8")

    return old, base, True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Point a checkpoint's head.pt base at a local path.")
    ap.add_argument("--run", required=True, help="checkpoint directory containing head.pt")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"new base path (default: {DEFAULT_BASE})")
    ap.add_argument("--check", action="store_true", help="print the current base without writing")
    args = ap.parse_args(argv)

    if not Path(args.run, "head.pt").is_file():
        print(f"repoint_checkpoint: {args.run} has no head.pt", file=sys.stderr)
        return 1
    if args.check:
        print(read_meta(args.run).base)
        return 0

    old, base, changed = repoint(args.run, args.base)
    print(f"{old} -> {base}" if changed else f"already {old}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
