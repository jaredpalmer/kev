"""A private Hugging Face mirror of full-weight checkpoints: long-term storage for the snapshots (kev.full_ft.SnapshotWriter)
and final checkpoints that full-weight trials keep on the Modal runs volume. The volume copy stays primary: reads and
benchmarks load /runs/... paths, and a failed upload only logs (loudly) and never fails training.

    mirror("/runs/r19-27b/00-trial-0/snapshots/step-0000389/checkpoint", "jaredpalmer/kev-snapshots")
    -> the directory's files at <study>/<trial>/step-0000389/ in the repo (a final checkpoint: <study>/<trial>/final/)

Refuses a repo that is not private (it creates a missing one private). The upload commit is recorded next to the files
on the volume: in a snapshot's snapshot.json under "hub", in a final checkpoint's hub.json. modal_app.run_mirror runs
this in its own CPU container with the HF token of the Modal secret it mounts (HF_TOKEN in the environment; never printed).
"""
import datetime
import re
from pathlib import Path

from .full_ft import SNAPSHOT_INFO
from .suite import read_json, write_json

DEFAULT_REPO = "jaredpalmer/kev-snapshots"
RECORD = "hub.json"   # a final checkpoint's upload record (a snapshot's goes into its snapshot.json)
IGNORED = ["resume/*", ".*", RECORD]   # resume points are optimizer state; dotfiles are partial writes
ATTEMPTS = 2          # an upload is tried once more after a failure


def destination(path, root="/runs"):
    """Where a checkpoint directory goes in the repo: <study>/<trial>/<step-N> for a snapshot, <...>/final otherwise."""
    rel = Path(path).relative_to(root).as_posix()
    if match := re.fullmatch(r"(.+)/snapshots/(step-\d+)/checkpoint", rel): return f"{match[1]}/{match[2]}"
    return rel.removesuffix("/checkpoint") + "/final"


def is_snapshot(path):
    return (Path(path) / SNAPSHOT_INFO).exists()


def recorded(path):
    """The upload record a checkpoint directory carries, or None."""
    path = Path(path)
    if is_snapshot(path): return read_json(path / SNAPSHOT_INFO).get("hub")
    return read_json(path / RECORD) if (path / RECORD).exists() else None


def record(path, entry):
    path = Path(path)
    if is_snapshot(path): write_json(path / SNAPSHOT_INFO, {**read_json(path / SNAPSHOT_INFO), "hub": entry}, atomic=True)   # stays complete throughout
    else: write_json(path / RECORD, entry, atomic=True)


def ensure_private(api, repo):
    """Create `repo` private if it is missing; refuse (PermissionError) if it exists and is not private."""
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="model").private:
        raise PermissionError(f"{repo} is not a private repo; checkpoints are only mirrored to private repos")


def mirror(path, repo=DEFAULT_REPO, api=None, root="/runs", force=False, log=print):
    """Upload one complete checkpoint directory (a snapshot with its snapshot.json, or a checkpoint with head.pt) to the
    private `repo` and record the commit. -> the record, or None when it was not uploaded (every failure is logged loudly
    and nothing is raised: the volume copy is primary). Skips a directory already recorded for this repo unless `force`."""
    path = Path(path)
    if not (path / "head.pt").exists() or (path.parent.parent.name == "snapshots" and not is_snapshot(path)):
        log(f"!!! mirror: {path} is not a complete checkpoint (no head.pt, or a snapshot without {SNAPSHOT_INFO}); not uploaded"); return None
    if not force and (previous := recorded(path)) and previous.get("repo") == repo:
        log(f"mirror: {path} is at {repo}/{previous['path']} already (commit {previous['commit']})"); return previous
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()   # HF_TOKEN from the environment (the Modal secret)
    dest = destination(path, root)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            ensure_private(api, repo)
            info = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(path), path_in_repo=dest,
                                     ignore_patterns=IGNORED, commit_message=f"mirror {dest}")
        except PermissionError as error:   # not a transient failure: never retried, never uploaded
            log(f"!!! mirror of {path} refused: {error}"); return None
        except Exception as error:   # noqa: BLE001 - loud, retried once, never raised
            log(f"!!! mirror of {path} to {repo} failed (attempt {attempt}/{ATTEMPTS}): {type(error).__name__}: {str(error)[:300]}")
            continue
        entry = {"repo": repo, "path": dest, "commit": info.oid, "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        record(path, entry)
        log(f"mirror: {path} -> {repo}/{dest} (commit {info.oid})")
        return entry
    log(f"!!! mirror of {path} to {repo} gave up after {ATTEMPTS} attempts; the volume copy is unaffected (retry: modal_app.py::mirror_snapshots)")
    return None
