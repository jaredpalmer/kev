"""Rows that must not be public (they carry a private corpus's record ids and option keys, e.g. a trial's `sft-v1`
development rows) live in a private Hub dataset; git holds a pointer manifest with each file's sha256 and private path.

    uv run python scripts/private_rows.py upload  --manifest runs/r19-readout/private-rows.json runs/r19-27b-lr2e6/00-trial-0/development/rows.json ...
    uv run python scripts/private_rows.py restore --manifest runs/r19-readout/private-rows.json   # into place, sha256-checked

`upload` pushes the files in one commit to DATASET under `runs/r<N>/<local path below runs/>` and writes the manifest,
pinned to that commit. `restore` downloads every file that is not in place yet (accounts with access: `hf auth login` or
HF_TOKEN) and raises PermissionError naming the dataset for everyone else, as kev.suite.load_split does for a private
suite partition.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import digest, read_json, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET = "jaredpalmer/kev-private-train"   # the SFT corpus's private dataset (PLAN.md, "Data policy for the SFT work")


def private_path(local, round_number):
    """runs/r19-27b-lr2e6/00-trial-0/development/rows.json -> runs/r19/r19-27b-lr2e6/00-trial-0/development/rows.json"""
    return f"runs/r{round_number}/{Path(local).relative_to('runs').as_posix()}"


def upload(files, manifest, round_number, dataset=DATASET, root=ROOT):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    if not api.repo_info(dataset, repo_type="dataset").private: raise SystemExit(f"{dataset} is not private; refusing to upload")
    ops = [CommitOperationAdd(private_path(f, round_number), str(root / f)) for f in files]
    commit = api.create_commit(dataset, ops, commit_message=f"round {round_number}: rows that carry private record ids", repo_type="dataset")
    write_json(root / manifest, {"dataset": dataset, "revision": commit.oid, "note": "rows carrying private record ids and option keys; "
                                 "restore with scripts/private_rows.py restore --manifest <this file>",
                                 "files": [{"path": f, "private_path": private_path(f, round_number), "sha256": digest(root / f)} for f in files]})
    print(f"uploaded {len(files)} files to {dataset}@{commit.oid[:10]}; wrote {manifest}")


def restore(manifest, root=ROOT):
    """Put every file of the manifest in place under root (sha256-checked); -> the paths fetched."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    m, fetched = read_json(Path(root) / manifest), []
    for f in m["files"]:
        target = Path(root) / f["path"]
        if target.exists() and digest(target) == f["sha256"]: continue
        try:
            cached = hf_hub_download(m["dataset"], f["private_path"], repo_type="dataset", revision=m["revision"])
        except (RepositoryNotFoundError, GatedRepoError) as e:   # a private dataset answers "not found" to anyone without access
            raise PermissionError(f"{f['path']} is only in {m['dataset']}, which is private to its owner; `hf auth login` with access") from e
        if digest(cached) != f["sha256"]: raise ValueError(f"{f['private_path']}@{m['revision'][:10]} does not match the manifest's sha256")
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(cached, target); fetched.append(f["path"])
    return fetched


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upload"); up.add_argument("--manifest", required=True); up.add_argument("--round", type=int, required=True); up.add_argument("files", nargs="+")
    down = sub.add_parser("restore"); down.add_argument("--manifest", required=True)
    a = ap.parse_args()
    if a.cmd == "upload": upload(a.files, a.manifest, a.round)
    else: print(f"restored {restore(a.manifest) or 'nothing (every file in place)'}")


if __name__ == "__main__":
    main()
