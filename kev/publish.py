"""Publish a trained run to the Hugging Face Hub.

Repo naming: jaredpalmer/kev-<base size>, e.g. kev-0.5b (Qwen2.5-0.5B), kev-0.6b (Qwen3-0.6B), kev-4b, kev-8b.
The exact base checkpoint is recorded in the model card's `base_model` field and in head.pt.

    uv run python -m kev.publish --run runs/kev --repo jaredpalmer/kev-0.5b
    uv run python -m kev.publish --run runs/kev2 --repo jaredpalmer/kev-0.5b --message "v0.2: none-of-the-above fix"

Uploads: the adapter (or, for a full-weight run, config.json and every model*.safetensors shard with its index), head.pt,
tokenizer files, eval.json, training log (if found), and the model card (--card) as README.md with the repo id and run
name filled in. Requires `hf auth login`.
"""
import argparse, os, re, shutil, tempfile
from pathlib import Path
from huggingface_hub import HfApi
from .checkpoint import Checkpoint
from .suite import read_json, write_json

FILES = ["head.pt", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "eval.json"]
WEIGHTS = {False: ["adapter_config.json", "adapter_model.safetensors"], True: ["config.json", "model.safetensors.index.json"]}   # by Checkpoint.full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True, help="e.g. jaredpalmer/kev-0.5b")
    ap.add_argument("--message", default=None)
    ap.add_argument("--card", required=True, help="model card markdown, e.g. docs/model-cards/kev-4b.md")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--tag", help="create this Hub tag on the uploaded commit (versioned release, e.g. v0.2)")
    ap.add_argument("--revision", help="upload to this branch instead of main (created if missing); for candidates that must not replace the released weights")
    a = ap.parse_args()

    checkpoint, run_name = Checkpoint(a.run), os.path.basename(a.run.rstrip("/"))
    base, full = checkpoint.meta.base, checkpoint.full
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)

    with tempfile.TemporaryDirectory() as tmp:
        for f in FILES + WEIGHTS[full] + [p.name for p in checkpoint.shards() if full]:
            src = f"{a.run}/{f}"
            if os.path.exists(src): shutil.copy(src, tmp)
            else: print(f"skip {f} (not found)")
        if not full:   # runs before task_type was set saved null; the Hub warns about it and PEFT treats both the same for a bare backbone
            cfg_path = f"{tmp}/adapter_config.json"; cfg = read_json(cfg_path)
            if not cfg.get("task_type"): cfg["task_type"] = "FEATURE_EXTRACTION"; write_json(cfg_path, cfg)
        if os.path.exists(f"runs/logs/train_{run_name}.log"):   # standalone runs keep their log there (see .gitignore)
            shutil.copy(f"runs/logs/train_{run_name}.log", f"{tmp}/train.log")
        # research trials: runs/<study>/<trial>/checkpoint -> ship the trial's result, provenance and training log too
        trial = os.path.dirname(a.run.rstrip("/")) if run_name == "checkpoint" else None
        if trial:
            run_name = os.path.relpath(trial, "runs")
            for f in ("result.json", "provenance.json", "train.log", "training_config.json", "training_metrics.json"):
                for src in (f"{trial}/{f}", f"{a.run}/{f}"):
                    if os.path.exists(src): shutil.copy(src, f"{tmp}/{f}"); break

        # the card's prose names the Hub repo and trial itself; only the frontmatter is filled from the checkpoint
        card = re.sub(r"^base_model: .*$", f"base_model: {base}", Path(a.card).read_text(encoding="utf-8"), flags=re.M)
        if "base_model_relation:" not in card: card = card.replace(f"base_model: {base}", f"base_model: {base}\nbase_model_relation: {'finetune' if full else 'adapter'}")
        Path(tmp, "README.md").write_text(card, encoding="utf-8")

        ev = read_json(f"{tmp}/eval.json") if os.path.exists(f"{tmp}/eval.json") else {}
        acc = ev.get("accuracy_calibration", {}).get("ALL", {})
        if os.path.exists(f"{tmp}/result.json"):
            clean = read_json(f"{tmp}/result.json").get("clean", {})   # research trials; other result files (kev-finetune runs) just skip the figures
            acc = {"acc": clean.get("acc", float("nan")), "ece": clean.get("ece", float("nan"))}
        msg = a.message or f"Upload {run_name} (base {base}; acc {acc.get('acc', float('nan')):.3f}, ECE {acc.get('ece', float('nan')):.3f})"
        if a.revision: api.create_branch(a.repo, branch=a.revision, repo_type="model", exist_ok=True)
        info = api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="model", commit_message=msg, revision=a.revision)
        print(info)
        if a.tag:
            api.create_tag(a.repo, tag=a.tag, repo_type="model", tag_message=msg, exist_ok=False, revision=a.revision)
            print(f"tagged {a.repo}@{a.tag}")


if __name__ == "__main__":
    main()
