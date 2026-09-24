"""Quantize a Kev base model to AWQ int4 and publish it to the Hub as a new base repo. This is the first step
toward an AWQ-served Kev checkpoint (e.g. kev-9b-awq): run this once against the base, then publish a checkpoint
whose head.pt `base` field points at the produced repo (same adapter/head as the existing bf16 checkpoint; see
kev.checkpoint.Meta.quantization and kev.publish, which needs no changes to carry it).

Needs a CUDA GPU and the `awq` extra (`uv sync --extra awq`; AutoAWQ has no MPS/CPU kernels) and `hf auth login`
for the push. Not run as part of CI or any train/publish path; a maintainer (or a Modal job, see the kev-modal-study
skill) runs it by hand.

    uv run --extra awq python scripts/quantize_awq.py --base Qwen/Qwen3.5-9B-Base --base_revision 68c46c4b \
        --calib evals/v7/decision-v7 --repo jaredpalmer/qwen3.5-9b-base-awq

After this, refit the checkpoint's temperature against the AWQ base (scripts/calibrate_checkpoint.py) before
publishing it — quantization noise can shift calibration even though it rarely changes the argmax.
"""
import argparse, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.suite import load_split  # noqa: E402


def calibration_texts(suite, split, n, seed):
    import random
    records = load_split(suite, split)
    rng = random.Random(seed)
    sample = rng.sample(records, min(n, len(records)))
    return [r["state"] for r in sample]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--base_revision", help="pin the base to this revision (recorded in the quantized repo's config)")
    ap.add_argument("--calib", required=True, help="a frozen suite directory (evals/<version>/<name>) to draw calibration text from")
    ap.add_argument("--calib_split", default="train")
    ap.add_argument("--calib_n", type=int, default=512, help="AutoAWQ's own default; more rarely helps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--repo", required=True, help="e.g. jaredpalmer/qwen3.5-9b-base-awq")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi

    texts = calibration_texts(a.calib, a.calib_split, a.calib_n, a.seed)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision)
    model = AutoAWQForCausalLM.from_pretrained(a.base, revision=a.base_revision, low_cpu_mem_usage=True, use_cache=False)
    model.quantize(tok, quant_config={"zero_point": True, "q_group_size": a.group_size, "w_bit": a.bits, "version": "GEMM"}, calib_data=texts)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_quantized(tmp)
        tok.save_pretrained(tmp)
        api = HfApi()
        api.create_repo(a.repo, repo_type="model", exist_ok=True, private=a.private)
        info = api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="model",
                                  commit_message=f"AWQ {a.bits}-bit (group_size={a.group_size}) of {a.base}"
                                                 f"{f'@{a.base_revision}' if a.base_revision else ''}, calibrated on {len(texts)} records from {a.calib}/{a.calib_split}")
        print(info)


if __name__ == "__main__":
    main()
