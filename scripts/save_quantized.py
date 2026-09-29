"""Save a checkpoint's base quantized exactly as KEV_QUANT quantizes it at load (kev.model.quant_config), to serve with
KEV_BASE: a smaller download and nothing to quantize at load. The adapter and head stay in the checkpoint.

    python scripts/save_quantized.py --run jaredpalmer/kev-27b --quant int8 --out /content/Qwen3.8-27B-int8
    KEV_QUANT=int8 KEV_BASE=<that directory, or its Hub repo> python -m kev.serve --run jaredpalmer/kev-27b
"""
import argparse

import torch
from transformers import AutoModelForCausalLM

from kev.checkpoint import Checkpoint
from kev.device import default_device
from kev.model import load_tokenizer, quant_config


def save(run, quant, out):
    meta = Checkpoint(str(run)).meta
    lm = AutoModelForCausalLM.from_pretrained(meta.base, revision=meta.base_revision, dtype=torch.bfloat16, device_map={"": default_device()},
                                              quantization_config=quant_config(quant, torch.bfloat16))
    if quant == "int8": lm.to("cpu")   # torchao's safetensors export clones every tensor: on the GPU that is the model twice
    lm.save_pretrained(out)
    load_tokenizer(meta.base, revision=meta.base_revision).save_pretrained(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quant", choices=["int8", "nf4"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    save(a.run, a.quant, a.out)
    print("SAVED", a.out, flush=True)
