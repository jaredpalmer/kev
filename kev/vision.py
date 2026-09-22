"""Experimental Qwen2.5-VL image decisions using next-token letter logits, without generation.

This is a vision-language baseline, not a trained or calibrated Kev pointer checkpoint.
Run `python -m kev.vision --help`; see docs/experimental-vision.md for limits and examples.
"""
import argparse
import json
import string
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .api import SystemOneRequest, output_tokens, to_answers, to_record
from .device import default_device, sync
from .model import user_tokens

DEFAULT_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_VISION_PIXELS = 1024 * 28 * 28
MAX_INPUT_TOKENS = 8192
LETTERS = string.ascii_uppercase


def load_image(path):
    """Decode one local still image, rejecting oversized files before pixel allocation."""
    with Path(path).open("rb") as source:
        source.seek(0, 2)
        if source.tell() > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
        source.seek(0)
        with Image.open(source) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("only still images are supported")
            return ImageOps.exif_transpose(image).convert("RGB")


class VisionPredictor:
    """One image and shared text state, with an independent causal pass for each question.

    Construct with a processor and Qwen2.5-VL model, or use from_pretrained().
    No text prefix cache is used: equal image placeholders do not imply equal pixels.
    """

    def __init__(self, processor, model, base=DEFAULT_BASE, revision=None):
        self.processor, self.model = processor, model.eval()
        self.base, self.revision = base, revision
        self.device = model.device
        self.letter_ids = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in self.letter_ids):
            raise ValueError("vision baseline requires single-token option letters A-Z")
        self.letter_ids = [ids[0] for ids in self.letter_ids]
        if len(set(self.letter_ids)) != len(LETTERS) or set(self.letter_ids) & set(processor.tokenizer.all_special_ids):
            raise ValueError("option letters must be distinct, non-special tokens")

    @classmethod
    def from_pretrained(cls, base=DEFAULT_BASE, revision=None, device=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = device or default_device()
        processor = AutoProcessor.from_pretrained(base, revision=revision, min_pixels=4 * 28 * 28,
                                                 max_pixels=MAX_VISION_PIXELS)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, revision=revision, dtype=torch.float32, attn_implementation="sdpa").to(device)
        return cls(processor, model, base, revision)

    def inputs(self, state, question, image):
        # Reuse the text path's delimiter escaping, including vision and chat control tokens.
        tok = self.processor.tokenizer
        body = f"State:\n{state}\n\nQuestion:\n{question['instr']}\n\nOptions:\n"
        body += "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, question["options"]))
        body = tok.decode(user_tokens(tok, body), skip_special_tokens=False)
        messages = [
            {"role": "system", "content": "Choose the best option using the image and state. Reply with only its uppercase letter."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": body}]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", truncation=False)
        limit = min(MAX_INPUT_TOKENS, self.model.config.text_config.max_position_embeddings)
        if inputs["input_ids"].shape[1] > limit:
            raise ValueError(f"image and question exceed {limit} input tokens")
        return inputs.to(self.device)

    @torch.inference_mode()
    def answer(self, request: SystemOneRequest, image_path):
        rec, meta = to_record(request)
        if any(len(q["options"]) > len(LETTERS) for q in rec["questions"]):
            raise ValueError("vision baseline supports at most 26 options per question")
        image = load_image(image_path)
        sync(self.device.type)
        start = time.perf_counter()
        probabilities, input_count = [], 0
        for question in rec["questions"]:
            inputs = self.inputs(rec["state"], question, image)
            result = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = result.logits[0, -1, self.letter_ids[:len(question["options"])]]
            if not torch.isfinite(logits).all():
                raise ValueError("vision model returned non-finite option logits")
            probabilities.append(torch.softmax(logits.float(), -1).cpu().tolist())
            input_count += inputs["input_ids"].shape[1]
        sync(self.device.type)
        answers = to_answers(probabilities, meta)
        return {"model": self.base, "answers": answers,
                "usage": {"input_tokens": input_count, "output_tokens": output_tokens(self.processor.tokenizer, answers)},
                "latency_ms": round(1000 * (time.perf_counter() - start), 1),
                "experimental": {"readout": "next-token-letter-logits", "calibrated": False, "revision": self.revision}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="local still image, shared by all questions")
    parser.add_argument("--request", required=True, help="UTF-8 JSON SystemOneRequest")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Qwen2.5-VL checkpoint, not a Kev text checkpoint")
    parser.add_argument("--revision", help="Hub revision for both model and processor")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    request = SystemOneRequest.model_validate_json(Path(args.request).read_text(encoding="utf-8"))
    predictor = VisionPredictor.from_pretrained(args.base, args.revision, args.device)
    print(json.dumps(predictor.answer(request, args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
