"""Image baseline contracts and a tiny native vision forward pass; no downloaded weights."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BatchFeature, PreTrainedTokenizerFast

from kev.api import SystemOneRequest
from kev.vision import LETTERS, MAX_INPUT_TOKENS, VisionPredictor, load_image


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(path)
    return path


class Processor:
    def __init__(self):
        vocab = {letter: i for i, letter in enumerate(LETTERS)}
        vocab.update({"[UNK]": 26, "<|vision_start|>": 27, "<|image_pad|>": 28, "<|vision_end|>": 29})
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
                                                unk_token="[UNK]")
        self.messages = []
        self.length = 4

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "image question"

    def __call__(self, text, images, **kwargs):
        assert images[0].mode == "RGB"
        assert kwargs == {"return_tensors": "pt", "truncation": False}
        pixels = torch.tensor(images[0].getpixel((0, 0)), dtype=torch.float32).mean() / 255
        return BatchFeature({"input_ids": torch.tensor([[27, 28, 29] + [0] * (self.length - 3)]),
                             "attention_mask": torch.ones(1, self.length, dtype=torch.long),
                             "pixel_values": pixels.expand(4, 24).clone(),
                             "image_grid_thw": torch.tensor([[1, 2, 2]])})


class Model:
    device = torch.device("cpu")
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=MAX_INPUT_TOKENS))

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, **kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs["logits_to_keep"] == 1
        assert not torch.is_grad_enabled()
        self.calls.append(kwargs)
        return SimpleNamespace(logits=torch.arange(32, dtype=torch.float32).reshape(1, 1, 32))


def request(questions=None):
    return SystemOneRequest.model_validate({"state": "Inspect this parcel", "questions": questions or {
        "damage": {"type": "noul", "instructions": "Is it damaged?"},
        "route": {"type": "choice", "instructions": "Where should it go?", "criteria": {"accept": None, "review": None}},
        "severity": {"type": "score", "instructions": "Damage severity", "criteria": ["none", "minor", "major"]}}})


def test_image_answers_use_native_pixels_and_existing_response_mapping(image_path):
    processor, model = Processor(), Model()
    result = VisionPredictor(processor, model).answer(request(), image_path)
    assert len(model.calls) == 3
    assert all("pixel_values" in call and "image_grid_thw" in call for call in model.calls)
    assert result["answers"]["damage"]["noul"] == 0.73
    assert result["answers"]["route"]["choice"] == "review"
    assert result["answers"]["severity"]["score"] == 1.58
    assert result["usage"]["input_tokens"] == 12
    assert result["experimental"]["calibrated"] is False
    assert result["model"] != "kev-latest"
    assert len(processor.messages) == 3


@pytest.mark.parametrize("length, accepted", [(MAX_INPUT_TOKENS, True), (MAX_INPUT_TOKENS + 1, False)])
def test_context_limit_before_forward(image_path, length, accepted):
    processor, model = Processor(), Model()
    processor.length = length
    predictor = VisionPredictor(processor, model)
    if accepted:
        predictor.answer(request(), image_path)
        assert len(model.calls) == 3
    else:
        with pytest.raises(ValueError, match="input tokens"):
            predictor.answer(request(), image_path)
        assert model.calls == []


def test_too_many_options_rejected_before_image_io():
    predictor = VisionPredictor(Processor(), Model())
    req = request({"q": {"type": "choice", "instructions": "Pick", "criteria": {str(i): None for i in range(27)}}})
    with pytest.raises(ValueError, match="26 options"):
        predictor.answer(req, "does-not-exist.png")


def test_image_validation(image_path, monkeypatch):
    from kev import vision

    assert load_image(image_path).mode == "RGB"
    monkeypatch.setattr(vision, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="pixels"):
        load_image(image_path)
    monkeypatch.setattr(vision, "MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="bytes"):
        load_image(image_path)


def test_animated_image_rejected(tmp_path):
    path = tmp_path / "animated.gif"
    Image.new("RGB", (4, 4), "red").save(path, save_all=True, append_images=[Image.new("RGB", (4, 4), "blue")])
    with pytest.raises(ValueError, match="still images"):
        load_image(path)


def test_invalid_letter_tokenizer_rejected():
    processor = Processor()
    processor.tokenizer.encode = lambda *args, **kwargs: [1, 2]
    with pytest.raises(ValueError, match="single-token"):
        VisionPredictor(processor, Model())


def test_questions_have_independent_prompts(image_path, monkeypatch):
    from kev import vision

    bodies = []
    original = vision.user_tokens

    def capture(tok, text):
        bodies.append(text)
        return original(tok, text)

    monkeypatch.setattr(vision, "user_tokens", capture)
    VisionPredictor(Processor(), Model()).answer(request(), image_path)
    assert ["Is it damaged?" in text for text in bodies] == [True, False, False]
    assert ["Where should it go?" in text for text in bodies] == [False, True, False]
    assert all("Inspect this parcel" in text for text in bodies)


def test_nonfinite_logits_rejected(image_path):
    class NonfiniteModel(Model):
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.full((1, 1, 32), float("nan")))

    with pytest.raises(ValueError, match="non-finite"):
        VisionPredictor(Processor(), NonfiniteModel()).answer(request(), image_path)


def test_tiny_qwen_vision_forward_uses_pixels(image_path):
    from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                              Qwen2_5_VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor)

    torch.manual_seed(7)
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "max_position_embeddings": 128, "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                       "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "out_hidden_size": 32, "window_size": 4, "fullatt_block_indexes": [0]},
        image_token_id=28, video_token_id=30, vision_start_token_id=27, vision_end_token_id=29,
    )
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    tokenizer = Processor().tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]})
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                             min_pixels=16, max_pixels=16),
        tokenizer=tokenizer, video_processor=Qwen2VLVideoProcessor(),
        chat_template="{{ '<|vision_start|><|image_pad|><|vision_end|>A' }}",
    )
    predictor = VisionPredictor(processor, model)
    rec = {"instr": "Pick a color", "options": ["red", "blue"]}
    with torch.inference_mode():
        inputs = predictor.inputs("", rec, load_image(image_path))
        first = model(**inputs, use_cache=False, logits_to_keep=1).logits
        inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        second = model(**inputs, use_cache=False, logits_to_keep=1).logits
    assert first.shape == (1, 1, 32)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, second), "The native model must actually consume image pixels"
