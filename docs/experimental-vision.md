# Experimental image decisions

This experiment takes a local image and a Kev request, then returns Choice, Noul, and Score answers. It is an initial implementation for [issue #28](https://github.com/jaredpalmer/kev/issues/28).

```bash
uv run --extra vision python -m kev.vision \
  --image /path/to/product.png \
  --request examples/vision-request.json \
  --device cuda
```

The first run downloads `Qwen/Qwen2.5-VL-3B-Instruct`. It loads in fp32, which requires four bytes per parameter plus memory for image processing and inference. `--device` accepts `cpu`, `cuda`, or `mps`; omitting it uses Kev's device selection. Device performance and full-checkpoint accuracy have not been measured for this experiment.

`--base` accepts a Qwen2.5-VL checkpoint. `--revision` selects the same Hub revision for the model and processor; pin a commit for reproducible experiments. Released Kev text checkpoints cannot be loaded by this command. The existing HTTP server and playground continue to accept text requests only.

The request file uses `SystemOneRequest`. The example asks whether a product looks damaged and selects a review queue. One image is shared by all questions. The output uses Kev's existing answer mapping, includes the actual vision base in `model`, and identifies the experiment:

```json
"experimental": {
  "readout": "next-token-letter-logits",
  "calibrated": false,
  "revision": null
}
```

## How it scores images

The model's native processor converts the image to pixel tensors and image-grid metadata. Each question gets an independent causal forward pass with the shared image and text state. Options are labeled A through Z. The command selects the next-token logits for those letters and normalizes them over the supplied options. It requests logits only at the last input position and never calls `generate`. The underlying interfaces are documented in [Transformers' Qwen2.5-VL reference](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl).

This is a zero-shot vision-language baseline. It does not load Kev's trained pointer head or use its calibration temperature. Its normalized scores and confidence fields are not measured probabilities of correctness. Option order and letter preferences can influence answers. A trained Kev vision checkpoint would require labeled image decisions, a training path, and held-out calibration and evaluation.

## Limits

- One local still image, at most 20 MiB and 16 million source pixels. The loader applies EXIF orientation and converts to RGB. URLs and animated images are not supported.
- The processor resizes images to at most `1024 * 28 * 28` pixels. Small details may be lost.
- At most 26 options per question. Noul always uses two; Score uses its ordered level descriptions.
- Each image-plus-question input must fit within 8,192 tokens and the base model's context window. Oversized inputs raise an error without truncation or model inference for that question.
- Questions run sequentially, repeating image processing and vision computation. This command does not use Kev's text prefix cache. Token usage sums all question passes, including image placeholder tokens; output tokens measure serialized answers, not generated text.

## Verification

```bash
uv run --extra vision python -m pytest tests/test_vision.py -q
```

Tests use synthetic images and random tiny model weights. They check typed answer mapping, limits, image validation, and that Qwen2.5-VL consumes image pixels. They do not establish useful prediction quality. Before proposing a released model, compare predictions on a labeled image set, test option permutations and image substitutions, fit calibration on a separate partition, and measure memory and latency with the intended checkpoint and device.
