---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-0.5B
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - prototype
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
metrics:
  - accuracy
  - expected_calibration_error
  - nll
model-index:
  - name: Kev-0.5B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "held-out split of the six training sources (1,350 questions)" }
        metrics:
          - { type: accuracy, value: 0.799 }
          - { type: expected_calibration_error, value: 0.065, name: "ECE (10 bins)" }
          - { type: expected_calibration_error, value: 0.031, name: "ECE after temperature scaling (T=1.47)" }
---

# Kev-0.5B — prototype (superseded)

Kev-0.5B is a **decision model**. It takes one document (the *state*) and a set of typed questions, and returns a probability distribution for each question in one forward pass. It does not generate text.

It is a LoRA adapter plus a small pointer head on top of `Qwen/Qwen2.5-0.5B`. It reproduces the architecture that Archer Hume inferred for TypeSafe's Jev in [*Jev's Architecture Unmasked*](https://archerhume.com/posts/jevs-architecture-unmasked), and it serves TypeSafe's public `/v1/systemone` API contract.

This checkpoint is the **original prototype**, trained on a laptop in September 2026 to show the mechanism works. It is superseded by [Kev-0.8B](kev-0.8b.md), [Kev-4B](kev-4b.md) and [Kev-9B](kev-9b.md), which use Qwen3.5 bases, frozen checksummed suites, and a recipe found through ~110 controlled trials; on the same out-of-domain items (transfer-v4 dev) this model scores 0.561 against 0.643 / 0.794 / 0.812.620 / 0.790 / 0.796. It stays on the Hub for reference and reproducibility; use the current family for anything else.

- Hub: [jaredpalmer/kev-0.5b](https://huggingface.co/jaredpalmer/kev-0.5b) (tag `v0.1`)
- Code, training recipe, evaluation and demo: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
- Weights: [GitHub release `v0.1.0`](https://github.com/jaredpalmer/kev/releases/tag/v0.1.0), `kev-0.5b.tar.gz` (38 MB; LoRA adapter `adapter_model.safetensors`, head `head.pt`, tokenizer files, `eval.json`, training log). SHA-256 `15639f79…6e12f8`, full digest in the sidecar `.sha256`. Extract to `runs/kev/`. Weights are not committed to git.

## Model details

| | |
|---|---|
| Developed by | Jared Palmer, with Devin (Cognition) |
| Model type | Causal transformer, prefill-only, block-causal branch mask, pointer readout |
| Base model | `Qwen/Qwen2.5-0.5B` (494M parameters, frozen) |
| Adapter | LoRA rank 16, alpha 32, dropout 0.05, on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (all 24 layers) |
| Head | Two linear maps `896 → 256` (query from `<decide>`, key from each `</opt>`), scaled dot product, softmax over options |
| Trainable parameters | 9.3M (LoRA 8.8M + head 0.46M), 1.9% of the backbone |
| Precision | fp32 (training and serving on Apple MPS) |
| Context used in training | ≤ 384 state tokens, ≤ 1,024 tokens per question branch |
| Context allowed at serving | 8,192 per branch (backbone supports 32k) |
| Question types | `noul` (yes/no), `choice` (2–255 options), `score` (2–255 ordered levels) |
| Language | English |
| License | Apache-2.0 for the adapter and head. The base model is under the Qwen license (Apache-2.0 for Qwen2.5-0.5B). Datasets carry their own licenses. |
| Version | Kev-0.5B v0.1, trained 2026-09-17 |

## Intended use

**Intended.** Research on decision models: calibration of direct probability readouts, shared-state / isolated-question attention, option-order sensitivity, and API-level compatibility with TypeSafe's System One contract. Local demos and teaching.

**Not intended.** Any production decision that affects people: moderation, fraud, credit, hiring, medical or legal routing. The model's knowledge is limited to a 0.5B backbone, its calibration is only verified on the training distributions, and its outputs on unfamiliar tasks have not been measured.

## How the model is used

Input is one packed token sequence:

```
<state> …state…  <q> instr <opt> o1 </opt> <opt> o2 </opt> … <decide>  <q> … <decide>  …
```

- The attention mask lets a question token see the state and its own branch only. Questions cannot see each other.
- Each branch restarts position ids after the state.
- For each question, the head scores every `</opt>` hidden state against the `<decide>` hidden state and applies softmax.
- Application code turns the distributions into the API answer: `choice`/`confidence` for Choice, `p(yes)` for Noul, expected level for Score.

Reserved tokens are existing Qwen special tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`). User text is sanitized so it cannot produce them.

Serve with `python -m kev.serve --run runs/kev` and call `POST /v1/systemone`, or use `typesafe-sdk` with `base_url="http://127.0.0.1:8009"`.

## Training data

Six public datasets, converted to TypeSafe-shaped requests and rendered with the same code path used at serving time (`api.to_record()`). 1,500 records were sampled per source from the standard **train** splits, giving 9,000 records and 13,500 questions (4,500 Choice, 6,000 Noul, 3,000 Score).

| source | split | converted to | notes |
|---|---|---|---|
| Banking77 | train | Choice, K = 77 | intent names as option keys; templated descriptions, 50% `null` |
| BoolQ | train | Noul | passage as state; 40% with `true`/`false` criteria |
| AG News | train | Choice K = 4 + 2 Noul | derived yes/no questions packed with the topic question |
| MNLI | train | Choice K = 3 | premise as state, hypothesis in instructions |
| SST-5 | train | Score, 5 levels | |
| Yelp Review Full | train | Score 5 levels + Noul | text truncated to 220 words; `recommend` = stars ≥ 4 |

Rendering variation applied at conversion time: ~30% `null` option descriptions, ~10% structured `{"what": …}` descriptions, ~15% structured `{"question", "focus"}` instructions, ~32% states wrapped as objects or arrays (`{"document"}`, `{"ticket": {"channel","body"}}`, `[{"role","content"}]`).

Augmentation applied once per record before encoding: option order shuffled; with probability 0.10 the true option replaced by `other: None of the above`; with probability 0.15 an irrelevant distractor option added.

No LLM-generated data. No human annotation beyond the original datasets.

## Training procedure

| | |
|---|---|
| Objective | Cross-entropy over options, averaged over questions in a record |
| Optimizer | AdamW, lr 2e-4, weight decay 0.01, OneCycle schedule (10% warm-up) |
| Batch | 1 record per step, gradient accumulation 8, gradient clipping 1.0 |
| Epochs | 2 (2,250 optimizer steps) |
| Hardware | Apple M5, 32 GB unified memory, PyTorch 2.8 MPS backend |
| Wall clock | ~1h45m (~0.29 s per record) |
| Seed | 0 |
| Final training loss | 0.27 |

This checkpoint predates two loss terms that are now defaults in `kev/train.py`: the ordinal term for Score (`--ord_w`) and the permutation-consistency KL for Choice (`--perm_kl`). To reproduce this checkpoint exactly:

```bash
uv run python -m kev.train --n_per_source 1500 --epochs 2 --accum 8 --perm_kl 0 --ord_w 0 --out runs/kev
```

Note that augmentation is now re-applied every epoch rather than fixed at encode time, so a re-run will not be bit-identical.

## Evaluation

Held-out **test / validation** splits of the same six sources, 150 records per source, 1,350 questions, seed 1. Full results in `runs/kev/eval.json`.

### Accuracy and calibration

| source | K | zero-shot base | zero-shot Instruct | **Kev-0.5B** |
|---|---|---|---|---|
| | | acc / ECE | acc / ECE | acc / ECE / NLL |
| banking77 | 77 | – | – | 0.860 / 0.057 / 0.56 |
| agnews | 4 | 0.813 / 0.069 | 0.787 / 0.160 | 0.940 / 0.028 / 0.22 |
| agnews yes/no | 2 | 0.780 / 0.103 | 0.853 / 0.062 | 0.960 / 0.017 / 0.10 |
| boolq | 2 | 0.427 / 0.274 | 0.607 / 0.084 | 0.753 / 0.136 / 0.63 |
| mnli | 3 | 0.460 / 0.225 | 0.433 / 0.390 | 0.747 / 0.100 / 0.63 |
| sst5 | 5 | 0.373 / 0.083 | 0.447 / 0.344 | 0.533 / 0.121 / 1.17 (MAE 0.59 levels) |
| yelp | 5 | 0.313 / 0.043 | 0.353 / 0.078 | 0.553 / 0.118 / 0.95 (MAE 0.54 levels) |
| yelp yes/no | 2 | 0.833 / 0.129 | 0.833 / 0.066 | 0.887 / 0.084 / 0.33 |
| **all** | | | | **0.799 / 0.065** |

Baselines: `Qwen/Qwen2.5-0.5B` (raw) and `Qwen/Qwen2.5-0.5B-Instruct` (chat template), same rendered text, next-token logits over option letters A–H; not run for K = 77. ECE uses 10 equal-width bins on the top probability.

### Temperature scaling

Fit on even-indexed records, tested on odd-indexed: `T = 1.47`. Held-out NLL 0.505 → 0.481, ECE 0.057 → **0.031**. The model is mildly over-confident before scaling.

### Mechanism tests

| test | result |
|---|---|
| Isolation (secret in sibling question / absent / in state) | p = 0.03 / 0.03 / **0.99** |
| Packed vs separate, max abs probability difference | 3.7e-6 (2.0× faster packed, ~2.7 questions per request) |
| Permutation, 4 orders, Choice K ≥ 3 | argmax flips 7.4%; mean spread of p(correct) 0.065, p90 0.25 |
| IIA, append one irrelevant option | mean \|Δ log-odds\| top-2 = 0.13, p90 0.34 |
| Boundary forgery, option text with fake delimiters | option count unchanged; forged option p ≤ 0.09 |

## Limitations

- **In-distribution only.** All numbers above are on held-out splits of the training datasets. Out-of-source generalization has not been measured for this checkpoint.
- **Small backbone.** 0.5B parameters. On the TypeSafe docs' structured-criteria example the model picks `return_policy` where Jev picks `return_status`. Reading comprehension (BoolQ 0.75, MNLI 0.75) is far below state of the art.
- **Narrow task coverage.** Six datasets and about ten instruction templates. Code, tables, multi-turn chat, arithmetic, and multi-step conditions are untrained.
- **Order sensitivity remains.** 7% argmax flips and a p90 probability spread of 0.25 under option reordering. A threshold near a decision boundary can change the action.
- **Score confidence** is computed by the serving code, not the checkpoint. It was `1 − E|level − mode| / (L − 1)` when this card was written; it is now `max(0, 1 − E|level − mode| / D)`, D the mean absolute deviation of a uniform distribution over the levels, as in TypeSafe's reference adapter (`system-one-adapter` 0.2.1).
- **Calibration is not a guarantee.** ECE 0.03 after temperature scaling on these sources says nothing about calibration on a new workflow. Proper scoring rules give the right *incentive*; they do not remove the need for outcome data.
- **Inherited limitations** from Qwen2.5-0.5B and from the datasets, including their label noise, demographic skews (e.g. Yelp, banking intents), and English-only coverage.

## Bias, risks and recommendations

The training sets carry the biases of their sources: US-centric news categories, English banking terminology, restaurant reviews, and crowd-sourced NLI labels. The model will mirror them.

Direct probability outputs look authoritative. A `confidence: 0.92` from this model is a statistic about its own distribution over three options, not a verified probability of being right. Do not threshold on it for consequential decisions without measuring calibration on your own labelled outcomes first.

The question-isolation property is a real safety feature (one question's text cannot manipulate another's answer) and was verified. The delimiter-forgery protection was verified for the five reserved tokens. Other prompt-injection routes through the state text have not been studied.

## Environmental impact

One training run: ~1.75 h on a single Apple M5 laptop SoC at roughly 30–40 W, i.e. about 0.06 kWh. Evaluation and smoke runs add a similar amount. This is small.

## Citation

```bibtex
@software{kev2026,
  title  = {kev: a laptop-scale reconstruction of a Jev-style decision model},
  author = {Palmer, Jared},
  year   = {2026},
  url    = {https://github.com/jaredpalmer/kev}
}

@misc{hume2026jev,
  title  = {Jev's Architecture Unmasked},
  author = {Hume, Archer},
  year   = {2026},
  url    = {https://archerhume.com/posts/jevs-architecture-unmasked}
}
```

## Contact

Open an issue at [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev/issues).
