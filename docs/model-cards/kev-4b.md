---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - qwen3.5
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Kev-4B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,204 records; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.872 }
          - { type: expected_calibration_error, value: 0.075, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.797 }
          - { type: brier_score, value: 0.299 }
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.837 }
          - { type: brier_score, value: 0.255 }
---

# Kev-4B

Kev-4B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 33.8M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-4B-Base` (revision `1001bb4d`), serving TypeSafe's public `/v1/systemone` contract.

**The recommended Kev.** The best accuracy per byte: out of domain 0.797 on the development partition and **0.837 on the locked test**, Brier 0.255 on the test, held-out rule pairs 0.77–0.78. This checkpoint is the `decision-v7` recipe (trial `q35-4b-s23/00-trial-0`, seed 2, selected on development accuracy) followed by a 9-minute **delta fine-tune** (`--init_from`, lr 2e-5, one epoch) on 1,425 additional records — date-bearing policy cases rendered with explicit day counts, and evidence-free cases with uniform targets — mixed with 2,000 replayed training records. Against the pre-delta checkpoint on the locked test: +1.0 pp [−0.1, +2.1], Brier 0.266 → 0.255, `deadline` 0.65 → 0.75.

- Hub: `jaredpalmer/kev-4b` (this repo; trial `night2-4b-du/00-trial-0`). The pre-delta checkpoint is at revision `v7-base`; the Qwen3 generation at `qwen3` ([its card](kev-4b-qwen3.md)).
- Demo: [huggingface.co/spaces/jaredpalmer/kev](https://huggingface.co/spaces/jaredpalmer/kev) runs Kev-4B and Kev-0.8B on ZeroGPU with the same encoder and API code as `kev.serve`.
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN.md` (the Qwen3.5 port and this experiment are under History), `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-4B (Qwen3) | Kev-4B before the delta (`v7-base`) | **Kev-4B, raw logits** | **Kev-4B as served (T = 2.14)** | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.854 | 0.877 | 0.872 | 0.872 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.790 | 0.794 | **0.797** | 0.797 | 0.857 |
| out-of-domain Brier | 0.328 | 0.316 | 0.299 | **0.264** | 0.211 |
| out-of-domain ECE | 0.102 | 0.130 | 0.122 | **0.040** | 0.049 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 8.2% | 8.2% | 6.9% | **2.6%** | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | 0.31 | 0.54 | 0.54 | 0.57 | 0.70 |
| held-out policy structures, both siblings correct | 0.73 | 0.78 | 0.78 | 0.78 | 0.86 |
| unknowable items answered at ≥ 0.9 (lower is better; transfer-v9) | 0.44 | 0.19 | **0.00** | 0.00 | 0.09 |
| **locked test**, out-of-domain accuracy / Brier | 0.806 / 0.294 | 0.832 / 0.266 | **0.837 / 0.255** | – | – |
| **locked test**, in-distribution accuracy | 0.856 | 0.870 | 0.871 | – | – |

Per-source out-of-domain accuracy (Kev-4B / Jev): QNLI 0.91 / 0.93, SciQ 0.97 / 0.99, TweetEval-offensive 0.74 / 0.81, PAWS 0.74 / 0.79, MMLU 0.70 / 0.90, Emotion 0.56 / 0.59, deadline (3-level date arithmetic) 0.60 / 0.93 — **0.85 with the `date_facts` preprocessor** (below), (A or B) and C 0.91 / 0.91, (A and B) or not C 0.88 / 0.97, if A then not B else C 1.00 / 0.78.

**Calibration is built in.** `head.pt` carries a temperature (T = 2.14) fitted on this checkpoint's in-distribution development rows by minimising negative log-likelihood ([`scripts/calibrate_checkpoint.py`](https://github.com/jaredpalmer/kev/blob/main/scripts/calibrate_checkpoint.py)); the pointer head divides its logits by it at inference. Every loader — `kev.serve`, `kev.benchmark`, the Space, anyone's harness — gets the calibrated probabilities by default. It never changes an answer: the argmax is identical, so accuracy is the same in both columns; confidences are re-ordered only slightly across questions with different option counts, which is why coverage moves by a point or two. `KEV_TEMPERATURE=1.0` restores the raw logits; the raw column is what the training produced. Per-(type, option-count) temperatures were tested and are worse out of domain. The fit uses no out-of-domain or test data.

**`date_facts` preprocessor.** Kev, like every Kev before it, cannot subtract dates reliably (the untrained base can; LoRA training erodes it). It can use a stated day count. `KEV_DATE_FACTS=1` appends one sentence per pair of absolute dates found in the state ("June 26, 2026 is 8 days before July 4, 2026"); this checkpoint was trained on such renderings, so with it `deadline` goes from 0.60 to 0.85 and overall out-of-domain accuracy from 0.797 to 0.820. It is preprocessing, reported separately, never folded into the model's own numbers.

**What the delta cost.** MMLU-Pro fell 0.500 → 0.490 and scienthoon's ECE rose 0.086 → 0.116; coverage at ≤ 5% error was unchanged (0.54 development, 0.67 → 0.68 locked test) and confident errors fell (8.2% → 6.9%). The pre-registered criteria for the delta (`PLAN.md`, "Tonight's autoresearch") were met for dates and for the unknowable-confidence behaviour; the coverage criterion asked for +5 pp and got 0; the locked read decided promotion.

**Newer evaluation columns** (`transfer-v9` development, Kev-4B / Jev): MMLU-Pro (10-way) 0.490 / 0.840; state buried among unrelated records 0.67 / 0.70; unknowable share at ≥ 0.9 confidence 0.00 / 0.09 (intact controls 0.94).

**External suites** (same items as their published Jev numbers): SemIf's authored 144 — 0.896 before the delta (live Jev 0.965; SemIf's untrained Qwen3.5-4B 0.813); scienthoon's 900 tickets — queue 0.918, angry 0.790, ECE 0.116 (Jev 0.897, 0.914, 0.105). On ekzhang's 1,000-question MMLU-Pro sample the shipped checkpoint scores 0.468 over all 1,000 questions (8 exceed the state limit and count as wrong; live Jev 0.835 on the same items, ekzhang reports 0.829).

## How it was built

- **Base model**: Qwen3.5-4B-Base, a hybrid of 24 Gated DeltaNet (linear attention) layers and 8 full-attention layers. Because the recurrent layers cannot honour a block-causal mask, questions run as separate causal rows that continue from the shared state (`kev/model.py: forward_rows_batch`); isolation is exact by construction (together vs alone within 1e-5) and on attention-only models this form is bit-identical to the packed one.
- **Recipe**: `decision-v7`, two epochs, LoRA r=16 (attention, MLP and DeltaNet projections), lr 5e-5 — the same data and settings as every other Kev, so the Qwen3 → Qwen3.5 difference is the base (`PLAN.md`, Qwen3.5 port §10: locked test +7.3 pp [+2.8, +11.7] over Kev-8B).
- **Delta**: `kev.train --init_from jaredpalmer/kev-4b@v7-base --data evals/night2/dates_unknowable.jsonl --replay 2000 --lr 2e-5 --epochs 1`. The 1,425 new records are generated (no public dataset): 900 date-bearing policy cases, a third rendered plainly, a third with a relational day-count sentence, a third with a `date_facts` field; 255 cases with the deciding sentence removed and a uniform soft target over the options, plus their 270 intact controls. Record hashes are in `evals/night2/manifest.json`; the source checkpoint's hashes are in `training_config.json`.
- Why a delta and not a retrain: it is a controlled change (one fixed checkpoint, one data addition, 9 minutes), and the results section shows exactly what it moved.

## Known limits

- Use [Kev-9B](kev-9b.md) when accuracy and calibration matter more than memory: 0.852 vs 0.837 out of domain on the locked test, Brier 0.237 vs 0.255.

- **Slow on a Mac.** The DeltaNet kernels have no MPS implementation; PyTorch falls back to reference code. A five-question request that takes 0.17 s on the Qwen3 Kev-4B takes 0.78 s here in bf16 on an M5. On CUDA with `flash-linear-attention` installed it is fast. Use `jaredpalmer/kev-4b@qwen3` for low latency on Apple Silicon until an MLX path exists.
- Requires `transformers >= 5.17` (the `qwen3_5` architecture) and `peft >= 0.21`.
- Knowledge (MMLU 0.70 vs Jev 0.90; MMLU-Pro 0.490 vs 0.840), TweetEval (0.74 vs 0.81) and noisy-label Emotion (0.56 vs 0.59) are the remaining gap; knowledge is set by the base (the untrained Qwen3.5-4B scores the same).
- Date arithmetic without the preprocessor: `deadline` 0.60 (Jev 0.93). With `KEV_DATE_FACTS=1`: 0.85.
- The raw logits are over-confident out of domain; the built-in temperature (T = 2.14) fixes most of it without changing any answer. `KEV_TEMPERATURE=1.0` gives the raw values. Coverage at a 5% error budget is 0.54–0.68 against Jev's 0.70.
- 4B bf16 needs ~9 GB of GPU memory for serving; training took 56 min on one H100 (peak 24.6 GB).

## Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on `q/k/v/o_proj`, `gate/up/down_proj`, `in_proj_qkv/z/a/b`, `out_proj`; pointer head from scratch; cross-entropy on the option distribution; lr 5e-5 (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. Then the delta described above (one epoch, lr 2e-5, 3,937 records seen, 9 minutes on one H100). No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate (`runs/locked/kev-4b-night2-du-ungated/`; the pre-delta read is `runs/locked/kev-4b-q35/`). Every number carries suite hash, code hashes and git commit in `result.json`. Untrained-base baselines use zero-shot letter logits on the same items (`scripts/base_mmlu_probe.py`).

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008      # KEV_DTYPE=bf16 on a Mac; slow on MPS, see limits
KEV_DATE_FACTS=1 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008   # + date preprocessing; KEV_TEMPERATURE=1.0 for raw logits
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
