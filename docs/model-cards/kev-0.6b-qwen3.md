---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3-0.6B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - decision-model
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
  - name: Kev-0.6B (Qwen3)
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v4 development (1,204 records; ten trained public sources + programmatic policy pairs)" }
        metrics:
          - { type: accuracy, value: 0.801 }
          - { type: expected_calibration_error, value: 0.086, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.620 }
          - { type: brier_score, value: 0.536 }
---

# Kev-0.6B (Qwen3)

> **Previous generation (Qwen3).** Kept as the fast small option on Apple Silicon (0.12 s per five-question request vs 0.33 s for Kev-0.8B). For accuracy use [Kev-0.8B](kev-0.8b.md): on the locked test it scores 0.668 vs 0.642 out of domain against this model on the same items. Weights: `jaredpalmer/kev-0.6b`.

Kev-0.6B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16) plus a pointer head on `Qwen/Qwen3-0.6B-Base`, and it serves TypeSafe's public `/v1/systemone` contract.

**The small member of the Kev family.** It is the best 0.6B checkpoint under a frozen, checksummed evaluation protocol: the 4B/8B recipe's data (`decision-v7`) at lr 1e-4, three seeds (transfer 0.613 / 0.605 / **0.620**), after eight one-knob mutations and three seeds of the previous data found nothing better than 0.61. Out of domain it is a 0.6B model — use Kev-4B for accuracy; use this one where memory or latency rule the 4B out, and measure on your own data.

- Hub: `jaredpalmer/kev-0.6b` (this repo; trial `v7-06b/02-trial-2`, seed 2 of 3)
- Code, suites, results, and the full research log: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — see `PLAN.md` (full record at git tag `research-archive-2026-09-24`), `runs/leaderboard.md`, and `evals/v4/*/manifest.json`

## What changed since Kev-0.5B

| | Kev-0.5B | Kev-0.6B (this) |
|---|---|---|
| backbone | Qwen2.5-0.5B | Qwen3-0.6B-Base |
| training records | 9,000 (six sources) | 12,576 (ten public sources + 896 policy minimal pairs + 1,680 records from 60 random rule structures) |
| none-of-the-above | augmentation fix only | + minimal pairs: same state rendered with the true option present and removed |
| in-distribution accuracy (decision-v4 dev) | 0.712 | **0.801** |
| out-of-domain accuracy (transfer-v4 dev) | 0.561 | **0.620** |
| none-option present, accuracy | 0.25 (transfer-v1) | 0.80 |
| seeds behind the number | 1 | 3 (transfer 0.605–0.620) |

Jev (`typesafe-ai/jev` via Vercel AI Gateway) on the same frozen development sets: **0.845** in-distribution, **0.857** out-of-domain. Per-source transfer accuracy for this checkpoint: QNLI 0.85, SciQ 0.93, TweetEval-offensive 0.69, PAWS 0.59, Emotion 0.49, MMLU 0.50; held-out policy structures near chance.

## Known limits

- **Out of domain it is a 0.6B model.** Transfer accuracy is flat at ~0.60 across every hyperparameter we tried (eight one-knob mutations, three seeds). The same recipe at 4B reaches 0.72–0.75 and at 8B 0.74–0.77; capacity, not data, is the bottleneck at this size.
- **Held-out policy reasoning fails**: on programmatic policy pairs whose rule structure was never trained, both-siblings-correct is 6–11% (Kev-4B 0.73, Jev 0.86).
- **Ordinal hedging**: on 3-level Score questions with date arithmetic it collapses to the middle level.
- Confident-error rate out of domain is 11% (≥0.9 confidence and wrong); raw ECE 0.09 in-domain, 0.15 out of domain. Probabilities are usable in-domain; treat them as advisory elsewhere.
- **Locked test, read once** (`runs/locked/kev-06b-v7-ungated/`): in-distribution accuracy **0.808** (Brier 0.266, ECE 0.089), out-of-domain **0.642** (Brier 0.483, ECE 0.128, confident errors 7.9%). This partition will not be read again for this checkpoint.

## Architecture

Prefill-only causal LM with a block-causal attention mask: a shared state prefix, one isolated branch per question, and a pointer readout over option boundary tokens. Questions packed into one request get exactly the probabilities they would get alone (measured max delta 4e-6). Details in the repository README.

## Training

Frozen suite `evals/v7/decision-v7` (manifest pins dataset and base-model revisions): 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, and 1,680 records from 60 randomly generated rule structures, two epochs, LoRA r=16 on attention and MLP projections at lr 1e-4, pointer head from scratch, cross-entropy on the option distribution, bf16 autocast with fp32 master weights on one H100 (~12 min). Augmentation: option permutation, none-of-the-above insertion, distractors, and none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; a locked test partition exists and is read at most once per promoted candidate. Every number above carries the suite hash, code hashes, and git commit in `result.json`. Comparisons use a record-clustered paired bootstrap. See `PLAN.md` at git tag `research-archive-2026-09-24` ("Evidence and corrections") for the corrections we made to our own earlier claims.

## Use

```python
from typesafe import TypeSafeClient   # any TypeSafe-compatible client
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")
```

Serve with `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.6b --port 8008` from the repository.

## License

Apache-2.0 for the adapter and head. The base model is Apache-2.0 (Qwen3). Training datasets carry their own licenses.
