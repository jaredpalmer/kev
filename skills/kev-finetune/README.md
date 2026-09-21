# kev-finetune

Fine-tune an open Jev-style decision model on your own questions, get calibrated probabilities, and serve it as a
TypeSafe System One endpoint. No GPU on your machine; everything runs on Modal from six short scripts.

`SKILL.md` is the agent-facing version of this page. Install it into your coding agent with

```bash
npx skills add jaredpalmer/kev@kev-finetune
```

and say "fine-tune Kev on my support tickets". The agent will interview you, find the questions your code already asks,
generate data, train, show you the numbers, deploy, and clean up. The rest of this page is the same recipe for humans.

## What you need

- Python 3.10+ and [uv](https://docs.astral.sh/uv/); `uvx modal setup` once for a [Modal](https://modal.com) account
  (training costs about $1 per Kev-4B run on an H100; serving on an L4 scales to zero).
- Optional: an OpenAI-compatible API key to generate training data (~$0.25 per 1000 records with gpt-4.1-mini).

## The recipe

1. **Describe the decision** in `workload.json`: the input, and the questions in System One shape (`noul` yes/no,
   `choice` named options, `score` ordered levels). Start from `assets/workload.example.json`. If your code already
   calls Jev / TypeSafe, `python3 scripts/extract_workload.py path/to/repo --out workload.json` drafts it from the
   call sites.
2. **Size the dataset**: `python3 scripts/plan_size.py workload.json` tells you how many records make a +5 point gain
   over the released model measurable (about 1000 for three questions per record).
3. **Get labelled records** (see `references/data-generation.md`):
   - from data you have: `python3 scripts/convert_data.py workload.json tickets.csv --state body --label team=dept --out data/x.real.jsonl`
   - from an LLM: `KEV_GEN_API_KEY=... python3 scripts/generate_data.py workload.json --n 1000 --out data/x.jsonl --examples data/x.real.jsonl`
4. **Split**: `python3 scripts/split_data.py data/x.jsonl --out data/x [--holdout data/x.real.jsonl]`.
5. **Train + calibrate + score**: `modal run scripts/kev_modal.py::train --data data/x --name x-v1 --init-from jaredpalmer/kev-4b`.
   Prints baseline vs fine-tuned, raw vs calibrated; writes `runs/x-v1/result.json` and `errors.jsonl`.
6. **Deploy**: `KEV_SERVE_SECRET=kev-serve-key KEV_SERVE_RUN=x-v1 modal deploy scripts/kev_modal.py`, then point your
   TypeSafe client's `base_url` at the printed URL (`references/deploy.md`).
7. **Tear down**: `modal run scripts/kev_modal.py::teardown --everything --yes`.

Iterate between 3 and 5: read `errors.jsonl`, tighten the labelling rules in the spec, add records, retrain as `x-v2`,
`modal run scripts/kev_modal.py::compare --a x-v2 --b x-v1`. `references/hill-climbing.md` explains every number.

## Why fine-tune at all

The released Kev checkpoints already answer these questions zero-shot, and `train` scores that baseline for you. What
the fine-tune adds on a specific workload is (a) accuracy on your labels and (b) a temperature fitted to your data, so
the confidence you threshold on means what it says. On the example workload (support tickets, three questions, 400
records) the 4B went from automating 47% of decisions at a 5% error budget to 61%, with confident errors near zero.
A hosted model cannot be recalibrated to your data; that is the whole argument.

## Layout

```
SKILL.md                      agent instructions (phases, interview, gotchas)
scripts/extract_workload.py   find Jev/TypeSafe calls and labelled files in a codebase; draft the spec
scripts/convert_data.py       CSV/JSONL with labels -> Kev records
scripts/generate_data.py      spec -> labelled records via an OpenAI-compatible model
scripts/plan_size.py          how many records for a significant comparison; post-hoc from result.json
scripts/split_data.py         validate + split by state (optional real holdout)
scripts/kev_modal.py          Modal app: validate, train, evaluate, compare, pull, publish, teardown, Serve
references/                   data-format, data-generation, hill-climbing, deploy
assets/workload.example.json  a complete spec to copy
```

The scripts are standard-library Python (plus `modal`) and short on purpose: when your case does not fit, edit them.
