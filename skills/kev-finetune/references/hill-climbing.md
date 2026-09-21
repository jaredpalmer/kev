# Reading the numbers and improving the model

`train` and `evaluate` write `runs/<name>/result.json` and print a table. Every metric is computed over development
*questions* (one record with three questions is three rows) by `kev.metrics.metrics`.

## The table

| Row | Meaning | What to want |
| --- | --- | --- |
| accuracy | argmax == label | up; temperature never changes it |
| brier | sum of squared probability error | down; rewards being right *and* honest |
| nll | negative log-likelihood of the label | down; the temperature is fitted to minimize this on calibration |
| ece | expected calibration error (10 bins) | down; calibrated should be well under raw |
| confident errors (p>=0.9, wrong) | share of *all* questions answered wrong with >= 0.9 | down, ideally near 0: these are the silent failures |
| coverage at 5% error | largest share of questions that can be accepted, highest confidence first, with <= 5% error among them | up: how much of the workload can be automated |
| mean confidence | average top probability | should track accuracy; far above accuracy = overconfident |

Columns come in pairs: `raw` is the model's logits as trained; `calibrated` divides them by the temperature fitted
on `calibration.jsonl`. Both models get their own fit on your calibration slice, so the comparison is fair.

Why this matters against Jev: a hosted model's probabilities cannot be re-fitted to your data, so on your domain its
`confident errors` and `coverage at 5% error` are whatever they are. Kev's calibrated column is a number you control.

## Deciding

1. **Is the gain real?** `bootstrap` in `result.json` holds paired bootstrap deltas (fine-tuned minus baseline) for
   `acc`, `brier` and `ece` with 95% CIs, resampling development records. A CI that excludes zero is a real change on
   this development set. With 60 development records the CI is wide; if it straddles zero, get more data before tuning knobs.
2. **Did it forget?** `regression` scores both models on 300 public `decision-v7` development records (raw logits).
   Accept a drop of ~2 accuracy points; more means the delta drifted. Lower `--lr` (halve it), keep `--replay 2000`,
   do not add epochs.
3. **Is it honest?** Calibrated `confident_error_rate` of the fine-tuned model should be <= the baseline's, and
   `mean_conf` within a few points of `acc`. Overconfident after fine-tuning usually means too many epochs or
   near-duplicate training states (the model memorized). Check `split_data.py` output for duplicates.
4. **Per question.** `development.per_question` breaks metrics down by question id. A question that did not move is
   either already solved by the baseline or has inconsistent labels in your data; open `errors.jsonl` filtered on it.

## Levers, in order of payoff

1. **More and better data.** Read the top 20 lines of `errors.jsonl` (most confident mistakes). Typical causes and fixes:
   - Same kind of state labelled two ways -> tighten `guidance` in the spec; regenerate; re-split.
   - One option rarely correct -> raise its share (`generate_data.py` rebalances against what is already in the file;
     just ask for more `--n`).
   - Model right, label wrong -> the generator misread the rule; add an explicit rule and a reference example.
   - States too short to decide -> ask for more detail in the spec's `state` description, or accept and use `target`
     soft labels for genuinely undecidable cases.
   Doubling the training set usually beats any hyperparameter.
2. **Epochs.** `--epochs 2` when you have 1000+ records and the training loss in `train.log` is still falling at the
   end of epoch 1. Watch `mean_conf` vs `acc` afterwards.
3. **Learning rate.** Recipe defaults: 4e-5 (0.8B), 2e-5 (4B, 9B). Halve on regression, do not go above 5e-5 for a delta.
4. **Replay.** `--replay 2000` (default) mixes public records so the model keeps general skill. `--replay 500` if your
   data is large (3000+) and training time matters; `--replay 0` only for a throwaway experiment.
5. **Base size.** When two data rounds stop moving the 4B, run the same data on `jaredpalmer/kev-9b`
   (`--init-from jaredpalmer/kev-9b`, ~40 min). Serve it on `KEV_SERVE_GPU=A100-80GB` or H100.

Run each change as a new name (`support-v2`, `support-v3`) and compare with
`modal run scripts/kev_modal.py::compare --a support-v2 --b support-v1` (paired bootstrap on calibrated probabilities;
both runs must have scored the same `development.jsonl`).

## Comparing against Jev or another endpoint

`evaluate --remote <base_url> --remote-model <id>` (key in `KEV_REMOTE_API_KEY`) scores any System One-compatible
endpoint on the same development file, from a CPU container. Point it at your deployed Kev to confirm the served
numbers match, or at Jev if you have a TypeSafe key. Remote probabilities are taken as returned (no temperature fit),
so the comparison is exactly "what the service gives you" vs "what your calibrated Kev gives you".

## Files on the volume

```
/runs/<name>/data/{train,calibration,development}.jsonl   what was uploaded
/runs/<name>/config.json                                  resolved training config, base, kev commit
/runs/<name>/train.log                                    kev.train output (loss every 10 steps)
/runs/<name>/checkpoint/                                  LoRA adapter, head.pt (with temperature), tokenizer
/runs/<name>/{calibration,development}/                   predictions.jsonl, rows.json, report.json
/runs/<name>/baseline/{calibration,development}/          the init checkpoint on the same records
/runs/<name>/regression/{finetuned,baseline}/             public decision-v7 sample
/runs/<name>/result.json, errors.jsonl
```

`pull --name <name>` copies everything but the checkpoint and prediction dumps to `runs/<name>/`;
`pull --checkpoint` adds the weights (a few hundred MB) for local serving.
