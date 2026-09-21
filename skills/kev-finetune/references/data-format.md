# Labelled record format

One JSON object per line. Each record is a System One request (`state` + `questions`) with a `label` on every question.
This is the same shape the deployed endpoint accepts, minus the labels, so what you train on is what you serve.

```json
{"state": "Order 5521 arrived two weeks late and now I see two charges on my card. Fix this today.",
 "questions": {
   "department":  {"type": "choice", "instructions": "Which team should handle this ticket first?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"},
                   "label": "billing"},
   "escalate":    {"type": "noul", "instructions": "Does this need urgent attention from a human within the hour?",
                   "label": true},
   "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm or neutral", "Annoyed", "Angry or threatening to leave"],
                   "label": 2}}}
```

## Fields

| Field | Rules |
| --- | --- |
| `state` | A string, or a JSON object / array (rendered as `key: value` lines; field names are visible to the model). Under ~1400 characters (384 tokens). Non-empty. |
| `questions` | Object keyed by question id. 1 or more. Questions see the state but not each other. |
| `type` | `noul` (yes/no), `choice` (named options), `score` (ordered levels). |
| `instructions` | The question text. Required. Write it the way the endpoint will ask it. |
| `criteria` | `choice`: object `{name: description or null}`, 1-255 options, names are what the model returns. `score`: list of 2-255 level descriptions, low to high. `noul`: optional `{"true": ..., "false": ...}` descriptions. |
| `label` | `choice`: an option name. `noul`: `true` or `false`. `score`: level index as an integer from 0. |
| `target` (optional) | Soft label: `{option key: weight}`; keys are option names / `"true"`,`"false"` / level indices as strings. Use `{"false": 0.5, "true": 0.5}` for records whose evidence was deliberately removed, to teach "no evidence, no confidence". |

Whole request (state + all questions + options) must fit 2048 tokens; each question branch 1024. Records that do not
fit are dropped by the trainer with a printed count; `kev_modal.py::validate` reports them beforehand.

## What good training data looks like

- **Same questions as production.** Instructions and option names identical to what you will send at serving time.
  The fine-tune binds those exact strings to the behaviour; paraphrases at serving time still work but gain less.
- **Balanced per option.** Every option is the correct answer at least 5% of the time, and ideally roughly equal.
  `generate_data.py` targets this automatically; `split_data.py` warns when it fails.
- **Hard negatives.** States that mention two options but where the rule picks one; polite anger; short and long inputs.
  Put the rule in the spec's `guidance` so the generator labels consistently.
- **Distinct states.** Identical states with different labels are dropped as conflicts. Near-duplicates inflate scores;
  `split_data.py` groups exact matches only, so vary names, numbers and wording.
- **Several questions per record** are fine and cheaper (one state, several labels), as long as each label is right.
- **Real examples first.** If you have 50 real labelled cases, use them as `--examples` for the generator and keep
  a separate real-only development file for the final check.

## Sizes

| Records | Use |
| --- | --- |
| 100-200 | smoke test of the pipeline; scores are noisy (development ~30 records) |
| 300-600 | first real run; bootstrap CIs start to separate baseline and fine-tuned |
| 1000-3000 | production model; consider `--epochs 2` |

The split is 70 / 15 / 15 by default (`--calibration`, `--development`). Keep at least 40 records in calibration and
development; below that the temperature fit and the scores are unstable.

## Converting existing labels

Map each source row to one record. Keep the model's `instructions` and `criteria` constant across records (put them in
a small Python dict and reuse it). Convert labels to the required type: routing names -> `choice` option names, flags ->
`noul` booleans, ratings -> `score` indices (subtract 1 if your scale starts at 1). Then run
`python3 scripts/split_data.py yours.jsonl` to validate before splitting.
