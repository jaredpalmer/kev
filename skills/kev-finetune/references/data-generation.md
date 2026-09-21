# Getting labelled data

Four sources, best first. Mix them: real rows measure, synthetic rows train.

## 1. Existing labels (`convert_data.py`)

Anything with an input text and a human decision next to it: a ticket export with its routing, a moderation queue with
verdicts, a CRM with lead scores, a spreadsheet someone filled in.

```bash
python3 scripts/convert_data.py workload.json tickets.csv \
    --state subject,body \                        # one column = string state; several = object state {column: value}
    --label department=team --label escalate=was_escalated --label priority=prio_1_to_3 \
    --map department="Billing Ops:billing,Ship:shipping" --score-offset 1 \
    --out data/support.real.jsonl
```

Columns default to the question id. Labels are matched case-insensitively to option names or option descriptions;
yes/no/1/0 become noul booleans; score labels may be indices, 1-based ratings (`--score-offset 1`) or level texts.
Skipped rows are counted by reason with an example value, so a missing `--map` is obvious. JSONL input supports dotted
paths (`--label team=meta.team`).

Then decide what the real rows are for:

- **Few (< 200):** use them as the measuring stick. `split_data.py generated.jsonl --out data/x --holdout data/x.real.jsonl`
  puts the real rows half/half into calibration and development, never into train, and drops generated rows that share
  a state with them. Also pass them to the generator as `--examples` so the synthetic rows match their style.
- **Many (1000+):** you may not need synthetic data at all. `split_data.py real.jsonl --out data/x` and train.
  Generate only to fill options the real data rarely labels.

## 2. An LLM writes the records (`generate_data.py`)

```bash
export KEV_GEN_API_KEY=...                           # or OPENAI_API_KEY; AI_GATEWAY_API_KEY switches the base URL to the gateway automatically
export KEV_GEN_BASE_URL=https://api.openai.com/v1    # https://ai-gateway.vercel.sh/v1, http://localhost:11434/v1 (Ollama), any OpenAI-compatible URL
python3 scripts/generate_data.py workload.json --n 1000 --out data/x.jsonl --model gpt-4.1-mini --examples data/x.real.jsonl
```

Each call asks for a batch of 20 records with per-question label targets computed from what is already in the file,
so the output is balanced per option; states are deduplicated; the file is appended as batches arrive (re-run with a
higher `--n` to add more). Cost: ~$0.25 per 1000 records with gpt-4.1-mini; a stronger model (gpt-4.1, claude-sonnet)
labels ambiguous cases more consistently and is worth it for the final dataset.

What makes the generated data good is the spec:

- `guidance`: the rules a careful human would apply, including how to resolve the ambiguous cases ("a late delivery
  that also asks for a refund is shipping until the package is found"). Every recurring mistake in `errors.jsonl` is a
  missing sentence here.
- `variety`: axes to spread over (tone, length, language, presence of ids/dates, two-issue inputs). Without it the model
  writes 1000 variations of the same ticket.
- `state_example`: when inputs are objects (`{"subject": ..., "body": ..., "customer_tier": ...}`), give one; the
  generator reproduces the shape and Kev renders the field names into the text.

Targeted generation for hill-climbing: copy the spec, narrow `domain`/`variety` to the failing pattern (e.g. "tickets
that mention two departments"), generate 200 into the same output file, re-split.

## 3. You (the agent) write them

`python3 scripts/generate_data.py workload.json --dry-run` prints exactly the prompt the script would send, including
the label targets for the next batch. Answer it yourself, append the `records` as labelled JSONL (the
`{"state", "questions": {id: {..., "label"}}}` shape, see `references/data-format.md`), repeat. Practical up to ~100
records; beyond that use source 2.

## 4. Programmatic

When labels follow from rules over structured fields (policy windows, thresholds, date arithmetic), write a small
generator in Python that samples fields and computes the label, then renders the state. This is how Kev's own
contrastive policy data was built; it produces perfectly consistent labels and minimal pairs (same state, one fact
changed, label flips) that teach the model what the decision actually depends on. Emit the same JSONL and go through
`split_data.py`.

## How much

`python3 scripts/plan_size.py workload.json --baseline-acc <measured or 0.75> --min-gain 0.05` gives the record count
for a paired comparison at 80% power (McNemar approximation; the unpaired bound is printed too). Typical answers:
~1000 records for three questions per record and a 5-point target; ~300 for a 10-point target. After a run,
`plan_size.py --from-result runs/x/result.json` says whether more volume would make the observed gain significant or
whether the gain is too small to chase.

Calibration needs at least ~100 questions to fit a stable temperature; the planner enforces this.

## Adapting the scripts

They are short and standard-library. Common edits:

- **Other record shapes** (several inputs per record, extra metadata): add fields to `state` as an object; Kev renders
  them. Keep `questions` as is.
- **Soft labels** for undecidable cases: set `"target": {"true": 0.5, "false": 0.5}` on the question instead of relying
  on a hard label; `split_data.py` accepts it, the trainer uses it as a distribution.
- **Different generator API** (Anthropic Messages, Gemini): replace `chat()` in `generate_data.py` (one function,
  urllib); keep `parse_records`/`to_record`.
- **Different balancing** (match production's label distribution instead of uniform): change `batch_targets`.
- **Weighted splits** (stratify by a metadata field): change `split()`; it groups by state hash today.
