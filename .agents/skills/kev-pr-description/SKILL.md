---
name: kev-pr-description
description: Write a Kev pull request title and body that teaches the reader why the change exists and how it works. Use when opening, editing or reviewing a PR on this repo, or when the PR title will become the squash-merge commit message.
---

# Writing a Kev pull request

A Kev PR is read by three people: the reviewer today, someone reading `git log` next year to learn why a number changed,
and a reader who is not an ML researcher (an engineer integrating the API, a student, a curious normie). Write for the
third one. If the description only makes sense to whoever wrote the diff, it is a changelog, not a PR.

The models for this are Andrew Clark (`acdlite`) and Sebastian Markbåge (`sebmarkbage`) on facebook/react before 2023.
Their descriptions read as short essays: problem first, mechanism second, what could go wrong third, what was left out
last. Read a few before writing a big one:

- [react#18796 Initial Lanes implementation](https://github.com/facebook/react/pull/18796): a new model explained from
  the flaw in the old one (`priority >= batchPriority` could not express "a set of tasks"), a translation table from old
  field names to new, then "Stuff I intentionally omitted".
- [react#20890 Lazily propagate context changes](https://github.com/facebook/react/pull/20890): why eager propagation
  wastes work, the exceptions (Suspense, Offscreen) and why they are exceptions, credit to the RFC and where it deviates.
- [react#19703 Disable timeoutMs argument](https://github.com/facebook/react/pull/19703): a tl;dr, then the observation
  that motivated the removal (every transition is a load or a refresh), then what users do instead.
- [react#22644 useId](https://github.com/facebook/react/pull/22644): an algorithm taught with a bit diagram and the
  sentence "The leading 0s are important", followed by why.
- [react#20970 Basic Fizz Architecture](https://github.com/facebook/react/pull/20970): a sample of the output stream
  before any code, then Principles, Execution Model, Data Structures, Error Handling.
- [react#14182 Use unique thread ID for each partial render](https://github.com/facebook/react/pull/14182): the V8
  reasoning behind a data layout, and "This *should* be a fast approach, in theory, but I haven't actually confirmed".
- [react#21021 Don't delete trailing mismatches during hydration](https://github.com/facebook/react/pull/21021): a
  small fix described as the problem, the precedent it follows, the cost of the fix ("It's a bit unfortunate that we
  can't warn") and a link to the commit that shows the cost.
- [react#25571 Try assigning fetch to globalThis](https://github.com/facebook/react/pull/25571): the whole body is
  "In case it's a more modern yet rigid environment." Length follows novelty.

## What they do that we copy

1. **Start with the world as it is.** The first paragraph describes the behaviour or gap before the change, in present
   tense, without mentioning the diff. If a reader stopped after that paragraph they should know what problem exists.
2. **Explain the mechanism as an argument, not a list of edits.** "We do X because Y; that works because Z." One
   concrete artifact per idea: the line of code that changed meaning, a tiny table, a request/response, a number.
3. **Define the one term the change hinges on, the first time it appears.** In Kev that is usually a word like pointer
   head, temperature, permutation flip rate, calibration partition, AURC, prefix cache, hybrid backbone. One clause is
   enough ("the permutation flip rate, how often the argmax changes when the same options are shuffled"). Do not define
   everything; define what the reader needs to evaluate this PR.
4. **State the assumption the change relies on.** "These optimizations rely on the constraint that components are pure
   functions" (react#14569). For us: "rows are independent, so answers cannot change", "Score options are an ordered
   scale, so they are not rotated", "the temperature is fitted on development rows, never on the test partition".
5. **Say what is uncertain and what would falsify it.** "I do expect we will find regressions." "I could go either way
   on this." Write the decision rule down before the numbers arrive: which metric, which partition, which threshold.
6. **Say what is deliberately not in the PR** and where it goes next (a follow-up, a PLAN item, never). Scope stated
   is scope reviewable.
7. **Numbers come with their provenance**: checkpoint, suite and partition, device, n, and a pointer to the committed
   report or run directory. A bare "improves accuracy" is not a claim we can verify later (`docs/claims.json` exists for
   this reason).
8. **Length follows novelty.** A new loss or serving path earns headings (Motivation, Mechanism, What is not here). A
   one-line fix earns one sentence of reason. Headings only when there is enough text to navigate.

## Kev specifics

- The title is the squash commit subject. Write it as the change in a sentence, with the PLAN item when there is one:
  `benchmark --rotations: test-time cyclic option averaging for Choice (round 4.4)`, not `Add rotations flag`.
- Keep the `Test plan` checklist, last, as in the repo today. It is where the parity evidence from `kev-verify` goes
  (byte-identical `rows.json` against main, weight-backed suites, measured latency). Evidence, not "tests pass".
- If the change moves a published number, name the file the README or model card will cite.
- No scaffolding headings with one bullet under them (`## Changes` -> a list of file names is the diff, again). No
  adjectives that do the reader's judging for them (robust, comprehensive, clean, significant without a CI).
- Plain technical English (AGENTS.md > Writing). Address the reader; say "we" for decisions the project made and "I"
  for the author's judgement calls, the way both authors do.

## Examples

Two versions of the same change. The facts are from PR #58; only the writing differs.

### Weak

> ## Summary
> - Added `RotationAveraged` predictor wrapper in `kev/predictors.py`
> - Added `--rotations` flag to `kev.benchmark` (default 1)
> - `report.json` now records `rotations`
> - Added unit test for rotation averaging
>
> ## Test plan
> - [x] Tests pass

Every line restates the diff. Nothing says why anyone would rotate options, why rotations rather than random shuffles,
why logits are averaged instead of probabilities, why Score is excluded, what it costs, or what result would make this
a serving option. A reviewer has to reverse-engineer the intent from the code; a later reader cannot.

### Strong

> Kev answers a Choice question by pointing at one of the option tokens. The pointer should depend on what each option
> says, not on where it sits in the list, but models learn position habits (favour the first slot, or the last). We
> measure that as the permutation flip rate: score the same question under several option orders and count how often
> the argmax changes. `--perm_kl` fights this at training time. This PR adds the test-time counterpart so we can
> measure how much order dependence the released checkpoints still have, and whether cancelling it at inference is
> worth K forward passes.
>
> The mechanism: rotate the options (B C D A, C D A B, ...), score each rotation, average the logits per option key,
> softmax once. Rotations rather than random permutations because K rotations put every option in every slot exactly
> once, so a pure position bias cancels exactly. `test_rotation_averaging_cancels_a_position_bias` plants a +2 logit
> bonus on slot 0 and checks the content differences come back to the bit. We average logits, not probabilities: the
> geometric mean of softmaxes is the softmax of the mean logits, so the returned probabilities and logits stay
> consistent and `kev.calibrate` can still fit a temperature on them. A remote predictor that returns no logits gets
> the same average in log-probability space.
>
> Noul and Score are left alone. Their option order is part of the question (Score is an ordered scale), so rotating
> them changes the meaning, not the presentation.
>
> Cost is K passes per question instead of one, so this is a benchmark option and, at most, an opt-in serving mode,
> never the default. `--rotations 1` is byte-identical to main (`rows.json` on the smoke checkpoint, `evals/smoke-v1`);
> `report.json` records the value used so a result cannot be mistaken for a plain run.
>
> The gate registered in PLAN round 4.4: it becomes a serving option if transfer-v4 accuracy improves by >= 0.5 pp with
> the paired-bootstrap lower bound >= 0, or if the flip rate halves. On the deliberately weak smoke checkpoint, 4
> rotations take permutation mean max |dp| 0.81 -> 0.38 and flip rate 0.83 -> 0.50. Kev-9B / Kev-4B on transfer-v4
> development at 16 rotations (capped at each question's K) are running on Modal; the numbers will be added here and
> to the PLAN round-4 results.
>
> #### Test plan
> - [x] Fast suites: 123 passed
> - [x] `test_rotation_averaging_cancels_a_position_bias` (synthetic +2-logit first-slot bias, Noul untouched)
> - [x] Parity: `kev.benchmark` on `runs/smoke-hl` / `evals/smoke-v1` (CPU), `rows.json` byte-identical to `origin/main` at `--rotations 1`
> - [x] Smoke run with `--rotations 4`: `mean_max_delta` 0.81 -> 0.38, flip rate 0.83 -> 0.50

A reader who has never heard of a pointer head now knows what order dependence is, why this fixes it exactly rather than
approximately, what it costs, and what number decides its fate.

### Small change

Weak: `Fix n_perm validation on the permute endpoint.`

Strong: `n_perm` on `/v1/systemone/permute` is now `Field(ge=1, le=64)`. `n_perm=0` divided by an empty average and a
large count ran one request for minutes; 64 is the cap #30 proposed, and #53 sized the row budget on 64-question requests.

One sentence of problem, one of reason for the bound. That is the whole body.

## Before you open it

- Could someone outside the project say, from the first paragraph alone, what was wrong before?
- Is there one concrete artifact (number, line, request) per idea?
- Is the assumption the correctness depends on written down?
- Is the decision rule written before the result?
- Does "Test plan" carry evidence a reviewer could re-run?
- Did you cut every sentence that only repeats the diff?
