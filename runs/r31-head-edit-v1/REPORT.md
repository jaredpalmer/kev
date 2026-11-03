# Kev-specific analytic head edit: small probability gains, failed accuracy primary

## Decision

Do not ship. This is more directly related to Kev than round 30's donor-generation edit, and it improves hard/documents
Brier and log loss slightly. It does not produce a convincing decision-accuracy gain: full, half and coordinate-shuffled
gains all have identical panel accuracies. Hard accuracy's lower bound is zero, and learned-minus-control accuracy is
exactly zero. Both preregistered primary gates fail. The other calibration/regression gates pass. Stop here rather than
choosing a new dose or regularizer against these development results.

## Mechanism and registration

Registered in `PLAN.md` and `experiments/rounds/r31.json` at commit
`09a804f39a0d84c7a93d11b01de24d8eac568676`, before any feature fitting or evaluation inference.
Model: `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f`; base and kernel provenance in `report.json`.
Temperature stays at the shipped **2.3510958125672174** throughout the primary comparison.

Kev's pointer head projects the decision token and each option token into 256 channels, multiplies those features
channel-wise, and sums to score each option. For option i, define `x_i = q * k_i * scale / T`. Fit a change delta in
channel gains; logits become `z_i + x_i @ delta`. Unequal gains can change rankings, unlike a scalar temperature.

On 512 training records from each of decision-v7, hard-v1 and documents-v1 (1,536 records, 2,177 questions), accumulate
`b = sum w X.T @ (target - p)` and `H = sum w centered(X).T @ diag(p) @ centered(X)`.
Each suite, record within suite, and question within record receives equal mass at its respective level. Use canonical
soft targets. Records are chosen by source round-robin, hash-sorted ID, clean variant only, one per group; exclude IDs,
original groups and normalized exact states seen in any evaluation panel or another selected training suite. This is
not a semantic/near-duplicate contamination proof, nor a claim that the original release never trained on these sources.

Solve once for delta under `sum(delta)=0`, adding `0.1 * trace(H)/256` to the Hessian diagonal, then scale the whole
solution to `max(abs(delta)) <= 0.25`. This is **supervised analytic fitting**, not a data-free or literally derivative-free
method. No autograd, iterative optimizer or backbone update is used. Fold `1+delta` into each row of `head.q.weight` and
its bias. Mean-zero gains exclude a pure scalar gain, but can still alter confidence because channels contribute unequally.
The seed-0 coordinate permutation control preserves the gain multiset and L2 norm, not the weight-space norm.

The edit's query-weight relative norm is 6.6008%; bias 6.1306%. Weighted in-fit NLL falls 0.445918 -> 0.443412.
This training-set figure is a fitting diagnostic, not evidence of generalization.

## Development-only results

Clean, knowable questions; accuracy in percent. All other metrics are fractions (lower Brier/ECE/NLL/AURC is better).
Raw rows include the other variants and source/task breakdowns; each arm/panel `report.json` contains those summaries.

| Panel | n | Arm | Accuracy | Brier | ECE | NLL | AURC | Coverage at 5% error |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| transfer | 656 | baseline | 64.787 | .429775 | .048954 | .716541 | .170505 | .144817 |
| transfer | 656 | learned | 64.787 | .429767 | .048952 | .716398 | .170519 | .144817 |
| transfer | 656 | half | 64.787 | .429769 | .046363 | .716465 | .170478 | .144817 |
| transfer | 656 | shuffled | 64.787 | .429722 | .047826 | .716308 | .170437 | .144817 |
| hard | 1,083 | baseline | 59.372 | .530870 | .112470 | .973454 | .242555 | .086796 |
| hard | 1,083 | learned | 59.464 | .529890 | .110895 | .971386 | .241221 | .086796 |
| hard | 1,083 | half | 59.464 | .530376 | .112333 | .972414 | .241480 | .087719 |
| hard | 1,083 | shuffled | 59.464 | .530327 | .112080 | .972317 | .241499 | .087719 |
| documents | 920 | baseline | 84.239 | .235893 | .058648 | .481987 | .042856 | .677174 |
| documents | 920 | learned | 84.239 | .235506 | .056417 | .480572 | .042827 | .677174 |
| documents | 920 | half | 84.239 | .235697 | .057528 | .481272 | .042840 | .677174 |
| documents | 920 | shuffled | 84.239 | .235590 | .056906 | .480847 | .042848 | .677174 |

Full learned minus baseline, source-stratified original-record clustered paired percentile bootstrap, 2,000 samples,
seed 0, micro aggregation; sibling questions remain together:

| Panel | Accuracy delta, pp [95% CI] | Brier delta [95% CI] | ECE delta [95% CI] | NLL delta [95% CI] |
|---|---|---|---|---|
| transfer | 0 [0, 0] | −.000008 [−.000249, +.000230] | −.000002 [−.005764, +.007313] | −.000143 [−.000558, +.000298] |
| hard | +.092 [0, +.279] | −.000980 [−.001157, −.000808] | −.001575 [−.004509, +.000649] | −.002068 [−.002382, −.001745] |
| documents | 0 [0, 0] | −.000387 [−.000535, −.000231] | −.002231 [−.006198, +.002420] | −.001415 [−.001766, −.001064] |

Learned minus shuffled hard accuracy: **0 pp [0, 0]**. Control improves hard Brier by .000543 and documents Brier by
.000303 versus baseline, so probability improvements are not exclusive to the learned assignment. The learned edit's
hard Brier gain is larger, but that was not the primary attribution criterion; do not substitute it after seeing results.
Decision-v7 is diagnostic: all arms have 82.674% accuracy; full-dose Brier .248695 -> .248740 and NLL .453869 -> .454081.
No panel's ECE improvement is reliably separated from zero. Half-dose is exploratory, not a replacement primary.

## Integrity, evidence and limits

- All four arms evaluate 1,204 / 764 / 700 / 568 records on decision/transfer/hard/documents, with zero rejection or
  truncation. The clean metric populations above exclude non-clean/unknowable variants as in the canonical evaluator.
- Fresh baseline `rows.json` is byte-identical to round 30 on every panel; the model and inference paths are unchanged.
- Only `head.q.weight` and `head.q.bias` change. Backbone, LoRA and key projection remain hash-identical, and restoration
  in a finally block is exact. All four packing/sibling checks pass (maximum sibling probability delta < .000406,
  tolerance .001; packing delta 0).
- `fit-selection.json` records selected IDs/groups/sources/record hashes. `fit.safetensors` contains b, H and delta;
  `gains.safetensors` contains the four applied gain changes. These are edit/statistic artifacts, not a published model.
  Local CPU re-solving matches within 6.5e-15; cross-CPU linear algebra is not claimed bit-identical.
- `report.json` records script/source hashes, fitting input hashes, changed-parameter hashes, environment, frozen-T
  metrics and intervals. Unit tests reproduce all 36 baseline comparisons plus the control comparison from saved rows.
- `runs/r31-readout/round31.json` is the canonical secondary readout; its in-distribution decision-v7 temperature fit
  happens to return the shipped T for every arm. It is not held-out calibration and selects no candidate.
- No locked tests, checkpoint publication, Hub writes, change to shared inference code, or post-result search. A future
  positive candidate would still need independent calibration and locked-test qualification. This small null pilot does
  not establish that every higher-dimensional or targeted edit must fail.

## Cost and reproduction

[Modal app](https://modal.com/apps/jp-1083/main/ap-zmFFCn0bKxl7KTftcHxyBb), one H100, retries disabled, script 848.4 seconds.
Metered app compute **$1.03453412**, split across Nov 2/3 UTC; combined with round 30 **$2.24693682**, below the original
$10 cap. Storage/egress are separate. One-hour conservative compute bound $5.6752. Details: `cost.json`.

```sh
# CPU only: reproduce the fit and all frozen-T comparisons from committed evidence
uv run python -m pytest tests/test_head_weight_edit.py tests/test_weight_space_pilot.py -q
uv run python -m kev.rounds validate experiments/rounds/r31.json
uv run python -m kev.rounds readout experiments/rounds/r31.json

# Paid rerun only if wanted: use a new output name; this refuses an existing directory
uv run modal run --detach modal_app.py::script --script head_weight_edit.py \
  --name r31-head-edit-reproduction --gpu H100 --timeout 3600
```

Thermonuclear structural review: no shared model, training, serving, data or metric changes. The experiment reuses the
canonical loader, materializer, training-policy check, evaluator, mechanism checks, bootstrap and round readout; only
feature capture/analytic solve and bounded weight mutation are new. No new file exceeds 1,000 lines.
