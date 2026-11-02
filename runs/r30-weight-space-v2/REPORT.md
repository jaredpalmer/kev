# Weight-space programming on Kev-0.8B: no useful quality gain

**Recommendation: keep Kev-0.8B unchanged.** Transferring the paper's published sparse edit at full or half strength did
not meet the preregistered success rule. This rules out a useful gain from this particular transferred edit on these
development panels, not weight-space programming in general or a Kev-specific direction search.

## What was tested

[Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/) edits an instruction-tuned Qwen model to reduce
generation loops. Kev uses Qwen-Base and scores decisions without generating text. We verified the published edit against
the donor's exact bf16 tensor hashes, then transferred its **delta**, not its replacement weights:

```text
Kev base weight := original Kev base weight + dose * (edited donor weight - original donor weight)
dose = 0 (baseline), 1 (primary), 0.5 (exploratory)
```

The two `mlp.down_proj` matrices are layers 15 and 19, 64 columns each. Exactly 130,188 values changed (65,260 + 64,928;
the paper's 131,072 touched positions include values that round back to their original bf16 bits). Full-dose changes
were 10.619% and 8.093% of the respective Kev matrix norms. Every other model parameter, including LoRA and the pointer
head, retained its original hash; the original model was restored bit for bit after the last arm. No training, gradient
updates, checkpoint publishing, test-partition reads, or changes to the released model occurred.

## Primary result: shipped temperature unchanged

All reads use the same fp32, unmerged-LoRA model in one H100 process, T=2.3510958125672174. Accuracy is over clean,
knowable questions. Confidence intervals are paired, source-stratified original-record cluster bootstrap intervals
(2,000 resamples, seed 0, micro); sibling questions remain together. Deltas are candidate minus baseline.

| Development panel | Questions | Baseline accuracy | Full dose | Half dose | Full-dose delta, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| transfer-v4 | 656 | 64.787% | 64.634% | 64.634% | -0.152 [-0.457, 0.000] |
| hard-v1 | 1,083 | 59.372% | 59.280% | 59.280% | -0.092 [-0.552, +0.369] |
| documents-v1 | 920 | 84.239% | 84.348% | 84.348% | +0.109 [0.000, +0.333] |

Full dose changed 1/656 transfer decisions, 13/1,083 hard decisions, and 1/920 document decisions. The net change was
one fewer correct transfer answer, one fewer correct hard answer, and one more correct document answer. Half dose changed
1, 8, and 1 decisions respectively, with the same net accuracy changes. There is no broad capability increase.

| Panel / metric (lower is better) | Baseline | Full dose | Half dose | Full-dose delta [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| transfer Brier | 0.429775 | 0.429920 | 0.429875 | +0.000145 [-0.000226, +0.000518] |
| transfer ECE | 0.048954 | 0.043526 | 0.041966 | -0.005428 [-0.012867, +0.007531] |
| transfer NLL | 0.716541 | 0.716748 | 0.716697 | +0.000207 [-0.000418, +0.000827] |
| hard Brier | 0.530870 | 0.531040 | 0.530920 | +0.000171 [-0.000112, +0.000452] |
| documents Brier | 0.235893 | 0.235660 | 0.235802 | -0.000232 [-0.000456, -0.000017] |

The small transfer ECE reduction is not separated from zero. Documents Brier and NLL improve slightly, but this does
not rescue the failed primary: transfer accuracy does not improve, and its Brier upper bound is positive. Half dose
was preregistered as exploratory and supplies no alternative success. These are development-only intervals, not a
multiple-comparison-adjusted confirmation. All precise values, intervals, NLL/ECE, task metrics and selective metrics
are in `report.json` and each arm's suite reports.

## Integrity and diagnostics

- Complete coverage for all 12 reads: 9,708 records, 12,705 question predictions, zero rejections or truncations.
- Transfer packing/isolation passes all arms: packed-vs-solo max delta 0; sibling max delta <= 0.000410, below 0.001.
- Transfer permutation flip rate remains 2/36 (5.556%); decision-v7 remains 3/60 (5%). The hard/documents panels have
  no permutation variants, so no permutation claim is made for them.
- Transfer selective coverage at 5% empirical error: 14.482% baseline, 14.787% full, 14.634% half. Transfer AURC:
  0.170505, 0.170883, 0.170728. These tiny changes are not a deployed selective-risk guarantee.
- `kev.rounds`' secondary temperature diagnostic refits each arm on decision-v7 development rows (1,264 clean scored
  questions). The same temperature-grid value is selected for all arms, so it does not change the conclusion. This is
  an in-distribution diagnostic, **not** a held-out calibration fit or a temperature to newly ship. Both arms are
  report-only; the canonical readout's candidate is `null`.

## Provenance and reproduction

- Kev: `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f` (v1.0).
- Kev base: `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Donor: `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
- Published edit: `RikiyaT/lm_understand-logs@4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`, `exp8_math/runs/p50c/edit/`.
- Registration: `experiments/rounds/r30.json` and PLAN, committed before evaluation. Initial registration commit:
  `8632761`; executed source: `32472dc`. Later changes add a torch-backend assertion and tests, not a numerical change.
- Executed script SHA256: `9699a368fff6eda60a6f5c48e2723a2a5441f6e7ea8b8bb9bae14f4002ea2a96`.
- Kernel set: NVIDIA H100 80GB HBM3, torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, FLA 0.5.2,
  Triton 3.8.0, causal-conv1d 1.7.0, fp32 SDPA; exact environment and source hashes in `report.json`.
- Successful [Modal run](https://modal.com/apps/jp-1083/main/ap-u7rypySYFabAuSBdNwZX0W): 888.20 s script wall time.
  Volume `kev-runs:/scripts/r30-weight-space-v2` retains all outputs, predictions, the source edit artifacts and
  `delta.safetensors`; Git retains benchmark rows, reports, provenance and the canonical readout, not donor/model weights.
- [Attempt v1](https://modal.com/apps/jp-1083/main/ap-mXz13pYEjg4OGAkREIRtOC) failed before model loading/evaluation
  because the donor uses an indexed shard filename. Fixed by resolving the pinned index; no hypotheses or panels changed.

```bash
# New immutable name required for a new GPU run; no retraining or model publication.
uv run modal run --detach modal_app.py::script --script weight_space_pilot.py \
  --name r30-weight-space-reproduction --gpu H100 --timeout 3600

# Local analysis of committed JSON rows, no GPU or database/SQL needed.
uv run python -m kev.rounds validate experiments/rounds/r30.json
uv run python -m kev.rounds readout experiments/rounds/r30.json
uv run python -m pytest tests/test_weight_space_pilot.py -q
```

The preregistered $10 cap was not expanded. Metered app compute was **$1.2124**: $1.0977 for the successful pilot and
$0.1147 for the failed pre-read attempt, excluding separately metered storage/egress and before credits. The conservative
successful-call compute bound was $5.6752; per-app billing provenance is in `cost.json`. Billing can lag; no billable
GPU calls remain running. No follow-on search is justified by this pilot alone. A new experiment should identify a
Kev-specific decision-error direction, not assume that a generation-loop direction transfers.
