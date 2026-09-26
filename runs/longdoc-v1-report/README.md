# longdoc-v1: how Kev-27B holds up past its trained state length (report only)

Kev-27B was trained on states of up to 7,552 tokens. `evals/longdoc-v1` (development partition, 1,200 records, 4,654
questions; the locked test was not read) asks the same kinds of questions over states of ~4k (the control, inside the
trained range), ~8k, ~16k, ~32k and ~64k tokens under the Qwen3.8-27B tokenizer. Two parts per bucket, 120 records each:
**cuad** (real SEC-filed contracts, CUAD's expert labels, a target contract padded with other contracts and named by its
filing title) and **synthetic** (generated bundles of service agreements: locate one, a detail at 10/50/90 % depth, a
two-hop credit lookup, a stated-or-absent term). Scored with `kev.benchmark` on one H200 in bf16.

**Scoring path.** Kev-27B is read on main's one long-state rule (#149, `kev.predictors.LocalPredictor`): a record whose
longest row (state + one question) is at most `kev.model.ROW_PASS_TOKENS` (16,384) takes the exact path (row form, math
attention kernel, no TF32): every 4k, 8k and 16k record. A longer one runs its state once (`kev.shared_prefix`) under SDPA's
fused kernels and carries `kernels: efficient` in its rows: every 32k and 64k record (1,866 questions, `long_rows` in the
read's report). The round-19 SFT arm's rows were scored before #149 merged, by this branch's first version of the long path
(state once + fused kernels at every bucket, since removed); the parity below says how much that path moves a probability.

Decision rule, written into `scripts/longdoc_report.py` before the reads: a bucket "falls" when the 95 % bootstrap
interval of its accuracy minus the 4k control's lies below zero (2,000 record-clustered resamples, seed 0). For CUAD the
cleaner read is the paired one against 8k: the 8k-64k buckets ask the same questions about the same target contracts and
differ only in padding (the 4k bucket needs targets short enough for it, a different, shorter subset).

## Accuracy / ECE / Brier by bucket (`report.json`, `report.md`)

| system | part | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|---|
| Kev-27B, shipped T = 1.38 | all | 0.930 / 0.025 / 0.097 | 0.921 / 0.026 / 0.116 | 0.919 / 0.025 / 0.115 | 0.923 / 0.027 / 0.116 | 0.918 / 0.028 / 0.122 |
| | cuad | 0.853 / 0.053 / 0.201 | 0.837 / 0.066 / 0.237 | 0.834 / 0.063 / 0.235 | 0.841 / 0.070 / 0.237 | 0.832 / 0.072 / 0.248 |
| | synthetic | 1.000 / 0.012 / 0.001 | 1.000 / 0.013 / 0.001 | 1.000 / 0.012 / 0.001 | 1.000 / 0.013 / 0.001 | 1.000 / 0.019 / 0.003 |
| r19 SFT arm (a), raw T = 1 (earlier path) | all | 0.932 / 0.049 / 0.112 | 0.924 / 0.056 / 0.129 | 0.923 / 0.050 / 0.126 | 0.922 / 0.050 / 0.128 | 0.917 / 0.055 / 0.136 |
| | cuad | 0.858 / 0.101 / 0.233 | 0.843 / 0.115 / 0.265 | 0.841 / 0.103 / 0.261 | 0.839 / 0.104 / 0.263 | 0.830 / 0.114 / 0.280 |
| | synthetic | 1.000 / 0.000 / 0.000 | 1.000 / 0.001 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.001 / 0.000 | 1.000 / 0.001 / 0.000 |
| r19 SFT arm (a), T = 1.59 (exploratory held-out-dataset T) | all | 0.932 / 0.025 / 0.102 | 0.924 / 0.032 / 0.119 | 0.923 / 0.026 / 0.117 | 0.922 / 0.028 / 0.117 | 0.917 / 0.034 / 0.125 |
| | cuad | 0.858 / 0.055 / 0.212 | 0.843 / 0.070 / 0.244 | 0.841 / 0.057 / 0.242 | 0.839 / 0.063 / 0.241 | 0.830 / 0.076 / 0.256 |
| Jev (as returned) | all | 0.937 / 0.033 / 0.102 | 0.921 / 0.035 / 0.119 | 0.919 / 0.037 / 0.121 | 0.920 / 0.032 / 0.113 | refused (0 / 240 answered) |
| | cuad | 0.869 / 0.073 / 0.212 | 0.837 / 0.074 / 0.245 | 0.839 / 0.077 / 0.244 | 0.839 / 0.070 / 0.227 | refused |
| | synthetic | 1.000 / 0.003 / 0.002 | 1.000 / 0.001 / 0.000 | 0.996 / 0.003 / 0.006 | 0.996 / 0.006 / 0.005 | refused |

Questions per bucket: 923 / 933 / 932 / 934 / 932 (4k ... 64k). Temperature only rescales, so the SFT arm's accuracy is the
same at T = 1 and T = 1.59; 1.59 is the exploratory temperature fitted on held-out datasets, not this suite.

**Where accuracy starts to fall: nowhere we can resolve.** No bucket falls for any system. Kev-27B, 64k minus 4k: all
-1.1 pp [-3.5, +1.3], cuad -2.1 pp [-6.7, +2.4]; cuad paired against 8k: 16k -0.2 pp [-1.7, +1.4], 32k 0.0 [-1.6, +1.7],
64k -0.4 [-2.0, +1.3]. The SFT arm drifts down a little more on CUAD (paired vs 8k: 32k -0.9 pp [-2.6, +0.7], 64k -1.3
[-3.3, +0.7]) but inside the interval. Jev drops 3.2 pp from 4k to 8k on CUAD ([-7.6, +1.0]) and is flat after. Calibration
moves more than accuracy on CUAD: Kev-27B's ECE 0.053 (4k) -> 0.072 (64k), Brier 0.201 -> 0.248; the SFT arm at T = 1 is
over-confident everywhere (CUAD ECE 0.10-0.11) and T = 1.59 halves it.

**CUAD with and without the targets that overlap the SFT corpus.** 17 of 102 development CUAD targets contain a LEDGAR
provision that the SFT corpus holds (`evals/longdoc-v1/overlap.json`, `cuad_targets.sft_v1_ledgar`; LEDGAR and CUAD are both
clauses of SEC-filed contracts). Kev-27B's CUAD accuracy, all targets / without those 17:

| | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|
| all targets | 0.853 | 0.837 | 0.834 | 0.841 | 0.832 |
| without the 17 | 0.859 | 0.837 | 0.834 | 0.839 | 0.831 |

The same shape; the contamination does not carry the result.

**The synthetic half is saturated.** All three systems score 1.000 (Jev 0.996 at 16k-32k) at every length, on every kind
(locate, depth 10/50/90 %, two-hop, absent). It shows that planted facts in a regular agreement bundle are still found at
64k, and nothing about how fast harder long-document reasoning degrades. A longdoc-v2 should replace it with items that need
the whole document, not one sentence of it:
- aggregation across many documents (total of a term over every agreement that meets a condition; the maximum or earliest);
- conflicting clauses with precedence rules (an order of precedence clause, a schedule that overrides the body, a later
  clause that narrows an earlier one) so the answer depends on applying the rule, not finding the sentence;
- counting (how many agreements, sites or clauses satisfy a condition, with near misses);
- cross-document multi-hop with distractors (customer -> master agreement -> order form -> rate card, where every hop has
  look-alike parties and values elsewhere in the bundle);
- amendment chains (an original agreement plus dated amendments that replace, delete or restore terms; the answer is the term
  in force on a given date).

## Parity of the scoring paths (`report.json` -> `parity`)

Kev-27B read twice on the same records: this branch's first path (state once + fused kernels everywhere,
`runs/longdoc-v1-kev-27b-prefix-path/`) against main's rule (`runs/longdoc-v1-kev-27b/`):

| bucket | main's path | questions | max abs(dp) | mean abs(dp) | argmax flips |
|---|---|---|---|---|---|
| 4k | exact | 923 | 0.014 | 0.0007 | 0 |
| 8k | exact | 933 | 0.013 | 0.0008 | 1 |
| 16k | exact | 932 | 0.013 | 0.0008 | 1 |
| 32k | shared prefix, fused (`efficient`) | 934 | 0.016 | 0.0006 | 0 |
| 64k | shared prefix, fused (`efficient`) | 932 | 0.032 | 0.0008 | 0 |

At 4k-16k this is the fused state-once path against the exact one: 2 flips in 2,788 questions, both CUAD presence questions at
p = 0.50 +/- 0.01 on both paths. The SFT arm's earlier-path rows are therefore comparable to main's to within that noise,
and were not re-read.

## Serving cost (`../longdoc-v1-serving-27b-h200/report.json`)

Kev-27B through `kev.serve.Server` as `kev.serve` loads it on CUDA (bf16, fused kernels, CUDA graphs), one H200, 6 requests
per bucket (4 questions each), each a new state with the prefix cache cleared. Measured before #149 with the server's request
limits raised to 65,536 (main's defaults now admit these states); the eager serving path it measures is unchanged by #149.

| bucket | state tokens (median) | latency, new state (median / max) | ms per 1k state tokens | peak GPU memory (max) | over resident | cached state |
|---|---|---|---|---|---|---|
| 4k | 3,581 | 385 / 442 ms | 107 | 66.7 GB | 1.1 GB | 59 ms |
| 8k | 7,382 | 930 / 1,325 ms | 126 | 68.0 GB | 2.4 GB | 261 ms |
| 16k | 14,403 | 1,968 / 2,594 ms | 132 | 69.6 GB | 4.0 GB | 508 ms |
| 32k | 28,318 | 3,589 / 3,724 ms | 124 | 73.5 GB | 7.9 GB | 529 ms |
| 64k | 59,642 | 8,257 / 8,443 ms | 138 | 81.2 GB | 15.7 GB | 548 ms |

Resident 66.0 GB (weights + graph buffers) of 150 GB. States past 4,096 tokens run eagerly (CUDA graphs cover states up to
the graph bank's width), so cost is roughly linear in state length at ~0.13 s per 1k tokens; a repeated state costs ~0.5 s
at any length. The benchmark on main's rule (unfused, adapter unmerged, exact path up to 16k) took a median 6.6 s and p95
19.6 s per record. The report file also holds the first version's parity section (its long path vs the exact path at 4k-8k:
max 0.009, 0 flips in 127 questions), superseded by the table above.

## Coverage, contamination, spend

- Kev-27B and the SFT arm answered every record. Jev (`kev.jev --count-refusals --attempts 10`, one run) answered all 960
  records at 4k-32k and none of the 240 at 64k: 56 HTTP 400 and 184 HTTP 503 on requests past 1.5 x Jev's ~32k context,
  which kev.jev now counts as refusals (the gateway answers an oversize request with either).
- Overlap (`evals/longdoc-v1/overlap.json`, counts only): 0 records contain a JevBench public item, a Kev development or
  test item, or a ContractNLI document at >= 0.5 of its word 8-grams; the LEDGAR overlap is above.
- Spend: Modal (app `kev-longdoc`, H200, billing report by app, 2026-09-26 05:20 UTC): $25.98 in all; the first two reads and
  the serving run $12.46, the Kev-27B re-read on main's rule $13.52 (2 h 46 min: the exact path recomputes the state per
  question up to 16k). AI Gateway (Jev): about $1.3 in all (two complete reads at $0.59 each, two discarded partial runs and a
  40-record probe).
