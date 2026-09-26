# longdoc-v1: how Kev-27B holds up past its trained state length (report only)

Kev-27B was trained on states of up to 7,552 tokens. `evals/longdoc-v1` (development partition, 1,200 records, 4,654
questions; the locked test was not read) asks the same kinds of questions over states of ~4k (the control, inside the
trained range), ~8k, ~16k, ~32k and ~64k tokens under the Qwen3.8-27B tokenizer. Two parts per bucket, 120 records each:
**cuad** (real SEC-filed contracts, CUAD's expert labels, a target contract padded with other contracts and named by its
filing title) and **synthetic** (generated bundles of service agreements: locate one, a detail at 10/50/90 % depth, a
two-hop credit lookup, a stated-or-absent term). Scored with `kev.benchmark` on one H200 in bf16 through the suite's
long-document path (state once, rows from its cache, fused attention kernels), whose parity with the exact path is below.

Decision rule, written into `scripts/longdoc_report.py` before the reads: a bucket "falls" when the 95 % bootstrap
interval of its accuracy minus the 4k control's lies below zero (2,000 record-clustered resamples, seed 0). For CUAD the
cleaner read is the paired one against 8k: the 8k-64k buckets ask the same questions about the same target contracts and
differ only in padding (the 4k bucket needs targets short enough for it, a different subset).

## Accuracy / ECE / Brier by bucket (`report.json`, `report.md`)

| system | part | 4k | 8k | 16k | 32k | 64k |
|---|---|---|---|---|---|---|
| Kev-27B, shipped T = 1.38 | all | 0.930 / 0.026 / 0.097 | 0.920 / 0.026 / 0.116 | 0.921 / 0.025 / 0.114 | 0.923 / 0.027 / 0.116 | 0.918 / 0.026 / 0.122 |
| | cuad | 0.853 / 0.056 / 0.201 | 0.834 / 0.067 / 0.237 | 0.836 / 0.065 / 0.235 | 0.841 / 0.070 / 0.237 | 0.832 / 0.072 / 0.248 |
| | synthetic | 1.000 / 0.012 / 0.001 | 1.000 / 0.013 / 0.001 | 1.000 / 0.012 / 0.001 | 1.000 / 0.013 / 0.001 | 1.000 / 0.019 / 0.003 |
| r19 SFT arm (a), raw T = 1 | all | 0.932 / 0.049 / 0.112 | 0.924 / 0.056 / 0.129 | 0.923 / 0.050 / 0.126 | 0.922 / 0.050 / 0.128 | 0.917 / 0.055 / 0.136 |
| | cuad | 0.858 / 0.101 / 0.233 | 0.843 / 0.115 / 0.265 | 0.841 / 0.103 / 0.261 | 0.839 / 0.104 / 0.263 | 0.830 / 0.114 / 0.280 |
| | synthetic | 1.000 / 0.000 / 0.000 | 1.000 / 0.001 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.001 / 0.000 | 1.000 / 0.001 / 0.000 |
| r19 SFT arm (a), T = 1.59 (exploratory held-out-dataset T) | all | 0.932 / 0.025 / 0.102 | 0.924 / 0.032 / 0.119 | 0.923 / 0.026 / 0.117 | 0.922 / 0.028 / 0.117 | 0.917 / 0.034 / 0.125 |
| | cuad | 0.858 / 0.055 / 0.212 | 0.843 / 0.070 / 0.244 | 0.841 / 0.057 / 0.242 | 0.839 / 0.063 / 0.241 | 0.830 / 0.076 / 0.256 |
| Jev (as returned) | all | 0.938 / 0.035 / 0.102 | 0.921 / 0.034 / 0.119 | 0.919 / 0.036 / 0.122 | 0.919 / 0.033 / 0.113 | refused (0 / 240 answered) |
| | cuad | 0.871 / 0.075 / 0.211 | 0.837 / 0.072 / 0.244 | 0.836 / 0.077 / 0.245 | 0.835 / 0.073 / 0.229 | refused |
| | synthetic | 1.000 / 0.003 / 0.002 | 1.000 / 0.001 / 0.000 | 0.998 / 0.005 / 0.006 | 0.998 / 0.005 / 0.004 | refused |

Questions per bucket: 923 / 933 / 932 / 934 / 932 (4k ... 64k). Temperature only rescales, so the SFT arm's accuracy is the
same at T = 1 and T = 1.59; 1.59 is the exploratory temperature fitted on held-out datasets, not this suite.

**Where accuracy starts to fall: nowhere we can resolve.** No bucket falls for any system. Kev-27B, 64k minus 4k: all
-1.1 pp [-3.5, +1.3], cuad -2.1 pp [-6.7, +2.4]; cuad paired against 8k: 16k +0.2 pp [-1.4, +2.2], 32k +0.2 [-1.6, +2.3],
64k -0.2 [-2.0, +1.6]. The r19 SFT arm drifts down a little more on CUAD (paired vs 8k: 32k -0.9 pp [-2.6, +0.7], 64k -1.3
[-3.3, +0.7]) but inside the interval. Jev drops 3.5 pp from 4k to 8k on CUAD ([-7.8, +0.8]) and is flat after. Calibration
moves more than accuracy on CUAD: Kev-27B's ECE 0.056 (4k) -> 0.072 (64k), Brier 0.201 -> 0.248; the SFT arm at T = 1 is
over-confident everywhere (CUAD ECE 0.10-0.11) and T = 1.59 halves it.

**What this probe can and cannot see.** The synthetic part is at ceiling for all three systems at every length (Kev-27B
1.000 on locate, depth 10/50/90 %, multihop and absent at 64k), so it only shows that planted facts in a regular document
bundle are still found at 64k, not how fast harder long-document reasoning degrades; a v2 needs harder synthetic items.
CUAD carries the signal: governing law is 1.000 everywhere, clause presence ~0.80 and clause category ~0.75-0.83 at every
length.

## Serving cost (`../longdoc-v1-serving-27b-h200/report.json`)

Kev-27B through `kev.serve.Server` as `kev.serve` loads it on CUDA (bf16, fused kernels, CUDA graphs), one H200, request
limits raised to 65,536 tokens, 6 requests per bucket (4 questions each), each a new state with the prefix cache cleared:

| bucket | state tokens (median) | latency, new state (median / max) | ms per 1k state tokens | peak GPU memory (max) | over resident | cached state |
|---|---|---|---|---|---|---|
| 4k | 3,581 | 385 / 442 ms | 107 | 66.7 GB | 1.1 GB | 59 ms |
| 8k | 7,382 | 930 / 1,325 ms | 126 | 68.0 GB | 2.4 GB | 261 ms |
| 16k | 14,403 | 1,968 / 2,594 ms | 132 | 69.6 GB | 4.0 GB | 508 ms |
| 32k | 28,318 | 3,589 / 3,724 ms | 124 | 73.5 GB | 7.9 GB | 529 ms |
| 64k | 59,642 | 8,257 / 8,443 ms | 138 | 81.2 GB | 15.7 GB | 548 ms |

Resident 66.0 GB (weights + graph buffers) of 150 GB. States past 4,096 tokens run eagerly (CUDA graphs cover states up to
the graph bank's width), so cost is roughly linear in state length at ~0.13 s per 1k tokens; a repeated state costs ~0.5 s
at any length. The benchmark path (unfused, adapter unmerged) took a median 2.5 s and p95 9.9 s per record.

**Parity of the long-document scoring path** (same report, `parity_exact_vs_long_path`): 8 records per part from the 4k
and 8k buckets scored by the exact path (row form, math attention kernel) and by the long path (state once, fused
kernels), Kev-27B at T = 1.38: 4k max |dp| 0.0070, mean 0.0008, 0 argmax flips in 63 questions; 8k max 0.0090, mean 0.0008,
0 flips in 64.

## Coverage, contamination, spend

- Kev-27B and the SFT arm answered every record. Jev answered all 960 records at 4k-32k and none of the 240 at 64k: 212
  HTTP 400 and 28 HTTP 503 on both tries (the gateway answers past-context requests with either). kev.jev keeps retrying a
  503, so the read was finished by `scripts/longdoc_jev_finish.py` from the first 960 answered records of a kev.jev run
  (`usage.json` records both halves); two earlier kev.jev runs stopped on transient 503s and were discarded unread.
- Overlap (`evals/longdoc-v1/overlap.json`, counts only): 0 records contain a JevBench public item, a Kev development or
  test item, or a ContractNLI document at >= 0.5 of its word 8-grams. LEDGAR (in the SFT corpus, also SEC contract
  clauses) does occur inside CUAD contracts: 17 of 102 development targets contain a LEDGAR provision that the SFT corpus
  holds. Without those targets Kev-27B's CUAD accuracy is 0.859 / 0.834 / 0.836 / 0.839 / 0.831 (4k ... 64k), the same
  shape (`cuad.no_sft_ledgar`).
- Spend: Modal for this probe (app `kev-longdoc`): two benchmark containers (~75 min each) and one serving container
  (~15 min) on H200, about $15 (billing report by app; the workspace total also moves with round 20). AI Gateway (Jev):
  about $0.72 (14.1M input tokens answered, plus the two discarded runs and a 40-record probe).
