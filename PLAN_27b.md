# Plan: a bigger Kev — Qwen3.8-27B, question-side LoRA, long documents

Status: **proposal, 2026-09-21; gated on [`PLAN.md` Round 4](PLAN.md#round-4--current-architecture-levers-before-the-27b-registered-2026-09-22) as of 2026-09-22.** Nothing here has been started. Written after reading [DoccyHealth/Solomon](https://huggingface.co/DoccyHealth/Solomon) (card and code), [Bonsai 2 27B](https://x.com/PrismML/status/2100692248480596348), Archer Hume's release thread ([1](https://x.com/4rcherhume/status/2101888238357237798), [2](https://x.com/4rcherhume/status/2101965358047596823)), and the night-2 results in [`PLAN.md`](PLAN.md). Budget assumption: Modal credits up to $10k; this plan uses ~$3–4k and says where each dollar goes. Every number below is either ours (linked to a result file) or theirs (linked to the card). Working notes were in `scratchpad.txt` (deleted; in git history up to f3bae08).

## Gating addendum (2026-09-22, after the Solomon / SemIf comparison)

Round 4 in `PLAN.md` runs every cheap lever at the current sizes first. Three of its results change this plan and must be read before Phase A spends anything:

- **Which base to pin (4.8).** If the post-trained `Qwen/Qwen3.5-9B` keeps its date arithmetic through Kev training, B1 pins a post-trained 27B with evidence instead of by analogy to the 35B-A3B, and A1 (question-side LoRA as a way to protect base skills) drops in priority — the cheaper protection is the base itself. If it erodes like the Base checkpoints, A1 becomes the first Phase-A run.
- **What the probe must contain (A2).** SemIf reports zero-shot Qwen3.8-27B at 0.958 balanced accuracy on its authored-144 vs 0.813 for zero-shot 4B; trained Kev-4B scores 0.847 there ([`runs/kev-4b-semif-v1`](runs/kev-4b-semif-v1/report.json)). The A2 probe therefore scores `evals/external/semif-v1` and `evals/external/wanli-v1` alongside `transfer-v4`/`v9`, and the Phase-B gate adds: authored-144 ≥ 0.90 zero-shot. A base ceiling matters more there than on our synthetic transfer set.
- **What long context costs (4.12).** The 4B long-state delta reports the short-state accuracy cost of lifting `MAX_STATE`; B2's budget is sized from that curve, not assumed.

Struck from §5: "calibration in training rather than after it" — the round-3 matched loss screen was negative ([`PLAN.md`](PLAN.md#round-3-screening-result--stopped-at-the-registered-gate)); the remaining calibration levers are ordering (round-4 4.9 soft targets, 4.10 reliability head) and per-workload temperature at the deployer (4.1). Kept from the Solomon reading and still true: MMLU 72.9 vs base 71.8 at 27B — do not sell B1 on knowledge.

## Review addendum (after the calibration audit)

This remains a **proposal, not an active large-model run**. The controlled loss screen in [`PLAN.md`](PLAN.md#round-3-screening-result--stopped-at-the-registered-gate) produced no qualifying candidate; it does not justify scaling that recipe. The current authorization is **$1,000 total**, not the hypothetical $10,000 credit grant or the full budget below.

Corrections to assumptions in the original proposal:
- Temperature preserves within-question argmax but can reorder confidence across multiclass questions. Selective coverage needs tie-aware metrics and a full-statistic clustered bootstrap; use the audited implementation and fresh final-panel protocol.
- Similar knowledge scores across the recipes tried do not prove that no training can improve knowledge retrieval or reasoning. Base capacity is a hypothesis, not an identified hard ceiling.
- Question-side LoRA may preserve an unmodified prefix, but computation after the question can still change. Cache sharing requires identical base weights, tokenizer, prefix encoding and precision. Implementation size and performance need measurement, not a line-count promise.
- Attention heatmaps are diagnostic signals, not verified evidence or causal explanations. Evidence pointers need independent labels and sufficiency/removal checks.
- Reserved outcomes and multi-label outputs require trained semantics and evaluation, not just adding API fields.
- A ternary backbone is not inherently untrainable: adapter training depends on the available differentiable runtime. Bonsai compatibility, retention, memory and latency must be probed rather than assumed.
- Historical test partitions exposed during repeated model selection are regression sets now. Do not use them as fresh confirmatory evidence, and do not transfer Solomon's accuracy/memory claims to Kev without measurement.

The staged questions below are still useful, but their timing, dollar and accuracy forecasts are unverified planning assumptions. A new run needs an explicit protocol and a measured memory/throughput check before its budget is committed.

## 1. What we learned from Solomon, and what it changes

Solomon is a LoRA (r=64, fp32, 870 MB) plus trained linear heads on **`Qwen/Qwen3.8-27B`** ([card, "How it works"](https://huggingface.co/DoccyHealth/Solomon#how-it-works)). The base is a post-trained dense hybrid — 64 layers, 48 Gated DeltaNet + 16 attention, 262k context, natively multimodal ([config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json): `qwen3_5_text`). That is the architecture `DecisionModel` already runs ([`kev/model.py`](kev/model.py), `hybrid=True`, row-batched branches). 54 GB in bf16.

Four things they do that we don't:

| idea | what it is | our take |
|---|---|---|
| **Question-side LoRA** | adapter off while the document is prefilled, on from the question onward ([`adapter/config.json: placement`](https://huggingface.co/DoccyHealth/Solomon/blob/main/adapter/config.json); [`engine_contract.py: SwitchLoRA`](https://huggingface.co/DoccyHealth/Solomon/blob/main/src/solomon/engine_contract.py)) | **Test first.** Two possible wins: (a) the base reads the document unmodified, so its skills in *reading* (date arithmetic, MMLU) may survive training — our erosion finding ([`PLAN.md`, Qwen3.5 port §10](PLAN.md#qwen35-port-2026-09-20), "The deadline hypothesis") may be a document-side effect; (b) the prefix cache becomes adapter-agnostic (one cached state serves any adapter). Cost of the ablation: ~$10. |
| **Reserved outcomes** | "the document does not state this" / "gives conflicting answers" as real options; a four-state boolean; multi-label ([`engine_contract.py: RESERVED, LABEL, OPTION`](https://huggingface.co/DoccyHealth/Solomon/blob/main/src/solomon/engine_contract.py)) | Product work, not model work. Our unknowable soft-target training ([`evals/night2`](evals/night2/manifest.json)) is the model-side half; an explicit `not_stated` option and a multi-label type are API additions. Later. |
| **Evidence pointers** | a trained relevance head ranks three sentences ([card, "Evidence"](https://huggingface.co/DoccyHealth/Solomon#evidence-experimental)) | Kev can get a first version for free: the `<decide>` token's attention over state tokens in the 8 attention layers is a heat-map over the document. Zero training; demo-able. |
| **Real documents, long context, page images** | 200 public documents, AI labels (Qwen3.8-2.4T, two blind passes, no human review), base-anchor KL, replay ([card, "Training data"](https://huggingface.co/DoccyHealth/Solomon#training-data-v11)) | **This is the real gap.** Kev trains at `MAX_STATE = 384` tokens ([`kev/model.py`](kev/model.py)). Solomon and Jev are used on documents. |

Their numbers, read with our eyes ([card, "What it scores"](https://huggingface.co/DoccyHealth/Solomon#what-it-scores)):
- Real-document test panel, 802 questions over 54 documents, AI-generated labels never checked by a human: Solomon 88.0 % vs the external reference (Jev) 86.0 %. The v1.1-over-v1.0 gain is +3.4 pp, 95 % CI [−1.3, +7.4]; they say the interval includes zero.
- MMLU + MMLU-Pro (400 + 400): **Solomon 72.9 %, base Qwen3.8-27B 71.8 %, Jev 87.1 %.** The readout training moved knowledge by 1 pp at 27B — the same finding as ours at 4B, 9B and 35B-A3B ([`PLAN.md`, night-2 results #7, #8](PLAN.md)). Knowledge is the base's; no amount of Kev training changes it.
- Per-type temperatures did not improve held-out calibration for them; a single temperature did for us ([`scripts/temperature_groups.py`](scripts/temperature_groups.py)). Different heads, different behaviour.

**Bonsai 2 27B** ([PrismML](https://x.com/PrismML/status/2100692248480596348)) is a ternary quantization of the same Qwen3.8-27B: 5.9 GB, 98.2 % retention, a WebGPU demo. Not a training base (ternary weights), but the natural path to a *laptop* Kev-27B later: a LoRA side-path on the ternary base. Parked under §5.

## 2. Where Kev stands, honestly

- Kev-9B: 0.822 development / **0.852 locked test** out of domain ([`runs/locked/kev-9b-night2-du-ungated`](runs/locked/kev-9b-night2-du-ungated/summary.json)); Jev 0.857 on the development items. Gap: knowledge (MMLU 0.74 vs 0.90, MMLU-Pro 0.52 vs 0.84), calibration (coverage at ≤ 5 % error 0.47–0.62 vs 0.70), and nothing else large.
- Every base-scaling step so far bought 1–2 pp (4B → 9B: [`PLAN.md`, Qwen3.5 port §10](PLAN.md#qwen35-port-2026-09-20); 9B → 35B-A3B: [`PLAN.md` #8](PLAN.md)); recipe and data changes bought 5–10 pp each (learning rate, rule trees, the dates/unknowable delta). Credits spent on base scale alone would be low-leverage.
- The structural limit is the 384-token state. Everything else — exact isolation, prefix cache, hybrid support, frozen suites, human-labelled evaluation, live-Jev comparisons, locked test — is in place and is stricter than what Solomon reports against.

## 3. Phase A — cheap, decisive probes (~$60, one day)

| # | question | run | decides |
|---|---|---|---|
| A1 | **Does question-side LoRA preserve the base's skills?** | 4B × 2 seeds, 9B × 1 seed, v7 recipe, adapter applied only to branch positions (`DecisionModel` gets a `lora_placement="question"` switch: scale LoRA output by a per-position mask; ~30 lines; parity test that `full` reproduces today's numbers). Read: `deadline` raw (base 0.68 / 0.82; trained today 0.55 / 0.72), MMLU, transfer, pairs. | If deadline ≥ 0.70 at 4B or ≥ 0.80 at 9B with transfer within 1 pp → question-side becomes the default placement for every later run, and the prefix cache becomes adapter-agnostic. |
| A2 | **Is Qwen3.8-27B a better base for Kev?** | Zero-shot probe on `transfer-v4` and `v9`, plain and SemIf prompts (`modal_app.py::base_probe`, H200, ~$6), plus `::smoke` for memory and step time. Reference: Qwen3.6-35B-A3B instruct scored 0.812 zero-shot with the SemIf prompt and trained to 0.823. | Gate for Phase B: zero-shot ≥ 0.80 **and** MMLU-Pro ≥ 0.65 (the 35B-A3B was 0.59 and trained to 0.55). |
| A3 | **How badly does the 384-token limit hurt?** | Current Kev-9B and Kev-4B on `transfer-v4` items with the state embedded in 1k / 2k / 4k tokens of unrelated text (the `buried` generator at scale; eval only; `MAX_STATE` lifted for evaluation). | The degradation curve sizes the long-context track in Phase B. |

## 4. Phase B — the bigger run (~$1.5–3k, one to two weeks)

**B1. Kev-27B**, only if A2 passes. Qwen3.8-27B, v7 recipe + `evals/night2/dates_unknowable.jsonl` folded into training, `--weights_dtype bf16`, LoRA on attention + DeltaNet + MLP, question-side placement if A1 won. H200 (54 GB weights + LoRA; budget 6 h timeouts — the 35B-A3B took 2.3 h at 3B active, a dense 27B will be 3–5×). 3 seeds × lr {5e-5, 2e-5} ≈ 6 trials × ~$50. Pre-registered ship rule: `transfer-v4` dev ≥ Kev-9B + 2 pp with a paired CI excluding zero, MMLU-Pro ≥ 0.65, no held-out family below Kev-9B by more than 3 pp; one locked read. Expected from what we have measured: MMLU ~0.83, deadline ~0.95, transfer 0.84–0.86 — Jev level on our suite is plausible for the first time; MMLU-Pro will still trail Jev's 0.84. Card must say the base is post-trained, not a Base checkpoint. Serving: 8-bit ≈ 30 GB (Solomon's number), bf16 54 GB; a laptop path waits for Bonsai (§5).

**B2. Long-document track**, independent of B1 and more important.
- Raise the training state limit to 4k tokens (`MAX_STATE`), keeping the packed limit consistent; the row form makes this a memory question, not an architecture one.
- Data: real public-domain and open-licence documents (US federal works, UK OGL, AU CC-BY — the classes Solomon lists), questions generated per document, labels from an open teacher (Qwen3.8-2.4T via OpenRouter, as Solomon did) with two independent passes and agreement filtering. **No Jev outputs.** A few hundred items get human-verified labels and become a frozen `documents-v1` suite with a locked test — so we never report against unverified labels the way Solomon had to.
- Train Kev-9B (and 27B if B1 passes) on `decision-v7` + the document set; report short-state suites unchanged, document suite separately.
- ~$1k: teacher labels ~$200, trials ~$800.

**B3. Autoresearch at 9B** over the enlarged data (~$700): 100 trials at ~$7 with the existing hill-climber ([`kev/autoresearch.py`](kev/autoresearch.py)), knobs = lr, LoRA rank/targets, replay fraction, soft-target weight, placement. Today's incumbents were found at 4B and carried up; 9B has never had its own search.

## 5. Phase C — follow-ons (~$500)

- Evidence pointers from `<decide>` attention over state tokens; report agreement with Solomon-style sentence pointers on the document suite.
- Multi-label type and an explicit `not_stated` option in `/v1/systemone` (the API is TypeSafe's; this would be an extension, flagged as such).
- Mac serving: MLX for the hybrid, and a Bonsai-ternary base + LoRA side-path for a 6 GB Kev-27B.
- Calibration in training rather than after it: label smoothing or a proper-scoring calibration term, judged by coverage at ≤ 5 % error (temperature cannot move it; only reordering confidences can).

## 6. What not to do

- Don't buy base scale without the probe. Three generations of evidence say the knowledge column is the base's and everything else is recipe.
- Don't train on Jev outputs, and don't report against AI labels without a human-verified subset.
- Don't fold preprocessing (`date_facts`) or temperature into the model's raw numbers in the tables; report them as rows.

## 7. Budget

| phase | cost | wall clock |
|---|---|---|
| A: probes | ~$60 | 1 day |
| B1: Kev-27B | ~$300–600 | 2–3 days |
| B2: documents | ~$1,000 | 1–2 weeks (labels are the long pole) |
| B3: 9B autoresearch | ~$700 | 2 nights |
| C: follow-ons | ~$500 | as time allows |
| **total** | **~$3–4k** | |
