# Plan: a bigger Kev — Qwen3.8-27B, question-side LoRA, long documents

Status: **proposal, 2026-09-21; gated on [`PLAN.md` Round 4](PLAN.md#round-4--current-architecture-levers-before-the-27b-registered-2026-09-22) as of 2026-09-22.** Nothing here has been started. Written after reading [DoccyHealth/Solomon](https://huggingface.co/DoccyHealth/Solomon) (card and code), [Bonsai 2 27B](https://x.com/PrismML/status/2100692248480596348), Archer Hume's release thread ([1](https://x.com/4rcherhume/status/2101888238357237798), [2](https://x.com/4rcherhume/status/2101965358047596823)), and the night-2 results in [`PLAN.md`](PLAN.md). Budget assumption: Modal credits up to $10k; this plan uses ~$3–4k and says where each dollar goes. Every number below is either ours (linked to a result file) or theirs (linked to the card). Working notes were in `scratchpad.txt` (deleted; in git history up to f3bae08).

## Gating addendum (2026-09-22, after the Solomon / SemIf comparison)

Round 4 in `PLAN.md` ran every cheap lever at the current sizes first (closed 2026-09-22, $58). What it decided for this plan:

- **Base to pin (4.8): no evidence for a post-trained base.** Kev training cut the post-trained Qwen3.5-9B's `deadline` from 0.70 zero-shot to 0.47 (the Base-trained 9B: 0.72) and cost 2.6 pp of transfer accuracy against the released Kev-9B. The 35B-A3B's surviving date skill was specific to that model. A1 (question-side LoRA) is now the only open test of whether the erosion happens on the document side, and it runs before B1.
- **Long documents (4.12): a data problem at the current sizes.** The released Kev-4B / 9B lose 22–43 pp when the state is buried in 1–4k tokens of unrelated records (`evals/round4/longstate-v1`). A $3 one-epoch 4B delta on 1,800 buried records cut that to 8 / 14 / 30 pp with no measurable short-state cost against the release. B2 starts from that recipe (more 4k-plus records, then real documents) at 4B / 9B, before any 27B spend, and A3 is answered.
- **What the probe must contain (A2), unchanged:** SemIf reports zero-shot Qwen3.8-27B at 0.958 balanced accuracy on its authored-144 vs 0.813 for zero-shot 4B; trained Kev-4B scores 0.847 there ([`runs/kev-4b-semif-v1`](runs/kev-4b-semif-v1/report.json)). The A2 probe therefore scores `evals/external/semif-v1` and `evals/external/wanli-v1` alongside `transfer-v4`/`v9`, and the Phase-B gate adds authored-144 ≥ 0.90 zero-shot.

- **A2 read (2026-09-22, H200, ~$10; `runs/probes/qwen38-27b-*`).** Qwen3.8-27B zero-shot: `transfer-v4` dev **0.812** with the SemIf prompt (0.761 plain) — gate ≥ 0.80 met; SemIf-144 **0.944** (0.910 plain) — gate ≥ 0.90 met; MMLU-Pro on `transfer-v9` **0.635** — gate ≥ 0.65 missed; MMLU 0.762, `deadline` 0.95, WANLI-256 0.766 (the trained Kev-9B: 0.703), unknowable share at ≥ 0.9 0.36. Jared authorized B1 trials on 2026-09-22 despite the MMLU-Pro miss (a human override, recorded in `PLAN.md` Round 6). **B1 round-6 result:** the one-epoch v7-recipe trial at lr 5e-5 (`runs/r6-27b/00-trial-0`) reached `transfer-v4` dev 0.849 (+2.7 pp vs Kev-9B, paired CI [−0.2, +5.6]), `deadline` 0.975, MMLU 0.863, MMLU-Pro 0.655, held-out pairs 0.92, coverage at ≤ 5 % error 0.72, Brier 0.215, and every external suite above Kev-9B (SemIf 0.951, scienthoon 0.773, WANLI-v2 0.751, TypeSafe 0.843); it missed the registered candidate rule on the paired lower bound (−0.0015, needs > 0) and on `composition_held_conditional` (−6.2 pp, 2 of 32 questions; rule allows −3 pp), so it had no fresh-panel or locked read. Its isolation gate fails as every bf16-weights trial does. **A1 (question-side LoRA) is negative at 4B, 9B and 27B** (`PLAN.md` Round 6): it preserves nothing of the base's date arithmetic and costs 3–6 pp of transfer accuracy, so placement stays `full`.

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

### B1 v2 and B2 revised (registered 2026-09-23, before any data is built or training launched)

**Why.** Round 6 (`PLAN.md`) left three findings that change B1 and B2. (1) Three Kev-27B seeds agree on about +2.5 pp over Kev-9B on `transfer-v4` development, with much better calibration, knowledge and date arithmetic, but B1's rule (paired lower bound > 0 on 656 questions, and a 3 pp floor on 32-question tasks) cannot resolve a gain of that size. (2) The 27B trials skipped a part of B1's own recipe: `evals/night2/dates_unknowable.jsonl` was not folded in. (3) No 27B has seen a long state, and long-state records are the largest lever measured (+17–20 pp on buried states at 9B), with MNLI-free, threshold-0.8 soft targets keeping short states intact.

**B1 v2 recipe** (study `r6-27b-v2`, `experiments/round6/27b-v2.json`): `Qwen/Qwen3.8-27B@1d4bf0f2`, trained from the base (no `init_from`), one epoch, lr 5e-5, `weights_dtype bf16, dtype bf16, checkpointing 1, batch 1, accum 8` (batch 1 because 6k-token rows at 27B; the effective batch of 8 is unchanged), `p_none_pair 0.25, lora 16, lora_targets all, max_state 7552`, seeds 1 and 2, H200, timeout 28,800 s. Data `evals/round6/b1v2/train.jsonl` (`scripts/build_round6_data.py --b1v2`): the decision-v7 training partition (12,576 records) with the round-6 threshold-0.8 ambiguity targets applied in place and the MNLI records' targets removed, plus `dates_unknowable.jsonl` (1,425), plus 1,400 long-state records (200 / 200 / 500 / 500 of `longstate-v2/train.jsonl`, the `soft-half` subsample). Expected about 4 h and $25 per seed.

**B1 v2 rule** (replaces B1's for these two trials; written before any of them trains):
- *Selection (development sets):* `transfer-v4` development accuracy ≥ 0.842 (point); MMLU-Pro on `transfer-v9` development ≥ 0.65; unknowable share at p ≥ 0.9 ≤ 0.05; held-out pairs ≥ 0.75; buried questions of `longstate-v2` development ≥ Kev-9B's (0.572) + 10 pp with a paired lower bound > 0; pooled external accuracy (SemIf, scienthoon, wanli-v2, TypeSafe) point estimate ≥ Kev-9B's. The seed with the higher `transfer-v4` accuracy among those passing is the candidate.
- *Confirmation, read once for the candidate and Kev-9B:* `evals/round6/transfer-r6/test.jsonl` (1,260 unread questions) accuracy paired lower bound > 0 against Kev-9B and no task whose paired interval lies entirely below −3 pp; `evals/round6/longstate-v3/development.jsonl` buried accuracy lower bound > 0 and point ≥ +10 pp against Kev-9B. Reading these panels spends them for every later candidate, 9B included.
- *Locked read, once:* `kev-27b-v2`: pass if locked `transfer-v4` ≥ 0.862 and served Brier ≤ 0.237 (B1's numbers).
- A release additionally needs the bf16 serving checks (isolation tolerance and flip rate measured in bf16, a fitted temperature in `head.pt`) and a card stating that the base is post-trained and that the A2 MMLU-Pro gate was overridden.

**B2 revised protocol** (replaces "human-verified labels" with a procedure one person can run):
- *Documents:* real public documents with an open licence, preferring sources that carry their own labels (for example the US CFPB consumer-complaint narratives, public domain, with product / issue / company-response fields), plus open-licence policy and notice documents whose questions are templated. A frozen `evals/documents-v1` suite with train / development / test partitions, split by document, sha256 in the manifest.
- *Training labels:* two open-weight teachers from different families through the Vercel AI Gateway (DeepSeek V3.2 and Qwen3-235B; licences re-checked before use); a question is kept only where both agree (and agree with the native label where one exists). No majority vote, no Jev in any role.
- *Evaluation labels (development and test):* the native label where one exists, checked by a three-judge panel from families other than the teacher and Kev's base (Claude Opus 4.5, GPT-5, Gemini 3); unanimous agreement verifies an item, otherwise the item is adjudicated against the document with a written reason, and items still unresolved are dropped.
- *Adjudication, amended 2026-09-23 before any adjudication was combined:* 658 of 2,181 development/test questions (30 %) lacked a unanimous panel. Each is adjudicated twice, independently, by Devin subagents (Claude-family models, the same family as one judge; noted in the manifest) that read the full document and write accept / relabel / drop with a reason. An item is decided only where both adjudications agree (accept + accept keeps the native label; relabel + relabel to the same option relabels; drop + drop drops); every disagreement is dropped. The suite is therefore the subset whose label is clear to all readers, and its manifest reports the adjudicator agreement rate.
- *Human spot-check:* Jared reviews a random 50 items of the frozen test split in `tools/review`; the suite is described as "AI-adjudicated, human spot-checked (k/50 agreement)". If fewer than 47 of 50 hold up, the suite is not frozen and the protocol is revisited.
- *Spend:* hard cap $80 on the AI Gateway key `kev-documents-v1-labels` ($100 limit, 30-day expiry), enforced in the labelling script.

### B1 v2 result (2026-09-24; read once per the registered rule; `scripts/b1v2_readout.py`, `runs/r6-verdict/`)

| seed (1 epoch, v7 + dates/unknowable + long states + soft targets) | transfer-v4 dev | MMLU-Pro | unknowable | pairs | longstate-v2 buried | pooled externals | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.845 | **0.630** | 0.000 | 0.94 | 0.831 (Kev-9B 0.572) | 0.771 (0.761) | fails MMLU-Pro ≥ 0.65 |
| 2 | 0.848 | 0.665 | 0.000 | 0.89 | 0.849 (+27.7 [+23.3, …]) | 0.787 (0.761) | **candidate** |

**Confirmation (seed 2 vs Kev-9B, fresh panels read once):** `transfer-r6` test 0.842 → 0.863 (+2.1 pp [+0.35, +3.8], 1,260 questions, no task entirely below −3 pp); `longstate-v3` 0.556 → 0.833 (+27.7 [+23.1, +32.5]). **Locked read** (`kev-27b-v2-ungated`; first attempt failed mid-read with no summary, completed on the tool's interrupted-read path): `transfer-v4` locked accuracy **0.896** (≥ 0.862), served Brier **0.160** (≤ 0.237), ECE 0.018, coverage at ≤ 5 % error 0.835; `decision-v7` locked 0.870 / 0.185. Fitted temperature 1.38 (out-of-fold ECE 0.039 → 0.022). **B1 v2 passes every registered criterion.** Release blockers left: the bf16 serving checks (isolation / flip rate in bf16, latency; `runs/serving-27b-h200`) and a card stating the post-trained base and the A2 MMLU-Pro gate override. Two process notes: seed 1 failing MMLU-Pro by 0.02 on 200 questions while seed 2 passes is the resolution limit of that gate; and the network drop at ~21:00 killed local clients, not the detached remote reads, which were pulled from the volume unchanged.

### bf16 serving check (registered 2026-09-24T03:20Z, before the isolation measurement)

Measured already (`runs/serving-27b-h200`, H200, 200 decision-v7 development records, 280 questions): served bf16 against the fp32 evaluation path max |Δp| 0.018, mean 0.0013, 0 argmax flips (eager); with CUDA graphs max 0.014, 1 flip; 55 GB resident, 24 s load; latency with graphs 72 / 112 / 140 / 386 ms for a new state (2 questions short, 6 short, 5 on 370 tokens, 5 on 2,200 tokens), 39-85 ms on a cached state; eager 450-540 ms. The in-trial fp32 mechanism check (8 records) reports sibling max |Δp| 0.0126 against its 1e-3 tolerance because the frozen backbone is bf16. **Registered tolerance for the release:** on the same 200 records through the served path (`scripts/serving_bench.py --isolation`), each question alone against the full request and against the unrelated sibling probe must stay within the precision band just measured: max |Δp| ≤ 0.03 and at most 1 argmax flip per 280 questions, for both comparisons. Kev-9B is measured the same way for reference and does not gate.

### documents-v1 result (frozen 2026-09-23; development read, test locked and unread)

`evals/documents-v1`: 5,219 / 568 / 574 real CFPB complaint narratives (train / development / test), split evenly across 9 products and short (< 1,200 chars) / medium / long (4,000-28,000 chars, about 1k-7k tokens); two Choice questions each (product, main issue). Train: 7,488 questions whose consumer label both open-weight teachers (DeepSeek V3.2, Qwen3-235B) chose. Development / test: 920 / 936 questions, 70 % verified by a unanimous blind judge panel (Claude Opus 4.5, GPT-5, Gemini 3 Flash), the rest decided by two independent adjudications that agreed (562 of 658 agreed, 85 %; 179 consumer labels corrected, 96 disagreements and 229 agreed drops removed). Human spot check: 47 / 50 (Wilson 95 % [0.84, 0.98]), exactly the registered bar; the three disagreements were all panel-verified items. Label spend $32.26 of the $80 cap. `scripts/{build,label,freeze}_documents_v1.py`; labels, adjudications and reviews under `runs/documents-v1-work/`.

**Development read (selection set, fp32, each checkpoint's shipped or fitted temperature; `runs/docs1-*`):**

| model | all (920) | short | medium | long | product | issue | Brier | cov ≤ 5 % | vs Kev-9B (paired) |
|---|---|---|---|---|---|---|---|---|---|
| Kev-0.8B | 0.633 | 0.623 | 0.606 | 0.669 | 0.696 | 0.540 | 0.515 | 0.09 | −20.0 pp [−23.1, −16.7] |
| Kev-4B | 0.811 | 0.826 | 0.810 | 0.797 | 0.890 | 0.695 | 0.283 | 0.62 | −2.2 [−4.2, 0.0] |
| **Kev-9B (released)** | 0.833 | 0.839 | 0.845 | 0.813 | 0.890 | 0.749 | 0.254 | 0.66 | — |
| 9B round-5 C9 (long states + soft) | 0.828 | 0.830 | 0.842 | 0.813 | 0.892 | 0.735 | 0.237 | 0.68 | −0.4 [−1.6, +0.8] |
| 9B `soft-nomnli-thr08` | 0.825 | 0.823 | 0.842 | 0.810 | 0.890 | 0.730 | 0.248 | 0.62 | −0.8 [−2.2, +0.7] |
| Kev-27B trial A (1 epoch, no long states) | **0.851** | 0.866 | 0.855 | 0.833 | 0.905 | 0.773 | **0.196** | **0.79** | +1.8 [−0.6, +4.2] |
| **Jev** (`typesafe-ai/jev` via AI Gateway, `runs/jev-documents-v1`, $0.03) | **0.868** | 0.872 | 0.848 | 0.885 | 0.901 | 0.821 | **0.175** | **0.80** | +3.6 [+1.2, +6.0] |

**Gap to Jev (paired, 920 questions).** Kev-9B trails Jev by 3.6 pp accuracy [1.2, 6.0] and 0.079 Brier [0.057, 0.099] (coverage at ≤ 5 % error 0.66 vs 0.80); Kev-4B by 5.8 pp; Kev-27B trial A by 1.7 pp [−3.9, +0.3] and 0.021 Brier [−0.001, +0.042], the only Kev whose intervals include parity. The gap is concentrated where Jev is strongest: the issue question (Kev-9B 0.749 vs 0.821; 52 questions only Jev gets right vs 25 only Kev-9B) and long narratives (0.813 vs 0.885; 30 vs 8). On short and medium narratives and on the product question Kev-9B is within 1-3 pp. Jev's probabilities are near-certain on half the rows (466 of 920 at ≥ 0.999999); its ECE is 0.043 vs Kev-9B's 0.106 here.

**What it says.** (1) Real long documents do not collapse the way buried synthetic states do: the released Kev-9B loses 2.6 pp from short to long narratives, not the 22-43 pp measured on `longstate-v1`. The synthetic gap is about a state buried among unrelated records, not about length. (2) The round-5/6 long-state training buys nothing on real documents (both 9B candidates within ±1 pp of the release), so a long-document release has to be judged on real documents, and `documents-v1` train is the data to try next. (3) Kev-27B is the best model here too, with the clearest margin in calibration (Brier −0.058, coverage 0.79 vs 0.66). (4) The issue question (5 overlapping options) is where every model loses most; product is near ceiling at 9B.

### documents-v2 (private held-out test, frozen 2026-09-23)

`evals/documents-v2`: 577 CFPB complaint narratives, 953 questions, test partition only, drawn with the v1 builder from narratives v1 never drew (excluded by text hash and complaint id; `scripts/build_documents_v2.py`). Labels by the same protocol: 800 questions verified by a unanimous blind judge panel, 238 of 289 queued items decided by two agreeing adjudications (82 %; 80 consumer labels corrected), the rest dropped; label spend $7.86. Human spot check 50 / 50 (a second, independent read by another agent session also found nothing it would change). Partitions and every work file (candidates, judge labels, adjudications, reviews) live only in the private `jaredpalmer/kev-private-evals` (pinned `a54a79c4`); the public repo has the manifest alone (hashes and counts, no ids). Round 7 gives it one read per confirmed candidate, not gating. **Known limits shared with v1:** a few panel-verified product labels follow the consumer's filing where the adjudication guideline would say `credit_reporting` (credit-report disputes about a loan); some narratives carry redaction markers, CFPB page text, or are mostly pasted statute text; the product question has no none/other option. The next version should filter text quality and add that option, not v1 / v2 after the fact.

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
