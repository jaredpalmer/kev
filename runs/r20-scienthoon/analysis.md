# Where the full-weight arms lose scienthoon (rounds 19-20; report only, no new reads)

Numbers: `runs/r20-scienthoon/drift.json`, from `uv run python scripts/scienthoon_drift.py --out runs/r20-scienthoon`.
Only committed rows are used: this checkout's, plus `git show research-archive-2026-09-24:<path>` for the archived 27B
LoRA reads and AutoJev. Deltas are paired record-clustered bootstraps against Kev-27B (`kev.rounds.paired`, 2,000
resamples, seed 0, micro), in pp. Argmax accuracy does not depend on temperature, so rows are compared as saved.

## Headline

The scienthoon cost of the full-weight arms comes from one binary judgement: whether a neutrally worded complaint
"sounds angry". Queue routing does not move (+0.3 pp for every arm). The arms call a calm ticket about a real problem
("The box for order #8223 was crushed and the item inside is broken.") angry 28-54 times where Kev-27B does so 9 times.
The gold labels on those questions agree with the text. Kev-27B is not the norm, though: it is the best of six LoRA
checkpoints trained from the same base on Kev's data. Its own recipe's other seed makes 31 of these errors, and none of
the five other checkpoints would pass round 19's scienthoon guard or pooled-externals guard against it. The full-weight
arms land at the LoRA family's average (0.757-0.778 against a mean of 0.765).

## The suite

`evals/external/scienthoon-v1` has 291 synthetic support tickets and three questions per ticket (873 questions):
- `queue` (Choice of 4), routing by topic.
- `priority` (Score 0-3). The label follows an organisation rule that is absent from the text (template urgency + angry +
  gold/enterprise tier; the manifest's protocol note), so it cannot be learned from the ticket.
- `angry` (Noul: "The customer sounds angry.").

Every body is an issue sentence plus an optional closing phrase. The phrase is either angry ("This is unacceptable and I
want it fixed today.", "정말 화가 납니다. 당장 처리해 주세요.", "... Ridiculous.", five in all) or neutral ("Appreciate any help.",
"답변 기다리겠습니다.", five in all), so the text class of every `angry` label can be read off the body. 95 bodies carry an angry
phrase and 108 labels say angry. **15 labels contradict the text** (14 neutral bodies labelled angry, 1 angry body
labelled calm; ids in `drift.json` → `tickets`). Those 15 are label noise that no text-faithful model gets right, so the
`angry` ceiling is 276 / 291 = 0.948. Every 27B checkpoint misses 8-15 of them. Models that over-call anger get a few
right by accident.

## Where the loss is

| checkpoint (kind) | scienthoon | queue | angry | priority | angry false positives on neutral text | Δ vs Kev-27B, all | Δ without `angry` |
|---|---|---|---|---|---|---|---|
| **Kev-27B** (B1 v2 s2, LoRA) | **0.796** | 0.948 | 0.911 | 0.529 | 9 | - | - |
| B1 v2 s1 (same recipe, other seed) | 0.769 | 0.952 | 0.832 | 0.522 | 31 | −2.7 [−4.2, −1.4] | −0.2 [−1.5, +1.2] |
| B1 trial A (LoRA, v7) | 0.773 | 0.952 | 0.873 | 0.495 | 15 | −2.3 [−3.9, −0.8] | −1.5 [−3.4, +0.3] |
| B1 trial B (LoRA, v7) | 0.742 | 0.952 | 0.804 | 0.471 | 43 | −5.4 [−7.7, −3.4] | −2.7 [−5.2, −0.5] |
| 2 epochs s1 (LoRA) | 0.772 | 0.952 | 0.849 | 0.515 | 24 | −2.4 [−4.0, −0.8] | −0.5 [−2.6, +1.4] |
| 2 epochs s2 (LoRA) | 0.740 | 0.952 | 0.746 | 0.522 | 59 | −5.6 [−7.7, −3.6] | −0.2 [−2.1, +1.5] |
| round 10 skills (LoRA delta from Kev-27B) | 0.778 | 0.945 | 0.897 | 0.491 | 15 | −1.8 [−3.0, −0.7] | −2.1 [−3.6, −0.5] |
| round 17 (a) (LoRA delta from Kev-27B) | 0.781 | 0.931 | 0.918 | 0.495 | 9 | −1.5 [−2.7, −0.2] | −2.6 [−4.5, −0.9] |
| round 19 (a) full, `sft-v1`, lr 2e-6 | 0.767 | 0.952 | 0.852 | 0.498 | 28 | −2.9 [−4.5, −1.4] | −1.4 [−3.3, +0.3] |
| round 19 (b) full, `sft-v1`, lr 5e-6 | 0.761 | 0.952 | 0.794 | 0.536 | 45 | −3.6 [−5.3, −1.7] | +0.5 [−1.2, +2.4] |
| round 19 (c) full, Kev's own data | 0.759 | 0.952 | 0.780 | 0.546 | 54 | −3.7 [−6.0, −1.5] | +1.0 [−1.2, +3.3] |
| round 20 a-w85 / a-w70 / a-w50 | 0.764 / 0.763 / 0.757 | 0.952 | 0.842 / 0.825 / 0.797 | 0.498 / 0.512 / 0.522 | 31 / 36 / 46 | −3.2 / −3.3 / −3.9 | −1.4 / −0.7 / −0.2 |
| round 20 b-w85 / b-w70 / b-w50 | 0.758 / 0.766 / 0.778 | 0.952 | 0.794 / 0.818 / 0.845 | 0.529 / 0.529 / 0.536 | 46 / 40 / 31 | −3.8 / −3.0 / −1.8 | +0.2 / +0.2 / +0.5 |
| AutoJev-27B (full, other data) | 0.769 | 0.931 | 0.938 | 0.436 | 0 | −2.7 [−4.9, −0.6] | −5.5 [−8.4, −2.7] |
| Jev (converted from the suite's own Jev read) | 0.753 | 0.897 | 0.914 | 0.447 | 8 | −4.4 [−6.3, −2.4] | −6.7 [−9.6, −3.8] |

- **By question type.** For every round-19/20 arm the loss is `angry`: −5.8 to −13.1 pp. `queue` gains +0.3 pp (one
  question) for every arm. `priority` ranges from −3.1 to +1.7 pp and is not learnable from the text. With `angry` removed,
  the arms sit at −1.4 to +1.0 pp.
- **By label (confusion against Kev-27B).** The arms over-predict "angry". They call 123-154 tickets angry, where the
  gold labels say 108, the text 95 and Kev-27B 102. Kev-27B's mean p(angry) on calm, correctly labelled tickets is 0.137;
  the arms give 0.19-0.33. False negatives on angry text stay at 0-2 for everyone. On `priority` the arms predict higher
  levels than Kev-27B: mean predicted − gold is −0.04 to +0.03, against −0.12. Their accuracy there is about the same.
- **By ticket kind and language.** 49 of the 55 flipped questions are problem tickets (missing package, damage, refund,
  double charge, crash), not inquiries. Kev-27B's own 9 false positives are all Korean-issue tickets, and Jev's 8 are too.
  The arms add English ones: (a) has 11 English and 17 Korean, (b) 18 and 27.
- **Option count and state length** do not vary here. `angry` is always 2 options, `queue` 4 and `priority` 4, and every
  state is under 100 tokens. The loss is not about format or length.

## The flipped questions (Kev-27B right, a round-19 arm wrong)

There are 55 such `angry` questions across the three round-19 arms; 17 flip in all three. 53 are calm-text tickets with
gold "not angry": the gold label agrees with the text, and the arm is wrong. The other 2 are angry-text tickets (arm (b)
only). A sample of 20 with each side's p(angry) is in `drift.json` → `flips.sample`. Typical rows:

| id | ticket (subject / body) | Kev-27B | (a) | (b) | (c) |
|---|---|---|---|---|---|
| 0007 | 배송 완료라는데 못 받았습니다 / 송장은 배송 완료인데 집에 아무것도 없어요. Let me know what you need from me. | 0.40 | 0.69 | 0.64 | 0.78 |
| 0016 | 환불이 아직 안 들어왔습니다 / 17일 전에 반품했는데 환불이 안 됐어요. | 0.39 | 0.68 | 0.65 | 0.71 |
| 0080 | Package marked delivered but not here / Tracking for order #8990 says delivered 11 days ago. Nothing arrived. | 0.36 | 0.50 | 0.84 | 0.77 |
| 0094 | Damaged on arrival / The box for order #8223 was crushed and the item inside is broken. | 0.24 | 0.56 | 0.79 | 0.77 |
| 0082 | Where is my order? / Order #4589 was placed 5 days ago and tracking has not updated. | 0.21 | 0.38 | 0.82 | 0.60 |
| 0102 | Charged twice for order #7977 / My card shows two charges of $300.0 for one order. 확인 부탁드립니다. | 0.12 | 0.21 | 0.58 | 0.43 |

These are borderline for every model: Kev-27B puts them at 0.2-0.46, just under the 0.5 line, and the arms at 0.5-0.85.
The pattern is a boundary, not a taxonomy convention. A ticket that describes a real problem calmly gets read as anger.
Nothing in Kev's older data teaches this boundary: arm (c) was trained on Kev-27B's own data and is the worst (54 false
positives). LoRA seeds on that same data range from 9 to 59.

## Is the base closer to Kev-27B or to the arms?

No scienthoon read of the untrained `Qwen/Qwen3.8-27B` exists: the `runs/probes/qwen38-27b-*` probes cover semif-v1,
transfer-v4/v9 and wanli-v1. Round 20's interpolations are the only indirect evidence, and they disagree. Moving arm (a)
toward the base adds false positives (28 → 31 → 36 → 46 at α 1 → 0.5). Moving arm (b) toward the base removes them
(45 → 46 → 40 → 31). The interpolation keeps the SFT pointer head, so neither path ends at the base's own judgement.
Settling it would take one zero-shot base read of scienthoon: one H200, about $6, the A2 probe path.

## The pooled externals and short states

- **Pooled externals** (SemIf + scienthoon + WANLI-v2 + TypeSafe, 2,108 questions). The arms' losses are scienthoon
  losses, and those are `angry` losses. (a) is −1.2 [−2.4, 0.0]: −25 questions on scienthoon (−17 on `angry`), −1 on
  SemIf, 0 on WANLI-v2, +1 on TypeSafe. Without `angry` it is −0.4 [−1.8, +0.8]. For (b), −1.6 [−2.9, −0.3] becomes
  0.0 [−1.3, +1.3]. For (c), −1.2 becomes +0.7. For b-w50, −0.5 [−1.7, +0.7] becomes +0.5 [−0.8, +1.8]; b-w50 also
  loses 6 SemIf questions (candidate selection) and gains 11 on WANLI-v2.
- **Short states** (transfer-v4 dev + transfer-r3 test, 1,806 questions). This is a different, spread-out loss. (a) is
  −18 questions net (−1.0 [−2.2, +0.2]): qnli −6, composition_held_or_not −6 / 72, emotion −3, tweet_offensive −3. (b)
  and b-w50 are −2 and −3. (c) is −55, with paws −11, emotion −9 and composition_held_and_or −9. Anger and tone are not
  its driver.

## What Kev-27B's own family says about the guard

Six LoRA checkpoints were trained from the base on Kev's decision-v7-family data (B1 v2 s1/s2, B1 trials A/B, two-epoch
s1/s2). Their scienthoon mean is 0.765, sd 0.021, range 0.740-0.796. Kev-27B is the maximum. Its seed was chosen on
transfer-v4 among seeds that passed a pooled-externals floor against Kev-9B (`A:PLAN_27b.md` B1 v2 rule), which is a
mild selection on this suite. Against Kev-27B under round 19's guards (scienthoon lower bound ≥ −2 pp, pooled externals
lower bound ≥ −1.5 pp), **none of the five siblings passes either one**:
- B1 v2 s1: −2.7 [−4.2, −1.4] / −1.5 [−2.6, −0.5]
- B1 trial A: −2.3 [−3.9, −0.8] / −0.9 [−1.9, +0.1]
- 2 epochs s1: −2.4 [−4.0, −0.8] / −1.0 [−2.0, +0.1]
- B1 trial B and 2 epochs s2: −5.4 / −2.4 and −5.6 / −2.7

The round-19/20 arms score 0.757-0.778, at or above that family mean. Their angry false positives (28-54) are within the
family's range (9-59, mean 30). So the guard compares each arm with the luckiest draw of the recipe. Measured against the
recipe itself, full weights on `sft-v1` cost nothing on scienthoon.

The earlier 27B scienthoon failures had a different cause. Round 10's skills delta and round 17's arm (a) keep `angry`
(15 and 9 false positives). They lose `queue` and `priority` instead: −2.1 and −2.6 pp without `angry`.

## Recommendation for round 21

The options, with the evidence for and against each:

- **(d) Accept, and fix the guard: recommended.** The gold labels on the flipped questions are sound, so these are real
  errors. But they sit on one borderline boundary, and on it Kev-27B is a favourable draw of its own recipe (rank 1 of 6,
  +1.5 sd). The other 3 pp of scienthoon's spread is `priority`, which by construction is not in the text. A −2 pp paired
  bound against that single checkpoint fails every LoRA sibling, and would have failed Kev-27B's own second seed. Round 21
  should register one of these (Jared's call, before any read):
  1. scienthoon and the pooled externals gated against the LoRA family (for example, the arm's point estimate ≥ the
     family mean, and a lower bound ≥ −2 pp against the family's median checkpoint, B1 trial A); or
  2. scienthoon scored without `priority` (unknowable from the text), with `angry` false positives on calm text reported
     as the diagnostic.

  Kev-27B remains the parent for every other criterion. This is a change of rule, and it has to be registered as one, not
  applied to rounds 19-20 after the fact.
- **(a) Data: a cheap addition, not the fix.** AutoJev (full weights, other data) makes 0 false positives on calm text,
  so full-weight SFT can hold this boundary when the data covers it. A small open-weight-generated family of tone
  minimal pairs would target it directly: the same problem report written calm or angry, labelled by construction, in
  domains other than retail support tickets (banking, telecom, SaaS, travel), in English and Korean. It must not
  paraphrase scienthoon, and must be screened against scienthoon, breadth-v1 and transfer-r3's `emotion` /
  `tweet_offensive` holdouts. It belongs in the data extension already planned for round 21. It also tests whether the
  boundary moves at all.
- **(c) Lower lr / fewer steps: a free secondary.** At the final checkpoint, lr 2e-6 made fewer false positives than lr
  5e-6 (28 vs 45). But arm (a)'s interpolations toward the base got worse, so "less movement" does not help on its own.
  Snapshots now exist, and picking among them by a development criterion costs nothing. They should be read as a report,
  not selected on scienthoon.
- **(b) Distillation anchor toward Kev-27B: not recommended now.** It is mechanically close. `kev.train --anchor` takes
  any `{record_id: {qid: {option key: p}}}` file (`kev/anchors.py` writes the base's; `anchor_loss` only needs the same
  option keys). `batch_loss` applies it on the full-weight path, except with `--row_budget`. A converter from a
  `kev.benchmark` rows.json (`id`, `question`, `keys`, `p` at the served T) plus one Kev-27B read of the replay pool
  (about 1-2 H200 hours for about 20k records) would produce the targets. Questions that get a none-option inserted are
  skipped. The evidence argues against the payoff: the boundary Kev-27B gets right is not fixed by its training data
  (seed 1 on the same data makes 31 false positives), so distilling Kev-27B's answers on that data would not fix it
  either. Anchoring toward the base was also negative before (PLAN.md finding 13).
