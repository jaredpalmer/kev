# Overnight night 3 (2026-09-23 evening → 2026-09-24): instructions as given

Unlike round 6 (`overnight-round6.md`, a prepared prompt for a fresh session), night 3 continued an interactive session: the
work plan was built up in conversation during the evening of 2026-09-23 and then handed over for the night with the
messages below. This file records them verbatim, with the standing constraints that applied, so the night's record in
`PLAN.md` ("Night 3") can be read against what was asked.

## The hand-over (Jared, 2026-09-23, ~23:45 local)

After the Kev-4B round-8 release had been published and the Kev-27B serving checks proposed:

> Go for it. Then get to work tonight. You may need do even more thing outside of the plan. You have all night. YOu can burn $1000 in modal credits if you need to. $100 on AI Gateway. After you've gotten to good state continue with autoresearch. Believe in your self. Hill climb. Use worktrees. Nurse/babysit PRs etc. Do thermonuclear-code-reviews as you go. Use subagents to parallelize work and work faster. Keep a scratchpad in ./tmp/scratchpad.txt. Keep plans updated in PLan.md and PLan27.md where relevant. VErify and work methodically like an ai researcher. Go for it. You got this.

Earlier the same evening, the approvals that set the night's starting point: "okay go ahead with round 8"; "add hard target calibration as target. also should we generate similar syntethic data to where we we failed on this bench?"; "go for it and publish" (the Kev-4B round-8 release only).

## Standing constraints (from the session and the repo, not restated in the hand-over)

- **Budgets:** Modal $1,000 on top of the metered reading at the start ($1,085.32); AI Gateway $100 (Jev reference reads and nothing else; no Jev output in any training file).
- **Registration:** every round's rule is written into `PLAN.md` and committed before its training or reads; selection on development partitions only; each test / locked partition read once per candidate, after the rule that decides.
- **Publishing:** public Hub changes need Jared's explicit OK. The night therefore published Kev-27B privately, kept the new candidates in private Hub repos, and left draft PR #106 unmerged.
- **Code to main:** through reviewed PRs (strict review, CI green, squash merge); research work stays on `research/overnight-r6`, pushed after every commit.
- **Frozen files stay frozen;** new data is a new directory with a manifest.

## Where the record is

- `PLAN.md`, "Night 3": outcome, spend, incidents, pending jobs, and every round's registration and result in order.
- `PLAN_27b.md`: B1 v2 result, bf16 serving check, JevBench read of Kev-27B.
- `runs/night3-state.json`: spend readings, studies with spawn ids and bounds, candidates, PRs.
- The working scratchpad (`tmp/scratchpad.txt`) is not committed: several of its clock stamps were written ahead of time; `PLAN.md` uses commit times.
