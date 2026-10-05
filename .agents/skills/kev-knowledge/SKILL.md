---
name: kev-knowledge
description: Search the Kev knowledge graph (private repo jaredpalmer/kev-knowledge) before re-deriving history. Use when asked what was tried, decided, measured or learned in earlier Kev sessions, why a rule or number exists, what a round or PR did, or when starting a research, serving, release or data task that earlier sessions may have covered. Also covers how the graph is refreshed (nightly Devin Cloud automation) and how to add to it.
allowed-tools:
  - read
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(gh repo clone jaredpalmer/kev-knowledge)
    - Exec(rg)
    - Exec(qmd)
    - Read(/tmp/kev-knowledge/**)
    - Read(~/dev/kev-knowledge/**)
---

# Kev knowledge graph

Every Devin session on this repo since 17 Sep 2026 is distilled into the private repo
`jaredpalmer/kev-knowledge`: hand-written topic notes (what we know), one note per session (what happened), a
day-by-day timeline, PR and issue indexes with back-links, and the extracted transcripts (`digest/`). A nightly Devin
Cloud automation reads new sessions through the Devin API and writes the notes (`automation/nightly.md` there). Use it
instead of re-deriving history from git or guessing why a rule exists. It is a separate private repo on purpose:
transcripts hold private names and verbatim conversation; never copy `digest/` text into this repo or any public
surface.

## When to look

- "Did we try X?", "why is Y the rule?", "what did round N find?", "what did PR #N change and why?", "what did Jared
  decide about Z?": search first, then cite the note.
- Before planning a round, a serving change, a data family, a release or a benchmark: read the matching topic note.
- When AGENTS.md or PLAN.md states a rule without the incident behind it, the topic notes have the incident.

## Get it

```sh
[ -d /tmp/kev-knowledge ] || gh repo clone jaredpalmer/kev-knowledge /tmp/kev-knowledge   # or ~/dev/kev-knowledge on Jared's Mac
git -C /tmp/kev-knowledge pull --ff-only -q
```

If the clone is refused, the session has no access to the private repo: say so and fall back to PLAN.md, the git log
and `research-archive-2026-09-24`; do not guess at what the graph would have said.

## Search it

`rg` always works and is enough for most questions; qmd adds semantic search if it is installed.

```sh
K=/tmp/kev-knowledge
cat $K/index.md                                   # the map: topics with session counts, sessions by date
rg -n -i "<term>" $K/topics $K/sessions           # identifiers, PR numbers, flags, suite names
rg -n -i "<term>" $K/timeline.md $K/prs.md        # when, and which sessions touched a PR
rg -n -i "<exact phrase>" $K/digest               # quotes from transcripts (long files; read with sed -n 'a,bp')
```

```sh
# optional: npm install -g --allow-scripts=node-llama-cpp @tobilu/qmd  (models download on first use)
cd $K && qmd update && qmd embed -c kev           # the checked-in .qmd/index.yml builds a project-local index; no trust prompt
qmd search "<keywords>" -c kev -n 8               # BM25, instant
qmd vsearch "<question>" -c kev -n 8              # semantic, seconds
qmd query "<question>" -c kev -n 8                # hybrid + LLM rerank, best quality, slow on CPU
qmd get "kev/topics/<slug>.md"
qmd search "<phrase>" -c kev-transcripts -n 5     # transcripts are excluded from default search
```

Read order: `topics/` (distilled, trustworthy) -> `sessions/<id>.md` (what happened, the exact asks and outcomes)
-> `digest/<id>.md` (transcript, for quotes).

## Map

| file | use |
|---|---|
| `index.md`, `timeline.md` | hub; the story since 17 Sep 2026 with open threads |
| `topics/architecture.md`, `training-recipe.md`, `full-weight-sft.md`, `calibration.md` | the model and how it is trained; what moved the needle |
| `topics/evaluation-suites.md`, `research-rounds.md`, `autoresearch-program.md` | suites, gates, the audit that removed scienthoon/wanli/typesafe; every round's verdict; the standing rules |
| `topics/serving-performance.md`, `deployment-paths.md`, `long-context.md`, `modal-infrastructure.md` | CUDA graphs, batching, fused kernels, MLX, the vLLM decision, 64k states, GPUs and spend |
| `topics/releases.md`, `docs-and-writing.md`, `data-and-synthetic.md`, `competitors.md`, `base-models.md` | what shipped when; README/card rules; data policy and teachers; Jev/AutoJev/Clef; Qwen -> Gemma/Inkling |
| `topics/skills-and-workflow.md`, `code-quality.md` | worktrees, subagents, the thermonuclear standard |
| `sessions/warm-lute.md` | the nine-day CLI session that ran rounds 4-29 and every 27B release |
| `sessions/cloud-<12 hex>.md` | Devin Cloud sessions; each links its app.devin.ai URL |
| `prs.md`, `issues.md` | every PR/issue -> the sessions that discussed it (`prs.md#pr-<n>`) |

## Citing

Name the note path (`kev-knowledge/topics/calibration.md`) or the session id. Numbers in the notes were written by
hand from transcripts; for a published number prefer `docs/claims.json`, PLAN.md or the model card here, and say
which you used.

## Adding to it

The nightly automation picks up every Devin Cloud session on its own; nothing to do for normal work. To record
something immediately, or to add a lesson the automation would not infer, open a PR on `jaredpalmer/kev-knowledge`
editing `tools/session_notes.py` or `topics/*.md`, run `python3 tools/build_graph.py` there, and never edit the
generated files (`sessions/*.md`, `index.md`, `prs.md`, `issues.md`) by hand. The procedure the automation follows,
and the by-hand refresh (`tools/extract_cloud_sessions.py`, needs `DEVIN_API_KEY`), are in that repo's README and
`automation/nightly.md`.
