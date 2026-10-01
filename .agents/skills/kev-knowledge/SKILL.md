---
name: kev-knowledge
description: Search the Kev knowledge graph (~/dev/kev-knowledge, indexed by qmd) before re-deriving history. Use when asked what was tried, decided, measured or learned in earlier Kev sessions, why a rule or number exists, what a round or PR did, or when starting a research, serving, release or data task that earlier sessions may have covered. Also covers refreshing the graph after new sessions.
allowed-tools:
  - read
  - grep
  - glob
  - exec
permissions:
  allow:
    - Exec(qmd)
    - Read(~/dev/kev-knowledge/**)
---

# Kev knowledge graph (qmd)

Every Devin CLI session on this repo since 17 Sep 2026 is distilled into `~/dev/kev-knowledge`: hand-written topic
notes (what we know), one note per session (what happened), a day-by-day timeline, PR and issue indexes with
back-links, and the extracted transcripts. It is indexed by `qmd` as the collection `kev`. Use it instead of
re-deriving history from git or guessing why a rule exists. It lives outside the repo on purpose (transcripts are
noisy and may hold private names); never copy its `raw/` or `digest/` text into the repo.

## When to look

- "Did we try X?", "why is Y the rule?", "what did round N find?", "what did PR #N change and why?", "what did Jared
  decide about Z?" - search first, then cite the note.
- Before planning a round, a serving change, a data family, a release or a benchmark: read the matching topic note.
- When AGENTS.md or PLAN.md states a rule without the incident behind it, the topic notes have the incident.

## How to search

Two collections: `kev` (the distilled notes: index, timeline, topics, sessions, prs, issues; searched by default) and
`kev-transcripts` (`digest/` + `raw/`; excluded from default queries, name it with `-c`).

```sh
qmd search "<keywords>" -c kev -n 8            # BM25, instant; best for identifiers (PR numbers, flags, suite names)
qmd vsearch "<question>" -c kev -n 8           # semantic, ~3 s
qmd query "<question>" -c kev -n 8             # hybrid + LLM rerank, best quality; ~20 s on this Mac (CPU, no Metal)
qmd query "<question>" -c kev --files --min-score 0.3   # paths only
qmd get "kev/topics/<slug>.md"                 # read a note (line-numbered)
qmd multi-get "kev/sessions/*.md" -l 30        # skim the first 30 lines of every session note
qmd search "<exact phrase>" -c kev-transcripts -n 5     # quotes from the transcripts
```

Read order: `topics/` (distilled, trustworthy) -> `sessions/<id>.md` (what happened, with the exact asks and
outcomes) -> `digest/<id>.md` (prose transcript, for quotes) -> `raw/<id>.md` (tool calls, truncated outputs; last
resort). Transcript hits are long; read them with `qmd get "kev-transcripts/digest/<id>.md:<line>:<count>"` rather
than whole.

Without qmd: `rg -n "<term>" ~/dev/kev-knowledge/topics ~/dev/kev-knowledge/sessions`, then open
`~/dev/kev-knowledge/index.md`.

## Map

| file | use |
|---|---|
| `index.md` | hub: topics with session counts, sessions by date |
| `timeline.md` | the story 17 Sep - 1 Oct 2026, with open threads |
| `topics/architecture.md`, `training-recipe.md`, `full-weight-sft.md`, `calibration.md` | the model and how it is trained; what moved the needle |
| `topics/evaluation-suites.md`, `research-rounds.md`, `autoresearch-program.md` | suites, gates, the audit that removed scienthoon/wanli/typesafe; every round's verdict; the standing rules |
| `topics/serving-performance.md`, `deployment-paths.md`, `long-context.md`, `modal-infrastructure.md` | CUDA graphs, batching, fused kernels, MLX, vLLM decision, 64k states, GPUs and spend |
| `topics/releases.md`, `docs-and-writing.md`, `data-and-synthetic.md`, `competitors.md`, `base-models.md` | what shipped when; README/card rules; data policy and teachers; Jev/AutoJev/Clef; Qwen -> Gemma/Inkling |
| `topics/skills-and-workflow.md`, `code-quality.md` | worktrees, subagents, Devin CLI tips, the thermonuclear standard |
| `sessions/warm-lute.md` | the nine-day marathon that ran rounds 4-29 and every 27B release |
| `prs.md`, `issues.md` | every PR/issue -> sessions that discussed it (`prs.md#pr-<n>`) |

## Citing

Quote the note path (`~/dev/kev-knowledge/topics/calibration.md`) or the session id (`devin -r warm-lute` resumes it).
Numbers in the notes were written by hand from the transcripts; for a published number, prefer `docs/claims.json`,
PLAN.md or the model card, and say which you used.

## Refreshing after new sessions

```sh
cd ~/dev/kev-knowledge
python3 tools/extract_sessions.py        # read-only over ~/.local/share/devin/cli/sessions.db
python3 tools/build_graph.py             # exits non-zero naming any session that lacks a note
qmd update && qmd embed -c kev && qmd embed -c kev-transcripts   # the second is slow (tens of minutes); optional
```

If `qmd` is missing: `npm install -g --allow-scripts=node-llama-cpp @tobilu/qmd`, then
`qmd collection add ~/dev/kev-knowledge --name kev --mask "*.md,topics/**/*.md,sessions/**/*.md"` and
`qmd collection add ~/dev/kev-knowledge --name kev-transcripts --mask "digest/**/*.md,raw/**/*.md"` +
`qmd collection exclude kev-transcripts`; the contexts are listed in `~/dev/kev-knowledge/README.md`.

For each new session: read `digest/<id>.md`, add an entry to `tools/session_notes.py` (title, summary, asks,
outcomes, lessons, topics), extend the topic notes it taught something new, add the day to `timeline.md`, rebuild.
`tools/fetch_github.sh` refreshes `data/prs.tsv` / `data/issues.tsv` (needs `gh`). Do not edit `sessions/*.md`,
`index.md`, `prs.md` or `issues.md` by hand; they are regenerated. `README.md` there documents the layout.
