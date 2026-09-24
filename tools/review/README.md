# Kev label review

A keyboard-first page for one person to accept, relabel or drop AI-proposed labels, one item at a time.
Nothing leaves the browser: decisions autosave to localStorage (keyed by a hash of the file), so a reload resumes where you were.

```sh
npm install && npm run dev
```

Open `http://localhost:5173/?file=sample.jsonl` (files in `public/`), or drop / open any `.jsonl` file. Each line is
`{id, document, source, question: {type, instructions, options}, proposed_label, label_origin, judges: [{model, label, rationale}], adjudication}`;
blank lines are skipped, malformed or duplicate-id lines are counted and skipped.

| Key | Action |
| --- | --- |
| `Enter` / `y` | accept the proposed label, next |
| `1`-`9`, `0`, then `a`-`z` | relabel to that option, next (letters `e j k n u x y` are commands, so they are skipped; options past the 29th are click-only) |
| click an option | relabel to it, next (choosing the proposed option counts as accept) |
| `x` | drop (ambiguous / unanswerable), next |
| `n` | edit the note (`Enter` saves, `Esc` cancels) |
| `←` / `k`, `→` / `j` | previous / next without deciding |
| `u` | undo the last decision |
| `e` | export `reviews.jsonl` |
| `?` | toggle shortcut help |

Filters: undecided (default), all, disagree (any judge differs from the proposed label), decided. Spot-check mode
shuffles with a fixed seed (42) and keeps the first N items (default 50), so the sample is the same on every machine.

Export writes one line per decided item, in file order:

```json
{"id": "...", "verdict": "accept|relabel|drop", "label": "final label or null for drop", "proposed_label": "...", "note": "", "reviewed_at": "ISO timestamp"}
```

The header shows accepted / relabelled / dropped and the agreement rate, accept / (accept + relabel).
