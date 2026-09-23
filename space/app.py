"""Kev on Hugging Face Spaces (Gradio, ZeroGPU).

One document (the state) and a set of typed questions in, a probability per option out, in one forward pass. No text is
generated. Loads Kev-4B and Kev-0.8B (jaredpalmer/kev-4b, jaredpalmer/kev-0.8b) through kev.checkpoint, exactly as
kev.serve does: LoRA merged into the base in fp32, pointer head on top. kev/model.py, kev/api.py and kev/checkpoint.py
are copied from the repo at publish time (scripts/publish_space.sh), so the Space runs the same encoder, mask, loader and
API code as kev.serve.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # noqa: E402  (must come before torch)

import html, json, random, time  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from kev.api import SystemOneRequest, output_tokens, render, to_answers, to_record, with_date_facts  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE  # noqa: E402
from presets import PRESETS  # noqa: E402

MODELS = {"Kev-4B": "jaredpalmer/kev-4b", "Kev-0.8B": "jaredpalmer/kev-0.8b"}
DEFAULT_MODEL = "Kev-4B"


# ------------------------------------------------------------------ load
# The shared loader runs on CPU in fp32 (the exact path every reported number uses: LoRA merged in fp32, pointer head on
# top), then the finished module moves to "cuda" once at module scope, which is what ZeroGPU expects. The head is loaded
# raw (T=1); the Calibrated toggle sets the checkpoint's fitted temperature per request.

def load(repo):
    ck = Checkpoint(repo)
    tok, m = ck.load("cpu", LoadOptions(attn="sdpa", temperature=1.0))
    m.device = "cuda"; m.to("cuda")
    print(f"[kev] {repo}: base={ck.meta.base}@{ck.meta.base_revision} hybrid={m.hybrid} temperature={ck.meta.temperature:.2f}", flush=True)
    return tok, m, ck.meta.temperature


LOADED = {name: load(repo) for name, repo in MODELS.items()}   # name -> (tokenizer, model, fitted temperature)


# ------------------------------------------------------------------ inference

def parse_state(text):
    """Plain text, or a JSON object / array (same rule as the playground)."""
    t = (text or "").strip()
    if t[:1] in "{[":
        try: return json.loads(t)
        except json.JSONDecodeError: pass
    return text or ""


def build_request(state_text, questions_json):
    try: questions = json.loads(questions_json or "")
    except json.JSONDecodeError as e: raise gr.Error(f"Questions must be valid JSON: {e}") from None
    if not isinstance(questions, dict) or not questions:
        raise gr.Error('Questions must be a non-empty JSON object, e.g. {"topic": {"type": "choice", ...}}')
    try: return SystemOneRequest(state=parse_state(state_text), model="kev-latest", questions=questions)
    except ValidationError as e:
        first = e.errors()[0]
        raise gr.Error(f"Invalid question at `{'.'.join(str(x) for x in first.get('loc', ()))}`: {first.get('msg')}") from None


def probs(name, req, calibrated):
    """One request through one model. Returns (probabilities per question, per-question metadata, token count, ms).
    calibrated: apply the temperature the checkpoint carries (the pointer head divides its logits by it); else raw."""
    tok, model, fitted = LOADED[name]
    model.head.temperature = fitted if calibrated else 1.0   # argmax unchanged either way; requests run one at a time (Gradio queue)
    rec, meta = to_record(req)
    try: enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    except ValueError as e: raise gr.Error(str(e)) from None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ps = model.probs(enc)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
    return [p.tolist() for p in ps], meta, len(enc["ids"]), ms


def systemone(name, req, calibrated):
    """The /v1/systemone response kev.serve would return for this request."""
    tok = LOADED[name][0]
    ps, meta, n_tokens, ms = probs(name, req, calibrated)
    answers = to_answers(ps, meta)
    return {"model": MODELS[name], "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": output_tokens(tok, answers)}, "latency_ms": round(ms, 1)}


def stability(name, req, calibrated, n_perm):
    """Re-run the first Choice question with 2+ options under shuffled option orders (kev.serve /v1/systemone/permute)."""
    target = next((qid for qid, q in req.questions.items() if q.type == "choice" and len(q.criteria) >= 2), None)
    if target is None: return "_No Choice question with two or more options to permute._"
    q = req.questions[target]; keys = list(q.criteria); rng = random.Random(0); runs = []
    for i in range(max(2, n_perm)):
        order = list(keys)
        if i > 0: rng.shuffle(order)
        one = SystemOneRequest(state=req.state, model=req.model, questions={target: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})})
        ps, meta, _, _ = probs(name, one, calibrated)
        a = to_answers(ps, meta)[target]
        runs.append((order, a["choice"], a["probabilities"]))
    spread = {k: max(r[2][k] for r in runs) - min(r[2][k] for r in runs) for k in keys}
    stable = len({r[1] for r in runs}) == 1
    lines = [f"**{name}, `{target}` under {len(runs)} option orders:** argmax {'stable' if stable else 'flips'}, "
             f"largest probability spread {max(spread.values()):.2f}", "",
             "| order | picked | " + " | ".join(f"`{k}`" for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    lines += [f"| {' → '.join(order)} | `{choice}` | " + " | ".join(f"{p[k]:.2f}" for k in keys) + " |" for order, choice, p in runs]
    return "\n".join(lines)


# ------------------------------------------------------------------ rendering

CARD = "border:1px solid var(--border-color-primary);border-radius:10px;padding:12px 14px;margin-bottom:10px;background:var(--background-fill-secondary);"
TRACK = "position:relative;display:block;height:6px;border-radius:999px;background:var(--border-color-primary);overflow:hidden;"


def bar(label, p, top):
    fill = "var(--body-text-color)" if top else "var(--body-text-color-subdued)"; w = "600" if top else "400"
    return ('<div style="display:grid;grid-template-columns:minmax(0,11rem) minmax(0,1fr) 3rem;align-items:center;gap:0 12px;font-size:12px;line-height:22px">'
            f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:{w}" title="{html.escape(label)}">{html.escape(label)}</span>'
            f'<span style="{TRACK}"><span style="position:absolute;top:0;bottom:0;left:0;border-radius:999px;background:{fill};width:{max(0.0, min(1.0, p)) * 100:.1f}%"></span></span>'
            f'<span style="text-align:right;font-variant-numeric:tabular-nums;font-weight:{w}">{p:.2f}</span></div>')


def render_answers(name, req, resp):
    answers, usage = resp["answers"], resp["usage"]
    parts = ['<div style="font-size:12px;color:var(--body-text-color-subdued);margin:6px 0 10px">'
             f'<b>{name}</b> · {len(answers)} question(s) · {usage["input_tokens"]} input tokens · {resp["latency_ms"]:.0f} ms on GPU · no text generated</div>']
    for qid, a in answers.items():
        q = req.questions.get(qid); instr = render(q.instructions) if q is not None else ""
        if a["type"] == "noul":
            p = a["noul"]; headline, detail = ("yes" if p >= 0.5 else "no"), f"p(yes) {p:.2f}"
            rows = bar("yes", p, p >= 0.5) + bar("no", 1 - p, p < 0.5)
        elif a["type"] == "choice":
            headline, detail = a["choice"], f"confidence {a['confidence']:.2f}"
            rows = "".join(bar(k, v, k == a["choice"]) for k, v in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]))
        else:
            legend, dist = a["legend"], a["probabilities"]; top = max(dist, key=dist.get)
            headline, detail = f"{a['score']:.2f} of {len(legend) - 1}", f"confidence {a['confidence']:.2f}"
            rows = "".join(bar(f"{i}. {legend[str(i)]}", dist[str(i)], str(i) == top) for i in range(len(legend)))
        parts.append(f'<div style="{CARD}"><div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap">'
                     f'<span style="font-family:var(--font-mono);font-size:13px;font-weight:600">{html.escape(qid)}</span>'
                     f'<span style="font-size:11px;color:var(--body-text-color-subdued)">· {a["type"]}</span></div>'
                     f'<div style="font-size:13px;margin:2px 0 6px">{html.escape(instr)}</div>'
                     '<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:8px">'
                     f'<span style="font-size:17px;font-weight:600">{html.escape(str(headline))}</span>'
                     f'<span style="font-size:12px;color:var(--body-text-color-subdued)">{detail}</span></div>{rows}</div>')
    return "".join(parts)


# ------------------------------------------------------------------ handler

def duration(state_text, questions_json, model_choice=DEFAULT_MODEL, calibrated=False, date_facts=False, check_stability=False, n_perm=4):
    n_models = len(MODELS) if model_choice == "Both" else 1
    return 15 * n_models + (6 * int(n_perm or 4) * n_models if check_stability else 0)


@spaces.GPU(duration=duration)
def decide(state_text, questions_json, model_choice=DEFAULT_MODEL, calibrated=False, date_facts=False, check_stability=False, n_perm=4):
    """Answer typed questions about one document with a Kev decision model, in a single forward pass.

    Args:
        state_text: the document the model reads (the "state"): plain text, or a JSON object/array.
        questions_json: JSON object mapping a question id to a typed question. Each question is one of
            {"type": "choice", "instructions": str, "criteria": {name: description}},
            {"type": "noul", "instructions": str, "criteria": {"true": str, "false": str}} (criteria optional),
            {"type": "score", "instructions": str, "criteria": [level0, level1, ...]}.
        model_choice: "Kev-4B" (recommended), "Kev-0.8B", or "Both" to compare them on the same request.
        calibrated: apply the temperature fitted for the chosen checkpoint (stored in its head.pt; about 2.1–2.4). Argmax is unchanged.
        date_facts: append the day count between every pair of absolute dates in the state before the model reads it.
        check_stability: also re-run the first Choice question under shuffled option orders.
        n_perm: how many option orders to try when check_stability is on.

    Returns:
        Rendered answers, the raw /v1/systemone response(s), and the option-order report.
    """
    req = build_request(state_text, questions_json)
    if date_facts: req = req.model_copy(update={"state": with_date_facts(req.state)})
    calibrated = bool(calibrated)
    names = list(MODELS) if model_choice == "Both" else [model_choice]
    responses = {name: systemone(name, req, calibrated) for name in names}
    rendered = "".join(render_answers(name, req, r) for name, r in responses.items())
    report = "\n\n".join(stability(name, req, calibrated, int(n_perm or 4)) for name in names) if check_stability else ""
    raw = responses[names[0]] if len(names) == 1 else responses
    return rendered, raw, report


# ------------------------------------------------------------------ UI

def as_text(s): return s if isinstance(s, str) else json.dumps(s, indent=2)


EXAMPLES = [[as_text(p["state"]), json.dumps(p["questions"], indent=2)] for p in PRESETS]

INTRO = """# Kev

Small decision models: one document (the **state**) and a set of typed questions in, a probability for every option
out, in one forward pass. Nothing is generated. Pick an example or paste your own, then press **Decide**.
"""

ABOUT = """Kev is a LoRA adapter plus a pointer head on a Qwen3.5 base model. The pointer head scores each question's
`<decide>` token against its option spans; a block-causal layout means questions share the state but cannot read each
other. Three question types: `choice` (named options), `noul` (yes/no), `score` (ordered levels). Requests and
responses follow TypeSafe's `/v1/systemone` contract.

This Space runs [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) and
[`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b), loaded exactly as `kev.serve` does (LoRA merged
in fp32). Code, training recipe and frozen evaluation suites: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev).
"""

PLACEHOLDER = ('<div style="border:1px dashed var(--border-color-primary);border-radius:10px;padding:28px 16px;text-align:center;'
               'color:var(--body-text-color-subdued);font-size:13px">Answers appear here. One card per question, with a bar per option.</div>')

SCHEMA = """```jsonc
{
  "department": {                       // choice: named options, probabilities by name
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": { "returns": "Refunds and wrong items", "billing": "Charges and invoices" }
  },
  "escalate": {                         // noul: yes/no, answer is p(true)
    "type": "noul",
    "instructions": "Does this need urgent human attention?",
    "criteria": { "true": "optional description", "false": "optional description" }
  },
  "frustration": {                      // score: ordered levels, answer is the expected level
    "type": "score",
    "instructions": "How frustrated is the customer?",
    "criteria": ["Calm", "Frustrated", "Very angry"]
  }
}
```
Descriptions may be `null`, a string, or a nested JSON object/array. The state box takes plain text or a JSON object/array.
"""

FOOTER = ("Raw probabilities are not perfectly calibrated out of domain (Kev-4B: ECE 0.12 raw, 0.05 with the T=2.0 option). "
          "Measure on your own inputs before relying on the numbers. Example inputs are the repo's playground presets. "
          "Apache-2.0.")

OPTIONS_HELP = """- **Calibrated**: one temperature per checkpoint, fitted on its in-distribution development set and stored in `head.pt`
  (Kev-4B 2.14, Kev-0.8B 2.41). This is what `kev.serve` does by default; `KEV_TEMPERATURE=1.0` gives the raw logits. The argmax is
  unchanged; out-of-domain ECE on Kev-4B goes from 0.12 to 0.04.
- **date_facts**: Kev cannot subtract dates by itself. This appends the day count between every pair of absolute dates in the
  state before the model reads it (`KEV_DATE_FACTS=1`). The *Return window* example is wrong without it and right with it.
- **Option-order check**: re-run the first Choice question under *n* shuffled option orders and report whether the argmax flips
  and how far each probability moves.
- **Both**: run Kev-4B and Kev-0.8B on the same request and show the answers one above the other.
"""

CSS = """
.gradio-container { overflow: visible !important; }   /* the default overflow:hidden disables position:sticky */
#col-container { max-width: 1180px; margin: 0 auto; }
/* sticky control bar: one bordered card; Gradio's auto-inserted .form wrappers and per-block boxes are flattened */
#controls { position: sticky; top: 0; z-index: 50; background: var(--body-background-fill); padding: 10px 0 12px;
            border-bottom: 1px solid var(--border-color-primary); gap: 6px; }
#controls .form, #controls .block { border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; }
#controls .form { gap: 0; }
#controls-top { align-items: end; gap: 16px; }
#controls-options { align-items: center; gap: 8px 28px; flex-wrap: wrap; }
#controls-options > .block { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; margin: 0 !important; }
#controls-options > .block:last-child { width: 140px !important; margin-left: auto !important; }   /* order count sits under the button */
#controls-options label { margin: 0; }
#decide-btn { min-height: 44px; }
#examples .gallery { gap: 8px; }
#questions .cm-editor { max-height: 420px; }   /* gr.Code ignores max_lines; keep the editor scrollable instead of page-long */
"""

with gr.Blocks(title="Kev decision models") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(INTRO)

        # inputs are created here so the examples and the control bar can reference them; they are rendered further down
        state_box = gr.Textbox(label="State (the document the model reads)", lines=6, max_lines=14, value=EXAMPLES[0][0],
                               placeholder="Paste a support message, a review, an article, or a JSON object.", render=False)
        questions_box = gr.Code(label="Questions (JSON)", language="json", lines=14, value=EXAMPLES[0][1], render=False, elem_id="questions")

        with gr.Column(elem_id="controls"):
            with gr.Row(elem_id="controls-top", equal_height=True):
                model_radio = gr.Radio(list(MODELS) + ["Both"], value=DEFAULT_MODEL, label="Model", scale=4)
                run_btn = gr.Button("Decide", variant="primary", size="lg", scale=1, min_width=160, elem_id="decide-btn")
            with gr.Row(elem_id="controls-options"):
                calibrated_cb = gr.Checkbox(label="Calibrated (T = 2)", value=False, container=False)
                dates_cb = gr.Checkbox(label="date_facts", value=False, container=False)
                stability_cb = gr.Checkbox(label="Option-order check", value=False, container=False)
                n_perm_sl = gr.Dropdown([(f"{n} orders", n) for n in (2, 3, 4, 6, 8)], value=4, label="orders", container=False, filterable=False)

        gr.Examples(examples=EXAMPLES, example_labels=[p["name"] for p in PRESETS], inputs=[state_box, questions_box], label="Examples", elem_id="examples")

        with gr.Row():
            with gr.Column(scale=1):
                state_box.render()
                questions_box.render()
            with gr.Column(scale=1):
                answers_html = gr.HTML(value=PLACEHOLDER)
                stability_md = gr.Markdown()
                with gr.Accordion("Raw /v1/systemone response", open=False): raw_json = gr.JSON()

        with gr.Accordion("About Kev", open=False): gr.Markdown(ABOUT)
        with gr.Accordion("What the examples show", open=False):
            gr.Markdown("\n".join(f"- **{p['name']}**: {p['blurb']}" for p in PRESETS))
        with gr.Accordion("What the options do", open=False): gr.Markdown(OPTIONS_HELP)
        with gr.Accordion("Question schema", open=False): gr.Markdown(SCHEMA)
        gr.Markdown(FOOTER)

    run_btn.click(fn=decide, inputs=[state_box, questions_box, model_radio, calibrated_cb, dates_cb, stability_cb, n_perm_sl],
                  outputs=[answers_html, raw_json, stability_md], api_name="decide")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome(), css=CSS, mcp_server=True)
