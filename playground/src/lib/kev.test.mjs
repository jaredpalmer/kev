import assert from "node:assert/strict";
import test from "node:test";
import { parseRequestDraft, PRESETS } from "./kev.ts";

test("incomplete question drafts stay recoverable until the choice is complete", () => {
  const preset = PRESETS.find((p) => p.name === "News article");
  const stateText = JSON.stringify(preset.state);
  const drafts = [
    '{"topic":',
    '{"topic":{"type":"choice","instructions":"Topic?"}}',
    '{"topic":{"type":"choice","criteria":null}}',
    '{"topic":null}',
    'null',
  ];

  for (const draft of drafts) {
    const parsed = parseRequestDraft(stateText, draft);
    assert.equal(parsed.req, undefined, draft);
    assert.ok(parsed.err, draft);
    assert.deepEqual(parsed.choiceIds, [], draft);
  }

  const recovered = parseRequestDraft(stateText, JSON.stringify(preset.questions));
  assert.equal(recovered.err, undefined);
  assert.deepEqual(recovered.req, { state: preset.state, model: "kev-latest", questions: preset.questions });
  assert.deepEqual(recovered.choiceIds, ["topic"]);
});

test("presets keep their requests and only choices with multiple options can permute", () => {
  const expectedChoices = {
    "Support triage": ["department", "return_reason", "requested_resolution", "tone"],
    "News article": ["topic"],
    "Review rating": [],
    "Isolation probe": ["which_code"],
    "Boundary forgery": ["topic"],
  };
  for (const preset of PRESETS) {
    const stateText = typeof preset.state === "string" ? preset.state : JSON.stringify(preset.state);
    const parsed = parseRequestDraft(stateText, JSON.stringify(preset.questions));
    assert.equal(parsed.err, undefined, preset.name);
    assert.deepEqual(parsed.req, { state: preset.state, model: "kev-latest", questions: preset.questions });
    assert.deepEqual(parsed.choiceIds, expectedChoices[preset.name], preset.name);
  }

  const questions = {
    empty: { type: "choice", criteria: {} },
    single: { type: "choice", criteria: { only: null } },
    multiple: { type: "choice", criteria: { first: null, second: null } },
  };
  const parsed = parseRequestDraft("{unfinished state", JSON.stringify(questions));
  assert.equal(parsed.req.state, "{unfinished state");
  assert.deepEqual(parsed.choiceIds, ["multiple"]);
});
