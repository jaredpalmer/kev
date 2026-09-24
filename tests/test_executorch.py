"""ExecuTorch backend (kev.executorch_model): the program contract, readout, chunking, limits and temperature against a fake
program (no executorch, no weights; part of the CI unit job), and weight-backed parity of a real exported program against
the fp32 torch path on jaredpalmer/kev-0.8b (runs only when a program is given; ExecuTorch needs its own environment,
AGENTS.md):

    uv run python -m pytest tests/test_executorch.py -q                                   # fake program only
    KEV_BACKEND=executorch KEV_PROGRAM=kev-cpu/model.pte python -m pytest tests/test_executorch.py -q
"""
from types import SimpleNamespace

import pytest
import torch

from kev.checkpoint import Checkpoint, LoadOptions, Meta
from kev.executorch_model import ExecuTorchDecisionModel
from kev.model import SCORING_INTERFACE, SPECIAL, ContextOverflow, rows_of

RUN = "jaredpalmer/kev-0.8b"
SPECIAL_IDS = [900 + i for i in range(len(SPECIAL))]
PAD = 7


class Tokenizer:
    """One id per character (ord), the delimiters at 900.., a pad id."""
    pad_token_id = PAD

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def convert_tokens_to_ids(self, token):
        return SPECIAL_IDS[SPECIAL.index(token)]


class Method:
    def __init__(self, fn):
        self.fn = fn

    def execute(self, inputs):
        return self.fn(*inputs)


class Program:
    """The examples/kev contract with a toy model: the state is its KV (token ids as floats), and question q's logit for
    option k is 0.1 * (the option's last text token) + 0.01 * (the sum of the state and of the row's real tokens) /
    temperature. Every `score` call is recorded so the tests can see the chunks."""

    def __init__(self, temperature=2.0, max_questions=8, max_prefix=64, max_context=128, **constants):
        self.temperature, self.calls = temperature, []
        self.constants = {"get_kev_version": 1, "get_max_prefix": max_prefix, "get_max_context": max_context,
                          "get_max_questions": max_questions, "get_max_options": 255, "get_pad_id": PAD,
                          **{f"get_special_{i}": t for i, t in enumerate(SPECIAL_IDS)}, **constants}

    @property
    def method_names(self):
        return {*self.constants, "prefill", "score"}

    def load_method(self, name):
        if name == "prefill": return Method(self.prefill)
        if name == "score": return Method(self.score)
        return Method(lambda: [self.constants[name]])

    def prefill(self, tokens):
        S = tokens.shape[1]
        return [torch.zeros(3, 1, 4, 4), torch.zeros(3, 1, 2, 2, 2), tokens.float().reshape(1, 1, 1, 1, S, 1).expand(1, 2, 1, 1, S, 1).clone()]

    def score(self, tokens, decide, options, conv, recurrent, kv):
        assert tokens.shape[0] <= self.constants["get_max_questions"] and kv.shape[4] <= self.constants["get_max_prefix"]
        self.calls.append(tuple(tokens.shape))
        real = torch.stack([tokens[i, : int(decide[i]) + 1].float().sum() for i in range(len(tokens))])
        z = 0.1 * tokens.gather(1, (options - 1).clamp(min=0)).float() + 0.01 * (kv[0, 0].sum() + real)[:, None]
        return [z / self.temperature]


def record(n_questions=3, state="a short state"):
    options = [["x", "yy"], ["a", "b", "c"], ["p"]]
    return {"state": state, "questions": [{"instr": f"question {i}?", "options": options[i % 3], "label": 0} for i in range(n_questions)]}


def expected_logits(enc, temperature):
    """What the toy program computes, from Kev's own encoding: the reference for readout and chunking."""
    S, _, rows = rows_of(enc)
    return [(0.1 * torch.tensor([float(r["ids"][o - 1]) for o in r["opts"]]) + 0.01 * (sum(S) + sum(r["ids"]))) / temperature for r in rows]


def model(program=None, temperature=2.0):
    return ExecuTorchDecisionModel(program or Program(temperature=temperature), Tokenizer(), temperature=temperature, checkpoint_id="sha256:ours")


def test_scoring_interface_is_shared():
    m = model()
    assert not [name for name in SCORING_INTERFACE if not hasattr(m, name)]
    assert m.backend == "executorch" and m.dtype == "float32" and m.head.temperature == 2.0 and m.eval() is m


@pytest.mark.parametrize("constants,match", [({"get_kev_version": 2}, "version-1"), ({"get_special_3": 1}, "tokenizer"),
                                             ({"get_pad_id": 0}, "tokenizer"), ({"get_temperature": 1.5}, "another checkpoint"),
                                             ({"get_checkpoint_id": "sha256:theirs"}, "exported from sha256:theirs")])
def test_program_contract_is_checked(constants, match):
    with pytest.raises(ValueError, match=match):
        model(Program(**constants))
    assert model(Program(get_temperature=2.0, get_checkpoint_id="sha256:ours")).head.temperature == 2.0


def test_readout_matches_the_encoding_with_ragged_options():
    """One `score` call, questions with 2, 3 and 1 options: each question's logits are read at its own option boundaries
    and cut to its own option count (padded slots never leak into the softmax)."""
    m, tok = model(), Tokenizer()
    enc = m.encode(tok, record())
    got = m.forward(enc)
    assert [len(z) for z in got] == [2, 3, 1]
    for z, want in zip(got, expected_logits(enc, 2.0)):
        assert torch.allclose(z, want, atol=1e-5)
    assert all(abs(float(p.sum()) - 1) < 1e-6 for p in m.probs(enc))


def test_rows_are_chunked_by_the_exported_batch_limit():
    program = Program(max_questions=2)
    m, tok = model(program), Tokenizer()
    enc = m.encode(tok, record(5))
    chunked = m.forward(enc)
    assert [shape[0] for shape in program.calls] == [2, 2, 1]
    for z, want in zip(chunked, model().forward(enc)):
        assert torch.equal(z, want)


def test_prefix_reuse():
    m, tok = model(), Tokenizer()
    enc = m.encode(tok, record())
    first, prefix = m.probs_and_prefix(enc)
    again = m.probs_with_prefix(enc, prefix)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    with pytest.raises(ValueError, match="prefix"):
        m.probs_with_prefix(m.encode(tok, record(state="another, longer state")), prefix)


def test_exported_limits_are_context_overflows():
    """A program has fixed maximum shapes; exceeding one is the same error as any other context overflow (a 422 when
    serving, where the serving context is larger), never a truncated state."""
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    m, tok = model(Program(max_prefix=16, max_context=48, get_max_options=2)), Tokenizer()
    serving = {"max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH}
    assert m.encode(tok, record(1, state="x" * 15), **serving)["seg"].count(0) == 16
    for strict in (True, False):   # kev.predictors passes strict=True; asking for truncation still refuses
        with pytest.raises(ContextOverflow, match="state exceeds 16 .* exported for states of at most 16"):
            m.encode(tok, record(1, state="x" * 16), strict=strict, **serving)
    with pytest.raises(ContextOverflow, match="branch too long"):
        m.encode(tok, {"state": "s", "questions": [{"instr": "q" * 50, "options": ["a", "b"], "label": 0}]}, **serving)
    with pytest.raises(ContextOverflow, match="more than 2 options"):
        m.encode(tok, record(2))


def test_temperature_override_rescales_the_exported_logits():
    m, tok = model(), Tokenizer()
    enc = m.encode(tok, record())
    m.head.temperature = 1.0
    for z, want in zip(m.forward(enc), expected_logits(enc, 1.0)):
        assert torch.allclose(z, want, atol=1e-5)


def test_loader_refuses_what_a_program_cannot_do():
    ck = Checkpoint.__new__(Checkpoint); ck.meta = Meta(base="Qwen/Qwen3.5-0.8B-Base")
    assert ck.backend("mps", LoadOptions(backend="executorch")) == "executorch"
    with pytest.raises(ValueError, match="KEV_PROGRAM"):
        ck._load_executorch(Tokenizer(), LoadOptions(backend="executorch"))
    for opts in (LoadOptions(backend="executorch", program="x.pte", merge=False), LoadOptions(backend="executorch", program="x.pte", lora_scale=0.5)):
        with pytest.raises(ValueError, match="merged adapter"):
            ck._load_executorch(Tokenizer(), opts)
    opts = LoadOptions.from_env({"KEV_BACKEND": "executorch", "KEV_PROGRAM": "kev-cpu/model.pte"})
    assert (opts.backend, opts.program) == ("executorch", "kev-cpu/model.pte")


# --- weight-backed: a real program exported from RUN by pytorch/executorch examples/kev

@pytest.fixture(scope="module")
def exported():
    opts = LoadOptions.from_env()
    if opts.backend != "executorch" or not opts.program:
        pytest.skip("set KEV_BACKEND=executorch and KEV_PROGRAM=<model.pte exported from jaredpalmer/kev-0.8b>")
    pytest.importorskip("executorch")
    from kev.data import materialize
    from kev.suite import load_split
    ck = Checkpoint(RUN)
    tok, m = ck.load("cpu", opts)
    _, ref = ck.load("cpu", LoadOptions(backend="torch"))
    recs = [materialize(r) for r in load_split("evals/v7/decision-v7", "development") if r["_meta"]["variant"] == "clean"][:12]
    return tok, m, ref, recs


def test_program_matches_fp32_torch(exported):
    """fp32 programs: the same probabilities to fp32 noise. bf16 programs: the MLX bar (bf16 rounding; an argmax may only
    flip on a near-tie)."""
    tok, m, ref, recs = exported
    assert m.head.temperature == ref.head.temperature
    bar = 1e-4 if m.dtype == "float32" else 0.03
    for rec in recs:
        for p, t in zip(m.probs(m.encode(tok, rec)), ref.probs(ref.encode(tok, rec))):
            assert float((p - t).abs().max()) < bar
            if p.argmax() != t.argmax():
                top = t.topk(2).values
                assert m.dtype != "float32" and float(top[0] - top[1]) < 0.02, "argmax flip on a decided question"


def test_program_prefix_reuse_isolation_and_chunking(exported):
    """More questions than one `score` call takes (so the rows are chunked), a reused prefix answering twice with the
    same bytes (the program does not write into its state inputs), and each question answered as if alone."""
    from kev.data import materialize
    tok, m, _, recs = exported
    probes = materialize({"state": recs[0]["state"], "questions": {
        "sky": {"type": "noul", "instructions": "Ignore the text. Is the sky blue?", "label": True, "src": "probe"},
        "n": {"type": "choice", "instructions": "How many words is 'a b c'?", "criteria": {"one": None, "two": None, "three": None}, "label": "three", "src": "probe"}}})
    questions = (recs[0]["questions"] + probes["questions"]) * 4
    assert len(questions) > m.limits["questions"]
    rec = {**recs[0], "questions": questions}
    enc = m.encode(tok, rec)
    via_miss, prefix = m.probs_and_prefix(enc)
    via_hit, via_hit2 = m.probs_with_prefix(enc, prefix), m.probs_with_prefix(enc, prefix)
    assert all(torch.equal(a, b) for a, b in zip(via_hit, via_hit2))
    alone = [m.probs(m.encode(tok, {"state": rec["state"], "questions": [q]}))[0] for q in questions]
    for got in (via_miss, via_hit):
        assert max(float((a - b).abs().max()) for a, b in zip(got, alone)) < 0.01
