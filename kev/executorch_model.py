"""ExecuTorch backend: a Kev program exported by pytorch/executorch's `examples/kev` (XNNPACK on CPU or MLX on Apple GPUs,
FP32 or unquantized BF16), scored from Python under Kev's own encoder.

The program is the whole model, frozen at export: the Qwen3.5 backbone with the LoRA merged, the pointer head and the
checkpoint's temperature. It has two methods,

    prefill(tokens [1, S])                                    -> conv, recurrent, kv      (the state prefix)
    score(tokens [Q, L], decide [Q], options [Q, K], conv, recurrent, kv) -> logits [Q, K]   (fp32, already / temperature)

and constant methods for the contract: `get_kev_version` (1), the limits the shapes were exported with (`get_max_prefix`,
`get_max_context` = state + one branch, `get_max_questions` rows per `score` call, `get_max_options`), `get_pad_id` and
`get_special_0..4` (the ids of kev.model.SPECIAL), `get_temperature` (the one baked into the logits, the checkpoint's)
and `get_checkpoint_id` ("sha256:" + the sha256 of the head.pt it was exported from). The last two are checked against
the checkpoint when present (programs exported before pytorch/executorch#23023 merged lack them), so a program cannot
be served under the wrong checkpoint's name.

This is the computation the other backends run: `kev.model.encode` builds the tokens, `rows_of` splits them into the state
and one causal row per question, `prefill` is the state prefix (the DeltaNet conv + recurrent state and the attention KV)
and `score` runs up to `max_questions` rows on it without changing it. The C++ runner in the example re-implements the
encoder and links a different tokenizer; this module does neither, so a parity read through it measures the lowered
program alone (scripts/backend_parity.py --native_tokenizer measures the tokenizer separately).

ExecuTorch 1.5 needs torch 2.14 and Kev pins torch<2.9, so this runs from its own environment with kev installed
--no-deps (AGENTS.md). Selected by `LoadOptions(backend="executorch", program=...)` in kev.checkpoint; never by "auto".
"""
import torch

from .model import MAX_BRANCH, MAX_STATE, SPECIAL, ContextOverflow, PrefixScorer, encode, pad_id, rows_of, rows_per_pass

VERSION = 1
LIMITS = ("prefix", "context", "questions", "options")


def load_program(path):
    """The ExecuTorch program at `path`, with the kernel library the XNNPACK export calls into (llama::gated_delta_rule
    runs outside the delegate) registered first."""
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401
    from executorch.runtime import Runtime
    return Runtime.get().load_program(str(path))


class ProgramHead:
    """The pointer head lives inside the program; what Kev can still see and set is its temperature. Setting a value other
    than the exported one rescales the program's logits by exported / requested, which is exact up to fp32 rounding: the
    division by the exported temperature is the program's last operation."""

    def __init__(self, temperature):
        self.temperature = temperature


class ExecuTorchDecisionModel(PrefixScorer):
    """Prefill-only scorer over an exported program; same scoring interface as DecisionModel and MLXDecisionModel."""
    backend, hybrid, option_isolation = "executorch", True, False
    device = "cpu"          # where inputs and outputs live; the delegate inside the program decides where the work runs
    prefix_min_tokens = 0   # the state always runs once, as `prefill`: the row form would recompute it per question

    def __init__(self, program, tok, temperature, checkpoint_id=None):
        """`temperature` is the checkpoint's: the one the program's logits are divided by. `checkpoint_id` is
        "sha256:<head.pt sha256>" of the checkpoint the program is meant to be."""
        names = set(program.method_names)
        const = lambda name: program.load_method(name).execute([])[0]
        if "get_kev_version" not in names or const("get_kev_version") != VERSION:
            raise ValueError(f"not a version-{VERSION} Kev program (pytorch/executorch examples/kev export)")
        self.limits = {k: int(const(f"get_max_{k}")) for k in LIMITS}
        specials = [const(f"get_special_{i}") for i in range(len(SPECIAL))]
        if specials != [tok.convert_tokens_to_ids(t) for t in SPECIAL] or const("get_pad_id") != pad_id(tok):
            raise ValueError("the program was exported with a different tokenizer than this checkpoint's base")
        self.exported_temperature = float(temperature)
        if checkpoint_id and "get_checkpoint_id" in names and const("get_checkpoint_id") != checkpoint_id:
            raise ValueError(f"the program was exported from {const('get_checkpoint_id')}, not this checkpoint ({checkpoint_id})")
        if "get_temperature" in names and abs(float(const("get_temperature")) - self.exported_temperature) > 1e-6:
            raise ValueError(f"the program's temperature is {const('get_temperature')}, the checkpoint's {temperature}: it was exported from another checkpoint or calibration")
        self.head = ProgramHead(self.exported_temperature)
        self.pad_id = pad_id(tok)
        self._prefill, self._score = program.load_method("prefill"), program.load_method("score")
        # the backbone precision the program was exported with, read off its attention KV (one-token prefill)
        self.dtype = str(self._prefill.execute([torch.tensor([[specials[0]]])])[2].dtype).removeprefix("torch.")

    def eval(self):
        return self

    def encode(self, tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=True, **kw):
        """kev.model.encode within the program's exported shapes, always strictly (`strict` is accepted for the callers
        that pass it and ignored): a state or a question row that does not fit is a ContextOverflow (a 422 from kev.serve,
        a rejected record in kev.benchmark), never a truncated state. The scoring methods take encodings from here."""
        limits = self.limits
        try:
            enc = encode(tok, rec, max_state=min(max_state, limits["prefix"]), max_branch=min(max_branch, limits["context"]), strict=True, option_isolation=False, **kw)
        except ContextOverflow as e:
            raise ContextOverflow(f"{e}; this program was exported for states of at most {limits['prefix']} tokens and {limits['context']} with a question") from None
        if max(map(len, enc["opt_idx"]), default=0) > limits["options"]:
            raise ContextOverflow(f"a question has more than {limits['options']} options, the most this program was exported for")
        return enc

    def prefix(self, enc):
        Ls = enc["seg"].count(0)
        return Ls, tuple(self._prefill.execute([torch.tensor([enc["ids"][:Ls]])]))

    def _branch_logits(self, enc, state):
        """The branches as right-padded rows on the state, at most max_questions (and rows_per_pass) rows per `score` call.
        Pads sit after every real token and both layer kinds are causal, so no real token sees one; padded option slots
        point at token 0 and are cut off before the softmax."""
        Ls = enc["seg"].count(0)
        _, _, rows = rows_of(enc)
        chunk = min(self.limits["questions"], rows_per_pass([r["ids"] for r in rows], Ls))
        scale = self.exported_temperature / self.head.temperature
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids = torch.full((len(part), max(len(r["ids"]) for r in part)), self.pad_id, dtype=torch.long)
            opts = torch.zeros((len(part), max(len(r["opts"]) for r in part)), dtype=torch.long)
            for i, r in enumerate(part):
                ids[i, : len(r["ids"])] = torch.tensor(r["ids"]); opts[i, : len(r["opts"])] = torch.tensor(r["opts"])
            z = self._score.execute([ids, torch.tensor([r["decide"] for r in part]), opts, *state])[0]
            out += [z[i, : len(r["opts"])] * scale for i, r in enumerate(part)]   # scale 1.0 (no override) is exact
        return out
