"""Ascend NPU backend (kev.npu_qwen35 and the NPU branches of kev.device, kev.model, kev.checkpoint and kev.serve).

Most of this needs no NPU and runs in CI: kev.npu_qwen35 imports torch only (torch_npu and vllm-ascend load lazily), and a
fake `torch.npu` stands in for the device calls. The tests taking the `npu` fixture need an Ascend card with torch_npu and
vllm-ascend and skip anywhere else. Run them on a free card (docs/ascend-npu.md: `npu-smi info`, and do not `uv sync` over
the system torch), from a directory outside the repo (the Ascend compiler writes kernel_meta/ and fusion_result.json there):
    ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<repo>:$PYTHONPATH python -m pytest <repo>/tests/test_npu.py -q
(keep CANN's PYTHONPATH from set_env.sh: the op compiler imports `tbe` from it.)
"""
import copy
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kev import npu_qwen35
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import DecisionModel


def fake_npu(monkeypatch, **extra):
    """A `torch.npu` that records what kev.device and kev.npu_qwen35 ask of it."""
    calls = []
    ns = SimpleNamespace(synchronize=lambda: calls.append("synchronize"), empty_cache=lambda: calls.append("empty_cache"),
                         max_memory_allocated=lambda: 123, graph_pool_handle=lambda: "pool", **extra)
    monkeypatch.setattr(torch, "npu", ns, raising=False)
    return calls


def test_device_helpers_reach_torch_npu(monkeypatch):
    """kev.device is the one home of synchronize / empty_cache / memory; "npu:0" and "npu:3" dispatch like "cuda" does."""
    from kev import device
    calls = fake_npu(monkeypatch)
    device.sync("npu:0"); device.empty_cache("npu:3")
    assert calls == ["synchronize", "empty_cache"]
    assert device.allocated_bytes("npu:0") == 123 and device.allocated_bytes("cpu") == 0
    device.sync("cpu"); device.empty_cache("cpu")
    assert calls == ["synchronize", "empty_cache"]   # a CPU run never touches the NPU


class Rows(DecisionModel):
    """DecisionModel's real probs / probs_and_prefix / probs_with_prefix with the passes replaced by recorders."""
    def __init__(self, device):
        torch.nn.Module.__init__(self)
        self.device, self.passes = device, []
    def rows_form(self, encs): return True
    def forward(self, enc):
        self.passes.append("forward")
        return [torch.tensor([0.0, 1.0]), torch.tensor([2.0, 0.0, 0.0])]
    def prefix(self, enc):
        self.passes.append("prefix")
        return 2, "cache", None
    def _branch_rows_from_prefix(self, enc, cache):
        self.passes.append("rows from cache")
        return ["from cache"]


ENC = {"seg": [0, 0, 1, 1]}


def test_hybrid_rows_are_recomputed_on_npu():
    """transformers 5.5.4 continues a cached DeltaNet state with a decode kernel kev.npu_qwen35 does not replace, so on an
    NPU a hybrid record runs its rows from scratch: no state pass, no prefix kept, and a prefix handed in is not used."""
    expected = [F.softmax(z, -1) for z in Rows("npu:0").forward(ENC)]
    m = Rows("npu:0")
    probs, prefix = m.probs_and_prefix(ENC)
    assert prefix is None and m.passes == ["forward"]
    assert all(torch.equal(a, b) for a, b in zip(probs, expected))
    m = Rows("npu:0")
    assert all(torch.equal(a, b) for a, b in zip(m.probs_with_prefix(ENC, (2, "cache", None)), expected))
    assert m.passes == ["forward"]
    m = Rows("npu:0")
    assert all(torch.equal(a, b) for a, b in zip(m.probs(ENC), expected)) and m.passes == ["forward"]


def test_hybrid_rows_still_use_the_prefix_off_npu():
    """The NPU branch is the only change: on any other device the miss path is a state pass kept as the prefix plus the
    branch rows from its cache, and a hit runs only the rows."""
    m = Rows("cpu")
    probs, prefix = m.probs_and_prefix(ENC)
    assert probs == ["from cache"] and prefix == (2, "cache", None) and m.passes == ["prefix", "rows from cache"]
    m = Rows("cpu")
    assert m.probs_with_prefix(ENC, (2, "cache", None)) == ["from cache"] and m.passes == ["rows from cache"]


def test_server_keeps_no_prefix_cache_on_npu(monkeypatch):
    """A cached state would be continued through the recurrent kernel, so the NPU server builds its cache with size 0
    (and DecisionModel recomputes, see above); every other device keeps the configured size."""
    from kev import serve
    monkeypatch.setattr(serve, "PREFIX_CACHE_SIZE", 4)

    class Model:
        prefix_min_tokens = 0

    for device, size in (("npu:0", 0), ("cpu", 4)):
        s = serve.Server(SimpleNamespace(release_date=lambda: "2026-01-01"), None, Model(), device)
        try:
            assert s.prefix_cache.size == size, device
        finally:
            s.close()


def ascend(monkeypatch, *, merged=True, hybrid=True):
    """-> (calls, stand-in Checkpoint, tokenizer) for Checkpoint._load_torch with kev.npu_qwen35's entry points recorded,
    in the order they happen: prewarm before the model is built, then fuse and accelerate on its language model."""
    calls = []
    model = SimpleNamespace(hybrid=hybrid, lm="lm", pad_id=7)

    def build(tok, device, opts):
        calls.append("build")
        return model, merged

    ck = SimpleNamespace(full=False, _adapted_torch=build, _full_torch=build)
    monkeypatch.setattr(npu_qwen35, "prewarm", lambda: calls.append("prewarm"))
    monkeypatch.setattr(npu_qwen35, "fuse", lambda lm, pad_id=None, graphs=True: calls.append(("fuse", lm, pad_id, graphs)))
    monkeypatch.setattr(npu_qwen35, "accelerate", lambda lm: calls.append(("accelerate", lm)))
    return calls, ck, model


def test_load_prewarms_then_fuses_and_accelerates_on_npu(monkeypatch):
    """prewarm() must run before the backbone initialises the device (vllm-ascend's custom ops fail to launch otherwise);
    fuse needs plain weights, so a merged adapter or full weights, and only when asked for; accelerate always."""
    calls, ck, model = ascend(monkeypatch)
    assert Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions(fused=True)) is model
    assert calls == ["prewarm", "build", ("fuse", "lm", 7, True), ("accelerate", "lm")]

    calls, ck, _ = ascend(monkeypatch)
    Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions(fused=True, npu_graphs=False))
    assert calls == ["prewarm", "build", ("fuse", "lm", 7, False), ("accelerate", "lm")]   # fused layers, every kernel enqueued

    calls, ck, _ = ascend(monkeypatch)
    Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions())
    assert calls == ["prewarm", "build", ("accelerate", "lm")]                  # fused is opt-in at the library level

    calls, ck, _ = ascend(monkeypatch, merged=False)
    Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions(fused=True))
    assert calls == ["prewarm", "build", ("accelerate", "lm")]                  # an unmerged adapter is not fused

    calls, ck, _ = ascend(monkeypatch, hybrid=False)
    Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions(fused=True))
    assert calls == ["prewarm", "build"]                                        # the kernels are for the Gated DeltaNet


def test_load_can_decline_the_ascend_kernels(monkeypatch):
    """npu_kernels=False keeps the reference Gated DeltaNet: nothing of kev.npu_qwen35 is touched. A failed import or
    prewarm (no vllm-ascend) warns and loads the same way."""
    calls, ck, _ = ascend(monkeypatch)
    Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions(fused=True, npu_kernels=False))
    assert calls == ["build"]

    calls, ck, _ = ascend(monkeypatch)
    def refuse(): raise RuntimeError("vllm_ascend custom ops are unavailable")
    monkeypatch.setattr(npu_qwen35, "prewarm", refuse)
    with pytest.warns(UserWarning, match="kev.npu_qwen35 unavailable"):
        Checkpoint._load_torch(ck, "tok", "npu:0", LoadOptions(fused=True))
    assert calls == ["build"]


def test_the_npu_switches_come_from_load_options(monkeypatch):
    """Both are `KEV_*`, so they are read where every other one is (LoadOptions.from_env), not inside the backend; on by
    default, which is what every NPU number was measured with."""
    assert (LoadOptions().npu_kernels, LoadOptions().npu_graphs) == (True, True)
    fresh = LoadOptions.from_env({})
    assert (fresh.npu_kernels, fresh.npu_graphs) == (True, True)
    off = LoadOptions.from_env({"KEV_NPU_KERNELS": "0", "KEV_NPU_GRAPHS": "0"})
    assert (off.npu_kernels, off.npu_graphs) == (False, False)


def test_device_select_resolves_an_ascend_card(monkeypatch):
    """kev.device.select is where a --device string is checked and where torch_npu is imported, before anything can
    touch the device (kev.benchmark and kev.serve both go through it)."""
    from kev import device
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())   # the import select owes an Ascend device
    for ok in ("cpu", "mps", "cuda", "cuda:1", "npu", "npu:7"):
        assert device.select(ok) == ok
    for bad in ("gpu", "npu:x", "npu:0:1", ""):
        with pytest.raises(ValueError, match="unknown device"):
            device.select(bad)


class FakeGatedDeltaNet(torch.nn.Module):
    def __init__(self, chunk):
        super().__init__()
        self.chunk_gated_delta_rule = chunk


def test_report_environment_names_the_ascend_kernels(monkeypatch):
    """A read's report.json says which kernels its logits came from (kev.predictors.kernel_environment). On an NPU that
    is the card and the chunk kernel accelerate bound on each layer; other devices' environments are unchanged."""
    from kev.predictors import kernel_environment
    fake_npu(monkeypatch, get_device_name=lambda device: "Ascend910B2")
    lm = torch.nn.Sequential(FakeGatedDeltaNet(npu_qwen35.chunk_gated_delta_rule))
    lm.config = SimpleNamespace(_attn_implementation="eager")
    model = SimpleNamespace(backend="torch", dtype="torch.bfloat16", hybrid=True, lm=lm)
    env = kernel_environment(model, "npu:0")
    assert env["gpu"] == "Ascend910B2"
    assert env["deltanet"]["chunk_gated_delta_rule"] == "kev.npu_qwen35.chunk_gated_delta_rule"
    off = kernel_environment(model, "cpu")
    assert off["gpu"] is None and "chunk_gated_delta_rule" not in off["deltanet"]


def test_other_devices_do_not_touch_the_ascend_kernels(monkeypatch):
    calls, ck, _ = ascend(monkeypatch)
    Checkpoint._load_torch(ck, "tok", "cpu", LoadOptions(fused=True, cuda_graphs=True))
    assert calls == ["build"]


def graphs(monkeypatch, pad_id=7):
    fake_npu(monkeypatch, mem_get_info=lambda: (npu_qwen35.GRAPH_MIN_FREE * 2, 0))
    return npu_qwen35.Graphs(None, pad_id)


def test_graphs_pad_each_pass_to_a_rung_of_the_ladder(monkeypatch):
    """A graph is captured per padded shape, so a pass is rounded up to the next rung and a pass past the ladder runs
    eagerly. The token axis is the second of ids, positions and masks, the second of embeddings, the third of mrope planes."""
    g = graphs(monkeypatch)
    ids, emb, planes = torch.zeros(2, 100, dtype=torch.long), torch.zeros(2, 100, 8), torch.zeros(3, 2, 100, dtype=torch.long)
    assert [npu_qwen35._token_axis(t) for t in (ids, emb, planes)] == [1, 1, 2]
    rungs = npu_qwen35.GRAPH_LENGTHS
    assert g.bucket(torch.zeros(1, 1)) == rungs[0] and g.bucket(ids) == 128
    assert g.bucket(torch.zeros(1, rungs[0] + 1)) == rungs[1] and g.bucket(torch.zeros(1, rungs[-1])) == rungs[-1]
    assert g.bucket(torch.zeros(1, rungs[-1] + 1)) is None
    mask = torch.ones(2, 100, dtype=torch.long)
    assert g.key((ids, ids, mask), 128) == ((2, 128, torch.long), (2, 128, torch.long), (2, 128, torch.long))
    assert g.key((emb, planes, None), 192) == ((2, 192, 8, torch.float32), (3, 2, 192, torch.long), None)
    assert g.fills(ids) == (7, 0, 0) and g.fills(emb) == (0, 0, 0)   # padded tokens, or zero embeddings; position 0; masked out


def test_graphs_replay_pads_the_buffers_and_copies_the_result(monkeypatch):
    """What a replayed pass reads: the real tokens at the front of the captured buffers, pad / position 0 / mask 0 behind
    them (not what the last pass left), and an output cloned out of the buffer the next replay overwrites."""
    g = graphs(monkeypatch)
    tokens, pos, mask = torch.tensor([[1, 2, 3]]), torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.long)
    key = g.key((tokens, pos, mask), 128)
    buffers = [torch.full((1, 128), 99), torch.full((1, 128), 5), torch.full((1, 128), 1)]   # a previous, longer pass
    out, replays = torch.arange(4.0), []
    g.entries[key] = (SimpleNamespace(replay=lambda: replays.append(1)), buffers, out)
    got = g.run(tokens, pos, mask)
    assert replays == [1] and torch.equal(got, out) and got.data_ptr() != out.data_ptr()
    assert buffers[0][0, :3].tolist() == [1, 2, 3] and set(buffers[0][0, 3:].tolist()) == {7}
    assert buffers[1][0, :3].tolist() == [0, 1, 2] and set(buffers[1][0, 3:].tolist()) == {0}
    assert buffers[2][0, :3].tolist() == [1, 1, 1] and set(buffers[2][0, 3:].tolist()) == {0}


def test_graphs_leave_a_pass_eager_when_they_cannot_serve_it(monkeypatch):
    """None = run eagerly: past the ladder, a shape whose capture failed, and no memory left for another graph."""
    g = graphs(monkeypatch)
    long = torch.zeros(1, npu_qwen35.GRAPH_LENGTHS[-1] + 1, dtype=torch.long)
    assert g.run(long, long, long) is None
    tokens = torch.zeros(1, 10, dtype=torch.long)
    g.refused.add(g.key((tokens, tokens, tokens), 128))
    assert g.run(tokens, tokens, tokens) is None
    g = graphs(monkeypatch)
    monkeypatch.setattr(torch.npu, "mem_get_info", lambda: (npu_qwen35.GRAPH_MIN_FREE, 0))
    assert not g.room() and g.run(tokens, tokens, tokens) is None and g.entries == {}


def test_layer_type_reads_either_transformers_spelling():
    """transformers 5.5 names a decoder layer's mixer `layer_type`, later versions `block_type`."""
    assert npu_qwen35._layer_type(types.SimpleNamespace(layer_type="linear_attention")) == "linear_attention"
    assert npu_qwen35._layer_type(types.SimpleNamespace(block_type="full_attention")) == "full_attention"


@pytest.fixture(scope="module")
def npu():
    """The first visible Ascend card, with vllm-ascend's ops registered before anything initialises it (prewarm)."""
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm_ascend")
    if not torch.npu.is_available():
        pytest.skip("no Ascend card visible")
    npu_qwen35.prewarm()
    return torch.device("npu:0")


def relative_error(got, want):
    return ((got - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("B,T", [(1, 40), (3, 100), (2, 257)])
def test_chunk_kernel_matches_the_reference_recurrence(npu, B, T):
    """The Ascend chunk against transformers' fp32 torch_chunk_gated_delta_rule, on the non-contiguous [B, T, H, D] views
    the model hands in and with a ragged last chunk (T not a multiple of 64). The kernels run in bf16: measured on a
    910B2 at 0.8-1.1 % of the output's largest value."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    torch.manual_seed(0)
    H, D = 8, 128
    q, k, v = (torch.randn(B, H, T, D).transpose(1, 2) for _ in range(3))
    g, beta = -torch.rand(B, T, H), torch.rand(B, T, H)
    want, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    got, state = npu_qwen35.chunk_gated_delta_rule(q.to(npu), k.to(npu), v.to(npu), g=g.to(npu), beta=beta.to(npu),
                                                   use_qk_l2norm_in_kernel=True)
    got = got.float().cpu()
    assert state is None and got.shape == want.shape and torch.isfinite(got).all()
    assert relative_error(got, want) < 0.03


@torch.no_grad()
def test_fused_model_and_its_graphs_match_the_reference(npu):
    """A two-layer Qwen3.5 text model (one Gated DeltaNet, one attention layer) through fuse, eagerly and then from a
    captured graph, against the same bf16 weights run in fp32 on the CPU, on right-padded rows as DecisionModel._pad_rows
    builds them. Measured on a 910B2: 0.86 % of the largest hidden value eagerly; a replay is padded to its bucket, which
    tiles the GEMMs differently (0.4 % from eager), and is bit-identical from one replay to the next."""
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=256, linear_num_value_heads=8,
                            linear_num_key_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
                            layer_types=["linear_attention", "full_attention"], pad_token_id=0)
    torch.manual_seed(1)
    ref = Qwen3_5TextModel(cfg).eval()
    for p in ref.parameters():   # the weights only: the rotary buffers stay fp32 as a loaded checkpoint keeps them
        p.data = p.data.bfloat16()
    lm = copy.deepcopy(ref).to(npu)
    ref = ref.float()
    rows, L = 3, 50
    ids = torch.randint(2, 64, (rows, L))
    mask = torch.ones(rows, L, dtype=torch.long)
    mask[1, 30:] = 0; mask[2, 41:] = 0
    ids[mask == 0] = 0
    real = mask.bool()
    want = ref(input_ids=ids, attention_mask=mask).last_hidden_state[real]

    def run():   # a replayed pass comes back at its bucket length; kev slices each row to its own
        return lm(input_ids=ids.to(npu), attention_mask=mask.to(npu)).last_hidden_state.float().cpu()[:, :L]

    npu_qwen35.fuse(lm, pad_id=0, graphs=False)
    eager = run()
    assert torch.isfinite(eager).all() and relative_error(eager[real], want) < 0.02

    lm.graphs = npu_qwen35.Graphs(lm, 0)
    captured, replayed = run(), run()
    assert len(lm.graphs.entries) == 1 and not lm.graphs.refused
    assert torch.equal(replayed, captured)
    assert relative_error(replayed[real], want) < 0.02
