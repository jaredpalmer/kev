"""Indexed DeltaNet state loads against the pinned FLA serving calculation."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('rows,length', [(1, 1), (3, 17), (8, 64), (17, 129)])
def test_indexed_delta_matches_gather_and_replays(dtype, rows, length):
    pytest.importorskip('fla')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from kev.indexed_delta import indexed_delta_rule

    torch.manual_seed(41)
    H, HV, K, V = 2, 4, 128, 128
    q, k = [torch.randn(rows, length, H, K, device='cuda', dtype=dtype) for _ in range(2)]
    v = torch.randn(rows, length, HV, V, device='cuda', dtype=dtype)
    g, beta = [torch.randn(rows, length, HV, device='cuda', dtype=dtype) for _ in range(2)]
    A_log = torch.randn(HV, device='cuda')
    dt_bias = torch.randn(HV, device='cuda')
    bank = torch.randn(5, HV, K, V, device='cuda')
    before = bank.clone()
    upload = torch.zeros(rows, 37, device='cuda', dtype=torch.long)
    entry = upload[:, -1]
    entry.copy_(torch.arange(rows, device='cuda') % 5)

    def candidate():
        return indexed_delta_rule(q, k, v, g, beta, bank, entry, A_log, dt_bias)

    def reference():
        return chunk_gated_delta_rule(q, k, v, g, beta, initial_state=bank.index_select(0, entry),
            output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            A_log=A_log, dt_bias=dt_bias, use_beta_sigmoid_in_kernel=True)[0]

    expected = reference()
    assert torch.equal(candidate(), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): candidate()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual = candidate()
    for indices in (torch.zeros_like(entry), (entry + 2) % 5, entry.flip(0)):
        entry.copy_(indices)
        expected = reference()
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.equal(bank, before)
