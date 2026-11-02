from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kev.model import PointerHead
from kev.suite import read_json
from scripts.head_weight_edit import (MAX_GAIN_CHANGE, apply_gains, capture_features, curvature,
                                     fit_sample, identities, solve_gains)


def record(i, source="a", group=None, state=None):
    return {"state": state or f"state {i}", "questions": {"q": {"type": "choice", "instructions": "Pick",
            "criteria": {"a": "A", "b": "B"}, "label": "a", "src": source}},
            "_meta": {"id": str(i), "group_id": str(i) if group is None else group, "source": source, "variant": "clean"}}


def test_fit_selection_is_label_independent_source_balanced_and_disjoint():
    records = [record(i, str(i % 2)) for i in range(30)]
    blocked = record(0)
    records += [record(100, group="0"), record(101, state="  STATE   0 "), record(102, state="state 1")]
    picked = fit_sample(records, identities(blocked), count=12)
    assert len(picked) == 12
    assert sum(r["_meta"]["source"] == "0" for r in picked) == 6
    assert all(not identities(r) & identities(blocked) for r in picked)
    assert len({r["_meta"]["group_id"] for r in picked}) == 12
    assert len({next(k[1] for k in identities(r) if k[0] == "state") for r in picked}) == 12
    changed = deepcopy(records)
    for r in changed:
        r["questions"]["q"]["label"] = "b"
    assert [r["_meta"]["id"] for r in picked] == [r["_meta"]["id"] for r in fit_sample(changed[::-1], identities(blocked), count=12)]
    with pytest.raises(ValueError, match="disjoint"):
        fit_sample(records, set(), count=1000)


def test_analytic_curvature_equals_autograd_and_step_reduces_loss():
    torch.manual_seed(8)
    x = torch.randn(4, 6, dtype=torch.float64)
    target = torch.tensor([0.1, 0.2, 0.6, 0.1], dtype=torch.float64)
    z = x.sum(-1)
    rhs, hessian = curvature(x, z, target)
    gains = torch.ones(6, dtype=torch.float64, requires_grad=True)
    def loss(g): return -(target * (x @ g).log_softmax(-1)).sum()
    assert torch.allclose(-torch.autograd.functional.jacobian(loss, gains), rhs)
    assert torch.allclose(torch.autograd.functional.hessian(loss, gains), hessian)
    delta, damping = solve_gains(rhs, hessian)
    assert damping > 0 and delta.abs().max() <= MAX_GAIN_CHANGE + 1e-12
    assert abs(float(delta.sum())) < 1e-12
    assert loss(gains + delta) < loss(gains)
    assert torch.all(1 + delta > 0)


def test_mean_zero_solve_blocks_scalar_temperature_change():
    delta, _ = solve_gains(torch.ones(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64))
    assert torch.equal(delta, torch.zeros_like(delta))


@pytest.mark.parametrize("ridge,bound", [(0, .25), (float("nan"), .25), (.1, 1), (.1, -1)])
def test_bad_solve_bounds_fail(ridge, bound):
    with pytest.raises(ValueError):
        solve_gains(torch.ones(3), torch.eye(3), ridge, bound)


def test_nonfinite_and_empty_curvature_fail():
    with pytest.raises(ValueError):
        solve_gains(torch.ones(3), torch.zeros(3, 3))
    with pytest.raises(ValueError):
        curvature(torch.ones(2, 3), torch.tensor([float("nan"), 0.0]), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError):
        curvature(torch.ones(2, 3), torch.zeros(2), torch.tensor([1.0, 1.0]))


def test_folded_gains_match_feature_logits_restore_and_reject_before_mutation():
    torch.manual_seed(3)
    head = PointerHead(12, dp=6).eval()
    head.temperature = 2.35
    query, options = torch.randn(12), torch.randn(4, 12)
    original = {"weight": head.q.weight.detach().clone(), "bias": head.q.bias.detach().clone()}
    keys_before = {k: v.clone() for k, v in head.k.state_dict().items()}
    logits = head(query, options).detach().clone()
    x = head.q(query) * head.k(options) * head.scale / head.temperature
    delta = torch.tensor([.2, -.2, .1, -.1, .05, -.05])
    apply_gains(head, original, delta)
    assert torch.allclose(head(query, options), logits + x @ delta, atol=2e-7)
    assert all(torch.equal(head.k.state_dict()[k], v) for k, v in keys_before.items())
    before = head.q.weight.clone()
    with pytest.raises(ValueError):
        apply_gains(head, original, delta * 5)
    assert torch.equal(head.q.weight, before)
    apply_gains(head, original, torch.zeros(6))
    assert torch.equal(head.q.weight, original["weight"]) and torch.equal(head.q.bias, original["bias"])
    assert torch.equal(head(query, options), logits)


def test_feature_capture_uses_real_head_and_removes_hook_on_failure():
    torch.manual_seed(4)
    head = PointerHead(10, dp=4).eval()
    head.temperature = 2.35
    class Predictor:
        model = SimpleNamespace(head=head)
        def __call__(self, r):
            z = head(torch.randn(10), torch.randn(2, 10))
            return {"logits": {"q": dict(zip(("a", "b"), z.tolist()))}}
    r = record(0)
    r["questions"]["q"]["target"] = {"a": 1, "b": 3}
    x, z, target = capture_features(Predictor(), r)[0]
    assert torch.allclose(x.sum(-1), z, atol=1e-7)
    assert torch.equal(target, torch.tensor([.25, .75], dtype=torch.float64))
    assert not head._forward_pre_hooks
    class Broken(Predictor):
        def __call__(self, r): raise RuntimeError("deliberate")
    with pytest.raises(RuntimeError, match="deliberate"):
        capture_features(Broken(), r)
    assert not head._forward_pre_hooks


def test_shuffled_control_preserves_gain_multiset_and_not_assignments():
    delta = torch.linspace(-.25, .25, 256, dtype=torch.float64)
    shuffled = delta[torch.from_numpy(np.random.default_rng(0).permutation(len(delta)))]
    assert torch.equal(delta.sort().values, shuffled.sort().values)
    assert not torch.equal(delta, shuffled)


def test_round31_is_preregistered_report_only():
    from pathlib import Path
    from kev.rounds import load, validate
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/rounds/r31.json"
    spec = load(path)
    assert not validate(spec, root, rows=False).problems
    assert all(not arm["select"] for arm in read_json(path)["arms"].values())
