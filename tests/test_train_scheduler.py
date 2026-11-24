import copy
import math

import pytest
import torch

from kev.train import _one_cycle_scheduler


MAX_LR = [0.03, 0.007]


def _optimizer(initial=None):
    values = initial or (torch.tensor([1.0, -2.0], dtype=torch.float64),
                         torch.tensor([0.5, 3.0], dtype=torch.float64))
    params = [torch.nn.Parameter(value.clone()) for value in values]
    opt = torch.optim.AdamW([{"params": [params[0]], "lr": MAX_LR[0]},
                             {"params": [params[1]], "lr": MAX_LR[1]}],
                            weight_decay=0.01)
    return params, opt


def _step(params, opt, sched, index, trace):
    opt.zero_grad(set_to_none=True)
    loss = (params[0] * (index + 1)).square().sum() + (params[1] * (index + 2)).square().sum()
    loss.backward()
    opt.step()
    sched.step()
    trace.append((tuple(group["lr"] for group in opt.param_groups),
                  tuple(group["betas"][0] for group in opt.param_groups)))


def _run(steps, legacy=False):
    params, opt = _optimizer()
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=steps, pct_start=0.1)
             if legacy else _one_cycle_scheduler(opt, MAX_LR, steps))
    trace = []
    for i in range(steps):
        _step(params, opt, sched, i, trace)
    return params, trace, sched


def test_ten_step_legacy_schedule_fails_and_factory_completes_finitely():
    params, opt = _optimizer()
    with pytest.raises(ZeroDivisionError):
        torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=10, pct_start=0.1)

    params, opt = _optimizer()
    sched = _one_cycle_scheduler(opt, MAX_LR, 10)
    trace = []
    for i in range(10):
        _step(params, opt, sched, i, trace)
    assert sched.last_epoch == 10
    assert all(math.isfinite(lr) and math.isfinite(beta1) for lrs, betas in trace for lr, beta1 in zip(lrs, betas))
    assert all(torch.isfinite(param).all() for param in params)


@pytest.mark.parametrize("steps", range(1, 65))
def test_short_one_cycle_schedules_complete_with_finite_optimizer_state(steps):
    params, trace, sched = _run(steps)
    assert sched.last_epoch == steps
    assert len(trace) == steps
    assert all(math.isfinite(lr) and math.isfinite(beta1) for lrs, betas in trace for lr, beta1 in zip(lrs, betas))
    assert all(torch.isfinite(param).all() for param in params)


def test_32_step_schedule_matches_legacy_lr_beta_and_tensor_updates_exactly():
    actual_params, actual_trace, _ = _run(32)
    legacy_params, legacy_trace, _ = _run(32, legacy=True)
    assert actual_trace == legacy_trace
    assert all(torch.equal(actual, legacy) for actual, legacy in zip(actual_params, legacy_params))


def test_ten_step_scheduler_resume_matches_uninterrupted_run():
    expected_params, expected_trace, _ = _run(10)

    params, opt = _optimizer()
    sched = _one_cycle_scheduler(opt, MAX_LR, 10)
    first_trace = []
    for i in range(4):
        _step(params, opt, sched, i, first_trace)
    saved_values = [param.detach().clone() for param in params]
    optimizer_state = copy.deepcopy(opt.state_dict())
    scheduler_state = copy.deepcopy(sched.state_dict())

    resumed_params, resumed_opt = _optimizer(saved_values)
    resumed_sched = _one_cycle_scheduler(resumed_opt, MAX_LR, 10)
    resumed_opt.load_state_dict(optimizer_state)
    resumed_sched.load_state_dict(scheduler_state)
    resumed_trace = first_trace.copy()
    for i in range(4, 10):
        _step(resumed_params, resumed_opt, resumed_sched, i, resumed_trace)

    assert resumed_trace == expected_trace
    assert resumed_sched.state_dict() == _run(10)[2].state_dict()
    assert all(torch.equal(actual, expected) for actual, expected in zip(resumed_params, expected_params))
