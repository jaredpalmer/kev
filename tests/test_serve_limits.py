"""The serving context caps are deployment policy, not a property of the checkpoint.

A host with the memory for longer states than the 8,192 kev.model defaults to has to be able to raise the caps
without editing a tracked file on every upgrade. The training and suite contexts are not deployment policy and stay
where they are.
"""
import importlib

import kev.model


def _reload(monkeypatch, **env):
    for k in ("KEV_MAX_STATE", "KEV_MAX_BRANCH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import kev.serve
    return importlib.reload(kev.serve)


def test_limits_default_to_the_model_serving_context(monkeypatch):
    s = _reload(monkeypatch)
    assert (s.SERVE_MAX_STATE, s.SERVE_MAX_BRANCH) == (kev.model.SERVE_MAX_STATE, kev.model.SERVE_MAX_BRANCH)


def test_limits_follow_the_environment(monkeypatch):
    s = _reload(monkeypatch, KEV_MAX_STATE="32768", KEV_MAX_BRANCH="40960")
    assert (s.SERVE_MAX_STATE, s.SERVE_MAX_BRANCH) == (32768, 40960)


def test_each_limit_moves_on_its_own(monkeypatch):
    s = _reload(monkeypatch, KEV_MAX_STATE="32768")
    assert (s.SERVE_MAX_STATE, s.SERVE_MAX_BRANCH) == (32768, kev.model.SERVE_MAX_BRANCH)


def test_the_training_and_suite_contexts_do_not_move(monkeypatch):
    _reload(monkeypatch, KEV_MAX_STATE="32768", KEV_MAX_BRANCH="40960")
    assert (kev.model.SERVE_MAX_STATE, kev.model.SERVE_MAX_BRANCH) == (8192, 8192)
