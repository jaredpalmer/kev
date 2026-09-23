"""The state-prefix cache holds GPU memory between requests, so *when* a prefix is displaced matters.

A cached prefix for a long state is gigabytes. Holding the one being replaced through the pass that
replaces it doubles that, and on a card with a per-process ceiling the pass is what runs out of
memory -- which leaves the displaced prefix in the cache, because the eviction is downstream of the
failure. Every request after it then meets the same wall, and the process never recovers.
"""
import kev.serve as serve


class Probs:
    def __init__(self, v): self.v = v
    def tolist(self): return self.v


class FakeModel:
    """Stands in for the checkpoint, and records how full the cache was when each miss began its pass."""
    prefix_min_tokens = 4

    def __init__(self):
        self.server = None
        self.cache_at_pass = []
        self.passes = 0

    def encode(self, tok, rec, max_state, max_branch):
        n = rec["state_tokens"]
        return {"ids": [rec["salt"] + i for i in range(n + 2)], "seg": [0] * n + [1, 1]}

    def probs(self, enc):
        return [Probs([0.5, 0.5])]

    def probs_and_prefix(self, enc):
        self.cache_at_pass.append(len(self.server.prefix_cache))
        self.passes += 1
        return [Probs([0.5, 0.5])], f"prefix-{self.passes}"

    def probs_with_prefix(self, enc, prefix):
        return [Probs([0.5, 0.5])]


def make_server(monkeypatch, size=1):
    monkeypatch.setattr(serve, "PREFIX_CACHE_SIZE", size)
    monkeypatch.setattr(serve, "PREFIX_MIN_TOKENS", None)
    model = FakeModel()
    server = serve.Server(None, None, model, "cpu", release_date="2026-01-01")
    model.server = server
    return server, model


def state(salt, tokens=16):
    return {"salt": salt, "state_tokens": tokens}


def test_a_miss_frees_the_prefix_it_displaces_before_the_pass_that_replaces_it(monkeypatch):
    server, model = make_server(monkeypatch, size=1)
    server.probs(state(0))
    server.probs(state(1000))
    assert model.cache_at_pass == [0, 0]


def test_nothing_is_displaced_while_the_cache_has_room(monkeypatch):
    server, model = make_server(monkeypatch, size=2)
    server.probs(state(0))
    server.probs(state(1000))
    server.probs(state(2000))
    assert model.cache_at_pass == [0, 1, 1]
    assert len(server.prefix_cache) == 2


def test_a_repeated_state_still_hits(monkeypatch):
    server, model = make_server(monkeypatch, size=1)
    server.probs(state(0))
    _, meta = server.probs(state(0))
    assert meta["prefix_cache_hit"] is True
    assert (server.prefix_hits, server.prefix_misses) == (1, 1)


def test_a_state_below_the_threshold_is_never_cached(monkeypatch):
    server, model = make_server(monkeypatch, size=1)
    server.probs(state(0, tokens=2))
    assert server.prefix_cache == {}
    assert model.cache_at_pass == []
