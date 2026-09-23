"""What the load leaves reserved on the card.

Merging the LoRA builds the backbone in fp32 and casts it afterwards (checkpoint.load): for kev-4b that is ~16 GB the
caching allocator keeps reserved for the life of the process, because the allocator only ever grows. On a card shared
with a much larger server that reservation is most of the serving budget, and the request that needs it has no way to
ask for it back. The load hands it back itself, before the first request.
"""
from kev import serve
from kev.checkpoint import LoadOptions


class FakeCheckpoint:
    """Records that it was asked to load, without touching a card."""

    def __init__(self, run, events):
        self.requested, self.path, self.events = run, f"/cache/{run}", events

    def load(self, dev, opts):
        self.events.append(("load", dev))
        return "tok", "model"

    def release_date(self):
        return "2026-01-01"


def build(monkeypatch, events):
    monkeypatch.setattr(serve, "Checkpoint", lambda run: FakeCheckpoint(run, events))
    monkeypatch.setattr(serve, "empty_cache", lambda dev: events.append(("empty_cache", dev)))
    return serve.build_server("runs/kev", "cuda", LoadOptions())


def test_the_load_hands_back_what_it_reserved(monkeypatch):
    events = []
    build(monkeypatch, events)
    assert events == [("load", "cuda"), ("empty_cache", "cuda")]


def test_the_server_still_carries_what_was_loaded(monkeypatch):
    events = []
    srv = build(monkeypatch, events)
    assert (srv.tok, srv.model, srv.device) == ("tok", "model", "cuda")
    assert srv.checkpoint.requested == "runs/kev"
