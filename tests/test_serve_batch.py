import kev.serve as serve
import numpy as np


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, tok, rec, **kwargs):
        return {
            "ids": [11, 12, 13],
            "seg": [0, 0, 1],
            "option_isolation": False,
        }

    def probs_and_prefix_batch(self, encs):
        self.calls.append(("miss_batch", len(encs)))
        return [[np.array([0.7, 0.3])] for _ in encs], (2, "prefix")

    def probs_with_prefix_batch(self, encs, prefix):
        self.calls.append(("hit_batch", len(encs)))
        return [[np.array([0.6, 0.4])] for _ in encs]

    def probs_with_prefix(self, enc, prefix):
        self.calls.append(("hit_one", 1))
        return [np.array([0.6, 0.4])]


def test_same_state_batch_reuses_one_prefix(monkeypatch):
    model = FakeModel()
    state = {
        "tok": object(),
        "model": model,
        "dev": "mlx",
        "prefix_cache": {},
        "prefix_hits": 0,
        "prefix_misses": 0,
        "prefix_coalesced": 0,
    }
    monkeypatch.setattr(serve, "STATE", state)
    monkeypatch.setattr(serve, "PREFIX_CACHE_SIZE", 4)
    monkeypatch.setattr(serve, "PREFIX_MIN_TOKENS", 384)
    monkeypatch.setattr(serve, "TEMPERATURE", 1.0)
    records = [
        {"state": "same", "questions": [{"instr": "a", "options": ["x", "y"]}]},
        {"state": "same", "questions": [{"instr": "b", "options": ["x", "y"]}]},
    ]

    values = serve._probs_core_many(records)
    assert len(values) == 2
    assert model.calls == [("miss_batch", 2)]
    assert state["prefix_misses"] == 1
    assert state["prefix_coalesced"] == 1
    assert all(item[1]["prefix_cache_hit"] is False for item in values)

    one = serve._probs_core_many([records[0]])
    assert model.calls[-1] == ("hit_one", 1)
    assert one[0][1]["prefix_cache_hit"] is True
    assert one[0][0] == [[0.6, 0.4]]
