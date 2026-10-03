import sys
import warnings

import pytest

from kev import compare
from kev.suite import read_json, write_json


@pytest.mark.parametrize("variants", [(), ("none_present",), ("none_absent",), ("none_present", "none_absent")])
def test_comparison_writes_missing_none_diagnostics(tmp_path, monkeypatch, variants):
    row = {"id": "one", "question": "q", "source": "fixture", "group": "one", "task": "fixture",
           "type": "choice", "variant": "clean", "keys": ["answer", "none_of_these"], "label": 0, "p": [0.75, 0.25]}
    rows = [row] + [dict(row, variant=variant, p=p) for variant in variants
                    for p in ([0.75, 0.25], [0.5, 0.5], [0.25, 0.75])]
    for name in ("candidate", "reference"):
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / "rows.json", rows)
        write_json(directory / "report.json", {"suite_sha256": "fixture", "clean": {"n": 1}, "tasks": {}})
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["kev.compare", "--candidate", str(tmp_path / "candidate"),
                                     "--reference", str(tmp_path / "reference"), "--out", str(output)])
    # Exercise the real bootstrap and strict JSON writer, including empty NumPy warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compare.main()
    result = read_json(output)
    for side in ("candidate", "reference"):
        for variant in ("none_present", "none_absent"):
            expected = ({"n": 3, "mean_p_none": 0.5, "p_none_above_half": 1 / 3} if variant in variants
                        else {"n": 0, "mean_p_none": None, "p_none_above_half": None})
            assert result["none_of_the_above"][side][variant] == expected
    for metric in ("acc", "nll", "brier"):
        assert result["paired"][metric][f"macro_{metric}_delta"] == 0
