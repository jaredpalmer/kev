"""Offline guards for the one-shot Inkling released-weight plumbing smoke."""

import importlib.util
import io
import json
import stat
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import inkling_weight_smoke as smoke


def test_launch_marker_is_durable_and_exclusive(tmp_path):
    marker = tmp_path / "launch.marker.json"
    metadata = {"run_id": "one", "status": "launching"}
    smoke.claim_marker(marker, metadata)
    assert json.loads(marker.read_text(encoding="utf-8")) == metadata
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        smoke.claim_marker(marker, {"run_id": "two"})
    smoke.update_marker(marker, {"sandbox_id": "sb-test", "status": "running"})
    updated = json.loads(marker.read_text(encoding="utf-8"))
    assert updated == {"run_id": "one", "status": "running", "sandbox_id": "sb-test"}
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_report_is_written_privately(tmp_path):
    report = tmp_path / "smoke.json"
    smoke.write_report(report, {"status": "passed"})
    assert json.loads(report.read_text(encoding="utf-8")) == {"status": "passed"}
    assert stat.S_IMODE(report.stat().st_mode) == 0o600


def test_loading_info_only_allows_explicit_visual_sentinel():
    clean = smoke.summarize_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": ["model.visual.patch_embed.weight"],
            "mismatched_keys": [],
            "error_msgs": [],
        }
    )
    assert smoke.loading_info_passes(clean)
    for info in (
        {"missing_keys": ["embed_tokens.weight"]},
        {"unexpected_keys": ["lm_head.weight"]},
        {"mismatched_keys": ["layers.0.weight"]},
        {"error_msgs": ["load failed"]},
    ):
        assert not smoke.loading_info_passes(smoke.summarize_loading_info(info))


def test_cost_ceiling_includes_sandbox_build_allowance_and_egress():
    costs = smoke.cost_breakdown()
    assert costs["sandbox"] == pytest.approx(14.856)
    assert costs["image_build_allowance"] == pytest.approx(0.03965)
    assert costs["egress"] == pytest.approx(0.0001171875)
    assert smoke.cost_bound() == pytest.approx(14.8957671875)
    assert smoke.cost_bound() < smoke.MAX_COST_USD
    assert smoke.cost_breakdown(1500)["sandbox"] == pytest.approx(7.428)
    assert smoke.SANDBOX_SECONDS <= 3600
    assert smoke.END_TO_END_WATCHDOG_SECONDS <= 3600


def test_example_is_one_benign_choice_without_eval_payload():
    assert len(smoke.BENIGN_RECORD["questions"]) == 1
    question = smoke.BENIGN_RECORD["questions"][0]
    assert len(question["options"]) == 2
    assert question["label"] == 0
    assert smoke.TRANSFORMERS_REVISION == "6173a2f679ff4ed393afc499aefc569da7e68bc1"
    assert smoke.MODEL_REVISION == "6e7c8248f75265bdd4fbab5273304243b5d8c2cb"


def test_remote_runner_with_tiny_local_release_fixture(monkeypatch, tmp_path):
    if importlib.util.find_spec("transformers.models.inkling2_loop") is None:
        pytest.skip("Inkling2Loop architecture is unavailable in this Transformers installation")

    import torch
    from transformers import (
        AutoTokenizer,
        Inkling2LoopConfig,
        Inkling2LoopForConditionalGeneration,
        Inkling2LoopTextConfig,
        Inkling2VisionConfig,
    )

    class TinyTokenizer:
        unk_token_id = 1
        pad_token_id = 0

        def __init__(self):
            self.size = 68
            self.token_ids = {}

        def __len__(self):
            return self.size

        def add_special_tokens(self, tokens):
            added = 0
            for token in tokens["additional_special_tokens"]:
                if token not in self.token_ids:
                    self.token_ids[token] = self.size
                    self.size += 1
                    added += 1
            return added

        def convert_tokens_to_ids(self, token):
            return self.token_ids.get(token, self.unk_token_id)

        def __call__(self, text, add_special_tokens=False):
            ids = [2 + sum(map(ord, token)) % 58 for token in text.split()]
            return SimpleNamespace(input_ids=ids)

    text_config = Inkling2LoopTextConfig(
        vocab_size=68,
        unpadded_vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=4,
        rel_extent=32,
        n_layers_per_attn_res_block=1,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        eos_token_id=None,
        layer_types=["hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding"],
        loop_layer_index_start=1,
        loop_layer_index_end=3,
        loop_num_passes=2,
        loop_gate_rank=4,
        use_cache=False,
    )
    vision_config = Inkling2VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=2,
        spatial_merge_size=2,
        posemb_grid_size=4,
        projection_hidden_dim=16,
        text_hidden_size=32,
    )
    release = Inkling2LoopForConditionalGeneration(
        Inkling2LoopConfig(text_config=text_config, vision_config=vision_config, image_token_id=4)
    )
    release.save_pretrained(tmp_path, safe_serialization=True)

    tokenizer = TinyTokenizer()
    monkeypatch.setattr(smoke, "_install_runtime", lambda: None)
    monkeypatch.setattr(smoke, "MODEL_ID", str(tmp_path))
    monkeypatch.setattr(
        smoke,
        "EXPECTED_TEXT_CONFIG",
        {"model_type": "inkling2_loop_text", "hidden_size": 32, "num_hidden_layers": 4},
    )
    monkeypatch.setattr(smoke, "REMOTE_MODEL_PATH", Path(__file__).parents[1] / "kev/model.py")
    monkeypatch.setattr(smoke, "REMOTE_DEVICE_PATH", Path(__file__).parents[1] / "kev/device.py")
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: tokenizer),
    )
    monkeypatch.setitem(sys.modules, "kev", None)
    monkeypatch.setitem(sys.modules, "peft", None)

    grad_enabled = torch.is_grad_enabled()
    try:
        result = smoke.run_remote_smoke(device="cpu")
    finally:
        torch.set_grad_enabled(grad_enabled)

    assert result["status"] == "passed", result
    assert result["loaded_class"] == "Inkling2LoopForCausalLM"
    assert result["extracted_backbone"] == "model"
    assert result["lm_head_shape"] == [64, 32]
    assert result["loading_info"]["missing_count"] == 0
    assert result["loading_info"]["unexpected_visual_count"] > 0
    assert result["loading_info"]["unexpected_non_visual_count"] == 0
    assert result["embedding_class"] == "Inkling2LoopNormedEmbedding"
    assert result["embedding_rows"] == len(tokenizer)
    assert result["backbone_parameter_dtype"] == "torch.bfloat16"
    assert result["pointer_head_parameter_dtype"] == "torch.float32"
    assert result["use_cache"] is False
    assert result["independent_causal_rows"] == 1
    assert result["option_count"] == 2
    assert result["label_used_for_scoring"] is False
    assert result["training_steps"] == 0
    assert result["dataset_reads"] == 0


def test_modal_app_wrapper_delegates_without_remote_launch(monkeypatch, tmp_path):
    from scripts import inkling_weight_smoke_launch

    calls = []
    expected = {"status": "mocked"}

    def fake_launch(report_path=inkling_weight_smoke_launch.DEFAULT_REPORT_PATH, *, app_name):
        calls.append((report_path, app_name))
        return expected

    import modal_app

    monkeypatch.setattr(modal_app, "APP_NAME", "kev-smoke-nondefault")
    monkeypatch.setattr(inkling_weight_smoke_launch, "launch", fake_launch)
    report_path = tmp_path / "wrapper-report.json"
    assert modal_app.inkling_weight_smoke(report_path) is expected
    assert calls == [(report_path, "kev-smoke-nondefault")]


def test_launch_uses_app_name_for_lookup_without_creating_sandbox(monkeypatch, tmp_path):
    import modal
    from scripts import inkling_weight_smoke_launch as launcher

    marker = tmp_path / "durable-marker.json"
    report = tmp_path / "smoke-report.json"
    calls = []
    monkeypatch.setattr(launcher, "MARKER_PATH", marker)
    monkeypatch.setattr(launcher, "verify_pinned_hub_metadata", lambda: {"resolved_sha": "test"})
    monkeypatch.setattr(modal.App, "lookup", lambda name: calls.append(name) or object())
    monkeypatch.setattr(launcher, "_has_active_smoke_sandbox", lambda: True)
    monkeypatch.setattr(modal.Sandbox, "create", lambda **kwargs: pytest.fail("must not create a Sandbox"))

    with pytest.raises(RuntimeError, match="an Inkling smoke sandbox exists"):
        launcher.launch(report, app_name="kev-smoke-nondefault")
    assert calls == ["kev-smoke-nondefault"]
    assert not marker.exists()
    assert not report.exists()


def test_named_termination_uses_requested_app_name(monkeypatch):
    import modal
    from scripts import inkling_weight_smoke_launch as launcher

    calls = []

    class Sandbox:
        def terminate(self, wait):
            assert wait
            return 0

    monkeypatch.setattr(
        modal.Sandbox,
        "from_name",
        lambda app_name, name: calls.append((app_name, name)) or Sandbox(),
    )
    assert launcher.terminate(None, "inkling-sandbox", "kev-smoke-nondefault") == {"exit_code": 0}
    assert calls == [("kev-smoke-nondefault", "inkling-sandbox")]


def test_modal_wrapper_command_exits_nonzero_for_failed_report():
    command = (
        "import json, sys; "
        "from scripts import inkling_weight_smoke_launch as launcher; "
        "launcher.launch = lambda *args, **kwargs: {'status': 'failed'}; "
        "import modal_app; report = modal_app.inkling_weight_smoke(); "
        "print(json.dumps(report, sort_keys=True)); "
        "raise SystemExit(0 if report.get('status') == 'passed' else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {"status": "failed"}


def test_active_sandbox_guard_handles_sdk_listing(monkeypatch):
    import modal
    from scripts import inkling_weight_smoke_launch

    def sandbox_list(**kwargs):
        yield object()

    monkeypatch.setattr(modal.Sandbox, "list", sandbox_list)
    assert inkling_weight_smoke_launch._has_active_smoke_sandbox()


def test_metadata_404_fails_before_modal_create_and_leaves_marker_untouched(monkeypatch, tmp_path):
    import modal
    from scripts import inkling_weight_smoke_launch as launcher

    marker = tmp_path / "durable-marker.json"
    report = tmp_path / "smoke-report.json"
    monkeypatch.setattr(launcher, "MARKER_PATH", marker)

    def hub_404(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":"Invalid rev id"}'),
        )

    def unexpected(*args, **kwargs):
        raise AssertionError("Modal must not be consulted after pinned metadata fails")

    monkeypatch.setattr(launcher.urllib.request, "urlopen", hub_404)
    monkeypatch.setattr(modal.App, "lookup", unexpected)
    monkeypatch.setattr(modal.Sandbox, "create", unexpected)
    with pytest.raises(launcher.HubMetadataError, match="HTTP 404:.*Invalid rev id"):
        launcher.launch(report, app_name="test-app")

    assert not marker.exists()
    assert not report.exists()

    marker_contents = b"existing marker must remain untouched\n"
    marker.write_bytes(marker_contents)
    monkeypatch.setattr(
        launcher.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("metadata must not be queried")),
    )
    with pytest.raises(FileExistsError):
        launcher.launch(report, app_name="test-app")
    assert marker.read_bytes() == marker_contents


def test_pinned_metadata_requires_revision_and_tokenizer_files(monkeypatch):
    from scripts import inkling_weight_smoke_launch as launcher

    requests = []

    class Response:
        status = 200

        def __init__(self, body=b""):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return self.body[:size]

    def hub_ok(request, timeout):
        requests.append((request.method, request.full_url))
        if request.method == "GET":
            return Response(
                json.dumps(
                    {
                        "id": smoke.MODEL_ID,
                        "sha": smoke.MODEL_REVISION,
                        "private": False,
                        "gated": False,
                    }
                ).encode()
            )
        return Response()

    monkeypatch.setattr(launcher.urllib.request, "urlopen", hub_ok)
    result = launcher.verify_pinned_hub_metadata()
    assert result["resolved_sha"] == smoke.MODEL_REVISION
    assert result["required_files"] == {name: 200 for name in launcher.REQUIRED_HUB_FILES}
    assert [method for method, _ in requests] == ["GET", "HEAD", "HEAD", "HEAD"]


def test_required_tokenizer_404_fails_before_modal_and_marker(monkeypatch, tmp_path):
    import modal
    from scripts import inkling_weight_smoke_launch as launcher

    marker = tmp_path / "durable-marker.json"
    report = tmp_path / "smoke-report.json"
    monkeypatch.setattr(launcher, "MARKER_PATH", marker)

    class Response:
        status = 200

        def __init__(self, body=b""):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size=-1):
            return self.body[:size]

    def hub_response(request, timeout):
        if request.method == "GET":
            return Response(
                json.dumps(
                    {
                        "id": smoke.MODEL_ID,
                        "sha": smoke.MODEL_REVISION,
                        "private": False,
                        "gated": False,
                    }
                ).encode()
            )
        if request.full_url.endswith("/tokenizer.json"):
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "Not Found",
                {},
                io.BytesIO(b'{"error":"not found"}'),
            )
        return Response()

    def unexpected(*args, **kwargs):
        raise AssertionError("Modal must not be consulted after a required file 404")

    monkeypatch.setattr(launcher.urllib.request, "urlopen", hub_response)
    monkeypatch.setattr(modal.App, "lookup", unexpected)
    monkeypatch.setattr(modal.Sandbox, "create", unexpected)
    with pytest.raises(launcher.HubMetadataError, match="tokenizer.json returned HTTP 404"):
        launcher.launch(report, app_name="test-app")
    assert not marker.exists()
    assert not report.exists()
