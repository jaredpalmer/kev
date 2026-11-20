"""Tiny, synthetic compatibility gates; import only via the pinned-source launcher."""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils.other import ModulesToSaveWrapper
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Inkling2LoopMergeMessageQueries
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopTextModel

from kev.model import DecisionModel, PointerHead, SPECIAL, encode, is_hybrid, rows_of
from kev.suite import write_json
from inkling_compat import REQUIRED_GATES, exit_code, loading_info_json

SEED = 20261120
ATTENTION_TARGETS = ("q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear")


class ResearchDecisionModel(DecisionModel):
    """Reuse Kev row/head methods while deliberately bypassing its production loader."""

    def __init__(self, backbone, head, pad_id):
        nn.Module.__init__(self)
        self.lm, self.head = backbone, head
        self.device, self.pad_id = torch.device("cpu"), pad_id
        self.hybrid, self.option_isolation = is_hybrid(backbone.config), False


def run_gate(report, name, run, required=True):
    gate = {"name": name, "status": "untested", "required": required}
    report["gates"].append(gate)
    try:
        result = run()
        if not isinstance(result, dict):
            raise TypeError("gate must return JSON-ready fields")
        gate.update(result)
        if "status" not in result:
            gate["status"] = "pass"
    except Exception as exc:
        gate.update(status="fail", error=f"{type(exc).__name__}: {exc}")
    return gate


def max_delta(left, right):
    if left.numel() == 0:
        return 0.0
    value = float((left.detach().float() - right.detach().float()).abs().max().item())
    return value if math.isfinite(value) else None


def tiny_tokenizer():
    words = ("<unk>", "<pad>", "alpha", "beta", "gamma", "question", "option", "delta", "epsilon", "state")
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>")


def tiny_model(vocab_size, pad_id):
    config = Inkling2LoopTextConfig(
        vocab_size=vocab_size, unpadded_vocab_size=vocab_size, hidden_size=32, num_hidden_layers=13,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, swa_num_attention_heads=2,
        swa_num_key_value_heads=1, swa_head_dim=16, sliding_window_size=8, d_rel=4, rel_extent=32,
        n_layers_per_attn_res_block=6, moe_intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2,
        pad_token_id=pad_id, eos_token_id=None,
        layer_types=["hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding" for i in range(13)],
        loop_config={"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4}, use_cache=False,
    )
    config.validate_architecture()
    return Inkling2LoopTextModel(config).to(dtype=torch.float32)


def record(state="alpha beta", first="alpha"):
    return {"state": state, "questions": [
        {"instr": "question alpha", "options": [f"option {first}", "option beta"], "label": 0},
        {"instr": "question gamma", "options": ["option gamma", "option delta"], "label": 1},
    ]}


def flatten_logits(rows):
    return torch.cat([x.detach().float().reshape(-1) for qs in rows for x in qs])


def cached_tokenizer_metadata():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "thinkingmachines/Inkling-2-Mini", revision="6e7c8248f75265bdd4fbab5273304243b5d8c2cb",
            local_files_only=True, trust_remote_code=False,
        )
    except Exception as exc:
        return {"status": "untested", "reason": f"not cached ({type(exc).__name__})", "weights_loaded": False}
    vocab = tokenizer.get_vocab()
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL}
    missing = [token for token, value in ids.items() if token not in vocab or value == tokenizer.unk_token_id]
    return {"status": "pass", "revision": "6e7c8248f75265bdd4fbab5273304243b5d8c2cb",
            "boundary_ids": ids, "missing_specials": missing, "weights_loaded": False}


def row_execution(tokenizer, model, pad_id):
    encoded = encode(tokenizer, record())
    state, state_pos, branches = rows_of(encoded)
    model.config.use_cache = False
    view = ResearchDecisionModel(model, PointerHead(model.config.hidden_size, dp=16), pad_id).eval()
    alternate = encode(tokenizer, record(first="gamma"))
    with torch.no_grad():
        first = view.forward_rows_batch([encoded], shared_prefix=False)[0][1]
        changed = view.forward_rows_batch([alternate], shared_prefix=False)[0][1]
        longer = encode(tokenizer, record(state="alpha beta gamma delta epsilon"))
        batched = view.forward_rows_batch([encoded, longer], shared_prefix=False)
        separate = [view.forward_rows_batch([x], shared_prefix=False)[0] for x in (encoded, longer)]
        packed_a, packed_b = view.forward_batch([encoded], shared_prefix=False)[0][1], view.forward_batch([alternate], shared_prefix=False)[0][1]
        hidden_a, hidden_b = view.hidden_batch([encoded])[0], view.hidden_batch([alternate])[0]
    independent_delta = max_delta(first, changed)
    batch_delta = max(max_delta(a, b) for ga, gb in zip(batched, separate) for a, b in zip(ga, gb))
    lo, hi = encoded["seg"].index(2), encoded["decide_idx"][1] + 1
    packed_logit, packed_hidden = max_delta(packed_a, packed_b), max_delta(hidden_a[lo:hi], hidden_b[lo:hi])
    packed_leak = max(packed_logit or 0.0, packed_hidden or 0.0) > 1e-6
    checks = {"independent_rows": independent_delta is not None and independent_delta <= 1e-6,
              "padding_batching": batch_delta is not None and batch_delta <= 1e-6, "packed_negative_control": packed_leak}
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
            "questions": len(branches), "state_tokens": len(state), "state_positions": len(state_pos),
            "causal_row_logit_delta": independent_delta, "padding_batch_max_delta": batch_delta,
            "packed_changed_branch_logit_delta": packed_logit, "packed_hidden_delta": packed_hidden,
            "tolerance": 1e-6, "shared_prefix": False, "use_cache": model.config.use_cache,
            "helpers": ["kev.model.encode", "kev.model.rows_of", "DecisionModel.forward_rows_batch"],
            "packed_path": "negative control only"}, encoded


def converter_numeric():
    op = Inkling2LoopDequantizeExperts()
    names = ["w13_weight", "w13_weight_scale", "w13_weight_amax", "w13_weight_outer_scale_denominator"]
    packed = torch.full((2, 1, 8), 0x21, dtype=torch.uint8)
    scale = torch.full((2, 1, 1), 2.0, dtype=torch.float8_e4m3fn)
    amax, denominator = torch.tensor([8.0, 6.0]), torch.tensor([4.0, 3.0])
    nv = op.convert(dict(zip(names, [packed, scale, amax, denominator])), source_patterns=names, target_patterns=["gate_up_proj"])["gate_up_proj"]
    e2m1 = [0.0, .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]
    nv_delta = max_delta(nv, torch.tensor([e2m1[1], e2m1[2]] * 8).repeat(2, 1, 1) * 4)
    names = ["w2_weight", "w2_weight_scale", "w2_weight_amax", "w2_weight_outer_scale_denominator"]
    weight = torch.linspace(-2, 2, 64).reshape(2, 1, 32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[[127]], [[128]]], dtype=torch.uint8)
    mx = op.convert({names[0]: weight, names[1]: scales}, source_patterns=names, target_patterns=["down_proj"])["down_proj"]
    mx_delta = max_delta(mx, torch.stack([weight[i].float() * 2.0**i for i in range(2)]))
    q, kv = torch.arange(16).reshape(4, 4), torch.arange(100, 116).reshape(4, 4)
    merged = Inkling2LoopMergeMessageQueries().convert({"queries": q, "kv_queries": kv},
        source_patterns=["queries", "kv_queries"], target_patterns=["message_input.query"])["message_input.query"]
    reference = torch.stack([torch.stack([q[2*i], kv[2*i], kv[2*i+1], q[2*i+1]]) for i in range(2)])
    query_delta = max_delta(merged, reference)
    ok = nv_delta == mx_delta == query_delta == 0
    return {"status": "pass" if ok else "fail", "nvfp4_max_abs_delta": nv_delta,
            "mxfp8_max_abs_delta": mx_delta, "message_query_max_abs_delta": query_delta}


def embedding_metrics(base, wrapped, token_ids):
    original = base.get_input_embeddings()
    with torch.no_grad():
        normalized, output = original(token_ids), wrapped.get_input_embeddings()(token_ids)
        raw = F.embedding(token_ids, original.weight, original.padding_idx, original.max_norm,
                          original.norm_type, original.scale_grad_by_freq, original.sparse)
        restored = original.embed_norm(output)
    return {"peft_vs_normed_base_max_abs_delta": max_delta(output, normalized),
            "peft_vs_raw_embedding_max_abs_delta": max_delta(output, raw),
            "norm_restored_peft_vs_base_max_abs_delta": max_delta(restored, normalized),
            "raw_vs_normed_base_max_abs_delta": max_delta(raw, normalized)}


def peft_probe(model, tokenizer, special_ids, encoded, workdir):
    head = PointerHead(model.config.hidden_size, dp=16).eval()
    base = copy.deepcopy(model).eval()
    baseline = ResearchDecisionModel(base, head, tokenizer.pad_token_id).eval()
    with torch.no_grad(): base_logits = flatten_logits(baseline.forward_rows_batch([encoded], shared_prefix=False))
    config = LoraConfig(r=2, lora_alpha=4, target_modules=list(ATTENTION_TARGETS), bias="none",
                        trainable_token_indices={"embed_tokens": list(special_ids.values())})
    token_adapter = get_peft_model(copy.deepcopy(model), config).eval()
    ids = torch.tensor([*special_ids.values(), 2, 3], dtype=torch.long)
    before_embed = embedding_metrics(model, token_adapter, ids)
    with torch.no_grad():
        token_logits = flatten_logits(ResearchDecisionModel(token_adapter, copy.deepcopy(head), tokenizer.pad_token_id).eval().forward_rows_batch([encoded], shared_prefix=False))
    before_logits = max_delta(base_logits, token_logits)
    untrained_path = workdir / "untrained-adapter"
    token_adapter.save_pretrained(untrained_path)
    restored_token = PeftModel.from_pretrained(copy.deepcopy(model), untrained_path).eval()
    after_embed = embedding_metrics(model, restored_token, ids)
    with torch.no_grad():
        after_logits = max_delta(base_logits, flatten_logits(ResearchDecisionModel(restored_token, copy.deepcopy(head), tokenizer.pad_token_id).eval().forward_rows_batch([encoded], shared_prefix=False)))
    parity = all(x is not None and x <= 1e-6 for x in (before_embed["peft_vs_normed_base_max_abs_delta"], before_logits,
        after_embed["peft_vs_normed_base_max_abs_delta"], after_logits))
    control = all(x["peft_vs_raw_embedding_max_abs_delta"] <= 1e-6 and x["norm_restored_peft_vs_base_max_abs_delta"] <= 1e-6
        and x["raw_vs_normed_base_max_abs_delta"] > 1e-6 for x in (before_embed, after_embed))

    return {"status": "pass" if parity else "fail",
            "selective_token_measurement": "embedding output immediately before backbone; zero-delta tolerance 1e-6",
            "selective_token_parity_before": {"embedding": before_embed, "logits_max_abs_delta": before_logits},
            "selective_token_parity_after_reload": {"embedding": after_embed, "logits_max_abs_delta": after_logits},
            "norm_bypass_control": control,
            "root_cause": {"peft": "TrainableTokensLayer.forward_adapters uses F.embedding on raw base weights (peft/tuners/trainable_tokens/layer.py:232-264)",
                           "inkling": "Inkling2LoopNormedEmbedding.forward applies embed_norm (modeling_inkling2_loop.py:1240-1246)"}}


def modules_to_save_probe(model, tokenizer, special_ids, encoded, workdir):
    head = PointerHead(model.config.hidden_size, dp=16).eval()
    baseline = ResearchDecisionModel(copy.deepcopy(model).eval(), head, tokenizer.pad_token_id).eval()
    with torch.no_grad():
        baseline_logits = flatten_logits(baseline.forward_rows_batch([encoded], shared_prefix=False))
    config = LoraConfig(r=2, lora_alpha=4, target_modules=list(ATTENTION_TARGETS),
                        modules_to_save=["embed_tokens"], bias="none")
    candidate = get_peft_model(copy.deepcopy(model), config).eval()
    input_embedding = candidate.get_input_embeddings()
    if not isinstance(input_embedding, ModulesToSaveWrapper):
        raise TypeError(f"expected ModulesToSaveWrapper, got {type(input_embedding).__name__}")
    saved_embedding = input_embedding.modules_to_save[input_embedding.active_adapters[0]]
    if not isinstance(saved_embedding, type(model.get_input_embeddings())) or not hasattr(saved_embedding, "embed_norm"):
        raise TypeError("modules_to_save did not preserve the complete normed embedding module")
    delimiter_ids = list(special_ids.values())
    probe_ids = torch.tensor([*delimiter_ids, 2, 3], dtype=torch.long)
    with torch.no_grad():
        original_embedding = model.get_input_embeddings()(probe_ids)
        initial_embedding = candidate.get_input_embeddings()(probe_ids)
        initial_logits = flatten_logits(ResearchDecisionModel(candidate, copy.deepcopy(head), tokenizer.pad_token_id)
                                       .eval().forward_rows_batch([encoded], shared_prefix=False))
    initial_embedding_delta = max_delta(initial_embedding, original_embedding)
    initial_logit_delta = max_delta(initial_logits, baseline_logits)
    target_names = sorted(name for name, module in model.named_modules()
                          if isinstance(module, nn.Linear) and name.endswith(ATTENTION_TARGETS))
    injected_names = sorted(name.removeprefix("base_model.model.") for name, module in candidate.named_modules()
                            if hasattr(module, "lora_A") and hasattr(module, "lora_B"))
    target_match = bool(target_names) and injected_names == target_names

    train_head = copy.deepcopy(head)
    view = ResearchDecisionModel(candidate, train_head, tokenizer.pad_token_id)
    view.train()
    frozen = {name: p.detach().clone() for name, p in candidate.named_parameters() if not p.requires_grad}
    lora = [p for name, p in candidate.named_parameters() if "lora_" in name and p.requires_grad]
    lora_before = [p.detach().clone() for p in lora]
    trainable = [p for p in candidate.parameters() if p.requires_grad] + list(train_head.parameters())
    embedding_weight = saved_embedding.weight
    rows_before = embedding_weight[delimiter_ids].detach().clone()
    optimizer = torch.optim.SGD(trainable, lr=.01)
    optimizer.zero_grad(set_to_none=True)
    logits = view.forward_rows_batch([encoded], shared_prefix=False)[0]
    loss = F.cross_entropy(torch.stack(logits), torch.tensor([0, 1]))
    loss.backward()
    lora_grads = [p.grad for p in lora if p.grad is not None]
    lora_grad_norm = math.sqrt(sum(float(g.float().pow(2).sum()) for g in lora_grads)) if lora_grads else 0.0
    lora_gradients_ok = bool(lora_grads) and all(bool(torch.isfinite(g).all()) for g in lora_grads) and lora_grad_norm > 0
    row_grads = embedding_weight.grad[delimiter_ids]
    grad_norms = [float(torch.linalg.vector_norm(row.float()).item()) for row in row_grads]
    gradients_ok = all(math.isfinite(value) and value > 0 for value in grad_norms)
    optimizer.step()
    lora_update = max((max_delta(before, after) for before, after in zip(lora_before, lora)), default=0.0)
    row_updates = [float(torch.linalg.vector_norm(row.float()).item())
                   for row in (embedding_weight[delimiter_ids] - rows_before)]
    updates_ok = all(math.isfinite(value) and value > 0 for value in row_updates)
    after_params = dict(candidate.named_parameters())
    frozen_delta = max((max_delta(value, after_params[name]) for name, value in frozen.items()), default=0.0)

    view.eval()
    adapter_path = workdir / "modules-to-save-adapter"
    candidate.save_pretrained(adapter_path)
    head_path = workdir / "modules-to-save-head.experimental.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in train_head.state_dict().items()}, str(head_path))
    tokenizer_path = workdir / "modules-to-save-tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    reloaded_ids = {token: loaded_tokenizer.convert_tokens_to_ids(token) for token in SPECIAL}
    loaded = PeftModel.from_pretrained(copy.deepcopy(model), adapter_path).eval()
    loaded_head = PointerHead(model.config.hidden_size, dp=16).eval()
    loaded_head.load_state_dict(load_file(str(head_path)))
    with torch.no_grad():
        saved_embedding_delta = max_delta(candidate.get_input_embeddings()(probe_ids), loaded.get_input_embeddings()(probe_ids))
        expected_logits = flatten_logits(view.eval().forward_rows_batch([encoded], shared_prefix=False))
        loaded_logits = flatten_logits(ResearchDecisionModel(loaded, loaded_head, tokenizer.pad_token_id)
                                       .eval().forward_rows_batch([encoded], shared_prefix=False))
    reload_logit_delta = max_delta(expected_logits, loaded_logits)
    reload_ok = saved_embedding_delta == 0 and reload_logit_delta == 0 and reloaded_ids == special_ids
    parity_ok = initial_embedding_delta is not None and initial_embedding_delta <= 1e-6 and initial_logit_delta is not None and initial_logit_delta <= 1e-6
    frozen_ok = frozen_delta == 0
    params_saved = sum(parameter.numel() for parameter in saved_embedding.parameters())
    lora_ok = lora_gradients_ok and lora_update is not None and lora_update > 0
    ok = parity_ok and target_match and lora_ok and gradients_ok and updates_ok and frozen_ok and reload_ok
    return {"status": "pass" if ok else "fail", "strategy": "attention LoRA + modules_to_save=['embed_tokens']",
            "initial_embedding_max_abs_delta": initial_embedding_delta, "initial_logit_max_abs_delta": initial_logit_delta,
            "initial_parity_tolerance": 1e-6, "embedding_module_class": type(saved_embedding).__name__,
            "saved_embedding_parameter_count": params_saved, "delimiter_ids": special_ids,
            "delimiter_gradient_norms": grad_norms, "delimiter_update_norms": row_updates,
            "all_delimiter_rows_have_finite_nonzero_gradients": gradients_ok,
            "all_delimiter_rows_updated": updates_ok, "frozen_non_target_base_max_abs_delta": frozen_delta,
            "attention_lora": {"targets_expected": len(target_names), "targets_injected": len(injected_names),
                "targets_match": target_match, "finite_nonzero_gradients": lora_gradients_ok,
                "gradient_norm": lora_grad_norm, "max_abs_update": lora_update},
            "embedding_reload_max_abs_delta": saved_embedding_delta,
            "head_reload_logit_max_abs_delta": reload_logit_delta, "tokenizer_id_contract_preserved": reloaded_ids == special_ids,
            "modules_to_save_caveat": "trains and stores the entire normed vocabulary embedding, not only five delimiters; more memory/checkpoint storage than selective tokens; random synthetic data only, no quality claim",
            "optimizer_steps": 1, "loss": float(loss.detach())}


def finalize(report):
    for name in REQUIRED_GATES:
        if not any(g["name"] == name for g in report["gates"]):
            report["gates"].append({"name": name, "status": "untested", "required": True, "reason": "not reached"})
    failed = [g["name"] for g in report["gates"] if g.get("required") and g["status"] == "fail"]
    untested = [g["name"] for g in report["gates"] if g.get("required") and g["status"] == "untested"]
    optional = [g["name"] for g in report["gates"] if not g.get("required") and g["status"] == "untested"]
    report["summary"] = {"status": "failed_required_gates" if failed or untested else "required_gates_passed_partial_source_load",
                         "passed": sum(g["status"] == "pass" for g in report["gates"]), "failed": len(failed),
                         "untested": len(untested) + len(optional), "required_failures": failed,
                         "required_untested": untested, "optional_untested": optional}
    return exit_code(report)


def run(report_path, source_info):
    torch.set_num_threads(1); torch.manual_seed(SEED)
    report = {"schema_version": 1, "source": source_info,
        "runtime": {"python": sys.version.split()[0], "torch": str(torch.__version__), "transformers": transformers.__version__,
                    "peft": peft.__version__, "device": "cpu", "seed": SEED},
        "scope": {"weights": "tiny random only; no pretrained weights", "data": "synthetic in-memory records only; this probe reads no datasets",
            "training": "one CPU update on random initialization; no pretrained fine-tuning", "quality_verdict": "none",
            "execution": "use_cache=False; independent causal rows; no shared_prefix or serving integration",
            "source_load": "canonical synthetic text-only round trip tested; original multimodal release checkpoint load untested",
            "checkpoint": "temporary experimental adapter/head only; no loadable Kev checkpoint"}, "gates": []}
    run_gate(report, "source_pin", lambda: {"revision": source_info["revision"], "clean_worktree": True,
        "transformers_file": source_info["transformers_file"]})
    state = {}
    def architecture():
        tokenizer = tiny_tokenizer(); torch.manual_seed(SEED)
        model = tiny_model(len(tokenizer), tokenizer.pad_token_id)
        state.update(tokenizer=tokenizer, model=model, initial_vocab=len(tokenizer))
        c = model.config
        ok = c.loop_num_passes == 3 and c.loop_layer_index_start == 6 and c.loop_layer_index_end == 12 and c.num_hidden_layers == 13 and c.total_num_hidden_layers == 25 and {"hybrid", "hybrid_sliding"}.issubset(set(c.layer_types)) and c.use_cache is False
        return {"status": "pass" if ok else "fail", "model": "Inkling2LoopTextModel", "hidden_size": c.hidden_size,
                "physical_layers": 13, "pre_loop": 6, "loop_core": [6, 12], "passes": 3, "post_loop": 1,
                "total_layer_visits": c.total_num_hidden_layers, "layer_types": list(c.layer_types), "use_cache": c.use_cache}
    run_gate(report, "tiny_architecture", architecture)
    if "model" in state:
        tokenizer, model, initial_vocab = state["tokenizer"], state["model"], state["initial_vocab"]
        with tempfile.TemporaryDirectory(prefix="inkling-compat-") as temp:
            workdir = Path(temp); ids_state = {}
            def tokens():
                try:
                    public = cached_tokenizer_metadata()
                except Exception as exc:
                    public = {"status": "untested", "reason": type(exc).__name__, "weights_loaded": False}
                added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL})
                ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL}; ids_state.update(ids)
                add_ok = added == 5 and len(set(ids.values())) == 5 and len(tokenizer) - initial_vocab == 5
                emb = model.get_input_embeddings(); ordinary = torch.tensor([[2, 3]])
                with torch.no_grad(): before = emb(ordinary).clone()
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                emb = model.get_input_embeddings()
                with torch.no_grad(): old_delta = max_delta(before, emb(ordinary)); norm_delta = max_delta(emb(torch.tensor(list(ids.values()))), emb.embed_norm(emb.weight[list(ids.values())]))
                path = workdir / "tokenizer"; tokenizer.save_pretrained(path)
                loaded = AutoTokenizer.from_pretrained(path, local_files_only=True)
                roundtrip = {token: loaded.convert_tokens_to_ids(token) for token in SPECIAL}
                resize_ok = type(emb).__name__ == "Inkling2LoopNormedEmbedding" and hasattr(emb, "embed_norm") and emb.num_embeddings == len(tokenizer) and old_delta == norm_delta == 0
                ok = add_ok and resize_ok and roundtrip == ids
                return {"status": "pass" if ok else "fail", "cached_public_tokenizer": public, "added_count": added,
                    "boundary_ids": ids, "roundtrip_ids": roundtrip, "resize_class": type(emb).__name__,
                    "preserved_existing_rows_delta": old_delta, "normed_new_rows_delta": norm_delta, "resize_contract": resize_ok}
            token_gate = run_gate(report, "tokenizer_embedding", tokens)
            def canonical():
                path = workdir / "canonical-text-model"; model.save_pretrained(path, safe_serialization=True)
                loaded, info = Inkling2LoopTextModel.from_pretrained(path, local_files_only=True, dtype=torch.float32, output_loading_info=True)
                loading = loading_info_json(info)
                delta = max((max_delta(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()), default=0.0)
                no_mm = not hasattr(loaded, "lm_head") and not hasattr(loaded, "vision_tower")
                ok = not any(loading.values()) and delta == 0 and no_mm
                return {"status": "pass" if ok else "fail", **loading, "state_max_abs_delta": delta, "no_lm_head_or_vision": no_mm}
            run_gate(report, "canonical_text_loader", canonical)
            if token_gate["status"] == "pass":
                def rows():
                    data, encoded = row_execution(tokenizer, model, tokenizer.pad_token_id)
                    state["encoded"] = encoded
                    return data
                row_gate = run_gate(report, "row_execution", rows)
                if row_gate["status"] == "pass":
                    run_gate(report, "selective_token_parity", lambda: peft_probe(model, tokenizer, ids_state, state["encoded"], workdir))
                    run_gate(report, "peft_modules_to_save", lambda: modules_to_save_probe(model, tokenizer, ids_state, state["encoded"], workdir))
                else:
                    report["gates"].append({"name": "selective_token_parity", "status": "untested", "required": True, "reason": "row gate failed"})
                    report["gates"].append({"name": "peft_modules_to_save", "status": "untested", "required": True, "reason": "row gate failed"})
            else:
                report["gates"].append({"name": "row_execution", "status": "untested", "required": True, "reason": "tokenizer gate failed"})
                report["gates"].append({"name": "selective_token_parity", "status": "untested", "required": True, "reason": "tokenizer gate failed"})
                report["gates"].append({"name": "peft_modules_to_save", "status": "untested", "required": True, "reason": "tokenizer gate failed"})
            run_gate(report, "converter_numeric", converter_numeric)
            report["gates"].append({"name": "original_release_loader", "status": "untested", "required": False,
                "reason": "synthetic converter operations only; no complete original-format checkpoint fixture"})
    code = finalize(report)
    report_path.parent.mkdir(parents=True, exist_ok=True); write_json(report_path, report)
    print(json.dumps(report["summary"], sort_keys=True)); print(f"report={report_path}")
    return code
