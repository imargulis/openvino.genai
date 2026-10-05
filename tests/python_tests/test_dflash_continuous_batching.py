# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DFlash continuous batching with several requests, on the stateful SDPA and the PagedAttention draft backends.

Greedy speculative decoding must reproduce the target alone, whatever the draft proposes. Tiny random drafts
reject almost every candidate. In the constant setup every token embedding carries a large channel 0 that only
token 0 reads out, so the target and all drafts predict token 0 and every candidate is accepted.
"""

import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("optimum.exporters.openvino")

TINY_QWEN3 = "optimum-intel-internal-testing/tiny-random-qwen3"
NUM_ASSISTANT_TOKENS = 4
MAX_NEW_TOKENS = 40
# prompts cross the 32-token blocks of the CPU draft KV cache
PROMPT_LENGTHS = (5, 37, 70)
PRECISION = {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32"}
VARIANTS = ("dflash", "dspark", "dflash2")
BACKENDS = ("SDPA", "PA")


def _keep_first_channel(norm):
    # ties would leave the argmax to TopK and sampler tie-breaking, which differ
    with torch.no_grad():
        norm.weight.zero_()
        norm.weight[0] = 1.0


def _save_checkpoint(model, output_dir: Path, architectures):
    from transformers import AutoTokenizer

    model.save_pretrained(output_dir)
    model.config.architectures = architectures
    model.config.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(TINY_QWEN3).save_pretrained(output_dir)


def _build_target(output_dir: Path, constant: bool):
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = AutoConfig.from_pretrained(TINY_QWEN3)
    if constant:
        config.tie_word_embeddings = False
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(config).eval()
    if constant:
        with torch.no_grad():
            model.model.embed_tokens.weight[:, 0] = 50.0
            model.lm_head.weight[:, 0] = 0.0
            model.lm_head.weight[0, 0] = 1.0
        _keep_first_channel(model.model.norm)
    _save_checkpoint(model, output_dir, ["Qwen3ForCausalLM"])
    return config


def _build_draft(output_dir: Path, variant: str, target_config, constant: bool):
    from optimum.exporters.openvino.model_patcher import (
        Qwen3DFlash2ForCausalLM,
        Qwen3DFlashForCausalLM,
        Qwen3DSparkForCausalLM,
    )

    config = deepcopy(target_config)
    num_target_layers = target_config.num_hidden_layers
    target_layer_ids = [0, num_target_layers - 1]
    mask_token_id = config.vocab_size - 1
    config.num_hidden_layers = 2
    config.num_target_layers = num_target_layers
    if getattr(config, "layer_types", None):
        config.layer_types = list(config.layer_types[: config.num_hidden_layers])
    config.is_causal = False
    if variant == "dflash":
        architectures = ["DFlashDraftModel"]
        config.dflash_config = {"mask_token_id": mask_token_id, "target_layer_ids": target_layer_ids}
        model_class = Qwen3DFlashForCausalLM
    elif variant == "dflash2":
        architectures = ["DFlash2DraftModel"]
        config.dflash_config = {
            "conv_group_size": 8 if config.hidden_size % 8 == 0 else 1,
            "conv_kernel_size": 2,
            "mask_token_id": mask_token_id,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": target_layer_ids,
        }
        model_class = Qwen3DFlash2ForCausalLM
    else:
        architectures = ["Qwen3DSparkModel"]
        config.markov_rank = 0
        config.mask_token_id = mask_token_id
        config.target_layer_ids = target_layer_ids
        model_class = Qwen3DSparkForCausalLM
    config.architectures = architectures
    torch.manual_seed(0)
    model = model_class(config).eval()
    if constant:
        # the draft reads out through the target embedding and LM head
        _keep_first_channel(model.norm)
    _save_checkpoint(model, output_dir, architectures)


def _export(checkpoint_dir: Path, output_dir: Path):
    from optimum.exporters.openvino import main_export

    main_export(str(checkpoint_dir), output_dir, task="text-generation-with-past", convert_tokenizer=True)
    return output_dir


@pytest.fixture(scope="module", params=("random", "constant"))
def dflash_models(request, tmp_path_factory):
    constant = request.param == "constant"
    root = tmp_path_factory.mktemp(f"dflash_{request.param}")
    target_config = _build_target(root / "target_hf", constant)
    models = {"target": _export(root / "target_hf", root / "target"), "constant": constant}
    for variant in VARIANTS:
        _build_draft(root / f"{variant}_hf", variant, target_config, constant)
        models[variant] = _export(root / f"{variant}_hf", root / variant)
    return models


def _prompts(lengths=PROMPT_LENGTHS):
    rng = np.random.default_rng(7)
    return [ov.Tensor(rng.integers(10, 1000, size=(1, length), dtype=np.int64)) for length in lengths]


def _generation_config():
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.ignore_eos = True
    config.num_assistant_tokens = NUM_ASSISTANT_TOKENS
    return config


def _target_config():
    config = _generation_config()
    config.num_assistant_tokens = 0
    return config


def _scheduler_config(max_num_batched_tokens=256):
    config = ov_genai.SchedulerConfig()
    config.max_num_batched_tokens = max_num_batched_tokens
    config.enable_prefix_caching = False
    return config


def _pipeline(models, variant=None, backend="SDPA", scheduler_config=None):
    properties = dict(PRECISION)
    if variant is not None:
        draft_properties = dict(PRECISION, ATTENTION_BACKEND=backend)
        properties["draft_model"] = ov_genai.draft_model(models[variant], "CPU", **draft_properties)
        if variant == "dflash2":
            properties["selector_model"] = ov_genai.selector_model(
                models[variant] / "openvino_selector_model.xml", "CPU", **PRECISION
            )
    return ov_genai.ContinuousBatchingPipeline(
        models["target"], scheduler_config or _scheduler_config(), "CPU", properties
    )


def _generate(pipe, prompts, config):
    results = pipe.generate(prompts, [config] * len(prompts))
    return [result.m_generation_ids[0] for result in results], results


@pytest.fixture(scope="module")
def target_outputs(dflash_models):
    outputs, _ = _generate(_pipeline(dflash_models), _prompts(), _target_config())
    assert all(len(tokens) == MAX_NEW_TOKENS for tokens in outputs)
    return outputs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_requests_drafted_together_match_target(dflash_models, target_outputs, variant, backend):
    pipe = _pipeline(dflash_models, variant, backend)
    outputs, results = _generate(pipe, _prompts(), _generation_config())
    assert outputs == target_outputs

    metrics = results[0].extended_perf_metrics
    assert metrics.get_num_draft_tokens() > 0
    if dflash_models["constant"]:
        assert metrics.get_num_accepted_tokens() == metrics.get_num_draft_tokens()
    draft_stages = metrics.draft_model_metrics.raw_metrics.m_batch_sizes
    draft_steps = metrics.draft_step_metrics.raw_metrics.m_batch_sizes
    if backend == "PA":
        # one draft inference per step for all requests
        assert len(draft_stages) == len(draft_steps)
        assert max(draft_stages) > NUM_ASSISTANT_TOKENS
    else:
        assert len(draft_stages) > len(draft_steps)

    single_outputs = [_generate(pipe, [prompt], _generation_config())[0][0] for prompt in _prompts()]
    assert single_outputs == target_outputs


@pytest.mark.parametrize("variant", VARIANTS)
def test_paged_draft_proposes_as_stateful_draft(dflash_models, variant):
    accepted = {}
    for backend in BACKENDS:
        _, results = _generate(_pipeline(dflash_models, variant, backend), _prompts(), _generation_config())
        metrics = results[0].extended_perf_metrics
        accepted[backend] = (metrics.get_num_draft_tokens(), metrics.get_num_accepted_tokens())
    assert accepted["PA"] == accepted["SDPA"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_staggered_requests_match_target(dflash_models, target_outputs, variant, backend):
    pipe = _pipeline(dflash_models, variant, backend)
    prompts = _prompts()
    handles = {}
    # a new request arrives while the earlier ones are prefilling, validating or drafting
    arrival_steps = {0: 0, 1: 3, 2: 7}
    step = 0
    while len(handles) < len(prompts) or pipe.has_non_finished_requests():
        for request_id, arrival in arrival_steps.items():
            if arrival == step:
                handles[request_id] = pipe.add_request(request_id, prompts[request_id], _generation_config())
        if pipe.has_non_finished_requests():
            pipe.step()
        step += 1
    outputs = [handles[request_id].read_all()[0].generated_ids for request_id in range(len(prompts))]
    assert outputs == target_outputs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_token_budget_limits_drafted_candidates(dflash_models, target_outputs, variant, backend):
    # one full validation window fits, so later requests draft fewer candidates or wait
    scheduler_config = _scheduler_config(max_num_batched_tokens=NUM_ASSISTANT_TOKENS + 3)
    pipe = _pipeline(dflash_models, variant, backend, scheduler_config)
    outputs, _ = _generate(pipe, _prompts(), _generation_config())
    assert outputs == target_outputs


def test_draft_with_token_type_ids_requires_layout_marker(dflash_models, tmp_path):
    unmarked_dir = tmp_path / "unmarked"
    shutil.copytree(dflash_models["dflash"], unmarked_dir)
    xml_path = unmarked_dir / "openvino_model.xml"
    xml = xml_path.read_text(encoding="utf-8")
    unmarked_xml = re.sub(r'\s*<input_layout value="per_row"\s*/>', "", xml)
    assert unmarked_xml != xml
    xml_path.write_text(unmarked_xml, encoding="utf-8")
    with pytest.raises(RuntimeError, match="no dflash/input_layout metadata"):
        _pipeline({**dflash_models, "dflash": unmarked_dir}, "dflash")
