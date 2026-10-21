// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"
#include "model_desc.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> make_annotated_stateful_sdpa_model() {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});
    auto embedding_weights =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 4}, std::vector<float>(16 * 4, 1.0f));
    auto embeddings =
        std::make_shared<ov::op::v8::Gather>(embedding_weights,
                                             input_ids,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->set_friendly_name("beam_idx");
    beam_idx->output(0).set_names({"beam_idx"});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 1, -1});
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto make_cache = [&](const std::string& variable_id) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(4), ov::element::f32, variable_id});
        auto initial = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 4}, {0.0f});
        auto read = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
        auto past = std::make_shared<ov::op::v8::Gather>(read, beam_idx, gather_axis);
        auto current = std::make_shared<ov::op::v1::Reshape>(embeddings, reshape_pattern, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, current}, 1);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto transposed = std::make_shared<ov::op::v1::Transpose>(concat, transpose_order);
        return std::make_tuple(transposed, assign);
    };

    auto [key, key_assign] = make_cache("dflash_test_key_cache");
    auto [value, value_assign] = make_cache("dflash_test_value_cache");
    auto query =
        std::make_shared<ov::op::v0::Unsqueeze>(embeddings,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto mask = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2, 3}, {0.0f});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);

    auto attention_transposed = std::make_shared<ov::op::v1::Transpose>(sdpa, transpose_order);
    auto attention_hidden =
        std::make_shared<ov::op::v1::Reshape>(attention_transposed,
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, -1}),
                                              true);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 4}, {1.0f});
    auto decoder_layer = std::make_shared<ov::op::v1::Add>(embeddings, attention_hidden);
    decoder_layer->set_friendly_name("decoder_layer_0");
    auto final_norm = std::make_shared<ov::op::v1::Multiply>(decoder_layer, bias);
    final_norm->set_friendly_name("final_norm");
    auto lm_head_weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, {1.0f});
    auto lm_head = std::make_shared<ov::op::v0::MatMul>(final_norm, lm_head_weights);
    lm_head->set_friendly_name("lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(lm_head);
    logits_result->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{logits_result},
                                             ov::SinkVector{key_assign, value_assign},
                                             ov::ParameterVector{input_ids, beam_idx});
    model->set_rt_info(
        std::string(
            R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0},"1":{"producer":"final_norm","output_index":0}}})"),
        "hidden_states_decoder_layers");
    return model;
}

std::shared_ptr<ov::Model> make_eagle3_pattern_hidden_state_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    ov::Output<ov::Node> current = input->output(0);

    for (size_t layer_idx = 0; layer_idx < 5; ++layer_idx) {
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 2, 4}, 1.0f);
        auto multiplied = std::make_shared<ov::op::v1::Multiply>(current, scale);
        auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4, 4}, 1.0f);
        auto projection = std::make_shared<ov::op::v0::MatMul>(multiplied, weight);
        auto layer = std::make_shared<ov::op::v1::Add>(current, projection);
        layer->set_friendly_name("model.layers." + std::to_string(layer_idx) + "/residual_add");
        current = layer->output(0);
    }

    auto result = std::make_shared<ov::op::v0::Result>(current);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

size_t count_outputs_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    size_t count = 0;
    for (const auto& output : model->outputs()) {
        if (output.get_names().count(name) != 0) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST(DFlashModelTransforms, AppliesAndExtractsDraftRtInfo) {
    auto model = make_annotated_stateful_sdpa_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("1,12,23,34,45"), {"dflash", "target_layer_ids"});

    // Verify default legacy candidate_position_offset when omitted
    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);

    ASSERT_TRUE(properties.at("dflash_mode").as<bool>());
    ASSERT_EQ(properties.at("dflash_mask_token_id").as<int64_t>(), 151669);
    ASSERT_EQ(properties.at("dflash_target_layer_ids").as<std::vector<int32_t>>(),
              (std::vector<int32_t>{1, 12, 23, 34, 45}));

    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
    ASSERT_EQ(rt_info.mask_token_id, 151669);
    ASSERT_EQ(rt_info.target_layer_ids, (std::vector<int32_t>{1, 12, 23, 34, 45}));
    ASSERT_EQ(rt_info.candidate_position_offset, 1);
    ASSERT_TRUE(properties.empty());

    // Verify explicit candidate_position_offset
    model->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);
    ASSERT_EQ(properties.at("dflash_candidate_position_offset").as<size_t>(), 0);

    auto deepspec_rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(deepspec_rt_info.candidate_position_offset, 0);
    ASSERT_TRUE(properties.empty());

    // Verify invalid values throw in extraction
    properties["dflash_mode"] = true;
    properties["dflash_mask_token_id"] = int64_t(151669);
    properties["dflash_target_layer_ids"] = std::vector<int32_t>{1, 12, 23, 34, 45};
    properties["dflash_candidate_position_offset"] = size_t(2);
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(properties), ov::Exception);
}

TEST(DFlashModelTransforms, FallsBackToEagle3LayerPatternWithoutAnnotations) {
    auto model = make_eagle3_pattern_hidden_state_model();
    const std::vector<int32_t> target_layer_ids = {0, 2, 4};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);

    ASSERT_FALSE(retained_locators);
    ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    const auto hidden_state = model->output("last_hidden_state");
    ASSERT_EQ(hidden_state.get_partial_shape(), ov::PartialShape({1, 2, 12}));
}

TEST(DFlashModelTransforms, ValidatesLocatorsAcrossPagedAttentionAndGatherBeforeAddingResult) {
    auto model = make_annotated_stateful_sdpa_model();
    const auto original_result_count = model->get_results().size();
    const std::vector<int32_t> target_layer_ids = {0, 1};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);
    ASSERT_TRUE(retained_locators);
    const auto& locators = *retained_locators;
    ASSERT_EQ(locators.size(), 2);
    ASSERT_EQ(locators.front().producer, "decoder_layer_0");
    ASSERT_EQ(locators.back().producer, "final_norm");

    ov::pass::SDPAToPagedAttention(/*use_per_layer_block_indices_inputs=*/false,
                                   /*use_score_outputs=*/false,
                                   /*allow_score_aggregation=*/true,
                                   /*allow_cache_rotation=*/false)
        .run_on_model(model);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 0);
    ASSERT_EQ(model->get_results().size(), original_result_count);
    const auto transformed_ops = model->get_ordered_ops();
    ASSERT_TRUE(std::any_of(transformed_ops.begin(), transformed_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return std::string(node->get_type_name()).find("PagedAttention") != std::string::npos;
    }));
    for (const auto& locator : locators) {
        const auto shape = locator.output.get_partial_shape();
        ASSERT_TRUE(shape[1].is_static());
        ASSERT_EQ(shape[1].get_length(), 1);
    }

    ov::genai::utils::apply_gather_before_matmul_transformation(model);
    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 0);
    ASSERT_EQ(model->get_results().size(), original_result_count);
    const auto parameters = model->get_parameters();
    ASSERT_TRUE(
        std::any_of(parameters.begin(), parameters.end(), [](const std::shared_ptr<ov::op::v0::Parameter>& parameter) {
            return parameter->get_friendly_name() == "sampled_tokens_indices";
        }));
    const auto transformed_lm_head = std::get<0>(ov::genai::utils::find_llm_matmul(model));
    ASSERT_TRUE(transformed_lm_head);
    ASSERT_TRUE(ov::as_type_ptr<ov::op::v8::Gather>(transformed_lm_head->input_value(0).get_node_shared_ptr()));

    ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids);

    ASSERT_EQ(count_outputs_with_name(model, "last_hidden_state"), 1);
    ASSERT_EQ(model->get_results().size(), original_result_count + 1);
    ASSERT_EQ(model->output("last_hidden_state").get_partial_shape()[2].get_length(), 8);
}

TEST(DFlashModelTransforms, RejectsMalformedRequestedLocators) {
    const std::vector<std::pair<std::string, std::vector<int32_t>>> invalid_annotations = {
        {R"({"layers":{"0":{"producer":1,"output_index":0}}})", {0}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":-1}}})", {0}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0}}})", {1}},
        {R"({"layers":{"0":{"producer":"decoder_layer_0","output_index":0},"1":{"producer":"decoder_layer_0","output_index":0}}})",
         {0, 1}},
    };
    for (const auto& [annotation, layer_ids] : invalid_annotations) {
        SCOPED_TRACE(annotation);
        auto model = make_annotated_stateful_sdpa_model();
        model->set_rt_info(annotation, "hidden_states_decoder_layers");
        EXPECT_THROW(ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, layer_ids), ov::Exception);
    }
}

TEST(DFlashModelTransforms, RejectsReplacedRetainedLocatorWhileOriginalRemainsLive) {
    auto model = make_annotated_stateful_sdpa_model();
    const std::vector<int32_t> target_layer_ids = {0};
    auto retained_locators =
        ov::genai::utils::dflash::resolve_target_hidden_state_locators(model, target_layer_ids);
    ASSERT_TRUE(retained_locators);
    const auto& locators = *retained_locators;
    const auto original_output = locators.front().output;
    const auto original = original_output.get_node_shared_ptr();

    auto keep_original_live = std::make_shared<ov::op::v0::Result>(original_output);
    keep_original_live->set_friendly_name("original_decoder_layer_0_result");
    model->add_results({keep_original_live});

    auto replacement = std::make_shared<ov::op::v1::Add>(original->input_value(0), original->input_value(1));
    const auto original_name = original->get_friendly_name();
    for (auto consumer : original_output.get_target_inputs()) {
        if (consumer.get_node() != keep_original_live.get()) {
            consumer.replace_source_output(replacement->output(0));
        }
    }
    original->set_friendly_name("replaced_decoder_layer_0");
    replacement->set_friendly_name(original_name);
    model->validate_nodes_and_infer_types();

    EXPECT_THROW(
        ov::genai::utils::dflash::expose_target_hidden_states(model, retained_locators, target_layer_ids),
        ov::Exception);
}

TEST(DFlashCBGenerationConfig, AcceptsSupportedVlmOptions) {
    ov::genai::GenerationConfig config;

    EXPECT_NO_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config));
}

TEST(DFlashCBGenerationConfig, RejectsUnsupportedVlmOptions) {
    {
        ov::genai::GenerationConfig config;
        config.pruning_ratio = 1;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.echo = true;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.logprobs = 1;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
    {
        ov::genai::GenerationConfig config;
        config.return_omni_outputs = true;
        EXPECT_THROW(ov::genai::dflash_cb::ensure_vlm_generation_config(config), ov::Exception);
    }
}

TEST(DFlashCBCandidatePlanning, KeepsDraftWindowStableUntilGenerationEnds) {
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 8, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 9, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(3, 10, 10), 0);
}

TEST(DFlashCBCandidatePlanning, ClampsValidationWindowToTargetLength) {
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 9, 10), 0);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(3, 10, 10), 0);
}

TEST(DFlashCBCandidatePlanning, SupportsSingleAssistantToken) {
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 0, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::draft_candidate_count(1, 9, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(1, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::validation_candidate_count(1, 9, 10), 0);
}

TEST(DFlashCBDraftWindow, SelectsUnseenTargetRows) {
    using ov::genai::dflash_cb::select_new_target_rows;
    // prefill chunk: the target exported 4 rows ending at position 8, the draft knows 4 positions
    auto prefill = select_new_target_rows(0, 4, 8, 4, 0);
    ASSERT_EQ(prefill.begin, 0);
    ASSERT_EQ(prefill.count, 4);

    // validation rows [seed, c1, c2, c3] with c1 accepted: rows of the seed and c1
    auto partial = select_new_target_rows(0, 10, 12, 4, 2);
    ASSERT_EQ(partial.begin, 0);
    ASSERT_EQ(partial.count, 2);

    auto full = select_new_target_rows(0, 10, 14, 4, 0);
    ASSERT_EQ(full.begin, 0);
    ASSERT_EQ(full.count, 4);

    // the target skipped the request this step: its published rows are stale
    ASSERT_EQ(select_new_target_rows(0, 12, 12, 4, 3).count, 0);
}

TEST(DFlashCBDraftWindow, RejectsTargetRowsBehindTheDraft) {
    using ov::genai::dflash_cb::select_new_target_rows;
    // the target recomputed preempted tokens and reports fewer positions than the draft holds
    ASSERT_THROW(select_new_target_rows(0, 12, 10, 4, 0), ov::Exception);
    // the target exported fewer rows than the draft needs
    ASSERT_THROW(select_new_target_rows(0, 4, 10, 4, 0), ov::Exception);
}

TEST(DFlashCBDraftWindow, PlansBlockOnceHiddenStatesReachTheSeed) {
    using ov::genai::dflash_cb::plan_draft_block;
    // prompt 8, the seed at position 8, hidden states for positions 0..7
    auto block = plan_draft_block(3, 1, 100, 8, 8, /*candidate_position_offset=*/1);
    ASSERT_EQ(block.block_rows, 4);
    ASSERT_EQ(block.candidate_rows, 3);

    auto deepspec = plan_draft_block(3, 1, 100, 8, 8, /*candidate_position_offset=*/0);
    ASSERT_EQ(deepspec.block_rows, 3);
    ASSERT_EQ(deepspec.candidate_rows, 3);

    // the block keeps its trained size while fewer candidates are validated near max_new_tokens
    auto near_end = plan_draft_block(3, 8, 10, 8, 15, 1);
    ASSERT_EQ(near_end.block_rows, 4);
    ASSERT_EQ(near_end.candidate_rows, 1);

    // no seed yet, hidden states behind the seed, drafting off, or only the last token left
    ASSERT_EQ(plan_draft_block(3, 0, 100, 8, 4, 1).block_rows, 0);
    ASSERT_EQ(plan_draft_block(3, 2, 100, 8, 8, 1).block_rows, 0);
    ASSERT_EQ(plan_draft_block(0, 1, 100, 8, 8, 1).block_rows, 0);
    ASSERT_EQ(plan_draft_block(3, 9, 10, 8, 16, 1).block_rows, 0);
}

TEST(DFlashCBDraftWindow, BoundsDraftBatchBudget) {
    using ov::genai::dflash_cb::draft_max_num_batched_tokens;
    using ov::genai::dflash_cb::MAX_DRAFT_BLOCK_ROWS;
    ASSERT_EQ(draft_max_num_batched_tokens(256, 4), 256 + 4 * MAX_DRAFT_BLOCK_ROWS);
    ASSERT_EQ(draft_max_num_batched_tokens(std::numeric_limits<size_t>::max(), 4), std::numeric_limits<size_t>::max());
}

namespace {

constexpr int64_t MASK_TOKEN = 99;
constexpr size_t HIDDEN_SIZE = 2;

std::shared_ptr<ov::Model> make_dflash_draft_inputs_model() {
    ov::ParameterVector params;
    auto add_input = [&](const std::string& name, const ov::element::Type& type, const ov::PartialShape& shape) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
        parameter->output(0).get_tensor().set_names({name});
        params.push_back(parameter);
    };
    add_input("token_type_ids", ov::element::i64, ov::PartialShape{-1, 1});
    add_input("hidden_states", ov::element::f32, ov::PartialShape{-1, 1, static_cast<int64_t>(HIDDEN_SIZE)});
    add_input("sampled_tokens_indices", ov::element::i64, ov::PartialShape{-1});
    auto logits = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 8}, {0.0f});
    logits->output(0).get_tensor().set_names({"logits"});
    return std::make_shared<ov::Model>(ov::OutputVector{logits}, params);
}

// Hidden-state rows [rows, 1, HIDDEN_SIZE] whose values count up from `first`.
ov::Tensor make_hidden_rows(size_t rows, float first) {
    ov::Tensor tensor(ov::element::f32, {rows, 1, HIDDEN_SIZE});
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.get_size(), first);
    return tensor;
}

ov::genai::GeneratedSequences target_step(const std::vector<int64_t>& tokens,
                                          size_t num_processed_tokens,
                                          const ov::Tensor& hidden_states) {
    ov::genai::GeneratedSequences sequences;
    sequences.emplace(0,
                      ov::genai::GeneratedSequence(tokens,
                                                   std::vector<float>(tokens.size(), 0.0f),
                                                   num_processed_tokens,
                                                   hidden_states));
    return sequences;
}

}  // namespace

class DFlashCBPipelineTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class DraftInstance : public ContinuousBatchingForDFlashDecodingImpl {
    public:
        explicit DraftInstance(size_t candidate_position_offset = 1) {
            m_sampler = std::make_shared<ov::genai::Sampler>();
            dflash_mode_enabled = true;
            m_dflash_mask_token_id = MASK_TOKEN;
            m_dflash_candidate_position_offset = candidate_position_offset;
            ov::Core core;
            auto request = core.compile_model(make_dflash_draft_inputs_model(), "CPU").create_infer_request();
            m_model_runner = std::make_shared<ov::genai::ModelRunner>(request, /*block_size=*/4);
            m_model_runner->enable_dflash_draft(candidate_position_offset);
        }

        ov::genai::SequenceGroup::Ptr add(uint64_t request_id,
                                          const std::vector<int64_t>& prompt,
                                          size_t num_assistant_tokens,
                                          size_t max_new_tokens = 100) {
            auto config = ov::genai::utils::get_greedy_config();
            config.num_assistant_tokens = num_assistant_tokens;
            config.max_new_tokens = max_new_tokens;
            auto group = std::make_shared<ov::genai::SequenceGroup>(request_id, prompt, config);
            group->pause_generation(true);
            m_requests.push_back(group);
            return group;
        }

        void remove(uint64_t request_id) {
            m_requests.erase(std::remove_if(m_requests.begin(), m_requests.end(),
                                            [&](const ov::genai::SequenceGroup::Ptr& group) {
                                                return group->get_request_id() == request_id;
                                            }),
                             m_requests.end());
        }

        using ContinuousBatchingForDFlashDecodingImpl::finish_dflash_draft_step;

        auto& state(uint64_t request_id) {
            return m_dflash_draft_states.at(request_id);
        }

        // Forwards `scheduled_rows` of the request's window (all by default) and finishes the step the way the
        // draft multistep does.
        void forward_window(const ov::genai::SequenceGroup::Ptr& group,
                            const ov::Tensor& logits,
                            std::optional<size_t> scheduled_rows = std::nullopt) {
            auto& draft_state = state(group->get_request_id());
            const size_t context_rows = draft_state.pending_rows ? draft_state.pending_rows.get_shape()[0] : 0;
            group->schedule_tokens(scheduled_rows.value_or(context_rows + draft_state.block_rows));
            group->finish_iteration();
            finish_dflash_draft_step(group, draft_state, logits, 0);
        }
    };

    // Logits whose argmax per row is the given token.
    static ov::Tensor make_logits(const std::vector<int64_t>& argmax_tokens, size_t vocab_size = 128) {
        ov::Tensor logits(ov::element::f32, {argmax_tokens.size(), 1, vocab_size});
        std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
        for (size_t row = 0; row < argmax_tokens.size(); ++row) {
            logits.data<float>()[row * vocab_size + argmax_tokens[row]] = 1.0f;
        }
        return logits;
    }

    // Prompt of 8 tokens processed by the target in one chunk; the target sampled seed 7.
    ov::genai::SequenceGroup::Ptr start_with_block(DraftInstance& draft, size_t num_assistant_tokens = 3) {
        auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, num_assistant_tokens);
        draft.update_request(0, target_step({7}, 8, make_hidden_rows(8, 0.0f)), true);
        return group;
    }

    static std::vector<int64_t> generated(const ov::genai::SequenceGroup::Ptr& group) {
        return group->get_running_sequences().front()->get_generated_ids();
    }
};

TEST_F(DFlashCBPipelineTest, PrefillChunkOffersOnlyTheExportedRows) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);

    auto result = draft.update_request(0, target_step({}, 4, make_hidden_rows(4, 0.0f)), true);

    ASSERT_EQ(result.inserted_tokens_cnt, 0);
    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_FALSE(group->is_waiting());
    // a window inside the prompt is scheduled in the prompt phase, capped to the exported rows
    ASSERT_FALSE(group->can_generate_tokens());
    ASSERT_EQ(group->get_num_processed_tokens(), 0);
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 4);
    ASSERT_EQ(draft.state(0).block_rows, 0);
}

TEST_F(DFlashCBPipelineTest, WindowCrossingThePromptEndIsOneGeneratePhaseWindow) {
    DraftInstance draft;
    auto group = start_with_block(draft);

    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_TRUE(group->can_generate_tokens());
    ASSERT_EQ(group->get_num_processed_tokens(), 0);
    // 8 context rows + seed + 3 masks
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 12);
    ASSERT_FALSE(group->requires_sampling());
}

TEST_F(DFlashCBPipelineTest, DeepSpecBlockStartsCandidatesAtTheSeed) {
    DraftInstance draft(/*candidate_position_offset=*/0);
    auto group = start_with_block(draft);

    // 8 context rows + seed + 2 masks
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 11);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, MASK_TOKEN, MASK_TOKEN}));
}

TEST_F(DFlashCBPipelineTest, CandidatesReplaceThePlaceholders) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.forward_window(group, make_logits({20, 21, 22}));

    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 20, 21, 22}));
    ASSERT_EQ(draft.state(0).committed_rows, 8);
    ASSERT_FALSE(draft.state(0).pending_rows);
    // the block stays in the draft KV until the next update rewinds it
    ASSERT_EQ(group->get_num_processed_tokens(), 12);
}

TEST_F(DFlashCBPipelineTest, RejectsASplitBlock) {
    DraftInstance draft;
    auto group = start_with_block(draft);

    // 8 context rows and 2 of the 4 block rows
    ASSERT_THROW(draft.forward_window(group, make_logits({20, 21, 22}), 10), ov::Exception);
}

TEST_F(DFlashCBPipelineTest, PrefillChunkCommitsOnlyTheForwardedRows) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);
    draft.update_request(0, target_step({}, 4, make_hidden_rows(4, 0.0f)), true);

    // the KV cache fit only 3 of the 4 exported rows
    draft.forward_window(group, {}, 3);

    ASSERT_EQ(draft.state(0).committed_rows, 3);
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 1);
    ASSERT_TRUE(draft.state(0).owns_pending_rows);
}

TEST_F(DFlashCBPipelineTest, PartialRejectionKeepsAcceptedRowsAndRewindsTheBlock) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.forward_window(group, make_logits({20, 21, 22}));

    // the target accepted 20, rejected 21 and corrected it to 30; it exported rows for [7, 20, 21, 22]
    auto result = draft.update_request(0, target_step({7, 20, 30}, 10, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 2);
    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 20, 30, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(group->get_num_processed_tokens(), 8);
    // rows of the seed and the accepted candidate, then the next block
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 2);
    ASSERT_EQ(draft.state(0).pending_rows.data<float>()[0], 100.0f);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 2 + 4);
}

TEST_F(DFlashCBPipelineTest, FullAcceptanceTakesAllValidatedRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.forward_window(group, make_logits({20, 21, 22}));

    auto result = draft.update_request(0, target_step({7, 20, 21, 22, 23}, 12, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 4);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 4 + 4);
}

TEST_F(DFlashCBPipelineTest, FullRejectionTakesOnlyTheSeedRow) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.forward_window(group, make_logits({20, 21, 22}));

    auto result = draft.update_request(0, target_step({7, 30}, 9, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 3);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 30, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 1);
}

TEST_F(DFlashCBPipelineTest, DeferredTargetPublishesNoNewRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.forward_window(group, make_logits({20, 21, 22}));

    // the target did not run the request: it still reports 8 processed positions and stale rows
    auto result = draft.update_request(0, target_step({7, 20, 21, 22}, 8, make_hidden_rows(8, 0.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_FALSE(draft.state(0).pending_rows);
    ASSERT_EQ(draft.state(0).block_rows, 0);
    ASSERT_TRUE(group->is_waiting());
}

TEST_F(DFlashCBPipelineTest, NearMaxNewTokensForwardsOnlyTheContext) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3}, 3, /*max_new_tokens=*/2);
    draft.update_request(0, target_step({3}, 4, make_hidden_rows(4, 0.0f)), true);

    // one token left for the target: no block, only the context rows are forwarded
    ASSERT_EQ(draft.state(0).block_rows, 0);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{3}));
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 4);
}

TEST_F(DFlashCBPipelineTest, UnscheduledWindowDropsPlaceholdersAndOwnsItsRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    ov::Tensor view = draft.state(0).pending_rows;

    draft.finish_dflash_draft_step(group, draft.state(0), {}, 0);

    ASSERT_EQ(generated(group), (std::vector<int64_t>{7}));
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 8);
    ASSERT_TRUE(draft.state(0).owns_pending_rows);
    ASSERT_NE(draft.state(0).pending_rows.data(), view.data());
}

TEST_F(DFlashCBPipelineTest, ReusedRequestIdStartsFromAFreshState) {
    DraftInstance draft;
    auto first = start_with_block(draft);
    draft.forward_window(first, make_logits({20, 21, 22}));
    draft.remove(0);

    auto second = draft.add(0, {5, 6, 7, 8}, 3);
    draft.update_request(0, target_step({}, 2, make_hidden_rows(2, 0.0f)), true);

    ASSERT_EQ(draft.state(0).committed_rows, 0);
    ASSERT_EQ(draft.state(0).pending_rows.get_shape()[0], 2);
}

class DFlashStrategyTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class StrategyInstance : public DFlashDecodingImpl {
    public:
        using DFlashDecodingImpl::make_draft_generation_config;
    };

    static void expect_rejected(const ov::genai::GenerationConfig& config, const std::string& expected_message) {
        // Unsupported configurations must be rejected before add_request touches the pipelines.
        StrategyInstance strategy;
        try {
            strategy.add_request(0, ov::Tensor{}, config);
            FAIL() << "Expected the DFlash request to be rejected";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find(expected_message), std::string::npos) << exception.what();
        }
    }
};

TEST_F(DFlashStrategyTest, RejectsSampling) {
    auto config = ov::genai::utils::get_greedy_config();
    config.do_sample = true;
    expect_rejected(config, "greedy decoding only");
}

TEST_F(DFlashStrategyTest, RejectsConfidenceThreshold) {
    auto config = ov::genai::utils::get_greedy_config();
    config.assistant_confidence_threshold = 0.5f;
    expect_rejected(config, "assistant_confidence_threshold must be 0.f");
}

TEST_F(DFlashStrategyTest, RejectsAdapters) {
    auto config = ov::genai::utils::get_greedy_config();
    config.adapters = ov::genai::AdapterConfig{};
    expect_rejected(config, "does not support adapters");
}

TEST_F(DFlashStrategyTest, DraftConfigOnlyProposesCandidates) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 3;
    config.stop_strings = {"stop"};
    config.stop_token_ids = {5};
    config.echo = true;

    const auto draft_config = StrategyInstance::make_draft_generation_config(config);

    EXPECT_TRUE(draft_config.ignore_eos);
    EXPECT_TRUE(draft_config.stop_strings.empty());
    EXPECT_TRUE(draft_config.stop_token_ids.empty());
    EXPECT_FALSE(draft_config.echo);
    EXPECT_EQ(draft_config.max_new_tokens, SIZE_MAX - 1);
    EXPECT_NO_THROW(draft_config.validate());
}

TEST_F(DFlashStrategyTest, RejectsADraftWithoutTokenTypeIds) {
    auto make_model = [] {
        auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
        input_ids->output(0).set_names({"input_ids"});
        auto result = std::make_shared<ov::op::v0::Result>(input_ids);
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids});
    };
    ov::genai::utils::dflash::DFlashRTInfo rt_info;
    rt_info.dflash_mode = true;
    rt_info.mask_token_id = MASK_TOKEN;
    rt_info.target_layer_ids = {0};
    const ov::genai::ModelDesc main_desc(make_model(), ov::genai::Tokenizer{});
    const ov::genai::ModelDesc draft_desc(make_model(), ov::genai::Tokenizer{});

    try {
        DFlashDecodingImpl strategy(main_desc, draft_desc, rt_info);
        FAIL() << "Expected the stateful DFlash draft to be rejected";
    } catch (const ov::Exception& exception) {
        EXPECT_NE(std::string(exception.what()).find("re-export the draft"), std::string::npos) << exception.what();
    }
}

