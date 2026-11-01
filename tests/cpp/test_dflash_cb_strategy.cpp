// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "model_desc.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_lattice.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> make_annotated_stateful_sdpa_model() {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});
    auto embedding_weights =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 4}, std::vector<float>(16 * 4, 1.0f));
    embedding_weights->set_friendly_name("model.embed_tokens.weight");
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

constexpr size_t DSPARK_TEST_VOCAB_SIZE = 16;
constexpr size_t DSPARK_TEST_MARKOV_RANK = 4;
constexpr int64_t DSPARK_TEST_TOP_K = 2;

void set_dspark_markov_head_rt_info(const std::shared_ptr<ov::Model>& model, const std::string& omitted_key = "") {
    const std::vector<std::pair<std::string, std::string>> rt_info = {
        {"top_k", std::to_string(DSPARK_TEST_TOP_K)},
        {"vocab_size", std::to_string(DSPARK_TEST_VOCAB_SIZE)},
        {"markov_rank", std::to_string(DSPARK_TEST_MARKOV_RANK)},
        {"markov_head_type", "vanilla"},
    };
    for (const auto& [name, value] : rt_info) {
        if (name != omitted_key) {
            model->set_rt_info(value, {"dflash", name});
        }
    }
}

// The dense step biases token j after token p by codebook[p] . codebook[j]. The candidate scorer keeps the Top-K
// draft logits and scores candidate i followed by candidate j with the sum of their logits. As in the optimum-intel
// export, every K-sized shape follows the TopK.
std::shared_ptr<ov::Model> make_dspark_markov_head_model(const std::string& omitted_key = "") {
    auto prev_token_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    prev_token_ids->output(0).set_names({"prev_token_ids"});
    const ov::Shape codebook_shape{DSPARK_TEST_VOCAB_SIZE, DSPARK_TEST_MARKOV_RANK};
    std::vector<float> codebook_values(ov::shape_size(codebook_shape));
    std::iota(codebook_values.begin(), codebook_values.end(), 0.0f);
    auto codebook = ov::op::v0::Constant::create(ov::element::f32, codebook_shape, codebook_values);
    auto codes = std::make_shared<ov::op::v8::Gather>(codebook,
                                                      prev_token_ids,
                                                      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto markov_bias = std::make_shared<ov::op::v0::MatMul>(codes, codebook, false, true);
    auto markov_bias_result = std::make_shared<ov::op::v0::Result>(markov_bias);
    markov_bias_result->output(0).set_names({"markov_bias"});

    auto draft_logits =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{-1, static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE)});
    draft_logits->output(0).set_names({"draft_logits"});
    auto anchor_token_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    anchor_token_ids->output(0).set_names({"anchor_token_ids"});
    auto subsequence_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    subsequence_begins->output(0).set_names({"subsequence_begins"});
    auto top_k = std::make_shared<ov::op::v11::TopK>(
        draft_logits,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {DSPARK_TEST_TOP_K}),
        -1,
        ov::op::v11::TopK::Mode::MAX,
        ov::op::v11::TopK::SortType::SORT_VALUES);
    auto candidate_ids = std::make_shared<ov::op::v0::Convert>(top_k->output(1), ov::element::i64);
    auto predecessor_logits =
        std::make_shared<ov::op::v0::Unsqueeze>(top_k->output(0),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    auto successor_logits =
        std::make_shared<ov::op::v0::Unsqueeze>(top_k->output(0),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto edge_scores = std::make_shared<ov::op::v1::Add>(predecessor_logits, successor_logits);
    auto edge_scores_result = std::make_shared<ov::op::v0::Result>(edge_scores);
    edge_scores_result->output(0).set_names({"edge_scores"});
    auto candidate_ids_result = std::make_shared<ov::op::v0::Result>(candidate_ids);
    candidate_ids_result->output(0).set_names({"candidate_ids"});

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{markov_bias_result, edge_scores_result, candidate_ids_result},
        ov::ParameterVector{prev_token_ids, draft_logits, anchor_token_ids, subsequence_begins});
    set_dspark_markov_head_rt_info(model, omitted_key);
    return model;
}

constexpr size_t DFLASH2_TEST_HIDDEN_SIZE = 4;

// The DFlash-2 selector is a gated Markov head: its candidate scorer also reads the draft hidden states, and it has no
// dense step. Here the hidden state of a position adds the sum of its values to every edge score of that position.
std::shared_ptr<ov::Model> make_dflash2_markov_head_model() {
    auto draft_logits =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{-1, static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE)});
    draft_logits->output(0).set_names({"draft_logits"});
    auto draft_hidden_states =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{-1, static_cast<int64_t>(DFLASH2_TEST_HIDDEN_SIZE)});
    draft_hidden_states->output(0).set_names({"draft_hidden_states"});
    auto anchor_token_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    anchor_token_ids->output(0).set_names({"anchor_token_ids"});
    auto subsequence_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    subsequence_begins->output(0).set_names({"subsequence_begins"});
    auto top_k = std::make_shared<ov::op::v11::TopK>(
        draft_logits,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {DSPARK_TEST_TOP_K}),
        -1,
        ov::op::v11::TopK::Mode::MAX,
        ov::op::v11::TopK::SortType::SORT_VALUES);
    auto candidate_ids = std::make_shared<ov::op::v0::Convert>(top_k->output(1), ov::element::i64);
    auto predecessor_logits =
        std::make_shared<ov::op::v0::Unsqueeze>(top_k->output(0),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    auto successor_logits =
        std::make_shared<ov::op::v0::Unsqueeze>(top_k->output(0),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto unary_scores = std::make_shared<ov::op::v1::Add>(predecessor_logits, successor_logits);
    auto hidden_sum = std::make_shared<ov::op::v0::MatMul>(
        draft_hidden_states,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{DFLASH2_TEST_HIDDEN_SIZE, 1}, {1.0f}));
    auto gate =
        std::make_shared<ov::op::v0::Unsqueeze>(hidden_sum,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    auto edge_scores = std::make_shared<ov::op::v1::Add>(unary_scores, gate);
    auto edge_scores_result = std::make_shared<ov::op::v0::Result>(edge_scores);
    edge_scores_result->output(0).set_names({"edge_scores"});
    auto candidate_ids_result = std::make_shared<ov::op::v0::Result>(candidate_ids);
    candidate_ids_result->output(0).set_names({"candidate_ids"});

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{edge_scores_result, candidate_ids_result},
        ov::ParameterVector{draft_logits, draft_hidden_states, anchor_token_ids, subsequence_begins});
    set_dspark_markov_head_rt_info(model);
    model->set_rt_info(std::string("dflash2"), {"dflash", "markov_head_type"});
    return model;
}

ov::AnyMap select_markov_computation(std::optional<int64_t> top_k,
                                     const std::shared_ptr<ov::Model>& markov_head_model) {
    ov::AnyMap properties;
    ov::genai::utils::dflash::select_markov_head_computation(markov_head_model, top_k, properties);
    return properties;
}

std::shared_ptr<ov::Model> selected_computation(const ov::AnyMap& properties) {
    return properties.at("dflash_markov_head_computation").as<std::shared_ptr<ov::Model>>();
}

// The K of the selected candidate scorer, or TOP_K_FOR_MARKOV_STEP for the dense step.
int64_t selected_top_k(const ov::AnyMap& properties) {
    const auto computation = selected_computation(properties);
    if (ov::genai::utils::dflash::is_markov_step(computation)) {
        return ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP;
    }
    return std::stoll(computation->get_rt_info<std::string>(std::vector<std::string>{"dflash", "top_k"}));
}

void set_markov_head_computation(ov::AnyMap& properties, const std::shared_ptr<ov::Model>& computation) {
    properties["dflash_markov_head_computation"] = computation;
}

std::shared_ptr<ov::Model> make_dspark_markov_step() {
    return selected_computation(select_markov_computation(int64_t{0}, make_dspark_markov_head_model()));
}

std::shared_ptr<ov::Model> make_dspark_candidate_scorer(bool with_extra_input = false,
                                                        const std::string& omitted_key = "") {
    auto model = selected_computation(select_markov_computation(std::nullopt, make_dspark_markov_head_model()));
    if (with_extra_input) {
        auto draft_hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
        draft_hidden_states->output(0).set_names({"draft_hidden_states"});
        model->add_parameters({draft_hidden_states});
    }
    if (!omitted_key.empty()) {
        model->get_rt_info()["dflash"].as<ov::AnyMap>().erase(omitted_key);
    }
    return model;
}

ov::genai::utils::dflash::DFlashRTInfo make_dspark_backbone_info() {
    ov::genai::utils::dflash::DFlashRTInfo info;
    info.markov_rank = DSPARK_TEST_MARKOV_RANK;
    info.markov_head_type = "vanilla";
    return info;
}

ov::AnyMap make_dspark_draft_properties(size_t markov_rank, const std::string& head_type) {
    ov::AnyMap properties;
    properties["dflash_mode"] = true;
    properties["dflash_mask_token_id"] = int64_t{15};
    properties["dflash_target_layer_ids"] = std::vector<int32_t>{0};
    properties["dflash_candidate_position_offset"] = size_t{0};
    properties["dflash_markov_rank"] = markov_rank;
    properties["dflash_markov_head_type"] = head_type;
    return properties;
}

std::shared_ptr<ov::Model> make_dflash2_candidate_scorer() {
    return selected_computation(select_markov_computation(std::nullopt, make_dflash2_markov_head_model()));
}

ov::genai::utils::dflash::DFlashRTInfo make_dflash2_backbone_info() {
    ov::genai::utils::dflash::DFlashRTInfo info;
    info.dflash_version = 2;
    info.markov_rank = DSPARK_TEST_MARKOV_RANK;
    info.markov_head_type = "dflash2";
    return info;
}

ov::AnyMap make_dflash2_draft_properties(size_t markov_rank, const std::string& head_type) {
    auto properties = make_dspark_draft_properties(markov_rank, head_type);
    properties["dflash_version"] = int64_t{2};
    properties["dflash_gpu_min_activations_scale_factor"] = 32.0f;
    return properties;
}

std::shared_ptr<ov::Model> make_dflash_export_backbone_model() {
    auto inputs_embeds = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 4});
    inputs_embeds->output(0).set_names({"inputs_embeds"});
    auto identity =
        std::make_shared<ov::op::v1::Add>(inputs_embeds,
                                          ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}));
    auto result = std::make_shared<ov::op::v0::Result>(identity);
    result->output(0).set_names({"last_hidden_state"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{inputs_embeds});
}

std::vector<float> tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    return std::vector<float>(data, data + tensor.get_size());
}

std::vector<int64_t> int64_tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + tensor.get_size());
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

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& prefix)
        : m_path(std::filesystem::temp_directory_path() /
                 (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::remove_all(m_path);
        std::filesystem::create_directories(m_path);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    const std::filesystem::path& path() const {
        return m_path;
    }

private:
    std::filesystem::path m_path;
};

}  // namespace

TEST(DFlashModelTransforms, AppliesAndExtractsDraftRtInfo) {
    auto model = make_annotated_stateful_sdpa_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("1,12,23,34,45"), {"dflash", "target_layer_ids"});
    model->set_rt_info(std::string("2"), {"dflash", "version"});
    model->set_rt_info(std::string("32.0"), {"dflash", "gpu_min_activations_scale_factor"});
    model->set_rt_info(std::string("1.25"), {"dflash", "input_embedding_scale"});
    model->set_rt_info(std::string("0.75"), {"dflash", "output_multiplier"});
    model->set_rt_info(std::string("8.0"), {"dflash", "final_logit_softcapping"});
    model->set_rt_info(std::to_string(DSPARK_TEST_MARKOV_RANK), {"dflash", "markov_rank"});
    model->set_rt_info(std::string("dflash2"), {"dflash", "markov_head_type"});
    const auto candidate_scorer = make_dflash2_candidate_scorer();

    // Verify default legacy candidate_position_offset when omitted
    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);

    ASSERT_TRUE(properties.at("dflash_mode").as<bool>());
    ASSERT_EQ(properties.at("dflash_mask_token_id").as<int64_t>(), 151669);
    ASSERT_EQ(properties.at("dflash_target_layer_ids").as<std::vector<int32_t>>(),
              (std::vector<int32_t>{1, 12, 23, 34, 45}));
    ASSERT_FLOAT_EQ(properties.at("dflash_gpu_min_activations_scale_factor").as<float>(), 32.0f);
    ASSERT_EQ(properties.at("dflash_markov_head_type").as<std::string>(), "dflash2");

    set_markov_head_computation(properties, candidate_scorer);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
    ASSERT_EQ(rt_info.dflash_version, 2);
    ASSERT_EQ(rt_info.mask_token_id, 151669);
    ASSERT_EQ(rt_info.target_layer_ids, (std::vector<int32_t>{1, 12, 23, 34, 45}));
    ASSERT_EQ(rt_info.candidate_position_offset, 1);
    ASSERT_FLOAT_EQ(rt_info.gpu_min_activations_scale_factor, 32.0f);
    ASSERT_FLOAT_EQ(rt_info.input_embedding_scale, 1.25f);
    ASSERT_FLOAT_EQ(rt_info.output_multiplier, 0.75f);
    ASSERT_FLOAT_EQ(rt_info.final_logit_softcapping, 8.0f);
    ASSERT_EQ(rt_info.markov_rank, DSPARK_TEST_MARKOV_RANK);
    ASSERT_EQ(rt_info.markov_head_type, "dflash2");
    ASSERT_EQ(rt_info.markov_head_computation, candidate_scorer);
    ASSERT_TRUE(properties.empty());

    // Verify explicit candidate_position_offset
    model->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});
    ov::genai::utils::dflash::apply_dflash_rt_info(model, properties);
    ASSERT_EQ(properties.at("dflash_candidate_position_offset").as<size_t>(), 0);

    set_markov_head_computation(properties, candidate_scorer);
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

TEST(DFlashModelTransforms, ValidatesAndPromotesDFlash2GpuActivationScale) {
    ov::AnyMap without_scale;
    without_scale["dflash_mode"] = true;
    without_scale["dflash_version"] = int64_t{2};
    without_scale["dflash_mask_token_id"] = int64_t{151669};
    without_scale["dflash_target_layer_ids"] = std::vector<int32_t>{0};
    EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(without_scale), ov::Exception);

    auto model = make_annotated_stateful_sdpa_model();
    model->set_rt_info(true, "dflash_mode");
    model->set_rt_info(std::string("2"), {"dflash", "version"});
    model->set_rt_info(std::string("151669"), {"dflash", "mask_token_id"});
    model->set_rt_info(std::string("0"), {"dflash", "target_layer_ids"});
    model->set_rt_info(std::string("0"), {"dflash", "gpu_min_activations_scale_factor"});
    ov::AnyMap zero_scale;
    EXPECT_THROW(ov::genai::utils::dflash::apply_dflash_rt_info(model, zero_scale), ov::Exception);

    // At compile time on GPU, the scale is the exported one unless a larger one is requested.
    ov::genai::utils::dflash::DFlashRTInfo info;
    info.dflash_mode = true;
    info.dflash_version = 2;
    info.gpu_min_activations_scale_factor = 32.0f;

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_gpu_compile_properties(info, "GPU", false, properties);
    ASSERT_FLOAT_EQ(properties.at(ov::hint::activations_scale_factor.name()).as<float>(), 32.0f);

    properties[ov::hint::activations_scale_factor.name()] = 64.0f;
    ov::genai::utils::dflash::apply_dflash_gpu_compile_properties(info, "GPU.0", false, properties);
    ASSERT_FLOAT_EQ(properties.at(ov::hint::activations_scale_factor.name()).as<float>(), 64.0f);

    // A smaller draft scale is rejected; a smaller target scale, which a draft without properties inherits, is raised.
    properties[ov::hint::activations_scale_factor.name()] = 8.0f;
    EXPECT_THROW(ov::genai::utils::dflash::apply_dflash_gpu_compile_properties(info, "GPU", false, properties),
                 ov::Exception);
    ov::genai::utils::dflash::apply_dflash_gpu_compile_properties(info, "GPU", true, properties);
    ASSERT_FLOAT_EQ(properties.at(ov::hint::activations_scale_factor.name()).as<float>(), 32.0f);
}

TEST(DFlashModelTransforms, GathersTheLastHiddenStateRowsOfTheLogits) {
    auto target = make_annotated_stateful_sdpa_model();
    auto draft = make_dflash_export_backbone_model();
    ov::genai::utils::dflash::attach_target_lm_head_to_draft(target, draft, true);
    EXPECT_THROW(ov::genai::utils::dflash::gather_last_hidden_state_with_logits(draft), ov::Exception);

    ov::genai::utils::apply_gather_before_matmul_transformation(draft);
    ov::genai::utils::dflash::gather_last_hidden_state_with_logits(draft);

    auto request = ov::genai::utils::singleton_core().compile_model(draft, "CPU").create_infer_request();
    std::vector<float> inputs_embeds(3 * 4);
    std::iota(inputs_embeds.begin(), inputs_embeds.end(), 0.0f);
    std::vector<int64_t> sampled_rows{2, 0};
    request.set_tensor("inputs_embeds", ov::Tensor(ov::element::f32, ov::Shape{1, 3, 4}, inputs_embeds.data()));
    request.set_tensor("sampled_tokens_indices", ov::Tensor(ov::element::i64, ov::Shape{2}, sampled_rows.data()));
    request.infer();

    const auto hidden_state = request.get_tensor("last_hidden_state");
    ASSERT_EQ(hidden_state.get_shape(), ov::Shape({1, 2, 4}));
    EXPECT_EQ(tensor_values(hidden_state), (std::vector<float>{8.0f, 9.0f, 10.0f, 11.0f, 0.0f, 1.0f, 2.0f, 3.0f}));
    EXPECT_EQ(request.get_tensor("logits").get_shape(), ov::Shape({1, 2, 8}));
}

TEST(DFlashModelTransforms, AppliesDraftEmbeddingScale) {
    const auto has_embedding_scale = [](const std::shared_ptr<ov::Model>& draft) {
        const auto ordered_ops = draft->get_ordered_ops();
        return std::any_of(ordered_ops.begin(), ordered_ops.end(), [](const std::shared_ptr<ov::Node>& node) {
            return node->get_friendly_name() == "dflash_draft_embedding_scale";
        });
    };

    // a text target's embedding is grafted into the draft
    auto draft = make_dflash_export_backbone_model();
    ov::genai::utils::dflash::attach_target_embedding_to_draft(make_annotated_stateful_sdpa_model(), draft, 1.25f);
    ASSERT_TRUE(ov::genai::utils::has_input(draft, "input_ids"));
    ASSERT_FALSE(ov::genai::utils::has_input(draft, "inputs_embeds"));
    ASSERT_TRUE(has_embedding_scale(draft));

    // the draft of a VLM target keeps its external inputs_embeds
    auto vlm_draft = make_dflash_export_backbone_model();
    ov::genai::utils::dflash::scale_draft_inputs_embeds(vlm_draft, 1.25f);
    ASSERT_TRUE(ov::genai::utils::has_input(vlm_draft, "inputs_embeds"));
    ASSERT_TRUE(has_embedding_scale(vlm_draft));
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

TEST(DSparkModelTransforms, AppliesAndExtractsMarkovHeadRtInfo) {
    auto backbone = make_dflash_export_backbone_model();
    backbone->set_rt_info(true, "dflash_mode");
    backbone->set_rt_info(std::string("15"), {"dflash", "mask_token_id"});
    backbone->set_rt_info(std::string("0,1"), {"dflash", "target_layer_ids"});
    backbone->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});
    backbone->set_rt_info(std::string("4"), {"dflash", "markov_rank"});
    backbone->set_rt_info(std::string("vanilla"), {"dflash", "markov_head_type"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(backbone, properties);
    ASSERT_EQ(properties.at("dflash_markov_rank").as<size_t>(), 4);
    ASSERT_EQ(properties.at("dflash_markov_head_type").as<std::string>(), "vanilla");

    auto markov_step = make_dspark_markov_step();
    set_markov_head_computation(properties, markov_step);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.candidate_position_offset, 0);
    ASSERT_EQ(rt_info.markov_rank, 4);
    ASSERT_EQ(rt_info.markov_head_type, "vanilla");
    ASSERT_EQ(rt_info.markov_head_computation, markov_step);
    ASSERT_TRUE(ov::genai::utils::dflash::is_markov_step(rt_info.markov_head_computation));
    ASSERT_TRUE(properties.empty());
}

TEST(DSparkModelTransforms, TreatsMarkovRankZeroAsPlainDFlash) {
    // DeepSpec publishes its DFlash drafts as DSpark checkpoints with markov_rank=0; their export has no Markov keys.
    auto backbone = make_dflash_export_backbone_model();
    backbone->set_rt_info(true, "dflash_mode");
    backbone->set_rt_info(std::string("15"), {"dflash", "mask_token_id"});
    backbone->set_rt_info(std::string("0,1"), {"dflash", "target_layer_ids"});
    backbone->set_rt_info(std::string("0"), {"dflash", "candidate_position_offset"});

    ov::AnyMap properties;
    ov::genai::utils::dflash::apply_dflash_rt_info(backbone, properties);
    ASSERT_EQ(properties.count("dflash_markov_rank"), 0);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_TRUE(rt_info.dflash_mode);
    ASSERT_EQ(rt_info.candidate_position_offset, 0);
    ASSERT_EQ(rt_info.markov_rank, 0);
    ASSERT_EQ(rt_info.markov_head_computation, nullptr);
    ASSERT_TRUE(properties.empty());

    auto explicit_rank_zero = make_dspark_draft_properties(0, "vanilla");
    rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(explicit_rank_zero);
    ASSERT_EQ(rt_info.markov_rank, 0);
    ASSERT_EQ(rt_info.markov_head_computation, nullptr);
    ASSERT_TRUE(explicit_rank_zero.empty());
}

TEST(DSparkModelTransforms, ExtractsSelectedCandidateScorer) {
    auto candidate_scorer = make_dspark_candidate_scorer();
    auto properties = make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "vanilla");
    set_markov_head_computation(properties, candidate_scorer);
    auto rt_info = ov::genai::utils::dflash::extract_dflash_info_from_config(properties);
    ASSERT_EQ(rt_info.markov_head_computation, candidate_scorer);
    ASSERT_FALSE(ov::genai::utils::dflash::is_markov_step(rt_info.markov_head_computation));
    ASSERT_TRUE(properties.empty());
}

TEST(DSparkModelTransforms, RejectsInconsistentDraftProperties) {
    const auto with_computation = [](ov::AnyMap properties, const std::shared_ptr<ov::Model>& computation) {
        set_markov_head_computation(properties, computation);
        return properties;
    };
    const auto vanilla = [](size_t markov_rank) {
        return make_dspark_draft_properties(markov_rank, "vanilla");
    };
    auto unsupported_version = vanilla(0);
    unsupported_version["dflash_version"] = int64_t{3};
    auto gated_candidate_scorer = make_dspark_candidate_scorer();
    gated_candidate_scorer->set_rt_info(std::string("gated"), {"dflash", "markov_head_type"});
    const std::vector<std::pair<std::string, ov::AnyMap>> cases = {
        {"unsupported draft version", unsupported_version},
        {"Markov head without its computation", vanilla(DSPARK_TEST_MARKOV_RANK)},
        {"gated Markov head type",
         with_computation(make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "gated"), make_dspark_markov_step())},
        {"rnn Markov head type",
         with_computation(make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "rnn"), make_dspark_markov_step())},
        {"computation of another head type",
         with_computation(vanilla(DSPARK_TEST_MARKOV_RANK), gated_candidate_scorer)},
        {"computation of another rank", with_computation(vanilla(8), make_dspark_candidate_scorer())},
        {"DFlash v2 draft without a Markov head", make_dflash2_draft_properties(0, "dflash2")},
        {"DFlash v2 draft with a vanilla Markov head",
         with_computation(make_dflash2_draft_properties(DSPARK_TEST_MARKOV_RANK, "vanilla"),
                          make_dspark_candidate_scorer())},
        {"DFlash v1 draft with a gated Markov head",
         with_computation(make_dspark_draft_properties(DSPARK_TEST_MARKOV_RANK, "dflash2"),
                          make_dflash2_candidate_scorer())},
    };
    for (auto [name, properties] : cases) {
        SCOPED_TRACE(name);
        EXPECT_THROW(ov::genai::utils::dflash::extract_dflash_info_from_config(properties), ov::Exception);
    }
}

TEST(DSparkModelTransforms, RejectsMarkovStepThatDoesNotMatchTheBackbone) {
    auto info = make_dspark_backbone_info();
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_markov_step(make_dspark_markov_step(), info));

    auto mismatched_rank = info;
    mismatched_rank.markov_rank = 8;
    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(make_dspark_markov_step(), mismatched_rank),
                 ov::Exception);

    auto wrong_vocab = make_dspark_markov_step();
    wrong_vocab->set_rt_info(std::string("32"), {"dflash", "vocab_size"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(wrong_vocab, info), ov::Exception);

    EXPECT_THROW(ov::genai::utils::dflash::validate_markov_step(make_dflash_export_backbone_model(), info),
                 ov::Exception);
}

TEST(DSparkModelTransforms, ReadsTheMarkovHeadModelNextToTheDraft) {
    const TemporaryDirectory temporary_directory("genai_dspark_markov_");
    const auto& draft_dir = temporary_directory.path();
    const auto info = make_dspark_backbone_info();

    ov::AnyMap without_markov_head{{"dflash_markov_rank", size_t{0}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, without_markov_head);
    ASSERT_EQ(without_markov_head.count("dflash_markov_head_computation"), 0);

    ov::AnyMap top_k_without_markov_head{{"dflash_markov_rank", size_t{0}}, {"markov_head_top_k", 16}};
    EXPECT_THROW(ov::genai::utils::dflash::read_markov_head(draft_dir, top_k_without_markov_head), ov::Exception);

    ov::AnyMap missing_markov_head_model{{"dflash_markov_rank", size_t{4}}};
    EXPECT_THROW(ov::genai::utils::dflash::read_markov_head(draft_dir, missing_markov_head_model), ov::Exception);

    ov::save_model(make_dspark_markov_head_model(), draft_dir / "openvino_markov_head_model.xml");
    ov::AnyMap rank_zero_next_to_markov_head{{"dflash_markov_rank", size_t{0}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, rank_zero_next_to_markov_head);
    ASSERT_EQ(rank_zero_next_to_markov_head.count("dflash_markov_head_computation"), 0);

    ov::AnyMap exported_top_k{{"dflash_markov_rank", size_t{4}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, exported_top_k);
    ASSERT_EQ(selected_top_k(exported_top_k), DSPARK_TEST_TOP_K);
    const auto candidate_scorer = selected_computation(exported_top_k);
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_candidate_scorer(candidate_scorer, info));
    EXPECT_EQ(candidate_scorer->output("candidate_ids").get_partial_shape()[1].get_length(), DSPARK_TEST_TOP_K);

    // markov_head_top_k is consumed here so that it never reaches compile_model.
    ov::AnyMap dense_walk{{"dflash_markov_rank", size_t{4}}, {"markov_head_top_k", 0}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, dense_walk);
    ASSERT_EQ(dense_walk.count("markov_head_top_k"), 0);
    ASSERT_EQ(selected_top_k(dense_walk), ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP);
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_markov_step(selected_computation(dense_walk), info));

    ov::AnyMap replaced_top_k{{"dflash_markov_rank", size_t{4}}, {"markov_head_top_k", int64_t{3}}};
    ov::genai::utils::dflash::read_markov_head(draft_dir, replaced_top_k);
    ASSERT_EQ(replaced_top_k.count("markov_head_top_k"), 0);
    ASSERT_EQ(selected_top_k(replaced_top_k), 3);
}

TEST(DSparkModelTransforms, SelectsDenseStepOrCandidateScorerAndReplacesTopK) {
    const auto markov_head_model = make_dspark_markov_head_model();
    auto& core = ov::genai::utils::singleton_core();

    const auto markov_step = selected_computation(select_markov_computation(int64_t{0}, markov_head_model));
    ASSERT_EQ(markov_step->inputs().size(), 1);
    ASSERT_EQ(markov_step->outputs().size(), 1);
    EXPECT_FALSE(markov_step->has_rt_info(std::vector<std::string>{"dflash", "top_k"}));
    auto markov_step_request = core.compile_model(markov_step, "CPU").create_infer_request();
    int64_t previous_token = 1;
    markov_step_request.set_tensor("prev_token_ids", ov::Tensor(ov::element::i64, ov::Shape{1}, &previous_token));
    markov_step_request.infer();
    const auto markov_bias = tensor_values(markov_step_request.get_tensor("markov_bias"));
    ASSERT_EQ(markov_bias.size(), DSPARK_TEST_VOCAB_SIZE);
    // codebook[1] = [4, 5, 6, 7], codebook[0] = [0, 1, 2, 3], codebook[2] = [8, 9, 10, 11].
    EXPECT_FLOAT_EQ(markov_bias[0], 38.0f);
    EXPECT_FLOAT_EQ(markov_bias[2], 214.0f);

    constexpr size_t num_positions = 2;
    std::vector<float> draft_logits(num_positions * DSPARK_TEST_VOCAB_SIZE);
    for (size_t position = 0; position < num_positions; ++position) {
        for (size_t token = 0; token < DSPARK_TEST_VOCAB_SIZE; ++token) {
            draft_logits[position * DSPARK_TEST_VOCAB_SIZE + token] =
                static_cast<float>((token * 5 + position * 3) % DSPARK_TEST_VOCAB_SIZE);
        }
    }
    int64_t anchor_token = 0;
    int64_t subsequence_begins[] = {0, static_cast<int64_t>(num_positions)};
    for (const auto& [top_k, expected_top_k] : std::vector<std::pair<std::optional<int64_t>, size_t>>{
             {std::nullopt, DSPARK_TEST_TOP_K},
             {int64_t{3}, 3},
             {static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE), DSPARK_TEST_VOCAB_SIZE}}) {
        SCOPED_TRACE(expected_top_k);
        const auto selected = select_markov_computation(top_k, markov_head_model);
        ASSERT_EQ(selected_top_k(selected), static_cast<int64_t>(expected_top_k));
        const auto candidate_scorer = selected_computation(selected);
        EXPECT_EQ(candidate_scorer->get_rt_info<std::string>(std::vector<std::string>{"dflash", "top_k"}),
                  std::to_string(expected_top_k));
        auto request = core.compile_model(candidate_scorer, "CPU").create_infer_request();
        request.set_tensor(
            "draft_logits",
            ov::Tensor(ov::element::f32, ov::Shape{num_positions, DSPARK_TEST_VOCAB_SIZE}, draft_logits.data()));
        request.set_tensor("anchor_token_ids", ov::Tensor(ov::element::i64, ov::Shape{1}, &anchor_token));
        request.set_tensor("subsequence_begins", ov::Tensor(ov::element::i64, ov::Shape{2}, subsequence_begins));
        request.infer();
        const auto candidate_ids = int64_tensor_values(request.get_tensor("candidate_ids"));
        const auto edge_scores = tensor_values(request.get_tensor("edge_scores"));
        ASSERT_EQ(candidate_ids.size(), num_positions * expected_top_k);
        ASSERT_EQ(edge_scores.size(), num_positions * expected_top_k * expected_top_k);
        for (size_t position = 0; position < num_positions; ++position) {
            const auto* logits = draft_logits.data() + position * DSPARK_TEST_VOCAB_SIZE;
            std::vector<int64_t> ranked_tokens(DSPARK_TEST_VOCAB_SIZE);
            std::iota(ranked_tokens.begin(), ranked_tokens.end(), 0);
            std::sort(ranked_tokens.begin(), ranked_tokens.end(), [&](int64_t lhs, int64_t rhs) {
                return logits[lhs] > logits[rhs];
            });
            for (size_t i = 0; i < expected_top_k; ++i) {
                ASSERT_EQ(candidate_ids[position * expected_top_k + i], ranked_tokens[i]);
                for (size_t j = 0; j < expected_top_k; ++j) {
                    EXPECT_FLOAT_EQ(edge_scores[(position * expected_top_k + i) * expected_top_k + j],
                                    logits[ranked_tokens[i]] + logits[ranked_tokens[j]]);
                }
            }
        }
    }

    // Selection works on a copy: the Markov head model keeps its exported K.
    EXPECT_EQ(markov_head_model->output("candidate_ids").get_partial_shape()[1].get_length(), DSPARK_TEST_TOP_K);
    EXPECT_EQ(markov_head_model->get_rt_info<std::string>(std::vector<std::string>{"dflash", "top_k"}),
              std::to_string(DSPARK_TEST_TOP_K));

    for (const int64_t invalid_top_k : {int64_t{-1}, static_cast<int64_t>(DSPARK_TEST_VOCAB_SIZE) + 1}) {
        SCOPED_TRACE(invalid_top_k);
        EXPECT_THROW(select_markov_computation(invalid_top_k, markov_head_model), ov::Exception);
    }
    for (const std::string key : {"top_k", "vocab_size"}) {
        SCOPED_TRACE(key);
        const auto incomplete = make_dspark_markov_head_model(key);
        EXPECT_THROW(select_markov_computation(std::nullopt, incomplete), ov::Exception);
        EXPECT_THROW(select_markov_computation(int64_t{0}, incomplete), ov::Exception);
    }
}

TEST(DSparkModelTransforms, RejectsCandidateScorerThatDoesNotMatchTheBackbone) {
    const auto info = make_dspark_backbone_info();
    EXPECT_NO_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(), info));

    auto mismatched_rank = info;
    mismatched_rank.markov_rank = 8;
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(), mismatched_rank),
                 ov::Exception);

    auto wrong_top_k = make_dspark_candidate_scorer();
    wrong_top_k->set_rt_info(std::string("3"), {"dflash", "top_k"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(wrong_top_k, info), ov::Exception);

    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(true), info),
                 ov::Exception);

    auto stateful = make_annotated_stateful_sdpa_model();
    set_dspark_markov_head_rt_info(stateful);
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(stateful, info), ov::Exception);

    for (const std::string key : {"top_k", "vocab_size", "markov_rank", "markov_head_type"}) {
        SCOPED_TRACE(key);
        EXPECT_THROW(
            ov::genai::utils::dflash::validate_candidate_scorer(make_dspark_candidate_scorer(false, key), info),
            ov::Exception);
    }
}

TEST(DFlash2ModelTransforms, KeepsTheDraftHiddenStatesOfTheGatedCandidateScorer) {
    const auto info = make_dflash2_backbone_info();
    const auto markov_head_model = make_dflash2_markov_head_model();
    auto& core = ov::genai::utils::singleton_core();

    constexpr size_t num_positions = 2;
    std::vector<float> draft_logits(num_positions * DSPARK_TEST_VOCAB_SIZE);
    std::iota(draft_logits.begin(), draft_logits.end(), 0.0f);
    std::vector<float> draft_hidden_states(num_positions * DFLASH2_TEST_HIDDEN_SIZE, 1.0f);
    std::fill_n(draft_hidden_states.begin() + DFLASH2_TEST_HIDDEN_SIZE, DFLASH2_TEST_HIDDEN_SIZE, -1.0f);
    int64_t anchor_token = 0;
    int64_t subsequence_begins[] = {0, static_cast<int64_t>(num_positions)};
    for (const auto& [top_k, expected_top_k] :
         std::vector<std::pair<std::optional<int64_t>, size_t>>{{std::nullopt, DSPARK_TEST_TOP_K}, {int64_t{3}, 3}}) {
        SCOPED_TRACE(expected_top_k);
        const auto candidate_scorer = selected_computation(select_markov_computation(top_k, markov_head_model));
        ASSERT_EQ(candidate_scorer->inputs().size(), 4);
        EXPECT_NO_THROW(ov::genai::utils::dflash::validate_candidate_scorer(candidate_scorer, info));

        auto request = core.compile_model(candidate_scorer, "CPU").create_infer_request();
        request.set_tensor(
            "draft_logits",
            ov::Tensor(ov::element::f32, ov::Shape{num_positions, DSPARK_TEST_VOCAB_SIZE}, draft_logits.data()));
        request.set_tensor("draft_hidden_states",
                           ov::Tensor(ov::element::f32,
                                      ov::Shape{num_positions, DFLASH2_TEST_HIDDEN_SIZE},
                                      draft_hidden_states.data()));
        request.set_tensor("anchor_token_ids", ov::Tensor(ov::element::i64, ov::Shape{1}, &anchor_token));
        request.set_tensor("subsequence_begins", ov::Tensor(ov::element::i64, ov::Shape{2}, subsequence_begins));
        request.infer();
        const auto edge_scores = tensor_values(request.get_tensor("edge_scores"));
        ASSERT_EQ(edge_scores.size(), num_positions * expected_top_k * expected_top_k);
        // The best pair of a position is its top logit twice, gated by the sum of the position's hidden state.
        EXPECT_FLOAT_EQ(edge_scores[0], 2 * 15.0f + 4.0f);
        EXPECT_FLOAT_EQ(edge_scores[expected_top_k * expected_top_k], 2 * 31.0f - 4.0f);
    }

    // Without the hidden states input the scorer is not the one of a gated Markov head.
    auto vanilla_scorer = make_dspark_candidate_scorer();
    vanilla_scorer->set_rt_info(std::string("dflash2"), {"dflash", "markov_head_type"});
    EXPECT_THROW(ov::genai::utils::dflash::validate_candidate_scorer(vanilla_scorer, info), ov::Exception);
}

TEST(DFlash2ModelTransforms, RejectsTheDenseMarkovWalkOfTheGatedMarkovHead) {
    const auto markov_head_model = make_dflash2_markov_head_model();
    EXPECT_THROW(select_markov_computation(ov::genai::utils::dflash::TOP_K_FOR_MARKOV_STEP, markov_head_model),
                 ov::Exception);

    const TemporaryDirectory temporary_directory("genai_dflash2_markov_");
    ov::save_model(markov_head_model, temporary_directory.path() / "openvino_markov_head_model.xml");
    ov::AnyMap dense_walk{{"dflash_markov_rank", DSPARK_TEST_MARKOV_RANK}, {"markov_head_top_k", 0}};
    EXPECT_THROW(ov::genai::utils::dflash::read_markov_head(temporary_directory.path(), dense_walk), ov::Exception);
}

namespace {

class RecordingDrafter : public ov::genai::dflash_cb::BlockDrafter {
public:
    explicit RecordingDrafter(ov::genai::GenerationConfig config = ov::genai::utils::get_greedy_config(),
                              std::vector<std::optional<int64_t>> sampled_tokens = {})
        : m_config(std::move(config)),
          m_sampled_tokens(std::move(sampled_tokens)) {}

    const ov::genai::GenerationConfig& config() const override {
        return m_config;
    }

    void append(int64_t token, float log_prob, ov::genai::DraftProposalPtr proposal) override {
        tokens.push_back(token);
        log_probs.push_back(log_prob);
        proposals.push_back(std::move(proposal));
    }

    std::optional<int64_t> sample(const ov::Tensor& row) override {
        sampled_rows.emplace_back(row.data<const float>(), row.data<const float>() + row.get_size());
        OPENVINO_ASSERT(sampled_rows.size() <= m_sampled_tokens.size(), "The drafter has no scripted token left.");
        const auto token = m_sampled_tokens[sampled_rows.size() - 1];
        if (token) {
            tokens.push_back(*token);
        }
        return token;
    }

    std::mt19937& lattice_rng() override {
        return m_lattice_rng;
    }

    std::vector<int64_t> tokens;
    std::vector<float> log_probs;
    std::vector<ov::genai::DraftProposalPtr> proposals;
    std::vector<std::vector<float>> sampled_rows;

private:
    ov::genai::GenerationConfig m_config;
    std::vector<std::optional<int64_t>> m_sampled_tokens;
    std::mt19937 m_lattice_rng;
};

// [rows, 1, vocab] logits whose row r has the value 1 at argmax_tokens[r] and 0 elsewhere.
ov::Tensor make_argmax_logits(const std::vector<int64_t>& argmax_tokens, size_t vocab_size) {
    ov::Tensor logits(ov::element::f32, {argmax_tokens.size(), 1, vocab_size});
    std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
    for (size_t row = 0; row < argmax_tokens.size(); ++row) {
        logits.data<float>()[row * vocab_size + argmax_tokens[row]] = 1.0f;
    }
    return logits;
}

// A two-block lattice with K = 2: edge_scores[s, i, j] scores candidate j at row s after candidate i at row s - 1.
std::pair<ov::Tensor, ov::Tensor> make_two_block_lattice() {
    ov::Tensor edge_scores(ov::element::f32, {4, 2, 2});
    ov::Tensor candidate_ids(ov::element::i64, {4, 2});
    const std::vector<float> scores = {
        0.0f, 1.0f, /*unused*/ 5.0f, 5.0f,  // row 0 from the anchor: candidate 1
        3.0f, 2.0f, 0.0f, 4.0f,             // row 1 after candidate 1: candidate 1
        2.0f, 1.0f, /*unused*/ 0.0f, 9.0f,  // row 2 from the anchor: candidate 0
        1.0f, 1.0f, 0.0f, 0.0f,             // row 3 after candidate 0: a tie takes candidate 0
    };
    std::copy(scores.begin(), scores.end(), edge_scores.data<float>());
    const std::vector<int64_t> ids = {10, 11, 12, 13, 20, 21, 22, 23};
    std::copy(ids.begin(), ids.end(), candidate_ids.data<int64_t>());
    return {edge_scores, candidate_ids};
}

}  // namespace

TEST(DFlashCBCandidateSelection, ArgmaxDraftsEveryBlockFromItsOwnRows) {
    using ov::genai::dflash_cb::DraftBlock;
    // rows 0-2 belong to the first block, rows 3-4 to a split block that drafts nothing, rows 5-7 to the last block,
    // which validates two candidates near max_new_tokens
    const auto logits = make_argmax_logits({10, 11, 12, 0, 0, 20, 21, 22}, 32);
    RecordingDrafter first, last;
    ov::genai::dflash_cb::CandidateSelector{}.select({DraftBlock{7, 0, 3, 3}, DraftBlock{8, 5, 3, 2}},
                                                     logits,
                                                     {},
                                                     {&first, &last});
    ASSERT_EQ(first.tokens, (std::vector<int64_t>{10, 11, 12}));
    ASSERT_EQ(last.tokens, (std::vector<int64_t>{20, 21}));
    ASSERT_TRUE(first.sampled_rows.empty());
    ASSERT_TRUE(std::all_of(first.proposals.begin(), first.proposals.end(), [](const auto& proposal) {
        return !proposal;
    }));
}

TEST(DFlashCBCandidateSelection, RejectsBlockRowsPastTheOutput) {
    using ov::genai::dflash_cb::DraftBlock;
    ov::Tensor logits(ov::element::f32, {4, 1, 8});
    std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
    RecordingDrafter drafter;
    ASSERT_THROW(ov::genai::dflash_cb::CandidateSelector{}.select({DraftBlock{7, 2, 3, 3}}, logits, {}, {&drafter}),
                 ov::Exception);
    ASSERT_THROW(ov::genai::dflash_cb::CandidateSelector{}.select({DraftBlock{7, 0, 2, 3}}, logits, {}, {&drafter}),
                 ov::Exception);
    ASSERT_THROW(ov::genai::dflash_cb::CandidateSelector{}.select({DraftBlock{7, 0, 3, 3}}, logits, {}, {}),
                 ov::Exception);
}

TEST(DFlashCBCandidateSelection, RejectsDraftOutputsWithEmptyRows) {
    using ov::genai::dflash_cb::DraftBlock;
    const ov::Tensor logits(ov::element::f32, {3, 1, 0});
    RecordingDrafter drafter;
    ASSERT_THROW(ov::genai::dflash_cb::CandidateSelector{}.select({DraftBlock{7, 0, 3, 3}}, logits, {}, {&drafter}),
                 ov::Exception);
}

TEST(DFlashCBCandidateSelection, DraftSamplerDrawsWhenTheArgmaxIsNotTheDraft) {
    using ov::genai::dflash_cb::DraftBlock;
    const auto logits = make_argmax_logits({10, 11, 12}, 16);
    auto penalized = ov::genai::utils::get_greedy_config();
    penalized.repetition_penalty = 1.5f;
    ov::genai::GenerationConfig sampled;
    sampled.do_sample = true;
    // the sampler drew 4 from the second row, then appended no token, which ends the block
    RecordingDrafter penalized_drafter(penalized, {5, 4, std::nullopt});
    RecordingDrafter sampled_drafter(sampled, {6, 7, 8});

    ov::genai::dflash_cb::CandidateSelector selector;
    selector.select({DraftBlock{7, 0, 3, 3}}, logits, {}, {&penalized_drafter});
    selector.select({DraftBlock{7, 0, 3, 3}}, logits, {}, {&sampled_drafter});

    ASSERT_EQ(penalized_drafter.tokens, (std::vector<int64_t>{5, 4}));
    ASSERT_EQ(penalized_drafter.sampled_rows.size(), 3);
    ASSERT_EQ(sampled_drafter.tokens, (std::vector<int64_t>{6, 7, 8}));
    // every draw gets its own [1, 1, vocab] row of the backbone logits
    for (size_t row = 0; row < 3; ++row) {
        const float* expected = logits.data<const float>() + row * 16;
        ASSERT_EQ(sampled_drafter.sampled_rows[row], std::vector<float>(expected, expected + 16));
    }
}

TEST(DFlashCBCandidateSelection, GreedyPathWalksEachBlockFromItsAnchor) {
    const auto [edge_scores, candidate_ids] = make_two_block_lattice();
    using ov::genai::dflash_cb::greedy_candidate_path;
    ASSERT_EQ(greedy_candidate_path(edge_scores, candidate_ids, 0, 2), (std::vector<int64_t>{11, 13}));
    ASSERT_EQ(greedy_candidate_path(edge_scores, candidate_ids, 2, 2), (std::vector<int64_t>{20, 22}));
    ASSERT_EQ(greedy_candidate_path(edge_scores, candidate_ids, 2, 1), (std::vector<int64_t>{20}));
}

TEST(DFlashCBCandidateSelection, SampledLatticePathProposesTheSoftmaxOfEveryRow) {
    const auto [edge_scores, candidate_ids] = make_two_block_lattice();
    const float temperature = 0.5f;
    std::mt19937 rng(3);
    const auto candidates =
        ov::genai::dflash_cb::sample_candidate_path(edge_scores, candidate_ids, 0, 2, temperature, rng);

    ASSERT_EQ(candidates.size(), 2);
    const auto* scores = edge_scores.data<const float>();
    const auto* ids = candidate_ids.data<const int64_t>();
    size_t previous_index = 0;
    for (size_t position = 0; position < 2; ++position) {
        const auto& candidate = candidates[position];
        ASSERT_TRUE(candidate.proposal);
        ASSERT_EQ(candidate.proposal->token_ids, std::vector<int64_t>(ids + position * 2, ids + position * 2 + 2));
        // the row after the candidate drafted before it
        const float* row = scores + (position * 2 + previous_index) * 2;
        const float first = 1.0f / (1.0f + std::exp((row[1] - row[0]) / temperature));
        ASSERT_NEAR(candidate.proposal->probabilities[0], first, 1e-6f);
        ASSERT_NEAR(candidate.proposal->probabilities[1], 1.0f - first, 1e-6f);
        ASSERT_NO_THROW(ov::genai::detail::validate_draft_proposal(*candidate.proposal));
        const auto index = ov::genai::dflash_cb::find_candidate(candidate_ids, position, candidate.token);
        ASSERT_TRUE(index);
        ASSERT_NEAR(candidate.log_prob, std::log(candidate.proposal->probabilities[*index]), 1e-6f);
        previous_index = *index;
    }

    std::mt19937 same_seed(3);
    const auto repeated =
        ov::genai::dflash_cb::sample_candidate_path(edge_scores, candidate_ids, 0, 2, temperature, same_seed);
    ASSERT_EQ(repeated[0].token, candidates[0].token);
    ASSERT_EQ(repeated[1].token, candidates[1].token);
    ASSERT_THROW(ov::genai::dflash_cb::sample_candidate_path(edge_scores, candidate_ids, 0, 2, 0.0f, rng),
                 ov::Exception);
}

TEST(DFlashCBCandidateSelection, LogitProcessorsKeepTheArgmaxOnlyWithoutHistoryOrSampling) {
    using ov::genai::dflash_cb::logit_processors_keep_argmax;
    auto config = ov::genai::utils::get_greedy_config();
    ASSERT_TRUE(logit_processors_keep_argmax(config));
    config.logprobs = 2;
    ASSERT_TRUE(logit_processors_keep_argmax(config));

    for (const auto& change : std::vector<std::function<void(ov::genai::GenerationConfig&)>>{
             [](auto& c) { c.repetition_penalty = 1.1f; },
             [](auto& c) { c.presence_penalty = 0.1f; },
             [](auto& c) { c.frequency_penalty = 0.1f; },
             [](auto& c) { c.min_new_tokens = 1; },
             [](auto& c) { c.structured_output_config = ov::genai::StructuredOutputConfig(); },
             [](auto& c) { c.do_sample = true; },
         }) {
        auto changed = ov::genai::utils::get_greedy_config();
        change(changed);
        ASSERT_FALSE(logit_processors_keep_argmax(changed));
    }
}

TEST(DFlashCBCandidateSelection, ValidatesSubsequenceBegins) {
    const auto begins = [](std::vector<int64_t> values) {
        ov::Tensor tensor(ov::element::i64, ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), tensor.data<int64_t>());
        return tensor;
    };
    using ov::genai::dflash_cb::validate_subsequence_begins;
    EXPECT_NO_THROW(validate_subsequence_begins(begins({0, 3}), 1, 3));
    EXPECT_NO_THROW(validate_subsequence_begins(begins({0, 3, 4, 8}), 3, 8));

    EXPECT_THROW(validate_subsequence_begins(begins({0}), 0, 0), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 3, 8}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({1, 3, 4, 8}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 3, 4, 7}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 3, 3, 8}), 3, 8), ov::Exception);
    EXPECT_THROW(validate_subsequence_begins(begins({0, 5, 4, 8}), 3, 8), ov::Exception);
    int32_t i32_values[] = {0, 3};
    EXPECT_THROW(validate_subsequence_begins(ov::Tensor(ov::element::i32, ov::Shape{2}, i32_values), 1, 3),
                 ov::Exception);
}

namespace {

// A two-position lattice over a vocabulary of 6 with K = 2. Position 0 keeps tokens {1, 3}, position 1 tokens {0, 4}.
// The best path drafts 3 then 4; after token 1, token 0 is the better candidate at position 1.
struct CandidateLattice {
    ov::Tensor backbone_logits{ov::element::f32, ov::Shape{2, 1, 6}};
    ov::Tensor edge_scores{ov::element::f32, ov::Shape{2, 2, 2}};
    ov::Tensor candidate_ids{ov::element::i64, ov::Shape{2, 2}};
    ov::Tensor row{ov::element::f32, ov::Shape{1, 1, 6}};

    CandidateLattice() {
        const std::vector<float> logits = {
            1.0f, 5.0f, 0.0f, 4.0f, 2.0f, 3.0f,
            6.0f, 0.0f, 2.0f, 1.0f, 5.0f, 3.0f,
        };
        const std::vector<float> scores = {
            2.0f, 3.0f,
            2.0f, 3.0f,
            8.0f, 1.0f,
            0.0f, 9.0f,
        };
        const std::vector<int64_t> ids = {1, 3, 0, 4};
        std::copy(logits.begin(), logits.end(), backbone_logits.data<float>());
        std::copy(scores.begin(), scores.end(), edge_scores.data<float>());
        std::copy(ids.begin(), ids.end(), candidate_ids.data<int64_t>());
    }

    // Walks the lattice as the candidate selector does for a greedy request whose logit processors can change the
    // argmax: sample_token draws from the lattice row of every position, spread over the vocabulary.
    template <typename SampleToken>
    std::vector<int64_t> walk(SampleToken&& sample_token) {
        std::vector<int64_t> drafted;
        ov::genai::dflash_cb::walk_candidate_lattice(
            candidate_ids,
            0,
            2,
            [&](size_t position, size_t previous_index) -> std::optional<int64_t> {
                ov::genai::dflash_cb::fill_candidate_row(backbone_logits.data<const float>() + position * 6,
                                                         6,
                                                         edge_scores,
                                                         candidate_ids,
                                                         position,
                                                         previous_index,
                                                         row.data<float>());
                const auto token = sample_token(row);
                if (token) {
                    drafted.push_back(*token);
                }
                return token;
            });
        return drafted;
    }
};

std::optional<int64_t> argmax_token(const ov::Tensor& row) {
    const auto* data = row.data<const float>();
    return std::distance(data, std::max_element(data, data + row.get_size()));
}

// Drafts through a greedy sampler the way the draft pipeline does, with the prompt {3} and the anchor token 2.
std::vector<int64_t> draft_through_sampler(CandidateLattice& lattice, float repetition_penalty) {
    ov::genai::GenerationConfig config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = 8;
    config.repetition_penalty = repetition_penalty;
    ov::genai::Sampler sampler;
    auto group = std::make_shared<ov::genai::SequenceGroup>(0, ov::genai::TokenIds{3}, config);
    auto sequence = (*group)[0];
    sequence->append_token(2, 0.0f);
    sequence->set_status(ov::genai::SequenceStatus::RUNNING);
    group->update_processed_tokens_num(2);
    return lattice.walk([&](const ov::Tensor& row) -> std::optional<int64_t> {
        group->schedule_tokens(1);
        group->set_output_seq_len(1);
        group->set_num_validated_tokens(0);
        sampler.sample({group}, row, false);
        group->finish_iteration();
        return sequence->get_generated_ids().back();
    });
}

}  // namespace

TEST(DFlashCBCandidateSelection, WalksTheRowOfTheDraftedCandidate) {
    CandidateLattice lattice;
    EXPECT_EQ(lattice.walk(argmax_token), (std::vector<int64_t>{3, 4}));

    // A processor that masks token 3 moves the walk to the row of token 1.
    const auto mask_token_3 = [](ov::Tensor row) {
        row.data<float>()[3] = -std::numeric_limits<float>::infinity();
        return argmax_token(row);
    };
    EXPECT_EQ(lattice.walk(mask_token_3), (std::vector<int64_t>{1, 0}));

    // With both candidates masked, the token comes from the remaining backbone logits and ends the block.
    const auto mask_candidates_1_and_3 = [](ov::Tensor row) {
        row.data<float>()[1] = -std::numeric_limits<float>::infinity();
        row.data<float>()[3] = -std::numeric_limits<float>::infinity();
        return argmax_token(row);
    };
    EXPECT_EQ(lattice.walk(mask_candidates_1_and_3), (std::vector<int64_t>{5}));

    size_t calls = 0;
    const auto stop_after_one = [&](const ov::Tensor& row) -> std::optional<int64_t> {
        return calls++ == 0 ? argmax_token(row) : std::nullopt;
    };
    EXPECT_EQ(lattice.walk(stop_after_one), (std::vector<int64_t>{3}));
}

TEST(DFlashCBCandidateSelection, BreaksTiesByTheSmallerTokenIdLikeTheSampler) {
    // Position 0 ties tokens 3 and 1, listed in that order; token 1 then leads to token 4, token 3 to token 0.
    CandidateLattice lattice;
    const std::vector<int64_t> ids = {3, 1, 0, 4};
    const std::vector<float> scores = {
        5.0f, 5.0f,
        5.0f, 5.0f,
        9.0f, 0.0f,
        0.0f, 9.0f,
    };
    std::copy(ids.begin(), ids.end(), lattice.candidate_ids.data<int64_t>());
    std::copy(scores.begin(), scores.end(), lattice.edge_scores.data<float>());

    EXPECT_EQ(ov::genai::dflash_cb::greedy_candidate_path(lattice.edge_scores, lattice.candidate_ids, 0, 2),
              (std::vector<int64_t>{1, 4}));
    EXPECT_EQ(draft_through_sampler(lattice, 1.0f), (std::vector<int64_t>{1, 4}));
}

TEST(DFlashCBCandidateSelection, AppliesTheRequestLogitProcessors) {
    CandidateLattice lattice;
    EXPECT_EQ(draft_through_sampler(lattice, 1.0f), (std::vector<int64_t>{3, 4}));
    // The penalty halves the score of the prompt token 3 (3 -> 1.5), so token 1 (2) wins and token 0 follows it.
    EXPECT_EQ(draft_through_sampler(lattice, 2.0f), (std::vector<int64_t>{1, 0}));
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
            m_sampler->set_collect_draft_proposals(true);
            dflash_mode_enabled = true;
            m_dflash_mask_token_id = MASK_TOKEN;
            m_dflash_candidate_position_offset = candidate_position_offset;
            ov::Core core;
            auto request = core.compile_model(make_dflash_draft_inputs_model(), "CPU").create_infer_request();
            m_model_runner = std::make_shared<ov::genai::ModelRunner>(request, /*block_size=*/4);
            m_model_runner->enable_dflash_draft(candidate_position_offset);
            m_dflash_candidate_selector = std::make_shared<ov::genai::dflash_cb::CandidateSelector>();
        }

        ov::genai::SequenceGroup::Ptr add(uint64_t request_id,
                                          const std::vector<int64_t>& prompt,
                                          size_t num_assistant_tokens,
                                          size_t max_new_tokens = 100) {
            auto config = ov::genai::utils::get_greedy_config();
            config.num_assistant_tokens = num_assistant_tokens;
            config.max_new_tokens = max_new_tokens;
            return add(request_id, prompt, config);
        }

        ov::genai::SequenceGroup::Ptr add(uint64_t request_id,
                                          const std::vector<int64_t>& prompt,
                                          const ov::genai::GenerationConfig& config) {
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

        bool has_window(const ov::genai::SequenceGroup::Ptr& group) const {
            return m_dflash_draft_windows.count(sequence_id(group)) != 0;
        }

        const ov::genai::detail::DFlashDraftWindow& window(const ov::genai::SequenceGroup::Ptr& group) const {
            return m_dflash_draft_windows.at(sequence_id(group));
        }

        // Leaves the request as the draft multistep does after its window ran whole and sampled `candidates`.
        void finish_window(const ov::genai::SequenceGroup::Ptr& group, const std::vector<int64_t>& candidates) {
            const auto planned = window(group);
            auto sequence = group->get_running_sequences().front();
            sequence->remove_last_tokens(planned.block_rows - 1);
            auto& logit_processor = m_sampler->get_logit_processor(group->get_request_id());
            for (const int64_t token : candidates) {
                sequence->append_token(token, 0.0f);
                logit_processor.register_new_generated_token(token);
            }
            group->update_processed_tokens_num(planned.committed_rows + planned.context_rows);
            m_dflash_draft_windows.clear();
        }

        // Runs the request's whole window and drafts its block from `logits`, the gathered block rows, as the draft
        // multistep does. Returns the number of candidates drafted.
        size_t draft_window(const ov::genai::SequenceGroup::Ptr& group, const ov::Tensor& logits) {
            const auto planned = window(group);
            group->schedule_tokens(planned.context_rows + planned.block_rows);
            group->finish_iteration();
            auto sequence = group->get_running_sequences().front();
            sequence->remove_last_tokens(planned.block_rows - 1);
            const ov::genai::dflash_cb::DraftBlock block{sequence->get_generated_ids().back(),
                                                         0,
                                                         logits.get_shape()[0],
                                                         planned.candidate_rows};
            const size_t num_candidates = draft_dflash_blocks({block}, {group}, logits, {});
            group->update_processed_tokens_num(planned.committed_rows + planned.context_rows);
            m_dflash_draft_windows.clear();
            return num_candidates;
        }

        // Leaves the request as the draft multistep does after the scheduler did not run its window whole.
        void drop_window(const ov::genai::SequenceGroup::Ptr& group) {
            const auto planned = window(group);
            group->get_running_sequences().front()->remove_last_tokens(planned.block_rows - 1);
            group->update_processed_tokens_num(
                std::min(group->get_num_processed_tokens(), planned.committed_rows + planned.context_rows));
            m_dflash_draft_windows.clear();
        }

    private:
        static uint64_t sequence_id(const ov::genai::SequenceGroup::Ptr& group) {
            return group->get_running_sequences().front()->get_id();
        }
    };

    // Prompt of 8 tokens processed by the target in one chunk; the target sampled seed 7.
    ov::genai::SequenceGroup::Ptr start_with_block(DraftInstance& draft, size_t num_assistant_tokens = 3) {
        auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, num_assistant_tokens);
        draft.update_request(0, target_step({7}, 8, make_hidden_rows(8, 0.0f)), true);
        return group;
    }

    ov::genai::SequenceGroup::Ptr start_with_block(DraftInstance& draft, const ov::genai::GenerationConfig& config) {
        auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, config);
        draft.update_request(0, target_step({7}, 8, make_hidden_rows(8, 0.0f)), true);
        return group;
    }

    // Logits whose argmax per row is the given token.
    static ov::Tensor make_logits(const std::vector<int64_t>& argmax_tokens, size_t vocab_size = 128) {
        return make_argmax_logits(argmax_tokens, vocab_size);
    }

    // Logits whose row r has value v at token t for every (t, v) in rows[r], and 0 elsewhere.
    static ov::Tensor make_scored_logits(const std::vector<std::vector<std::pair<int64_t, float>>>& rows,
                                         size_t vocab_size = 128) {
        ov::Tensor logits(ov::element::f32, {rows.size(), 1, vocab_size});
        std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
        for (size_t row = 0; row < rows.size(); ++row) {
            for (const auto& [token, value] : rows[row]) {
                logits.data<float>()[row * vocab_size + token] = value;
            }
        }
        return logits;
    }

    static ov::genai::GenerationConfig sampled_config(size_t num_assistant_tokens, size_t max_new_tokens = 100) {
        ov::genai::GenerationConfig config;
        config.do_sample = true;
        config.temperature = 0.8f;
        config.top_k = 4;
        config.rng_seed = 3;
        config.num_assistant_tokens = num_assistant_tokens;
        config.max_new_tokens = max_new_tokens;
        return config;
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
    ASSERT_EQ(draft.window(group).context_rows, 4);
    ASSERT_EQ(draft.window(group).block_rows, 0);
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

TEST_F(DFlashCBPipelineTest, PartialRejectionKeepsAcceptedRowsAndRewindsTheBlock) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    // the target accepted 20, rejected 21 and corrected it to 30; it exported rows for [7, 20, 21, 22]
    auto result = draft.update_request(0, target_step({7, 20, 30}, 10, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 2);
    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 20, 30, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(group->get_num_processed_tokens(), 8);
    // rows of the seed and the accepted candidate, then the next block
    ASSERT_EQ(draft.window(group).committed_rows, 8);
    ASSERT_EQ(draft.window(group).context_rows, 2);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 2 + 4);
}

TEST_F(DFlashCBPipelineTest, FullAcceptanceTakesAllValidatedRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    auto result = draft.update_request(0, target_step({7, 20, 21, 22, 23}, 12, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(draft.window(group).context_rows, 4);
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 4 + 4);
}

TEST_F(DFlashCBPipelineTest, FullRejectionTakesOnlyTheSeedRow) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    auto result = draft.update_request(0, target_step({7, 30}, 9, make_hidden_rows(4, 100.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 3);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 30, MASK_TOKEN, MASK_TOKEN, MASK_TOKEN}));
    ASSERT_EQ(draft.window(group).context_rows, 1);
}

TEST_F(DFlashCBPipelineTest, DeferredTargetPublishesNoNewRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    // the target did not run the request: it still reports 8 processed positions and stale rows
    auto result = draft.update_request(0, target_step({7, 20, 21, 22}, 8, make_hidden_rows(8, 0.0f)), true);

    ASSERT_EQ(result.removed_tokens_cnt, 0);
    ASSERT_FALSE(draft.has_window(group));
    ASSERT_TRUE(group->is_waiting());
}

TEST_F(DFlashCBPipelineTest, BlockFollowsANewContextRow) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    // the block drafted no candidates, and the target did not run the request
    draft.finish_window(group, {});

    draft.update_request(0, target_step({7}, 8, make_hidden_rows(8, 0.0f)), true);

    // the block would follow the block rows of the request forwarded before it
    ASSERT_FALSE(draft.has_window(group));
    ASSERT_TRUE(group->is_waiting());
}

TEST_F(DFlashCBPipelineTest, TargetRecomputeReusesTheCommittedRows) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.finish_window(group, {20, 21, 22});

    // the target was preempted and recomputed positions [0, 4) before validating the candidates
    draft.update_request(0, target_step({7, 20, 21, 22}, 4, make_hidden_rows(4, 0.0f)), true);
    ASSERT_FALSE(draft.has_window(group));
    ASSERT_EQ(group->get_num_processed_tokens(), 8);

    // it recomputed [4, 10) and validated the candidates: 20 accepted, 21 corrected to 30
    draft.update_request(0, target_step({7, 20, 30}, 10, make_hidden_rows(6, 100.0f)), true);
    ASSERT_EQ(draft.window(group).committed_rows, 8);
    ASSERT_EQ(draft.window(group).context_rows, 2);
}

TEST_F(DFlashCBPipelineTest, MissedWindowLeavesTheRequestToTheTarget) {
    DraftInstance draft;
    auto group = start_with_block(draft);
    draft.drop_window(group);

    // the target generated 30 alone and exported only the seed row; the prompt rows are gone
    auto result = draft.update_request(0, target_step({7, 30}, 9, make_hidden_rows(1, 100.0f)), true);

    ASSERT_EQ(result.inserted_tokens_cnt, 1);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 30}));
    ASSERT_FALSE(draft.has_window(group));
    ASSERT_TRUE(group->is_waiting());
    ASSERT_EQ(group->get_num_processed_tokens(), 0);
}

TEST_F(DFlashCBPipelineTest, NearMaxNewTokensForwardsOnlyTheContext) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3}, 3, /*max_new_tokens=*/2);
    draft.update_request(0, target_step({3}, 4, make_hidden_rows(4, 0.0f)), true);

    // one token left for the target: no block, only the context rows are forwarded
    ASSERT_EQ(draft.window(group).block_rows, 0);
    ASSERT_EQ(generated(group), (std::vector<int64_t>{3}));
    ASSERT_EQ(group->get_num_available_tokens_for_batching(), 4);
}

TEST_F(DFlashCBPipelineTest, RestoredPrefixIsAlreadyCommitted) {
    DraftInstance draft;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, 3);
    // prefix caching restored 4 positions of both models before the first step
    group->update_processed_tokens_num(4);

    draft.update_request(0, target_step({}, 6, make_hidden_rows(2, 0.0f)), true);

    ASSERT_EQ(draft.window(group).committed_rows, 4);
    ASSERT_EQ(draft.window(group).context_rows, 2);
    ASSERT_EQ(group->get_num_processed_tokens(), 4);
}

TEST_F(DFlashCBPipelineTest, ReusedRequestIdStartsFromAFreshWindow) {
    DraftInstance draft;
    auto first = start_with_block(draft);
    draft.remove(0);

    auto second = draft.add(0, {5, 6, 7, 8}, 3);
    draft.update_request(0, target_step({}, 2, make_hidden_rows(2, 0.0f)), true);

    ASSERT_EQ(draft.window(second).committed_rows, 0);
    ASSERT_EQ(draft.window(second).context_rows, 2);
    ASSERT_EQ(draft.window(second).block_rows, 0);
}

TEST_F(DFlashCBPipelineTest, CandidatesReplaceThePlaceholders) {
    DraftInstance draft;
    auto group = start_with_block(draft);

    ASSERT_EQ(draft.draft_window(group, make_logits({20, 21, 22})), 3);

    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 20, 21, 22}));
    // the next window rewrites the block KV
    ASSERT_EQ(group->get_num_processed_tokens(), 8);
}

TEST_F(DFlashCBPipelineTest, SampledCandidatesCarryTheirProposals) {
    DraftInstance draft;
    auto group = start_with_block(draft, sampled_config(3));
    draft.draft_window(group, make_logits({20, 21, 22}));

    const auto sequence = group->get_running_sequences().front();
    const auto tokens = sequence->get_generated_ids();
    ASSERT_EQ(tokens.size(), 4);
    const auto& proposals = sequence->get_draft_proposals();
    ASSERT_EQ(proposals.size(), 4);
    // the seed is the target's token
    ASSERT_FALSE(proposals[0]);
    for (size_t i = 1; i < tokens.size(); ++i) {
        ASSERT_TRUE(proposals[i]);
        ASSERT_NO_THROW(ov::genai::detail::validate_draft_proposal(*proposals[i]));
        // top_k = 4 keeps four tokens
        ASSERT_EQ(proposals[i]->token_ids.size(), 4);
        ASSERT_GT(ov::genai::detail::proposal_probability(*proposals[i], tokens[i]), 0.0f);
    }
    // the draft KV ends at the context rows, whatever the sampler advanced
    ASSERT_EQ(group->get_num_processed_tokens(), 8);
    ASSERT_FALSE(sequence->has_finished());

    // the target accepted the first candidate and corrected the second: the proposals are not sent again
    draft.update_request(0, target_step({7, tokens[1], 200}, 10, make_hidden_rows(4, 100.0f)), true);
    ASSERT_EQ(generated(group)[2], 200);
    ASSERT_TRUE(sequence->get_draft_proposals().empty());
}

TEST_F(DFlashCBPipelineTest, BlockNearMaxNewTokensDraftsTheValidatedCandidates) {
    DraftInstance draft;
    // one token generated, four allowed: the target validates two candidates and samples the last token
    auto group = start_with_block(draft, sampled_config(3, 4));
    ASSERT_EQ(draft.window(group).candidate_rows, 2);
    const size_t block_rows = draft.window(group).block_rows;

    ASSERT_EQ(draft.draft_window(group, make_logits(std::vector<int64_t>(block_rows - 1, 20))), 2);

    ASSERT_EQ(generated(group).size(), 3);
    ASSERT_FALSE(group->get_running_sequences().front()->has_finished());
}

TEST_F(DFlashCBPipelineTest, RepetitionPenaltyCountsTheTargetTokens) {
    DraftInstance draft;
    auto config = ov::genai::utils::get_greedy_config();
    config.num_assistant_tokens = 3;
    config.repetition_penalty = 2.0f;
    auto group = draft.add(0, {0, 1, 2, 3, 4, 5, 6, 7}, config);
    // the target sampled 50, which is not a prompt token
    draft.update_request(0, target_step({50}, 8, make_hidden_rows(8, 0.0f)), true);

    // the seed is penalized below 51 in the first row
    draft.draft_window(group, make_scored_logits({{{50, 1.0f}, {51, 0.9f}}, {{21, 1.0f}}, {{22, 1.0f}}}));

    ASSERT_EQ(generated(group), (std::vector<int64_t>{50, 51, 21, 22}));
}

TEST_F(DFlashCBPipelineTest, RejectedCandidatesLeaveThePenaltyHistory) {
    DraftInstance draft;
    auto config = ov::genai::utils::get_greedy_config();
    config.num_assistant_tokens = 3;
    config.repetition_penalty = 2.0f;
    auto group = start_with_block(draft, config);
    draft.draft_window(group, make_logits({20, 21, 22}));
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 20, 21, 22}));

    // the target rejected all three candidates and sampled 30
    draft.update_request(0, target_step({7, 30}, 9, make_hidden_rows(4, 100.0f)), true);
    draft.draft_window(group,
                       make_scored_logits({{{20, 1.0f}, {23, 0.9f}}, {{21, 1.0f}, {24, 0.9f}}, {{30, 1.0f}, {25, 0.9f}}}));

    // 20 and 21 are no longer generated tokens; 30 is
    ASSERT_EQ(generated(group), (std::vector<int64_t>{7, 30, 20, 21, 25}));
}

class DFlashStrategyTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class StrategyInstance : public DFlashDecodingImpl {
    public:
        explicit StrategyInstance(ov::genai::ModelInputType input_type = ov::genai::ModelInputType::TOKENS) {
            m_model_input_type = input_type;
        }
        using DFlashDecodingImpl::make_draft_generation_config;
        using DFlashDecodingImpl::validate_request_config;
    };

    static void expect_rejected(const ov::genai::GenerationConfig& config,
                                const std::string& expected_message,
                                ov::genai::ModelInputType input_type = ov::genai::ModelInputType::TOKENS) {
        // Unsupported configurations must be rejected before add_request touches the pipelines.
        StrategyInstance strategy(input_type);
        try {
            strategy.add_request(0, ov::Tensor{}, config);
            FAIL() << "Expected the DFlash request to be rejected";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find(expected_message), std::string::npos) << exception.what();
        }
    }
};

TEST_F(DFlashStrategyTest, RejectsUnsupportedRequests) {
    const std::vector<std::pair<std::function<void(ov::genai::GenerationConfig&)>, std::string>> cases = {
        {[](ov::genai::GenerationConfig& config) { config = ov::genai::utils::get_beam_search_config(); },
         "greedy and multinomial decoding only"},
        {[](ov::genai::GenerationConfig& config) {
             config.do_sample = true;
             config.num_return_sequences = 2;
         },
         "one sequence per request"},
        {[](ov::genai::GenerationConfig& config) { config.assistant_confidence_threshold = 0.5f; },
         "assistant_confidence_threshold must be 0.f"},
        {[](ov::genai::GenerationConfig& config) { config.adapters = ov::genai::AdapterConfig{}; },
         "does not support adapters"},
    };
    for (const auto& [make_unsupported, expected_message] : cases) {
        SCOPED_TRACE(expected_message);
        auto config = ov::genai::utils::get_greedy_config();
        make_unsupported(config);
        expect_rejected(config, expected_message);
    }
}

TEST_F(DFlashStrategyTest, ValidatesVlmRequests) {
    // The target runs these modes as it does without a draft.
    auto config = ov::genai::utils::get_greedy_config();
    config.pruning_ratio = 1;
    config.echo = true;
    config.logprobs = 1;
    EXPECT_NO_THROW(StrategyInstance(ov::genai::ModelInputType::EMBEDDINGS).validate_request_config(config));

    config.return_omni_outputs = true;
    expect_rejected(config, "Omni intermediate outputs", ov::genai::ModelInputType::EMBEDDINGS);
}

TEST_F(DFlashStrategyTest, DraftConfigOnlyProposesCandidates) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 3;
    config.stop_strings = {"stop"};
    config.stop_token_ids = {5};
    config.min_new_tokens = 2;
    config.echo = true;
    config.do_sample = true;
    config.repetition_penalty = 1.2f;
    config.rng_seed = 7;

    const auto draft_config = StrategyInstance::make_draft_generation_config(config);

    EXPECT_TRUE(draft_config.ignore_eos);
    EXPECT_TRUE(draft_config.stop_strings.empty());
    EXPECT_TRUE(draft_config.stop_token_ids.empty());
    EXPECT_EQ(draft_config.min_new_tokens, 0);
    EXPECT_FALSE(draft_config.echo);
    EXPECT_EQ(draft_config.max_new_tokens, SIZE_MAX - 1);
    // the draft samples like the target, from draws independent of the target's
    EXPECT_TRUE(draft_config.is_multinomial());
    EXPECT_EQ(draft_config.repetition_penalty, 1.2f);
    EXPECT_EQ(draft_config.rng_seed, ov::genai::detail::proposal_rng_seed(7));
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

