// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

#include "eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

namespace {

constexpr const char* DFLASH_HIDDEN_STATES_RT_INFO_KEY = "hidden_states_decoder_layers";
constexpr const char* LAST_HIDDEN_STATE_OUTPUT_NAME = "last_hidden_state";

void add_dflash_hidden_state_result(std::shared_ptr<ov::Model>& model,
                                    const std::vector<ov::Output<ov::Node>>& hidden_state_outputs) {
    ov::Output<ov::Node> output_to_operate;
    if (hidden_state_outputs.size() > 1) {
        auto concat = std::make_shared<ov::op::v0::Concat>(hidden_state_outputs, -1);
        concat->set_friendly_name("dflash_hidden_states_concat");
        output_to_operate = concat->output(0);
    } else {
        output_to_operate = hidden_state_outputs[0];
    }

    auto result = std::make_shared<ov::op::v0::Result>(output_to_operate);
    result->output(0).set_names({LAST_HIDDEN_STATE_OUTPUT_NAME});
    result->set_friendly_name(LAST_HIDDEN_STATE_OUTPUT_NAME);
    result->get_rt_info()["manually_added_output"] = true;
    model->add_results({result});
}

bool is_rank3_hidden_output(const ov::Output<ov::Node>& output,
                            std::optional<size_t> expected_hidden_size = std::nullopt) {
    const auto shape = output.get_partial_shape();
    return shape.rank().is_static() && shape.rank().get_length() == 3 && shape[2].is_static() &&
           (!expected_hidden_size || static_cast<size_t>(shape[2].get_length()) == *expected_hidden_size);
}

std::shared_ptr<ov::Node> resolve_unique_live_node(const std::vector<std::shared_ptr<ov::Node>>& live_nodes,
                                                   const std::string& producer) {
    std::shared_ptr<ov::Node> match;
    size_t node_count = 0;
    for (const auto& node : live_nodes) {
        if (node->get_friendly_name() == producer) {
            match = node;
            ++node_count;
        }
    }
    OPENVINO_ASSERT(node_count == 1, "Hidden-state producer ", producer, " resolved to ", node_count, " live nodes.");
    return match;
}

ov::Output<ov::Node> resolve_live_hidden_state_output(
    const std::vector<std::shared_ptr<ov::Node>>& live_nodes,
    const std::string& producer,
    nlohmann::json::number_unsigned_t output_index,
    std::optional<size_t> expected_hidden_size = std::nullopt) {
    const auto node = resolve_unique_live_node(live_nodes, producer);
    OPENVINO_ASSERT(output_index < node->get_output_size(),
                    "Hidden-state producer ",
                    producer,
                    " has no output ",
                    output_index,
                    ".");
    const auto output = node->output(static_cast<size_t>(output_index));
    OPENVINO_ASSERT(is_rank3_hidden_output(output, expected_hidden_size),
                    "Hidden-state producer ",
                    producer,
                    " output ",
                    output_index,
                    " must be rank-3 with a static hidden width",
                    expected_hidden_size ? " matching the retained hidden width." : ".");
    return output;
}

template <typename T>
std::optional<T> get_rt_info_value(const std::shared_ptr<ov::Model>& model, const std::vector<std::string>& path) {
    if (!model->has_rt_info(path)) {
        return std::nullopt;
    }
    return model->get_rt_info<T>(path);
}

std::vector<int32_t> parse_layer_ids(const std::string& raw) {
    std::vector<int32_t> result;
    std::stringstream stream(raw);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        std::istringstream item_stream(item);
        int32_t layer_id{};
        if (!(item_stream >> layer_id) || (item_stream >> std::ws && !item_stream.eof())) {
            OPENVINO_THROW("Malformed dflash_target_layer_ids RT info value: '",
                           raw,
                           "'. Invalid layer id: '",
                           item,
                           "'.");
        }
        result.push_back(layer_id);
    }
    return result;
}

std::shared_ptr<ov::op::v0::Parameter> find_hidden_states_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "hidden_states" ||
            parameter->output(0).get_names().count("hidden_states") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::op::v0::Parameter> find_inputs_embeds_parameter(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == "inputs_embeds" ||
            parameter->output(0).get_names().count("inputs_embeds") != 0) {
            return parameter;
        }
    }
    return nullptr;
}

std::vector<ov::Input<ov::Node>> get_live_consumers(const std::shared_ptr<ov::Model>& model,
                                                     const ov::Output<ov::Node>& output) {
    std::unordered_set<const ov::Node*> live_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        live_nodes.insert(node.get());
    }

    std::vector<ov::Input<ov::Node>> consumers;
    for (auto consumer : output.get_target_inputs()) {
        if (live_nodes.count(consumer.get_node()) != 0) {
            consumers.push_back(consumer);
        }
    }
    return consumers;
}

ov::Output<ov::Node> scale_draft_embeddings(const ov::Output<ov::Node>& embeddings,
                                             float input_embedding_scale) {
    if (input_embedding_scale == 1.0f) {
        return embeddings;
    }
    auto scale = ov::op::v0::Constant::create(embeddings.get_element_type(),
                                               ov::Shape{},
                                               std::vector<float>{input_embedding_scale});
    auto scaled_embeddings = std::make_shared<ov::op::v1::Multiply>(embeddings, scale);
    scaled_embeddings->set_friendly_name("dflash_draft_embedding_scale");
    return scaled_embeddings;
}

}  // namespace

std::optional<std::vector<DFlashHiddenStateLocator>> resolve_target_hidden_state_locators(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    if (!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY)) {
        return std::nullopt;
    }

    nlohmann::json annotation;
    try {
        annotation = nlohmann::json::parse(model->get_rt_info<std::string>(DFLASH_HIDDEN_STATES_RT_INFO_KEY));
    } catch (const nlohmann::json::exception& error) {
        OPENVINO_THROW("Malformed hidden-state annotation metadata in model rt_info: ", error.what());
    }

    std::set<std::pair<std::string, nlohmann::json::number_unsigned_t>> producer_outputs;
    std::vector<DFlashHiddenStateLocator> resolved;
    resolved.reserve(target_layer_ids.size());
    try {
        const auto& layers = annotation.at("layers");
        const auto live_nodes = model->get_ordered_ops();
        for (const auto layer_id : target_layer_ids) {
            const auto key = std::to_string(layer_id);
            const auto& locator = layers.at(key);
            const auto producer = locator.at("producer").get<std::string>();
            const auto output_index =
                locator.at("output_index").get_ref<const nlohmann::json::number_unsigned_t&>();
            OPENVINO_ASSERT(producer_outputs.emplace(producer, output_index).second,
                            "Duplicate producer/output locator for requested decoder layers: ",
                            producer,
                            "[",
                            output_index,
                            "].");

            const auto output = resolve_live_hidden_state_output(live_nodes, producer, output_index);
            resolved.push_back({producer, output, static_cast<size_t>(output.get_partial_shape()[2].get_length())});
        }
    } catch (const nlohmann::json::exception& error) {
        OPENVINO_THROW("Malformed hidden-state annotation metadata: ", error.what());
    }
    return resolved;
}

void expose_target_hidden_states(std::shared_ptr<ov::Model>& model,
                                 const std::optional<std::vector<DFlashHiddenStateLocator>>& retained_locators,
                                 const std::vector<int32_t>& target_layer_ids) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    OPENVINO_ASSERT(!target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    std::vector<ov::Output<ov::Node>> outputs;

    if (retained_locators) {
        OPENVINO_ASSERT(!retained_locators->empty(), "DFlash hidden-state locators cannot be empty.");
        OPENVINO_ASSERT(retained_locators->size() == target_layer_ids.size(),
                        "DFlash hidden-state locator count does not match target_layer_ids.");
        const auto live_nodes = model->get_ordered_ops();
        outputs.reserve(retained_locators->size());
        for (const auto& retained : *retained_locators) {
            const auto current_output = resolve_live_hidden_state_output(
                live_nodes,
                retained.producer,
                retained.output.get_index(),
                retained.hidden_size);
            OPENVINO_ASSERT(current_output.get_node_shared_ptr() == retained.output.get_node_shared_ptr(),
                            "DFlash hidden-state producer ",
                            retained.producer,
                            " no longer resolves to the retained output identity.");
            outputs.push_back(retained.output);
        }
    } else {
        OPENVINO_ASSERT(!model->has_rt_info(DFLASH_HIDDEN_STATES_RT_INFO_KEY),
                        "DFlash hidden-state fallback cannot be used when target metadata is present.");
        outputs = eagle3::find_decoder_layer_hidden_state_outputs(model, target_layer_ids);
        OPENVINO_ASSERT(outputs.size() == target_layer_ids.size(),
                        "DFlash target model has no hidden-state RT info and Eagle3 fallback found ",
                        outputs.size(),
                        " outputs for ",
                        target_layer_ids.size(),
                        " requested target layers.");
    }
    add_dflash_hidden_state_result(model, outputs);
}

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (!model->has_rt_info("dflash_mode") || !model->get_rt_info<bool>("dflash_mode")) {
        return;
    }

    properties["dflash_mode"] = true;
    if (auto version = get_rt_info_value<std::string>(model, {"dflash", "version"})) {
        properties["dflash_version"] = static_cast<int64_t>(std::stoll(*version));
    }
    if (auto mask_token_id = get_rt_info_value<std::string>(model, {"dflash", "mask_token_id"})) {
        properties["dflash_mask_token_id"] = static_cast<int64_t>(std::stoll(*mask_token_id));
    }
    if (auto target_layer_ids = get_rt_info_value<std::string>(model, {"dflash", "target_layer_ids"})) {
        properties["dflash_target_layer_ids"] = parse_layer_ids(*target_layer_ids);
    }
    if (auto input_scale = get_rt_info_value<std::string>(model, {"dflash", "input_embedding_scale"})) {
        properties["dflash_input_embedding_scale"] = std::stof(*input_scale);
    }
    if (auto output_multiplier = get_rt_info_value<std::string>(model, {"dflash", "output_multiplier"})) {
        properties["dflash_output_multiplier"] = std::stof(*output_multiplier);
    }
    if (auto softcap = get_rt_info_value<std::string>(model, {"dflash", "final_logit_softcapping"})) {
        properties["dflash_final_logit_softcapping"] = std::stof(*softcap);
    }
}

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config) {
    DFlashRTInfo info;
    auto mode_it = config.find("dflash_mode");
    if (mode_it == config.end()) {
        return info;
    }

    info.dflash_mode = mode_it->second.as<bool>();
    config.erase(mode_it);
    if (!info.dflash_mode) {
        return info;
    }

    if (auto version_it = config.find("dflash_version"); version_it != config.end()) {
        info.dflash_version = version_it->second.as<int64_t>();
        config.erase(version_it);
    }
    OPENVINO_ASSERT(info.dflash_version == 1 || info.dflash_version == 2,
                    "Unsupported DFlash version: ",
                    info.dflash_version,
                    ". Supported versions are 1 and 2.");

    auto mask_it = config.find("dflash_mask_token_id");
    OPENVINO_ASSERT(mask_it != config.end(), "DFlash draft model is missing dflash_mask_token_id RT info.");
    info.mask_token_id = mask_it->second.as<int64_t>();
    config.erase(mask_it);

    auto layers_it = config.find("dflash_target_layer_ids");
    OPENVINO_ASSERT(layers_it != config.end(), "DFlash draft model is missing dflash_target_layer_ids RT info.");
    info.target_layer_ids = layers_it->second.as<std::vector<int32_t>>();
    config.erase(layers_it);
    OPENVINO_ASSERT(!info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");

    if (auto input_scale_it = config.find("dflash_input_embedding_scale"); input_scale_it != config.end()) {
        info.input_embedding_scale = input_scale_it->second.as<float>();
        config.erase(input_scale_it);
    }
    if (auto output_multiplier_it = config.find("dflash_output_multiplier"); output_multiplier_it != config.end()) {
        info.output_multiplier = output_multiplier_it->second.as<float>();
        config.erase(output_multiplier_it);
    }
    if (auto softcap_it = config.find("dflash_final_logit_softcapping"); softcap_it != config.end()) {
        info.final_logit_softcapping = softcap_it->second.as<float>();
        config.erase(softcap_it);
    }

    return info;
}

void apply_dflash_selector_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (!model->has_rt_info("dflash_selector_mode") ||
        !model->get_rt_info<bool>("dflash_selector_mode")) {
        return;
    }
    properties["dflash_selector_mode"] = true;
    if (auto version = get_rt_info_value<std::string>(model, {"dflash_selector", "dflash_version"})) {
        properties["dflash_selector_version"] = static_cast<int64_t>(std::stoll(*version));
    }
    if (auto semantics = get_rt_info_value<std::string>(model, {"dflash_selector", "score_semantics"})) {
        properties["dflash_selector_score_semantics"] = *semantics;
    }
    if (auto top_k = get_rt_info_value<std::string>(model, {"dflash_selector", "top_k"})) {
        properties["dflash_selector_top_k"] = static_cast<size_t>(std::stoull(*top_k));
    }
    if (auto hidden_size = get_rt_info_value<std::string>(model, {"dflash_selector", "hidden_size"})) {
        properties["dflash_selector_hidden_size"] = static_cast<size_t>(std::stoull(*hidden_size));
    }
    if (auto vocab_size = get_rt_info_value<std::string>(model, {"dflash_selector", "vocab_size"})) {
        properties["dflash_selector_vocab_size"] = static_cast<size_t>(std::stoull(*vocab_size));
    }
}

DFlashSelectorRTInfo extract_dflash_selector_info_from_config(ov::AnyMap& config) {
    DFlashSelectorRTInfo info;
    auto mode_it = config.find("dflash_selector_mode");
    if (mode_it == config.end()) {
        return info;
    }
    info.selector_mode = mode_it->second.as<bool>();
    config.erase(mode_it);
    if (!info.selector_mode) {
        return info;
    }

    auto extract_required = [&config](const std::string& name) {
        auto it = config.find(name);
        OPENVINO_ASSERT(it != config.end(), "DFlash-2 selector is missing ", name, " RT info.");
        auto value = it->second;
        config.erase(it);
        return value;
    };
    info.dflash_version = extract_required("dflash_selector_version").as<int64_t>();
    info.score_semantics = extract_required("dflash_selector_score_semantics").as<std::string>();
    info.top_k = extract_required("dflash_selector_top_k").as<size_t>();
    info.hidden_size = extract_required("dflash_selector_hidden_size").as<size_t>();
    info.vocab_size = extract_required("dflash_selector_vocab_size").as<size_t>();
    return info;
}

void validate_dflash_selector_model(const std::shared_ptr<ov::Model>& model,
                                    const DFlashSelectorRTInfo& info) {
    OPENVINO_ASSERT(model, "DFlash-2 selector model cannot be null.");
    OPENVINO_ASSERT(info.selector_mode, "DFlash-2 selector RT info must enable selector_mode.");
    OPENVINO_ASSERT(info.dflash_version == 2, "DFlash-2 selector version must be 2.");
    OPENVINO_ASSERT(info.score_semantics == "unary_inclusive",
                    "DFlash-2 selector score_semantics must be 'unary_inclusive'.");
    OPENVINO_ASSERT(info.top_k > 0 && info.hidden_size > 0 && info.vocab_size > 0,
                    "DFlash-2 selector structural metadata must be positive.");
    OPENVINO_ASSERT(info.top_k <= info.vocab_size,
                    "DFlash-2 selector top_k cannot exceed vocab_size.");
    OPENVINO_ASSERT(model->get_variables().empty(),
                    "DFlash-2 selector must be stateless and cannot contain variables.");
    for (const auto& node : model->get_ordered_ops()) {
        const std::string type_name = node->get_type_name();
        OPENVINO_ASSERT(type_name != "ReadValue" && type_name != "Assign",
                        "DFlash-2 selector must be stateless and cannot contain ",
                        type_name,
                        " operations.");
    }

    auto validate_input = [&model](const std::string& name,
                                   const ov::element::Type& element_type,
                                   int64_t rank) {
        OPENVINO_ASSERT(utils::has_input(model, name), "DFlash-2 selector is missing input '", name, "'.");
        const auto input = model->input(name);
        OPENVINO_ASSERT(input.get_element_type() == element_type,
                        "DFlash-2 selector input '", name, "' has an incompatible element type.");
        const auto shape = input.get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() == rank,
                        "DFlash-2 selector input '", name, "' has an incompatible rank.");
        return shape;
    };
    const auto candidate_shape = validate_input("candidate_ids", ov::element::i64, 3);
    const auto unary_shape = validate_input("unary_logits", ov::element::f32, 3);
    const auto hidden_shape = validate_input("draft_hidden_states", ov::element::f32, 3);
    validate_input("anchor_token_ids", ov::element::i64, 1);
    for (const auto& shape : {candidate_shape, unary_shape}) {
        OPENVINO_ASSERT(shape[2].is_dynamic() ||
                            static_cast<size_t>(shape[2].get_length()) == info.top_k,
                        "DFlash-2 selector candidate K dimension does not match metadata.");
    }
    OPENVINO_ASSERT(hidden_shape[2].is_dynamic() ||
                        static_cast<size_t>(hidden_shape[2].get_length()) == info.hidden_size,
                    "DFlash-2 selector hidden size does not match metadata.");
    OPENVINO_ASSERT(model->outputs().size() == 1,
                    "DFlash-2 selector must expose exactly one output.");
    const auto output = model->output();
    OPENVINO_ASSERT(output.get_names().count("edge_scores") != 0,
                    "DFlash-2 selector output must be named 'edge_scores'.");
    OPENVINO_ASSERT(output.get_element_type() == ov::element::f32,
                    "DFlash-2 selector edge_scores must be FP32.");
    const auto output_shape = output.get_partial_shape();
    OPENVINO_ASSERT(output_shape.rank().is_static() && output_shape.rank().get_length() == 4,
                    "DFlash-2 selector edge_scores must have rank 4.");
    for (size_t axis : {2, 3}) {
        OPENVINO_ASSERT(output_shape[axis].is_dynamic() ||
                            static_cast<size_t>(output_shape[axis].get_length()) == info.top_k,
                        "DFlash-2 selector edge-score K dimensions do not match metadata.");
    }
}

void reshape_draft_hidden_states_input_for_cb(std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash draft model cannot be null.");

    auto hidden_states = find_hidden_states_parameter(model);
    OPENVINO_ASSERT(hidden_states, "DFlash draft model must have 'hidden_states' input.");

    const auto draft_shape = hidden_states->get_partial_shape();
    OPENVINO_ASSERT(draft_shape.rank().is_static() && draft_shape.rank().get_length() == 3,
                    "DFlash draft hidden_states input must have rank 3.");
    OPENVINO_ASSERT(draft_shape[0].is_dynamic() || draft_shape[0].get_length() == 1,
                    "DFlash draft hidden_states input must use exported shape [1, seq_len, hidden].");

    auto original_consumers = get_live_consumers(model, hidden_states->output(0));
    OPENVINO_ASSERT(!original_consumers.empty(),
                    "DFlash draft hidden_states input has no live consumers.");
    hidden_states->set_partial_shape(ov::PartialShape({draft_shape[1], ov::Dimension(1), draft_shape[2]}));

    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                     ov::Shape{3},
                                                     std::vector<int64_t>{1, -1, 0});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(hidden_states, reshape_shape, true);
    reshape->set_friendly_name("dflash_hidden_states_cb_to_draft_layout");

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(reshape->output(0));
    }
    model->validate_nodes_and_infer_types();
}

void attach_target_lm_head_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                    const std::shared_ptr<ov::Model>& draft_model,
                                    bool keep_hidden_state) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash lm_head graft requires both target and draft models.");

    auto target_head = std::get<0>(ov::genai::utils::find_llm_matmul(main_model));
    auto target_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(target_head);
    OPENVINO_ASSERT(target_matmul,
                    "DFlash: could not locate the target lm_head MatMul to graft onto the draft.");

    // Clone ONLY the weight side (input(1)); input(0) is the target activation and is not reused.
    // Cloning recursively carries any INT4 decompression subgraph intact.
    auto weight_source = target_matmul->input_value(1);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target lm_head weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    std::shared_ptr<ov::op::v0::Result> hidden_result;
    for (const auto& result : draft_model->get_results()) {
        if (result->output(0).get_names().count(LAST_HIDDEN_STATE_OUTPUT_NAME) != 0 ||
            result->get_friendly_name() == LAST_HIDDEN_STATE_OUTPUT_NAME) {
            hidden_result = result;
            break;
        }
    }
    OPENVINO_ASSERT(hidden_result,
                    "DFlash draft model must expose a 'last_hidden_state' output to graft the target lm_head.");

    // Feed the draft post-norm hidden into a bare MatMul with the target head weight + transpose flags.
    auto draft_hidden = hidden_result->input_value(0);
    auto logits_matmul = std::make_shared<ov::op::v0::MatMul>(draft_hidden,
                                                              cloned_weight_out,
                                                              target_matmul->get_transpose_a(),
                                                              target_matmul->get_transpose_b());
    logits_matmul->set_friendly_name("dflash_grafted_lm_head");

    auto logits_result = std::make_shared<ov::op::v0::Result>(logits_matmul);
    logits_result->output(0).set_names({"logits"});
    logits_result->set_friendly_name("logits");
    logits_result->get_rt_info()["manually_added_output"] = true;

    draft_model->add_results({logits_result});
    if (!keep_hidden_state) {
        draft_model->remove_result(hidden_result);
    }
    draft_model->validate_nodes_and_infer_types();
}

void attach_target_embedding_to_draft(const std::shared_ptr<ov::Model>& main_model,
                                      const std::shared_ptr<ov::Model>& draft_model,
                                      float input_embedding_scale) {
    OPENVINO_ASSERT(main_model && draft_model, "DFlash embedding attach requires both target and draft models.");

    auto gather = eagle3::find_embedding_gather(main_model);
    OPENVINO_ASSERT(gather,
                    "DFlash: could not find the target token-embedding Gather to attach onto the draft.");

    // Clone the embedding weight (input(0)). The recursive clone is precision-agnostic: it copies a
    // plain f16/f32 Constant or an int8/int4 dequant chain intact, so the draft owns its embedding.
    auto weight_source = gather->input_value(0);
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_weight = eagle3::clone_node_recursive(weight_source.get_node_shared_ptr(), cloned_nodes);
    OPENVINO_ASSERT(cloned_weight, "DFlash: failed to clone the target embedding weight subgraph.");
    auto cloned_weight_out = cloned_weight->output(weight_source.get_index());

    auto inputs_embeds_param = find_inputs_embeds_parameter(draft_model);
    OPENVINO_ASSERT(inputs_embeds_param,
                    "DFlash draft model must expose an 'inputs_embeds' input to attach the target embedding.");

    // Snapshot the live consumers of inputs_embeds before rewiring them onto the new Gather.
    auto original_consumers = get_live_consumers(draft_model, inputs_embeds_param->output(0));
    OPENVINO_ASSERT(!original_consumers.empty(), "DFlash draft inputs_embeds input has no live consumers.");

    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto embed_gather = std::make_shared<ov::op::v8::Gather>(cloned_weight_out, input_ids, axis);
    embed_gather->set_friendly_name("dflash_draft_embedding");
    auto embedding_output = scale_draft_embeddings(embed_gather, input_embedding_scale);

    for (auto consumer : original_consumers) {
        consumer.replace_source_output(embedding_output);
    }
    draft_model->add_parameters({input_ids});
    draft_model->remove_parameter(inputs_embeds_param);
    draft_model->validate_nodes_and_infer_types();
}

void scale_draft_inputs_embeds(const std::shared_ptr<ov::Model>& draft_model,
                               float input_embedding_scale) {
    OPENVINO_ASSERT(draft_model, "DFlash draft model cannot be null.");
    if (input_embedding_scale == 1.0f) {
        return;
    }
    auto inputs_embeds = find_inputs_embeds_parameter(draft_model);
    OPENVINO_ASSERT(inputs_embeds, "DFlash VLM draft model must expose an 'inputs_embeds' input.");
    auto original_consumers = get_live_consumers(draft_model, inputs_embeds->output(0));
    OPENVINO_ASSERT(!original_consumers.empty(), "DFlash VLM draft inputs_embeds input has no live consumers.");
    const auto scaled_embeddings = scale_draft_embeddings(inputs_embeds->output(0), input_embedding_scale);
    for (auto consumer : original_consumers) {
        consumer.replace_source_output(scaled_embeddings);
    }
    draft_model->validate_nodes_and_infer_types();
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
