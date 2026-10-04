// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <openvino/op/matmul.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
#include <openvino/runtime/properties.hpp>

#include "continuous_batching/paged_attention_transformations.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace {

struct DraftCandidateToken {
    int64_t token_id;
    float log_prob;
};

struct DFlashDraftOutputs {
    ov::Tensor logits;
    ov::Tensor hidden_states;
};

struct DFlashProposalResult {
    std::vector<DraftCandidateToken> candidates;
    std::vector<DraftProposal> proposals;
};

constexpr const char* ACTIVATIONS_SCALE_FACTOR_PROPERTY = "ACTIVATIONS_SCALE_FACTOR";
constexpr float DFLASH2_GPU_MIN_ACTIVATIONS_SCALE_FACTOR = 32.0f;

bool dflash_timing_profile_enabled() {
    const char* value = std::getenv("OPENVINO_DFLASH_TIMING_PROFILE");
    return value != nullptr && std::string(value) != "0";
}

bool dflash_device_profile_enabled() {
    const char* value = std::getenv("OPENVINO_DFLASH_DEVICE_PROFILE");
    return value != nullptr && std::string(value) != "0";
}

bool dflash_component_profile_enabled() {
    const char* value = std::getenv("OPENVINO_DFLASH_COMPONENT_PROFILE");
    return value != nullptr && std::string(value) != "0";
}

std::vector<float> zero_log_probs(size_t count) {
    return std::vector<float>(count, 0.0f);
}

float get_dflash2_gpu_activations_scale_factor(const std::shared_ptr<ov::Model>& model) {
    const std::vector<std::string> rt_info_path = {"runtime_options", ACTIVATIONS_SCALE_FACTOR_PROPERTY};
    float scale_factor = DFLASH2_GPU_MIN_ACTIVATIONS_SCALE_FACTOR;
    if (model->has_rt_info(rt_info_path)) {
        const float ir_scale_factor = model->get_rt_info<float>(rt_info_path);
        OPENVINO_ASSERT(std::isfinite(ir_scale_factor) && ir_scale_factor > 0.0f,
                        "DFlash-2 IR ACTIVATIONS_SCALE_FACTOR must be a finite positive number.");
        scale_factor = std::max(ir_scale_factor, DFLASH2_GPU_MIN_ACTIVATIONS_SCALE_FACTOR);
    }
    return scale_factor;
}

bool has_compiled_input(const ov::CompiledModel& model, const std::string& name) {
    const auto inputs = model.inputs();
    return std::find_if(inputs.begin(), inputs.end(), [&](const ov::Output<const ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != inputs.end();
}

bool model_has_output(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto outputs = model->outputs();
    return std::find_if(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != outputs.end();
}

std::shared_ptr<ov::Model> build_lm_head_probe_model(const std::shared_ptr<ov::Model>& main_model) {
    auto target_head = std::get<0>(utils::find_llm_matmul(main_model));
    auto target_matmul = ov::as_type_ptr<ov::op::v0::MatMul>(target_head);
    OPENVINO_ASSERT(target_matmul, "DFlash component probe could not locate the target lm_head MatMul.");

    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           target_matmul->input_value(0).get_partial_shape());
    hidden->set_friendly_name("dflash_component_probe_hidden");
    hidden->output(0).set_names({"dflash_component_probe_hidden"});
    auto lm_head = std::make_shared<ov::op::v0::MatMul>(hidden,
                                                        target_matmul->input_value(1),
                                                        target_matmul->get_transpose_a(),
                                                        target_matmul->get_transpose_b());
    lm_head->set_friendly_name("dflash_component_probe_lm_head");
    auto result = std::make_shared<ov::op::v0::Result>(lm_head);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden});
}

void validate_target_has_no_unmanaged_state(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    std::vector<std::string> unmanaged_state_ops;
    for (const auto& op : model->get_ordered_ops()) {
        const std::string type_name = op->get_type_name();
        if (type_name == "ReadValue" || type_name == "Assign") {
            unmanaged_state_ops.push_back(op->get_friendly_name());
        }
    }
    OPENVINO_ASSERT(unmanaged_state_ops.empty(),
                    "DFlash CB/PA target model contains unmanaged state op(s) after paging conversion. ",
                    "First unmanaged op: ", unmanaged_state_ops.empty() ? std::string{} : unmanaged_state_ops.front(),
                    ". Convert KV, conv, and GatedDeltaNet state to managed paging before enabling DFlash.");
}

}  // namespace

class ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashCBDraftRunner {
public:
    DFlashCBDraftRunner(const ov::genai::ModelDesc& model_desc,
                        const Tokenizer& tokenizer,
                        const ov::genai::utils::dflash::DFlashRTInfo& rt_info,
                        const ov::genai::ModelDesc& selector_model_desc,
                        const ov::genai::utils::dflash::DFlashSelectorRTInfo& selector_rt_info,
                        bool selector_enabled,
                        EmbeddingsModel::Ptr embedding_model = nullptr,
                        std::optional<ov::genai::ModelDesc> backbone_probe_model_desc = std::nullopt,
                        std::optional<ov::genai::ModelDesc> lm_head_probe_model_desc = std::nullopt)
        : m_tokenizer(tokenizer),
          m_embedding_model(std::move(embedding_model)),
          m_request(create_draft_infer_request(model_desc,
                                               static_cast<bool>(m_embedding_model),
                                               selector_enabled,
                                               rt_info.dflash_version)),
          m_sampler(tokenizer),
          m_mask_token_id(rt_info.mask_token_id),
          m_candidate_position_offset(rt_info.candidate_position_offset),
          m_selector_enabled(selector_enabled),
          m_selector_info(selector_rt_info),
          m_timing_profile_enabled(dflash_timing_profile_enabled() || dflash_device_profile_enabled() ||
                                   dflash_component_profile_enabled()),
          m_device_profile_enabled(dflash_device_profile_enabled()),
          m_component_profile_enabled(dflash_component_profile_enabled()) {
        if (m_component_profile_enabled && backbone_probe_model_desc.has_value()) {
            m_backbone_probe_request = create_backbone_probe_request(*backbone_probe_model_desc,
                                                                      static_cast<bool>(m_embedding_model),
                                                                      rt_info.dflash_version);
            m_backbone_probe_has_beam_idx =
                has_compiled_input(m_backbone_probe_request->get_compiled_model(), "beam_idx");
        }
        if (m_component_profile_enabled && lm_head_probe_model_desc.has_value()) {
            m_lm_head_probe_request = create_lm_head_probe_request(*lm_head_probe_model_desc);
        }
        if (m_selector_enabled) {
            auto selector_desc = selector_model_desc;
            if (selector_desc.device.empty()) {
                selector_desc.device = model_desc.device;
            }
            if (m_device_profile_enabled) {
                selector_desc.properties[ov::enable_profiling.name()] = true;
            }
            m_selector_request = utils::singleton_core()
                                     .compile_model(selector_desc.model,
                                                    selector_desc.device,
                                                    selector_desc.properties)
                                     .create_infer_request();
        }
        m_has_beam_idx = has_compiled_input(m_request.get_compiled_model(), "beam_idx");
        if (m_has_beam_idx) {
            m_beam_idx = ov::Tensor(ov::element::i32, {BATCH_SIZE});
            std::fill_n(m_beam_idx.data<int32_t>(), m_beam_idx.get_size(), 0);
        }
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
    }

    void initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config) {
        const auto shape = input_ids.get_shape();
        OPENVINO_ASSERT(shape.size() == 2 && shape[0] == BATCH_SIZE && shape[1] > 0,
                        "Expected DFlash input_ids shape [1, seq_len].");
        const int64_t* ids_data = input_ids.data<const int64_t>();
        TokenIds prompt_ids(ids_data, ids_data + shape[1]);
        initialize_sampler_sequence(std::move(prompt_ids), config);
    }

    void initialize_sequence(size_t prompt_length, const GenerationConfig& config) {
        initialize_sampler_sequence(
            dflash_cb::build_placeholder_prompt_ids(prompt_length, m_tokenizer.get_pad_token_id()),
            config);
    }

    void sync_generated_tokens(const std::vector<int64_t>& target_generated_tokens) {
        auto seq = (*m_sequence_group)[0];
        if (seq->get_generated_len() > 0) {
            seq->remove_last_tokens(seq->get_generated_len());
        }
        for (auto token : target_generated_tokens) {
            seq->append_token(token, 0.0f);
        }
        m_sequence_group->update_processed_tokens_num(m_prompt_length + target_generated_tokens.size());
        seq->set_status(SequenceStatus::RUNNING);
    }

    DFlashDraftOutputs infer(int64_t seed_token, const ov::Tensor& hidden_delta, size_t candidate_count) {
        const auto preparation_start = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(hidden_delta && hidden_delta.get_size() > 0, "DFlash hidden delta must be provided.");
        const auto hidden_delta_shape = hidden_delta.get_shape();
        OPENVINO_ASSERT(hidden_delta_shape.size() == 3 && hidden_delta_shape[1] == BATCH_SIZE,
                        "DFlash draft hidden_states input must have shape [seq_len, 1, hidden].");
        const size_t hidden_delta_length = hidden_delta_shape[0];

        auto input_ids = build_input_ids(seed_token, candidate_count);
        auto position_ids = build_position_ids(hidden_delta_length, candidate_count);
        auto attention_mask = build_attention_mask(hidden_delta_length, candidate_count);
        m_request.set_tensor("hidden_states", hidden_delta);
        m_request.set_tensor("position_ids", position_ids);
        m_request.set_tensor("attention_mask", attention_mask);
        if (m_backbone_probe_request) {
            m_backbone_probe_request->set_tensor("hidden_states", hidden_delta);
            m_backbone_probe_request->set_tensor("position_ids", position_ids);
            m_backbone_probe_request->set_tensor("attention_mask", attention_mask);
        }
        if (m_embedding_model) {
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(
                m_embedding_model->get_request_queue().get());
            ov::Tensor input_embeds = m_embedding_model->infer(embeddings_request_guard.get(), input_ids);
            m_request.set_tensor("inputs_embeds", input_embeds);
            if (m_backbone_probe_request) {
                m_backbone_probe_request->set_tensor("inputs_embeds", input_embeds);
            }
            // The embeddings request owns input_embeds. Keep it reserved until
            // synchronous draft inference has consumed that tensor.
            record_input_preparation(preparation_start);
            update_inference_time(execute_inference());
            execute_backbone_probe();
        } else {
            m_request.set_tensor("input_ids", input_ids);
            if (m_backbone_probe_request) {
                m_backbone_probe_request->set_tensor("input_ids", input_ids);
            }
            record_input_preparation(preparation_start);
            update_inference_time(execute_inference());
            execute_backbone_probe();
        }
        m_committed_context_length += hidden_delta_length;
        const auto output_materialization_start = std::chrono::steady_clock::now();
        DFlashDraftOutputs outputs;
        outputs.logits = m_request.get_tensor("logits");
        if (m_selector_enabled) {
            outputs.hidden_states = m_request.get_tensor("last_hidden_state");
        }
        record_output_materialization(output_materialization_start);
        return outputs;
    }

    std::vector<DraftCandidateToken> sample_candidates(const ov::Tensor& logits, size_t candidate_count) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<DraftCandidateToken> candidates;
        candidates.reserve(candidate_count);
        const auto shape = logits.get_shape();
        for (size_t idx = 0; idx < candidate_count; ++idx) {
            ov::Tensor one_position(logits,
                                    ov::Coordinate{0, idx, 0},
                                    ov::Coordinate{1, idx + 1, shape[2]});
            auto sampled = sample_one_candidate(one_position);
            candidates.insert(candidates.end(), sampled.begin(), sampled.end());
        }
        m_raw_perf_metrics.m_batch_sizes.emplace_back(candidates.size());
        record_path_duration(start);
        return candidates;
    }

    DFlashProposalResult select_candidates(const DFlashDraftOutputs& outputs,
                                           int64_t anchor_token,
                                           size_t candidate_count) {
        OPENVINO_ASSERT(m_selector_enabled && m_selector_request,
                        "DFlash-2 selector mode is not initialized.");
        OPENVINO_ASSERT(outputs.logits.get_element_type() == ov::element::f32,
                        "DFlash-2 selector requires FP32 unary logits.");
        OPENVINO_ASSERT(outputs.hidden_states && outputs.hidden_states.get_element_type() == ov::element::f32,
                        "DFlash-2 selector requires FP32 draft hidden states.");

        const auto logits_shape = outputs.logits.get_shape();
        const auto hidden_shape = outputs.hidden_states.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == BATCH_SIZE,
                        "DFlash-2 logits must have shape [1, S, vocab].");
        OPENVINO_ASSERT(hidden_shape.size() == 3 && hidden_shape[0] == BATCH_SIZE,
                        "DFlash-2 hidden states must have shape [1, S, hidden].");
        OPENVINO_ASSERT(logits_shape[1] == hidden_shape[1],
                        "DFlash-2 logits and hidden-state proposal lengths must match.");
        OPENVINO_ASSERT(candidate_count <= logits_shape[1],
                        "DFlash-2 requested candidates exceed the draft output length.");
        OPENVINO_ASSERT(logits_shape[2] == m_selector_info.vocab_size,
                        "DFlash-2 draft vocabulary does not match selector metadata.");
        OPENVINO_ASSERT(hidden_shape[2] == m_selector_info.hidden_size,
                        "DFlash-2 draft hidden size does not match selector metadata.");

        const size_t top_k = m_selector_info.top_k;
        ov::Tensor draft_logits(outputs.logits,
                                ov::Coordinate{0, 0, 0},
                                ov::Coordinate{BATCH_SIZE, candidate_count, logits_shape[2]});

        ov::Tensor hidden_states(outputs.hidden_states,
                                 ov::Coordinate{0, 0, 0},
                                 ov::Coordinate{1, candidate_count, hidden_shape[2]});
        ov::Tensor anchor_ids(ov::element::i64, {BATCH_SIZE});
        anchor_ids.data<int64_t>()[0] = anchor_token;
        m_selector_request->set_tensor("draft_logits", draft_logits);
        m_selector_request->set_tensor("draft_hidden_states", hidden_states);
        m_selector_request->set_tensor("anchor_token_ids", anchor_ids);
        update_inference_time(execute_selector_inference());

        const auto path_start = std::chrono::steady_clock::now();
        const auto edge_scores = m_selector_request->get_tensor("edge_scores");
        const auto candidate_ids = m_selector_request->get_tensor("candidate_ids");
        OPENVINO_ASSERT(edge_scores.get_element_type() == ov::element::f32,
                        "DFlash-2 selector edge_scores must be FP32.");
        OPENVINO_ASSERT(candidate_ids.get_element_type() == ov::element::i64,
                        "DFlash-2 selector candidate_ids must be I64.");
        const auto edge_shape = edge_scores.get_shape();
        OPENVINO_ASSERT(edge_shape == ov::Shape({BATCH_SIZE, candidate_count, top_k, top_k}),
                        "DFlash-2 selector edge_scores shape does not match [1, S, K, K].");
        OPENVINO_ASSERT(candidate_ids.get_shape() == ov::Shape({BATCH_SIZE, candidate_count, top_k}),
                        "DFlash-2 selector candidate_ids shape does not match [1, S, K].");
        const auto* candidate_data = candidate_ids.data<const int64_t>();
        DFlashProposalResult result;
        result.candidates.reserve(candidate_count);
        const auto& sampling_params = m_sequence_group->get_sampling_parameters();
        if (!sampling_params.do_sample) {
            const auto selected_indices = dflash_cb::greedy_selector_path(edge_scores);
            for (size_t position = 0; position < candidate_count; ++position) {
                const size_t selected_index = selected_indices[position];
                result.candidates.push_back({candidate_data[position * top_k + selected_index], 0.0f});
            }
        } else {
            OPENVINO_ASSERT(sampling_params.temperature > 0.0f,
                            "DFlash-2 sampled selector requires temperature > 0.");
            result.proposals.reserve(candidate_count);
            const auto* edge_data = edge_scores.data<const float>();
            size_t previous_index = 0;
            for (size_t position = 0; position < candidate_count; ++position) {
                const auto* row = edge_data + (position * top_k + previous_index) * top_k;
                const float max_score = *std::max_element(row, row + top_k);
                DraftProposal proposal;
                proposal.token_ids.resize(top_k);
                proposal.probabilities.resize(top_k);
                float probability_sum = 0.0f;
                for (size_t index = 0; index < top_k; ++index) {
                    proposal.token_ids[index] = candidate_data[position * top_k + index];
                    proposal.probabilities[index] =
                        std::exp((row[index] - max_score) / sampling_params.temperature);
                    probability_sum += proposal.probabilities[index];
                }
                OPENVINO_ASSERT(probability_sum > 0.0f && std::isfinite(probability_sum),
                                "DFlash-2 selector produced an invalid proposal distribution.");
                for (auto& probability : proposal.probabilities) {
                    probability /= probability_sum;
                }
                std::discrete_distribution<size_t> distribution(proposal.probabilities.begin(),
                                                                proposal.probabilities.end());
                const size_t selected_index = distribution(m_selector_rng);
                result.candidates.push_back({proposal.token_ids[selected_index],
                                             std::log(proposal.probabilities[selected_index])});
                result.proposals.push_back(std::move(proposal));
                previous_index = selected_index;
            }
        }
        m_raw_perf_metrics.m_batch_sizes.emplace_back(result.candidates.size());
        record_path_duration(path_start);
        return result;
    }

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

    void print_timing_profile() const {
        if (!m_timing_profile_enabled || m_draft_inference_us.empty()) {
            return;
        }

        const auto mean_us = [](const std::vector<uint64_t>& values, size_t skip = 0) {
            const auto begin = values.begin() + std::min(skip, values.size() - 1);
            return static_cast<double>(std::accumulate(begin, values.end(), uint64_t{0})) /
                   std::distance(begin, values.end());
        };
        const auto draft_mean = mean_us(m_draft_inference_us);
        const auto selector_mean =
            m_selector_inference_us.empty() ? 0.0 : mean_us(m_selector_inference_us);
        const auto path_mean = m_path_selection_us.empty() ? 0.0 : mean_us(m_path_selection_us);
        const auto steady_draft_mean = mean_us(m_draft_inference_us, 2);
        const auto steady_selector_mean =
            m_selector_inference_us.empty() ? 0.0 : mean_us(m_selector_inference_us, 2);
        const auto steady_path_mean =
            m_path_selection_us.empty() ? 0.0 : mean_us(m_path_selection_us, 2);
        const auto backbone_mean =
            m_backbone_profile_us.empty() ? 0.0 : mean_us(m_backbone_profile_us);
        const auto lm_head_mean =
            m_lm_head_profile_us.empty() ? 0.0 : mean_us(m_lm_head_profile_us);

        std::cout << "DFlash timing profile (mean per proposal stage, ms): draft infer="
                  << draft_mean / 1000.0
                  << ", selector=" << selector_mean / 1000.0
                  << ", host candidate/path=" << path_mean / 1000.0
                  << ", total before verification=" << (draft_mean + selector_mean + path_mean) / 1000.0
                  << "\n";
        std::cout << "DFlash timing profile (steady state after first two proposal stages, ms): draft infer="
                  << steady_draft_mean / 1000.0
                  << ", selector=" << steady_selector_mean / 1000.0
                  << ", host candidate/path=" << steady_path_mean / 1000.0
                  << ", total before verification="
                  << (steady_draft_mean + steady_selector_mean + steady_path_mean) / 1000.0
                  << "\n";
        if (!m_backbone_profile_us.empty()) {
            const auto steady_backbone_mean = mean_us(m_backbone_profile_us, 2);
            const auto steady_lm_head_mean = mean_us(m_lm_head_profile_us, 2);
            std::cout << "DFlash device profile (mean per draft infer, ms): backbone="
                      << backbone_mean / 1000.0
                      << ", grafted lm_head=" << lm_head_mean / 1000.0
                      << ", profile sum=" << (backbone_mean + lm_head_mean) / 1000.0
                      << "\n";
            std::cout << "DFlash device profile (steady state after first two draft infers, ms): backbone="
                      << steady_backbone_mean / 1000.0
                      << ", grafted lm_head=" << steady_lm_head_mean / 1000.0
                      << ", profile sum=" << (steady_backbone_mean + steady_lm_head_mean) / 1000.0
                      << "\n";
        }
    }

    void record_hidden_materialization(std::chrono::steady_clock::time_point start) {
        if (!m_component_profile_enabled) {
            return;
        }
        m_hidden_materialization_us.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
    }

    void record_proposal_envelope(std::chrono::steady_clock::time_point start) {
        if (!m_component_profile_enabled) {
            return;
        }
        const auto measured_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        const auto backbone_probe_us =
            m_backbone_probe_inference_us.empty() ? uint64_t{0} : m_backbone_probe_inference_us.back();
        const auto lm_head_probe_us =
            m_lm_head_probe_inference_us.empty() ? uint64_t{0} : m_lm_head_probe_inference_us.back();
        OPENVINO_ASSERT(measured_us >= backbone_probe_us + lm_head_probe_us,
                        "DFlash component probe duration exceeds the enclosing proposal duration.");
        m_proposal_envelope_us.push_back(measured_us - backbone_probe_us - lm_head_probe_us);
    }

    void print_component_profile() const {
        if (!m_component_profile_enabled || m_draft_inference_us.empty() ||
            m_backbone_probe_inference_us.empty() || m_lm_head_probe_inference_us.empty()) {
            return;
        }

        const auto mean_us = [](const std::vector<uint64_t>& values, size_t skip = 0) {
            const auto begin = values.begin() + std::min(skip, values.size() - 1);
            return static_cast<double>(std::accumulate(begin, values.end(), uint64_t{0})) /
                   std::distance(begin, values.end());
        };
        const auto full_draft_us = mean_us(m_draft_inference_us, 2);
        const auto backbone_us = mean_us(m_backbone_probe_inference_us, 2);
        const auto lm_head_probe_us = mean_us(m_lm_head_probe_inference_us, 2);
        const auto integration_us = full_draft_us - backbone_us - lm_head_probe_us;
        const auto hidden_materialization_us =
            m_hidden_materialization_us.empty() ? 0.0 : mean_us(m_hidden_materialization_us, 2);
        const auto input_preparation_us =
            m_input_preparation_us.empty() ? 0.0 : mean_us(m_input_preparation_us, 2);
        const auto output_materialization_us =
            m_output_materialization_us.empty() ? 0.0 : mean_us(m_output_materialization_us, 2);
        const auto selector_us =
            m_selector_inference_us.empty() ? 0.0 : mean_us(m_selector_inference_us, 2);
        const auto path_us =
            m_path_selection_us.empty() ? 0.0 : mean_us(m_path_selection_us, 2);
        const auto total_us = hidden_materialization_us + input_preparation_us + full_draft_us +
                              output_materialization_us + selector_us + path_us;
        const auto proposal_envelope_us =
            m_proposal_envelope_us.empty() ? 0.0 : mean_us(m_proposal_envelope_us, 2);

        std::cout << "DFlash component wall profile (steady state after first two proposal stages, ms): "
                  << "hidden materialization=" << hidden_materialization_us / 1000.0
                  << ", draft input preparation=" << input_preparation_us / 1000.0
                  << ", backbone-only mirror=" << backbone_us / 1000.0
                  << ", standalone lm_head mirror=" << lm_head_probe_us / 1000.0
                  << ", graph integration delta=" << integration_us / 1000.0
                  << ", full draft infer=" << full_draft_us / 1000.0
                  << ", output materialization=" << output_materialization_us / 1000.0
                  << ", selector=" << selector_us / 1000.0
                  << ", host candidate/path=" << path_us / 1000.0
                  << ", total before verification=" << total_us / 1000.0
                  << ", measured proposal envelope=" << proposal_envelope_us / 1000.0
                  << "\n";
    }

    size_t get_consumed_hidden_states() const {
        return m_committed_context_length;
    }

private:
    void initialize_sampler_sequence(TokenIds prompt_ids, const GenerationConfig& config) {
        m_prompt_length = prompt_ids.size();
        // Sampler state is keyed by request_id; we reuse request_id=1, so clear per-request context.
        m_sampler.clear_request_info(1);
        m_sequence_group = std::make_shared<SequenceGroup>(1, prompt_ids, config);
        m_selector_rng.seed(dflash_cb::selector_rng_seed(config.rng_seed));
        m_sequence_group->update_processed_tokens_num(m_prompt_length);
        m_committed_context_length = 0;
        m_request.reset_state();
        if (m_has_beam_idx) {
            m_request.set_tensor("beam_idx", m_beam_idx);
        }
        if (m_backbone_probe_request) {
            m_backbone_probe_request->reset_state();
            if (m_backbone_probe_has_beam_idx) {
                m_backbone_probe_request->set_tensor("beam_idx", m_beam_idx);
            }
        }
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.m_durations.clear();
        m_raw_perf_metrics.m_batch_sizes.clear();
        m_draft_inference_us.clear();
        m_selector_inference_us.clear();
        m_path_selection_us.clear();
        m_backbone_profile_us.clear();
        m_lm_head_profile_us.clear();
        m_backbone_probe_inference_us.clear();
        m_lm_head_probe_inference_us.clear();
        m_input_preparation_us.clear();
        m_output_materialization_us.clear();
        m_hidden_materialization_us.clear();
        m_proposal_envelope_us.clear();
    }

    static ov::InferRequest create_draft_infer_request(const ov::genai::ModelDesc& model_desc,
                                                        bool use_external_embeddings,
                                                        bool selector_enabled,
                                                        int64_t dflash_version) {
        OPENVINO_ASSERT(model_desc.model, "DFlash draft model cannot be null.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "hidden_states"),
                        "DFlash CB/PA draft model must have 'hidden_states' input.");
        if (use_external_embeddings) {
            OPENVINO_ASSERT(utils::has_input(model_desc.model, "inputs_embeds"),
                            "DFlash VLM draft model must have an 'inputs_embeds' input.");
            OPENVINO_ASSERT(!utils::has_input(model_desc.model, "input_ids"),
                            "DFlash VLM draft model must not have an 'input_ids' input.");
        } else {
            OPENVINO_ASSERT(utils::has_input(model_desc.model, "input_ids"),
                            "DFlash CB/PA draft model must have an 'input_ids' input after load-time transforms.");
        }
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "position_ids"),
                        "DFlash CB/PA draft model must have a 'position_ids' input.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "attention_mask"),
                        "DFlash CB/PA draft model must have an 'attention_mask' input.");
        OPENVINO_ASSERT(model_has_output(model_desc.model, "logits"),
                        "DFlash CB/PA draft model must expose 'logits'.");
        if (selector_enabled) {
            OPENVINO_ASSERT(model_has_output(model_desc.model, "last_hidden_state"),
                            "DFlash-2 selector mode requires draft 'last_hidden_state'.");
        }
        auto compile_properties = model_desc.properties;
        if (dflash_version == 2 && model_desc.device.find("GPU") != std::string::npos &&
            compile_properties.count(ACTIVATIONS_SCALE_FACTOR_PROPERTY) == 0) {
            // Explicit compile properties take precedence. Otherwise promote
            // the model's IR recommendation because the GPU plugin may classify
            // the grafted stateful draft as an LLM and skip this RT-info hint.
            // Legacy values below the validated safe minimum are clamped.
            compile_properties[ACTIVATIONS_SCALE_FACTOR_PROPERTY] =
                get_dflash2_gpu_activations_scale_factor(model_desc.model);
        }
        if (dflash_device_profile_enabled()) {
            compile_properties[ov::enable_profiling.name()] = true;
        }
        if (model_desc.device == "NPU") {
            auto kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
            auto npu_compile_result = utils::compile_decoder_for_npu(model_desc.model, compile_properties, kv_axes_pos);
            return npu_compile_result.first.create_infer_request();
        }
        return utils::singleton_core()
            .compile_model(model_desc.model, model_desc.device, compile_properties)
            .create_infer_request();
    }

    static ov::InferRequest create_lm_head_probe_request(const ov::genai::ModelDesc& model_desc) {
        OPENVINO_ASSERT(model_desc.model, "DFlash lm_head probe model cannot be null.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "dflash_component_probe_hidden"),
                        "DFlash lm_head probe has an invalid hidden-state input.");
        OPENVINO_ASSERT(model_has_output(model_desc.model, "logits"),
                        "DFlash lm_head probe must expose logits.");
        return utils::singleton_core()
            .compile_model(model_desc.model, model_desc.device, model_desc.properties)
            .create_infer_request();
    }

    static ov::InferRequest create_backbone_probe_request(const ov::genai::ModelDesc& model_desc,
                                                          bool use_external_embeddings,
                                                          int64_t dflash_version) {
        OPENVINO_ASSERT(model_desc.model, "DFlash backbone probe model cannot be null.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "hidden_states"),
                        "DFlash backbone probe must have a 'hidden_states' input.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "position_ids") &&
                            utils::has_input(model_desc.model, "attention_mask"),
                        "DFlash backbone probe must have position_ids and attention_mask inputs.");
        OPENVINO_ASSERT(model_has_output(model_desc.model, "last_hidden_state"),
                        "DFlash backbone probe must expose 'last_hidden_state'.");
        if (use_external_embeddings) {
            OPENVINO_ASSERT(utils::has_input(model_desc.model, "inputs_embeds"),
                            "DFlash VLM backbone probe must have an 'inputs_embeds' input.");
        } else {
            OPENVINO_ASSERT(utils::has_input(model_desc.model, "input_ids"),
                            "DFlash backbone probe must have an 'input_ids' input.");
        }

        auto compile_properties = model_desc.properties;
        if (dflash_version == 2 && model_desc.device.find("GPU") != std::string::npos &&
            compile_properties.count(ACTIVATIONS_SCALE_FACTOR_PROPERTY) == 0) {
            compile_properties[ACTIVATIONS_SCALE_FACTOR_PROPERTY] =
                get_dflash2_gpu_activations_scale_factor(model_desc.model);
        }
        return utils::singleton_core()
            .compile_model(model_desc.model, model_desc.device, compile_properties)
            .create_infer_request();
    }

    ov::Tensor build_input_ids(int64_t seed_token, size_t candidate_count) const {
        return dflash_cb::build_draft_input_ids(
            seed_token, m_mask_token_id, candidate_count, m_candidate_position_offset);
    }

    ov::Tensor build_position_ids(size_t hidden_delta_length, size_t candidate_count) const {
        return dflash_cb::build_draft_position_ids(
            m_committed_context_length, hidden_delta_length, candidate_count, m_candidate_position_offset);
    }

    ov::Tensor build_attention_mask(size_t hidden_delta_length, size_t candidate_count) const {
        return dflash_cb::build_draft_attention_mask(
            m_committed_context_length, hidden_delta_length, candidate_count, m_candidate_position_offset);
    }

    std::vector<DraftCandidateToken> sample_one_candidate(const ov::Tensor& logits) {
        const auto sequence = (*m_sequence_group)[0];
        const size_t generated_before = sequence->get_generated_len();
        m_sequence_group->schedule_tokens(1);
        m_sequence_group->set_output_seq_len(1);
        m_sequence_group->set_num_validated_tokens(0);
        m_sampler.sample({m_sequence_group}, logits, false);
        m_sequence_group->finish_iteration();

        const auto& generated = sequence->get_generated_ids();
        if (generated.size() <= generated_before) {
            return {};
        }
        const auto& log_probs = sequence->get_generated_log_probs();
        OPENVINO_ASSERT(log_probs.size() >= generated.size(), "Generated token log-probs are out of sync.");
        std::vector<DraftCandidateToken> candidates;
        candidates.reserve(generated.size() - generated_before);
        for (size_t idx = generated_before; idx < generated.size(); ++idx) {
            candidates.push_back({generated[idx], log_probs[idx]});
        }
        return candidates;
    }

    uint64_t execute_inference() {
        auto start = std::chrono::steady_clock::now();
        m_request.infer();
        const auto duration_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        record_draft_profile(duration_us);
        return duration_us;
    }

    void execute_backbone_probe() {
        if (!m_backbone_probe_request) {
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        m_backbone_probe_request->infer();
        if (m_component_profile_enabled) {
            m_backbone_probe_inference_us.push_back(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
        }
        if (m_lm_head_probe_request) {
            m_lm_head_probe_request->set_tensor("dflash_component_probe_hidden",
                                                m_backbone_probe_request->get_tensor("last_hidden_state"));
            const auto lm_head_start = std::chrono::steady_clock::now();
            m_lm_head_probe_request->infer();
            if (m_component_profile_enabled) {
                m_lm_head_probe_inference_us.push_back(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - lm_head_start).count()));
            }
        }
    }

    uint64_t execute_selector_inference() {
        auto start = std::chrono::steady_clock::now();
        m_selector_request->infer();
        const auto duration_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        if (m_timing_profile_enabled) {
            m_selector_inference_us.push_back(duration_us);
        }
        return duration_us;
    }

    void record_draft_profile(uint64_t wall_duration_us) {
        if (!m_timing_profile_enabled) {
            return;
        }
        m_draft_inference_us.push_back(wall_duration_us);

        if (!m_device_profile_enabled) {
            return;
        }

        uint64_t profile_total_us = 0;
        uint64_t lm_head_us = 0;
        for (const auto& info : m_request.get_profiling_info()) {
            const auto duration_us = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(info.real_time).count());
            profile_total_us += duration_us;
            // The GPU plugin may report the grafted MatMul under an internal
            // name/type rather than dflash_grafted_lm_head. It is still the
            // uniquely longest draft kernel: [B, S, hidden] x [hidden, vocab].
            lm_head_us = std::max(lm_head_us, duration_us);
        }
        m_backbone_profile_us.push_back(profile_total_us - lm_head_us);
        m_lm_head_profile_us.push_back(lm_head_us);
    }

    void record_input_preparation(std::chrono::steady_clock::time_point start) {
        if (!m_component_profile_enabled) {
            return;
        }
        m_input_preparation_us.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
    }

    void record_output_materialization(std::chrono::steady_clock::time_point start) {
        if (!m_component_profile_enabled) {
            return;
        }
        m_output_materialization_us.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
    }

    void record_path_duration(std::chrono::steady_clock::time_point start) {
        if (!m_timing_profile_enabled) {
            return;
        }
        m_path_selection_us.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
    }

    void update_inference_time(uint64_t inference_time_us) {
        m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
        m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
    }

    static constexpr size_t BATCH_SIZE = 1;
    Tokenizer m_tokenizer;
    EmbeddingsModel::Ptr m_embedding_model;
    mutable ov::InferRequest m_request;
    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    bool m_has_beam_idx = false;
    ov::Tensor m_beam_idx;
    size_t m_prompt_length = 0;
    size_t m_committed_context_length = 0;
    int64_t m_mask_token_id = -1;
    size_t m_candidate_position_offset = 1;
    bool m_selector_enabled = false;
    ov::genai::utils::dflash::DFlashSelectorRTInfo m_selector_info;
    std::optional<ov::InferRequest> m_selector_request;
    std::mt19937 m_selector_rng;
    bool m_timing_profile_enabled = false;
    bool m_device_profile_enabled = false;
    bool m_component_profile_enabled = false;
    std::optional<ov::InferRequest> m_backbone_probe_request;
    bool m_backbone_probe_has_beam_idx = false;
    std::optional<ov::InferRequest> m_lm_head_probe_request;
    std::vector<uint64_t> m_draft_inference_us;
    std::vector<uint64_t> m_selector_inference_us;
    std::vector<uint64_t> m_path_selection_us;
    std::vector<uint64_t> m_backbone_profile_us;
    std::vector<uint64_t> m_lm_head_profile_us;
    std::vector<uint64_t> m_backbone_probe_inference_us;
    std::vector<uint64_t> m_lm_head_probe_inference_us;
    std::vector<uint64_t> m_input_preparation_us;
    std::vector<uint64_t> m_output_materialization_us;
    std::vector<uint64_t> m_hidden_materialization_us;
    std::vector<uint64_t> m_proposal_envelope_us;
};

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const ov::genai::utils::dflash::DFlashRTInfo& rt_info,
    const ov::genai::ModelDesc& selector_model_desc,
    const ov::genai::utils::dflash::DFlashSelectorRTInfo& selector_rt_info)
    : m_rt_info(rt_info),
      m_selector_enabled(static_cast<bool>(selector_model_desc.model)) {
    OPENVINO_ASSERT(m_rt_info.dflash_mode, "DFlash continuous batching requires dflash_mode=true.");
    OPENVINO_ASSERT(!m_rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    OPENVINO_ASSERT(!main_model_desc.scheduler_config.enable_prefix_caching,
                    "DFlash CB/PA does not support scheduler_config.enable_prefix_caching.");
    OPENVINO_ASSERT(m_rt_info.input_embedding_scale > 0.0f,
                    "DFlash input_embedding_scale must be positive.");
    OPENVINO_ASSERT(m_rt_info.output_multiplier > 0.0f,
                    "DFlash output_multiplier must be positive.");
    OPENVINO_ASSERT(m_rt_info.final_logit_softcapping >= 0.0f,
                    "DFlash final_logit_softcapping cannot be negative.");

    OPENVINO_ASSERT(main_model_desc.model && draft_model_desc.model,
                    "DFlash requires both target and draft models.");
    OPENVINO_ASSERT(!utils::is_npu_requested(main_model_desc.device, main_model_desc.properties) &&
                        !utils::is_npu_requested(draft_model_desc.device, draft_model_desc.properties),
                    "DFlash speculative decoding requires the Continuous Batching/Paged Attention backend on CPU or GPU; "
                    "NPU is not supported.");
    if (m_selector_enabled) {
        OPENVINO_ASSERT(m_rt_info.dflash_version == 2,
                        "selector_model() requires a DFlash version 2 backbone.");
        utils::dflash::validate_dflash_selector_model(selector_model_desc.model, selector_rt_info);
        const auto& selector_device =
            selector_model_desc.device.empty() ? draft_model_desc.device : selector_model_desc.device;
        OPENVINO_ASSERT(!utils::is_npu_requested(selector_device, selector_model_desc.properties),
                        "DFlash-2 selector_model() does not support NPU.");
    }

    // DFlash adds outputs and rewrites inputs on both graphs. Keep the caller's
    // models reusable if construction later fails or another pipeline is built.
    auto main_model = main_model_desc.model->clone();
    auto draft_model_desc_for_runner = draft_model_desc;
    draft_model_desc_for_runner.model = draft_model_desc.model->clone();
    const bool is_vlm_dflash = static_cast<bool>(main_model_desc.inputs_embedder);
    if (is_vlm_dflash) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_vision_registry = std::make_shared<VisionRegistry>();
    }
    const auto draft_embedding_model =
        m_inputs_embedder ? m_inputs_embedder->get_embedding_model() : nullptr;

    // Detect each draft capability independently from its I/O signature (no RT-info markers) and
    // apply only the missing transform(s); a well-formed draft has exactly one input/output per pair.
    const bool needs_embedding_attach = utils::has_input(draft_model_desc_for_runner.model, "inputs_embeds");
    const bool needs_lm_head_graft = model_has_output(draft_model_desc_for_runner.model, "last_hidden_state");
    OPENVINO_ASSERT(needs_embedding_attach != utils::has_input(draft_model_desc_for_runner.model, "input_ids"),
                    "DFlash draft model must have exactly one of 'inputs_embeds' or 'input_ids' input.");
    OPENVINO_ASSERT(needs_lm_head_graft != model_has_output(draft_model_desc_for_runner.model, "logits"),
                    "DFlash draft model must have exactly one of 'last_hidden_state' or 'logits' output.");

    // Resolve authoritative metadata before mutating either model so a malformed target leaves the draft reusable.
    auto retained_hidden_state_locators =
        utils::dflash::resolve_target_hidden_state_locators(main_model, m_rt_info.target_layer_ids);

    // Preserve the VLM draft's external inputs_embeds bridge. The legacy LLM
    // path still grafts an in-graph target embedding and remains input_ids-native.
    if (is_vlm_dflash) {
        OPENVINO_ASSERT(needs_embedding_attach,
                        "DFlash VLM draft model must be inputs_embeds-native.");
        utils::dflash::scale_draft_inputs_embeds(draft_model_desc_for_runner.model, m_rt_info.input_embedding_scale);
    } else if (needs_embedding_attach) {
        utils::dflash::attach_target_embedding_to_draft(main_model,
                                                        draft_model_desc_for_runner.model,
                                                        m_rt_info.input_embedding_scale);
    }
    std::shared_ptr<ov::Model> backbone_probe_model;
    std::shared_ptr<ov::Model> lm_head_probe_model;
    if (dflash_component_profile_enabled()) {
        backbone_probe_model = draft_model_desc_for_runner.model->clone();
        lm_head_probe_model = build_lm_head_probe_model(main_model);
    }
    if (needs_lm_head_graft) {
        utils::dflash::attach_target_lm_head_to_draft(main_model,
                                                      draft_model_desc_for_runner.model,
                                                      m_selector_enabled || dflash_component_profile_enabled());
    }

    const bool target_has_linear_attention = utils::get_cache_types(*main_model).has_linear();
    const bool allow_score_aggregation = true;
    const bool allow_cache_rotation = false;
    const bool allow_xattention = false;
    const bool allow_adaptive_rkv = false;
    const bool allow_qq_bias = main_model_desc.properties.count("query_to_query_bias") > 0 &&
                               main_model_desc.properties.at("query_to_query_bias").as<bool>();
    ov::pass::SDPAToPagedAttention(main_model_desc.scheduler_config.use_cache_eviction,
                                   main_model_desc.scheduler_config.use_cache_eviction,
                                   allow_score_aggregation,
                                   allow_cache_rotation,
                                   allow_xattention,
                                   allow_adaptive_rkv,
                                   allow_qq_bias)
        .run_on_model(main_model);
    utils::apply_gather_before_matmul_transformation(main_model);
    validate_target_has_no_unmanaged_state(main_model);
    utils::dflash::expose_target_hidden_states(
        main_model,
        retained_hidden_state_locators,
        m_rt_info.target_layer_ids);

    m_tokenizer = main_model_desc.tokenizer;
    auto main_generation_config = main_model_desc.generation_config;
    dflash_cb::ensure_num_assistant_tokens_is_set(main_generation_config);
    OPENVINO_ASSERT(main_model_desc.scheduler_config.max_num_batched_tokens >=
                        main_generation_config.num_assistant_tokens.value() + 1,
                    "DFlash CB/PA requires max_num_batched_tokens >= num_assistant_tokens + 1 while it is limited ",
                    "to one active request and one running sequence.");
    m_generation_config = main_generation_config;
    auto target_scheduler_config = main_model_desc.scheduler_config;
    target_scheduler_config.num_linear_attention_blocks =
        dflash_cb::adjusted_linear_attention_block_count(target_scheduler_config.num_linear_attention_blocks,
                                                          main_generation_config.num_assistant_tokens.value(),
                                                          target_has_linear_attention);
    if (draft_model_desc_for_runner.device.empty()) {
        draft_model_desc_for_runner.device = main_model_desc.device;
    }
    utils::dflash::reshape_draft_hidden_states_input_for_cb(draft_model_desc_for_runner.model);
    if (backbone_probe_model) {
        utils::dflash::reshape_draft_hidden_states_input_for_cb(backbone_probe_model);
    }

    std::optional<ov::genai::ModelDesc> backbone_probe_model_desc;
    if (backbone_probe_model) {
        backbone_probe_model_desc = draft_model_desc_for_runner;
        backbone_probe_model_desc->model = backbone_probe_model;
    }
    std::optional<ov::genai::ModelDesc> lm_head_probe_model_desc;
    if (lm_head_probe_model) {
        lm_head_probe_model_desc = draft_model_desc_for_runner;
        lm_head_probe_model_desc->model = lm_head_probe_model;
    }

    m_draft = std::make_shared<DFlashCBDraftRunner>(draft_model_desc_for_runner,
                                                    m_tokenizer,
                                                    m_rt_info,
                                                    selector_model_desc,
                                                    selector_rt_info,
                                                    m_selector_enabled,
                                                    draft_embedding_model,
                                                    std::move(backbone_probe_model_desc),
                                                    std::move(lm_head_probe_model_desc));

    if (is_vlm_dflash) {
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
            main_model,
            m_inputs_embedder,
            main_model_desc.tokenizer,
            main_generation_config,
            target_scheduler_config,
            main_model_desc.device,
            main_model_desc.properties,
            true);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
            main_model,
            main_model_desc.tokenizer,
            main_generation_config,
            target_scheduler_config,
            main_model_desc.device,
            main_model_desc.properties,
            true);
    }
    m_main_pipeline->set_hidden_state_export_needed(true);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

GenerationConfig ContinuousBatchingPipeline::DFlashDecodingImpl::make_draft_generation_config(
    const GenerationConfig& config) const {
    auto draft_config = config;
    draft_config.ignore_eos = true;
    draft_config.stop_strings = {};
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        // The target sampler retains these options and sees verified token
        // history. The placeholder-backed draft sampler must not apply state
        // that it cannot reconstruct after a speculative rejection.
        draft_config.repetition_penalty = 1.0f;
        draft_config.presence_penalty = 0.0f;
        draft_config.frequency_penalty = 0.0f;
        draft_config.no_repeat_ngram_size = std::numeric_limits<size_t>::max();
        draft_config.min_new_tokens = 0;
        draft_config.stop_token_ids = {};
        draft_config.include_stop_str_in_output = false;
        draft_config.structured_output_config.reset();
        draft_config.parsers.clear();
    }
    draft_config.max_new_tokens = config.max_new_tokens + config.num_assistant_tokens.value();
    draft_config.num_assistant_tokens = 0;
    return draft_config;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::append_pending_hidden_delta(RequestState& state,
                                                                                const ov::Tensor& hidden_delta,
                                                                                bool copy_data) {
    state.pending_hidden_deltas.append(hidden_delta, copy_data);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_pending_hidden_delta(const RequestState& state) {
    return !state.pending_hidden_deltas.empty();
}

ov::Tensor ContinuousBatchingPipeline::DFlashDecodingImpl::materialize_pending_hidden_delta(
    const RequestState& state) {
    return state.pending_hidden_deltas.materialize();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::clear_pending_hidden_delta(RequestState& state) {
    state.pending_hidden_deltas.clear();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_hidden_prefix_length(const RequestState& state) const {
    OPENVINO_ASSERT(!state.generated_tokens.empty(),
                    "DFlash hidden prefix can only be validated after target generated a seed token.");
    const size_t expected = state.prompt_len + state.generated_tokens.size() - 1;
    const size_t actual = m_draft->get_consumed_hidden_states() + state.pending_hidden_deltas.token_count();
    OPENVINO_ASSERT(actual == expected, "DFlash hidden prefix length mismatch before draft inference.");
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_active_request_state() const {
    for (const auto& [_, state] : m_request_states) {
        if (!state.finished) {
            return true;
        }
    }
    return false;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_finished_request_states() {
    for (auto state_it = m_request_states.begin(); state_it != m_request_states.end();) {
        if (state_it->second.finished) {
            state_it = m_request_states.erase(state_it);
        } else {
            ++state_it;
        }
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    const bool is_vlm_dflash = m_model_input_type == ModelInputType::EMBEDDINGS;
    OPENVINO_ASSERT(sampling_params.is_greedy_decoding() ||
                        (m_selector_enabled && sampling_params.is_multinomial()),
                    "Sampled DFlash decoding requires the DFlash-2 selector.");
    OPENVINO_ASSERT(sampling_params.num_beams == 1, "DFlash CB/PA does not support beam search.");
    OPENVINO_ASSERT(sampling_params.num_return_sequences == 1, "DFlash CB/PA supports one sequence per request.");
    OPENVINO_ASSERT(!sampling_params.adapters.has_value(),
                    "DFlash CB/PA does not support adapters until target and draft adapter parity is validated.");
    if (is_vlm_dflash) {
        dflash_cb::ensure_vlm_generation_config(sampling_params);
    }
    drop_finished_request_states();
    OPENVINO_ASSERT(!has_active_request_state() && !m_main_pipeline->has_non_finished_requests(),
                    "DFlash CB/PA POC supports only one active request. Wait for the current request to finish before adding another.");

    const auto input_shape = input_ids.get_shape();
    if (is_vlm_dflash) {
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::f32 &&
                            input_shape.size() == 3 && input_shape[0] == 1 && input_shape[1] > 0,
                        "DFlash VLM expects main input embeddings with shape [1, rows, hidden].");
    } else {
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::i64 &&
                            input_shape.size() == 2 && input_shape[0] == 1 && input_shape[1] > 0,
                        "Expected DFlash input_ids shape [1, seq_len].");
    }
    auto sampling_params_copy = sampling_params;
    dflash_cb::ensure_num_assistant_tokens_is_set(sampling_params_copy);
    RequestState state;
    state.generation_config = sampling_params_copy;
    state.prompt_len = input_shape[1];
    if (is_vlm_dflash) {
        // FIXME: The draft's token-mode SequenceGroup derives prompt length from
        // prompt_ids.size(), but DFlash receives the full prompt through target
        // hidden states rather than prompt IDs. This creates placeholder IDs for
        // sampler-length bookkeeping only; they are never fed to either model.
        // Replace this when SequenceGroup supports a sampler-only logical prompt length.
        m_draft->initialize_sequence(state.prompt_len, make_draft_generation_config(sampling_params_copy));
    } else {
        m_draft->initialize_sequence(input_ids, make_draft_generation_config(sampling_params_copy));
    }
    m_request_states[request_id] = std::move(state);

    // The draft sampler and request state are initialized above. If target
    // request creation fails, erase the state so a later request is not
    // rejected as a stale active DFlash request.
    try {
        return m_main_pipeline->add_request(request_id,
                                            input_ids,
                                            sampling_params_copy,
                                            prompt_ids,
                                            lm_extra_inputs);
    } catch (...) {
        m_request_states.erase(request_id);
        throw;
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        return ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(
            request_id,
            prompt,
            {},
            sampling_params);
    }
    auto input_ids = m_tokenizer.encode(prompt).input_ids;
    return add_request(request_id, input_ids, sampling_params);
}

bool ContinuousBatchingPipeline::DFlashDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;
    const auto step_start = std::chrono::steady_clock::now();

    m_main_pipeline->pull_awaiting_requests();

    std::map<uint64_t, size_t> draft_generated_by_request;
    const auto draft_start = std::chrono::steady_clock::now();
    for (auto& [request_id, state] : m_request_states) {
        if (state.finished) {
            clear_pending_hidden_delta(state);
            state.draft_generated = 0;
            continue;
        }
        if (!has_pending_hidden_delta(state) || state.generated_tokens.empty()) {
            state.draft_generated = 0;
            continue;
        }

        const size_t generated_len = state.generated_tokens.size();
        if (generated_len >= state.generation_config.max_new_tokens) {
            clear_pending_hidden_delta(state);
            state.draft_generated = 0;
            continue;
        }

        const size_t draft_count =
            dflash_cb::draft_candidate_count(state.generation_config.num_assistant_tokens.value(),
                                            generated_len,
                                            state.generation_config.max_new_tokens);
        const size_t validation_count =
            dflash_cb::validation_candidate_count(draft_count, generated_len, state.generation_config.max_new_tokens);
        if (validation_count == 0) {
            clear_pending_hidden_delta(state);
            state.draft_generated = 0;
            continue;
        }

        const int64_t seed_token = state.generated_tokens.back();
        validate_hidden_prefix_length(state);
        const auto proposal_start = std::chrono::steady_clock::now();
        const auto hidden_materialization_start = std::chrono::steady_clock::now();
        auto hidden_delta = materialize_pending_hidden_delta(state);
        m_draft->record_hidden_materialization(hidden_materialization_start);
        auto draft_outputs = m_draft->infer(seed_token, hidden_delta, draft_count);
        clear_pending_hidden_delta(state);
        std::vector<DraftCandidateToken> candidates;
        std::vector<DraftProposal> draft_proposals;
        if (m_selector_enabled) {
            auto proposal_result = m_draft->select_candidates(draft_outputs, seed_token, validation_count);
            candidates = std::move(proposal_result.candidates);
            draft_proposals = std::move(proposal_result.proposals);
        } else {
            candidates = m_draft->sample_candidates(draft_outputs.logits, validation_count);
        }

        state.generated_before_draft = state.generated_tokens.size();
        state.draft_generated = candidates.size();
        OPENVINO_ASSERT(!candidates.empty(),
                        "DFlash draft sampler produced no candidates despite requested validation candidates. ",
                        "request_id=", request_id,
                        ", generated_before_draft=", state.generated_before_draft,
                        ", draft_count=", draft_count,
                        ", validation_count=", validation_count);
        draft_generated_by_request[request_id] = candidates.size();

        auto candidate_tokens = state.generated_tokens;
        auto candidate_log_probs = zero_log_probs(candidate_tokens.size());
        std::vector<DraftProposal> aligned_proposals;
        for (const auto& candidate : candidates) {
            candidate_tokens.push_back(candidate.token_id);
            candidate_log_probs.push_back(candidate.log_prob);
        }
        if (!draft_proposals.empty()) {
            aligned_proposals.resize(state.generated_tokens.size());
            aligned_proposals.insert(aligned_proposals.end(),
                                     std::make_move_iterator(draft_proposals.begin()),
                                     std::make_move_iterator(draft_proposals.end()));
            OPENVINO_ASSERT(aligned_proposals.size() == candidate_tokens.size(),
                            "DFlash-2 sparse proposal rows must align with candidate tokens.");
        }
        OPENVINO_ASSERT(candidate_tokens.size() == candidate_log_probs.size(),
                        "DFlash draft candidate tokens and log-probs must stay aligned.");
        GeneratedSequences candidate_sequences;
        candidate_sequences.emplace(0,
                                    GeneratedSequence(candidate_tokens,
                                                      candidate_log_probs,
                                                      0,
                                                      {},
                                                      nullptr,
                                                      std::move(aligned_proposals)));
        m_draft->record_proposal_envelope(proposal_start);
        m_main_pipeline->update_request(request_id, candidate_sequences, false);
    }
    const auto draft_end = std::chrono::steady_clock::now();
    m_sd_metrics.draft_duration += PerfMetrics::get_microsec(draft_end - draft_start) / 1e6;

    const auto main_start = std::chrono::steady_clock::now();
    // Main VLM validation consumes generated IDs as embeddings, not raw IDs.
    // Synchronize after candidate insertion and before target validation.
    m_main_pipeline->sync_generated_embeddings();
    m_main_pipeline->step();
    const auto main_end = std::chrono::steady_clock::now();
    const auto main_duration = PerfMetrics::get_microsec(main_end - main_start);
    m_sd_metrics.main_duration += main_duration / 1e6;

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    update_draft_states_from_main(main_generated_requests);
    for (auto& [request_id, state] : m_request_states) {
        if (main_generated_requests.find(request_id) == main_generated_requests.end()) {
            state.finished = true;
        }
    }

    for (const auto& [request_id, draft_generated] : draft_generated_by_request) {
        auto state_it = m_request_states.find(request_id);
        if (state_it == m_request_states.end()) {
            continue;
        }
        auto& state = state_it->second;
        const auto accounting =
            dflash_cb::validation_accounting(draft_generated, state.generated_before_draft, state.generated_tokens.size());
        if (!accounting.target_extended) {
            continue;
        }
        const float acceptance_rate =
            draft_generated > 0 ? static_cast<float>(accounting.accepted) / draft_generated * 100.0f : 0.0f;
        m_sd_metrics.update_draft_generated_len(request_id, draft_generated);
        m_sd_metrics.update_draft_accepted_tokens(request_id, accounting.accepted);
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate);
    }

    m_pipeline_metrics = m_main_pipeline->get_metrics();

    const auto step_end = std::chrono::steady_clock::now();
    const auto step_microsec_duration = PerfMetrics::get_microsec(step_end - step_start);
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        raw_perf_counters.m_token_infer_durations.emplace_back(step_microsec_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(step_microsec_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_end);
        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);

        auto main_pipeline_metrics = m_main_pipeline->get_metrics();
        main_raw_perf_counters.m_durations.push_back(MicroSeconds(main_duration));
        main_raw_perf_counters.m_inference_durations[0] += MicroSeconds(main_pipeline_metrics.inference_duration);
        main_raw_perf_counters.m_batch_sizes.push_back(num_generated_tokens);
        m_sd_metrics.update_generated_len(num_generated_tokens);
    }

    if (main_generated_requests.empty()) {
        drop_finished_request_states();
        if (utils::env_setup_for_print_debug_info()) {
            m_sd_metrics.print(true);
            m_sd_metrics.clean_up();
        }
    }

}

void ContinuousBatchingPipeline::DFlashDecodingImpl::update_draft_states_from_main(
    const GeneratedRequests& main_generated_requests) {
    for (const auto& [request_id, generated_sequences] : main_generated_requests) {
        auto state_it = m_request_states.find(request_id);
        if (state_it == m_request_states.end() || generated_sequences.empty()) {
            continue;
        }

        auto& state = state_it->second;
        const auto& generated_sequence = generated_sequences.begin()->second;
        const auto accounting =
            dflash_cb::validation_accounting(state.draft_generated,
                                             state.generated_before_draft,
                                             generated_sequence.token_ids.size());

        auto hidden_delta = dflash_cb::truncate_normalized_hidden_state_from_end(generated_sequence.hidden_states,
                                                                                 accounting.rejected);
        append_pending_hidden_delta(state, hidden_delta, generated_sequence.token_ids.empty());
        state.generated_tokens = generated_sequence.token_ids;
        m_draft->sync_generated_tokens(state.generated_tokens);
        state.draft_generated = 0;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_requests() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    if (m_main_pipeline) {
        m_main_pipeline->finish_request();
    }
    m_request_states.clear();
}

ov::genai::RawPerfMetrics ContinuousBatchingPipeline::DFlashDecodingImpl::collect_draft_raw_metrics() {
    ov::genai::RawPerfMetrics raw_metrics;
    raw_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    if (!m_draft) {
        return raw_metrics;
    }

    m_draft->print_timing_profile();
    m_draft->print_component_profile();
    const auto& draft_metrics = m_draft->get_raw_perf_metrics();
    const size_t inferences_per_stage = m_selector_enabled ? 2 : 1;
    OPENVINO_ASSERT(draft_metrics.m_durations.size() ==
                        draft_metrics.m_batch_sizes.size() * inferences_per_stage,
                    "DFlash inference timings must contain one backbone inference and, when enabled, "
                    "one selector inference per proposal stage.");

    for (size_t stage = 0; stage < draft_metrics.m_batch_sizes.size(); ++stage) {
        const auto first_inference = draft_metrics.m_durations.begin() + stage * inferences_per_stage;
        const auto stage_duration =
            std::accumulate(first_inference,
                            first_inference + inferences_per_stage,
                            MicroSeconds(0.0f));
        raw_metrics.m_durations.push_back(stage_duration);
        raw_metrics.m_batch_sizes.push_back(draft_metrics.m_batch_sizes[stage]);
        raw_metrics.m_inference_durations[0] += stage_duration;
    }
    return raw_metrics;
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    if (position_ids.has_value()) {
        OPENVINO_ASSERT(m_model_input_type == ModelInputType::EMBEDDINGS,
                        "DFlash CB/PA only accepts explicit position_ids in VLM embedding mode.");
        OPENVINO_ASSERT(position_ids->size() == input_ids.size(),
                        "DFlash VLM position_ids count must match input embeddings.");
    }
    OPENVINO_ASSERT(!has_non_finished_requests(),
                    "Generate cannot be called while ContinuousBatchingPipeline is already running");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());
    OPENVINO_ASSERT(input_ids.size() == 1, "DFlash CB/PA POC supports batch size 1 only.");

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.main_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_perf_metrics.draft_model_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    auto start_time = std::chrono::steady_clock::now();

    auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);
    OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                        (input_ids.size() == 1 && sampling_params[0].is_greedy_decoding()),
                    "DFlash CB/PA streaming only supports batch size=1 with greedy decoding.");

    std::vector<GenerationHandle> main_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(input_ids[request_id].get_shape().at(0) == 1, "Use multiple tensors to pass a batch.");
        if (position_ids.has_value()) {
            const auto& [main_position_ids, rope_delta] = (*position_ids)[request_id];
            m_inputs_embedder->set_position_ids(main_position_ids);
            m_inputs_embedder->set_rope_delta(rope_delta.value_or(compute_rope_delta(main_position_ids)));
        }
        const bool has_valid_prompt_ids = prompt_ids.has_value() && request_id < prompt_ids->size();
        const bool has_valid_lm_extra_inputs = lm_extra_inputs_list.has_value() && request_id < lm_extra_inputs_list->size();
        main_generations.push_back(add_request(
            request_id,
            input_ids[request_id],
            sampling_params[request_id],
            has_valid_prompt_ids ? std::make_optional((*prompt_ids)[request_id]) : std::nullopt,
            has_valid_lm_extra_inputs ? std::make_optional((*lm_extra_inputs_list)[request_id]) : std::nullopt));
    }

    auto all_requests = m_main_pipeline->get_awaiting_requests();
    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();
    while (has_non_finished_requests()) {
        try {
            step();
        } catch (...) {
            drop_requests();
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }
    streamer_ptr->end();

    OPENVINO_ASSERT(m_main_pipeline->is_requests_empty(),
                    "Internal error: current request is supposed to be dropped within step() function as completed");

    m_perf_metrics.draft_model_metrics.raw_metrics = collect_draft_raw_metrics();
    uint64_t generate_duration_us = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time);

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto cfg = request->get_sampling_parameters();
        const auto& seqs = request->get_finished_sequences();
        size_t num_out = std::min(cfg.num_return_sequences, seqs.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_out);
        result.m_scores.resize(num_out);
        result.m_finish_reasons.resize(num_out, GenerationFinishReason::NONE);
        result.m_status = main_generations[request_id]->get_status();

        for (size_t i = 0; i < num_out; ++i) {
            const auto& seq = seqs[i];
            float score = cfg.is_beam_search() ? seq->get_beam_search_score(cfg) : seq->get_cumulative_log_prob();
            const auto& gen_ids = seq->get_generated_ids();
            if (cfg.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(gen_ids.begin(), gen_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
            result.m_finish_reasons[i] = seq->get_finish_reason();
            if (result.m_finish_reasons[i] == GenerationFinishReason::NONE && request->handle_stopped()) {
                result.m_finish_reasons[i] = request->get_generation_stream()->get_finish_reason();
            }
        }

        m_perf_metrics.raw_metrics.generate_durations.clear();
        m_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_duration_us);
        m_perf_metrics.num_input_tokens = request->get_prompt_len();
        m_perf_metrics.num_prefix_cache_hit_tokens = request->get_num_prefix_cache_hit_tokens();
        m_perf_metrics.evaluate_statistics(start_time);

        result.perf_metrics = m_perf_metrics;
        result.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_perf_metrics);
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}

}  // namespace ov::genai
