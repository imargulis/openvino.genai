// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>

#include "sequence_group.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

size_t num_rows(const ov::Tensor& rows) {
    return rows && rows.get_size() > 0 ? rows.get_shape()[0] : 0;
}

ov::Tensor slice_rows(const ov::Tensor& rows, size_t begin, size_t end) {
    auto [roi_begin, roi_end] = utils::make_roi(rows.get_shape(), 0, begin, end);
    return ov::Tensor(rows, roi_begin, roi_end);
}

ov::Tensor copy_rows(const ov::Tensor& rows) {
    ov::Tensor owned(rows.get_element_type(), rows.get_shape());
    rows.copy_to(owned);
    return owned;
}

ov::Tensor concat_rows(const ov::Tensor& head, const ov::Tensor& tail) {
    auto shape = head.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && tail.get_shape().size() == 3 && shape[2] == tail.get_shape()[2] &&
                        head.get_element_type() == tail.get_element_type(),
                    "DFlash target hidden-state rows must share the [rows, 1, hidden] layout.");
    const size_t head_rows = shape[0];
    shape[0] += tail.get_shape()[0];
    ov::Tensor merged(head.get_element_type(), shape);
    head.copy_to(slice_rows(merged, 0, head_rows));
    tail.copy_to(slice_rows(merged, head_rows, shape[0]));
    return merged;
}

}  // namespace

ContinuousBatchingPipeline::ContinuousBatchingForDFlashDecodingImpl::ContinuousBatchingForDFlashDecodingImpl(
    const std::shared_ptr<ov::Model>& model,
    const Tokenizer& tokenizer,
    const GenerationConfig& generation_config,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_validation_mode_enabled,
    int64_t mask_token_id,
    size_t candidate_position_offset)
    : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                   tokenizer,
                                                   generation_config,
                                                   scheduler_config,
                                                   device,
                                                   plugin_config,
                                                   is_validation_mode_enabled) {
    dflash_mode_enabled = true;
    if (!is_validation_mode_enabled) {
        enable_dflash_draft(mask_token_id, candidate_position_offset);
    }
}

ContinuousBatchingPipeline::ContinuousBatchingForDFlashDecodingImpl::ContinuousBatchingForDFlashDecodingImpl(
    const std::shared_ptr<ov::Model>& model,
    std::shared_ptr<InputsEmbedder> inputs_embedder,
    const Tokenizer& tokenizer,
    const GenerationConfig& generation_config,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_validation_mode_enabled,
    int64_t mask_token_id,
    size_t candidate_position_offset)
    : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                   std::move(inputs_embedder),
                                                   tokenizer,
                                                   generation_config,
                                                   scheduler_config,
                                                   device,
                                                   plugin_config,
                                                   is_validation_mode_enabled) {
    dflash_mode_enabled = true;
    if (!is_validation_mode_enabled) {
        // The draft is a text model: sequential positions instead of the target's M-RoPE positions.
        m_model_runner->enable_mtp_draft_positions(true);
        enable_dflash_draft(mask_token_id, candidate_position_offset);
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::enable_dflash_draft(
    int64_t mask_token_id,
    size_t candidate_position_offset) {
    OPENVINO_ASSERT(mask_token_id >= 0, "DFlash draft requires a valid mask token id.");
    OPENVINO_ASSERT(m_scheduler->get_config().dynamic_split_fuse,
                    "DFlash draft pipeline requires scheduler_config.dynamic_split_fuse=true.");
    // candidates are read at fixed logits rows, which needs the gathered logits
    const auto inputs = m_model_runner->get_infer_request().get_compiled_model().inputs();
    OPENVINO_ASSERT(std::any_of(inputs.begin(), inputs.end(),
                                [](const ov::Output<const ov::Node>& input) {
                                    return input.get_names().count("sampled_tokens_indices") > 0;
                                }),
                    "DFlash draft model has no 'sampled_tokens_indices' input, so its candidate logits cannot be gathered.");
    m_dflash_mask_token_id = mask_token_id;
    m_dflash_candidate_position_offset = candidate_position_offset;
    m_model_runner->enable_dflash_draft(candidate_position_offset);
}

size_t ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_dflash_draft_step(
    SequenceGroup::Ptr request,
    dflash_cb::DraftState& state,
    const ov::Tensor& logits,
    size_t first_logits_row) {
    const uint64_t request_id = request->get_request_id();
    // update_dflash_draft_request left the processed length at the committed rows; a forward advances it
    OPENVINO_ASSERT(request->get_num_processed_tokens() >= state.committed_rows,
                    "DFlash draft request ", request_id, " was preempted; the draft cannot recompute its rows. "
                    "Increase the cache size or use dynamic cache allocation.");
    const size_t forwarded_rows = request->get_num_processed_tokens() - state.committed_rows;
    const size_t pending_rows = num_rows(state.pending_rows);
    const size_t context_rows = std::min(forwarded_rows, pending_rows);
    const bool forwarded_block = forwarded_rows > pending_rows;
    OPENVINO_ASSERT(!forwarded_block || forwarded_rows == pending_rows + state.block_rows,
                    "DFlash draft window of request ", request_id, " (", pending_rows, " context rows + ",
                    state.block_rows, " block rows) was split into ", forwarded_rows,
                    " scheduled rows; increase the cache size or max_num_batched_tokens.");

    if (context_rows > 0) {
        state.committed_rows += context_rows;
        state.pending_rows = context_rows < pending_rows
                                 ? copy_rows(slice_rows(state.pending_rows, context_rows, pending_rows))
                                 : ov::Tensor{};
        state.owns_pending_rows = static_cast<bool>(state.pending_rows);
    }
    if (state.pending_rows && !state.owns_pending_rows) {
        // The rows view the target's output tensor, which the next target inference overwrites.
        state.pending_rows = copy_rows(state.pending_rows);
        state.owns_pending_rows = true;
    }
    m_model_runner->set_initial_hidden_state(request_id, state.pending_rows);

    auto running_sequences = request->get_running_sequences();
    if (running_sequences.empty() || state.block_rows == 0) {
        return 0;
    }
    Sequence::Ptr sequence = running_sequences.front();
    sequence->remove_last_tokens(state.block_rows - 1);
    // the model runner gathers every block row after the candidate position offset
    const size_t logits_rows = forwarded_block ? state.block_rows - m_dflash_candidate_position_offset : 0;
    if (forwarded_block) {
        const size_t vocab_size = logits.get_shape().back();
        const float* logits_data = logits.data<const float>();
        for (size_t candidate = 0; candidate < state.candidate_rows; ++candidate) {
            const float* row = logits_data + (first_logits_row + candidate) * vocab_size;
            sequence->append_token(std::max_element(row, row + vocab_size) - row, 0.0f);
        }
    }
    state.block_rows = 0;
    state.candidate_rows = 0;
    return logits_rows;
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::dflash_multistep() {
    const bool has_window = std::any_of(m_requests.begin(), m_requests.end(), [](const SequenceGroup::Ptr& request) {
        return !request->is_waiting() && !request->has_finished();
    });
    ov::Tensor logits;
    float step_duration = 0.0f;
    if (has_window) {
        const auto step_start = std::chrono::steady_clock::now();
        step();
        step_duration = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
        logits = m_model_runner->get_infer_request().get_tensor("logits");
    }

    // Only block windows produce logits rows, and they are scheduled in request order, as the sampler
    // assumes for every pipeline.
    bool forwarded = false;
    size_t logits_row = 0;
    size_t num_candidates = 0;
    for (auto& request : m_requests) {
        const uint64_t request_id = request->get_request_id();
        auto state_it = m_dflash_draft_states.find(request_id);
        const auto running_sequences = request->get_running_sequences();
        if (state_it != m_dflash_draft_states.end() && !running_sequences.empty() &&
            state_it->second.sequence_id != running_sequences.front()->get_id()) {
            // left by an earlier request with the same id; this request gets its state at its first update
            m_dflash_draft_states.erase(state_it);
            state_it = m_dflash_draft_states.end();
        }
        if (state_it != m_dflash_draft_states.end()) {
            dflash_cb::DraftState& state = state_it->second;
            forwarded |= request->get_num_processed_tokens() > state.committed_rows;
            const size_t candidate_rows = state.candidate_rows;
            const size_t logits_rows = finish_dflash_draft_step(request, state, logits, logits_row);
            logits_row += logits_rows;
            num_candidates += logits_rows > 0 ? candidate_rows : 0;
        }
        request->pause_generation(true);
        m_scheduler->clear_expected_num_scheduled_tokens(request_id);
    }
    for (auto state_it = m_dflash_draft_states.begin(); state_it != m_dflash_draft_states.end();) {
        const bool is_live = std::any_of(m_requests.begin(), m_requests.end(), [&](const SequenceGroup::Ptr& request) {
            return request->get_request_id() == state_it->first;
        });
        if (!is_live) {
            m_model_runner->set_initial_hidden_state(state_it->first, ov::Tensor{});
        }
        state_it = is_live ? std::next(state_it) : m_dflash_draft_states.erase(state_it);
    }
    // Candidates need embeddings before the next update can remove the rejected ones.
    sync_generated_embeddings();

    if (forwarded) {
        raw_perf_metrics.m_durations.emplace_back(step_duration);
        raw_perf_metrics.m_inference_durations[0] += MicroSeconds(get_metrics().inference_duration);
        raw_perf_metrics.m_batch_sizes.emplace_back(num_candidates);
    }
}

UpdateRequestResult ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::update_dflash_draft_request(
    SequenceGroup::Ptr request,
    const GeneratedSequences& candidates) {
    UpdateRequestResult result{0, 0};
    auto running_sequences = request->get_running_sequences();
    if (candidates.empty() || running_sequences.empty()) {
        return result;
    }
    OPENVINO_ASSERT(running_sequences.size() == 1 && candidates.size() == 1,
                    "DFlash draft supports one sequence per request.");
    const uint64_t request_id = request->get_request_id();
    Sequence::Ptr sequence = running_sequences.front();
    const GeneratedSequence& target = candidates.begin()->second;
    dflash_cb::DraftState& state = m_dflash_draft_states[request_id];
    if (state.sequence_id != sequence->get_id()) {
        state = dflash_cb::DraftState{};
        state.sequence_id = sequence->get_id();
    }

    // Keep the tokens both models agree on, then take the target's tokens: the accepted candidates stay,
    // the rejected ones are replaced by the target's correction.
    const auto& draft_ids = sequence->get_generated_ids();
    const size_t draft_len = draft_ids.size();
    size_t common_len = 0;
    while (common_len < std::min(draft_len, target.token_ids.size()) &&
           draft_ids[common_len] == target.token_ids[common_len]) {
        ++common_len;
    }
    result.removed_tokens_cnt = draft_len - common_len;
    sequence->remove_last_tokens(result.removed_tokens_cnt);
    for (size_t i = common_len; i < target.token_ids.size(); ++i) {
        sequence->append_token(target.token_ids[i], target.log_probs[i]);
    }
    result.inserted_tokens_cnt = target.token_ids.size() - common_len;

    const auto new_rows = dflash_cb::select_new_target_rows(request_id,
                                                            state.committed_rows + num_rows(state.pending_rows),
                                                            target.num_processed_tokens,
                                                            num_rows(target.hidden_states),
                                                            result.removed_tokens_cnt);
    if (new_rows.count > 0) {
        ov::Tensor rows = slice_rows(target.hidden_states, new_rows.begin, new_rows.begin + new_rows.count);
        if (state.pending_rows) {
            state.pending_rows = concat_rows(state.pending_rows, rows);
            state.owns_pending_rows = true;
        } else {
            state.pending_rows = rows;
            state.owns_pending_rows = false;
        }
    }

    const size_t known_rows = state.committed_rows + num_rows(state.pending_rows);
    const auto block = dflash_cb::plan_draft_block(request->get_sampling_parameters().num_assistant_tokens.value_or(0),
                                                   sequence->get_generated_len(),
                                                   request->get_max_new_tokens(),
                                                   request->get_prompt_len(),
                                                   known_rows,
                                                   m_dflash_candidate_position_offset);
    for (size_t i = 1; i < block.block_rows; ++i) {
        sequence->append_token(m_dflash_mask_token_id, 0.0f);
    }
    state.block_rows = block.block_rows;
    state.candidate_rows = block.candidate_rows;

    // Mark the window [committed rows, window end) for recompute, so the scheduler offers exactly the window.
    // The previous block's KV past the committed rows is overwritten.
    const size_t window_rows = num_rows(state.pending_rows) + block.block_rows;
    const size_t window_end = state.committed_rows + window_rows;
    request->update_processed_tokens_num(window_end);
    request->preempt_tokens(window_rows);
    m_model_runner->set_initial_hidden_state(request_id, state.pending_rows);
    if (m_scheduler && window_rows > 0 && window_end < request->get_prompt_len()) {
        m_scheduler->set_expected_num_scheduled_tokens(request_id, window_rows);
    }
    request->pause_generation(window_rows == 0);
    return result;
}

}  // namespace ov::genai
