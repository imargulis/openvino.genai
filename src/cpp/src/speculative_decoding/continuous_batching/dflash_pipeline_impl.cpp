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
    size_t candidate_position_offset,
    std::shared_ptr<dflash_cb::CandidateSelector> candidate_selector)
    : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                   tokenizer,
                                                   generation_config,
                                                   scheduler_config,
                                                   device,
                                                   plugin_config,
                                                   is_validation_mode_enabled) {
    dflash_mode_enabled = true;
    if (!is_validation_mode_enabled) {
        enable_dflash_draft(mask_token_id, candidate_position_offset, std::move(candidate_selector));
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
    size_t candidate_position_offset,
    std::shared_ptr<dflash_cb::CandidateSelector> candidate_selector)
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
        enable_dflash_draft(mask_token_id, candidate_position_offset, std::move(candidate_selector));
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::enable_dflash_draft(
    int64_t mask_token_id,
    size_t candidate_position_offset,
    std::shared_ptr<dflash_cb::CandidateSelector> candidate_selector) {
    OPENVINO_ASSERT(mask_token_id >= 0, "DFlash draft requires a valid mask token id.");
    OPENVINO_ASSERT(m_scheduler->get_config().dynamic_split_fuse,
                    "DFlash draft pipeline requires scheduler_config.dynamic_split_fuse=true.");
    // candidates are read at fixed logits rows, which needs the gathered logits
    const auto compiled_model = m_model_runner->get_infer_request().get_compiled_model();
    const auto has_port = [](const auto& ports, const std::string& name) {
        return std::any_of(ports.begin(), ports.end(), [&](const ov::Output<const ov::Node>& port) {
            return port.get_names().count(name) > 0;
        });
    };
    OPENVINO_ASSERT(has_port(compiled_model.inputs(), "sampled_tokens_indices"),
                    "DFlash draft model has no 'sampled_tokens_indices' input, so its candidate logits cannot be gathered.");
    m_dflash_candidate_selector =
        candidate_selector ? std::move(candidate_selector) : std::make_shared<dflash_cb::CandidateSelector>();
    OPENVINO_ASSERT(!m_dflash_candidate_selector->reads_hidden_states() ||
                        has_port(compiled_model.outputs(), "last_hidden_state"),
                    "A DFlash draft with a gated Markov head must return 'last_hidden_state' with its logits.");
    m_dflash_mask_token_id = mask_token_id;
    m_dflash_candidate_position_offset = candidate_position_offset;
    m_model_runner->enable_dflash_draft(candidate_position_offset);
    // The draft builds its KV from target hidden states it does not keep, so it cannot recompute a preempted
    // request. A window that does not fit the cache stays unscheduled, and the request decodes without candidates.
    m_scheduler->disable_preemption();
}

size_t ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_dflash_draft_step(
    SequenceGroup::Ptr request,
    dflash_cb::DraftState& state,
    size_t first_output_row,
    std::optional<dflash_cb::DraftBlockRows>& block) {
    const uint64_t request_id = request->get_request_id();
    // update_dflash_draft_request left the processed length at the committed rows; a forward advances it, and the
    // draft scheduler never preempts
    OPENVINO_ASSERT(request->get_num_processed_tokens() >= state.committed_rows,
                    "DFlash draft request ", request_id, " lost KV rows it cannot recompute.");
    const size_t forwarded_rows = request->get_num_processed_tokens() - state.committed_rows;
    const size_t pending_rows = num_rows(state.pending_rows);
    const size_t context_rows = std::min(forwarded_rows, pending_rows);
    const size_t forwarded_block_rows = forwarded_rows - context_rows;
    // the model runner gathers the forwarded block rows after the first candidate_position_offset ones
    const size_t gathered_rows = forwarded_block_rows > m_dflash_candidate_position_offset
                                     ? forwarded_block_rows - m_dflash_candidate_position_offset
                                     : 0;

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
        return gathered_rows;
    }
    Sequence::Ptr sequence = running_sequences.front();
    sequence->remove_last_tokens(state.block_rows - 1);
    // A block that did not fit the cache or the batch, whole or in part, gives no candidates: the target decodes
    // the request alone this step, and the next window starts again at the committed rows.
    if (forwarded_block_rows == state.block_rows) {
        block = dflash_cb::DraftBlockRows{sequence->get_generated_ids().back(),
                                          first_output_row,
                                          gathered_rows,
                                          state.candidate_rows};
    }
    state.block_rows = 0;
    state.candidate_rows = 0;
    return gathered_rows;
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::dflash_multistep() {
    const bool has_window = std::any_of(m_requests.begin(), m_requests.end(), [](const SequenceGroup::Ptr& request) {
        return !request->is_waiting() && !request->has_finished();
    });
    const auto step_start = std::chrono::steady_clock::now();
    if (has_window) {
        step();
    }

    // Only block windows produce output rows, and they are scheduled in request order, as the sampler
    // assumes for every pipeline.
    bool forwarded = false;
    size_t output_row = 0;
    std::vector<dflash_cb::DraftBlockRows> blocks;
    std::vector<Sequence::Ptr> block_sequences;
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
            std::optional<dflash_cb::DraftBlockRows> block;
            output_row += finish_dflash_draft_step(request, state, output_row, block);
            if (block) {
                blocks.push_back(*block);
                block_sequences.push_back(request->get_running_sequences().front());
            }
        }
        request->pause_generation(true);
        m_scheduler->clear_expected_num_scheduled_tokens(request_id);
    }

    size_t num_candidates = 0;
    float selection_inference_duration = 0.0f;
    if (!blocks.empty()) {
        auto& infer_request = m_model_runner->get_infer_request();
        const ov::Tensor logits = infer_request.get_tensor("logits");
        OPENVINO_ASSERT(logits.get_size() == output_row * logits.get_shape().back(),
                        "DFlash draft returned ", logits.get_size() / logits.get_shape().back(),
                        " logits rows, but its blocks gathered ", output_row, ".");
        const ov::Tensor hidden_states = m_dflash_candidate_selector->reads_hidden_states()
                                             ? infer_request.get_tensor("last_hidden_state")
                                             : ov::Tensor{};
        const auto candidates = m_dflash_candidate_selector->select(blocks, logits, hidden_states);
        selection_inference_duration = m_dflash_candidate_selector->get_inference_duration();
        for (size_t idx = 0; idx < blocks.size(); ++idx) {
            for (const int64_t token : candidates[idx]) {
                block_sequences[idx]->append_token(token, 0.0f);
            }
            num_candidates += candidates[idx].size();
        }
    }
    const float step_duration = PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
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
        raw_perf_metrics.m_inference_durations[0] +=
            MicroSeconds(get_metrics().inference_duration + selection_inference_duration);
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
        // a restored prefix is already committed: the draft request waits for its first window until now
        state.committed_rows = request->get_num_processed_tokens();
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
    // Every block follows at least one context row, which keeps the blocks of different requests apart in the
    // packed block rows that the DFlash-2 convolutions run over. A target still recomputing preempted tokens
    // catches up before the request drafts again.
    const auto block =
        num_rows(state.pending_rows) > 0 && target.num_processed_tokens == known_rows
            ? dflash_cb::plan_draft_block(request->get_sampling_parameters().num_assistant_tokens.value_or(0),
                                          sequence->get_generated_len(),
                                          request->get_max_new_tokens(),
                                          request->get_prompt_len(),
                                          known_rows,
                                          m_dflash_candidate_position_offset)
            : dflash_cb::DraftBlockPlan{};
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
