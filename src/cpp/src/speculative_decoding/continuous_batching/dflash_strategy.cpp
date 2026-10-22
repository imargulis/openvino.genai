// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <openvino/pass/sdpa_to_paged_attention.hpp>

#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace {

struct DFlashDraftOutputs {
    // [rows, 1, vocab]: the candidate rows of all sequences, one sequence after another
    ov::Tensor logits;
    // [rows, 1, hidden], only for a Markov head that reads the draft hidden states
    ov::Tensor hidden_states;
};

bool model_has_output(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto outputs = model->outputs();
    return std::find_if(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& port) {
        return port.get_names().count(name) != 0;
    }) != outputs.end();
}

void validate_target_has_no_unmanaged_state(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        const std::string type_name = op->get_type_name();
        OPENVINO_ASSERT(type_name != "ReadValue" && type_name != "Assign",
                        "DFlash CB/PA target model contains unmanaged state op(s) after paging conversion. ",
                        "First unmanaged op: ",
                        op->get_friendly_name(),
                        ". Convert KV, conv, and GatedDeltaNet state to managed paging before enabling DFlash.");
    }
}

size_t hidden_delta_rows(const ov::Tensor& hidden_delta) {
    OPENVINO_ASSERT(hidden_delta && hidden_delta.get_size() > 0, "DFlash hidden delta must be provided.");
    const auto shape = hidden_delta.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                    "DFlash draft hidden_states input must have shape [seq_len, 1, hidden].");
    return shape[0];
}

// The draft runner sets its PagedAttention inputs by hand; reject any input it would leave unset.
void validate_paged_draft_inputs(const ov::CompiledModel& model, bool use_external_embeddings) {
    const std::set<std::string> runner_inputs = {"hidden_states",
                                                 "position_ids",
                                                 "token_type_ids",
                                                 "past_lens",
                                                 "subsequence_begins",
                                                 "block_indices",
                                                 "block_indices_begins",
                                                 "max_context_len",
                                                 "sampled_tokens_indices",
                                                 use_external_embeddings ? "inputs_embeds" : "input_ids"};
    std::set<std::string> found;
    for (const auto& input : model.inputs()) {
        const auto& name = input.get_any_name();
        if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0) {
            continue;
        }
        OPENVINO_ASSERT(runner_inputs.count(name) != 0,
                        "DFlash paged draft has an input the runner does not set: ",
                        name);
        found.insert(name);
    }
    for (const auto& name : runner_inputs) {
        OPENVINO_ASSERT(found.count(name) != 0, "DFlash paged draft is missing the input: ", name);
    }
}

// A gated Markov head (DFlash-2) scores the candidates from the draft hidden states as well as the logits.
bool markov_head_reads_hidden_states(const ov::genai::utils::dflash::DFlashRTInfo& rt_info) {
    return rt_info.markov_head_computation &&
           utils::has_input(rt_info.markov_head_computation, "draft_hidden_states");
}

// A dense [1, count, N] view of rows of a contiguous [rows, 1, N] draft output.
ov::Tensor output_rows(ov::Tensor output, size_t first_row, size_t count) {
    OPENVINO_ASSERT(output.is_continuous(), "DFlash draft outputs must be contiguous.");
    const auto& shape = output.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1 && first_row + count <= shape[0],
                    "DFlash draft output rows exceed the [rows, 1, N] output.");
    const size_t row_bytes = shape[2] * output.get_element_type().size();
    return ov::Tensor(output.get_element_type(),
                      {1, count, shape[2]},
                      static_cast<uint8_t*>(output.data()) + first_row * row_bytes);
}

}  // namespace

// Drafts the sequences of all DFlash requests on PagedAttention, one inference per step: their rows
// [context delta ; block] go one after another along the token axis, and the draft returns the candidate rows
// of every sequence in the same order. A draft with a Markov head, DSpark or DFlash-2, then scores the candidate
// rows of all sequences with one candidate scorer inference, or walks them with the dense Markov step.
class ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashCBDraftRunner {
public:
    DFlashCBDraftRunner(const ov::genai::ModelDesc& model_desc,
                        const Tokenizer& tokenizer,
                        const ov::genai::utils::dflash::DFlashRTInfo& rt_info,
                        EmbeddingsModel::Ptr embedding_model = nullptr)
        : m_tokenizer(tokenizer),
          m_embedding_model(std::move(embedding_model)),
          m_gated_markov_head(markov_head_reads_hidden_states(rt_info)),
          m_compiled_model(compile_draft_model(model_desc,
                                               static_cast<bool>(m_embedding_model),
                                               m_gated_markov_head,
                                               rt_info)),
          m_request(m_compiled_model.create_infer_request()),
          m_kv_cache(std::make_unique<KVCacheManager>(m_request)),
          m_sampler(tokenizer),
          m_mask_token_id(rt_info.mask_token_id),
          m_candidate_position_offset(rt_info.candidate_position_offset),
          m_output_multiplier(rt_info.output_multiplier),
          m_final_logit_softcapping(rt_info.final_logit_softcapping) {
        m_sampler.set_collect_draft_proposals(true);
        validate_paged_draft_inputs(m_compiled_model, static_cast<bool>(m_embedding_model));
        if (rt_info.markov_head_computation) {
            auto markov_head_request =
                utils::singleton_core()
                    .compile_model(rt_info.markov_head_computation, model_desc.device, model_desc.properties)
                    .create_infer_request();
            if (utils::dflash::is_markov_step(rt_info.markov_head_computation)) {
                m_markov_step_request = std::move(markov_head_request);
            } else {
                m_candidate_scorer_request = std::move(markov_head_request);
            }
        }
        reset_perf_metrics();
    }

    void add_sequence(uint64_t request_id, const ov::Tensor& input_ids, const GenerationConfig& config) {
        const auto shape = input_ids.get_shape();
        OPENVINO_ASSERT(shape.size() == 2 && shape[0] == BATCH_SIZE && shape[1] > 0,
                        "Expected DFlash input_ids shape [1, seq_len].");
        const int64_t* ids_data = input_ids.data<const int64_t>();
        add_sampler_sequence(request_id, TokenIds(ids_data, ids_data + shape[1]), config);
    }

    void add_sequence(uint64_t request_id, size_t prompt_length, const GenerationConfig& config) {
        add_sampler_sequence(request_id,
                             dflash_cb::build_placeholder_prompt_ids(prompt_length, m_tokenizer.get_pad_token_id()),
                             config);
    }

    void remove_sequence(uint64_t request_id) {
        auto sequence_it = m_sequences.find(request_id);
        if (sequence_it == m_sequences.end()) {
            return;
        }
        const auto& blocks = sequence_it->second.blocks;
        m_free_blocks.insert(m_free_blocks.end(), blocks.begin(), blocks.end());
        m_sampler.clear_request_info(request_id);
        m_sequences.erase(sequence_it);
    }

    void sync_generated_tokens(uint64_t request_id, const std::vector<int64_t>& target_generated_tokens) {
        auto& sequence = get_sequence(request_id);
        auto seq = (*sequence.group)[0];
        if (seq->get_generated_len() > 0) {
            seq->remove_last_tokens(seq->get_generated_len());
        }
        for (auto token : target_generated_tokens) {
            seq->append_token(token, 0.0f);
        }
        sequence.group->update_processed_tokens_num(sequence.prompt_length + target_generated_tokens.size());
        seq->set_status(SequenceStatus::RUNNING);
    }

    size_t get_consumed_hidden_states(uint64_t request_id) const {
        return get_sequence(request_id).committed_context_length;
    }

    std::vector<DraftCandidates> propose(const std::vector<DraftInput>& inputs) {
        const auto outputs = infer(inputs);
        std::vector<DraftCandidates> proposals;
        if (m_candidate_scorer_request) {
            proposals = select_scored_candidates(inputs, outputs);
        } else if (m_markov_step_request) {
            proposals = walk_markov_steps(inputs, outputs.logits);
        } else {
            proposals.reserve(inputs.size());
            size_t first_row = 0;
            for (const auto& input : inputs) {
                proposals.push_back(sample_candidates(get_sequence(input.request_id),
                                                      output_rows(outputs.logits, first_row, input.validation_count)));
                first_row += input.validation_count;
            }
        }
        size_t num_candidates = 0;
        for (const auto& proposal : proposals) {
            num_candidates += proposal.token_ids.size();
        }
        finish_stage(num_candidates);
        return proposals;
    }

    // One stage per draft inference, with its Markov head inferences.
    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

    void reset_perf_metrics() {
        m_raw_perf_metrics = ov::genai::RawPerfMetrics();
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_stage_inference_us = 0;
    }

private:
    struct DraftSequence {
        SequenceGroup::Ptr group;
        size_t prompt_length = 0;
        size_t committed_context_length = 0;
        std::mt19937 proposal_rng;
        // draft KV cache blocks, in context order
        std::vector<int32_t> blocks;
    };

    void add_sampler_sequence(uint64_t request_id, TokenIds prompt_ids, const GenerationConfig& config) {
        OPENVINO_ASSERT(m_sequences.find(request_id) == m_sequences.end(),
                        "DFlash draft already has a sequence for request ", request_id, ".");
        DraftSequence sequence;
        sequence.prompt_length = prompt_ids.size();
        m_sampler.clear_request_info(request_id);
        sequence.group = std::make_shared<SequenceGroup>(request_id, prompt_ids, config);
        sequence.group->update_processed_tokens_num(sequence.prompt_length);
        // config is the draft config, whose seed is already distinct from the target sampler's.
        sequence.proposal_rng.seed(static_cast<std::mt19937::result_type>(config.rng_seed));
        m_sequences.emplace(request_id, std::move(sequence));
    }

    DraftSequence& get_sequence(uint64_t request_id) {
        auto sequence_it = m_sequences.find(request_id);
        OPENVINO_ASSERT(sequence_it != m_sequences.end(), "DFlash draft has no sequence for request ", request_id, ".");
        return sequence_it->second;
    }

    const DraftSequence& get_sequence(uint64_t request_id) const {
        auto sequence_it = m_sequences.find(request_id);
        OPENVINO_ASSERT(sequence_it != m_sequences.end(), "DFlash draft has no sequence for request ", request_id, ".");
        return sequence_it->second;
    }

    // The committed context of every sequence precedes its past_lens, so the block written by the
    // previous call is overwritten, which rolls it back. The draft returns the rows that sampled_tokens_indices
    // selects: the validation_count candidate rows of every sequence.
    DFlashDraftOutputs infer(const std::vector<DraftInput>& inputs) {
        OPENVINO_ASSERT(!inputs.empty(), "DFlash paged draft call needs at least one sequence.");
        const size_t num_sequences = inputs.size();
        std::vector<size_t> context_rows(num_sequences);
        std::vector<size_t> num_rows(num_sequences);
        size_t total_rows = 0;
        size_t total_candidates = 0;
        for (size_t idx = 0; idx < num_sequences; ++idx) {
            OPENVINO_ASSERT(inputs[idx].validation_count > 0 &&
                                inputs[idx].validation_count <= inputs[idx].candidate_count,
                            "DFlash validation_count must be between 1 and the draft block's candidate_count.");
            // every sequence brings context rows, so the block rows of a sequence never open a subsequence
            context_rows[idx] = hidden_delta_rows(inputs[idx].hidden_delta);
            num_rows[idx] = context_rows[idx] + inputs[idx].candidate_count + m_candidate_position_offset;
            total_rows += num_rows[idx];
            total_candidates += inputs[idx].validation_count;
        }
        reserve_blocks(inputs, num_rows);

        const auto& first_delta = inputs.front().hidden_delta;
        const auto hidden_type = first_delta.get_element_type();
        const size_t hidden_size = first_delta.get_shape()[2];
        ov::Tensor input_ids(ov::element::i64, {total_rows});
        ov::Tensor hidden_states(hidden_type, {total_rows, 1, hidden_size});
        ov::Tensor position_ids(ov::element::i64, {total_rows});
        ov::Tensor token_type_ids(ov::element::i64, {total_rows, 1});
        ov::Tensor sampled_tokens_indices(ov::element::i64, {total_candidates});
        ov::Tensor past_lens(ov::element::i32, {num_sequences});
        ov::Tensor subsequence_begins(ov::element::i32, {num_sequences + 1});
        ov::Tensor block_indices_begins(ov::element::i32, {num_sequences + 1});
        std::vector<int32_t> block_indices;
        size_t max_context_len = 0;

        auto* ids = input_ids.data<int64_t>();
        auto* hidden = static_cast<uint8_t*>(hidden_states.data());
        auto* positions = position_ids.data<int64_t>();
        auto* token_types = token_type_ids.data<int64_t>();
        auto* candidate_rows = sampled_tokens_indices.data<int64_t>();
        const size_t row_bytes = hidden_size * hidden_type.size();
        const size_t block_size = m_kv_cache->get_block_size();
        subsequence_begins.data<int32_t>()[0] = 0;
        block_indices_begins.data<int32_t>()[0] = 0;
        size_t row = 0;
        for (size_t idx = 0; idx < num_sequences; ++idx) {
            const auto& input = inputs[idx];
            const auto& sequence = get_sequence(input.request_id);
            const size_t context = context_rows[idx];
            const size_t rows = num_rows[idx];
            OPENVINO_ASSERT(input.hidden_delta.get_element_type() == hidden_type &&
                                input.hidden_delta.get_shape()[2] == hidden_size,
                            "DFlash hidden deltas of one paged draft call must share element type and hidden size.");

            const auto block_ids = dflash_cb::build_draft_input_ids(input.seed_token,
                                                                    m_mask_token_id,
                                                                    input.candidate_count,
                                                                    m_candidate_position_offset);
            // context rows read the target hidden states; their embedding is never used
            std::fill_n(ids + row, context, m_mask_token_id);
            std::copy_n(block_ids.data<const int64_t>(), block_ids.get_size(), ids + row + context);
            ov::Tensor context_states(hidden_states,
                                      ov::Coordinate{row, 0, 0},
                                      ov::Coordinate{row + context, 1, hidden_size});
            input.hidden_delta.copy_to(context_states);
            // block rows read their own embeddings; their hidden states are never used
            std::memset(hidden + (row + context) * row_bytes, 0, (rows - context) * row_bytes);
            std::iota(positions + row, positions + row + rows, static_cast<int64_t>(sequence.committed_context_length));
            std::fill_n(token_types + row, context, 0);
            std::fill_n(token_types + row + context, rows - context, 1);
            std::iota(candidate_rows,
                      candidate_rows + input.validation_count,
                      static_cast<int64_t>(row + context + m_candidate_position_offset));
            candidate_rows += input.validation_count;

            const size_t context_length = sequence.committed_context_length + rows;
            const size_t num_blocks = (context_length + block_size - 1) / block_size;
            past_lens.data<int32_t>()[idx] = static_cast<int32_t>(sequence.committed_context_length);
            block_indices.insert(block_indices.end(), sequence.blocks.begin(), sequence.blocks.begin() + num_blocks);
            row += rows;
            subsequence_begins.data<int32_t>()[idx + 1] = static_cast<int32_t>(row);
            block_indices_begins.data<int32_t>()[idx + 1] = static_cast<int32_t>(block_indices.size());
            max_context_len = std::max(max_context_len, context_length);
        }

        ov::Tensor block_indices_tensor(ov::element::i32, {block_indices.size()});
        std::copy(block_indices.begin(), block_indices.end(), block_indices_tensor.data<int32_t>());
        ov::Tensor max_context_len_tensor(ov::element::i32, {});
        max_context_len_tensor.data<int32_t>()[0] = static_cast<int32_t>(max_context_len);
        m_request.set_tensor("hidden_states", hidden_states);
        m_request.set_tensor("position_ids", position_ids);
        m_request.set_tensor("token_type_ids", token_type_ids);
        m_request.set_tensor("sampled_tokens_indices", sampled_tokens_indices);
        m_request.set_tensor("past_lens", past_lens);
        m_request.set_tensor("subsequence_begins", subsequence_begins);
        m_request.set_tensor("block_indices", block_indices_tensor);
        m_request.set_tensor("block_indices_begins", block_indices_begins);
        m_request.set_tensor("max_context_len", max_context_len_tensor);
        if (m_embedding_model) {
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(
                m_embedding_model->get_request_queue().get());
            ov::Tensor row_ids(ov::element::i64, {1, total_rows}, input_ids.data());
            ov::Tensor input_embeds = m_embedding_model->infer(embeddings_request_guard.get(), row_ids);
            const size_t embedding_size = input_embeds.get_shape().back();
            m_request.set_tensor("inputs_embeds",
                                 ov::Tensor(input_embeds.get_element_type(),
                                            {total_rows, embedding_size},
                                            input_embeds.data()));
            execute_inference(m_request);
        } else {
            m_request.set_tensor("input_ids", input_ids);
            execute_inference(m_request);
        }
        for (size_t idx = 0; idx < num_sequences; ++idx) {
            get_sequence(inputs[idx].request_id).committed_context_length += context_rows[idx];
        }

        DFlashDraftOutputs outputs;
        outputs.logits = m_request.get_tensor("logits");
        const auto logits_shape = outputs.logits.get_shape();
        OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == total_candidates && logits_shape[1] == 1,
                        "DFlash paged draft must return the ", total_candidates,
                        " candidate rows of all sequences as [rows, 1, vocab], got ", logits_shape, ".");
        if (m_gated_markov_head) {
            outputs.hidden_states = m_request.get_tensor("last_hidden_state");
            const auto hidden_shape = outputs.hidden_states.get_shape();
            OPENVINO_ASSERT(hidden_shape.size() == 3 && hidden_shape[0] == total_candidates && hidden_shape[1] == 1,
                            "DFlash paged draft must return the hidden states of its ", total_candidates,
                            " candidate rows as [rows, 1, hidden], got ", hidden_shape, ".");
        }
        return outputs;
    }

    void reserve_blocks(const std::vector<DraftInput>& inputs, const std::vector<size_t>& num_rows) {
        const size_t block_size = m_kv_cache->get_block_size();
        auto required_blocks = [&](size_t idx) {
            const size_t context_length = get_sequence(inputs[idx].request_id).committed_context_length + num_rows[idx];
            return (context_length + block_size - 1) / block_size;
        };
        size_t num_missing = 0;
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            const size_t num_owned = get_sequence(inputs[idx].request_id).blocks.size();
            num_missing += std::max(required_blocks(idx), num_owned) - num_owned;
        }
        if (num_missing > m_free_blocks.size()) {
            const size_t num_allocated = m_kv_cache->get_num_allocated_blocks();
            const size_t num_blocks = std::max(num_allocated + num_missing - m_free_blocks.size(), 2 * num_allocated);
            m_kv_cache->allocate_cache_if_needed(num_blocks);
            for (size_t block = num_blocks; block > num_allocated; --block) {
                m_free_blocks.push_back(static_cast<int32_t>(block - 1));
            }
        }
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            auto& blocks = get_sequence(inputs[idx].request_id).blocks;
            while (blocks.size() < required_blocks(idx)) {
                blocks.push_back(m_free_blocks.back());
                m_free_blocks.pop_back();
            }
        }
    }

    DraftCandidates sample_candidates(DraftSequence& sequence, const ov::Tensor& logits) {
        const auto shape = logits.get_shape();
        DraftCandidates candidates;
        candidates.token_ids.reserve(shape[1]);
        candidates.log_probs.reserve(shape[1]);
        for (size_t idx = 0; idx < shape[1]; ++idx) {
            ov::Tensor one_position(logits, ov::Coordinate{0, idx, 0}, ov::Coordinate{1, idx + 1, shape[2]});
            sample_one_candidate(sequence, apply_backbone_logit_transform(one_position), candidates);
        }
        return candidates;
    }

    // Drafts one position of every sequence per Markov step inference: the position's logits get the Markov bias
    // that follows the token drafted before it, the anchor at the first position, so drafting is autoregressive.
    std::vector<DraftCandidates> walk_markov_steps(const std::vector<DraftInput>& inputs, const ov::Tensor& logits) {
        OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32, "The Markov walk requires FP32 draft logits.");
        const size_t vocab_size = logits.get_shape()[2];
        std::vector<DraftCandidates> proposals(inputs.size());
        std::vector<size_t> first_rows(inputs.size());
        std::vector<int64_t> previous_tokens(inputs.size());
        std::vector<size_t> walking(inputs.size());
        size_t row = 0;
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            first_rows[idx] = row;
            row += inputs[idx].validation_count;
            previous_tokens[idx] = inputs[idx].seed_token;
        }
        std::iota(walking.begin(), walking.end(), size_t{0});
        // reused for every drafted position: each row is sampled before the next one is computed
        ov::Tensor corrected_logits(ov::element::f32, {BATCH_SIZE, 1, vocab_size});
        for (size_t position = 0; !walking.empty(); ++position) {
            ov::Tensor prev_token_ids(ov::element::i64, {walking.size()});
            for (size_t step = 0; step < walking.size(); ++step) {
                prev_token_ids.data<int64_t>()[step] = previous_tokens[walking[step]];
            }
            m_markov_step_request->set_tensor("prev_token_ids", prev_token_ids);
            execute_inference(*m_markov_step_request);
            const auto markov_bias = m_markov_step_request->get_tensor("markov_bias");
            OPENVINO_ASSERT(markov_bias.get_element_type() == ov::element::f32 &&
                                markov_bias.get_shape() == ov::Shape({walking.size(), vocab_size}),
                            "The Markov bias must be one FP32 row over the draft vocabulary per sequence.");
            std::vector<size_t> still_walking;
            for (size_t step = 0; step < walking.size(); ++step) {
                const size_t idx = walking[step];
                const auto backbone_logits =
                    apply_backbone_logit_transform(output_rows(logits, first_rows[idx] + position, 1));
                const auto* backbone = backbone_logits.data<const float>();
                std::transform(backbone,
                               backbone + vocab_size,
                               markov_bias.data<const float>() + step * vocab_size,
                               corrected_logits.data<float>(),
                               std::plus<float>());
                auto& candidates = proposals[idx];
                const size_t num_drafted = candidates.token_ids.size();
                sample_one_candidate(get_sequence(inputs[idx].request_id), corrected_logits, candidates);
                // the next position's Markov bias needs the token drafted here
                if (candidates.token_ids.size() == num_drafted) {
                    continue;
                }
                previous_tokens[idx] = candidates.token_ids.back();
                if (position + 1 < inputs[idx].validation_count) {
                    still_walking.push_back(idx);
                }
            }
            walking = std::move(still_walking);
        }
        return proposals;
    }

    // Scores the candidate rows of all sequences with one candidate scorer inference. The scorer takes the rows
    // packed back to back, sequence n from subsequence_begins[n] after its anchor anchor_token_ids[n], and returns
    // a Top-K lattice for every row: candidate_ids [rows, K], the backbone's K best tokens, and edge_scores
    // [rows, K, K], where edge_scores[s, i, j] scores candidate j at row s after candidate i at row s - 1.
    std::vector<DraftCandidates> select_scored_candidates(const std::vector<DraftInput>& inputs,
                                                          const DFlashDraftOutputs& outputs) {
        const auto& logits = outputs.logits;
        OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32 && logits.is_continuous(),
                        "The candidate scorer requires contiguous FP32 draft logits.");
        const size_t num_rows = logits.get_shape()[0];
        const size_t vocab_size = logits.get_shape()[2];
        const size_t num_sequences = inputs.size();
        ov::Tensor anchor_token_ids(ov::element::i64, {num_sequences});
        ov::Tensor subsequence_begins(ov::element::i64, {num_sequences + 1});
        auto* begins = subsequence_begins.data<int64_t>();
        begins[0] = 0;
        for (size_t idx = 0; idx < num_sequences; ++idx) {
            anchor_token_ids.data<int64_t>()[idx] = inputs[idx].seed_token;
            begins[idx + 1] = begins[idx] + static_cast<int64_t>(inputs[idx].validation_count);
        }
        dflash_cb::validate_subsequence_begins(subsequence_begins, num_sequences, num_rows);
        auto& scorer = *m_candidate_scorer_request;
        scorer.set_tensor("draft_logits", ov::Tensor(ov::element::f32, {num_rows, vocab_size}, logits.data()));
        if (m_gated_markov_head) {
            const auto& hidden_states = outputs.hidden_states;
            OPENVINO_ASSERT(hidden_states.get_element_type() == ov::element::f32 && hidden_states.is_continuous(),
                            "The gated Markov head requires contiguous FP32 draft hidden states.");
            scorer.set_tensor("draft_hidden_states",
                              ov::Tensor(ov::element::f32,
                                         {num_rows, hidden_states.get_shape()[2]},
                                         hidden_states.data()));
        }
        scorer.set_tensor("anchor_token_ids", anchor_token_ids);
        scorer.set_tensor("subsequence_begins", subsequence_begins);
        execute_inference(scorer);

        const auto edge_scores = scorer.get_tensor("edge_scores");
        const auto candidate_ids = scorer.get_tensor("candidate_ids");
        dflash_cb::validate_candidate_lattice(edge_scores, candidate_ids, num_rows, vocab_size);
        const size_t top_k = candidate_ids.get_shape()[1];
        std::vector<DraftCandidates> proposals;
        proposals.reserve(num_sequences);
        for (size_t idx = 0; idx < num_sequences; ++idx) {
            const size_t first_row = static_cast<size_t>(begins[idx]);
            const size_t count = inputs[idx].validation_count;
            const ov::Tensor sequence_scores(ov::element::f32,
                                             {count, top_k, top_k},
                                             edge_scores.data<float>() + first_row * top_k * top_k);
            const ov::Tensor sequence_ids(ov::element::i64,
                                          {count, top_k},
                                          candidate_ids.data<int64_t>() + first_row * top_k);
            proposals.push_back(select_from_lattice(get_sequence(inputs[idx].request_id),
                                                    output_rows(logits, first_row, count),
                                                    sequence_scores,
                                                    sequence_ids));
        }
        return proposals;
    }

    // Drafts one sequence from its lattice. A greedy draft takes the best candidate of every row when the logit
    // processors cannot change the argmax, and otherwise walks the lattice through the draft sampler so that they
    // apply. A sampled draft draws from the K candidates of every row; see sample_candidate_lattice().
    DraftCandidates select_from_lattice(DraftSequence& sequence,
                                        const ov::Tensor& backbone_logits,
                                        const ov::Tensor& edge_scores,
                                        const ov::Tensor& candidate_ids) {
        const auto& config = sequence.group->get_sampling_parameters();
        if (config.is_multinomial()) {
            return sample_candidate_lattice(sequence, edge_scores, candidate_ids);
        }
        DraftCandidates candidates;
        if (dflash_cb::logit_processors_keep_argmax(config)) {
            for (const int64_t token : dflash_cb::greedy_candidate_path(edge_scores, candidate_ids)) {
                // Greedy validation compares token ids and never reads the draft log-prob.
                candidates.token_ids.push_back(token);
                candidates.log_probs.push_back(0.0f);
            }
            return candidates;
        }
        const size_t vocab_size = backbone_logits.get_shape()[2];
        const ov::Shape row_shape{BATCH_SIZE, 1, vocab_size};
        if (!m_candidate_row || m_candidate_row.get_shape() != row_shape) {
            m_candidate_row = ov::Tensor(ov::element::f32, row_shape);
        }
        dflash_cb::walk_candidate_lattice(apply_backbone_logit_transform(backbone_logits),
                                          edge_scores,
                                          candidate_ids,
                                          candidate_ids.get_shape()[0],
                                          m_candidate_row,
                                          [&](const ov::Tensor& row) -> std::optional<int64_t> {
                                              const size_t num_drafted = candidates.token_ids.size();
                                              sample_one_candidate(sequence, row, candidates);
                                              if (candidates.token_ids.size() == num_drafted) {
                                                  return std::nullopt;
                                              }
                                              return candidates.token_ids.back();
                                          });
        return candidates;
    }

    // Draws every position from the softmax, at the request temperature, of the lattice row that follows the
    // candidate drafted before it, and proposes that distribution over the K candidates to the target.
    DraftCandidates sample_candidate_lattice(DraftSequence& sequence,
                                             const ov::Tensor& edge_scores,
                                             const ov::Tensor& candidate_ids) {
        const float temperature = sequence.group->get_sampling_parameters().temperature;
        OPENVINO_ASSERT(temperature > 0.0f, "Sampled DFlash drafts with a Markov head require temperature > 0.");
        const size_t num_positions = candidate_ids.get_shape()[0];
        const size_t top_k = candidate_ids.get_shape()[1];
        const auto* scores = edge_scores.data<const float>();
        const auto* ids = candidate_ids.data<const int64_t>();
        DraftCandidates candidates;
        candidates.token_ids.reserve(num_positions);
        candidates.log_probs.reserve(num_positions);
        candidates.proposals.reserve(num_positions);
        size_t previous_index = 0;
        for (size_t position = 0; position < num_positions; ++position) {
            const auto* row = scores + (position * top_k + previous_index) * top_k;
            const float max_score = *std::max_element(row, row + top_k);
            DraftProposal proposal;
            proposal.token_ids.assign(ids + position * top_k, ids + (position + 1) * top_k);
            proposal.probabilities.resize(top_k);
            float probability_sum = 0.0f;
            for (size_t index = 0; index < top_k; ++index) {
                proposal.probabilities[index] = std::exp((row[index] - max_score) / temperature);
                probability_sum += proposal.probabilities[index];
            }
            OPENVINO_ASSERT(probability_sum > 0.0f && std::isfinite(probability_sum),
                            "The DFlash candidate lattice produced an invalid proposal distribution.");
            for (auto& probability : proposal.probabilities) {
                probability /= probability_sum;
            }
            std::discrete_distribution<size_t> distribution(proposal.probabilities.begin(),
                                                            proposal.probabilities.end());
            const size_t selected_index = distribution(sequence.proposal_rng);
            candidates.token_ids.push_back(proposal.token_ids[selected_index]);
            candidates.log_probs.push_back(std::log(proposal.probabilities[selected_index]));
            candidates.proposals.push_back(std::make_shared<const DraftProposal>(std::move(proposal)));
            previous_index = selected_index;
        }
        return candidates;
    }

    static ov::CompiledModel compile_draft_model(const ov::genai::ModelDesc& model_desc,
                                                 bool use_external_embeddings,
                                                 bool gather_hidden_states,
                                                 const ov::genai::utils::dflash::DFlashRTInfo& rt_info) {
        const auto& model = model_desc.model;
        OPENVINO_ASSERT(model, "DFlash draft model cannot be null.");
        OPENVINO_ASSERT(utils::has_input(model, "hidden_states"),
                        "DFlash CB/PA draft model must have 'hidden_states' input.");
        OPENVINO_ASSERT(utils::has_input(model, "token_type_ids"),
                        "DFlash draft model must have a 'token_type_ids' input: DFlash drafts run on PagedAttention, "
                        "which takes every per-token input over the rows [context ; block]. Re-export the draft with "
                        "a current optimum-intel.");
        if (use_external_embeddings) {
            OPENVINO_ASSERT(utils::has_input(model, "inputs_embeds"),
                            "DFlash VLM draft model must have an 'inputs_embeds' input.");
            OPENVINO_ASSERT(!utils::has_input(model, "input_ids"),
                            "DFlash VLM draft model must not have an 'input_ids' input.");
        } else {
            OPENVINO_ASSERT(utils::has_input(model, "input_ids"),
                            "DFlash CB/PA draft model must have an 'input_ids' input after load-time transforms.");
        }
        OPENVINO_ASSERT(utils::has_input(model, "position_ids"),
                        "DFlash CB/PA draft model must have a 'position_ids' input.");
        OPENVINO_ASSERT(model_has_output(model, "logits"), "DFlash CB/PA draft model must expose 'logits'.");
        if (gather_hidden_states) {
            OPENVINO_ASSERT(model_has_output(model, "last_hidden_state"),
                            "A DFlash draft with a gated Markov head must expose 'last_hidden_state'.");
        }
        ov::pass::SDPAToPagedAttention().run_on_model(model);
        // only the candidate rows need logits; sampled_tokens_indices selects them
        utils::apply_gather_before_matmul_transformation(model);
        OPENVINO_ASSERT(utils::has_input(model, "sampled_tokens_indices"),
                        "DFlash paged draft: failed to insert the candidate-row gather before the lm_head.");
        if (gather_hidden_states) {
            utils::dflash::gather_last_hidden_state_with_logits(model);
        }
        auto compile_properties = model_desc.properties;
        utils::dflash::apply_dflash_gpu_compile_properties(rt_info, model_desc.device, compile_properties);
        return utils::singleton_core().compile_model(model, model_desc.device, compile_properties);
    }

    ov::Tensor apply_backbone_logit_transform(const ov::Tensor& logits) const {
        if (m_output_multiplier == 1.0f && m_final_logit_softcapping == 0.0f) {
            return logits;
        }
        OPENVINO_ASSERT(logits.get_element_type() == ov::element::f32,
                        "DFlash backbone sampling requires FP32 logits.");
        ov::Tensor transformed_logits(logits.get_element_type(), logits.get_shape());
        const auto* source = logits.data<const float>();
        auto* destination = transformed_logits.data<float>();
        for (size_t index = 0; index < logits.get_size(); ++index) {
            float value = source[index] * m_output_multiplier;
            if (m_final_logit_softcapping > 0.0f) {
                value = std::tanh(value / m_final_logit_softcapping) * m_final_logit_softcapping;
            }
            destination[index] = value;
        }
        return transformed_logits;
    }

    void sample_one_candidate(DraftSequence& sequence, const ov::Tensor& logits, DraftCandidates& candidates) {
        const auto& group = sequence.group;
        const auto seq = (*group)[0];
        const size_t generated_before = seq->get_generated_len();
        group->schedule_tokens(1);
        group->set_output_seq_len(1);
        group->set_num_validated_tokens(0);
        m_sampler.sample({group}, logits);
        group->finish_iteration();

        const auto& generated = seq->get_generated_ids();
        const auto& log_probs = seq->get_generated_log_probs();
        candidates.token_ids.insert(candidates.token_ids.end(), generated.begin() + generated_before, generated.end());
        candidates.log_probs.insert(candidates.log_probs.end(), log_probs.begin() + generated_before, log_probs.end());
        if (group->get_sampling_parameters().is_multinomial()) {
            const auto& proposals = seq->get_draft_proposals();
            OPENVINO_ASSERT(proposals.size() == generated.size(),
                            "DFlash backbone sampler proposals are out of sync with generated tokens.");
            candidates.proposals.insert(candidates.proposals.end(),
                                        proposals.begin() + generated_before,
                                        proposals.end());
        }
    }

    void execute_inference(ov::InferRequest& request) {
        const auto start = std::chrono::steady_clock::now();
        request.infer();
        m_stage_inference_us += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
    }

    void finish_stage(size_t num_candidates) {
        const MicroSeconds duration(static_cast<float>(m_stage_inference_us));
        m_raw_perf_metrics.m_durations.emplace_back(duration);
        m_raw_perf_metrics.m_batch_sizes.emplace_back(num_candidates);
        m_raw_perf_metrics.m_inference_durations[0] += duration;
        m_stage_inference_us = 0;
    }

    static constexpr size_t BATCH_SIZE = 1;
    Tokenizer m_tokenizer;
    EmbeddingsModel::Ptr m_embedding_model;
    // the Markov head reads the hidden states of the candidate rows, so the draft returns them with the logits
    bool m_gated_markov_head = false;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_request;
    std::unique_ptr<KVCacheManager> m_kv_cache;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    uint64_t m_stage_inference_us = 0;
    int64_t m_mask_token_id = -1;
    size_t m_candidate_position_offset = 1;
    float m_output_multiplier = 1.0f;
    float m_final_logit_softcapping = 0.0f;
    // at most one of them, for a draft with a Markov head
    std::optional<ov::InferRequest> m_candidate_scorer_request;
    std::optional<ov::InferRequest> m_markov_step_request;
    // reused for every lattice row walked through the sampler
    ov::Tensor m_candidate_row;
    std::map<uint64_t, DraftSequence> m_sequences;
    std::vector<int32_t> m_free_blocks;
};

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const ov::genai::utils::dflash::DFlashRTInfo& rt_info) {
    OPENVINO_ASSERT(rt_info.dflash_mode, "DFlash continuous batching requires dflash_mode=true.");
    OPENVINO_ASSERT(!rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    OPENVINO_ASSERT(!main_model_desc.scheduler_config.enable_prefix_caching,
                    "DFlash CB/PA does not support scheduler_config.enable_prefix_caching.");
    OPENVINO_ASSERT(rt_info.input_embedding_scale > 0.0f,
                    "DFlash input_embedding_scale must be positive.");
    OPENVINO_ASSERT(rt_info.output_multiplier > 0.0f,
                    "DFlash output_multiplier must be positive.");
    OPENVINO_ASSERT(rt_info.final_logit_softcapping >= 0.0f,
                    "DFlash final_logit_softcapping cannot be negative.");

    OPENVINO_ASSERT(main_model_desc.model && draft_model_desc.model,
                    "DFlash requires both target and draft models.");
    OPENVINO_ASSERT(!utils::is_npu_requested(main_model_desc.device, main_model_desc.properties) &&
                        !utils::is_npu_requested(draft_model_desc.device, draft_model_desc.properties),
                    "DFlash speculative decoding requires the Continuous Batching/Paged Attention backend on CPU or GPU; "
                    "NPU is not supported.");

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

    // PagedAttention is the only DFlash draft backend.
    if (auto backend_it = draft_model_desc_for_runner.properties.find("ATTENTION_BACKEND");
        backend_it != draft_model_desc_for_runner.properties.end()) {
        const auto backend = backend_it->second.as<std::string>();
        OPENVINO_ASSERT(backend == PA_BACKEND,
                        "DFlash drafts run on PagedAttention only; the draft ATTENTION_BACKEND must be '", PA_BACKEND,
                        "', got '", backend, "'.");
        draft_model_desc_for_runner.properties.erase(backend_it);
    }

    // Resolve authoritative metadata before mutating either model so a malformed target leaves the draft reusable.
    auto retained_hidden_state_locators =
        utils::dflash::resolve_target_hidden_state_locators(main_model, rt_info.target_layer_ids);

    // Preserve the VLM draft's external inputs_embeds bridge. The legacy LLM
    // path still grafts an in-graph target embedding and remains input_ids-native.
    if (is_vlm_dflash) {
        OPENVINO_ASSERT(needs_embedding_attach,
                        "DFlash VLM draft model must be inputs_embeds-native.");
        utils::dflash::scale_draft_inputs_embeds(draft_model_desc_for_runner.model, rt_info.input_embedding_scale);
    } else if (needs_embedding_attach) {
        utils::dflash::attach_target_embedding_to_draft(main_model,
                                                        draft_model_desc_for_runner.model,
                                                        rt_info.input_embedding_scale);
    }
    if (needs_lm_head_graft) {
        utils::dflash::attach_target_lm_head_to_draft(main_model,
                                                      draft_model_desc_for_runner.model,
                                                      markov_head_reads_hidden_states(rt_info));
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
        rt_info.target_layer_ids);

    m_tokenizer = main_model_desc.tokenizer;
    auto main_generation_config = main_model_desc.generation_config;
    dflash_cb::ensure_num_assistant_tokens_is_set(main_generation_config);
    m_max_num_batched_tokens = main_model_desc.scheduler_config.max_num_batched_tokens;
    OPENVINO_ASSERT(m_max_num_batched_tokens >= main_generation_config.num_assistant_tokens.value() + 1,
                    "DFlash CB/PA requires max_num_batched_tokens >= num_assistant_tokens + 1 to fit a validation "
                    "window.");
    m_generation_config = main_generation_config;
    auto target_scheduler_config = main_model_desc.scheduler_config;
    target_scheduler_config.num_linear_attention_blocks =
        dflash_cb::adjusted_linear_attention_block_count(target_scheduler_config.num_linear_attention_blocks,
                                                          main_generation_config.num_assistant_tokens.value(),
                                                          target_has_linear_attention);
    if (draft_model_desc_for_runner.device.empty()) {
        draft_model_desc_for_runner.device = main_model_desc.device;
    }

    m_draft = std::make_shared<DFlashCBDraftRunner>(draft_model_desc_for_runner,
                                                    m_tokenizer,
                                                    rt_info,
                                                    draft_embedding_model);

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
}

GenerationConfig ContinuousBatchingPipeline::DFlashDecodingImpl::make_draft_generation_config(
    const GenerationConfig& config) const {
    auto draft_config = config;
    draft_config.rng_seed = detail::proposal_rng_seed(config.rng_seed);
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
    dflash_cb::add_draft_assistant_token_headroom(draft_config, config);
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

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_hidden_prefix_length(uint64_t request_id,
                                                                                  const RequestState& state) const {
    OPENVINO_ASSERT(!state.generated_tokens.empty(),
                    "DFlash hidden prefix can only be validated after target generated a seed token.");
    const size_t expected = state.prompt_len + state.generated_tokens.size() - 1;
    const size_t actual = m_draft->get_consumed_hidden_states(request_id) + state.pending_hidden_deltas.token_count();
    OPENVINO_ASSERT(actual == expected,
                    "DFlash hidden prefix length mismatch before draft inference of request ", request_id,
                    ": expected ", expected, " rows, got ", actual, ".");
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::append_new_hidden_rows(uint64_t request_id,
                                                                           RequestState& state,
                                                                           const ov::Tensor& hidden_state,
                                                                           size_t num_processed_tokens) {
    if (!hidden_state || hidden_state.get_size() == 0) {
        return;
    }
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                    "DFlash target hidden states must have shape [seq_len, 1, hidden].");
    const size_t num_rows = shape[0];
    const size_t num_received =
        m_draft->get_consumed_hidden_states(request_id) + state.pending_hidden_deltas.token_count();
    // The rows end at the last processed token. A request the target did not schedule republishes its
    // previous rows, and a recomputed one replays rows received already.
    if (num_processed_tokens <= num_received) {
        return;
    }
    OPENVINO_ASSERT(num_processed_tokens >= num_rows && num_processed_tokens - num_rows <= num_received,
                    "DFlash target did not publish the hidden states of request ", request_id,
                    " from position ", num_received, ".");
    const size_t num_new_rows = num_processed_tokens - num_received;
    if (num_new_rows == num_rows) {
        append_pending_hidden_delta(state, hidden_state, false);
        return;
    }
    append_pending_hidden_delta(
        state,
        ov::Tensor(hidden_state, ov::Coordinate{num_rows - num_new_rows, 0, 0}, ov::Coordinate{num_rows, 1, shape[2]}),
        false);
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
            m_draft->remove_sequence(state_it->first);
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
    OPENVINO_ASSERT(sampling_params.is_greedy_decoding() || sampling_params.is_multinomial(),
                    "DFlash supports greedy or multinomial sampling only.");
    OPENVINO_ASSERT(sampling_params.num_beams == 1, "DFlash CB/PA does not support beam search.");
    OPENVINO_ASSERT(sampling_params.num_return_sequences == 1, "DFlash CB/PA supports one sequence per request.");
    OPENVINO_ASSERT(!sampling_params.adapters.has_value(),
                    "DFlash CB/PA does not support adapters until target and draft adapter parity is validated.");
    if (is_vlm_dflash) {
        dflash_cb::ensure_vlm_generation_config(sampling_params);
    }
    drop_finished_request_states();
    if (is_vlm_dflash) {
        OPENVINO_ASSERT(!has_active_request_state() && !m_main_pipeline->has_non_finished_requests(),
                        "DFlash VLM supports only one active request. Wait for the current request to finish before "
                        "adding another.");
    }
    OPENVINO_ASSERT(m_request_states.find(request_id) == m_request_states.end(),
                    "DFlash request ", request_id, " is already active.");

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
        m_draft->add_sequence(request_id, state.prompt_len, make_draft_generation_config(sampling_params_copy));
    } else {
        m_draft->add_sequence(request_id, input_ids, make_draft_generation_config(sampling_params_copy));
    }
    m_request_states[request_id] = std::move(state);

    // If target request creation fails, drop the draft state so the request ID can be reused.
    try {
        return m_main_pipeline->add_request(request_id,
                                            input_ids,
                                            sampling_params_copy,
                                            prompt_ids,
                                            lm_extra_inputs);
    } catch (...) {
        m_request_states.erase(request_id);
        m_draft->remove_sequence(request_id);
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

std::vector<ContinuousBatchingPipeline::DFlashDecodingImpl::DraftInput>
ContinuousBatchingPipeline::DFlashDecodingImpl::plan_draft_inputs() {
    std::vector<DraftInput> inputs;
    // The scheduler serves the generate phase first, in this order, and splits a validation window that
    // exceeds the remaining token budget. Drafting fewer candidates keeps every window whole.
    size_t token_budget = m_max_num_batched_tokens;
    for (const auto& progress : m_main_pipeline->get_requests_progress()) {
        if (progress.num_generate_tokens == 0) {
            continue;
        }
        size_t num_scheduled_tokens = progress.num_generate_tokens;
        auto state_it = m_request_states.find(progress.request_id);
        // a request with candidates pending validation or tokens to recompute catches up before drafting
        if (state_it != m_request_states.end() && progress.num_generate_tokens == 1 && token_budget > 1) {
            const uint64_t request_id = progress.request_id;
            auto& state = state_it->second;
            const auto& config = state.generation_config;
            const size_t generated_len = state.generated_tokens.size();
            if (!state.finished && has_pending_hidden_delta(state) && generated_len > 0) {
                const size_t draft_count = dflash_cb::draft_candidate_count(config.num_assistant_tokens.value(),
                                                                            generated_len,
                                                                            config.max_new_tokens);
                const size_t validation_count =
                    dflash_cb::validation_candidate_count(draft_count, generated_len, config.max_new_tokens);
                if (validation_count == 0) {
                    clear_pending_hidden_delta(state);
                } else {
                    validate_hidden_prefix_length(request_id, state);
                    DraftInput input;
                    input.request_id = request_id;
                    input.seed_token = state.generated_tokens.back();
                    input.hidden_delta = materialize_pending_hidden_delta(state);
                    input.candidate_count = draft_count;
                    input.validation_count = std::min(validation_count, token_budget - 1);
                    clear_pending_hidden_delta(state);
                    state.processed_before_validation = progress.num_processed_tokens;
                    num_scheduled_tokens = input.validation_count + 1;
                    inputs.push_back(std::move(input));
                }
            }
        }
        token_budget -= std::min(token_budget, num_scheduled_tokens);
    }
    return inputs;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::submit_candidates(const DraftInput& input,
                                                                      DraftCandidates candidates) {
    auto& state = m_request_states.at(input.request_id);
    OPENVINO_ASSERT(!candidates.token_ids.empty(),
                    "DFlash draft sampler produced no candidates despite requested validation candidates. ",
                    "request_id=", input.request_id,
                    ", generated_before_draft=", state.generated_tokens.size(),
                    ", draft_count=", input.candidate_count,
                    ", validation_count=", input.validation_count);
    OPENVINO_ASSERT(candidates.token_ids.size() == candidates.log_probs.size(),
                    "DFlash draft candidate tokens and log-probs must stay aligned.");
    state.generated_before_draft = state.generated_tokens.size();
    state.draft_generated = candidates.token_ids.size();

    auto candidate_tokens = state.generated_tokens;
    candidate_tokens.insert(candidate_tokens.end(), candidates.token_ids.begin(), candidates.token_ids.end());
    std::vector<float> candidate_log_probs(state.generated_tokens.size(), 0.0f);
    candidate_log_probs.insert(candidate_log_probs.end(), candidates.log_probs.begin(), candidates.log_probs.end());
    std::vector<DraftProposalPtr> aligned_proposals;
    if (!candidates.proposals.empty()) {
        aligned_proposals.resize(state.generated_tokens.size());
        aligned_proposals.insert(aligned_proposals.end(),
                                 std::make_move_iterator(candidates.proposals.begin()),
                                 std::make_move_iterator(candidates.proposals.end()));
    }
    GeneratedSequences candidate_sequences;
    candidate_sequences.emplace(0,
                                GeneratedSequence(candidate_tokens,
                                                  candidate_log_probs,
                                                  0,
                                                  {},
                                                  nullptr,
                                                  std::move(aligned_proposals)));
    m_main_pipeline->update_request(input.request_id, candidate_sequences, false);
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;
    const auto step_start = std::chrono::steady_clock::now();

    m_main_pipeline->pull_awaiting_requests();
    drop_finished_request_states();

    const auto draft_start = std::chrono::steady_clock::now();
    const auto draft_inputs = plan_draft_inputs();
    if (!draft_inputs.empty()) {
        auto proposals = m_draft->propose(draft_inputs);
        size_t num_candidates = 0;
        for (size_t idx = 0; idx < draft_inputs.size(); ++idx) {
            num_candidates += proposals[idx].token_ids.size();
            submit_candidates(draft_inputs[idx], std::move(proposals[idx]));
        }
        const auto draft_step_duration =
            MicroSeconds(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - draft_start));
        auto& draft_step_raw_metrics = m_perf_metrics.draft_step_metrics.raw_metrics;
        draft_step_raw_metrics.m_durations.push_back(draft_step_duration);
        draft_step_raw_metrics.m_batch_sizes.push_back(num_candidates);
        draft_step_raw_metrics.m_inference_durations[0] += draft_step_duration;
    }
    for (auto& [_, state] : m_request_states) {
        state.pending_hidden_deltas.own_data();
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

    m_pipeline_metrics = m_main_pipeline->get_metrics();

    const auto step_end = std::chrono::steady_clock::now();
    const auto step_microsec_duration = PerfMetrics::get_microsec(step_end - step_start);
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        raw_perf_counters.m_token_infer_durations.emplace_back(step_microsec_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(step_microsec_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_end);
        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);

        main_raw_perf_counters.m_durations.push_back(MicroSeconds(main_duration));
        main_raw_perf_counters.m_inference_durations[0] += MicroSeconds(m_pipeline_metrics.inference_duration);
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
    std::map<uint64_t, RequestProgress> progress_by_request;
    for (const auto& progress : m_main_pipeline->get_requests_progress()) {
        progress_by_request.emplace(progress.request_id, progress);
    }
    for (const auto& [request_id, generated_sequences] : main_generated_requests) {
        auto state_it = m_request_states.find(request_id);
        auto progress_it = progress_by_request.find(request_id);
        if (state_it == m_request_states.end() || progress_it == progress_by_request.end() ||
            generated_sequences.empty()) {
            continue;
        }

        auto& state = state_it->second;
        const auto& progress = progress_it->second;
        if (state.draft_generated > 0) {
            if (progress.num_tokens_to_validate > 0) {
                // the target deferred the validation window; its generated tokens still end with the candidates
                OPENVINO_ASSERT(progress.num_processed_tokens == state.processed_before_validation,
                                "DFlash validation window of request ", request_id,
                                " was scheduled partially; max_num_batched_tokens must fit every validation window.");
                continue;
            }
            OPENVINO_ASSERT(progress.num_processed_tokens > state.processed_before_validation,
                            "DFlash validation window of request ", request_id,
                            " was dropped without validation, which happens when the KV cache cannot grow.");
        }

        const auto& generated_sequence = generated_sequences.begin()->second;
        const auto accounting =
            dflash_cb::validation_accounting(state.draft_generated,
                                             state.generated_before_draft,
                                             generated_sequence.token_ids.size());
        if (accounting.target_extended) {
            const float acceptance_rate = static_cast<float>(accounting.accepted) / state.draft_generated * 100.0f;
            m_sd_metrics.update_draft_generated_len(request_id, state.draft_generated);
            m_sd_metrics.update_draft_accepted_tokens(request_id, accounting.accepted);
            m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate);
            m_perf_metrics.num_draft_tokens += state.draft_generated;
            m_perf_metrics.num_accepted_tokens += accounting.accepted;
        }

        auto hidden_delta = dflash_cb::truncate_normalized_hidden_state_from_end(generated_sequence.hidden_states,
                                                                                 accounting.rejected);
        append_new_hidden_rows(request_id, state, hidden_delta, progress.num_processed_tokens);
        state.generated_tokens = generated_sequence.token_ids;
        m_draft->sync_generated_tokens(request_id, state.generated_tokens);
        state.draft_generated = 0;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_requests() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    if (m_main_pipeline) {
        m_main_pipeline->finish_request();
    }
    for (const auto& [request_id, _] : m_request_states) {
        m_draft->remove_sequence(request_id);
    }
    m_request_states.clear();
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
    OPENVINO_ASSERT(m_model_input_type != ModelInputType::EMBEDDINGS || input_ids.size() == 1,
                    "DFlash VLM supports batch size 1 only.");

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft->reset_perf_metrics();
    auto start_time = std::chrono::steady_clock::now();

    auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);
    OPENVINO_ASSERT(
        !streamer_ptr->has_callback() ||
            (input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial())),
        "DFlash CB/PA streaming only supports batch size=1 with greedy/multinomial decoding.");

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

    m_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();
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
