// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include "utils.hpp"

namespace ov::genai::dflash_cb {

namespace {

size_t row_width(const ov::Tensor& rows) {
    OPENVINO_ASSERT(rows && rows.get_element_type() == ov::element::f32 && rows.is_continuous() &&
                        !rows.get_shape().empty(),
                    "DFlash candidate selection requires contiguous FP32 draft outputs.");
    return rows.get_shape().back();
}

const float* row_data(const ov::Tensor& rows, size_t row) {
    return rows.data<const float>() + row * row_width(rows);
}

int64_t argmax(const float* row, size_t vocab_size) {
    return std::max_element(row, row + vocab_size) - row;
}

bool blocks_are_packed(const std::vector<DraftBlockRows>& blocks, size_t num_rows) {
    size_t next_row = 0;
    for (const auto& block : blocks) {
        if (block.first_row != next_row) {
            return false;
        }
        next_row += block.num_rows;
    }
    return next_row == num_rows;
}

// [rows of all blocks, width]: a view when the blocks fill the output back to back, else a copy.
ov::Tensor packed_rows(const ov::Tensor& rows, const std::vector<DraftBlockRows>& blocks, size_t num_packed_rows) {
    const size_t width = row_width(rows);
    if (blocks_are_packed(blocks, rows.get_size() / width)) {
        return ov::Tensor(ov::element::f32, {num_packed_rows, width}, const_cast<void*>(rows.data()));
    }
    ov::Tensor packed(ov::element::f32, {num_packed_rows, width});
    float* destination = packed.data<float>();
    for (const auto& block : blocks) {
        std::copy_n(row_data(rows, block.first_row), block.num_rows * width, destination);
        destination += block.num_rows * width;
    }
    return packed;
}

}  // namespace

CandidateSelector::CandidateSelector(const utils::dflash::DFlashRTInfo& rt_info,
                                     const std::string& device,
                                     const ov::AnyMap& properties)
    : m_output_multiplier(rt_info.output_multiplier),
      m_final_logit_softcapping(rt_info.final_logit_softcapping) {
    const auto& markov_head = rt_info.markov_head_computation;
    if (!markov_head) {
        return;
    }
    auto request = utils::singleton_core().compile_model(markov_head, device, properties).create_infer_request();
    if (utils::dflash::is_markov_step(markov_head)) {
        m_markov_step = std::move(request);
    } else {
        m_reads_hidden_states = utils::has_input(markov_head, "draft_hidden_states");
        m_candidate_scorer = std::move(request);
    }
}

void CandidateSelector::select(const std::vector<DraftBlockRows>& blocks,
                               const ov::Tensor& logits,
                               const ov::Tensor& hidden_states,
                               const std::vector<BlockDrafter*>& drafters) {
    m_inference_duration = 0.0f;
    if (blocks.empty()) {
        return;
    }
    OPENVINO_ASSERT(drafters.size() == blocks.size(), "Every DFlash draft block needs a drafter.");
    const size_t num_rows = logits.get_size() / row_width(logits);
    for (const auto& block : blocks) {
        OPENVINO_ASSERT(block.num_candidates > 0 && block.num_candidates <= block.num_rows &&
                            block.first_row + block.num_rows <= num_rows,
                        "DFlash draft block rows exceed the ", num_rows, " gathered draft rows.");
    }
    if (m_candidate_scorer) {
        select_scored_candidates(blocks, logits, hidden_states, drafters);
    } else if (m_markov_step) {
        walk_markov_steps(blocks, logits, drafters);
    } else {
        draft_rows(blocks, logits, drafters);
    }
}

ov::Tensor& CandidateSelector::backbone_row(const float* logits, size_t vocab_size) {
    if (!m_row || m_row.get_size() != vocab_size) {
        m_row = ov::Tensor(ov::element::f32, {1, 1, vocab_size});
    }
    float* row = m_row.data<float>();
    for (size_t token = 0; token < vocab_size; ++token) {
        float value = logits[token] * m_output_multiplier;
        if (m_final_logit_softcapping > 0.0f) {
            value = std::tanh(value / m_final_logit_softcapping) * m_final_logit_softcapping;
        }
        row[token] = value;
    }
    return m_row;
}

void CandidateSelector::draft_rows(const std::vector<DraftBlockRows>& blocks,
                                   const ov::Tensor& logits,
                                   const std::vector<BlockDrafter*>& drafters) {
    const size_t vocab_size = row_width(logits);
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        auto& drafter = *drafters[idx];
        const bool takes_argmax = logit_processors_keep_argmax(drafter.config());
        for (size_t position = 0; position < blocks[idx].num_candidates; ++position) {
            const float* row = row_data(logits, blocks[idx].first_row + position);
            if (takes_argmax) {
                // output_multiplier > 0 and the soft cap keep the argmax of the backbone logits
                drafter.append(argmax(row, vocab_size), 0.0f, nullptr);
            } else if (!drafter.sample(backbone_row(row, vocab_size))) {
                break;
            }
        }
    }
}

// Drafts one position of every block per Markov step inference: the position's logits get the Markov bias that
// follows the token drafted before it, the anchor at the first position.
void CandidateSelector::walk_markov_steps(const std::vector<DraftBlockRows>& blocks,
                                          const ov::Tensor& logits,
                                          const std::vector<BlockDrafter*>& drafters) {
    const size_t vocab_size = row_width(logits);
    std::vector<int64_t> previous_tokens(blocks.size());
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        previous_tokens[idx] = blocks[idx].anchor_token;
    }
    std::vector<size_t> walking(blocks.size());
    std::iota(walking.begin(), walking.end(), size_t{0});
    for (size_t position = 0; !walking.empty(); ++position) {
        ov::Tensor prev_token_ids(ov::element::i64, {walking.size()});
        for (size_t step = 0; step < walking.size(); ++step) {
            prev_token_ids.data<int64_t>()[step] = previous_tokens[walking[step]];
        }
        m_markov_step->set_tensor("prev_token_ids", prev_token_ids);
        infer(*m_markov_step);
        const auto markov_bias = m_markov_step->get_tensor("markov_bias");
        OPENVINO_ASSERT(markov_bias.get_element_type() == ov::element::f32 &&
                            markov_bias.get_shape() == ov::Shape({walking.size(), vocab_size}),
                        "The Markov bias must be one FP32 row over the draft vocabulary per block.");
        std::vector<size_t> still_walking;
        for (size_t step = 0; step < walking.size(); ++step) {
            const size_t idx = walking[step];
            auto& drafter = *drafters[idx];
            auto& row = backbone_row(row_data(logits, blocks[idx].first_row + position), vocab_size);
            float* corrected = row.data<float>();
            std::transform(corrected,
                           corrected + vocab_size,
                           markov_bias.data<const float>() + step * vocab_size,
                           corrected,
                           std::plus<float>());
            std::optional<int64_t> token;
            if (logit_processors_keep_argmax(drafter.config())) {
                token = argmax(corrected, vocab_size);
                drafter.append(*token, 0.0f, nullptr);
            } else {
                token = drafter.sample(row);
            }
            // the next position's Markov bias needs the token drafted here
            if (!token) {
                continue;
            }
            previous_tokens[idx] = *token;
            if (position + 1 < blocks[idx].num_candidates) {
                still_walking.push_back(idx);
            }
        }
        walking = std::move(still_walking);
    }
}

// Scores the rows of all blocks with one candidate scorer inference and walks each block's Top-K lattice: greedily, by
// drawing from the K candidates of every row for a sampled request, or through the draft sampler when the request's
// logit processors can change the argmax.
void CandidateSelector::select_scored_candidates(const std::vector<DraftBlockRows>& blocks,
                                                 const ov::Tensor& logits,
                                                 const ov::Tensor& hidden_states,
                                                 const std::vector<BlockDrafter*>& drafters) {
    const size_t vocab_size = row_width(logits);
    const size_t num_blocks = blocks.size();
    ov::Tensor anchor_token_ids(ov::element::i64, {num_blocks});
    ov::Tensor subsequence_begins(ov::element::i64, {num_blocks + 1});
    auto* begins = subsequence_begins.data<int64_t>();
    begins[0] = 0;
    for (size_t idx = 0; idx < num_blocks; ++idx) {
        anchor_token_ids.data<int64_t>()[idx] = blocks[idx].anchor_token;
        begins[idx + 1] = begins[idx] + static_cast<int64_t>(blocks[idx].num_rows);
    }
    const size_t num_rows = static_cast<size_t>(begins[num_blocks]);
    validate_subsequence_begins(subsequence_begins, num_blocks, num_rows);

    auto& scorer = *m_candidate_scorer;
    scorer.set_tensor("draft_logits", packed_rows(logits, blocks, num_rows));
    if (m_reads_hidden_states) {
        scorer.set_tensor("draft_hidden_states", packed_rows(hidden_states, blocks, num_rows));
    }
    scorer.set_tensor("anchor_token_ids", anchor_token_ids);
    scorer.set_tensor("subsequence_begins", subsequence_begins);
    infer(scorer);

    const auto edge_scores = scorer.get_tensor("edge_scores");
    const auto candidate_ids = scorer.get_tensor("candidate_ids");
    validate_candidate_lattice(edge_scores, candidate_ids, num_rows, vocab_size);
    for (size_t idx = 0; idx < num_blocks; ++idx) {
        auto& drafter = *drafters[idx];
        const auto& config = drafter.config();
        const size_t first_row = static_cast<size_t>(begins[idx]);
        const size_t num_candidates = blocks[idx].num_candidates;
        if (config.is_multinomial()) {
            for (auto& candidate : sample_candidate_path(edge_scores,
                                                         candidate_ids,
                                                         first_row,
                                                         num_candidates,
                                                         config.temperature,
                                                         drafter.lattice_rng())) {
                drafter.append(candidate.token, candidate.log_prob, std::move(candidate.proposal));
            }
        } else if (logit_processors_keep_argmax(config)) {
            for (const int64_t token : greedy_candidate_path(edge_scores, candidate_ids, first_row, num_candidates)) {
                drafter.append(token, 0.0f, nullptr);
            }
        } else {
            walk_candidate_lattice(candidate_ids,
                                   first_row,
                                   num_candidates,
                                   [&](size_t position, size_t previous_index) -> std::optional<int64_t> {
                                       const size_t logits_row = blocks[idx].first_row + position - first_row;
                                       auto& row = backbone_row(row_data(logits, logits_row), vocab_size);
                                       fill_candidate_row(row.data<float>(),
                                                          vocab_size,
                                                          edge_scores,
                                                          candidate_ids,
                                                          position,
                                                          previous_index,
                                                          row.data<float>());
                                       return drafter.sample(row);
                                   });
        }
    }
}

void CandidateSelector::infer(ov::InferRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    m_inference_duration += std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace ov::genai::dflash_cb
