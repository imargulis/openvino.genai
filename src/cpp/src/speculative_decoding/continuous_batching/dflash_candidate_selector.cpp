// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

std::vector<std::vector<int64_t>> CandidateSelector::select(const std::vector<DraftBlockRows>& blocks,
                                                            const ov::Tensor& logits,
                                                            const ov::Tensor& hidden_states) {
    m_inference_duration = 0.0f;
    if (blocks.empty()) {
        return {};
    }
    const size_t num_rows = logits.get_size() / row_width(logits);
    for (const auto& block : blocks) {
        OPENVINO_ASSERT(block.num_candidates > 0 && block.num_candidates <= block.num_rows &&
                            block.first_row + block.num_rows <= num_rows,
                        "DFlash draft block rows exceed the ", num_rows, " gathered draft rows.");
    }
    if (m_candidate_scorer) {
        return select_scored_candidates(blocks, logits, hidden_states);
    }
    if (m_markov_step) {
        return walk_markov_steps(blocks, logits);
    }
    return select_argmax(blocks, logits);
}

std::vector<std::vector<int64_t>> CandidateSelector::select_argmax(const std::vector<DraftBlockRows>& blocks,
                                                                   const ov::Tensor& logits) const {
    // output_multiplier > 0 and the soft cap keep the argmax of the backbone logits
    const size_t vocab_size = row_width(logits);
    std::vector<std::vector<int64_t>> candidates(blocks.size());
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        for (size_t position = 0; position < blocks[idx].num_candidates; ++position) {
            const float* row = row_data(logits, blocks[idx].first_row + position);
            candidates[idx].push_back(std::max_element(row, row + vocab_size) - row);
        }
    }
    return candidates;
}

// Drafts one position of every block per Markov step inference: the position's logits get the Markov bias that
// follows the token drafted before it, the anchor at the first position.
std::vector<std::vector<int64_t>> CandidateSelector::walk_markov_steps(const std::vector<DraftBlockRows>& blocks,
                                                                       const ov::Tensor& logits) {
    const size_t vocab_size = row_width(logits);
    std::vector<std::vector<int64_t>> candidates(blocks.size());
    std::vector<size_t> walking(blocks.size());
    std::iota(walking.begin(), walking.end(), size_t{0});
    for (size_t position = 0; !walking.empty(); ++position) {
        ov::Tensor prev_token_ids(ov::element::i64, {walking.size()});
        for (size_t step = 0; step < walking.size(); ++step) {
            const size_t idx = walking[step];
            prev_token_ids.data<int64_t>()[step] = position == 0 ? blocks[idx].anchor_token : candidates[idx].back();
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
            const float* backbone = row_data(logits, blocks[idx].first_row + position);
            const float* bias = markov_bias.data<const float>() + step * vocab_size;
            size_t best = 0;
            float best_value = -std::numeric_limits<float>::infinity();
            for (size_t token = 0; token < vocab_size; ++token) {
                float value = backbone[token] * m_output_multiplier;
                if (m_final_logit_softcapping > 0.0f) {
                    value = std::tanh(value / m_final_logit_softcapping) * m_final_logit_softcapping;
                }
                value += bias[token];
                if (value > best_value) {
                    best_value = value;
                    best = token;
                }
            }
            candidates[idx].push_back(static_cast<int64_t>(best));
            if (position + 1 < blocks[idx].num_candidates) {
                still_walking.push_back(idx);
            }
        }
        walking = std::move(still_walking);
    }
    return candidates;
}

// Scores the rows of all blocks with one candidate scorer inference and walks each block's Top-K lattice greedily.
std::vector<std::vector<int64_t>> CandidateSelector::select_scored_candidates(const std::vector<DraftBlockRows>& blocks,
                                                                              const ov::Tensor& logits,
                                                                              const ov::Tensor& hidden_states) {
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
    std::vector<std::vector<int64_t>> candidates;
    candidates.reserve(num_blocks);
    for (size_t idx = 0; idx < num_blocks; ++idx) {
        candidates.push_back(greedy_candidate_path(edge_scores,
                                                   candidate_ids,
                                                   static_cast<size_t>(begins[idx]),
                                                   blocks[idx].num_candidates));
    }
    return candidates;
}

void CandidateSelector::infer(ov::InferRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    m_inference_duration += std::chrono::duration<float, std::micro>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace ov::genai::dflash_cb
