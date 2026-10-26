// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai::dflash_cb {

// Drafts the greedy candidates of the blocks a draft step forwarded, from the draft outputs of their rows: the
// argmax of every row, or, for a draft with a Markov head (DSpark, DFlash-2), a walk that makes every candidate
// depend on the one drafted before it. The Markov head runs once per step for the blocks of all requests: the Top-K
// candidate scorer scores every row in one inference, the dense Markov step one position of every block per
// inference.
class CandidateSelector {
public:
    // argmax of every row
    CandidateSelector() = default;

    CandidateSelector(const utils::dflash::DFlashRTInfo& rt_info, const std::string& device, const ov::AnyMap& properties);

    // A gated Markov head (DFlash-2) also scores the draft's last_hidden_state of every row.
    bool reads_hidden_states() const {
        return m_reads_hidden_states;
    }

    // logits [rows, 1, vocab] and, when reads_hidden_states(), hidden_states [rows, 1, hidden] hold the rows of
    // every block at block.first_row. Returns the block.num_candidates tokens of every block.
    std::vector<std::vector<int64_t>> select(const std::vector<DraftBlockRows>& blocks,
                                             const ov::Tensor& logits,
                                             const ov::Tensor& hidden_states);

    // Microseconds of Markov head inference in the last select().
    float get_inference_duration() const {
        return m_inference_duration;
    }

private:
    std::vector<std::vector<int64_t>> select_argmax(const std::vector<DraftBlockRows>& blocks,
                                                    const ov::Tensor& logits) const;
    std::vector<std::vector<int64_t>> walk_markov_steps(const std::vector<DraftBlockRows>& blocks,
                                                        const ov::Tensor& logits);
    std::vector<std::vector<int64_t>> select_scored_candidates(const std::vector<DraftBlockRows>& blocks,
                                                               const ov::Tensor& logits,
                                                               const ov::Tensor& hidden_states);
    void infer(ov::InferRequest& request);

    bool m_reads_hidden_states = false;
    // the dense Markov step adds its bias to the backbone logits scaled by output_multiplier, then soft-capped
    float m_output_multiplier = 1.0f;
    float m_final_logit_softcapping = 0.0f;
    // at most one of them, for a draft with a Markov head
    std::optional<ov::InferRequest> m_candidate_scorer;
    std::optional<ov::InferRequest> m_markov_step;
    float m_inference_duration = 0.0f;
};

}  // namespace ov::genai::dflash_cb
