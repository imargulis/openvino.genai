// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "sampling/draft_proposal.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai::dflash_cb {

// Appends the candidates of one block to its request's draft sequence.
class BlockDrafter {
public:
    virtual ~BlockDrafter() = default;

    // The draft request's sampling parameters.
    virtual const GenerationConfig& config() const = 0;

    // Appends a candidate the selector chose; a sampled candidate carries the distribution it was drawn from.
    virtual void append(int64_t token, float log_prob, DraftProposalPtr proposal) = 0;

    // Draws a candidate from a [1, 1, vocab] logits row with the request's draft sampler, which applies the request's
    // logit processors and records the distribution of a sampled candidate, and appends it. nullopt when the sampler
    // appended no token, which ends the block.
    virtual std::optional<int64_t> sample(const ov::Tensor& row) = 0;

    // Draws the candidates of a sampled request from a Top-K candidate lattice.
    virtual std::mt19937& lattice_rng() = 0;
};

// Drafts the candidates of the blocks a draft step forwarded, from the draft outputs of their rows: from every row, or,
// for a draft with a Markov head (DSpark, DFlash-2), with a walk that makes every candidate depend on the one drafted
// before it. The Markov head runs once per step for the blocks of all requests: the Top-K candidate scorer scores
// every row in one inference, the dense Markov step one position of every block per inference.
//
// A greedy request whose logit processors keep the argmax takes the best token of every row. Every other request
// draws its candidates with its draft sampler, so that its penalties and filters apply and the target can verify a
// sampled candidate against the distribution it was drawn from; a sampled request with a candidate scorer instead
// draws from the K candidates of every lattice row.
class CandidateSelector {
public:
    // drafts from every row
    CandidateSelector() = default;

    CandidateSelector(const utils::dflash::DFlashRTInfo& rt_info, const std::string& device, const ov::AnyMap& properties);

    // A gated Markov head (DFlash-2) also scores the draft's last_hidden_state of every row.
    bool reads_hidden_states() const {
        return m_reads_hidden_states;
    }

    // logits [rows, 1, vocab] and, when reads_hidden_states(), hidden_states [rows, 1, hidden] hold the rows of
    // every block at block.first_row. Drafts at most block.num_candidates tokens of every block with its drafter.
    void select(const std::vector<DraftBlockRows>& blocks,
                const ov::Tensor& logits,
                const ov::Tensor& hidden_states,
                const std::vector<BlockDrafter*>& drafters);

    // Microseconds of Markov head inference in the last select().
    float get_inference_duration() const {
        return m_inference_duration;
    }

private:
    void draft_rows(const std::vector<DraftBlockRows>& blocks,
                    const ov::Tensor& logits,
                    const std::vector<BlockDrafter*>& drafters);
    void walk_markov_steps(const std::vector<DraftBlockRows>& blocks,
                           const ov::Tensor& logits,
                           const std::vector<BlockDrafter*>& drafters);
    void select_scored_candidates(const std::vector<DraftBlockRows>& blocks,
                                  const ov::Tensor& logits,
                                  const ov::Tensor& hidden_states,
                                  const std::vector<BlockDrafter*>& drafters);
    // [1, 1, vocab] row of the backbone logits scaled by output_multiplier, then soft-capped
    ov::Tensor& backbone_row(const float* logits, size_t vocab_size);
    void infer(ov::InferRequest& request);

    bool m_reads_hidden_states = false;
    // the dense Markov step adds its bias to the backbone logits scaled by output_multiplier, then soft-capped
    float m_output_multiplier = 1.0f;
    float m_final_logit_softcapping = 0.0f;
    // at most one of them, for a draft with a Markov head
    std::optional<ov::InferRequest> m_candidate_scorer;
    std::optional<ov::InferRequest> m_markov_step;
    float m_inference_duration = 0.0f;
    // the row a candidate is drawn from by a draft sampler
    ov::Tensor m_row;
};

}  // namespace ov::genai::dflash_cb
