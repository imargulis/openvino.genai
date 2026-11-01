// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai::dflash_cb {

// One request's forwarded draft block. The draft logits hold the gathered rows of all blocks of a step one block
// after another.
struct DraftBlock {
    // the token before the first candidate, the seed
    int64_t anchor_token = 0;
    size_t first_row = 0;
    size_t num_rows = 0;
    // the leading rows the target validates, at most num_rows
    size_t num_candidates = 0;
};

// True when the logit processors of a draft sampler with this config cannot change which token has the highest logit,
// so a greedy draft can take the best token of every row without the sampler. Follows the transforms that
// LogitProcessor adds for greedy decoding; the log-prob transforms keep the argmax.
inline bool logit_processors_keep_argmax(const GenerationConfig& config) {
    return config.is_greedy_decoding() && config.repetition_penalty == 1.0f && config.presence_penalty == 0.0f &&
           config.frequency_penalty == 0.0f && config.min_new_tokens == 0 && !config.is_structured_output_generation();
}

// Appends the candidates of one block to its request's draft sequence.
class BlockDrafter {
public:
    virtual ~BlockDrafter() = default;

    // The draft request's sampling parameters.
    virtual const GenerationConfig& config() const = 0;

    // Appends a candidate the selector chose.
    virtual void append(int64_t token) = 0;

    // Draws a candidate from a [1, 1, vocab] logits row with the request's draft sampler, which applies the request's
    // logit processors and records the distribution of a sampled candidate, and appends it. nullopt when the sampler
    // appended no token, which ends the block.
    virtual std::optional<int64_t> sample(const ov::Tensor& row) = 0;
};

// Drafts the candidates of the blocks a draft step forwarded, from the draft logits of their rows. A greedy request
// whose logit processors keep the argmax takes the best token of every row. Every other request draws its candidates
// with its draft sampler, so that its penalties and filters apply and the target can verify a sampled candidate
// against the distribution it was drawn from.
class CandidateSelector {
public:
    // logits [rows, 1, vocab] hold the rows of every block at block.first_row. Drafts at most block.num_candidates
    // tokens of every block with its drafter.
    void select(const std::vector<DraftBlock>& blocks,
                const ov::Tensor& logits,
                const std::vector<BlockDrafter*>& drafters);

private:
    void draft_rows(const std::vector<DraftBlock>& blocks,
                    const ov::Tensor& logits,
                    const std::vector<BlockDrafter*>& drafters);
    // [1, 1, vocab] copy of a logits row, which the draft sampler may rewrite
    ov::Tensor& backbone_row(const float* logits, size_t vocab_size);

    // the row a candidate is drawn from by a draft sampler
    ov::Tensor m_row;
};

}  // namespace ov::genai::dflash_cb
