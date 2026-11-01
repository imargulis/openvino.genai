// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov::genai::dflash_cb {

namespace {

size_t row_width(const ov::Tensor& rows) {
    OPENVINO_ASSERT(rows && rows.get_element_type() == ov::element::f32 && rows.is_continuous() &&
                        !rows.get_shape().empty() && rows.get_shape().back() > 0,
                    "DFlash candidate selection requires contiguous FP32 draft outputs with nonempty rows.");
    return rows.get_shape().back();
}

const float* row_data(const ov::Tensor& rows, size_t row) {
    return rows.data<const float>() + row * row_width(rows);
}

int64_t argmax(const float* row, size_t vocab_size) {
    return std::max_element(row, row + vocab_size) - row;
}

}  // namespace

void CandidateSelector::select(const std::vector<DraftBlock>& blocks,
                               const ov::Tensor& logits,
                               const std::vector<BlockDrafter*>& drafters) {
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
    draft_rows(blocks, logits, drafters);
}

ov::Tensor& CandidateSelector::backbone_row(const float* logits, size_t vocab_size) {
    if (!m_row || m_row.get_size() != vocab_size) {
        m_row = ov::Tensor(ov::element::f32, {1, 1, vocab_size});
    }
    std::copy_n(logits, vocab_size, m_row.data<float>());
    return m_row;
}

void CandidateSelector::draft_rows(const std::vector<DraftBlock>& blocks,
                                   const ov::Tensor& logits,
                                   const std::vector<BlockDrafter*>& drafters) {
    const size_t vocab_size = row_width(logits);
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        auto& drafter = *drafters[idx];
        const bool takes_argmax = logit_processors_keep_argmax(drafter.config());
        for (size_t position = 0; position < blocks[idx].num_candidates; ++position) {
            const float* row = row_data(logits, blocks[idx].first_row + position);
            if (takes_argmax) {
                drafter.append(argmax(row, vocab_size));
            } else if (!drafter.sample(backbone_row(row, vocab_size))) {
                break;
            }
        }
    }
}

}  // namespace ov::genai::dflash_cb
