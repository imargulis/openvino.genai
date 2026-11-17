// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "sampling/draft_proposal.hpp"

namespace ov::genai {

struct RowDraw {
    int64_t token = 0;
    float log_prob = 0.0f;
    DraftProposalPtr proposal;
};

// Draws from softmax((multiplier * logits + bias) / temperature) over full vocabulary rows the way the sampler draws a
// row whose logit processors only apply the temperature. The softmax and the sums of its blocks run on the OpenVINO
// CPU plugin, which picks its kernels for the host's instruction set; the draw scans the block sums, then one block.
// The probabilities go to a pooled buffer that the dense proposal returns to the pool when it is released.
class TemperatureRowSampler {
public:
    // with_bias: every row takes a bias row. num_streams: the rows the CPU plugin softmaxes at once, one thread each.
    TemperatureRowSampler(size_t vocab_size, bool with_bias, size_t num_streams);

    size_t vocab_size() const {
        return m_vocab_size;
    }

    bool with_bias() const {
        return m_with_bias;
    }

    // u is the uniform in [0, 1) that SamplingDistribution::draw() takes from the request's rng. bias is null unless
    // with_bias(); multiplier > 0. nullopt when the row has no mass. Callable concurrently: each call takes an infer
    // request of its own.
    std::optional<RowDraw> sample(const float* logits,
                                  const float* bias,
                                  float multiplier,
                                  float temperature,
                                  double u);

private:
    ov::InferRequest take_request();
    void give_request(ov::InferRequest&& request);

    size_t m_vocab_size = 0;
    bool m_with_bias = false;
    // the probabilities per block sum
    size_t m_block = 0;
    ov::CompiledModel m_compiled;
    std::mutex m_mutex;
    std::vector<ov::InferRequest> m_free_requests;
};

// The first index of the largest multiplier * logits + bias, multiplier > 0. bias may be null.
int64_t row_argmax(const float* logits, const float* bias, size_t size, float multiplier);

// Draws from softmax(values / temperature) over k candidates, values[i] the logit of token ids[i]. The proposal is
// sparse over the k ids. nullopt when the candidates have no mass.
std::optional<RowDraw> sample_temperature_candidates(const float* values,
                                                     const int32_t* ids,
                                                     size_t k,
                                                     float temperature,
                                                     double u);

}  // namespace ov::genai
