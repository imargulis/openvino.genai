// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
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

class ProposalBufferPool;

// Draws from softmax((multiplier * logits + bias) / temperature) over full vocabulary rows the way the sampler draws a
// row whose logit processors only apply the temperature. The softmax runs on the OpenVINO CPU plugin, which picks its
// kernels for the host's instruction set, into pooled buffers that the dense proposals give back when they are
// released. A row's token is drawn from the proposal it records, to double precision.
class TemperatureRowSampler {
public:
    struct Row {
        const float* logits = nullptr;
        // null unless with_bias()
        const float* bias = nullptr;
        // the uniform in [0, 1) that SamplingDistribution::draw() takes from the request's rng
        double u = 0.0;
    };

    // with_bias: every row takes a bias row. num_streams: the CPU plugin streams, which softmax as many rows at once;
    // the plugin keeps at most one per performance core.
    TemperatureRowSampler(size_t vocab_size, bool with_bias, size_t num_streams);

    size_t vocab_size() const {
        return m_vocab_size;
    }

    bool with_bias() const {
        return m_with_bias;
    }

    // the rows the CPU plugin softmaxes at once
    size_t num_streams() const {
        return m_num_streams;
    }

    // Draws every row from its uniform, with multiplier > 0, on as many streams as are free; nullopt for a row without
    // mass. Callable concurrently.
    std::vector<std::optional<RowDraw>> sample(const std::vector<Row>& rows, float multiplier, float temperature);

    // Keeps at most num_buffers of the buffers that released proposals give back, for the next rows.
    void keep_buffers(size_t num_buffers);

private:
    class Lease;

    std::optional<RowDraw> draw(std::vector<float>&& probabilities, double u);

    size_t m_vocab_size = 0;
    bool m_with_bias = false;
    ov::CompiledModel m_compiled;
    // shared with the proposals, which can outlive the sampler
    std::shared_ptr<ProposalBufferPool> m_buffers;
    std::mutex m_mutex;
    std::condition_variable m_request_returned;
    std::vector<ov::InferRequest> m_free_requests;
    // the requests that exist, free or leased, at most one per stream
    size_t m_num_requests = 0;
    size_t m_num_streams = 0;
};

struct WeightedDraw {
    size_t index = 0;
    double total = 0.0;
};

// The index at u of the distribution proportional to the weights, which SamplingDistribution::draw() draws with one
// running sum, from sums of blocks to double precision, and the total weight. A u at which the threshold rounds up to
// the total takes the last weight with mass. nullopt when the weights have no positive finite mass.
std::optional<WeightedDraw> draw_weighted(const float* weights, size_t size, double u);

// Draws from softmax(values / temperature) over k candidates, values[i] the logit of token ids[i]. The proposal is
// sparse over the k ids. nullopt when the candidates have no mass.
std::optional<RowDraw> sample_temperature_candidates(const float* values,
                                                     const int32_t* ids,
                                                     size_t k,
                                                     float temperature,
                                                     double u);

}  // namespace ov::genai
