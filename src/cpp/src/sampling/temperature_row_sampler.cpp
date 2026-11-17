// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sampling/temperature_row_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/runtime/properties.hpp"
#include "sampling/logit_transformers.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {

constexpr size_t MAX_POOLED_BUFFERS = 128;
// A block of the draw is the largest power of two from MIN_DRAW_BLOCK to MAX_DRAW_BLOCK that divides the vocabulary,
// so the softmax rows need no padding, else MAX_DRAW_BLOCK.
constexpr size_t MAX_DRAW_BLOCK = 1024;
constexpr size_t MIN_DRAW_BLOCK = 64;

constexpr const char* LOGITS = "logits";
constexpr const char* LOGIT_SCALE = "logit_scale";
constexpr const char* BIAS = "bias";
constexpr const char* BIAS_SCALE = "bias_scale";
constexpr const char* PROBABILITIES = "probabilities";
constexpr const char* BLOCK_SUMS = "block_sums";

class BufferPool {
public:
    std::vector<float> take(size_t size) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            while (!m_free.empty()) {
                std::vector<float> buffer = std::move(m_free.back());
                m_free.pop_back();
                if (buffer.size() == size) {
                    return buffer;
                }
            }
        }
        return std::vector<float>(size);
    }

    void give(std::vector<float>&& buffer) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_free.size() < MAX_POOLED_BUFFERS) {
            m_free.push_back(std::move(buffer));
        }
    }

private:
    std::mutex m_mutex;
    std::vector<std::vector<float>> m_free;
};

// Never destroyed: a proposal can be released during static destruction.
BufferPool& buffer_pool() {
    static auto* pool = new BufferPool;
    return *pool;
}

DraftProposalPtr make_pooled_proposal(std::vector<float>&& probabilities) {
    auto* proposal = new DraftProposal;
    proposal->probabilities = std::move(probabilities);
    return DraftProposalPtr(proposal, [](const DraftProposal* released) {
        auto* owned = const_cast<DraftProposal*>(released);
        buffer_pool().give(std::move(owned->probabilities));
        delete owned;
    });
}

size_t draw_block_size(size_t vocab_size) {
    for (size_t block = MAX_DRAW_BLOCK; block >= MIN_DRAW_BLOCK; block /= 2) {
        if (vocab_size % block == 0) {
            return block;
        }
    }
    return MAX_DRAW_BLOCK;
}

std::shared_ptr<ov::op::v0::Parameter> make_parameter(const char* name, const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
    return parameter;
}

std::shared_ptr<ov::op::v0::Result> make_result(const ov::Output<ov::Node>& output, const char* name) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    result->output(0).set_names({name});
    result->set_friendly_name(name);
    return result;
}

// softmax((multiplier * logits + bias) / temperature) as softmax(logit_scale * logits + bias_scale * bias), with
// logit_scale = multiplier / temperature and bias_scale = 1 / temperature, and the sums of its blocks.
std::shared_ptr<ov::Model> make_softmax_model(size_t vocab_size, bool with_bias, size_t block) {
    const ov::Shape row{1, vocab_size};
    const ov::Shape scalar{1, 1};
    auto logits = make_parameter(LOGITS, row);
    auto logit_scale = make_parameter(LOGIT_SCALE, scalar);
    ov::ParameterVector parameters{logits, logit_scale};
    ov::Output<ov::Node> values = std::make_shared<ov::op::v1::Multiply>(logits, logit_scale);
    if (with_bias) {
        auto bias = make_parameter(BIAS, row);
        auto bias_scale = make_parameter(BIAS_SCALE, scalar);
        parameters.push_back(bias);
        parameters.push_back(bias_scale);
        values = std::make_shared<ov::op::v1::Add>(values, std::make_shared<ov::op::v1::Multiply>(bias, bias_scale));
    }
    const auto probabilities = std::make_shared<ov::op::v8::Softmax>(values, -1);

    const size_t num_blocks = (vocab_size + block - 1) / block;
    ov::Output<ov::Node> padded = probabilities;
    if (num_blocks * block != vocab_size) {
        const auto padding = static_cast<int64_t>(num_blocks * block - vocab_size);
        padded = std::make_shared<ov::op::v1::Pad>(
            probabilities,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, padding}),
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f}),
            ov::op::PadMode::CONSTANT);
    }
    const std::vector<int64_t> blocks_shape{1, static_cast<int64_t>(num_blocks), static_cast<int64_t>(block)};
    const auto blocks = std::make_shared<ov::op::v1::Reshape>(
        padded,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, blocks_shape),
        false);
    const auto block_sums = std::make_shared<ov::op::v1::ReduceSum>(
        blocks,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
        false);
    return std::make_shared<ov::Model>(
        ov::ResultVector{make_result(probabilities, PROBABILITIES), make_result(block_sums, BLOCK_SUMS)},
        parameters,
        "dflash_temperature_softmax");
}

// The index at threshold in [0, total) of the cumulative probabilities; a threshold that rounds up to the total takes
// the last probability with mass, as SamplingDistribution::draw() does.
size_t draw_from_blocks(const float* probabilities,
                        size_t size,
                        size_t block_size,
                        const std::vector<double>& block_sums,
                        double threshold) {
    size_t block = 0;
    for (size_t idx = 0; idx < block_sums.size(); ++idx) {
        if (!(block_sums[idx] > 0.0)) {
            continue;
        }
        block = idx;
        if (threshold < block_sums[idx]) {
            break;
        }
        threshold -= block_sums[idx];
    }
    const size_t begin = block * block_size;
    const size_t end = std::min(size, begin + block_size);
    double cumulative = 0.0;
    size_t selected = begin;
    for (size_t i = begin; i < end; ++i) {
        if (!(probabilities[i] > 0.0f)) {
            continue;
        }
        selected = i;
        cumulative += probabilities[i];
        if (cumulative > threshold) {
            break;
        }
    }
    return selected;
}

int64_t first_biased_argmax(const float* logits, const float* bias, size_t size, float multiplier) {
    size_t best = 0;
    float best_value = logits[0] * multiplier + bias[0];
    for (size_t i = 1; i < size; ++i) {
        const float value = logits[i] * multiplier + bias[i];
        if (value > best_value) {
            best = i;
            best_value = value;
        }
    }
    return static_cast<int64_t>(best);
}

}  // namespace

TemperatureRowSampler::TemperatureRowSampler(size_t vocab_size, bool with_bias, size_t num_streams)
    : m_vocab_size(vocab_size),
      m_with_bias(with_bias),
      m_block(draw_block_size(vocab_size)) {
    OPENVINO_ASSERT(vocab_size > 0 && num_streams > 0, "The temperature row sampler needs rows and streams.");
    const auto streams = static_cast<int32_t>(num_streams);
    // f32: the CPU plugin defaults to bf16 or f16 inference on hosts that support them. No pinning: the sampler's
    // threads and the GPU pipeline's share the cores.
    m_compiled = utils::singleton_core().compile_model(make_softmax_model(vocab_size, with_bias, m_block),
                                                       "CPU",
                                                       ov::AnyMap{ov::hint::inference_precision(ov::element::f32),
                                                                  ov::hint::enable_cpu_pinning(false),
                                                                  ov::num_streams(ov::streams::Num(streams)),
                                                                  ov::inference_num_threads(streams)});
}

ov::InferRequest TemperatureRowSampler::take_request() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_free_requests.empty()) {
            ov::InferRequest request = std::move(m_free_requests.back());
            m_free_requests.pop_back();
            return request;
        }
    }
    return m_compiled.create_infer_request();
}

void TemperatureRowSampler::give_request(ov::InferRequest&& request) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_free_requests.push_back(std::move(request));
}

std::optional<RowDraw> TemperatureRowSampler::sample(const float* logits,
                                                     const float* bias,
                                                     float multiplier,
                                                     float temperature,
                                                     double u) {
    OPENVINO_ASSERT((bias != nullptr) == m_with_bias,
                    "The temperature row sampler was compiled ",
                    m_with_bias ? "with" : "without",
                    " a bias row.");
    const ov::Shape row{1, m_vocab_size};
    std::vector<float> probabilities = buffer_pool().take(m_vocab_size);
    ov::InferRequest request = take_request();
    // const_cast is safe: the softmax only reads its inputs.
    request.set_tensor(LOGITS, ov::Tensor(ov::element::f32, row, const_cast<float*>(logits)));
    request.get_tensor(LOGIT_SCALE).data<float>()[0] = multiplier / temperature;
    if (m_with_bias) {
        request.set_tensor(BIAS, ov::Tensor(ov::element::f32, row, const_cast<float*>(bias)));
        request.get_tensor(BIAS_SCALE).data<float>()[0] = 1.0f / temperature;
    }
    request.set_tensor(PROBABILITIES, ov::Tensor(ov::element::f32, row, probabilities.data()));
    request.infer();
    thread_local std::vector<double> block_sums;
    const ov::Tensor sums = request.get_tensor(BLOCK_SUMS);
    block_sums.assign(sums.data<const float>(), sums.data<const float>() + sums.get_size());
    give_request(std::move(request));

    const double total = std::accumulate(block_sums.begin(), block_sums.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total)) {
        buffer_pool().give(std::move(probabilities));
        return std::nullopt;
    }
    const size_t token = draw_from_blocks(probabilities.data(), m_vocab_size, m_block, block_sums, u * total);
    // The softmax normalizes in single precision; the proposal is the distribution the token was drawn from.
    const auto inverse_total = static_cast<float>(1.0 / total);
    for (float& probability : probabilities) {
        probability *= inverse_total;
    }
    RowDraw draw;
    draw.token = static_cast<int64_t>(token);
    draw.log_prob = std::log(probabilities[token]);
    draw.proposal = make_pooled_proposal(std::move(probabilities));
    return draw;
}

int64_t row_argmax(const float* logits, const float* bias, size_t size, float multiplier) {
    if (!bias) {
        // multiplier > 0 keeps the argmax of the logits
        return static_cast<int64_t>(std::max_element(logits, logits + size) - logits);
    }
    return first_biased_argmax(logits, bias, size, multiplier);
}

std::optional<RowDraw> sample_temperature_candidates(const float* values,
                                                     const int32_t* ids,
                                                     size_t k,
                                                     float temperature,
                                                     double u) {
    float max = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < k; ++i) {
        max = std::max(max, values[i]);
    }
    if (!(max > -std::numeric_limits<float>::infinity())) {
        return std::nullopt;
    }
    auto proposal = std::make_shared<DraftProposal>();
    proposal->token_ids.resize(k);
    proposal->probabilities.resize(k);
    const float scale = 1.0f / temperature;
    double total = 0.0;
    for (size_t i = 0; i < k; ++i) {
        proposal->token_ids[i] = ids[i];
        proposal->probabilities[i] = exp_flush_tail((values[i] - max) * scale);
        total += proposal->probabilities[i];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        return std::nullopt;
    }
    const double threshold = u * total;
    double cumulative = 0.0;
    size_t selected = 0;
    for (size_t i = 0; i < k; ++i) {
        if (!(proposal->probabilities[i] > 0.0f)) {
            continue;
        }
        selected = i;
        cumulative += proposal->probabilities[i];
        if (cumulative > threshold) {
            break;
        }
    }
    const auto inverse_total = static_cast<float>(1.0 / total);
    for (auto& probability : proposal->probabilities) {
        probability *= inverse_total;
    }
    RowDraw draw;
    draw.token = proposal->token_ids[selected];
    draw.log_prob = std::log(proposal->probabilities[selected]);
    draw.proposal = std::move(proposal);
    return draw;
}

}  // namespace ov::genai
