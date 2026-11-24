// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sampling/temperature_row_sampler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/runtime/properties.hpp"
#include "sampling/logit_transformers.hpp"
#include "utils.hpp"

namespace ov::genai {

// Buffers of one size for the dense proposals, which give theirs back when they are released.
class ProposalBufferPool {
public:
    explicit ProposalBufferPool(size_t size) : m_size(size) {}

    std::vector<float> take() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_free.empty()) {
                std::vector<float> buffer = std::move(m_free.back());
                m_free.pop_back();
                return buffer;
            }
        }
        return std::vector<float>(m_size);
    }

    void give(std::vector<float>&& buffer) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_free.size() < m_capacity) {
            m_free.push_back(std::move(buffer));
        }
    }

    void set_capacity(size_t capacity) {
        std::vector<std::vector<float>> dropped;
        std::lock_guard<std::mutex> lock(m_mutex);
        m_capacity = capacity;
        while (m_free.size() > capacity) {
            dropped.push_back(std::move(m_free.back()));
            m_free.pop_back();
        }
    }

private:
    const size_t m_size;
    std::mutex m_mutex;
    size_t m_capacity = 0;
    std::vector<std::vector<float>> m_free;
};

namespace {

// The draw sums blocks of the probabilities, then scans the block sums and one block.
constexpr size_t DRAW_BLOCK = 512;

constexpr const char* LOGITS = "logits";
constexpr const char* LOGIT_SCALE = "logit_scale";
constexpr const char* BIAS = "bias";
constexpr const char* BIAS_SCALE = "bias_scale";
constexpr const char* PROBABILITIES = "probabilities";

DraftProposalPtr make_pooled_proposal(std::shared_ptr<ProposalBufferPool> pool, std::vector<float>&& probabilities) {
    auto* proposal = new DraftProposal;
    proposal->probabilities = std::move(probabilities);
    return DraftProposalPtr(proposal, [pool = std::move(pool)](const DraftProposal* released) {
        auto* owned = const_cast<DraftProposal*>(released);
        pool->give(std::move(owned->probabilities));
        delete owned;
    });
}

std::shared_ptr<ov::op::v0::Parameter> make_parameter(const char* name, const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
    return parameter;
}

// softmax((multiplier * logits + bias) / temperature) as softmax(logit_scale * logits + bias_scale * bias), with
// logit_scale = multiplier / temperature and bias_scale = 1 / temperature.
std::shared_ptr<ov::Model> make_softmax_model(size_t vocab_size, bool with_bias) {
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
    auto result = std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v8::Softmax>(values, -1));
    result->output(0).set_names({PROBABILITIES});
    result->set_friendly_name(PROBABILITIES);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, parameters, "dflash_temperature_softmax");
}

// The sum of the weights in double. Eight running sums let the additions overlap, which one running sum serializes.
double sum_of(const float* weights, size_t size) {
    std::array<double, 8> sums{};
    size_t i = 0;
    for (; i + sums.size() <= size; i += sums.size()) {
        for (size_t lane = 0; lane < sums.size(); ++lane) {
            sums[lane] += weights[i + lane];
        }
    }
    double sum = 0.0;
    for (; i < size; ++i) {
        sum += weights[i];
    }
    for (const double lane_sum : sums) {
        sum += lane_sum;
    }
    return sum;
}

}  // namespace

std::optional<WeightedDraw> draw_weighted(const float* weights, size_t size, double u) {
    std::vector<double> block_sums((size + DRAW_BLOCK - 1) / DRAW_BLOCK);
    for (size_t block = 0; block < block_sums.size(); ++block) {
        const size_t begin = block * DRAW_BLOCK;
        block_sums[block] = sum_of(weights + begin, std::min(DRAW_BLOCK, size - begin));
    }
    const double total = std::accumulate(block_sums.begin(), block_sums.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total)) {
        return std::nullopt;
    }
    double threshold = u * total;
    size_t block = block_sums.size();
    size_t last_block_with_mass = 0;
    for (size_t idx = 0; idx < block_sums.size(); ++idx) {
        if (!(block_sums[idx] > 0.0)) {
            continue;
        }
        last_block_with_mass = idx;
        if (threshold < block_sums[idx]) {
            block = idx;
            break;
        }
        threshold -= block_sums[idx];
    }
    if (block == block_sums.size()) {
        block = last_block_with_mass;
        threshold = std::numeric_limits<double>::infinity();
    }
    // A threshold past the running sum of the block, which can differ from its block sum in the last bits, takes the
    // block's last weight with mass.
    const size_t begin = block * DRAW_BLOCK;
    const size_t end = std::min(size, begin + DRAW_BLOCK);
    double cumulative = 0.0;
    size_t selected = begin;
    for (size_t i = begin; i < end; ++i) {
        if (!(weights[i] > 0.0f)) {
            continue;
        }
        selected = i;
        cumulative += weights[i];
        if (cumulative > threshold) {
            break;
        }
    }
    return WeightedDraw{selected, total};
}

// The requests that a sample() call takes, at least one and at most one per row. A call that holds none waits for
// one, and a call that holds one never waits for more. A call that unwinds drops its requests instead of giving them
// back: destroying a request waits for its inference without throwing, so no inference writes to a freed buffer.
class TemperatureRowSampler::Lease {
public:
    Lease(TemperatureRowSampler& sampler, size_t max_requests) : m_sampler(sampler) {
        std::unique_lock<std::mutex> lock(sampler.m_mutex);
        sampler.m_request_returned.wait(lock, [&] {
            return !sampler.m_free_requests.empty() || sampler.m_num_requests < sampler.m_num_streams;
        });
        while (m_requests.size() < max_requests) {
            if (!sampler.m_free_requests.empty()) {
                m_requests.push_back(std::move(sampler.m_free_requests.back()));
                sampler.m_free_requests.pop_back();
            } else if (sampler.m_num_requests < sampler.m_num_streams) {
                // replaces a request that an unwinding call dropped
                m_requests.push_back(sampler.m_compiled.create_infer_request());
                ++sampler.m_num_requests;
            } else {
                break;
            }
        }
    }

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    ~Lease() {
        if (m_requests.empty()) {
            return;
        }
        const size_t num_dropped = m_requests.size();
        m_requests.clear();
        std::lock_guard<std::mutex> lock(m_sampler.m_mutex);
        m_sampler.m_num_requests -= num_dropped;
        m_sampler.m_request_returned.notify_all();
    }

    size_t size() const {
        return m_requests.size();
    }

    ov::InferRequest& operator[](size_t idx) {
        return m_requests[idx];
    }

    // after every inference of the call finished
    void give_back() {
        std::lock_guard<std::mutex> lock(m_sampler.m_mutex);
        for (auto& request : m_requests) {
            m_sampler.m_free_requests.push_back(std::move(request));
        }
        m_requests.clear();
        m_sampler.m_request_returned.notify_all();
    }

private:
    TemperatureRowSampler& m_sampler;
    std::vector<ov::InferRequest> m_requests;
};

TemperatureRowSampler::TemperatureRowSampler(size_t vocab_size, bool with_bias, size_t num_streams)
    : m_vocab_size(vocab_size),
      m_with_bias(with_bias),
      m_buffers(std::make_shared<ProposalBufferPool>(vocab_size)) {
    OPENVINO_ASSERT(vocab_size > 0 && num_streams > 0, "The temperature row sampler needs rows and streams.");
    const auto streams = static_cast<int32_t>(num_streams);
    // f32: the CPU plugin defaults to bf16 or f16 inference on hosts that support them. Performance cores only: the
    // rows of a block finish with the slowest of them, and an efficient core takes about twice as long for a row. No
    // pinning: the sampler's threads and the GPU pipeline's share the cores.
    m_compiled = utils::singleton_core().compile_model(
        make_softmax_model(vocab_size, with_bias),
        "CPU",
        ov::AnyMap{ov::hint::inference_precision(ov::element::f32),
                   ov::hint::enable_cpu_pinning(false),
                   ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY),
                   ov::num_streams(ov::streams::Num(streams)),
                   ov::inference_num_threads(streams)});
    const int32_t compiled_streams = m_compiled.get_property(ov::num_streams);
    m_num_streams = static_cast<size_t>(std::max<int32_t>(1, compiled_streams));
    for (size_t idx = 0; idx < m_num_streams; ++idx) {
        m_free_requests.push_back(m_compiled.create_infer_request());
    }
    m_num_requests = m_num_streams;
}

void TemperatureRowSampler::keep_buffers(size_t num_buffers) {
    m_buffers->set_capacity(num_buffers);
}

std::vector<std::optional<RowDraw>> TemperatureRowSampler::sample(const std::vector<Row>& rows,
                                                                  float multiplier,
                                                                  float temperature) {
    for (const auto& row : rows) {
        OPENVINO_ASSERT(row.logits && (row.bias != nullptr) == m_with_bias,
                        "The temperature row sampler was compiled ",
                        m_with_bias ? "with" : "without",
                        " a bias row.");
    }
    std::vector<std::optional<RowDraw>> draws(rows.size());
    if (rows.empty()) {
        return draws;
    }
    const ov::Shape row_shape{1, m_vocab_size};
    // the buffer each request softmaxes into, which outlives the lease's inferences
    std::vector<std::vector<float>> probabilities;
    Lease lease(*this, rows.size());
    probabilities.resize(lease.size());
    // row r runs on request r % lease.size(), so the rows finish in their order
    const auto start = [&](size_t row) {
        const size_t slot = row % lease.size();
        probabilities[slot] = m_buffers->take();
        auto& request = lease[slot];
        // const_cast is safe: the softmax only reads its inputs.
        request.set_tensor(LOGITS, ov::Tensor(ov::element::f32, row_shape, const_cast<float*>(rows[row].logits)));
        request.get_tensor(LOGIT_SCALE).data<float>()[0] = multiplier / temperature;
        if (m_with_bias) {
            request.set_tensor(BIAS, ov::Tensor(ov::element::f32, row_shape, const_cast<float*>(rows[row].bias)));
            request.get_tensor(BIAS_SCALE).data<float>()[0] = 1.0f / temperature;
        }
        request.set_tensor(PROBABILITIES, ov::Tensor(ov::element::f32, row_shape, probabilities[slot].data()));
        request.start_async();
    };
    for (size_t row = 0; row < std::min(lease.size(), rows.size()); ++row) {
        start(row);
    }
    for (size_t row = 0; row < rows.size(); ++row) {
        const size_t slot = row % lease.size();
        lease[slot].wait();
        std::vector<float> row_probabilities = std::move(probabilities[slot]);
        if (row + lease.size() < rows.size()) {
            start(row + lease.size());
        }
        draws[row] = draw(std::move(row_probabilities), rows[row].u);
    }
    lease.give_back();
    return draws;
}

std::optional<RowDraw> TemperatureRowSampler::draw(std::vector<float>&& probabilities, double u) {
    const auto drawn = draw_weighted(probabilities.data(), probabilities.size(), u);
    if (!drawn) {
        m_buffers->give(std::move(probabilities));
        return std::nullopt;
    }
    // the proposal is the distribution the token was drawn from
    const double inverse_total = 1.0 / drawn->total;
    for (float& probability : probabilities) {
        probability = static_cast<float>(probability * inverse_total);
    }
    RowDraw draw;
    draw.token = static_cast<int64_t>(drawn->index);
    draw.log_prob = std::log(probabilities[drawn->index]);
    draw.proposal = make_pooled_proposal(m_buffers, std::move(probabilities));
    return draw;
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
    const double inverse_total = 1.0 / total;
    for (auto& probability : proposal->probabilities) {
        probability = static_cast<float>(probability * inverse_total);
    }
    RowDraw draw;
    draw.token = proposal->token_ids[selected];
    draw.log_prob = std::log(proposal->probabilities[selected]);
    draw.proposal = std::move(proposal);
    return draw;
}

}  // namespace ov::genai
