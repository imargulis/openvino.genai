// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include "sampling/temperature_row_sampler.hpp"

using ov::genai::draw_weighted;
using ov::genai::RowDraw;
using ov::genai::sample_temperature_candidates;
using ov::genai::TemperatureRowSampler;

namespace {

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float MULTIPLIER = 1.25f;
constexpr float TEMPERATURE = 0.7f;
// a vocabulary over two draw blocks and a shorter third one
constexpr size_t VOCAB_SIZE = 2 * 512 + 100;

// Normal logits, every seventh one -inf as for a token outside a pruned draft vocabulary.
std::vector<float> make_logits(size_t size, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 2.0f);
    std::vector<float> logits(size);
    for (size_t i = 0; i < size; ++i) {
        logits[i] = i % 7 == 3 ? -INF : normal(rng);
    }
    return logits;
}

// softmax((MULTIPLIER * logits + bias) / TEMPERATURE) in double
std::vector<double> reference_softmax(const std::vector<float>& logits, const std::vector<float>& bias = {}) {
    std::vector<double> values(logits.size());
    double max = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        values[i] = (double(MULTIPLIER) * logits[i] + (bias.empty() ? 0.0 : bias[i])) / TEMPERATURE;
        max = std::max(max, values[i]);
    }
    double total = 0.0;
    for (double& value : values) {
        value = std::exp(value - max);
        total += value;
    }
    for (double& value : values) {
        value /= total;
    }
    return values;
}

std::optional<RowDraw> sample_row(TemperatureRowSampler& sampler,
                                  const std::vector<float>& logits,
                                  double u,
                                  const float* bias = nullptr) {
    return sampler.sample({{logits.data(), bias, u}}, MULTIPLIER, TEMPERATURE).front();
}

// The index at u that SamplingDistribution::draw() takes with one running sum.
size_t scan(const std::vector<float>& weights, double u) {
    double total = 0.0;
    for (const float weight : weights) {
        total += weight;
    }
    const double threshold = u * total;
    double cumulative = 0.0;
    size_t selected = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        if (!(weights[i] > 0.0f)) {
            continue;
        }
        selected = i;
        cumulative += weights[i];
        if (cumulative > threshold) {
            break;
        }
    }
    return selected;
}

}  // namespace

TEST(TemperatureRowSampler, DrawsTheIndexThatOneRunningSumDraws) {
    // log-normal weights over 64 draw blocks and a shorter last one, with zeros
    const size_t size = 64 * 512 + 100;
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0f, 4.0f);
    std::vector<float> weights(size);
    for (size_t i = 0; i < size; ++i) {
        weights[i] = i % 7 == 3 ? 0.0f : std::exp(normal(rng));
    }
    double total = 0.0;
    for (const float weight : weights) {
        total += weight;
    }

    // The middle of the interval of the cumulative distribution that a weight takes draws its index: at the ends of
    // the draw blocks, where block sums in single precision lost some, and at every 31st index. The weights too
    // small for their middle to differ from the interval's ends in double are left out.
    double cumulative = 0.0;
    size_t checked = 0;
    for (size_t i = 0; i < size; ++i) {
        const size_t offset = i % 512;
        if (weights[i] > 1e-12 * total && (offset < 2 || offset > 509 || i % 31 == 0)) {
            const auto drawn = draw_weighted(weights.data(), size, (cumulative + weights[i] / 2.0) / total);
            ASSERT_TRUE(drawn);
            EXPECT_EQ(drawn->index, i);
            ++checked;
        }
        cumulative += weights[i];
    }
    EXPECT_GT(checked, 500u);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (size_t draw = 0; draw < 2000; ++draw) {
        const double u = uniform(rng);
        EXPECT_EQ(draw_weighted(weights.data(), size, u)->index, scan(weights, u)) << u;
    }
    EXPECT_NEAR(draw_weighted(weights.data(), size, 0.5)->total, total, 1e-12 * total);
}

TEST(TemperatureRowSampler, AThresholdAtTheTotalTakesTheLastWeightWithMass) {
    std::vector<float> weights(3 * 512, 0.0f);
    weights[5] = 1.0f;
    weights[600] = 2.0f;
    weights[1030] = 3.0f;
    weights[1040] = 4.0f;
    EXPECT_EQ(draw_weighted(weights.data(), weights.size(), 1.0)->index, 1040u);
    EXPECT_EQ(draw_weighted(weights.data(), weights.size(), 0.0)->index, 5u);
    EXPECT_EQ(draw_weighted(weights.data(), weights.size(), 0.3)->index, 1030u);
}

TEST(TemperatureRowSampler, WeightsWithoutPositiveFiniteMassDrawNothing) {
    std::vector<float> weights(700, 0.0f);
    EXPECT_FALSE(draw_weighted(weights.data(), weights.size(), 0.5));
    weights[3] = 1.0f;
    weights[650] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(draw_weighted(weights.data(), weights.size(), 0.5));
    weights[650] = INF;
    EXPECT_FALSE(draw_weighted(weights.data(), weights.size(), 0.5));
}

TEST(TemperatureRowSampler, RecordsTheSoftmaxOfTheScaledLogitsAndTheBias) {
    const auto logits = make_logits(VOCAB_SIZE, 1);
    const auto bias = make_logits(VOCAB_SIZE, 2);
    for (const bool with_bias : {false, true}) {
        TemperatureRowSampler sampler(VOCAB_SIZE, with_bias, 1);
        const auto draw = sample_row(sampler, logits, 0.37, with_bias ? bias.data() : nullptr);
        ASSERT_TRUE(draw);
        const auto expected = with_bias ? reference_softmax(logits, bias) : reference_softmax(logits);
        const auto& probabilities = draw->proposal->probabilities;
        ASSERT_EQ(probabilities.size(), VOCAB_SIZE);
        EXPECT_TRUE(draw->proposal->token_ids.empty());
        double total = 0.0;
        for (size_t i = 0; i < VOCAB_SIZE; ++i) {
            if (expected[i] == 0.0) {
                EXPECT_EQ(probabilities[i], 0.0f) << i;
            } else if (expected[i] > 1e-30) {
                EXPECT_NEAR(probabilities[i] / expected[i], 1.0, 1e-5) << i;
            }
            total += probabilities[i];
        }
        EXPECT_NEAR(total, 1.0, 1e-6);
        EXPECT_GT(probabilities[draw->token], 0.0f);
        EXPECT_EQ(draw->log_prob, std::log(probabilities[draw->token]));
    }
}

TEST(TemperatureRowSampler, DrawsEveryTokenOfAUniformRow) {
    std::vector<float> logits(VOCAB_SIZE, 0.5f);
    std::vector<size_t> tokens;
    for (size_t i = 0; i < VOCAB_SIZE; ++i) {
        if (i % 5 == 0) {
            logits[i] = -INF;
        } else {
            tokens.push_back(i);
        }
    }
    TemperatureRowSampler sampler(VOCAB_SIZE, false, 1);
    for (size_t rank = 0; rank < tokens.size(); ++rank) {
        const auto draw = sample_row(sampler, logits, (rank + 0.5) / tokens.size());
        ASSERT_TRUE(draw);
        EXPECT_EQ(draw->token, static_cast<int64_t>(tokens[rank]));
    }
}

TEST(TemperatureRowSampler, ARowWithoutMassDrawsNothing) {
    TemperatureRowSampler sampler(VOCAB_SIZE, false, 1);
    EXPECT_FALSE(sample_row(sampler, std::vector<float>(VOCAB_SIZE, -INF), 0.5));
    auto logits = make_logits(VOCAB_SIZE, 4);
    logits[700] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(sample_row(sampler, logits, 0.5));
}

TEST(TemperatureRowSampler, RowsDrawnTogetherMatchRowsDrawnOneByOne) {
    constexpr size_t num_rows = 13;
    std::vector<std::vector<float>> logits;
    std::vector<TemperatureRowSampler::Row> rows;
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (size_t row = 0; row < num_rows; ++row) {
        logits.push_back(make_logits(VOCAB_SIZE, 10 + static_cast<uint32_t>(row)));
    }
    // a row without mass in the middle draws nothing and leaves the rows after it alone
    std::fill(logits[6].begin(), logits[6].end(), -INF);
    for (size_t row = 0; row < num_rows; ++row) {
        rows.push_back({logits[row].data(), nullptr, uniform(rng)});
    }

    TemperatureRowSampler one_stream(VOCAB_SIZE, false, 1);
    std::vector<std::optional<RowDraw>> expected;
    for (const auto& row : rows) {
        expected.push_back(one_stream.sample({row}, MULTIPLIER, TEMPERATURE).front());
    }
    TemperatureRowSampler streams(VOCAB_SIZE, false, 4);
    const auto check = [&](const std::vector<std::optional<RowDraw>>& draws) {
        ASSERT_EQ(draws.size(), num_rows);
        for (size_t row = 0; row < num_rows; ++row) {
            ASSERT_EQ(draws[row].has_value(), expected[row].has_value()) << row;
            if (draws[row]) {
                EXPECT_EQ(draws[row]->token, expected[row]->token) << row;
                EXPECT_EQ(draws[row]->proposal->probabilities, expected[row]->proposal->probabilities) << row;
            }
        }
    };
    EXPECT_FALSE(expected[6]);
    check(streams.sample(rows, MULTIPLIER, TEMPERATURE));

    // two calls at once share the streams
    std::vector<std::optional<RowDraw>> first;
    std::vector<std::optional<RowDraw>> second;
    std::thread other([&] {
        first = streams.sample(rows, MULTIPLIER, TEMPERATURE);
    });
    second = streams.sample(rows, MULTIPLIER, TEMPERATURE);
    other.join();
    check(first);
    check(second);
}

TEST(TemperatureRowSampler, ReusesTheBuffersOfReleasedProposalsAndOutlivesItsProposals) {
    const auto logits = make_logits(VOCAB_SIZE, 6);
    auto sampler = std::make_unique<TemperatureRowSampler>(VOCAB_SIZE, false, 1);
    sampler->keep_buffers(1);
    auto draw = sample_row(*sampler, logits, 0.5);
    ASSERT_TRUE(draw);
    const float* buffer = draw->proposal->probabilities.data();
    draw.reset();
    draw = sample_row(*sampler, logits, 0.5);
    ASSERT_TRUE(draw);
    EXPECT_EQ(draw->proposal->probabilities.data(), buffer);

    const auto probabilities = draw->proposal->probabilities;
    sampler.reset();
    EXPECT_EQ(draw->proposal->probabilities, probabilities);
    draw.reset();
}

TEST(TemperatureRowSampler, DrawsEveryCandidateFromTheSparseProposalItRecords) {
    const std::vector<float> values = {1.0f, -INF, 3.0f, 0.5f, 3.0f};
    const std::vector<int32_t> ids = {40, 7, 12, 3, 99};
    const auto first = sample_temperature_candidates(values.data(), ids.data(), values.size(), TEMPERATURE, 0.5);
    ASSERT_TRUE(first);
    EXPECT_EQ(first->proposal->token_ids, (std::vector<int64_t>{40, 7, 12, 3, 99}));
    const auto& probabilities = first->proposal->probabilities;
    EXPECT_EQ(probabilities[1], 0.0f);
    double total = 0.0;
    for (const float probability : probabilities) {
        total += probability;
    }
    double cumulative = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        if (probabilities[i] > 0.0f) {
            const double u = (cumulative + probabilities[i] / 2.0) / total;
            const auto draw = sample_temperature_candidates(values.data(), ids.data(), values.size(), TEMPERATURE, u);
            ASSERT_TRUE(draw);
            EXPECT_EQ(draw->token, ids[i]);
            EXPECT_EQ(draw->log_prob, std::log(probabilities[i]));
        }
        cumulative += probabilities[i];
    }
    const std::vector<float> no_mass(3, -INF);
    EXPECT_FALSE(sample_temperature_candidates(no_mass.data(), ids.data(), no_mass.size(), TEMPERATURE, 0.5));
}
