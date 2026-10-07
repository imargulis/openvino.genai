// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "sampling/sampler.hpp"
#include "sequence_group.hpp"

TEST(RejectionSampling, UsesASeparateDeterministicSamplingStream) {
    constexpr size_t generation_seed = 42;
    EXPECT_NE(ov::genai::detail::proposal_rng_seed(generation_seed),
              static_cast<std::mt19937::result_type>(generation_seed));
}

TEST(RejectionSampling, KeepsSparseProposalRowsAlignedWithSequenceTokens) {
    auto sequence = ov::genai::Sequence::create(1);
    sequence->append_token(10, -0.2f);
    sequence->append_token(20, -0.3f);
    ov::genai::DraftProposal proposal{{20, 21}, {0.7f, 0.3f}};
    sequence->set_draft_proposal(1, proposal);

    ASSERT_TRUE(sequence->get_draft_proposal(0).empty());
    ASSERT_EQ(sequence->get_draft_proposal(1).token_ids,
              (std::vector<int64_t>{20, 21}));
    sequence->clear_draft_proposal(1);
    ASSERT_TRUE(sequence->get_draft_proposal(1).empty());
    sequence->remove_last_tokens(1);
    ASSERT_EQ(sequence->get_generated_len(), 1);
}

TEST(RejectionSampling, SamplesExactResidualOutsideRejectedProposalMass) {
    ov::genai::DraftProposal proposal{{1}, {1.0f}};
    ASSERT_FLOAT_EQ(ov::genai::detail::proposal_probability(proposal, 1), 1.0f);
    ASSERT_FLOAT_EQ(ov::genai::detail::proposal_probability(proposal, 0), 0.0f);

    std::mt19937 rng(0);
    const std::vector<float> target_probabilities{0.25f, 0.75f};
    const auto sampled =
        ov::genai::detail::sample_residual_distribution(target_probabilities, proposal, rng);
    ASSERT_EQ(sampled.m_index, 0);
    ASSERT_NEAR(std::exp(sampled.m_log_prob), 0.25f, 1e-6f);
}

TEST(RejectionSampling, SupportsDenseDraftDistributions) {
    ov::genai::DraftProposal proposal{{}, {0.6f, 0.4f, 0.0f}};

    ASSERT_TRUE(proposal.is_dense());
    EXPECT_NO_THROW(ov::genai::detail::validate_draft_proposal(proposal, 3));
    ASSERT_FLOAT_EQ(ov::genai::detail::proposal_probability(proposal, 0), 0.6f);
    ASSERT_FLOAT_EQ(ov::genai::detail::proposal_probability(proposal, 2), 0.0f);

    std::mt19937 rng(0);
    const std::vector<float> target_probabilities{0.1f, 0.2f, 0.7f};
    const auto sampled =
        ov::genai::detail::sample_residual_distribution(target_probabilities, proposal, rng);
    ASSERT_EQ(sampled.m_index, 2);
}

TEST(RejectionSampling, RejectsInvalidDistributions) {
    const ov::genai::DraftProposal duplicate_ids{{0, 0}, {0.5f, 0.5f}};
    const ov::genai::DraftProposal incomplete_mass{{0}, {0.75f}};
    EXPECT_THROW(ov::genai::detail::validate_draft_proposal(duplicate_ids, 2), ov::Exception);
    EXPECT_THROW(ov::genai::detail::validate_draft_proposal(incomplete_mass, 2), ov::Exception);
}

TEST(RejectionSampling, UsesUniformFallbackForInvalidSamplingLogits) {
    float logits_data[] = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    ov::genai::Logits logits(logits_data, 2);

    const auto probabilities = ov::genai::detail::materialize_sampling_probabilities(logits, 2);

    ASSERT_EQ(probabilities.size(), 2);
    EXPECT_FLOAT_EQ(probabilities[0], 0.5f);
    EXPECT_FLOAT_EQ(probabilities[1], 0.5f);
}

TEST(RejectionSampling, CapturesTheDistributionUsedToSampleDraftTokens) {
    ov::genai::GenerationConfig config;
    config.do_sample = true;
    config.max_new_tokens = 8;
    auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(1, ov::genai::TokenIds{0}, config);
    sequence_group->update_processed_tokens_num(1);
    sequence_group->schedule_tokens(1);

    ov::Tensor logits(ov::element::f32, ov::Shape{1, 1, 3});
    const std::vector<float> values{0.0f, 1.0f, 2.0f};
    std::copy(values.begin(), values.end(), logits.data<float>());
    ov::genai::Sampler sampler;

    sampler.sample({sequence_group}, logits, false, true);

    const auto sequence = (*sequence_group)[0];
    ASSERT_EQ(sequence->get_generated_len(), 1);
    const auto& proposal = sequence->get_draft_proposal(0);
    ASSERT_TRUE(proposal.is_dense());
    EXPECT_NO_THROW(ov::genai::detail::validate_draft_proposal(proposal, 3));
    EXPECT_GT(proposal.probabilities[2], proposal.probabilities[1]);
    EXPECT_GT(proposal.probabilities[1], proposal.probabilities[0]);
}

TEST(RejectionSampling, RejectsSampledSpeculationWithoutCompleteProposalRows) {
    ov::genai::GenerationConfig config;
    config.do_sample = true;
    config.max_new_tokens = 8;
    auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(1, ov::genai::TokenIds{0}, config);
    (*sequence_group)[0]->append_token(1, 0.0f);
    sequence_group->update_processed_tokens_num(1);
    sequence_group->schedule_tokens(2);
    sequence_group->set_num_validated_tokens(1);

    ov::Tensor logits(ov::element::f32, ov::Shape{1, 2, 3});
    const std::vector<float> values{0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f};
    std::copy(values.begin(), values.end(), logits.data<float>());
    ov::genai::Sampler sampler;

    EXPECT_THROW(sampler.sample({sequence_group}, logits, true, false), ov::Exception);
}

TEST(RejectionSampling, AppliesExactProposalAcceptanceRatio) {
    std::mt19937 rng(0);
    ASSERT_TRUE(ov::genai::detail::accept_draft_token(0.5f, 0.25f, rng));
    ASSERT_FALSE(ov::genai::detail::accept_draft_token(0.0f, 1.0f, rng));
}

TEST(RejectionSampling, AcceptsAtTheExpectedRate) {
    std::mt19937 rng(0);
    size_t accepted = 0;
    constexpr size_t trials = 100'000;
    for (size_t trial = 0; trial < trials; ++trial) {
        accepted += ov::genai::detail::accept_draft_token(0.2f, 0.5f, rng);
    }

    EXPECT_NEAR(static_cast<float>(accepted) / trials, 0.4f, 0.01f);
}

TEST(RejectionSampling, RecoversTheTargetDistribution) {
    constexpr std::array<float, 3> target{0.1f, 0.2f, 0.7f};
    const ov::genai::DraftProposal proposal{{}, {0.6f, 0.4f, 0.0f}};
    std::mt19937 proposal_rng(ov::genai::detail::proposal_rng_seed(42));
    std::mt19937 target_rng(42);
    const std::vector<float> target_probabilities(target.begin(), target.end());
    std::discrete_distribution<size_t> sample_proposal(proposal.probabilities.begin(),
                                                        proposal.probabilities.end());
    std::array<size_t, target.size()> samples{};
    constexpr size_t trials = 100'000;

    for (size_t trial = 0; trial < trials; ++trial) {
        const size_t draft_token = sample_proposal(proposal_rng);
        if (ov::genai::detail::accept_draft_token(target[draft_token],
                                                   proposal.probabilities[draft_token],
                                                   target_rng)) {
            ++samples[draft_token];
        } else {
            const auto residual_token =
                ov::genai::detail::sample_residual_distribution(target_probabilities, proposal, target_rng);
            ++samples[static_cast<size_t>(residual_token.m_index)];
        }
    }

    for (size_t token = 0; token < target.size(); ++token) {
        EXPECT_NEAR(static_cast<float>(samples[token]) / trials, target[token], 0.01f);
    }
}
