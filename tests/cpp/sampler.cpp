// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "sampling/sampler.hpp"
#include "openvino/genai/generation_config.hpp"
#include "utils.hpp"


using namespace ov::genai;

TEST(SamplerStopTokenIdsTest, single_stop_token_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = {9};
    ASSERT_TRUE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_token_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = {7, 8, 9};
    ASSERT_TRUE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, single_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = { 10 };
    ASSERT_FALSE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = { 10, 10, 11 };
    ASSERT_FALSE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerValidationMode, gen_phase_to_cut_whole_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    constexpr size_t processed_before = 5;
    sequence_groups.front()->update_processed_tokens_num(processed_before);

    // append candidates [ 2, 3, 4 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i + 1, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 2, 3, 4]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
    EXPECT_EQ(sequence_groups.front()->get_num_processed_tokens(), processed_before + 1)
        << "Full rejection must retain the target replacement token as accepted depth one";
}

TEST(SamplerValidationMode, gen_phase_to_cut_part_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates [ 1, 2, 2 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        int64_t token_id = i == num_validated_tokens ? i - 1 : i;
        sequence_groups.front()->get_sequences().front()->append_token(token_id, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2, 2]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, gen_phase) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates [ 1, 2, 3 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2, 3]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3, 4};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase_to_cut_part_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 0, 1, 1 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        int64_t token_id = i + 1 == num_validated_tokens ? i - 1 : i;
        sequence_groups.front()->get_sequences().front()->append_token(token_id, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 1]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase_to_cut_whole_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 1, 2, 3 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i + 1, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [1, 2, 3]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 0, 1, 2 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

namespace {
// Emulates a processed prompt [0, 1, 2, 3, 4] followed by the main model's token 0, then appends the draft candidates
// and schedules them for validation.
SequenceGroup::Ptr make_validation_group(const GenerationConfig& config, const std::vector<int64_t>& candidates) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    auto group = std::make_shared<SequenceGroup>(0, input_tensor, config);
    auto sequence = group->get_sequences().front();
    sequence->append_token(0, 1.f);
    group->update_processed_tokens_num(input_vector.size());
    for (const auto candidate : candidates) {
        sequence->append_token(candidate, 1.f);
    }
    group->set_num_validated_tokens(candidates.size());
    group->schedule_tokens(group->get_num_available_tokens_for_batching());
    return group;
}
}  // namespace

// min_new_tokens = 2 masks the stop token for the first two generated tokens only, so the stop candidate drafted as
// the fourth token must be accepted.
TEST(SamplerValidationMode, lifts_min_new_tokens_mask_inside_validation_window) {
    auto config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = 30;
    config.min_new_tokens = 2;
    config.stop_token_ids = {4};
    auto group = make_validation_group(config, {1, 2, 4});
    std::vector<float> logits = {
        0, 5.f, 0, 0, 0,
        0, 0, 5.f, 0, 0,
        1.f, 0, 0, 0, 5.f,
        5.f, 0, 0, 0, 0,
    };
    Sampler sampler;
    sampler.sample({group}, ov::Tensor(ov::element::f32, ov::Shape{4, 1, 5}, logits.data()), true);

    EXPECT_EQ(group->get_sequences().front()->get_generated_ids(), (TokenIds{0, 1, 2, 4}));
}

// min_new_tokens = 5 masks the stop token for the first five generated tokens. Rejecting candidate 2 in favour of 3
// leaves three generated tokens, so the stop token must stay masked at the fourth even though its logit is largest.
TEST(SamplerValidationMode, keeps_min_new_tokens_mask_after_rejection) {
    auto config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = 30;
    config.min_new_tokens = 5;
    config.stop_token_ids = {4};
    auto group = make_validation_group(config, {1, 2});
    auto sequence = group->get_sequences().front();
    std::vector<float> window_logits = {
        0, 5.f, 0, 0, 0,
        0, 0, 0, 5.f, 0,
        5.f, 0, 0, 0, 0,
    };
    Sampler sampler;
    sampler.sample({group}, ov::Tensor(ov::element::f32, ov::Shape{3, 1, 5}, window_logits.data()), true);
    ASSERT_EQ(sequence->get_generated_ids(), (TokenIds{0, 1, 3}));

    group->schedule_tokens(group->get_num_available_tokens_for_batching());
    std::vector<float> next_logits = {0, 0, 4.f, 0, 5.f};
    sampler.sample({group}, ov::Tensor(ov::element::f32, ov::Shape{1, 1, 5}, next_logits.data()), true);

    EXPECT_EQ(sequence->get_generated_ids(), (TokenIds{0, 1, 3, 2}));
    EXPECT_FALSE(sequence->has_finished());
}

// Accepted candidates report the main model's log-probabilities, and the sequence score must follow them.
TEST(SamplerValidationMode, accepted_candidates_update_sequence_score) {
    auto config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = 30;
    auto group = make_validation_group(config, {1, 2});
    auto sequence = group->get_sequences().front();
    std::vector<float> logits = {
        0, 2.f, 0, 0, 0,
        0, 0, 3.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };
    Sampler sampler;
    sampler.sample({group}, ov::Tensor(ov::element::f32, ov::Shape{3, 1, 5}, logits.data()), true);
    ASSERT_EQ(sequence->get_generated_ids(), (TokenIds{0, 1, 2, 3}));

    const auto& log_probs = sequence->get_generated_log_probs();
    float sum = 0.0f;
    for (const auto log_prob : log_probs) {
        sum += log_prob;
    }
    EXPECT_FLOAT_EQ(sequence->get_cumulative_log_prob(), sum);
}

// The parallel samples of a request share one logit processor. Each must still see the request's generated length,
// so the stop token stays masked for all of them until min_new_tokens tokens are generated.
TEST(SamplerMinNewTokens, masks_stop_token_for_every_parallel_sample) {
    GenerationConfig config;
    config.max_new_tokens = 10;
    config.do_sample = true;
    config.top_k = 1;
    config.num_return_sequences = 3;
    config.min_new_tokens = 3;
    config.stop_token_ids = {4};
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    auto group = std::make_shared<SequenceGroup>(0, input_tensor, config);
    Sampler sampler;

    group->schedule_tokens(input_vector.size());
    group->set_output_seq_len(1);
    std::vector<float> prompt_logits = {0, 5.f, 0, 0, 4.f};
    sampler.sample({group}, ov::Tensor(ov::element::f32, ov::Shape{1, 1, 5}, prompt_logits.data()));
    ASSERT_EQ(group->num_running_seqs(), 3);

    for (size_t step = 0; step < 2; ++step) {
        group->schedule_tokens(1);
        std::vector<float> logits;
        for (size_t sample = 0; sample < 3; ++sample) {
            logits.insert(logits.end(), {0, 0, 4.f, 0, 5.f});
        }
        sampler.sample({group}, ov::Tensor(ov::element::f32, ov::Shape{3, 1, 5}, logits.data()));
    }

    for (const auto& sequence : group->get_sequences()) {
        EXPECT_EQ(sequence->get_generated_ids(), (TokenIds{1, 2, 2}));
    }
}
