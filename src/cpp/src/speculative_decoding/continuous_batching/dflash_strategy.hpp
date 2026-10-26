// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fast_draft_strategy.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai {

// DFlash speculative decoding on continuous batching: the target validates candidates and exports the
// hidden states of the requested layers; the draft proposes each request's candidates in one forward
// over the window [target hidden-state context rows ; seed + mask block rows].
class ContinuousBatchingPipeline::DFlashDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                       const ov::genai::ModelDesc& draft_model_desc,
                       const ov::genai::utils::dflash::DFlashRTInfo& rt_info);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

protected:
    DFlashDecodingImpl() = default;
    void align_request_pair_processed_prefix(uint64_t request_id) override;
    void validate_awaiting_requests(const std::vector<SequenceGroup::Ptr>& main_awaiting_requests,
                                    const std::vector<SequenceGroup::Ptr>& draft_awaiting_requests) const override;
    void validate_request_config(const GenerationConfig& config) const;
    static GenerationConfig make_draft_generation_config(const GenerationConfig& config);

    ov::genai::utils::dflash::DFlashRTInfo m_rt_info;
};

// Continuous-batching pipeline for both DFlash models. The validating main pipeline publishes its
// processed length with the exported hidden states. The draft pipeline forwards each request's window
// once per step and drafts the candidates of every forwarded block with the candidate selector; its logic
// is in ContinuousBatchingForSpeculativeDecodingImpl behind dflash_mode_enabled.
class ContinuousBatchingPipeline::ContinuousBatchingForDFlashDecodingImpl
    : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
public:
    ContinuousBatchingForDFlashDecodingImpl() = default;

    ContinuousBatchingForDFlashDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                            const Tokenizer& tokenizer,
                                            const GenerationConfig& generation_config,
                                            const SchedulerConfig& scheduler_config,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config,
                                            bool is_validation_mode_enabled,
                                            int64_t mask_token_id = -1,
                                            size_t candidate_position_offset = 1,
                                            std::shared_ptr<dflash_cb::CandidateSelector> candidate_selector = nullptr);

    ContinuousBatchingForDFlashDecodingImpl(const std::shared_ptr<ov::Model>& model,
                                            std::shared_ptr<InputsEmbedder> inputs_embedder,
                                            const Tokenizer& tokenizer,
                                            const GenerationConfig& generation_config,
                                            const SchedulerConfig& scheduler_config,
                                            const std::string& device,
                                            const ov::AnyMap& plugin_config,
                                            bool is_validation_mode_enabled,
                                            int64_t mask_token_id = -1,
                                            size_t candidate_position_offset = 1,
                                            std::shared_ptr<dflash_cb::CandidateSelector> candidate_selector = nullptr);

    bool supports_embedding_prefix_verification() const override {
        return true;
    }

    // Only prompt blocks are published: the draft KV past the prompt holds mask and rewound block rows.
    // DFlashDecodingImpl restores main and draft to a common prefix after both requests are added.
    void initialize_prefix_cache(const SequenceGroup::Ptr& group) override {
        for (const auto& sequence : group->get_sequences()) {
            sequence->set_prefix_cache_policy(group->get_prompt_len());
        }
    }

protected:
    bool _can_publish_kv_only_completed_blocks() const override {
        return true;
    }
};

}  // namespace ov::genai
