// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fast_draft_strategy.hpp"
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

    void validate_request(
        uint64_t request_id,
        const ov::Tensor& input_ids,
        const ov::genai::GenerationConfig& sampling_params,
        const std::optional<ov::Tensor>& prompt_ids,
        const std::optional<std::unordered_map<std::string, ov::Tensor>>& lm_extra_inputs) const override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;

protected:
    DFlashDecodingImpl() = default;
    void align_request_pair_processed_prefix(uint64_t request_id) override;
    void validate_awaiting_requests(const std::vector<SequenceGroup::Ptr>& main_awaiting_requests,
                                    const std::vector<SequenceGroup::Ptr>& draft_awaiting_requests) const override;
    void validate_request_config(const GenerationConfig& config) const;
    // An unset num_assistant_tokens takes the default; an explicit 0 runs the request on the target alone.
    GenerationConfig make_main_generation_config(const GenerationConfig& config) const;
    // With temperature_only, the draft keeps only the request's temperature and seed.
    static GenerationConfig make_draft_generation_config(const GenerationConfig& config, bool temperature_only = false);

    ov::genai::utils::dflash::DFlashRTInfo m_rt_info;
    // OV_GENAI_DFLASH_DRAFT_TEMPERATURE_ONLY=1 for a draft that draws temperature-only rows without the sampler
    bool m_draft_temperature_only = false;
};

}  // namespace ov::genai
