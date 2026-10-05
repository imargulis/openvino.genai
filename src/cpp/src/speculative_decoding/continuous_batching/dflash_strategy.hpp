// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fast_draft_strategy.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::DFlashDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                       const ov::genai::ModelDesc& draft_model_desc,
                       const ov::genai::utils::dflash::DFlashRTInfo& rt_info,
                       const ov::genai::ModelDesc& selector_model_desc = {},
                       const ov::genai::utils::dflash::DFlashSelectorRTInfo& selector_rt_info = {});

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;
    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;

private:
    class DFlashCBDraftRunner;

    using RequestProgress = ContinuousBatchingForSpeculativeDecodingImpl::RequestProgress;

    // One sequence of a draft call.
    struct DraftInput {
        uint64_t request_id = 0;
        int64_t seed_token = 0;
        // target hidden states of the context rows, [rows, 1, hidden]
        ov::Tensor hidden_delta;
        // candidates of the draft block
        size_t candidate_count = 0;
        // candidates proposed to the target, at most candidate_count
        size_t validation_count = 0;
    };

    struct DraftCandidates {
        std::vector<int64_t> token_ids;
        std::vector<float> log_probs;
        // the distribution every candidate was sampled from, empty for greedy proposals
        std::vector<DraftProposalPtr> proposals;
    };

    struct RequestState {
        dflash_cb::HiddenDeltaBuffer pending_hidden_deltas;
        std::vector<int64_t> generated_tokens;
        size_t prompt_len = 0;
        size_t generated_before_draft = 0;
        size_t draft_generated = 0;
        // target processed tokens when the pending candidates were proposed
        size_t processed_before_validation = 0;
        bool finished = false;
        GenerationConfig generation_config;
    };

    GenerationConfig make_draft_generation_config(const GenerationConfig& config) const;
    static void append_pending_hidden_delta(RequestState& state, const ov::Tensor& hidden_delta, bool copy_data);
    static bool has_pending_hidden_delta(const RequestState& state);
    static ov::Tensor materialize_pending_hidden_delta(const RequestState& state);
    static void clear_pending_hidden_delta(RequestState& state);
    void validate_hidden_prefix_length(uint64_t request_id, const RequestState& state) const;
    void append_new_hidden_rows(uint64_t request_id,
                                RequestState& state,
                                const ov::Tensor& hidden_state,
                                size_t num_processed_tokens);
    bool has_active_request_state() const;
    void drop_finished_request_states();
    std::vector<DraftInput> plan_draft_inputs();
    void submit_candidates(const DraftInput& input, DraftCandidates candidates);
    void update_draft_states_from_main(const GeneratedRequests& main_generated_requests);
    void drop_requests();
    ov::genai::RawPerfMetrics collect_draft_raw_metrics();

    std::shared_ptr<DFlashCBDraftRunner> m_draft;
    ov::genai::utils::dflash::DFlashRTInfo m_rt_info;
    bool m_selector_enabled = false;
    size_t m_max_num_batched_tokens = 0;
    std::map<uint64_t, RequestState> m_request_states;
};

}  // namespace ov::genai
