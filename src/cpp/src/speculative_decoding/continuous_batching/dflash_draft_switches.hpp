// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>

namespace ov::genai::dflash_cb {

// Prototype switches for temperature-only DFlash drafting, read once per process. All off reproduces #4848.
struct DraftSwitches {
    // OV_GENAI_DFLASH_DRAFT_TEMPERATURE_ONLY=1: the draft keeps only the request's temperature and seed; top_k, top_p,
    // min_p, the penalties and min_new_tokens apply on the target only.
    bool temperature_only = false;
    // OV_GENAI_DFLASH_DENSE_SAMPLER=1: temperature-only full-vocabulary rows (DFlash v1, the dense Markov step) are
    // drawn from a softmax on the OpenVINO CPU plugin instead of the LogitProcessor, into pooled proposal buffers.
    bool dense_sampler = false;
    // OV_GENAI_DFLASH_PARALLEL_ROWS=1: the rows of a DFlash v1 block are drawn in parallel.
    bool parallel_rows = false;
    // OV_GENAI_DFLASH_DRAFT_TOPK=K: a DFlash v1 draft returns its K best logits per row and their ids.
    size_t draft_topk = 0;
    // OV_GENAI_DFLASH_ROW_STREAMS=N: CPU plugin streams of the dense sampler, the rows it softmaxes at once; 0 takes
    // min(8, hardware threads).
    size_t row_streams = 0;
};

inline const DraftSwitches& draft_switches() {
    static const DraftSwitches switches = [] {
        const auto flag = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && std::string(value) == "1";
        };
        const auto count = [](const char* name) -> size_t {
            const char* value = std::getenv(name);
            return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : 0;
        };
        DraftSwitches result;
        result.temperature_only = flag("OV_GENAI_DFLASH_DRAFT_TEMPERATURE_ONLY");
        result.dense_sampler = flag("OV_GENAI_DFLASH_DENSE_SAMPLER");
        result.parallel_rows = flag("OV_GENAI_DFLASH_PARALLEL_ROWS");
        result.draft_topk = count("OV_GENAI_DFLASH_DRAFT_TOPK");
        result.row_streams = count("OV_GENAI_DFLASH_ROW_STREAMS");
        return result;
    }();
    return switches;
}

}  // namespace ov::genai::dflash_cb
