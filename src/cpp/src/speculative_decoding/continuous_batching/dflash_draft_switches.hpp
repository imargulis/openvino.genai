// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <system_error>
#include <thread>

#include "openvino/core/except.hpp"

namespace ov::genai::dflash_cb {

// Prototype switches for temperature-only DFlash drafting, read once per process. All off reproduces #4848.
struct DraftSwitches {
    // OV_GENAI_DFLASH_DRAFT_TEMPERATURE_ONLY=1: a draft that draws temperature-only rows without the sampler, from its
    // candidate scorer, the dense sampler or the Top-K head, keeps only the request's temperature and seed; top_k,
    // top_p, min_p, the penalties and min_new_tokens apply on the target only. Any other draft keeps them, since the
    // sampler draws a row faster with the request's top_k than without one.
    bool temperature_only = false;
    // OV_GENAI_DFLASH_DENSE_SAMPLER=1: temperature-only full-vocabulary rows (DFlash v1, the dense Markov step) of a
    // draft without a soft cap are drawn from a softmax on the OpenVINO CPU plugin instead of the LogitProcessor, into
    // pooled proposal buffers.
    bool dense_sampler = false;
    // OV_GENAI_DFLASH_PARALLEL_ROWS=1: the dense sampler softmaxes the rows of a DFlash v1 block at once, on CPU
    // plugin streams, and draws them on as many threads.
    bool parallel_rows = false;
    // OV_GENAI_DFLASH_DRAFT_TOPK=K: a DFlash v1 draft returns its K best logits per row and their ids. Needs
    // OV_GENAI_DFLASH_DRAFT_TEMPERATURE_ONLY=1.
    size_t draft_topk = 0;
    // OV_GENAI_DFLASH_ROW_STREAMS=N: the CPU plugin streams of OV_GENAI_DFLASH_PARALLEL_ROWS=1; 0 takes min(8,
    // hardware threads). The plugin keeps at most one per performance core.
    size_t row_streams = 0;

    // the CPU plugin streams that the dense sampler asks for
    size_t dense_sampler_streams() const {
        if (!parallel_rows) {
            return 1;
        }
        return row_streams > 0 ? row_streams : std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 8);
    }
};

inline const DraftSwitches& draft_switches() {
    static const DraftSwitches switches = [] {
        const auto value_of = [](const char* name) {
            const char* value = std::getenv(name);
            return std::string(value != nullptr ? value : "");
        };
        const auto flag = [&](const char* name) {
            const std::string value = value_of(name);
            OPENVINO_ASSERT(value.empty() || value == "0" || value == "1", name, " must be 0 or 1, got '", value, "'.");
            return value == "1";
        };
        const auto count = [&](const char* name) -> size_t {
            const std::string value = value_of(name);
            size_t result = 0;
            const char* end = value.data() + value.size();
            const auto parsed = std::from_chars(value.data(), end, result);
            OPENVINO_ASSERT(value.empty() || (parsed.ec == std::errc{} && parsed.ptr == end),
                            name,
                            " must be a non-negative integer, got '",
                            value,
                            "'.");
            return result;
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
