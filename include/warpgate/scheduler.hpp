#pragma once

#include "warpgate/backend.hpp"
#include "warpgate/latency_model.hpp"
#include "warpgate/types.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace warpgate {

struct BackendSlot {
    std::shared_ptr<SearchBackend> backend;
    LatencyModel latency;
};

struct SchedulerConfig {
    std::size_t max_micro_batch{64};
    std::uint64_t max_batch_wait_us{100};
    std::uint64_t safety_margin_us{50};
};

class AdaptiveScheduler {
public:
    explicit AdaptiveScheduler(SchedulerConfig config = {});

    void add_backend(std::shared_ptr<SearchBackend> backend, double cold_start_us = 1000.0);
    void submit(SearchQuery query);

    [[nodiscard]] std::size_t pending() const noexcept;
    [[nodiscard]] bool should_dispatch(std::uint64_t now_us) const;

    BatchExecution plan(std::uint64_t now_us) const;
    std::vector<SearchResult> dispatch(std::uint64_t now_us);

    void observe(std::string_view backend_name, std::size_t batch_size, double latency_us);

private:
    [[nodiscard]] std::size_t candidate_batch_size(std::uint64_t now_us) const;
    [[nodiscard]] std::size_t choose_backend(std::size_t batch_size,
                                             std::uint64_t now_us,
                                             std::uint64_t earliest_deadline_us) const;

    SchedulerConfig config_;
    std::deque<SearchQuery> queue_;
    std::vector<BackendSlot> backends_;
};

} // namespace warpgate
