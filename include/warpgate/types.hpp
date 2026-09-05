#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace warpgate {

using QueryId = std::uint64_t;

struct SearchQuery {
    QueryId id{};
    std::vector<float> vector;
    std::size_t top_k{10};
    std::uint64_t enqueue_time_us{};
    std::uint64_t deadline_us{};
};

struct Neighbor {
    std::size_t index{};
    float distance{};
};

using SearchResult = std::vector<Neighbor>;

struct BatchExecution {
    const char* backend_name{};
    std::size_t batch_size{};
    double predicted_latency_us{};
    std::uint64_t dispatch_time_us{};
};

} // namespace warpgate
