#pragma once
#include "warpgate/types.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <span>
#include <thread>
#include <vector>
namespace warpgate {
template <class Fn>
std::vector<SearchResult> parallel_query_map(std::span<const SearchQuery> queries, std::size_t requested_workers, Fn&& fn) {
    if (queries.empty()) return {};
    std::size_t workers = requested_workers;
    if (workers == 0) workers = std::max(1u, std::thread::hardware_concurrency());
    workers = std::min(workers, queries.size());
    if (workers <= 1 || queries.size() < 2) {
        std::vector<SearchResult> out; out.reserve(queries.size());
        for (const auto& q : queries) out.push_back(fn(q));
        return out;
    }
    std::vector<SearchResult> out(queries.size());
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool; pool.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        pool.emplace_back([&] {
            while (true) {
                const auto i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= queries.size()) break;
                out[i] = fn(queries[i]);
            }
        });
    }
    for (auto& t : pool) t.join();
    return out;
}
}
