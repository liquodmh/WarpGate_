#pragma once
#include "warpgate/backend.hpp"
#include "warpgate/distance.hpp"
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>
namespace warpgate {
struct HnswConfig {
    std::size_t m{16};
    std::size_t ef_construction{100};
    std::size_t ef_search{64};
    std::uint32_t seed{42};
    DistanceMetric metric{DistanceMetric::L2};
    std::size_t workers{0};
};
class HnswBackend final : public SearchBackend {
public:
    HnswBackend(std::vector<float> dataset, std::size_t rows, std::size_t dim, HnswConfig config = {});
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::size_t preferred_max_batch() const noexcept override;
    std::vector<SearchResult> search(std::span<const SearchQuery> queries) override;
    void save(const std::string& path) const;
    [[nodiscard]] static HnswBackend load(const std::string& path, std::size_t workers = 0);
    [[nodiscard]] std::size_t size() const noexcept { return rows_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }
    [[nodiscard]] DistanceMetric metric() const noexcept { return config_.metric; }
private:
    struct EmptyTag {};
    explicit HnswBackend(EmptyTag) : rng_(42) {}
    struct Node { int level{}; std::vector<std::vector<std::size_t>> links; };
    struct Candidate { std::size_t id{}; float distance{}; };
    [[nodiscard]] std::span<const float> vector_at(std::size_t id) const;
    [[nodiscard]] int sample_level();
    [[nodiscard]] float distance_to(std::span<const float> query, std::size_t id) const;
    [[nodiscard]] std::size_t greedy_at_level(std::span<const float> query, std::size_t entry, int level) const;
    [[nodiscard]] std::vector<Candidate> search_layer(std::span<const float> query, std::size_t entry, std::size_t ef, int level) const;
    void prune(std::size_t node, int level);
    void connect_bidirectional(std::size_t a, std::size_t b, int level);
    void insert_node(std::size_t id);
    [[nodiscard]] SearchResult search_one(const SearchQuery& query) const;
    std::vector<float> dataset_; std::size_t rows_{}; std::size_t dim_{}; HnswConfig config_{};
    std::vector<Node> nodes_; std::size_t entry_point_{}; int max_level_{-1}; std::mt19937 rng_;
};
}
