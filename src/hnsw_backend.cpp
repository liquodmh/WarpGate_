#include "warpgate/hnsw_backend.hpp"
#include "warpgate/distance.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
namespace warpgate {
HnswBackend::HnswBackend(std::vector<float> dataset, std::size_t rows, std::size_t dim, HnswConfig config)
    : dataset_(std::move(dataset)), rows_(rows), dim_(dim), config_(config), rng_(config.seed) {
    if (rows_ == 0 || dim_ == 0 || dataset_.size() != rows_ * dim_) throw std::invalid_argument("dataset shape does not match rows * dim");
    if (config_.m < 2) throw std::invalid_argument("HNSW m must be >= 2");
    config_.ef_construction = std::max(config_.ef_construction, config_.m);
    config_.ef_search = std::max<std::size_t>(config_.ef_search, 1);
    nodes_.reserve(rows_);
    for (std::size_t id = 0; id < rows_; ++id) insert_node(id);
}
std::string_view HnswBackend::name() const noexcept { return "hnsw-cpu"; }
std::size_t HnswBackend::preferred_max_batch() const noexcept { return 32; }
std::span<const float> HnswBackend::vector_at(std::size_t id) const { return {dataset_.data() + id * dim_, dim_}; }
float HnswBackend::distance_to(std::span<const float> q, std::size_t id) const { return l2_squared(q, vector_at(id)); }
int HnswBackend::sample_level() {
    std::uniform_real_distribution<double> u(std::numeric_limits<double>::min(), 1.0);
    return static_cast<int>(-std::log(u(rng_)) / std::log(static_cast<double>(config_.m)));
}
std::size_t HnswBackend::greedy_at_level(std::span<const float> q, std::size_t entry, int level) const {
    std::size_t current = entry; float best = distance_to(q, current); bool improved = true;
    while (improved) {
        improved = false;
        if (level > nodes_[current].level) break;
        for (const auto n : nodes_[current].links[static_cast<std::size_t>(level)]) {
            const float d = distance_to(q, n);
            if (d < best) { best = d; current = n; improved = true; }
        }
    }
    return current;
}
std::vector<HnswBackend::Candidate> HnswBackend::search_layer(std::span<const float> q, std::size_t entry, std::size_t ef, int level) const {
    struct MinCmp { bool operator()(const Candidate& a, const Candidate& b) const { return a.distance > b.distance; } };
    struct MaxCmp { bool operator()(const Candidate& a, const Candidate& b) const { return a.distance < b.distance; } };
    std::priority_queue<Candidate, std::vector<Candidate>, MinCmp> candidates;
    std::priority_queue<Candidate, std::vector<Candidate>, MaxCmp> best;
    std::unordered_set<std::size_t> visited; visited.reserve(ef * 8 + 16);
    const Candidate start{entry, distance_to(q, entry)};
    candidates.push(start); best.push(start); visited.insert(entry);
    while (!candidates.empty()) {
        const auto current = candidates.top();
        if (best.size() >= ef && current.distance > best.top().distance) break;
        candidates.pop();
        if (level > nodes_[current.id].level) continue;
        for (const auto n : nodes_[current.id].links[static_cast<std::size_t>(level)]) {
            if (!visited.insert(n).second) continue;
            Candidate next{n, distance_to(q, n)};
            if (best.size() < ef || next.distance < best.top().distance) {
                candidates.push(next); best.push(next); if (best.size() > ef) best.pop();
            }
        }
    }
    std::vector<Candidate> out; out.reserve(best.size());
    while (!best.empty()) { out.push_back(best.top()); best.pop(); }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });
    return out;
}
void HnswBackend::prune(std::size_t node, int level) {
    auto& links = nodes_[node].links[static_cast<std::size_t>(level)];
    if (links.size() <= config_.m) return;
    const auto base = vector_at(node);
    std::sort(links.begin(), links.end(), [&](std::size_t a, std::size_t b) {
        return l2_squared(base, vector_at(a)) < l2_squared(base, vector_at(b));
    });
    links.resize(config_.m);
}
void HnswBackend::connect_bidirectional(std::size_t a, std::size_t b, int level) {
    auto& la = nodes_[a].links[static_cast<std::size_t>(level)];
    auto& lb = nodes_[b].links[static_cast<std::size_t>(level)];
    if (std::find(la.begin(), la.end(), b) == la.end()) la.push_back(b);
    if (std::find(lb.begin(), lb.end(), a) == lb.end()) lb.push_back(a);
    prune(a, level); prune(b, level);
}
void HnswBackend::insert_node(std::size_t id) {
    const int level = sample_level();
    nodes_.push_back(Node{level, std::vector<std::vector<std::size_t>>(static_cast<std::size_t>(level + 1))});
    if (id == 0) { entry_point_ = 0; max_level_ = level; return; }
    const auto q = vector_at(id); std::size_t entry = entry_point_;
    for (int l = max_level_; l > level; --l) entry = greedy_at_level(q, entry, l);
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        const auto candidates = search_layer(q, entry, config_.ef_construction, l);
        const auto count = std::min(config_.m, candidates.size());
        for (std::size_t i = 0; i < count; ++i) connect_bidirectional(id, candidates[i].id, l);
        if (!candidates.empty()) entry = candidates.front().id;
    }
    if (level > max_level_) { entry_point_ = id; max_level_ = level; }
}
SearchResult HnswBackend::search_one(const SearchQuery& query) const {
    if (query.vector.size() != dim_) throw std::invalid_argument("query dimension mismatch");
    const auto q = std::span<const float>(query.vector.data(), query.vector.size());
    std::size_t entry = entry_point_;
    for (int l = max_level_; l > 0; --l) entry = greedy_at_level(q, entry, l);
    const auto candidates = search_layer(q, entry, std::max({config_.ef_search, query.top_k, std::size_t{1}}), 0);
    const auto k = std::min(query.top_k, candidates.size());
    SearchResult out; out.reserve(k);
    for (std::size_t i = 0; i < k; ++i) out.push_back({candidates[i].id, candidates[i].distance});
    return out;
}
std::vector<SearchResult> HnswBackend::search(std::span<const SearchQuery> queries) {
    std::vector<SearchResult> out; out.reserve(queries.size());
    for (const auto& q : queries) out.push_back(search_one(q));
    return out;
}
}
