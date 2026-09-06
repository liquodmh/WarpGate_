#include "warpgate/cpu_exact_backend.hpp"
#include "warpgate/distance.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
namespace warpgate {
CpuExactBackend::CpuExactBackend(std::vector<float> dataset, std::size_t rows, std::size_t dim)
    : dataset_(std::move(dataset)), rows_(rows), dim_(dim) {
    if (rows_ == 0 || dim_ == 0 || dataset_.size() != rows_ * dim_) {
        throw std::invalid_argument("dataset shape does not match rows * dim");
    }
}
std::string_view CpuExactBackend::name() const noexcept { return "cpu-exact"; }
std::size_t CpuExactBackend::preferred_max_batch() const noexcept { return 16; }
SearchResult CpuExactBackend::search_one(const SearchQuery& query) const {
    if (query.vector.size() != dim_) throw std::invalid_argument("query dimension mismatch");
    const auto k = std::min(query.top_k, rows_);
    SearchResult out;
    out.reserve(rows_);
    const auto q = std::span<const float>(query.vector.data(), query.vector.size());
    for (std::size_t row = 0; row < rows_; ++row) {
        const auto candidate = std::span<const float>(dataset_.data() + row * dim_, dim_);
        out.push_back({row, l2_squared(q, candidate)});
    }
    if (k < out.size()) {
        std::nth_element(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k), out.end(),
                         [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
        out.resize(k);
    }
    std::sort(out.begin(), out.end(), [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
    return out;
}
std::vector<SearchResult> CpuExactBackend::search(std::span<const SearchQuery> queries) {
    std::vector<SearchResult> results;
    results.reserve(queries.size());
    for (const auto& query : queries) results.push_back(search_one(query));
    return results;
}
}
