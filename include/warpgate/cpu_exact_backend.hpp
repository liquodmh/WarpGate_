#pragma once

#include "warpgate/backend.hpp"

#include <cstddef>
#include <vector>

namespace warpgate {

class CpuExactBackend final : public SearchBackend {
public:
    CpuExactBackend(std::vector<float> dataset, std::size_t rows, std::size_t dim);

    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::size_t preferred_max_batch() const noexcept override;
    std::vector<SearchResult> search(std::span<const SearchQuery> queries) override;

private:
    [[nodiscard]] SearchResult search_one(const SearchQuery& query) const;

    std::vector<float> dataset_;
    std::size_t rows_{};
    std::size_t dim_{};
};

} // namespace warpgate
