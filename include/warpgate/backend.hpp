#pragma once

#include "warpgate/types.hpp"

#include <span>
#include <string_view>
#include <vector>

namespace warpgate {

class SearchBackend {
public:
    virtual ~SearchBackend() = default;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual std::size_t preferred_max_batch() const noexcept = 0;
    virtual std::vector<SearchResult> search(std::span<const SearchQuery> queries) = 0;
};

} // namespace warpgate
