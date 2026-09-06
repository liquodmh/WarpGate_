#pragma once
#include <span>
#include <string_view>
namespace warpgate {
[[nodiscard]] float l2_squared_scalar(std::span<const float> a, std::span<const float> b);
[[nodiscard]] float l2_squared(std::span<const float> a, std::span<const float> b);
[[nodiscard]] std::string_view distance_kernel_name() noexcept;
}
