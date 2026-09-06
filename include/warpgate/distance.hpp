#pragma once
#include <span>
#include <string_view>
namespace warpgate {
enum class DistanceMetric { L2 = 0, Cosine = 1 };
[[nodiscard]] float l2_squared_scalar(std::span<const float> a, std::span<const float> b);
[[nodiscard]] float cosine_distance_scalar(std::span<const float> a, std::span<const float> b);
[[nodiscard]] float l2_squared(std::span<const float> a, std::span<const float> b);
[[nodiscard]] float cosine_distance(std::span<const float> a, std::span<const float> b);
[[nodiscard]] float vector_distance(std::span<const float> a, std::span<const float> b, DistanceMetric metric);
[[nodiscard]] std::string_view distance_kernel_name() noexcept;
[[nodiscard]] std::string_view distance_metric_name(DistanceMetric metric) noexcept;
}
