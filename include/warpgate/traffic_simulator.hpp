#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace warpgate {

enum class TrafficPattern { Steady, Bursty, MixedSla };
enum class BatchPolicy { Immediate, Fixed, Adaptive };

struct TrafficRequest {
    std::uint64_t id{};
    std::uint64_t arrival_us{};
    std::uint64_t deadline_us{};
};

struct LatencyPoint {
    std::size_t batch_size{};
    double latency_us{};
};

struct ServiceModel {
    double launch_overhead_us{180.0};
    double per_query_us{35.0};
    double batch_exponent{0.70};
    std::vector<LatencyPoint> measured_profile;

    [[nodiscard]] double predict(std::size_t batch_size) const;
    [[nodiscard]] bool uses_measured_profile() const noexcept { return !measured_profile.empty(); }
    [[nodiscard]] static ServiceModel from_csv(const std::string& path);
};

struct TrafficSimConfig {
    BatchPolicy policy{BatchPolicy::Adaptive};
    std::size_t max_batch{32};
    std::size_t fixed_batch{16};
    std::uint64_t max_wait_us{100};
    std::uint64_t safety_margin_us{50};
};

struct TrafficMetrics {
    std::size_t requests{};
    double throughput_qps{};
    double p50_us{};
    double p95_us{};
    double p99_us{};
    double max_us{};
    double sla_miss_rate{};
    double mean_batch_size{};
    std::size_t max_batch_size{};
    std::size_t max_queue_depth{};
    std::uint64_t makespan_us{};
};

[[nodiscard]] std::vector<TrafficRequest> make_traffic_trace(
    std::size_t request_count,
    TrafficPattern pattern,
    double offered_qps,
    std::uint64_t default_sla_us = 5000,
    std::uint32_t seed = 42);

[[nodiscard]] TrafficMetrics simulate_traffic(
    const std::vector<TrafficRequest>& trace,
    const ServiceModel& service,
    const TrafficSimConfig& config);

[[nodiscard]] std::string_view traffic_pattern_name(TrafficPattern pattern) noexcept;
[[nodiscard]] std::string_view batch_policy_name(BatchPolicy policy) noexcept;

} // namespace warpgate
