#pragma once

#include <array>
#include <cstddef>

namespace warpgate {

class LatencyModel {
public:
    explicit LatencyModel(double cold_start_us = 1000.0, double alpha = 0.20);

    void observe(std::size_t batch_size, double latency_us);
    [[nodiscard]] double predict(std::size_t batch_size) const;

private:
    static constexpr std::size_t kBuckets = 8;
    [[nodiscard]] static std::size_t bucket_for(std::size_t batch_size);

    std::array<double, kBuckets> ewma_{};
    std::array<bool, kBuckets> seen_{};
    double cold_start_us_{};
    double alpha_{};
};

} // namespace warpgate
