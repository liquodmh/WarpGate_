#include "warpgate/latency_model.hpp"

#include <algorithm>

namespace warpgate {

LatencyModel::LatencyModel(double cold_start_us, double alpha)
    : cold_start_us_(cold_start_us), alpha_(std::clamp(alpha, 0.01, 1.0)) {
    ewma_.fill(cold_start_us_);
    seen_.fill(false);
}

std::size_t LatencyModel::bucket_for(std::size_t batch_size) {
    if (batch_size <= 1) return 0;
    if (batch_size <= 2) return 1;
    if (batch_size <= 4) return 2;
    if (batch_size <= 8) return 3;
    if (batch_size <= 16) return 4;
    if (batch_size <= 32) return 5;
    if (batch_size <= 64) return 6;
    return 7;
}

void LatencyModel::observe(std::size_t batch_size, double latency_us) {
    const auto bucket = bucket_for(batch_size);
    if (!seen_[bucket]) {
        ewma_[bucket] = latency_us;
        seen_[bucket] = true;
        return;
    }
    ewma_[bucket] = alpha_ * latency_us + (1.0 - alpha_) * ewma_[bucket];
}

double LatencyModel::predict(std::size_t batch_size) const {
    const auto bucket = bucket_for(batch_size);
    if (seen_[bucket]) return ewma_[bucket];

    for (std::size_t radius = 1; radius < kBuckets; ++radius) {
        if (bucket >= radius && seen_[bucket - radius]) {
            const double scale = static_cast<double>(batch_size) / static_cast<double>(1ULL << (bucket - radius));
            return ewma_[bucket - radius] * std::max(1.0, scale * 0.60);
        }
        if (bucket + radius < kBuckets && seen_[bucket + radius]) {
            return ewma_[bucket + radius];
        }
    }
    return cold_start_us_;
}

} // namespace warpgate
