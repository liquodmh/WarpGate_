#include "warpgate/latency_model.hpp"
#include <algorithm>
namespace warpgate {
LatencyModel::LatencyModel(double cold_start_us, double alpha)
    : cold_start_us_(cold_start_us), alpha_(std::clamp(alpha, 0.01, 1.0)) {
    ewma_.fill(cold_start_us_); seen_.fill(false);
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
    const auto b = bucket_for(batch_size);
    if (!seen_[b]) { ewma_[b] = latency_us; seen_[b] = true; return; }
    ewma_[b] = alpha_ * latency_us + (1.0 - alpha_) * ewma_[b];
}
double LatencyModel::predict(std::size_t batch_size) const {
    const auto b = bucket_for(batch_size);
    if (seen_[b]) return ewma_[b];
    for (std::size_t r = 1; r < kBuckets; ++r) {
        if (b >= r && seen_[b-r]) return ewma_[b-r] * std::max(1.0, static_cast<double>(batch_size) / static_cast<double>(std::size_t{1} << (b-r)) * 0.60);
        if (b + r < kBuckets && seen_[b+r]) return ewma_[b+r];
    }
    return cold_start_us_;
}
}
