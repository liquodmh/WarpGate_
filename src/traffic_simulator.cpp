#include "warpgate/traffic_simulator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace warpgate {
namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = p * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(index));
    const auto hi = static_cast<std::size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lo);
    return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

std::size_t select_batch_size(const std::vector<TrafficRequest>& pending,
                              std::uint64_t now_us,
                              std::uint64_t next_arrival_us,
                              const ServiceModel& service,
                              const TrafficSimConfig& config,
                              bool& should_wait) {
    should_wait = false;
    if (pending.empty()) return 0;

    const auto hard_cap = std::min(config.max_batch, pending.size());
    if (config.policy == BatchPolicy::Immediate) return 1;

    const auto& earliest = pending.front();
    const auto waited_us = now_us >= earliest.arrival_us ? now_us - earliest.arrival_us : 0;

    if (config.policy == BatchPolicy::Fixed) {
        const auto target = std::min(config.fixed_batch, config.max_batch);
        if (pending.size() >= target || waited_us >= config.max_wait_us) {
            return std::min(target, pending.size());
        }
        if (next_arrival_us != std::numeric_limits<std::uint64_t>::max()) {
            const auto wait_deadline = earliest.arrival_us + config.max_wait_us;
            if (next_arrival_us <= wait_deadline) should_wait = true;
        }
        return std::min(target, pending.size());
    }

    const double predicted_us = service.predict(hard_cap);
    const double safe_finish_us = static_cast<double>(now_us) + predicted_us +
                                  static_cast<double>(config.safety_margin_us);
    const bool deadline_pressure = safe_finish_us >= static_cast<double>(earliest.deadline_us);
    if (deadline_pressure || hard_cap == config.max_batch || waited_us >= config.max_wait_us) {
        return hard_cap;
    }

    const auto wait_deadline = std::min(
        earliest.arrival_us + config.max_wait_us,
        earliest.deadline_us > config.safety_margin_us
            ? earliest.deadline_us - config.safety_margin_us
            : earliest.deadline_us);

    if (next_arrival_us != std::numeric_limits<std::uint64_t>::max() &&
        next_arrival_us <= wait_deadline) {
        should_wait = true;
    }
    return hard_cap;
}

} // namespace

double ServiceModel::predict(std::size_t batch_size) const {
    if (batch_size == 0) return 0.0;
    if (!measured_profile.empty()) {
        if (batch_size <= measured_profile.front().batch_size) return measured_profile.front().latency_us;
        for (std::size_t i = 1; i < measured_profile.size(); ++i) {
            if (batch_size <= measured_profile[i].batch_size) {
                const auto& a = measured_profile[i - 1];
                const auto& b = measured_profile[i];
                const double width = static_cast<double>(b.batch_size - a.batch_size);
                const double x = static_cast<double>(batch_size - a.batch_size) / width;
                return a.latency_us + x * (b.latency_us - a.latency_us);
            }
        }
        const auto& last = measured_profile.back();
        return last.latency_us * static_cast<double>(batch_size) / static_cast<double>(last.batch_size);
    }
    if (launch_overhead_us < 0.0 || per_query_us < 0.0 || batch_exponent <= 0.0) {
        throw std::invalid_argument("invalid service model");
    }
    return launch_overhead_us + per_query_us * std::pow(static_cast<double>(batch_size), batch_exponent);
}

ServiceModel ServiceModel::from_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open latency profile");
    ServiceModel model;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("batch_size") != std::string::npos) continue;
        std::stringstream ss(line);
        std::string batch_text;
        std::string latency_text;
        if (!std::getline(ss, batch_text, ',') || !std::getline(ss, latency_text)) {
            throw std::runtime_error("invalid latency profile row");
        }
        const auto batch = static_cast<std::size_t>(std::stoull(batch_text));
        const auto latency = std::stod(latency_text);
        if (batch == 0 || !(latency > 0.0)) throw std::runtime_error("invalid latency profile value");
        model.measured_profile.push_back({batch, latency});
    }
    if (model.measured_profile.empty()) throw std::runtime_error("latency profile is empty");
    std::sort(model.measured_profile.begin(), model.measured_profile.end(), [](const LatencyPoint& a, const LatencyPoint& b) {
        return a.batch_size < b.batch_size;
    });
    for (std::size_t i = 1; i < model.measured_profile.size(); ++i) {
        if (model.measured_profile[i - 1].batch_size == model.measured_profile[i].batch_size) {
            throw std::runtime_error("duplicate batch size in latency profile");
        }
    }
    return model;
}

std::vector<TrafficRequest> make_traffic_trace(std::size_t request_count,
                                               TrafficPattern pattern,
                                               double offered_qps,
                                               std::uint64_t default_sla_us,
                                               std::uint32_t seed) {
    if (request_count == 0) return {};
    if (!(offered_qps > 0.0) || default_sla_us == 0) {
        throw std::invalid_argument("offered_qps and SLA must be positive");
    }

    const double interval_us = 1'000'000.0 / offered_qps;
    std::vector<TrafficRequest> trace;
    trace.reserve(request_count);
    std::mt19937 rng(seed);
    std::uint64_t t = 0;

    for (std::size_t i = 0; i < request_count; ++i) {
        if (pattern == TrafficPattern::Bursty) {
            constexpr std::size_t kBurstSize = 32;
            if (i != 0 && i % kBurstSize == 0) {
                t += static_cast<std::uint64_t>(std::llround(interval_us * static_cast<double>(kBurstSize)));
            }
        } else {
            if (i != 0) t += static_cast<std::uint64_t>(std::llround(interval_us));
        }

        std::uint64_t sla = default_sla_us;
        if (pattern == TrafficPattern::MixedSla) {
            std::uniform_int_distribution<int> lane_dist(0, 9);
            const auto lane = lane_dist(rng);
            if (lane < 2) sla = std::max<std::uint64_t>(800, default_sla_us / 5);
            else if (lane < 6) sla = std::max<std::uint64_t>(2000, default_sla_us / 2);
            else sla = default_sla_us;
        }

        trace.push_back({static_cast<std::uint64_t>(i), t, t + sla});
    }
    return trace;
}

TrafficMetrics simulate_traffic(const std::vector<TrafficRequest>& trace,
                                const ServiceModel& service,
                                const TrafficSimConfig& config) {
    if (trace.empty()) return {};
    if (config.max_batch == 0 || config.fixed_batch == 0) {
        throw std::invalid_argument("batch sizes must be positive");
    }
    if (!std::is_sorted(trace.begin(), trace.end(),
                        [](const TrafficRequest& a, const TrafficRequest& b) {
                            return a.arrival_us < b.arrival_us;
                        })) {
        throw std::invalid_argument("traffic trace must be sorted by arrival time");
    }

    std::vector<TrafficRequest> pending;
    pending.reserve(config.max_batch * 4);
    std::vector<double> latencies;
    latencies.reserve(trace.size());

    std::size_t next = 0;
    std::size_t completed = 0;
    std::size_t misses = 0;
    std::size_t batches = 0;
    std::size_t batch_items = 0;
    std::size_t max_batch = 0;
    std::size_t max_queue = 0;
    std::uint64_t now_us = trace.front().arrival_us;
    const auto first_arrival = trace.front().arrival_us;

    auto enqueue_arrived = [&] {
        while (next < trace.size() && trace[next].arrival_us <= now_us) {
            pending.push_back(trace[next++]);
        }
        std::sort(pending.begin(), pending.end(), [](const TrafficRequest& a, const TrafficRequest& b) {
            if (a.deadline_us != b.deadline_us) return a.deadline_us < b.deadline_us;
            return a.arrival_us < b.arrival_us;
        });
        max_queue = std::max(max_queue, pending.size());
    };

    while (completed < trace.size()) {
        enqueue_arrived();
        if (pending.empty()) {
            now_us = trace[next].arrival_us;
            enqueue_arrived();
        }

        const auto next_arrival = next < trace.size()
            ? trace[next].arrival_us
            : std::numeric_limits<std::uint64_t>::max();

        bool should_wait = false;
        auto batch_size = select_batch_size(pending, now_us, next_arrival, service, config, should_wait);

        if (should_wait && next_arrival != std::numeric_limits<std::uint64_t>::max()) {
            now_us = next_arrival;
            continue;
        }

        batch_size = std::min(batch_size, pending.size());
        if (batch_size == 0) throw std::runtime_error("simulator selected empty batch");

        const auto service_us = static_cast<std::uint64_t>(std::ceil(service.predict(batch_size)));
        const auto finish_us = now_us + service_us;
        for (std::size_t i = 0; i < batch_size; ++i) {
            const auto& request = pending[i];
            latencies.push_back(static_cast<double>(finish_us - request.arrival_us));
            if (finish_us > request.deadline_us) ++misses;
        }

        pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(batch_size));
        now_us = finish_us;
        completed += batch_size;
        ++batches;
        batch_items += batch_size;
        max_batch = std::max(max_batch, batch_size);
    }

    const auto makespan = now_us - first_arrival;
    TrafficMetrics metrics;
    metrics.requests = completed;
    metrics.throughput_qps = makespan == 0 ? 0.0 :
        static_cast<double>(completed) * 1'000'000.0 / static_cast<double>(makespan);
    metrics.p50_us = percentile(latencies, 0.50);
    metrics.p95_us = percentile(latencies, 0.95);
    metrics.p99_us = percentile(latencies, 0.99);
    metrics.max_us = *std::max_element(latencies.begin(), latencies.end());
    metrics.sla_miss_rate = static_cast<double>(misses) / static_cast<double>(completed);
    metrics.mean_batch_size = batches == 0 ? 0.0 :
        static_cast<double>(batch_items) / static_cast<double>(batches);
    metrics.max_batch_size = max_batch;
    metrics.max_queue_depth = max_queue;
    metrics.makespan_us = makespan;
    return metrics;
}

std::string_view traffic_pattern_name(TrafficPattern pattern) noexcept {
    switch (pattern) {
        case TrafficPattern::Steady: return "steady";
        case TrafficPattern::Bursty: return "bursty";
        case TrafficPattern::MixedSla: return "mixed-sla";
    }
    return "unknown";
}

std::string_view batch_policy_name(BatchPolicy policy) noexcept {
    switch (policy) {
        case BatchPolicy::Immediate: return "immediate";
        case BatchPolicy::Fixed: return "fixed";
        case BatchPolicy::Adaptive: return "adaptive";
    }
    return "unknown";
}

} // namespace warpgate
