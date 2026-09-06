#include "warpgate/traffic_simulator.hpp"

#include <cassert>
#include <iostream>

int main() {
    const auto trace = warpgate::make_traffic_trace(2000, warpgate::TrafficPattern::Steady, 20000.0, 5000, 42);
    const warpgate::ServiceModel service{};

    warpgate::TrafficSimConfig immediate;
    immediate.policy = warpgate::BatchPolicy::Immediate;
    const auto a = warpgate::simulate_traffic(trace, service, immediate);

    warpgate::TrafficSimConfig adaptive;
    adaptive.policy = warpgate::BatchPolicy::Adaptive;
    adaptive.max_batch = 32;
    adaptive.max_wait_us = 100;
    adaptive.safety_margin_us = 50;
    const auto b = warpgate::simulate_traffic(trace, service, adaptive);

    assert(a.requests == trace.size());
    assert(b.requests == trace.size());
    assert(a.mean_batch_size == 1.0);
    assert(b.mean_batch_size > 1.0);
    assert(b.throughput_qps > a.throughput_qps);
    assert(b.sla_miss_rate < a.sla_miss_rate);
    assert(b.p99_us < a.p99_us);

    const auto bursty = warpgate::make_traffic_trace(320, warpgate::TrafficPattern::Bursty, 10000.0, 5000, 42);
    const auto c = warpgate::simulate_traffic(bursty, service, adaptive);
    assert(c.requests == bursty.size());
    assert(c.max_batch_size > 1);

    const auto mixed = warpgate::make_traffic_trace(500, warpgate::TrafficPattern::MixedSla, 8000.0, 5000, 42);
    const auto d = warpgate::simulate_traffic(mixed, service, adaptive);
    assert(d.requests == mixed.size());
    assert(d.p99_us >= d.p50_us);

    std::cout << "traffic simulator tests passed\n";
}
