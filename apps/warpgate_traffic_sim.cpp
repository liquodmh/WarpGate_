#include "warpgate/traffic_simulator.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
warpgate::TrafficPattern parse_pattern(const std::string& value) {
    if (value == "steady") return warpgate::TrafficPattern::Steady;
    if (value == "bursty") return warpgate::TrafficPattern::Bursty;
    if (value == "mixed" || value == "mixed-sla") return warpgate::TrafficPattern::MixedSla;
    throw std::invalid_argument("pattern must be steady, bursty, or mixed-sla");
}

void print_row(std::string_view policy, const warpgate::TrafficMetrics& m, bool json) {
    if (json) {
        std::cout << "{\"policy\":\"" << policy
                  << "\",\"requests\":" << m.requests
                  << ",\"throughput_qps\":" << m.throughput_qps
                  << ",\"p50_us\":" << m.p50_us
                  << ",\"p95_us\":" << m.p95_us
                  << ",\"p99_us\":" << m.p99_us
                  << ",\"max_us\":" << m.max_us
                  << ",\"sla_miss_rate\":" << m.sla_miss_rate
                  << ",\"mean_batch_size\":" << m.mean_batch_size
                  << ",\"max_batch_size\":" << m.max_batch_size
                  << ",\"max_queue_depth\":" << m.max_queue_depth
                  << ",\"makespan_us\":" << m.makespan_us << "}\n";
        return;
    }

    std::cout << std::left << std::setw(10) << policy
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(12) << m.throughput_qps
              << std::setw(10) << m.p50_us
              << std::setw(10) << m.p95_us
              << std::setw(10) << m.p99_us
              << std::setw(10) << (m.sla_miss_rate * 100.0)
              << std::setw(10) << m.mean_batch_size
              << std::setw(10) << m.max_queue_depth << '\n';
}
}

int main(int argc, char** argv) {
    try {
        std::size_t requests = 5000;
        double offered_qps = 20000.0;
        auto pattern = warpgate::TrafficPattern::Steady;
        bool json = false;
        std::string profile_path;

        if (argc > 1) requests = static_cast<std::size_t>(std::stoull(argv[1]));
        if (argc > 2) pattern = parse_pattern(argv[2]);
        if (argc > 3) offered_qps = std::stod(argv[3]);
        for (int i = 4; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--json") json = true;
            else profile_path = arg;
        }

        const auto trace = warpgate::make_traffic_trace(requests, pattern, offered_qps, 5000, 42);
        const auto service = profile_path.empty() ? warpgate::ServiceModel{} : warpgate::ServiceModel::from_csv(profile_path);

        if (!json) {
            std::cout << "Synthetic online traffic simulation (NOT a GPU benchmark)\n"
                      << "pattern=" << warpgate::traffic_pattern_name(pattern)
                      << " offered_qps=" << offered_qps
                      << " requests=" << requests
                      << " service=" << (service.uses_measured_profile() ? "measured" : "synthetic") << "\n\n"
                      << std::left << std::setw(10) << "policy"
                      << std::right << std::setw(12) << "throughput"
                      << std::setw(10) << "p50_us"
                      << std::setw(10) << "p95_us"
                      << std::setw(10) << "p99_us"
                      << std::setw(10) << "SLA_miss%"
                      << std::setw(10) << "mean_b"
                      << std::setw(10) << "max_q" << '\n';
        }

        for (const auto policy : {warpgate::BatchPolicy::Immediate,
                                  warpgate::BatchPolicy::Fixed,
                                  warpgate::BatchPolicy::Adaptive}) {
            warpgate::TrafficSimConfig config;
            config.policy = policy;
            config.max_batch = 32;
            config.fixed_batch = 16;
            config.max_wait_us = 100;
            config.safety_margin_us = 50;
            const auto metrics = warpgate::simulate_traffic(trace, service, config);
            print_row(warpgate::batch_policy_name(policy), metrics, json);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "warpgate_traffic_sim: " << e.what() << '\n';
        return 2;
    }
}
