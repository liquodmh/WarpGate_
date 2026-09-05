#include "warpgate/cpu_exact_backend.hpp"
#include "warpgate/scheduler.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace {
std::uint64_t now_us() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
}

int main(int argc, char** argv) {
    std::size_t rows = 50'000;
    std::size_t dim = 128;
    std::size_t queries = 200;
    if (argc > 1) rows = static_cast<std::size_t>(std::stoull(argv[1]));
    if (argc > 2) dim = static_cast<std::size_t>(std::stoull(argv[2]));
    if (argc > 3) queries = static_cast<std::size_t>(std::stoull(argv[3]));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> dataset(rows * dim);
    for (auto& x : dataset) x = dist(rng);

    auto cpu = std::make_shared<warpgate::CpuExactBackend>(std::move(dataset), rows, dim);
    warpgate::AdaptiveScheduler scheduler({.max_micro_batch = 16, .max_batch_wait_us = 50, .safety_margin_us = 100});
    scheduler.add_backend(cpu, 5'000.0);

    std::vector<float> query_vector(dim);
    for (std::size_t q = 0; q < queries; ++q) {
        for (auto& x : query_vector) x = dist(rng);
        const auto t = now_us();
        scheduler.submit({q, query_vector, 10, t, t + 50'000});
    }

    const auto start = std::chrono::steady_clock::now();
    std::size_t completed = 0;
    while (scheduler.pending() > 0) {
        const auto plan = scheduler.plan(now_us());
        auto result = scheduler.dispatch(now_us());
        completed += result.size();
        std::cout << "dispatch backend=" << plan.backend_name
                  << " batch=" << plan.batch_size
                  << " predicted_us=" << std::fixed << std::setprecision(1)
                  << plan.predicted_latency_us << '\n';
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "\nWarpGate scheduler microbenchmark\n"
              << "rows=" << rows << " dim=" << dim << " queries=" << completed << '\n'
              << "elapsed_s=" << std::fixed << std::setprecision(4) << seconds << '\n'
              << "qps=" << std::fixed << std::setprecision(2) << (completed / seconds) << '\n'
              << "NOTE: CPU exact-search only; this is NOT a CUDA/cuVS performance claim.\n";
    return 0;
}
