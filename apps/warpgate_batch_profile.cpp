#include "warpgate/hnsw_backend.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const auto mid = values.size() / 2;
    if (values.size() % 2 == 1) return values[mid];
    return (values[mid - 1] + values[mid]) / 2.0;
}
}

int main(int argc, char** argv) {
    try {
        std::size_t rows = 5000;
        std::size_t dim = 128;
        std::size_t max_batch = 32;
        std::string output = "warpgate_latency_profile.csv";
        if (argc > 1) rows = static_cast<std::size_t>(std::stoull(argv[1]));
        if (argc > 2) dim = static_cast<std::size_t>(std::stoull(argv[2]));
        if (argc > 3) max_batch = static_cast<std::size_t>(std::stoull(argv[3]));
        if (argc > 4) output = argv[4];
        if (rows == 0 || dim == 0 || max_batch == 0) throw std::invalid_argument("rows, dim and max_batch must be positive");

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        std::vector<float> data(rows * dim);
        for (auto& v : data) v = dist(rng);

        warpgate::HnswBackend backend(std::move(data), rows, dim, {16, 100, 64, 42, warpgate::DistanceMetric::Cosine, 0});

        std::vector<warpgate::SearchQuery> queries;
        queries.reserve(max_batch);
        for (std::size_t i = 0; i < max_batch; ++i) {
            std::vector<float> q(dim);
            for (auto& v : q) v = dist(rng);
            queries.push_back({i, std::move(q), 10, 1, 1000000});
        }

        std::ofstream out(output, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open profile output");
        out << "batch_size,latency_us\n";

        for (std::size_t batch = 1; batch <= max_batch; batch *= 2) {
            const auto span = std::span<const warpgate::SearchQuery>(queries.data(), batch);
            for (int warmup = 0; warmup < 2; ++warmup) (void)backend.search(span);

            std::vector<double> samples;
            samples.reserve(7);
            for (int rep = 0; rep < 7; ++rep) {
                const auto start = std::chrono::steady_clock::now();
                (void)backend.search(span);
                const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                samples.push_back(elapsed);
            }
            const auto latency = median(std::move(samples));
            out << batch << ',' << latency << '\n';
            std::cout << "batch=" << batch << " median_us=" << latency << '\n';
            if (batch > max_batch / 2) break;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "warpgate_batch_profile: " << e.what() << '\n';
        return 2;
    }
}
