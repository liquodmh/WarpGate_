#include "warpgate/backend.hpp"
#include "warpgate/latency_model.hpp"
#include "warpgate/scheduler.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <string_view>

class FakeBackend final : public warpgate::SearchBackend {
public:
    FakeBackend(std::string_view name, std::size_t max_batch) : name_(name), max_batch_(max_batch) {}
    std::string_view name() const noexcept override { return name_; }
    std::size_t preferred_max_batch() const noexcept override { return max_batch_; }
    std::vector<warpgate::SearchResult> search(std::span<const warpgate::SearchQuery> q) override {
        return std::vector<warpgate::SearchResult>(q.size());
    }
private:
    std::string_view name_;
    std::size_t max_batch_;
};

int main() {
    {
        warpgate::LatencyModel model(1000.0, 0.5);
        assert(model.predict(1) == 1000.0);
        model.observe(1, 200.0);
        assert(model.predict(1) == 200.0);
        model.observe(1, 100.0);
        assert(model.predict(1) == 150.0);
    }

    {
        warpgate::AdaptiveScheduler scheduler({.max_micro_batch = 8, .max_batch_wait_us = 100, .safety_margin_us = 20});
        auto cpu = std::make_shared<FakeBackend>("cpu", 8);
        auto gpu = std::make_shared<FakeBackend>("gpu", 64);
        scheduler.add_backend(cpu, 300.0);
        scheduler.add_backend(gpu, 700.0);
        scheduler.observe("cpu", 8, 600.0);
        scheduler.observe("gpu", 8, 250.0);

        for (std::size_t i = 0; i < 8; ++i) {
            scheduler.submit({i, {0.0F}, 1, 1'000, 10'000});
        }
        const auto plan = scheduler.plan(1'100);
        assert(plan.batch_size == 8);
        assert(std::string_view(plan.backend_name) == "gpu");
    }

    {
        warpgate::AdaptiveScheduler scheduler({.max_micro_batch = 16, .max_batch_wait_us = 100, .safety_margin_us = 50});
        scheduler.add_backend(std::make_shared<FakeBackend>("cpu", 16), 100.0);
        scheduler.submit({1, {0.0F}, 1, 1'000, 1'040});
        assert(scheduler.should_dispatch(1'001));
        assert(scheduler.plan(1'001).batch_size == 1);
    }

    std::cout << "all tests passed\n";
    return 0;
}
