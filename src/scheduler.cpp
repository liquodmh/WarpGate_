#include "warpgate/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace warpgate {

AdaptiveScheduler::AdaptiveScheduler(SchedulerConfig config) : config_(config) {
    if (config_.max_micro_batch == 0) {
        throw std::invalid_argument("max_micro_batch must be > 0");
    }
}

void AdaptiveScheduler::add_backend(std::shared_ptr<SearchBackend> backend, double cold_start_us) {
    if (!backend) throw std::invalid_argument("backend must not be null");
    backends_.push_back({std::move(backend), LatencyModel(cold_start_us)});
}

void AdaptiveScheduler::submit(SearchQuery query) {
    if (query.deadline_us <= query.enqueue_time_us) {
        throw std::invalid_argument("deadline must be after enqueue time");
    }
    queue_.push_back(std::move(query));
}

std::size_t AdaptiveScheduler::pending() const noexcept { return queue_.size(); }

std::size_t AdaptiveScheduler::candidate_batch_size(std::uint64_t now_us) const {
    if (queue_.empty()) return 0;
    const auto hard_cap = std::min(config_.max_micro_batch, queue_.size());
    const auto waited = now_us - queue_.front().enqueue_time_us;
    if (waited >= config_.max_batch_wait_us) return hard_cap;

    const auto earliest_deadline = queue_.front().deadline_us;
    for (const auto& q : queue_) {
        if (q.deadline_us < earliest_deadline) {
            // FIFO is intentional for now; deadline ordering becomes v0.2.
            break;
        }
    }

    const auto time_left = earliest_deadline > now_us ? earliest_deadline - now_us : 0;
    if (time_left <= config_.safety_margin_us) return 1;
    return hard_cap;
}

bool AdaptiveScheduler::should_dispatch(std::uint64_t now_us) const {
    if (queue_.empty()) return false;
    if (queue_.size() >= config_.max_micro_batch) return true;
    if (now_us - queue_.front().enqueue_time_us >= config_.max_batch_wait_us) return true;
    return queue_.front().deadline_us <= now_us + config_.safety_margin_us;
}

std::size_t AdaptiveScheduler::choose_backend(std::size_t batch_size,
                                              std::uint64_t now_us,
                                              std::uint64_t earliest_deadline_us) const {
    if (backends_.empty()) throw std::runtime_error("no backends registered");

    std::size_t best = 0;
    double best_prediction = std::numeric_limits<double>::infinity();
    const double budget = earliest_deadline_us > now_us
        ? static_cast<double>(earliest_deadline_us - now_us - std::min(config_.safety_margin_us, earliest_deadline_us - now_us))
        : 0.0;

    bool found_within_budget = false;
    for (std::size_t i = 0; i < backends_.size(); ++i) {
        const auto effective_batch = std::min(batch_size, backends_[i].backend->preferred_max_batch());
        const double prediction = backends_[i].latency.predict(effective_batch);
        const bool within_budget = prediction <= budget;

        if ((within_budget && !found_within_budget) ||
            (within_budget == found_within_budget && prediction < best_prediction)) {
            best = i;
            best_prediction = prediction;
            found_within_budget = within_budget;
        }
    }
    return best;
}

BatchExecution AdaptiveScheduler::plan(std::uint64_t now_us) const {
    if (queue_.empty()) throw std::runtime_error("cannot plan empty queue");
    const auto batch = candidate_batch_size(now_us);
    const auto deadline = queue_.front().deadline_us;
    const auto backend_index = choose_backend(batch, now_us, deadline);
    const auto actual_batch = std::min(batch, backends_[backend_index].backend->preferred_max_batch());

    return {
        backends_[backend_index].backend->name().data(),
        actual_batch,
        backends_[backend_index].latency.predict(actual_batch),
        now_us,
    };
}

std::vector<SearchResult> AdaptiveScheduler::dispatch(std::uint64_t now_us) {
    const auto execution = plan(now_us);

    std::size_t backend_index = 0;
    for (; backend_index < backends_.size(); ++backend_index) {
        if (backends_[backend_index].backend->name() == execution.backend_name) break;
    }
    if (backend_index == backends_.size()) throw std::runtime_error("planned backend disappeared");

    std::vector<SearchQuery> batch;
    batch.reserve(execution.batch_size);
    for (std::size_t i = 0; i < execution.batch_size; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }

    const auto start = std::chrono::steady_clock::now();
    auto results = backends_[backend_index].backend->search(batch);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::micro>(end - start).count();
    backends_[backend_index].latency.observe(batch.size(), elapsed);
    return results;
}

void AdaptiveScheduler::observe(std::string_view backend_name,
                                std::size_t batch_size,
                                double latency_us) {
    for (auto& slot : backends_) {
        if (slot.backend->name() == backend_name) {
            slot.latency.observe(batch_size, latency_us);
            return;
        }
    }
    throw std::invalid_argument("unknown backend");
}

} // namespace warpgate
