#include "warpgate/scheduler.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
namespace warpgate {
AdaptiveScheduler::AdaptiveScheduler(SchedulerConfig config) : config_(config) {
    if (config_.max_micro_batch == 0) throw std::invalid_argument("max_micro_batch must be > 0");
}
void AdaptiveScheduler::add_backend(std::shared_ptr<SearchBackend> backend, double cold_start_us) {
    if (!backend) throw std::invalid_argument("backend must not be null");
    backends_.push_back({std::move(backend), LatencyModel(cold_start_us)});
}
void AdaptiveScheduler::submit(SearchQuery query) {
    if (query.deadline_us <= query.enqueue_time_us) throw std::invalid_argument("deadline must be after enqueue time");
    const auto pos = std::upper_bound(queue_.begin(), queue_.end(), query.deadline_us,
        [](std::uint64_t deadline, const SearchQuery& q) { return deadline < q.deadline_us; });
    queue_.insert(pos, std::move(query));
}
std::size_t AdaptiveScheduler::pending() const noexcept { return queue_.size(); }
std::size_t AdaptiveScheduler::candidate_batch_size(std::uint64_t now_us) const {
    if (queue_.empty()) return 0;
    const auto hard_cap = std::min(config_.max_micro_batch, queue_.size());
    const auto waited = now_us >= queue_.front().enqueue_time_us ? now_us - queue_.front().enqueue_time_us : 0;
    if (waited >= config_.max_batch_wait_us) return hard_cap;
    const auto time_left = queue_.front().deadline_us > now_us ? queue_.front().deadline_us - now_us : 0;
    if (time_left <= config_.safety_margin_us) return 1;
    return hard_cap;
}
bool AdaptiveScheduler::should_dispatch(std::uint64_t now_us) const {
    if (queue_.empty()) return false;
    if (queue_.size() >= config_.max_micro_batch) return true;
    const auto waited = now_us >= queue_.front().enqueue_time_us ? now_us - queue_.front().enqueue_time_us : 0;
    if (waited >= config_.max_batch_wait_us) return true;
    return queue_.front().deadline_us <= now_us + config_.safety_margin_us;
}
std::size_t AdaptiveScheduler::choose_backend(std::size_t batch_size, std::uint64_t now_us, std::uint64_t deadline) const {
    if (backends_.empty()) throw std::runtime_error("no backends registered");
    std::size_t best = 0; double best_pred = std::numeric_limits<double>::infinity(); bool found_fit = false;
    const auto remaining = deadline > now_us ? deadline - now_us : 0;
    const double budget = static_cast<double>(remaining > config_.safety_margin_us ? remaining - config_.safety_margin_us : 0);
    for (std::size_t i = 0; i < backends_.size(); ++i) {
        const auto effective = std::min(batch_size, backends_[i].backend->preferred_max_batch());
        const double pred = backends_[i].latency.predict(effective);
        const bool fit = pred <= budget;
        if ((fit && !found_fit) || (fit == found_fit && pred < best_pred)) {
            best = i; best_pred = pred; found_fit = fit;
        }
    }
    return best;
}
BatchExecution AdaptiveScheduler::plan(std::uint64_t now_us) const {
    if (queue_.empty()) throw std::runtime_error("cannot plan empty queue");
    const auto batch = candidate_batch_size(now_us);
    const auto bi = choose_backend(batch, now_us, queue_.front().deadline_us);
    const auto actual = std::min(batch, backends_[bi].backend->preferred_max_batch());
    return {backends_[bi].backend->name().data(), actual, backends_[bi].latency.predict(actual), now_us};
}
std::vector<SearchResult> AdaptiveScheduler::dispatch(std::uint64_t now_us) {
    const auto execution = plan(now_us);
    std::size_t bi = 0;
    for (; bi < backends_.size(); ++bi) if (backends_[bi].backend->name() == execution.backend_name) break;
    if (bi == backends_.size()) throw std::runtime_error("planned backend disappeared");
    std::vector<SearchQuery> batch; batch.reserve(execution.batch_size);
    for (std::size_t i = 0; i < execution.batch_size; ++i) { batch.push_back(std::move(queue_.front())); queue_.pop_front(); }
    const auto start = std::chrono::steady_clock::now();
    auto result = backends_[bi].backend->search(batch);
    const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    backends_[bi].latency.observe(batch.size(), elapsed);
    return result;
}
void AdaptiveScheduler::observe(std::string_view backend_name, std::size_t batch_size, double latency_us) {
    for (auto& slot : backends_) if (slot.backend->name() == backend_name) { slot.latency.observe(batch_size, latency_us); return; }
    throw std::invalid_argument("unknown backend");
}
}
