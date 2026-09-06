# Architecture
For deadline `D` at time `t`, `wait_budget = D - t - predicted_execution_time - safety_margin`. A request may wait only while that budget is positive. CPU exact search is the correctness oracle; HNSW is the ANN backend; both share the distance-kernel abstraction. Requests are stored in earliest-deadline-first order and measured execution latency is fed back into EWMA batch buckets.
