# Roadmap

## Gate 1 — Correctness
- [x] Backend abstraction
- [x] Reference exact search
- [x] Deadline-aware scheduler
- [x] Batch latency learning
- [x] Unit tests
- [ ] EDF queue
- [ ] deterministic replay of traffic traces

## Gate 2 — CPU baseline
- [ ] HNSW implementation from scratch
- [ ] AVX2 L2/cosine kernels
- [ ] optional AVX-512 kernels
- [ ] thread pool
- [ ] perf + cache-miss profile

## Gate 3 — NVIDIA GPU
- [ ] cuVS/CAGRA adapter
- [ ] CUDA stream pool
- [ ] pinned memory allocator
- [ ] micro-batch copy/launch pipeline
- [ ] Nsight Systems traces
- [ ] Nsight Compute kernel analysis

## Gate 4 — Paper-grade benchmark
Datasets:
- [ ] SIFT1M
- [ ] GIST1M
- [ ] Deep1B subset
- [ ] text-embedding workload (e.g. 768D)

Metrics:
- [ ] Recall@10
- [ ] QPS
- [ ] p50/p95/p99 latency
- [ ] GPU utilization
- [ ] GPU memory footprint
- [ ] energy/query if counters are available

Workloads:
- [ ] client batch=1 steady load
- [ ] bursty traffic
- [ ] mixed SLA tenants
- [ ] overload / backpressure

## Gate 5 — Open-source credibility
- [ ] reproducible Docker/devcontainer
- [ ] public benchmark scripts
- [ ] technical report
- [ ] upstream issue/PR backed by measurements
