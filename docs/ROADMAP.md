# Roadmap

## Gate 1 — Runtime correctness ✅
Backend abstraction, EDF scheduling, SLA-aware batching, EWMA latency model, unit tests.

## Gate 2 — CPU ANN engine ✅
HNSW from scratch, scalar/AVX2 L2 and cosine, multi-core batch execution, exact ground truth, Recall@K/QPS benchmarks, versioned index persistence, sanitizer CI.

## Gate 3 — Online workload science ✅
Deterministic traffic-trace replay, p50/p95/p99 end-to-end latency, SLA miss rate, queue depth, batch-size distribution and throughput under steady/bursty/mixed-SLA traffic. The current service model is synthetic by design and will be replaced by measured GPU timings in Gate 4.

## Gate 4 — NVIDIA GPU
cuVS/CAGRA adapter, pinned host buffers, bounded CUDA stream pool, async H2D/D2H, Nsight Systems/Compute traces, batch=1 vs adaptive micro-batch benchmark on real NVIDIA hardware.

## Gate 5 — Scale-out
Multi-GPU routing, per-GPU pressure model, sharding/replication, NIXL/GPUDirect experiments where hardware permits.

## Gate 6 — Open-source credibility
Reproducible public datasets, technical report, benchmark artifacts, and an upstream NVIDIA/cuVS issue or PR supported by measurements.
