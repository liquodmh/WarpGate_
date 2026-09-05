# WarpGate

**SLA-aware heterogeneous runtime for real-time vector search on CPU + NVIDIA GPU.**

WarpGate is not another vector database. It is a runtime layer for the awkward production case where clients send online vector-search requests one at a time while GPUs achieve their best efficiency on larger batches.

The core idea is simple: **wait only when waiting is safe**. WarpGate learns backend latency by batch size, watches each request deadline, forms micro-batches when there is latency budget, and routes work to the backend predicted to meet the SLA most efficiently.

> Status: v0.1 scheduler/reference implementation. CUDA/cuVS numbers are intentionally not claimed until measured on real NVIDIA hardware.

## Why this project exists

Offline ANN benchmarks commonly use convenient batch sizes. Production traffic often does not. A service can receive many independent `batch=1` requests, creating launch/copy/queue overhead and poor accelerator utilization.

WarpGate targets this runtime gap:

```text
batch=1 clients
      |
      v
+-------------------+
| WarpGate runtime  |
|-------------------|
| deadline guard    |
| micro-batcher     |
| latency learner   |
| CPU/GPU router    |
+---------+---------+
          |
    +-----+------+
    |            |
    v            v
CPU HNSW/SIMD  NVIDIA cuVS/CAGRA
```

## v0.1 implemented now

- C++20 backend abstraction
- exact L2 CPU reference backend
- SLA/deadline-aware queueing
- adaptive micro-batch planning
- per-backend EWMA latency model by batch bucket
- backend selection from observed latency
- scheduler microbenchmark
- unit tests + GitHub Actions CI

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the local reference benchmark:

```bash
./build/warpgate_bench 50000 128 200
```

Arguments are `rows dimension queries`.

## The benchmark rule

**No invented GPU numbers.** Every performance table in this repository must be generated from a committed benchmark command and record:

- GPU model + driver
- CUDA version
- cuVS version/commit
- CPU model
- dataset + dimension + top-k
- arrival pattern
- recall target
- p50/p95/p99
- QPS
- GPU utilization

The headline experiment for the NVIDIA path is:

```text
online workload from client: batch size = 1

A) immediate request-by-request cuVS/CAGRA
B) concurrent streams without adaptive batching
C) WarpGate adaptive micro-batching

constraint: same recall target and same p99 SLA
objective: maximize throughput without violating p99
```

## What would count as a real result?

Not “GPU is faster than CPU.” That is boring.

A valuable result is something like:

> Under batch=1 online traffic, WarpGate increases throughput at the same p99 latency target by dynamically deciding when to wait, batch, route, or execute immediately.

The percentage belongs here only after measurement.

## NVIDIA-focused roadmap

1. CPU baseline: HNSW + AVX2/AVX-512 + profiling.
2. Add cuVS/CAGRA backend.
3. Add CUDA stream pool and pinned-memory pipeline.
4. Benchmark batch=1 online traffic against naive cuVS execution.
5. Add multi-GPU routing and per-GPU queue models.
6. Experiment with NIXL / GPUDirect paths where hardware permits.
7. Publish traces, benchmark scripts, and a technical report.
8. Use the measurements to contribute an issue, benchmark, or PR upstream.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [ROADMAP.md](docs/ROADMAP.md).

## Intended engineering depth

- Modern C++ and memory ownership
- data structures and ANN algorithms
- SIMD and CPU cache behavior
- CUDA concurrency and stream scheduling
- accelerator-aware queueing
- latency/throughput/SLA tradeoffs
- reproducible systems benchmarking
- multi-GPU and high-performance data movement

## License

MIT
