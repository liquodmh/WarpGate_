# WarpGate

**SLA-aware heterogeneous runtime for real-time vector search.**

WarpGate targets a production mismatch: online clients often send vector-search requests one at a time while high-throughput backends are most efficient on batches. The runtime learns backend latency, orders work by deadline, batches only when SLA budget allows, and routes across heterogeneous search backends.

> **v0.5 measured-model calibration:** HNSW, AVX2, cosine/L2 metrics, parallel batch search, index persistence, EDF scheduling, reproducible JSON benchmarks, and sanitizer CI are implemented. CUDA/cuVS remains intentionally gated on real NVIDIA hardware measurements.

## Implemented

- C++20
- exact L2/cosine reference backend
- hierarchical **HNSW from scratch**
- scalar + optional **AVX2** kernels for L2 and cosine
- multi-core query-batch execution
- persistent HNSW index save/load
- earliest-deadline-first queue
- SLA-aware adaptive micro-batching
- EWMA latency model by batch size
- exact-vs-HNSW Recall@K/QPS benchmark
- machine-readable JSON benchmark output
- deterministic online traffic simulator with p50/p95/p99, SLA misses, batching and queue-depth metrics
- measured batch-latency profile generation/loading for hardware-backed scheduler calibration
- scalar + AVX2 GitHub Actions matrix
- ASan + UBSan CI

```text
batch=1 clients -> EDF queue -> adaptive micro-batcher -> SLA router
                                                    /           \
                                      exact/HNSW + SIMD      cuVS/CAGRA
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Enable AVX2:

```bash
cmake -S . -B build-avx2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DWARPGATE_ENABLE_AVX2=ON
```

## Benchmarks

```bash
./build/warpgate_ann_bench 20000 128 200 l2
./build/warpgate_ann_bench 20000 128 200 cosine --json
```

Output records rows, dimension, metric, active kernel, HNSW build time, exact QPS, HNSW QPS, and Recall@10. Synthetic CPU results are regression data, not NVIDIA/GPU marketing claims.

## Persistence

```cpp
warpgate::HnswBackend index(vectors, rows, dim, config);
index.save("index.wghnsw");
auto restored = warpgate::HnswBackend::load("index.wghnsw");
```

The current binary format is versioned (`WGHNSW01`) and intended for WarpGate-controlled deployments.

## Online traffic simulation

```bash
./build/warpgate_traffic_sim 5000 steady 20000
./build/warpgate_traffic_sim 5000 bursty 20000 --json
./build/warpgate_traffic_sim 5000 mixed-sla 12000 --json
```

The simulator replays the same trace through immediate, fixed-batch, and adaptive policies. By default its service curve is synthetic and is **not a GPU benchmark**. WarpGate can also consume a measured batch-latency CSV.

```bash
./build/warpgate_batch_profile 5000 128 32 profile.csv
./build/warpgate_traffic_sim 5000 steady 12000 --json profile.csv
```

Today the profiler targets the CPU HNSW backend; the same calibration path is intended for real cuVS/CAGRA measurements. See [docs/ONLINE_SIMULATION.md](docs/ONLINE_SIMULATION.md).

## NVIDIA path

`SearchBackend` is the adapter boundary for cuVS/CAGRA. The hardware experiment must compare immediate batch=1 CAGRA, concurrent CUDA streams, and WarpGate micro-batching under the **same recall target and p99 SLA**. No GPU performance claim belongs in this repository until reproduced on real NVIDIA hardware.

See [docs/CUDA_CUVS.md](docs/CUDA_CUVS.md).

## Status

- [x] exact CPU baseline
- [x] HNSW from scratch
- [x] L2 + cosine
- [x] AVX2
- [x] multi-core batch execution
- [x] index persistence
- [x] EDF scheduler
- [x] adaptive latency model
- [x] recall/QPS + JSON benchmark
- [x] sanitizer CI
- [x] p50/p95/p99 online traffic simulation
- [x] steady/bursty/mixed-SLA trace replay
- [x] measured batch-latency profiles + CSV calibration
- [ ] cuVS/CAGRA adapter on NVIDIA hardware
- [ ] CUDA stream pool + pinned memory
- [ ] multi-GPU routing
- [ ] NIXL / GPUDirect experiments
- [ ] upstream benchmark/PR backed by measurements

## Benchmark integrity

Always record hardware, driver/CUDA/cuVS versions, dataset, dimension, metric, top-k, arrival pattern, recall, p50/p95/p99, QPS, memory, and utilization.

## License
MIT
