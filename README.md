# WarpGate

**SLA-aware CPU/GPU runtime for real-time vector search.**

WarpGate targets a production mismatch: clients often submit vector queries one at a time while accelerators are most efficient with batches. The runtime learns backend latency, orders work by deadline, forms micro-batches only when SLA budget allows, and routes across heterogeneous search backends.

> **v0.2 CPU production core:** HNSW + AVX2 are implemented and tested. CUDA/cuVS is the hardware-backed next adapter; no GPU performance is claimed without real NVIDIA measurements.

## Implemented

- C++20
- exact L2 reference backend
- hierarchical **HNSW from scratch**
- scalar + optional **AVX2** distance kernels
- earliest-deadline-first queue
- SLA-aware adaptive micro-batching
- EWMA latency model by batch size
- exact-vs-HNSW Recall@K/QPS benchmark
- scalar + AVX2 GitHub Actions matrix
- ASan + UBSan CI

```text
batch=1 clients -> EDF queue -> adaptive micro-batcher -> SLA router
                                                    /           \
                                            exact/HNSW+SIMD  cuVS/CAGRA
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

AVX2: add `-DWARPGATE_ENABLE_AVX2=ON`.

## Benchmarks

```bash
./build/warpgate_bench 50000 128 200
./build/warpgate_ann_bench 20000 128 200
```

The ANN benchmark reports build time, exact QPS, HNSW QPS, Recall@K, and active distance kernel. Synthetic CPU results are regression data, not marketing claims.

## NVIDIA path

`SearchBackend` is the adapter boundary for cuVS/CAGRA. The required hardware experiment compares immediate batch=1 CAGRA, concurrent streams, and WarpGate micro-batching under the **same recall target and p99 SLA**. See [docs/CUDA_CUVS.md](docs/CUDA_CUVS.md).

## Status

- [x] exact CPU baseline
- [x] HNSW from scratch
- [x] AVX2
- [x] EDF scheduler
- [x] adaptive latency model
- [x] recall/QPS benchmark
- [x] sanitizer CI
- [ ] cuVS/CAGRA on NVIDIA hardware
- [ ] CUDA stream pool + pinned memory
- [ ] multi-GPU routing
- [ ] NIXL / GPUDirect experiments
- [ ] upstream benchmark/PR backed by measurements

**Benchmark integrity:** record hardware, driver/CUDA/cuVS versions, dataset, dimension, top-k, arrival pattern, recall, p50/p95/p99, QPS, and memory/utilization.

## License
MIT
