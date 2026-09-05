# WarpGate Architecture

WarpGate is a deadline-aware runtime that sits in front of heterogeneous vector-search backends.

```text
clients
   |
   v
request queue ---> deadline guard
   |                  |
   +------> adaptive micro-batcher
                    |
                    v
               SLA scheduler
               /          \
              v            v
         CPU backend   GPU backend
        HNSW/SIMD      cuVS/CAGRA
              \            /
               v          v
              metrics + EWMA
                    |
                    +--> continuously updated latency model
```

## Core invariant

A query may wait only when the predicted benefit from a larger batch does not violate its latency budget.

For a query with deadline `D` at time `t`:

`wait_budget = D - t - predicted_execution_time - safety_margin`

The runtime can micro-batch only while `wait_budget > 0`.

## v0.1

- FIFO request queue
- per-backend batch latency model using EWMA buckets
- deadline guard
- adaptive backend selection
- exact CPU reference backend
- reproducible scheduler microbenchmark

## v0.2

- earliest-deadline-first queue
- throughput-aware scoring, not latency-only scoring
- load feedback and queue-depth prediction
- Prometheus metrics

## v0.3 — NVIDIA path

- cuVS/CAGRA backend
- CUDA streams
- pinned host buffers
- asynchronous H2D/D2H
- real batch-size-1 vs micro-batched online benchmark
- Nsight Systems / Nsight Compute profiles

## v0.4

- multi-GPU routing
- per-GPU queue model
- memory-pressure-aware placement
- index sharding / replication policies

## v0.5

- NIXL-based transfer path
- GPUDirect Storage experiments
- GPUDirect RDMA experiments where hardware is available
