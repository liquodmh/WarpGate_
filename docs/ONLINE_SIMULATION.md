# Online Traffic Simulation

WarpGate v0.4 adds a deterministic discrete-event simulator for scheduler policy experiments **before** NVIDIA hardware is attached.

This is intentionally **not** a GPU benchmark. The simulator uses a documented synthetic service curve:

`service_us(batch) = launch_overhead + per_query * batch^exponent`

The default model is meant to represent a backend with non-trivial launch overhead and sub-linear batch cost. It exists to test scheduling behavior, not to predict a specific GPU.

## Workloads

- `steady`: regular batch=1 arrivals.
- `bursty`: groups of requests arrive together, followed by idle gaps.
- `mixed-sla`: deterministic seeded traffic with tight, medium, and relaxed deadlines.

## Policies

- `immediate`: execute every request as batch=1.
- `fixed`: batch toward a fixed target with a maximum wait.
- `adaptive`: use EDF order, deadline pressure, a safety margin, maximum wait, and the service model to decide whether to wait or dispatch.

## Metrics

Every run reports:

- throughput QPS
- p50 / p95 / p99 / max latency
- SLA miss rate
- mean and maximum batch size
- maximum queue depth
- makespan

## Run

```bash
./build/warpgate_traffic_sim 5000 steady 20000
./build/warpgate_traffic_sim 5000 bursty 20000 --json
./build/warpgate_traffic_sim 5000 mixed-sla 12000 --json
```

The same trace is replayed through all policies so comparisons are apples-to-apples. Real NVIDIA claims must replace the synthetic service model with measured cuVS/CAGRA timings and end-to-end CUDA execution.

## Measured service profiles

The simulator can replace the synthetic service curve with measured batch latency:

```bash
./build/warpgate_batch_profile 5000 128 32 profile.csv
./build/warpgate_traffic_sim 5000 steady 12000 --json profile.csv
```

Profile format:

```csv
batch_size,latency_us
1,210.5
2,235.1
4,278.9
8,340.2
```

WarpGate linearly interpolates between measured points and uses a conservative proportional extrapolation beyond the largest point. This makes the simulator ready to consume real cuVS/CAGRA batch measurements later without changing scheduling code.
