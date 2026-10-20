---
scripts:
  - scripts/tpu-performance-optimization/llmd_telemetry.py
  - scripts/tpu-model-onboarding/llmd/parse_bench_results.py
  - scripts/tpu-model-onboarding/llmd/benchmark.py
references:
  - references/tpu-performance-optimization/SERVING_BENCHMARK.md
  - references/tpu-model-onboarding/llmd/distributed-serving-benchmark.md
---

# Distributed Serving Telemetry Contract (llm-d and Dynamo)

This reference defines the telemetry contract, metrics schema, Prometheus counter mappings, and aggregation workflow for distributed LLM serving on Cloud TPU and GPU clusters using `{{schema:llmd-telemetry.json}}` and `{{script:tpu-performance-optimization/llmd_telemetry.py}}`.

In distributed serving architectures (such as GKE Cloud TPU `llm-d` or NVIDIA Dynamo), single-worker loopback metrics are insufficient to characterize system efficiency. Comprehensive optimization studies require capturing the interaction between cluster routing policies (for example, prefix cache scoring and queue-based load spreading), accelerator HBM KV cache utilization, host DRAM KV offloading, and request-level tail latencies under service-level agreement (SLA) targets.

## Telemetry Schema Architecture

Every executed benchmark arm produces a canonical telemetry payload validating against `{{schema:llmd-telemetry.json}}`. The schema is organized into five functional sections:

### Request Latency Distributions

Latency percentiles are tracked in milliseconds across four key phases of request lifecycle:

- `ttft_ms`: Time To First Token latency covering p50, p90, p95, p99, mean, and median. Evaluates prompt prefill efficiency, routing stickiness, and prefix cache reuse.
- `tpot_ms`: Time Per Output Token latency covering mean, median, p90, and p99. Captures token generation speed and memory bandwidth saturation during the decode phase.
- `itl_ms`: Inter-Token Latency covering mean and p99. Measures token chunk pacing consistency and scheduler bubble overhead.
- `e2e_ms`: End-to-end request turnaround duration covering p50, p90, p99, mean, and median. Captures total client perceived latency from dispatch to completion.

### Throughput and Goodput Under SLA

Serving throughput is separated into raw volumes and SLA-constrained goodput:

- `request_throughput_rps`: Completed requests per second.
- `output_token_throughput_tps`: Generated decode tokens per second across all active workers.
- `total_token_throughput_tps`: Total prompt prefill plus decode output tokens processed per second.
- `goodput`: Structured evaluation against target SLA bounds (`sla_ttft_target_ms`, `sla_tpot_target_ms`, `sla_e2e_target_ms`), reporting the exact `requests_meeting_sla` count and the overall `goodput_percentage` (0.0 to 100.0).

### Cache and Memory Hierarchy

Tracks accelerator high-bandwidth memory (HBM) KV cache and prefix cache utilization:

- `kv_cache`: Tracks `usage_factor_mean` and `usage_factor_peak` (0.0 to 1.0) along with `total_blocks` and `free_blocks` allocated across accelerator devices.
- `prefix_cache`: Captures `hit_rate_tokens` (cached tokens divided by total prompt tokens) and `hit_rate_requests` (requests finding at least one cached block), along with query and hit counts.

### System Reliability and Queuing Dynamics

Monitors error rates, HTTP status breakdowns, and queue wait times:

- `error_rate` and `timeout_rate`: Fraction of dispatched requests that failed or timed out (0.0 to 1.0).
- `http_status_codes`: Status code frequency map (for example, counts of 200, 429, 500, 503 responses).
- `queue`: Tracks router queue depth (`router_queue_depth_mean`, `router_queue_depth_max`), client wait time before worker execution (`waiting_time_ms_mean`, `waiting_time_ms_p99`), and active worker execution time (`execution_time_ms_mean`, `execution_time_ms_p99`).

### Arm Metadata

Captures experiment lineage and hardware provenance:

- `arm_id`: Unique identifier of the tuning arm (for example, `baseline`, `arm-001`).
- `scenario`: Benchmark profile executed (`inferencex-agentx-mvp`, `mooncake_trace`, `guidellm`, `custom`).
- `model`: Served model repository or name (for example, `Qwen/Qwen3-32B`).
- `accelerator_type`: Chip architecture under test (for example, `v6e`, `v5e`, `v7x`).
- `replica_topology`: Serving layout (`mode`, `prefill_replicas`, `decode_replicas`, `total_workers`, `chips_per_replica`, `topology_shape`).
- `routing_algorithm`: Routing policy or scoring plugin applied at the gateway (`precise-prefix-cache-scorer`, `kv-cache-utilization-scorer`, `round-robin`).
- `timestamp`: UTC ISO-8601 execution timestamp.

## Prometheus Counter and Gauge Definitions

Telemetry aggregation ingests metrics exposed by the serving engine and the routing proxy:

| Metric Name | Type | Source Component | Description |
|---|---|---|---|
| `vllm:num_requests_running` | Gauge | Worker Model Server | Number of requests currently executing on the accelerator. |
| `vllm:num_requests_waiting` | Gauge | Worker Model Server | Requests queued on worker awaiting KV block allocation. |
| `vllm:gpu_cache_usage_factor` | Gauge | Worker Model Server | Fraction of accelerator HBM KV cache currently allocated. |
| `vllm:kv_cache_usage_perc` | Gauge | Worker Model Server | Percentile KV cache occupancy across worker instances. |
| `vllm:prefix_cache_hits_total` | Counter | Worker Model Server | Total prompt tokens matched in prefix cache blocks. |
| `vllm:prefix_cache_queries_total` | Counter | Worker Model Server | Total prompt tokens queried against prefix cache index. |
| `vllm:host_memory_evicted_bytes_total` | Counter | Worker Model Server | Total bytes of KV cache blocks evicted to host DRAM. |
| `vllm:host_memory_reloaded_bytes_total` | Counter | Worker Model Server | Total bytes of KV cache blocks reloaded from host DRAM. |
| `vllm:kv_transfer_duration_seconds_bucket` | Histogram | Worker Model Server | Latency buckets for host-to-device and device-to-host KV transfers. |
| `vllm:pcie_bandwidth_utilization` | Gauge | Worker Model Server | Host-to-device bus utilization fraction during KV offload. |
| `vllm:request_queue_time_seconds_bucket` | Histogram | Worker Model Server | Time requests spent waiting in worker queue before prefill. |
| `vllm:request_inference_time_seconds_bucket` | Histogram | Worker Model Server | Time spent in active prefill and decode execution. |
| `llmd_router_queue_depth` | Gauge | Endpoint Picker Proxy | Instantaneous request queue depth at the routing gateway. |
| `llmd_router_request_latencies_bucket` | Histogram | Endpoint Picker Proxy | End-to-end turnaround latency measured at the router ingress. |
| `llmd_router_waiting_time_seconds_bucket` | Histogram | Endpoint Picker Proxy | Request dispatch queuing latency inside the router. |
| `sglang:cached_tokens` | Counter | SGLang Worker | Cached prompt tokens reused via RadixAttention. |
| `sglang:prompt_tokens` | Counter | SGLang Worker | Total prompt tokens processed by SGLang worker. |
| `sglang:token_usage` | Gauge | SGLang Worker | Current KV token memory usage fraction in SGLang. |

## Scraping and Aggregation Mechanisms

The telemetry scraper `{{script:tpu-performance-optimization/llmd_telemetry.py}}` accepts both benchmark client logs and server Prometheus feeds:

```bash
python3 {{script:tpu-performance-optimization/llmd_telemetry.py}} \
  --benchmark-output <work>/profile_export_aiperf.json \
  --prometheus-metrics <work>/server_metrics_export.json \
  --output-json <work>/telemetry.json \
  --arm-id arm-001 \
  --scenario inferencex-agentx-mvp \
  --model Qwen/Qwen3-32B \
  --accelerator-type v6e \
  --sla-ttft-ms 800.0 \
  --sla-tpot-ms 25.0 \
  --validate
```

### Live Prometheus Ingestion Across Worker Roles

When scraping live Kubernetes pods via HTTP URLs, note the port separation in disaggregated deployments:

- Prefill pods (`llm-d.ai/role=prefill`): Expose Prometheus metrics on port 8000 (`http://<pod-ip>:8000/metrics`).
- Decode pods (`llm-d.ai/role=decode`): The native sidecar proxy occupies port 8000, while the underlying model server container exposes Prometheus metrics on port 8200 (`http://<pod-ip>:8200/metrics`).
- Homogeneous pods: Listen directly on port 8000 (`http://<pod-ip>:8000/metrics`).

Multiple endpoints are passed to `{{script:tpu-performance-optimization/llmd_telemetry.py}}` as a comma-separated list via `--prometheus-metrics`. The scraper aggregates counters additively, evaluates peak and mean occupancy across gauges, and calculates worker load skew ratios.

## Downstream Optimization Agent Integration

The telemetry contract serves as the data foundation for automated performance optimization studies:

### Executor Arm Verification

During each study round, `tpu-performance-optimization-executor` executes an arm hypothesis, gathers client and server telemetry, and runs `{{script:tpu-performance-optimization/llmd_telemetry.py}} --validate` to generate `<work>/telemetry.json`. This contract guarantees that latency percentiles and goodput numbers are standardized before entering `arm-result.json`.

### Researcher Bottleneck Decomposition

The `tpu-performance-optimization-researcher` inspects `<work>/telemetry.json` to attribute performance bottlenecks:

- Cache Capacity Bottlenecks: When `kv_cache.usage_factor_peak` reaches 1.0 or `vllm:num_requests_waiting` spikes, the researcher recommends adjusting max batch size or enabling host DRAM offloading.
- Routing and Affinity Imbalance: When `worker_load_skew_ratio` exceeds 1.25x or prefix cache hit rate falls below baseline, the researcher generates hypotheses adjusting `EndpointPickerConfig` scorer weights (`precise-prefix-cache-scorer` vs `kv-cache-utilization-scorer`).
- Host DRAM Transfer Overhead: When `host_memory_offload.transfer_latency_ms.p99` elevates TTFT tail latency, the researcher tunes offload chunking thresholds or prefetching parameters.
- SLA Compliance Gates: The researcher uses `goodput.goodput_percentage` as the primary admissibility gate before qualifying an arm as a win against the baseline.
