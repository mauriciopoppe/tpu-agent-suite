---
scripts:
  - scripts/tpu-model-onboarding/llmd/benchmark.py
  - scripts/tpu-model-onboarding/llmd/parse_bench_results.py
---

# Distributed Serving Benchmark Workflow (llm-d and Dynamo)

This reference defines the benchmarking workflow for distributed serving deployments across **NVIDIA Dynamo** (`DynamoGraphDeployment`) and **GKE Cloud TPU `llm-d`** (`Gateway` + `EndpointPickerConfig` + `InferencePool`, in both homogeneous `optimized-baseline` and disaggregated `pd-disaggregation` modes).

Unlike single-replica local loopback benchmarks (`http://127.0.0.1:8000` driven by `vllm bench serve` or `sglang.bench_serving`), distributed serving benchmarks must exercise the **cluster-level routing control plane** (Dynamo's `Frontend` KV-router or `llm-d`'s Endpoint Picker `precise-prefix-cache-scorer`, `kv-cache-utilization-scorer`, and `queue-scorer`) while simultaneously scraping **per-worker Prometheus `/metrics`** from every prefill and decode pod.

## Supported Benchmark Harnesses and When to Use Each

Use `{{script:tpu-model-onboarding/llmd/benchmark.py}}` to generate the benchmark configuration and Kubernetes `batch/v1` `Job` manifest, and `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}` to parse client latency, token throughput, per-worker prefix cache hit rate, KV cache occupancy, load skew, and SLA compliance.

| Harness (`--harness`) | Primary Use Case | Dataset / Workload Synthesis | Router & Prefix Cache Coverage |
|---|---|---|---|
| `inference-perf` | Staged constant-rate saturation sweep (`25%` / `50%` / `85%` of cluster QPS) with synthetic shared prefixes | Native `shared_prefix` (`32` groups x `32` prompts) when `prefix_len` is set | Exercises `precise-prefix-cache-scorer` without external trace downloads |
| `aiperf-agentx` | Multi-turn coding agent trace replay (`--scenario inferencex-agentx-mvp`) for end-to-end Dynamo vs. `llm-d` comparison | Multi-turn session affinity headers + per-worker `--server-metrics` | Full multi-turn session affinity headers + per-worker `--server-metrics` |
| `aiperf-mooncake` | High-KV-reuse agentic trace replay (`--custom-dataset-type mooncake_trace`) | Multi-turn `session_id` and `--server-metrics` | Multi-turn `session_id` + `--server-metrics` |

## Distributed Topology Discovery and Endpoint Contract

Before launching a benchmark run, resolve both the **entry-point HTTP URL** (`--url` / `target_url`) and the **per-worker Prometheus `/metrics` endpoints** (`--server-metrics`) for the active serving stack.

### Entry-Point Routing Target (`target_url`)

- **NVIDIA Dynamo (`--stack dynamo`)**: Target the `Frontend` Kubernetes Service at `http://<release-name>-frontend:8000`. When running inside the cluster, `{{script:tpu-model-onboarding/llmd/benchmark.py}}` automatically attaches a `podAffinity` rule (`nvidia.com/dynamo-component-type=frontend`, `nvidia.com/dynamo-graph-deployment-name=<release-name>`) on `topologyKey: kubernetes.io/hostname` so the benchmark client pod is co-located on the same node as the Dynamo frontend to eliminate cross-node client hop jitter.
- **GKE Cloud TPU `llm-d` (`--stack llm-d`)**: Target the Kubernetes Gateway Service at `http://<release-name>-inference-gateway:80` (default `gateway` mode) or the standalone Endpoint Picker proxy at `http://<release-name>-epp:9002` (`standalone` mode). Never point a distributed benchmark directly at an individual worker pod IP, as doing so bypasses `EndpointPickerConfig` scoring and disaggregated prefill/decode routing.

### Per-Worker Prometheus `/metrics` Port Contract (`8000` vs. `8200`)

Passing `--server-metrics` to `aiperf profile` instructs the harness to scrape Prometheus counters and gauges from every worker pod before, during, and after the run, emitting `<bench_work>/server_metrics_export.json` alongside `<bench_work>/profile_export_aiperf.json`.

- **NVIDIA Dynamo Workers (`--stack dynamo`)**: Select worker pods with `-l nvidia.com/dynamo-graph-deployment-name=<release-name>,nvidia.com/dynamo-component-type=worker`. Every worker pod exposes `/metrics` on port `8000` (`http://<pod-ip>:8000/metrics`).
- **GKE Cloud TPU `llm-d` Homogeneous Workers (`--stack llm-d`, `optimized-baseline`)**: Select worker pods with `-l llm-d.ai/inferenceServing=true`. Every worker pod exposes vLLM `/metrics` on port `8000` (`http://<pod-ip>:8000/metrics`).
- **GKE Cloud TPU `llm-d` Disaggregated Workers (`--stack llm-d --disagg`, `pd-disaggregation`)**:
  - **Prefill pods** (`-l llm-d.ai/inferenceServing=true,llm-d.ai/role=prefill`) run the `modelserver` container directly on port `8000` and expose Prometheus metrics at `http://<pod-ip>:8000/metrics`.
  - **Decode pods** (`-l llm-d.ai/inferenceServing=true,llm-d.ai/role=decode`) run the `llm-d-routing-sidecar` (`routing-proxy`) native sidecar on port `8000` while the underlying `modelserver` vLLM container listens on `--port 8200`. You must scrape decode pod metrics on port **`8200`** (`http://<pod-ip>:8200/metrics`).

Generate the exact worker discovery command for your topology with `{{script:tpu-model-onboarding/llmd/benchmark.py}}`:

```bash
python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
  --model Qwen/Qwen3-32B \
  --stack llm-d \
  --release-name llm-d \
  --namespace default \
  --disagg \
  --emit discovery-cmd
```

## Multi-Turn Session Affinity and KV Cache Hygiene

Agentic workloads (`aiperf-agentx` and `aiperf-mooncake`) replay multi-turn sessions where each follow-up turn appends tool outputs and assistant reasoning onto the previous turn's prompt prefix.

### Session Affinity Headers

When `{{script:tpu-model-onboarding/llmd/benchmark.py}}` builds an `aiperf-agentx` or `aiperf-mooncake` Job, it injects the session-affinity environment variables into the benchmark container so `aiperf` propagates each session's correlation ID across HTTP headers recognized by Dynamo, SGLang Model Gateway (SMG), and `llm-d`:

```yaml
env:
  - name: AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID
    value: "1"
  - name: AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID
    value: "1"
  - name: AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID
    value: "1"
  - name: AIPERF_DATASET_CONFIGURATION_TIMEOUT
    value: "3600"
  - name: AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT
    value: "3600"
  - name: AIPERF_HTTP_CONNECTION_LIMIT
    value: "2500"
```

### Flushing Prefix Caches Between Concurrency Sweep Points

When sweeping multiple concurrency points (for example `c=8, 16, 24`) or comparing two `EndpointPickerConfig` scoring profiles on a warm cluster, residual blocks in the vLLM or SGLang prefix cache from run `N` will artificially inflate the prefix cache hit rate and lower TTFT on run `N+1`. Always flush worker prefix caches between runs using the commands emitted by `--emit cache-reset-cmd`:

```bash
# Emit and run in-place HTTP prefix cache reset across all worker pods
# (uses /reset_prefix_cache for vLLM or /flush_cache for SGLang, respecting port 8200 on disagg decode pods)
python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
  --model Qwen/Qwen3-32B \
  --stack llm-d \
  --release-name llm-d \
  --namespace default \
  --disagg \
  --emit cache-reset-cmd
```

If the serving container does not enable the administrative cache reset endpoint (`VLLM_SERVER_DEV_MODE=1`), re-run `--emit cache-reset-cmd` and poll `/v1/models` until all replicas report Ready before starting the next concurrency point.

## Running Benchmarks Across Harnesses

Always invoke `{{script:tpu-model-onboarding/llmd/benchmark.py}}` with `--execute`, `--logs-out <serving_work>/bench-logs.txt`, `--parse-out <serving_work>/bench-results.json`, and `--evidence-dir <serving_work>/evidence` so the script submits the benchmark `ConfigMap` and `Job` to the target namespace, waits for completion (`--timeout <seconds>`, default `900`), writes the benchmark pod logs, parses the structured telemetry summary, and captures post-benchmark cluster and replica traffic distribution evidence automatically.

### Mode 1: AgentX Multi-Turn Benchmark (`aiperf-agentx`)

The `aiperf-agentx` harness runs the `inferencex-agentx-mvp` scenario against `/v1/chat/completions` with `--streaming`, `--use-server-token-count`, `--burst-phase-starts`, `--extra-inputs ignore_eos:true`, and `--random-seed 42`:

- **Rule-locked benchmark mode (default)**: Runs for `--benchmark-duration 1800` (30 minutes) without `--unsafe-override`. Use this mode when recording baseline vs. onboarded TPU comparison numbers.
- **Smoke override mode (`--smoke`)**: Appends `--unsafe-override --num-dataset-entries 12 --benchmark-duration 120` for rapid 2-minute routing and disaggregation verification during iterative bringup. `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}` marks smoke runs as `submission_valid: false` in `<serving_work>/bench-results.json` so smoke numbers are never mistaken for full runs.
- **Context length selection**: Models with `model_context_length <= 262144` (256K) automatically select `semianalysis_cc_traces_weka_062126_256k`; larger context windows select `semianalysis_cc_traces_weka_062126`.

```bash
# Execute a smoke AgentX benchmark Job on GKE Cloud TPU llm-d and capture logs, parsed results, and cluster evidence
python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
  --model Qwen/Qwen3-32B \
  --harness aiperf-agentx \
  --stack llm-d \
  --release-name llm-d \
  --namespace default \
  --smoke \
  --concurrency 4 \
  --ttft-sla 2000 --tpot-sla 40 \
  --execute \
  --timeout 900 \
  --logs-out <serving_work>/bench-logs.txt \
  --parse-out <serving_work>/bench-results.json \
  --evidence-dir <serving_work>/evidence
```

Pass `--kubeconfig <path>` and `--context <ctx>` when targeting a non-default Kubernetes context.

### Mode 2: Mooncake Agentic Trace Replay (`aiperf-mooncake`)

The `aiperf-mooncake` harness mirrors the `schemaVersion: "2.0"` `perf.yaml` recipe pattern from NVIDIA Dynamo (`recipes/*/perf/perf.yaml`). The generated Job automatically downloads and verifies the SHA256 checksum (`f20d3f2bc83dd1306cda659fbe34e7c4d85ca5497626c98bc0b1c4d2211379d0`) of `64k_400_90kv_agent_new_noschedule_short_15perc.jsonl` into `/model-cache/traces/`, executes a warmup pass (`--num-requests 32`), and runs the main `mooncake_trace` profile with `--export-http-trace`.

```bash
python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
  --model Qwen/Qwen3-32B \
  --harness aiperf-mooncake \
  --stack llm-d \
  --release-name llm-d \
  --namespace default \
  --disagg \
  --concurrency 16 \
  --execute \
  --timeout 900 \
  --logs-out <serving_work>/bench-logs.txt \
  --parse-out <serving_work>/bench-results.json \
  --evidence-dir <serving_work>/evidence
```

When testing custom multi-turn session inter-turn think times or length-aware decode scheduling (`agent_hints.osl`) in Python, call `prepare_mooncake_trace(records, delay_override_ms=..., add_expected_osl=True)` from `{{script:tpu-model-onboarding/llmd/benchmark.py}}`.

### Mode 3: Staged Shared-Prefix Sweep (`inference-perf`) and `guidellm`

When invoked with `--harness inference-perf` or `--harness guidellm`, `{{script:tpu-model-onboarding/llmd/benchmark.py}}` synthesizes a warmup stage (`rate=1, duration=30s`) followed by three constant-rate SLA validation stages at `25%`, `50%`, and `85%` of estimated cluster capacity. For `high-prefix-share` workloads with `prefix_len`, always prefer `inference-perf` over `guidellm` so the `shared_prefix` generator exercises `precise-prefix-cache-scorer`.

```bash
python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
  --model Qwen/Qwen3-32B \
  --harness inference-perf \
  --stack llm-d \
  --release-name llm-d \
  --namespace default \
  --execute \
  --timeout 900 \
  --logs-out <serving_work>/bench-logs.txt \
  --parse-out <serving_work>/bench-results.json \
  --evidence-dir <serving_work>/evidence
```

## Collecting Logs, Cluster Evidence, and Parsing Client + Server Telemetry

When `--execute` is passed alongside `--logs-out <serving_work>/bench-logs.txt`, `--parse-out <serving_work>/bench-results.json`, and `--evidence-dir <serving_work>/evidence`, `{{script:tpu-model-onboarding/llmd/benchmark.py}}` writes:

- `<serving_work>/bench-logs.txt`: Complete stdout/stderr logs from the completed benchmark pod.
- `<serving_work>/bench-results.json` and `<serving_work>/bench-results.json.md`: Structured JSON summary and Markdown report produced by `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}`.
- `<serving_work>/evidence/target_cluster_state.txt`: Wide cluster resource inventory (`LeaderWorkerSet`, `Deployment`, `Pod`, `Service`, `Workload`, `ProvisioningRequest`, and `Job`) in the target namespace.
- `<serving_work>/evidence/traffic_distribution.json`: Per-pod counts of `POST /v1/completions` and `POST /v1/chat/completions` across all serving worker replicas plus percentage traffic split (`replica_split_pct`).
- `<serving_work>/evidence/worker_pod_logs/<pod_name>.log`: Startup patch verification and serving logs from each worker pod.
- `<serving_work>/evidence/epp_router.log`: Endpoint Picker (`llm-d-epp`) routing logs.

To re-evaluate saved benchmark logs or an extracted artifact directory against updated SLA thresholds, invoke `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}`:

```bash
python3 {{script:tpu-model-onboarding/llmd/parse_bench_results.py}} \
  --logs <serving_work>/bench-logs.txt \
  --ttft-sla 800 \
  --tpot-sla 25 \
  --e2e-sla 5000 \
  --prefix-hit-rate-min 0.70 \
  --max-context-overflow-pct 0.0 \
  --summary-json <serving_work>/bench-results.json
```

You can also parse an extracted directory containing `<serving_work>/profile_export_aiperf.json` and `<serving_work>/server_metrics_export.json` directly using `--artifacts-dir <serving_work>/`:

```bash
python3 {{script:tpu-model-onboarding/llmd/parse_bench_results.py}} \
  --artifacts-dir <serving_work>/ \
  --ttft-sla 800 \
  --tpot-sla 25 \
  --summary-json <serving_work>/bench-results.json
```

### Diagnostic Interpretation of Combined Client + Server Metrics

| Signal in `<serving_work>/bench-results.json` | Expected Healthy Range | Root Cause When Out of Range |
|---|---|---|
| `server_metrics.prefix_cache_hit_rate` (`vllm:prefix_cache_hits / vllm:prefix_cache_queries` or `sglang:cached_tokens / sglang:prompt_tokens`) | `>= 0.70` on `high-prefix-share`, `aiperf-agentx`, and `aiperf-mooncake` | Session-affinity headers missing, `precise-prefix-cache-scorer` weight too low, or worker HBM KV pool evicting cached prefixes (`kv_cache_usage_p95 > 0.95`) |
| `server_metrics.worker_load_skew_ratio` (`max / mean` requests across workers) | `1.0x - 1.35x` | `precise-prefix-cache-scorer` pinning hot sessions to a single replica without sufficient `queue-scorer` / `kv-cache-utilization-scorer` counterweight |
| `server_metrics.kv_cache_usage_p95` (`vllm:kv_cache_usage_perc`) | `<= 0.90` | Insufficient total replica count or `gpu-memory-utilization` set too low for target concurrency |
| `agentx.context_overflow_pct` | `0.0%` | Serving engine `--max-model-len` is smaller than the dataset trace window (`256K` vs. `1M`), or chat template extra tokens overflow context |
| Exit code of `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}` | `0` (all SLAs met) | `1` indicates an SLA breach (tune EPP scorer weights, disaggregation ratio, or engine flags); `2` indicates missing delimiter or malformed JSON |
