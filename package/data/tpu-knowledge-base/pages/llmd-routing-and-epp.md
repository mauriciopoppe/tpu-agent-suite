<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: llmd-routing-and-epp
keyword: endpointpicker
url: not applicable -- see Evidence
authority: first-party-verified
retrieved: 2026-10-12
source_updated: not applicable
generator: authored
---

# LLM-d Routing and EndpointPicker (EPP)

Distributed serving orchestration in llm-d separates traffic routing from accelerator execution. An external processor (ext-proc) plugin, known as the EndpointPicker (EPP), runs inside the Envoy ingress gateway or Kubernetes Gateway API to make latency-aware and cache-aware dispatch decisions across model server replicas.

## Architecture and Request Lifecycle

The routing path follows a multi-tier structure that decouples network ingress from model execution. External inference requests arrive at the ingress gateway (Envoy L4/L7). Before forwarding the payload to a backend worker, the gateway passes request metadata and prompt tokens to the EPP filter over gRPC. The EPP evaluates candidate workers in the target InferencePool, selects the optimal replica, and annotates the request headers. The request is then forwarded directly to the selected model server replica (such as vLLM or SGLang running behind a local routing proxy).

## EPP Scoring Pipeline and Scheduling Algorithms

EPP evaluates candidate endpoints using a composable scoring and filtering pipeline configured via EndpointPickerConfig. The total score for an endpoint replica p is computed as a weighted linear combination of individual scoring plugins: TotalScore(p) = sum(w_i * Scorer_i(p)).

The prefix-cache-scorer calculates MatchRatio = MatchedBlocks / TotalBlocks (evaluating to 0.0 when TotalBlocks is 0). It prioritizes routing prompts to workers that already hold matching KV cache blocks in memory. For multi-turn conversational agents or shared system prompts where cache hits exceed 80%, this scorer dramatically cuts Time to First Token (TTFT).

The token-load-scorer calculates Score(p) = 1.0 - min(InFlightTokens(p) + UncachedTokens(p), T_thresh) / T_thresh. Rather than balancing raw request counts, this scorer accounts for the actual computational weight of active requests, preventing worker overload when handling long sequences.

The queue-scorer evaluates Score(p) = (Q_max - Q(p)) / (Q_max - Q_min), where Q(p) is the instantaneous pending queue depth reported by the replica. It rapidly sheds traffic away from nodes experiencing deep queues or temporary scheduling delays.

The kv-cache-utilization-scorer evaluates Score(p) = 1.0 - KVCacheUsagePercent(p). This serves as a memory safety valve, steering traffic away from workers nearing HBM capacity to prevent page evictions or out-of-memory aborts.

The no-hit-lru-scorer rotates cold prompts across least-recently-used replicas during cache misses, distributing initialization load evenly across the cluster.

## Queue-Depth Thresholding and Utilization Filtering

To prevent head-of-line blocking and catastrophic tail latency spikes, EPP incorporates hard filtering mechanisms alongside soft scoring:

The utilization-filter evaluates configurable threshold conditions (such as waiting-queue, active-requests, or kv-cache-utilization). When an endpoint's waiting queue exceeds a defined limit (for example, waiting-queue > 0 or max_queue_depth > 16), the endpoint is pruned from the candidate set before scoring occurs. Pruning saturated endpoints avoids upstream 429 load-shedding and queue buildup.

The prefix-cache-affinity-filter classifies candidate endpoints into Sticky (MatchRatio >= affinityThreshold, typically 0.80) and NonSticky groups. It predicts queue delay for sticky endpoints as Delay_ms = (InFlightTokens(p) / PeakPrefillThroughput) * 1000. If the predicted delay on the warm replica is smaller than the cost of re-computing the full prompt on a cold replica, the request remains pinned to the sticky worker. If the queue delay exceeds maxTTFTPenaltyMs, traffic spills over to an idle replica.

## Session Affinity and Headers

Multi-turn chat sessions and iterative reasoning workflows benefit from session-level affinity:

Session identifiers provided in request headers (such as `x-session-id`, `x-llmd-session-id`, or user context cookies) allow EPP to establish sticky routing to the replica servicing prior turns of the same conversation.

Sticky routing ensures consecutive turns reuse the accumulated KV cache without broadcasting prefix state across replicas. When a pinned replica becomes unavailable or exceeds its queue threshold, EPP falls back to prefix matching across remaining workers, triggering re-prefill or remote KV transfer.

## Multi-Tier Routing Topologies

Production deployments organize routing into three complementary tiers:

Tier 1 operates at the edge or regional ingress gateway, handling TLS termination, rate limiting, and tenant path routing to specific model families.

Tier 2 operates at the cluster InferencePool level, where the EPP filter inspects request prompts, evaluates prefix hash matches, checks real-time worker queue depths, and assigns requests to specific model server pods.

Tier 3 operates locally inside each model server pod as the HTTP serving endpoint exposing OpenAI-compatible completion and metrics routes.

## Telemetry-to-Bottleneck Mapping and Tuning Heuristics

When optimizing a distributed `llm-d` deployment, client latency quantiles and Prometheus router/worker counters in `llmd-telemetry.json` map directly to bottleneck classes and manifest edits across `01-endpointpickerconfig.yaml` and `03-serving-deployment.yaml`:

| `llmd-telemetry.json` Signal | Diagnostic Condition | Bottleneck `kind` | Concrete `EndpointPickerConfig` / `LeaderWorkerSet` YAML Edit |
| :--- | :--- | :--- | :--- |
| `cache_memory.prefix_cache.hit_rate_tokens` | `< 0.50` on multi-turn or shared-prefix workloads (`routing_telemetry.prefix_cache_scorer_weight` too low) | `request-routing` | Increase `prefix-cache-scorer` weight (e.g. `weight: 80`), lower `token-load-scorer` and `queue-depth-scorer` relative dominance, and tighten `prefix-cache-affinity-filter.affinityThreshold` (e.g. `affinityThreshold: 0.80`) in `01-endpointpickerconfig.yaml`. |
| `worker_telemetry.worker_load_skew_ratio` or `latencies.queue_wait_ms.p99` | `worker_load_skew_ratio > 1.5` or high router queue wait while average KV usage is moderate | `request-routing` or `load-imbalance` | Increase `queue-depth-scorer` and `kv-utilization-scorer` weights (e.g. `weight: 60` and `weight: 50`) and tighten `saturationDetector.maxQueueDepth` (e.g. `maxQueueDepth: 48`) in `01-endpointpickerconfig.yaml`. |
| `cache_memory.kv_cache.usage_factor_peak` | `usage_factor_peak > 0.92` or KV recomputation/preemption | `memory-capacity` | Raise `--gpu-memory-utilization` (up to `0.95` within HBM headroom), tune `--block-size` (`64` or `128`), cap `--max-num-seqs` per replica in `03-serving-deployment.yaml`, and raise `kv-utilization-scorer` weight (e.g. `weight: 70`) in `01-endpointpickerconfig.yaml`. |
| `throughput.output_token_throughput_tps` vs `latencies.tpot_ms.p99` | Low tokens/step batching efficiency or XLA dynamic shape padding overhead on TPU MXU | `scheduler-batching` or `recompilation-padding` | Tune `--max-num-batched-tokens` (aligned to TPU MXU lane dim `128` on v5e/v5p/v7x or `256` on v6e, e.g. `8192` or `16384`) and `--max-num-seqs` (aligned to MXU sublane dim `8`, e.g. `128` or `256`) in `03-serving-deployment.yaml`. |

## ConfigMap Synchronization and TPU MXU Batching Alignment

When modifying `01-endpointpickerconfig.yaml` and `03-serving-deployment.yaml` in an `llm-d` Kustomize bundle, enforce the following structural invariants:

1. **Nested `ConfigMap` Key Synchronization**: When `01-endpointpickerconfig.yaml` defines a `ConfigMap` containing both `endpoint-picker-config.yaml` and `default-plugins.yaml` keys, keep both embedded YAML strings synchronized whenever scorer weights, filter parameters, or `saturationDetector` thresholds are updated.
2. **Scorer Weight Invariants**: Every scorer plugin entry in `EndpointPickerConfig` must have a finite, non-negative `weight`, and the sum of active scorer weights in a profile must be strictly positive (`> 0`).
3. **TPU MXU Batching Alignment**: On Cloud TPU `LeaderWorkerSet` deployments (`03-serving-deployment.yaml`), align `--max-num-batched-tokens` to a multiple of the TPU MXU lane dimension (`128` on v5e, v5p, and v7x; `256` on v6e) and `--max-num-seqs` to a multiple of the MXU sublane dimension (`8`), with `--gpu-memory-utilization` in `(0, 1]` and a positive integer `--block-size` (`64` or `128`).

## Standard Distributed Serving Admissibility Gates

Distributed `llm-d` studies pre-register SLA admissibility gates (`same_execution: true`) alongside the primary throughput or goodput objective so candidate configurations that improve throughput at the cost of tail latency or errors are rejected:

- `ttft_p99_ms` (`comparator: "lte"`, e.g. `bound: 800.0` ms): Guards against prefill queueing or routing hotspots.
- `tpot_p99_ms` (`comparator: "lte"`, e.g. `bound: 35.0` ms): Guards against decode batch over-sizing or prefill interference.
- `error_rate_pct` (`comparator: "lte"`, e.g. `bound: 1.0` %): Rejects configurations that trigger HTTP 5xx/429 overload drops or OOM restarts.
- `goodput_pct` (`comparator: "gte"`, e.g. `bound: 90.0` %): Ensures a high fraction of requests complete within joint `(TTFT, TPOT, E2E)` SLAs.
