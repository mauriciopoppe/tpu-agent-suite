---
title: "LLM-d Routing and EndpointPicker (EPP)"
type: concept
tags: [workload, performance, framework]
created: 2026-10-12
updated: 2026-10-12
sources: 2
human_reviewed: false
aliases: ["EPP", "EndpointPicker", "llm-d routing", "prefix cache routing", "ext-proc"]
---

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

## See also

- [KV cache](../concepts/kv-cache.md)
- [Prefill vs decode](../concepts/prefill-vs-decode.md)

## Sources

- [scaling-book, Part 7: Inference](../sources/2025-scaling-book-inference.md)
- [scaling-book, Part 8: Serving LLaMA 3](../sources/2025-scaling-book-applied-inference.md)
