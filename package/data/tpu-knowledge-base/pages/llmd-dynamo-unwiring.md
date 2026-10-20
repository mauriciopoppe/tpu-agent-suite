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
entry: llmd-dynamo-unwiring
keyword: dynamo-unwiring
url: not applicable -- see Evidence
authority: first-party-verified
verified: 2026-10-19
retrieved: 2026-10-19
source_updated: not applicable
generator: authored
---

# Unwiring NVIDIA Dynamo to GKE Cloud TPU llm-d

Onboarding and unwiring a distributed serving stack from NVIDIA Dynamo on GPU Kubernetes clusters to `llm-d` on Google Kubernetes Engine (GKE) Cloud TPUs replaces proprietary Dynamo custom resources (`DynamoGraphDeployment`, `DynamoComponentDeployment`), NIXL/UCX GPU memory transports, and CUDA worker settings with Kubernetes Gateway API Inference Extension (`InferencePool`, `EndpointPickerConfig`) and `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`) running `vllm-tpu` on Cloud TPU slices.

## Control Plane and Custom Resource Mapping

NVIDIA Dynamo deployments declare serving topologies using `DynamoGraphDeployment` (`nvidia.com/v1alpha1`), `DynamoComponentDeployment`, or legacy `DynamoNim` / `DynamoGraph` manifests with `Frontend`, `Router`, and `VllmWorker` components. On GKE Cloud TPU, `tpu-dynamo-unwiring` (`dynamo_translator.py`) translates these into standard `llm-d` Kubernetes manifests:

- **Gateway (`gateway.networking.k8s.io/v1`) and HTTPRoute**: Replaces the Dynamo `Frontend` service entry point with a regional or cluster L7 Envoy Gateway (`gke-l7-regional-external-managed` or `istio`) routing `/v1/completions` and `/v1/chat/completions` to the target `InferencePool`.
- **EndpointPickerConfig (`inference.networking.x-k8s.io/v1alpha1`) and InferencePool (`inference.networking.x-k8s.io/v1alpha2`)**: Replaces the Dynamo `Router` component (`DYN_ROUTER_MODE` and Dynamo worker discovery over etcd/NATS) with the `llm-d` EndpointPicker (EPP) external processor. The `InferencePool` selects only the leader pods of the worker groups via `leaderworkerset.sigs.k8s.io/worker-index: "0"` alongside the model's `llm-d.ai/model` and `llm-d.ai/role` labels.
- **LeaderWorkerSet (`leaderworkerset.x-k8s.io/v1`)**: Replaces Dynamo's GPU worker pods (`VllmWorker`). Each replica group (`spec.replicas`) provisions a coherent TPU slice via `cloud.google.com/gke-tpu-accelerator` and `cloud.google.com/gke-tpu-topology`, with `spec.leaderWorkerTemplate.size` matching the number of TPU hosts in the slice (`1` for single-host slices such as `tpu7x` `2x2x1` or `v6e` `2x4`).

## Router Mode and Session Affinity Translation

Dynamo's `DYN_ROUTER_MODE` and router flags map directly to `EndpointPickerConfig` scoring plugins:

- **`DYN_ROUTER_MODE=kv` (or `--router-mode kv`)**: Translates to an EPP scoring profile combining `prefix-cache-scorer` (weight `50`), `token-load-scorer` (weight `25`), `queue-scorer` (weight `15`), and `kv-cache-utilization-scorer` (weight `10`), plus `utilization-filter` (`maxWaitingQueueDepth: 16`) and `prefix-cache-affinity-filter` (`affinityThreshold: "0.80"`, `maxTTFTPenaltyMs: 250`).
- **`DYN_ROUTER_MODE=round_robin` / `random`**: Translates to balanced `queue-scorer` (weight `50`), `token-load-scorer` (weight `30`), and `kv-cache-utilization-scorer` (weight `20`).
- **Session Affinity Headers (`X-Session-ID`, `X-Correlation-ID`, `x-session-id`)**: Preserved in `EndpointPickerConfig` via `sessionAffinity.headerNames` and the `session-affinity` plugin (`weight: 100`) so multi-turn conversational sessions remain pinned to the replica holding their warm KV cache blocks.

## Hardware Sizing and Parallelism Translation (GPU to TPU)

GPU `nvidia.com/gpu` resource requests and `CUDA_VISIBLE_DEVICES` environment variables are stripped and replaced with GKE TPU node selectors and `google.com/tpu` limits:

- **TPU7x (Ironwood, `tpu-v7-chip` / `tpu7x`)**: Each physical `tpu7x` chip has 192 GB of HBM and exposes 2 logical TensorCores (`devices` in vLLM when `TPU_MULTIHOST_BACKEND` / dual-core mode is active, or 1 chip = 2 cores on a 4-chip `2x2x1` slice providing 8 logical TensorCores). When translating a GPU deployment with `--tensor-parallel-size 2` and `--data-parallel-size 4` (or `replicas: 4` of 2 GPUs) onto a 4-chip `tpu7x` (`2x2x1`) node (`google.com/tpu: 4`), a single `LeaderWorkerSet` replica runs `--tensor-parallel-size 2` and `--data-parallel-size 4` across the 8 logical TensorCores, or multiple LWS replicas each claim their own TPU slice.
- **TPU v6e (Trillium, `tpu-v6e-slice`)**: Each `v6e` chip has 32 GB of HBM and 1 TensorCore (`2x4` topology = 8 chips on a single host).
- **Model Support on TPU7x and v6e**: Dense and MoE checkpoints including `Qwen/Qwen3-32B` (`Qwen3ForCausalLM`), `google/gemma-4-31B-it`, `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, and `openai/gpt-oss-120b` are supported in `vllm-tpu` (`tpu-inference`) and `vllm-torchtpu`.

## Pallas Attention Block Size and Environment Unwiring

CUDA-specific flags and environment variables (`NCCL_*`, `UCX_*`, `NIXL_*`, `CUDA_VISIBLE_DEVICES`, `--kv-Events-config`, `--block-size 16` or `64`) are unwired for TPU execution:

- **Pallas RPA Block Size (`--block-size 768` or `128`)**: Small GPU paged-attention block sizes (such as `16`, `32`, or `64`) cause high VMEM metadata overhead and sub-optimal DMA transfers on TPU. The Dynamo unwiring translator rewrites `--block-size` to a TPU Pallas-aligned multiple (`768` by default) and sets `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`.

## Structured Intake for Distributed Serving & Onboard-and-Optimize

When a user asks to onboard or unwire an NVIDIA Dynamo deployment (`dynamo_unwiring`) to `llm-d` on GKE Cloud TPU or asks for an **onboard-and-optimize** study (for example, `"Unwire our NVIDIA Dynamo deployment for Qwen 3.5 on GKE TPU v7x and optimize throughput for AgentX"`), `tpu-agent` distinguishes two request classes during query intake:

1. **Onboarding or Unwiring Only (`onboard_only`)**:
   - The user asks only to onboard a model or unwire an NVIDIA Dynamo deployment to `llm-d` on GKE TPU (`onboarding_path: "dynamo_unwiring"`).
   - Deliverable is **Stage 1: Baseline Onboarding (`tpu-model-onboarding`) only** (`mode: "inference"` for single-model vLLM, or `mode: "distributed"` for `dynamo_unwiring`). Do not add unrequested benchmarking sweeps, Stage 2 profiling, Stage 3 optimization study loops, `protocol.json`, `traffic_profile`, or `sla_constraints`.
2. **Onboard-and-Optimize (`onboard_and_optimize`)**:
   - The user asks to both onboard (or unwire NVIDIA Dynamo to `llm-d`) and optimize throughput, latency, or cost efficiency.
   - Deliverable is a **3-Stage Plan**:
     1. **Stage 1: Baseline Onboarding (`tpu-model-onboarding`)** -- bring up and verify the baseline deployment (`mode: "inference"` or `mode: "distributed"`).
     2. **Stage 2: Empirical Profiling & Telemetry Attribution (`tpu-diagnostics` or `llmd-telemetry.json`)** -- capture an in-place profile on the verified baseline (or, for distributed `llm-d` deployments on `dynamo_unwiring`, read the empirical orchestration signals from `output_files.llmd_telemetry`) to rank bottleneck `levers`.
     3. **Stage 3: Study Loop (`tpu-performance-optimization`)** -- compile the user's intent, hardware topology, and latency SLAs into a native `protocol.json` conforming to `framework/schemas/study-protocol.json` (`primary_metric`, `admissibility_gates`, `measurement_cell`, and `stopping_rule`) via `study_protocol.py`.

### Starting Point, Hardware Budget, and Cluster Discovery (`dynamo_unwiring`)

1. **Starting Point & Onboarding Path (`onboarding_path`)**:
   - `dynamo_unwiring`: Unwiring NVIDIA Dynamo (`DynamoGraphDeployment` / `DynamoComponentDeployment` or GPU serving deployments) into native GKE TPU `llm-d` manifests (`mode: "distributed"`, `dynamo_manifests` or `source_namespace`).
   - `existing_deployment`: Existing healthy GKE TPU deployment (skips Stage 1 directly to Stage 2 profiling and Stage 3 study loop).
2. **Hardware Budget & Topology (`hardware_budget`)**:
   - Normalize accelerator generation (`"TPU v7x"`, `"v7x"`, `"Ironwood"` -> `tpu7x`; `"v6e"`, `"Trillium"` -> `v6e`; `v5p`; `v5e`; `v4`) and slice topology (`2x2x2` = 8 chips on `tpu7x`, `2x4` = 8 chips on `v6e`/`v5e`, `2x2x1` = 4 chips on `v4`/`v5p`/`v6e`/`tpu7x`).
   - Record `max_slices` for the homogeneous `LeaderWorkerSet` worker pool.
3. **NVIDIA Dynamo Unwiring Cluster Discovery & Probing (`deployment inspect`)**:
   - If the user has not yet specified how to reach the source GPU deployment (Kubernetes `source_context` / `kubeconfig` + `source_namespace`, or an exported `dynamo_manifests` path) and the target GKE TPU cluster (Kubernetes `target_context` / `kubeconfig` + `target_namespace`), ask the user for those cluster coordinates.
   - Once the source GPU namespace/manifests and target TPU namespace/cluster are known, run `deployment inspect` against both clusters:
     - **Source GPU deployment**: `tpu-agent-suite deployment inspect --namespace <source_namespace> [--context <source_context>] [--kubeconfig <source_kubeconfig>] --json` (or `--manifests <dynamo_manifests> --json`) to auto-discover the `DynamoGraphDeployment` / worker deployment name, `model_id`, worker pool replica count (`replicas.homogeneous_pools`), GPU resources (`accelerator.chips_per_node`, `accelerator.total_chips`), and `DYN_ROUTER_MODE` (`router_config.kv_routing` and `router_config.session_headers`).
     - **Target GKE TPU cluster**: `tpu-agent-suite deployment inspect --target-cluster-only --namespace <target_namespace> [--context <target_context>] [--kubeconfig <target_kubeconfig>] --json` to auto-discover installed CRDs (`LeaderWorkerSet`, `InferencePool`, `Gateway`) and TPU node pool accelerator (`tpu7x`, `v6e`, etc.) and topology (`2x2x1`, `2x2x2`, etc.) without listing or dumping unrelated workloads in the target namespace.
   - Present the auto-discovered source GPU topology and target GKE TPU cluster configuration in the plan alongside optional intake questions (whether the user has single-VM TPU carry-over files `deployments.json` or `.patch`; default: none unless supplied).
   - **Missing CRDs or Cluster Resource Blockers (`--standalone-epp` and `status: "needs_input"`)**:
     - If `installed_crds.inference_pool` or `installed_crds.gateway_api` is `false` in the target cluster inspection, surface this in the Stage 1 `ask_question` confirmation, asking whether to deploy in standalone EPP mode (`standalone_epp: true` / `--standalone-epp`, which omits `00-gateway.yaml` and `02-inferencepool.yaml`) and whether a specific container image or `ServiceAccount` (`epp_service_account`) should be used.
     - Never scavenge or reuse unowned resources (`PersistentVolumeClaim`, `Secret`, `ServiceAccount`, `RoleBinding`, `ConfigMap`, or node-cached proprietary images) from a shared namespace. If `tpu-model-onboarding` encounters an image pull (`ErrImagePull` / `ImagePullBackOff` / `403 Forbidden`), RBAC (`Forbidden`), or missing resource/credential error during rollout and returns `status: "needs_input"`, `tpu-agent` asks the user how to proceed via `ask_question` before re-invoking `tpu-model-onboarding`.

### Traffic Profiles & Study Protocol Compilation (`onboard_and_optimize`)

When optimization is requested (`onboard_and_optimize`), resolve `traffic_profile` and `framework/schemas/study-protocol.json`:

- **Traffic Profile (`traffic_profile`)**:
  - `agentx`: `prefix-repetition`, ISL=`8192`, OSL=`1024`, prefix cache=`0.70`, `multi_turn: true`, concurrency=`[8, 16, 32, 64]`, interactivity tier `50_to_75` (`TPOT <= 20 ms`, `TTFT <= 2000 ms`).
  - `agentic-codegen`: `prefix-repetition`, ISL=`16384`, OSL=`1024`, prefix cache=`0.80`, `multi_turn: true`, concurrency=`[4, 8, 16, 32]`, interactivity tier `50_to_75`.
  - `mooncake-trace`: `mooncake-trace`, ISL=`8192`, OSL=`512`, prefix cache=`0.65`, `multi_turn: true`, concurrency=`[16, 32, 64, 128]`, interactivity tier `25_to_50`.
  - `conversational`: `sharegpt`, ISL=`1024`, OSL=`512`, prefix cache=`0.30`, `multi_turn: true`, concurrency=`[16, 32, 64, 128]`, interactivity tier `25_to_50`.
  - `rag`: `prefix-repetition`, ISL=`8192`, OSL=`256`, prefix cache=`0.50`, `multi_turn: false`, concurrency=`[8, 16, 32, 64]`, interactivity tier `25_to_50`.
  - `deep-research`: `random`, ISL=`32768`, OSL=`2048`, prefix cache=`0.40`, `multi_turn: true`, concurrency=`[2, 4, 8, 16]`, interactivity tier `sub_25`.
- **Study Protocol & SLA Admissibility Gates (`framework/schemas/study-protocol.json`)**:
  - **Primary Metric (`primary_metric`)**: `output_tokens_per_s`, `total_tokens_per_s`, `tokens_per_dollar`, `output_tokens_per_s_per_user`, `ttft_p50_ms`, or `tpot_p50_ms`, with `unit` (`tok/s`, `tok/$`, `tok/s/user`, `ms`) and `direction` (`maximize` or `minimize`).
  - **Interactivity Tier & Admissibility Gates (`admissibility_gates`)**: `sub_25` (`< 25 tok/s/user`), `25_to_50` (`25-50 tok/s/user`), `50_to_75` (`50-75 tok/s/user`), or `75_plus` (`75+ tok/s/user`), compiled into co-located `same_execution: true` gates (`ttft-p99-ceiling`, `tpot-p99-ceiling`, `error-rate-ceiling`).
  - **Measurement Cell & Stopping Rule**: `measurement_cell` (`topology`, `shape`, `concurrency`), `replicate_count` (default `3`), `noise_floor_pct` (default `1.5`), `stopping_rule` (`max_rounds: 12`, `plateau_rounds: 4`), and `mode` (`approve` or `auto`).
- **Request Schema Mapping (`onboarding-request.json` and `study-request.json`)**:
  - For `tpu-model-onboarding` (`onboarding-request.json`): pass `mode: "distributed"`, `onboarding_path: "dynamo_unwiring"`, `hardware_budget`, `dynamo_manifests` or `source_namespace` / `source_context`, `target_namespace` / `target_context`, optional `deployments_json` / `vllm_patch`, and (when optimizing) `traffic_profile` and `sla_constraints`.
  - For `tpu-performance-optimization` (`study-request.json`): pass `study_id`, `run_id`, `target` (model ID or deployed `output_files.bundle_dir`), `namespace` (`target_namespace`), `mode` (`"auto"` when the user confirmed end-to-end execution or replied `"go"`), `hardware_budget`, `traffic_profile`, `protocol` / `protocol_path` (`protocol.json`), `baseline_metrics` (`output_files.llmd_telemetry`), and `levers` (`["request-routing", "load-imbalance", "scheduler-batching"]`).
