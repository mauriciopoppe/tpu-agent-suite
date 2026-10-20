# NVIDIA Dynamo to GKE TPU llm-d Translation Guide

This reference defines the translation contract between NVIDIA Dynamo (or SGLang / vLLM router configurations on GPUs) and the GKE Cloud TPU distributed serving stack built on Kubernetes Gateway API and [llm-d](https://github.com/llm-d/llm-d).

## Architecture Comparison

| Architectural Layer | NVIDIA Dynamo (GPU) | GKE Cloud TPU (llm-d) |
| :--- | :--- | :--- |
| **Declarative Orchestration** | `DynamoGraphDeployment` (DGD) / `DGDR` | Kubernetes Gateway API + `InferencePool` + `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`) |
| **Ingress & Gateway** | Dynamo Frontend / Ingress Controller | Kubernetes Gateway API (`Gateway`, `HTTPRoute`) |
| **Inference Router** | Dynamo Scheduler / `KvRouter` (Rust) | Gateway API Inference Extension (`InferencePool`) + `EndpointPickerConfig` (EPP) |
| **Model Server Runtime** | In-process `dynamo.vllm` / `dynamo.sglang` | `vLLM-TorchTPU` (or `SGLang-TorchTPU`) on `torch_tpu` |
| **Compute Acceleration** | `nvidia.com/gpu` (H100, H200, B200) | `google.com/tpu` with topology (`v6e`, `v7x`) |
| **KV Cache Architecture** | GPU HBM with CUDA PagedAttention (16/32 block size) | TPU HBM with Pallas PagedAttention (768 block size) + Unified Block Pool |

## Custom Resource Definition (CRD) Translation

Dynamo manages multi-replica deployments using operator CRDs under `nvidia.com/v1beta1`:

- **`DynamoGraphDeployment` (DGD)**: Defines frontend routing, worker pools, shared memory size, and multinode gang scheduling. Translated directly into:
  - `00-gateway.yaml`: Gateway API `Gateway` and `HTTPRoute`.
  - `01-endpointpickerconfig.yaml`: `EndpointPickerConfig` with calibrated TPU throughput and scoring profiles.
  - `02-inferencepool.yaml`: `InferencePool` linking routing selectors strictly to leader pods (`leaderworkerset.sigs.k8s.io/worker-index: "0"`).
  - `03-serving-deployment.yaml` (`base-modelserver.yaml` + `patch-vllm-tpu-workers.yaml`): TPU `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`) standardized across both single-host slices (`leaderWorkerTemplate.size: 1`) and multi-host slices (`leaderWorkerTemplate.size > 1`) with exclusive nodepool topology placement (`leaderworkerset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool`).
  - `04-vllm-patches.yaml`: Optional `vllm-tpu-patches` `ConfigMap` carrying over `vllm_torchtpu` source patches (`apply_vllm_patch.sh` + `<patch_name>`) when the model matches the `vllm_patches` registry (`registry.json`) or `--vllm-patch` is supplied.
- **`DynamoGraphDeploymentRequest` (DGDR)**: High-level SLA contract (`model`, `backend`, `sla.ttft`, `sla.itl`). In GKE Cloud TPU, this translates into an offline throughput sweep using `tpu-roofline` and pre-calibrated peak throughput tables (`18500` tok/s for v6e, `42000` tok/s for v7x).

## Parameter Translation Mapping

### 1. Router and Scheduling Rules

Dynamo routes requests using configurable scheduler policies. In `llm-d`, routing is executed by the Endpoint Picker (EPP) plugins configured in an `EndpointPickerConfig` resource:

| Dynamo Router Policy | llm-d EPP Plugin | TPU Calibration & Configuration |
| :--- | :--- | :--- |
| **Prefix Cache Affinity** | `prefix-cache-affinity-filter`, `prefix-cache-scorer` | Weight: 100. Directs requests with matching prefix hashes to replicas holding warm KV blocks. |
| **Throughput / Token Load** | `token-load-scorer` | Weight: 50. Replaces GPU-inherited rates with TPU peak prefill throughput: `18500` tok/s for v6e, `42000` tok/s for v7x. |
| **Queue Depth Balancing** | `queue-scorer`, `active-request-scorer` | Weight: 20-30. Penalizes replicas with deep inflight queues. |
| **KV Cache Saturation** | `kv-cache-utilization-scorer` | Weight: 30. Steers new sequences away from replicas near HBM exhaustion. |
| **Endpoint Selection** | `max-score-picker` / `weighted-random-picker` | Selects highest aggregated scoring replica or samples proportionally. |
| **Flow Control / Admission** | `utilization-detector` + `probabilistic-admitter` | Detects fleet-wide saturation (e.g. threshold 0.95) and sheds excess load gracefully. |

### 2. Hardware Resource & Pod Topology

| GPU Construct (Dynamo) | TPU Construct (llm-d) | Notes |
| :--- | :--- | :--- |
| `resources.limits["nvidia.com/gpu"]: 4` | `resources.limits["google.com/tpu"]: 4` | Allocates 4 TPU TensorCore chips per pod. |
| GPU node selector | `nodeSelector["cloud.google.com/gke-tpu-accelerator"]: "v6e"` | Specifies TPU generation (`v5e`, `v6e`, `v7x`). |
| GPU topology / interconnect | `annotations["cloud.google.com/gke-tpu-topology"]: "2x2x1"` | Specifies 2D/3D torus/mesh TPU topology. |
| Multi-node / Single-node workload | `LeaderWorkerSet` (`size: 1` single-host, `size > 1` multi-host) | Standardized `LeaderWorkerSet` with `InferencePool` selecting `leaderworkerset.sigs.k8s.io/worker-index: "0"`. |
| Shared memory `/dev/shm` | `emptyDir: {medium: "Memory", sizeLimit: "64Gi"}` | Required for host DRAM KV offloading and IPC. |

### 3. Serving Engine Flags (vLLM-TorchTPU) & Single-VM Carry-Over

| Parameter | GPU Setting (Dynamo) | TPU Setting (llm-d) | Rationale |
| :--- | :--- | :--- | :--- |
| **PagedAttention Block Size** | `--block-size 16` or `32` | `--block-size 768` | Matches TPU vector/matrix core tiling dimensions in Pallas. |
| **Tensor Parallelism** | `--tensor-parallel-size <N>` | `--tensor-parallel-size <chips>` | Matches chip count in the local slice (or overridden via `--deployments-json`). |
| **Prefix Caching** | `--enable-prefix-caching` | `--enable-prefix-caching` | Reuses cached prompt KV blocks across requests. |
| **Memory Utilization** | `--gpu-memory-utilization 0.90` | `--gpu-memory-utilization 0.95` | Allocates TPU HBM pool for activations and KV cache. |
| **Unified Block Pool** | N/A | `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1` | Unifies KV cache allocation across attention layers and latent states. |
| **Single-VM `deployments.json`** | N/A | `--deployments-json <path>` | Carries over tuned `tensor_parallel_size`, `data_parallel_size`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `attention_backend`, and `env` overrides into the `LeaderWorkerSet` container. |
| **Single-VM `vllm_patches`** | N/A | `04-vllm-patches.yaml` (`vllm-tpu-patches`) | Automatically looks up `model_id` in the `vllm_patches` registry (`registry.json`, or accepts `--vllm-patch`), mounts `/app/patches`, and runs `apply_vllm_patch.sh` before `exec vllm serve`. |

## Validation Contract

Translation is considered complete and successful when:
1. `dynamo_translator.py` produces the complete manifest bundle in `<work>/dynamo_unwiring/<model_slug>/llmd_bundle/`.
2. All generated manifests pass Kubernetes dry-run validation via `dynamo_translator.py --dry-run`.
3. `mapping_report.json` contains full traceability between source Dynamo entities and target `llm-d` entities.
