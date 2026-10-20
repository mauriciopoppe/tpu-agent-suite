---
references:
  - references/tpu-model-onboarding/llmd/dynamo-to-llmd-mapping.md
  - references/tpu-model-onboarding/llmd/distributed-serving-benchmark.md
  - references/tpu-model-onboarding/preflight-check.md
  - references/common/vllm-engine-compatibility.md
scripts:
  - scripts/tpu-model-onboarding/inspect_deployment.py
  - scripts/tpu-model-onboarding/dynamo_translator.py
  - scripts/tpu-model-onboarding/llmd/benchmark.py
  - scripts/tpu-model-onboarding/llmd/parse_bench_results.py
  - scripts/tpu-performance-optimization/llmd_telemetry.py
  - scripts/common/vllm_patches/registry.json
  - scripts/common/vllm_patches/apply_vllm_patch.sh
  - scripts/common/vllm_patches/gpt_oss_mxfp4.py
---

# Distributed Serving Workflow for GKE Cloud TPU llm-d

This workflow governs the end-to-end translation, onboarding, and deployment of distributed multi-replica model serving on Google Kubernetes Engine (GKE) Cloud TPU by unwiring NVIDIA Dynamo deployments (`dynamo_unwiring`) into Kubernetes Gateway API and `llm-d` manifests.

## Overview and Execution Scope

Distributed serving orchestrates multiple TPU replicas behind an intelligent inference router. In single-slice onboarding, a model is deployed directly on a single TPU host or slice. In distributed serving, the deployment comprises a Gateway API `Gateway` and `HTTPRoute`, an `EndpointPickerConfig` (EPP) ConfigMap and EPP `Deployment`, an `InferencePool`, and GKE Cloud TPU worker replicas (`LeaderWorkerSet` and `ServiceAccount` rendered via Kustomize overlays) configured with Pallas PagedAttention (`--block-size 768`), unified block pooling (`TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`), and calibrated prefill throughput rates across the supported TPU VM shapes (`v4`, `v5e`, `v5p`, `v6e`, `tpu7x`).

### Paths and Artifacts

`<artifacts> = {{runs}}/<run_id>/artifacts`
`<work> = <artifacts>/tpu-model-onboarding`
`<serving_work> = <work>/distributed_serving`
`<bundle_dir> = <serving_work>/llmd_bundle`
`<kustomize_dir> = <bundle_dir>/kustomize`

Every log, report, Kustomize overlay, and generated manifest bundle must be written under `<serving_work>`, keeping the working tree clean and preserving reproducibility.

### Operational Modes

Read the request carefully before executing the workflow:

- **Serving mode**: The user explicitly asked to serve, host, deploy, spin up, onboard, or maintain a running distributed endpoint. In this mode, the service persists after verification, and the handoff report provides connection details and manual cancellation commands.
- **Verification mode**: Default mode when the request asks to test, onboard, or verify distributed serving without explicitly requesting an enduring service. The deployment is rolled out, verified with smoke inference requests, and then torn down to release TPU accelerator resources.

## NVIDIA Dynamo Manifest Unwiring (`dynamo_unwiring`)

The workflow ingests exported NVIDIA Dynamo Kubernetes manifests (`dynamo_manifests`) or a live GPU cluster namespace (`source_namespace`) and delegates to `{{agent:tpu-dynamo-unwiring}}` (following `{{reference:tpu-model-onboarding/llmd/dynamo-to-llmd-mapping.md}}`) to translate GPU constructs, routing algorithms, and memory configurations into a self-contained GKE Cloud TPU `llm-d` Kustomize bundle (carrying over optional single-VM `deployments.json` flags and `{{script:common/vllm_patches/registry.json}}` / `{{script:common/vllm_patches/gpt_oss_mxfp4.py}}` / `{{script:common/vllm_patches/apply_vllm_patch.sh}}` runtime patches).

## Cluster Resource Ownership & Blocker Escalation (`status: "needs_input"`)

- **Never scavenge or reuse unowned cluster resources**: Never query `pvc,secret,sa,role,rolebinding,configmap` in a shared namespace to discover and reuse pre-existing resources (`PersistentVolumeClaim`, `Secret`, `ServiceAccount`, `RoleBinding`, `ConfigMap`) or node-cached proprietary images that were not created by this run or explicitly provided by the user.
- **Stop and ask the user when stuck (`status: "needs_input"`)**: If `preflight_inspection.json` or Kustomize `-k` rollout / pod startup hits a blocker (such as missing `InferencePool` / `Gateway` CRDs when `standalone_epp` was not requested, RBAC `Forbidden` on cluster resources or pod watching, `ErrImagePull` / `ImagePullBackOff` / `403 Forbidden` on the container image, or missing Hugging Face credentials / storage volumes), do NOT improvise workarounds or attach unowned resources from the namespace. Instead, immediately stop and return `status: "needs_input"` in `onboarding-result.json` with a clear diagnostic summary of what failed and the specific options/parameters needed from the user (e.g., container image URI, whether to use `--standalone-epp` and which `ServiceAccount` to bind, or whether to mount a user-owned PVC/Secret) so `tpu-agent` can prompt the user for a decision.

## Step-by-Step Workflow Checklist

Follow each step in sequence, recording intermediate outputs in `<serving_work>`. Execute cluster inspection, manifest translation, dry-run validation, and load benchmarking through the packaged Python scripts, and deploy the rendered Kustomize bundle via `kubectl` and the runtime environment.

### Step 1: Cluster Preflight and Topology Inspection

Inspect the target Kubernetes cluster capabilities via the deployment inspection script using `--target-cluster-only` (which inspects only installed CRDs and TPU node pools without listing, aggregating, or dumping unrelated workloads in the target namespace):

1. Run the deployment inspection tool in target-cluster-only mode:
   ```bash
   python3 {{script:tpu-model-onboarding/inspect_deployment.py}} \
     --target-cluster-only \
     --namespace <target_namespace> \
     --json > <serving_work>/preflight_inspection.json
   ```
2. Verify required Custom Resource Definitions (CRDs) from `preflight_inspection.json` (`installed_crds`):
   - Gateway API: `gateway.networking.k8s.io`
   - LeaderWorkerSet: `leaderworkerset.x-k8s.io`
   - Inference Extension: `inference.networking.x-k8s.io`
3. Inspect TPU node pool labels and capacity across supported TPU VM shapes in `preflight_inspection.json` (`accelerator`):
   - Accelerator family: `cloud.google.com/gke-tpu-accelerator` (`tpu-v4-podslice`, `tpu-v5-lite-podslice`, `tpu-v5p-slice`, `tpu-v6e-slice`, `tpu7x`)
   - Accelerator topology: `cloud.google.com/gke-tpu-topology` (`1x1`, `2x2`, `2x4`, `4x4`, `4x8`, `1x1x1`, `2x2x1`, `2x2x2`, `2x4x1`, `2x2x4`)
   - Device resource and node availability: `google.com/tpu` (`total_chips`, `chips_per_node`, `total_nodes`, `ready_nodes`)

### Step 2: Manifest Translation (`tpu-dynamo-unwiring`)

Translate the source NVIDIA Dynamo deployment into a complete `llm-d` manifest bundle:

1. Delegate to `{{agent:tpu-dynamo-unwiring}}` using the exact `dynamo-unwiring-request.json` field names:
   - `run_id`: Active run identifier
   - `dynamo_manifests` (local path to exported GPU Dynamo manifests) or `source_namespace` (live source GPU namespace) and optional `source_context`
   - `target_namespace`: Target Kubernetes namespace for `llm-d` on TPU (e.g. `"vllm-serving"`)
   - `target_tpu_type`: Target TPU family (`"v5e"`, `"v5p"`, `"v6e"`, `"tpu7x"`)
   - `target_tpu_topology`: Target TPU topology (e.g. `"2x2x1"`, `"2x2x2"`, `"2x4"`)
   - `model_id`: Hugging Face model identifier
   - `deployments_json`: Optional path to single-VM `deployments.json` to carry over tuned `vllm serve` flags and environment variables
   - `vllm_patch`: Optional path to single-VM `.patch` or `.py` file to carry over
   - `standalone_epp`: Boolean flag to emit a standalone EPP `Deployment` + `Service`
   - `dry_run`: `true`
2. Ensure the synthesized Kustomize directory (`<bundle_dir>/kustomize`) contains all required manifests and satisfies cross-resource selector contracts:
   - `00-gateway.yaml`: Gateway resource (`llm-d-inference-gateway`) and HTTPRoute configuration (`backendRefs` pointing to `InferencePool` on port `8080`).
   - `01-endpointpickerconfig.yaml`: `EndpointPickerConfig` ConfigMap, EPP `ServiceAccount`, EPP `Deployment`, and EPP `Service` defining prefix cache scoring, queue depth balancing, and calibrated peak prefill throughput (`7000` for `v4`, `8500` for `v5e`, `16000` for `v5p`, `18500` for `v6e`, `42000` for `tpu7x`).
   - `02-inferencepool.yaml`: `InferencePool` whose `spec.selector.matchLabels` (`llm-d.ai/inferenceServing: "true"`, `llm-d.ai/guide: optimized-baseline`, and `leaderworkerset.sigs.k8s.io/worker-index: "0"`) matches the TPU modelserver leader pod template labels and whose `spec.targetPorts[0].number` (`8000`) matches the container port.
   - `base-modelserver.yaml` and `patch-vllm-tpu-workers.yaml`: TPU serving `ServiceAccount` and `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`, `leaderWorkerTemplate.size: 1` for single-host slices and `size: num_hosts` for multi-host slices) configured with Pallas block size `768`, `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`, carried-over `deployments.json` flags, and optional `04-vllm-patches.yaml` runtime patch `ConfigMap`.
   - `kustomization.yaml`: Kustomization listing `["00-gateway.yaml", "01-endpointpickerconfig.yaml", "02-inferencepool.yaml", "base-modelserver.yaml"]` when `standalone_epp` is `false`, or `["01-endpointpickerconfig.yaml", "base-modelserver.yaml"]` when `standalone_epp` is `true` (excluding `00-gateway.yaml` and `02-inferencepool.yaml` so Kustomize `-k` rollout succeeds directly on clusters without `Gateway` or `InferencePool` CRDs), plus optional `04-vllm-patches.yaml` in `resources:` alongside `patches:`.
   - Never overwrite `01-endpointpickerconfig.yaml` with ad-hoc scorer plugins or weights after `{{agent:tpu-dynamo-unwiring}}` returns, and never rewrite `01-endpointpickerconfig.yaml` or `base-modelserver.yaml` with ad-hoc Python scripts or `yaml.safe_dump` calls (if image or serving flag overrides are needed, pass them via `--deployments-json` to `dynamo_translator.py`). `dynamo_translator.py` and `epp_config.py` serialize all multiline `ConfigMap.data` entries using YAML literal block scalars (`|`).

### Step 3: Dry-Run Verification

Validate the translated configuration and Kustomize bundle without running raw `kubectl kustomize`:

1. `{{agent:tpu-dynamo-unwiring}}` executes `{{script:tpu-model-onboarding/dynamo_translator.py}} --dry-run` (which validates Kustomize rendering and writes `kustomize_rendered.yaml`). `dynamo_translator.py` only translates and dry-runs manifests; it never applies manifests to a cluster.
2. Verify that `mapping_report.json` has `dry_run_verified: true`.

### Step 4 and Step 5: Cluster Rollout, Readiness Wait, Smoke Check, and Evidence Capture

After the Kustomize bundle is translated and dry-run validated (`<bundle_dir>/kustomize` returned by `{{agent:tpu-dynamo-unwiring}}`), deploy the manifests to the target cluster, wait for readiness, run greedy smoke inference, and capture cluster evidence:

1. Apply the rendered Kustomize bundle to `<target_namespace>`:
   ```bash
   kubectl -n <target_namespace> apply -k <kustomize_dir>
   ```
2. Wait for the `llm-d-epp` router `Deployment` and TPU `LeaderWorkerSet` pods to reach `Ready`, always scoping `kubectl get pods`, `kubectl wait`, and pod log queries to the deployed workload label selector (`-l app=vllm-tpu-homogeneous` or `-l llm-d.ai/inferenceServing=true` / `-l leaderworkerset.sigs.k8s.io/name=vllm-tpu-workers`) so unrelated pods in a shared namespace are never inspected or waited on:
   ```bash
   kubectl rollout status deployment/llm-d-epp -n <target_namespace> --timeout=600s
   kubectl wait --for=condition=Ready pod -l app=vllm-tpu-homogeneous -n <target_namespace> --timeout=600s
   ```
3. Probe `/v1/models` and send a greedy (`temperature: 0.0`) `/v1/completions` smoke request against `service/llm-d-epp` (via `{{runtime}} compute run` or `{{runtime}} compute port-forward`), writing the verification result to `<serving_work>/smoke_verification.json`.
4. Capture post-rollout cluster state, manifests, and pod logs into `<serving_work>/evidence`:
   ```bash
   python3 {{script:tpu-model-onboarding/inspect_deployment.py}} \
     --namespace <target_namespace> \
     --store-dir <serving_work>/evidence \
     --capture-logs \
     --json > <serving_work>/evidence/post_rollout_inspection.json
   ```

### Step 6: Load Generator Execution and Telemetry Parsing

When SLA validation or load benchmarking (`inference-perf`, `aiperf-mooncake`, or `aiperf-agentx`) is requested, follow `{{reference:tpu-model-onboarding/llmd/distributed-serving-benchmark.md}}` to execute the benchmark Job, parse the resulting log stream using `{{script:tpu-model-onboarding/llmd/benchmark.py}}` and `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}`, and aggregate baseline telemetry into `<serving_work>/llmd-telemetry.json` conforming to `{{schema:llmd-telemetry.json}}` via `{{script:tpu-performance-optimization/llmd_telemetry.py}}`:

```bash
python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
  --model <model_id> \
  --harness <inference-perf|aiperf-mooncake|aiperf-agentx> \
  --namespace <target_namespace> \
  --target-url http://llm-d-epp.<target_namespace>.svc.cluster.local:8000 \
  --execute \
  --logs-out <serving_work>/bench-logs.txt \
  --parse-out <serving_work>/bench-results.json \
  --evidence-dir <serving_work>/evidence

python3 {{script:tpu-model-onboarding/llmd/parse_bench_results.py}} \
  --logs <serving_work>/bench-logs.txt \
  --json > <serving_work>/bench-summary.json

python3 {{script:tpu-performance-optimization/llmd_telemetry.py}} \
  --benchmark-output <serving_work>/bench-logs.txt \
  --output-json <serving_work>/llmd-telemetry.json \
  --arm-id baseline \
  --model <model_id> \
  --accelerator-type <target_tpu_type> \
  --validate
```

### Step 7: Handoff and Downstream Optimization

Finalize onboarding outputs and prepare for performance optimization:

1. Set `output_files.deployments_json` in the result payload pointing to the primary serving deployment manifest.
2. Set `output_files.launch_recipe` and `output_files.bundle_dir` pointing to the rendered Kustomize bundle directory (`<bundle_dir>`).
3. Set `output_files.llmd_telemetry` pointing to `<serving_work>/llmd-telemetry.json` (when Step 6 ran) so downstream Stage 2 attribution and Stage 3 (`tpu-performance-optimization`) consume the baseline telemetry directly.
4. Provide the handoff summary:
   - Deployed serving topology classification (`optimized-baseline` or `pd-disaggregation`).
   - Gateway or EPP endpoint URL and model identifier.
   - Worker replica count and TPU chip configuration.
   - Next steps for downstream performance optimization: pass the deployment and `output_files.llmd_telemetry` to `tpu-performance-optimization` to execute the Make-It-Fast autoresearch loop (EPP scheduling tuning, flow control, and data-layer tuning).
