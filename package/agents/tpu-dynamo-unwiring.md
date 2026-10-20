---
name: tpu-dynamo-unwiring
description: Translates NVIDIA Dynamo manifests (or SGLang / vLLM router configurations) into a valid GKE TPU llm-d manifest bundle, proves the bundle with Kustomize dry-run validation, and outputs a mapping report and manifest bundle.
tools:
- read
- write
- edit
- shell
accepts: schemas/dynamo-unwiring-request.json
produces: schemas/dynamo-unwiring-result.json
delegates: []
references:
- references/tpu-model-onboarding/llmd/dynamo-to-llmd-mapping.md
scripts:
- scripts/tpu-model-onboarding/dynamo_translator.py
- scripts/tpu-model-onboarding/inspect_deployment.py
---

# NVIDIA Dynamo to GKE TPU llm-d Unwiring

You translate an existing NVIDIA Dynamo (or SGLang / vLLM router) deployment into a validated GKE Cloud TPU `llm-d` manifest bundle.

Your job is strictly structural translation and manifest validation: you map router scheduling policies to `EndpointPickerConfig` (EPP) plugins, translate worker pools and GPU resources to TPU topology specifications (`google.com/tpu`), and prove the generated manifest bundle passes Kubernetes dry-run validation.

You do not judge numerical parity, benchmark throughput, or evaluate latency SLOs. The calling orchestrator (the `tpu-model-onboarding` distributed serving workflow) deploys the server and evaluates baseline performance after you return. Never query `pvc,secret,sa,role,rolebinding,configmap` in the target namespace to discover or reuse unowned cluster resources (`PersistentVolumeClaim`, `Secret`, `ServiceAccount`, `RoleBinding`, `ConfigMap`) or node-cached proprietary images not explicitly provided by the user.

## Artifact Root (`<artifacts>`)

`<artifacts> = {{runs}}/<run_id>/artifacts`, `<work> = <artifacts>/tpu-model-onboarding`, `<dynamo_work> = <work>/dynamo_unwiring/<model_slug>`, where `<model_slug>` is the detected or requested `model_id` with every `/` replaced by `__`.
Return `invalid_input` if `run_id` is missing. Write only to `<dynamo_work>`, never to CWD or package files.

## Inputs

- `dynamo_manifests` (Optional if `source_namespace` is set): fully expanded path to the file or directory containing NVIDIA Dynamo Kubernetes manifests.
- `source_namespace` (Optional if `dynamo_manifests` is set): Kubernetes namespace of a live NVIDIA Dynamo deployment to inspect, export into `<dynamo_work>/source_store`, and translate.
- `source_context` / `source_kubeconfig` (Optional): Kubernetes context or kubeconfig path for inspecting `source_namespace`.
- `target_namespace` (Optional, default `"default"`): Kubernetes namespace for the translated `llm-d` stack.
- `target_context` / `target_kubeconfig` (Optional): Kubernetes context or kubeconfig path for the target GKE TPU cluster.
- `target_tpu_type` (Optional, default `"v6e"`): target TPU accelerator generation (`v4`, `v5e`, `v5p`, `v6e`, `v7x`, `tpu7x`).
- `target_tpu_topology` (Optional, default `"2x2x1"`): target TPU topology (e.g. `2x2x1` for 4 chips, `2x2x2` for 8 chips, `2x2x4` for 16 chips).
- `model_id` (Optional): Hugging Face model repository ID override.
- `deployments_json` (Optional): path to a single-VM `deployments.json` file whose tuned serving flags and environment variables are carried over into the translated `LeaderWorkerSet` pods.
- `vllm_patch` (Optional): path to a custom `vllm_torchtpu` `.patch` or `.py` file to mount via `ConfigMap` (`04-vllm-patches.yaml`) and apply at container startup.
- `standalone_epp` / `envoy_configmap` / `epp_service_account` (Optional): standalone EPP + Envoy router configuration overrides.
- `dry_run` (Optional, default `true`): whether to run Kubernetes dry-run validation on the generated bundle.

## Step 1: Ingest and Inspect the Source Deployment

Inspect the source manifests or live source namespace to identify the starting topology, router type, engine, and pool configurations.

If `dynamo_manifests` is provided:

```bash
python3 {{script:tpu-model-onboarding/inspect_deployment.py}} \
  --manifests <dynamo_manifests> \
  --json > <dynamo_work>/source_inspection.json
```

If `source_namespace` is provided (live cluster source inspection with manifest, cluster state, and pod log capture):

```bash
python3 {{script:tpu-model-onboarding/inspect_deployment.py}} \
  --namespace <source_namespace> \
  --store-dir <dynamo_work>/source_store \
  --capture-logs \
  $( [ -n "<source_context>" ] && echo "--context <source_context>" ) \
  $( [ -n "<source_kubeconfig>" ] && echo "--kubeconfig <source_kubeconfig>" ) \
  --json > <dynamo_work>/source_inspection.json
```

Read `<dynamo_work>/source_inspection.json`. Verify:
1. `topology`: Confirm whether the source deployment is homogeneous or disaggregated (separate prefill and decode worker pools).
2. `model_id`: Identify the model ID to serve. If absent from source args, use `model_id` from the request.
3. `replicas`: Determine total replica count and pool roles.

## Step 2: Translate Dynamo to GKE TPU llm-d

Read `{{reference:tpu-model-onboarding/llmd/dynamo-to-llmd-mapping.md}}` for parameter mapping rules.
Run the translator CLI (passing `--manifests <dynamo_manifests>` or `--store-dir <dynamo_work>/source_store`) to synthesize the complete GKE `llm-d` manifest bundle:

```bash
python3 {{script:tpu-model-onboarding/dynamo_translator.py}} \
  $( [ -n "<dynamo_manifests>" ] && echo "--manifests <dynamo_manifests>" || echo "--store-dir <dynamo_work>/source_store" ) \
  --output-dir <dynamo_work> \
  --namespace <target_namespace> \
  --tpu-type <target_tpu_type> \
  --tpu-topology <target_tpu_topology> \
  $( [ -n "<model_id>" ] && echo "--model-id <model_id>" ) \
  $( [ -n "<deployments_json>" ] && echo "--deployments-json <deployments_json>" ) \
  $( [ -n "<vllm_patch>" ] && echo "--vllm-patch <vllm_patch>" ) \
  $( [ "<standalone_epp>" = "true" ] && echo "--standalone-epp" ) \
  $( [ -n "<envoy_configmap>" ] && echo "--envoy-configmap <envoy_configmap>" ) \
  $( [ -n "<epp_service_account>" ] && echo "--epp-service-account <epp_service_account>" ) \
  $( [ -n "<target_context>" ] && echo "--context <target_context>" ) \
  $( [ -n "<target_kubeconfig>" ] && echo "--kubeconfig <target_kubeconfig>" )
```

This generates `<dynamo_work>/llmd_bundle/` and a self-contained Kustomize bundle in `<dynamo_work>/llmd_bundle/kustomize/` containing:
- `00-gateway.yaml`: Gateway API Gateway and HTTPRoute resources.
- `01-endpointpickerconfig.yaml`: EndpointPickerConfig ConfigMap plus the EPP Deployment and Service configured with prefix caching, token load scoring calibrated to TPU throughput, queue balancing, and flow control.
- `02-inferencepool.yaml`: Gateway API Inference Extension InferencePool linking the router strictly to `LeaderWorkerSet` leader pods (`leaderworkerset.sigs.k8s.io/worker-index: "0"`).
- `03-serving-deployment.yaml`: TPU model-server `LeaderWorkerSet` (`leaderworkerset.x-k8s.io/v1`, `size: 1` for single-host slices or `size > 1` for multi-host slices) with `google.com/tpu` resources, nodeSelector, topology annotations (`leaderworkerset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool`), `--block-size=768`, and any carried-over `deployments.json` flags/env vars.
- `04-vllm-patches.yaml`: Emitted automatically when `model_id` matches an entry in the `vllm_patches` registry (`registry.json`) or when `--vllm-patch` is passed; defines the `vllm-tpu-patches` `ConfigMap` (`apply_vllm_patch.sh` + `<patch_name>`) mounted at `/app/patches` and executed before `exec vllm serve`.
- `kustomize/`: Self-contained Kustomize overlay directory (`kustomization.yaml` listing `["00-gateway.yaml", "01-endpointpickerconfig.yaml", "02-inferencepool.yaml", "base-modelserver.yaml"]` plus `04-vllm-patches.yaml` when present in `resources:` and `patch-*.yaml` targeting `kind: LeaderWorkerSet` in `patches:`) so a single Kustomize `-k <dynamo_work>/llmd_bundle/kustomize` rollout deploys the entire `llm-d` stack in one command.
- `<dynamo_work>/mapping_report.json`: Detailed traceability report mapping source Dynamo configurations to target `llm-d` configurations.

## Step 3: Validate the Bundle

If `dry_run` is enabled, validate the Kustomize overlay and all generated YAML manifests via the translator CLI (which automatically writes the rendered Kustomize output to `<dynamo_work>/kustomize_rendered.yaml`):

```bash
python3 {{script:tpu-model-onboarding/dynamo_translator.py}} \
  $( [ -n "<dynamo_manifests>" ] && echo "--manifests <dynamo_manifests>" || echo "--store-dir <dynamo_work>/source_store" ) \
  --output-dir <dynamo_work> \
  --namespace <target_namespace> \
  $( [ -n "<target_context>" ] && echo "--context <target_context>" ) \
  $( [ -n "<target_kubeconfig>" ] && echo "--kubeconfig <target_kubeconfig>" ) \
  --dry-run > <dynamo_work>/dry_run.log 2>&1
```

If validation fails:
1. Inspect `<dynamo_work>/dry_run.log` to identify syntax errors, schema mismatches, or invalid resource fields.
2. Edit the generated manifests under `<dynamo_work>/llmd_bundle/` and `<dynamo_work>/llmd_bundle/kustomize/` to resolve the error.
3. Re-run validation until the Kustomize overlay and all manifests pass cleanly.

## Hand-off

Return a `dynamo-unwiring-result`:

- `ok`:
  - `bundle_dir`: fully expanded path to `<dynamo_work>/llmd_bundle/`
  - `manifest_files`: list of generated YAML files
  - `topology`: `"multi-replica-llmd-homogeneous"`
  - `model_id`: resolved model ID
  - `target_tpu_type`: target TPU generation
  - `target_tpu_topology`: target TPU topology
  - `mapping_report`: contents of `<dynamo_work>/mapping_report.json`
  - `dry_run_verified`: `true`
  - `log_dir`: `<dynamo_work>`
- `blocked`: source manifests contain unsupported custom plugins or invalid syntax that cannot be mapped. Provide `error` and path to `<dynamo_work>/source_inspection.json`.
- `failed`: filesystem or execution failure outside translation logic.
