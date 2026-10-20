---
references:
  - references/tpu-performance-optimization/llmd-telemetry-contract.md
scripts:
  - scripts/common/study_protocol.py
  - scripts/tpu-knowledge-base/kb_lookup.py
  - scripts/tpu-model-onboarding/llmd/benchmark.py
  - scripts/tpu-model-onboarding/llmd/parse_bench_results.py
  - scripts/tpu-performance-optimization/epp_config.py
  - scripts/tpu-performance-optimization/evaluate_arm.py
  - scripts/tpu-performance-optimization/hypothesis_bank.py
  - scripts/tpu-performance-optimization/llmd_telemetry.py
---

# Distributed Serving (`llm-d`) Optimization Reference

When the target is a distributed `llm-d` Kustomize bundle (`bundle_dir` containing `00-gateway.yaml`, `01-endpointpickerconfig.yaml`, `02-inferencepool.yaml`, and `03-serving-deployment.yaml`), the standard `tpu-performance-optimization` workflow (`tpu-performance-optimization` -> `tpu-performance-optimization-researcher` -> `tpu-performance-optimization-executor`) uses the following script contracts.

## Study Initialization & Baseline (`tpu-performance-optimization`)

1. If `{{runtime}} study status --study-id <study_id>` indicates the study has not been created yet, compile `<run>/protocol.json` (when not already provided via `protocol` or `protocol_path`) and start the study:
   ```bash
   python3 {{script:common/study_protocol.py}} \
     --prompt "<prompt>" --study-id <study_id> --run-id <run_id> \
     --output-dir <run> --json
   {{runtime}} study start      --study-id <study_id> --protocol <run>/protocol.json --package tpu-agent-suite --target <target>
   {{runtime}} study attach-run --study-id <study_id> --run-id <run_id>
   ```
2. Inspect the initial `llm-d` bundle to record its `scorer_weights`, `filters`, `pools`, and `recipe_fingerprint`:
   ```bash
   python3 {{script:tpu-performance-optimization/epp_config.py}} inspect \
     --bundle-dir <bundle_dir> --json > <run>/baseline_bundle_inspect.json
   ```
3. If the study has no baseline yet, reuse `baseline_metrics` if provided on disk or run the baseline load sweep against the Gateway or EPP endpoint, parse client outputs via `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}`, and aggregate telemetry into `{{schema:llmd-telemetry.json}}` (`{{reference:tpu-performance-optimization/llmd-telemetry-contract.md}}`):
   ```bash
   python3 {{script:tpu-model-onboarding/llmd/benchmark.py}} \
     --model <model_id> --harness inference-perf --stack llm-d \
     --workload-class high-prefix-share --namespace <namespace> \
     --target-url <service_url> --concurrency <c> --isl <isl> --osl <osl> --prefix-len <prefix_len> \
     --execute --logs-out <run>/baseline_bench.log --parse-out <run>/baseline_summary.json \
     --evidence-dir <run>/baseline_evidence
   python3 {{script:tpu-performance-optimization/llmd_telemetry.py}} \
     --benchmark-output <run>/baseline_summary.json \
     --arm-id baseline --model <model_id> --accelerator-type <tpu_type> \
     --output-json <run>/baseline_telemetry.json --validate
   ```
4. Write `<run>/baseline.json` (with `profile` pointing to `<run>/baseline_telemetry.json` and `fingerprint` set to `recipe_fingerprint` from `epp_config.py inspect`) and register it via `{{runtime}} study set-baseline --study-id <study_id> --baseline <run>/baseline.json`. Both `<run>/baseline_telemetry.json` and `epp_config.py inspect` provide empirical bottleneck attribution when no XProf trace is present.

## Hypothesis Formulation (`tpu-performance-optimization-researcher`)

1. Query `llmd-routing-and-epp` and `llmd-dynamo-unwiring` in the TPU Knowledge Base for the telemetry-to-bottleneck mapping table, nested `ConfigMap` synchronization rules, TPU MXU batching alignment, and SLA admissibility gates:
   ```bash
   echo '{"query": ["llmd-routing-and-epp", "llmd-dynamo-unwiring"]}' | python3 {{script:tpu-knowledge-base/kb_lookup.py}}
   ```
2. Inspect the current manifests via `python3 {{script:tpu-performance-optimization/epp_config.py}} inspect --bundle-dir <bundle_dir> --json`, correlate `{{schema:llmd-telemetry.json}}` signals to bottleneck `kind` (`request-routing`, `load-imbalance`, `scheduler-batching`, `recompilation-padding`, `memory-capacity`), and register 2 to 3 orthogonal hypotheses with `epp_config` and/or `target_parameters`:
   ```bash
   python3 {{script:tpu-performance-optimization/hypothesis_bank.py}} \
     --study-id <study_id> --run-id <run_id> \
     --hypotheses-json <work>/hypotheses.json \
     --tpu-version <tpu_type> --repo-root <bundle_dir>
   ```

## Arm Execution, Validation, and Rollback (`tpu-performance-optimization-executor`)

1. **Checkpoint & Direct YAML Edit (Phase 1)**: Before modifying any manifest in `<bundle_dir>`, save a bitwise-exact checkpoint:
   ```bash
   python3 {{script:tpu-performance-optimization/epp_config.py}} checkpoint \
     --bundle-dir <bundle_dir> --checkpoint-dir <work>/checkpoint --json
   ```
   Edit `01-endpointpickerconfig.yaml` (keeping both `endpoint-picker-config.yaml` and `default-plugins.yaml` in the `ConfigMap` synchronized when both keys exist) and/or `03-serving-deployment.yaml` directly using `edit` or `write`.
2. **Mandatory Validation Gate (Phase 2 Staged Triage)**: For Kustomize bundle orchestration arms, Stage 2 triage validates YAML syntax, nested `ConfigMap` `EndpointPickerConfig` schemas, non-negative scorer weights with positive sum, `LeaderWorkerSet` `vllm serve` flag invariants, and Kustomize dry-run rendering:
   ```bash
   python3 {{script:tpu-performance-optimization/epp_config.py}} validate \
     --bundle-dir <bundle_dir> --json
   ```
   If `"valid": false`, fix the reported `errors` in the YAML files and re-run `validate` until `"valid": true`.
3. **Cluster Rollout & Co-located Benchmark (Phase 3)**: Apply the validated bundle with `kubectl -n <namespace> apply -k <bundle_dir>` (and `kubectl -n <namespace> rollout restart deployment/llm-d-epp` if needed so the updated `ConfigMap` takes effect), wait for pod readiness, re-run `{{script:tpu-model-onboarding/llmd/benchmark.py}}` with the exact baseline workload flags, parse via `{{script:tpu-model-onboarding/llmd/parse_bench_results.py}}`, and aggregate into `<work>/llmd_telemetry.json` via `{{script:tpu-performance-optimization/llmd_telemetry.py}}`.
4. **Evaluation & Rollback on Non-Win (Phase 4)**: Evaluate the arm against the baseline and all SLA admissibility gates:
   ```bash
   python3 {{script:tpu-performance-optimization/evaluate_arm.py}} \
     --arm <arm> --execution-id <exec_id> \
     --telemetry <work>/llmd_telemetry.json \
     --protocol <protocol.json> --baseline <baseline.json> \
     > <work>/eval.json
   ```
   If `verdict` is `loss`, `tie`, or `invalid`, immediately restore the pre-mutation checkpoint and cluster state:
   ```bash
   python3 {{script:tpu-performance-optimization/epp_config.py}} rollback \
     --bundle-dir <bundle_dir> \
     --checkpoint-dir <work>/checkpoint \
     --apply --namespace <namespace>
   ```
   If `verdict` is `win`, retain the mutated bundle as the new incumbency baseline.
