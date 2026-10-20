# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for NVIDIA Dynamo unwiring and deployment inspection."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import yaml

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load dynamo_translator.py
_TRANSLATOR_SCRIPT = (_PACKAGE_ROOT / "scripts" / "tpu-model-onboarding" /
                      "dynamo_translator.py")
_trans_spec = importlib.util.spec_from_file_location("dynamo_translator",
                                                     _TRANSLATOR_SCRIPT)
dynamo_translator = importlib.util.module_from_spec(_trans_spec)
_trans_spec.loader.exec_module(dynamo_translator)

# Load inspect_deployment.py
_INSPECT_SCRIPT = (_PACKAGE_ROOT / "scripts" / "tpu-model-onboarding" /
                   "inspect_deployment.py")
_inspect_spec = importlib.util.spec_from_file_location("inspect_deployment",
                                                       _INSPECT_SCRIPT)
inspect_deployment = importlib.util.module_from_spec(_inspect_spec)
_inspect_spec.loader.exec_module(inspect_deployment)


class TestDynamoTranslator(unittest.TestCase):
    """Verifies Dynamo to GKE TPU llm-d manifest translation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_translate_homogeneous_dynamo_deployment(self):
        dynamo_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-frontend
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: frontend
        image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:latest
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-worker
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: worker
        image: vllm/vllm-openai:latest
        args: ["--model", "Qwen/Qwen3.5-27B", "--tensor-parallel-size", "4"]
        resources:
          limits:
            nvidia.com/gpu: 4
"""
        manifest_path = self.work_dir / "dynamo.yaml"
        manifest_path.write_text(dynamo_yaml, encoding="utf-8")

        out_dir = self.work_dir / "output"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="llm-serving",
            target_tpu_type="v6e",
            target_tpu_topology="2x2x1",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["topology"], "multi-replica-llmd-homogeneous")
        self.assertEqual(result["model_id"], "Qwen/Qwen3.5-27B")

        bundle_dir = pathlib.Path(result["bundle_dir"])
        self.assertTrue((bundle_dir / "00-gateway.yaml").exists())
        self.assertTrue((bundle_dir / "01-endpointpickerconfig.yaml").exists())
        self.assertTrue((bundle_dir / "02-inferencepool.yaml").exists())
        self.assertTrue((bundle_dir / "03-serving-deployment.yaml").exists())

        dep_content = (bundle_dir / "03-serving-deployment.yaml").read_text(
            encoding="utf-8")
        dep_doc = yaml.safe_load(dep_content)
        self.assertEqual(dep_doc["kind"], "LeaderWorkerSet")
        self.assertEqual(dep_doc["apiVersion"], "leaderworkerset.x-k8s.io/v1")
        self.assertEqual(dep_doc["spec"]["replicas"], 4)
        self.assertEqual(dep_doc["spec"]["leaderWorkerTemplate"]["size"], 1)
        worker_tmpl = dep_doc["spec"]["leaderWorkerTemplate"]["workerTemplate"]
        pod_spec = worker_tmpl["spec"]

        container = pod_spec["containers"][0]
        self.assertEqual(container["resources"]["limits"]["google.com/tpu"],
                         "4")
        self.assertEqual(
            pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"],
            "v6e",
        )
        self.assertEqual(
            pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-topology"],
            "2x2x1",
        )
        self.assertEqual(
            worker_tmpl["metadata"]["annotations"]
            ["cloud.google.com/gke-tpu-topology"],
            "2x2x1",
        )
        self.assertEqual(
            worker_tmpl["metadata"]["annotations"]
            ["cloud.google.com/gke-tpu-accelerator-label"],
            "tpu-v6e-slice",
        )
        self.assertNotIn(
            "leaderworkerset.sigs.k8s.io/exclusive-topology",
            worker_tmpl["metadata"]["annotations"],
        )

        pool_doc = yaml.safe_load(
            (bundle_dir / "02-inferencepool.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            pool_doc["spec"]["selector"]["matchLabels"]
            ["leaderworkerset.sigs.k8s.io/worker-index"],
            "0",
        )

        epp_content = (bundle_dir / "01-endpointpickerconfig.yaml").read_text(
            encoding="utf-8")
        epp_docs = [
            d for d in yaml.safe_load_all(epp_content) if isinstance(d, dict)
        ]
        self.assertEqual(len(epp_docs), 3)
        epp_doc = epp_docs[0]
        self.assertEqual(epp_doc["kind"], "ConfigMap")
        self.assertIn("endpoint-picker-config.yaml", epp_doc["data"])
        self.assertIn(
            "peakPrefillThroughput: 18500",
            epp_doc["data"]["endpoint-picker-config.yaml"],
        )
        self.assertEqual(epp_docs[1]["kind"], "Deployment")
        self.assertEqual(epp_docs[1]["metadata"]["name"], "llm-d-epp")
        self.assertEqual(epp_docs[2]["kind"], "Service")
        self.assertEqual(epp_docs[2]["metadata"]["name"], "llm-d-epp")

    def test_worker_service_not_misclassified_as_worker_deployment(self):
        """Regression test ensuring kind: Service with 'dynamo' and 'worker' in name is not counted as a worker Deployment."""
        dynamo_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-32b-dynamo-frontend
  labels:
    ai.dynamo/component: frontend
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: frontend
        image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-32b-dynamo-worker-cb4de3b1
  labels:
    ai.dynamo/component: worker
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: main
        image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1
        args: ["--model", "Qwen/Qwen3-32B", "--tensor-parallel-size", "2"]
        resources:
          limits:
            nvidia.com/gpu: 2
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-32b-dynamo-worker-cb4de3b1
  labels:
    ai.dynamo/component: worker
spec:
  ports:
  - name: system-port
    port: 9090
    targetPort: 9090
"""
        manifest_path = self.work_dir / "dynamo_with_worker_svc.yaml"
        manifest_path.write_text(dynamo_yaml, encoding="utf-8")

        out_dir = self.work_dir / "output_with_worker_svc"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="llm-d",
            target_tpu_type="tpu7x",
            target_tpu_topology="2x2x1",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(len(result["mapping_report"]["source_pools"]), 1)
        self.assertEqual(
            result["mapping_report"]["translated_pools"][0]["replicas"], 4)

        bundle_dir = pathlib.Path(result["bundle_dir"])
        dep_doc = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))
        self.assertEqual(dep_doc["kind"], "LeaderWorkerSet")
        self.assertEqual(dep_doc["spec"]["replicas"], 4)

    def test_translate_multinode_dynamographdeployment_to_leaderworkerset(
            self):
        dgd_yaml = """apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-5-397b-dgd
spec:
  services:
    frontend:
      image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.0
  components:
    worker:
      replicas: 2
      multinode:
        nodeCount: 2
      template:
        spec:
          containers:
            - name: worker
              image: vllm/vllm
              args:
                - --model
                - Qwen/Qwen3.5-397B-A17B-FP8
              resources:
                limits:
                  nvidia.com/gpu: 8
"""
        manifest_path = self.work_dir / "dgd_multinode.yaml"
        manifest_path.write_text(dgd_yaml, encoding="utf-8")

        out_dir = self.work_dir / "output_lws"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="lws-serving",
            target_tpu_type="v7x",
            target_tpu_topology="2x2x4",
        )

        self.assertEqual(result["status"], "ok")
        bundle_dir = pathlib.Path(result["bundle_dir"])
        serving_lws = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))

        self.assertEqual(serving_lws["kind"], "LeaderWorkerSet")
        self.assertEqual(serving_lws["apiVersion"],
                         "leaderworkerset.x-k8s.io/v1")
        self.assertEqual(serving_lws["spec"]["replicas"], 2)
        self.assertEqual(serving_lws["spec"]["leaderWorkerTemplate"]["size"],
                         2)
        w_tmpl = serving_lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]
        self.assertEqual(
            w_tmpl["metadata"]["annotations"]
            ["leaderworkerset.sigs.k8s.io/exclusive-topology"],
            "cloud.google.com/gke-nodepool",
        )
        self.assertEqual(
            w_tmpl["spec"]["nodeSelector"]
            ["cloud.google.com/gke-tpu-topology"],
            "2x2x4",
        )
        self.assertEqual(
            w_tmpl["spec"]["containers"][0]["resources"]["limits"]
            ["google.com/tpu"],
            "8",
        )

    def test_end_to_end_gateway_inferencepool_pod_selector_and_port_contract(
            self):
        """Verifies cross-resource compatibility across Gateway, HTTPRoute, InferencePool, EPP, and Workloads."""
        dgd_yaml = """apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: homog-contract-test
spec:
  services:
    frontend:
      image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.0
  components:
    worker:
      replicas: 2
      template:
        spec:
          containers:
            - name: worker
              image: vllm/vllm
              args: ["--model", "Qwen/Qwen3.5-27B"]
"""
        manifest_path = self.work_dir / "contract.yaml"
        manifest_path.write_text(dgd_yaml, encoding="utf-8")

        out_dir = self.work_dir / "output_contract"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="contract-ns",
            target_tpu_type="v6e",
            target_tpu_topology="2x2x1",
        )
        bundle_dir = pathlib.Path(result["bundle_dir"])

        gw_docs = list(
            yaml.safe_load_all(
                (bundle_dir / "00-gateway.yaml").read_text(encoding="utf-8")))
        gateway = gw_docs[0]
        httproute = gw_docs[1]
        pool = yaml.safe_load(
            (bundle_dir / "02-inferencepool.yaml").read_text(encoding="utf-8"))
        serving_dep = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))

        self.assertEqual(
            httproute["spec"]["parentRefs"][0]["name"],
            gateway["metadata"]["name"],
        )
        backend_ref = httproute["spec"]["rules"][0]["backendRefs"][0]
        self.assertEqual(backend_ref["kind"], "InferencePool")
        self.assertEqual(backend_ref["name"], pool["metadata"]["name"])

        match_labels = pool["spec"]["selector"]["matchLabels"]
        self.assertEqual(
            match_labels.get("leaderworkerset.sigs.k8s.io/worker-index"), "0")
        target_port = pool["spec"]["targetPorts"][0]["number"]

        self.assertEqual(serving_dep["kind"], "LeaderWorkerSet")
        worker_tmpl = serving_dep["spec"]["leaderWorkerTemplate"][
            "workerTemplate"]
        pod_labels = worker_tmpl["metadata"]["labels"]
        for k, v in match_labels.items():
            if k == "leaderworkerset.sigs.k8s.io/worker-index":
                continue
            self.assertEqual(
                pod_labels.get(k),
                v,
                f"Workload {serving_dep['metadata']['name']} missing InferencePool selector {k}={v}",
            )

        pod_spec = worker_tmpl["spec"]
        exposed_ports = [
            p["containerPort"] for c in pod_spec.get("containers", [])
            for p in c.get("ports", [])
        ]
        self.assertIn(target_port, exposed_ports)

    def test_kustomize_overlay_and_kubectl_kustomize_execution(self):
        """Verifies Kustomize overlay generation and executes `kubectl kustomize`."""
        dynamo_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: vllm/vllm-openai:latest
        args: ["--model", "Qwen/Qwen3.5-27B"]
"""
        manifest_path = self.work_dir / "kustomize_input.yaml"
        manifest_path.write_text(dynamo_yaml, encoding="utf-8")

        out_dir = self.work_dir / "output_kustomize"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="kustomize-ns",
            target_tpu_type="v6e",
            target_tpu_topology="2x2x1",
        )
        kustomize_dir = pathlib.Path(result["kustomize_dir"])
        self.assertTrue((kustomize_dir / "kustomization.yaml").exists())
        self.assertTrue((kustomize_dir / "00-gateway.yaml").exists())
        self.assertTrue(
            (kustomize_dir / "01-endpointpickerconfig.yaml").exists())
        self.assertTrue((kustomize_dir / "02-inferencepool.yaml").exists())
        self.assertTrue((kustomize_dir / "base-modelserver.yaml").exists())

        kustomization_doc = yaml.safe_load(
            (kustomize_dir / "kustomization.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            kustomization_doc["resources"],
            [
                "00-gateway.yaml",
                "01-endpointpickerconfig.yaml",
                "02-inferencepool.yaml",
                "base-modelserver.yaml",
            ],
        )
        self.assertEqual(len(kustomization_doc.get("patches", [])), 1)
        self.assertEqual(
            kustomization_doc["patches"][0]["target"]["kind"],
            "LeaderWorkerSet",
        )

        if shutil.which("kubectl"):
            proc = subprocess.run(
                ["kubectl", "kustomize",
                 str(kustomize_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            rendered_docs = [
                d for d in yaml.safe_load_all(proc.stdout)
                if isinstance(d, dict)
            ]
            kinds = [d["kind"] for d in rendered_docs]
            self.assertIn("Gateway", kinds)
            self.assertIn("HTTPRoute", kinds)
            self.assertIn("ConfigMap", kinds)
            self.assertIn("Service", kinds)
            self.assertIn("InferencePool", kinds)
            self.assertIn("ServiceAccount", kinds)
            self.assertIn("Deployment", kinds)
            self.assertIn("LeaderWorkerSet", kinds)
            workload_lws = [
                d for d in rendered_docs if d["kind"] == "LeaderWorkerSet"
            ]
            self.assertEqual(len(workload_lws), 1)
            self.assertEqual(
                workload_lws[0]["metadata"]["annotations"]
                ["llm-d.ai/translated-from"],
                "nvidia-dynamo",
            )
            epp_deps = [
                d for d in rendered_docs if d["kind"] == "Deployment"
                and d["metadata"]["name"] == "llm-d-epp"
            ]
            self.assertEqual(len(epp_deps), 1)

            dry_run_success, dry_run_error = (
                dynamo_translator.validate_bundle_dry_run(
                    bundle_dir=pathlib.Path(result["bundle_dir"]),
                    kustomize_dir=kustomize_dir,
                    namespace="kustomize-ns",
                ))
            self.assertTrue(dry_run_success, dry_run_error)

    def test_translate_real_life_agentx_dynamo_to_llmd(self):
        """Verifies end-to-end inspection and translation of a real-life SemiAnalysis AgentX GPU Dynamo deployment into GKE Cloud TPU llm-d manifests."""
        agentx_dynamo_yaml = """apiVersion: v1
kind: Service
metadata:
  name: agentx-dynamo-frontend
  namespace: agentx-bench
  labels:
    app: agentx-dynamo
    ai.dynamo/component: frontend
spec:
  selector:
    app: agentx-dynamo
    ai.dynamo/component: frontend
  ports:
    - name: http
      port: 8000
      targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentx-dynamo-frontend
  namespace: agentx-bench
  labels:
    app: agentx-dynamo
    ai.dynamo/component: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agentx-dynamo
      ai.dynamo/component: frontend
  template:
    metadata:
      labels:
        app: agentx-dynamo
        ai.dynamo/component: frontend
    spec:
      containers:
        - name: dynamo-frontend
          image: nvcr.io/nvidia/ai-dynamo/dynamo-frontend:v0.3.0
          args:
            - --router-mode=kv
            - --http-port=8000
          env:
            - name: DYN_ROUTER_MODE
              value: kv
            - name: AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID
              value: "1"
          ports:
            - name: http
              containerPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentx-vllm-worker
  namespace: agentx-bench
  labels:
    app: agentx-dynamo
    ai.dynamo/component: worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agentx-dynamo
      ai.dynamo/component: worker
  template:
    metadata:
      labels:
        app: agentx-dynamo
        ai.dynamo/component: worker
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-h100-80gb
        cloud.google.com/gke-nodepool: a3-highgpu-8g
      containers:
        - name: vllm-worker
          image: vllm/vllm-openai:v0.8.5
          args:
            - --model=Qwen/Qwen2.5-72B-Instruct
            - --tensor-parallel-size=8
            - --enable-prefix-caching
            - --max-model-len=32768
            - --gpu-memory-utilization=0.90
          env:
            - name: NCCL_IB_DISABLE
              value: "0"
            - name: UCX_TLS
              value: rc,sm,cuda_copy,cuda_ipc
          resources:
            limits:
              nvidia.com/gpu: 8
            requests:
              nvidia.com/gpu: 8
          ports:
            - name: http
              containerPort: 8000
"""
        manifest_path = self.work_dir / "agentx_dynamo_gpu.yaml"
        manifest_path.write_text(agentx_dynamo_yaml, encoding="utf-8")

        source_report = inspect_deployment.inspect_deployment(
            manifests=manifest_path)
        self.assertEqual(source_report["topology"], "dynamo-gpu")
        self.assertEqual(source_report["model_id"],
                         "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(source_report["engine"], "vllm")
        self.assertEqual(source_report["router"], "dynamo-frontend")
        self.assertTrue(source_report["router_config"]["session_affinity"])
        self.assertTrue(source_report["router_config"]["kv_routing"])
        self.assertEqual(
            source_report["router_config"]["session_headers"],
            ["X-Session-ID", "X-Correlation-ID"],
        )
        self.assertEqual(source_report["router_config"]["frontend_port"], 8000)
        self.assertTrue(source_report["accelerator"]["is_gpu"])
        self.assertEqual(source_report["accelerator"]["chips_per_node"], 8)
        self.assertEqual(source_report["accelerator"]["total_chips"], 16)
        self.assertEqual(
            source_report["accelerator"]["gke_accelerator"],
            "nvidia-h100-80gb",
        )
        self.assertEqual(source_report["accelerator"]["node_pool"],
                         "a3-highgpu-8g")

        out_dir = self.work_dir / "output_agentx_llmd"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="agentx-tpu",
            target_tpu_type="v6e",
            target_tpu_topology="2x4",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["topology"], "multi-replica-llmd-homogeneous")
        self.assertEqual(result["model_id"], "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(result["target_tpu_type"], "v6e")
        self.assertEqual(result["target_tpu_topology"], "2x4")

        mapping_report = result["mapping_report"]
        self.assertEqual(mapping_report["source_router"], "dynamo-frontend")
        self.assertTrue(mapping_report["frontend_config"]["session_affinity"])
        self.assertTrue(mapping_report["frontend_config"]["kv_routing"])
        self.assertEqual(mapping_report["frontend_config"]["port"], 8000)
        self.assertEqual(
            mapping_report["session_affinity"]["headers"],
            ["X-Session-ID", "X-Correlation-ID"],
        )

        bundle_dir = pathlib.Path(result["bundle_dir"])
        kustomize_dir = pathlib.Path(result["kustomize_dir"])
        for expected_file in (
                "00-gateway.yaml",
                "01-endpointpickerconfig.yaml",
                "02-inferencepool.yaml",
                "03-serving-deployment.yaml",
        ):
            self.assertTrue((bundle_dir / expected_file).exists(),
                            f"Missing {expected_file} in bundle_dir")
        for expected_kust_file in (
                "00-gateway.yaml",
                "01-endpointpickerconfig.yaml",
                "02-inferencepool.yaml",
                "base-modelserver.yaml",
                "patch-vllm-tpu-workers.yaml",
                "kustomization.yaml",
        ):
            self.assertTrue((kustomize_dir / expected_kust_file).exists(),
                            f"Missing {expected_kust_file} in kustomize_dir")

        gw_docs = [
            d for d in yaml.safe_load_all((bundle_dir /
                                           "00-gateway.yaml").read_text(
                                               encoding="utf-8"))
            if isinstance(d, dict)
        ]
        self.assertEqual(len(gw_docs), 2)
        self.assertEqual(gw_docs[0]["apiVersion"],
                         "gateway.networking.k8s.io/v1")
        self.assertEqual(gw_docs[0]["kind"], "Gateway")
        self.assertEqual(gw_docs[1]["kind"], "HTTPRoute")
        backend_ref = gw_docs[1]["spec"]["rules"][0]["backendRefs"][0]
        self.assertEqual(backend_ref["kind"], "InferencePool")
        self.assertEqual(backend_ref["name"], "llm-d-pool")
        self.assertEqual(backend_ref["port"], 8000)

        epp_docs = [
            d for d in yaml.safe_load_all((
                bundle_dir /
                "01-endpointpickerconfig.yaml").read_text(encoding="utf-8"))
            if isinstance(d, dict)
        ]
        self.assertEqual(len(epp_docs), 3)
        epp_cfg = yaml.safe_load(
            epp_docs[0]["data"]["endpoint-picker-config.yaml"])
        self.assertEqual(epp_cfg["kind"], "EndpointPickerConfig")
        filters = {f["name"]: f for f in epp_cfg["scheduling"]["filters"]}
        self.assertIn("prefix-cache-affinity-filter", filters)
        self.assertEqual(
            filters["prefix-cache-affinity-filter"]["parameters"]
            ["sessionAffinityHeaders"],
            ["X-Session-ID", "X-Correlation-ID"],
        )
        scorers = {s["name"]: s for s in epp_cfg["scheduling"]["scorers"]}
        self.assertIn("prefix-cache-scorer", scorers)
        self.assertEqual(scorers["prefix-cache-scorer"]["weight"], 100)
        epp_env = {
            e["name"]: e["value"]
            for e in epp_docs[1]["spec"]["template"]["spec"]["containers"]
            [0].get("env", [])
        }
        self.assertEqual(
            epp_env.get("SESSION_AFFINITY_HEADERS"),
            "X-Session-ID,X-Correlation-ID",
        )

        pool_doc = yaml.safe_load(
            (bundle_dir / "02-inferencepool.yaml").read_text(encoding="utf-8"))
        self.assertEqual(pool_doc["kind"], "InferencePool")
        self.assertEqual(pool_doc["spec"]["targetPortNumber"], 8000)
        self.assertEqual(pool_doc["spec"]["targetPorts"], [{"number": 8000}])
        self.assertEqual(pool_doc["spec"]["endpointPickerConfigRef"]["name"],
                         "epp-config")

        dep_doc = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))
        self.assertEqual(dep_doc["kind"], "LeaderWorkerSet")
        self.assertEqual(dep_doc["spec"]["replicas"], 2)
        self.assertEqual(dep_doc["spec"]["leaderWorkerTemplate"]["size"], 1)
        pod_tmpl = dep_doc["spec"]["leaderWorkerTemplate"]["workerTemplate"]
        self.assertEqual(
            pod_tmpl["metadata"]["annotations"]
            ["cloud.google.com/gke-tpu-topology"],
            "2x4",
        )
        self.assertEqual(
            pod_tmpl["metadata"]["annotations"]
            ["cloud.google.com/gke-tpu-accelerator-label"],
            "tpu-v6e-slice",
        )
        pod_spec = pod_tmpl["spec"]
        self.assertEqual(
            pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"],
            "v6e",
        )
        container = pod_spec["containers"][0]
        self.assertEqual(container["resources"]["limits"]["google.com/tpu"],
                         "8")
        self.assertEqual(container["resources"]["requests"]["google.com/tpu"],
                         "8")
        c_args = container["args"]
        self.assertIn("Qwen/Qwen2.5-72B-Instruct", c_args)
        self.assertIn("--enable-prefix-caching", c_args)
        max_len_idx = c_args.index("--max-model-len")
        self.assertEqual(c_args[max_len_idx + 1], "32768")
        tp_idx = c_args.index("--tensor-parallel-size")
        self.assertEqual(c_args[tp_idx + 1], "8")

        if shutil.which("kubectl"):
            kust_proc = subprocess.run(
                ["kubectl", "kustomize",
                 str(kustomize_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            rendered_docs = [
                d for d in yaml.safe_load_all(kust_proc.stdout)
                if isinstance(d, dict)
            ]
            rendered_kinds = [d["kind"] for d in rendered_docs]
            for expected_kind in (
                    "Gateway",
                    "HTTPRoute",
                    "ConfigMap",
                    "Deployment",
                    "LeaderWorkerSet",
                    "Service",
                    "InferencePool",
                    "ServiceAccount",
            ):
                self.assertIn(expected_kind, rendered_kinds)

            dry_run_ok, dry_run_err = (
                dynamo_translator.validate_bundle_dry_run(
                    bundle_dir=bundle_dir,
                    kustomize_dir=kustomize_dir,
                    namespace="agentx-tpu",
                ))
            self.assertTrue(dry_run_ok, dry_run_err)

        translated_report = inspect_deployment.inspect_deployment(
            manifests=bundle_dir)
        self.assertEqual(translated_report["topology"],
                         "multi-replica-llmd-homogeneous")
        self.assertEqual(translated_report["model_id"],
                         "Qwen/Qwen2.5-72B-Instruct")
        self.assertEqual(translated_report["engine"], "vllm")
        self.assertEqual(translated_report["router"], "llm-d-epp")
        self.assertTrue(translated_report["router_config"]["session_affinity"])
        self.assertTrue(translated_report["router_config"]["kv_routing"])
        self.assertEqual(
            translated_report["router_config"]["session_headers"],
            ["X-Session-ID", "X-Correlation-ID"],
        )
        self.assertTrue(translated_report["accelerator"]["is_tpu"])
        self.assertEqual(translated_report["accelerator"]["type"], "v6e")
        self.assertEqual(translated_report["accelerator"]["topology"], "2x4")
        self.assertEqual(translated_report["accelerator"]["chips_per_node"], 8)
        self.assertEqual(translated_report["accelerator"]["total_chips"], 16)

    def test_cluster_live_inspect_and_translate(self):
        """End-to-end live cluster inspection -> store_dir -> Kustomize translation -> round-trip & redeploy."""
        live_cluster_resources = [
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "dynamo-frontend",
                    "namespace": "dynamo-prod",
                    "uid": "svc-uid-1001",
                    "resourceVersion": "94821",
                    "managedFields": [{
                        "manager": "kubectl"
                    }],
                    "labels": {
                        "ai.dynamo/component": "frontend",
                        "app": "dynamo-frontend",
                    },
                },
                "spec": {
                    "selector": {
                        "app": "dynamo-frontend",
                        "ai.dynamo/component": "frontend",
                    },
                    "ports": [{
                        "name": "http",
                        "port": 8000,
                        "targetPort": 8000
                    }],
                },
                "status": {
                    "loadBalancer": {}
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "dynamo-frontend",
                    "namespace": "dynamo-prod",
                    "uid": "dep-uid-2001",
                    "resourceVersion": "94822",
                    "labels": {
                        "ai.dynamo/component": "frontend",
                        "app": "dynamo-frontend",
                    },
                },
                "spec": {
                    "replicas": 1,
                    "selector": {
                        "matchLabels": {
                            "app": "dynamo-frontend",
                            "ai.dynamo/component": "frontend",
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "dynamo-frontend",
                                "ai.dynamo/component": "frontend",
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name":
                                "frontend",
                                "image":
                                "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.0",
                                "args": [
                                    "--router-mode=kv",
                                    "--port=8000",
                                ],
                                "env": [
                                    {
                                        "name":
                                        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
                                        "value": "1",
                                    },
                                    {
                                        "name": "DYN_ROUTER_MODE",
                                        "value": "kv",
                                    },
                                ],
                                "ports": [{
                                    "name": "http",
                                    "containerPort": 8000
                                }],
                            }]
                        },
                    },
                },
                "status": {
                    "availableReplicas": 1
                },
            },
        ]

        for pod_idx in range(4):
            live_cluster_resources.append({
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": f"dynamo-worker-{pod_idx}",
                    "namespace": "dynamo-prod",
                    "uid": f"pod-uid-300{pod_idx}",
                    "resourceVersion": f"9490{pod_idx}",
                    "labels": {
                        "ai.dynamo/component": "worker",
                        "app": "dynamo-worker",
                    },
                },
                "spec": {
                    "nodeSelector": {
                        "cloud.google.com/gke-accelerator": "nvidia-h100-80gb",
                        "cloud.google.com/gke-nodepool": "h100-gpu-pool",
                    },
                    "containers": [{
                        "name":
                        "worker",
                        "image":
                        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0",
                        "args": [
                            "--model",
                            "Qwen/Qwen3-32B",
                            "--tensor-parallel-size",
                            "2",
                            "--max-model-len",
                            "32768",
                            "--enable-prefix-caching",
                        ],
                        "env": [
                            {
                                "name": "NCCL_IB_DISABLE",
                                "value": "0"
                            },
                            {
                                "name": "UCX_TLS",
                                "value": "rc,cuda_copy"
                            },
                        ],
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": "2"
                            }
                        },
                    }],
                },
                "status": {
                    "phase": "Running"
                },
            })

        store_dir = self.work_dir / "cluster_runtime_store"

        with unittest.mock.patch.object(
                inspect_deployment,
                "query_cluster_resources",
                return_value=live_cluster_resources,
        ) as mock_query:
            live_report = inspect_deployment.inspect_deployment(
                namespace="dynamo-prod",
                store_dir=store_dir,
            )
            mock_query.assert_called_once_with(
                "dynamo-prod",
                kubeconfig=None,
                context=None,
            )

        self.assertEqual(live_report["topology"], "dynamo-gpu")
        self.assertEqual(live_report["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(live_report["engine"], "vllm")
        self.assertEqual(live_report["router"], "dynamo-frontend")
        self.assertTrue(live_report["router_config"]["session_affinity"])
        self.assertTrue(live_report["router_config"]["kv_routing"])
        self.assertEqual(
            live_report["router_config"]["session_headers"],
            ["X-Session-ID", "X-Correlation-ID"],
        )
        self.assertTrue(live_report["accelerator"]["is_gpu"])
        self.assertEqual(
            live_report["accelerator"]["gke_accelerator"],
            "nvidia-h100-80gb",
        )
        self.assertEqual(live_report["accelerator"]["chips_per_node"], 2)
        self.assertEqual(live_report["accelerator"]["total_chips"], 8)
        self.assertEqual(live_report["replicas"]["total"], 4)

        self.assertTrue((store_dir / "inspection_report.json").is_file())
        manifests_dir = store_dir / "manifests"
        self.assertTrue(manifests_dir.is_dir())
        captured_files = sorted(manifests_dir.glob("*.yaml"))
        self.assertEqual(len(captured_files), 6)
        first_captured = yaml.safe_load(
            captured_files[0].read_text(encoding="utf-8"))
        self.assertNotIn("status", first_captured)
        self.assertNotIn("uid", first_captured["metadata"])
        self.assertNotIn("managedFields", first_captured["metadata"])

        trans_result = dynamo_translator.translate_dynamo_to_llmd(
            store_dir=store_dir,
            target_namespace="llmd-tpu-prod",
            target_tpu_type="v6e",
            target_tpu_topology="2x4",
            dry_run=bool(shutil.which("kubectl")),
        )
        self.assertEqual(trans_result["status"], "ok")
        self.assertEqual(trans_result["store_dir"], str(store_dir))
        self.assertEqual(trans_result["topology"],
                         "multi-replica-llmd-homogeneous")
        self.assertEqual(trans_result["model_id"], "Qwen/Qwen3-32B")

        kustomize_dir = pathlib.Path(trans_result["kustomize_dir"])
        self.assertEqual(kustomize_dir, store_dir / "kustomize")
        for expected_file in (
                "00-gateway.yaml",
                "01-endpointpickerconfig.yaml",
                "02-inferencepool.yaml",
                "base-modelserver.yaml",
                "patch-vllm-tpu-workers.yaml",
                "kustomization.yaml",
        ):
            self.assertTrue(
                (kustomize_dir / expected_file).is_file(),
                f"Missing {expected_file} in {kustomize_dir}",
            )

        mapping_report = trans_result["mapping_report"]
        self.assertEqual(mapping_report["source_inspection_topology"],
                         "dynamo-gpu")
        self.assertEqual(mapping_report["translated_pools"][0]["replicas"], 4)

        if shutil.which("kubectl"):
            kust_proc = subprocess.run(
                ["kubectl", "kustomize",
                 str(kustomize_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            rendered_docs = [
                d for d in yaml.safe_load_all(kust_proc.stdout)
                if isinstance(d, dict)
            ]
            rendered_kinds = [d["kind"] for d in rendered_docs]
            for expected_kind in (
                    "Gateway",
                    "HTTPRoute",
                    "ConfigMap",
                    "Deployment",
                    "LeaderWorkerSet",
                    "Service",
                    "InferencePool",
                    "ServiceAccount",
            ):
                self.assertIn(expected_kind, rendered_kinds)

            worker_lws = [
                d for d in rendered_docs if d["kind"] == "LeaderWorkerSet"
                and d["metadata"]["name"] == "vllm-tpu-workers"
            ]
            self.assertEqual(len(worker_lws), 1)
            worker_dep = worker_lws[0]
            self.assertEqual(worker_dep["spec"]["replicas"], 4)
            self.assertEqual(
                worker_dep["metadata"]["annotations"].get(
                    "llm-d.ai/translated-from"),
                "nvidia-dynamo",
            )
            self.assertTrue(trans_result["dry_run_verified"])

        roundtrip_report = inspect_deployment.inspect_deployment(
            manifests=kustomize_dir)
        self.assertEqual(roundtrip_report["topology"],
                         "multi-replica-llmd-homogeneous")
        self.assertEqual(roundtrip_report["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(roundtrip_report["engine"], "vllm")
        self.assertEqual(roundtrip_report["router"], "llm-d-epp")
        self.assertTrue(roundtrip_report["router_config"]["session_affinity"])
        self.assertTrue(roundtrip_report["router_config"]["kv_routing"])
        self.assertTrue(roundtrip_report["accelerator"]["is_tpu"])
        self.assertEqual(roundtrip_report["accelerator"]["type"], "v6e")
        self.assertEqual(roundtrip_report["accelerator"]["topology"], "2x4")
        self.assertEqual(roundtrip_report["accelerator"]["chips_per_node"], 8)
        self.assertEqual(roundtrip_report["accelerator"]["total_chips"], 32)
        self.assertEqual(roundtrip_report["replicas"]["total"], 4)

    def test_target_cluster_only_preflight_does_not_dump_namespace_workloads(
            self):
        """Verifies target_cluster_only=True only inspects CRDs and TPU node capacity without dumping namespace manifests."""
        target_cluster_resources = [
            {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {
                    "name": "gke-tpu7x-pool-node-0",
                    "labels": {
                        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                        "cloud.google.com/gke-tpu-topology": "2x2x1",
                        "cloud.google.com/gke-nodepool": "tpu7x-pool",
                    },
                },
                "status": {
                    "capacity": {
                        "google.com/tpu": "4"
                    },
                    "conditions": [{
                        "type": "Ready",
                        "status": "True"
                    }],
                },
            },
            {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": "leaderworkersets.leaderworkerset.x-k8s.io"
                },
            },
            {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": "inferencepools.inference.networking.k8s.io"
                },
            },
            {
                "apiVersion": "apiextensions.k8s.io/v1",
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": "gateways.gateway.networking.k8s.io"
                },
            },
        ]
        preflight_store = self.work_dir / "preflight_store"
        with unittest.mock.patch.object(
                inspect_deployment,
                "query_target_cluster_resources",
                return_value=target_cluster_resources,
        ) as mock_target_query, unittest.mock.patch.object(
                inspect_deployment,
                "query_cluster_resources",
        ) as mock_ns_query:
            report = inspect_deployment.inspect_deployment(
                namespace="vllm-serving",
                store_dir=preflight_store,
                target_cluster_only=True,
            )
            mock_target_query.assert_called_once_with(
                kubeconfig=None,
                context=None,
            )
            mock_ns_query.assert_not_called()

        self.assertEqual(report["topology"], "target-cluster-preflight")
        self.assertTrue(report["target_cluster_only"])
        self.assertEqual(
            report["installed_crds"],
            ["Gateway", "InferencePool", "LeaderWorkerSet"],
        )
        self.assertTrue(report["accelerator"]["is_tpu"])
        self.assertEqual(report["accelerator"]["type"], "tpu7x")
        self.assertEqual(report["accelerator"]["topology"], "2x2x1")
        self.assertEqual(report["accelerator"]["chips_per_node"], 4)
        self.assertEqual(report["accelerator"]["total_chips"], 4)
        self.assertEqual(report["accelerator"]["total_nodes"], 1)
        self.assertEqual(report["accelerator"]["ready_nodes"], 1)
        self.assertEqual(report["replicas"]["total"], 0)
        self.assertTrue((preflight_store / "inspection_report.json").is_file())
        self.assertFalse((preflight_store / "manifests").exists())

    def test_dynamo_to_llmd_golden_snapshot(self):
        """Snapshot regression test comparing translated Dynamo GPU Qwen3 fixtures against golden TPU v6e llm-d overlay."""
        fixtures_root = _PACKAGE_ROOT / "tests" / "fixtures"
        golden_src_dir = fixtures_root / "dynamo_gpu_qwen3"
        golden_target_dir = fixtures_root / "llmd_tpu_v6e_qwen3"

        self.assertTrue(golden_src_dir.is_dir())
        self.assertTrue(golden_target_dir.is_dir())

        src_report = inspect_deployment.inspect_deployment(
            manifests=golden_src_dir)
        self.assertEqual(src_report["topology"], "dynamo-gpu")
        self.assertEqual(src_report["model_id"], "Qwen/Qwen3-32B")
        self.assertEqual(src_report["engine"], "vllm")
        self.assertEqual(src_report["router"], "dynamo-frontend")
        self.assertTrue(src_report["router_config"]["session_affinity"])
        self.assertTrue(src_report["router_config"]["kv_routing"])
        self.assertEqual(src_report["replicas"]["total"], 4)
        self.assertEqual(src_report["accelerator"]["total_chips"], 8)
        self.assertEqual(src_report["accelerator"]["chips_per_node"], 2)
        self.assertEqual(
            src_report["accelerator"]["gke_accelerator"],
            "nvidia-h100-80gb",
        )

        out_dir = self.work_dir / "snapshot_out"
        trans_result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=golden_src_dir,
            output_dir=out_dir,
            target_namespace="llmd-qwen3-prod",
            target_tpu_type="v6e",
            target_tpu_topology="2x4",
            dry_run=bool(shutil.which("kubectl")),
        )
        self.assertEqual(trans_result["status"], "ok")
        generated_kustomize_dir = pathlib.Path(trans_result["kustomize_dir"])
        bundle_dir = pathlib.Path(trans_result["bundle_dir"])

        generated_files = sorted(
            f.name for f in generated_kustomize_dir.glob("*.yaml"))
        golden_files = sorted(f.name for f in golden_target_dir.glob("*.yaml"))
        self.assertEqual(
            generated_files,
            [
                "00-gateway.yaml",
                "01-endpointpickerconfig.yaml",
                "02-inferencepool.yaml",
                "base-modelserver.yaml",
                "kustomization.yaml",
                "patch-vllm-tpu-workers.yaml",
            ],
        )
        self.assertEqual(generated_files, golden_files)

        for fname in golden_files:
            gen_docs = [
                d for d in yaml.safe_load_all((generated_kustomize_dir /
                                               fname).read_text(
                                                   encoding="utf-8"))
                if isinstance(d, dict)
            ]
            gold_docs = [
                d for d in yaml.safe_load_all((golden_target_dir /
                                               fname).read_text(
                                                   encoding="utf-8"))
                if isinstance(d, dict)
            ]
            self.assertEqual(
                gen_docs,
                gold_docs,
                f"Generated Kustomize manifest {fname} diverged from golden fixture",
            )

            if fname == "01-endpointpickerconfig.yaml":
                gen_epp_cfg = yaml.safe_load(
                    gen_docs[0]["data"]["endpoint-picker-config.yaml"])
                gold_epp_cfg = yaml.safe_load(
                    gold_docs[0]["data"]["endpoint-picker-config.yaml"])
                self.assertEqual(gen_epp_cfg, gold_epp_cfg)

        if shutil.which("kubectl"):
            gen_kust_proc = subprocess.run(
                ["kubectl", "kustomize",
                 str(generated_kustomize_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            gold_kust_proc = subprocess.run(
                ["kubectl", "kustomize",
                 str(golden_target_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            gen_rendered = [
                d for d in yaml.safe_load_all(gen_kust_proc.stdout)
                if isinstance(d, dict)
            ]
            gold_rendered = [
                d for d in yaml.safe_load_all(gold_kust_proc.stdout)
                if isinstance(d, dict)
            ]
            self.assertEqual(gen_rendered, gold_rendered)

            self.assertTrue(trans_result["dry_run_verified"])
            gold_dry_ok, gold_dry_err = (
                dynamo_translator.validate_bundle_dry_run(
                    bundle_dir=bundle_dir,
                    kustomize_dir=golden_target_dir,
                    namespace="llmd-qwen3-prod",
                ))
            self.assertTrue(gold_dry_ok, gold_dry_err)

        for overlay_dir in (generated_kustomize_dir, golden_target_dir):
            roundtrip = inspect_deployment.inspect_deployment(
                manifests=overlay_dir)
            self.assertEqual(roundtrip["topology"],
                             "multi-replica-llmd-homogeneous")
            self.assertEqual(roundtrip["model_id"], "Qwen/Qwen3-32B")
            self.assertEqual(roundtrip["engine"], "vllm")
            self.assertEqual(roundtrip["router"], "llm-d-epp")
            self.assertTrue(roundtrip["router_config"]["session_affinity"])
            self.assertTrue(roundtrip["router_config"]["kv_routing"])
            self.assertTrue(roundtrip["accelerator"]["is_tpu"])
            self.assertEqual(roundtrip["accelerator"]["type"], "v6e")
            self.assertEqual(roundtrip["accelerator"]["topology"], "2x4")
            self.assertEqual(roundtrip["accelerator"]["chips_per_node"], 8)
            self.assertEqual(roundtrip["accelerator"]["total_chips"], 32)
            self.assertEqual(roundtrip["replicas"]["total"], 4)

    def test_translate_with_deployments_json_and_store_dir_autodiscovery(self):
        """Verifies single-VM deployments.json flag and env carry-over into translated LeaderWorkerSet."""
        dynamo_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: vllm/vllm-openai:latest
        args: ["--model", "Qwen/Qwen3-32B", "--tensor-parallel-size", "2", "--max-model-len", "8192"]
        resources:
          limits:
            nvidia.com/gpu: 2
"""
        store_dir = self.work_dir / "store_with_deployments"
        manifests_dir = store_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        (manifests_dir / "dynamo.yaml").write_text(dynamo_yaml,
                                                   encoding="utf-8")

        deployments_payload = {
            "deployments": [{
                "model_id": "Qwen/Qwen3-32B",
                "engine": "vllm",
                "port": 8000,
                "tensor_parallel_size": 4,
                "data_parallel_size": 2,
                "max_model_len": 16384,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 8192,
                "attention_backend": "CUSTOM",
                "enable_chunked_prefill": True,
                "env": {
                    "VLLM_XLA_CACHE_PATH": "/tmp/xla_cache",
                    "TPU_SKIP_MDS_QUERY": "1",
                },
            }]
        }
        deployments_file = store_dir / "deployments.json"
        deployments_file.write_text(json.dumps(deployments_payload, indent=2),
                                    encoding="utf-8")

        result = dynamo_translator.translate_dynamo_to_llmd(
            store_dir=store_dir,
            target_namespace="tuned-ns",
            target_tpu_type="v6e",
            target_tpu_topology="2x4",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deployments_json"], str(deployments_file))
        self.assertEqual(
            result["mapping_report"]["deployments_json"],
            str(deployments_file),
        )

        bundle_dir = pathlib.Path(result["bundle_dir"])
        lws_doc = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lws_doc["kind"], "LeaderWorkerSet")
        container = lws_doc["spec"]["leaderWorkerTemplate"]["workerTemplate"][
            "spec"]["containers"][0]
        args = container["args"]

        def _flag_val(flag: str) -> str:
            idx = args.index(flag)
            return args[idx + 1]

        self.assertEqual(_flag_val("--tensor-parallel-size"), "4")
        self.assertEqual(_flag_val("--data-parallel-size"), "2")
        self.assertEqual(_flag_val("--max-model-len"), "16384")
        self.assertEqual(_flag_val("--max-num-seqs"), "256")
        self.assertEqual(_flag_val("--max-num-batched-tokens"), "8192")
        self.assertEqual(_flag_val("--attention-backend"), "CUSTOM")
        self.assertIn("--enable-chunked-prefill", args)
        self.assertNotIn("--model-id", args)
        self.assertNotIn("--engine", args)

        env_map = {e["name"]: e["value"] for e in container["env"]}
        self.assertEqual(env_map["VLLM_XLA_CACHE_PATH"], "/tmp/xla_cache")
        self.assertEqual(env_map["TPU_SKIP_MDS_QUERY"], "1")
        self.assertEqual(env_map["VLLM_TORCHTPU_ACCELERATOR"], "v6e")

    def test_translate_with_vllm_patch_registry_and_explicit_flag(self):
        """Verifies automatic registry.json vllm_patches lookup and ConfigMap + startup wrapper emission."""
        dynamo_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpt-oss-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: vllm/vllm-openai:latest
        args: ["--model", "openai/gpt-oss-20b", "--tensor-parallel-size", "4"]
        resources:
          limits:
            nvidia.com/gpu: 4
"""
        manifest_path = self.work_dir / "gpt_oss_dynamo.yaml"
        manifest_path.write_text(dynamo_yaml, encoding="utf-8")

        out_dir = self.work_dir / "gpt_oss_llmd_out"
        result = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="gpt-oss-ns",
            target_tpu_type="v6e",
            target_tpu_topology="2x2x1",
            dry_run=bool(shutil.which("kubectl")),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model_id"], "openai/gpt-oss-20b")
        self.assertEqual(result["vllm_patch"], "gpt_oss_mxfp4.py")
        self.assertEqual(result["mapping_report"]["vllm_patch"],
                         "gpt_oss_mxfp4.py")

        bundle_dir = pathlib.Path(result["bundle_dir"])
        kustomize_dir = pathlib.Path(result["kustomize_dir"])
        self.assertTrue((bundle_dir / "04-vllm-patches.yaml").is_file())
        self.assertTrue((kustomize_dir / "04-vllm-patches.yaml").is_file())

        cm_doc = yaml.safe_load(
            (bundle_dir / "04-vllm-patches.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cm_doc["kind"], "ConfigMap")
        self.assertEqual(cm_doc["metadata"]["name"], "vllm-tpu-patches")
        self.assertEqual(cm_doc["metadata"]["namespace"], "gpt-oss-ns")
        self.assertIn("apply_vllm_patch.sh", cm_doc["data"])
        self.assertIn("gpt_oss_mxfp4.py", cm_doc["data"])

        kustomization_doc = yaml.safe_load(
            (kustomize_dir / "kustomization.yaml").read_text(encoding="utf-8"))
        self.assertIn("04-vllm-patches.yaml", kustomization_doc["resources"])

        lws_doc = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lws_doc["kind"], "LeaderWorkerSet")
        pod_spec = lws_doc["spec"]["leaderWorkerTemplate"]["workerTemplate"][
            "spec"]
        volumes = {v["name"]: v for v in pod_spec.get("volumes", [])}
        self.assertIn("vllm-patches", volumes)
        self.assertEqual(volumes["vllm-patches"]["configMap"]["name"],
                         "vllm-tpu-patches")
        self.assertEqual(volumes["vllm-patches"]["configMap"]["defaultMode"],
                         0o755)

        container = pod_spec["containers"][0]
        mounts = {m["name"]: m for m in container.get("volumeMounts", [])}
        self.assertIn("vllm-patches", mounts)
        self.assertEqual(mounts["vllm-patches"]["mountPath"], "/app/patches")
        self.assertTrue(mounts["vllm-patches"]["readOnly"])

        self.assertEqual(container["command"], ["/bin/bash", "-lc"])
        self.assertEqual(len(container["args"]), 1)
        shell_cmd = container["args"][0]
        self.assertIn(
            "bash /app/patches/apply_vllm_patch.sh /app/patches/gpt_oss_mxfp4.py",
            shell_cmd,
        )
        self.assertIn("exec vllm serve --model openai/gpt-oss-20b", shell_cmd)

        if shutil.which("kubectl"):
            self.assertTrue(result["dry_run_verified"])

        rt_report = inspect_deployment.inspect_deployment(manifests=bundle_dir)
        self.assertEqual(rt_report["topology"],
                         "multi-replica-llmd-homogeneous")
        self.assertEqual(rt_report["model_id"], "openai/gpt-oss-20b")
        self.assertTrue(rt_report["accelerator"]["is_tpu"])

        custom_patch = self.work_dir / "custom_hotfix.patch"
        custom_patch.write_text(
            "--- a/vllm/foo.py\n+++ b/vllm/foo.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
            encoding="utf-8",
        )
        custom_out = self.work_dir / "custom_patch_out"
        custom_res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=custom_out,
            target_namespace="gpt-oss-ns",
            target_tpu_type="v6e",
            target_tpu_topology="2x2x1",
            vllm_patch=custom_patch,
        )
        self.assertEqual(custom_res["vllm_patch"], "custom_hotfix.patch")
        custom_cm = yaml.safe_load(
            (pathlib.Path(custom_res["bundle_dir"]) /
             "04-vllm-patches.yaml").read_text(encoding="utf-8"))
        self.assertIn("custom_hotfix.patch", custom_cm["data"])


class TestDeploymentInspector(unittest.TestCase):
    """Verifies deployment inspection and topology classification."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inspect_single_host_deployment(self):
        doc = {
            "kind": "Deployment",
            "metadata": {
                "name": "single-vllm"
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "nodeSelector": {
                            "cloud.google.com/gke-tpu-accelerator": "v6e"
                        },
                        "containers": [{
                            "name":
                            "vllm",
                            "image":
                            "vllm/vllm-torchtpu",
                            "args": ["--model", "meta-llama/Llama-3.1-8B"],
                            "resources": {
                                "limits": {
                                    "google.com/tpu": "4"
                                }
                            },
                        }],
                    }
                },
            },
        }
        report = inspect_deployment.inspect_resources([doc])
        self.assertEqual(report["topology"], "single-host")
        self.assertEqual(report["model_id"], "meta-llama/Llama-3.1-8B")
        self.assertEqual(report["accelerator"]["type"], "v6e")

    def test_inspect_dynamo_deployment(self):
        docs = [
            {
                "kind": "Service",
                "metadata": {
                    "name": "dynamo-frontend"
                },
            },
            {
                "kind": "Deployment",
                "metadata": {
                    "name": "dynamo-worker"
                },
                "spec": {
                    "replicas": 4,
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "worker",
                                "image": "vllm/vllm",
                                "resources": {
                                    "limits": {
                                        "nvidia.com/gpu": "4"
                                    }
                                },
                            }]
                        }
                    },
                },
            },
        ]
        report = inspect_deployment.inspect_resources(docs)
        self.assertEqual(report["topology"], "dynamo-gpu")
        self.assertEqual(report["router"], "dynamo-frontend")
        self.assertTrue(report["accelerator"]["is_gpu"])

    def test_inspect_llmd_homogeneous_deployment(self):
        docs = [
            {
                "kind": "InferencePool",
                "metadata": {
                    "name": "llm-d-pool"
                },
                "spec": {
                    "selector": {
                        "app": "vllm-worker"
                    }
                },
            },
            {
                "kind": "Deployment",
                "metadata": {
                    "name": "vllm-workers"
                },
                "spec": {
                    "replicas": 4,
                    "template": {
                        "spec": {
                            "nodeSelector": {
                                "cloud.google.com/gke-tpu-accelerator": "v6e"
                            },
                            "containers": [{
                                "name":
                                "worker",
                                "image":
                                "vllm-torchtpu",
                                "args": [
                                    "--model",
                                    "Qwen/Qwen3.5-27B",
                                ],
                                "resources": {
                                    "limits": {
                                        "google.com/tpu": "4"
                                    }
                                },
                            }],
                        }
                    },
                },
            },
        ]
        report = inspect_deployment.inspect_resources(docs)
        self.assertEqual(report["topology"], "multi-replica-llmd-homogeneous")
        self.assertEqual(report["router"], "llm-d-epp")
        self.assertTrue(report["accelerator"]["is_tpu"])

    def test_runtime_cli_deployment_inspect(self):
        doc = {
            "kind": "Deployment",
            "metadata": {
                "name": "single-vllm"
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "nodeSelector": {
                            "cloud.google.com/gke-tpu-accelerator": "v6e"
                        },
                        "containers": [{
                            "name":
                            "vllm",
                            "image":
                            "vllm/vllm-torchtpu",
                            "args": ["--model", "meta-llama/Llama-3.1-8B"],
                            "resources": {
                                "limits": {
                                    "google.com/tpu": "4"
                                }
                            },
                        }],
                    }
                },
            },
        }
        manifest_file = self.work_dir / "single.yaml"
        manifest_file.write_text(yaml.dump(doc), encoding="utf-8")
        store_dir = self.work_dir / "inspect_store"

        runtime_script = _REPO_ROOT / "framework" / "runtime.py"
        cmd = [
            sys.executable,
            str(runtime_script),
            "deployment",
            "inspect",
            "--manifests",
            str(manifest_file),
            "--namespace",
            "vllm-serving",
            "--context",
            "gke-tpu-ctx",
            "--kubeconfig",
            "/tmp/kubeconfig",
            "--store-dir",
            str(store_dir),
            "--capture-logs",
            "--json",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        self.assertEqual(data["topology"], "single-host")
        self.assertEqual(data["model_id"], "meta-llama/Llama-3.1-8B")
        self.assertEqual(data["store_dir"], str(store_dir))
        self.assertIn("pod_logs_dir", data)

        cluster_env_docs = [
            {
                "kind": "Node",
                "metadata": {
                    "name": "gke-tpu-node-1",
                    "labels": {
                        "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                        "cloud.google.com/gke-tpu-topology": "2x2x1",
                        "cloud.google.com/gke-nodepool": "tpu7x-pool",
                    },
                },
                "status": {
                    "capacity": {
                        "google.com/tpu": "4"
                    }
                },
            },
            {
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": "leaderworkersets.leaderworkerset.x-k8s.io"
                },
            },
            {
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": "inferencepools.inference.networking.k8s.io"
                },
            },
            {
                "kind": "CustomResourceDefinition",
                "metadata": {
                    "name": "gateways.gateway.networking.k8s.io"
                },
            },
        ]
        env_report = inspect_deployment.inspect_resources(cluster_env_docs)
        self.assertTrue(env_report["accelerator"]["is_tpu"])
        self.assertEqual(env_report["accelerator"]["type"], "tpu7x")
        self.assertEqual(env_report["accelerator"]["topology"], "2x2x1")
        self.assertEqual(env_report["accelerator"]["node_pool"], "tpu7x-pool")
        self.assertEqual(
            env_report["installed_crds"],
            ["Gateway", "InferencePool", "LeaderWorkerSet"],
        )

    def test_translate_tpu7x_lws_and_standalone_epp(self):
        dgd_doc = {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {
                "name": "qwen3-multi-vm-dgd"
            },
            "spec": {
                "services": {
                    "frontend": {
                        "image":
                        "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.0",
                    }
                },
                "components": {
                    "worker": {
                        "replicas": 2,
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name":
                                    "worker",
                                    "image":
                                    "vllm/vllm",
                                    "args": ["--model", "Qwen/Qwen3-0.6B"],
                                    "resources": {
                                        "limits": {
                                            "nvidia.com/gpu": "4"
                                        }
                                    },
                                }]
                            }
                        },
                    }
                },
            },
        }
        manifest_path = self.work_dir / "qwen3_multi_vm_dgd.yaml"
        manifest_path.write_text(yaml.dump(dgd_doc), encoding="utf-8")
        dep_json = self.work_dir / "deployments.json"
        dep_json.write_text(
            json.dumps({
                "deployments": [{
                    "model_id": "Qwen/Qwen3-0.6B",
                    "image":
                    "us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torchtpu-vllm-dev:nightly-20260804",
                    "pvc_name": "model-cache-pvc",
                    "hf_secret_name": "hf-token-secret",
                    "max-model-len": 4096,
                    "max-num-seqs": 64,
                }]
            }),
            encoding="utf-8",
        )
        out_dir = self.work_dir / "out_tpu7x_standalone"
        res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_path,
            output_dir=out_dir,
            target_namespace="vllm-serving",
            target_tpu_type="tpu7x",
            target_tpu_topology="2x2x1",
            deployments_json=dep_json,
            standalone_epp=True,
            epp_service_account="custom-epp-sa",
        )
        self.assertEqual(res["status"], "ok")
        bundle_dir = pathlib.Path(res["bundle_dir"])
        lws_doc = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lws_doc["kind"], "LeaderWorkerSet")
        self.assertEqual(lws_doc["spec"]["replicas"], 2)
        self.assertEqual(lws_doc["spec"]["leaderWorkerTemplate"]["size"], 1)
        w_tmpl = lws_doc["spec"]["leaderWorkerTemplate"]["workerTemplate"]
        self.assertEqual(
            w_tmpl["spec"]["containers"][0]["resources"]["limits"]
            ["google.com/tpu"],
            "4",
        )
        epp_raw = (bundle_dir /
                   "01-endpointpickerconfig.yaml").read_text(encoding="utf-8")
        self.assertIn("endpoint-picker-config.yaml: |", epp_raw)
        self.assertIn("default-plugins.yaml: |", epp_raw)
        self.assertIn("envoy.yaml: |", epp_raw)
        self.assertNotIn('endpoint-picker-config.yaml: "', epp_raw)
        self.assertNotIn('default-plugins.yaml: "', epp_raw)
        self.assertNotIn('envoy.yaml: "', epp_raw)
        epp_docs = list(yaml.safe_load_all(epp_raw))
        self.assertIn("default-plugins.yaml", epp_docs[0]["data"])
        self.assertIn("envoy.yaml", epp_docs[0]["data"])
        self.assertEqual(
            len(epp_docs[1]["spec"]["template"]["spec"]["containers"]), 2)
        kust_doc = yaml.safe_load(
            (pathlib.Path(res["kustomize_dir"]) /
             "kustomization.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            kust_doc["resources"],
            ["01-endpointpickerconfig.yaml", "base-modelserver.yaml"],
        )

    def test_inspect_deployment_cluster_state_and_capture_logs(self):
        """Verifies cluster_state.txt and pod_logs/<pod_name>.log capture in inspect_deployment."""
        store_dir = self.work_dir / "source_store"
        resources = [{
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "dynamo-worker-0",
                "namespace": "dynamo-gpu-ns",
                "labels": {
                    "ai.dynamo/component": "worker"
                },
            },
            "spec": {
                "containers": [{
                    "name": "worker",
                    "image": "vllm/vllm",
                    "args": ["--model", "Qwen/Qwen3-0.6B"],
                    "resources": {
                        "limits": {
                            "nvidia.com/gpu": "4"
                        }
                    },
                }]
            },
        }]

        def fake_subprocess_run(cmd, *args, **kwargs):
            if "get" in cmd and "-o" in cmd and "wide" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=
                    ("NAME                  READY   STATUS    RESTARTS   AGE\n"
                     "pod/dynamo-worker-0   1/1     Running   0          5m\n"
                     ),
                    stderr="",
                )
            if "logs" in cmd and "dynamo-worker-0" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout="INFO: Application startup complete.\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps({"items": resources}),
                stderr="",
            )

        with unittest.mock.patch.object(inspect_deployment.subprocess,
                                        "run",
                                        side_effect=fake_subprocess_run):
            report = inspect_deployment.inspect_deployment(
                namespace="dynamo-gpu-ns",
                store_dir=store_dir,
                capture_logs=True,
                kubeconfig="/tmp/kubeconfig",
                context="src-ctx",
            )

        self.assertEqual(report["model_id"], "Qwen/Qwen3-0.6B")
        state_file = store_dir / "cluster_state.txt"
        self.assertTrue(state_file.is_file())
        self.assertIn("pod/dynamo-worker-0",
                      state_file.read_text(encoding="utf-8"))
        self.assertEqual(report["cluster_state_path"], str(state_file))

        pod_log_file = store_dir / "pod_logs" / "dynamo-worker-0.log"
        self.assertTrue(pod_log_file.is_file())
        self.assertIn(
            "Application startup complete",
            pod_log_file.read_text(encoding="utf-8"),
        )
        self.assertEqual(report["captured_pod_logs"], ["dynamo-worker-0.log"])

    def test_translator_dry_run_rendered_and_no_apply_flags(self):
        """Verifies kustomize_rendered.yaml emission on --dry-run and confirms dynamo_translator does not expose --apply / rollout flags."""
        dgd_doc = {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "metadata": {
                "name": "qwen3-dgd"
            },
            "spec": {
                "services": {
                    "frontend": {
                        "image":
                        "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.5.0"
                    }
                },
                "components": {
                    "worker": {
                        "replicas": 2,
                        "template": {
                            "spec": {
                                "containers": [{
                                    "name":
                                    "worker",
                                    "image":
                                    "vllm/vllm",
                                    "args": ["--model", "Qwen/Qwen3-0.6B"],
                                    "resources": {
                                        "limits": {
                                            "nvidia.com/gpu": "4"
                                        }
                                    },
                                }]
                            }
                        },
                    }
                },
            },
        }
        manifest_path = self.work_dir / "dgd.yaml"
        manifest_path.write_text(yaml.dump(dgd_doc), encoding="utf-8")
        out_dir = self.work_dir / "translated_out"

        if shutil.which("kubectl"):
            res = dynamo_translator.translate_dynamo_to_llmd(
                manifest_path=manifest_path,
                output_dir=out_dir,
                target_namespace="llmd-serving",
                target_tpu_type="tpu7x",
                target_tpu_topology="2x2x1",
                dry_run=True,
            )
            self.assertTrue(res["dry_run_verified"])
            rendered_file = out_dir / "kustomize_rendered.yaml"
            self.assertTrue(rendered_file.is_file())
            self.assertEqual(res["kustomize_rendered"], str(rendered_file))
            rendered_docs = [
                d for d in yaml.safe_load_all(
                    rendered_file.read_text(encoding="utf-8"))
                if isinstance(d, dict)
            ]
            self.assertTrue(
                any(d.get("kind") == "LeaderWorkerSet" for d in rendered_docs))
        else:
            dynamo_translator.translate_dynamo_to_llmd(
                manifest_path=manifest_path,
                output_dir=out_dir,
                target_namespace="llmd-serving",
                target_tpu_type="tpu7x",
                target_tpu_topology="2x2x1",
                dry_run=False,
            )

        for removed_fn in (
                "apply_kustomize_bundle",
                "wait_for_deployment_ready",
                "run_smoke_check",
                "capture_deployment_evidence",
        ):
            self.assertFalse(hasattr(dynamo_translator, removed_fn))

    def test_dynamo_unwiring_request_schema_allows_source_namespace(self):
        """Verifies dynamo-unwiring-request.json validates both dynamo_manifests and source_namespace inputs."""
        from framework import utils

        schema_path = _PACKAGE_ROOT / "schemas" / "dynamo-unwiring-request.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            utils.JsonSchemaValidator.validate_schema_definition(schema), [])

        live_ns_req = {
            "source_namespace": "vllm-serving",
            "run_id": "20261016T160000Z-000000000000",
            "target_namespace": "vllm-serving",
            "target_tpu_type": "tpu7x",
            "target_tpu_topology": "2x2x1",
            "standalone_epp": True,
            "dry_run": True,
        }
        errors = utils.JsonSchemaValidator.validate_instance(
            live_ns_req, schema)
        self.assertEqual(errors, [])

    def test_single_slice_deployments_json_sibling_autodiscovery_and_carryover(
            self):
        """Verifies auto-discovery of sibling <work>/inference/deployments.json and carry-over of image, TP, max_model_len, and vllm_patch."""
        onboarding_work = self.work_dir / "runs" / "run-123" / "artifacts" / "tpu-model-onboarding"
        inference_work = onboarding_work / "inference"
        dynamo_work = onboarding_work / "dynamo_unwiring" / "openai__gpt-oss-20b"
        inference_work.mkdir(parents=True, exist_ok=True)
        dynamo_work.mkdir(parents=True, exist_ok=True)

        dep_json_path = inference_work / "deployments.json"
        dep_payload = {
            "deployments": [{
                "model_id": "openai/gpt-oss-20b",
                "engine": "vllm",
                "port": 8000,
                "image":
                "us-docker.pkg.dev/cloud-tpu-images/vllm/vllm-torchtpu:verified-sha",
                "vllm_patch": "gpt_oss_mxfp4.py",
                "tensor_parallel_size": 4,
                "max_model_len": 4096,
                "parallelism": {
                    "source": "registry",
                    "reason": "validated stack",
                },
            }]
        }
        dep_json_path.write_text(json.dumps(dep_payload), encoding="utf-8")

        # Also verify inference.py DeploymentBase accepts image and vllm_patch
        custom_dep_path = self.work_dir / "custom_deployments.json"
        custom_dep_path.write_text(
            json.dumps({
                "deployments": [{
                    "model_id": "openai/gpt-oss-20b",
                    "engine": "custom_fastapi",
                    "port": 8000,
                    "image":
                    "us-docker.pkg.dev/cloud-tpu-images/vllm/vllm-torchtpu:verified-sha",
                    "vllm_patch": "gpt_oss_mxfp4.py",
                    "max_model_len": 4096,
                }]
            }),
            encoding="utf-8",
        )
        inf_script = (_PACKAGE_ROOT / "scripts" / "tpu-model-onboarding" /
                      "inference.py")
        inf_text = inf_script.read_text(encoding="utf-8")
        self.assertIn("image: Optional[str] = None", inf_text)
        self.assertIn("vllm_patch: Optional[str] = None", inf_text)

        dynamo_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-worker
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: vllm/vllm-openai:latest
        args: ["--model", "openai/gpt-oss-20b"]
        resources:
          limits:
            nvidia.com/gpu: 4
"""
        manifest_file = self.work_dir / "dynamo_gpt_oss.yaml"
        manifest_file.write_text(dynamo_yaml, encoding="utf-8")

        res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=manifest_file,
            output_dir=dynamo_work,
            target_namespace="vllm-serving",
            target_tpu_type="tpu7x",
            target_tpu_topology="2x2x1",
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["deployments_json"], str(dep_json_path))
        self.assertEqual(res["vllm_patch"], "gpt_oss_mxfp4.py")
        self.assertEqual(
            res["mapping_report"]["resolved_tpu_image"],
            "us-docker.pkg.dev/cloud-tpu-images/vllm/vllm-torchtpu:verified-sha",
        )
        self.assertEqual(
            res["mapping_report"]["resolved_tpu_image_source"],
            "deployments_json",
        )

        bundle_dir = pathlib.Path(res["bundle_dir"])
        self.assertTrue((bundle_dir / "04-vllm-patches.yaml").is_file())
        lws_doc = yaml.safe_load(
            (bundle_dir /
             "03-serving-deployment.yaml").read_text(encoding="utf-8"))
        container = (lws_doc["spec"]["leaderWorkerTemplate"]["workerTemplate"]
                     ["spec"]["containers"][0])
        self.assertEqual(
            container["image"],
            "us-docker.pkg.dev/cloud-tpu-images/vllm/vllm-torchtpu:verified-sha",
        )
        joined_args = " ".join(str(a) for a in container["args"])
        self.assertIn("--tensor-parallel-size 4", joined_args)
        self.assertIn("--max-model-len 4096", joined_args)
        self.assertNotIn("--image", joined_args)
        self.assertNotIn("--vllm-patch", joined_args)
        self.assertNotIn("--parallelism", joined_args)

    def test_resolve_tpu_base_image_priority_hierarchy(self):
        """Verifies resolve_tpu_base_image priority: deployments.json > TPU_VLLM_IMAGE/TASK_DOCKER > .tpu-agent-suite/environment.json > default."""
        ws_dir = self.work_dir / "workspace"
        suite_dir = ws_dir / ".tpu-agent-suite"
        suite_dir.mkdir(parents=True, exist_ok=True)
        fake_home = self.work_dir / "fake_home"
        fake_home.mkdir(parents=True, exist_ok=True)

        clean_env = {
            k: v
            for k, v in dynamo_translator.os.environ.items() if k not in (
                "TPU_VLLM_IMAGE",
                "TASK_DOCKER",
                "TPU_AGENT_SUITE_HOME",
            )
        }

        with unittest.mock.patch.dict(dynamo_translator.os.environ,
                                      clean_env,
                                      clear=True), unittest.mock.patch.object(
                                          dynamo_translator.Path,
                                          "home",
                                          return_value=fake_home):
            # 1. Default fallback when no env or environment.json exists
            img_default, src_default = (
                dynamo_translator._resolve_tpu_base_image_with_source(
                    deployments_config={},
                    workspace=ws_dir,
                ))
            self.assertEqual(img_default,
                             dynamo_translator.DEFAULT_TPU_VLLM_IMAGE)
            self.assertEqual(src_default, "default_fallback")

            # 2. .tpu-agent-suite/environment.json gke_job profile params.image
            env_json_path = suite_dir / "environment.json"
            env_json_path.write_text(
                json.dumps({
                    "revision": 1,
                    "default": "gke-tpu7x",
                    "profiles": {
                        "gke-tpu7x": {
                            "backend": "gke_job",
                            "params": {
                                "image":
                                "us-docker.pkg.dev/example/vllm/env-json-image:v1",
                                "namespace": "vllm-serving",
                            },
                        }
                    },
                }),
                encoding="utf-8",
            )
            img_env_json, src_env_json = (
                dynamo_translator._resolve_tpu_base_image_with_source(
                    deployments_config={},
                    workspace=ws_dir,
                ))
            self.assertEqual(
                img_env_json,
                "us-docker.pkg.dev/example/vllm/env-json-image:v1",
            )
            self.assertEqual(src_env_json, "environment_json")

            # 3. TASK_DOCKER and TPU_VLLM_IMAGE environment variables outrank environment.json
            dynamo_translator.os.environ[
                "TASK_DOCKER"] = "us-docker.pkg.dev/example/vllm/task-docker:v2"
            self.assertEqual(
                dynamo_translator.resolve_tpu_base_image(
                    deployments_config={},
                    workspace=ws_dir,
                ),
                "us-docker.pkg.dev/example/vllm/task-docker:v2",
            )
            dynamo_translator.os.environ[
                "TPU_VLLM_IMAGE"] = "us-docker.pkg.dev/example/vllm/tpu-vllm-env:v3"
            img_env_var, src_env_var = (
                dynamo_translator._resolve_tpu_base_image_with_source(
                    deployments_config={},
                    workspace=ws_dir,
                ))
            self.assertEqual(
                img_env_var,
                "us-docker.pkg.dev/example/vllm/tpu-vllm-env:v3",
            )
            self.assertEqual(src_env_var, "env_var")

            # 4. deployments.json "image" outranks environment variables and environment.json
            img_dep, src_dep = (
                dynamo_translator._resolve_tpu_base_image_with_source(
                    deployments_config={
                        "image":
                        "us-docker.pkg.dev/example/vllm/deployments-json:v4"
                    },
                    workspace=ws_dir,
                ))
            self.assertEqual(
                img_dep,
                "us-docker.pkg.dev/example/vllm/deployments-json:v4",
            )
            self.assertEqual(src_dep, "deployments_json")

            # 5. Explicit --image override outranks everything
            img_cli, src_cli = (
                dynamo_translator._resolve_tpu_base_image_with_source(
                    deployments_config={
                        "image":
                        "us-docker.pkg.dev/example/vllm/deployments-json:v4"
                    },
                    image_override=
                    "us-docker.pkg.dev/example/vllm/cli-override:v5",
                    workspace=ws_dir,
                ))
            self.assertEqual(
                img_cli,
                "us-docker.pkg.dev/example/vllm/cli-override:v5",
            )
            self.assertEqual(src_cli, "cli_override")


if __name__ == "__main__":
    unittest.main()
