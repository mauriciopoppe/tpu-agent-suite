#!/usr/bin/env python3
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
"""inspect_deployment.py: Inspects Kubernetes deployments or manifest files to detect serving topology.

Classifies workloads into:
- single-host
- multi-host-single-replica
- multi-replica-llmd-homogeneous
- dynamo-gpu
- dynamo-tpu
- target-cluster-preflight
- unknown
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    """Loads all YAML documents from a file or directory."""
    import yaml

    documents: list[dict[str, Any]] = []
    files: list[Path] = []
    is_kustomize_dir = False
    if path.is_file():
        files = [path]
    elif path.is_dir():
        is_kustomize_dir = ((path / "kustomization.yaml").is_file()
                            or (path / "kustomization.yml").is_file())
        for ext in ("*.yaml", "*.yml", "*.json"):
            files.extend(path.glob(ext))
    else:
        return documents

    for f in sorted(files):
        if is_kustomize_dir and (f.name
                                 in ("kustomization.yaml", "kustomization.yml")
                                 or f.name.startswith("patch-")):
            continue
        try:
            content = f.read_text(encoding="utf-8")
            for doc in yaml.safe_load_all(content):
                if isinstance(doc, dict) and "kind" in doc:
                    if str(doc.get("kind", "")).lower() == "kustomization":
                        continue
                    doc["_source_file"] = str(f)
                    documents.append(doc)
        except Exception as exc:
            print(f"Warning: failed to read {f}: {exc}", file=sys.stderr)

    return documents


def sanitize_k8s_resource(resource: dict[str, Any]) -> dict[str, Any]:
    """Strips ephemeral runtime cluster fields before persisting a manifest to disk."""
    cleaned = {
        k: v
        for k, v in resource.items() if k not in ("status", "_source_file")
    }
    if isinstance(cleaned.get("metadata"), dict):
        ephemeral_meta = {
            "managedFields",
            "resourceVersion",
            "uid",
            "creationTimestamp",
            "generation",
            "selfLink",
        }
        cleaned["metadata"] = {
            k: v
            for k, v in cleaned["metadata"].items() if k not in ephemeral_meta
        }
    return cleaned


def _format_resources_table(resources: list[dict[str, Any]],
                            namespace: str) -> str:
    """Formats a human-readable summary table of resources as a fallback when kubectl is unavailable."""
    lines = ["KIND\tNAME\tNAMESPACE\tSTATUS"]
    for res in resources:
        if not isinstance(res, dict) or "kind" not in res:
            continue
        kind = str(res.get("kind", "Resource"))
        meta = res.get("metadata", {}) if isinstance(res.get("metadata"),
                                                     dict) else {}
        name = str(meta.get("name", "unknown"))
        ns = str(meta.get("namespace") or namespace)
        status_obj = res.get("status", {}) if isinstance(
            res.get("status"), dict) else {}
        phase = str(
            status_obj.get("phase") or status_obj.get("readyReplicas")
            or "Active")
        lines.append(f"{kind}\t{name}\t{ns}\t{phase}")
    return "\n".join(lines) + "\n"


def capture_cluster_state_text(
    namespace: str,
    store_dir: Path,
    kubeconfig: str | None = None,
    context: str | None = None,
    resources: list[dict[str, Any]] | None = None,
    offline: bool = False,
) -> Path:
    """Captures human-readable cluster state into <store_dir>/cluster_state.txt."""
    store_dir.mkdir(parents=True, exist_ok=True)
    state_path = store_dir / "cluster_state.txt"

    if not offline:
        for resource_list in ("deploy,lws,pods,svc", "deploy,pods,svc"):
            cmd = [
                "kubectl", "get", resource_list, "-n", namespace, "-o", "wide"
            ]
            if kubeconfig:
                cmd.extend(["--kubeconfig", kubeconfig])
            if context:
                cmd.extend(["--context", context])
            try:
                proc = subprocess.run(cmd,
                                      capture_output=True,
                                      text=True,
                                      check=True,
                                      timeout=15)
                output = (proc.stdout or "").strip()
                if output:
                    state_path.write_text(output + "\n", encoding="utf-8")
                    return state_path
            except (subprocess.CalledProcessError, FileNotFoundError,
                    subprocess.TimeoutExpired):
                continue

    fallback_text = _format_resources_table(resources or [], namespace)
    state_path.write_text(fallback_text, encoding="utf-8")
    return state_path


def capture_pod_logs(
    namespace: str,
    store_dir: Path,
    kubeconfig: str | None = None,
    context: str | None = None,
    resources: list[dict[str, Any]] | None = None,
    tail: int = 500,
) -> list[str]:
    """Saves pod logs (<store_dir>/pod_logs/<pod_name>.log) for all pods in the namespace."""
    pod_logs_dir = store_dir / "pod_logs"
    pod_logs_dir.mkdir(parents=True, exist_ok=True)

    pod_names: list[str] = []
    for res in resources or []:
        if isinstance(res, dict) and str(res.get("kind", "")).lower() == "pod":
            name = res.get("metadata", {}).get("name")
            if name and str(name) not in pod_names:
                pod_names.append(str(name))

    if not pod_names:
        cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
        if kubeconfig:
            cmd.extend(["--kubeconfig", kubeconfig])
        if context:
            cmd.extend(["--context", context])
        try:
            proc = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  check=True,
                                  timeout=15)
            data = json.loads(proc.stdout)
            for item in data.get("items", []):
                if isinstance(item, dict):
                    name = item.get("metadata", {}).get("name")
                    if name and str(name) not in pod_names:
                        pod_names.append(str(name))
        except Exception:
            pass

    saved_logs: list[str] = []
    for pod_name in pod_names:
        log_cmd = [
            "kubectl", "logs", pod_name, "-n", namespace, f"--tail={tail}"
        ]
        if kubeconfig:
            log_cmd.extend(["--kubeconfig", kubeconfig])
        if context:
            log_cmd.extend(["--context", context])
        log_file = pod_logs_dir / f"{pod_name}.log"
        try:
            proc = subprocess.run(log_cmd,
                                  capture_output=True,
                                  text=True,
                                  check=True,
                                  timeout=30)
            log_file.write_text(proc.stdout or "", encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            log_file.write_text((exc.stdout or "") + (exc.stderr or ""),
                                encoding="utf-8")
        except Exception as exc:
            log_file.write_text(
                f"Failed to capture logs for {pod_name}: {exc}\n",
                encoding="utf-8")
        saved_logs.append(log_file.name)

    return saved_logs


def store_captured_resources(
    resources: list[dict[str, Any]],
    store_dir: Path,
    report: dict[str, Any],
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    capture_logs: bool = False,
    offline: bool = False,
    target_cluster_only: bool = False,
) -> dict[str, Any]:
    """Persists captured cluster resources, cluster_state.txt, optional pod logs, and inspection report into store_dir."""
    import yaml

    store_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = store_dir / "manifests"
    saved_files: list[str] = []

    if target_cluster_only:
        state_path = store_dir / "cluster_state.txt"
        accel = report.get("accelerator", {})
        crds = ", ".join(report.get("installed_crds", [])) or "none"
        state_lines = [
            "ACCELERATOR\tTOPOLOGY\tNODE_POOL\tTOTAL_CHIPS\tCHIPS_PER_NODE\tINSTALLED_CRDS",
            (f"{accel.get('type', 'unknown')}\t"
             f"{accel.get('topology', '')}\t"
             f"{accel.get('node_pool', '')}\t"
             f"{accel.get('total_chips', 0)}\t"
             f"{accel.get('chips_per_node', 0)}\t"
             f"{crds}"),
        ]
        state_path.write_text("\n".join(state_lines) + "\n", encoding="utf-8")
        report["store_dir"] = str(store_dir)
        report["captured_manifest_files"] = saved_files
        report["cluster_state_path"] = str(state_path)
    else:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for idx, res in enumerate(resources):
            if not isinstance(res, dict) or "kind" not in res:
                continue
            cleaned = sanitize_k8s_resource(res)
            kind_slug = re.sub(r"[^a-z0-9]+", "-",
                               str(cleaned.get("kind",
                                               "resource")).lower()).strip("-")
            raw_name = str(
                cleaned.get("metadata", {}).get("name", f"item-{idx}"))
            name_slug = re.sub(r"[^a-z0-9.-]+", "-",
                               raw_name.lower()).strip("-")
            filename = f"{idx:02d}-{kind_slug}-{name_slug}.yaml"
            (manifests_dir / filename).write_text(yaml.dump(cleaned,
                                                            sort_keys=False),
                                                  encoding="utf-8")
            saved_files.append(filename)

        state_path = capture_cluster_state_text(
            namespace=namespace,
            store_dir=store_dir,
            kubeconfig=kubeconfig,
            context=context,
            resources=resources,
            offline=offline,
        )

        report["store_dir"] = str(store_dir)
        report["manifests_dir"] = str(manifests_dir)
        report["captured_manifest_files"] = saved_files
        report["cluster_state_path"] = str(state_path)

        if capture_logs:
            saved_logs = capture_pod_logs(
                namespace=namespace,
                store_dir=store_dir,
                kubeconfig=kubeconfig,
                context=context,
                resources=resources,
            )
            report["pod_logs_dir"] = str(store_dir / "pod_logs")
            report["captured_pod_logs"] = saved_logs

    report_path = store_dir / "inspection_report.json"
    report["inspection_report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def query_target_cluster_resources(
    kubeconfig: str | None = None,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """Queries cluster-level capabilities only (installed CRDs and TPU node pools)."""
    items: list[dict[str, Any]] = []
    for cluster_group in ("nodes", "crd"):
        cmd = ["kubectl", "get", cluster_group, "-o", "json"]
        if kubeconfig:
            cmd.extend(["--kubeconfig", kubeconfig])
        if context:
            cmd.extend(["--context", context])
        try:
            proc = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  check=True,
                                  timeout=15)
            data = json.loads(proc.stdout)
            if "items" in data and isinstance(data["items"], list):
                items.extend(data["items"])
        except Exception:
            continue
    return items


def query_cluster_resources(
    namespace: str,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """Queries Kubernetes resources from a live cluster via kubectl."""
    cmd = [
        "kubectl",
        "get",
        "pods,deployments,statefulsets,services,inferencepools,gateways,httproutes,configmaps,leaderworkersets",
        "-n",
        namespace,
        "-o",
        "json",
    ]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    if context:
        cmd.extend(["--context", context])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
        items = data.get("items", [])
    except subprocess.CalledProcessError:
        items = []
        for kind_group in (
                "pods,deployments,statefulsets,services,configmaps",
                "leaderworkersets",
                "inferencepools",
                "gateways",
                "httproutes",
                "dynamographdeployments",
                "dynamocomponentdeployments",
        ):
            sub_cmd = [
                "kubectl", "get", kind_group, "-n", namespace, "-o", "json"
            ]
            if kubeconfig:
                sub_cmd.extend(["--kubeconfig", kubeconfig])
            if context:
                sub_cmd.extend(["--context", context])
            try:
                sub_proc = subprocess.run(sub_cmd,
                                          capture_output=True,
                                          text=True,
                                          check=True)
                sub_data = json.loads(sub_proc.stdout)
                if "items" in sub_data and isinstance(sub_data["items"], list):
                    items.extend(sub_data["items"])
                elif isinstance(sub_data, dict) and "kind" in sub_data:
                    items.append(sub_data)
            except Exception:
                continue
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Warning: kubectl query failed: {exc}", file=sys.stderr)
        return []

    workload_kinds = {
        "deployment",
        "statefulset",
        "leaderworkerset",
        "pod",
        "dynamographdeployment",
        "dynamocomponentdeployment",
    }
    if not any(
            str(r.get("kind", "")).lower() in workload_kinds
            for r in items if isinstance(r, dict)):
        for extra_group in ("nodes", "crd"):
            extra_cmd = ["kubectl", "get", extra_group, "-o", "json"]
            if kubeconfig:
                extra_cmd.extend(["--kubeconfig", kubeconfig])
            if context:
                extra_cmd.extend(["--context", context])
            try:
                extra_proc = subprocess.run(extra_cmd,
                                            capture_output=True,
                                            text=True,
                                            check=True,
                                            timeout=10)
                extra_data = json.loads(extra_proc.stdout)
                if "items" in extra_data and isinstance(
                        extra_data["items"], list):
                    items.extend(extra_data["items"])
            except Exception:
                continue

    return items


def inspect_resources(
    resources: list[dict[str, Any]],
    target_cluster_only: bool = False,
) -> dict[str, Any]:
    """Analyzes a list of Kubernetes resources to classify topology and extract config."""
    if target_cluster_only:
        resources = [
            r for r in resources
            if isinstance(r, dict) and str(r.get("kind", "")).lower() in (
                "node", "customresourcedefinition")
        ]
    deployments: list[dict[str, Any]] = []
    statefulsets: list[dict[str, Any]] = []
    services: list[dict[str, Any]] = []
    inference_pools: list[dict[str, Any]] = []
    gateways: list[dict[str, Any]] = []
    http_routes: list[dict[str, Any]] = []
    configmaps: list[dict[str, Any]] = []
    lws_list: list[dict[str, Any]] = []
    pods: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    installed_crds: set[str] = set()
    has_dynamo_frontend = False
    has_dynamo_worker = False
    session_affinity_enabled = False
    kv_routing_enabled = False
    session_headers: list[str] = []
    frontend_port: int | None = None
    detected_models: set[str] = set()
    detected_engines: set[str] = set()

    def _scan_router_text(text: str) -> None:
        nonlocal session_affinity_enabled, kv_routing_enabled
        lower = text.lower()
        if ("aiperf_http_x_dynamo_session_id_from_correlation_id=1" in lower
                or "x-dynamo-session-id" in lower or "x-session-id" in lower
                or "x-correlation-id" in lower
                or "sessionaffinityheaders" in lower
                or "prefix-cache-affinity-filter" in lower):
            session_affinity_enabled = True
            for hdr in ("X-Session-ID", "X-Correlation-ID"):
                if hdr not in session_headers:
                    session_headers.append(hdr)
        if ("dyn_router_mode=kv" in lower or "--router-mode=kv" in lower
                or "--router-mode kv" in lower or "kv-router" in lower
                or "prefix-cache-scorer" in lower):
            kv_routing_enabled = True

    for r in resources:
        kind = r.get("kind", "").lower()
        if kind == "deployment":
            deployments.append(r)
        elif kind == "statefulset":
            statefulsets.append(r)
        elif kind == "service":
            services.append(r)
        elif kind == "inferencepool":
            inference_pools.append(r)
        elif kind == "gateway":
            gateways.append(r)
        elif kind == "httproute":
            http_routes.append(r)
        elif kind == "configmap":
            configmaps.append(r)
        elif kind == "leaderworkerset":
            lws_list.append(r)
            installed_crds.add("LeaderWorkerSet")
        elif kind == "pod":
            pods.append(r)
        elif kind == "node":
            nodes.append(r)
        elif kind == "customresourcedefinition":
            crd_name = str(r.get("metadata", {}).get("name", "")).lower()
            if "leaderworkerset" in crd_name:
                installed_crds.add("LeaderWorkerSet")
            elif "inferencepool" in crd_name:
                installed_crds.add("InferencePool")
            elif "gateway" in crd_name:
                installed_crds.add("Gateway")
        elif kind in (
                "dynamographdeployment",
                "dynamocomponentdeployment",
                "dynamographdeploymentrequest",
        ):
            spec = r.get("spec", {})
            model = spec.get("model")
            if model:
                detected_models.add(model)
            backend = spec.get("backend")
            if backend:
                detected_engines.add(backend)
            raw_comps: dict[str, Any] = {}
            for field in ("services", "components"):
                val = spec.get(field, {})
                if isinstance(val, dict):
                    raw_comps.update(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "name" in item:
                            raw_comps[item["name"]] = item
            for c_name, c_spec in raw_comps.items():
                if not isinstance(c_spec, dict):
                    continue
                c_name_lower = c_name.lower()
                c_type = c_spec.get("type", "").lower()
                c_comp_type = c_spec.get("componentType", "").lower()
                comp_labels = dict(c_spec.get("labels", {}))

                if ("frontend" in c_name_lower or c_type == "frontend"
                        or c_comp_type == "frontend" or "epp" in c_name_lower
                        or c_type == "epp" or
                        comp_labels.get("ai.dynamo/component") == "frontend"):
                    has_dynamo_frontend = True
                    f_env = list(c_spec.get("env", [])) + list(
                        c_spec.get("envs", []))
                    f_args = list(c_spec.get("args", []))
                    extra_pod_spec = c_spec.get("extraPodSpec", {})
                    if isinstance(extra_pod_spec, dict):
                        main_c = extra_pod_spec.get("mainContainer", {})
                        if isinstance(main_c, dict):
                            f_env += list(main_c.get("env", []))
                            f_args += list(main_c.get("args", []))
                    pod_tmpl = c_spec.get("podTemplate") or c_spec.get(
                        "template", {})
                    f_tmpl_containers = (pod_tmpl.get("spec", {}).get(
                        "containers", [])
                                         if isinstance(pod_tmpl, dict) else [])
                    for fc in f_tmpl_containers:
                        f_env = list(f_env) + list(fc.get("env", []))
                        f_args = list(f_args) + list(fc.get("args", []))
                        for p in fc.get("ports", []):
                            if isinstance(p, dict) and p.get("containerPort"):
                                frontend_port = int(p["containerPort"])
                    if c_spec.get("port"):
                        frontend_port = int(c_spec["port"])
                    f_text = " ".join([str(a) for a in f_args] + [
                        f"{e.get('name', '')}={e.get('value', '')}"
                        for e in f_env if isinstance(e, dict)
                    ])
                    _scan_router_text(f_text)
                else:
                    has_dynamo_worker = True
                    pod_template = c_spec.get("podTemplate") or c_spec.get(
                        "template", {})
                    pod_tmpl_spec = (dict(pod_template.get("spec", {}))
                                     if isinstance(pod_template, dict) else {})
                    containers = pod_tmpl_spec.get("containers", [])
                    extra_pod_spec = c_spec.get("extraPodSpec", {})
                    if (not containers and isinstance(extra_pod_spec, dict)
                            and "mainContainer" in extra_pod_spec):
                        main_c = extra_pod_spec.get("mainContainer", {})
                        c_args = main_c.get("args", [])
                        c_cmd = main_c.get("command", [])
                        c_env = (list(c_spec.get("env", [])) +
                                 list(c_spec.get("envs", [])) +
                                 list(main_c.get("env", [])))
                        c_img = main_c.get("image") or c_spec.get(
                            "image", "vllm/vllm")
                        c_res = dict(c_spec.get("resources", {}))
                        if main_c.get("resources"):
                            c_res.update(main_c["resources"])
                        limits = dict(c_res.get("limits", {}))
                        requests = dict(c_res.get("requests", {}))
                        if "gpu" in limits and "nvidia.com/gpu" not in limits:
                            limits["nvidia.com/gpu"] = limits["gpu"]
                        if "gpu" in requests and "nvidia.com/gpu" not in requests:
                            requests["nvidia.com/gpu"] = requests["gpu"]
                        c_res["limits"] = limits
                        c_res["requests"] = requests
                        containers = [{
                            "name": main_c.get("name", c_name),
                            "image": c_img,
                            "args": c_args,
                            "command": c_cmd,
                            "env": c_env,
                            "resources": c_res,
                        }]
                    if not containers:
                        c_res = dict(c_spec.get("resources", {}))
                        limits = dict(c_res.get("limits", {}))
                        requests = dict(c_res.get("requests", {}))
                        if "gpu" in limits and "nvidia.com/gpu" not in limits:
                            limits["nvidia.com/gpu"] = limits["gpu"]
                        if "gpu" in requests and "nvidia.com/gpu" not in requests:
                            requests["nvidia.com/gpu"] = requests["gpu"]
                        c_res["limits"] = limits
                        c_res["requests"] = requests
                        containers = [{
                            "name":
                            c_name,
                            "image":
                            c_spec.get("image", "vllm/vllm"),
                            "args":
                            c_spec.get("args", []),
                            "command":
                            c_spec.get("command", []),
                            "env":
                            list(c_spec.get("env", [])) +
                            list(c_spec.get("envs", [])),
                            "resources":
                            c_res,
                        }]
                    pod_tmpl_spec["containers"] = containers
                    if c_spec.get("nodeSelector"
                                  ) and "nodeSelector" not in pod_tmpl_spec:
                        pod_tmpl_spec["nodeSelector"] = c_spec["nodeSelector"]
                    worker_doc = {
                        "kind": "Deployment",
                        "metadata": {
                            "name": c_name,
                            "labels": dict(comp_labels)
                        },
                        "spec": {
                            "replicas": c_spec.get("replicas", 1),
                            "template": {
                                "metadata":
                                dict(pod_template.get("metadata", {}))
                                if isinstance(pod_template, dict) else {},
                                "spec": pod_tmpl_spec,
                            },
                            "multinode": c_spec.get("multinode"),
                        },
                    }
                    deployments.append(worker_doc)

    has_llmd_router = False
    has_inference_pool = len(inference_pools) > 0
    endpoint_picker_config: dict[str, Any] = {}

    for cm in configmaps:
        name = cm.get("metadata", {}).get("name", "")
        data = cm.get("data", {})
        if "endpointpickerconfig" in name.lower() or "epp" in name.lower():
            has_llmd_router = True
            for k, v in data.items():
                if "config" in k.lower() or k.endswith((".yaml", ".json")):
                    endpoint_picker_config[k] = v
                    _scan_router_text(str(v))

    for svc in services:
        meta = svc.get("metadata", {})
        name = meta.get("name", "").lower()
        svc_labels = meta.get("labels", {})
        svc_selector = svc.get("spec", {}).get("selector", {})
        if ("dynamo-frontend" in name or "dynamo-router" in name
                or svc_labels.get("ai.dynamo/component") == "frontend"
                or svc_selector.get("ai.dynamo/component") == "frontend"):
            has_dynamo_frontend = True
            for p in svc.get("spec", {}).get("ports", []):
                if isinstance(p, dict) and p.get("port"):
                    frontend_port = int(p["port"])

    controllers = deployments + statefulsets + lws_list
    has_worker_controller = False
    for ctrl in controllers:
        ctrl_meta = ctrl.get("metadata", {})
        ctrl_name = ctrl_meta.get("name", "").lower()
        ctrl_labels = dict(ctrl_meta.get("labels", {}))
        ctrl_tmpl = ctrl.get("spec", {}).get("template", {})
        if isinstance(ctrl_tmpl, dict):
            ctrl_labels.update(ctrl_tmpl.get("metadata", {}).get("labels", {}))
        is_router_ctrl = ("dynamo-frontend" in ctrl_name
                          or "dynamo-router" in ctrl_name or
                          ctrl_labels.get("ai.dynamo/component") == "frontend"
                          or ctrl_labels.get("llm-d.ai/component") == "epp"
                          or ctrl_name == "llm-d-epp"
                          or ctrl_name.endswith("-epp"))
        if not is_router_ctrl:
            has_worker_controller = True
            break

    effective_pods = [] if has_worker_controller else pods
    workload_objects = controllers + effective_pods
    total_replicas = 0
    accelerator_type: str = "cpu"
    accelerator_topology: str = ""
    accelerator_chips: int = 0
    chips_per_node: int = 0
    gke_accelerator: str = ""
    node_pool: str = ""
    is_tpu = False
    is_gpu = False
    homogeneous_pools: list[str] = []

    for obj in workload_objects:
        metadata = obj.get("metadata", {})
        spec = obj.get("spec", {})
        name = metadata.get("name", "").lower()

        replicas = spec.get("replicas", 1)
        lws_tmpl = spec.get("leaderWorkerTemplate", {})
        multinode_cfg = spec.get("multinode", {})
        lws_size = int(
            (lws_tmpl.get("size") if isinstance(lws_tmpl, dict) else None)
            or (multinode_cfg.get("nodeCount") if isinstance(
                multinode_cfg, dict) else None) or 1)

        lws_worker_tmpl = (lws_tmpl.get("workerTemplate", {}) if isinstance(
            lws_tmpl, dict) else {})
        template_obj = spec.get("template") or lws_worker_tmpl
        pod_spec = (template_obj.get("spec", spec) if isinstance(
            template_obj, dict) else spec)
        containers = pod_spec.get("containers", [])

        node_selector = pod_spec.get("nodeSelector", {})
        annotations = dict(metadata.get("annotations", {}))
        labels = dict(metadata.get("labels", {}))
        if isinstance(template_obj, dict):
            tmpl_meta = template_obj.get("metadata", {})
            annotations.update(tmpl_meta.get("annotations", {}))
            labels.update(tmpl_meta.get("labels", {}))

        if labels.get("ai.dynamo/component") == "worker":
            has_dynamo_worker = True

        if (labels.get("llm-d.ai/component") == "epp" or name == "llm-d-epp"
                or name.endswith("-epp")):
            has_llmd_router = True
            for fc in containers:
                f_env = fc.get("env", [])
                f_args = fc.get("args", [])
                _scan_router_text(" ".join([str(a) for a in f_args] + [
                    f"{e.get('name', '')}={e.get('value', '')}"
                    for e in f_env if isinstance(e, dict)
                ]))
            continue
        if ("dynamo-frontend" in name or "dynamo-router" in name
                or labels.get("ai.dynamo/component") == "frontend"):
            has_dynamo_frontend = True
            for fc in containers:
                f_env = fc.get("env", [])
                f_args = fc.get("args", [])
                for p in fc.get("ports", []):
                    if isinstance(p, dict) and p.get("containerPort"):
                        frontend_port = int(p["containerPort"])
                _scan_router_text(" ".join([str(a) for a in f_args] + [
                    f"{e.get('name', '')}={e.get('value', '')}"
                    for e in f_env if isinstance(e, dict)
                ]))
            continue

        total_replicas += replicas

        if "cloud.google.com/gke-tpu-accelerator" in node_selector:
            is_tpu = True
            accelerator_type = node_selector[
                "cloud.google.com/gke-tpu-accelerator"]
        if "cloud.google.com/gke-accelerator" in node_selector:
            is_gpu = True
            gke_accelerator = node_selector["cloud.google.com/gke-accelerator"]
        if "cloud.google.com/gke-nodepool" in node_selector:
            node_pool = node_selector["cloud.google.com/gke-nodepool"]
        if "cloud.google.com/gke-tpu-topology" in annotations:
            accelerator_topology = annotations[
                "cloud.google.com/gke-tpu-topology"]
        elif "cloud.google.com/gke-tpu-topology" in node_selector:
            accelerator_topology = node_selector[
                "cloud.google.com/gke-tpu-topology"]

        for c in containers:
            c_name = c.get("name", "").lower()
            image = c.get("image", "").lower()
            args = c.get("args", [])
            args_str = " ".join(str(a) for a in args)
            command = c.get("command", [])
            cmd_str = " ".join(str(a) for a in command)
            env_list = c.get("env", [])
            env_str = " ".join(f"{e.get('name', '')}={e.get('value', '')}"
                               for e in env_list if isinstance(e, dict))
            raw_exec = f"{cmd_str} {args_str}"
            all_exec = f"{raw_exec} {env_str}".lower()

            if "dynamo" in image or "dynamo" in c_name:
                if "worker" in c_name or "worker" in image:
                    has_dynamo_worker = True
                if "frontend" in c_name or "frontend" in image:
                    has_dynamo_frontend = True
                    _scan_router_text(all_exec)
                    continue

            if "llm-d" in image or "llmd" in c_name or "epp" in c_name:
                has_llmd_router = True
                if "epp" in c_name or "inference-scheduler" in image:
                    _scan_router_text(all_exec)
                    continue

            if "vllm" in image or "vllm" in all_exec:
                detected_engines.add("vllm")
            elif "sglang" in image or "sglang" in all_exec:
                detected_engines.add("sglang")

            env_map = {
                e.get("name", ""): str(e.get("value", ""))
                for e in env_list if isinstance(e, dict)
            }
            match = re.search(r"--model(?:=|\s+)([^\s]+)", raw_exec,
                              re.IGNORECASE)
            if match:
                raw_m = match.group(1).strip("'\"")
                if raw_m.startswith("$"):
                    var_n = raw_m.strip("${}()")
                    resolved_m = env_map.get(var_n)
                    detected_models.add(resolved_m if resolved_m else raw_m)
                else:
                    detected_models.add(raw_m)
            else:
                for k in ("MODEL_PATH", "MODEL_NAME", "SERVED_MODEL_NAME"):
                    if env_map.get(k) and not env_map[k].startswith("$"):
                        detected_models.add(env_map[k])
                        break

            resources_req = c.get("resources", {})
            limits = resources_req.get("limits", {})
            if "google.com/tpu" in limits:
                is_tpu = True
                try:
                    per_node = int(limits["google.com/tpu"])
                    chips_per_node = max(chips_per_node, per_node)
                    accelerator_chips += per_node * replicas * lws_size
                except ValueError:
                    pass
            elif "nvidia.com/gpu" in limits:
                is_gpu = True
                accelerator_type = "gpu"
                try:
                    per_node = int(limits["nvidia.com/gpu"])
                    chips_per_node = max(chips_per_node, per_node)
                    accelerator_chips += per_node * replicas * lws_size
                except ValueError:
                    pass

            homogeneous_pools.append(name)

    topology = "unknown"
    if has_dynamo_frontend or has_dynamo_worker:
        topology = "dynamo-tpu" if is_tpu else "dynamo-gpu"
    elif (has_inference_pool or has_llmd_router or len(gateways) > 0
          or len(http_routes) > 0):
        topology = "multi-replica-llmd-homogeneous"
    elif len(workload_objects) > 0:
        if len(lws_list) > 0:
            lws_size = (lws_list[0].get("spec",
                                        {}).get("leaderWorkerTemplate",
                                                {}).get("size", 1))
            topology = "multi-host-single-replica" if lws_size > 1 else "single-host"
        elif total_replicas == 1:
            topology = "single-host"
        elif total_replicas > 1:
            topology = "multi-host-single-replica"

    router_type = "none"
    if has_dynamo_frontend:
        router_type = "dynamo-frontend"
    elif has_inference_pool or has_llmd_router:
        router_type = "llm-d-epp"

    if has_inference_pool:
        installed_crds.add("InferencePool")
    if len(gateways) > 0 or len(http_routes) > 0:
        installed_crds.add("Gateway")

    tpu_total_nodes = 0
    tpu_ready_nodes = 0
    if not is_tpu and not is_gpu and nodes:
        for n in nodes:
            n_labels = n.get("metadata", {}).get("labels", {})
            if "cloud.google.com/gke-tpu-accelerator" in n_labels:
                is_tpu = True
                tpu_total_nodes += 1
                n_spec = n.get("spec", {}) if isinstance(n.get("spec"),
                                                         dict) else {}
                n_status = (n.get("status", {}) if isinstance(
                    n.get("status"), dict) else {})
                conditions = n_status.get("conditions", [])
                if not conditions:
                    is_ready = not bool(n_spec.get("unschedulable"))
                else:
                    is_ready = not bool(n_spec.get("unschedulable")) and any(
                        isinstance(c, dict) and c.get("type") == "Ready"
                        and str(c.get("status")).lower() == "true"
                        for c in conditions)
                if is_ready:
                    tpu_ready_nodes += 1
                accelerator_type = str(
                    n_labels["cloud.google.com/gke-tpu-accelerator"])
                if "cloud.google.com/gke-tpu-topology" in n_labels:
                    accelerator_topology = str(
                        n_labels["cloud.google.com/gke-tpu-topology"])
                if "cloud.google.com/gke-nodepool" in n_labels:
                    node_pool = str(n_labels["cloud.google.com/gke-nodepool"])
                alloc = (n_status.get("allocatable", {}) if isinstance(
                    n_status.get("allocatable"), dict) else {})
                cap = (n_status.get("capacity", {}) if isinstance(
                    n_status.get("capacity"), dict) else {})
                raw_tpu_chips = alloc.get("google.com/tpu") or cap.get(
                    "google.com/tpu")
                if raw_tpu_chips is not None:
                    try:
                        per_node = int(raw_tpu_chips)
                        chips_per_node = max(chips_per_node, per_node)
                        accelerator_chips += per_node
                    except ValueError:
                        pass

    accel_info: dict[str, Any] = {
        "is_tpu": is_tpu,
        "is_gpu": is_gpu,
        "type": accelerator_type,
        "topology": accelerator_topology,
        "total_chips": accelerator_chips,
        "chips_per_node": chips_per_node,
    }
    if gke_accelerator:
        accel_info["gke_accelerator"] = gke_accelerator
    if node_pool:
        accel_info["node_pool"] = node_pool
    if tpu_total_nodes > 0:
        accel_info["total_nodes"] = tpu_total_nodes
        accel_info["ready_nodes"] = tpu_ready_nodes

    result: dict[str, Any] = {
        "topology": topology,
        "model_id": list(detected_models)[0] if detected_models else "unknown",
        "engine": list(detected_engines)[0] if detected_engines else "unknown",
        "router": router_type,
        "router_config": {
            "session_affinity": session_affinity_enabled,
            "session_headers": session_headers,
            "kv_routing": kv_routing_enabled,
            "frontend_port": frontend_port or 8000,
        },
        "accelerator": accel_info,
        "replicas": {
            "total": total_replicas,
            "homogeneous_pools": homogeneous_pools,
        },
        "resource_counts": {
            "deployments": len(deployments),
            "statefulsets": len(statefulsets),
            "leaderworkersets": len(lws_list),
            "inferencepools": len(inference_pools),
            "gateways": len(gateways),
            "httproutes": len(http_routes),
            "services": len(services),
            "configmaps": len(configmaps),
            "pods": len(pods),
        },
    }

    if target_cluster_only:
        result["topology"] = "target-cluster-preflight"
        result["target_cluster_only"] = True
    elif endpoint_picker_config:
        result["endpoint_picker_config"] = endpoint_picker_config
    if installed_crds:
        result["installed_crds"] = sorted(installed_crds)

    return result


def inspect_deployment(
    manifests: Path | None = None,
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
    store_dir: Path | None = None,
    capture_logs: bool = False,
    target_cluster_only: bool = False,
) -> dict[str, Any]:
    """Inspects a deployment either from manifest files or from a live Kubernetes cluster."""
    is_offline = bool(manifests and manifests.exists())
    if is_offline:
        assert manifests is not None
        resources = load_yaml_documents(manifests)
    elif target_cluster_only:
        resources = query_target_cluster_resources(kubeconfig=kubeconfig,
                                                   context=context)
    else:
        resources = query_cluster_resources(namespace,
                                            kubeconfig=kubeconfig,
                                            context=context)

    report = inspect_resources(resources,
                               target_cluster_only=target_cluster_only)
    if store_dir is not None:
        store_captured_resources(
            resources,
            store_dir,
            report,
            namespace=namespace,
            kubeconfig=kubeconfig,
            context=context,
            capture_logs=capture_logs,
            offline=is_offline,
            target_cluster_only=target_cluster_only,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect GKE serving deployment topologies.")
    parser.add_argument(
        "--manifests",
        type=Path,
        default=None,
        help="Path to manifest file or directory to inspect offline",
    )
    parser.add_argument("--namespace",
                        default="default",
                        help="Kubernetes namespace to inspect")
    parser.add_argument("--kubeconfig",
                        default=None,
                        help="Path to kubeconfig")
    parser.add_argument("--context", default=None, help="Kubectl context")
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=None,
        help=
        "Directory to store captured cluster manifests, cluster_state.txt, and inspection_report.json",
    )
    parser.add_argument(
        "--capture-logs",
        action="store_true",
        default=False,
        help=
        "Capture pod logs (<store-dir>/pod_logs/<pod_name>.log) when --store-dir is set",
    )
    parser.add_argument(
        "--target-cluster-only",
        action="store_true",
        default=False,
        help=
        "Inspect only cluster-level capabilities (installed CRDs and TPU node pools) without listing or dumping namespace workloads",
    )
    parser.add_argument("--json",
                        action="store_true",
                        default=True,
                        help="Print result as JSON")

    args = parser.parse_args()
    report = inspect_deployment(
        manifests=args.manifests,
        namespace=args.namespace,
        kubeconfig=args.kubeconfig,
        context=args.context,
        store_dir=args.store_dir,
        capture_logs=args.capture_logs,
        target_cluster_only=args.target_cluster_only,
    )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
