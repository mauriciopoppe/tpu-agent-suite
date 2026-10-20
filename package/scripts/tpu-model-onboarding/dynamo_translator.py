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
"""dynamo_translator.py: Translates NVIDIA Dynamo manifests into homogeneous GKE TPU llm-d manifests.

Maps:
- Dynamo frontend / scheduler -> GKE Gateway API + EndpointPickerConfig (EPP)
- Dynamo GPU worker pools -> GKE TPU homogeneous LeaderWorkerSets
- GPU vLLM / SGLang flags -> TPU-calibrated vLLM-TorchTPU flags and env vars
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class _LiteralBlockDumper(yaml.SafeDumper):
    """YAML SafeDumper that renders multiline strings using literal block style (`|`)."""


def _represent_str_literal_block(dumper: yaml.SafeDumper,
                                 data: str) -> yaml.ScalarNode:
    if "\n" in data:
        cleaned = "\n".join(line.rstrip() for line in data.splitlines())
        if data.endswith("\n"):
            cleaned += "\n"
        return dumper.represent_scalar("tag:yaml.org,2002:str",
                                       cleaned,
                                       style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralBlockDumper.add_representer(str, _represent_str_literal_block)


def _dump_yaml(doc: Mapping[str, Any]) -> str:
    """Serialize a single YAML document using literal block scalars for multiline strings."""
    return yaml.dump(
        dict(doc),
        Dumper=_LiteralBlockDumper,
        sort_keys=False,
        default_flow_style=False,
        width=4096,
    )


def _dump_yaml_all(docs: Sequence[Mapping[str, Any]]) -> str:
    """Serialize multiple YAML documents using literal block scalars for multiline strings."""
    return yaml.dump_all(
        [dict(d) for d in docs],
        Dumper=_LiteralBlockDumper,
        sort_keys=False,
        default_flow_style=False,
        width=4096,
    )


_VLLM_PATCHES_DIR = Path(
    __file__).resolve().parents[1] / "common" / "vllm_patches"

DEFAULT_TPU_VLLM_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-images/vllm/vllm-torchtpu:latest")

_DEPLOYMENTS_JSON_SKIP_KEYS = frozenset({
    "model_id",
    "model",
    "engine",
    "port",
    "parallelism",
    "env",
    "envs",
    "vllm_patch",
    "patch",
    "image",
    "image_pull_secret",
    "pvc_name",
    "pvc_mount_path",
    "hf_secret_name",
    "hf_secret_key",
    "epp_service_account",
    "envoy_configmap",
})

TPU_PEAK_PREFILL_RATES = {
    "v4": 7000,
    "v5e": 8500,
    "v5p": 16000,
    "v6e": 18500,
    "v7x": 42000,
    "tpu7x": 42000,
}

TPU_TOPOLOGY_CHIPS = {
    "1x1": 1,
    "2x2": 4,
    "2x4": 8,
    "4x4": 16,
    "4x8": 32,
    "1x1x1": 1,
    "2x2x1": 4,
    "2x2x2": 8,
    "2x4x1": 8,
    "2x2x4": 16,
    "2x4x4": 32,
    "4x4x4": 64,
}

TPU_CHIPS_PER_HOST = {
    "v4": 4,
    "v5e": 8,
    "v5p": 4,
    "v6e": 8,
    "v7x": 8,
    "tpu7x": 8,
}

TPU_GKE_ACCELERATOR_LABELS = {
    "v4": "tpu-v4-podslice",
    "v5e": "tpu-v5-lite-podslice",
    "v5p": "tpu-v5p-slice",
    "v6e": "tpu-v6e-slice",
    "v7x": "tpu7x",
    "tpu7x": "tpu7x",
}


def parse_dynamo_manifests(manifest_path: Path) -> dict[str, Any]:
    """Parses Dynamo manifests from a file or directory into structured components."""
    docs: list[dict[str, Any]] = []
    if manifest_path.is_file():
        content = manifest_path.read_text(encoding="utf-8")
        docs = [d for d in yaml.safe_load_all(content) if isinstance(d, dict)]
    elif manifest_path.is_dir():
        for ext in ("*.yaml", "*.yml", "*.json"):
            for f in sorted(manifest_path.glob(ext)):
                content = f.read_text(encoding="utf-8")
                docs.extend([
                    d for d in yaml.safe_load_all(content)
                    if isinstance(d, dict)
                ])

    frontend_objs: list[dict[str, Any]] = []
    controller_worker_objs: list[dict[str, Any]] = []
    pod_worker_objs: list[dict[str, Any]] = []
    config_objs: list[dict[str, Any]] = []
    other_objs: list[dict[str, Any]] = []
    sla_targets: dict[str, Any] = {}

    for d in docs:
        kind = d.get("kind", "").lower()
        meta = d.get("metadata", {})
        name = meta.get("name", "").lower()
        top_labels = meta.get("labels", {})
        if kind in ("dynamographdeployment", "dynamocomponentdeployment"):
            spec = d.get("spec", {})
            if isinstance(spec.get("sla"), dict):
                sla_targets.update(spec["sla"])
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
                    frontend_objs.append({
                        "kind": "Frontend",
                        "metadata": {
                            "name": c_name,
                            "labels": dict(comp_labels)
                        },
                        "spec": c_spec,
                    })
                else:
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
                    controller_worker_objs.append(worker_doc)
        elif kind == "dynamographdeploymentrequest":
            spec = d.get("spec", {})
            if isinstance(spec.get("sla"), dict):
                sla_targets.update(spec["sla"])
            model_id = spec.get("model", "meta-llama/Llama-3.1-8B-Instruct")
            backend = spec.get("backend", "vllm")
            frontend_objs.append({
                "kind": "Frontend",
                "metadata": {
                    "name": "dynamo-frontend"
                },
                "spec": {},
            })
            worker_doc = {
                "kind": "Deployment",
                "metadata": {
                    "name": "dynamo-worker"
                },
                "spec": {
                    "replicas": 2,
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "worker",
                                "image": f"{backend}/{backend}",
                                "args": ["--model", model_id],
                                "resources": {
                                    "limits": {
                                        "nvidia.com/gpu": 4
                                    }
                                },
                            }]
                        }
                    },
                },
            }
            controller_worker_objs.append(worker_doc)
        elif kind == "pod":
            containers = d.get("spec", {}).get("containers", [])
            comp_label = top_labels.get("ai.dynamo/component")
            is_frontend = (
                "frontend" in name or "router" in name
                or comp_label == "frontend"
                or any("frontend" in c.get("name", "").lower()
                       or "router" in c.get("name", "").lower()
                       or "dynamo-frontend" in c.get("image", "").lower()
                       for c in containers))
            is_worker = ("worker" in name or comp_label == "worker"
                         or any("worker" in c.get("name", "").lower()
                                or "vllm" in c.get("image", "").lower()
                                or "sglang" in c.get("image", "").lower()
                                for c in containers))
            if is_frontend and not is_worker:
                frontend_objs.append(d)
            elif is_worker:
                pod_worker_objs.append(d)
            else:
                other_objs.append(d)
        elif kind == "configmap":
            config_objs.append(d)
        elif ("dynamo" in name
              and "frontend" in name) or (top_labels.get("ai.dynamo/component")
                                          == "frontend"):
            frontend_objs.append(d)
        elif kind not in (
                "service",
                "secret",
                "serviceaccount",
                "ingress",
                "httproute",
                "gateway",
                "endpoints",
                "endpointslice",
                "poddisruptionbudget",
                "horizontalpodautoscaler",
                "role",
                "rolebinding",
                "clusterrole",
                "clusterrolebinding",
                "persistentvolumeclaim",
                "networkpolicy",
                "namespace",
        ) and (("dynamo" in name and "worker" in name) or
               (top_labels.get("ai.dynamo/component") == "worker")):
            controller_worker_objs.append(d)
        elif kind in ("deployment", "statefulset", "leaderworkerset"):
            lws_worker_tmpl = (d.get("spec",
                                     {}).get("leaderWorkerTemplate",
                                             {}).get("workerTemplate", {}))
            tmpl = d.get("spec", {}).get("template") or lws_worker_tmpl
            tmpl_labels = (tmpl.get("metadata", {}).get("labels", {})
                           if isinstance(tmpl, dict) else {})
            spec = (tmpl.get("spec", d.get("spec", {})) if isinstance(
                tmpl, dict) else d.get("spec", {}))
            containers = spec.get("containers", [])
            comp_label = top_labels.get(
                "ai.dynamo/component") or tmpl_labels.get(
                    "ai.dynamo/component")
            is_worker = comp_label == "worker" or any(
                "worker" in c.get("name", "").lower()
                or "vllm" in c.get("image", "").lower()
                or "sglang" in c.get("image", "").lower() for c in containers)
            is_frontend = comp_label == "frontend" or any(
                "frontend" in c.get("name", "").lower()
                or "router" in c.get("name", "").lower()
                or "dynamo-frontend" in c.get("image", "").lower()
                for c in containers)
            if is_frontend and not is_worker:
                frontend_objs.append(d)
            elif is_worker:
                controller_worker_objs.append(d)
            else:
                other_objs.append(d)
        else:
            other_objs.append(d)

    worker_objs = controller_worker_objs if controller_worker_objs else pod_worker_objs

    return {
        "frontend": frontend_objs,
        "workers": worker_objs,
        "configs": config_objs,
        "other": other_objs,
        "sla_targets": sla_targets,
        "all_docs": docs,
    }


def extract_frontend_config(
        frontend_objs: list[dict[str, Any]]) -> dict[str, Any]:
    """Extracts router mode, session affinity headers, and port from Dynamo frontend manifests."""
    session_affinity = False
    kv_routing = False
    port = 8000
    image = ""
    session_headers = ["X-Session-ID", "X-Correlation-ID"]

    for obj in frontend_objs:
        spec = obj.get("spec", {})
        if not isinstance(spec, dict):
            continue
        if spec.get("image") and not image:
            image = str(spec["image"])
        if spec.get("port"):
            port = int(spec["port"])
        for p in spec.get("ports", []):
            if isinstance(p, dict) and p.get("port"):
                port = int(p["port"])

        tmpl_spec = spec.get("template", {}).get("spec", {})
        containers = list(tmpl_spec.get("containers", []))
        if spec.get("env") or spec.get("args"):
            containers.append({
                "env": spec.get("env", []),
                "args": spec.get("args", []),
                "image": spec.get("image", ""),
            })
        for c in containers:
            if c.get("image") and not image:
                image = str(c["image"])
            for p in c.get("ports", []):
                if isinstance(p, dict) and p.get("containerPort"):
                    port = int(p["containerPort"])
            env_list = c.get("env", [])
            args_list = c.get("args", [])
            text = " ".join([str(a) for a in args_list] + [
                f"{e.get('name', '')}={e.get('value', '')}"
                for e in env_list if isinstance(e, dict)
            ]).lower()
            if ("aiperf_http_x_dynamo_session_id_from_correlation_id=1" in text
                    or "x-dynamo-session-id" in text or "x-session-id" in text
                    or "x-correlation-id" in text or "session" in text):
                session_affinity = True
            if ("dyn_router_mode=kv" in text or "--router-mode=kv" in text
                    or "--router-mode kv" in text or "kv" in text):
                kv_routing = True

    return {
        "image": image,
        "port": port,
        "session_affinity": session_affinity,
        "session_headers": session_headers,
        "kv_routing": kv_routing,
    }


def extract_worker_config(worker_doc: dict[str, Any]) -> dict[str, Any]:
    """Extracts model, args, resources, multinode count, and environment from a worker manifest."""
    spec = worker_doc.get("spec", {})
    replicas = spec.get("replicas", 1)
    lws_tmpl = spec.get("leaderWorkerTemplate", {})
    pod_tmpl = spec.get("template") or lws_tmpl.get("workerTemplate", {})
    pod_spec = pod_tmpl.get("spec", spec) if isinstance(pod_tmpl,
                                                        dict) else spec
    containers = pod_spec.get("containers", [])
    node_selector = dict(pod_spec.get("nodeSelector", {}))

    multinode = spec.get("multinode")
    node_count = 1
    if isinstance(multinode, dict) and multinode.get("nodeCount"):
        node_count = int(multinode["nodeCount"])
    elif isinstance(lws_tmpl, dict) and lws_tmpl.get("size"):
        node_count = int(lws_tmpl["size"])

    main_container: dict[str, Any] = containers[0] if containers else {}
    for c in containers:
        c_name_lower = c.get("name", "").lower()
        c_img_lower = c.get("image", "").lower()
        if ("worker" in c_name_lower or "vllm" in c_img_lower
                or "sglang" in c_img_lower or "main" in c_name_lower):
            main_container = c
            break

    args = main_container.get("args", [])
    args_str = " ".join(str(a) for a in args)

    env_list = main_container.get("env", [])
    env_map = {
        e.get("name", ""): str(e.get("value", ""))
        for e in env_list if isinstance(e, dict)
    }

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    match = re.search(r"--model(?:=|\s+)([^\s]+)", args_str)
    if match:
        raw_m = match.group(1).strip("'\"")
        if raw_m.startswith("$"):
            var_n = raw_m.strip("${}()")
            model_id = env_map.get(var_n, raw_m)
        else:
            model_id = raw_m
    else:
        for k in ("MODEL_PATH", "MODEL_NAME", "SERVED_MODEL_NAME"):
            if env_map.get(k) and not env_map[k].startswith("$"):
                model_id = env_map[k]
                break

    if model_id.startswith("$"):
        var_n = model_id.strip("${}()")
        model_id = env_map.get(var_n, "meta-llama/Llama-3.1-8B-Instruct")

    max_model_len: int | None = None
    len_match = re.search(r"--max-model-len(?:=|\s+)(\d+)", args_str)
    if len_match:
        max_model_len = int(len_match.group(1))

    tp_size: int | None = None
    tp_match = re.search(r"--tensor-parallel-size(?:=|\s+)(\d+)", args_str)
    if tp_match:
        tp_size = int(tp_match.group(1))

    limits = main_container.get("resources", {}).get("limits", {})
    gpu_count = int(limits.get("nvidia.com/gpu") or limits.get("gpu", 4))

    return {
        "name": worker_doc.get("metadata", {}).get("name", "worker"),
        "role": "homogeneous",
        "model_id": model_id,
        "replicas": replicas,
        "gpu_count": gpu_count,
        "node_count": node_count,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tp_size,
        "node_selector": node_selector,
        "args_str": args_str,
        "env": env_list,
    }


def generate_endpoint_picker_config(
    namespace: str,
    target_tpu_type: str,
    include_standalone_config: bool = False,
) -> dict[str, Any]:
    """Generates the Kubernetes ConfigMap holding the homogeneous llm-d EndpointPickerConfig."""
    peak_rate = TPU_PEAK_PREFILL_RATES.get(target_tpu_type, 18500)
    epp_yaml = f"""apiVersion: config.llm-d.io/v1alpha1
kind: EndpointPickerConfig
metadata:
  name: epp-config
  namespace: {namespace}
scheduling:
  filters:
    - name: prefix-cache-affinity-filter
      parameters:
        sessionAffinityHeaders:
          - X-Session-ID
          - X-Correlation-ID
    - name: utilization-filter
  scorers:
    - name: prefix-cache-scorer
      weight: 100
    - name: token-load-scorer
      weight: 50
      parameters:
        peakPrefillThroughput: {peak_rate}
    - name: active-request-scorer
      weight: 20
    - name: kv-cache-utilization-scorer
      weight: 30
  picker:
    name: max-score-picker
flowControl:
  saturationDetector:
    name: utilization-detector
    parameters:
      threshold: 0.95
  admitter:
    name: probabilistic-admitter
    parameters:
      power: 2.0
"""

    data: dict[str, str] = {"endpoint-picker-config.yaml": epp_yaml}
    if include_standalone_config:
        data["default-plugins.yaml"] = """apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: queue-scorer
- type: kv-cache-utilization-scorer
- type: prefix-cache-scorer
- type: metrics-data-source
  parameters:
    scheme: "http"
    path: "/metrics"
    insecureSkipVerify: true
- type: core-metrics-extractor
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 2
  - pluginRef: kv-cache-utilization-scorer
    weight: 2
  - pluginRef: prefix-cache-scorer
    weight: 3
"""
        data["envoy.yaml"] = """admin:
  address:
    socket_address: { address: 127.0.0.1, port_value: 19000 }
static_resources:
  listeners:
    - name: http-gateway-listener
      address:
        socket_address: { address: 0.0.0.0, port_value: 8081 }
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: http-8081
                route_config:
                  name: vllm_route
                  virtual_hosts:
                    - name: vllm_host
                      domains: ["*"]
                      routes:
                        - match: { prefix: "/" }
                          route:
                            cluster: original_destination_cluster
                            timeout: 3600s
                            idle_timeout: 3600s
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      failure_mode_allow: true
                      grpc_service:
                        envoy_grpc:
                          cluster_name: ext_proc
                          authority: localhost:9002
                        timeout: 10s
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SEND
                        request_body_mode: FULL_DUPLEX_STREAMED
                        response_body_mode: FULL_DUPLEX_STREAMED
                        request_trailer_mode: SEND
                        response_trailer_mode: SEND
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
  clusters:
    - name: original_destination_cluster
      type: ORIGINAL_DST
      connect_timeout: 1000s
      lb_policy: CLUSTER_PROVIDED
      original_dst_lb_config:
        use_http_header: true
        http_header_name: x-gateway-destination-endpoint
    - name: ext_proc
      type: STATIC
      connect_timeout: 10s
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options: {}
      transport_socket:
        name: "envoy.transport_sockets.tls"
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext
          common_tls_context:
            validation_context: {}
      load_assignment:
        cluster_name: ext_proc
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address: { address: 127.0.0.1, port_value: 9002 }
"""

    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "epp-config",
            "namespace": namespace,
        },
        "data": data,
    }


def generate_endpoint_picker_manifests(
    namespace: str,
    target_tpu_type: str,
    pool_name: str = "llm-d-pool",
    standalone_epp: bool = False,
    envoy_configmap: str | None = None,
    epp_service_account: str | None = None,
) -> list[dict[str, Any]]:
    """Generates the EPP ConfigMap, Deployment, and Service manifests for homogeneous llm-d."""
    use_standalone = bool(standalone_epp or envoy_configmap
                          or epp_service_account)
    epp_cm = generate_endpoint_picker_config(
        namespace,
        target_tpu_type,
        include_standalone_config=use_standalone,
    )
    epp_labels: dict[str, str] = {
        "app": "llm-d-epp",
        "llm-d.ai/component": "epp",
    }

    if use_standalone:
        epp_container: dict[str, Any] = {
            "name":
            "epp",
            "image":
            "ghcr.io/llm-d/llm-d-router-endpoint-picker:main",
            "args": [
                "--endpoint-selector",
                "llm-d.ai/guide=optimized-baseline,leaderworkerset.sigs.k8s.io/worker-index=0",
                "--endpoint-target-ports",
                "8000",
                "--zap-encoder=json",
                "--config-file=/etc/epp/default-plugins.yaml",
                "--grpc-health-port=9003",
                "--v=2",
                "--tracing=false",
            ],
            "env": [
                {
                    "name": "SESSION_AFFINITY_HEADERS",
                    "value": "X-Session-ID,X-Correlation-ID",
                },
                {
                    "name": "NAMESPACE",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": "metadata.namespace"
                        }
                    },
                },
                {
                    "name": "POD_NAME",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": "metadata.name"
                        }
                    },
                },
            ],
            "ports": [
                {
                    "name": "grpc",
                    "containerPort": 9002
                },
                {
                    "name": "health",
                    "containerPort": 9003
                },
                {
                    "name": "metrics",
                    "containerPort": 9090
                },
            ],
            "resources": {
                "limits": {
                    "memory": "8Gi"
                },
                "requests": {
                    "cpu": "1",
                    "memory": "2Gi"
                },
            },
            "volumeMounts": [{
                "name": "epp-config-vol",
                "mountPath": "/etc/epp",
                "readOnly": True,
            }],
        }
        envoy_container: dict[str, Any] = {
            "name":
            "envoy-proxy",
            "image":
            "docker.io/envoyproxy/envoy:distroless-v1.33.2",
            "command": ["envoy"],
            "args": [
                "--service-node",
                "envoy-proxy",
                "--log-level",
                "warn",
                "--cpuset-threads",
                "--drain-strategy",
                "immediate",
                "--drain-time-s",
                "60",
                "-c",
                "/etc/envoy/envoy.yaml",
            ],
            "ports": [
                {
                    "name": "http-8081",
                    "containerPort": 8081
                },
                {
                    "name": "metrics-19001",
                    "containerPort": 19001
                },
            ],
            "resources": {
                "limits": {
                    "memory": "4Gi"
                },
                "requests": {
                    "cpu": "500m",
                    "memory": "1Gi"
                },
            },
            "volumeMounts": [{
                "name": "envoy-config-vol",
                "mountPath": "/etc/envoy",
                "readOnly": True,
            }],
        }
        containers = [epp_container, envoy_container]
        volumes = [
            {
                "name": "epp-config-vol",
                "configMap": {
                    "name": "epp-config"
                }
            },
            {
                "name": "envoy-config-vol",
                "configMap": {
                    "name": envoy_configmap or "epp-config"
                },
            },
        ]
    else:
        containers = [{
            "name":
            "epp",
            "image":
            "ghcr.io/llm-d/llm-d-inference-scheduler:latest",
            "args": [
                "--pool-name",
                pool_name,
                "--pool-namespace",
                namespace,
                "--config-file",
                "/etc/epp/endpoint-picker-config.yaml",
                "--grpc-port",
                "9002",
                "--grpc-health-port",
                "9003",
            ],
            "env": [{
                "name": "SESSION_AFFINITY_HEADERS",
                "value": "X-Session-ID,X-Correlation-ID",
            }],
            "ports": [
                {
                    "name": "grpc",
                    "containerPort": 9002
                },
                {
                    "name": "health",
                    "containerPort": 9003
                },
                {
                    "name": "metrics",
                    "containerPort": 9090
                },
            ],
            "volumeMounts": [{
                "name": "epp-config-vol",
                "mountPath": "/etc/epp",
                "readOnly": True,
            }],
        }]
        volumes = [{
            "name": "epp-config-vol",
            "configMap": {
                "name": "epp-config"
            }
        }]

    tmpl_spec: dict[str, Any] = {
        "containers": containers,
        "volumes": volumes,
    }
    if epp_service_account:
        tmpl_spec["serviceAccountName"] = epp_service_account

    epp_deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "llm-d-epp",
            "namespace": namespace,
            "labels": dict(epp_labels),
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": "llm-d-epp"
                }
            },
            "template": {
                "metadata": {
                    "labels": dict(epp_labels)
                },
                "spec": tmpl_spec,
            },
        },
    }
    service_ports = [
        {
            "name": "grpc",
            "port": 9002,
            "targetPort": 9002
        },
        {
            "name": "metrics",
            "port": 9090,
            "targetPort": 9090
        },
    ]
    if use_standalone:
        service_ports.extend([
            {
                "name": "http",
                "port": 8000,
                "targetPort": 8081
            },
            {
                "name": "http-80",
                "port": 80,
                "targetPort": 8081
            },
        ])
    epp_service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "llm-d-epp",
            "namespace": namespace,
            "labels": dict(epp_labels),
        },
        "spec": {
            "selector": {
                "app": "llm-d-epp"
            },
            "ports": service_ports,
        },
    }
    return [epp_cm, epp_deployment, epp_service]


def generate_gateway_and_route(namespace: str,
                               service_name: str) -> list[dict[str, Any]]:
    """Generates Gateway API Gateway and HTTPRoute manifests for llm-d."""
    gateway = {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {
            "name": "llm-d-gateway",
            "namespace": namespace,
        },
        "spec": {
            "gatewayClassName":
            "gke-l7-regional-external-managed",
            "listeners": [{
                "name": "http",
                "protocol": "HTTP",
                "port": 80,
                "allowedRoutes": {
                    "namespaces": {
                        "from": "Same"
                    }
                },
            }],
        },
    }

    http_route = {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": "llm-d-route",
            "namespace": namespace,
        },
        "spec": {
            "parentRefs": [{
                "name": "llm-d-gateway"
            }],
            "rules": [{
                "matches": [{
                    "path": {
                        "type": "PathPrefix",
                        "value": "/v1"
                    }
                }],
                "backendRefs": [{
                    "group": "inference.networking.x-k8s.io",
                    "kind": "InferencePool",
                    "name": service_name,
                    "port": 8000,
                }],
            }],
        },
    }

    return [gateway, http_route]


def generate_inference_pool(
    namespace: str,
    pool_name: str,
    match_labels: dict[str, str],
) -> dict[str, Any]:
    """Generates an InferencePool custom resource for llm-d."""
    effective_labels = dict(match_labels)
    effective_labels.setdefault("leaderworkerset.sigs.k8s.io/worker-index",
                                "0")
    return {
        "apiVersion": "inference.networking.x-k8s.io/v1",
        "kind": "InferencePool",
        "metadata": {
            "name": pool_name,
            "namespace": namespace,
        },
        "spec": {
            "selector": {
                "matchLabels": dict(effective_labels),
                **effective_labels,
            },
            "targetPortNumber": 8000,
            "targetPorts": [{
                "number": 8000
            }],
            "endpointPickerConfigRef": {
                "name": "epp-config"
            },
        },
    }


def load_deployments_json(
    deployments_json: Path | str | dict[str, Any] | None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Loads and selects a single-VM deployment entry from deployments.json."""
    if deployments_json is None:
        return {}
    if isinstance(deployments_json, dict):
        raw = deployments_json
    else:
        path = Path(deployments_json)
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    if not isinstance(raw, dict):
        return {}
    entries = raw.get("deployments")
    if isinstance(entries, list) and entries:
        valid = [d for d in entries if isinstance(d, dict)]
        if model_id:
            matched = next(
                (d for d in valid
                 if d.get("model_id") == model_id or d.get("model") == model_id
                 ),
                None,
            )
            if matched is not None:
                return dict(matched)
        return dict(valid[0]) if valid else {}
    return dict(raw)


def _discover_sibling_deployments_json(
    search_roots: Sequence[Path | None], ) -> Path | None:
    """Discovers deployments.json in store_dir, manifest_path, or sibling <work>/inference/deployments.json."""
    seen: set[Path] = set()
    for root in search_roots:
        if root is None:
            continue
        base_dir = root if root.is_dir() else root.parent
        direct = base_dir / "deployments.json"
        if direct.is_file():
            return direct
        for anc in (base_dir, *base_dir.parents):
            if anc in seen:
                continue
            seen.add(anc)
            inf_candidate = anc / "inference" / "deployments.json"
            if inf_candidate.is_file():
                return inf_candidate
            if anc.name == "tpu-model-onboarding":
                top_cand = anc / "deployments.json"
                if top_cand.is_file():
                    return top_cand
                break
            if anc.name in ("artifacts", "runs", ".tpu-agent-suite"):
                break
    return None


def _extract_image_from_env_doc(env_doc: Mapping[str, Any]) -> str | None:
    """Extracts a container image from a parsed environment.json document."""
    profiles = env_doc.get("profiles")
    if isinstance(profiles, Mapping):
        default_prof_name = env_doc.get("default")
        if isinstance(default_prof_name,
                      str) and default_prof_name in profiles:
            default_prof = profiles[default_prof_name]
            if isinstance(default_prof, Mapping):
                params = default_prof.get("params")
                img = (params.get("image") if isinstance(params, Mapping) else
                       None) or default_prof.get("image")
                if isinstance(img, str) and img.strip():
                    return img.strip()
        for prof in profiles.values():
            if not isinstance(prof, Mapping):
                continue
            backend = str(prof.get("backend") or "")
            if backend and backend not in ("gke_job", "gke_pod"):
                continue
            params = prof.get("params")
            img = (params.get("image") if isinstance(params, Mapping) else
                   None) or prof.get("image")
            if isinstance(img, str) and img.strip():
                return img.strip()
    top_img = env_doc.get("image")
    if isinstance(top_img, str) and top_img.strip():
        return top_img.strip()
    return None


def _resolve_tpu_base_image_with_source(
    deployments_config: Mapping[str, Any] | None = None,
    image_override: str | None = None,
    workspace: Path | None = None,
) -> tuple[str, str]:
    """Resolves the target TPU vLLM container image and its resolution source."""
    if isinstance(image_override, str) and image_override.strip():
        return image_override.strip(), "cli_override"
    if isinstance(deployments_config, Mapping):
        dep_img = deployments_config.get("image")
        if isinstance(dep_img, str) and dep_img.strip():
            return dep_img.strip(), "deployments_json"

    env_img = os.environ.get("TPU_VLLM_IMAGE") or os.environ.get("TASK_DOCKER")
    if isinstance(env_img, str) and env_img.strip():
        return env_img.strip(), "env_var"

    env_file_name = "environment" + ".json"
    candidate_files: list[Path] = []
    start_dir = (workspace if workspace is not None else Path.cwd())
    if start_dir.is_file():
        start_dir = start_dir.parent
    try:
        resolved_start = start_dir.resolve()
    except Exception:
        resolved_start = start_dir
    for anc in (resolved_start, *resolved_start.parents):
        candidate_files.append(anc / ".tpu-agent-suite" / env_file_name)

    suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
    if suite_home:
        sh_path = Path(suite_home).expanduser()
        candidate_files.append(sh_path / env_file_name)
        candidate_files.append(sh_path / ".tpu-agent-suite" / env_file_name)
    candidate_files.append(Path.home() / ".tpu-agent-suite" / env_file_name)

    visited: set[Path] = set()
    for env_path in candidate_files:
        if env_path in visited or not env_path.is_file():
            continue
        visited.add(env_path)
        try:
            env_doc = json.loads(env_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(env_doc, Mapping):
            extracted = _extract_image_from_env_doc(env_doc)
            if extracted:
                return extracted, "environment_json"

    return DEFAULT_TPU_VLLM_IMAGE, "default_fallback"


def resolve_tpu_base_image(
    deployments_config: Mapping[str, Any] | None = None,
    image_override: str | None = None,
    workspace: Path | None = None,
) -> str:
    """Resolves the target TPU vLLM container image across CLI, deployments.json, env, and environment.json."""
    resolved_image, _ = _resolve_tpu_base_image_with_source(
        deployments_config=deployments_config,
        image_override=image_override,
        workspace=workspace,
    )
    return resolved_image


def _set_or_append_cli_flag(
    args: list[str],
    flag: str,
    value: Any,
) -> None:
    """Sets or appends a CLI flag in container args without duplicating existing flags."""
    if value is None or value is False:
        if value is False and flag in args:
            args.remove(flag)
        return
    if value is True:
        if flag not in args and not any(
                a.startswith(f"{flag}=") for a in args):
            args.append(flag)
        return

    val_str = json.dumps(value) if isinstance(value,
                                              (dict, list)) else str(value)
    for idx, tok in enumerate(args):
        if tok == flag:
            if idx + 1 < len(args) and not str(args[idx + 1]).startswith("--"):
                args[idx + 1] = val_str
            else:
                args.insert(idx + 1, val_str)
            return
        if tok.startswith(f"{flag}="):
            args[idx] = f"{flag}={val_str}"
            return
    args.extend([flag, val_str])


def apply_deployments_config_to_container(
    container_args: list[str],
    env_vars: list[dict[str, str]],
    deployments_config: dict[str, Any],
) -> None:
    """Applies tuned flags and environment variables from deployments.json onto container args and env."""
    for key, val in deployments_config.items():
        if key in _DEPLOYMENTS_JSON_SKIP_KEYS:
            continue
        flag = "--" + str(key).lstrip("-").replace("_", "-")
        _set_or_append_cli_flag(container_args, flag, val)

    env_overrides = deployments_config.get("env") or deployments_config.get(
        "envs")
    existing_idx = {
        e["name"]: idx
        for idx, e in enumerate(env_vars)
        if isinstance(e, dict) and "name" in e
    }
    if isinstance(env_overrides, dict):
        for ek, ev in env_overrides.items():
            name_str, val_str = str(ek), str(ev)
            if name_str in existing_idx:
                env_vars[existing_idx[name_str]]["value"] = val_str
            else:
                existing_idx[name_str] = len(env_vars)
                env_vars.append({"name": name_str, "value": val_str})
    elif isinstance(env_overrides, list):
        for item in env_overrides:
            if isinstance(item, dict) and "name" in item:
                name_str = str(item["name"])
                val_str = str(item.get("value", ""))
                if name_str in existing_idx:
                    env_vars[existing_idx[name_str]]["value"] = val_str
                else:
                    existing_idx[name_str] = len(env_vars)
                    env_vars.append({"name": name_str, "value": val_str})


def resolve_vllm_patch(
    model_id: str,
    vllm_patch: Path | str | None = None,
    deployments_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolves a vllm_torchtpu runtime patch from --vllm-patch, deployments.json, or registry.json."""
    candidate: Path | str | None = vllm_patch
    if not candidate and deployments_config:
        candidate = deployments_config.get(
            "vllm_patch") or deployments_config.get("patch")

    registry_entry: dict[str, Any] | None = None
    registry_path = _VLLM_PATCHES_DIR / "registry.json"
    if registry_path.is_file():
        try:
            reg_data = json.loads(registry_path.read_text(encoding="utf-8"))
            models_map = reg_data.get("models", {})
            if isinstance(models_map, dict) and model_id in models_map:
                registry_entry = models_map[model_id]
                if not candidate and isinstance(registry_entry, dict):
                    candidate = registry_entry.get("patch")
        except Exception:
            pass

    if not candidate:
        return None

    cand_path = Path(str(candidate))
    if cand_path.is_file():
        patch_path = cand_path.resolve()
    elif (_VLLM_PATCHES_DIR / cand_path.name).is_file():
        patch_path = (_VLLM_PATCHES_DIR / cand_path.name).resolve()
    elif (_VLLM_PATCHES_DIR / str(candidate)).is_file():
        patch_path = (_VLLM_PATCHES_DIR / str(candidate)).resolve()
    else:
        raise FileNotFoundError(f"vLLM patch file not found: {candidate}")

    apply_script_path = _VLLM_PATCHES_DIR / "apply_vllm_patch.sh"
    apply_script_content = apply_script_path.read_text(encoding="utf-8")
    res: dict[str, Any] = {
        "patch_name":
        patch_path.name,
        "patch_path":
        str(patch_path),
        "patch_content":
        patch_path.read_text(encoding="utf-8"),
        "apply_script_content":
        apply_script_content,
        "provenance": (registry_entry.get("provenance", "custom")
                       if isinstance(registry_entry, dict) else "custom"),
    }
    if isinstance(registry_entry, dict) and "validated" in registry_entry:
        res["validated"] = registry_entry["validated"]
    return res


def generate_vllm_patches_configmap(
    namespace: str,
    patch_info: dict[str, Any],
) -> dict[str, Any]:
    """Generates the vllm-tpu-patches ConfigMap holding apply_vllm_patch.sh and the patch file."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "vllm-tpu-patches",
            "namespace": namespace,
            "labels": {
                "llm-d.ai/component": "vllm-patches",
                "llm-d.ai/accelerator-variant": "tpu",
            },
        },
        "data": {
            "apply_vllm_patch.sh": patch_info["apply_script_content"],
            patch_info["patch_name"]: patch_info["patch_content"],
        },
    }


def generate_tpu_serving_deployment(
    namespace: str,
    name: str,
    model_id: str,
    replicas: int,
    target_tpu_type: str,
    target_tpu_topology: str,
    node_count: int = 1,
    max_model_len: int | None = None,
    deployments_json: Path | str | dict[str, Any] | None = None,
    vllm_patch_name: str | None = None,
    image: str | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Generates a homogeneous Kubernetes LeaderWorkerSet targeting Cloud TPUs."""
    chip_count = TPU_TOPOLOGY_CHIPS.get(target_tpu_topology, 4)
    chips_per_host = TPU_CHIPS_PER_HOST.get(target_tpu_type, 8)
    topo_hosts = max(1, chip_count // chips_per_host)
    effective_node_count = max(node_count, topo_hosts)
    if effective_node_count > 1:
        per_pod_chips = max(chips_per_host, chip_count // effective_node_count)
    else:
        per_pod_chips = max(1, chip_count)

    vllm_port = 8000
    resolved_max_len = str(max_model_len) if max_model_len else "8192"

    dep_cfg = (load_deployments_json(deployments_json, model_id=model_id)
               if deployments_json is not None else {})
    resolved_image, resolved_image_source = _resolve_tpu_base_image_with_source(
        deployments_config=dep_cfg,
        image_override=image,
        workspace=workspace,
    )
    image_pull_secret = (str(dep_cfg["image_pull_secret"])
                         if dep_cfg.get("image_pull_secret") else None)
    pvc_name = str(dep_cfg["pvc_name"]) if dep_cfg.get("pvc_name") else None
    pvc_mount_path = str(dep_cfg.get("pvc_mount_path") or "/cache")
    hf_secret_name = (str(dep_cfg["hf_secret_name"])
                      if dep_cfg.get("hf_secret_name") else None)
    hf_secret_key = str(dep_cfg.get("hf_secret_key") or "HF_TOKEN")

    container_args = [
        "--model",
        model_id,
        f"--port={vllm_port}",
        "--tensor-parallel-size",
        str(chip_count),
        "--block-size",
        "768",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.95",
        "--max-model-len",
        resolved_max_len,
    ]

    env_vars: list[dict[str, Any]] = [
        {
            "name": "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL",
            "value": "1"
        },
        {
            "name": "VLLM_TORCHTPU_ACCELERATOR",
            "value": target_tpu_type
        },
    ]

    volumes: list[dict[str, Any]] = [{
        "name": "dshm",
        "emptyDir": {
            "medium": "Memory",
            "sizeLimit": "64Gi"
        },
    }]
    volume_mounts: list[dict[str, Any]] = [{
        "name": "dshm",
        "mountPath": "/dev/shm"
    }]

    if pvc_name:
        volumes.append({
            "name": "shared-pvc",
            "persistentVolumeClaim": {
                "claimName": pvc_name
            },
        })
        volume_mounts.append({
            "name": "shared-pvc",
            "mountPath": pvc_mount_path
        })

    if dep_cfg:
        apply_deployments_config_to_container(container_args, env_vars,
                                              dep_cfg)
        if not vllm_patch_name and (dep_cfg.get("vllm_patch")
                                    or dep_cfg.get("patch")):
            vllm_patch_name = Path(
                str(dep_cfg.get("vllm_patch") or dep_cfg.get("patch"))).name

    if hf_secret_name and not any(
            e.get("name") == "HF_TOKEN" for e in env_vars):
        env_vars.append({
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": hf_secret_name,
                    "key": hf_secret_key,
                    "optional": True,
                }
            },
        })

    if vllm_patch_name:
        volumes.append({
            "name": "vllm-patches",
            "configMap": {
                "name": "vllm-tpu-patches",
                "defaultMode": 0o755,
            },
        })
        volume_mounts.append({
            "name": "vllm-patches",
            "mountPath": "/app/patches",
            "readOnly": True,
        })
        container_command = ["/bin/bash", "-lc"]
        quoted_args = " ".join(shlex.quote(str(a)) for a in container_args)
        container_exec_args = [
            f"set -euo pipefail && bash /app/patches/apply_vllm_patch.sh /app/patches/{vllm_patch_name} && exec vllm serve {quoted_args}"
        ]
    else:
        container_command = ["vllm", "serve"]
        container_exec_args = container_args

    pod_labels = {
        "app": "vllm-tpu-homogeneous",
        "role": "homogeneous",
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/guide": "optimized-baseline",
        "llm-d.ai/role": "decode",
        "llm-d.ai/accelerator-variant": "tpu",
        "llm-d.ai/accelerator-vendor": "google",
    }

    pod_annotations = {
        "cloud.google.com/gke-tpu-topology":
        target_tpu_topology,
        "cloud.google.com/gke-tpu-accelerator-label":
        TPU_GKE_ACCELERATOR_LABELS.get(target_tpu_type, target_tpu_type),
    }

    container_resources = {
        "limits": {
            "google.com/tpu": str(per_pod_chips),
            "memory": "180Gi",
            "cpu": "32",
        },
        "requests": {
            "google.com/tpu": str(per_pod_chips),
            "memory": "120Gi",
            "cpu": "16",
        },
    }

    container_spec: dict[str, Any] = {
        "name": "vllm-tpu",
        "image": resolved_image,
        "command": container_command,
        "args": container_exec_args,
        "env": env_vars,
        "resources": container_resources,
        "volumeMounts": volume_mounts,
        "ports": [{
            "name": "http",
            "containerPort": vllm_port
        }],
        "readinessProbe": {
            "httpGet": {
                "path": "/health",
                "port": vllm_port
            },
            "initialDelaySeconds": 60,
            "periodSeconds": 10,
        },
    }
    if resolved_image_source != "default_fallback":
        container_spec["imagePullPolicy"] = "IfNotPresent"

    pod_spec: dict[str, Any] = {
        "nodeSelector": {
            "cloud.google.com/gke-tpu-accelerator": target_tpu_type,
            "cloud.google.com/gke-tpu-topology": target_tpu_topology,
        },
        "volumes": volumes,
        "containers": [container_spec],
    }
    if image_pull_secret:
        pod_spec["imagePullSecrets"] = [{"name": image_pull_secret}]

    lws_annotations = dict(pod_annotations)
    if effective_node_count > 1:
        lws_annotations[
            "leaderworkerset.sigs.k8s.io/exclusive-topology"] = "cloud.google.com/gke-nodepool"
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": dict(pod_labels),
        },
        "spec": {
            "replicas": replicas,
            "leaderWorkerTemplate": {
                "size": effective_node_count,
                "workerTemplate": {
                    "metadata": {
                        "labels": dict(pod_labels),
                        "annotations": lws_annotations,
                    },
                    "spec": pod_spec,
                },
            },
        },
    }


def render_kustomize_overlay(
    kustomize_dir: Path,
    namespace: str,
    workloads: list[dict[str, Any]],
    gateway_docs: list[dict[str, Any]] | None = None,
    epp_docs: list[dict[str, Any]] | None = None,
    inference_pool_doc: dict[str, Any] | None = None,
    target_tpu_type: str = "v6e",
    pool_name: str = "llm-d-pool",
    vllm_patches_doc: dict[str, Any] | None = None,
    standalone_epp: bool = False,
) -> Path:
    """Writes a self-contained Kustomize directory for the homogeneous llm-d stack."""
    kustomize_dir.mkdir(parents=True, exist_ok=True)
    guide_name = "optimized-baseline"

    if gateway_docs is None:
        gateway_docs = generate_gateway_and_route(namespace, pool_name)
    if epp_docs is None:
        epp_docs = generate_endpoint_picker_manifests(
            namespace,
            target_tpu_type,
            pool_name=pool_name,
            standalone_epp=standalone_epp,
        )
    if inference_pool_doc is None:
        pool_selector = {
            "llm-d.ai/inferenceServing": "true",
            "llm-d.ai/guide": guide_name,
            "leaderworkerset.sigs.k8s.io/worker-index": "0",
        }
        inference_pool_doc = generate_inference_pool(namespace, pool_name,
                                                     pool_selector)

    (kustomize_dir / "00-gateway.yaml").write_text(
        _dump_yaml_all(gateway_docs), encoding="utf-8")
    (kustomize_dir / "01-endpointpickerconfig.yaml").write_text(
        _dump_yaml_all(epp_docs), encoding="utf-8")
    (kustomize_dir / "02-inferencepool.yaml").write_text(
        _dump_yaml(inference_pool_doc), encoding="utf-8")

    sa_doc = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": f"{guide_name}-sa",
            "namespace": namespace,
            "labels": {
                "llm-d.ai/guide": guide_name,
                "llm-d.ai/accelerator-variant": "tpu",
            },
        },
    }
    base_docs = [sa_doc] + workloads
    (kustomize_dir / "base-modelserver.yaml").write_text(
        _dump_yaml_all(base_docs), encoding="utf-8")

    if standalone_epp:
        resources_list = [
            "01-endpointpickerconfig.yaml",
            "base-modelserver.yaml",
        ]
    else:
        resources_list = [
            "00-gateway.yaml",
            "01-endpointpickerconfig.yaml",
            "02-inferencepool.yaml",
            "base-modelserver.yaml",
        ]
    if vllm_patches_doc is not None:
        (kustomize_dir / "04-vllm-patches.yaml").write_text(
            _dump_yaml(vllm_patches_doc), encoding="utf-8")
        resources_list.append("04-vllm-patches.yaml")

    patches_meta: list[dict[str, Any]] = []
    for w in workloads:
        w_name = w["metadata"]["name"]
        w_kind = w["kind"]
        role = w["metadata"]["labels"].get("llm-d.ai/role", "decode")
        patch_filename = f"patch-{w_name}.yaml"
        patch_doc = {
            "apiVersion": w["apiVersion"],
            "kind": w_kind,
            "metadata": {
                "name": w_name,
                "namespace": namespace,
                "annotations": {
                    "llm-d.ai/translated-from": "nvidia-dynamo",
                },
            },
        }
        (kustomize_dir / patch_filename).write_text(_dump_yaml(patch_doc),
                                                    encoding="utf-8")
        patches_meta.append({
            "target": {
                "kind": w_kind,
                "name": w_name,
                "labelSelector": f"llm-d.ai/role={role}",
            },
            "path": patch_filename,
        })

    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "namespace": namespace,
        "resources": resources_list,
        "patches": patches_meta,
    }
    (kustomize_dir / "kustomization.yaml").write_text(
        _dump_yaml(kustomization), encoding="utf-8")
    return kustomize_dir


def _kubectl_base_cmd(
    kubeconfig: str | Path | None = None,
    context: str | None = None,
) -> list[str]:
    """Builds the base kubectl command with optional kubeconfig and context flags."""
    cmd = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", str(kubeconfig)])
    if context:
        cmd.extend(["--context", str(context)])
    return cmd


def _is_crd_discovery_warning(stderr: str) -> bool:
    """Returns True if kubectl dry-run failed only because offline CRDs are not registered."""
    lower = stderr.lower()
    return ("ensure crds are installed first" in lower
            or "no matches for kind" in lower or "connection refused" in lower
            or "unable to recognize" in lower)


def validate_bundle_dry_run(
    bundle_dir: Path,
    kustomize_dir: Path,
    namespace: str,
    kubeconfig: str | Path | None = None,
    context: str | None = None,
    rendered_output_path: Path | None = None,
) -> tuple[bool, str]:
    """Validates `kubectl kustomize <kustomize_dir>`, Kustomize `-k` client dry-run, and per-file YAML manifests."""
    base_cmd = _kubectl_base_cmd(kubeconfig, context)
    try:
        kust_proc = subprocess.run(
            base_cmd + ["kustomize", str(kustomize_dir)],
            capture_output=True,
            check=True,
            text=True,
        )
        rendered_docs = [
            d for d in yaml.safe_load_all(kust_proc.stdout)
            if isinstance(d, dict)
        ]
        if not rendered_docs:
            return False, "kubectl kustomize produced no Kubernetes documents"
        if rendered_output_path is not None:
            rendered_output_path.parent.mkdir(parents=True, exist_ok=True)
            rendered_output_path.write_text(kust_proc.stdout, encoding="utf-8")
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        return False, f"kubectl kustomize failed: {stderr}".strip()

    apply_k_cmd = base_cmd + [
        "apply",
        "--dry-run=client",
        "-k",
        str(kustomize_dir),
        "-n",
        namespace,
    ]
    try:
        subprocess.run(apply_k_cmd, capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        if not _is_crd_discovery_warning(exc.stderr or ""):
            return False, (exc.stderr or str(exc)).strip()
    except Exception as exc:
        return False, str(exc)

    for f in sorted(bundle_dir.glob("*.yaml")):
        cmd = base_cmd + [
            "apply",
            "--dry-run=client",
            "-f",
            str(f),
            "-n",
            namespace,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            if not _is_crd_discovery_warning(exc.stderr or ""):
                return False, (exc.stderr or str(exc)).strip()
        except Exception as exc:
            return False, str(exc)

    return True, ""


def translate_dynamo_to_llmd(
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    target_namespace: str = "default",
    target_tpu_type: str = "v6e",
    target_tpu_topology: str = "2x2x1",
    model_id_override: str | None = None,
    store_dir: Path | None = None,
    dry_run: bool = False,
    deployments_json: Path | str | dict[str, Any] | None = None,
    vllm_patch: Path | str | None = None,
    standalone_epp: bool = False,
    envoy_configmap: str | None = None,
    epp_service_account: str | None = None,
    kubeconfig: str | Path | None = None,
    context: str | None = None,
    image: str | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Translates Dynamo manifests into a homogeneous llm-d manifest bundle."""
    if store_dir is not None:
        store_dir = Path(store_dir)
        if manifest_path is None:
            manifest_path = ((store_dir / "manifests") if
                             (store_dir / "manifests").exists() else store_dir)
        if output_dir is None:
            output_dir = store_dir

    if manifest_path is None:
        raise ValueError("Either manifest_path or store_dir must be provided")
    manifest_path = Path(manifest_path)
    if (manifest_path.is_dir() and (manifest_path / "manifests").is_dir()
            and not any(manifest_path.glob("*.yaml"))):
        if store_dir is None:
            store_dir = manifest_path
        manifest_path = manifest_path / "manifests"

    if output_dir is None:
        output_dir = store_dir if store_dir is not None else manifest_path.parent
    output_dir = Path(output_dir)

    if deployments_json is None:
        deployments_json = _discover_sibling_deployments_json((
            store_dir,
            output_dir,
            manifest_path if manifest_path.is_dir() else manifest_path.parent,
        ))

    inspection_report: dict[str, Any] = {}
    for candidate_dir in (store_dir, output_dir, manifest_path):
        if candidate_dir is not None and candidate_dir.is_dir():
            rep_file = candidate_dir / "inspection_report.json"
            if rep_file.is_file():
                try:
                    inspection_report = json.loads(
                        rep_file.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass

    parsed = parse_dynamo_manifests(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / "llmd_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    initial_dep_cfg = load_deployments_json(deployments_json,
                                            model_id=model_id_override)
    frontend_config = extract_frontend_config(parsed["frontend"])
    worker_configs = [extract_worker_config(w) for w in parsed["workers"]]
    fallback_model = (
        model_id_override or initial_dep_cfg.get("model_id") or
        (inspection_report.get("model_id") if inspection_report.get("model_id")
         not in (None, "", "unknown") else None)
        or "meta-llama/Llama-3.1-8B-Instruct")
    if not worker_configs:
        worker_configs = [{
            "name": "worker",
            "role": "homogeneous",
            "model_id": fallback_model,
            "replicas": 2,
            "gpu_count": 4,
            "node_count": 1,
            "max_model_len": None,
            "tensor_parallel_size": None,
            "node_selector": {},
            "args_str": "",
            "env": [],
        }]

    model_id = (model_id_override or initial_dep_cfg.get("model_id")
                or worker_configs[0]["model_id"])
    deployments_config = load_deployments_json(deployments_json,
                                               model_id=model_id)
    effective_envoy_cm = envoy_configmap or (
        str(deployments_config["envoy_configmap"])
        if deployments_config.get("envoy_configmap") else None)
    effective_epp_sa = epp_service_account or (
        str(deployments_config["epp_service_account"])
        if deployments_config.get("epp_service_account") else None)
    effective_standalone_epp = bool(standalone_epp or effective_envoy_cm
                                    or effective_epp_sa)

    patch_info = resolve_vllm_patch(
        model_id=model_id,
        vllm_patch=vllm_patch,
        deployments_config=deployments_config,
    )
    vllm_patch_name = patch_info["patch_name"] if patch_info else None

    topology = "multi-replica-llmd-homogeneous"
    guide_name = "optimized-baseline"
    source_max_len = next(
        (w["max_model_len"] for w in worker_configs if w.get("max_model_len")),
        None,
    )

    generated_files: list[str] = []
    pool_name = "llm-d-pool"

    epp_docs = generate_endpoint_picker_manifests(
        target_namespace,
        target_tpu_type,
        pool_name=pool_name,
        standalone_epp=effective_standalone_epp,
        envoy_configmap=effective_envoy_cm,
        epp_service_account=effective_epp_sa,
    )
    epp_file = bundle_dir / "01-endpointpickerconfig.yaml"
    epp_file.write_text(_dump_yaml_all(epp_docs), encoding="utf-8")
    generated_files.append(epp_file.name)

    pool_selector = {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/guide": guide_name,
        "leaderworkerset.sigs.k8s.io/worker-index": "0",
    }
    inference_pool = generate_inference_pool(target_namespace, pool_name,
                                             pool_selector)
    pool_file = bundle_dir / "02-inferencepool.yaml"
    pool_file.write_text(_dump_yaml(inference_pool), encoding="utf-8")
    generated_files.append(pool_file.name)

    gateways = generate_gateway_and_route(target_namespace, pool_name)
    gw_file = bundle_dir / "00-gateway.yaml"
    gw_file.write_text(_dump_yaml_all(gateways), encoding="utf-8")
    generated_files.append(gw_file.name)

    total_replicas = sum(w["replicas"] for w in worker_configs)
    max_nodes = max((w.get("node_count", 1) for w in worker_configs),
                    default=1)
    effective_workspace = workspace if workspace is not None else output_dir
    resolved_tpu_image, resolved_tpu_image_source = (
        _resolve_tpu_base_image_with_source(
            deployments_config=deployments_config,
            image_override=image,
            workspace=effective_workspace,
        ))
    dep = generate_tpu_serving_deployment(
        namespace=target_namespace,
        name="vllm-tpu-workers",
        model_id=model_id,
        replicas=total_replicas,
        target_tpu_type=target_tpu_type,
        target_tpu_topology=target_tpu_topology,
        node_count=max_nodes,
        max_model_len=source_max_len,
        deployments_json=deployments_config,
        vllm_patch_name=vllm_patch_name,
        image=image,
        workspace=effective_workspace,
    )
    dep_file = bundle_dir / "03-serving-deployment.yaml"
    dep_file.write_text(_dump_yaml(dep), encoding="utf-8")
    generated_files.append(dep_file.name)
    workload_manifests = [dep]
    translated_pools = [{
        "name":
        "vllm-tpu-workers",
        "kind":
        dep["kind"],
        "role":
        "homogeneous",
        "tpu_type":
        target_tpu_type,
        "tpu_topology":
        target_tpu_topology,
        "replicas":
        total_replicas,
        "node_count":
        dep["spec"]["leaderWorkerTemplate"]["size"],
    }]

    vllm_patches_doc: dict[str, Any] | None = None
    if patch_info is not None:
        vllm_patches_doc = generate_vllm_patches_configmap(
            target_namespace, patch_info)
        patches_file = bundle_dir / "04-vllm-patches.yaml"
        patches_file.write_text(_dump_yaml(vllm_patches_doc), encoding="utf-8")
        generated_files.append(patches_file.name)

    bundle_kustomize_dir = render_kustomize_overlay(
        kustomize_dir=bundle_dir / "kustomize",
        namespace=target_namespace,
        workloads=workload_manifests,
        gateway_docs=gateways,
        epp_docs=epp_docs,
        inference_pool_doc=inference_pool,
        target_tpu_type=target_tpu_type,
        pool_name=pool_name,
        vllm_patches_doc=vllm_patches_doc,
        standalone_epp=effective_standalone_epp,
    )
    if store_dir is not None:
        kustomize_dir = render_kustomize_overlay(
            kustomize_dir=store_dir / "kustomize",
            namespace=target_namespace,
            workloads=workload_manifests,
            gateway_docs=gateways,
            epp_docs=epp_docs,
            inference_pool_doc=inference_pool,
            target_tpu_type=target_tpu_type,
            pool_name=pool_name,
            vllm_patches_doc=vllm_patches_doc,
            standalone_epp=effective_standalone_epp,
        )
    else:
        kustomize_dir = bundle_kustomize_dir

    endpoint_plugins = [
        "prefix-cache-affinity-filter",
        "prefix-cache-scorer",
        "token-load-scorer",
        "active-request-scorer",
        "kv-cache-utilization-scorer",
        "max-score-picker",
    ]

    mapping_report: dict[str, Any] = {
        "source_router":
        "dynamo-frontend" if parsed["frontend"] else "direct-service",
        "frontend_config": frontend_config,
        "session_affinity": {
            "enabled": True,
            "headers": ["X-Session-ID", "X-Correlation-ID"],
        },
        "kv_routing": True,
        "source_pools": worker_configs,
        "translated_pools": translated_pools,
        "kustomize_dir": str(kustomize_dir),
        "endpoint_picker_plugins": endpoint_plugins,
        "resolved_tpu_image": resolved_tpu_image,
        "resolved_tpu_image_source": resolved_tpu_image_source,
    }
    if deployments_config:
        mapping_report["deployments_json"] = (
            str(deployments_json) if isinstance(deployments_json,
                                                (Path,
                                                 str)) else deployments_config)
        mapping_report["deployments_config"] = deployments_config
    if patch_info is not None:
        mapping_report["vllm_patch"] = patch_info["patch_name"]
        mapping_report["vllm_patch_provenance"] = patch_info.get(
            "provenance", "custom")
    if parsed.get("sla_targets"):
        mapping_report["sla_targets"] = parsed["sla_targets"]
    if inspection_report:
        mapping_report["source_inspection_topology"] = inspection_report.get(
            "topology", "unknown")

    report_file = output_dir / "mapping_report.json"
    report_file.write_text(json.dumps(mapping_report, indent=2),
                           encoding="utf-8")

    result: dict[str, Any] = {
        "status": "ok",
        "bundle_dir": str(bundle_dir),
        "kustomize_dir": str(kustomize_dir),
        "manifest_files": generated_files,
        "topology": topology,
        "model_id": model_id,
        "target_tpu_type": target_tpu_type,
        "target_tpu_topology": target_tpu_topology,
        "mapping_report": mapping_report,
        "log_dir": str(output_dir),
    }
    if deployments_config and deployments_json is not None:
        result["deployments_json"] = (str(deployments_json) if isinstance(
            deployments_json, (Path, str)) else "inline")
    if patch_info is not None:
        result["vllm_patch"] = patch_info["patch_name"]
    if store_dir is not None:
        result["store_dir"] = str(store_dir)

    if dry_run:
        rendered_output_path = output_dir / "kustomize_rendered.yaml"
        dry_run_success, dry_run_error = validate_bundle_dry_run(
            bundle_dir=bundle_dir,
            kustomize_dir=kustomize_dir,
            namespace=target_namespace,
            kubeconfig=kubeconfig,
            context=context,
            rendered_output_path=rendered_output_path,
        )
        result["dry_run_verified"] = dry_run_success
        if rendered_output_path.is_file():
            result["kustomize_rendered"] = str(rendered_output_path)
        if not dry_run_success:
            result["dry_run_error"] = dry_run_error

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        "Translate NVIDIA Dynamo manifests to homogeneous GKE TPU llm-d stack."
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=None,
        help="Path to Dynamo manifest file or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to emit llm-d manifest bundle",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=None,
        help=
        "Cluster runtime store directory containing manifests/ and inspection_report.json",
    )
    parser.add_argument("--namespace",
                        default="default",
                        help="Target Kubernetes namespace")
    parser.add_argument(
        "--tpu-type",
        default="v6e",
        choices=["v4", "v5e", "v5p", "v6e", "v7x", "tpu7x"],
        help="Target TPU type",
    )
    parser.add_argument(
        "--tpu-topology",
        default="2x2x1",
        help="Target TPU topology (e.g. 2x2x1, 2x2x2)",
    )
    parser.add_argument("--model-id",
                        default=None,
                        help="Hugging Face model ID override")
    parser.add_argument(
        "--deployments-json",
        type=Path,
        default=None,
        help=
        "Path to single-VM deployments.json to carry over tuned vLLM flags and env vars",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional TPU vLLM container image override",
    )
    parser.add_argument(
        "--vllm-patch",
        type=Path,
        default=None,
        help=
        "Path to a vllm_torchtpu .patch file to inject via ConfigMap and apply at container startup",
    )
    parser.add_argument(
        "--standalone-epp",
        action="store_true",
        help="Generate standalone EPP + Envoy proxy router deployment",
    )
    parser.add_argument(
        "--envoy-configmap",
        default=None,
        help="Optional Envoy ConfigMap name for standalone EPP router",
    )
    parser.add_argument(
        "--epp-service-account",
        default=None,
        help="Optional ServiceAccount name for standalone EPP router",
    )
    parser.add_argument(
        "--kubeconfig",
        default=None,
        help="Optional path to kubeconfig file for kubectl commands",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional Kubernetes context name for kubectl commands",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=
        "Validate generated manifests with kubectl dry-run and write kustomize_rendered.yaml",
    )

    args = parser.parse_args()
    if args.manifests is None and args.store_dir is None:
        parser.error("Either --manifests or --store-dir is required")

    result = translate_dynamo_to_llmd(
        manifest_path=args.manifests,
        output_dir=args.output_dir,
        target_namespace=args.namespace,
        target_tpu_type=args.tpu_type,
        target_tpu_topology=args.tpu_topology,
        model_id_override=args.model_id,
        store_dir=args.store_dir,
        dry_run=args.dry_run,
        deployments_json=args.deployments_json,
        vllm_patch=args.vllm_patch,
        standalone_epp=args.standalone_epp,
        envoy_configmap=args.envoy_configmap,
        epp_service_account=args.epp_service_account,
        kubeconfig=args.kubeconfig,
        context=args.context,
        image=args.image,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
