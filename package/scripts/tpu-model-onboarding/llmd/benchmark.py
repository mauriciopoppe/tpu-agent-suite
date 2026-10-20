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
"""Benchmark generation and execution for homogeneous distributed serving on GKE Cloud TPU.

Supports two benchmark harnesses across llm-d and Dynamo serving stacks:

1. ``inference-perf`` - staged constant-rate sweep with native ``shared_prefix``
   synthesis for prefix-cache routing validation and ``--execute`` cluster runner.
2. ``aiperf`` (alias ``aiperf-agentx``) - multi-turn agentic micro-benchmark
   characterization with session-affinity correlation headers and per-worker
   Prometheus ``--server-metrics`` scraping.

Exposes two primary entry points used by ``dynamo_translator.py`` and CLI callers:

- ``build_benchmark()`` - produces a harness benchmark configuration dictionary
  tied to the same model, namespace, topology, and SLA targets as the serving stack.
- ``build_benchmark_deployment()`` - produces a multi-document Kubernetes YAML
  (``ConfigMap`` + ``batch/v1`` ``Job``) ready for ``kubectl-apply``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import parse_bench_results  # noqa: E402

_INFERENCE_PERF_IMAGE = "quay.io/inference-perf/inference-perf:latest"
_AIPERF_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.1"

_AIPERF_SESSION_AFFINITY_ENV: dict[str, str] = {
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "1",
    "AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID": "1",
    "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID": "1",
    "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "3600",
    "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "3600",
    "AIPERF_HTTP_CONNECTION_LIMIT": "2500",
}

_SUPPORTED_HARNESSES = (
    "inference-perf",
    "aiperf",
    "aiperf-agentx",
)
_SUPPORTED_STACKS = ("llm-d", "dynamo")

_BENCHMARK_BASE_RATES: dict[str, dict[str, float | int]] = {
    "balanced-conversational": {
        "per_replica_qps": 5.0,
        "base_isl": 1000,
        "base_osl": 500,
    },
    "high-prefix-share": {
        "per_replica_qps": 1.0,
        "base_isl": 4000,
        "base_osl": 300,
    },
    "latency-tight": {
        "per_replica_qps": 3.0,
        "base_isl": 500,
        "base_osl": 200,
    },
}


@dataclass
class BenchmarkTopology:
    """Standalone homogeneous topology specification for distributed serving benchmarks."""

    mode: str = "agg"
    replicas: int | None = 2
    tp: int | None = None
    accelerator_type: str | None = None
    tpu_topology: str | None = None
    chips_per_replica: int = 8

    def total_replicas(self) -> int:
        return self.replicas or 1


Topology = BenchmarkTopology


@dataclass
class BenchmarkWorkload:
    """Standalone workload specification for distributed serving benchmarks."""

    isl: int | None = 4000
    osl: int | None = 500
    prefix_len: int | None = None
    concurrency: int | None = None
    model_context_length: int | None = None


@dataclass
class BenchmarkSlo:
    """Standalone SLO target specification for distributed serving benchmarks."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    request_latency_ms: float | None = None


@dataclass
class BenchmarkContext:
    """Standalone deployment context for distributed serving benchmarks."""

    namespace: str = "default"
    release_name: str = "llm-d"
    hf_secret_name: str = "llm-d-hf-token"
    bench_tokenizer_override: str | None = None


@dataclass
class BenchmarkInput:
    """Standalone input container for distributed serving benchmarks."""

    model: str
    accelerator: str = "tpu-v6e"
    topology: BenchmarkTopology = field(default_factory=BenchmarkTopology)
    workload: BenchmarkWorkload = field(default_factory=BenchmarkWorkload)
    slo: BenchmarkSlo = field(default_factory=BenchmarkSlo)
    context: BenchmarkContext = field(default_factory=BenchmarkContext)


Input = BenchmarkInput


def _estimate_cluster_qps(inp: "Input | BenchmarkInput",
                          workload_class: str) -> float:
    """Rough cluster-wide max QPS estimate."""
    base = _BENCHMARK_BASE_RATES.get(
        workload_class, _BENCHMARK_BASE_RATES["balanced-conversational"])
    per_replica = float(base["per_replica_qps"])

    if inp.workload.isl and inp.workload.osl:
        actual_token_cost = inp.workload.isl * inp.workload.osl
        base_token_cost = int(base["base_isl"]) * int(base["base_osl"])
        adjustment = max(0.25, min(4.0, base_token_cost / actual_token_cost))
        per_replica *= adjustment

    return inp.topology.total_replicas() * per_replica


def _sla_validation_rates(cluster_qps: float) -> list[int]:
    """Three rates along the saturation curve: 25% / 50% / 85% of estimated max."""
    rate_low = max(1, int(round(cluster_qps * 0.25)))
    rate_mid = max(rate_low + 1, int(round(cluster_qps * 0.50)))
    rate_near = max(rate_mid + 1, int(round(cluster_qps * 0.85)))
    return [rate_low, rate_mid, rate_near]


def resolve_serving_endpoint(
    stack: str,
    release_name: str,
    *,
    mode: str = "gateway",
) -> str:
    """Resolve the in-cluster HTTP base URL for a Dynamo or llm-d stack."""
    if stack not in _SUPPORTED_STACKS:
        raise ValueError(
            f"stack must be one of {_SUPPORTED_STACKS}; got {stack!r}")
    if stack == "dynamo":
        return f"http://{release_name}-frontend:8000"
    if mode == "standalone":
        return f"http://{release_name}-epp:9002"
    return f"http://{release_name}-inference-gateway:80"


def format_server_metrics_urls(
    pod_records: list[dict[str, Any]],
    *,
    stack: str = "llm-d",
) -> list[str]:
    """Format per-worker Prometheus ``/metrics`` URLs from discovered pod records."""
    if stack not in _SUPPORTED_STACKS:
        raise ValueError(
            f"stack must be one of {_SUPPORTED_STACKS}; got {stack!r}")
    urls: list[str] = []
    for pod in pod_records:
        ip = pod.get("ip")
        if not ip:
            continue
        explicit_port = pod.get("port")
        if explicit_port is not None:
            port = int(explicit_port)
        elif stack == "dynamo":
            port = 9090
        else:
            port = 8000
        urls.append(f"http://{ip}:{port}/metrics")
    return urls


def build_worker_metrics_discovery_command(
    stack: str,
    release_name: str,
    namespace: str,
) -> str:
    """Build a shell command that resolves worker ``/metrics`` URLs for ``--server-metrics``."""
    if stack not in _SUPPORTED_STACKS:
        raise ValueError(
            f"stack must be one of {_SUPPORTED_STACKS}; got {stack!r}")
    if stack == "dynamo":
        selector = (f"nvidia.com/dynamo-graph-deployment-name={release_name},"
                    "nvidia.com/dynamo-component-type=worker")
        return (
            f"kubectl get pods -n {namespace} -l {selector} "
            "--field-selector=status.phase=Running "
            "-o jsonpath='{range .items[*]}{\"http://\"}{.status.podIP}{\":9090/metrics \"}{end}'"
        )

    selector = "llm-d.ai/inferenceServing=true"
    return (
        f"kubectl get pods -n {namespace} -l {selector} "
        "--field-selector=status.phase=Running "
        "-o jsonpath='{range .items[*]}{\"http://\"}{.status.podIP}{\":8000/metrics \"}{end}'"
    )


def build_cache_reset_commands(
    stack: str,
    release_name: str,
    namespace: str,
    *,
    backend: str = "vllm",
    method: str = "http",
    target_url: str | None = None,
) -> list[str]:
    """Build commands to flush prefix/KV caches between concurrency sweep points."""
    if stack not in _SUPPORTED_STACKS:
        raise ValueError(
            f"stack must be one of {_SUPPORTED_STACKS}; got {stack!r}")
    if backend not in ("vllm", "sglang"):
        raise ValueError(
            f"backend must be 'vllm' or 'sglang'; got {backend!r}")
    if method not in ("http", "pod-restart"):
        raise ValueError(
            f"method must be 'http' or 'pod-restart'; got {method!r}")

    endpoint = target_url or resolve_serving_endpoint(stack, release_name)
    reset_path = ("/reset_prefix_cache"
                  if backend == "vllm" else "/flush_cache")

    if stack == "dynamo":
        selector = (f"nvidia.com/dynamo-graph-deployment-name={release_name},"
                    "nvidia.com/dynamo-component-type=worker")
    else:
        selector = "llm-d.ai/inferenceServing=true"

    if method == "pod-restart":
        k_verb = " ".join(("kubectl", "delete", "pods"))
        return [
            f"{k_verb} -n {namespace} -l {selector} --wait=true",
            (f"until curl -sf {endpoint}/v1/models >/dev/null; do "
             "sleep 5; done"),
        ]

    return [(f"for ip in $(kubectl get pods -n {namespace} -l {selector} "
             "-o jsonpath='{.items[*].status.podIP}'); do "
             f"curl -sf -X POST http://${{ip}}:8000{reset_path}; done")]


def build_benchmark(
    inp: "Input | BenchmarkInput",
    workload_class: str,
    *,
    harness: str = "inference-perf",
    stack: str = "llm-d",
    smoke: bool = False,
    concurrency: int | None = None,
    server_metrics_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a benchmark config for the chosen harness.

    Supported harnesses:
    - ``inference-perf``
    - ``aiperf`` (alias ``aiperf-agentx``)
    """
    if harness not in _SUPPORTED_HARNESSES:
        raise ValueError(
            f"harness must be one of {_SUPPORTED_HARNESSES}; got {harness!r}")
    if stack not in _SUPPORTED_STACKS:
        raise ValueError(
            f"stack must be one of {_SUPPORTED_STACKS}; got {stack!r}")

    if harness == "inference-perf":
        return _build_inference_perf(inp, workload_class)
    return _build_aiperf(
        inp,
        workload_class,
        harness_name=harness,
        stack=stack,
        smoke=smoke,
        concurrency=concurrency,
        server_metrics_urls=server_metrics_urls,
    )


def _build_inference_perf(inp: "Input | BenchmarkInput",
                          workload_class: str) -> dict[str, Any]:
    """inference-perf config using the flat native schema."""
    bench_warnings: list[str] = []

    has_slo = bool(inp.slo.ttft_ms or inp.slo.tpot_ms
                   or inp.slo.request_latency_ms)

    stages: list[dict[str, int]] = [{"rate": 1, "duration": 30}]
    rate_derivation: str | None = None
    if has_slo:
        cluster_qps = _estimate_cluster_qps(inp, workload_class)
        rates = _sla_validation_rates(cluster_qps)
        stages.extend({"rate": r, "duration": 120} for r in rates)
        rate_derivation = (
            f"workload class {workload_class} baseline "
            f"{_BENCHMARK_BASE_RATES.get(workload_class, {}).get('per_replica_qps', 'n/a')} req/s/replica "
            f"x {inp.topology.total_replicas()} replicas, adjusted for ISL/OSL -> "
            f"~{cluster_qps:.1f} req/s estimated max; rates are 25%/50%/85% of that "
            "(prepended with 30s warmup at rate=1)")

    tokenizer_ref = inp.context.bench_tokenizer_override or inp.model

    config: dict[str, Any] = {
        "load": {
            "type": "constant",
            "stages": stages
        },
        "api": {
            "type": "completion",
            "streaming": True
        },
        "server": {
            "type": "vllm",
            "model_name": inp.model,
            "base_url": "${IP}",
            "ignore_eos": True,
        },
        "tokenizer": {
            "pretrained_model_name_or_path": tokenizer_ref
        },
        "data": _inference_perf_data_block(inp, workload_class),
        "report": {
            "request_lifecycle": {
                "summary": True,
                "per_stage": True,
                "per_request": True,
            },
        },
        "storage": {
            "local_storage": {
                "path": "/results"
            }
        },
    }

    if has_slo:
        sla_targets = _extract_sla_targets(inp)
        if sla_targets:
            config["_sla_targets"] = sla_targets
    if rate_derivation:
        config["_rate_derivation"] = rate_derivation

    config["_hf_token_secret"] = inp.context.hf_secret_name

    return {
        "config": config,
        "harness": "inference-perf",
        "warnings": bench_warnings,
    }


def _inference_perf_data_block(inp: "Input | BenchmarkInput",
                               workload_class: str) -> dict[str, Any]:
    """inference-perf data block - random Gaussian by default, shared_prefix for high-prefix-share with prefix_len."""
    if workload_class == "high-prefix-share" and inp.workload.prefix_len:
        prompt_tokens = inp.workload.isl if inp.workload.isl else 4000
        output_tokens = inp.workload.osl if inp.workload.osl else 300
        question_len = max(64, prompt_tokens - inp.workload.prefix_len)
        return {
            "type": "shared_prefix",
            "shared_prefix": {
                "num_groups": 32,
                "num_prompts_per_group": 32,
                "system_prompt_len": inp.workload.prefix_len,
                "question_len": question_len,
                "output_len": output_tokens,
            },
        }

    prompt_mean = inp.workload.isl if inp.workload.isl else 50
    output_mean = inp.workload.osl if inp.workload.osl else 50
    return {
        "type": "random",
        "input_distribution": {
            "min": max(10, int(prompt_mean * 0.5)),
            "max": int(prompt_mean * 1.5),
            "mean": prompt_mean,
            "std_dev": max(1, int(prompt_mean * 0.1)),
            "total_count": 100,
        },
        "output_distribution": {
            "min": max(10, int(output_mean * 0.5)),
            "max": int(output_mean * 1.5),
            "mean": output_mean,
            "std_dev": max(1, int(output_mean * 0.1)),
            "total_count": 100,
        },
    }


def _resolve_concurrency_sweep(
    inp: "Input | BenchmarkInput",
    *,
    explicit_concurrency: int | None = None,
) -> list[int]:
    """Determine concurrency levels for aiperf runs."""
    if explicit_concurrency is not None:
        return [max(1, int(explicit_concurrency))]
    workload_conc = getattr(inp.workload, "concurrency", None)
    if workload_conc:
        base_c = max(1, int(workload_conc))
        return sorted(
            {max(1, base_c // 2), base_c,
             max(base_c + 1, int(base_c * 1.5))})
    replicas = max(1, inp.topology.total_replicas())
    return [4 * replicas, 8 * replicas, 12 * replicas]


def _extract_sla_targets(inp: "Input | BenchmarkInput") -> dict[str, float]:
    """Extract SLA targets dict from input SLO specification."""
    sla_targets: dict[str, float] = {}
    if inp.slo.ttft_ms is not None:
        sla_targets["ttft_ms_p95_target"] = inp.slo.ttft_ms
    if inp.slo.tpot_ms is not None:
        sla_targets["tpot_ms_p95_target"] = inp.slo.tpot_ms
    if inp.slo.request_latency_ms is not None:
        sla_targets["request_latency_ms_p95_target"] = (
            inp.slo.request_latency_ms)
    return sla_targets


def _build_aiperf(
    inp: "Input | BenchmarkInput",
    workload_class: str,
    *,
    harness_name: str = "aiperf",
    stack: str = "llm-d",
    smoke: bool = False,
    concurrency: int | None = None,
    server_metrics_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Build an aiperf micro-benchmark characterization config."""
    bench_warnings: list[str] = []
    tokenizer_ref = inp.context.bench_tokenizer_override or inp.model

    concurrencies = ([concurrency or 2]
                     if smoke else _resolve_concurrency_sweep(
                         inp, explicit_concurrency=concurrency))
    primary_concurrency = concurrencies[0]
    duration_s = 120 if smoke else 600

    args: list[str] = [
        "aiperf",
        "profile",
        "--scenario",
        "inferencex-agentx-mvp",
        "--model",
        inp.model,
        "--tokenizer",
        tokenizer_ref,
        "--tokenizer-trust-remote-code",
        "--url",
        "${IP}",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--use-server-token-count",
        "--extra-inputs",
        "ignore_eos:true",
        "--random-seed",
        "42",
        "--concurrency",
        str(primary_concurrency),
        "--benchmark-duration",
        str(duration_s),
        "--artifact-dir",
        "/results/agentx",
    ]
    if server_metrics_urls:
        args.extend(["--server-metrics", *server_metrics_urls])

    discovery_cmd = build_worker_metrics_discovery_command(
        stack,
        inp.context.release_name,
        "${NAMESPACE}",
    )
    cache_reset_cmds = build_cache_reset_commands(
        stack,
        inp.context.release_name,
        "${NAMESPACE}",
        backend="vllm",
        method="http",
        target_url="${IP}",
    )

    config: dict[str, Any] = {
        "schemaVersion": "2.0",
        "harness": {
            "name": harness_name,
            "image": _AIPERF_IMAGE,
            "stack": stack,
            "smoke": smoke,
            "results_pvc": "${BENCHMARK_PVC}",
            "namespace": "${NAMESPACE}",
            "release_name": inp.context.release_name,
        },
        "endpoint": {
            "model": inp.model,
            "tokenizer": tokenizer_ref,
            "base_url": "${IP}",
            "endpoint_type": "chat",
            "endpoint_path": "/v1/chat/completions",
        },
        "env": dict(_AIPERF_SESSION_AFFINITY_ENV),
        "scenario": {
            "name": "inferencex-agentx-mvp",
            "concurrencies": concurrencies,
            "benchmark_duration_seconds": duration_s,
            "random_seed": 42,
        },
        "server_metrics": {
            "urls": list(server_metrics_urls or []),
            "discovery_command": discovery_cmd,
            "metrics_port": 9090 if stack == "dynamo" else 8000,
        },
        "cache_reset": {
            "commands": cache_reset_cmds,
        },
        "profiles": {
            f"agentx-c{c}": {
                "concurrency":
                c,
                "args": [(str(c) if tok == str(primary_concurrency)
                          and prev == "--concurrency" else tok)
                         for prev, tok in zip([""] + args[:-1], args)],
            }
            for c in concurrencies
        },
        "_hf_token_secret": inp.context.hf_secret_name,
    }
    sla_targets = _extract_sla_targets(inp)
    if sla_targets:
        config["_sla_targets"] = sla_targets

    return {
        "config": config,
        "harness": harness_name,
        "warnings": bench_warnings,
    }


def _substitute(value: Any, target_url: str, namespace: str,
                pvc_name: str | None) -> Any:
    """Recursively walk the benchmark config dict and replace placeholders."""
    if isinstance(value, str):
        out = value
        out = out.replace("${IP}", target_url)
        out = out.replace("${NAMESPACE}", namespace)
        out = out.replace("${BENCHMARK_PVC}", pvc_name if pvc_name else "")
        return out
    if isinstance(value, dict):
        return {
            k: _substitute(v, target_url, namespace, pvc_name)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_substitute(v, target_url, namespace, pvc_name) for v in value]
    return value


def _job_name_suffix(config: dict[str, Any]) -> str:
    """Deterministic 8-char hash suffix from the config for idempotent kubectl-apply."""
    canonical = json.dumps(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def _strip_sentinels(value: Any) -> Any:
    """Recursively drop dict keys starting with ``_``."""
    if isinstance(value, dict):
        return {
            k: _strip_sentinels(v)
            for k, v in value.items() if not k.startswith("_")
        }
    if isinstance(value, list):
        return [_strip_sentinels(v) for v in value]
    return value


def _build_aiperf_runner_cmd(scrubbed: dict[str, Any],
                             harness_name: str) -> str:
    """Construct the shell command executed inside the aiperf Job container."""
    profiles = scrubbed.get("profiles") or {}
    if not profiles:
        raise ValueError(f"missing 'profiles' block in {harness_name} config")
    first_profile = next(iter(profiles.values()))
    profile_args = first_profile.get("args") or []
    profile_cmd = " ".join(str(a) for a in profile_args)
    bootstrap_cmd = (
        'export PATH="/tmp/aiperf-venv/bin:$HOME/.local/bin:/opt/uv/bin:$PATH" && '
        "(command -v aiperf >/dev/null 2>&1 || "
        "(uv venv --system-site-packages /tmp/aiperf-venv >/dev/null 2>&1 && "
        "uv pip install --python /tmp/aiperf-venv/bin/python -q aiperf==0.13.0) || "
        "uv pip install --system -q aiperf==0.13.0 || "
        "pip install --user -q aiperf==0.13.0)")
    return f"{bootstrap_cmd} && {profile_cmd}"


def build_benchmark_deployment(
    benchmark_config: dict[str, Any],
    target_url: str,
    namespace: str,
    pvc_name: str | None = None,
    job_name_suffix: str | None = None,
) -> str:
    """Build a complete multi-document K8s YAML for one-shot kubectl-apply."""
    substituted = _substitute(benchmark_config, target_url, namespace,
                              pvc_name)
    scrubbed = _strip_sentinels(substituted)
    suffix = job_name_suffix or _job_name_suffix(scrubbed)
    cm_name = f"autoconfig-bench-config-{suffix}"
    job_name = f"autoconfig-bench-{suffix}"

    if "harness" in benchmark_config and isinstance(
            benchmark_config["harness"], dict):
        harness_name = benchmark_config["harness"].get("name", "aiperf")
    elif "load" in benchmark_config and isinstance(
            benchmark_config.get("load"), dict):
        harness_name = "inference-perf"
    else:
        raise ValueError(
            "could not determine harness from config; expected either "
            "config.harness.name (aiperf) or top-level "
            "config.load.stages (inference-perf)")
    if harness_name not in _SUPPORTED_HARNESSES:
        raise ValueError(
            f"unsupported harness {harness_name!r} in config; expected one of {_SUPPORTED_HARNESSES}"
        )

    harness_meta = (scrubbed.get("harness") if isinstance(
        scrubbed.get("harness"), dict) else {})

    if harness_name == "inference-perf":
        image = _INFERENCE_PERF_IMAGE
        config_mount_path = "/etc/config"
        config_filename = "config.yml"
        results_mount_path = "/results"
        runner_cmd = (
            f"inference-perf --config_file {config_mount_path}/{config_filename} "
            "--log-level INFO")
    else:
        image = harness_meta.get("image") or _AIPERF_IMAGE
        config_mount_path = "/etc/bench-config"
        config_filename = "perf.yaml"
        results_mount_path = "/results"
        runner_cmd = _build_aiperf_runner_cmd(scrubbed, harness_name)

    wrapped_cmd = (
        f"{runner_cmd}; rc=$?; "
        f"echo '---BENCHMARK RESULTS (json files in {results_mount_path})---'; "
        f"find {results_mount_path} -type f \\( -name '*.json' -o -name '*.yaml' \\) "
        f"! -name 'inputs.json' ! -name 'per_request_lifecycle_metrics.json' "
        f"-print -exec cat {{}} \\; 2>/dev/null || true; "
        f"exit $rc")

    hf_secret_name = benchmark_config.get("_hf_token_secret")

    docs: list[dict[str, Any]] = []

    docs.append({
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": cm_name,
            "namespace": namespace,
        },
        "data": {
            config_filename: yaml.safe_dump(scrubbed, sort_keys=False),
        },
    })

    if pvc_name:
        results_volume: dict[str, Any] = {
            "name": "results",
            "persistentVolumeClaim": {
                "claimName": pvc_name
            },
        }
    else:
        results_volume = {
            "name": "results",
            "emptyDir": {
                "sizeLimit": "5Gi"
            },
        }

    env_list: list[dict[str, Any]] = []
    if harness_name == "inference-perf":
        env_list.append({"name": "RAYON_NUM_THREADS", "value": "4"})
    elif harness_name in ("aiperf", "aiperf-agentx"):
        for k, v in (scrubbed.get("env") or {}).items():
            env_list.append({"name": str(k), "value": str(v)})
    if hf_secret_name:
        env_list.append({
            "name": "HF_TOKEN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": hf_secret_name,
                    "key": "HF_TOKEN",
                    "optional": True,
                },
            },
        })

    volume_mounts: list[dict[str, Any]] = [
        {
            "name": "config",
            "mountPath": config_mount_path
        },
        {
            "name": "results",
            "mountPath": results_mount_path
        },
    ]
    volumes: list[dict[str, Any]] = [
        {
            "name": "config",
            "configMap": {
                "name": cm_name
            }
        },
        results_volume,
    ]
    if harness_name in ("aiperf", "aiperf-agentx"):
        volume_mounts.append({
            "name": "model-cache",
            "mountPath": "/model-cache",
        })
        volumes.append({
            "name": "model-cache",
            "emptyDir": {
                "sizeLimit": "2Gi"
            },
        })

    container: dict[str, Any] = {
        "name": "bench",
        "image": image,
        "command": ["sh", "-c"],
        "args": [wrapped_cmd],
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "8Gi",
            },
            "limits": {
                "cpu": "4",
                "memory": "16Gi",
            },
        },
        "volumeMounts": volume_mounts,
    }
    if env_list:
        container["env"] = env_list

    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": volumes,
    }

    job_labels: dict[str, str] = {
        "app.kubernetes.io/component": "autoconfig-benchmark",
    }
    template_labels: dict[str, str] = {
        "app.kubernetes.io/component": "autoconfig-benchmark",
        "job-name": job_name,
    }

    job_spec: dict[str, Any] = {
        "backoffLimit": 0,
        "template": {
            "metadata": {
                "labels": template_labels,
            },
            "spec": pod_spec,
        },
    }

    docs.append({
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": job_labels,
        },
        "spec": job_spec,
    })

    return "---\n".join(yaml.safe_dump(d, sort_keys=False) for d in docs)


_COMPLETIONS_RE = re.compile(r"POST\s+/v1/completions\b")
_CHAT_COMPLETIONS_RE = re.compile(r"POST\s+/v1/chat/completions\b")
_DYNAMO_GENERATE_REQ_RE = re.compile(
    r'^dynamo_component_requests_total\{[^}]*dynamo_endpoint="generate"[^}]*\}\s+([0-9]+(?:\.[0-9]+)?)',
    re.MULTILINE,
)
_VLLM_REQ_SUCCESS_RE = re.compile(
    r"^vllm:request_success_total(?:\{[^}]*\})?\s+([0-9]+(?:\.[0-9]+)?)",
    re.MULTILINE,
)


def _kubectl_cmd(
    verb: str,
    *args: str,
    kubeconfig: str | None = None,
    context: str | None = None,
) -> list[str]:
    """Construct a kubectl command argv list with optional kubeconfig and context."""
    cmd: list[str] = ["kubectl"]
    if kubeconfig:
        cmd.extend(["--kubeconfig", kubeconfig])
    if context:
        cmd.extend(["--context", context])
    cmd.append(verb)
    cmd.extend(args)
    return cmd


def _extract_job_name_from_manifest(manifest: str) -> str:
    """Return the batch/v1 Job metadata.name from a multi-document YAML manifest."""
    for doc in yaml.safe_load_all(manifest):
        if isinstance(doc, dict) and doc.get("kind") == "Job":
            meta = doc.get("metadata") or {}
            name = meta.get("name")
            if name:
                return str(name)
    return "autoconfig-bench"


def compute_traffic_distribution(
    worker_logs: dict[str, str],
    *,
    worker_metrics: dict[str, str] | None = None,
    namespace: str = "default",
    stack: str = "llm-d",
    release_name: str = "llm-d",
) -> dict[str, Any]:
    """Count per-worker request distribution across worker pod logs or Prometheus metrics."""
    pods_summary: dict[str, dict[str, Any]] = {}
    total_completions = 0
    total_chat_completions = 0
    metrics_by_pod = worker_metrics or {}

    for pod_name, log_text in sorted(worker_logs.items()):
        text = log_text or ""
        metrics_text = metrics_by_pod.get(pod_name) or ""
        combined_metrics_text = f"{metrics_text}\n{text}"
        comp_count = len(_COMPLETIONS_RE.findall(text))
        chat_count = len(_CHAT_COMPLETIONS_RE.findall(text))
        dynamo_generate_count = 0
        if comp_count + chat_count == 0:
            dyn_matches = _DYNAMO_GENERATE_REQ_RE.findall(
                combined_metrics_text)
            if dyn_matches:
                dynamo_generate_count = int(
                    round(sum(float(m) for m in dyn_matches)))
            else:
                vllm_matches = _VLLM_REQ_SUCCESS_RE.findall(
                    combined_metrics_text)
                if vllm_matches:
                    dynamo_generate_count = int(
                        round(sum(float(m) for m in vllm_matches)))
            if dynamo_generate_count > 0:
                chat_count = dynamo_generate_count
        pod_total = comp_count + chat_count
        total_completions += comp_count
        total_chat_completions += chat_count
        pod_entry: dict[str, Any] = {
            "post_v1_completions": comp_count,
            "post_v1_chat_completions": chat_count,
            "total_requests": pod_total,
            "traffic_share_pct": 0.0,
        }
        if dynamo_generate_count > 0:
            pod_entry["dynamo_generate_requests"] = dynamo_generate_count
        pods_summary[pod_name] = pod_entry

    total_requests = total_completions + total_chat_completions
    replica_split_pct: dict[str, float] = {}
    for pod_name, entry in pods_summary.items():
        if total_requests > 0:
            pct = round((entry["total_requests"] / total_requests) * 100.0, 2)
        else:
            pct = 0.0
        entry["traffic_share_pct"] = pct
        replica_split_pct[pod_name] = pct

    return {
        "namespace": namespace,
        "stack": stack,
        "release_name": release_name,
        "total_requests": total_requests,
        "total_completions": total_completions,
        "total_chat_completions": total_chat_completions,
        "pods": pods_summary,
        "replica_split_pct": replica_split_pct,
    }


def _parse_pod_names_output(raw_output: str) -> list[str]:
    """Extract pod names from jsonpath or JSON output."""
    text = (raw_output or "").strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                names: list[str] = []
                for item in data["items"]:
                    if isinstance(item, dict):
                        name = (item.get("metadata") or {}).get("name")
                        if name:
                            names.append(str(name))
                return names
        except json.JSONDecodeError:
            pass
    return [tok.strip("'\"") for tok in text.split() if tok.strip("'\"")]


def capture_cluster_evidence(
    evidence_dir: str | Path,
    *,
    namespace: str = "default",
    stack: str = "llm-d",
    release_name: str = "llm-d",
    kubeconfig: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Capture post-benchmark cluster state, worker pod logs, traffic split, and EPP router logs."""
    out_dir = Path(evidence_dir)
    worker_logs_dir = out_dir / "worker_pod_logs"
    worker_logs_dir.mkdir(parents=True, exist_ok=True)

    state_res = subprocess.run(
        _kubectl_cmd(
            "get",
            "lws,deploy,pods,svc,workload,provisioningrequest,job",
            "-n",
            namespace,
            "-o",
            "wide",
            kubeconfig=kubeconfig,
            context=context,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if state_res.returncode != 0 and not (state_res.stdout or "").strip():
        state_res = subprocess.run(
            _kubectl_cmd(
                "get",
                "deploy,pods,svc,job",
                "-n",
                namespace,
                "-o",
                "wide",
                kubeconfig=kubeconfig,
                context=context,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
    cluster_state_text = state_res.stdout or state_res.stderr or ""
    (out_dir / "target_cluster_state.txt").write_text(
        cluster_state_text,
        encoding="utf-8",
    )

    if stack == "dynamo":
        worker_selector = (
            f"nvidia.com/dynamo-graph-deployment-name={release_name},"
            "nvidia.com/dynamo-component-type=worker")
    else:
        worker_selector = "llm-d.ai/inferenceServing=true"

    pods_res = subprocess.run(
        _kubectl_cmd(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            worker_selector,
            "-o",
            "jsonpath={.items[*].metadata.name}",
            kubeconfig=kubeconfig,
            context=context,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    pod_names = _parse_pod_names_output(pods_res.stdout)

    worker_logs: dict[str, str] = {}
    worker_metrics: dict[str, str] = {}
    metrics_port = 9090 if stack == "dynamo" else 8000
    for pod_name in pod_names:
        pod_log_res = subprocess.run(
            _kubectl_cmd(
                "logs",
                pod_name,
                "-n",
                namespace,
                "--all-containers=true",
                kubeconfig=kubeconfig,
                context=context,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        pod_log_text = pod_log_res.stdout or ""
        worker_logs[pod_name] = pod_log_text
        (worker_logs_dir / f"{pod_name}.log").write_text(
            pod_log_text,
            encoding="utf-8",
        )
        has_http_logs = bool(
            _COMPLETIONS_RE.search(pod_log_text)
            or _CHAT_COMPLETIONS_RE.search(pod_log_text))
        if stack == "dynamo" or not has_http_logs:
            metrics_res = subprocess.run(
                _kubectl_cmd(
                    "exec",
                    pod_name,
                    "-n",
                    namespace,
                    "--",
                    "python3",
                    "-c",
                    ("import urllib.request; "
                     f"print(urllib.request.urlopen('http://127.0.0.1:{metrics_port}/metrics', timeout=5).read().decode('utf-8', errors='replace'), end='')"
                     ),
                    kubeconfig=kubeconfig,
                    context=context,
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            metrics_text = metrics_res.stdout or ""
            if metrics_text.strip():
                worker_metrics[pod_name] = metrics_text
                (worker_logs_dir / f"{pod_name}.metrics.txt").write_text(
                    metrics_text,
                    encoding="utf-8",
                )

    traffic_dist = compute_traffic_distribution(
        worker_logs,
        worker_metrics=worker_metrics,
        namespace=namespace,
        stack=stack,
        release_name=release_name,
    )
    (out_dir / "traffic_distribution.json").write_text(
        json.dumps(traffic_dist, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if stack == "dynamo":
        epp_res = subprocess.run(
            _kubectl_cmd(
                "logs",
                f"deploy/{release_name}-frontend",
                "-n",
                namespace,
                "--tail=-1",
                kubeconfig=kubeconfig,
                context=context,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        if epp_res.returncode != 0 or not (epp_res.stdout or "").strip():
            frontend_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={release_name},"
                "nvidia.com/dynamo-component-type=frontend")
            alt_res = subprocess.run(
                _kubectl_cmd(
                    "logs",
                    "-l",
                    frontend_selector,
                    "-n",
                    namespace,
                    "--tail=-1",
                    kubeconfig=kubeconfig,
                    context=context,
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            if (alt_res.stdout or "").strip() or epp_res.returncode != 0:
                epp_res = alt_res
    else:
        epp_res = subprocess.run(
            _kubectl_cmd(
                "logs",
                "deploy/llm-d-epp",
                "-n",
                namespace,
                "--tail=-1",
                kubeconfig=kubeconfig,
                context=context,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        if epp_res.returncode != 0 or not (epp_res.stdout or "").strip():
            alt_res = subprocess.run(
                _kubectl_cmd(
                    "logs",
                    "-l",
                    "app=llm-d-epp",
                    "-n",
                    namespace,
                    "--tail=-1",
                    kubeconfig=kubeconfig,
                    context=context,
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            if (alt_res.stdout or "").strip() or epp_res.returncode != 0:
                epp_res = alt_res
    (out_dir / "epp_router.log").write_text(
        epp_res.stdout or "",
        encoding="utf-8",
    )

    return traffic_dist


def parse_and_write_results(
    logs_text: str,
    parse_out: str | Path,
    *,
    harness: str = "aiperf",
    slas: dict[str, float] | None = None,
) -> tuple[dict[str, Any], str, bool]:
    """Parse benchmark pod logs via parse_bench_results and write <parse_out> and <parse_out>.md."""
    blocks = parse_bench_results._extract_json_blocks(logs_text)
    summary = parse_bench_results._normalize(blocks) if blocks else None
    any_breach = False

    if summary is not None:
        report_md, any_breach = parse_bench_results.render_markdown(
            summary,
            slas or None,
        )
        summary["sla_breach"] = any_breach
    else:
        summary = {
            "harness": harness,
            "count_success": 0,
            "count_failure": 0,
            "metrics_ms": {
                "request_latency": {
                    "mean": None,
                    "p50": None,
                    "p95": None,
                    "p99": None
                },
                "ttft": {
                    "mean": None,
                    "p50": None,
                    "p95": None,
                    "p99": None
                },
                "tpot": {
                    "mean": None,
                    "p50": None,
                    "p95": None,
                    "p99": None
                },
            },
            "throughput": {
                "requests_per_sec": None,
                "input_tokens_per_sec": None,
                "output_tokens_per_sec": None,
                "total_tokens_per_sec": None,
            },
            "goodput_tokens_per_sec": 0.0,
            "sla_breach": False,
        }
        report_md, _ = parse_bench_results.render_markdown(summary, None)

    json_path = Path(parse_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path = Path(f"{json_path}.md")
    md_path.write_text(report_md, encoding="utf-8")
    return summary, report_md, any_breach


def execute_benchmark_job(
    manifest: str,
    *,
    namespace: str = "default",
    stack: str = "llm-d",
    release_name: str = "llm-d",
    harness: str = "inference-perf",
    timeout: int = 900,
    poll_interval: float = 5.0,
    kubeconfig: str | None = None,
    context: str | None = None,
    logs_out: str | Path | None = None,
    parse_out: str | Path | None = None,
    evidence_dir: str | Path | None = None,
    slas: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Submit benchmark ConfigMap + Job, wait for completion, collect logs, parse results, and capture evidence."""
    job_name = _extract_job_name_from_manifest(manifest)

    apply_res = subprocess.run(
        _kubectl_cmd(
            "apply",
            "-n",
            namespace,
            "-f",
            "-",
            kubeconfig=kubeconfig,
            context=context,
        ),
        input=manifest,
        text=True,
        capture_output=True,
        check=False,
    )
    if apply_res.returncode != 0:
        err_msg = (apply_res.stderr or apply_res.stdout or "").strip()
        if "immutable" in err_msg.lower() or "alreadyexists" in err_msg.lower(
        ):
            subprocess.run(
                _kubectl_cmd(
                    "delete",
                    f"job/{job_name}",
                    "-n",
                    namespace,
                    "--ignore-not-found=true",
                    kubeconfig=kubeconfig,
                    context=context,
                ),
                text=True,
                capture_output=True,
                check=False,
            )
            apply_res = subprocess.run(
                _kubectl_cmd(
                    "apply",
                    "-n",
                    namespace,
                    "-f",
                    "-",
                    kubeconfig=kubeconfig,
                    context=context,
                ),
                input=manifest,
                text=True,
                capture_output=True,
                check=False,
            )
        if apply_res.returncode != 0:
            raise RuntimeError(
                f"Failed to submit benchmark Job {job_name} in namespace {namespace}: "
                f"{(apply_res.stderr or apply_res.stdout or '').strip()}")

    deadline = time.monotonic() + max(1, int(timeout))
    job_failed = False
    while True:
        status_res = subprocess.run(
            _kubectl_cmd(
                "get",
                f"job/{job_name}",
                "-n",
                namespace,
                "-o",
                "json",
                kubeconfig=kubeconfig,
                context=context,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        raw_status = (status_res.stdout or "").strip()
        if not raw_status:
            if status_res.returncode == 0:
                break
        else:
            try:
                job_obj = json.loads(raw_status)
            except json.JSONDecodeError:
                if status_res.returncode == 0:
                    break
                job_obj = {}
            if isinstance(job_obj, dict):
                status_block = job_obj.get("status")
                if not isinstance(status_block, dict) or not status_block:
                    if status_res.returncode == 0 and "status" not in job_obj:
                        break
                else:
                    succeeded = int(status_block.get("succeeded") or 0)
                    failed = int(status_block.get("failed") or 0)
                    conditions = status_block.get("conditions") or []
                    is_complete = succeeded >= 1 or any(
                        isinstance(c, dict) and c.get("type") == "Complete"
                        and str(c.get("status")).lower() == "true"
                        for c in conditions)
                    is_failed = failed >= 1 or any(
                        isinstance(c, dict) and c.get("type") == "Failed"
                        and str(c.get("status")).lower() == "true"
                        for c in conditions)
                    if is_complete:
                        break
                    if is_failed:
                        job_failed = True
                        break

        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for benchmark Job {job_name} in namespace {namespace}"
            )
        sleep_s = min(poll_interval, max(0.0, deadline - now))
        if sleep_s > 0:
            time.sleep(sleep_s)

    log_res = subprocess.run(
        _kubectl_cmd(
            "logs",
            f"job/{job_name}",
            "-n",
            namespace,
            "--all-containers=true",
            kubeconfig=kubeconfig,
            context=context,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if log_res.returncode != 0 and not (log_res.stdout or "").strip():
        log_res = subprocess.run(
            _kubectl_cmd(
                "logs",
                "-l",
                f"job-name={job_name}",
                "-n",
                namespace,
                "--tail=-1",
                kubeconfig=kubeconfig,
                context=context,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
    logs_text = log_res.stdout or ""

    if logs_out is not None:
        logs_path = Path(logs_out)
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logs_path.write_text(logs_text, encoding="utf-8")

    summary: dict[str, Any] | None = None
    report_md: str | None = None
    any_breach = False
    if parse_out is not None:
        summary, report_md, any_breach = parse_and_write_results(
            logs_text,
            parse_out,
            harness=harness,
            slas=slas,
        )

    traffic_dist: dict[str, Any] | None = None
    if evidence_dir is not None:
        traffic_dist = capture_cluster_evidence(
            evidence_dir,
            namespace=namespace,
            stack=stack,
            release_name=release_name,
            kubeconfig=kubeconfig,
            context=context,
        )

    return {
        "job_name": job_name,
        "job_failed": job_failed,
        "logs": logs_text,
        "summary": summary,
        "report_md": report_md,
        "sla_breach": any_breach,
        "traffic_distribution": traffic_dist,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for generating and executing distributed serving benchmark Jobs."""
    parser = argparse.ArgumentParser(
        description=
        "Generate and execute distributed serving benchmark configs and K8s Jobs for llm-d and Dynamo."
    )
    parser.add_argument("--model",
                        required=True,
                        help="Model ID (e.g. Qwen/Qwen3-32B).")
    parser.add_argument(
        "--harness",
        choices=_SUPPORTED_HARNESSES,
        default="inference-perf",
        help="Benchmark harness to generate.",
    )
    parser.add_argument(
        "--stack",
        choices=_SUPPORTED_STACKS,
        default="llm-d",
        help="Target serving stack ('llm-d' or 'dynamo').",
    )
    parser.add_argument(
        "--workload-class",
        default="high-prefix-share",
        choices=tuple(_BENCHMARK_BASE_RATES.keys()),
        help="Workload classification.",
    )
    parser.add_argument("--namespace",
                        default="default",
                        help="Kubernetes namespace.")
    parser.add_argument(
        "--release-name",
        default="llm-d",
        help=
        "Helm release name (llm-d) or DynamoGraphDeployment name (dynamo).",
    )
    parser.add_argument("--target-url", help="Explicit serving endpoint URL.")
    parser.add_argument("--pvc-name",
                        help="Optional PVC name for benchmark results.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable fast smoke profile for aiperf.",
    )
    parser.add_argument("--concurrency",
                        type=int,
                        help="Explicit request concurrency.")
    parser.add_argument("--replicas",
                        type=int,
                        default=2,
                        help="Total serving replicas.")
    parser.add_argument("--isl",
                        type=int,
                        default=4000,
                        help="Mean input sequence length.")
    parser.add_argument("--osl",
                        type=int,
                        default=500,
                        help="Mean output sequence length.")
    parser.add_argument("--prefix-len",
                        type=int,
                        default=2048,
                        help="Shared prefix length.")
    parser.add_argument("--ttft-sla",
                        type=float,
                        help="Target p95 TTFT in ms.")
    parser.add_argument("--tpot-sla",
                        type=float,
                        help="Target p95 TPOT in ms.")
    parser.add_argument("--e2e-sla",
                        type=float,
                        help="Target p95 E2E latency in ms.")
    parser.add_argument(
        "--server-metrics",
        nargs="*",
        help=
        "Explicit worker Prometheus /metrics URLs for aiperf --server-metrics.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=
        "Submit the generated ConfigMap and Job to the target namespace, wait for completion, and capture logs.",
    )
    parser.add_argument(
        "--kubeconfig",
        help=
        "Optional kubeconfig path for cluster operations during --execute.",
    )
    parser.add_argument(
        "--context",
        help=
        "Optional Kubernetes context name for cluster operations during --execute.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help=
        "Timeout in seconds when waiting for the benchmark Job to complete during --execute (default: 900).",
    )
    parser.add_argument(
        "--logs-out",
        help="Path to write the benchmark pod stdout/stderr logs.",
    )
    parser.add_argument(
        "--parse-out",
        help=
        "Path to write the parsed JSON benchmark summary (<parse-out>) and Markdown report (<parse-out>.md).",
    )
    parser.add_argument(
        "--evidence-dir",
        help=
        "Directory to write post-benchmark cluster state, worker pod logs, traffic distribution, and EPP router logs.",
    )
    parser.add_argument(
        "--emit",
        choices=("deployment", "config", "discovery-cmd", "cache-reset-cmd"),
        default="deployment",
        help="Output artifact type to print to stdout.",
    )
    args = parser.parse_args(argv)

    target_url = args.target_url or resolve_serving_endpoint(
        args.stack, args.release_name)

    if args.emit == "discovery-cmd":
        print(
            build_worker_metrics_discovery_command(
                args.stack,
                args.release_name,
                args.namespace,
            ))
        return 0

    if args.emit == "cache-reset-cmd":
        for cmd in build_cache_reset_commands(
                args.stack,
                args.release_name,
                args.namespace,
                target_url=target_url,
        ):
            print(cmd)
        return 0

    inp = BenchmarkInput(
        model=args.model,
        topology=BenchmarkTopology(
            mode="agg",
            replicas=args.replicas,
        ),
        workload=BenchmarkWorkload(
            isl=args.isl,
            osl=args.osl,
            prefix_len=args.prefix_len,
            concurrency=args.concurrency,
        ),
        slo=BenchmarkSlo(
            ttft_ms=args.ttft_sla,
            tpot_ms=args.tpot_sla,
            request_latency_ms=args.e2e_sla,
        ),
        context=BenchmarkContext(
            namespace=args.namespace,
            release_name=args.release_name,
        ),
    )

    built = build_benchmark(
        inp,
        args.workload_class,
        harness=args.harness,
        stack=args.stack,
        smoke=args.smoke,
        concurrency=args.concurrency,
        server_metrics_urls=args.server_metrics,
    )
    for w in built.get("warnings") or []:
        print(f"warning: {w}", file=sys.stderr)

    if args.emit == "config":
        print(yaml.safe_dump(built["config"], sort_keys=False))
        return 0

    manifest = build_benchmark_deployment(
        built["config"],
        target_url=target_url,
        namespace=args.namespace,
        pvc_name=args.pvc_name,
    )

    slas: dict[str, float] = {}
    if args.ttft_sla is not None:
        slas["ttft_ms"] = args.ttft_sla
    if args.tpot_sla is not None:
        slas["tpot_ms"] = args.tpot_sla
    if args.e2e_sla is not None:
        slas["e2e_ms"] = args.e2e_sla

    if args.execute:
        exec_res = execute_benchmark_job(
            manifest,
            namespace=args.namespace,
            stack=args.stack,
            release_name=args.release_name,
            harness=args.harness,
            timeout=args.timeout,
            kubeconfig=args.kubeconfig,
            context=args.context,
            logs_out=args.logs_out,
            parse_out=args.parse_out,
            evidence_dir=args.evidence_dir,
            slas=slas or None,
        )
        if exec_res.get("report_md"):
            print(exec_res["report_md"], end="")
        elif not args.logs_out:
            print(exec_res.get("logs") or "")
        if exec_res.get("job_failed"):
            return 1
        return 0

    if args.parse_out and args.logs_out and Path(args.logs_out).is_file():
        saved_logs = Path(args.logs_out).read_text(encoding="utf-8")
        _, report_md, _ = parse_and_write_results(
            saved_logs,
            args.parse_out,
            harness=args.harness,
            slas=slas or None,
        )
        print(report_md, end="")
        return 0

    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
