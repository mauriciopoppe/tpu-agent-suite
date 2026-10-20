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
"""Deterministic Study Protocol and request payload compiler for distributed serving."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

TPU_DEFAULT_TOPOLOGIES: dict[str, tuple[str, int]] = {
    "v4": ("2x2x1", 4),
    "v5e": ("2x4", 8),
    "v5p": ("2x2x1", 4),
    "v6e": ("2x4", 8),
    "tpu7x": ("2x2x2", 8),
}

TRAFFIC_PRESETS: dict[str, dict[str, Any]] = {
    "agentx": {
        "preset": "agentx",
        "workload_kind": "prefix-repetition",
        "input_seq_len": 8192,
        "output_seq_len": 1024,
        "prefix_cache_hit_ratio": 0.7,
        "multi_turn": True,
        "concurrency_sweep": [8, 16, 32, 64],
        "num_prompts_per_cell": 200,
        "default_ttft_p99_ms": 2000.0,
        "default_tpot_p99_ms": 20.0,
    },
    "custom": {
        "preset": "custom",
        "workload_kind": "random",
        "input_seq_len": 2048,
        "output_seq_len": 512,
        "prefix_cache_hit_ratio": 0.0,
        "multi_turn": False,
        "concurrency_sweep": [8, 16, 32, 64],
        "num_prompts_per_cell": 200,
        "default_ttft_p99_ms": 2000.0,
        "default_tpot_p99_ms": 40.0,
    },
}

PRIMARY_METRICS: dict[str, dict[str, str]] = {
    "output_tokens_per_s": {
        "name": "output_tokens_per_s",
        "unit": "tok/s",
        "direction": "maximize",
    },
    "total_tokens_per_s": {
        "name": "total_tokens_per_s",
        "unit": "tok/s",
        "direction": "maximize",
    },
    "ttft_p50_ms": {
        "name": "ttft_p50_ms",
        "unit": "ms",
        "direction": "minimize",
    },
    "tpot_p50_ms": {
        "name": "tpot_p50_ms",
        "unit": "ms",
        "direction": "minimize",
    },
}


def chips_from_topology(topology: str) -> int:
    """Compute physical chip count from a 2D or 3D topology string."""
    parts = [int(part) for part in topology.split("x") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid topology string: {topology!r}")
    return math.prod(parts)


def normalize_tpu_type(raw: str | None) -> str:
    """Normalize user-supplied TPU generation strings to schema enum values."""
    if not raw:
        return "v6e"
    cleaned = raw.strip().lower().replace("_",
                                          "").replace("-",
                                                      "").replace(" ", "")
    if cleaned in ("v7x", "tpu7x", "tpuv7x", "v7", "ironwood"):
        return "tpu7x"
    if cleaned in ("v6e", "tpuv6e", "trillium"):
        return "v6e"
    if cleaned in ("v5p", "tpuv5p"):
        return "v5p"
    if cleaned in ("v5e", "tpuv5e", "v5litepod"):
        return "v5e"
    if cleaned in ("v4", "tpuv4"):
        return "v4"
    raise ValueError(f"Unsupported TPU type: {raw!r}")


def slugify_study_id(text: str) -> str:
    """Convert an identifier string into a hyphen-case study_id slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return (slug or "tpu-study")[:64].rstrip("-")


def parse_user_query(prompt: str) -> dict[str, Any]:
    """Extract target model, TPU type, onboarding path, and SLA bounds from a prompt."""
    lower = prompt.lower()
    hf_match = re.search(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b", prompt)
    model_id = "Qwen/Qwen3-32B"
    if hf_match and not hf_match.group(1).lower().startswith(
        ("http/", "https/", "tok/s")):
        model_id = hf_match.group(1)
    elif re.search(r"\bgpt[\s_-]*oss[\s_-]*20b\b", lower):
        model_id = "openai/gpt-oss-20b"
    elif re.search(r"\bqwen[\s/-]*3\.5\b", lower):
        model_id = "Qwen/Qwen3.5-32B"
    elif re.search(r"\bqwen[\s/-]*3\b", lower):
        model_id = "Qwen/Qwen3-32B"

    tpu_match = re.search(
        r"\b(tpu[\s_-]*v?7x|v7x|ironwood|v6e|trillium|v5p|v5e|v4)(?:-(\d+))?\b",
        lower)
    tpu_type = normalize_tpu_type(tpu_match.group(1)) if tpu_match else "v6e"
    topo_match = re.search(r"\b(\d+x\d+(?:x\d+)?)\b", lower)
    if topo_match:
        topology = topo_match.group(1)
    elif tpu_match and tpu_match.group(2) == "8":
        topology = "2x4" if tpu_type in ("v6e", "v5e") else "2x2x2"
    elif tpu_match and tpu_match.group(2) == "4":
        topology = "2x2" if tpu_type in ("v6e", "v5e") else "2x2x1"
    else:
        topology = TPU_DEFAULT_TOPOLOGIES[tpu_type][0]

    has_onboard = bool(
        re.search(r"\b(onboard|onboarding|deploy|unwire|dynamo|translate)\b",
                  lower))
    has_optimize = bool(
        re.search(
            r"\b(optimize|maximize|minimize|throughput|latency|ttft|tpot|tune|study)\b",
            lower))
    intent = ("onboard_and_optimize" if has_onboard and has_optimize else
              ("optimize_only"
               if has_optimize and not has_onboard else "onboard_only"))
    has_dist = bool(
        re.search(r"\b(dynamo|llm-d|llmd|epp|unwire|agentx)\b", lower))
    onboarding_path = ("dynamo_unwiring" if has_dist else (
        "existing_deployment" if intent == "optimize_only" else None))

    primary_metric = "output_tokens_per_s"
    if "throughput" not in lower:
        if "ttft" in lower:
            primary_metric = "ttft_p50_ms"
        elif "tpot" in lower:
            primary_metric = "tpot_p50_ms"

    out: dict[str, Any] = {
        "prompt": prompt,
        "intent": intent,
        "target": model_id,
        "tpu_type": tpu_type,
        "topology": topology,
        "onboarding_path": onboarding_path,
        "traffic_preset":
        "agentx" if "agentx" in lower or onboarding_path else "custom",
        "primary_metric": primary_metric,
    }
    ttft_m = re.search(
        r"ttft(?:\s+p\d+)?\s*(?:<=|<|=)\s*(\d+(?:\.\d+)?)\s*(ms|s)\b", lower)
    if ttft_m:
        val = float(ttft_m.group(1))
        out["ttft_p99_ms"] = val * 1000.0 if ttft_m.group(2) == "s" else val
    tpot_m = re.search(
        r"tpot(?:\s+p\d+)?\s*(?:<=|<|=)\s*(\d+(?:\.\d+)?)\s*(ms|s)\b", lower)
    if tpot_m:
        val = float(tpot_m.group(1))
        out["tpot_p99_ms"] = val * 1000.0 if tpot_m.group(2) == "s" else val
    return out


def build_query_intake(
    *,
    prompt: str | None = None,
    study_id: str | None = None,
    target: str | None = None,
    namespace: str = "default",
    onboarding_path: str | None = None,
    tpu_type: str | None = None,
    topology: str | None = None,
    max_slices: int = 2,
    traffic_preset: str | None = None,
    input_seq_len: int | None = None,
    output_seq_len: int | None = None,
    prefix_cache_hit_ratio: float | None = None,
    concurrency_sweep: list[int] | None = None,
    primary_metric: str | None = None,
    ttft_p50_ms: float | None = None,
    ttft_p99_ms: float | None = None,
    tpot_p99_ms: float | None = None,
    replicate_count: int = 3,
    noise_floor_pct: float = 1.5,
    max_rounds: int = 12,
    plateau_rounds: int = 4,
    target_improvement_pct: float | None = None,
    mode: str = "approve",
) -> dict[str, Any]:
    """Synthesize query intake context and a study-protocol.json dictionary."""
    parsed = parse_user_query(prompt) if prompt else {}
    resolved_target = target or parsed.get("target") or "Qwen/Qwen3-32B"
    resolved_tpu = normalize_tpu_type(tpu_type or parsed.get("tpu_type")
                                      or "v6e")
    resolved_topology = topology or parsed.get(
        "topology") or TPU_DEFAULT_TOPOLOGIES[resolved_tpu][0]
    chips_per_slice = chips_from_topology(resolved_topology)
    resolved_path = onboarding_path or parsed.get(
        "onboarding_path") or "dynamo_unwiring"

    preset_name = traffic_preset or parsed.get("traffic_preset") or "agentx"
    if preset_name not in TRAFFIC_PRESETS:
        raise ValueError(f"Unsupported traffic preset: {preset_name!r}")
    preset_cfg = TRAFFIC_PRESETS[preset_name]

    isl = input_seq_len if input_seq_len is not None else int(
        preset_cfg["input_seq_len"])
    osl = output_seq_len if output_seq_len is not None else int(
        preset_cfg["output_seq_len"])
    cache_ratio = (prefix_cache_hit_ratio if prefix_cache_hit_ratio is not None
                   else float(preset_cfg["prefix_cache_hit_ratio"]))
    sweep = list(concurrency_sweep) if concurrency_sweep is not None else list(
        preset_cfg["concurrency_sweep"])

    metric_name = primary_metric or parsed.get(
        "primary_metric") or "output_tokens_per_s"
    if metric_name not in PRIMARY_METRICS:
        raise ValueError(f"Unsupported primary metric: {metric_name!r}")
    metric_cfg = dict(PRIMARY_METRICS[metric_name])

    resolved_ttft_p99 = float(
        ttft_p99_ms if ttft_p99_ms is not None else parsed.
        get("ttft_p99_ms", preset_cfg["default_ttft_p99_ms"]))
    resolved_tpot_p99 = float(
        tpot_p99_ms if tpot_p99_ms is not None else parsed.
        get("tpot_p99_ms", preset_cfg["default_tpot_p99_ms"]))

    model_slug = resolved_target.split("/")[-1]
    resolved_study_id = slugify_study_id(
        study_id or f"{model_slug}-{resolved_tpu}-{preset_name}-opt")
    ref_concurrency = int(sweep[min(len(sweep) // 2, len(sweep) - 1)])
    measurement_cell = {
        "topology": f"{resolved_tpu}-{resolved_topology}",
        "shape": f"{isl}-{osl}",
        "concurrency": ref_concurrency,
    }

    admissibility_gates: list[dict[str, Any]] = []
    if ttft_p50_ms is not None:
        admissibility_gates.append({
            "name": "ttft-p50-ceiling",
            "metric": "ttft_p50_ms",
            "comparator": "lte",
            "bound": float(ttft_p50_ms),
            "unit": "ms",
            "same_execution": True,
        })
    admissibility_gates.extend([
        {
            "name": "ttft-p99-ceiling",
            "metric": "ttft_p99_ms",
            "comparator": "lte",
            "bound": resolved_ttft_p99,
            "unit": "ms",
            "same_execution": True,
        },
        {
            "name": "tpot-p99-ceiling",
            "metric": "tpot_p99_ms",
            "comparator": "lte",
            "bound": resolved_tpot_p99,
            "unit": "ms",
            "same_execution": True,
        },
        {
            "name": "error-rate-ceiling",
            "metric": "error_rate_pct",
            "comparator": "lte",
            "bound": 0.0,
            "unit": "%",
            "same_execution": True,
        },
    ])

    stopping_rule: dict[str, Any] = {
        "max_rounds": int(max_rounds),
        "plateau_rounds": int(plateau_rounds),
    }
    if target_improvement_pct is not None:
        stopping_rule["target_improvement_pct"] = float(target_improvement_pct)

    protocol: dict[str, Any] = {
        "primary_metric": metric_cfg,
        "measurement_cell": measurement_cell,
        "replicate_count": int(replicate_count),
        "noise_floor_pct": float(noise_floor_pct),
        "stopping_rule": stopping_rule,
        "admissibility_gates": admissibility_gates,
        "mode": mode,
    }

    intent = parsed.get("intent", "onboard_and_optimize")
    stages = (["baseline_onboarding"] if intent == "onboard_only" else (
        ["empirical_profiling", "study_loop"] if intent == "optimize_only" else
        ["baseline_onboarding", "empirical_profiling", "study_loop"]))
    resolved_path_out = onboarding_path if onboarding_path is not None else parsed.get(
        "onboarding_path")
    if resolved_path_out is None and intent != "onboard_only":
        resolved_path_out = "dynamo_unwiring"

    return {
        "study_id": resolved_study_id,
        "target": resolved_target,
        "namespace": namespace,
        "onboarding_path": resolved_path_out,
        "staged_plan": {
            "intent": intent,
            "stages": stages,
        },
        "hardware_budget": {
            "tpu_type": resolved_tpu,
            "topology": resolved_topology,
            "chips_per_slice": chips_per_slice,
            "max_slices": int(max_slices),
        },
        "traffic_profile": {
            "preset": preset_name,
            "workload_kind": preset_cfg["workload_kind"],
            "input_seq_len": isl,
            "output_seq_len": osl,
            "prefix_cache_hit_ratio": cache_ratio,
            "multi_turn": bool(preset_cfg["multi_turn"]),
            "concurrency_sweep": sweep,
            "num_prompts_per_cell": int(preset_cfg["num_prompts_per_cell"]),
        },
        "sla_constraints": {
            "ttft_p99_ms": resolved_ttft_p99,
            "tpot_p99_ms": resolved_tpot_p99,
            "max_error_rate_pct": 0.0,
        },
        "protocol": protocol,
        **protocol,
    }


def compile_study_protocol(intake: dict[str, Any] | None = None,
                           **kwargs: Any) -> dict[str, Any]:
    """Compile query intake or keyword arguments into a study-protocol.json dict."""
    if intake is None:
        intake = build_query_intake(**kwargs)
    proto = intake.get("protocol", intake)
    return {
        "primary_metric":
        dict(proto["primary_metric"]),
        "measurement_cell":
        dict(proto["measurement_cell"]),
        "replicate_count":
        int(proto["replicate_count"]),
        "noise_floor_pct":
        float(proto["noise_floor_pct"]),
        "stopping_rule":
        dict(proto["stopping_rule"]),
        "admissibility_gates":
        [dict(gate) for gate in proto["admissibility_gates"]],
        "mode":
        proto.get("mode", "approve"),
    }


build_study_protocol = compile_study_protocol


def compile_onboarding_request(
    intake: dict[str, Any],
    *,
    prompt: str,
    run_id: str | None = None,
    compute_target: str | None = None,
    dynamo_manifests: list[str] | None = None,
) -> dict[str, Any]:
    """Compile query intake into an onboarding-request.json dict."""
    path = intake.get("onboarding_path")
    payload: dict[str, Any] = {
        "prompt": prompt,
        "model": {
            "model_id": intake["target"],
            "task": "text-generation"
        },
        "mode": "distributed" if path == "dynamo_unwiring" else "inference",
        "hardware_budget": dict(intake["hardware_budget"]),
        "traffic_profile": dict(intake["traffic_profile"]),
    }
    if path in ("dynamo_unwiring", "existing_deployment"):
        payload["onboarding_path"] = path
    if run_id:
        payload["run_id"] = str(run_id)
    if compute_target:
        payload["compute_target"] = str(compute_target)
    if "sla_constraints" in intake:
        payload["sla_constraints"] = dict(intake["sla_constraints"])
    if dynamo_manifests:
        payload["dynamo_manifests"] = list(dynamo_manifests)
    return payload


def compile_study_request(
    intake: dict[str, Any],
    *,
    prompt: str,
    run_id: str | None = None,
    baseline_metrics: str | None = None,
    levers: list[str] | None = None,
    protocol: dict[str, Any] | None = None,
    protocol_path: str | None = None,
) -> dict[str, Any]:
    """Compile query intake into a study-request.json dict."""
    payload: dict[str, Any] = {
        "prompt": prompt,
        "study_id": intake["study_id"],
        "target": intake["target"],
        "namespace": intake.get("namespace", "default"),
        "mode": intake.get("mode", "approve"),
        "hardware_budget": dict(intake["hardware_budget"]),
        "traffic_profile": dict(intake["traffic_profile"]),
    }
    if run_id:
        payload["run_id"] = str(run_id)
    if baseline_metrics:
        payload["baseline_metrics"] = str(baseline_metrics)
    if levers:
        payload["levers"] = list(levers)
    if protocol is not None:
        payload["protocol"] = dict(protocol)
    if protocol_path:
        payload["protocol_path"] = str(protocol_path)
    return payload


def write_compiled_artifacts(
    intake: dict[str, Any],
    output_dir: Path,
    *,
    prompt: str,
    run_id: str | None = None,
    baseline_metrics: str | None = None,
    levers: list[str] | None = None,
    dynamo_manifests: list[str] | None = None,
) -> dict[str, str]:
    """Write protocol.json, onboarding-request.json, and study-request.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    proto_path = output_dir / "protocol.json"
    onboard_path = output_dir / "onboarding-request.json"
    study_path = output_dir / "study-request.json"

    protocol = compile_study_protocol(intake)
    onboarding_req = compile_onboarding_request(
        intake,
        prompt=prompt,
        run_id=run_id,
        dynamo_manifests=dynamo_manifests)
    study_req = compile_study_request(
        intake,
        prompt=prompt,
        run_id=run_id,
        baseline_metrics=baseline_metrics,
        levers=levers,
        protocol=protocol,
        protocol_path=str(proto_path),
    )

    proto_path.write_text(json.dumps(protocol, indent=2) + "\n",
                          encoding="utf-8")
    onboard_path.write_text(json.dumps(onboarding_req, indent=2) + "\n",
                            encoding="utf-8")
    study_path.write_text(json.dumps(study_req, indent=2) + "\n",
                          encoding="utf-8")
    return {
        "study_protocol": str(proto_path),
        "onboarding_request": str(onboard_path),
        "study_request": str(study_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=
        "Compile user query intake into study-protocol.json and delegation payloads."
    )
    parser.add_argument("--prompt",
                        required=True,
                        help="Natural-language user request.")
    parser.add_argument("--study-id",
                        help="Optional explicit study identifier.")
    parser.add_argument("--run-id", help="Optional session run identifier.")
    parser.add_argument("--target",
                        help="Optional explicit model ID or target service.")
    parser.add_argument("--namespace",
                        default="default",
                        help="Target Kubernetes namespace.")
    parser.add_argument("--onboarding-path",
                        choices=["dynamo_unwiring", "existing_deployment"])
    parser.add_argument(
        "--tpu-type",
        help="TPU generation override (v4, v5e, v5p, v6e, tpu7x).")
    parser.add_argument("--topology",
                        help="TPU topology override (e.g. 2x4, 2x2x2).")
    parser.add_argument("--max-slices", type=int, default=2)
    parser.add_argument("--traffic-preset",
                        choices=list(TRAFFIC_PRESETS.keys()))
    parser.add_argument("--input-seq-len", type=int)
    parser.add_argument("--output-seq-len", type=int)
    parser.add_argument("--prefix-cache-hit-ratio", type=float)
    parser.add_argument("--primary-metric",
                        choices=list(PRIMARY_METRICS.keys()))
    parser.add_argument("--ttft-p50-ms", type=float)
    parser.add_argument("--ttft-p99-ms", type=float)
    parser.add_argument("--tpot-p99-ms", type=float)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--plateau-rounds", type=int, default=4)
    parser.add_argument("--mode",
                        default="approve",
                        choices=["approve", "auto"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    intake = build_query_intake(
        prompt=args.prompt,
        study_id=args.study_id,
        target=args.target,
        namespace=args.namespace,
        onboarding_path=args.onboarding_path,
        tpu_type=args.tpu_type,
        topology=args.topology,
        max_slices=args.max_slices,
        traffic_preset=args.traffic_preset,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        prefix_cache_hit_ratio=args.prefix_cache_hit_ratio,
        primary_metric=args.primary_metric,
        ttft_p50_ms=args.ttft_p50_ms,
        ttft_p99_ms=args.ttft_p99_ms,
        tpot_p99_ms=args.tpot_p99_ms,
        max_rounds=args.max_rounds,
        plateau_rounds=args.plateau_rounds,
        mode=args.mode,
    )
    protocol = compile_study_protocol(intake)
    onboarding_req = compile_onboarding_request(intake,
                                                prompt=args.prompt,
                                                run_id=args.run_id)
    study_req = compile_study_request(intake,
                                      prompt=args.prompt,
                                      run_id=args.run_id,
                                      protocol=protocol)
    artifact_paths = (write_compiled_artifacts(
        intake, args.output_dir, prompt=args.prompt, run_id=args.run_id)
                      if args.output_dir is not None else {})
    bundle = {
        "query_intake": intake,
        "study_protocol": protocol,
        "onboarding_request": onboarding_req,
        "study_request": study_req,
        "artifact_paths": artifact_paths,
    }
    print(json.dumps(bundle, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
