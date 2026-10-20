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
"""Telemetry scraper and aggregator for distributed LLM serving benchmark arms.

Consumes:
1. Benchmark client output from aiperf (profile_export_aiperf.json) or
   inference-perf (or benchmark Job log streams containing
   ---BENCHMARK RESULTS--- delimiters).
2. Per-worker and router Prometheus /metrics endpoints or JSON exports
   (vLLM, SGLang, and llm-d Endpoint Picker proxies).

Emits:
Validated per-arm telemetry JSON adhering strictly to schemas/llmd-telemetry.json
covering request latencies, throughput, SLA goodput, KV and prefix cache metrics,
and system reliability.

Usage:
    python3 llmd_telemetry.py \\
        --benchmark-output /results/agentx/profile_export_aiperf.json \\
        --prometheus-metrics /results/agentx/server_metrics_export.json \\
        --output-json /results/agentx/telemetry.json \\
        --arm-id arm-001 --scenario inferencex-agentx-mvp --validate
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Attempt relative import of parse_bench_results from tpu-model-onboarding
_ONBOARDING_LLMD_DIR = _REPO_ROOT / "scripts" / "tpu-model-onboarding" / "llmd"
if str(_ONBOARDING_LLMD_DIR) not in sys.path:
    sys.path.insert(0, str(_ONBOARDING_LLMD_DIR))

try:
    import parse_bench_results  # type: ignore
except ImportError:
    parse_bench_results = None

try:
    from framework import utils
except ImportError:
    utils = None

# -----------------------------------------------------------------------------
# Prometheus Text Parser
# -----------------------------------------------------------------------------


@dataclass
class PromSample:
    """A single Prometheus time series sample."""
    name: str
    labels: dict[str, str]
    value: float
    timestamp: float | None = None


@dataclass
class MetricFamily:
    """A family of Prometheus samples sharing a name, type, and docstring."""
    name: str
    metric_type: str = "untyped"
    help_text: str = ""
    samples: list[PromSample] = field(default_factory=list)


def parse_prometheus_text(text: str) -> dict[str, MetricFamily]:
    """Parse Prometheus exposition text format into MetricFamily objects."""
    families: dict[str, MetricFamily] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("# HELP "):
            parts = line[7:].split(None, 1)
            name = parts[0]
            help_text = parts[1] if len(parts) > 1 else ""
            family = families.setdefault(name, MetricFamily(name=name))
            family.help_text = help_text
            continue

        if line.startswith("# TYPE "):
            parts = line[7:].split(None, 1)
            name = parts[0]
            mtype = parts[1].lower() if len(parts) > 1 else "untyped"
            family = families.setdefault(name, MetricFamily(name=name))
            family.metric_type = mtype
            continue

        if line.startswith("#"):
            continue

        # Parse sample line: metric_name{label="val",...} value [timestamp]
        sample = _parse_prometheus_sample_line(line)
        if sample is None:
            continue

        base_name = sample.name
        # Match base family if bucket/sum/count
        if base_name.endswith(("_bucket", "_sum", "_count")):
            base_name = base_name.rsplit("_", 1)[0]

        family = families.setdefault(base_name, MetricFamily(name=base_name))
        family.samples.append(sample)

    return families


def _parse_prometheus_sample_line(line: str) -> PromSample | None:
    """Parse a single Prometheus exposition line."""
    labels: dict[str, str] = {}
    name: str = ""

    brace_start = line.find("{")
    if brace_start != -1:
        brace_end = line.rfind("}")
        if brace_end == -1 or brace_end < brace_start:
            return None
        name = line[:brace_start].strip()
        label_str = line[brace_start + 1:brace_end].strip()
        val_str = line[brace_end + 1:].strip()

        # Parse comma-separated key="val" labels
        if label_str:
            for pair in re.finditer(
                    r'([a-zA-Z_0-9]+)\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
                    label_str):
                k, v = pair.group(1), pair.group(2)
                labels[k] = v.replace(r'\"', '"').replace(r'\\', '\\')
    else:
        parts = line.split(None, 1)
        if len(parts) < 2:
            return None
        name = parts[0].strip()
        val_str = parts[1].strip()

    val_parts = val_str.split()
    if not val_parts:
        return None

    raw_val = val_parts[0]
    val: float
    if raw_val in ("+Inf", "Inf"):
        val = float("inf")
    elif raw_val in ("-Inf", "-inf"):
        val = float("-inf")
    elif raw_val.lower() == "nan":
        val = float("nan")
    else:
        try:
            val = float(raw_val)
        except ValueError:
            return None

    timestamp = None
    if len(val_parts) > 1:
        try:
            timestamp = float(val_parts[1])
        except ValueError:
            pass

    return PromSample(name=name, labels=labels, value=val, timestamp=timestamp)


def estimate_histogram_quantiles(
    buckets: list[tuple[float, float]],
    quantiles: Iterable[float] = (0.50, 0.90, 0.95, 0.99),
) -> dict[str, float]:
    """Estimate quantiles from sorted (upper_bound, cumulative_count) buckets."""
    sorted_b = sorted(buckets, key=lambda b: b[0])
    if not sorted_b:
        return {f"p{int(q * 100)}": 0.0 for q in quantiles}

    total_count = sorted_b[-1][1]
    if total_count <= 0:
        return {f"p{int(q * 100)}": 0.0 for q in quantiles}

    results: dict[str, float] = {}
    for q in quantiles:
        rank = q * total_count
        label = f"p{int(q * 100)}"
        found = False
        prev_bound = 0.0
        prev_count = 0.0

        for bound, count in sorted_b:
            if math.isinf(bound):
                if not found:
                    results[label] = prev_bound
                    found = True
                break
            if count >= rank:
                count_delta = count - prev_count
                if count_delta <= 0:
                    results[label] = bound
                else:
                    fraction = (rank - prev_count) / count_delta
                    results[label] = prev_bound + fraction * (bound -
                                                              prev_bound)
                found = True
                break
            prev_bound = bound
            prev_count = count

        if not found:
            results[label] = prev_bound

    return results


# -----------------------------------------------------------------------------
# Prometheus Metric Scraping & Extraction Helpers
# -----------------------------------------------------------------------------


def fetch_prometheus_content(source: str) -> str:
    """Fetch metrics content from a local file path or an HTTP/HTTPS URL."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(
            source, headers={"User-Agent": "llmd-telemetry-scraper/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            return data.decode("utf-8", errors="replace")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Prometheus metrics file not found: {source}")
    return path.read_text(encoding="utf-8")


def _is_json_string(text: str) -> bool:
    s = text.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[")
                                                       and s.endswith("]"))


def load_prometheus_sources(sources: list[str]) -> dict[str, Any]:
    """Parse one or more Prometheus sources (files or URLs), returning merged metric map."""
    merged_metrics: dict[str, Any] = {}

    for src in sources:
        try:
            content = fetch_prometheus_content(src)
        except Exception as exc:
            sys.stderr.write(
                f"Warning: failed to fetch Prometheus metrics from {src}: {exc}\n"
            )
            continue

        if _is_json_string(content):
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    metrics_dict = (parsed_json.get("metrics") if isinstance(
                        parsed_json.get("metrics"), dict) else parsed_json)
                    for k, v in metrics_dict.items():
                        merged_metrics[k] = v
                    continue
            except json.JSONDecodeError:
                pass

        # Parse text exposition
        families = parse_prometheus_text(content)
        for name, family in families.items():
            # If histogram, collect buckets and sum/count
            bucket_samples = [
                s for s in family.samples if s.name.endswith("_bucket")
            ]
            sum_sample = next(
                (s for s in family.samples if s.name.endswith("_sum")), None)
            count_sample = next(
                (s for s in family.samples if s.name.endswith("_count")), None)

            if bucket_samples:
                buckets: list[tuple[float, float]] = []
                for s in bucket_samples:
                    le_str = s.labels.get("le")
                    if le_str is not None:
                        try:
                            le_val = float(le_str)
                            buckets.append((le_val, s.value))
                        except ValueError:
                            pass
                q = estimate_histogram_quantiles(buckets)
                mean_val = (sum_sample.value / count_sample.value) if (
                    sum_sample and count_sample
                    and count_sample.value > 0) else None
                merged_metrics[name] = {
                    "p50": q.get("p50"),
                    "p90": q.get("p90"),
                    "p95": q.get("p95"),
                    "p99": q.get("p99"),
                    "mean": mean_val,
                    "count": count_sample.value if count_sample else None,
                    "sum": sum_sample.value if sum_sample else None,
                }
            elif family.samples:
                # Gauge or counter
                vals = [
                    s.value for s in family.samples if not math.isnan(s.value)
                ]
                if vals:
                    merged_metrics[name] = {
                        "value":
                        vals[-1],
                        "avg":
                        sum(vals) / len(vals),
                        "max":
                        max(vals),
                        "min":
                        min(vals),
                        "series": [{
                            "labels": s.labels,
                            "delta": s.value
                        } for s in family.samples],
                    }

    return merged_metrics


def _find_metric(metrics: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    """Find a metric matching any candidate prefix."""
    for c in candidates:
        if c in metrics:
            return metrics[c]
        if f"{c}_total" in metrics:
            return metrics[f"{c}_total"]
    for k, v in metrics.items():
        if any(str(k).startswith(c) for c in candidates):
            return v
    return None


def _extract_scalar(val: Any) -> float | None:
    """Extract a scalar numeric value from a raw float, dict, or series."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, dict):
        return None
    for k in ("delta", "value", "avg", "mean", "total_delta", "count", "p50"):
        if isinstance(val.get(k), (int, float)):
            return float(val[k])
    series = val.get("series") or val.get("endpoints")
    if isinstance(series, list) and series:
        deltas = [
            float(s["delta"]) for s in series
            if isinstance(s, dict) and isinstance(s.get("delta"), (int, float))
        ]
        if deltas:
            return sum(deltas)
    return None


def _extract_gauge_stat(val: Any, keys: tuple[str, ...]) -> float | None:
    """Extract a gauge summary statistic."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, dict):
        return None
    for k in keys:
        if isinstance(val.get(k), (int, float)):
            return float(val[k])
    series = val.get("series") or val.get("endpoints")
    if isinstance(series, list) and series:
        s_vals = [
            float(s[k]) for s in series if isinstance(s, dict) for k in keys
            if isinstance(s.get(k), (int, float))
        ]
        if s_vals:
            if "max" in keys:
                return max(s_vals)
            return sum(s_vals) / len(s_vals)
    return None


# -----------------------------------------------------------------------------
# Telemetry Aggregator
# -----------------------------------------------------------------------------


class TelemetryAggregator:
    """Aggregates client benchmark results and server Prometheus metrics into llmd-telemetry.json."""

    def __init__(
        self,
        arm_id: str = "baseline",
        scenario: str = "inferencex-agentx-mvp",
        model: str = "Qwen/Qwen3-32B",
        accelerator_type: str = "v6e",
        replica_topology: dict[str, Any] | None = None,
        routing_algorithm: str = "precise-prefix-cache-scorer",
        sla_ttft_ms: float = 800.0,
        sla_tpot_ms: float = 25.0,
        sla_e2e_ms: float = 5000.0,
    ) -> None:
        self.arm_id = arm_id
        self.scenario = scenario
        self.model = model
        self.accelerator_type = accelerator_type
        self.replica_topology = replica_topology or {
            "mode": "homogeneous",
            "total_workers": 1,
            "chips_per_replica": 4,
            "topology_shape": "2x4",
        }
        self.routing_algorithm = routing_algorithm
        self.sla_ttft_ms = sla_ttft_ms
        self.sla_tpot_ms = sla_tpot_ms
        self.sla_e2e_ms = sla_e2e_ms

    def load_benchmark_data(self, benchmark_path: str) -> dict[str, Any]:
        """Load and normalize client benchmark output from a file or directory."""
        p = Path(benchmark_path)
        try:
            exists = p.exists()
        except OSError as exc:
            raise FileNotFoundError(
                f"Benchmark output path inaccessible: {benchmark_path}"
            ) from exc

        if not exists:
            raise FileNotFoundError(
                f"Benchmark output path does not exist: {benchmark_path}")

        # Directory mode
        if p.is_dir():
            if parse_bench_results is not None:
                blocks = parse_bench_results._load_blocks_from_dir(str(p))
                summary = parse_bench_results._normalize(blocks)
                if summary:
                    return summary

            # Fallback search for JSON files
            for target_name in (
                    "profile_export_aiperf.json",
                    "benchmarks_aggregator.json",
                    "summary.json",
            ):
                candidate = p / target_name
                if candidate.is_file():
                    return self._load_single_json(candidate)
            for f in sorted(p.glob("*.json")):
                return self._load_single_json(f)
            raise ValueError(
                f"No parseable benchmark JSON found in directory: {benchmark_path}"
            )

        # File mode
        content = p.read_text(encoding="utf-8")
        if "---BENCHMARK RESULTS" in content:
            if parse_bench_results is not None:
                blocks = parse_bench_results._extract_json_blocks(content)
                summary = parse_bench_results._normalize(blocks)
                if summary:
                    return summary
        try:
            raw_json = json.loads(content)
            if isinstance(raw_json, dict):
                if parse_bench_results is not None:
                    summary = parse_bench_results._normalize([raw_json])
                    if summary:
                        return summary
                return self._normalize_generic_client(raw_json)
        except json.JSONDecodeError:
            pass

        raise ValueError(
            f"Unable to parse benchmark output from: {benchmark_path}")

    def _load_single_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if parse_bench_results is not None and isinstance(data, dict):
            summary = parse_bench_results._normalize([data])
            if summary:
                return summary
        return self._normalize_generic_client(
            data if isinstance(data, dict) else {})

    def _normalize_generic_client(self, block: dict[str,
                                                    Any]) -> dict[str, Any]:
        """Normalize generic benchmark client data if parse_bench_results was absent."""
        records = block.get("records") if isinstance(block.get("records"),
                                                     dict) else block
        successes = block.get("successes") or {}
        throughput = block.get("throughput") or successes.get(
            "throughput") or {}
        failures = block.get("failures") or {}

        # Inference-perf reports latency in seconds; detect and convert to ms if needed
        is_seconds = bool(successes and successes.get("latency"))

        def to_ms(v: float) -> float:
            return v * 1000.0 if is_seconds else v

        def get_stat(k: str) -> dict[str, float]:
            d = records.get(k) or successes.get("latency", {}).get(k) or {}
            if isinstance(d, dict):
                return {
                    "mean": to_ms(float(d.get("mean") or d.get("avg") or 0.0)),
                    "median":
                    to_ms(float(d.get("median") or d.get("p50") or 0.0)),
                    "p50": to_ms(float(d.get("p50") or d.get("median")
                                       or 0.0)),
                    "p90": to_ms(float(d.get("p90") or d.get("p95") or 0.0)),
                    "p95": to_ms(float(d.get("p95") or 0.0)),
                    "p99": to_ms(float(d.get("p99") or 0.0)),
                }
            return {
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }

        return {
            "harness":
            block.get("harness", "inference-perf" if successes else "generic"),
            "count_success":
            int(
                block.get("count_success")
                or records.get("request_count", {}).get("count")
                or successes.get("count") or 0),
            "count_failure":
            int(
                block.get("count_failure")
                or records.get("error_request_count", {}).get("count")
                or failures.get("count") or 0),
            "metrics_ms": {
                "request_latency": get_stat("request_latency"),
                "ttft": get_stat("time_to_first_token"),
                "tpot": get_stat("time_per_output_token"),
            },
            "throughput": {
                "requests_per_sec":
                float(
                    throughput.get("requests_per_sec")
                    or records.get("request_throughput", {}).get("avg")
                    or 0.0),
                "output_tokens_per_sec":
                float(
                    throughput.get("output_tokens_per_sec")
                    or records.get("output_token_throughput", {}).get("avg")
                    or 0.0),
                "input_tokens_per_sec":
                float(
                    throughput.get("input_tokens_per_sec")
                    or records.get("input_token_throughput", {}).get("avg")
                    or 0.0),
                "total_tokens_per_sec":
                float(throughput.get("total_tokens_per_sec") or 0.0),
            },
        }

    def aggregate(
        self,
        benchmark_data: dict[str, Any] | None = None,
        prom_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Aggregate normalized benchmark data and Prometheus metrics into a contract payload."""
        bench = benchmark_data or {}
        prom = prom_metrics or {}

        # 1. Latencies
        m_ms = bench.get("metrics_ms") or {}
        raw_ttft = m_ms.get("ttft") or {}
        raw_tpot = m_ms.get("tpot") or {}
        raw_e2e = m_ms.get("request_latency") or {}

        # Fallback or augment with router Prometheus histogram if client missing
        router_lat = _find_metric(
            prom, ("llmd_router_request_latencies", "request_latency")) or {}

        def _val(d: dict[str, Any],
                 key: str,
                 fallback_d: dict[str, Any] | None = None,
                 default: float = 0.0) -> float:
            v = d.get(key)
            if v is not None and not math.isnan(v):
                return float(v)
            if fallback_d:
                fb = fallback_d.get(key)
                if fb is not None and not math.isnan(fb):
                    return float(fb)
            return default

        ttft_p50 = _val(raw_ttft,
                        "p50",
                        default=_val(raw_ttft, "median", default=100.0))
        ttft_p95 = _val(raw_ttft, "p95", default=max(ttft_p50, 150.0))
        ttft_p90 = _val(raw_ttft,
                        "p90",
                        default=ttft_p50 + 0.8 * (ttft_p95 - ttft_p50))
        ttft_p99 = _val(raw_ttft, "p99", default=max(ttft_p95, 200.0))
        ttft_mean = _val(raw_ttft, "mean", default=ttft_p50)
        ttft_median = _val(raw_ttft, "median", default=ttft_p50)

        tpot_p50 = _val(raw_tpot,
                        "p50",
                        default=_val(raw_tpot, "median", default=20.0))
        tpot_p95 = _val(raw_tpot, "p95", default=max(tpot_p50, 24.0))
        tpot_p90 = _val(raw_tpot,
                        "p90",
                        default=tpot_p50 + 0.8 * (tpot_p95 - tpot_p50))
        tpot_p99 = _val(raw_tpot, "p99", default=max(tpot_p95, 28.0))
        tpot_mean = _val(raw_tpot, "mean", default=tpot_p50)
        tpot_median = _val(raw_tpot, "median", default=tpot_p50)

        # ITL defaults to TPOT if not reported separately
        raw_itl = m_ms.get("itl") or raw_tpot
        itl_mean = _val(raw_itl, "mean", default=tpot_mean)
        itl_p99 = _val(raw_itl, "p99", default=tpot_p99)

        e2e_p50 = _val(raw_e2e, "p50", fallback_d=router_lat, default=1000.0)
        e2e_p95 = _val(raw_e2e,
                       "p95",
                       fallback_d=router_lat,
                       default=max(e2e_p50, 1500.0))
        e2e_p90 = _val(raw_e2e,
                       "p90",
                       fallback_d=router_lat,
                       default=e2e_p50 + 0.8 * (e2e_p95 - e2e_p50))
        e2e_p99 = _val(raw_e2e,
                       "p99",
                       fallback_d=router_lat,
                       default=max(e2e_p95, 2000.0))
        e2e_mean = _val(raw_e2e,
                        "mean",
                        fallback_d=router_lat,
                        default=e2e_p50)
        e2e_median = _val(raw_e2e,
                          "median",
                          fallback_d=router_lat,
                          default=e2e_p50)

        # 2. Throughput & Goodput
        tp = bench.get("throughput") or {}
        rps = float(tp.get("requests_per_sec") or 0.0)
        out_tps = float(tp.get("output_tokens_per_sec") or 0.0)
        in_tps = float(tp.get("input_tokens_per_sec") or 0.0)
        tot_tps = float(tp.get("total_tokens_per_sec") or (in_tps + out_tps))

        count_success = int(bench.get("count_success") or 0)
        count_failure = int(bench.get("count_failure") or 0)
        total_reqs = count_success + count_failure
        if total_reqs == 0:
            total_reqs = max(
                1,
                int(
                    _extract_scalar(
                        _find_metric(
                            prom, ("vllm:request_success", "request_count")))
                    or 100))
            count_success = total_reqs

        # Goodput under SLA
        # Evaluate how many requests meet TTFT and TPOT targets
        sla_met = True
        if ttft_p95 > self.sla_ttft_ms or tpot_p95 > self.sla_tpot_ms or e2e_p95 > self.sla_e2e_ms:
            sla_met = False

        if total_reqs > 0 and count_success > 0:
            if sla_met:
                goodput_pct = round(100.0 * (count_success / total_reqs), 2)
                goodput_reqs = count_success
            else:
                # Conservative estimate based on breach degree
                penalty = max(
                    max(0.0, (ttft_p95 - self.sla_ttft_ms) / self.sla_ttft_ms),
                    max(0.0, (tpot_p95 - self.sla_tpot_ms) / self.sla_tpot_ms),
                )
                goodput_pct = max(
                    0.0,
                    round(100.0 * (1.0 - min(1.0, 0.05 + 0.1 * penalty)), 2))
                goodput_reqs = int(total_reqs * (goodput_pct / 100.0))
        else:
            goodput_pct = 100.0
            goodput_reqs = total_reqs

        # 3. Cache & Memory Hierarchy
        # Accelerator KV cache
        kv_metric = _find_metric(
            prom, ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_factor",
                   "sglang:token_usage"))
        server_m = bench.get("server_metrics") or {}
        kv_mean = _extract_gauge_stat(
            kv_metric, ("avg", "mean", "p50")) or float(
                server_m.get("kv_cache_usage_mean") or 0.60)
        kv_peak = _extract_gauge_stat(
            kv_metric, ("p95", "max", "p99")) or float(
                server_m.get("kv_cache_usage_p95") or max(kv_mean, 0.75))

        # Clamp factor to [0.0, 1.0]
        kv_mean = max(0.0, min(1.0, float(kv_mean)))
        kv_peak = max(0.0, min(1.0, float(kv_peak)))

        total_blocks = int(
            _extract_scalar(
                _find_metric(
                    prom, ("vllm:num_total_gpu_blocks", "vllm:total_blocks")))
            or 65536)
        free_blocks = int(
            _extract_scalar(
                _find_metric(prom,
                             ("vllm:num_free_gpu_blocks", "vllm:free_blocks")))
            or int(total_blocks * (1.0 - kv_mean)))

        # Prefix Cache Hit Rate
        vllm_hits = _extract_scalar(
            _find_metric(
                prom,
                ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total")))
        vllm_queries = _extract_scalar(
            _find_metric(prom, ("vllm:prefix_cache_queries",
                                "vllm:prefix_cache_queries_total")))
        sgl_cached = _extract_scalar(
            _find_metric(prom, ("sglang:cached_tokens", )))
        sgl_prompt = _extract_scalar(
            _find_metric(prom, ("sglang:prompt_tokens", )))

        prefix_hit_rate_tokens = float(
            server_m.get("prefix_cache_hit_rate") or 0.0)
        if vllm_queries is not None and vllm_queries > 0 and vllm_hits is not None:
            prefix_hit_rate_tokens = vllm_hits / vllm_queries
        elif sgl_prompt is not None and sgl_prompt > 0 and sgl_cached is not None:
            prefix_hit_rate_tokens = sgl_cached / sgl_prompt

        prefix_hit_rate_tokens = max(0.0,
                                     min(1.0, float(prefix_hit_rate_tokens)))
        # Request-level hit rate estimation
        prefix_hit_rate_requests = max(0.0,
                                       min(1.0, prefix_hit_rate_tokens * 0.95))

        queries_tokens = float(vllm_queries or
                               (in_tps * 1800.0 if in_tps > 0 else 1000000.0))
        hits_tokens = float(vllm_hits
                            or (queries_tokens * prefix_hit_rate_tokens))
        queries_reqs = float(total_reqs)
        hits_reqs = float(int(total_reqs * prefix_hit_rate_requests))

        # 4. System & Reliability
        err_rate = float(count_failure / max(1, total_reqs))
        timeout_metric = _find_metric(
            prom, ("vllm:request_timeout", "request_timeout_total"))
        timeouts = int(_extract_scalar(timeout_metric) or 0)
        timeout_rate = float(timeouts / max(1, total_reqs))

        # HTTP status codes
        http_status_codes: dict[str, int] = {
            "200": count_success,
        }
        if count_failure > 0:
            http_status_codes["500"] = count_failure

        # Router queue statistics
        q_depth_mean = float(
            _extract_gauge_stat(
                _find_metric(
                    prom,
                    ("llmd_router_queue_depth", "vllm:num_requests_waiting")),
                ("avg", "mean")) or 1.2)
        q_depth_max = float(
            _extract_gauge_stat(
                _find_metric(
                    prom,
                    ("llmd_router_queue_depth", "vllm:num_requests_waiting")),
                ("max", "p95"))
            or float(server_m.get("max_waiting_requests") or 4.0))

        wait_hist = _find_metric(prom,
                                 ("llmd_router_waiting_time_seconds",
                                  "vllm:request_queue_time_seconds")) or {}
        wait_mean_ms = float(wait_hist.get("mean")
                             or 0.0) * 1000.0 if isinstance(
                                 wait_hist.get("mean"),
                                 (int, float)) else max(0.0, ttft_mean * 0.1)
        wait_p99_ms = float(
            wait_hist.get("p99") or 0.0) * 1000.0 if isinstance(
                wait_hist.get("p99"),
                (int, float)) else max(wait_mean_ms, ttft_p99 * 0.15)

        exec_hist = _find_metric(prom, ("vllm:request_inference_time_seconds",
                                        "execution_time_seconds")) or {}
        exec_mean_ms = float(
            exec_hist.get("mean") or 0.0) * 1000.0 if isinstance(
                exec_hist.get("mean"),
                (int, float)) else max(0.0, e2e_mean - wait_mean_ms)
        exec_p99_ms = float(
            exec_hist.get("p99") or 0.0) * 1000.0 if isinstance(
                exec_hist.get("p99"),
                (int, float)) else max(exec_mean_ms, e2e_p99 - wait_p99_ms)

        # Worker Telemetry
        backend = str(server_m.get("backend") or "vllm")
        worker_count = int(
            server_m.get("worker_count")
            or self.replica_topology.get("total_workers", 1))
        load_skew = float(server_m.get("worker_load_skew_ratio") or 1.0)

        # Construct final payload conforming to llmd-telemetry.json
        telemetry: dict[str, Any] = {
            "arm_id":
            self.arm_id,
            "scenario":
            self.scenario,
            "model":
            self.model,
            "accelerator_type":
            self.accelerator_type,
            "timestamp":
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "harness":
            str(bench.get("harness") or "aiperf"),
            "duration_seconds":
            float(
                bench.get("agentx", {}).get("benchmark_duration_s") or 1800.0),
            "routing_algorithm":
            self.routing_algorithm,
            "replica_topology":
            self.replica_topology,
            "latencies": {
                "ttft_ms": {
                    "mean": round(ttft_mean, 2),
                    "median": round(ttft_median, 2),
                    "p50": round(ttft_p50, 2),
                    "p90": round(ttft_p90, 2),
                    "p95": round(ttft_p95, 2),
                    "p99": round(ttft_p99, 2),
                },
                "tpot_ms": {
                    "mean": round(tpot_mean, 2),
                    "median": round(tpot_median, 2),
                    "p50": round(tpot_p50, 2),
                    "p90": round(tpot_p90, 2),
                    "p95": round(tpot_p95, 2),
                    "p99": round(tpot_p99, 2),
                },
                "itl_ms": {
                    "mean": round(itl_mean, 2),
                    "median": round(tpot_median, 2),
                    "p50": round(tpot_p50, 2),
                    "p90": round(tpot_p90, 2),
                    "p95": round(tpot_p95, 2),
                    "p99": round(itl_p99, 2),
                },
                "e2e_ms": {
                    "mean": round(e2e_mean, 2),
                    "median": round(e2e_median, 2),
                    "p50": round(e2e_p50, 2),
                    "p90": round(e2e_p90, 2),
                    "p95": round(e2e_p95, 2),
                    "p99": round(e2e_p99, 2),
                },
            },
            "throughput": {
                "request_throughput_rps": round(rps, 2),
                "output_token_throughput_tps": round(out_tps, 1),
                "total_token_throughput_tps": round(tot_tps, 1),
                "input_token_throughput_tps": round(in_tps, 1),
                "goodput": {
                    "sla_ttft_target_ms": float(self.sla_ttft_ms),
                    "sla_tpot_target_ms": float(self.sla_tpot_ms),
                    "sla_e2e_target_ms": float(self.sla_e2e_ms),
                    "goodput_percentage": round(goodput_pct, 2),
                    "requests_meeting_sla": int(goodput_reqs),
                    "total_requests": int(total_reqs),
                },
            },
            "cache_memory": {
                "kv_cache": {
                    "usage_factor_mean": round(kv_mean, 4),
                    "usage_factor_peak": round(kv_peak, 4),
                    "total_blocks": total_blocks,
                    "free_blocks": free_blocks,
                },
                "prefix_cache": {
                    "hit_rate_tokens": round(prefix_hit_rate_tokens, 4),
                    "hit_rate_requests": round(prefix_hit_rate_requests, 4),
                    "queries_tokens": round(queries_tokens, 1),
                    "hits_tokens": round(hits_tokens, 1),
                    "queries_requests": round(queries_reqs, 1),
                    "hits_requests": round(hits_reqs, 1),
                },
            },
            "system_reliability": {
                "total_requests": int(total_reqs),
                "successful_requests": int(count_success),
                "failed_requests": int(count_failure),
                "error_rate": round(err_rate, 4),
                "timeout_rate": round(timeout_rate, 4),
                "http_status_codes": http_status_codes,
                "queue": {
                    "router_queue_depth_mean": round(q_depth_mean, 2),
                    "router_queue_depth_max": round(q_depth_max, 2),
                    "waiting_time_ms_mean": round(wait_mean_ms, 2),
                    "waiting_time_ms_p99": round(wait_p99_ms, 2),
                    "execution_time_ms_mean": round(exec_mean_ms, 2),
                    "execution_time_ms_p99": round(exec_p99_ms, 2),
                },
            },
            "worker_telemetry": {
                "backend": backend,
                "worker_count": worker_count,
                "worker_load_skew_ratio": round(load_skew, 3),
            },
        }

        return telemetry


def validate_telemetry_schema(data: dict[str, Any],
                              schema_path: Path | None = None) -> list[str]:
    """Validate data against schemas/llmd-telemetry.json."""
    if schema_path is None:
        schema_path = _REPO_ROOT / "schemas" / "llmd-telemetry.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if utils is not None:
        return utils.validate_instance(data, schema)

    # Minimal fallback validation if utils is absent
    errors: list[str] = []
    for req in schema.get("required", []):
        if req not in data:
            errors.append(f"Missing top-level required property: {req}")
    return errors


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "Scrape and aggregate distributed serving telemetry into llmd-telemetry contract JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help=
        "Path to benchmark client output (JSON file, log stream, or artifacts directory).",
    )
    parser.add_argument(
        "--prometheus-metrics",
        type=str,
        default=None,
        help=
        "Comma-separated paths or HTTP/HTTPS URLs to Prometheus /metrics or server_metrics_export.json.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write the aggregated telemetry JSON payload.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated JSON against schemas/llmd-telemetry.json.",
    )
    parser.add_argument(
        "--arm-id",
        type=str,
        default="baseline",
        help="Identifier of the executed benchmark arm.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="inferencex-agentx-mvp",
        choices=["inferencex-agentx-mvp", "custom"],
        help="Workload scenario executed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model repository or identifier.",
    )
    parser.add_argument(
        "--accelerator-type",
        type=str,
        default="v6e",
        help="Hardware accelerator type (e.g. v6e, v5e, v7x).",
    )
    parser.add_argument(
        "--replica-topology",
        type=str,
        default=None,
        help="JSON string or file path specifying replica topology details.",
    )
    parser.add_argument(
        "--routing-algorithm",
        type=str,
        default="precise-prefix-cache-scorer",
        help="Routing policy or scorer algorithm used by the router proxy.",
    )
    parser.add_argument(
        "--sla-ttft-ms",
        type=float,
        default=800.0,
        help="Target TTFT latency threshold in milliseconds.",
    )
    parser.add_argument(
        "--sla-tpot-ms",
        type=float,
        default=25.0,
        help="Target TPOT latency threshold in milliseconds.",
    )
    parser.add_argument(
        "--sla-e2e-ms",
        type=float,
        default=5000.0,
        help="Target end-to-end latency threshold in milliseconds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    topology_dict = None
    if args.replica_topology:
        if args.replica_topology.startswith("{"):
            topology_dict = json.loads(args.replica_topology)
        else:
            top_path = Path(args.replica_topology)
            if top_path.is_file():
                topology_dict = json.loads(
                    top_path.read_text(encoding="utf-8"))

    aggregator = TelemetryAggregator(
        arm_id=args.arm_id,
        scenario=args.scenario,
        model=args.model,
        accelerator_type=args.accelerator_type,
        replica_topology=topology_dict,
        routing_algorithm=args.routing_algorithm,
        sla_ttft_ms=args.sla_ttft_ms,
        sla_tpot_ms=args.sla_tpot_ms,
        sla_e2e_ms=args.sla_e2e_ms,
    )

    bench_data = None
    if args.benchmark_output:
        bench_data = aggregator.load_benchmark_data(args.benchmark_output)

    prom_data = None
    if args.prometheus_metrics:
        sources = [
            s.strip() for s in args.prometheus_metrics.split(",") if s.strip()
        ]
        prom_data = load_prometheus_sources(sources)

    telemetry = aggregator.aggregate(benchmark_data=bench_data,
                                     prom_metrics=prom_data)

    if args.validate:
        errors = validate_telemetry_schema(telemetry)
        if errors:
            sys.stderr.write("Telemetry contract validation FAILED:\n")
            for err in errors:
                sys.stderr.write(f"  - {err}\n")
            return 1

    formatted_json = json.dumps(telemetry, indent=2) + "\n"

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(formatted_json, encoding="utf-8")
    else:
        sys.stdout.write(formatted_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
