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
"""Parse distributed serving benchmark Job logs or artifact directories.

Consumes either:
1. The stdout log stream from an ``autoconfig-benchmark`` Job (finding the
   ``---BENCHMARK RESULTS---`` delimiter emitted by ``benchmark.py`` and
   extracting the JSON result payloads that follow), or
2. An ``--artifacts-dir`` containing ``aiperf`` (``profile_export_aiperf.json``
   and ``server_metrics_export.json``) or ``inference-perf`` JSON files.

Normalizes ``aiperf`` and ``inference-perf`` client latency and throughput
shapes, joins per-worker Prometheus ``server_metrics_export.json`` telemetry
(prefix cache hit rate, KV cache occupancy, worker load skew, queue depth)
across vLLM and SGLang backends, and emits a Markdown summary table (plus an
optional ``--summary-json`` artifact).

Usage:
    python3 parse_bench_results.py --logs bench-logs.txt

    # With SLA comparison (milliseconds) and prefix cache hit rate gate:
    python3 parse_bench_results.py --logs bench-logs.txt \\
        --ttft-sla 800 --tpot-sla 25 --e2e-sla 5000 \\
        --prefix-hit-rate-min 0.70 --summary-json summary.json

Exit codes:
    0  parse succeeded, no SLAs supplied OR all SLAs met
    1  parse succeeded, one or more SLAs breached
    2  parse failed (no delimiter, no JSON, malformed input)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_DELIMITER_RE = re.compile(r"^---BENCHMARK RESULTS.*?---$", re.MULTILINE)


def _read_input(logs_path: str | None) -> str:
    if logs_path:
        with open(logs_path, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _extract_json_blocks(text: str) -> list[dict[str, Any]]:
    """Find the delimiter, then pull every top-level JSON object that follows."""
    match = _DELIMITER_RE.search(text)
    if not match:
        return []

    tail = text[match.end():]
    blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(tail):
        if tail[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        start = i
        while i < len(tail):
            ch = tail[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = tail[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                blocks.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        i += 1
                        break
            i += 1
        else:
            break
    return blocks


def _load_blocks_from_dir(artifacts_dir: str) -> list[dict[str, Any]]:
    """Load JSON blocks from an artifacts directory in deterministic order."""
    root = Path(artifacts_dir)
    if not root.exists():
        return []
    blocks: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if "warmup" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                blocks.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return blocks


def _is_prometheus_key(key: str) -> bool:
    return key.startswith(("vllm:", "sglang:", "dynamo_"))


def _detect_format(block: dict[str, Any]) -> str | None:
    """Return 'inference-perf', 'aiperf', 'aiperf-server', or None."""
    if isinstance(block.get("successes"), dict) and isinstance(
            block["successes"].get("latency"), dict):
        return "inference-perf"

    metrics_dict = block.get("metrics")
    if isinstance(metrics_dict, dict) and any(
            _is_prometheus_key(str(k)) for k in metrics_dict):
        return "aiperf-server"
    if any(_is_prometheus_key(str(k)) for k in block):
        return "aiperf-server"

    records = block.get("records") if isinstance(block.get("records"),
                                                 dict) else block
    if isinstance(records, dict) and (
            isinstance(records.get("time_to_first_token"), dict)
            or isinstance(records.get("request_latency"), dict)
            or isinstance(records.get("inter_token_latency"), dict)
            or isinstance(records.get("time_per_output_token"), dict)):
        return "aiperf"

    return None


def _normalize_inference_perf(block: dict[str, Any]) -> dict[str, Any] | None:
    """Pull common metrics from an inference-perf summary JSON."""
    successes = block.get("successes") or {}
    latency = successes.get("latency") or {}
    throughput = successes.get("throughput") or {}
    failures = block.get("failures") or {}

    def pct(metric: str, key: str) -> float | None:
        d = latency.get(metric) or {}
        return d.get(key)

    def to_ms(v: float | None) -> float | None:
        return v * 1000.0 if v is not None else None

    return {
        "harness": "inference-perf",
        "count_success": successes.get("count"),
        "count_failure": failures.get("count", 0),
        "metrics_ms": {
            "request_latency": {
                "mean": to_ms(pct("request_latency", "mean")),
                "p50": to_ms(pct("request_latency", "median")),
                "p95": to_ms(pct("request_latency", "p95")),
                "p99": to_ms(pct("request_latency", "p99")),
            },
            "ttft": {
                "mean": to_ms(pct("time_to_first_token", "mean")),
                "p50": to_ms(pct("time_to_first_token", "median")),
                "p95": to_ms(pct("time_to_first_token", "p95")),
                "p99": to_ms(pct("time_to_first_token", "p99")),
            },
            "tpot": {
                "mean": to_ms(pct("time_per_output_token", "mean")),
                "p50": to_ms(pct("time_per_output_token", "median")),
                "p95": to_ms(pct("time_per_output_token", "p95")),
                "p99": to_ms(pct("time_per_output_token", "p99")),
            },
        },
        "throughput": {
            "requests_per_sec": throughput.get("requests_per_sec"),
            "input_tokens_per_sec": throughput.get("input_tokens_per_sec"),
            "output_tokens_per_sec": throughput.get("output_tokens_per_sec"),
            "total_tokens_per_sec": throughput.get("total_tokens_per_sec"),
        },
    }


def _to_ms_unit(val: float | int | None, unit: str | None) -> float | None:
    """Convert an aiperf latency statistic into milliseconds."""
    if val is None:
        return None
    fval = float(val)
    u = (unit or "ms").strip().lower()
    if u in ("s", "sec", "second", "seconds"):
        return fval * 1000.0
    if u in ("us", "usec", "microsecond", "microseconds"):
        return fval * 1e-3
    if u in ("ns", "nsec", "nanosecond", "nanoseconds"):
        return fval * 1e-6
    return fval


def _extract_aiperf_stat_ms(
        records: dict[str, Any],
        metric_names: tuple[str, ...]) -> dict[str, float | None]:
    """Extract mean/p50/p95/p99 in milliseconds from an aiperf metric entry."""
    for name in metric_names:
        m = records.get(name)
        if isinstance(m, dict):
            unit = m.get("unit")
            mean_raw = m.get("avg") if m.get("avg") is not None else m.get(
                "mean")
            p50_raw = (m.get("p50")
                       if m.get("p50") is not None else m.get("median"))
            return {
                "mean": _to_ms_unit(mean_raw, unit),
                "p50": _to_ms_unit(p50_raw, unit),
                "p95": _to_ms_unit(m.get("p95"), unit),
                "p99": _to_ms_unit(m.get("p99"), unit),
            }
    return {"mean": None, "p50": None, "p95": None, "p99": None}


def _extract_aiperf_scalar(records: dict[str, Any], top: dict[str, Any],
                           keys: tuple[str, ...]) -> float | None:
    """Extract a scalar throughput or count value from aiperf records."""
    for key in keys:
        for container in (records, top):
            val = container.get(key)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                for stat_key in ("avg", "mean", "value", "total", "count",
                                 "p50"):
                    sval = val.get(stat_key)
                    if isinstance(sval, (int, float)):
                        return float(sval)
    return None


def _extract_metric_delta(metric_obj: Any) -> float | None:
    """Extract total counter delta across one or more worker series."""
    if isinstance(metric_obj, (int, float)):
        return float(metric_obj)
    if not isinstance(metric_obj, dict):
        return None
    if isinstance(metric_obj.get("delta"), (int, float)):
        return float(metric_obj["delta"])
    if isinstance(metric_obj.get("total_delta"), (int, float)):
        return float(metric_obj["total_delta"])
    if isinstance(metric_obj.get("value"), (int, float)):
        return float(metric_obj["value"])
    series = metric_obj.get("series") or metric_obj.get("endpoints")
    if isinstance(series, list) and series:
        deltas = [
            float(s["delta"]) for s in series
            if isinstance(s, dict) and isinstance(s.get("delta"), (int, float))
        ]
        if deltas:
            return sum(deltas)
    if isinstance(series, dict) and series:
        deltas = [
            float(v["delta"]) if isinstance(v, dict)
            and isinstance(v.get("delta"), (int, float)) else float(v)
            for v in series.values()
            if (isinstance(v, dict) and isinstance(v.get("delta"), (
                int, float))) or isinstance(v, (int, float))
        ]
        if deltas:
            return sum(deltas)
    return None


def _extract_metric_per_worker_deltas(metric_obj: Any) -> list[float]:
    """Extract per-worker counter deltas to evaluate router load skew."""
    if not isinstance(metric_obj, dict):
        return []
    series = metric_obj.get("series") or metric_obj.get("endpoints")
    if isinstance(series, list):
        return [
            float(s["delta"]) for s in series
            if isinstance(s, dict) and isinstance(s.get("delta"), (int, float))
        ]
    if isinstance(series, dict):
        out: list[float] = []
        for v in series.values():
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, dict) and isinstance(v.get("delta"),
                                                    (int, float)):
                out.append(float(v["delta"]))
        return out
    return []


def _extract_gauge_stat(metric_obj: Any,
                        stat_keys: tuple[str, ...]) -> float | None:
    """Extract a gauge statistic (e.g. p95, mean, max) across worker series."""
    if isinstance(metric_obj, (int, float)):
        return float(metric_obj)
    if not isinstance(metric_obj, dict):
        return None
    for k in stat_keys:
        if isinstance(metric_obj.get(k), (int, float)):
            return float(metric_obj[k])
    series = metric_obj.get("series") or metric_obj.get("endpoints")
    vals: list[float] = []
    if isinstance(series, list):
        for s in series:
            if isinstance(s, dict):
                for k in stat_keys:
                    if isinstance(s.get(k), (int, float)):
                        vals.append(float(s[k]))
                        break
    elif isinstance(series, dict):
        for s in series.values():
            if isinstance(s, (int, float)):
                vals.append(float(s))
            elif isinstance(s, dict):
                for k in stat_keys:
                    if isinstance(s.get(k), (int, float)):
                        vals.append(float(s[k]))
                        break
    if not vals:
        return None
    if "max" in stat_keys or "p95" in stat_keys:
        return max(vals)
    return sum(vals) / len(vals)


def _find_prom_metric(metrics_map: dict[str, Any],
                      prefixes: tuple[str, ...]) -> Any:
    """Find a Prometheus metric entry matching any prefix (handling :total suffixes)."""
    for prefix in prefixes:
        if prefix in metrics_map:
            return metrics_map[prefix]
        if f"{prefix}_total" in metrics_map:
            return metrics_map[f"{prefix}_total"]
    for k, v in metrics_map.items():
        if any(str(k).startswith(p) for p in prefixes):
            return v
    return None


def _parse_server_metrics(
        server_block: dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse Prometheus server metrics from vLLM, SGLang, and Dynamo."""
    if not server_block or not isinstance(server_block, dict):
        return None
    metrics_map = (server_block.get("metrics") if isinstance(
        server_block.get("metrics"), dict) else server_block)
    if not isinstance(metrics_map, dict):
        return None

    vllm_hits = _extract_metric_delta(
        _find_prom_metric(metrics_map, ("vllm:prefix_cache_hits", )))
    vllm_queries = _extract_metric_delta(
        _find_prom_metric(metrics_map, ("vllm:prefix_cache_queries", )))
    sglang_cached = _extract_metric_delta(
        _find_prom_metric(metrics_map, ("sglang:cached_tokens", )))
    sglang_prompt = _extract_metric_delta(
        _find_prom_metric(metrics_map, ("sglang:prompt_tokens", )))

    prefix_hit_rate: float | None = None
    backend = "unknown"
    if vllm_queries is not None and vllm_queries > 0 and vllm_hits is not None:
        prefix_hit_rate = max(0.0, min(1.0, vllm_hits / vllm_queries))
        backend = "vllm"
    elif (sglang_prompt is not None and sglang_prompt > 0
          and sglang_cached is not None):
        prefix_hit_rate = max(0.0, min(1.0, sglang_cached / sglang_prompt))
        backend = "sglang"

    kv_metric = _find_prom_metric(
        metrics_map,
        (
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "sglang:token_usage",
        ),
    )
    kv_p95 = _extract_gauge_stat(kv_metric, ("p95", "max", "avg", "mean"))
    kv_mean = _extract_gauge_stat(kv_metric, ("avg", "mean", "p50"))
    if backend == "unknown" and kv_metric is not None:
        backend = ("sglang" if _find_prom_metric(
            metrics_map, ("sglang:token_usage", )) is not None else "vllm")

    waiting_metric = _find_prom_metric(
        metrics_map, ("vllm:num_requests_waiting", "sglang:num_queue_reqs"))
    max_waiting = _extract_gauge_stat(waiting_metric,
                                      ("max", "p95", "avg", "mean"))

    req_metric = _find_prom_metric(
        metrics_map,
        (
            "vllm:request_success",
            "vllm:prefix_cache_queries",
            "sglang:prompt_tokens",
        ),
    )
    worker_deltas = [
        d for d in _extract_metric_per_worker_deltas(req_metric) if d >= 0
    ]
    load_skew_ratio: float | None = None
    if worker_deltas:
        mean_load = sum(worker_deltas) / len(worker_deltas)
        if mean_load > 0:
            load_skew_ratio = max(worker_deltas) / mean_load

    if (prefix_hit_rate is None and kv_p95 is None and max_waiting is None
            and load_skew_ratio is None):
        return None

    return {
        "backend": backend,
        "prefix_cache_hit_rate": prefix_hit_rate,
        "kv_cache_usage_p95": kv_p95,
        "kv_cache_usage_mean": kv_mean,
        "max_waiting_requests": max_waiting,
        "worker_load_skew_ratio": load_skew_ratio,
        "worker_count": len(worker_deltas) if worker_deltas else None,
    }


def _normalize_aiperf(
    client_block: dict[str, Any],
    server_block: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Normalize an aiperf profile_export_aiperf.json block and optional server metrics."""
    records = (client_block.get("records") if isinstance(
        client_block.get("records"), dict) else client_block)
    if not isinstance(records, dict):
        return None

    req_lat = _extract_aiperf_stat_ms(records,
                                      ("request_latency", "e2e_latency"))
    ttft = _extract_aiperf_stat_ms(records, ("time_to_first_token", "ttft"))
    tpot = _extract_aiperf_stat_ms(
        records,
        ("time_per_output_token", "inter_token_latency", "tpot", "itl"),
    )

    rps = _extract_aiperf_scalar(records, client_block,
                                 ("request_throughput", "requests_per_sec"))
    out_tps = _extract_aiperf_scalar(
        records,
        client_block,
        ("output_token_throughput", "output_tokens_per_sec"),
    )
    in_tps = _extract_aiperf_scalar(
        records,
        client_block,
        ("input_token_throughput", "input_tokens_per_sec"),
    )
    tot_tps = _extract_aiperf_scalar(
        records,
        client_block,
        ("total_token_throughput", "total_tokens_per_sec"),
    )
    if tot_tps is None and in_tps is not None and out_tps is not None:
        tot_tps = in_tps + out_tps

    raw_success = _extract_aiperf_scalar(
        records,
        client_block,
        ("request_count", "successful_requests", "completed_requests"),
    )
    raw_failure = _extract_aiperf_scalar(
        records,
        client_block,
        ("error_request_count", "failed_requests", "errored_requests"),
    )
    count_success = int(raw_success) if raw_success is not None else 0
    count_failure = int(raw_failure) if raw_failure is not None else 0

    embedded_server = (client_block.get("server_metrics") if isinstance(
        client_block.get("server_metrics"), dict) else None)
    server_summary = _parse_server_metrics(server_block or embedded_server)

    result: dict[str, Any] = {
        "harness": "aiperf",
        "count_success": count_success,
        "count_failure": count_failure,
        "metrics_ms": {
            "request_latency": req_lat,
            "ttft": ttft,
            "tpot": tpot,
        },
        "throughput": {
            "requests_per_sec": rps,
            "input_tokens_per_sec": in_tps,
            "output_tokens_per_sec": out_tps,
            "total_tokens_per_sec": tot_tps,
        },
    }
    if server_summary is not None:
        result["server_metrics"] = server_summary
    return result


def _normalize(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Try each block until one matches a known format, joining aiperf server metrics when present."""
    aiperf_clients: list[dict[str, Any]] = []
    aiperf_servers: list[dict[str, Any]] = []

    for block in blocks:
        fmt = _detect_format(block)
        if fmt == "aiperf":
            aiperf_clients.append(block)
        elif fmt == "aiperf-server":
            aiperf_servers.append(block)

    if aiperf_clients:
        client_block = aiperf_clients[-1]
        server_block = aiperf_servers[-1] if aiperf_servers else None
        return _normalize_aiperf(client_block, server_block)

    for block in blocks:
        fmt = _detect_format(block)
        if fmt == "inference-perf":
            return _normalize_inference_perf(block)
    return None


def _fmt(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def render_markdown(summary: dict[str, Any],
                    slas: dict[str, float] | None = None) -> tuple[str, bool]:
    """Render the summary as a markdown report. Returns (text, any_sla_breach)."""
    m = summary["metrics_ms"]
    tp = summary["throughput"]
    any_breach = False
    lines: list[str] = []

    lines.append(f"**Benchmark harness:** {summary['harness']}")
    fail_count = summary.get("count_failure") or 0
    lines.append(
        f"**Requests:** {summary.get('count_success', 'n/a')} successful, "
        f"{fail_count} failed")
    lines.append("")
    lines.append("| Metric | mean | p50 | p95 | p99 |")
    lines.append("|---|---|---|---|---|")
    for label, key in [
        ("Request latency (ms)", "request_latency"),
        ("TTFT (ms)", "ttft"),
        ("TPOT (ms)", "tpot"),
    ]:
        d = m.get(key) or {}
        lines.append(
            f"| {label} | {_fmt(d.get('mean'))} | {_fmt(d.get('p50'))} | "
            f"{_fmt(d.get('p95'))} | {_fmt(d.get('p99'))} |")
    lines.append("")
    rps = tp.get("requests_per_sec")
    in_tps = tp.get("input_tokens_per_sec")
    out_tps = tp.get("output_tokens_per_sec")
    tot_tps = tp.get("total_tokens_per_sec")
    lines.append(f"**Throughput:** {_fmt(rps, 2)} req/s, "
                 f"{_fmt(in_tps, 0)} input tokens/s, "
                 f"{_fmt(out_tps, 0)} output tokens/s" +
                 (f", {_fmt(tot_tps, 0)} total tokens/s"
                  if tot_tps is not None else ""))

    srv = summary.get("server_metrics")
    if isinstance(srv, dict):
        lines.append("")
        lines.append("**Distributed Router & Worker Telemetry:**")
        hit_rate = srv.get("prefix_cache_hit_rate")
        if hit_rate is not None:
            lines.append(
                f"- Prefix cache hit rate ({srv.get('backend', 'worker')}): {hit_rate * 100:.1f}%"
            )
        kv_p95 = srv.get("kv_cache_usage_p95")
        if kv_p95 is not None:
            lines.append(f"- KV cache occupancy (p95): {kv_p95 * 100:.1f}%")
        skew = srv.get("worker_load_skew_ratio")
        if skew is not None:
            lines.append(f"- Worker load skew ratio (max/mean): {skew:.2f}x")
        waiting = srv.get("max_waiting_requests")
        if waiting is not None:
            lines.append(f"- Max queued requests: {waiting:.0f}")

    if slas:
        lines.append("")
        lines.append("**SLA validation (p95):**")
        for label, sla_key, metric_key in [
            ("TTFT", "ttft_ms", "ttft"),
            ("TPOT", "tpot_ms", "tpot"),
            ("End-to-end", "e2e_ms", "request_latency"),
        ]:
            target = slas.get(sla_key)
            if target is None:
                continue
            measured = (m.get(metric_key) or {}).get("p95")
            if measured is None:
                lines.append(f"- {label}: target {target} ms, measured n/a")
                continue
            if measured <= target:
                lines.append(
                    f"- {label}: {measured:.0f} ms (target: {target} ms) [PASS]"
                )
            else:
                breach_pct = (measured - target) / target * 100
                lines.append(
                    f"- {label}: {measured:.0f} ms (target: {target} ms) "
                    f"[EXCEEDS by {breach_pct:.0f}%]")
                any_breach = True

        min_hit = slas.get("prefix_hit_rate_min")
        if min_hit is not None:
            measured_hit = (srv.get("prefix_cache_hit_rate") if isinstance(
                srv, dict) else None)
            if measured_hit is None:
                lines.append(
                    f"- Prefix cache hit rate: target >= {min_hit * 100:.1f}%, measured n/a [FAIL]"
                )
                any_breach = True
            elif measured_hit >= min_hit:
                lines.append(
                    f"- Prefix cache hit rate: {measured_hit * 100:.1f}% "
                    f"(target: >= {min_hit * 100:.1f}%) [PASS]")
            else:
                lines.append(
                    f"- Prefix cache hit rate: {measured_hit * 100:.1f}% "
                    f"(target: >= {min_hit * 100:.1f}%) [BELOW TARGET]")
                any_breach = True

    goodput = (tp.get("output_tokens_per_sec")
               or 0.0) if not any_breach else 0.0
    summary["goodput_tokens_per_sec"] = goodput

    return "\n".join(lines) + "\n", any_breach


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--logs",
        help=
        "Path to a saved logs file. If omitted and --artifacts-dir is not set, reads from stdin.",
    )
    parser.add_argument(
        "--artifacts-dir",
        help=
        "Path to a directory of benchmark JSON artifacts (e.g. profile_export_aiperf.json + server_metrics_export.json).",
    )
    parser.add_argument(
        "--ttft-sla",
        type=float,
        help="TTFT p95 target in ms.",
    )
    parser.add_argument(
        "--tpot-sla",
        type=float,
        help="TPOT p95 target in ms.",
    )
    parser.add_argument(
        "--e2e-sla",
        type=float,
        help="End-to-end request latency p95 target in ms.",
    )
    parser.add_argument(
        "--prefix-hit-rate-min",
        type=float,
        help="Minimum acceptable worker prefix cache hit rate (0.0 to 1.0).",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write the normalized JSON summary.",
    )
    args = parser.parse_args(argv)

    if args.artifacts_dir:
        blocks = _load_blocks_from_dir(args.artifacts_dir)
    else:
        text = _read_input(args.logs)
        blocks = _extract_json_blocks(text)

    if not blocks:
        print(
            "error: no JSON blocks found after `---BENCHMARK RESULTS---` delimiter "
            "(or in --artifacts-dir). Verify the benchmark Job completed and check "
            "the saved log output.",
            file=sys.stderr,
        )
        return 2

    summary = _normalize(blocks)
    if summary is None:
        print(
            "error: extracted JSON blocks but none matched aiperf or inference-perf "
            "format. Run with --logs to a file and inspect manually.",
            file=sys.stderr,
        )
        return 2

    slas: dict[str, float] = {}
    if args.ttft_sla is not None:
        slas["ttft_ms"] = args.ttft_sla
    if args.tpot_sla is not None:
        slas["tpot_ms"] = args.tpot_sla
    if args.e2e_sla is not None:
        slas["e2e_ms"] = args.e2e_sla
    if args.prefix_hit_rate_min is not None:
        slas["prefix_hit_rate_min"] = args.prefix_hit_rate_min

    report, any_breach = render_markdown(summary, slas or None)
    summary["sla_breach"] = any_breach
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(report)
    return 1 if any_breach else 0


if __name__ == "__main__":
    sys.exit(main())
