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
"""Tests for llmd-telemetry schema validation and telemetry aggregator."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PERF_SCRIPTS_DIR = _PACKAGE_ROOT / "scripts" / "tpu-performance-optimization"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from unittest import mock  # noqa: E402

import llmd_telemetry  # noqa: E402

from framework import compiler, utils  # noqa: E402


class TestLlmdTelemetry(unittest.TestCase):
    """Unit and regression tests for llmd_telemetry.py and schemas/llmd-telemetry.json."""

    def setUp(self) -> None:
        self.schema_path = _PACKAGE_ROOT / "schemas" / "llmd-telemetry.json"
        self.assertTrue(self.schema_path.is_file(),
                        f"Schema missing: {self.schema_path}")
        self.schema = utils.load_json(self.schema_path)

    def test_schema_definition_is_structurally_valid(self) -> None:
        errors = utils.validate_schema_definition(self.schema)
        self.assertEqual(errors, [],
                         f"Schema definition validation failed: {errors}")

    def test_valid_telemetry_instance_passes_validation(self) -> None:
        aggregator = llmd_telemetry.TelemetryAggregator(
            arm_id="arm-001",
            scenario="inferencex-agentx-mvp",
            model="Qwen/Qwen3-32B",
            accelerator_type="v6e",
            replica_topology={
                "mode": "homogeneous",
                "total_workers": 2,
                "chips_per_replica": 8,
                "topology_shape": "2x2x2",
            },
            routing_algorithm="precise-prefix-cache-scorer",
            sla_ttft_ms=800.0,
            sla_tpot_ms=25.0,
            sla_e2e_ms=5000.0,
        )
        telemetry = aggregator.aggregate()
        errors = utils.validate_instance(telemetry, self.schema)
        self.assertEqual(errors, [], f"Instance validation failed: {errors}")

    def test_schema_rejects_missing_required_properties(self) -> None:
        aggregator = llmd_telemetry.TelemetryAggregator()
        telemetry = aggregator.aggregate()

        for req in [
                "arm_id",
                "scenario",
                "model",
                "accelerator_type",
                "timestamp",
                "latencies",
                "throughput",
                "cache_memory",
                "system_reliability",
        ]:
            bad_telemetry = dict(telemetry)
            del bad_telemetry[req]
            errors = utils.validate_instance(bad_telemetry, self.schema)
            self.assertTrue(
                any("required" in e.lower() and req in e for e in errors),
                f"Expected validation error for missing {req}, got: {errors}")

    def test_schema_rejects_out_of_range_values(self) -> None:
        aggregator = llmd_telemetry.TelemetryAggregator()
        telemetry = aggregator.aggregate()

        # Cache usage factor must be <= 1.0
        telemetry["cache_memory"]["kv_cache"]["usage_factor_mean"] = 1.5
        errors = utils.validate_instance(telemetry, self.schema)
        self.assertTrue(
            any("maximum" in e.lower() or "additional" in e.lower()
                for e in errors),
            f"Expected error for usage_factor_mean > 1.0, got: {errors}")

    def test_parse_prometheus_text_format(self) -> None:
        prom_text = (
            "# HELP vllm:gpu_cache_usage_factor Accelerator KV cache usage factor\n"
            "# TYPE vllm:gpu_cache_usage_factor gauge\n"
            'vllm:gpu_cache_usage_factor{model="Qwen/Qwen3-32B"} 0.68\n'
            "# HELP vllm:prefix_cache_hits_total Total prefix hits\n"
            "# TYPE vllm:prefix_cache_hits_total counter\n"
            'vllm:prefix_cache_hits_total{model="Qwen/Qwen3-32B"} 9200\n'
            'vllm:prefix_cache_queries_total{model="Qwen/Qwen3-32B"} 10000\n'
            'llmd_router_queue_depth 3.0\n'
            'llmd_router_request_latencies_bucket{le="100.0"} 25\n'
            'llmd_router_request_latencies_bucket{le="500.0"} 80\n'
            'llmd_router_request_latencies_bucket{le="1000.0"} 95\n'
            'llmd_router_request_latencies_bucket{le="+Inf"} 100\n'
            'llmd_router_request_latencies_sum 32500.0\n'
            'llmd_router_request_latencies_count 100\n')

        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as f:
            f.write(prom_text)
            tmp_path = f.name

        metrics = llmd_telemetry.load_prometheus_sources([tmp_path])
        pathlib.Path(tmp_path).unlink(missing_ok=True)

        self.assertIn("vllm:gpu_cache_usage_factor", metrics)
        self.assertAlmostEqual(metrics["vllm:gpu_cache_usage_factor"]["value"],
                               0.68)
        self.assertAlmostEqual(
            metrics["vllm:prefix_cache_hits_total"]["value"], 9200.0)

        # Verify histogram quantile calculation
        self.assertIn("llmd_router_request_latencies", metrics)
        hist = metrics["llmd_router_request_latencies"]
        self.assertGreater(hist["p50"], 100.0)
        self.assertLessEqual(hist["p50"], 500.0)
        self.assertAlmostEqual(hist["mean"], 325.0)

    def test_parse_prometheus_json_format(self) -> None:
        prom_json = {
            "metrics": {
                "vllm:prefix_cache_hits": {
                    "delta": 8500
                },
                "vllm:prefix_cache_queries": {
                    "delta": 10000
                },
                "vllm:kv_cache_usage_perc": {
                    "avg": 0.62,
                    "p95": 0.78,
                    "max": 0.85
                },
                "vllm:num_requests_waiting": {
                    "avg": 1.5,
                    "max": 4.0
                },
                "vllm:request_success": {
                    "series": [
                        {
                            "endpoint": "http://10.0.0.1:8000/metrics",
                            "delta": 100
                        },
                        {
                            "endpoint": "http://10.0.0.2:8000/metrics",
                            "delta": 95
                        },
                    ]
                },
            }
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            f.write(json.dumps(prom_json))
            tmp_path = f.name

        metrics = llmd_telemetry.load_prometheus_sources([tmp_path])
        pathlib.Path(tmp_path).unlink(missing_ok=True)

        self.assertIn("vllm:prefix_cache_hits", metrics)
        self.assertEqual(metrics["vllm:prefix_cache_hits"]["delta"], 8500)
        self.assertEqual(metrics["vllm:kv_cache_usage_perc"]["p95"], 0.78)

    def test_aggregate_aiperf_client_and_server_metrics(self) -> None:
        client_data = {
            "harness": "aiperf",
            "count_success": 200,
            "count_failure": 2,
            "metrics_ms": {
                "request_latency": {
                    "mean": 1200.0,
                    "p50": 1100.0,
                    "p90": 1600.0,
                    "p95": 1800.0,
                    "p99": 2100.0
                },
                "ttft": {
                    "mean": 250.0,
                    "p50": 200.0,
                    "p90": 400.0,
                    "p95": 450.0,
                    "p99": 600.0
                },
                "tpot": {
                    "mean": 14.0,
                    "p50": 13.5,
                    "p90": 18.0,
                    "p95": 19.0,
                    "p99": 22.0
                },
            },
            "throughput": {
                "requests_per_sec": 12.5,
                "input_tokens_per_sec": 25000.0,
                "output_tokens_per_sec": 3200.0,
                "total_tokens_per_sec": 28200.0,
            },
            "server_metrics": {
                "backend": "vllm",
                "prefix_cache_hit_rate": 0.85,
                "kv_cache_usage_mean": 0.62,
                "kv_cache_usage_p95": 0.78,
                "max_waiting_requests": 3.0,
                "worker_load_skew_ratio": 1.05,
                "worker_count": 2,
            },
        }

        aggregator = llmd_telemetry.TelemetryAggregator(
            arm_id="arm-prefix-scorer",
            scenario="inferencex-agentx-mvp",
            model="Qwen/Qwen3-32B",
            accelerator_type="v6e",
            sla_ttft_ms=800.0,
            sla_tpot_ms=25.0,
        )

        telemetry = aggregator.aggregate(benchmark_data=client_data)
        self.assertEqual(telemetry["arm_id"], "arm-prefix-scorer")
        self.assertEqual(telemetry["latencies"]["ttft_ms"]["p95"], 450.0)
        self.assertEqual(telemetry["latencies"]["tpot_ms"]["p95"], 19.0)
        self.assertEqual(telemetry["throughput"]["request_throughput_rps"],
                         12.5)
        self.assertEqual(
            telemetry["cache_memory"]["prefix_cache"]["hit_rate_tokens"], 0.85)
        self.assertEqual(
            telemetry["cache_memory"]["kv_cache"]["usage_factor_peak"], 0.78)
        self.assertEqual(telemetry["system_reliability"]["total_requests"],
                         202)
        self.assertEqual(telemetry["system_reliability"]["failed_requests"], 2)
        self.assertAlmostEqual(telemetry["system_reliability"]["error_rate"],
                               2 / 202,
                               places=4)

        # Validate against schema
        errs = utils.validate_instance(telemetry, self.schema)
        self.assertEqual(errs, [])

    def test_goodput_calculation_with_sla_breach(self) -> None:
        client_data = {
            "harness": "aiperf",
            "count_success": 100,
            "count_failure": 0,
            "metrics_ms": {
                # TTFT p95 of 1200ms breaches 800ms SLA
                "ttft": {
                    "p50": 600.0,
                    "p90": 1000.0,
                    "p95": 1200.0,
                    "p99": 1500.0
                },
                "tpot": {
                    "mean": 20.0,
                    "median": 20.0,
                    "p90": 22.0,
                    "p99": 24.0
                },
                "request_latency": {
                    "p50": 1500.0,
                    "p90": 2000.0,
                    "p99": 2500.0
                },
            },
            "throughput": {
                "requests_per_sec": 10.0,
                "output_tokens_per_sec": 2000.0
            },
        }

        aggregator = llmd_telemetry.TelemetryAggregator(
            sla_ttft_ms=800.0,
            sla_tpot_ms=25.0,
        )
        telemetry = aggregator.aggregate(benchmark_data=client_data)
        goodput = telemetry["throughput"]["goodput"]
        self.assertLess(goodput["goodput_percentage"], 100.0)
        self.assertLess(goodput["requests_meeting_sla"], 100)

    def test_cli_execution_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            bench_file = tmp_path / "bench.json"
            prom_file = tmp_path / "server.json"
            out_file = tmp_path / "telemetry.json"

            bench_file.write_text(
                json.dumps({
                    "records": {
                        "request_latency": {
                            "p50": 1000.0,
                            "p95": 1500.0,
                            "p99": 1800.0,
                            "avg": 1100.0
                        },
                        "time_to_first_token": {
                            "p50": 150.0,
                            "p95": 250.0,
                            "p99": 350.0,
                            "avg": 160.0
                        },
                        "time_per_output_token": {
                            "p50": 15.0,
                            "p95": 18.0,
                            "p99": 21.0,
                            "avg": 15.5
                        },
                        "request_throughput": {
                            "avg": 8.0
                        },
                        "output_token_throughput": {
                            "avg": 2500.0
                        },
                        "request_count": {
                            "count": 150
                        },
                        "error_request_count": {
                            "count": 0
                        },
                    }
                }),
                encoding="utf-8",
            )

            prom_file.write_text(
                json.dumps({
                    "vllm:gpu_cache_usage_factor": 0.65,
                    "vllm:prefix_cache_hits_total": 5000,
                    "vllm:prefix_cache_queries_total": 7000,
                }),
                encoding="utf-8",
            )

            exit_code = llmd_telemetry.main([
                "--benchmark-output",
                str(bench_file),
                "--prometheus-metrics",
                str(prom_file),
                "--output-json",
                str(out_file),
                "--arm-id",
                "arm-epp-utilization",
                "--scenario",
                "inferencex-agentx-mvp",
                "--model",
                "Qwen/Qwen3-32B",
                "--accelerator-type",
                "v6e",
                "--validate",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.is_file())

            result_json = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(result_json["arm_id"], "arm-epp-utilization")
            self.assertEqual(result_json["scenario"], "inferencex-agentx-mvp")
            self.assertEqual(result_json["accelerator_type"], "v6e")
            self.assertEqual(
                result_json["throughput"]["request_throughput_rps"], 8.0)

            # Re-verify against schema
            errs = utils.validate_instance(result_json, self.schema)
            self.assertEqual(errs, [])

    def test_schema_is_registered_by_compiler(self) -> None:
        manifest_path = (_PACKAGE_ROOT / "package.release.json" if
                         (_PACKAGE_ROOT / "package.release.json").is_file()
                         else _PACKAGE_ROOT / "package.json")
        manifest = compiler.resolve_manifest_composition(
            _PACKAGE_ROOT, utils.load_json(manifest_path))
        declared_schemas = set(manifest.get("schemas", []))
        self.assertIn("schemas/llmd-telemetry.json", declared_schemas)

    def test_aggregate_inference_perf_data(self) -> None:
        inf_perf_block = {
            "successes": {
                "count": 500,
                "latency": {
                    "request_latency": {
                        "mean": 1.25,
                        "median": 1.10,
                        "p95": 1.80,
                        "p99": 2.20
                    },
                    "time_to_first_token": {
                        "mean": 0.22,
                        "median": 0.18,
                        "p95": 0.35,
                        "p99": 0.50
                    },
                    "time_per_output_token": {
                        "mean": 0.015,
                        "median": 0.014,
                        "p95": 0.020,
                        "p99": 0.025
                    },
                },
                "throughput": {
                    "requests_per_sec": 15.0,
                    "input_tokens_per_sec": 30000.0,
                    "output_tokens_per_sec": 3500.0,
                    "total_tokens_per_sec": 33500.0,
                },
            },
            "failures": {
                "count": 0
            },
        }

        aggregator = llmd_telemetry.TelemetryAggregator(
            arm_id="arm-inference-perf",
            scenario="custom",
        )
        telemetry = aggregator.aggregate(
            benchmark_data=aggregator._normalize_generic_client(
                inf_perf_block))
        self.assertEqual(telemetry["arm_id"], "arm-inference-perf")
        self.assertEqual(telemetry["scenario"], "custom")
        self.assertEqual(telemetry["throughput"]["request_throughput_rps"],
                         15.0)
        self.assertEqual(telemetry["latencies"]["ttft_ms"]["p50"], 180.0)
        errs = utils.validate_instance(telemetry, self.schema)
        self.assertEqual(errs, [])

    def test_fetch_prometheus_http_mock(self) -> None:
        sample_prom = ("# TYPE vllm:gpu_cache_usage_factor gauge\n"
                       "vllm:gpu_cache_usage_factor 0.74\n")
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = sample_prom.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            metrics = llmd_telemetry.load_prometheus_sources(
                ["http://10.0.0.1:8000/metrics"])
            self.assertIn("vllm:gpu_cache_usage_factor", metrics)
            self.assertEqual(metrics["vllm:gpu_cache_usage_factor"]["value"],
                             0.74)

    def test_error_handling_nonexistent_files(self) -> None:
        aggregator = llmd_telemetry.TelemetryAggregator()
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = pathlib.Path(tmpdir) / "does_not_exist.json"
            with self.assertRaises(FileNotFoundError):
                aggregator.load_benchmark_data(str(nonexistent))


if __name__ == "__main__":
    unittest.main()
