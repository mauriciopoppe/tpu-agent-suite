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
"""Unit tests for distributed serving benchmark generation and result parsing."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

_LLMD_DIR = (Path(__file__).resolve().parent.parent / "scripts" /
             "tpu-model-onboarding" / "llmd")
if str(_LLMD_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMD_DIR))

import benchmark  # noqa: E402
import parse_bench_results  # noqa: E402


def _sample_input(
    *,
    prefix_len: int | None = 2048,
    context_len: int | None = None,
    ttft_ms: float | None = 800.0,
    tpot_ms: float | None = 25.0,
) -> benchmark.BenchmarkInput:
    return benchmark.BenchmarkInput(
        model="Qwen/Qwen3-32B",
        accelerator="tpu-v6e",
        topology=benchmark.BenchmarkTopology(replicas=2),
        workload=benchmark.BenchmarkWorkload(
            isl=4000,
            osl=500,
            prefix_len=prefix_len,
            concurrency=8,
            model_context_length=context_len,
        ),
        slo=benchmark.BenchmarkSlo(ttft_ms=ttft_ms, tpot_ms=tpot_ms),
        context=benchmark.BenchmarkContext(
            namespace="bench-ns",
            release_name="llmd-qwen",
            hf_secret_name="hf-sec",
        ),
    )


class TestDistributedBenchmarkGeneration(unittest.TestCase):
    """Tests for benchmark.py across inference-perf and aiperf."""

    def test_inference_perf_shared_prefix_manifest(self) -> None:
        inp = _sample_input(prefix_len=2048)
        ip_res = benchmark.build_benchmark(
            inp,
            "high-prefix-share",
            harness="inference-perf",
        )
        self.assertEqual(ip_res["harness"], "inference-perf")
        self.assertEqual(ip_res["config"]["data"]["type"], "shared_prefix")
        ip_yaml = benchmark.build_benchmark_deployment(
            ip_res["config"],
            target_url="http://llmd-qwen-inference-gateway:80",
            namespace="bench-ns",
            pvc_name="bench-pvc",
        )
        ip_docs = list(yaml.safe_load_all(ip_yaml))
        self.assertEqual(len(ip_docs), 2)
        self.assertEqual(ip_docs[0]["kind"], "ConfigMap")
        self.assertEqual(ip_docs[1]["kind"], "Job")
        job_spec = ip_docs[1]["spec"]["template"]["spec"]
        env_names = {e["name"] for e in job_spec["containers"][0]["env"]}
        self.assertIn("RAYON_NUM_THREADS", env_names)
        self.assertIn("HF_TOKEN", env_names)

    def test_aiperf_submission_and_smoke_modes(self) -> None:
        inp_1m = _sample_input(context_len=1048576)
        res_sub = benchmark.build_benchmark(
            inp_1m,
            "high-prefix-share",
            harness="aiperf",
            stack="llm-d",
            smoke=False,
            concurrency=8,
            server_metrics_urls=[
                "http://10.0.0.1:8000/metrics",
                "http://10.0.0.2:8000/metrics",
            ],
        )
        cfg_sub = res_sub["config"]
        self.assertFalse(cfg_sub["harness"]["smoke"])
        self.assertEqual(
            cfg_sub["scenario"]["name"],
            "inferencex-agentx-mvp",
        )
        self.assertEqual(
            cfg_sub["scenario"]["benchmark_duration_seconds"],
            600,
        )
        profile_args = cfg_sub["profiles"]["agentx-c8"]["args"]
        self.assertIn("inferencex-agentx-mvp", profile_args)
        self.assertIn("600", profile_args)

        inp_256k = _sample_input(context_len=262144)
        res_smoke = benchmark.build_benchmark(
            inp_256k,
            "high-prefix-share",
            harness="aiperf-agentx",
            stack="dynamo",
            smoke=True,
            concurrency=2,
        )
        cfg_smoke = res_smoke["config"]
        self.assertTrue(cfg_smoke["harness"]["smoke"])
        self.assertEqual(
            cfg_smoke["scenario"]["benchmark_duration_seconds"],
            120,
        )
        smoke_args = cfg_smoke["profiles"]["agentx-c2"]["args"]
        self.assertIn("120", smoke_args)

        dynamo_yaml = benchmark.build_benchmark_deployment(
            cfg_smoke,
            target_url="http://llmd-qwen-frontend:8000",
            namespace="bench-ns",
        )
        docs = list(yaml.safe_load_all(dynamo_yaml))
        pod_spec = docs[1]["spec"]["template"]["spec"]
        env_map = {
            e["name"]: e.get("value")
            for e in pod_spec["containers"][0]["env"] if "value" in e
        }
        self.assertEqual(
            env_map["AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"],
            "1")
        self.assertEqual(
            env_map["AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID"], "1")
        self.assertEqual(
            env_map["AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID"], "1")

    def test_topology_discovery_and_cache_reset_helpers(self) -> None:
        self.assertEqual(
            benchmark.resolve_serving_endpoint("dynamo", "qwen-dgd"),
            "http://qwen-dgd-frontend:8000",
        )
        self.assertEqual(
            benchmark.resolve_serving_endpoint("llm-d",
                                               "llmd-rel",
                                               mode="gateway"),
            "http://llmd-rel-inference-gateway:80",
        )
        self.assertEqual(
            benchmark.resolve_serving_endpoint("llm-d",
                                               "llmd-rel",
                                               mode="standalone"),
            "http://llmd-rel-epp:9002",
        )

        pods = [
            {
                "ip": "10.0.0.10"
            },
            {
                "ip": "10.0.0.11"
            },
        ]
        urls_llmd = benchmark.format_server_metrics_urls(pods, stack="llm-d")
        self.assertEqual(
            urls_llmd,
            [
                "http://10.0.0.10:8000/metrics",
                "http://10.0.0.11:8000/metrics",
            ],
        )
        urls_dynamo = benchmark.format_server_metrics_urls(pods,
                                                           stack="dynamo")
        self.assertEqual(
            urls_dynamo,
            [
                "http://10.0.0.10:9090/metrics",
                "http://10.0.0.11:9090/metrics",
            ],
        )

        reset_vllm = benchmark.build_cache_reset_commands(
            "llm-d",
            "llmd-rel",
            "bench-ns",
            backend="vllm",
            method="http",
        )
        self.assertEqual(len(reset_vllm), 1)
        self.assertIn(":8000/reset_prefix_cache", reset_vllm[0])

        reset_sglang = benchmark.build_cache_reset_commands(
            "dynamo",
            "qwen-dgd",
            "bench-ns",
            backend="sglang",
            method="http",
        )
        self.assertIn(":8000/flush_cache", reset_sglang[0])

        reset_restart = benchmark.build_cache_reset_commands(
            "dynamo",
            "qwen-dgd",
            "bench-ns",
            method="pod-restart",
        )
        self.assertEqual(len(reset_restart), 2)
        self.assertIn("kubectl delete pods", reset_restart[0])
        self.assertIn("/v1/models", reset_restart[1])

    def test_cli_entrypoints(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = benchmark.main([
                "--model",
                "Qwen/Qwen3-32B",
                "--harness",
                "aiperf",
                "--stack",
                "llm-d",
                "--emit",
                "discovery-cmd",
            ])
        self.assertEqual(rc, 0)

    def test_execute_logs_parse_and_evidence_capture(self) -> None:
        sample_client_json = {
            "records": {
                "request_latency": {
                    "unit": "ms",
                    "avg": 900.0,
                    "p50": 850.0,
                    "p95": 1100.0,
                    "p99": 1300.0,
                },
                "time_to_first_token": {
                    "unit": "ms",
                    "avg": 180.0,
                    "p50": 160.0,
                    "p95": 240.0,
                    "p99": 310.0,
                },
                "time_per_output_token": {
                    "unit": "ms",
                    "avg": 12.0,
                    "p50": 11.5,
                    "p95": 15.0,
                    "p99": 18.0,
                },
                "request_throughput": {
                    "avg": 8.0
                },
                "input_token_throughput": {
                    "avg": 16000.0
                },
                "output_token_throughput": {
                    "avg": 2400.0
                },
                "request_count": {
                    "count": 40
                },
                "error_request_count": {
                    "count": 0
                },
            }
        }
        bench_log_text = ("Starting benchmark pod...\n"
                          "---BENCHMARK RESULTS (json files in /results)---\n"
                          "/results/agentx/profile_export_aiperf.json\n"
                          f"{json.dumps(sample_client_json)}\n")

        pod0_logs = (
            "Applying runtime patch gpt-oss-mxfp4...\n"
            'INFO: 10.0.0.5:41234 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
            'INFO: 10.0.0.5:41235 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
            'INFO: 10.0.0.5:41236 - "POST /v1/completions HTTP/1.1" 200 OK\n')
        pod1_logs = (
            "Applying runtime patch gpt-oss-mxfp4...\n"
            'INFO: 10.0.0.5:41240 - "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
        )

        recorded_cmds: list[list[str]] = []

        def fake_run(cmd: list[str],
                     **kwargs: object) -> subprocess.CompletedProcess[str]:
            recorded_cmds.append(list(cmd))
            joined = " ".join(cmd)
            if "apply" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="job.batch/created\n", stderr="")
            if "get" in cmd and any(a.startswith("job/") for a in cmd):
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=json.dumps({"status": {
                        "succeeded": 1
                    }}),
                    stderr="",
                )
            if "logs" in cmd and any(a.startswith("job/") for a in cmd):
                return subprocess.CompletedProcess(cmd,
                                                   0,
                                                   stdout=bench_log_text,
                                                   stderr="")
            if "get" in cmd and "lws,deploy,pods,svc,workload,provisioningrequest,job" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=
                    ("NAME                              READY   STATUS    RESTARTS   AGE\n"
                     "pod/vllm-tpu-workers-0            1/1     Running   0          10m\n"
                     "pod/vllm-tpu-workers-1            1/1     Running   0          10m\n"
                     ),
                    stderr="",
                )
            if "get" in cmd and "pods" in cmd and "jsonpath" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="vllm-tpu-workers-0 vllm-tpu-workers-1",
                    stderr="",
                )
            if "logs" in cmd and "vllm-tpu-workers-0" in cmd:
                return subprocess.CompletedProcess(cmd,
                                                   0,
                                                   stdout=pod0_logs,
                                                   stderr="")
            if "logs" in cmd and "vllm-tpu-workers-1" in cmd:
                return subprocess.CompletedProcess(cmd,
                                                   0,
                                                   stdout=pod1_logs,
                                                   stderr="")
            if "logs" in cmd and ("deploy/llm-d-epp" in cmd
                                  or "app=llm-d-epp" in joined):
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=
                    "epp router: selected pod vllm-tpu-workers-0 score=0.92\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            logs_out = tmp_path / "bench-logs.txt"
            parse_out = tmp_path / "bench-results.json"
            evidence_dir = tmp_path / "evidence"

            buf = io.StringIO()
            with patch("benchmark.subprocess.run",
                       side_effect=fake_run), redirect_stdout(buf):
                rc = benchmark.main([
                    "--model",
                    "Qwen/Qwen3-32B",
                    "--harness",
                    "aiperf",
                    "--stack",
                    "llm-d",
                    "--namespace",
                    "bench-ns",
                    "--release-name",
                    "llmd-qwen",
                    "--smoke",
                    "--concurrency",
                    "4",
                    "--ttft-sla",
                    "500",
                    "--tpot-sla",
                    "25",
                    "--kubeconfig",
                    "/tmp/fake-kubeconfig",
                    "--context",
                    "gke-tpu7x",
                    "--timeout",
                    "120",
                    "--execute",
                    "--logs-out",
                    str(logs_out),
                    "--parse-out",
                    str(parse_out),
                    "--evidence-dir",
                    str(evidence_dir),
                ])

            self.assertEqual(rc, 0)
            self.assertTrue(logs_out.is_file())
            self.assertIn("profile_export_aiperf.json",
                          logs_out.read_text(encoding="utf-8"))

            self.assertTrue(parse_out.is_file())
            parsed_summary = json.loads(parse_out.read_text(encoding="utf-8"))
            self.assertEqual(parsed_summary["harness"], "aiperf")
            self.assertEqual(parsed_summary["count_success"], 40)
            self.assertFalse(parsed_summary["sla_breach"])
            self.assertAlmostEqual(parsed_summary["metrics_ms"]["ttft"]["p95"],
                                   240.0)

            md_report_path = Path(f"{parse_out}.md")
            self.assertTrue(md_report_path.is_file())
            self.assertIn("TTFT (ms)",
                          md_report_path.read_text(encoding="utf-8"))

            cluster_state_path = evidence_dir / "target_cluster_state.txt"
            self.assertTrue(cluster_state_path.is_file())
            self.assertIn("pod/vllm-tpu-workers-0",
                          cluster_state_path.read_text(encoding="utf-8"))

            traffic_path = evidence_dir / "traffic_distribution.json"
            self.assertTrue(traffic_path.is_file())
            traffic = json.loads(traffic_path.read_text(encoding="utf-8"))
            self.assertEqual(traffic["total_requests"], 4)
            self.assertEqual(traffic["total_completions"], 1)
            self.assertEqual(traffic["total_chat_completions"], 3)
            self.assertEqual(
                traffic["pods"]["vllm-tpu-workers-0"]
                ["post_v1_chat_completions"],
                2,
            )
            self.assertEqual(
                traffic["pods"]["vllm-tpu-workers-0"]["post_v1_completions"],
                1,
            )
            self.assertAlmostEqual(
                traffic["replica_split_pct"]["vllm-tpu-workers-0"],
                75.0,
            )
            self.assertAlmostEqual(
                traffic["replica_split_pct"]["vllm-tpu-workers-1"],
                25.0,
            )

            self.assertTrue((evidence_dir / "worker_pod_logs" /
                             "vllm-tpu-workers-0.log").is_file())
            self.assertTrue((evidence_dir / "worker_pod_logs" /
                             "vllm-tpu-workers-1.log").is_file())
            self.assertTrue((evidence_dir / "epp_router.log").is_file())
            self.assertIn("selected pod vllm-tpu-workers-0",
                          (evidence_dir /
                           "epp_router.log").read_text(encoding="utf-8"))
            self.assertTrue(
                any("--kubeconfig" in c and "--context" in c
                    for c in recorded_cmds))

        dynamo_traffic = benchmark.compute_traffic_distribution(
            {
                "qwen3-worker-0": "starting worker 0...\n",
                "qwen3-worker-1": "starting worker 1...\n",
            },
            worker_metrics={
                "qwen3-worker-0":
                ('dynamo_component_requests_total{dynamo_component="backend",dynamo_endpoint="generate",model="Qwen/Qwen3-32B"} 15\n'
                 ),
                "qwen3-worker-1":
                ('dynamo_component_requests_total{dynamo_component="backend",dynamo_endpoint="generate",model="Qwen/Qwen3-32B"} 5\n'
                 ),
            },
            namespace="dynamo-gpu",
            stack="dynamo",
            release_name="qwen3-32b-agg-router",
        )
        self.assertEqual(dynamo_traffic["total_requests"], 20)
        self.assertEqual(dynamo_traffic["total_chat_completions"], 20)
        self.assertEqual(
            dynamo_traffic["pods"]["qwen3-worker-0"]
            ["dynamo_generate_requests"],
            15,
        )
        self.assertAlmostEqual(
            dynamo_traffic["replica_split_pct"]["qwen3-worker-0"],
            75.0,
        )
        self.assertAlmostEqual(
            dynamo_traffic["replica_split_pct"]["qwen3-worker-1"],
            25.0,
        )


class TestDistributedParseBenchResults(unittest.TestCase):
    """Tests for parse_bench_results.py across aiperf and inference-perf."""

    def test_parse_aiperf_with_vllm_server_metrics(self) -> None:
        client_json = {
            "records": {
                "request_latency": {
                    "unit": "ms",
                    "avg": 1200.0,
                    "p50": 1100.0,
                    "p95": 1800.0,
                    "p99": 2100.0,
                },
                "time_to_first_token": {
                    "unit": "s",
                    "avg": 0.25,
                    "p50": 0.20,
                    "p95": 0.45,
                    "p99": 0.60,
                },
                "time_per_output_token": {
                    "unit": "ms",
                    "avg": 14.0,
                    "p50": 13.5,
                    "p95": 19.0,
                    "p99": 22.0,
                },
                "request_throughput": {
                    "avg": 12.5
                },
                "input_token_throughput": {
                    "avg": 25000.0
                },
                "output_token_throughput": {
                    "avg": 3200.0
                },
                "request_count": {
                    "count": 200
                },
                "error_request_count": {
                    "count": 0
                },
            },
        }
        server_json = {
            "metrics": {
                "vllm:prefix_cache_hits": {
                    "delta": 8500
                },
                "vllm:prefix_cache_queries": {
                    "delta": 10000
                },
                "vllm:kv_cache_usage_perc": {
                    "avg": 0.62,
                    "p95": 0.78
                },
                "vllm:num_requests_waiting": {
                    "max": 2.0
                },
                "vllm:request_success": {
                    "series": [
                        {
                            "endpoint": "http://10.0.0.1:8000/metrics",
                            "delta": 105
                        },
                        {
                            "endpoint": "http://10.0.0.2:8000/metrics",
                            "delta": 95
                        },
                    ]
                },
            }
        }
        log_stream = ("benchmark starting...\n"
                      "---BENCHMARK RESULTS (json files in /results)---\n"
                      "/results/agentx/profile_export_aiperf.json\n"
                      f"{json.dumps(client_json)}\n"
                      "/results/agentx/server_metrics_export.json\n"
                      f"{json.dumps(server_json)}\n")
        blocks = parse_bench_results._extract_json_blocks(log_stream)
        self.assertEqual(len(blocks), 2)
        summary = parse_bench_results._normalize(blocks)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["harness"], "aiperf")
        self.assertAlmostEqual(summary["metrics_ms"]["ttft"]["p95"], 450.0)
        self.assertAlmostEqual(summary["metrics_ms"]["tpot"]["p95"], 19.0)
        self.assertAlmostEqual(
            summary["server_metrics"]["prefix_cache_hit_rate"], 0.85)
        self.assertAlmostEqual(summary["server_metrics"]["kv_cache_usage_p95"],
                               0.78)
        self.assertAlmostEqual(
            summary["server_metrics"]["worker_load_skew_ratio"], 1.05)

        report, breach = parse_bench_results.render_markdown(
            summary,
            {
                "ttft_ms": 800.0,
                "tpot_ms": 25.0,
                "prefix_hit_rate_min": 0.80,
            },
        )
        self.assertFalse(breach)
        self.assertIn("Prefix cache hit rate (vllm): 85.0%", report)
        self.assertAlmostEqual(summary["goodput_tokens_per_sec"], 3200.0)

    def test_parse_aiperf_sglang_metrics_and_sla_breach(self) -> None:
        client_json = {
            "records": {
                "request_latency": {
                    "unit": "ms",
                    "p95": 3000.0
                },
                "time_to_first_token": {
                    "unit": "ms",
                    "p95": 950.0
                },
                "inter_token_latency": {
                    "unit": "ms",
                    "p95": 20.0
                },
                "output_token_throughput": {
                    "avg": 1500.0
                },
                "request_count": 95,
                "error_request_count": 5,
            },
        }
        server_json = {
            "metrics": {
                "sglang:cached_tokens": {
                    "delta": 40000
                },
                "sglang:prompt_tokens": {
                    "delta": 100000
                },
                "sglang:token_usage": {
                    "p95": 0.88
                },
                "sglang:num_queue_reqs": {
                    "max": 7.0
                },
            }
        }
        summary = parse_bench_results._normalize([client_json, server_json])
        assert summary is not None
        self.assertEqual(summary["server_metrics"]["backend"], "sglang")
        self.assertAlmostEqual(
            summary["server_metrics"]["prefix_cache_hit_rate"], 0.40)

        report, breach = parse_bench_results.render_markdown(
            summary,
            {
                "ttft_ms": 800.0,
                "prefix_hit_rate_min": 0.70,
            },
        )
        self.assertTrue(breach)
        self.assertIn("[EXCEEDS by 19%]", report)
        self.assertIn("[BELOW TARGET]", report)
        self.assertEqual(summary["goodput_tokens_per_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
