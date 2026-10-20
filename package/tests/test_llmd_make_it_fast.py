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
"""Unit and closed-loop study tests for the llm-d Make-It-Fast optimization loop."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_PERF_SCRIPTS_DIR = _PACKAGE_ROOT / "scripts" / "tpu-performance-optimization"
_ONBOARD_SCRIPTS_DIR = _PACKAGE_ROOT / "scripts" / "tpu-model-onboarding"
_LLMD_SCRIPTS_DIR = _ONBOARD_SCRIPTS_DIR / "llmd"
_COMMON_SCRIPTS_DIR = _PACKAGE_ROOT / "scripts" / "common"

for _p in (
        _REPO_ROOT,
        _PERF_SCRIPTS_DIR,
        _ONBOARD_SCRIPTS_DIR,
        _LLMD_SCRIPTS_DIR,
        _COMMON_SCRIPTS_DIR,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dynamo_translator  # noqa: E402
import epp_config  # noqa: E402
import evaluate_arm  # noqa: E402
import hypothesis_bank  # noqa: E402
import llmd_telemetry  # noqa: E402
import study_protocol  # noqa: E402

from framework import study, utils  # noqa: E402

_HOMOGENEOUS_DYNAMO_YAML = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-worker
  namespace: llmd-perf
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: worker
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "Qwen/Qwen3-32B"
        - "--tensor-parallel-size"
        - "4"
        - "--max-model-len"
        - "16384"
        - "--max-num-batched-tokens"
        - "4096"
        - "--max-num-seqs"
        - "128"
        - "--gpu-memory-utilization"
        - "0.90"
        resources:
          limits:
            nvidia.com/gpu: 4
"""


def _build_telemetry_payload(
    *,
    arm_id: str,
    model: str = "Qwen/Qwen3-32B",
    accelerator_type: str = "v6e",
    routing_algorithm: str = "prefix-cache-scorer",
    output_tps: float = 1820.0,
    input_tps: float = 6400.0,
    rps: float = 3.55,
    ttft_p50_ms: float = 210.0,
    ttft_p99_ms: float = 640.0,
    tpot_p50_ms: float = 17.5,
    tpot_p99_ms: float = 24.5,
    e2e_p99_ms: float = 3100.0,
    prefix_hit_rate: float = 0.31,
    kv_usage_mean: float = 0.62,
    kv_usage_peak: float = 0.78,
    count_success: int = 200,
    count_failure: int = 0,
    sla_ttft_ms: float = 800.0,
    sla_tpot_ms: float = 35.0,
) -> dict:
    """Create a schema-valid llmd-telemetry.json dictionary using TelemetryAggregator."""
    client_data = {
        "harness": "aiperf",
        "count_success": count_success,
        "count_failure": count_failure,
        "metrics_ms": {
            "request_latency": {
                "mean": round(e2e_p99_ms * 0.65, 2),
                "p50": round(e2e_p99_ms * 0.60, 2),
                "p95": round(e2e_p99_ms * 0.92, 2),
                "p99": e2e_p99_ms,
            },
            "ttft": {
                "mean": round((ttft_p50_ms + ttft_p99_ms) / 2.0, 2),
                "p50": ttft_p50_ms,
                "p95": round(ttft_p99_ms * 0.92, 2),
                "p99": ttft_p99_ms,
            },
            "tpot": {
                "mean": round((tpot_p50_ms + tpot_p99_ms) / 2.0, 2),
                "p50": tpot_p50_ms,
                "p95": round(tpot_p99_ms * 0.94, 2),
                "p99": tpot_p99_ms,
            },
        },
        "throughput": {
            "requests_per_sec": rps,
            "input_tokens_per_sec": input_tps,
            "output_tokens_per_sec": output_tps,
            "total_tokens_per_sec": round(output_tps + input_tps, 2),
        },
        "server_metrics": {
            "backend": "vllm",
            "prefix_cache_hit_rate": prefix_hit_rate,
            "kv_cache_usage_mean": kv_usage_mean,
            "kv_cache_usage_p95": kv_usage_peak,
            "max_waiting_requests": 2.0,
            "worker_load_skew_ratio": 1.05,
            "worker_count": 2,
        },
    }
    agg = llmd_telemetry.TelemetryAggregator(
        arm_id=arm_id,
        scenario="inferencex-agentx-mvp",
        model=model,
        accelerator_type=accelerator_type,
        routing_algorithm=routing_algorithm,
        sla_ttft_ms=sla_ttft_ms,
        sla_tpot_ms=sla_tpot_ms,
    )
    return agg.aggregate(benchmark_data=client_data)


class TestEppConfigBundleOperations(unittest.TestCase):
    """Tests for epp_config.py inspect, checkpoint, mutate, and rollback."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_homogeneous_bundle(self) -> pathlib.Path:
        src = self.work_dir / "dynamo-homog.yaml"
        src.write_text(_HOMOGENEOUS_DYNAMO_YAML, encoding="utf-8")
        out_dir = self.work_dir / "bundle-homog"
        res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=src,
            output_dir=out_dir,
            target_tpu_type="v6e",
            target_tpu_topology="2x2",
        )
        return pathlib.Path(res["bundle_dir"])

    def test_inspect_homogeneous_bundle(self) -> None:
        homog_dir = self._create_homogeneous_bundle()
        snap = epp_config.inspect_bundle(homog_dir)
        self.assertEqual(snap["serving_mode"], "homogeneous")
        self.assertIn("prefix-cache-scorer", snap["scorers"])
        self.assertIn("queue-depth-scorer", snap["scorers"])
        self.assertIn("kv-utilization-scorer", snap["scorers"])
        self.assertIn("serving", snap["pools"])
        self.assertEqual(snap["pools"]["serving"]["replicas"], 2)
        self.assertEqual(len(snap["recipe_fingerprint"]), 64)

    def test_inspect_and_mutate_kustomize_overlay(self) -> None:
        auto_dir = self._create_homogeneous_bundle()
        before = epp_config.inspect_bundle(auto_dir)
        self.assertIn("prefix-cache-scorer", before["scorer_weights"])

        mut_res = epp_config.mutate_bundle(
            auto_dir,
            scorers={
                "precise-prefix-cache-scorer": 85.0,
                "active-request-scorer": 45.0,
            },
            replicas=3,
            vllm_args={"--max-num-batched-tokens": "8192"},
        )
        self.assertEqual(mut_res["status"], "completed")
        self.assertNotEqual(mut_res["fingerprint_before"],
                            mut_res["fingerprint_after"])

        after = epp_config.inspect_bundle(auto_dir)
        self.assertEqual(after["scorer_weights"]["prefix-cache-scorer"], 85.0)
        self.assertEqual(after["scorer_weights"]["queue-depth-scorer"], 45.0)
        self.assertEqual(after["pools"]["serving"]["replicas"], 3)
        self.assertEqual(after["vllm_flags"]["--max-num-batched-tokens"], 8192)

    def test_mutate_dry_run_leaves_bundle_untouched(self) -> None:
        homog_dir = self._create_homogeneous_bundle()
        before = epp_config.inspect_bundle(homog_dir)

        dry = epp_config.mutate_bundle(
            homog_dir,
            scorers={"prefix-cache-scorer": 95.0},
            dry_run=True,
        )
        self.assertTrue(dry["dry_run"])
        self.assertNotEqual(dry["fingerprint_before"],
                            dry["fingerprint_after"])

        after = epp_config.inspect_bundle(homog_dir)
        self.assertEqual(before["recipe_fingerprint"],
                         after["recipe_fingerprint"])
        self.assertEqual(
            before["scorer_weights"]["prefix-cache-scorer"],
            after["scorer_weights"]["prefix-cache-scorer"],
        )

    def test_mutate_rejects_invalid_invariants(self) -> None:
        homog_dir = self._create_homogeneous_bundle()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            epp_config.mutate_bundle(homog_dir,
                                     scorers={"prefix-cache-scorer": -10.0})

        with self.assertRaisesRegex(ValueError, "total active scorer weight"):
            epp_config.mutate_bundle(
                homog_dir,
                scorers={
                    "prefix-cache-scorer": 0.0,
                    "queue-depth-scorer": 0.0,
                    "kv-utilization-scorer": 0.0,
                    "token-load-scorer": 0.0,
                },
            )

        with self.assertRaisesRegex(ValueError, "gpu-memory-utilization"):
            epp_config.mutate_bundle(
                homog_dir, vllm_args={"--gpu-memory-utilization": "1.25"})

        with self.assertRaisesRegex(ValueError, "block-size"):
            epp_config.mutate_bundle(homog_dir,
                                     vllm_args={"--block-size": "0"})

        with self.assertRaisesRegex(ValueError, "replicas"):
            epp_config.mutate_bundle(homog_dir, replicas=0)

    def test_checkpoint_and_rollback_restore_exact_fingerprint_and_clean_extra_files(
        self, ) -> None:
        homog_dir = self._create_homogeneous_bundle()
        ckpt_dir = self.work_dir / "ckpt-baseline"

        ckpt_meta = epp_config.checkpoint_bundle(homog_dir, ckpt_dir)
        orig_fp = ckpt_meta["recipe_fingerprint"]

        epp_config.mutate_bundle(
            homog_dir,
            scorers={
                "prefix-cache-scorer": 90.0,
                "queue-depth-scorer": 15.0,
            },
            filters={
                "prefix-cache-affinity-filter": {
                    "affinityThreshold": 0.88
                }
            },
            vllm_args={
                "--max-num-batched-tokens": "16384",
                "--block-size": "64",
            },
            env_vars={"VLLM_TORCH_PROFILER_DIR": "/tmp/prof"},
            replicas=4,
        )
        stray_file = homog_dir / "99-experimental-canary.yaml"
        stray_file.write_text("kind: ConfigMap\n", encoding="utf-8")

        mutated_snap = epp_config.inspect_bundle(homog_dir)
        self.assertNotEqual(mutated_snap["recipe_fingerprint"], orig_fp)
        self.assertTrue(stray_file.exists())

        rb_meta = epp_config.rollback_bundle(homog_dir, ckpt_dir)
        self.assertTrue(rb_meta["fingerprint_matches_checkpoint"])
        self.assertEqual(rb_meta["recipe_fingerprint"], orig_fp)
        self.assertFalse(stray_file.exists())

    def test_cli_subcommands_inspect_checkpoint_mutate_rollback(self) -> None:
        homog_dir = self._create_homogeneous_bundle()
        ckpt_dir = self.work_dir / "ckpt-cli"
        script = _PERF_SCRIPTS_DIR / "epp_config.py"

        p_insp = subprocess.run(
            [
                sys.executable,
                str(script),
                "inspect",
                "--bundle-dir",
                str(homog_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        insp_data = json.loads(p_insp.stdout)
        self.assertEqual(insp_data["serving_mode"], "homogeneous")
        orig_fp = insp_data["recipe_fingerprint"]

        p_ckpt = subprocess.run(
            [
                sys.executable,
                str(script),
                "checkpoint",
                "--bundle-dir",
                str(homog_dir),
                "--checkpoint-dir",
                str(ckpt_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        ckpt_data = json.loads(p_ckpt.stdout)
        self.assertEqual(ckpt_data["recipe_fingerprint"], orig_fp)

        p_mut = subprocess.run(
            [
                sys.executable,
                str(script),
                "mutate",
                "--bundle-dir",
                str(homog_dir),
                "--set-scorer",
                "prefix-cache-scorer=85",
                "--set-filter",
                "prefix-cache-affinity-filter=affinityThreshold:0.82",
                "--set-vllm-arg",
                "--max-num-batched-tokens=16384",
                "--set-replicas",
                "3",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        mut_data = json.loads(p_mut.stdout)
        self.assertNotEqual(mut_data["fingerprint_after"], orig_fp)
        self.assertEqual(
            mut_data["snapshot"]["scorer_weights"]["prefix-cache-scorer"],
            85.0)

        p_rb = subprocess.run(
            [
                sys.executable,
                str(script),
                "rollback",
                "--bundle-dir",
                str(homog_dir),
                "--checkpoint-dir",
                str(ckpt_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rb_data = json.loads(p_rb.stdout)
        self.assertTrue(rb_data["fingerprint_matches_checkpoint"])
        self.assertEqual(rb_data["recipe_fingerprint"], orig_fp)

    def test_mutate_syncs_both_endpoint_picker_config_and_default_plugins_yaml(
        self, ) -> None:
        import yaml

        src = self.work_dir / "dynamo-standalone.yaml"
        src.write_text(_HOMOGENEOUS_DYNAMO_YAML, encoding="utf-8")
        out_dir = self.work_dir / "bundle-standalone"
        res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=src,
            output_dir=out_dir,
            target_tpu_type="v6e",
            target_tpu_topology="2x2",
            standalone_epp=True,
        )
        bundle_dir = pathlib.Path(res["bundle_dir"])
        script = _PERF_SCRIPTS_DIR / "epp_config.py"

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "mutate",
                "--bundle-dir",
                str(bundle_dir),
                "--set-scorer",
                "prefix-cache-scorer=80",
                "--set-scorer",
                "queue-depth-scorer=15",
                "--set-scorer",
                "kv-utilization-scorer=10",
                "--set-scorer",
                "active-request-scorer=25",
                "--set-scorer",
                "no-hit-lru-scorer=5",
                "--set-filter-param",
                "prefix-cache-affinity-filter.affinityThreshold=0.85",
                "--set-filter-param",
                "saturationDetector.maxKVCacheUtilization=0.92",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        mut_out = json.loads(proc.stdout)
        self.assertEqual(mut_out["status"], "completed")

        epp_yaml = bundle_dir / "01-endpointpickerconfig.yaml"
        epp_raw = epp_yaml.read_text(encoding="utf-8")
        self.assertIn("endpoint-picker-config.yaml: |", epp_raw)
        self.assertIn("default-plugins.yaml: |", epp_raw)
        self.assertIn("envoy.yaml: |", epp_raw)
        self.assertNotIn('endpoint-picker-config.yaml: "', epp_raw)
        self.assertNotIn('default-plugins.yaml: "', epp_raw)
        self.assertNotIn('envoy.yaml: "', epp_raw)
        outer_docs = list(yaml.safe_load_all(epp_raw))
        cm_doc = next(d for d in outer_docs if d.get("kind") == "ConfigMap")
        cm_data = cm_doc["data"]
        self.assertIn("endpoint-picker-config.yaml", cm_data)
        self.assertIn("default-plugins.yaml", cm_data)

        v0_doc = yaml.safe_load(cm_data["endpoint-picker-config.yaml"])
        v1_doc = yaml.safe_load(cm_data["default-plugins.yaml"])

        top_scorers = {
            s["name"]: s["weight"]
            for s in v0_doc["scheduling"]["scorers"]
        }
        self.assertEqual(top_scorers["prefix-cache-scorer"], 80)
        self.assertEqual(top_scorers["queue-depth-scorer"], 15)
        self.assertEqual(top_scorers["kv-cache-utilization-scorer"], 10)
        self.assertEqual(top_scorers["active-request-scorer"], 25)
        self.assertEqual(top_scorers["no-hit-lru-scorer"], 5)

        self.assertEqual(v1_doc["apiVersion"], "llm-d.ai/v1alpha1")
        self.assertNotIn("flowControl", v1_doc)
        top_plugin_types = []
        for p_entry in v1_doc["plugins"]:
            self.assertNotIn(
                "weight",
                p_entry,
                "Top-level plugins in llm-d.ai/v1alpha1 must never carry weight",
            )
            top_plugin_types.append(p_entry.get("type"))
        for expected_type in (
                "prefix-cache-scorer",
                "queue-scorer",
                "kv-cache-utilization-scorer",
                "active-request-scorer",
                "no-hit-lru-scorer",
                "prefix-cache-affinity-filter",
        ):
            self.assertIn(expected_type, top_plugin_types)

        sp_plugins = v1_doc["schedulingProfiles"][0]["plugins"]
        sp_weights = {}
        for ref_entry in sp_plugins:
            ref_name = ref_entry["pluginRef"]
            if "filter" in ref_name or "picker" in ref_name:
                self.assertNotIn(
                    "weight",
                    ref_entry,
                    f"Filter/picker {ref_name} must not have weight",
                )
            if "weight" in ref_entry:
                self.assertIsInstance(ref_entry["weight"], int)
                sp_weights[ref_name] = ref_entry["weight"]

        self.assertEqual(sp_weights["prefix-cache-scorer"], 80)
        self.assertEqual(sp_weights["queue-scorer"], 15)
        self.assertEqual(sp_weights["kv-cache-utilization-scorer"], 10)
        self.assertEqual(sp_weights["active-request-scorer"], 25)
        self.assertEqual(sp_weights["no-hit-lru-scorer"], 5)

    def test_apply_bundle_and_mutate_rollback_apply_restart_epp_deployment(
        self, ) -> None:
        src = self.work_dir / "dynamo-apply.yaml"
        src.write_text(_HOMOGENEOUS_DYNAMO_YAML, encoding="utf-8")
        out_dir = self.work_dir / "bundle-apply"
        res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=src,
            output_dir=out_dir,
            target_tpu_type="v6e",
            target_tpu_topology="2x2",
            standalone_epp=True,
        )
        bundle_dir = pathlib.Path(res["bundle_dir"])
        ckpt_dir = self.work_dir / "ckpt-apply"

        recorded_cmds: list[list[str]] = []

        def _fake_run(cmd, *args, **kwargs):
            del args, kwargs
            recorded_cmds.append(list(cmd))
            proc = mock.MagicMock()
            proc.returncode = 0
            proc.stdout = f"applied: {' '.join(cmd)}"
            proc.stderr = ""
            return proc

        with mock.patch.object(epp_config.subprocess,
                               "run",
                               side_effect=_fake_run):
            mut_res = epp_config.mutate_bundle(
                bundle_dir,
                checkpoint_dir=ckpt_dir,
                set_scorers=["prefix-cache-scorer=85"],
                apply=True,
                namespace="llmd-ns",
                context="gke-tpu-ctx",
                kubeconfig="/tmp/fake-kubeconfig",
            )
            self.assertTrue(mut_res["applied_to_cluster"])
            self.assertEqual(mut_res["restarted_deployments"], ["llm-d-epp"])
            self.assertEqual(len(recorded_cmds), 3)
            self.assertEqual(
                recorded_cmds[0],
                [
                    "kubectl",
                    "--kubeconfig",
                    "/tmp/fake-kubeconfig",
                    "--context",
                    "gke-tpu-ctx",
                    "-n",
                    "llmd-ns",
                    "apply",
                    "-k",
                    str((bundle_dir / "kustomize").resolve()),
                ],
            )
            self.assertEqual(
                recorded_cmds[1],
                [
                    "kubectl",
                    "--kubeconfig",
                    "/tmp/fake-kubeconfig",
                    "--context",
                    "gke-tpu-ctx",
                    "-n",
                    "llmd-ns",
                    "rollout",
                    "restart",
                    "deployment/llm-d-epp",
                ],
            )
            self.assertEqual(
                recorded_cmds[2],
                [
                    "kubectl",
                    "--kubeconfig",
                    "/tmp/fake-kubeconfig",
                    "--context",
                    "gke-tpu-ctx",
                    "-n",
                    "llmd-ns",
                    "rollout",
                    "status",
                    "deployment/llm-d-epp",
                    "--timeout=180s",
                ],
            )

            recorded_cmds.clear()
            rb_res = epp_config.rollback_bundle(
                bundle_dir,
                ckpt_dir,
                apply=True,
                namespace="llmd-ns",
                context="gke-tpu-ctx",
                kubeconfig="/tmp/fake-kubeconfig",
            )
            self.assertTrue(rb_res["applied_to_cluster"])
            self.assertEqual(rb_res["restarted_deployments"], ["llm-d-epp"])
            self.assertEqual(len(recorded_cmds), 3)
            self.assertIn("rollout", recorded_cmds[1])
            self.assertIn("deployment/llm-d-epp", recorded_cmds[1])

    def test_validate_bundle_and_cli_catches_corrupted_nested_configmap_and_invalid_weights(
        self, ) -> None:
        src = self.work_dir / "dynamo-validate.yaml"
        src.write_text(_HOMOGENEOUS_DYNAMO_YAML, encoding="utf-8")
        out_dir = self.work_dir / "bundle-validate"
        res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=src,
            output_dir=out_dir,
            target_tpu_type="v6e",
            target_tpu_topology="2x2",
            standalone_epp=True,
        )
        bundle_dir = pathlib.Path(res["bundle_dir"])
        script = _PERF_SCRIPTS_DIR / "epp_config.py"

        val_ok = epp_config.validate_bundle(bundle_dir)
        self.assertTrue(val_ok["valid"], msg=val_ok["errors"])
        self.assertEqual(val_ok["errors"], [])
        self.assertIn("prefix-cache-scorer", val_ok["scorer_weights"])
        self.assertEqual(len(val_ok["recipe_fingerprint"]), 64)

        p_ok = subprocess.run(
            [
                sys.executable,
                str(script),
                "validate",
                "--bundle-dir",
                str(bundle_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(p_ok.returncode, 0, msg=p_ok.stdout + p_ok.stderr)
        cli_ok = json.loads(p_ok.stdout)
        self.assertTrue(cli_ok["valid"])

        ckpt_dir = self.work_dir / "ckpt-validate"
        epp_config.checkpoint_bundle(bundle_dir, ckpt_dir)

        epp_file = bundle_dir / "01-endpointpickerconfig.yaml"
        epp_text = epp_file.read_text(encoding="utf-8")
        corrupted_yaml = epp_text.replace(
            "kind: EndpointPickerConfig",
            "kind: EndpointPickerConfig\n    broken_indent: [unclosed",
            1,
        )
        epp_file.write_text(corrupted_yaml, encoding="utf-8")

        val_bad_syntax = epp_config.validate_bundle(bundle_dir)
        self.assertFalse(val_bad_syntax["valid"])
        self.assertTrue(
            any("nested YAML syntax error" in e or "YAML syntax error" in e
                for e in val_bad_syntax["errors"]),
            msg=val_bad_syntax["errors"],
        )

        p_bad = subprocess.run(
            [
                sys.executable,
                str(script),
                "validate",
                "--bundle-dir",
                str(bundle_dir),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(p_bad.returncode, 1)
        cli_bad = json.loads(p_bad.stdout)
        self.assertFalse(cli_bad["valid"])

        epp_config.rollback_bundle(bundle_dir, ckpt_dir)
        bad_weight_yaml = epp_text.replace("weight: 100", "weight: -25", 1)
        epp_file.write_text(bad_weight_yaml, encoding="utf-8")
        val_neg = epp_config.validate_bundle(bundle_dir)
        self.assertFalse(val_neg["valid"])
        self.assertTrue(
            any("non-negative" in e for e in val_neg["errors"]),
            msg=val_neg["errors"],
        )

        epp_config.rollback_bundle(bundle_dir, ckpt_dir)
        self.assertTrue(epp_config.validate_bundle(bundle_dir)["valid"])


class TestHypothesisBankLlmd(unittest.TestCase):
    """Tests for hypothesis_bank.py with distributed serving (llm-d) arms."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.repo_root = self.root / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)
        (self.repo_root / "01-endpointpickerconfig.yaml").write_text(
            "kind: EndpointPickerConfig\n", encoding="utf-8")

        self.env_patcher = mock.patch.dict(
            os.environ,
            {"TPU_AGENT_SUITE_HOME": str(self.root / "suite_home")})
        self.env_patcher.start()

        self.study_id = "qwen3-32b-llmd-bank-test"
        intake = study_protocol.build_query_intake(
            prompt=
            "Optimize Qwen/Qwen3-32B throughput on v6e 2x4 with llm-d for agentx",
            study_id=self.study_id,
            primary_metric="output_tokens_per_s",
            replicate_count=1,
        )
        self.protocol = study_protocol.compile_study_protocol(intake)
        study.start_study(
            study_id=self.study_id,
            protocol=self.protocol,
            package="tpu-agent-suite",
            target="Qwen/Qwen3-32B",
        )

    def tearDown(self) -> None:
        self.env_patcher.stop()
        self.temp_dir.cleanup()

    def test_admits_and_deduplicates_llmd_epp_hypotheses(self) -> None:
        hypotheses = [
            {
                "id":
                "h1-prefix-routing",
                "kind":
                "load-imbalance",
                "claim":
                "Increase prefix-cache-scorer weight to 80 and queue-depth-scorer to 50.",
                "share":
                0.35,
                "headroom":
                0.45,
                "evidence": [{
                    "type":
                    "telemetry",
                    "ref":
                    "cache_memory.prefix_cache.hit_rate_tokens=0.31",
                    "note":
                    "Low prefix hit rate under shared 8k system prompt.",
                }],
                "code_refs": [{
                    "path": "01-endpointpickerconfig.yaml"
                }],
                "epp_config": {
                    "scorers": {
                        "prefix-cache-scorer": 80,
                        "queue-depth-scorer": 50,
                    },
                    "filters": {
                        "prefix-cache-affinity-filter": {
                            "affinityThreshold": 0.80
                        }
                    },
                },
            },
            {
                "id":
                "h2-kv-tiered-cache",
                "kind":
                "memory-capacity",
                "claim":
                "Enable block-size 64 and increase gpu-memory-utilization to 0.94.",
                "share":
                0.0,
                "headroom":
                0.25,
                "evidence": [{
                    "type":
                    "telemetry",
                    "ref":
                    "cache_memory.kv_cache.usage_factor_peak=0.96",
                }],
                "epp_config": {
                    "vllm_flags": {
                        "--block-size": 64,
                        "--gpu-memory-utilization": 0.94,
                    }
                },
            },
        ]

        res = hypothesis_bank.admit_hypotheses(
            study_id=self.study_id,
            hypotheses=hypotheses,
            tpu_version="v6e",
            repo_root=self.repo_root,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(len(res["admitted"]), 2)
        self.assertEqual(res["rejected"], [])

        hb_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                    "hypothesis-bank-result.json")
        self.assertEqual(utils.validate_instance(res, hb_schema), [])

        dup_hypotheses = [{
            "id":
            "h3-alias-duplicate",
            "kind":
            "load-imbalance",
            "claim":
            "Alias duplicate using precise-prefix-cache-scorer and active-request-scorer.",
            "share":
            0.30,
            "headroom":
            0.40,
            "evidence": [{
                "type": "telemetry",
                "ref": "prefix_cache_hit_rate"
            }],
            "target_parameters": {
                "scorers": {
                    "precise-prefix-cache-scorer": 80,
                    "active-request-scorer": 50,
                },
                "filters": {
                    "prefix-cache-affinity-filter": {
                        "affinityThreshold": 0.80
                    }
                },
            },
        }]
        res_dup = hypothesis_bank.admit_hypotheses(
            study_id=self.study_id,
            hypotheses=dup_hypotheses,
            tpu_version="v6e",
            repo_root=self.repo_root,
        )
        self.assertEqual(len(res_dup["admitted"]), 0)
        self.assertEqual(len(res_dup["rejected"]), 1)
        self.assertTrue(
            any("duplicate_recipe:h1-prefix-routing" in r
                for r in res_dup["rejected"][0]["reasons"]),
            msg=res_dup["rejected"],
        )

    def test_admits_request_routing_kind_and_validates_llmd_schemas(
        self, ) -> None:
        rr_hypotheses = [{
            "id":
            "h-request-routing-epp",
            "kind":
            "request-routing",
            "claim":
            "Tune EndpointPickerConfig prefix-cache-scorer=90 and queue-scorer=20.",
            "share":
            0.50,
            "headroom":
            0.40,
            "evidence": [{
                "type":
                "telemetry",
                "ref":
                "cache_memory.prefix_cache.hit_rate_tokens=0.28",
                "note":
                "Low prefix hit rate under shared system prompt.",
            }],
            "code_refs": [{
                "path": "01-endpointpickerconfig.yaml"
            }],
            "epp_config": {
                "scorers": {
                    "prefix-cache-scorer": 90,
                    "queue-depth-scorer": 20,
                }
            },
        }]
        res = hypothesis_bank.admit_hypotheses(
            study_id=self.study_id,
            hypotheses=rr_hypotheses,
            tpu_version="v6e",
            repo_root=self.repo_root,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(len(res["admitted"]), 1)
        self.assertEqual(res["admitted"][0]["hypothesis_id"],
                         "h-request-routing-epp")
        self.assertEqual(
            study.load_study(self.study_id)["arms"][0]["kind"],
            "request-routing")
        self.assertEqual(res["rejected"], [])

        arm_req_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                         "arm-request.json")
        study_req_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                           "study-request.json")
        prof_req_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                          "profiling-request.json")
        prof_res_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                          "profiling-result.json")

        arm_req_payload = {
            "prompt": "Execute request-routing arm.",
            "study_id": self.study_id,
            "run_id": "20261020T120000Z-abcdef012345",
            "arm": "h-request-routing-epp",
            "kind": "request-routing",
            "claim":
            "Tune EndpointPickerConfig prefix-cache-scorer=90 and queue-scorer=20.",
            "bundle_dir": str(self.repo_root),
            "namespace": "llmd-ns",
            "kubeconfig": "/tmp/kubeconfig",
            "context": "gke-tpu-ctx",
            "service_url": "http://10.0.0.1:8000",
        }
        self.assertEqual(
            utils.validate_instance(arm_req_payload, arm_req_schema), [])

        study_req_payload = {
            "prompt": "Tune llm-d deployment on GKE TPU v6e.",
            "study_id": self.study_id,
            "run_id": "20261020T120000Z-abcdef012345",
            "target": "Qwen/Qwen2.5-3B-Instruct",
            "bundle_dir": str(self.repo_root),
            "namespace": "llmd-ns",
            "kubeconfig": "/tmp/kubeconfig",
            "context": "gke-tpu-ctx",
            "service_url": "http://10.0.0.1:8000",
            "mode": "auto",
            "levers": ["request-routing", "scheduler-batching"],
        }
        self.assertEqual(
            utils.validate_instance(study_req_payload, study_req_schema), [])

        prof_req_payload = {
            "prompt":
            "Run initial llm-d diagnostics passthrough on bundle_dir.",
            "run_id": "20261020T120000Z-abcdef012345",
            "bundle_dir": str(self.repo_root),
            "namespace": "llmd-ns",
            "kubeconfig": "/tmp/kubeconfig",
            "context": "gke-tpu-ctx",
            "service_url": "http://10.0.0.1:8000",
        }
        self.assertEqual(
            utils.validate_instance(prof_req_payload, prof_req_schema), [])

        prof_res_payload = {
            "status":
            "reported",
            "summary":
            "Initial llm-d diagnostics passthrough completed; routed to request-routing and scheduler-batching levers.",
            "run_id":
            "20261020T120000Z-abcdef012345",
            "profiling_report_artifact":
            "/runs/20261020T120000Z-abcdef012345/artifacts/tpu-diagnostics/report.md",
            "xprof_version":
            "llmd-diagnostics-noop",
            "traces": [{
                "path":
                str(self.repo_root),
                "captured":
                False,
                "metrics": {},
                "levers": [
                    {
                        "kind": "request-routing",
                        "summary":
                        "EndpointPickerConfig scorer weights and prefix-cache affinity routing.",
                        "share_pct": 50,
                        "source_tool": "llmd-diagnostics-noop",
                        "next_step": "Tune EndpointPickerConfig weights.",
                    },
                    {
                        "kind": "scheduler-batching",
                        "summary": "vLLM continuous batching parameters.",
                        "share_pct": 30,
                        "source_tool": "llmd-diagnostics-noop",
                        "next_step": "Tune --max-num-batched-tokens.",
                    },
                ],
            }],
        }
        self.assertEqual(
            utils.validate_instance(prof_res_payload, prof_res_schema), [])


class TestEvaluateArmLlmdTelemetry(unittest.TestCase):
    """Tests for evaluate_arm.py with llmd-telemetry.json and study-protocol.json."""

    def setUp(self) -> None:
        intake = study_protocol.build_query_intake(
            prompt=
            "Maximize throughput for Qwen/Qwen3-32B on v6e 2x4 with llm-d for agentx under TTFT p99 <= 800 ms and TPOT p99 <= 35 ms",
            study_id="qwen3-eval-test",
            primary_metric="output_tokens_per_s",
            replicate_count=1,
            noise_floor_pct=1.5,
        )
        self.protocol = study_protocol.compile_study_protocol(intake)
        self.baseline_telemetry = _build_telemetry_payload(
            arm_id="baseline",
            output_tps=1820.0,
            ttft_p99_ms=640.0,
            tpot_p99_ms=24.5,
        )
        self.baseline = {
            "execution_id": "exec-base-01",
            "metric": {
                "name": "output_tokens_per_s",
                "value": 1820.0,
                "unit": "tok/s",
                "replicates": 1,
                "provenance": "measured",
            },
        }

    def test_evaluate_win_tie_loss_and_invalid_gate_violation(self) -> None:
        win_telemetry = _build_telemetry_payload(
            arm_id="arm-win",
            output_tps=2340.0,
            ttft_p99_ms=310.0,
            tpot_p99_ms=22.8,
            prefix_hit_rate=0.79,
        )
        win_res = evaluate_arm.evaluate_arm(
            arm="arm-win",
            execution_id="exec-win-01",
            protocol=self.protocol,
            baseline=self.baseline,
            telemetry=win_telemetry,
        )
        self.assertEqual(win_res["verdict"], "win")
        self.assertAlmostEqual(win_res["metric"]["delta_pct"],
                               28.5714,
                               places=3)
        self.assertTrue(all(g["passed"] for g in win_res["gates"]))

        tie_telemetry = _build_telemetry_payload(
            arm_id="arm-tie",
            output_tps=1830.0,
            ttft_p99_ms=630.0,
            tpot_p99_ms=24.2,
        )
        tie_res = evaluate_arm.evaluate_arm(
            arm="arm-tie",
            execution_id="exec-tie-01",
            protocol=self.protocol,
            baseline=self.baseline,
            telemetry=tie_telemetry,
        )
        self.assertEqual(tie_res["verdict"], "tie")

        loss_telemetry = _build_telemetry_payload(
            arm_id="arm-loss",
            output_tps=1600.0,
            ttft_p99_ms=710.0,
            tpot_p99_ms=26.0,
        )
        loss_res = evaluate_arm.evaluate_arm(
            arm="arm-loss",
            execution_id="exec-loss-01",
            protocol=self.protocol,
            baseline=self.baseline,
            telemetry=loss_telemetry,
        )
        self.assertEqual(loss_res["verdict"], "loss")
        self.assertLess(loss_res["metric"]["delta_pct"], -1.5)

        inadmissible_telemetry = _build_telemetry_payload(
            arm_id="arm-bad-tpot",
            output_tps=2510.0,
            ttft_p99_ms=420.0,
            tpot_p99_ms=64.0,
        )
        inv_res = evaluate_arm.evaluate_arm(
            arm="arm-bad-tpot",
            execution_id="exec-bad-tpot-01",
            protocol=self.protocol,
            baseline=self.baseline,
            telemetry=inadmissible_telemetry,
        )
        self.assertEqual(inv_res["verdict"], "invalid")
        tpot_gate = next(g for g in inv_res["gates"]
                         if g["name"] == "tpot-p99-ceiling")
        self.assertFalse(tpot_gate["passed"])
        self.assertEqual(tpot_gate["value"], 64.0)


class TestClosedLoopMakeItFastStudy(unittest.TestCase):
    """End-to-end multi-round Make-It-Fast closed loop test."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.env_patcher = mock.patch.dict(
            os.environ,
            {"TPU_AGENT_SUITE_HOME": str(self.root / "suite_home")})
        self.env_patcher.start()

    def tearDown(self) -> None:
        self.env_patcher.stop()
        self.temp_dir.cleanup()

    def test_multi_round_llmd_closed_loop_win_and_gate_violation_rollback(
        self, ) -> None:
        study_id = "qwen3-32b-v6e-agentx-make-it-fast"
        intake = study_protocol.build_query_intake(
            prompt=
            ("Onboard and maximize throughput for Qwen/Qwen3-32B on TPU v6e 2x2 with llm-d "
             "for agentx under TTFT p99 <= 800 ms and TPOT p99 <= 35 ms"),
            study_id=study_id,
            primary_metric="output_tokens_per_s",
            replicate_count=1,
            noise_floor_pct=1.5,
            max_rounds=6,
        )
        protocol = study_protocol.compile_study_protocol(intake)
        study.start_study(
            study_id=study_id,
            protocol=protocol,
            package="tpu-agent-suite",
            target="Qwen/Qwen3-32B",
        )

        dynamo_file = self.root / "dynamo-homog.yaml"
        dynamo_file.write_text(_HOMOGENEOUS_DYNAMO_YAML, encoding="utf-8")
        out_dir = self.root / "llmd-bundle"
        trans_res = dynamo_translator.translate_dynamo_to_llmd(
            manifest_path=dynamo_file,
            output_dir=out_dir,
            target_tpu_type="v6e",
            target_tpu_topology="2x2",
        )
        bundle_dir = pathlib.Path(trans_res["bundle_dir"])

        epp_config.mutate_bundle(
            bundle_dir,
            scorers={
                "prefix-cache-scorer": 10.0,
                "queue-depth-scorer": 10.0,
                "kv-utilization-scorer": 80.0,
            },
        )
        base_snap = epp_config.inspect_bundle(bundle_dir)
        self.assertEqual(base_snap["scorer_weights"]["prefix-cache-scorer"],
                         10.0)
        base_fp = base_snap["recipe_fingerprint"]

        base_telemetry = _build_telemetry_payload(
            arm_id="baseline",
            output_tps=1820.0,
            ttft_p99_ms=640.0,
            tpot_p99_ms=24.5,
            prefix_hit_rate=0.31,
        )
        base_telemetry_path = self.root / "baseline-llmd-telemetry.json"
        base_telemetry_path.write_text(json.dumps(base_telemetry, indent=2),
                                       encoding="utf-8")
        baseline_record = {
            "execution_id": "exec-llmd-base-00",
            "metric": {
                "name":
                "output_tokens_per_s",
                "value":
                base_telemetry["throughput"]["output_token_throughput_tps"],
                "unit":
                "tok/s",
                "replicates":
                1,
                "provenance":
                "measured",
            },
            "profile": str(base_telemetry_path),
            "fingerprint": {
                "commit_sha": "a2c9f7f3",
                "config_digest": f"sha256:{base_fp}",
                "cell": "v6e-2x2/8192-1024/c32",
            },
        }
        study.set_baseline(study_id, baseline_record)

        hypotheses = [
            {
                "id":
                "a1-prefix-affinity-routing",
                "kind":
                "load-imbalance",
                "claim":
                "Raise prefix-cache-scorer to 80 and queue-depth-scorer to 50 with 0.80 affinity threshold.",
                "share":
                0.42,
                "headroom":
                0.50,
                "evidence": [{
                    "type":
                    "telemetry",
                    "ref":
                    "cache_memory.prefix_cache.hit_rate_tokens=0.31",
                    "note":
                    "Low prefix hit rate causes redundant prefill recomputation across shared agent prefixes.",
                }],
                "code_refs": [{
                    "path": "01-endpointpickerconfig.yaml"
                }],
                "epp_config": {
                    "scorers": {
                        "prefix-cache-scorer": 80,
                        "queue-depth-scorer": 50,
                        "kv-utilization-scorer": 30,
                    },
                    "filters": {
                        "prefix-cache-affinity-filter": {
                            "affinityThreshold": 0.80
                        }
                    },
                },
            },
            {
                "id":
                "a2-overaggressive-batching",
                "kind":
                "scheduler-batching",
                "claim":
                "Increase max-num-batched-tokens to 32768 and max-num-seqs to 512 on serving pool.",
                "share":
                0.25,
                "headroom":
                0.30,
                "evidence": [{
                    "type":
                    "telemetry",
                    "ref":
                    "latencies.tpot_ms.p99=24.5",
                    "note":
                    "Test larger batch size to push output token throughput.",
                }],
                "code_refs": [{
                    "path": "03-serving-deployment.yaml"
                }],
                "epp_config": {
                    "vllm_flags": {
                        "--max-num-batched-tokens": 32768,
                        "--max-num-seqs": 512,
                    }
                },
            },
        ]
        hb_res = hypothesis_bank.admit_hypotheses(
            study_id=study_id,
            run_id="20261016T160000Z-000000000001",
            hypotheses=hypotheses,
            tpu_version="v6e",
            repo_root=bundle_dir,
        )
        self.assertEqual(
            [a["hypothesis_id"] for a in hb_res["admitted"]],
            [
                "a1-prefix-affinity-routing",
                "a2-overaggressive-batching",
            ],
        )

        # ROUND 1: Execute Arm 1 (a1-prefix-affinity-routing) -> WIN
        run_r1 = "20261016T161000Z-000000000001"
        study.attach_run_to_study(study_id, run_r1)
        claimed_r1 = study.claim_arm(study_id, worker_id="executor-1")
        self.assertEqual(claimed_r1["arm"], "a1-prefix-affinity-routing")

        ckpt_r1 = self.root / "checkpoints" / "round-1"
        epp_config.checkpoint_bundle(bundle_dir, ckpt_r1)

        epp_config.mutate_bundle(
            bundle_dir,
            scorers={
                "prefix-cache-scorer": 80.0,
                "queue-depth-scorer": 50.0,
                "kv-utilization-scorer": 30.0,
            },
            filters={
                "prefix-cache-affinity-filter": {
                    "affinityThreshold": 0.80
                }
            },
        )
        r1_snap = epp_config.inspect_bundle(bundle_dir)
        r1_winning_fp = r1_snap["recipe_fingerprint"]
        self.assertNotEqual(r1_winning_fp, base_fp)
        self.assertEqual(r1_snap["scorer_weights"]["prefix-cache-scorer"],
                         80.0)

        r1_telemetry = _build_telemetry_payload(
            arm_id="a1-prefix-affinity-routing",
            output_tps=2340.0,
            ttft_p50_ms=115.0,
            ttft_p99_ms=310.0,
            tpot_p50_ms=16.8,
            tpot_p99_ms=22.8,
            prefix_hit_rate=0.79,
        )
        r1_eval = evaluate_arm.evaluate_arm(
            arm="a1-prefix-affinity-routing",
            execution_id="exec-llmd-r1-01",
            protocol=protocol,
            baseline=baseline_record,
            telemetry=r1_telemetry,
        )
        self.assertEqual(r1_eval["verdict"], "win")

        doc_r1 = study.load_study(study_id)
        arms_r1 = []
        for a in doc_r1["arms"]:
            a_copy = dict(a)
            if a_copy["arm"] == "a1-prefix-affinity-routing":
                a_copy.pop("claimed_at", None)
                a_copy.pop("claimed_by", None)
                a_copy["status"] = "decided"
                a_copy["result"] = {
                    "verdict": r1_eval["verdict"],
                    "execution_id": r1_eval["execution_id"],
                    "metric": r1_eval["metric"],
                    "gates": r1_eval["gates"],
                }
            arms_r1.append(a_copy)
        study.record_round(study_id, {"arms": arms_r1})

        # ROUND 2: Execute Arm 2 (a2-overaggressive-batching) -> INVALID + ROLLBACK
        run_r2 = "20261016T162000Z-000000000002"
        study.attach_run_to_study(study_id, run_r2)
        claimed_r2 = study.claim_arm(study_id, worker_id="executor-1")
        self.assertEqual(claimed_r2["arm"], "a2-overaggressive-batching")

        ckpt_r2 = self.root / "checkpoints" / "round-2"
        ckpt_r2_meta = epp_config.checkpoint_bundle(bundle_dir, ckpt_r2)
        self.assertEqual(ckpt_r2_meta["recipe_fingerprint"], r1_winning_fp)

        epp_config.mutate_bundle(
            bundle_dir,
            vllm_args={
                "--max-num-batched-tokens": "32768",
                "--max-num-seqs": "512",
            },
        )
        r2_mutated_snap = epp_config.inspect_bundle(bundle_dir)
        self.assertNotEqual(r2_mutated_snap["recipe_fingerprint"],
                            r1_winning_fp)

        r2_telemetry = _build_telemetry_payload(
            arm_id="a2-overaggressive-batching",
            output_tps=2510.0,
            ttft_p50_ms=140.0,
            ttft_p99_ms=390.0,
            tpot_p50_ms=42.0,
            tpot_p99_ms=64.0,
            prefix_hit_rate=0.78,
        )
        r2_eval = evaluate_arm.evaluate_arm(
            arm="a2-overaggressive-batching",
            execution_id="exec-llmd-r2-01",
            protocol=protocol,
            baseline=baseline_record,
            telemetry=r2_telemetry,
        )
        self.assertEqual(r2_eval["verdict"], "invalid")

        rb_r2 = epp_config.rollback_bundle(bundle_dir, ckpt_r2)
        self.assertTrue(rb_r2["fingerprint_matches_checkpoint"])
        self.assertEqual(rb_r2["recipe_fingerprint"], r1_winning_fp)

        restored_snap = epp_config.inspect_bundle(bundle_dir)
        self.assertEqual(
            restored_snap["scorer_weights"]["prefix-cache-scorer"], 80.0)
        self.assertEqual(
            restored_snap["pools"]["serving"]["vllm_flags"].get(
                "--max-num-batched-tokens"),
            r1_snap["pools"]["serving"]["vllm_flags"].get(
                "--max-num-batched-tokens"),
        )

        doc_r2 = study.load_study(study_id)
        arms_r2 = []
        for a in doc_r2["arms"]:
            a_copy = dict(a)
            if a_copy["arm"] == "a2-overaggressive-batching":
                a_copy.pop("claimed_at", None)
                a_copy.pop("claimed_by", None)
                a_copy["status"] = "decided"
                a_copy["result"] = {
                    "verdict": r2_eval["verdict"],
                    "execution_id": r2_eval["execution_id"],
                    "metric": r2_eval["metric"],
                    "gates": r2_eval["gates"],
                }
            arms_r2.append(a_copy)
        final_doc = study.record_round(study_id, {"arms": arms_r2})

        self.assertEqual(study.study_round(final_doc), 2)
        by_arm = {a["arm"]: a for a in final_doc["arms"]}
        self.assertEqual(
            by_arm["a1-prefix-affinity-routing"]["result"]["verdict"], "win")
        self.assertEqual(
            by_arm["a2-overaggressive-batching"]["result"]["verdict"],
            "invalid",
        )


if __name__ == "__main__":
    unittest.main()
