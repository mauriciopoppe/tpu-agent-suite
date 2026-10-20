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
"""Tests for query intake and direct Study Protocol compilation."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import compiler, utils  # noqa: E402


def _load_study_protocol_module():
    script_path = _PACKAGE_ROOT / "scripts" / "common" / "study_protocol.py"
    spec = importlib.util.spec_from_file_location("study_protocol",
                                                  script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestQueryUnderstandingProtocol(unittest.TestCase):
    """Verifies query intake parsing and native study-protocol.json compilation."""

    @classmethod
    def setUpClass(cls):
        cls.proto_mod = _load_study_protocol_module()
        cls.onboarding_req_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                                    "onboarding-request.json")
        cls.study_req_schema = utils.load_json(_PACKAGE_ROOT / "schemas" /
                                               "study-request.json")
        cls.study_protocol_schema = utils.load_json(
            _REPO_ROOT / "framework" / "schemas" / "study-protocol.json")

    def test_schemas_are_structurally_valid_and_examples_pass(self):
        self.assertFalse(
            (_PACKAGE_ROOT / "schemas" / "objective-function.json").exists(),
            "objective-function.json must be retired in favor of framework/schemas/study-protocol.json",
        )
        for name, schema in (
            ("onboarding-request.json", self.onboarding_req_schema),
            ("study-request.json", self.study_req_schema),
            ("study-protocol.json", self.study_protocol_schema),
        ):
            with self.subTest(schema=name):
                errors = utils.validate_schema_definition(schema)
                self.assertEqual(errors, [],
                                 f"{name} invalid schema: {errors}")
                examples = schema.get("examples", [])
                self.assertTrue(examples, f"{name} must define examples")
                for idx, ex in enumerate(examples):
                    ex_errors = utils.validate_instance(ex, schema)
                    self.assertEqual(
                        ex_errors,
                        [],
                        f"{name} example #{idx} failed validation: {ex_errors}",
                    )

    def test_parse_agentx_onboard_and_optimize_query(self):
        prompt = (
            "Onboard Qwen 3.5 on GKE TPU v7x and optimize throughput for AgentX"
        )
        parsed = self.proto_mod.parse_user_query(prompt)
        self.assertEqual(parsed["intent"], "onboard_and_optimize")
        self.assertEqual(parsed["target"], "Qwen/Qwen3.5-32B")
        self.assertEqual(parsed["tpu_type"], "tpu7x")
        self.assertEqual(parsed["topology"], "2x2x2")
        self.assertEqual(parsed["onboarding_path"], "dynamo_unwiring")
        self.assertEqual(parsed["primary_metric"], "output_tokens_per_s")

    def test_parse_pure_onboarding_query_does_not_inflate_scope(self):
        prompt = "do onboarding on gpt oss 20b"
        parsed = self.proto_mod.parse_user_query(prompt)
        self.assertEqual(parsed["intent"], "onboard_only")
        self.assertEqual(parsed["target"], "openai/gpt-oss-20b")
        self.assertIsNone(parsed["onboarding_path"])

        intake = self.proto_mod.build_query_intake(prompt=prompt)
        self.assertEqual(intake["staged_plan"]["stages"],
                         ["baseline_onboarding"])
        onboarding_req = self.proto_mod.compile_onboarding_request(
            intake, prompt=prompt)
        self.assertEqual(onboarding_req["mode"], "inference")
        self.assertNotIn("onboarding_path", onboarding_req)
        self.assertEqual(
            utils.validate_instance(onboarding_req,
                                    self.onboarding_req_schema),
            [],
        )

    def test_parse_pure_dynamo_unwiring_query_does_not_inflate_to_study_loop(
            self):
        prompt = (
            "We have an NVIDIA Dynamo deployment running in namespace dynamo-gpu "
            "on our GPU Kubernetes cluster, and we want to unwire it to llm-d "
            "on our GKE TPU7x cluster tpu7x-serving-cluster in namespace vllm-serving"
        )
        parsed = self.proto_mod.parse_user_query(prompt)
        self.assertEqual(parsed["intent"], "onboard_only")
        self.assertEqual(parsed["onboarding_path"], "dynamo_unwiring")
        self.assertEqual(parsed["tpu_type"], "tpu7x")

        intake = self.proto_mod.build_query_intake(prompt=prompt)
        self.assertEqual(intake["staged_plan"]["stages"],
                         ["baseline_onboarding"])

    def test_parse_dynamo_unwiring_with_sla_constraints(self):
        prompt = (
            "Unwire our NVIDIA Dynamo deployment for Qwen/Qwen3-32B to llm-d on "
            "TPU v6e-8 and optimize throughput with TTFT <= 1.5s and TPOT <= 20ms"
        )
        parsed = self.proto_mod.parse_user_query(prompt)
        self.assertEqual(parsed["intent"], "onboard_and_optimize")
        self.assertEqual(parsed["target"], "Qwen/Qwen3-32B")
        self.assertEqual(parsed["tpu_type"], "v6e")
        self.assertEqual(parsed["topology"], "2x4")
        self.assertEqual(parsed["onboarding_path"], "dynamo_unwiring")
        self.assertEqual(parsed["primary_metric"], "output_tokens_per_s")
        self.assertEqual(parsed["ttft_p99_ms"], 1500.0)
        self.assertEqual(parsed["tpot_p99_ms"], 20.0)

        protocol = self.proto_mod.compile_study_protocol(prompt=prompt)
        self.assertEqual(
            utils.validate_instance(protocol, self.study_protocol_schema), [])
        gates_by_metric = {
            g["metric"]: g
            for g in protocol["admissibility_gates"]
        }
        self.assertEqual(gates_by_metric["ttft_p99_ms"]["bound"], 1500.0)
        self.assertEqual(gates_by_metric["tpot_p99_ms"]["bound"], 20.0)

    def test_parse_existing_deployment_latency_query(self):
        prompt = (
            "Optimize P99 TTFT on our existing Qwen/Qwen3-32B deployment on "
            "v6e-8")
        parsed = self.proto_mod.parse_user_query(prompt)
        self.assertEqual(parsed["intent"], "optimize_only")
        self.assertEqual(parsed["onboarding_path"], "existing_deployment")
        self.assertEqual(parsed["tpu_type"], "v6e")
        self.assertEqual(parsed["topology"], "2x4")
        self.assertEqual(parsed["primary_metric"], "ttft_p50_ms")

        intake = self.proto_mod.build_query_intake(prompt=prompt)
        self.assertEqual(
            intake["staged_plan"]["stages"],
            ["empirical_profiling", "study_loop"],
        )

        protocol = self.proto_mod.compile_study_protocol(intake)
        self.assertEqual(
            utils.validate_instance(protocol, self.study_protocol_schema), [])

    def test_end_to_end_compilation_validates_against_study_protocol_and_request_schemas(
            self):
        prompt = (
            "Onboard Qwen 3.5 on GKE TPU v7x and optimize throughput for AgentX"
        )
        intake = self.proto_mod.build_query_intake(prompt=prompt)
        self.assertEqual(
            intake["staged_plan"]["stages"],
            ["baseline_onboarding", "empirical_profiling", "study_loop"],
        )

        protocol = self.proto_mod.compile_study_protocol(intake)
        self.assertEqual(
            utils.validate_instance(protocol, self.study_protocol_schema), [])
        self.assertEqual(protocol["primary_metric"]["name"],
                         "output_tokens_per_s")
        self.assertEqual(protocol["primary_metric"]["direction"], "maximize")
        self.assertEqual(
            protocol["measurement_cell"],
            {
                "topology": "tpu7x-2x2x2",
                "shape": "8192-1024",
                "concurrency": 32,
            },
        )

        onboarding_req = self.proto_mod.compile_onboarding_request(
            intake,
            prompt=prompt,
            run_id="20261012T120000Z-000000000000",
        )
        self.assertEqual(
            utils.validate_instance(onboarding_req,
                                    self.onboarding_req_schema),
            [],
        )
        self.assertEqual(onboarding_req["mode"], "distributed")
        self.assertEqual(onboarding_req["onboarding_path"], "dynamo_unwiring")

        study_req = self.proto_mod.compile_study_request(
            intake,
            prompt=prompt,
            run_id="20261012T120000Z-000000000000",
            levers=["scheduler-batching", "kernel-block"],
            protocol=protocol,
        )
        self.assertEqual(
            utils.validate_instance(study_req, self.study_req_schema), [])
        self.assertEqual(study_req["study_id"], intake["study_id"])
        self.assertEqual(
            utils.validate_instance(study_req["protocol"],
                                    self.study_protocol_schema),
            [],
        )

    def test_cli_writes_all_compiled_artifacts(self):
        script = _PACKAGE_ROOT / "scripts" / "common" / "study_protocol.py"
        prompt = (
            "Onboard Qwen 3.5 on GKE TPU v7x and optimize throughput for AgentX"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = pathlib.Path(tmpdir) / "compiled"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--prompt",
                    prompt,
                    "--run-id",
                    "20261012T120000Z-000000000000",
                    "--output-dir",
                    str(out_dir),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            bundle = json.loads(proc.stdout)
            self.assertIn("study_protocol", bundle)
            self.assertIn("onboarding_request", bundle)
            self.assertIn("study_request", bundle)
            self.assertNotIn("objective_function", bundle)
            self.assertFalse((out_dir / "objective-function.json").exists())

            for filename, schema in (
                ("protocol.json", self.study_protocol_schema),
                ("onboarding-request.json", self.onboarding_req_schema),
                ("study-request.json", self.study_req_schema),
            ):
                artifact_path = out_dir / filename
                self.assertTrue(artifact_path.is_file(),
                                f"Missing {artifact_path}")
                payload = utils.load_json(artifact_path)
                self.assertEqual(
                    utils.validate_instance(payload, schema),
                    [],
                    f"Written {filename} failed schema validation",
                )

    def test_query_understanding_reference_and_skill_contracts(self):
        compiler.verify_package(_PACKAGE_ROOT)
        query_ref_path = (_PACKAGE_ROOT / "references" / "tpu-agent" /
                          "query-understanding.md")
        query_ref = query_ref_path.read_text(encoding="utf-8")
        self.assertLess(len(query_ref.splitlines()), 205)
        for expected in (
                '"dynamo-unwiring"',
                "llmd-dynamo-unwiring",
                "{{schema:onboarding-request.json}}",
                "{{schema:study-request.json}}",
                "{{script:common/study_protocol.py}}",
        ):
            self.assertIn(expected, query_ref)
        for retired in (
                "1a_single_model_vllm",
                "1b_llm_d_tpu",
                "1c_dynamo_unwiring",
                "single_model_vllm",
                "llmd_tpu",
                "objective-function.json",
        ):
            self.assertNotIn(retired, query_ref)

        kb_page = (_PACKAGE_ROOT / "data" / "tpu-knowledge-base" / "pages" /
                   "llmd-dynamo-unwiring.md").read_text(encoding="utf-8")
        for expected in (
                "dynamo_unwiring",
                "existing_deployment",
                "framework/schemas/study-protocol.json",
                "protocol.json",
        ):
            self.assertIn(expected, kb_page)


if __name__ == "__main__":
    unittest.main()
