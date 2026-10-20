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
"""Evaluates arm execution measurements against protocol baselines and gates.

Enforces:
1. Replicate count threshold satisfaction.
2. Directional delta calculation against baseline.
3. Co-located same_execution checks for all admissibility gates.
4. Noise floor gating to separate wins from ties.
5. Benchmark harness tamper detection (fail-closed git tracked check / SHA-256).
6. Canonical arm fingerprinting to detect duplicate hypotheses across rounds/journal.
7. Normalized run-directory artifact path containment (rejects .. traversal, CWD, and package writes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import posixpath
import re
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

_HERE = pathlib.Path(__file__).resolve()
# The directory holding the framework package: the package's own in a build,
# the repository root in the repository.
for _root in (_HERE.parents[2], _HERE.parents[3]):
    if (_root / "framework").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from framework import study  # noqa: E402

_NUMERIC_STR_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

_SCORER_ALIASES: dict[str, str] = {
    "prefix-cache-scorer": "prefix-cache-scorer",
    "precise-prefix-cache-scorer": "prefix-cache-scorer",
    "queue-depth-scorer": "queue-depth-scorer",
    "queue-scorer": "queue-depth-scorer",
    "active-request-scorer": "queue-depth-scorer",
    "kv-utilization-scorer": "kv-utilization-scorer",
    "kv-cache-utilization-scorer": "kv-utilization-scorer",
    "token-load-scorer": "token-load-scorer",
    "no-hit-lru-scorer": "no-hit-lru-scorer",
    "session-affinity-scorer": "session-affinity-scorer",
}


def _normalize_vllm_flag_key(key: str) -> str:
    cleaned = str(key).strip().replace("_", "-")
    if not cleaned.startswith("--"):
        cleaned = "--" + cleaned.lstrip("-")
    return cleaned


def _normalize_distributed_params(
    target_parameters: Mapping[str, Any] | None = None,
    epp_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize distributed-serving (`llm-d`) parameters and `epp_config`."""
    raw: dict[str, Any] = dict(target_parameters or {})
    nested_epp = raw.pop("epp_config", None)
    merged_epp: dict[str, Any] = {}
    if isinstance(nested_epp, Mapping):
        merged_epp.update(nested_epp)
    if isinstance(epp_config, Mapping):
        merged_epp.update(epp_config)
    for k, v in merged_epp.items():
        if k not in raw:
            raw[k] = v
        elif isinstance(raw[k], Mapping) and isinstance(v, Mapping):
            raw[k] = {**dict(raw[k]), **dict(v)}

    # Canonicalize scorer keys and weights if present
    if "scorers" in raw and isinstance(raw["scorers"], Mapping):
        norm_scorers: dict[str, Any] = {}
        for s_key, s_val in raw["scorers"].items():
            canon_s = _SCORER_ALIASES.get(
                str(s_key).strip(),
                str(s_key).strip())
            if isinstance(s_val, Mapping):
                w = s_val.get("weight", 1)
                params = s_val.get("parameters") or {}
                if params:
                    norm_scorers[canon_s] = {
                        "weight": w,
                        "parameters": dict(params)
                    }
                else:
                    norm_scorers[canon_s] = w
            else:
                norm_scorers[canon_s] = s_val
        raw["scorers"] = norm_scorers

    # Canonicalize vllm_flags keys if present
    if "vllm_flags" in raw and isinstance(raw["vllm_flags"], Mapping):
        raw["vllm_flags"] = {
            _normalize_vllm_flag_key(fk): fv
            for fk, fv in raw["vllm_flags"].items()
        }

    # Fold top-level vLLM flag keys (e.g. max_num_batched_tokens, max_num_seqs,
    # gpu_memory_utilization) into canonical form when distributed keys exist
    _TOP_VLLM_KEYS = (
        "max_num_batched_tokens",
        "max-num-batched-tokens",
        "--max-num-batched-tokens",
        "max_num_seqs",
        "max-num-seqs",
        "--max-num-seqs",
        "gpu_memory_utilization",
        "gpu-memory-utilization",
        "--gpu-memory-utilization",
    )
    has_llmd_context = any(k in raw for k in (
        "scorers",
        "filters",
        "vllm_flags",
        "saturation_thresholds",
        "serving_mode",
    )) or any(k in raw for k in _TOP_VLLM_KEYS)

    if has_llmd_context:
        vflags = dict(raw.get("vllm_flags") or {})
        for tk in list(raw.keys()):
            if tk in _TOP_VLLM_KEYS or (
                    tk in ("block_size", "block-size", "--block-size") and
                ("scorers" in raw or "vllm_flags" in raw or "filters" in raw)):
                vflags[_normalize_vllm_flag_key(tk)] = raw.pop(tk)
        if vflags:
            raw["vllm_flags"] = vflags

    return raw


def _lookup_dotted(data: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


_LLMD_METRIC_PATHS: dict[str, str] = {
    "output_token_throughput_tps": "throughput.output_token_throughput_tps",
    "output_tokens_per_s": "throughput.output_token_throughput_tps",
    "tokens_per_s": "throughput.output_token_throughput_tps",
    "total_token_throughput_tps": "throughput.total_token_throughput_tps",
    "total_tokens_per_s": "throughput.total_token_throughput_tps",
    "input_token_throughput_tps": "throughput.input_token_throughput_tps",
    "input_tokens_per_s": "throughput.input_token_throughput_tps",
    "request_throughput_rps": "throughput.request_throughput_rps",
    "requests_per_s": "throughput.request_throughput_rps",
    "rps": "throughput.request_throughput_rps",
    "goodput_pct": "throughput.goodput.goodput_percentage",
    "goodput_percentage": "throughput.goodput.goodput_percentage",
    "goodput-floor": "throughput.goodput.goodput_percentage",
    "goodput_rps": "throughput.goodput.completed_within_sla_rps",
    "completed_within_sla_rps": "throughput.goodput.completed_within_sla_rps",
    "ttft_p50_ms": "latencies.ttft_ms.p50",
    "ttft_median_ms": "latencies.ttft_ms.p50",
    "ttft_p90_ms": "latencies.ttft_ms.p90",
    "ttft_p95_ms": "latencies.ttft_ms.p95",
    "ttft_p99_ms": "latencies.ttft_ms.p99",
    "ttft_ms": "latencies.ttft_ms.p99",
    "ttft-ceiling": "latencies.ttft_ms.p99",
    "ttft-p99-ceiling": "latencies.ttft_ms.p99",
    "ttft_mean_ms": "latencies.ttft_ms.mean",
    "tpot_p50_ms": "latencies.tpot_ms.p50",
    "tpot_median_ms": "latencies.tpot_ms.p50",
    "tpot_p90_ms": "latencies.tpot_ms.p90",
    "tpot_p95_ms": "latencies.tpot_ms.p95",
    "tpot_p99_ms": "latencies.tpot_ms.p99",
    "tpot_ms": "latencies.tpot_ms.p99",
    "tpot-ceiling": "latencies.tpot_ms.p99",
    "tpot-p99-ceiling": "latencies.tpot_ms.p99",
    "tpot_mean_ms": "latencies.tpot_ms.mean",
    "itl_p50_ms": "latencies.itl_ms.p50",
    "itl_p90_ms": "latencies.itl_ms.p90",
    "itl_p95_ms": "latencies.itl_ms.p95",
    "itl_p99_ms": "latencies.itl_ms.p99",
    "itl_ms": "latencies.itl_ms.p99",
    "itl_mean_ms": "latencies.itl_ms.mean",
    "e2e_p50_ms": "latencies.e2e_ms.p50",
    "e2e_p90_ms": "latencies.e2e_ms.p90",
    "e2e_p95_ms": "latencies.e2e_ms.p95",
    "e2e_p99_ms": "latencies.e2e_ms.p99",
    "e2e_ms": "latencies.e2e_ms.p99",
    "e2e_mean_ms": "latencies.e2e_ms.mean",
    "queue_wait_p99_ms": "latencies.queue_wait_ms.p99",
    "queue_wait_ms": "latencies.queue_wait_ms.p99",
    "error_rate": "system_reliability.error_rate",
    "timeout_rate": "system_reliability.timeout_rate",
    "prefix_cache_hit_rate": "cache_memory.prefix_cache.hit_rate_tokens",
    "hit_rate_tokens": "cache_memory.prefix_cache.hit_rate_tokens",
    "kv_cache_usage_peak": "cache_memory.kv_cache.usage_factor_peak",
    "usage_factor_peak": "cache_memory.kv_cache.usage_factor_peak",
    "kv_cache_usage_mean": "cache_memory.kv_cache.usage_factor_mean",
    "usage_factor_mean": "cache_memory.kv_cache.usage_factor_mean",
    "worker_load_skew_ratio": "worker_telemetry.worker_load_skew_ratio",
}


def extract_llmd_metric(telemetry: Mapping[str, Any],
                        metric_name: str) -> float | None:
    """Extract a scalar metric value from an `llmd-telemetry.json` document."""
    key = str(metric_name).strip()
    if key in ("error_rate_pct", "error-rate-ceiling"):
        raw_err = _lookup_dotted(telemetry, "system_reliability.error_rate")
        if isinstance(raw_err, (int, float)) and not isinstance(raw_err, bool):
            return float(raw_err) * 100.0
        return None
    if key in ("timeout_rate_pct", "timeout-rate-ceiling"):
        raw_to = _lookup_dotted(telemetry, "system_reliability.timeout_rate")
        if isinstance(raw_to, (int, float)) and not isinstance(raw_to, bool):
            return float(raw_to) * 100.0
        return None

    dotted = _LLMD_METRIC_PATHS.get(key, key)
    val = _lookup_dotted(telemetry, dotted)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _load_telemetry_items(
    telemetry: Mapping[str, Any] | str | pathlib.Path | Sequence[Any] | None,
) -> list[dict[str, Any]]:
    """Normalize telemetry argument (dict, path, JSON string, or list) into a list of dicts."""
    if telemetry is None:
        return []
    if isinstance(telemetry, Mapping):
        return [dict(telemetry)]
    if isinstance(telemetry, (str, pathlib.Path)):
        text = str(telemetry).strip()
        if text.startswith(("{", "[")):
            parsed = json.loads(text)
        else:
            parsed = json.loads(pathlib.Path(text).read_text(encoding="utf-8"))
        if isinstance(parsed, Mapping):
            return [dict(parsed)]
        if isinstance(parsed, Sequence):
            return [dict(item) for item in parsed if isinstance(item, Mapping)]
        return []
    if isinstance(telemetry, Sequence):
        items: list[dict[str, Any]] = []
        for elem in telemetry:
            items.extend(_load_telemetry_items(elem))
        return items
    return []


def extract_llmd_gate_measurements(
    telemetry_items: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    execution_id: str,
) -> list[dict[str, Any]]:
    """Extract gate measurements from `llmd-telemetry.json` items for `protocol` gates."""
    if not telemetry_items:
        return []
    out: list[dict[str, Any]] = []
    for gate in protocol.get("admissibility_gates") or []:
        if not isinstance(gate, Mapping) or "name" not in gate:
            continue
        g_name = str(gate["name"])
        g_metric = str(gate.get("metric") or g_name)
        vals: list[float] = []
        for t_doc in telemetry_items:
            v = extract_llmd_metric(t_doc, g_metric)
            if v is None and g_metric != g_name:
                v = extract_llmd_metric(t_doc, g_name)
            # Note: if gate metric is `error_rate_pct` and bound <= 1.0 while raw error_rate
            # was intended as a fraction or percentage, `extract_llmd_metric` handles conversion.
            if v is not None:
                vals.append(v)
        if vals:
            out.append({
                "name": g_name,
                "value": float(statistics.median(vals)),
                "execution_id": execution_id,
            })
    return out


def compare_value(val: float, comparator: str, bound: float) -> bool:
    """Evaluate comparator between measurement value and bound."""
    if comparator in ("<=", "le", "lte"):
        return val <= bound
    if comparator in ("<", "lt"):
        return val < bound
    if comparator in (">=", "ge", "gte"):
        return val >= bound
    if comparator in (">", "gt"):
        return val > bound
    if comparator in ("==", "eq"):
        return abs(val - bound) < 1e-9
    raise ValueError(f"unsupported comparator: {comparator!r}")


def to_round_result(arm_result: dict[str, Any]) -> dict[str, Any]:
    """Project an arm-result.json envelope to a study-round.json result block."""
    allowed = ("verdict", "execution_id", "metric", "gates", "artifacts")
    return {k: arm_result[k] for k in allowed if k in arm_result}


def _canonicalize_number(num: float) -> int | float:
    if math.isfinite(num) and num.is_integer():
        return int(num)
    return round(num, 9)


def _canonicalize_value(value: Any,
                        *,
                        coerce_numeric_strings: bool = False) -> Any:
    """Normalize numbers, strings, lists, and dicts for deterministic hashing."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return _canonicalize_number(float(value))
    if isinstance(value, str):
        collapsed = " ".join(value.strip().split())
        if coerce_numeric_strings and _NUMERIC_STR_RE.match(collapsed):
            try:
                return _canonicalize_number(float(collapsed))
            except ValueError:
                pass
        return collapsed
    if isinstance(value, Mapping):
        return {
            str(k):
            _canonicalize_value(v,
                                coerce_numeric_strings=coerce_numeric_strings)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_value(v,
                                coerce_numeric_strings=coerce_numeric_strings)
            for v in value
        ]
    return str(value)


def compute_arm_fingerprint(
    kind: str,
    target_parameters: Mapping[str, Any] | None = None,
    claim: str = "",
    epp_config: Mapping[str, Any] | None = None,
) -> str:
    """Return a 16-hex-character SHA-256 fingerprint for an arm hypothesis.

    When target_parameters (or epp_config) is non-empty, the fingerprint is
    computed from (kind, canonical(target_parameters + epp_config)) so minor
    claim rewordings, scorer alias differences, or CLI numeric stringification
    ("128" vs 128 vs 128.0) do not defeat deduplication.
    When target_parameters and epp_config are empty, normalized claim text is included.
    """
    merged_params = _normalize_distributed_params(
        target_parameters=target_parameters,
        epp_config=epp_config,
    )
    norm_params = _canonicalize_value(merged_params,
                                      coerce_numeric_strings=True)
    payload: dict[str, Any] = {
        "kind": str(kind).strip().lower(),
        "target_parameters": norm_params,
    }
    if not norm_params:
        payload["claim"] = _canonicalize_value(claim).lower()
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def find_duplicate_arm(
    kind: str,
    target_parameters: Mapping[str, Any] | None,
    prior_arms: Sequence[Mapping[str, Any]],
    claim: str = "",
    current_arm_id: str | None = None,
    epp_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Find an already-evaluated or in-flight prior arm with the same fingerprint."""
    candidate_fp = compute_arm_fingerprint(
        kind=kind,
        target_parameters=target_parameters,
        claim=claim,
        epp_config=epp_config,
    )
    for entry in prior_arms:
        prior_id = entry.get("arm")
        if current_arm_id and prior_id == current_arm_id:
            continue
        prior_status = entry.get("status", "")
        prior_result = entry.get("result") or {}
        prior_verdict = prior_result.get("verdict")
        # Only decided arms (or arms carrying a verdict) and active in_flight arms
        # block re-execution; open/blocked arms have not been tested yet.
        if prior_status not in ("decided", "in_flight") and not prior_verdict:
            continue
        prior_kind = entry.get("kind", "")
        if not prior_kind:
            continue
        prior_fp = compute_arm_fingerprint(
            kind=prior_kind,
            target_parameters=entry.get("target_parameters"),
            claim=entry.get("claim", ""),
            epp_config=entry.get("epp_config"),
        )
        if prior_fp == candidate_fp:
            return dict(entry)
    return None


def _verify_git_tracked_clean(resolved: pathlib.Path) -> tuple[bool, str]:
    """Verify that resolved is a tracked file in a git worktree with zero local modifications.

    Returns (verified_clean, failure_reason).
    """
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--",
             str(resolved)],
            cwd=str(resolved.parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if tracked.returncode != 0:
            return False, "not tracked by git"

        status = subprocess.run(
            ["git", "status", "--porcelain", "--ignored", "--",
             str(resolved)],
            cwd=str(resolved.parent),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode != 0:
            return False, "git status failed"
        if status.stdout.strip():
            status_line = status.stdout.strip().splitlines()[0].strip()
            return False, f"modified in git ({status_line})"
        return True, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"git unavailable ({exc})"


def check_harness_integrity(
    protected_paths: Sequence[str | pathlib.Path],
    repo_root: str | pathlib.Path | None = None,
    expected_sha256: Mapping[str, str] | None = None,
) -> list[str]:
    """Verify that protected benchmark harness files are unmodified (fail-closed).

    A protected file is accepted ONLY if:
    1. It exists on disk, AND
    2. Either its SHA-256 matches an entry in expected_sha256 (or inline PATH:SHA256),
       OR it is a tracked file in a git repository with a clean working tree status.
    If neither verification source is available (e.g., an untracked or out-of-repo file
    with no expected_sha256), a violation is recorded so "unverifiable" never passes silently.
    """
    violations: list[str] = []
    base_dir = pathlib.Path(
        repo_root).resolve() if repo_root else pathlib.Path.cwd()
    checksums = dict(expected_sha256 or {})

    for item in protected_paths:
        raw_str = str(item)
        inline_hash: str | None = None
        if ":" in raw_str and not pathlib.Path(raw_str).exists():
            candidate_path, candidate_sha = raw_str.rsplit(":", 1)
            if len(candidate_sha) == 64 and all(c in "0123456789abcdefABCDEF"
                                                for c in candidate_sha):
                raw_str = candidate_path
                inline_hash = candidate_sha.lower()

        p = pathlib.Path(raw_str)
        resolved = p if p.is_absolute() else (base_dir / p)
        display = raw_str

        if not resolved.is_file():
            violations.append(f"protected benchmark file missing: {display}")
            continue

        expected_hash = (inline_hash or checksums.get(display)
                         or checksums.get(str(resolved)))
        if expected_hash:
            actual_hash = hashlib.sha256(
                resolved.read_bytes()).hexdigest().lower()
            if actual_hash != expected_hash.lower():
                violations.append(
                    f"protected benchmark file checksum mismatch: {display}")
                continue
            # Even when checksum matches, if the file is also tracked in git and
            # has uncommitted changes against HEAD, flag it.
            git_ok, git_reason = _verify_git_tracked_clean(resolved)
            if not git_ok and git_reason.startswith("modified in git"):
                violations.append(
                    f"protected benchmark file {git_reason}: {display}")
            continue

        # No expected_sha256 provided: must be verifiable as a clean git-tracked file.
        git_ok, git_reason = _verify_git_tracked_clean(resolved)
        if not git_ok:
            if git_reason.startswith("modified in git"):
                violations.append(
                    f"protected benchmark file {git_reason}: {display}")
            else:
                violations.append(
                    f"protected benchmark file unverifiable ({git_reason} and no expected_sha256 provided): {display}"
                )

    return violations


def _normalize_artifact_path(raw: str) -> tuple[str, str, bool]:
    """Return (scheme_prefix, posix_normalized_path, had_traversal_segment)."""
    slashed = raw.strip().replace("\\", "/")
    scheme = ""
    path_part = slashed
    if "://" in slashed:
        scheme_part, remainder = slashed.split("://", 1)
        scheme = f"{scheme_part}://"
        path_part = "/" + remainder.lstrip("/")

    segments = [seg for seg in path_part.split("/") if seg]
    had_traversal = ".." in segments or "." in segments
    normalized_path = posixpath.normpath(path_part)
    if scheme:
        norm_full = scheme + normalized_path.lstrip("/")
    else:
        norm_full = normalized_path
    return scheme, norm_full, had_traversal


def check_artifact_paths(
    artifacts: Sequence[str],
    run_id: str | None = None,
) -> list[str]:
    """Verify every artifact sits under the active run's artifacts directory.

    That one rule is also what keeps an artifact out of the installed package
    and out of the working directory: a path under `<run_id>/artifacts/` is in
    neither. So the run id is required. Without it nothing here can tell a
    run's directory from any other that has `artifacts` in its name.
    """
    if not run_id:
        return [
            "artifact path cannot be checked without the active run_id"
            f" (pass --run-id): {raw}" for raw in artifacts
        ]
    violations: list[str] = []
    for raw in artifacts:
        if not raw or not raw.strip():
            violations.append("empty artifact path is not allowed")
            continue

        _scheme, norm, had_traversal = _normalize_artifact_path(raw)
        if had_traversal or norm == ".." or norm.startswith(
                "../") or "/../" in f"/{norm}/":
            violations.append(
                f"path traversal ('..' or '.') in artifact path is forbidden: {raw}"
            )
            continue

        # Must land under a run's artifacts directory, never bare in CWD.
        if "/" not in norm or norm.startswith("./") or norm.startswith("../"):
            violations.append(
                f"bare/relative CWD artifact path is forbidden (must be under <run_id>/artifacts/): {raw}"
            )
            continue

        if "/artifacts/" not in norm and not norm.endswith("/artifacts"):
            violations.append(
                f"artifact path must reside under {{runs}}/<run_id>/artifacts/: {raw}"
            )
            continue

        # Verify that after normalization the path stays strictly inside <run_id>/artifacts/
        expected_anchor = f"/{run_id}/artifacts/"
        expected_rel_prefix = f"{run_id}/artifacts/"
        norm_with_trailing = norm if norm.endswith("/") else f"{norm}/"
        if (expected_anchor not in norm_with_trailing
                and not norm_with_trailing.startswith(expected_rel_prefix)):
            violations.append(
                f"artifact path does not reside under active run_id {run_id!r}: {raw}"
            )

    return violations


def evaluate_arm(
    arm: str,
    execution_id: str,
    replicates: Sequence[float] | None = None,
    protocol: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    gate_measurements: Sequence[dict[str, Any]] | None = None,
    *,
    telemetry: Mapping[str, Any] | str | pathlib.Path | Sequence[Any]
    | None = None,
    llmd_telemetry: Mapping[str, Any] | str | pathlib.Path | Sequence[Any]
    | None = None,
    epp_config: Mapping[str, Any] | None = None,
    protected_paths: Sequence[str | pathlib.Path] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    repo_root: str | pathlib.Path | None = None,
    kind: str | None = None,
    target_parameters: Mapping[str, Any] | None = None,
    claim: str = "",
    prior_arms: Sequence[Mapping[str, Any]] | None = None,
    artifacts: Sequence[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate arm execution and return an arm-result.json compliant dict."""
    protocol = protocol or {}
    primary = protocol.get("primary_metric") or {}
    metric_name = primary.get("name", "metric")
    direction = primary.get("direction", "maximize")
    unit = primary.get("unit", "")
    noise_floor_pct = float(protocol.get("noise_floor_pct", 0.0))
    required_reps = int(protocol.get("replicate_count", 1))

    telemetry_items = _load_telemetry_items(
        telemetry if telemetry is not None else llmd_telemetry)
    resolved_replicates: list[float] = [float(x) for x in (replicates or [])]
    if not resolved_replicates and telemetry_items:
        for t_doc in telemetry_items:
            val = extract_llmd_metric(t_doc, metric_name)
            if val is not None:
                resolved_replicates.append(val)

    integrity_reasons: list[str] = []

    # Check duplicate hypothesis against prior decided/in-flight arms if provided.
    if kind and prior_arms:
        dup = find_duplicate_arm(
            kind=kind,
            target_parameters=target_parameters,
            prior_arms=prior_arms,
            claim=claim,
            current_arm_id=arm,
            epp_config=epp_config,
        )
        if dup is not None:
            dup_verdict = (dup.get("result")
                           or {}).get("verdict", dup.get("status", "unknown"))
            integrity_reasons.append(
                f"duplicate of prior arm {dup.get('arm')!r} (verdict={dup_verdict})"
            )

    # Check immutable benchmark harness files if protected_paths provided.
    if protected_paths:
        integrity_reasons.extend(
            check_harness_integrity(
                protected_paths=protected_paths,
                repo_root=repo_root,
                expected_sha256=expected_sha256,
            ))

    # Check artifact paths for containment in the active run's directory.
    if artifacts is not None:
        integrity_reasons.extend(
            check_artifact_paths(artifacts=artifacts, run_id=run_id))

    # Evaluate admissibility gates first so declared gates are always reported.
    declared_gates = protocol.get("admissibility_gates") or []
    measured_gates_map: dict[str, dict[str, Any]] = {}
    if telemetry_items:
        for g_item in extract_llmd_gate_measurements(telemetry_items, protocol,
                                                     execution_id):
            measured_gates_map[g_item["name"]] = g_item
    for g_item in gate_measurements or []:
        measured_gates_map[g_item["name"]] = g_item

    evaluated_gates: list[dict[str, Any]] = []
    all_gates_passed = True
    telemetry_valid = len(integrity_reasons) == 0

    for gate in declared_gates:
        g_name = gate["name"]
        g_same_exec = bool(gate.get("same_execution", False))
        g_comp = gate["comparator"]
        g_bound = float(gate["bound"])

        m_gate = measured_gates_map.get(g_name)
        if not m_gate:
            evaluated_gates.append({
                "name": g_name,
                "passed": False,
                "execution_id": "unmeasured",
            })
            all_gates_passed = False
            telemetry_valid = False
            continue

        g_exec = m_gate.get("execution_id", "")
        if not g_exec:
            g_exec = "unattributed"
            if g_same_exec:
                all_gates_passed = False
                telemetry_valid = False
        elif g_same_exec and g_exec != execution_id:
            all_gates_passed = False
            telemetry_valid = False

        if "value" in m_gate and m_gate["value"] is not None:
            g_val = float(m_gate["value"])
            passed = compare_value(g_val, g_comp, g_bound)
            gate_entry: dict[str, Any] = {
                "name": g_name,
                "passed": passed
                and (not g_same_exec or g_exec == execution_id),
                "execution_id": g_exec,
                "value": g_val,
            }
        else:
            passed = False
            telemetry_valid = False
            gate_entry = {
                "name": g_name,
                "passed": False,
                "execution_id": g_exec,
            }

        if not gate_entry["passed"]:
            all_gates_passed = False

        evaluated_gates.append(gate_entry)

    # When no replicates were measured, never fabricate a metric with value 0.0,
    # replicates 0 (below schema minimum 1), or provenance "measured".
    if not resolved_replicates:
        summary = "no replicate measurements provided; metric was not evaluated"
        if integrity_reasons:
            summary = f"{summary}; {'; '.join(integrity_reasons)}"
        return {
            "status": "failed",
            "arm": arm,
            "verdict": "invalid",
            "execution_id": execution_id,
            "gates": evaluated_gates,
            "summary": summary,
        }

    rep_count = len(resolved_replicates)
    agg_value = float(statistics.median(resolved_replicates))

    # Verdict determination via framework.study.compute_arm_verdict
    b_val = 0.0
    if isinstance(baseline, Mapping):
        b_metric = baseline.get("metric")
        if isinstance(b_metric, Mapping) and "value" in b_metric:
            b_val = float(b_metric["value"])
        else:
            extracted_b = extract_llmd_metric(baseline, metric_name)
            if extracted_b is not None:
                b_val = float(extracted_b)

    verdict, delta_pct = study.compute_arm_verdict(
        metric_value=agg_value,
        baseline_value=b_val,
        direction=direction,
        noise_floor_pct=noise_floor_pct,
        gates_passed=all_gates_passed,
        replicates=rep_count,
        required_replicates=required_reps,
        telemetry_valid=telemetry_valid,
    )

    metric_dict: dict[str, Any] = {
        "name": metric_name,
        "value": agg_value,
        "unit": unit,
        "replicates": rep_count,
        "provenance": "measured",
    }
    if delta_pct is not None:
        metric_dict["delta_pct"] = delta_pct

    response: dict[str, Any] = {
        "status": "completed",
        "arm": arm,
        "verdict": verdict,
        "execution_id": execution_id,
        "metric": metric_dict,
        "gates": evaluated_gates,
    }
    if artifacts is not None:
        response["artifacts"] = list(artifacts)
    if integrity_reasons:
        response["summary"] = "integrity check failed: " + "; ".join(
            integrity_reasons)
    return response


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        "Evaluate arm execution against study protocol and baseline.")
    parser.add_argument("--arm", required=True, help="Arm identifier.")
    parser.add_argument("--execution-id",
                        required=True,
                        help="Compute execution ID.")
    parser.add_argument(
        "--replicates",
        nargs="*",
        type=float,
        default=(),
        help="Measured primary metric replicate values.",
    )
    parser.add_argument(
        "--telemetry",
        "--llmd-telemetry",
        dest="telemetry",
        nargs="*",
        default=None,
        help=
        "Optional llmd-telemetry.json file path(s) or inline JSON payloads.",
    )
    parser.add_argument(
        "--protocol",
        type=argparse.FileType("r"),
        required=True,
        help="JSON file containing study protocol.",
    )
    parser.add_argument(
        "--baseline",
        type=argparse.FileType("r"),
        default=None,
        help="Optional JSON file containing study baseline.",
    )
    parser.add_argument(
        "--gates",
        type=argparse.FileType("r"),
        default=None,
        help="Optional JSON file containing measured gates array.",
    )
    parser.add_argument(
        "--protected-paths",
        nargs="*",
        default=None,
        help=
        "Protected benchmark harness file paths (or PATH:SHA256 pairs) that must remain unmodified.",
    )
    parser.add_argument(
        "--expected-sha256",
        nargs="*",
        default=None,
        help=
        "Optional PATH=SHA256 pairs for verifying protected files outside git.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=None,
        help=
        "Artifact paths produced by the execution (validated against run directory).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Active session run_id. Required with --artifacts: every"
        " artifact must sit under this run.",
    )

    args = parser.parse_args()

    protocol = json.load(args.protocol)
    baseline = json.load(args.baseline) if args.baseline else None
    gates = json.load(args.gates) if args.gates else None

    sha_map: dict[str, str] = {}
    for entry in args.expected_sha256 or []:
        if "=" in entry:
            k, v = entry.split("=", 1)
            sha_map[k.strip()] = v.strip()

    result = evaluate_arm(
        arm=args.arm,
        execution_id=args.execution_id,
        replicates=args.replicates,
        protocol=protocol,
        baseline=baseline,
        gate_measurements=gates,
        telemetry=args.telemetry,
        protected_paths=args.protected_paths,
        expected_sha256=sha_map or None,
        artifacts=args.artifacts,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
