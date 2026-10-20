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
"""Declarative EPP and Homogeneous Distributed Serving Mutator + Checkpoint/Rollback CLI.

Inspects, checkpoints, rolls back, and declaratively mutates homogeneous `llm-d`
Kustomize bundles (`01-endpointpickerconfig.yaml`, `02-inferencepool.yaml`,
`03-serving-deployment.yaml`, `base-modelserver.yaml`,
`patch-vllm-tpu-workers.yaml`, `patch-decode.yaml`, and `kustomization.yaml`)
produced by `dynamo_translator.py`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCORER_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("prefix-cache-scorer", "precise-prefix-cache-scorer"),
    ("queue-depth-scorer", "queue-scorer", "active-request-scorer"),
    ("kv-utilization-scorer", "kv-cache-utilization-scorer"),
    ("token-load-scorer", ),
    ("no-hit-lru-scorer", ),
    ("session-affinity-scorer", ),
)

CANONICAL_SCORER_FOR_ALIAS: dict[str, str] = {}
ALIASES_FOR_SCORER: dict[str, tuple[str, ...]] = {}
for _group in SCORER_ALIAS_GROUPS:
    _canonical = _group[0]
    for _name in _group:
        CANONICAL_SCORER_FOR_ALIAS[_name] = _canonical
        ALIASES_FOR_SCORER[_name] = _group

EPP_CONFIGMAP_KEYS: tuple[str, ...] = (
    "endpoint-picker-config.yaml",
    "default-plugins.yaml",
    "precise-prefix-cache-routing-plugins.yaml",
)

LLM_D_V1ALPHA1_SCORER_TYPE_MAP: dict[str, str] = {
    "prefix-cache-scorer": "prefix-cache-scorer",
    "precise-prefix-cache-scorer": "prefix-cache-scorer",
    "queue-depth-scorer": "queue-scorer",
    "queue-scorer": "queue-scorer",
    "active-request-scorer": "active-request-scorer",
    "kv-utilization-scorer": "kv-cache-utilization-scorer",
    "kv-cache-utilization-scorer": "kv-cache-utilization-scorer",
    "token-load-scorer": "token-load-scorer",
    "no-hit-lru-scorer": "no-hit-lru-scorer",
    "session-affinity-scorer": "session-affinity-scorer",
}

BOOLEAN_VLLM_FLAGS: frozenset[str] = frozenset({
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
    "--async-scheduling",
    "--disable-log-requests",
    "--enable-chunked-prefill",
    "--no-enable-chunked-prefill",
    "--enforce-eager",
    "--trust-remote-code",
})


def _coerce_scalar(val: Any) -> Any:
    """Coerce string scalar representations to int, float, bool, JSON, or str."""
    if isinstance(val, (bool, int, float, dict, list)) or val is None:
        return val
    text = str(val).strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if (text.startswith("{")
            and text.endswith("}")) or (text.startswith("[")
                                        and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        if "." not in text and "e" not in text.lower():
            return int(text)
        num = float(text)
        if num.is_integer() and "." not in text:
            return int(num)
        return num
    except ValueError:
        return text


def _canonicalize_for_hash(value: Any) -> Any:
    """Recursively normalize numbers, dicts, and lists for deterministic hashing."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        fval = float(value)
        if fval.is_integer():
            return int(fval)
        return round(fval, 8)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                fval = float(stripped)
                if fval.is_integer():
                    return int(fval)
                return round(fval, 8)
            except ValueError:
                pass
        return value
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(v) for v in value]
    return str(value)


def compute_recipe_fingerprint(epp_config: Mapping[str, Any]) -> str:
    """Compute a deterministic 64-character SHA-256 hex digest of `epp_config`."""
    canonical = _canonicalize_for_hash(epp_config)
    encoded = json.dumps(canonical, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_flag_name(flag: str) -> str:
    """Ensure a CLI flag starts with `--` and uses hyphens."""
    cleaned = flag.strip()
    if not cleaned.startswith("--"):
        cleaned = "--" + cleaned.lstrip("-")
    return cleaned


def _extract_vllm_tokens_from_shell_cmd(cmd_str: str) -> tuple[str, list[str]]:
    """Split a `bash -lc '... exec vllm serve ...'` string into prefix and vllm args."""
    marker = "vllm serve"
    idx = cmd_str.find(marker)
    if idx == -1:
        return cmd_str, []
    prefix = cmd_str[:idx + len(marker)]
    tail = cmd_str[idx + len(marker):].strip()
    try:
        tokens = shlex.split(tail)
    except ValueError:
        tokens = tail.split()
    return prefix, tokens


def _parse_vllm_args_list(tokens: Sequence[str]) -> dict[str, Any]:
    """Parse a list of `vllm serve` CLI argument tokens into a flag map."""
    flags: dict[str, Any] = {}
    i = 0
    n = len(tokens)
    while i < n:
        tok = str(tokens[i])
        if not tok.startswith("-"):
            if "--model" not in flags and i == 0:
                flags["--model"] = tok
            i += 1
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            norm_k = _normalize_flag_name(k)
            flags[norm_k] = _coerce_scalar(v)
            i += 1
            continue
        norm_k = _normalize_flag_name(tok)
        if norm_k in BOOLEAN_VLLM_FLAGS or (i + 1 >= n or str(
                tokens[i + 1]).startswith("--")):
            if norm_k == "--no-enable-prefix-caching":
                flags["--enable-prefix-caching"] = False
                flags["--no-enable-prefix-caching"] = True
            elif norm_k == "--enable-prefix-caching":
                flags["--enable-prefix-caching"] = True
            else:
                flags[norm_k] = True
            i += 1
        else:
            flags[norm_k] = _coerce_scalar(tokens[i + 1])
            i += 2
    return flags


def _extract_container_vllm_flags(
        container: Mapping[str, Any]) -> dict[str, Any]:
    """Extract `vllm serve` CLI flags from a container spec."""
    cmd = container.get("command") or []
    args = container.get("args") or []
    if (len(cmd) >= 2 and "bash" in str(cmd[0]) and len(args) == 1
            and "vllm serve" in str(args[0])):
        _, tokens = _extract_vllm_tokens_from_shell_cmd(str(args[0]))
        return _parse_vllm_args_list(tokens)
    all_tokens = [str(x) for x in args]
    return _parse_vllm_args_list(all_tokens)


def _mutate_vllm_tokens(tokens: list[str], flag_name: str,
                        raw_val: Any) -> list[str]:
    """Mutate or append `flag_name` in a list of `vllm serve` tokens."""
    norm_flag = _normalize_flag_name(flag_name)
    coerced = _coerce_scalar(raw_val)
    updated: list[str] = []
    found = False
    i = 0
    n = len(tokens)

    if norm_flag in ("--enable-prefix-caching", "--no-enable-prefix-caching"):
        enable = (bool(coerced) if norm_flag == "--enable-prefix-caching" else
                  not bool(coerced))
        for tok in tokens:
            if tok in ("--enable-prefix-caching",
                       "--no-enable-prefix-caching"):
                continue
            updated.append(tok)
        updated.append("--enable-prefix-caching"
                       if enable else "--no-enable-prefix-caching")
        return updated

    if norm_flag in BOOLEAN_VLLM_FLAGS and isinstance(coerced, bool):
        for tok in tokens:
            if tok == norm_flag or tok.startswith(norm_flag + "="):
                continue
            updated.append(tok)
        if coerced:
            updated.append(norm_flag)
        return updated

    val_str = (json.dumps(coerced, separators=(",", ":")) if isinstance(
        coerced, (dict, list)) else str(coerced))

    while i < n:
        tok = str(tokens[i])
        if tok == norm_flag:
            found = True
            updated.append(norm_flag)
            if i + 1 < n and not str(tokens[i + 1]).startswith("--"):
                updated.append(val_str)
                i += 2
            else:
                updated.append(val_str)
                i += 1
            continue
        if tok.startswith(norm_flag + "="):
            found = True
            updated.append(f"{norm_flag}={val_str}")
            i += 1
            continue
        updated.append(tok)
        i += 1

    if not found:
        updated.extend([norm_flag, val_str])
    return updated


def _apply_vllm_flag_to_container(container: dict[str, Any], flag_name: str,
                                  raw_val: Any) -> None:
    """Apply a `vllm serve` flag mutation to a container spec."""
    cmd = container.get("command") or []
    args = list(container.get("args") or [])
    if (len(cmd) >= 2 and "bash" in str(cmd[0]) and len(args) == 1
            and "vllm serve" in str(args[0])):
        prefix, tokens = _extract_vllm_tokens_from_shell_cmd(str(args[0]))
        mutated_tokens = _mutate_vllm_tokens(tokens, flag_name, raw_val)
        rebuilt_tail = " ".join(shlex.quote(t) for t in mutated_tokens)
        container["args"] = [f"{prefix} {rebuilt_tail}".strip()]
    else:
        container["args"] = _mutate_vllm_tokens([str(x) for x in args],
                                                flag_name, raw_val)


def _extract_container_env(container: Mapping[str, Any]) -> dict[str, str]:
    """Extract container `env` list into an ordered dict of name -> value."""
    env_map: dict[str, str] = {}
    for entry in container.get("env") or []:
        if isinstance(entry, Mapping) and "name" in entry:
            if "value" in entry and entry["value"] is not None:
                env_map[str(entry["name"])] = str(entry["value"])
    return env_map


def _apply_env_to_container(container: dict[str, Any], key: str,
                            value: Any) -> None:
    """Update or append an environment variable in `container['env']`."""
    env_list = list(container.get("env") or [])
    val_str = str(value)
    for entry in env_list:
        if isinstance(entry, dict) and entry.get("name") == key:
            entry["value"] = val_str
            container["env"] = env_list
            return
    env_list.append({"name": key, "value": val_str})
    container["env"] = env_list


def _load_yaml_docs(path: Path) -> list[dict[str, Any]]:
    """Load all non-empty YAML documents from `path`."""
    text = path.read_text(encoding="utf-8")
    return [
        doc for doc in yaml.safe_load_all(text)
        if isinstance(doc, dict) and doc
    ]


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


def _dump_yaml_docs(path: Path, docs: Sequence[Mapping[str, Any]]) -> None:
    """Write one or more YAML documents back to `path` and validate round-trip."""
    rendered = _dump_yaml_all(docs)
    list(yaml.safe_load_all(rendered))
    path.write_text(rendered, encoding="utf-8")


def _iter_configmap_epp_keys(data: Mapping[str, Any]) -> list[str]:
    """Return ordered ConfigMap keys that contain EndpointPickerConfig YAML."""
    keys: list[str] = []
    for k in EPP_CONFIGMAP_KEYS:
        if k in data and isinstance(data[k], str):
            keys.append(k)
    for k, v in data.items():
        if k in keys or not isinstance(v, str):
            continue
        if str(k).endswith((".yaml", ".yml")):
            try:
                parsed = yaml.safe_load(v)
            except yaml.YAMLError:
                continue
            if isinstance(
                    parsed,
                    dict) and parsed.get("kind") == "EndpointPickerConfig":
                keys.append(str(k))
    return keys


def _extract_epp_subdocs(
    epp_yaml_path: Path,
) -> list[tuple[dict[str, Any], str | None, dict[str, Any]]]:
    """Return `(outer_doc, configmap_key_or_none, parsed_epp_config)` tuples."""
    docs = _load_yaml_docs(epp_yaml_path)
    results: list[tuple[dict[str, Any], str | None, dict[str, Any]]] = []
    for doc in docs:
        kind = doc.get("kind")
        if kind == "EndpointPickerConfig":
            results.append((doc, None, doc))
        elif kind == "ConfigMap":
            data = doc.get("data") or {}
            for key in _iter_configmap_epp_keys(data):
                inner = yaml.safe_load(data[key])
                if isinstance(inner, dict):
                    results.append((doc, key, inner))
    return results


def _parse_epp_document(epp_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a homogeneous `EndpointPickerConfig` document into normalized sections."""
    scorers: dict[str, dict[str, Any]] = {}
    filters: dict[str, dict[str, Any]] = {}
    flow_control: dict[str, Any] = {}

    scheduling = epp_doc.get("scheduling")
    if isinstance(scheduling, Mapping):
        for f_entry in scheduling.get("filters") or []:
            if isinstance(f_entry, Mapping) and f_entry.get("name"):
                fname = str(f_entry["name"])
                filters[fname] = {
                    "parameters":
                    copy.deepcopy(dict(f_entry.get("parameters") or {})),
                }
        for s_entry in scheduling.get("scorers") or []:
            if isinstance(s_entry, Mapping) and s_entry.get("name"):
                sname = str(s_entry["name"])
                scorers[sname] = {
                    "weight":
                    _coerce_scalar(s_entry.get("weight", 1)),
                    "parameters":
                    copy.deepcopy(dict(s_entry.get("parameters") or {})),
                }

    plugins_list = epp_doc.get("plugins")
    sched_profiles = epp_doc.get("schedulingProfiles")
    if isinstance(plugins_list, list) or isinstance(sched_profiles, list):
        plugin_meta: dict[str, dict[str, Any]] = {}
        for p_entry in plugins_list or []:
            if isinstance(p_entry, Mapping):
                ptype = str(p_entry.get("type") or p_entry.get("name") or "")
                pname = str(p_entry.get("name") or ptype)
                pparams = copy.deepcopy(dict(p_entry.get("parameters") or {}))
                plugin_meta[pname] = {"type": ptype, "parameters": pparams}
                if "filter" in ptype or "filter" in pname:
                    filters.setdefault(pname, {"parameters": pparams})

        for sp_entry in sched_profiles or []:
            if not isinstance(sp_entry, Mapping):
                continue
            for ref_entry in sp_entry.get("plugins") or []:
                if not isinstance(ref_entry, Mapping):
                    continue
                ref_name = str(
                    ref_entry.get("pluginRef") or ref_entry.get("name") or "")
                if not ref_name:
                    continue
                meta = plugin_meta.get(ref_name, {})
                if ("weight" in ref_entry or "scorer" in ref_name
                        or "scorer" in meta.get("type", "")):
                    weight = _coerce_scalar(ref_entry.get("weight", 1))
                    params = copy.deepcopy(meta.get("parameters") or {})
                    scorers.setdefault(
                        ref_name,
                        {
                            "weight": weight,
                            "parameters": copy.deepcopy(params)
                        },
                    )

        for pname, meta in plugin_meta.items():
            ptype = meta.get("type", "")
            if "scorer" in ptype or "scorer" in pname:
                scorers.setdefault(
                    pname,
                    {
                        "weight": 1,
                        "parameters": copy.deepcopy(
                            meta.get("parameters") or {}),
                    },
                )

    raw_fc = epp_doc.get("flowControl")
    if isinstance(raw_fc, Mapping):
        flow_control = copy.deepcopy(dict(raw_fc))

    for group in SCORER_ALIAS_GROUPS:
        present_name = next((n for n in group if n in scorers), None)
        if present_name is not None:
            base_entry = scorers[present_name]
            for alias in group:
                if alias not in scorers:
                    scorers[alias] = {
                        "weight":
                        base_entry["weight"],
                        "parameters":
                        copy.deepcopy(base_entry.get("parameters") or {}),
                        "alias_of":
                        present_name,
                    }

    return {
        "scorers": scorers,
        "filters": filters,
        "flow_control": flow_control,
    }


def _find_lws_container(lws_doc: Mapping[str, Any]) -> dict[str, Any] | None:
    """Locate the primary model-serving container inside a LeaderWorkerSet doc."""
    spec = lws_doc.get("spec")
    if not isinstance(spec, Mapping):
        return None
    lwt = spec.get("leaderWorkerTemplate")
    if not isinstance(lwt, Mapping):
        return None
    wt = lwt.get("workerTemplate")
    if not isinstance(wt, Mapping):
        return None
    pod_spec = wt.get("spec")
    if not isinstance(pod_spec, Mapping):
        return None
    containers = pod_spec.get("containers")
    if isinstance(containers, list) and containers:
        for c in containers:
            if isinstance(c, dict) and c.get("name") in (
                    "vllm-tpu",
                    "modelserver",
                    "vllm",
            ):
                return c
        if isinstance(containers[0], dict):
            return containers[0]
    return None


def _parse_pool_from_lws_docs(
        role: str, docs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge one or more LeaderWorkerSet docs (e.g. base + patch) into a pool spec."""
    name = "vllm-tpu-workers"
    replicas = 1
    size = 1
    vllm_flags: dict[str, Any] = {}
    env: dict[str, str] = {}

    for doc in docs:
        if doc.get("kind") != "LeaderWorkerSet":
            continue
        meta = doc.get("metadata") or {}
        if meta.get("name"):
            name = str(meta["name"])
        spec = doc.get("spec") or {}
        if "replicas" in spec and spec["replicas"] is not None:
            replicas = int(spec["replicas"])
        lwt = doc.get("spec", {}).get("leaderWorkerTemplate") or {}
        if "size" in lwt and lwt["size"] is not None:
            size = int(lwt["size"])
        container = _find_lws_container(doc)
        if container is not None:
            c_flags = _extract_container_vllm_flags(container)
            if c_flags:
                vllm_flags.update(c_flags)
            c_env = _extract_container_env(container)
            if c_env:
                env.update(c_env)

    return {
        "role": role,
        "name": name,
        "replicas": replicas,
        "size": size,
        "leaderWorkerTemplate": {
            "size": size
        },
        "vllm_flags": vllm_flags,
        "env": env,
    }


def _discover_pool_files(bundle_dir: Path) -> dict[str, list[Path]]:
    """Return mapping of homogeneous `serving` role -> list of LWS YAML files."""
    serving_file = bundle_dir / "03-serving-deployment.yaml"
    base_ms = bundle_dir / "base-modelserver.yaml"
    patch_workers = bundle_dir / "patch-vllm-tpu-workers.yaml"
    patch_decode = bundle_dir / "patch-decode.yaml"

    if serving_file.is_file():
        return {"serving": [serving_file]}

    if base_ms.is_file() or patch_workers.is_file() or patch_decode.is_file():
        files = [
            p for p in (base_ms, patch_workers, patch_decode) if p.is_file()
        ]
        return {"serving": files}

    nested_kustomize = bundle_dir / "kustomize"
    if nested_kustomize.is_dir():
        return _discover_pool_files(nested_kustomize)

    return {}


def _resolve_bundle_root(bundle_dir: str | Path) -> Path:
    """Resolve `bundle_dir` to the directory containing `01-endpointpickerconfig.yaml`."""
    bdir = Path(bundle_dir).resolve()
    if not bdir.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {bdir}")
    if (bdir / "01-endpointpickerconfig.yaml").is_file() or (
            bdir / "kustomize" / "01-endpointpickerconfig.yaml").is_file():
        return bdir
    if (bdir / "llmd_bundle" / "01-endpointpickerconfig.yaml").is_file():
        return bdir / "llmd_bundle"
    return bdir


def inspect_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Inspect a homogeneous `llm-d` bundle directory and return a canonical JSON snapshot."""
    bdir = _resolve_bundle_root(bundle_dir)

    epp_path = bdir / "01-endpointpickerconfig.yaml"
    if not epp_path.is_file() and (bdir / "kustomize" /
                                   "01-endpointpickerconfig.yaml").is_file():
        epp_path = bdir / "kustomize" / "01-endpointpickerconfig.yaml"
    if not epp_path.is_file():
        raise FileNotFoundError(
            f"Missing 01-endpointpickerconfig.yaml in {bdir}")

    subdocs = _extract_epp_subdocs(epp_path)
    if not subdocs:
        raise ValueError(f"No EndpointPickerConfig found inside {epp_path}")

    parsed_epp = _parse_epp_document(subdocs[0][2])
    for _, _, inner_doc in subdocs[1:]:
        extra = _parse_epp_document(inner_doc)
        for sname, sinfo in extra["scorers"].items():
            parsed_epp["scorers"].setdefault(sname, sinfo)
        for fname, finfo in extra["filters"].items():
            parsed_epp["filters"].setdefault(fname, finfo)
        if not parsed_epp["flow_control"] and extra["flow_control"]:
            parsed_epp["flow_control"] = extra["flow_control"]

    pool_files = _discover_pool_files(bdir)
    pools: dict[str, dict[str, Any]] = {}
    for role, paths in pool_files.items():
        all_docs: list[dict[str, Any]] = []
        for p in paths:
            all_docs.extend(_load_yaml_docs(p))
        pools[role] = _parse_pool_from_lws_docs(role, all_docs)

    serving_mode = "homogeneous"

    canonical_scorers: dict[str, Any] = {}
    for sname, sinfo in sorted(parsed_epp["scorers"].items()):
        if "alias_of" in sinfo:
            continue
        canonical_key = CANONICAL_SCORER_FOR_ALIAS.get(sname, sname)
        canonical_scorers[canonical_key] = {
            "weight": sinfo["weight"],
            "parameters": sinfo.get("parameters") or {},
        }

    canonical_filters: dict[str, Any] = {
        fname: finfo.get("parameters") or {}
        for fname, finfo in sorted(parsed_epp["filters"].items())
    }

    canonical_pools: dict[str, Any] = {}
    for role, pinfo in sorted(pools.items()):
        canonical_pools[role] = {
            "replicas": pinfo["replicas"],
            "leaderWorkerTemplate.size": pinfo["size"],
            "vllm_flags": dict(sorted(pinfo["vllm_flags"].items())),
            "env": dict(sorted(pinfo["env"].items())),
        }

    primary_pool = pools.get(
        "serving",
        {
            "replicas": 1,
            "size": 1,
            "vllm_flags": {},
            "env": {}
        },
    )

    saturation_detector = (
        (parsed_epp["flow_control"].get("saturationDetector") or {})
        if isinstance(parsed_epp["flow_control"], Mapping) else {})
    admitter = ((parsed_epp["flow_control"].get("admitter") or {})
                if isinstance(parsed_epp["flow_control"], Mapping) else {})

    canonical_epp_config: dict[str, Any] = {
        "serving_mode": serving_mode,
        "scorers": canonical_scorers,
        "filters": canonical_filters,
        "saturation_thresholds": {
            "saturationDetector": saturation_detector,
            "admitter": admitter,
        },
        "pools": canonical_pools,
    }

    fingerprint = compute_recipe_fingerprint(canonical_epp_config)

    replicas_summary = {role: p["replicas"] for role, p in pools.items()}
    replicas_summary["total"] = sum(p["replicas"] for p in pools.values())

    return {
        "bundle_dir": str(bdir),
        "serving_mode": serving_mode,
        "scorers": parsed_epp["scorers"],
        "scorer_weights": {
            k: v["weight"]
            for k, v in parsed_epp["scorers"].items()
        },
        "filters": parsed_epp["filters"],
        "flow_control": parsed_epp["flow_control"],
        "saturation_thresholds": {
            "saturationDetector": saturation_detector,
            "admitter": admitter,
        },
        "pools": pools,
        "replicas": replicas_summary,
        "leader_worker_template_size": primary_pool["size"],
        "vllm_flags": primary_pool["vllm_flags"],
        "env": primary_pool["env"],
        "epp_config": canonical_epp_config,
        "recipe_fingerprint": fingerprint,
    }


def _collect_bundle_yaml_files(bundle_dir: Path) -> list[Path]:
    """Collect all YAML files in `bundle_dir` and optional `bundle_dir/kustomize`."""
    files: list[Path] = []
    for p in sorted(bundle_dir.iterdir()):
        if p.is_file() and p.suffix in (".yaml", ".yml"):
            files.append(p)
    nested = bundle_dir / "kustomize"
    if nested.is_dir():
        for p in sorted(nested.iterdir()):
            if p.is_file() and p.suffix in (".yaml", ".yml"):
                files.append(p)
    return files


def checkpoint_bundle(bundle_dir: str | Path,
                      checkpoint_dir: str | Path) -> dict[str, Any]:
    """Save a bitwise-exact checkpoint of all YAML manifests in `bundle_dir`."""
    bdir = _resolve_bundle_root(bundle_dir)
    cdir = Path(checkpoint_dir).resolve()

    snapshot = inspect_bundle(bdir)
    cdir.mkdir(parents=True, exist_ok=True)
    files_manifest: dict[str, str] = {}

    for src in _collect_bundle_yaml_files(bdir):
        rel = src.relative_to(bdir).as_posix()
        dst = cdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw_bytes = src.read_bytes()
        dst.write_bytes(raw_bytes)
        files_manifest[rel] = hashlib.sha256(raw_bytes).hexdigest()

    manifest = {
        "bundle_dir": str(bdir),
        "checkpoint_dir": str(cdir),
        "recipe_fingerprint": snapshot["recipe_fingerprint"],
        "serving_mode": snapshot["serving_mode"],
        "files": files_manifest,
    }
    manifest_path = cdir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) +
                             "\n",
                             encoding="utf-8")
    return manifest


def _discover_epp_deployments(bundle_dir: Path) -> list[str]:
    """Discover EPP Deployment resource names in `bundle_dir`."""
    bdir = _resolve_bundle_root(bundle_dir)
    deployments: list[str] = []
    for yaml_path in _collect_bundle_yaml_files(bdir):
        try:
            docs = _load_yaml_docs(yaml_path)
        except Exception:
            continue
        for doc in docs:
            if doc.get("kind") != "Deployment":
                continue
            meta = doc.get("metadata") or {}
            name = str(meta.get("name") or "").strip()
            if not name:
                continue
            if (name in ("llm-d-epp", "vllm-epp-epp") or name.endswith("-epp")
                    or yaml_path.name
                    in ("00-gateway.yaml", "01-endpointpickerconfig.yaml")):
                if name not in deployments:
                    deployments.append(name)
    return deployments


def _build_kubectl_base_args(
    *,
    kubeconfig: str | Path | None = None,
    context: str | None = None,
    namespace: str | None = None,
) -> list[str]:
    """Build base `kubectl` command prefix with optional kubeconfig, context, and namespace."""
    args = ["kubectl"]
    if kubeconfig:
        args.extend(["--kubeconfig", str(kubeconfig)])
    if context:
        args.extend(["--context", str(context)])
    if namespace:
        args.extend(["-n", str(namespace)])
    return args


def apply_bundle(
    bundle_dir: str | Path,
    *,
    namespace: str | None = None,
    context: str | None = None,
    kubeconfig: str | Path | None = None,
    rollout_timeout: str = "180s",
    execute: bool = True,
) -> dict[str, Any]:
    """Apply Kustomize/YAML bundle to cluster and restart any standalone EPP Deployment."""
    bdir = _resolve_bundle_root(bundle_dir)
    base_args = _build_kubectl_base_args(
        kubeconfig=kubeconfig,
        context=context,
        namespace=namespace,
    )
    if (bdir / "kustomization.yaml").is_file():
        apply_args = [*base_args, "apply", "-k", str(bdir)]
    elif (bdir / "kustomize" / "kustomization.yaml").is_file():
        apply_args = [*base_args, "apply", "-k", str(bdir / "kustomize")]
    else:
        apply_args = [*base_args, "apply", "-f", str(bdir)]

    epp_deployments = _discover_epp_deployments(bdir)
    rollout_cmds: list[list[str]] = []
    for dep_name in epp_deployments:
        rollout_cmds.append(
            [*base_args, "rollout", "restart", f"deployment/{dep_name}"])
        rollout_cmds.append([
            *base_args,
            "rollout",
            "status",
            f"deployment/{dep_name}",
            f"--timeout={rollout_timeout}",
        ])

    apply_cmd_str = " ".join(apply_args)
    rollout_cmd_strs = [" ".join(cmd) for cmd in rollout_cmds]
    outputs: list[str] = []
    applied = False

    if execute:
        for cmd in [apply_args, *rollout_cmds]:
            cmd_str = " ".join(cmd)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Failed to run '{cmd_str}' (exit {proc.returncode}): {proc.stderr.strip()}"
                )
            if proc.stdout and proc.stdout.strip():
                outputs.append(proc.stdout.strip())
        applied = True

    return {
        "kubectl_apply_cmd": apply_cmd_str,
        "kubectl_rollout_cmds": rollout_cmd_strs,
        "restarted_deployments": epp_deployments,
        "applied_to_cluster": applied,
        "kubectl_output": "\n".join(outputs),
    }


def rollback_bundle(
    bundle_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    apply: bool = False,
    namespace: str | None = None,
    context: str | None = None,
    kubeconfig: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically restore `bundle_dir` from `checkpoint_dir` and verify fingerprint."""
    bdir = _resolve_bundle_root(bundle_dir)
    cdir = Path(checkpoint_dir).resolve()
    manifest_path = cdir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files: dict[str, str] = manifest.get("files") or {}
    expected_fp: str = str(manifest.get("recipe_fingerprint", ""))

    staged_bytes: dict[Path, bytes] = {}
    for rel, expected_sha in expected_files.items():
        src = cdir / rel
        if not src.is_file():
            raise FileNotFoundError(f"Corrupt checkpoint: missing file {src}")
        raw = src.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Corrupt checkpoint file {rel}: expected sha256 {expected_sha}, got {actual_sha}"
            )
        staged_bytes[bdir / rel] = raw

    for existing in _collect_bundle_yaml_files(bdir):
        rel = existing.relative_to(bdir).as_posix()
        if rel not in expected_files:
            existing.unlink()

    for dst, raw in staged_bytes.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw)

    restored_snapshot = inspect_bundle(bdir)
    restored_fp = restored_snapshot["recipe_fingerprint"]
    if expected_fp and restored_fp != expected_fp:
        raise ValueError(
            f"Rollback fingerprint mismatch: expected {expected_fp}, got {restored_fp}"
        )

    apply_info = apply_bundle(
        bdir,
        namespace=namespace,
        context=context,
        kubeconfig=kubeconfig,
        execute=bool(apply),
    )

    return {
        "status": "rolled_back",
        "bundle_dir": str(bdir),
        "checkpoint_dir": str(cdir),
        "restored_files": sorted(expected_files.keys()),
        "recipe_fingerprint": restored_fp,
        "fingerprint_verified": True,
        "fingerprint_matches_checkpoint": True,
        **apply_info,
    }


def _coerce_v1alpha1_weight(weight: float | int) -> int | float:
    """Coerce whole-number scorer weights to int for strict Go YAML decoding."""
    if isinstance(weight, bool):
        return int(weight)
    if isinstance(weight, int):
        return weight
    if isinstance(weight, float) and weight.is_integer():
        return int(weight)
    return weight


def _mutate_epp_inner_doc(
    epp_doc: dict[str, Any],
    *,
    scorer_updates: Sequence[tuple[str, float | int]],
    filter_updates: Sequence[tuple[str, str, Any]],
    saturation_updates: Mapping[str, Any],
) -> None:
    """Apply scorer, filter, and saturation mutations to an inner homogeneous EPP dict."""
    scheduling = epp_doc.get("scheduling")
    plugins_list = epp_doc.get("plugins")
    sched_profiles = epp_doc.get("schedulingProfiles")
    api_version = str(epp_doc.get("apiVersion") or "")
    supports_flow_control = ("flowControl" in epp_doc
                             or isinstance(scheduling, dict)
                             or api_version.startswith("config.llm-d.io/"))

    requested_scorer_names = {s for s, _ in scorer_updates}
    has_explicit_queue_and_active = (
        "active-request-scorer" in requested_scorer_names and
        bool(requested_scorer_names & {"queue-depth-scorer", "queue-scorer"}))

    for sname, weight in scorer_updates:
        if has_explicit_queue_and_active and sname == "active-request-scorer":
            aliases: tuple[str, ...] = ("active-request-scorer", )
        elif has_explicit_queue_and_active and sname in (
                "queue-depth-scorer",
                "queue-scorer",
        ):
            aliases = ("queue-depth-scorer", "queue-scorer")
        else:
            aliases = ALIASES_FOR_SCORER.get(sname, (sname, ))

        coerced_weight = _coerce_v1alpha1_weight(weight)

        if isinstance(scheduling, dict):
            if "scorers" in scheduling and isinstance(scheduling["scorers"],
                                                      list):
                matched = False
                for entry in scheduling["scorers"]:
                    if isinstance(entry,
                                  dict) and entry.get("name") in aliases:
                        entry["weight"] = coerced_weight
                        matched = True
                if not matched:
                    scheduling["scorers"].append({
                        "name": sname,
                        "weight": coerced_weight
                    })

        if isinstance(sched_profiles, list):
            v1_type = LLM_D_V1ALPHA1_SCORER_TYPE_MAP.get(sname, sname)
            v1_aliases = tuple(dict.fromkeys((*aliases, v1_type)))
            matched_sp = False
            for sp_entry in sched_profiles:
                if not isinstance(sp_entry, dict):
                    continue
                sp_plugins = sp_entry.setdefault("plugins", [])
                for ref_entry in sp_plugins:
                    if not isinstance(ref_entry, dict):
                        continue
                    ref_name = ref_entry.get("pluginRef") or ref_entry.get(
                        "name")
                    if ref_name in v1_aliases:
                        ref_entry["weight"] = coerced_weight
                        matched_sp = True
            if not matched_sp and sched_profiles:
                target_sp = sched_profiles[0]
                if isinstance(target_sp, dict):
                    target_sp.setdefault("plugins", []).append({
                        "pluginRef":
                        v1_type,
                        "weight":
                        coerced_weight
                    })
            if isinstance(plugins_list, list) and not any(
                    isinstance(p, dict) and
                (p.get("type") in v1_aliases or p.get("name") in v1_aliases)
                    for p in plugins_list):
                plugins_list.append({"type": v1_type})

    if isinstance(plugins_list, list) and api_version.startswith("llm-d.ai/"):
        for p_entry in plugins_list:
            if isinstance(p_entry, dict) and "weight" in p_entry:
                p_entry.pop("weight", None)

    for fname, param_key, param_val in filter_updates:
        if fname in ("saturationDetector", "flowControl.saturationDetector"):
            if supports_flow_control:
                fc = epp_doc.setdefault("flowControl", {})
                sd = fc.setdefault("saturationDetector", {})
                sd[param_key] = param_val
            continue
        if fname in ("admitter", "flowControl.admitter"):
            if supports_flow_control:
                fc = epp_doc.setdefault("flowControl", {})
                adm = fc.setdefault("admitter", {})
                adm[param_key] = param_val
            continue

        if isinstance(scheduling, dict):
            if "filters" in scheduling and isinstance(scheduling["filters"],
                                                      list):
                matched_f = False
                for f_entry in scheduling["filters"]:
                    if isinstance(f_entry,
                                  dict) and f_entry.get("name") == fname:
                        f_entry.setdefault("parameters",
                                           {})[param_key] = param_val
                        matched_f = True
                if not matched_f:
                    scheduling["filters"].append({
                        "name": fname,
                        "parameters": {
                            param_key: param_val
                        }
                    })

        if isinstance(plugins_list, list):
            matched_pl = False
            for p_entry in plugins_list:
                if isinstance(p_entry,
                              dict) and (p_entry.get("name") == fname
                                         or p_entry.get("type") == fname):
                    p_entry.setdefault("parameters", {})[param_key] = param_val
                    matched_pl = True
            if not matched_pl:
                plugins_list.append({
                    "type": fname,
                    "parameters": {
                        param_key: param_val
                    }
                })
            if isinstance(sched_profiles, list) and sched_profiles:
                first_sp = sched_profiles[0]
                if isinstance(first_sp, dict):
                    sp_plugins = first_sp.setdefault("plugins", [])
                    if not any(
                            isinstance(r, dict) and
                        (r.get("pluginRef") == fname or r.get("name") == fname)
                            for r in sp_plugins):
                        sp_plugins.append({"pluginRef": fname})

    if saturation_updates and supports_flow_control:
        fc = epp_doc.setdefault("flowControl", {})
        if "saturationDetector" in saturation_updates and isinstance(
                saturation_updates["saturationDetector"], Mapping):
            fc.setdefault("saturationDetector",
                          {}).update(saturation_updates["saturationDetector"])
        if "admitter" in saturation_updates and isinstance(
                saturation_updates["admitter"], Mapping):
            fc.setdefault("admitter",
                          {}).update(saturation_updates["admitter"])
        for k, v in saturation_updates.items():
            if k not in ("saturationDetector", "admitter"):
                fc.setdefault("saturationDetector", {})[k] = v


def validate_snapshot_invariants(snapshot: Mapping[str, Any]) -> list[str]:
    """Validate structural invariants on a post-mutation bundle snapshot."""
    errors: list[str] = []

    scorers = snapshot.get("scorers") or {}
    real_scorers = {
        k: v
        for k, v in scorers.items()
        if isinstance(v, Mapping) and "alias_of" not in v
    }
    if not real_scorers:
        errors.append("EndpointPickerConfig must declare at least one scorer.")
    else:
        total_weight = 0.0
        for sname, sinfo in real_scorers.items():
            w = sinfo.get("weight")
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                errors.append(
                    f"Scorer '{sname}' weight must be numeric, got {w!r}.")
            elif w < 0:
                errors.append(
                    f"Scorer '{sname}' weight must be non-negative, got {w}.")
            else:
                total_weight += float(w)
        if total_weight <= 0:
            errors.append(
                "At least one scorer in EndpointPickerConfig must have a positive weight (total active scorer weight > 0)."
            )

    sat = (snapshot.get("saturation_thresholds")
           or {}).get("saturationDetector") or {}
    if "maxKVCacheUtilization" in sat:
        kv_util = sat["maxKVCacheUtilization"]
        if (not isinstance(kv_util, (int, float)) or isinstance(kv_util, bool)
                or not (0.0 < float(kv_util) <= 1.0)):
            errors.append(
                f"saturationDetector.maxKVCacheUtilization must be in (0.0, 1.0], got {kv_util!r}."
            )
    if "maxQueueDepth" in sat:
        q_depth = sat["maxQueueDepth"]
        if (not isinstance(q_depth, (int, float)) or isinstance(q_depth, bool)
                or int(q_depth) <= 0):
            errors.append(
                f"saturationDetector.maxQueueDepth must be > 0, got {q_depth!r}."
            )

    for fname, finfo in (snapshot.get("filters") or {}).items():
        params = (finfo.get("parameters")
                  if isinstance(finfo, Mapping) else {}) or {}
        if "maxKVCacheUtilization" in params:
            val = params["maxKVCacheUtilization"]
            if (not isinstance(val, (int, float)) or isinstance(val, bool)
                    or not (0.0 < float(val) <= 1.0)):
                errors.append(
                    f"Filter '{fname}' maxKVCacheUtilization must be in (0.0, 1.0], got {val!r}."
                )

    pools = snapshot.get("pools") or {}
    for role, pinfo in pools.items():
        replicas = pinfo.get("replicas")
        if not isinstance(replicas, int) or isinstance(replicas,
                                                       bool) or replicas < 1:
            errors.append(
                f"Pool '{role}' replicas must be an integer >= 1, got {replicas!r}."
            )
        size = pinfo.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            errors.append(
                f"Pool '{role}' leaderWorkerTemplate.size must be an integer >= 1, got {size!r}."
            )
        vllm_flags = pinfo.get("vllm_flags") or {}
        if "--gpu-memory-utilization" in vllm_flags:
            gmu = vllm_flags["--gpu-memory-utilization"]
            if (not isinstance(gmu, (int, float)) or isinstance(gmu, bool)
                    or not (0.0 < float(gmu) <= 1.0)):
                errors.append(
                    f"Pool '{role}' --gpu-memory-utilization must be in (0.0, 1.0], got {gmu!r}."
                )
        if "--block-size" in vllm_flags:
            bs = vllm_flags["--block-size"]
            if not isinstance(bs, int) or isinstance(bs, bool) or bs <= 0:
                errors.append(
                    f"Pool '{role}' --block-size must be a positive integer (> 0), got {bs!r}."
                )
        if "--max-num-batched-tokens" in vllm_flags:
            mbt = vllm_flags["--max-num-batched-tokens"]
            if not isinstance(mbt, int) or isinstance(mbt, bool) or mbt <= 0:
                errors.append(
                    f"Pool '{role}' --max-num-batched-tokens must be a positive integer (> 0), got {mbt!r}."
                )
        if "--max-num-seqs" in vllm_flags:
            mns = vllm_flags["--max-num-seqs"]
            if not isinstance(mns, int) or isinstance(mns, bool) or mns <= 0:
                errors.append(
                    f"Pool '{role}' --max-num-seqs must be a positive integer (> 0), got {mns!r}."
                )

    return errors


def _validate_epp_inner_doc(inner_doc: Any, label: str) -> list[str]:
    """Validate a parsed EndpointPickerConfig mapping and return any errors."""
    errors: list[str] = []
    if not isinstance(inner_doc, Mapping):
        return [
            f"{label}: EndpointPickerConfig must parse into a YAML mapping, got {type(inner_doc).__name__}."
        ]
    if inner_doc.get("kind") != "EndpointPickerConfig":
        errors.append(
            f"{label}: expected kind 'EndpointPickerConfig', got {inner_doc.get('kind')!r}."
        )
    api_version = inner_doc.get("apiVersion")
    if not isinstance(api_version, str) or not api_version.strip():
        errors.append(
            f"{label}: EndpointPickerConfig must have a non-empty string 'apiVersion'."
        )
    api_ver_str = str(api_version or "").strip()

    scorer_weights_found: list[float] = []

    if "scheduling" in inner_doc:
        sched = inner_doc["scheduling"]
        if not isinstance(sched, Mapping):
            errors.append(
                f"{label}: 'scheduling' must be a mapping, got {type(sched).__name__}."
            )
        else:
            if "scorers" in sched:
                raw_scorers = sched["scorers"]
                if not isinstance(raw_scorers, list):
                    errors.append(
                        f"{label}: 'scheduling.scorers' must be a list, got {type(raw_scorers).__name__}."
                    )
                else:
                    for idx, s_entry in enumerate(raw_scorers):
                        if not isinstance(s_entry, Mapping):
                            errors.append(
                                f"{label}: scheduling.scorers[{idx}] must be a mapping."
                            )
                            continue
                        sname = s_entry.get("name")
                        if not isinstance(sname, str) or not sname.strip():
                            errors.append(
                                f"{label}: scheduling.scorers[{idx}] must have a non-empty 'name'."
                            )
                        w = s_entry.get("weight", 1)
                        if (not isinstance(w, (int, float))
                                or isinstance(w, bool)
                                or not math.isfinite(float(w))):
                            errors.append(
                                f"{label}: scorer {sname!r} weight must be a finite number, got {w!r}."
                            )
                        elif float(w) < 0:
                            errors.append(
                                f"{label}: scorer {sname!r} weight must be non-negative, got {w}."
                            )
                        else:
                            scorer_weights_found.append(float(w))
                        params = s_entry.get("parameters")
                        if params is not None and not isinstance(
                                params, Mapping):
                            errors.append(
                                f"{label}: scorer {sname!r} 'parameters' must be a mapping, got {type(params).__name__}."
                            )
            if "filters" in sched:
                raw_filters = sched["filters"]
                if not isinstance(raw_filters, list):
                    errors.append(
                        f"{label}: 'scheduling.filters' must be a list, got {type(raw_filters).__name__}."
                    )
                else:
                    for idx, f_entry in enumerate(raw_filters):
                        if not isinstance(f_entry, Mapping):
                            errors.append(
                                f"{label}: scheduling.filters[{idx}] must be a mapping."
                            )
                            continue
                        fname = f_entry.get("name")
                        if not isinstance(fname, str) or not fname.strip():
                            errors.append(
                                f"{label}: scheduling.filters[{idx}] must have a non-empty 'name'."
                            )
                        params = f_entry.get("parameters")
                        if params is not None and not isinstance(
                                params, Mapping):
                            errors.append(
                                f"{label}: filter {fname!r} 'parameters' must be a mapping, got {type(params).__name__}."
                            )

    if "plugins" in inner_doc:
        raw_plugins = inner_doc["plugins"]
        if not isinstance(raw_plugins, list):
            errors.append(
                f"{label}: 'plugins' must be a list, got {type(raw_plugins).__name__}."
            )
        else:
            for idx, p_entry in enumerate(raw_plugins):
                if not isinstance(p_entry, Mapping):
                    errors.append(
                        f"{label}: plugins[{idx}] must be a mapping.")
                    continue
                ptype = p_entry.get("type") or p_entry.get("name")
                if not isinstance(ptype, str) or not ptype.strip():
                    errors.append(
                        f"{label}: plugins[{idx}] must have a non-empty 'type' or 'name'."
                    )
                if api_ver_str.startswith("llm-d.ai/") and "weight" in p_entry:
                    errors.append(
                        f"{label}: top-level plugin {ptype!r} must not carry 'weight' in {api_ver_str}; weights belong in schedulingProfiles[].plugins."
                    )
                params = p_entry.get("parameters")
                if params is not None and not isinstance(params, Mapping):
                    errors.append(
                        f"{label}: plugin {ptype!r} 'parameters' must be a mapping, got {type(params).__name__}."
                    )

    if "schedulingProfiles" in inner_doc:
        raw_profiles = inner_doc["schedulingProfiles"]
        if not isinstance(raw_profiles, list) or not raw_profiles:
            errors.append(
                f"{label}: 'schedulingProfiles' must be a non-empty list.")
        else:
            for idx, sp_entry in enumerate(raw_profiles):
                if not isinstance(sp_entry, Mapping):
                    errors.append(
                        f"{label}: schedulingProfiles[{idx}] must be a mapping."
                    )
                    continue
                sp_plugins = sp_entry.get("plugins")
                if not isinstance(sp_plugins, list):
                    errors.append(
                        f"{label}: schedulingProfiles[{idx}].plugins must be a list."
                    )
                    continue
                for p_idx, ref_entry in enumerate(sp_plugins):
                    if not isinstance(ref_entry, Mapping):
                        errors.append(
                            f"{label}: schedulingProfiles[{idx}].plugins[{p_idx}] must be a mapping."
                        )
                        continue
                    ref_name = ref_entry.get("pluginRef") or ref_entry.get(
                        "name")
                    if not isinstance(ref_name, str) or not ref_name.strip():
                        errors.append(
                            f"{label}: schedulingProfiles[{idx}].plugins[{p_idx}] must have a non-empty 'pluginRef'."
                        )
                    if "weight" in ref_entry:
                        w = ref_entry["weight"]
                        if (not isinstance(w, (int, float))
                                or isinstance(w, bool)
                                or not math.isfinite(float(w))):
                            errors.append(
                                f"{label}: schedulingProfiles[{idx}] plugin {ref_name!r} weight must be a finite number, got {w!r}."
                            )
                        elif float(w) < 0:
                            errors.append(
                                f"{label}: schedulingProfiles[{idx}] plugin {ref_name!r} weight must be non-negative, got {w}."
                            )
                        else:
                            scorer_weights_found.append(float(w))
                    elif isinstance(ref_name, str) and "scorer" in ref_name:
                        scorer_weights_found.append(1.0)

    if not scorer_weights_found and not errors:
        errors.append(
            f"{label}: EndpointPickerConfig must declare at least one scorer.")
    elif scorer_weights_found and sum(scorer_weights_found) <= 0:
        errors.append(
            f"{label}: total active scorer weight in EndpointPickerConfig must be > 0."
        )

    return errors


def validate_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Validate all YAML manifests, nested ConfigMap EndpointPickerConfig docs, LWS flags, and Kustomize rendering."""
    errors: list[str] = []
    warnings: list[str] = []

    raw_path = Path(bundle_dir)
    if not raw_path.exists() or not raw_path.is_dir():
        return {
            "valid": False,
            "errors": [f"Bundle directory does not exist: {bundle_dir}"],
            "warnings": [],
            "scorer_weights": {},
            "filter_params": {},
            "pools": {},
            "recipe_fingerprint": "",
        }

    bdir = _resolve_bundle_root(raw_path)
    yaml_files = _collect_bundle_yaml_files(bdir)
    if not yaml_files:
        return {
            "valid": False,
            "errors": [f"No YAML manifests found in {bdir}"],
            "warnings": [],
            "scorer_weights": {},
            "filter_params": {},
            "pools": {},
            "recipe_fingerprint": "",
        }

    epp_docs_validated = 0
    for yaml_path in yaml_files:
        rel = yaml_path.relative_to(bdir).as_posix()
        try:
            text = yaml_path.read_text(encoding="utf-8")
            raw_docs = list(yaml.safe_load_all(text))
        except Exception as exc:
            errors.append(f"{rel}: YAML syntax error: {exc}")
            continue

        non_empty_docs = [d for d in raw_docs if d is not None]
        if not non_empty_docs:
            warnings.append(f"{rel}: file contains no YAML documents.")
            continue

        for doc_idx, doc in enumerate(non_empty_docs):
            if not isinstance(doc, Mapping):
                errors.append(
                    f"{rel} (doc #{doc_idx + 1}): YAML document must be a mapping, got {type(doc).__name__}."
                )
                continue

            kind = doc.get("kind")
            if kind == "EndpointPickerConfig":
                epp_docs_validated += 1
                errors.extend(_validate_epp_inner_doc(doc, rel))
            elif kind == "ConfigMap":
                data = doc.get("data")
                if data is not None and not isinstance(data, Mapping):
                    errors.append(
                        f"{rel}: ConfigMap 'data' must be a mapping, got {type(data).__name__}."
                    )
                elif isinstance(data, Mapping):
                    for key, val in data.items():
                        key_str = str(key)
                        is_known_epp_key = key_str in EPP_CONFIGMAP_KEYS
                        is_yaml_key = key_str.endswith((".yaml", ".yml"))
                        if not is_known_epp_key and not is_yaml_key:
                            continue
                        if not isinstance(val, str) or not val.strip():
                            errors.append(
                                f"{rel} [{key_str}]: ConfigMap YAML entry must be a non-empty string."
                            )
                            continue
                        try:
                            inner_docs = [
                                d for d in yaml.safe_load_all(val)
                                if d is not None
                            ]
                        except Exception as exc:
                            errors.append(
                                f"{rel} [{key_str}]: nested YAML syntax error: {exc}"
                            )
                            continue
                        if not inner_docs:
                            errors.append(
                                f"{rel} [{key_str}]: nested YAML string is empty."
                            )
                            continue
                        inner = inner_docs[0]
                        if not isinstance(inner, Mapping):
                            errors.append(
                                f"{rel} [{key_str}]: nested YAML must parse into a mapping, got {type(inner).__name__}."
                            )
                            continue
                        if is_known_epp_key or inner.get(
                                "kind") == "EndpointPickerConfig":
                            epp_docs_validated += 1
                            errors.extend(
                                _validate_epp_inner_doc(
                                    inner, f"{rel} [{key_str}]"))
            elif kind == "LeaderWorkerSet":
                container = _find_lws_container(doc)
                if container is not None:
                    cmd = container.get("command") or []
                    args = container.get("args") or []
                    if (len(cmd) >= 2 and "bash" in str(cmd[0])
                            and len(args) == 1
                            and "vllm serve" in str(args[0])):
                        cmd_str = str(args[0])
                        idx = cmd_str.find("vllm serve")
                        tail = cmd_str[idx + len("vllm serve"):].strip()
                        try:
                            shlex.split(tail)
                        except ValueError as exc:
                            errors.append(
                                f"{rel}: malformed shell quoting in vllm serve args: {exc}"
                            )

    if epp_docs_validated == 0:
        errors.append(
            f"No EndpointPickerConfig document or ConfigMap entry found in {bdir}."
        )

    root_epp = bdir / "01-endpointpickerconfig.yaml"
    kust_epp = bdir / "kustomize" / "01-endpointpickerconfig.yaml"
    if not errors and root_epp.is_file() and kust_epp.is_file():
        root_bytes = root_epp.read_bytes()
        kust_bytes = kust_epp.read_bytes()
        if root_bytes != kust_bytes:
            if root_epp.stat().st_mtime_ns >= kust_epp.stat().st_mtime_ns:
                kust_epp.write_bytes(root_bytes)
            else:
                root_epp.write_bytes(kust_bytes)
            warnings.append(
                "Synchronized 01-endpointpickerconfig.yaml between bundle root and kustomize/ overlay."
            )

    snapshot: dict[str, Any] = {}
    filter_params: dict[str, Any] = {}
    if not errors:
        try:
            snapshot = inspect_bundle(bdir)
            invariant_errors = validate_snapshot_invariants(snapshot)
            errors.extend(invariant_errors)
            filter_params = {
                fname:
                dict((finfo.get("parameters")
                      if isinstance(finfo, Mapping) else {}) or {})
                for fname, finfo in (snapshot.get("filters") or {}).items()
            }
        except Exception as exc:
            errors.append(f"Bundle inspection failed: {exc}")

    if not errors and shutil.which("kubectl") is not None:
        kustomize_target: Path | None = None
        if (bdir / "kustomize" / "kustomization.yaml").is_file():
            kustomize_target = bdir / "kustomize"
        elif (bdir / "kustomization.yaml").is_file():
            kustomize_target = bdir
        if kustomize_target is not None:
            proc = subprocess.run(
                ["kubectl", "kustomize",
                 str(kustomize_target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                err_detail = (proc.stderr or proc.stdout or "").strip()
                errors.append(
                    f"kubectl kustomize failed (exit {proc.returncode}): {err_detail}"
                )

    return {
        "valid":
        len(errors) == 0,
        "errors":
        errors,
        "warnings":
        warnings,
        "scorer_weights":
        snapshot.get("scorer_weights", {}) if snapshot else {},
        "filter_params":
        filter_params,
        "pools":
        snapshot.get("pools", {}) if snapshot else {},
        "recipe_fingerprint":
        snapshot.get("recipe_fingerprint", "") if snapshot else "",
    }


def _parse_kv(spec: str) -> tuple[str, str]:
    """Parse `<key>=<val>` into `(key, val)`, stripping any optional `serving:`/`default:` prefix."""
    if "=" not in spec:
        raise ValueError(f"Expected '<key>=<val>', got: {spec!r}")
    lhs, rhs = spec.split("=", 1)
    if ":" in lhs:
        _, lhs = lhs.split(":", 1)
    return lhs.strip(), rhs.strip()


def mutate_bundle(
    bundle_dir: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    patch_json: str | Path | Mapping[str, Any] | None = None,
    scorers: Mapping[str, float | int] | None = None,
    filters: Mapping[str, Mapping[str, Any]] | None = None,
    vllm_args: Mapping[str, Any] | None = None,
    env_vars: Mapping[str, Any] | None = None,
    replicas: int | None = None,
    set_scorers: Sequence[str] | None = None,
    set_filter_params: Sequence[str] | None = None,
    set_filters: Sequence[str] | None = None,
    set_vllm_flags: Sequence[str] | None = None,
    set_vllm_args: Sequence[str] | None = None,
    set_replicas: Sequence[str] | None = None,
    set_envs: Sequence[str] | None = None,
    set_lws_sizes: Sequence[str] | None = None,
    validate: bool = True,
    dry_run: bool = False,
    apply: bool = False,
    namespace: str | None = None,
    context: str | None = None,
    kubeconfig: str | Path | None = None,
) -> dict[str, Any]:
    """Apply declarative mutations to a homogeneous `llm-d` bundle with invariant validation."""
    del validate
    bdir = _resolve_bundle_root(bundle_dir)

    pre_snapshot = inspect_bundle(bdir)
    pre_fingerprint = pre_snapshot["recipe_fingerprint"]

    if checkpoint_dir is not None and not dry_run:
        checkpoint_bundle(bdir, checkpoint_dir)

    scorer_updates: list[tuple[str, float | int]] = []
    filter_updates: list[tuple[str, str, Any]] = []
    saturation_updates: dict[str, Any] = {}
    vllm_flag_updates: list[tuple[str, Any]] = []
    replica_updates: list[int] = []
    lws_size_updates: list[int] = []
    env_updates: list[tuple[str, Any]] = []

    if scorers:
        for sname, sval in scorers.items():
            scorer_updates.append((str(sname), _coerce_scalar(sval)))
    if filters:
        for fname, fparams in filters.items():
            if isinstance(fparams, Mapping):
                for pk, pv in fparams.items():
                    filter_updates.append(
                        (str(fname), str(pk), _coerce_scalar(pv)))
    if vllm_args:
        for fk, fv in vllm_args.items():
            flag_name, val_str = _parse_kv(f"{fk}={fv}")
            vllm_flag_updates.append((flag_name, _coerce_scalar(val_str)))
    if env_vars:
        for ek, ev in env_vars.items():
            env_key, env_val = _parse_kv(f"{ek}={ev}")
            env_updates.append((env_key, env_val))
    if replicas is not None:
        replica_updates.append(int(replicas))

    if patch_json is not None:
        if isinstance(patch_json, Mapping):
            patch_data = dict(patch_json)
        else:
            patch_str = str(patch_json).strip()
            if patch_str.startswith("{"):
                patch_data = json.loads(patch_str)
            else:
                patch_data = json.loads(
                    Path(patch_str).read_text(encoding="utf-8"))
        if "epp_config" in patch_data and isinstance(patch_data["epp_config"],
                                                     Mapping):
            patch_data = dict(patch_data["epp_config"])

        for sname, sval in (patch_data.get("scorers") or {}).items():
            if isinstance(sval, Mapping):
                w = _coerce_scalar(sval.get("weight", 1))
                scorer_updates.append((str(sname), w))
            else:
                scorer_updates.append((str(sname), _coerce_scalar(sval)))

        for fname, fparams in (patch_data.get("filters") or {}).items():
            if isinstance(fparams, Mapping):
                inner_p = (fparams.get("parameters") if "parameters" in fparams
                           and isinstance(fparams["parameters"], Mapping) else
                           fparams)
                for pk, pv in inner_p.items():
                    filter_updates.append(
                        (str(fname), str(pk), _coerce_scalar(pv)))

        if "saturation_thresholds" in patch_data and isinstance(
                patch_data["saturation_thresholds"], Mapping):
            saturation_updates.update(patch_data["saturation_thresholds"])
        if "flow_control" in patch_data and isinstance(
                patch_data["flow_control"], Mapping):
            saturation_updates.update(patch_data["flow_control"])

        for flag_k, flag_v in (patch_data.get("vllm_flags") or {}).items():
            vllm_flag_updates.append((str(flag_k), _coerce_scalar(flag_v)))

        if "replicas" in patch_data:
            rep_spec = patch_data["replicas"]
            if isinstance(rep_spec, int):
                replica_updates.append(rep_spec)
            elif isinstance(rep_spec, Mapping):
                for r_role, r_cnt in rep_spec.items():
                    if r_role != "total":
                        replica_updates.append(int(r_cnt))

        for ek, ev in (patch_data.get("env") or {}).items():
            env_updates.append((str(ek), ev))

        for _, ppatch in (patch_data.get("pools") or {}).items():
            if not isinstance(ppatch, Mapping):
                continue
            if "replicas" in ppatch:
                replica_updates.append(int(ppatch["replicas"]))
            if "leaderWorkerTemplate.size" in ppatch:
                lws_size_updates.append(
                    int(ppatch["leaderWorkerTemplate.size"]))
            elif "size" in ppatch:
                lws_size_updates.append(int(ppatch["size"]))
            for fk, fv in (ppatch.get("vllm_flags") or {}).items():
                vllm_flag_updates.append((str(fk), _coerce_scalar(fv)))
            for ek, ev in (ppatch.get("env") or {}).items():
                env_updates.append((str(ek), ev))

    for spec in set_scorers or []:
        sname, val_str = _parse_kv(spec)
        weight_val = _coerce_scalar(val_str)
        if not isinstance(weight_val,
                          (int, float)) or isinstance(weight_val, bool):
            raise ValueError(
                f"Scorer weight for '{sname}' must be numeric, got {val_str!r}"
            )
        scorer_updates.append((sname, weight_val))

    for spec in list(set_filter_params or []) + list(set_filters or []):
        if "=" not in spec:
            raise ValueError(
                f"Expected '<filter>.<param>=<val>' or '<filter>=<param>:<val>', got {spec!r}"
            )
        lhs, rhs = spec.split("=", 1)
        if "." in lhs:
            fname, param_key = lhs.rsplit(".", 1)
            filter_updates.append(
                (fname.strip(), param_key.strip(), _coerce_scalar(rhs)))
        elif ":" in rhs:
            param_key, param_val = rhs.split(":", 1)
            filter_updates.append(
                (lhs.strip(), param_key.strip(), _coerce_scalar(param_val)))
        else:
            raise ValueError(
                f"Expected '<filter>.<param>=<val>' or '<filter>=<param>:<val>', got {spec!r}"
            )

    for spec in list(set_vllm_flags or []) + list(set_vllm_args or []):
        flag_name, val_str = _parse_kv(spec)
        vllm_flag_updates.append((flag_name, _coerce_scalar(val_str)))

    for spec in set_replicas or []:
        val_part = spec.split("=", 1)[1] if "=" in spec else spec
        cnt = _coerce_scalar(val_part)
        if not isinstance(cnt, int) or isinstance(cnt, bool):
            raise ValueError(f"Replicas must be an integer, got {spec!r}")
        replica_updates.append(cnt)

    for spec in set_lws_sizes or []:
        val_part = spec.split("=", 1)[1] if "=" in spec else spec
        sz = _coerce_scalar(val_part)
        if not isinstance(sz, int) or isinstance(sz, bool):
            raise ValueError(
                f"leaderWorkerTemplate.size must be an integer, got {spec!r}")
        lws_size_updates.append(sz)

    for spec in set_envs or []:
        env_key, env_val = _parse_kv(spec)
        env_updates.append((env_key, env_val))

    with tempfile.TemporaryDirectory() as tmp_root:
        staging_dir = Path(tmp_root) / "bundle"
        shutil.copytree(bdir, staging_dir)

        changed_rel_paths: set[str] = set()

        if scorer_updates or filter_updates or saturation_updates:
            epp_candidates = [
                staging_dir / "01-endpointpickerconfig.yaml",
                staging_dir / "kustomize" / "01-endpointpickerconfig.yaml",
            ]
            for epp_file in epp_candidates:
                if not epp_file.is_file():
                    continue
                outer_docs = _load_yaml_docs(epp_file)
                for doc in outer_docs:
                    kind = doc.get("kind")
                    if kind == "EndpointPickerConfig":
                        _mutate_epp_inner_doc(
                            doc,
                            scorer_updates=scorer_updates,
                            filter_updates=filter_updates,
                            saturation_updates=saturation_updates,
                        )
                    elif kind == "ConfigMap":
                        data = doc.get("data") or {}
                        for key in _iter_configmap_epp_keys(data):
                            inner = yaml.safe_load(data[key])
                            if isinstance(inner, dict):
                                _mutate_epp_inner_doc(
                                    inner,
                                    scorer_updates=scorer_updates,
                                    filter_updates=filter_updates,
                                    saturation_updates=saturation_updates,
                                )
                                data[key] = _dump_yaml(inner)
                _dump_yaml_docs(epp_file, outer_docs)
                changed_rel_paths.add(
                    epp_file.relative_to(staging_dir).as_posix())

        if vllm_flag_updates or replica_updates or lws_size_updates or env_updates:
            target_dirs = [staging_dir]
            if (staging_dir / "kustomize").is_dir():
                target_dirs.append(staging_dir / "kustomize")

            for tdir in target_dirs:
                pool_files = _discover_pool_files(tdir)
                for _, paths in pool_files.items():
                    for lws_path in paths:
                        is_base = lws_path.name == "base-modelserver.yaml"
                        has_patch_with_spec = any(
                            p.name != "base-modelserver.yaml" and any(
                                "spec" in d for d in _load_yaml_docs(p)
                                if d.get("kind") == "LeaderWorkerSet")
                            for p in paths)
                        docs = _load_yaml_docs(lws_path)
                        file_modified = False

                        for doc in docs:
                            if doc.get("kind") != "LeaderWorkerSet":
                                continue
                            if not is_base and "spec" not in doc:
                                continue
                            spec = doc.setdefault("spec", {})
                            lwt = spec.setdefault("leaderWorkerTemplate", {})
                            container = _find_lws_container(doc)

                            for rep_cnt in replica_updates:
                                if not is_base or not has_patch_with_spec:
                                    spec["replicas"] = rep_cnt
                                    file_modified = True

                            for sz in lws_size_updates:
                                if is_base or not has_patch_with_spec or "size" in lwt:
                                    lwt["size"] = sz
                                    file_modified = True

                            if container is not None and (
                                    not is_base or not has_patch_with_spec):
                                for flag_name, flag_val in vllm_flag_updates:
                                    _apply_vllm_flag_to_container(
                                        container, flag_name, flag_val)
                                    file_modified = True

                                for env_k, env_v in env_updates:
                                    _apply_env_to_container(
                                        container, env_k, env_v)
                                    file_modified = True

                        if file_modified:
                            _dump_yaml_docs(lws_path, docs)
                            changed_rel_paths.add(
                                lws_path.relative_to(staging_dir).as_posix())

        post_snapshot = inspect_bundle(staging_dir)
        invariant_errors = validate_snapshot_invariants(post_snapshot)
        if invariant_errors:
            raise ValueError("Invalid EPP/LWS mutation invariants:\n- " +
                             "\n- ".join(invariant_errors))

        if not dry_run:
            for rel in sorted(changed_rel_paths):
                src = staging_dir / rel
                dst = bdir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())

    apply_info = apply_bundle(
        bdir,
        namespace=namespace,
        context=context,
        kubeconfig=kubeconfig,
        execute=bool(apply and not dry_run),
    )

    return {
        "status": "completed",
        "dry_run": dry_run,
        "bundle_dir": str(bdir),
        "serving_mode": post_snapshot["serving_mode"],
        "pre_fingerprint": pre_fingerprint,
        "post_fingerprint": post_snapshot["recipe_fingerprint"],
        "fingerprint_before": pre_fingerprint,
        "fingerprint_after": post_snapshot["recipe_fingerprint"],
        "recipe_fingerprint": post_snapshot["recipe_fingerprint"],
        "changed_files": sorted(changed_rel_paths),
        **apply_info,
        "scorers": post_snapshot["scorers"],
        "scorer_weights": post_snapshot["scorer_weights"],
        "filters": post_snapshot["filters"],
        "saturation_thresholds": post_snapshot["saturation_thresholds"],
        "pools": post_snapshot["pools"],
        "vllm_flags": post_snapshot["vllm_flags"],
        "replicas": post_snapshot["replicas"],
        "env": post_snapshot["env"],
        "epp_config": post_snapshot["epp_config"],
        "snapshot": post_snapshot,
    }


def _preprocess_argv(argv: Sequence[str]) -> list[str]:
    """Normalize `--set-vllm-flag --flag=val` and `--set-vllm-arg --flag=val` for argparse."""
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if (tok in ("--set-vllm-flag", "--set-vllm-arg") and i + 1 < n
                and argv[i + 1].startswith("--")):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "Inspect, validate, checkpoint, rollback, and mutate homogeneous llm-d Kustomize bundles."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_inspect = subparsers.add_parser(
        "inspect",
        help=
        "Parse EndpointPickerConfig and LeaderWorkerSet manifests and emit canonical JSON.",
    )
    p_inspect.add_argument("--bundle-dir",
                           required=True,
                           help="Path to the llm-d bundle directory.")
    p_inspect.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit JSON output (default: True).",
    )

    p_val = subparsers.add_parser(
        "validate",
        help=
        "Validate all YAML manifests, nested EndpointPickerConfig ConfigMaps, LWS flags, and Kustomize rendering.",
    )
    p_val.add_argument("--bundle-dir",
                       required=True,
                       help="Path to the llm-d bundle directory.")
    p_val.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit JSON output (default: True).",
    )

    p_ckpt = subparsers.add_parser(
        "checkpoint",
        help=
        "Save a bitwise-exact snapshot of all YAML manifests in --bundle-dir.",
    )
    p_ckpt.add_argument("--bundle-dir",
                        required=True,
                        help="Path to the llm-d bundle directory.")
    p_ckpt.add_argument(
        "--checkpoint-dir",
        required=True,
        help=
        "Directory where the checkpoint snapshot and manifest will be written.",
    )
    p_ckpt.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit JSON output (default: True).",
    )

    p_rb = subparsers.add_parser(
        "rollback",
        help=
        "Atomically restore --bundle-dir from --checkpoint-dir and verify recipe_fingerprint.",
    )
    p_rb.add_argument("--bundle-dir",
                      required=True,
                      help="Path to the llm-d bundle directory.")
    p_rb.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing a prior checkpoint.",
    )
    p_rb.add_argument(
        "--apply",
        action="store_true",
        help=
        "Apply restored Kustomize manifests to the cluster and restart EPP Deployment.",
    )
    p_rb.add_argument("--namespace",
                      default=None,
                      help="Optional Kubernetes namespace for --apply.")
    p_rb.add_argument("--context",
                      default=None,
                      help="Optional kubectl context for --apply.")
    p_rb.add_argument("--kubeconfig",
                      default=None,
                      help="Optional kubeconfig file path for --apply.")
    p_rb.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit JSON output (default: True).",
    )

    p_mut = subparsers.add_parser(
        "mutate",
        help=
        "Apply declarative mutations to EndpointPickerConfig and LeaderWorkerSet manifests.",
    )
    p_mut.add_argument("--bundle-dir",
                       required=True,
                       help="Path to the llm-d bundle directory.")
    p_mut.add_argument(
        "--checkpoint-dir",
        default=None,
        help=
        "Optional directory to save a pre-mutation checkpoint before mutating.",
    )
    p_mut.add_argument(
        "--patch-json",
        default=None,
        help="Inline JSON string or path to JSON patch file.",
    )
    p_mut.add_argument(
        "--set-scorer",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Set scorer weight (e.g. prefix-cache-scorer=80).",
    )
    p_mut.add_argument(
        "--set-filter-param",
        action="append",
        default=[],
        metavar="FILTER.PARAM=VAL",
        help=
        "Set filter parameter (e.g. prefix-cache-affinity-filter.affinityThreshold=0.85).",
    )
    p_mut.add_argument(
        "--set-filter",
        action="append",
        default=[],
        metavar="FILTER=PARAM:VAL",
        help=
        "Set filter parameter (e.g. prefix-cache-affinity-filter=affinityThreshold:0.85).",
    )
    p_mut.add_argument(
        "--set-vllm-flag",
        action="append",
        default=[],
        metavar="FLAG=VAL",
        help="Set vllm serve CLI flag (e.g. --max-num-batched-tokens=16384).",
    )
    p_mut.add_argument(
        "--set-vllm-arg",
        action="append",
        default=[],
        metavar="FLAG=VAL",
        help="Alias for --set-vllm-flag.",
    )
    p_mut.add_argument(
        "--set-replicas",
        action="append",
        default=[],
        metavar="INT",
        help="Set LeaderWorkerSet replica count (e.g. 4).",
    )
    p_mut.add_argument(
        "--set-lws-size",
        action="append",
        default=[],
        metavar="INT",
        help="Set leaderWorkerTemplate.size (e.g. 2).",
    )
    p_mut.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help=
        "Set container environment variable (e.g. VLLM_TORCH_PROFILER_DIR=/tmp/prof).",
    )
    p_mut.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate YAML and invariants after mutation (default: True).",
    )
    p_mut.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit JSON output (default: True).",
    )
    p_mut.add_argument(
        "--dry-run",
        action="store_true",
        help=
        "Validate mutations and compute post-mutation fingerprint without writing files.",
    )
    p_mut.add_argument(
        "--apply",
        action="store_true",
        help=
        "Apply mutated Kustomize manifests to the cluster after writing files.",
    )
    p_mut.add_argument("--namespace",
                       default=None,
                       help="Optional Kubernetes namespace for --apply.")
    p_mut.add_argument("--context",
                       default=None,
                       help="Optional kubectl context for --apply.")
    p_mut.add_argument("--kubeconfig",
                       default=None,
                       help="Optional kubeconfig file path for --apply.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    norm_argv = _preprocess_argv(raw_argv)
    parser = build_arg_parser()
    args = parser.parse_args(norm_argv)

    try:
        if args.subcommand == "inspect":
            result = inspect_bundle(args.bundle_dir)
        elif args.subcommand == "validate":
            result = validate_bundle(args.bundle_dir)
            print(json.dumps(result, indent=2))
            return 0 if result.get("valid") else 1
        elif args.subcommand == "checkpoint":
            result = checkpoint_bundle(args.bundle_dir, args.checkpoint_dir)
        elif args.subcommand == "rollback":
            result = rollback_bundle(
                args.bundle_dir,
                args.checkpoint_dir,
                apply=args.apply,
                namespace=args.namespace,
                context=args.context,
                kubeconfig=args.kubeconfig,
            )
        elif args.subcommand == "mutate":
            result = mutate_bundle(
                args.bundle_dir,
                checkpoint_dir=args.checkpoint_dir,
                patch_json=args.patch_json,
                set_scorers=args.set_scorer,
                set_filter_params=args.set_filter_param,
                set_filters=args.set_filter,
                set_vllm_flags=args.set_vllm_flag,
                set_vllm_args=args.set_vllm_arg,
                set_replicas=args.set_replicas,
                set_lws_sizes=args.set_lws_size,
                set_envs=args.set_env,
                validate=args.validate,
                dry_run=args.dry_run,
                apply=args.apply,
                namespace=args.namespace,
                context=args.context,
                kubeconfig=args.kubeconfig,
            )
        else:
            parser.error(f"Unknown subcommand: {args.subcommand}")
            return 2
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
