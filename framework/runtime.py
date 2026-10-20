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
"""Embedded runtime for TPU agent suite packages.

Provides runtime execution support for run namespace allocation and deterministic payload validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import datetime
import io
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Any, Callable, Iterator


def _is_hook_dispatch(argv: list[str]) -> bool:
    """Whether `argv` (without the program name) invokes `hook-dispatch`."""
    return argv[:1] == ["hook-dispatch"]


def _hook_dispatch_argv_hints(argv: list[str]) -> dict[str, str | None]:
    """Lenient, stdlib-only read of the `hook-dispatch` flags a failure reply needs.

    Used when the strict parser rejects the command line or the framework
    did not import. Unknown flags are ignored; absent flags are None; {} if
    even this read fails.
    """
    lenient = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ("--harness", "--hook", "--event", "--on-failure",
                 "--handback-event", "--workspace"):
        lenient.add_argument(flag, default=None)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            known, _ = lenient.parse_known_args(argv[1:])
    except SystemExit:
        return {}
    return vars(known)


# Run as a script, this file has its own directory on sys.path, not the one
# that holds the framework package: in the repository the root, in a build or
# the central store the package directory. Add it, so `framework` imports.
_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

try:
    from framework import compute, journal, study, trajectory, utils
except Exception as _import_error:  # noqa: BLE001 - becomes the hook's failure
    # A hook-dispatch whose package is half gone still exits by its policy
    # instead of a traceback (exit 1, which Antigravity refuses on). Not only
    # ImportError: `utils` raises at import when package.release.json is missing.
    # The policy is read from the command line; compiled commands always
    # state it. A hand-typed one that does not gets the traceback, which
    # is the more useful answer when debugging.
    if not (__name__ == "__main__" and _is_hook_dispatch(sys.argv[1:])):
        raise
    _hints = _hook_dispatch_argv_hints(sys.argv[1:])
    _on_failure = _hints.get("on_failure")
    if not _on_failure:
        raise
    print(
        f"runtime: [{_hints.get('hook') or '?'}] hook failed (startup): "
        f"{type(_import_error).__name__}: {_import_error}; "
        f"on_failure={_on_failure}",
        file=sys.stderr)
    raise SystemExit(0 if _on_failure == "allow" else 2)

Path = pathlib.Path

TpuAgentSuiteError = utils.TpuAgentSuiteError
DEFAULT_HOOK_TIMEOUT = utils.DEFAULT_HOOK_TIMEOUT
DEFAULT_HOOK_ON_FAILURE = utils.DEFAULT_HOOK_ON_FAILURE
TPU_AGENT_SUITE_VERSION = utils.TPU_AGENT_SUITE_VERSION
GOOGLE_HARNESSES = utils.GOOGLE_HARNESSES
GOOGLE_HARNESS_BRAIN_DIRS = utils.GOOGLE_HARNESS_BRAIN_DIRS
compute_digest = utils.compute_digest
event_mapping = utils.event_mapping
extract_all_json_objects = utils.extract_all_json_objects
extract_json_object = utils.extract_json_object
format_json = utils.format_json
load_schema_example = utils.load_schema_example
get_event_mappings = utils.get_event_mappings
get_hook_output_schema = utils.get_hook_output_schema
get_tool_mappings = utils.get_tool_mappings
harness_mapping = utils.harness_mapping
is_google_harness = utils.is_google_harness
subagent_handback_mapping = utils.subagent_handback_mapping
to_canonical_event = utils.to_canonical_event
load_json = utils.load_json
load_json_doc = utils.load_json_doc
rooted = utils.rooted
validate_instance = utils.validate_instance
validate_package_name = utils.validate_package_name
validate_run_id = utils.validate_run_id
utc_now_iso = utils.utc_now_iso
write_json = utils.write_json

# Ceiling on spooled context entries per session, oldest evicted first.
# Every live entry is replayed into emulated context on each invocation until
# pushed out by newer entries, so this bounds the per-call prompt overhead.
MAX_PENDING_CONTEXT_ENTRIES = 8

# Turns the `validate-schema` builtin into a no-op, so every delegation request
# and subagent deliverable is allowed through unchecked.
# Note: Hook dispatch executes the built runtime copy in the installed package
# directory (e.g. ~/.tpu-agent-suite/packages/<package_tag>/framework/runtime.py or
# the staged dist). Editing this constant in the source tree applies to future
# package builds; to disable verification in an already-built or installed
# environment via the constant, edit the built runtime.py directly.
# Alternatively, set the TPU_AGENT_SUITE_DISABLE_SCHEMA_VERIFIERS environment variable
# to disable verification for any existing install without editing files or
# rebuilding.
DISABLE_SCHEMA_VERIFIERS = False
DISABLE_SCHEMA_VERIFIERS_ENV_VAR = "TPU_AGENT_SUITE_DISABLE_SCHEMA_VERIFIERS"


def generate_run_id() -> str:
    """Generate a collision-resistant timestamped run identifier."""
    timestamp = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def get_installed_package_root(workspace: Path | str | None = None,
                               package: str = "") -> Path:
    """Resolve a package's installed root: the central versioned directory.

    The payload lives only in ~/.tpu-agent-suite/packages/<name>@<version>/; the
    workspace holds configuration, never code. A missing central directory is
    a broken or absent installation and fails loudly rather than falling back
    to a guess. Without a package, returns the TPU agent suite home itself.

    A bare (unversioned) name resolves to the highest installed version --
    the version is an install detail agents do not track -- via
    latest_installed_package; that is a deterministic central-store lookup,
    not a fallback between locations.

    The `workspace` parameter is retained for signature compatibility; the
    resolution is workspace-independent by design.
    """
    del workspace  # placement is declared, not searched for
    if not package:
        return utils.get_tpu_agent_suite_home()

    pkg_base, pkg_version = utils.parse_package_tag(package)
    central_pkg = utils.get_package_storage_root(pkg_base, pkg_version)
    if central_pkg.is_dir():
        return central_pkg
    if not pkg_version:
        installed = latest_installed_package(pkg_base)
        if installed is not None:
            return installed
    raise TpuAgentSuiteError(
        f"package {package} is not installed in"
        f" {utils.get_tpu_agent_suite_home() / utils.PACKAGES_DIRNAME} --"
        " install the distribution first")


def get_environment(workspace: Path | str | None = None,
                    package: str = "") -> tuple[dict[str, Any], Path | None]:
    """Resolve environment config via the single ownership-keyed resolver."""
    try:
        return utils.resolve_environment(package, workspace)
    except ValueError as exc:
        raise TpuAgentSuiteError(
            f"environment.json is not valid JSON: {exc}. Fix it, or delete "
            "it and set compute up again.") from exc


def get_installed_run_context(
        package: str | Path = "",
        run_id: str = "",
        workspace: Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    """Retrieve and validate the package root, run root directory, and run metadata document."""
    if isinstance(package, Path) and isinstance(workspace, str):
        workspace, package = package, workspace
    elif isinstance(package, Path) and not workspace:
        package, workspace = "", package

    validate_run_id(run_id)
    package_root = get_installed_package_root(workspace, str(package))

    # Runs live only in centralized storage (~/.tpu-agent-suite/runs/<run_id>);
    # start_run never allocates them anywhere else.
    run_root = utils.get_run_root(run_id)

    run_document = load_json(run_root /
                             "run.json") if (run_root /
                                             "run.json").is_file() else {}
    if run_document.get("run_id") != run_id:
        raise TpuAgentSuiteError(
            f"run metadata does not match the requested run ID: {run_id}")

    # Validate package isolation: ensure requested package matches run metadata
    pkg_str = str(package).strip()
    if pkg_str:
        doc_pkg = str(run_document.get("package", "")).strip()
        if doc_pkg and doc_pkg != pkg_str and doc_pkg.split(
                "@")[0] != pkg_str.split("@")[0]:
            raise TpuAgentSuiteError(
                f"run '{run_id}' was created by package '{doc_pkg}', refusing access from '{pkg_str}'"
            )

    return package_root, run_root, run_document


def start_run(
    package: str | Path = "",
    workspace: Path | str | None = None,
    compute_profile: str | None = None,
) -> dict[str, Any]:
    """Create a collision-resistant run namespace inside central store ~/.tpu-agent-suite/runs/."""
    if isinstance(package, Path) and isinstance(workspace, str):
        workspace, package = package, workspace

    package_str = str(package) if package else "no-package"
    # A run without a package has no environment to consult; the "no-package"
    # placeholder is a run-document label, not a resolvable package.
    environment: dict[str, Any] = {}
    if package:
        environment, _ = get_environment(workspace, package_str)

    profile_name = (compute_profile or "").strip()
    if compute_profile is not None and not profile_name:
        raise compute.ComputeConfigError(
            "--profile must be a non-empty profile name.",
            error_code="INVALID_FIELD")
    if profile_name:
        # Refuse a name the run could never resolve, rather than recording it
        # and failing at the first compute command.
        if not package:
            raise compute.ComputeConfigError(
                "start-run --profile needs --package: a run without a package "
                "has no environment.json to look the profile up in.",
                error_code="MISSING_FIELD")
        profiles, _ = compute.normalize_environment_profiles(environment)
        if profile_name not in profiles:
            raise compute.unknown_profile_error(profile_name, "--profile",
                                                profiles)

    # Always allocate run in central storage (~/.tpu-agent-suite/runs/)
    runs_root = utils.get_runs_root()
    runs_root.mkdir(parents=True, exist_ok=True)

    for _ in range(10):
        run_id = generate_run_id()
        run_root = utils.get_run_root(run_id)
        try:
            run_root.mkdir()
        except FileExistsError:
            continue

        (run_root / "artifacts").mkdir()
        document: dict[str, Any] = {
            "created_at": utc_now_iso(),
            # 0 means "no environment recorded", which is what a packageless
            # run has. A fresh install writes 1 and every setup or config
            # write bumps it, so 0 can never be a real revision -- whereas 1
            # is the most common one, and stamping it here would assert that
            # artifacts from a run with no configuration are comparable with
            # every artifact produced under the first real one.
            "environment_revision": environment.get("revision", 0),
            "version": TPU_AGENT_SUITE_VERSION,
            "package": package_str,
            "run_id": run_id,
        }
        if profile_name:
            document["compute_profile"] = profile_name
        write_json(run_root / "run.json", document)
        # The active run is a workspace's notion; a run started without one
        # (tests, ad-hoc tooling) has nowhere to be the active run of.
        if workspace:
            utils.record_active_run(run_id, workspace)
        return document

    raise TpuAgentSuiteError("could not allocate a unique run ID")


def _format_run_context(run_id: str) -> str:
    """Render the run-namespace context injected into the model conversation."""
    # Resolved rather than the home-anchored spelling compiled prose uses:
    # this string is built at runtime on the machine that owns the run, so it
    # can name where that run actually is. Spelling it `~/.tpu-agent-suite` would
    # point the model to the wrong directory whenever TPU_AGENT_SUITE_HOME moves
    # the store.
    try:
        artifacts_dir: Any = utils.get_run_artifacts_dir(run_id)
    except TpuAgentSuiteError:
        # A run_id we cannot parse still owes the caller its context, and the
        # portable spelling is right whenever the store has not been moved.
        artifacts_dir = f"{utils.RUNS_HOME_SPELLING}/{run_id}/artifacts"
    return (
        f"TPU agent suite run namespace: {run_id}\n"
        f"Artifacts: {artifacts_dir}/\n"
        f"Use this exact run_id for all subagent delegations and artifact creation."
    )


def validate_json_file(schema_path: Path, instance_path: Path) -> list[str]:
    """Validate a JSON instance against a JSON schema file."""
    instance = load_json(instance_path)
    schema = load_json(schema_path)
    errors = validate_instance(instance, schema)
    # Both rules below are conditional on the status a result claims, and the
    # schema validator has no conditional keywords, so they are enforced here:
    # a `completed` result must carry an artifact (only where the schema
    # declares `artifacts`), and each status in `status_requires` must carry
    # its fields. Demanding a success-shaped payload from a failure would push
    # agents toward reporting success.
    if not errors and "artifacts" in schema.get("properties", {}):
        errors = check_completed_has_artifact(instance)
    if not errors:
        errors = check_status_payload(schema, instance, schema_path.name)
    return errors


def resolve_workspace_root(
    workspace: Path | str | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> Path:
    """Determine the active workspace root directory across harnesses and contexts."""
    if workspace:
        return Path(workspace).resolve()

    # 1. Extract active workspace path directly from inbound harness payload if available
    if raw_payload and isinstance(raw_payload, dict):
        # Dynamically aggregate declared cwd/workspace candidate fields from event-mappings.json
        candidate_fields = ["workspace"]
        try:
            for h_info in get_event_mappings().get("SessionStart",
                                                   {}).get("harnesses",
                                                           {}).values():
                for field in h_info.get("fields", {}).get("cwd", []):
                    if field not in candidate_fields:
                        candidate_fields.append(field)
        except Exception:
            pass

        for key in candidate_fields:
            val = raw_payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], str):
                val = val[0]
            if isinstance(val, str) and val.strip():
                p = Path(val).resolve()
                if p.is_dir():
                    return p

    # 2. Fallback to process cwd and parent directory marker search
    cur = Path.cwd().resolve()
    if cur.name in (".agent", ".cursor", ".claude"):
        return cur.parent
    # The session-start hook records the active run where the harness says the
    # session is, so the nearest ancestor holding that file is the workspace.
    # A `.tpu-agent-suite` directory without one is not a marker: the installer
    # and the logging hooks create that directory in other places too.
    for p in [cur, *cur.parents]:
        if (p / utils.TPU_AGENT_SUITE_DIRNAME /
                utils.ACTIVE_RUN_FILE).is_file():
            return p
    return cur


def extract_last_transcript_message(transcript_path: str) -> str | None:
    """Extract assistant model output from a session transcript.

    Parses the JSONL transcript backwards to find the last MODEL entry containing
    content, enabling deliverable extraction when model output is not directly
    embedded in the hook event payload (used for Antigravity SubagentStop).

    Args:
        transcript_path: Filesystem path to the transcript.jsonl file.

    Returns:
        The content string of the latest MODEL message, or None if not found.
    """
    if not transcript_path:
        return None
    p = Path(transcript_path)
    if not p.is_file():
        return None
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("source") == "MODEL" and data.get("content"):
                    return data["content"]
            except Exception:
                continue
    except Exception:
        pass
    return None


def get_google_harness_subagent_name(
    conversation_id: str,
    transcript_path: str = "",
    workspace: Path | None = None,
) -> str | None:
    """Detect if a conversation represents a subagent and resolve its identifier.

    Inspects native Google harness brain descriptors located at
    ~/.gemini/<brain_root>/brain/*/.system_generated/subagents/<cid>.json
    to determine if the conversation was spawned as a subagent, extracting its
    'typeName' or 'role'. Also detects isolated worktree subagent sessions.

    Descriptors are keyed by conversation id rather than by harness, so every
    brain root registered for the family is searched: the hook knows which
    harness invoked it, but not which one wrote a descriptor it is resolving.

    Args:
        conversation_id: Unique conversation identifier.
        transcript_path: Optional path to the session transcript.
        workspace: Optional workspace directory path.

    Returns:
        The subagent name/type identifier if identified, or None if root session.
    """
    if transcript_path and "/worktrees/subagent-" in transcript_path:
        return "subagent"
    if not conversation_id:
        return None

    # Google harness native subagent descriptor JSON
    # Pattern: <brain_dir>/<parent_cid>/.system_generated/subagents/<conversation_id>.json
    for brain_root in sorted(set(utils.GOOGLE_HARNESS_BRAIN_DIRS.values())):
        brain_dir = Path.home() / ".gemini" / brain_root / "brain"
        if brain_dir.is_dir():
            matches = list(
                brain_dir.glob(
                    f"*/.system_generated/subagents/{conversation_id}.json"))
            if matches:
                try:
                    desc_data = json.loads(
                        matches[0].read_text(encoding="utf-8"))
                    sub_desc = desc_data.get("subagentDescriptor") or {}
                    name = sub_desc.get("typeName") or sub_desc.get("role")
                    if name:
                        return name
                except Exception:
                    pass

    return None


def google_harness_post_processing(
    subagent: str,
    subagent_output: Any,
    workspace: Path,
    session_id: str = "",
    transcript_path: str = "",
) -> tuple[str, Any]:
    """Resolve subagent identifier and deliverable for Google harness Stop events.

    Since the Google harnesses emit a generic 'Stop' lifecycle event rather
    than a native 'SubagentStop', this helper resolves missing subagent metadata
    from brain disk descriptors and extracts the subagent deliverable text from
    the session transcript when not explicitly provided in the event payload.

    Args:
        subagent: Current subagent identifier extracted from payload fields.
        subagent_output: Current subagent deliverable extracted from payload fields.
        workspace: Path to the active workspace.
        session_id: Resolved conversation/session ID.
        transcript_path: Resolved path to transcript file.

    Returns:
        A tuple of (subagent, subagent_output) with resolved values.
    """
    if not subagent and (session_id or transcript_path):
        subagent = get_google_harness_subagent_name(
            session_id, transcript_path, workspace) or ""
    if subagent_output is None and transcript_path:
        subagent_output = extract_last_transcript_message(transcript_path)
    return subagent, subagent_output


def get_google_harness_parent_conversation_id(
        conversation_id: str) -> str | None:
    """Resolve parent conversation identifier for a Google harness subagent conversation.

    Descriptors are keyed by conversation id rather than by harness, so every
    brain root registered for the family is searched, as in
    `get_google_harness_subagent_name`.
    """
    if not conversation_id:
        return None
    for brain_root in sorted(set(utils.GOOGLE_HARNESS_BRAIN_DIRS.values())):
        brain_dir = Path.home() / ".gemini" / brain_root / "brain"
        if brain_dir.is_dir():
            matches = list(
                brain_dir.glob(
                    f"*/.system_generated/subagents/{conversation_id}.json"))
            if matches:
                return matches[0].parents[2].name
    return None


def _safe_path_component(value: str) -> str:
    """Reduce an identifier to characters that are safe inside a file name."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def get_session_file(session_id: str) -> Path:
    """Return path to session metadata file under ~/.tpu-agent-suite/sessions/.

    Builds the path without creating it: readers outnumber writers here, and
    write_json creates the parent when there is something to store.
    """
    return utils.get_sessions_root(
    ) / f"{_safe_path_component(session_id)}.json"


def get_hook_context_file(session_id: str, hook_name: str) -> Path:
    """Return path to a hook's cached SessionStart context for a session.

    Each hook owns a distinct file, so hooks dispatched concurrently as separate
    processes never read-modify-write a document they share with one another.
    """
    contexts_dir = (utils.get_sessions_root() /
                    f"{_safe_path_component(session_id)}.contexts")
    return contexts_dir / f"{_safe_path_component(hook_name)}.json"


def record_session_run(
    session_id: str,
    run_id: str,
    package: str = "",
) -> None:
    """Record an active session-to-run mapping under ~/.tpu-agent-suite/sessions/."""
    if not session_id or not run_id:
        return
    session_file = get_session_file(session_id)
    existing = load_json_doc(session_file)
    session_doc = {
        "created_at": existing.get("created_at") or utc_now_iso(),
        "package": package,
        "run_id": run_id,
        "session_id": session_id,
    }
    if session_doc == existing:
        return
    write_json(session_file, session_doc)


def record_session_hook_context(
    session_id: str,
    hook_name: str,
    context_text: str,
    package: str = "",
) -> None:
    """Record cached hook context under ~/.tpu-agent-suite/sessions/<id>.contexts/."""
    if not session_id or not hook_name or not context_text:
        return
    write_json(
        get_hook_context_file(session_id, hook_name), {
            "context": context_text,
            "hook_name": hook_name,
            "package": package,
            "recorded_at": utc_now_iso(),
        })


def get_pending_context_dir(session_id: str) -> Path:
    """Return the spool directory of hook context awaiting delivery to a session.

    Reached only on the Google harnesses (antigravity today): their PostToolUse
    response is a bare `{}`, so a hook that produced context has nowhere to put
    it and spools it here instead, and the PreInvocation call they emulate
    SessionStart with hands it over later. Nothing in these three functions
    enforces that -- the `is_google_harness` gates in `dispatch_hook` do -- so
    widening the spool is a change of caller, not of mechanism.

    Separate from the `.contexts` cache next door, which holds one overwritable
    value per hook and is re-read idempotently. This is a queue: entries are
    added by one hook, delivered by another, and retired once delivered enough
    times.
    """
    return (utils.get_sessions_root() /
            f"{_safe_path_component(session_id)}.pending")


def record_pending_hook_context(
    session_id: str,
    hook_name: str,
    context_text: str,
    package: str = "",
) -> None:
    """Spool hook context for delivery on a later event.

    For hooks whose own event has no field to answer with, which today means
    PostToolUse on the Google harnesses; see `get_pending_context_dir`. Each
    entry gets its own file: the hooks that write here run as separate
    processes and several tool calls can finish before the next model call, so
    a shared document would be a read-modify-write race.
    """
    if not session_id or not context_text:
        return
    pending_dir = get_pending_context_dir(session_id)
    pending_dir.mkdir(parents=True, exist_ok=True)

    # One timestamp for both the name and the field, so the order entries are
    # read back in is the order they record having been written.
    recorded_at = utc_now_iso()
    entry_name = f"{recorded_at}-{uuid.uuid4().hex[:8]}.json"
    write_json(
        pending_dir / _safe_path_component(entry_name), {
            "context": context_text,
            "hook_name": hook_name,
            "package": package,
            "recorded_at": recorded_at,
        })

    # Bound the spool. Trim after writing so concurrent writers self-correct
    # and pull the spool back down to the cap. The oldest go first: the newest
    # context entries are the ones a result is most likely still to need.
    existing = sorted(pending_dir.glob("*.json"))
    overflow = len(existing) - MAX_PENDING_CONTEXT_ENTRIES
    if overflow > 0:
        for stale in existing[:overflow]:
            stale.unlink(missing_ok=True)


def replay_pending_hook_context(session_id: str | None) -> str:
    """Return spooled context for a session.

    Called only from the emulated-SessionStart bypass on the Google harnesses;
    see `get_pending_context_dir` for why the spool exists at all.

    Active context entries remain in the spool across rounds and are injected on
    each model invocation until pushed out by newer entries (capped by
    `MAX_PENDING_CONTEXT_ENTRIES`).

    No parent-conversation fallback, unlike `resolve_session_hook_context`:
    the spool is private to the session that wrote it so a subagent does not
    pollute or read its parent's context.
    """
    if not session_id:
        return ""
    pending_dir = get_pending_context_dir(session_id)
    if not pending_dir.is_dir():
        return ""

    texts: list[str] = []
    for entry_file in sorted(pending_dir.glob("*.json")):
        entry = load_json_doc(entry_file)
        context_text = entry.get("context")
        if not context_text:
            entry_file.unlink(missing_ok=True)
            continue
        texts.append(str(context_text))
    return "\n\n".join(texts)


def _iter_session_ids(session_id: str) -> Iterator[str]:
    """Yield the session ID, then its parent conversation's, if one exists.

    Google harness subagents run as child conversations with their own IDs,
    so a lookup that misses locally falls back to the session that spawned them.
    """
    yield session_id
    parent_cid = get_google_harness_parent_conversation_id(session_id)
    if parent_cid:
        yield parent_cid


def resolve_session_hook_context(
    session_id: str | None,
    hook_name: str,
) -> str | None:
    """Resolve cached hook context text for a session and hook name."""
    if not session_id or not hook_name:
        return None
    for cid in _iter_session_ids(session_id):
        context_text = load_json_doc(get_hook_context_file(
            cid, hook_name)).get("context")
        if context_text:
            return str(context_text)
    return None


def resolve_session_run_id(session_id: str | None = None) -> str | None:
    """Resolve the active run_id associated with a session or child conversation."""
    if not session_id:
        return None
    for cid in _iter_session_ids(session_id):
        run_id = load_json_doc(get_session_file(cid)).get("run_id")
        if run_id:
            return str(run_id)
    return None


def _build_event_context(
    workspace: Path,
    package: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Construct canonical context dictionary enriched with active session and run IDs."""
    context: dict[str, Any] = {
        "workspace": str(workspace),
        "package": package,
    }
    if session_id:
        context["session_id"] = session_id
        active_run_id = resolve_session_run_id(session_id)
        if active_run_id:
            context["run_id"] = active_run_id
    return context


def _get_event_session_id(event_payload: dict[str, Any]) -> str:
    """Extract canonical session_id from event context."""
    return str(event_payload.get("context", {}).get("session_id") or "")


def _extract_canonical_field(
    data: Any,
    candidate_spec: str | list[str],
    container: dict[str, Any] | None = None,
) -> Any:
    """Extract first matching field from data or optional container using dot-path traversal.

    Array-valued fields are unwrapped to their first element: several harnesses
    report a scalar concept as a one-element list (Cursor's `workspace_roots`,
    Antigravity's `workspacePaths`). An *empty* array carries no value at all,
    so it is treated as absent rather than returned as a list -- otherwise the
    caller receives `[]` where every candidate promised a scalar.

    Args:
        data: Primary dictionary or object to search.
        candidate_spec: Field name or list of field names in priority order.
        container: Optional secondary container to search if not found in data.

    Returns:
        The extracted value, or None if no candidate matches.
    """
    candidates = candidate_spec if isinstance(candidate_spec,
                                              list) else [candidate_spec]
    for field in candidates:
        val = _extract_arg_by_path(data, field)
        if val is None and container:
            val = _extract_arg_by_path(container, field)
        if val is None:
            continue
        if isinstance(val, list):
            if not val:
                continue
            val = val[0]
        return val
    return None


def _normalize_subagent_stop_event(
    raw_payload: dict[str, Any],
    harness: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize harness-specific subagent termination payload into canonical SubagentStop event.

    Args:
        raw_payload: The raw hook payload from the harness.
        harness: Target harness name (e.g., 'claude-code', 'antigravity', 'cursor', 'codex').
        package: Active package name.
        workspace: Path to the active workspace.

    Returns:
        Canonical normalized SubagentStop event dictionary.

    Raises:
        TpuAgentSuiteError: If harness is unsupported or configuration is missing.
    """
    event_mappings = get_event_mappings()
    harness_config = harness_mapping(
        event_mappings.get("SubagentStop", {}).get("harnesses", {}), harness)
    if not harness_config:
        raise TpuAgentSuiteError(
            f"unsupported harness for subagent stop: '{harness}'")

    fields = harness_config.get("fields", {})

    # Harnesses like Claude Code and Codex wrap subagent parameters inside a nested
    # dictionary rather than at the top level of the hook payload.
    # For example, Claude Code sends:
    #   {
    #     "last_assistant_message": "Kernel benchmark completed: 1.4x speedup",
    #     "tool_input": {
    #       "agent_type": "torch_coder",
    #       "task": "Optimize attention kernel"
    #     }
    #   }
    # In event-mappings.json, "container" is configured as ["tool_input"].
    # Here, container_dict extracts {"agent_type": "torch_coder", ...} so that:
    # 1. Candidate fields (e.g. subagent="agent_type") are resolved via container_dict fallback.
    # 2. Wrapped task arguments (e.g. "task") are unboxed and merged into normalized_args.
    container_dict: dict[str, Any] = {}
    for c_key in fields.get("container", []):
        val = raw_payload.get(c_key)
        if isinstance(val, dict):
            container_dict = val
            break

    subagent = str(
        _extract_canonical_field(
            raw_payload, fields.get("subagent", []), container=container_dict)
        or "")
    subagent_output = _extract_canonical_field(raw_payload,
                                               fields.get("output", []),
                                               container=container_dict)

    # Harness-specific disk/transcript fallbacks (e.g. Google harness emulated subagent stop)
    if is_google_harness(harness):
        cid = str(
            _extract_canonical_field(raw_payload,
                                     fields.get("session_id", []),
                                     container=container_dict) or "")
        tpath = str(
            _extract_canonical_field(raw_payload,
                                     fields.get("transcript_path", []),
                                     container=container_dict) or "")
        subagent, subagent_output = google_harness_post_processing(
            subagent=subagent,
            subagent_output=subagent_output,
            workspace=workspace,
            session_id=cid,
            transcript_path=tpath)

    # Assemble normalized arguments
    normalized_args: dict[str, Any] = {"subagent": subagent}
    if isinstance(raw_payload.get("args"), dict):
        normalized_args.update(raw_payload["args"])
    if container_dict:
        normalized_args.update(container_dict)
    for key, value in _passthrough_args(raw_payload,
                                        fields,
                                        skip=fields.get("container",
                                                        [])).items():
        normalized_args.setdefault(key, value)

    # Canonicalize stop/termination reason across harnesses using event-mappings.json
    stop_reason = _extract_canonical_field(raw_payload,
                                           fields.get("stop_reason", []),
                                           container=container_dict)
    if stop_reason is not None:
        normalized_args["stop_reason"] = str(stop_reason)

    # `fully_idle` separates "the model stopped talking" from "there is no
    # outstanding work". A harness that says so (Antigravity's `fullyIdle`)
    # lets a stop hook tell a turn boundary from a finished subagent without
    # reading its transcript. Absent means the harness did not say, and
    # silence must not be read as idle.
    fully_idle = _extract_idle_flag(raw_payload,
                                    fields,
                                    harness,
                                    "SubagentStop",
                                    container=container_dict)
    if fully_idle is not None:
        normalized_args["fully_idle"] = fully_idle

    session_id_val = str(
        _extract_canonical_field(raw_payload,
                                 fields.get("session_id", []),
                                 container=container_dict) or "")

    return _subagent_stop_event(subagent, subagent_output, normalized_args,
                                session_id_val, raw_payload, package,
                                workspace)


def _passthrough_args(
        raw_payload: dict[str, Any],
        fields: dict[str, Any],
        skip: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Raw payload keys a SubagentStop hook sees in `args` as sent.

    Everything the harness sent is kept beside the canonical fields, except
    what is already represented elsewhere in the event: the wrapper keys in
    `skip`, raw args, the event name, and the output (by its top-level key,
    e.g. 'toolCall.args' -> 'toolCall').
    """
    skip_keys = {"args", "event", "hook_event_name", *skip}
    skip_keys.update(spec.split(".")[0] for spec in fields.get("output", []))
    return {k: v for k, v in raw_payload.items() if k not in skip_keys}


def _subagent_stop_event(
    subagent: str,
    output: Any,
    args: dict[str, Any],
    session_id: str,
    raw_payload: dict[str, Any],
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """The canonical SubagentStop envelope, whichever road delivered it."""
    return {
        "event": "SubagentStop",
        "subagent": subagent,
        "tool": "delegate",
        "args": args,
        "output": output,
        "context": _build_event_context(workspace, package, session_id),
        "raw_input": raw_payload,
    }


def _match_handback(
    raw_payload: dict[str, Any],
    harness: str,
) -> dict[str, Any] | None:
    """The hand-back mapping `raw_payload` was fired for, if any.

    A hook registered for SubagentStop is also attached to the hand-back
    tool the mapping declares (see `utils.subagent_handback_mapping`), and
    both say `--event SubagentStop`. The hand-back's `--handback-event` is
    for a dispatch that fails before it gets here, and commands compiled
    before it lack it, so the payload decides. A tool event names its event
    and its tool. Both must match, so a PostToolUse on the same tool, or a
    PreToolUse on any other tool, is never mistaken for a delivery.
    """
    handback_mapping = subagent_handback_mapping(harness)
    if handback_mapping is None or not isinstance(raw_payload, dict):
        return None
    native_event = raw_payload.get("hook_event_name")
    if native_event and native_event != handback_mapping["event_name"]:
        return None
    tool_event = to_canonical_event(handback_mapping["event_name"], harness)
    tool_fields = event_mapping(tool_event, harness).get("fields", {})
    tool_name = _extract_canonical_field(raw_payload,
                                         tool_fields.get("tool", []))
    return (handback_mapping
            if tool_name == handback_mapping["matcher"] else None)


def _normalize_handback_event(
    raw_payload: dict[str, Any],
    handback_mapping: dict[str, Any],
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize a hand-back tool call into the canonical SubagentStop event.

    The result is the same envelope a native stop produces, so every hook
    body -- builtin or script -- reads `subagent` and `output` the same way
    on either road. `args.handback` marks the road taken: the reply has to
    be shaped for the tool event that actually fired, and a script may care.
    """
    fields = handback_mapping.get("fields", {})
    subagent = str(
        _extract_canonical_field(raw_payload, fields.get("subagent", []))
        or "")
    output = _extract_canonical_field(raw_payload, fields.get("output", []))
    session_id = str(
        _extract_canonical_field(raw_payload, fields.get("session_id", []))
        or "")
    args = {
        "subagent": subagent,
        "handback": True,
        "handback_tool": handback_mapping["matcher"],
        "handback_event": handback_mapping["event_name"],
    }
    for key, value in _passthrough_args(raw_payload, fields).items():
        args.setdefault(key, value)
    return _subagent_stop_event(subagent, output, args, session_id,
                                raw_payload, package, workspace)


def _is_handback_event(event_payload: dict[str, Any]) -> bool:
    """Whether a canonical SubagentStop arrived through a hand-back tool."""
    args = event_payload.get("args")
    return isinstance(args, dict) and args.get("handback") is True


def _jsonl_lines(path: str) -> list[str]:
    """The non-empty lines of a JSONL file, unparsed; unreadable is empty."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in map(str.strip, text.splitlines()) if line]


def _parse_record(line: str) -> Any:
    """The JSON value on a JSONL line, or None when the line is malformed."""
    try:
        return json.loads(line)
    except ValueError:
        return None


def _iter_nested_dicts(node: Any) -> Iterator[dict[str, Any]]:
    """Yield `node` and every dict nested anywhere inside it.

    Transcript lines wrap their records at depths that differ by harness
    version (`message.content[]`, ...). Walking the whole line keeps the
    reader indifferent to where the records sit.
    """
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_nested_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_nested_dicts(item)


def _is_call_to(node: dict[str, Any], tool: str) -> bool:
    """Whether a transcript record is a call to `tool` that names its id."""
    return (node.get("type") == "tool_use" and node.get("name") == tool
            and isinstance(node.get("id"), str))


def _current_run_lines(
    lines: list[str],
    prompt_id: str,
    prompt_id_key: str,
) -> list[str] | None:
    """The transcript lines of the run `prompt_id` names, or None.

    A resumed subagent appends its new run to the same transcript, and every
    record the harness writes for a run's prompts carries that run's prompt
    id, so the run starts at the first record bearing it. Without an id or a
    key, or when no record bears the id, the run cannot be told apart and
    None is returned. The whole transcript is not a safe stand-in: an earlier
    run's hand-back in it would exempt a later run that delivers natively.

    Only lines holding the id as text are parsed. The id is an ASCII UUID,
    which JSON writes verbatim, so a line without it cannot bear it; the
    parse then confirms it sits under `prompt_id_key` rather than in prose.
    """
    if prompt_id and prompt_id_key:
        for index, line in enumerate(lines):
            if prompt_id not in line:
                continue
            record = _parse_record(line)
            if isinstance(record,
                          dict) and record.get(prompt_id_key) == prompt_id:
                return lines[index:]
    return None


def _is_accepted_result(node: dict[str, Any], call_ids: set[str]) -> bool:
    """Whether a transcript record is a non-error result of one of `call_ids`."""
    return (node.get("type") == "tool_result"
            and node.get("tool_use_id") in call_ids
            and node.get("is_error") is not True)


def _handback_delivered(
    transcript_path: str,
    tool: str,
    prompt_id: str = "",
    prompt_id_key: str = "",
) -> bool:
    """Whether the current run delivered through the hand-back tool.

    Delivered: a call whose result is not an error. Both a hook's deny and
    the harness's own refusal come back as error results, and neither
    reaches the caller. An unreadable transcript proves nothing and answers
    no, which keeps the native stop as the delivery.

    Only the run being stopped counts (see `_current_run_lines`): a subagent
    that handed back in auto mode and is then resumed in default mode
    delivers the resumed run's plain final message, which must still be
    checked. A run that cannot be located answers no as well, and says so
    on stderr: a renamed id field would otherwise turn every closing remark
    after a hand-back into a check with nothing in the logs to explain it.

    This runs on native stops, once per hook, so only lines holding the
    tool name, or a call's id, as text are parsed and walked; both are ASCII
    and appear verbatim in any line holding them. The text match is a
    filter, not the answer: the name also turns up in prose (a system prompt
    quoting a git log, say), which the structural check rejects.
    """
    if not transcript_path or not tool:
        return False
    all_lines = _jsonl_lines(transcript_path)
    if not all_lines:
        return False
    lines = _current_run_lines(all_lines, prompt_id, prompt_id_key)
    if lines is None:
        print(
            f"runtime: cannot locate the stopping run in {transcript_path} "
            f"(prompt id {prompt_id or 'missing'!r}, transcript key "
            f"{prompt_id_key or 'missing'!r}); checking the native stop.",
            file=sys.stderr)
        return False
    call_ids = {
        node["id"]
        for line in lines if tool in line
        for node in _iter_nested_dicts(_parse_record(line))
        if _is_call_to(node, tool)
    }
    return any(
        _is_accepted_result(node, call_ids) for line in lines
        if any(call_id in line for call_id in call_ids)
        for node in _iter_nested_dicts(_parse_record(line)))


def _handback_offered(
    transcript_path: str,
    tool: str,
    offer: Any = None,
    prompt_id: str = "",
    prompt_id_key: str = "",
) -> bool | None:
    """Whether the current run was given the hand-back tool, or None if unknown.

    `offer` is the mapping's `transcript_offer`: how the harness records that
    it gave a run the tool, as a record whose `flag` key is true and whose
    line holds `text`. Claude Code announces the tool in a meta user record
    (`isMeta`) at the start of the run ("Your final report is delivered
    through SubagentHandback ..."), written before the subagent's first turn
    and so before any stop. A run that calls the tool has it too. Neither
    record exists in a run that was not given it: Claude Code 2.1.241 gives
    it to no subagent, in auto mode or out of it, headless or interactive,
    and delivers the plain final message instead.

    None means the harness declares no `offer`, the transcript could not be
    read, or the run could not be located, which proves nothing either way.

    The offer is matched on its wording, not on the tool name alone: Claude
    Code also writes a meta record naming the tool to withdraw it
    ("SubagentHandback is not available in this run"). Only lines holding the
    tool name as text are parsed, and a match must sit in a flagged record or
    a call, so the name turning up in prose does not count.
    """
    if not isinstance(offer, dict):
        return None
    flag = offer.get("flag")
    text = offer.get("text")
    if not transcript_path or not tool or not flag or not text:
        return None
    all_lines = _jsonl_lines(transcript_path)
    if not all_lines:
        return None
    lines = _current_run_lines(all_lines, prompt_id, prompt_id_key)
    if lines is None:
        return None
    for line in lines:
        if tool not in line:
            continue
        record = _parse_record(line)
        if (isinstance(record, dict) and record.get(flag) is True
                and text in line):
            return True
        if any(_is_call_to(node, tool) for node in _iter_nested_dicts(record)):
            return True
    return False


# Why a native SubagentStop is not held to the contract: the subagent
# delivers through the hand-back tool, which the run's transcript shows, or
# which the mode implies when the transcript cannot be read. The second is
# reported, because a run that was not given the tool goes unchecked then.
_HANDBACK_SKIP = ("subagent delivers through the hand-back tool; this native "
                  "stop is a turn boundary")
_HANDBACK_ASSUMED_SKIP = (
    "auto mode, but the transcript could not show whether the subagent was "
    "given the hand-back tool; this native stop is taken as a turn boundary "
    "and its result is not checked")


def _handback_skip_reason(
    event_payload: dict[str, Any],
    harness: str,
) -> str | None:
    """Why the stopping subagent can deliver only through a hand-back tool.

    Returns _HANDBACK_SKIP or _HANDBACK_ASSUMED_SKIP, or None when the native
    stop is the delivery and must be checked.

    A subagent given a hand-back tool can deliver only through it: the tool
    tells it that plain text at the end of its turn is not delivered, and
    Claude Code itself re-prompts a subagent that ends its turn without
    calling it. Every native stop of such a subagent is therefore a turn
    boundary around the delivery, never the delivery -- the stop before the
    call carries a draft the caller never sees, the stop after it a closing
    remark. Holding either to the `produces` contract checks the result a
    second time and, for the closing remark, blocks a subagent that already
    delivered on every retry. The hand-back hook is the one place the
    contract is enforced for it.

    Claude Code accepts the call only in auto mode (`requires_mode`), but
    being in that mode does not mean the run was given the tool: which
    versions offer it varies, and 2.1.241 offers it to no subagent. There the
    native stop is the delivery, and skipping it on the mode alone let every
    result through unchecked. So in auto mode the transcript decides (see
    `_handback_offered` and the mapping's `transcript_offer`): a run shown
    not to have the tool is checked. When the transcript cannot answer, or
    the harness declares no `transcript_offer`, the mode still decides, as
    before, and the reason says so. Checking instead would block a subagent
    that was given the tool and already delivered, on every retry, for a
    closing remark; an unchecked result is how every auto-mode stop behaved
    before the transcript was read, and the reason makes it visible.

    Outside that mode, or on a payload without the mode, a stop is checked
    unless the run already delivered through the tool: after a switch out of
    auto mode, the closing remark of a run that handed back is not its
    result. One case slips through: a run started outside auto mode and
    switched into it mid-run was never given the tool, so its plain final
    message goes unchecked.
    """
    handback_mapping = subagent_handback_mapping(harness)
    if handback_mapping is None:
        return None
    raw_input = event_payload.get("raw_input") or {}
    fields = event_mapping("SubagentStop", harness).get("fields", {})
    # Resolved through the mapping rather than read off `args`: the raw
    # payload's own `transcript_path` is the parent session's file, which
    # never holds the subagent's calls.
    transcript_path = str(
        _extract_canonical_field(raw_input, fields.get("transcript_path", []))
        or "")
    prompt_id = str(
        _extract_canonical_field(raw_input, fields.get("prompt_id", [])) or "")
    prompt_id_key = str(handback_mapping.get("transcript_prompt_id") or "")
    required_mode = handback_mapping.get("requires_mode")
    if required_mode and _extract_canonical_field(
            raw_input, fields.get("permission_mode", [])) == required_mode:
        offered = _handback_offered(transcript_path,
                                    handback_mapping["matcher"],
                                    handback_mapping.get("transcript_offer"),
                                    prompt_id=prompt_id,
                                    prompt_id_key=prompt_id_key)
        if offered is True:
            return _HANDBACK_SKIP
        if offered is None:
            return _HANDBACK_ASSUMED_SKIP
    if _handback_delivered(
            transcript_path,
            handback_mapping["matcher"],
            prompt_id=prompt_id,
            prompt_id_key=prompt_id_key,
    ):
        return _HANDBACK_SKIP
    return None


def _extract_arg_by_path(data: Any, path: str) -> Any:
    """Resolve a dot-separated path against data, unwrapping list containers automatically.

    Example:
        >>> _extract_arg_by_path({"Subagents": [{"TypeName": "torch_coder"}]}, "Subagents.TypeName")
        'torch_coder'
    """
    curr = data
    for part in path.split("."):
        if isinstance(curr, list) and curr:
            curr = curr[0]
        if isinstance(curr, dict):
            curr = curr.get(part)
        else:
            return None
        if curr is None:
            return None
    return curr


def _normalize_tool_event(
    raw_payload: dict[str, Any],
    harness: str,
    canonical_event: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize harness-specific PreToolUse or PostToolUse payload."""
    event_mappings = get_event_mappings()
    harness_config = harness_mapping(
        event_mappings.get(canonical_event, {}).get("harnesses", {}), harness)
    if not harness_config:
        raise TpuAgentSuiteError(
            f"unsupported harness for hook dispatch: '{harness}'")

    fields = harness_config.get("fields", {})

    raw_tool = str(
        _extract_canonical_field(raw_payload, fields.get("tool", [])) or "")
    raw_args_val = _extract_canonical_field(raw_payload,
                                            fields.get("args", []))
    raw_args: dict[str, Any] = raw_args_val if isinstance(raw_args_val,
                                                          dict) else {}
    raw_output = _extract_canonical_field(raw_payload,
                                          fields.get("output", []))

    tool_mappings = get_tool_mappings()
    normalized_tool = raw_tool
    normalized_args = dict(raw_args)

    # Reverse-lookup raw_tool in tool_mappings to find canonical tool name
    matched_canonical: str | None = None
    matched_harness_info: dict[str, Any] = {}
    for c_name, c_def in tool_mappings.items():
        h_info = harness_mapping(c_def.get("harnesses", {}), harness)
        tools_list = h_info.get("tools", [])
        matcher = h_info.get("matcher", "")
        if raw_tool in tools_list or (matcher and re.search(
                rf"^({matcher})$", raw_tool)):
            matched_canonical = c_name
            matched_harness_info = h_info
            break

    if matched_canonical:
        normalized_tool = matched_canonical
        for canonical_k, raw_spec in matched_harness_info.get("args",
                                                              {}).items():
            val = _extract_canonical_field(raw_args, raw_spec)
            if val is not None:
                if isinstance(val, str) and not val.strip():
                    continue
                normalized_args[canonical_k] = (str(val) if canonical_k
                                                in ("subagent",
                                                    "prompt") else val)

    session_id_val = str(
        _extract_canonical_field(raw_payload, fields.get("session_id", []))
        or "")

    subagent_val = normalized_args.get(
        "subagent") if normalized_tool == "delegate" else None

    event_dict: dict[str, Any] = {
        "event": canonical_event,
        "tool": normalized_tool,
        "subagent": subagent_val,
        "args": normalized_args,
        "output": raw_output if canonical_event == "PostToolUse" else None,
        "context": _build_event_context(workspace, package, session_id_val),
        "raw_input": raw_payload,
    }

    # Resolve execution success. Each harness declares the field that reports a
    # failure, and only that field -- see the "error" mapping in
    # event-mappings.json. An empty candidate list is not a gap in the mapping:
    # claude-code and cursor route failures to a separate event
    # (PostToolUseFailure, postToolUseFailure), so a PostToolUse that reaches us
    # from them succeeded by definition. Truthiness is the whole test, because
    # the harnesses that do report failures here send either a message string
    # that is empty when the call succeeded (antigravity's `error`) or a boolean
    # flag (the MCP call result's `isError`, which codex forwards verbatim).
    if canonical_event == "PostToolUse":
        event_dict["success"] = not _extract_canonical_field(
            raw_payload, fields.get("error", []))

    return event_dict


def latest_installed_package(package: str) -> Path | None:
    """Find the highest installed version of `package` under central storage.

    Packages install to `~/.tpu-agent-suite/packages/<name>@<version>/`, but callers
    routinely name a package without its version -- the version is an install
    detail, not something an agent tracks. An unversioned lookup used to miss
    the versioned directory entirely and fall through to a workspace path that
    holds only the subset of the package installed there, so anything living
    solely in central storage looked absent. Schemas are the case that bites:
    the package resolves, the schema underneath it does not, and the error
    blames the schema.
    """
    if not package:
        return None
    packages_root = utils.get_tpu_agent_suite_home() / "packages"
    if not packages_root.is_dir():
        return None

    candidates: list[tuple[utils.SemVer, Path]] = []
    for entry in packages_root.iterdir():
        if not entry.is_dir():
            continue
        name, _, version = entry.name.partition("@")
        if name != package or not version:
            continue
        try:
            candidates.append((utils.SemVer.parse(version), entry))
        except (ValueError, TpuAgentSuiteError):
            # An unparseable version is someone else's directory, not a
            # candidate. Skipping beats guessing at an ordering for it.
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def check_completed_has_artifact(instance: Any) -> list[str]:
    """Require a completed result to point at something durable.

    An agent's findings are supposed to land in an artifact and travel back as
    a reference, so the caller gets a reference rather than a payload and the
    work survives the conversation that produced it. Nothing enforced that, and
    the cheap path is to answer in prose: a recorded session spent 33 minutes
    measuring, wrote no files at all, and lost every number to a context
    truncation four times over.

    This lives here rather than in the schema because the schema validator
    supports no conditional keywords, and the rule is conditional: only a
    `completed` result must carry an artifact. A failure has nothing to show,
    and demanding one would push agents toward reporting success.
    """
    if not isinstance(instance, dict):
        return []
    if instance.get("status") != "completed":
        return []
    artifacts = instance.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return [
            "$: a completed result must carry at least one artifact;"
            " write the output under the run's artifacts directory"
        ]
    return []


def check_status_payload(schema: dict[str, Any],
                         instance: Any,
                         schema_name: str = "") -> list[str]:
    """Require a result to carry the fields the status it claims owes.

    The mirror of `check_completed_has_artifact`: that one stops a successful
    result from being prose, this one stops it from being empty. A stage whose
    work did not reach an answer reports the status for that and a
    `failure_reason`, rather than omitting the numbers and still claiming the
    outcome.

    The rule comes from the schema's own `status_requires`, keyed on the status
    value it applies to. It cannot sit in `required`, which would demand the
    fields of every outcome including the failures -- the very thing that made
    the stage schemas describe success and nothing else -- and the validator
    carries no `if`/`then` to express the condition.

    Keying on the status value rather than on a fixed success word is what lets
    a package name its own outcomes: `optimized`, `implemented` and `planned` say
    what a stage actually produced. Nothing here knows which statuses exist, so
    nothing here has to be edited when a package picks different ones.

    A listed field must be present *and* non-empty. An empty artifacts array or
    an empty summary satisfies presence while carrying nothing, which is the
    same failure this exists to catch.

    `schema_name` only labels the error; a schema that declares the keyword
    carries its rule wherever it is loaded, so both callers get it. That
    matters for the SubagentStop hook, which holds a loaded schema and no
    filename: `runtime.py validate` is a call an agent chooses to make, while
    the hook fires on every delegation whether or not it wants it.
    """
    if not isinstance(schema, dict) or not isinstance(instance, dict):
        return []
    rules = schema.get("status_requires")
    if not isinstance(rules, dict):
        return []
    expected = rules.get(str(instance.get("status")))
    if not expected or not isinstance(expected, list):
        return []

    missing = [
        field for field in expected
        if field not in instance or instance[field] in (None, "", [], {})
    ]
    if missing:
        label = str(schema.get("$id") or schema_name or "result")
        status = str(instance.get("status"))
        return [
            f"$: a {label} reporting '{status}' must carry"
            f" {', '.join(missing)}; report the status for work that did not"
            " reach an answer, with a failure_reason, rather than claiming"
            f" '{status}' having measured nothing"
        ]
    return []


def _extract_idle_flag(
    raw_payload: dict[str, Any],
    fields: dict[str, Any],
    harness: str,
    event_name: str,
    container: dict[str, Any] | None = None,
) -> bool | None:
    """Read the harness's idle flag, saying so on stderr when it is missing.

    Every gate that consults the flag reads absence permissively, which is the
    safe default but a silent one. On a harness whose mapping declares the
    field, absence does not mean the harness stayed quiet, it means the
    contract moved: the Google harnesses encode these payloads with protojson,
    so a rename or a casing change would make every stop look like a finished
    one and fire stop hooks on turn boundaries with nothing in the logs to
    explain it. A harness that declares no idle field never had one, and says
    nothing.
    """
    candidates = fields.get("fully_idle", [])
    if not candidates:
        return None

    value = _extract_canonical_field(raw_payload,
                                     candidates,
                                     container=container)
    if value is None:
        print(
            f"runtime: {harness} {event_name} payload carries none of "
            f"{list(candidates)}; treating the agent as possibly busy. If the "
            "harness renamed the field, stop hooks will fire on turn "
            "boundaries.",
            file=sys.stderr)
        return None
    return bool(value)


def _normalize_termination_event(
    raw_payload: dict[str, Any],
    harness: str,
    package: str,
    workspace: Path,
    event_name: str = "Stop",
) -> dict[str, Any]:
    """Normalize a root-loop termination into the canonical `Stop` or `SessionEnd` event.

    There is no tool call to describe. What a Stop handler needs is why the loop
    ended and how many times it has already been resumed, so it can tell a
    transient refusal worth retrying from a real failure, and can stop itself
    running forever.
    """
    fields = harness_mapping(
        utils.get_event_mappings().get(event_name, {}).get("harnesses", {}),
        harness).get("fields", {})

    def first(keys: list[str], default: Any = "") -> Any:
        for key in keys:
            if raw_payload.get(key) is not None:
                return raw_payload[key]
        return default

    cid = str(
        first(fields.get("conversation_id", []), "")
        or first(fields.get("session_id", []), ""))
    context = _build_event_context(workspace, package, cid)
    context["conversation_id"] = cid
    context["harness"] = harness

    fully_idle = _extract_idle_flag(raw_payload, fields, harness, event_name)
    return {
        "event": event_name,
        "tool": "",
        "args": {},
        "output": None,
        "raw": raw_payload,
        "termination": {
            "reason": str(first(fields.get("stop_reason", []))),
            "error": str(first(fields.get("error", []))),
            "execution_num":
            int(first(fields.get("execution_num", []), 0) or 0),
            # None when the harness did not say: a reader that needs to
            # know must not mistake silence for idle.
            "fully_idle": fully_idle,
        },
        "context": context,
    }


def _normalize_session_start_event(
    raw_payload: dict[str, Any],
    harness: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize harness-specific session start payload into canonical SessionStart event.

    Args:
        raw_payload: The raw hook payload from the harness.
        harness: Target harness name (e.g., 'claude-code', 'antigravity', 'cursor', 'codex').
        package: Active package name.
        workspace: Path to the active workspace.

    Returns:
        Canonical normalized SessionStart event dictionary.

    Raises:
        TpuAgentSuiteError: If harness is unsupported or configuration is missing.
    """
    event_mappings = get_event_mappings()
    harness_config = harness_mapping(
        event_mappings.get("SessionStart", {}).get("harnesses", {}), harness)
    if not harness_config:
        raise TpuAgentSuiteError(
            f"unsupported harness for session start: '{harness}'")

    fields = harness_config.get("fields", {})

    normalized_args: dict[str, Any] = {
        "session_id": "",
        "source": "startup",
        "cwd": str(workspace),
    }

    # Dynamically extract all canonical fields defined for this harness in event-mappings.json
    for canonical_key, candidate_fields in fields.items():
        val = _extract_canonical_field(raw_payload, candidate_fields)
        if val is not None:
            normalized_args[canonical_key] = val

    # These three name a single scalar, but harnesses disagree on how they
    # carry it: Cursor reports the working directory as `workspace_roots`, an
    # array. `_extract_canonical_field` unwraps the array, yet nothing forces
    # what it yields to be a string -- so coerce here rather than leave hooks
    # to discover the type varies by harness. An empty result is no answer at
    # all and must not displace the workspace fallback.
    for scalar_key in ("cwd", "session_id", "transcript_path"):
        if scalar_key not in normalized_args:
            continue
        coerced = str(normalized_args[scalar_key] or "")
        if scalar_key == "cwd" and not coerced:
            coerced = str(workspace)
        normalized_args[scalar_key] = coerced

    subagent: str | None = None
    if is_google_harness(harness):
        cid = str(normalized_args.get("session_id") or "")
        trans_path = str(normalized_args.get("transcript_path") or "")
        subagent = get_google_harness_subagent_name(cid, trans_path, workspace)

    session_id_val = str(normalized_args.get("session_id") or "")

    return {
        "event": "SessionStart",
        "tool": None,
        "subagent": subagent,
        "args": normalized_args,
        "output": None,
        "context": _build_event_context(workspace, package, session_id_val),
        "raw_input": raw_payload,
    }


def normalize_inbound_event(
    raw_payload: dict[str, Any],
    harness: str,
    event: str,
    package: str,
    workspace: Path,
) -> dict[str, Any]:
    """Normalize harness-specific stdin payload into canonical Hook Input format."""
    # A SubagentStop hook is also attached to the harness's hand-back tool,
    # if it has one, and fires there with that tool event's payload.
    if event == "SubagentStop":
        handback_mapping = _match_handback(raw_payload, harness)
        if handback_mapping is not None:
            return _normalize_handback_event(raw_payload, handback_mapping,
                                             package, workspace)
    # `Stop` is both a canonical event and the native name the Google
    # harnesses use for a subagent finishing, so the string alone cannot say
    # which is meant. Only the payload can: a subagent stop carries its role or
    # type name, a root-loop stop does not. An explicit `SubagentStop` is a
    # request rather than a guess and is honoured as given. `SessionEnd` has
    # no native event on those harnesses either and is emulated on the same
    # `Stop`, so a hook registered for it sees every subagent stop as well
    # and has to be told which ones those are.
    if event == "Stop" or (event == "SessionEnd"
                           and is_google_harness(harness)):
        subagent_fields = harness_mapping(
            utils.get_event_mappings().get("SubagentStop",
                                           {}).get("harnesses", {}),
            harness).get("fields", {})
        container: dict[str, Any] = {}
        for key in subagent_fields.get("container", []):
            value = raw_payload.get(key)
            if isinstance(value, dict):
                container = value
                break
        has_subagent = any(
            raw_payload.get(field) or container.get(field)
            for field in subagent_fields.get("subagent", []))
        if has_subagent:
            canonical_event = "SubagentStop"
        else:
            canonical_event = "SessionEnd" if event == "SessionEnd" else "Stop"
    elif event == "Stop":
        canonical_event = "Stop"
    else:
        canonical_event = to_canonical_event(event, harness)

    if canonical_event in ("Stop", "SessionEnd"):
        return _normalize_termination_event(raw_payload,
                                            harness,
                                            package,
                                            workspace,
                                            event_name=canonical_event)
    if canonical_event == "SubagentStop":
        return _normalize_subagent_stop_event(raw_payload, harness, package,
                                              workspace)
    if canonical_event == "SessionStart":
        return _normalize_session_start_event(raw_payload, harness, package,
                                              workspace)
    if canonical_event in ("PreToolUse", "PostToolUse"):
        return _normalize_tool_event(raw_payload, harness, canonical_event,
                                     package, workspace)
    raise TpuAgentSuiteError(
        f"unsupported canonical event: '{canonical_event}'")


def translate_canonical_output(
    canonical_result: dict[str, Any],
    harness: str,
    event: str,
    hook_name: str,
) -> dict[str, Any]:
    """Translate canonical hook output into target harness response format."""
    decision = canonical_result.get("decision", "allow")
    reason = canonical_result.get("reason", "")
    message = canonical_result.get("message")
    context_text = canonical_result.get("context")
    followup_msg = canonical_result.get("followup_message")
    updated_args = canonical_result.get("updated_args")
    run_id = canonical_result.get("run_id")

    if run_id and not context_text:
        context_text = _format_run_context(run_id)

    is_deny = decision == "deny"
    effective_reason = followup_msg or reason or f"Denied by hook {hook_name}"
    if event not in ("Stop", "SubagentStop", "SessionEnd"):
        event = to_canonical_event(event, harness)

    if harness == "claude-code":
        response: dict[str, Any] = {}
        if message:
            response["systemMessage"] = message

        if event == "SessionEnd":
            # The session is over; there is no loop left to block or resume,
            # so the only thing worth returning is the message.
            return response
        elif event == "SessionStart":
            if is_deny:
                response["decision"] = "block"
                response["reason"] = effective_reason
            h_out = {"hookEventName": "SessionStart"}
            if context_text:
                h_out["additionalContext"] = context_text
            response["hookSpecificOutput"] = h_out
            return response

        elif event == "SubagentStop":
            if is_deny:
                response["decision"] = "block"
                response["reason"] = effective_reason
            h_out = {"hookEventName": "SubagentStop"}
            if context_text:
                h_out["additionalContext"] = context_text
            response["hookSpecificOutput"] = h_out
            return response

        elif event == "Stop":
            if decision in ("retry", "deny"):
                response["decision"] = "block"
                response["reason"] = effective_reason
            h_out = {"hookEventName": "Stop"}
            if context_text:
                h_out["additionalContext"] = context_text
            response["hookSpecificOutput"] = h_out
            return response

        elif event == "PreToolUse":
            h_out: dict[str, Any] = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow" if not is_deny else "deny",
            }
            if is_deny and reason:
                h_out["permissionDecisionReason"] = reason
            if updated_args is not None:
                h_out["updatedInput"] = updated_args
            response["hookSpecificOutput"] = h_out
        elif event == "PostToolUse":
            if is_deny:
                response["decision"] = "block"
                if reason:
                    response["reason"] = reason
            h_out = {"hookEventName": "PostToolUse"}
            if context_text:
                h_out["additionalContext"] = context_text
            response["hookSpecificOutput"] = h_out
        return response

    elif is_google_harness(harness):
        if message:
            print(f"[{hook_name}] {message}", file=sys.stderr)

        if event in ("Stop", "SessionEnd"):
            # "continue" is what re-enters the loop, and it is the whole point:
            # the conversation survives, so a resumed run picks up where it
            # stopped instead of repeating everything it had already done.
            if decision == "retry":
                return {
                    "decision": "continue",
                    "reason": reason or f"Resumed by hook {hook_name}",
                }
            if is_deny:
                return {
                    "decision": "deny",
                    "reason": reason or f"Denied by hook {hook_name}",
                }
            return {"decision": "stop"}

        if event == "SessionStart":
            inject_steps: list[dict[str, Any]] = []
            if context_text:
                inject_steps.append({"ephemeralMessage": context_text})
            if is_deny:
                inject_steps.append({
                    "ephemeralMessage":
                    f"Session Start Denied: {effective_reason}"
                })
            return {"injectSteps": inject_steps}

        if event == "SubagentStop":
            if is_deny:
                return {
                    "decision": "continue",
                    "reason": effective_reason,
                }
            return {"decision": "allow"}

        if event == "PreToolUse":
            if is_deny:
                return {
                    "decision": "deny",
                    "reason": reason or f"Denied by hook {hook_name}",
                }
            res: dict[str, Any] = {"decision": "allow"}
            if updated_args is not None:
                res["overwrite"] = updated_args
            return res
        else:
            if is_deny:
                return {
                    "decision": "deny",
                    "reason": reason or f"Denied by hook {hook_name}",
                }
            return {}

    elif harness == "cursor":
        if message:
            print(f"[{hook_name}] {message}", file=sys.stderr)

        if event == "SessionStart":
            if is_deny:
                return {
                    "decision": "cancel",
                    "reason": reason or f"Cancelled by hook {hook_name}",
                }
            response: dict[str, Any] = {}
            if context_text:
                response["additional_context"] = context_text
            return response

        if event == "SubagentStop":
            if is_deny:
                return {
                    "followup_message": effective_reason,
                }
            return {}

        if event == "Stop":
            if decision in ("retry", "deny"):
                return {
                    "decision": "cancel",
                    "reason": reason or f"Cancelled by hook {hook_name}",
                }
            return {}

        if is_deny:
            return {
                "decision": "cancel",
                "reason": reason or f"Cancelled by hook {hook_name}",
            }
        response = {}
        if event == "PostToolUse" and context_text:
            response["additional_context"] = context_text
        return response

    elif harness == "codex":
        if message:
            print(f"[{hook_name}] {message}", file=sys.stderr)

        if event == "SessionStart":
            if is_deny:
                return {
                    "decision": "block",
                    "reason": effective_reason,
                }
            response: dict[str, Any] = {"decision": "allow"}
            if context_text:
                response["additional_context"] = context_text
            return response

        if event == "SubagentStop":
            if is_deny:
                return {
                    "decision": "block",
                    "reason": effective_reason,
                }
            response: dict[str, Any] = {"decision": "allow"}
            if context_text:
                response["additional_context"] = context_text
            return response

        if event == "Stop":
            if decision in ("retry", "deny"):
                return {
                    "decision": "block",
                    "reason": effective_reason,
                }
            response = {"decision": "allow"}
            if context_text:
                response["additional_context"] = context_text
            return response

        if is_deny:
            return {
                "decision": "deny",
                "reason": reason or f"Denied by hook {hook_name}",
            }
        response = {"decision": "allow"}
        if event == "PostToolUse" and context_text:
            response["additional_context"] = context_text
        if updated_args is not None:
            response["tool_input"] = updated_args
        return response

    else:
        raise TpuAgentSuiteError(
            f"unsupported harness for hook translation: '{harness}'")


_HOOK_FAILURE_REPORT_LOCK = threading.Lock()

# How much of a failure's reason is kept: the end, where a traceback names
# its error. An `exit` reason is the hook's whole stderr, which can be long
# and can hold secrets; a denial puts it in the model's context, and the log
# is never rotated.
_HOOK_FAILURE_REASON_CHARS = 1000


def _hook_failure(
    kind: str,
    reason: str,
    *,
    hook_name: str,
    on_failure: str,
    event_payload: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    """The verdict for a hook that failed to reach one of its own.

    `kind` says how it failed: `timeout`, `exit` (non-zero), `non_json`
    (stdout that does not parse as JSON), `schema` (JSON that is not an
    object, or not a valid reply), `exception` (a
    builtin raised, or a script or command could not be run or its output
    read), `missing` (its script cannot be found), `dispatch` (dispatch
    itself raised: reading the payload, a builtin that does not exist,
    translating the reply), or `startup` (the dispatcher could not parse
    its own command line). `on_failure` decides
    what that becomes. A hook that did answer never comes here, so an
    explicit deny is honoured whatever the policy.

    Said on stderr, and recorded in the workspace's hook failure log either
    way: a hook that fails open is otherwise silent, since the harnesses keep
    a hook's stderr out of sight when it exits 0. Recording is best-effort;
    it must not turn a verdict into a crash.
    """
    context = event_payload.get("context") or {}
    if len(reason) > _HOOK_FAILURE_REASON_CHARS:
        reason = "..." + reason[-_HOOK_FAILURE_REASON_CHARS:]
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hook": hook_name,
        "event": event_payload.get("event", ""),
        "tool": event_payload.get("tool"),
        "subagent": event_payload.get("subagent"),
        "kind": kind,
        "reason": reason,
        "on_failure": on_failure,
        "decision": on_failure,
        "session_id": context.get("session_id", ""),
        "run_id": context.get("run_id", ""),
        "package": context.get("package", ""),
        # One log serves every workspace, so each record names its own.
        "workspace": str(workspace),
    }
    # A batched delegation settles its subagents on parallel threads: one
    # line and one record at a time, never interleaved.
    with _HOOK_FAILURE_REPORT_LOCK:
        print(
            f"runtime: [{hook_name}] hook failed ({kind}): {reason}; "
            f"on_failure={on_failure}",
            file=sys.stderr)
        try:
            log_file = utils.get_tpu_agent_suite_home(
            ) / utils.HOOK_FAILURE_LOG
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # One write of the whole line: the lock does not reach the other
            # hook processes the harness runs alongside this one, and a
            # buffered text file may flush a long line in pieces between
            # theirs.
            line = (json.dumps(record) + "\n").encode("utf-8")
            fd = os.open(log_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                         0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except OSError:
            pass

    if on_failure == "allow":
        return {"decision": "allow"}
    return {"decision": "deny", "reason": reason}


# How much of a hook's non-JSON stdout its failure reason quotes.
_HOOK_OUTPUT_SHOWN_CHARS = 500


def _run_hook_process(args: str | list[str], *, shell: bool, stdin_text: str,
                      cwd: Path,
                      timeout: float) -> subprocess.CompletedProcess[str]:
    """Run a hook's process; at the timeout, kill it and all it started.

    `subprocess.run` kills only the process it started: for a command hook
    that is `sh`, and a script's own children are no better off, so whatever
    they had started ran on, orphaned. The hook leads a process group of its
    own instead, and the timeout kills the group. Only the timeout does: a
    hook that exits on its own may leave work running on purpose.

    Raises `subprocess.TimeoutExpired` once the group is killed.
    """
    with subprocess.Popen(args,
                          shell=shell,
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True,
                          cwd=str(cwd),
                          start_new_session=True) as proc:
        try:
            stdout, stderr = proc.communicate(stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                # Gone already, or (macOS) a group left only with zombies.
                proc.kill()
            # Not `communicate`: a descendant that left the group (`setsid`)
            # survives the kill and holds the pipes open, and reading them
            # would wait on it.
            proc.wait()
            raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def execute_single_hook_action(
    event_payload: dict[str, Any],
    hook_name: str,
    workspace: Path,
    package: str,
    command: str | None = None,
    script_rel: str | None = None,
    timeout: float = DEFAULT_HOOK_TIMEOUT,
    script_args: list[str] | None = None,
    on_failure: str = DEFAULT_HOOK_ON_FAILURE,
) -> dict[str, Any]:
    """Execute a single hook action (command or script) and validate its output."""
    # What every failure below reports alongside its kind and reason.
    failure = {
        "hook_name": hook_name,
        "on_failure": on_failure,
        "event_payload": event_payload,
        "workspace": workspace,
    }

    if not command and not script_rel:
        raise TpuAgentSuiteError(
            "Either script_rel or command must be provided")

    try:
        if command:
            p = _run_hook_process(command,
                                  shell=True,
                                  stdin_text=json.dumps(event_payload),
                                  cwd=workspace,
                                  timeout=timeout)
        else:
            # Payload is central-only (~/.tpu-agent-suite/packages/<tag>/); a
            # pre-centralization workspace-payload install must be
            # re-installed.
            try:
                package_dir = get_installed_package_root(workspace, package)
                script_path = rooted(package_dir, script_rel,
                                     f"hook '{hook_name}' script")
            except TpuAgentSuiteError as exc:
                # An uninstalled, renamed, or version-bumped package resolves
                # to no root at all. This branch already denies when a script
                # cannot be located, and denying here keeps that contract while
                # handing the harness a parseable response rather than an empty
                # stdout and a non-zero exit.
                return _hook_failure("missing", str(exc), **failure)

            if not script_path.is_file():
                return _hook_failure(
                    "missing",
                    f"Hook '{hook_name}' script not found at {script_rel}",
                    **failure)

            cmd_args = [sys.executable, str(script_path), *(script_args or [])]
            p = _run_hook_process(cmd_args,
                                  shell=False,
                                  stdin_text=json.dumps(event_payload),
                                  cwd=workspace,
                                  timeout=timeout)
    except subprocess.TimeoutExpired:
        return _hook_failure(
            "timeout", f"Hook '{hook_name}' timed out after {timeout} seconds",
            **failure)
    except (OSError, ValueError) as exc:
        # The hook could not be run (a missing interpreter, a null byte in
        # its arguments), or its output not read (`UnicodeDecodeError`): left
        # to escape, dispatch would exit without a verdict, and the harness
        # would let the tool through whatever the policy. Nothing broader: a
        # bug in this code is not the hook failing, and dispatch's own guard
        # records it as `dispatch`.
        return _hook_failure(
            "exception", f"Hook '{hook_name}' could not be run: "
            f"{type(exc).__name__}: {exc}", **failure)

    if p.returncode != 0:
        err_r = p.stderr.strip(
        ) or f"Hook script exited with code {p.returncode}"
        return _hook_failure("exit", err_r, **failure)

    stdout_clean = p.stdout.strip()
    if not stdout_clean:
        return {"decision": "allow"}

    try:
        out_obj = json.loads(stdout_clean)
    except json.JSONDecodeError:
        # What it printed is the one clue to why, usually a stray print
        # before the JSON, so the start is shown. Capped, since the reason
        # reaches the model's context.
        shown = stdout_clean[:_HOOK_OUTPUT_SHOWN_CHARS]
        if len(stdout_clean) > _HOOK_OUTPUT_SHOWN_CHARS:
            shown += "..."
        return _hook_failure(
            "non_json",
            f"Hook '{hook_name}' emitted non-JSON output on stdout: {shown!r}",
            **failure)

    if not isinstance(out_obj, dict):
        return _hook_failure(
            "schema", f"Hook '{hook_name}' output must be a JSON object",
            **failure)

    if "decision" not in out_obj:
        out_obj["decision"] = "allow"

    try:
        schema = get_hook_output_schema()
        s_errors = validate_instance(out_obj, schema)
    except Exception as exc:
        print(f"runtime: Warning - schema validation skipped: {exc}",
              file=sys.stderr)
        s_errors = []
    if s_errors:
        return _hook_failure(
            "schema",
            f"Hook '{hook_name}' output violated hook-output.schema.json: " +
            "; ".join(s_errors), **failure)

    return out_obj


# Builtin threads left running past their timeout. Dispatch must not shut the
# interpreter down normally while one is alive: see `abandoned_hook_threads`.
_ABANDONED_HOOK_THREADS: list[threading.Thread] = []


def abandoned_hook_threads() -> list[threading.Thread]:
    """Builtin threads abandoned at their timeout that are still running.

    Interpreter shutdown flushes the standard streams, and it aborts the
    process (`Fatal Python error: _enter_buffered_busy`, SIGABRT) when a
    daemon thread holds one of their locks -- as a builtin printing to stderr
    does. The harness then reads a crash, not the verdict: to Claude Code a
    non-zero exit is a non-blocking error, so the tool runs. A dispatch that
    abandoned a thread flushes its verdict and leaves by `os._exit` instead.
    """
    return [t for t in _ABANDONED_HOOK_THREADS if t.is_alive()]


def _run_builtin_on_thread(handler: Callable[..., dict[str, Any]],
                           kwargs: dict[str, Any], outcome: dict[str,
                                                                 Any]) -> None:
    """The body of a builtin hook's thread: its verdict or its exception.

    Both land in `outcome`, for `run_builtin_hook_action` to read once the
    thread has joined; an exception must not die with the thread unseen.
    """
    try:
        outcome["result"] = handler(**kwargs)
    except Exception as exc:  # noqa: BLE001 - becomes the hook's failure
        outcome["error"] = exc


def run_builtin_hook_action(
    handler: Callable[..., dict[str, Any]],
    *,
    package: str,
    workspace: Path,
    event_payload: dict[str, Any],
    hook_name: str,
    timeout: float = DEFAULT_HOOK_TIMEOUT,
    on_failure: str = DEFAULT_HOOK_ON_FAILURE,
) -> dict[str, Any]:
    """Run a builtin hook in-process under the same timeout and policy.

    A script hook's subprocess can be killed at its timeout; a builtin runs in
    this process, so it runs on a daemon thread instead and is abandoned if it
    outlives the timeout. Dispatch then exits as soon as the verdict is
    written, ending the thread wherever it is, much as the harness killing
    dispatch would have. So a builtin may stop partway: its files are written
    whole (`write_json` renames into place, study events are single locked
    appends), but a `start-run` can leave a run folder without its
    `run.json`, a `reap-jobs` can leave jobs for the lease sweep to reap, and
    any child process it started (`kubectl`, `gcloud`) outlives dispatch.

    The timeout is not a hard limit: `join` cannot return while the thread
    holds the GIL in a long C call, such as a runaway regular expression.
    """
    outcome: dict[str, Any] = {}
    handler_kwargs = {
        "package": package,
        "workspace": workspace,
        "event_payload": event_payload,
        "hook_name": hook_name,
        # start-run fits its compute check inside this.
        "timeout": timeout,
    }
    worker = threading.Thread(target=_run_builtin_on_thread,
                              args=(handler, handler_kwargs, outcome),
                              name=f"hook-{hook_name}",
                              daemon=True)
    worker.start()
    worker.join(timeout)

    failure = {
        "hook_name": hook_name,
        "on_failure": on_failure,
        "event_payload": event_payload,
        "workspace": workspace,
    }
    if worker.is_alive():
        _ABANDONED_HOOK_THREADS.append(worker)
        return _hook_failure(
            "timeout", f"Hook '{hook_name}' timed out after {timeout} seconds",
            **failure)
    if "error" in outcome:
        exc = outcome["error"]
        return _hook_failure(
            "exception",
            f"Hook '{hook_name}' raised {type(exc).__name__}: {exc}",
            **failure)
    return outcome["result"]


def _run_hook_action(
    event_payload: dict[str, Any],
    *,
    hook_name: str,
    workspace: Path,
    package: str,
    timeout: float,
    on_failure: str,
    builtin_handler: Callable[..., dict[str, Any]] | None,
    command: str | None,
    script_rel: str | None,
    script_args: list[str] | None,
) -> dict[str, Any]:
    """One hook action for one event: a builtin, a script or a command."""
    if builtin_handler:
        return run_builtin_hook_action(builtin_handler,
                                       package=package,
                                       workspace=workspace,
                                       event_payload=event_payload,
                                       hook_name=hook_name,
                                       timeout=timeout,
                                       on_failure=on_failure)
    return execute_single_hook_action(event_payload=event_payload,
                                      hook_name=hook_name,
                                      workspace=workspace,
                                      package=package,
                                      command=command,
                                      script_rel=script_rel,
                                      timeout=timeout,
                                      script_args=script_args,
                                      on_failure=on_failure)


def _run_hook_action_per_subagent(sub_events: list[dict[str, Any]],
                                  **action: Any) -> list[dict[str, Any]]:
    """The hook's verdict on each subagent of a batched delegation, in order.

    The harness waits `timeout` plus headroom for the whole dispatch, so the
    subagents are checked at once, not in turn: in turn, N slow ones would
    need N timeouts and the harness would kill dispatch before any verdict
    reached it. Each run is bounded by `timeout` on its own, so the batch
    takes as long as its slowest subagent, and a hung one costs only itself.
    """
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(sub_events),
            thread_name_prefix=f"hook-{action['hook_name']}") as pool:
        futures = [
            pool.submit(_run_hook_action, sub_event, **action)
            for sub_event in sub_events
        ]
        return [future.result() for future in futures]


def _builtin_start_run(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    timeout: float = DEFAULT_HOOK_TIMEOUT,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler to allocate a new run namespace and emit its run_id."""
    session_id = ""
    if event_payload:
        session_id = _get_event_session_id(event_payload)

    # 1. Idempotency check: reuse existing run_id if this session already initialized one
    if session_id:
        existing_run_id = resolve_session_run_id(session_id)
        if existing_run_id:
            # A resumed session is a new harness process picking the run
            # back up: it takes the lease over before the watcher reaps.
            _lease_run(package, workspace, existing_run_id)
            return _allow_run(
                package, workspace, existing_run_id,
                f"Reusing TPU agent suite run namespace: {existing_run_id}",
                timeout)

    # 2. First-time initialization for this session
    run_doc = start_run(package=package, workspace=workspace)
    run_id = run_doc["run_id"]
    if session_id:
        record_session_run(session_id, run_id, package)
    _lease_run(package, workspace, run_id)

    return _allow_run(package, workspace, run_id,
                      f"Initialized TPU agent suite run namespace: {run_id}",
                      timeout)


def _allow_run(package: str, workspace: Path, run_id: str, message: str,
               timeout: float) -> dict[str, Any]:
    """Allow the session, adding the compute check to the run context."""
    result = {"decision": "allow", "run_id": run_id, "message": message}
    try:
        note = compute.cli.check_target_at_session_start(
            package, workspace, run_id, timeout)
    except Exception as exc:  # noqa: BLE001 - must never fail session start
        print(f"warning: compute check at session start failed: {exc}",
              file=sys.stderr)
        note = None
    if note:
        run_context = _format_run_context(run_id)
        result["context"] = f"{run_context}\n\n{note}"
        # Later turns replay only the run context, since the check result can go stale.
        result["replay_context"] = run_context
    return result


def _lease_run(package: str, workspace: Path, run_id: str) -> None:
    """Lease the run to the harness process, then sweep orphaned runs.

    The sweep starts only once the claim is written, so a resumed run is never
    mistaken for one whose session ended. It reaps what no watcher did: after
    a reboot, or a watcher killed along with its session. A run whose owner
    cannot be identified goes unleased; nothing here may fail the start.
    """
    try:
        compute.lease.claim(run_id, compute.lease.session_owner(), workspace,
                            package)
        compute.lease.spawn_sweep(run_id)
    except Exception as exc:  # noqa: BLE001 - reported, the reply still owed
        print(f"runtime: could not lease run {run_id} -- {exc}",
              file=sys.stderr)


# Termination reasons that mean a subagent loop reached its own end.
# A subagent completes its single delegated task when it finishes without
# calling further tools (`no_tool_call`) or hits a terminal step/hook.
_COMPLETED_SUBAGENT_TERMINATION_REASONS = frozenset({
    "no_tool_call",
    "terminal_step_type",
    "terminal_custom_hook",
})

# Termination reasons that mean an entire session reached its actual end.
# In Google harnesses, per-turn Stop events emulate SessionEnd. An ordinary
# turn ending where the model stops talking to wait for user input has
# reason `no_tool_call`. Treating `no_tool_call` as a session end would cause
# `session-end-reap` to cancel jobs submitted in turn 1 before turn 2 can poll
# them. Only explicit terminal reasons mark a session's true end.
#
# The same holds for `user_canceled`, `halted_step` and budget exhaustion:
# they end a turn, and the conversation can still resume and poll its jobs.
# Their jobs are reaped when the session is really over, by the run lease
# (`compute/lease.py`): the detached watcher once the harness process exits,
# or failing that the orphan sweep the next session runs at start.
_COMPLETED_SESSION_TERMINATION_REASONS = frozenset({
    "terminal_step_type",
    "terminal_custom_hook",
})


def _agent_loop_completed(reason: Any) -> bool:
    """Whether a termination reason says the agent loop ran to its own end.

    Antigravity declares the reasons prefixed (`TERMINATION_REASON_NO_TOOL_CALL`)
    but puts the bare name on the wire (`NO_TOOL_CALL`); accept either spelling
    so a harness that stops stripping the prefix does not silently flip every
    decision that rests on this.
    """
    text = str(reason or "").strip().lower()
    return (text.removeprefix("termination_reason_")
            in _COMPLETED_SUBAGENT_TERMINATION_REASONS)


def _session_completed(reason: Any) -> bool:
    """Whether a termination reason says the root session ran to its actual end."""
    text = str(reason or "").strip().lower()
    return (text.removeprefix("termination_reason_")
            in _COMPLETED_SESSION_TERMINATION_REASONS)


def _is_session_over(event_payload: dict[str, Any]) -> bool:
    """Whether a canonical SessionEnd event marks the session's actual end.

    Native SessionEnd (Claude Code, Cursor, Codex) always does. The Google
    harnesses have no such event; theirs rides the per-turn Stop, so the
    payload has to say the loop is fully idle *and* stopped for a reason that
    is not just the turn ending. An absent idle flag is a turn boundary, not
    a session end, on those harnesses.
    """
    if event_payload.get("event") != "SessionEnd":
        return False
    if not is_google_harness(
            event_payload.get("context", {}).get("harness", "")):
        return True
    term = event_payload.get("termination", {})
    return (term.get("fully_idle") is True
            and _session_completed(term.get("reason")))


def _builtin_reap_jobs(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler: cancel compute jobs left running by the ended run.

    An executor that died between `compute submit` and recording the job id
    leaves a workload holding chips that nothing knows about. Every job record
    carries its run id, and the session maps to its run, so the run's end is
    the one place all of them can be found. Anything that goes wrong here is
    reported, never raised: the session is ending and the harness still needs
    a well-formed reply.
    """
    event_payload = event_payload or {}
    if not _is_session_over(event_payload):
        return {"decision": "allow"}
    session_id = _get_event_session_id(event_payload)
    run_id = (resolve_session_run_id(session_id) if session_id else None)
    if not run_id:
        return {"decision": "allow"}

    def _resolve_driver(job_id: str) -> Any:
        return compute.resolve_driver(workspace=workspace,
                                      package=package,
                                      run_id=run_id,
                                      job_id=job_id)

    try:
        reaped, failed = compute.reap_jobs(run_id, _resolve_driver)
    except Exception as exc:  # noqa: BLE001 - reported, the reply still owed
        print(f"runtime: reap of run {run_id} failed -- {exc}",
              file=sys.stderr)
        return {"decision": "allow"}
    for job_id, why in failed.items():
        print(
            f"runtime: could not cancel job {job_id} of run {run_id} -- "
            f"{why}",
            file=sys.stderr)
    result: dict[str, Any] = {"decision": "allow"}
    if reaped:
        result["message"] = (f"Cancelled {len(reaped)} compute job(s) left "
                             f"running by run {run_id}: {', '.join(reaped)}")
    return result


def _resolve_agent_contract(
    package: str,
    workspace: Path,
    agent_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve (accepts_schema, produces_schema) definitions for an agent contract."""
    if not agent_name:
        return None, None

    # Every other failure here degrades to "no contract" rather than raising,
    # because a hook that cannot find a contract has nothing to enforce but
    # still owes the harness a reply. Package resolution has to join them: an
    # uninstalled, renamed, or version-bumped package otherwise escapes as a
    # TpuAgentSuiteError and leaves the dispatch with an empty stdout. Say so on
    # stderr, since silently enforcing nothing is its own kind of failure.
    try:
        package_root = get_installed_package_root(workspace, package)
    except TpuAgentSuiteError as exc:
        print(f"runtime: contract validation skipped -- {exc}",
              file=sys.stderr)
        return None, None

    build_file = package_root / "tpu-agent-suite-build.json"
    if not build_file.is_file():
        return None, None

    try:
        build_info = load_json(build_file)
    except Exception:
        return None, None

    contracts = build_info.get("contracts", {})
    contract = contracts.get(agent_name)
    if not contract and ":" in agent_name:
        contract = contracts.get(agent_name.split(":", 1)[1])
    if not contract:
        return None, None

    def _load_schema(rel_path: str) -> dict[str, Any] | None:
        if not rel_path:
            return None
        candidate = package_root / rel_path
        if candidate.is_file():
            try:
                return load_json(candidate)
            except Exception:
                return None
        return None

    accepts_schema = _load_schema(contract.get("accepts", ""))
    produces_schema = _load_schema(contract.get("produces", ""))
    return accepts_schema, produces_schema


def _subagent_was_cut_short(event_payload: dict[str, Any],
                            harness: str) -> bool:
    """Whether the harness says this subagent was stopped before it finished.

    Only the Google harnesses report a reason drawn from a vocabulary this
    runtime can read, Antigravity's `TERMINATION_REASON_*`. The others map
    `stop_reason` onto a field of their own -- Cursor's is the tool `status` --
    whose values would every one of them read as "not a completion" and excuse
    every subagent from its contract, so they are not asked. Silence is not
    evidence of a halt either: a Google payload that carries no reason at all
    still has to answer for its result.
    """
    if not is_google_harness(harness):
        return False

    args = event_payload.get("args")
    reason = args.get("stop_reason") if isinstance(args, dict) else None
    return bool(reason) and not _agent_loop_completed(reason)


# Why a SubagentStop carrying no subagent name is skipped. Named so the
# dispatcher can tell this reason from the others when deciding what to log.
_NO_SUBAGENT_NAME_SKIP = "no subagent name (internal helper or root session)"


def _subagent_stop_skip_reason(event_payload: dict[str, Any],
                               harness: str) -> str | None:
    """Why a SubagentStop does not report a finished subagent, or None if it does.

    The event fires whenever a non-root agent ends a turn, and ending a turn
    is not the same as being done. A subagent that delegates has to end its
    turn to let the child run, so a harness that launches delegates
    asynchronously fires this mid-flight; Claude Code fires it for its own
    untyped helper agents as well. Running the package's stop hooks on those
    holds a subagent to a contract before it has reported anything -- one
    recorded run was blocked four times before the real result arrived.

    A result can arrive by either of two roads (see
    `utils.subagent_handback_mapping`), and "once" holds across both. A
    hand-back call is the delivery itself -- the subagent declared its result
    final -- so only a name is required of it. Otherwise any one of four
    signals is enough to say no:

    * no subagent name -- an internal helper or a root session, neither of
      which has a contract to hold;
    * cut short -- cancelled, halted, or out of budget, so there is no result
      to hold it to, and blocking the stop would only re-prompt an agent that
      was just stopped ("zombie subagent");
    * `fully_idle` false -- the harness says work of this agent's own is still
      outstanding. Absent means it did not say, which stays permissive;
    * the subagent was given the hand-back tool, so this native stop is a
      turn boundary around its delivery, never the delivery itself. Auto
      mode alone does not show that (see `_handback_skip_reason`).

    Claude Code's `background_tasks` is not a signal. It lists every live
    agent of the session with no parent link, so a running entry is as likely
    a sibling's child as this agent's own. Nor is it needed: the install plan
    disables background tasks, so every delegate runs in the foreground and
    has returned before the agent that spawned it can stop.
    """
    if not str(event_payload.get("subagent") or "").strip():
        return _NO_SUBAGENT_NAME_SKIP

    if _is_handback_event(event_payload):
        return None

    if _subagent_was_cut_short(event_payload, harness):
        return "subagent was cut short before finishing"

    args = event_payload.get("args")
    if isinstance(args, dict) and args.get("fully_idle") is False:
        return "subagent still has work outstanding (not fully idle)"

    handback_skip = _handback_skip_reason(event_payload, harness)
    if handback_skip:
        return handback_skip

    return None


def _delegate_routing_keys() -> set[str]:
    """Argument names that address a delegation rather than form its payload.

    Which agent to call and what to label the call are the harness's business,
    not the target agent's, so no accepts schema declares them. They reach the
    validator anyway because normalization preserves the harness's own argument
    names beside the canonical ones. Read the set off the mapping table instead
    of naming keys here, so teaching the framework a new harness cannot quietly
    reintroduce a rejection for every plain-text delegation.
    """
    keys: set[str] = set()
    try:
        harnesses = get_tool_mappings().get("delegate",
                                            {}).get("harnesses", {})
    except TpuAgentSuiteError:
        return keys

    for h_info in harnesses.values():
        for meta_key in h_info.get("metadata", []):
            if isinstance(meta_key, str) and meta_key:
                keys.add(meta_key)
        for canonical_key, raw_spec in h_info.get("args", {}).items():
            keys.add(canonical_key)
            specs = raw_spec if isinstance(raw_spec, list) else [raw_spec]
            for spec in specs:
                if isinstance(spec, str) and spec:
                    # A dotted spec addresses into a nested arg; the key that
                    # lands in the bag is its root.
                    keys.add(spec.split(".", 1)[0])
    return keys


def _delegation_denial_reason(
    subagent_name: str,
    errors: list[str],
    accepts_schema: dict[str, Any],
    ctx_run_id: str = "",
) -> str:
    """Compose a denial the caller can act on without guessing.

    The schema example is the most concrete thing in this message, so a caller
    repairing its payload tends to copy from it -- including the placeholder
    run_id it carries. That would swap a missing run namespace for a wrong one,
    which is harder to notice because it validates. Naming the session's own
    run_id first, and only when the run_id is what failed, gives the caller the
    value to send before it reaches the example.
    """
    reason = (f"Delegation payload to '{subagent_name}' violated accepts"
              " schema: " + "; ".join(errors))

    if ctx_run_id and any("run_id" in err for err in errors):
        reason += (f"\n\nThis session's run_id is {ctx_run_id}; send that"
                   " value. Do not copy the run_id shown in the example.")

    example = load_schema_example(accepts_schema)
    if example is not None:
        reason += ("\n\nExpected payload schema example:\n```json\n"
                   f"{json.dumps(example, indent=2)}\n```")
    return reason


def _schema_verifiers_disabled() -> bool:
    """Whether schema verification is turned off for this run."""
    return DISABLE_SCHEMA_VERIFIERS or bool(
        os.environ.get(DISABLE_SCHEMA_VERIFIERS_ENV_VAR, "").strip())


def _builtin_validate_schema(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    hook_name: str = "validate-schema",
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler to deterministically validate payload contracts and deliverables."""
    if not event_payload or _schema_verifiers_disabled():
        return {"decision": "allow"}

    event_type = event_payload.get("event", "")
    subagent_name = str(event_payload.get("subagent") or "")

    accepts_schema, produces_schema = _resolve_agent_contract(
        package, workspace, subagent_name)

    if event_type == "PreToolUse":
        tool_name = event_payload.get("tool", "")
        if tool_name != "delegate" or not accepts_schema:
            return {"decision": "allow"}

        args = event_payload.get("args", {})
        prompt_val = args.get("prompt")
        schema_props = accepts_schema.get("properties", {})
        candidates = extract_all_json_objects(prompt_val)

        routing_keys = {"subagent"}
        ctx_run_id = event_payload.get("context", {}).get("run_id")

        if candidates:
            # Check candidates in reverse order: the last JSON that satisfies the schema wins.
            val_errors: list[str] = []
            matched = False
            for cand in reversed(candidates):
                payload_to_test = {
                    k: v
                    for k, v in cand.items()
                    if k not in routing_keys or k in schema_props
                }
                cand_errors = validate_instance(payload_to_test,
                                                accepts_schema)
                if not cand_errors:
                    matched = True
                    break
                if not val_errors:
                    val_errors = cand_errors

            if not matched:
                err_msg = _delegation_denial_reason(subagent_name, val_errors,
                                                    accepts_schema, ctx_run_id)
                return {"decision": "deny", "reason": err_msg}
        else:
            if "prompt" in args or "run_id" in args:
                payload_to_validate = dict(args)
                routing_keys = _delegate_routing_keys()
            else:
                payload_to_validate = {"prompt": prompt_val or ""}
                routing_keys = set()

            payload_to_validate = {
                k: v
                for k, v in payload_to_validate.items()
                if k not in routing_keys or k in schema_props
            }

            val_errors = validate_instance(payload_to_validate, accepts_schema)
            if val_errors:
                err_msg = _delegation_denial_reason(subagent_name, val_errors,
                                                    accepts_schema, ctx_run_id)
                return {"decision": "deny", "reason": err_msg}

    elif event_type == "SubagentStop":
        # A SubagentStop that reaches a hook body is a finished subagent:
        # `dispatch_hook` filters turn boundaries out for every harness. What
        # is left to decide is whether this agent owes a result at all.
        if not produces_schema:
            return {"decision": "allow"}

        raw_out = event_payload.get("output")
        candidates = extract_all_json_objects(raw_out)
        if not candidates:
            err_msg = (
                f"Subagent '{subagent_name}' output must be a valid JSON object conforming to its produces schema."
            )
            example = load_schema_example(produces_schema)
            if example is not None:
                err_msg += f"\n\nExpected output schema example:\n```json\n{json.dumps(example, indent=2)}\n```"
            return {
                "decision": "deny",
                "reason": err_msg,
                "followup_message": err_msg
            }

        # Check candidates in reverse order: the last JSON that satisfies the schema wins.
        all_errors: list[str] = []
        for cand in reversed(candidates):
            val_errors = validate_instance(cand, produces_schema)
            # Only a schema that declares `artifacts` owes one. A schema that
            # reports its deliverable through a typed field and closes itself
            # with `additionalProperties: false` cannot satisfy both this rule
            # and its own shape, so demanding one would reject its own
            # declared example on every retry. `validate_json_file` guards the
            # same call for the same reason.
            artifact_errors = (check_completed_has_artifact(cand)
                               if "artifacts" in produces_schema.get(
                                   "properties", {}) else [])
            # The conditional payload rule has to run here too, not only in
            # `validate_json_file`. That one is a call an agent chooses to make;
            # this hook fires on every delegation. Enforcing the rule only on
            # the voluntary path leaves a `completed` result that measured
            # nothing free to reach the caller.
            payload_errors = check_status_payload(produces_schema, cand)
            cand_errors = list(val_errors) + list(artifact_errors) + list(
                payload_errors)
            if not cand_errors:
                return {"decision": "allow"}
            if not all_errors:
                # Save the errors of the last candidate for reporting if none pass
                all_errors = cand_errors

        err_msg = (
            f"Subagent '{subagent_name}' output violated produces contract: " +
            "; ".join(all_errors))
        example = load_schema_example(produces_schema)
        if example is not None:
            err_msg += f"\n\nExpected output schema example:\n```json\n{json.dumps(example, indent=2)}\n```"
        return {
            "decision": "deny",
            "reason": err_msg,
            "followup_message": err_msg
        }

    return {"decision": "allow"}


def _builtin_record_event(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler that appends a lifecycle event to every open study.

    Always allows. This handler observes and must never be able to block work:
    a monitor that can veto the thing it monitors turns a logging bug into an
    outage, and the whole point of recording from a hook is that it keeps
    working when the agent's own reporting does not.
    """
    payload = event_payload or {}
    context = payload.get("context") or {}
    record = {
        "event": payload.get("event", ""),
        "tool": payload.get("tool", ""),
        "subagent": payload.get("subagent", ""),
        "session_id": context.get("session_id", ""),
        "run_id": context.get("run_id", ""),
        "package": package,
    }
    termination = payload.get("termination") or {}
    if termination:
        # `fully_idle` is how a monitor tells "the model stopped talking" from
        # "there is no outstanding work", which are very different states when
        # deciding whether a study has stalled.
        record["stop_reason"] = termination.get("reason", "")
        record["fully_idle"] = termination.get("fully_idle")

    try:
        for study_id in study.studies_for_event(record.get("run_id", "")):
            study.append_study_event(study_id, "hook", **record)
    except (TpuAgentSuiteError, OSError):
        # Recording is best-effort by design; see the docstring.
        pass
    return {"decision": "allow"}


def get_active_skill_file(session_id: str) -> Path:
    """Return the file that records a session's active tool-limited skill."""
    # Kept out of the session file, which record_session_run rewrites whole.
    return (utils.get_sessions_root() /
            f"{_safe_path_component(session_id)}.active-skill.json")


def _installed_skill_allowed_tools(
        package: str, workspace: Path) -> dict[str, list[str] | None]:
    """Return the installed build's skill_allowed_tools table, or {} if unavailable."""
    try:
        package_root = get_installed_package_root(workspace, package)
        build_info = load_json(package_root / "tpu-agent-suite-build.json")
    except (TpuAgentSuiteError, OSError, ValueError) as exc:
        print(f"runtime: skill tool limits skipped -- {exc}", file=sys.stderr)
        return {}
    table = build_info.get("skill_allowed_tools")
    return table if isinstance(table, dict) else {}


def _is_setup_command(package: str, command: str) -> bool:
    """Whether `command` is only this package's `compute setup` or `config`."""
    if any(c in command for c in "$`\n\r"):
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    if any(set(t) <= set(lexer.punctuation_chars) for t in tokens):
        return False
    # The rendered guide names the runtime as `~/...`; errors use the full path.
    tokens = [os.path.expanduser(t) for t in tokens]
    prefixes = (compute.config.runtime_command(package, "compute", "setup"),
                compute.config.runtime_command(package, "config", ""))
    return any(tokens[:len(p)] == p for p in map(shlex.split, prefixes))


def _builtin_skill_tools(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Keep the main thread limited to its skill's tools for the rest of the session."""
    payload = event_payload or {}
    raw_input = payload.get("raw_input") or {}
    # Claude Code sets agent_id on events from subagents; only the main thread is limited.
    if isinstance(raw_input, dict) and raw_input.get("agent_id"):
        return {"decision": "allow"}
    session_id = _get_event_session_id(payload)
    if not session_id:
        return {"decision": "allow"}
    active_file = get_active_skill_file(session_id)
    tool = payload.get("tool", "")

    if payload.get("event") == "PostToolUse":
        # Skill isn't in tool-mappings, so it keeps Claude Code's native name and input.
        if tool != "Skill" or payload.get("success") is False:
            return {"decision": "allow"}
        args = payload.get("args") or {}
        invoked = str(args.get("skill") or args.get("command") or "")
        # Plugin installs name skills `<plugin>:<skill>`.
        name = invoked.rsplit(":", 1)[-1].lstrip("/").strip()
        table = _installed_skill_allowed_tools(package, workspace)
        if name not in table:
            # Skills from other packages don't change anything.
            return {"decision": "allow"}
        if table[name] is None:
            # A package skill with no tool limit (e.g. trajectory export) lifts it.
            active_file.unlink(missing_ok=True)
        else:
            write_json(
                active_file, {
                    "skill": name,
                    "allowed": list(table[name]),
                    "package": package,
                    "recorded_at": utc_now_iso(),
                })
        return {"decision": "allow"}

    if payload.get("event") != "PreToolUse" or not tool:
        return {"decision": "allow"}
    active = load_json_doc(active_file)
    allowed = active.get("allowed")
    # An allow list, so tools we've never heard of are blocked too.
    if not isinstance(allowed, list) or tool in allowed:
        return {"decision": "allow"}
    if tool == "shell" and _is_setup_command(
            package, str((payload.get("args") or {}).get("command") or "")):
        return {"decision": "allow"}
    skill = active.get("skill") or "the active skill"
    return {
        "decision":
        "deny",
        "reason":
        (f"The {skill} skill doesn't allow `{tool}`. Delegate this step to"
         " the specialist agent that handles it. If the user asked for a"
         " different skill, invoke that skill first."),
    }


BUILTIN_HOOK_ACTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "start-run": _builtin_start_run,
    "validate-schema": _builtin_validate_schema,
    "reap-jobs": _builtin_reap_jobs,
    "record-event": _builtin_record_event,
    "skill-tools": _builtin_skill_tools,
}


def _translate_hook_reply(
    canonical_result: dict[str, Any],
    harness: str,
    canonical_event: str,
    hook_name: str,
    normalized_event: dict[str, Any],
) -> dict[str, Any]:
    """Translate a hook result into the reply the firing native event expects.

    A SubagentStop delivered through a hand-back tool fired as that tool's
    event, and the harness reads the reply in that event's shape: a
    SubagentStop `decision: block` sent to a PreToolUse is ignored and the
    unchecked result goes through. So the verdict is re-expressed as a tool
    permission decision. The corrective text a stop hook addresses to the
    subagent (`followup_message`) becomes the denial reason, which is what
    the subagent reads back as the failed tool call's result.
    """
    if canonical_event == "SubagentStop" and _is_handback_event(
            normalized_event):
        args = normalized_event.get("args") or {}
        # Always set by `_normalize_handback_event` from the mapping, which
        # requires it. No fallback: a reply shaped for a guessed event may be
        # ignored by the harness, letting the unchecked result through.
        source_event = str(args["handback_event"])
        # Only `deny` refuses, as only `deny` blocks a native SubagentStop:
        # `retry` is a Stop decision, and the same reply has to mean the same
        # thing on either road.
        refused = canonical_result.get("decision") == "deny"
        reply: dict[str, Any] = {"decision": "deny" if refused else "allow"}
        # A refusal always carries text: the denial reason is all the
        # subagent reads back, and the native road's block has the same
        # default.
        reason = (canonical_result.get("followup_message")
                  or canonical_result.get("reason")
                  or (f"Denied by hook {hook_name}" if refused else None))
        if reason:
            reply["reason"] = reason
        if canonical_result.get("message"):
            reply["message"] = canonical_result["message"]
        # Deliberately not carried over: `context` (and a `run_id` rendered
        # into it) and `updated_args`. A stop hook's context is addressed to
        # an agent that keeps working, and a subagent handing back is done;
        # `updated_args` would become the tool's `updatedInput` and silently
        # rewrite the result on its way to the caller, which the native road
        # cannot do either.
        return translate_canonical_output(reply, harness, source_event,
                                          hook_name)
    return translate_canonical_output(canonical_result, harness,
                                      canonical_event, hook_name)


def dispatch_hook(
    harness: str,
    package: str,
    hook_name: str,
    event: str,
    script_rel: str | None = None,
    command: str | None = None,
    builtin: str | None = None,
    timeout: float | None = DEFAULT_HOOK_TIMEOUT,
    workspace: Path | None = None,
    raw_input_text: str | None = None,
    script_args: list[str] | None = None,
    matcher: str | None = None,
    on_failure: str = DEFAULT_HOOK_ON_FAILURE,
    handback_event: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Execute user hook script or command with normalized event and return harness-specific response.

    Dispatch always answers. An error in dispatch itself -- reading the
    payload, a builtin that does not exist, translating the reply -- is a
    `dispatch` failure, settled by `on_failure` like a failure of the hook.
    Left to escape, it would exit non-zero with nothing on stdout, and each
    harness would decide: Claude Code runs the tool, Antigravity blocks it.

    `handback_event` is the native tool event a SubagentStop hook fired on
    as a hand-back, as the compiled command states it. Only that failure
    reply reads it; the normal path recognizes the hand-back from the payload.
    """
    try:
        return _dispatch_hook_body(harness=harness,
                                   package=package,
                                   hook_name=hook_name,
                                   event=event,
                                   script_rel=script_rel,
                                   command=command,
                                   builtin=builtin,
                                   timeout=timeout,
                                   workspace=workspace,
                                   raw_input_text=raw_input_text,
                                   script_args=script_args,
                                   matcher=matcher,
                                   on_failure=on_failure)
    except Exception as exc:  # noqa: BLE001 - becomes the hook's failure
        return _dispatch_failure(exc,
                                 harness=harness,
                                 hook_name=hook_name,
                                 event=event,
                                 workspace=workspace,
                                 on_failure=on_failure,
                                 handback_event=handback_event)


def _dispatch_failure(exc: Exception,
                      *,
                      harness: str,
                      hook_name: str,
                      event: str,
                      workspace: Path | None,
                      on_failure: str,
                      handback_event: str | None,
                      kind: str = "dispatch") -> tuple[dict[str, Any], int]:
    """The reply for a dispatch that raised: its policy, in the harness's form.

    A hand-back is answered in its tool event's shape, not SubagentStop's:
    Claude Code rejects a reply naming another event as a hook error and
    runs the tool, so a `deny` would be lost. The command line says which
    event that is, as the payload may be what failed to read.

    `kind` labels the failure record: `dispatch` (the body raised) or
    `startup` (the command line did not parse).
    """
    try:
        event = to_canonical_event(event, harness)
    except Exception:  # noqa: BLE001 - the raw name still labels the record
        pass
    verdict = _hook_failure(
        kind,
        f"Dispatch of hook '{hook_name}' failed: {type(exc).__name__}: {exc}",
        hook_name=hook_name,
        on_failure=on_failure,
        event_payload={"event": event},
        workspace=workspace or Path.cwd())
    handback = ({
        "args": {
            "handback": True,
            "handback_event": handback_event
        }
    } if handback_event else {})
    return _translate_hook_reply(verdict, harness, event, hook_name,
                                 handback), 0


class HookDispatchStartupError(Exception):
    """A `hook-dispatch` command line the dispatcher could not parse."""


def _hook_dispatch_startup_failure(hints: dict[str, str | None],
                                   reason: str) -> int:
    """Settle a `hook-dispatch` whose arguments did not parse; the exit code.

    argparse would exit 2, which Claude Code reads as a block whatever the
    policy. If `hints` names a known harness and event, the reply takes the
    harness's shape and is recorded like any other failure. Otherwise the
    exit code carries the policy: 0 for `allow`, 2 for `deny`.
    """
    hook_name = hints.get("hook") or "?"
    harness = hints.get("harness") or ""
    event = hints.get("event") or ""
    on_failure = hints.get("on_failure")
    if on_failure not in utils.HOOK_FAILURE_POLICIES:
        on_failure = DEFAULT_HOOK_ON_FAILURE
    workspace = Path(hints["workspace"]) if hints.get("workspace") else None
    # Everything that reads the store or shapes a reply is in one try: the
    # schema files may be what is missing, and the log or the reply may be
    # what fails. The fall-through below uses nothing that can.
    try:
        if (harness in utils.supported_harnesses() and to_canonical_event(
                event, harness) in utils.get_supported_hook_events()):
            response, code = _dispatch_failure(
                HookDispatchStartupError(reason),
                harness=harness,
                hook_name=hook_name,
                event=event,
                workspace=workspace,
                on_failure=on_failure,
                handback_event=hints.get("handback_event"),
                kind="startup")
            print(json.dumps(response), flush=True)
            return code
    except Exception as exc:  # noqa: BLE001 - the exit code still carries the policy
        reason = f"{reason} (no reply could be shaped: {type(exc).__name__}: {exc})"
    print(
        f"runtime: [{hook_name}] hook failed (startup): {reason}; "
        f"on_failure={on_failure}",
        file=sys.stderr)
    return 0 if on_failure == "allow" else 2


def _dispatch_hook_body(
    harness: str,
    package: str,
    hook_name: str,
    event: str,
    script_rel: str | None = None,
    command: str | None = None,
    builtin: str | None = None,
    timeout: float | None = DEFAULT_HOOK_TIMEOUT,
    workspace: Path | None = None,
    raw_input_text: str | None = None,
    script_args: list[str] | None = None,
    matcher: str | None = None,
    on_failure: str = DEFAULT_HOOK_ON_FAILURE,
) -> tuple[dict[str, Any], int]:
    """`dispatch_hook` itself, before its guard against its own errors."""
    if timeout is None or timeout <= 0:
        timeout = DEFAULT_HOOK_TIMEOUT

    event = to_canonical_event(event, harness)

    # 1. Parse raw payload
    if raw_input_text is None:
        try:
            raw_input_text = sys.stdin.read()
        except Exception:
            raw_input_text = "{}"

    try:
        raw_payload = json.loads(
            raw_input_text) if raw_input_text.strip() else {}
    except Exception as exc:
        raw_payload = {"_raw": raw_input_text, "error": str(exc)}

    resolved_workspace = resolve_workspace_root(workspace, raw_payload)

    # 2. Inbound normalization
    normalized_event = normalize_inbound_event(
        raw_payload=raw_payload,
        harness=harness,
        event=event,
        package=package,
        workspace=resolved_workspace,
    )
    canonical_event = normalized_event.get("event") or event

    # Lifecycle short-circuits for Google harnesses (antigravity), where SessionStart,
    # SessionEnd, and SubagentStop are emulated via PreInvocation and Stop events.
    if is_google_harness(harness):
        # SessionEnd is emulated via native Stop, which also fires when a subagent finishes.
        # A subagent ending is not the session ending; bypass the hook.
        if event == "SessionEnd" and canonical_event == "SubagentStop":
            return translate_canonical_output({"decision": "allow"}, harness,
                                              event, hook_name), 0

        # SessionStart is emulated via native PreInvocation, which fires on every turn, retry,
        # and subagent invocation:
        # 1. Subagent conversations are not root user sessions; bypass the hook.
        # 2. `initialNumSteps` is the trajectory step count before this turn (1 for initial user prompt, > 1 for turn 2+).
        # 3. `invocationNum` is the 0-indexed model call within the current turn (0 for initial model call).
        # The hook runs only on Turn 1's first model call. Every later model call
        # re-injects its cached context instead: the context arrives as an
        # ephemeral message, which only the model call it is injected into sees.
        if canonical_event == "SessionStart":
            args = normalized_event.get("args", {})
            if "initial_num_steps" not in args or "invocation_num" not in args:
                raise TpuAgentSuiteError(
                    f"harness '{harness}' requires 'initial_num_steps' and 'invocation_num' "
                    f"to determine SessionStart phase")
            init_steps = args["initial_num_steps"]
            inv_num = args["invocation_num"]
            session_id = _get_event_session_id(normalized_event)

            is_subagent = bool(normalized_event.get("subagent"))
            is_subagent_or_retry = is_subagent or inv_num > 0
            is_turn_2_plus = init_steps > 1

            if is_subagent_or_retry or is_turn_2_plus:
                cached_ctx = ""
                if not is_subagent:
                    cached_ctx = resolve_session_hook_context(
                        session_id, hook_name)
                pending_ctx = replay_pending_hook_context(session_id)
                merged = "\n\n".join(t for t in (cached_ctx, pending_ctx) if t)
                canonical: dict[str, Any] = {"decision": "allow"}
                if merged:
                    canonical["context"] = merged
                return translate_canonical_output(canonical, harness,
                                                  canonical_event,
                                                  hook_name), 0

    # The canonical SubagentStop means a subagent finished and owes a result.
    # Turn boundaries, halts, and root sessions are all filtered out here, in
    # the one place the distinction is drawn, so every hook body downstream --
    # builtin or script, on any harness -- may assume the agent is done and
    # fires once -- whether the result arrived as a native stop or through a
    # hand-back tool call (see `_subagent_stop_skip_reason`).
    #
    # Only a nameless stop outside the Google harnesses is said on stderr.
    # There it is one of the harness's own internal agents, which used to
    # reach the hook bodies, so a record it no longer leaves should be
    # traceable. On the Google harnesses a nameless stop is every root turn,
    # and halts and pauses were always skipped; noting those is only noise.
    if canonical_event == "SubagentStop":
        skip_reason = _subagent_stop_skip_reason(normalized_event, harness)
        if skip_reason:
            if (skip_reason == _NO_SUBAGENT_NAME_SKIP
                    and not is_google_harness(harness)
                    or skip_reason == _HANDBACK_ASSUMED_SKIP):
                print(f"[{hook_name}] SubagentStop skipped: {skip_reason}",
                      file=sys.stderr)
            return _translate_hook_reply({"decision": "allow"}, harness,
                                         canonical_event, hook_name,
                                         normalized_event), 0

    # 3. Matcher filtering emulation: if a matcher pattern is specified (e.g. for the Google
    # harnesses, where the host engine ignores matchers on Stop events), verify that the target
    # matches before dispatching.
    if matcher and matcher != "*":
        target = ""
        if canonical_event == "SubagentStop":
            target = normalized_event.get("subagent", "")
        elif canonical_event in ("PreToolUse", "PostToolUse"):
            target = normalized_event.get("tool", "")

        matched = False
        if target:
            patterns = [p.strip() for p in matcher.split("|") if p.strip()]
            for p in patterns:
                if p == target or p == "*":
                    matched = True
                    break
                try:
                    if re.search(rf"^{p}$", target):
                        matched = True
                        break
                except re.error:
                    pass
        if not matched:
            return _translate_hook_reply({"decision": "allow"}, harness,
                                         canonical_event, hook_name,
                                         normalized_event), 0

    # 4. Execute hook action (with multi-subagent batch handling)

    # If the harness invokes multiple subagents in a single call (e.g. the Google harnesses'
    # Subagents array), run the hook for each subagent. If some are denied, filter them out via
    # PreToolUse 'overwrite'.
    subs_list = []
    subagents_key = None
    if normalized_event.get("tool") == "delegate":
        h_args = harness_mapping(
            get_tool_mappings().get("delegate", {}).get("harnesses", {}),
            harness).get("args", {})
        raw_subs_key = h_args.get("subagents")
        if raw_subs_key and isinstance(raw_subs_key, str):
            subagents_key = raw_subs_key
            raw_subs = normalized_event.get("args", {}).get(raw_subs_key)
            if raw_subs is None:
                raw_subs = _extract_arg_by_path(raw_payload, raw_subs_key)
            if isinstance(raw_subs, str):
                try:
                    raw_subs = json.loads(raw_subs)
                except Exception:
                    raw_subs = []
            if isinstance(raw_subs, list) and len(raw_subs) > 1:
                subs_list = raw_subs

    if subs_list and event == "PreToolUse" and subagents_key:
        allowed_subs = []
        deny_reasons = []
        h_args = harness_mapping(
            get_tool_mappings().get("delegate", {}).get("harnesses", {}),
            harness).get("args", {})
        sub_candidates = h_args.get("subagent", [])
        if isinstance(sub_candidates, str):
            sub_candidates = [sub_candidates]
        prompt_candidates = h_args.get("prompt", [])
        if isinstance(prompt_candidates, str):
            prompt_candidates = [prompt_candidates]

        b_handler = None
        if builtin:
            b_handler = BUILTIN_HOOK_ACTIONS.get(builtin)
            if not b_handler:
                raise TpuAgentSuiteError(
                    f"unsupported builtin hook action: '{builtin}', expected one of {sorted(BUILTIN_HOOK_ACTIONS)}"
                )

        # Each subagent gets its own copy of the event, nested dicts included:
        # the runs below are concurrent and must not share mutable state.
        sub_runs = []
        for sub_entry in subs_list:
            sub_event = copy.deepcopy(normalized_event)

            matched_sub_k = None
            for cand in sub_candidates:
                if cand in sub_entry:
                    matched_sub_k = cand
                    break
                short_k = cand.split(".")[-1]
                if short_k in sub_entry:
                    matched_sub_k = short_k
                    break

            matched_prompt_k = None
            for cand in prompt_candidates:
                if cand in sub_entry:
                    matched_prompt_k = cand
                    break
                short_k = cand.split(".")[-1]
                if short_k in sub_entry:
                    matched_prompt_k = short_k
                    break

            if matched_sub_k:
                subagent_name = str(sub_entry[matched_sub_k])
                # Set both the canonical event envelope metadata and the tool call arg bag.
                sub_event["subagent"] = subagent_name
                sub_event["args"]["subagent"] = subagent_name
            if matched_prompt_k:
                sub_event["args"]["prompt"] = str(sub_entry[matched_prompt_k])
            sub_runs.append(
                (sub_entry, sub_event, matched_sub_k, matched_prompt_k))

        sub_results = _run_hook_action_per_subagent(
            [sub_event for _, sub_event, _, _ in sub_runs],
            hook_name=hook_name,
            workspace=resolved_workspace,
            package=package,
            timeout=timeout,
            on_failure=on_failure,
            builtin_handler=b_handler,
            command=command,
            script_rel=script_rel,
            script_args=script_args,
        )

        for (sub_entry, sub_event, matched_sub_k,
             matched_prompt_k), sub_res in zip(sub_runs, sub_results):
            if sub_res.get("decision") == "deny":
                label = sub_event["args"].get("subagent") or "subagent"
                deny_reasons.append(
                    f"{label}: {sub_res.get('reason', 'denied')}")
            else:
                # If hook returned updated_args, reflect updates back to the subagent entry
                up_args = sub_res.get("updated_args")
                if isinstance(up_args, dict):
                    updated_entry = dict(sub_entry)
                    if matched_sub_k and "subagent" in up_args:
                        updated_entry[matched_sub_k] = up_args["subagent"]
                    if matched_prompt_k and "prompt" in up_args:
                        updated_entry[matched_prompt_k] = up_args["prompt"]
                    allowed_subs.append(updated_entry)
                else:
                    allowed_subs.append(sub_entry)

        if not allowed_subs:
            canonical_result = {
                "decision": "deny",
                "reason": "; ".join(deny_reasons)
                or "All subagents were denied",
            }
        elif deny_reasons:
            # Partial allowance: allow tool call but overwrite with only allowed subagents
            canonical_result = {
                "decision": "allow",
                "reason":
                f"Filtered out denied subagents: {'; '.join(deny_reasons)}",
                "updated_args": {
                    subagents_key: allowed_subs
                },
            }
        else:
            canonical_result = {"decision": "allow"}
            if any(s != o for s, o in zip(allowed_subs, subs_list)):
                canonical_result["updated_args"] = {
                    subagents_key: allowed_subs
                }
    elif builtin:
        handler = BUILTIN_HOOK_ACTIONS.get(builtin)
        if not handler:
            raise TpuAgentSuiteError(
                f"unsupported builtin hook action: '{builtin}', expected one of {sorted(BUILTIN_HOOK_ACTIONS)}"
            )
        canonical_result = run_builtin_hook_action(
            handler,
            package=package,
            workspace=resolved_workspace,
            event_payload=normalized_event,
            hook_name=hook_name,
            timeout=timeout,
            on_failure=on_failure,
        )
    else:
        canonical_result = execute_single_hook_action(
            event_payload=normalized_event,
            hook_name=hook_name,
            workspace=resolved_workspace,
            package=package,
            command=command,
            script_rel=script_rel,
            timeout=timeout,
            script_args=script_args,
            on_failure=on_failure,
        )

    if is_google_harness(harness):
        session_id = _get_event_session_id(normalized_event)

        # Cache Turn 1 SessionStart context so Turn 2+ can re-inject it without re-running
        # the hook. Only the Google harnesses emulate SessionStart per-turn; the rest fire
        # it once per session and never read the cache back.
        if canonical_event == "SessionStart":
            ctx_text = (canonical_result.get("replay_context")
                        or canonical_result.get("context"))
            run_id = canonical_result.get("run_id")
            if run_id and not ctx_text:
                # Mirror translate_canonical_output: a bare run_id also yields context.
                ctx_text = _format_run_context(run_id)
            if session_id and ctx_text:
                record_session_hook_context(
                    session_id=session_id,
                    hook_name=hook_name,
                    context_text=ctx_text,
                    package=package,
                )

        # Spool PostToolUse context where the event itself cannot answer with it.
        # Antigravity documents that response as a bare {}, so a hook that produced
        # context has nowhere to put it and it would simply be lost.
        # above drains it on a later invocation.
        elif canonical_event == "PostToolUse":
            deferred_text = canonical_result.get("context")
            if deferred_text and session_id:
                record_pending_hook_context(
                    session_id=session_id,
                    hook_name=hook_name,
                    context_text=str(deferred_text),
                    package=package,
                )

    # 4. Translate to target harness format
    harness_response = _translate_hook_reply(
        canonical_result,
        harness,
        canonical_event,
        hook_name,
        normalized_event,
    )

    return harness_response, 0


def purge_data(force: bool = False) -> None:
    """Purge central run history and session records (~/.tpu-agent-suite/{runs,sessions}/)."""
    # Session records are derived state: each maps a conversation to a run and
    # caches the context injected for it. Purging runs without them would leave
    # live sessions pointing at run namespaces that no longer exist. Both are
    # central, so this clears every workspace's, as it always did for runs.
    targets = [
        p for p in (utils.get_runs_root(), utils.get_sessions_root())
        if p.is_dir()
    ]
    if not targets:
        return
    if not force:
        listing = "\n".join(f"  {p}" for p in targets)
        try:
            choice = input(
                "Are you sure you want to permanently delete all run data, artifacts, "
                f"and session records in:\n{listing}\n[y/N]: ").strip().lower(
                )
        except (EOFError, KeyboardInterrupt):
            choice = "n"
        if choice not in ("y", "yes"):
            print("Purge data cancelled by user.", file=sys.stderr)
            return
    for target in targets:
        shutil.rmtree(target, ignore_errors=True)
    print("Purged all run data, artifacts, and session records.")


def resolve_trajectory_target(session_id: str = "",
                              run_id: str = "") -> tuple[str, str, Path]:
    """Return (session_id, run_id, run_root) for a trajectory export.

    - run_id only (what the trajectory export skill passes): session_id is
      looked up from the run_id.
    - session_id only: run_id is looked up from that session_id.
    - Both: both are used as given.

    Raises TpuAgentSuiteError if neither is given, if the lookup finds nothing,
    or if the run has no directory.
    """
    session_id = session_id.strip()
    run_id = run_id.strip()
    if not run_id:
        if not session_id:
            raise TpuAgentSuiteError("no run id given; pass --run-id")
        run_id = resolve_session_run_id(session_id) or ""
        if not run_id:
            raise TpuAgentSuiteError(
                f"no run is recorded for session {session_id}; pass --run-id")
    if not session_id:
        session_id = _session_for_run(run_id)

    run_root = utils.get_run_root(run_id)
    if not run_root.is_dir():
        raise TpuAgentSuiteError(
            f"run {run_id} has no directory at {run_root}")
    return session_id, run_id, run_root


def _session_for_run(run_id: str) -> str:
    """Return the id of the session whose record names `run_id`."""
    # Session records are keyed by session id, so the match runs the other
    # way: scan for records naming this run and take the newest.
    candidates = []
    for record_path in utils.get_sessions_root().glob("*.json"):
        document = load_json_doc(record_path)
        if document.get("run_id") == run_id and document.get("session_id"):
            candidates.append((str(document.get("created_at")
                                   or ""), str(document["session_id"])))
    if not candidates:
        raise TpuAgentSuiteError(f"no session is recorded for run {run_id}")
    return sorted(candidates)[-1][1]


def resolve_trajectory_user_id(configured: str = "") -> str:
    """Identifier that exported trajectories are filed under.

    A configured `user_id` wins. Otherwise the gcloud account, then $USER.
    Both fallbacks are best effort: this value only shapes the object path,
    so failing to find one must never block an export.
    """
    if configured.strip():
        return configured.strip()
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        account = (result.stdout or "").strip()
        # gcloud prints "(unset)" rather than nothing when no account is set.
        if account and account != "(unset)":
            return account
    except (OSError, subprocess.SubprocessError):
        pass
    return os.environ.get("USER") or "unknown"


def handle_trajectory_cli(args: argparse.Namespace) -> int:
    """Run `trajectory export`, reporting refusals as messages not traces."""
    # Checked first: an unsupported harness has nothing to configure, and the
    # refusal is the clearest thing the user can be told.
    if not trajectory.supported(args.harness):
        raise TpuAgentSuiteError(trajectory.unsupported_message(args.harness))

    workspace = resolve_workspace_root(args.workspace)
    package = args.package or ""
    # Without a package there is no environment to consult, which is not an
    # error: --bucket can still name a destination.
    environment: dict[str, Any] = {}
    if package:
        environment, _ = get_environment(workspace, package)

    # Explicit flags win; otherwise both values come from the settings the
    # user agreed to (`config set --user-confirmed`), and only while sharing
    # is on.
    params = (environment.get("trajectory") or {}).get("params") or {}
    if params.get("enabled") is not True:
        if params.get("enabled") is False and not (args.destination
                                                   or args.bucket):
            raise TpuAgentSuiteError(
                "trajectory sharing is off: the user chose not to share "
                "sessions, so nothing was exported. It can be turned on "
                "with `config set trajectory_enabled=true ... "
                "--user-confirmed`.")
        params = {}
    # The skill runs a fixed command with no --share-org, so the environment is
    # the only way a session exported from inside one can be shared.
    share_org = args.share_org or str(params.get("share_org") or "")
    # One sink is chosen, and only after agreeing to share at all.
    # Not having agreed leaves no bucket and no sink, which the refusal in
    # the export reports.
    destination = args.destination or ("harbor" if params.get("sink")
                                       == "harbor" else "bucket")

    session_id, run_id, run_root = resolve_trajectory_target(
        args.session_id, args.run_id or "")

    # Harbor replaces the bucket rather than racing it: the user picks one.
    # Nothing is attempted opportunistically, so a session is
    # never uploaded anywhere the user did not choose, and a failure here is
    # reported instead of being disguised as a fallback.
    if destination == "harbor":
        missing = trajectory.harbor_missing()
        if missing:
            raise TpuAgentSuiteError(
                f"trajectory destination is 'harbor' but {missing}.")
        # Resolved here, not inside the export, so the --all consent rule is
        # applied identically whichever destination was chosen.
        bundle = trajectory.resolve_for_export(args.harness, session_id,
                                               workspace, args.entrypoint,
                                               args.include_full_session)
        # Staged outside the run directory to prevent being read by agents
        # in later sessions.
        staging = pathlib.Path(
            tempfile.mkdtemp(prefix="tpu-agent-suite-harbor-"))
        harbor_summary = trajectory.export_to_harbor(
            bundle=bundle,
            job_dir=staging / session_id,
            run_root=run_root,
            run_id=run_id,
            message=args.note,
            package=package,
            environment=environment,
            share_org=share_org,
        )
        shutil.rmtree(staging, ignore_errors=True)
        for key in ("job_dir", "trajectory", "manifest", "bundle"):
            harbor_summary.pop(key, None)
        if args.json:
            print(format_json(harbor_summary))
        else:
            print(f"Logged session {session_id} to Harbor"
                  f" ({harbor_summary['steps']} steps):"
                  f" {harbor_summary['destination']}")
        return 0

    # Resolved only now: on the Harbor path these are never read, and the
    # user id lookup shells out to gcloud with a ten second timeout.
    bucket = args.bucket or str(params.get("bucket") or "")
    user_id = resolve_trajectory_user_id(args.user_id
                                         or str(params.get("user_id") or ""))

    summary = trajectory.export_trajectory(
        harness=args.harness,
        session_id=session_id,
        workspace=workspace,
        run_root=run_root,
        run_id=run_id,
        entrypoint=args.entrypoint,
        message=args.note,
        bucket=bucket,
        user_id=user_id,
        package=package,
        environment=environment,
        include_full_session=args.include_full_session,
    )

    if args.json:
        print(format_json(summary))
        return 0

    # A dry run is allowed without a bucket, so name that case rather than
    # printing "to " with nothing after it. Always name the destination: a
    # mistyped bucket must be visible to the user, not silently accepted.
    destination = summary["uri"] or "(no bucket configured)"
    count = len(summary["objects"])
    print(f"Exported {count} objects to {destination}")
    for entry in summary["objects"]:
        print(f"  {entry['name']} ({entry['bytes']} bytes)")
    print(f"  transcripts: {summary['transcripts']}, "
          f"scoped: {summary['cut']['basis']}")
    for entry in summary["dropped"]:
        print(f"  dropped {entry['path']}: {entry['reason']}")
    return 0


def build_runtime_parser() -> argparse.ArgumentParser:
    """Construct parser for standalone runtime commands."""
    parser = argparse.ArgumentParser(
        prog="runtime",
        description=
        "Embedded runtime execution operations for TPU agent suite packages.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # start-run
    p_start = sub.add_parser(
        "start-run", help="Initialize a new run namespace for a package")
    p_start.add_argument("--package", default="", help="Optional package name")
    p_start.add_argument("--workspace",
                         type=Path,
                         default=None,
                         help="Optional workspace root directory override")
    p_start.add_argument(
        "--profile",
        default=None,
        metavar="PROFILE",
        help="Pin this run to a named compute profile from environment.json",
    )

    # study
    study.register_study_parser(sub)

    # journal
    journal.register_journal_parser(sub)

    # validate
    p_validate = sub.add_parser(
        "validate", help="Validate a JSON instance against a schema")
    p_validate.add_argument("--schema",
                            type=Path,
                            required=True,
                            help="Path to schema JSON file")
    p_validate.add_argument("instance",
                            type=Path,
                            help="Path to JSON instance file to validate")

    # purge-data
    p_purge = sub.add_parser(
        "purge-data",
        help="Purge centralized run history, artifacts, and session records")
    p_purge.add_argument("--force",
                         action="store_true",
                         help="Purge without interactive confirmation")

    # hook-dispatch
    p_hook = sub.add_parser(
        "hook-dispatch",
        help="Normalize and dispatch lifecycle hook invocation to user script",
    )
    p_hook.add_argument(
        "--harness",
        required=True,
        choices=utils.supported_harnesses(),
        help="Target agent harness executing the hook",
    )
    p_hook.add_argument("--package", required=True, help="Package name")
    p_hook.add_argument("--hook", required=True, help="Hook name")
    p_hook.add_argument(
        "--event",
        required=True,
        choices=[
            "PreToolUse",
            "PostToolUse",
            "SubagentStop",
            "subagentStop",
            "Stop",
            "SessionStart",
            "sessionStart",
            "PreInvocation",
            "SessionEnd",
            "sessionEnd",
        ],
        help="Lifecycle phase",
    )
    grp_action = p_hook.add_mutually_exclusive_group(required=True)
    grp_action.add_argument(
        "--script",
        help="Script path relative to package directory",
    )
    grp_action.add_argument(
        "--command",
        help="Shell command line to execute directly",
    )
    grp_action.add_argument(
        "--builtin",
        help="Builtin runtime action name (e.g. start-run)",
    )
    p_hook.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_HOOK_TIMEOUT,
        help=
        f"Hook execution timeout in seconds (default: {DEFAULT_HOOK_TIMEOUT}s)",
    )
    p_hook.add_argument(
        "--on-failure",
        choices=utils.HOOK_FAILURE_POLICIES,
        default=DEFAULT_HOOK_ON_FAILURE,
        help=("Decision when the hook fails to reach one: times out, exits "
              "non-zero, or prints no valid reply "
              f"(default: {DEFAULT_HOOK_ON_FAILURE})"),
    )
    p_hook.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root directory path",
    )
    p_hook.add_argument(
        "--matcher",
        default=None,
        help="Optional subagent or tool matcher pattern for event filtering",
    )
    p_hook.add_argument(
        "--handback-event",
        default=None,
        help=("Native tool event a SubagentStop hook is registered on as a "
              "hand-back; shapes the reply when dispatch itself fails"),
    )
    p_hook.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Trailing arguments forwarded to the hook script",
    )

    # Common compute flags shared between the top-level 'compute' parser and its subcommands
    compute_common = argparse.ArgumentParser(add_help=False)
    compute_common.add_argument(
        "--workspace",
        type=Path,
        default=argparse.SUPPRESS,
        help="Optional workspace root directory override",
    )
    # Deliberately unconstrained by `choices`: the set of real backends is
    # whatever drivers registered, which is empty until driver modules ship and
    # varies with internal hydration. Validation happens in `resolve_driver`,
    # which answers with a structured INVALID_BACKEND payload naming both the
    # available backends and any driver module that failed to import -- more
    # useful to an agent than argparse's usage dump.
    compute_common.add_argument(
        "--package",
        default=argparse.SUPPRESS,
        help="Package whose environment.json supplies compute configuration",
    )
    compute_common.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        metavar="PROFILE",
        help=
        "Named compute profile from environment.json to use for this command",
    )
    compute_common.add_argument(
        "--backend",
        default=argparse.SUPPRESS,
        metavar="BACKEND",
        help=("Override compute backend selection. Registered backends: " +
              (", ".join(compute.get_valid_backends()) or "none")),
    )
    compute_common.add_argument(
        "--run-id",
        default=argparse.SUPPRESS,
        metavar="RUN_ID",
        help="Run ID associated with this compute operation",
    )

    # Dynamically register backend-specific CLI flags from all registered drivers
    for driver_cls in compute.get_registered_drivers().values():
        driver_cls.register_cli_args(compute_common)

    parser.set_defaults(
        workspace=None,
        backend=None,
        package=None,
        profile=None,
        run_id=None,
    )

    # compute
    p_compute = sub.add_parser(
        "compute",
        parents=[compute_common],
        help="Execute compute workloads and inspect accelerator hardware",
    )
    compute_sub = p_compute.add_subparsers(dest="compute_action",
                                           required=True,
                                           help="Compute operation")

    # compute inspect
    p_inspect = compute_sub.add_parser(
        "inspect",
        parents=[compute_common],
        help="Inspect hardware topology, chip allocation, and health")
    p_inspect.add_argument("--json",
                           action="store_true",
                           help="Output raw JSON format")

    # compute versions
    p_versions = compute_sub.add_parser(
        "versions",
        parents=[compute_common],
        help=("Releases of jax, jaxlib, libtpu, torch and torch_tpu on the "
              "target: asked live on a local target, as `compute setup "
              "verify` recorded them elsewhere, else unknown"))
    p_versions.add_argument("--json",
                            action="store_true",
                            help="Output raw JSON format")
    p_versions.add_argument(
        "--interpreter",
        default=None,
        help="Python to ask on a local target (default: the configured one)")

    # compute preflight
    p_preflight = compute_sub.add_parser(
        "preflight",
        parents=[compute_common],
        help=("Check the configured backend's tools, credentials and target; "
              "exits 1 when a check fails"))
    p_preflight.add_argument("--json",
                             action="store_true",
                             help="Output raw JSON format")

    # compute run
    p_run = compute_sub.add_parser(
        "run",
        parents=[compute_common],
        help="Synchronously execute code on the target accelerator")
    p_run.add_argument("--code",
                       required=True,
                       help="Path to code/script file to execute")
    p_run.add_argument("--timeout",
                       type=int,
                       default=None,
                       help=("Execution timeout in seconds. On gke_job it is "
                             "timed from when the pod starts, not including "
                             "Kueue queue time, which gets a separate budget "
                             "of the same length"))
    p_run.add_argument("--name", help="Optional job name (gke_job only)")
    p_run.add_argument(
        "--interpreter",
        default=None,
        help="Path to python interpreter (e.g. /opt/conda/bin/python)")
    p_run.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LOCAL[:REMOTE]",
        help=("Copy a local file next to the script before it runs; REMOTE "
              "is a path relative to the script's working directory and "
              "defaults to the file's name. Repeatable."))
    p_run.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="REMOTE[:LOCAL]",
        help=("Copy a file the script wrote back to this host after it "
              "exits; REMOTE is relative to the script's working directory, "
              "LOCAL defaults to the same spelling. Repeatable."))
    p_run.add_argument("--json",
                       action="store_true",
                       help="Output raw JSON format")
    p_run.add_argument("run_args",
                       nargs=argparse.REMAINDER,
                       help="Arguments passed to code")

    # compute submit
    p_submit = compute_sub.add_parser(
        "submit",
        parents=[compute_common],
        help="Submit asynchronous batch workload recipe")
    p_submit.add_argument("--recipe",
                          required=True,
                          help="Path to batch recipe script")
    p_submit.add_argument("--name", default=None, help="Optional job name")
    p_submit.add_argument(
        "--interpreter",
        default=None,
        help="Path to python interpreter (e.g. /opt/conda/bin/python)")
    p_submit.add_argument(
        "--serve",
        action="store_true",
        default=False,
        help=("Launch workload in long-lived serving mode; exempts the job "
              "from session-end automatic reaping."))
    p_submit.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LOCAL[:REMOTE]",
        help=("Copy a local file next to the recipe before it runs; REMOTE "
              "is a path relative to the recipe's working directory and "
              "defaults to the file's name. Repeatable."))
    p_submit.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="REMOTE",
        help=("Fetch a file the recipe wrote when `job artifacts` runs; "
              "REMOTE is relative to the recipe's working directory and "
              "lands at <artifact dir>/REMOTE (or <out-dir>/REMOTE). A "
              "':LOCAL' suffix is ignored. Repeatable."))
    p_submit.add_argument("--json",
                          action="store_true",
                          help="Output raw JSON format")
    p_submit.add_argument("submit_args",
                          nargs=argparse.REMAINDER,
                          help="Arguments passed to recipe")

    # compute port-forward
    p_port_forward = compute_sub.add_parser(
        "port-forward",
        parents=[compute_common],
        help="Forward a local port to a remote workload port",
    )
    p_port_forward.add_argument(
        "--port",
        required=True,
        help="Port mapping: [LOCAL_PORT:]REMOTE_PORT (e.g. 8000 or 8080:8000)",
    )
    p_port_forward.add_argument(
        "--job-id",
        default=None,
        help=
        "Target job ID (required for gke_job unless pod is directly specified)",
    )
    p_port_forward.add_argument(
        "--background",
        "-b",
        action="store_true",
        help="Run the port forward in the background and print its PID",
    )
    p_port_forward.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format",
    )

    # compute job
    p_job = compute_sub.add_parser(
        "job",
        parents=[compute_common],
        help="Manage asynchronous jobs and lifecycle")
    p_job.add_argument(
        "job_action",
        choices=list(compute.JOB_ACTIONS),
        help="Job lifecycle action",
    )
    p_job.add_argument("--job-id", default=None, help="Target job ID")
    p_job.add_argument("--status",
                       default=None,
                       help="Filter jobs by status (for list)")
    p_job.add_argument(
        "--until",
        help="For wait: running, done (default), or log:<regex>",
    )
    p_job.add_argument(
        "--fail-on-log",
        action="append",
        default=None,
        help="For wait: another log regex that ends the wait (repeatable)",
    )
    p_job.add_argument(
        "--no-default-fail-patterns",
        action="store_true",
        help="For wait: drop the built-in failure log patterns",
    )
    p_job.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="For wait: seconds before exit 3 (default: 540)",
    )
    p_job.add_argument(
        "--no-refresh",
        action="store_true",
        help=("For list: show each job's recorded status without asking its "
              "backend, e.g. when the cluster is unreachable"),
    )
    p_job.add_argument("--out-dir",
                       type=Path,
                       default=None,
                       help="Destination directory for artifacts")
    p_job.add_argument(
        "--include-serving",
        action="store_true",
        help=("For reap: also cancel jobs submitted with --serve, which the "
              "session-end reaper leaves running"),
    )
    p_job.add_argument(
        "--orphaned",
        action="store_true",
        help=("For reap: instead of one run, cancel the batch jobs of every "
              "run whose session has ended without reaping them"),
    )
    p_job.add_argument(
        "--watch",
        action="store_true",
        help=("For reap: wait until the run's session has been gone for the "
              "lease grace period, then reap its batch jobs (`submit` starts "
              "this detached)"),
    )
    p_job.add_argument("--json",
                       action="store_true",
                       help="Output raw JSON format")

    # Flags for `compute setup` and `config`. These don't reuse
    # compute_common: `setup apply` takes its values only from --answers, so
    # accepting --backend or driver flags such as --zone would silently ignore
    # them. default=SUPPRESS lets a flag go before or after the subcommand;
    # without it, `config --package X show` would have `show`'s own --package
    # default (None) overwrite X.
    setup_common = argparse.ArgumentParser(add_help=False)
    setup_common.add_argument("--workspace",
                              type=Path,
                              default=argparse.SUPPRESS,
                              help="Workspace root directory")
    setup_common.add_argument(
        "--package",
        default=argparse.SUPPRESS,
        help="Package whose environment.json to read or update")
    setup_common.add_argument("--json",
                              action="store_true",
                              default=argparse.SUPPRESS,
                              help="Output raw JSON format")
    config_common = argparse.ArgumentParser(add_help=False,
                                            parents=[setup_common])
    config_common.add_argument(
        "--profile",
        default=argparse.SUPPRESS,
        help="show: profile to resolve instead of the run's; set: profile to "
        "change; verify: profile to check; plan: profile to plan for, which "
        "may be new (default: the one this run uses)")
    config_common.add_argument(
        "--run-id",
        default=argparse.SUPPRESS,
        help="Run whose profile to use (default: the session's)")
    p_setup = compute_sub.add_parser("setup",
                                     help="Configure named compute profiles")
    setup_sub = p_setup.add_subparsers(dest="setup_action",
                                       required=True,
                                       help="Setup operation")
    p_setup_plan = setup_sub.add_parser(
        "plan",
        parents=[config_common],
        help="Show what a profile needs, with values suggested from this "
        "machine; writes nothing")
    p_setup_plan.add_argument("--backend",
                              default=argparse.SUPPRESS,
                              help="Backend to plan for (default: the "
                              "profile's)")
    p_setup_plan.add_argument(
        "--fields",
        type=lambda s: [f.strip() for f in s.split(",") if f.strip()],
        default=[],
        help="Comma-separated fields to ask about, instead of the backend's "
        "setup fields")
    p_setup_apply = setup_sub.add_parser(
        "apply",
        parents=[setup_common],
        help="Validate answers and write a complete profile to "
        "environment.json, or report what is missing and write nothing")
    p_setup_apply.add_argument(
        "--profile",
        required=True,
        help="Name of the compute profile to create or update")
    p_setup_apply.add_argument(
        "--answers",
        required=True,
        help="JSON object of {field: value}: a file, '-' for stdin, or inline")
    p_setup_apply.add_argument("--dry-run",
                               action="store_true",
                               help="Validate without writing")
    p_setup_verify = setup_sub.add_parser(
        "verify",
        parents=[config_common],
        help="Check that a profile's target works and record the result")
    p_setup_verify.add_argument(
        "--smoke-test",
        action="store_true",
        help="On backends that schedule Jobs (gke_job), also submit a "
        "smoke-test Job that checks the image and its software; it can take "
        "several minutes")

    p_config = sub.add_parser(
        "config",
        parents=[config_common],
        help="Show or change compute profiles and settings")
    config_sub = p_config.add_subparsers(
        dest="config_action", help="Config operation (default: show)")
    config_sub.add_parser(
        "show",
        parents=[config_common],
        help="Show the profiles and which one this run resolves to")
    p_config_set = config_sub.add_parser(
        "set",
        parents=[config_common],
        help="Change settings, validated like `compute setup apply`")
    p_config_set.add_argument("assignments",
                              nargs="+",
                              metavar="KEY=VALUE",
                              help="Settings to change; KEY= unsets KEY")
    p_config_set.add_argument(
        "--user-confirmed",
        action="store_true",
        help="The user agreed to this change; required for trajectory-"
        "sharing settings")
    p_config_use = config_sub.add_parser(
        "use",
        parents=[config_common],
        help="Point this run (or, with --default, every run) at a profile")
    p_config_use.add_argument("profile_name", help="Profile to use")
    p_config_use.add_argument(
        "--default",
        action="store_true",
        help="Change environment.json's default instead of this run")

    # trajectory
    p_trajectory = sub.add_parser(
        "trajectory", help="Export a session's trajectory to a bucket you own")
    trajectory_sub = p_trajectory.add_subparsers(dest="trajectory_action",
                                                 required=True,
                                                 help="Feedback operation")

    # trajectory export
    p_export = trajectory_sub.add_parser(
        "export", help="Bundle and export this session's trajectory")
    # Constrained to the harnesses the framework knows, as hook-dispatch does, so a
    # typo is rejected as an unknown name rather than reported as a harness
    # whose support has not landed yet.
    p_export.add_argument("--harness",
                          required=True,
                          choices=utils.supported_harnesses(),
                          help="Harness the session ran on")
    p_export.add_argument(
        "--note",
        default="",
        help="Optional note from the user, stored in the manifest")
    p_export.add_argument(
        "--entrypoint",
        default="tpu-agent",
        help="Entrypoint skill whose activation starts the bundle")
    p_export.add_argument("--session-id", default="", help="Session to export")
    # Carried explicitly rather than via compute_common, which would also pull
    # in every backend driver's flags.
    p_export.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root, defaulting to the current directory")
    p_export.add_argument(
        "--package",
        default=None,
        help="Package whose environment supplies the destination")
    p_export.add_argument("--run-id",
                          default=None,
                          help="Run whose artifacts ship with the trajectory")
    p_export.add_argument(
        "--bucket",
        default="",
        help="Destination bucket, overriding the configured one")
    p_export.add_argument(
        "--destination",
        default="",
        choices=["", "bucket", "harbor"],
        help="Where to send the trajectory, overriding the configured choice")
    p_export.add_argument(
        "--share-org",
        default="",
        help="Organization to share the uploaded Harbor job with")
    p_export.add_argument("--user-id",
                          default="",
                          help="User identifier used in the object path")
    # Consent for shipping a whole session when the activation point could not
    # be found. Never set this on the user's behalf.
    p_export.add_argument(
        "--all",
        action="store_true",
        dest="include_full_session",
        help="Send the entire session when its starting point cannot be found")
    p_export.add_argument("--json",
                          action="store_true",
                          help="Output raw JSON format")

    p_deployment = sub.add_parser(
        "deployment",
        help="Inspect and manage Kubernetes serving deployments and topologies",
    )
    deployment_sub = p_deployment.add_subparsers(
        dest="deployment_action",
        required=True,
        help="Deployment operation",
    )
    p_dep_inspect = deployment_sub.add_parser(
        "inspect",
        help=
        "Inspect deployment topology (single-host, llm-d homogeneous/disaggregated, Dynamo)",
    )
    p_dep_inspect.add_argument(
        "--manifests",
        type=Path,
        default=None,
        help="Path to manifest file or directory to inspect offline",
    )
    p_dep_inspect.add_argument(
        "--namespace",
        default="default",
        help="Kubernetes namespace to inspect",
    )
    p_dep_inspect.add_argument(
        "--kubeconfig",
        default=None,
        help="Path to kubeconfig",
    )
    p_dep_inspect.add_argument(
        "--context",
        default=None,
        help="Kubectl context",
    )
    p_dep_inspect.add_argument(
        "--store-dir",
        type=Path,
        default=None,
        help=
        "Directory to store captured cluster manifests and inspection_report.json",
    )
    p_dep_inspect.add_argument(
        "--capture-logs",
        action="store_true",
        default=False,
        help=
        "Capture pod logs (<store-dir>/pod_logs/<pod_name>.log) when --store-dir is set",
    )
    p_dep_inspect.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Output raw JSON format",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for standalone runtime operations."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not _is_hook_dispatch(argv):
        args = build_runtime_parser().parse_args(argv)
    else:
        # A usage error must be settled by the policy, not argparse's exit 2;
        # its stderr message becomes the reason. `--help` (exit 0) passes.
        # Building the parser reads schema files from the store, so it can
        # fail the same way the lookups below can.
        usage = io.StringIO()
        try:
            with contextlib.redirect_stderr(usage):
                args = build_runtime_parser().parse_args(argv)
        except SystemExit as exc:
            if exc.code == 0:
                raise
            reason = usage.getvalue().strip().splitlines()
            return _hook_dispatch_startup_failure(
                _hook_dispatch_argv_hints(argv),
                reason[-1] if reason else f"exit {exc.code}")
        except Exception as exc:  # noqa: BLE001 - becomes the hook's failure
            return _hook_dispatch_startup_failure(
                _hook_dispatch_argv_hints(argv),
                f"{type(exc).__name__}: {exc}")

    cmd = getattr(args, "subcommand", None)
    try:
        if cmd == "start-run":
            # An unnamed workspace means the current working directory: the
            # runtime CLI runs inside the workspace it serves (harnesses set
            # cwd = workspace root), and the resolver itself never guesses.
            # Under user scope cwd holds no install of its own, and resolution
            # falls to the central store.
            run_doc = start_run(
                package=args.package,
                workspace=args.workspace or Path.cwd(),
                compute_profile=getattr(args, "profile", None),
            )
            print(format_json(run_doc))

        elif cmd == "validate":
            errors = validate_json_file(args.schema, args.instance)
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 2

        elif cmd == "purge-data":
            purge_data(force=args.force)

        elif cmd == "hook-dispatch":
            script_args = (args.script_args[1:] if args.script_args
                           and args.script_args[0] == "--" else
                           (args.script_args or []))
            response, code = dispatch_hook(
                harness=args.harness,
                package=args.package,
                hook_name=args.hook,
                event=args.event,
                script_rel=args.script,
                command=args.command,
                builtin=args.builtin,
                timeout=args.timeout,
                workspace=args.workspace,
                script_args=script_args,
                matcher=args.matcher,
                on_failure=args.on_failure,
                handback_event=args.handback_event,
            )
            print(json.dumps(response), flush=True)
            if abandoned_hook_threads():
                # Shutdown could abort on a lock the thread holds and turn
                # the verdict into a crash; the verdict is out, so skip it.
                os._exit(code)
            return code

        elif cmd == "study":
            study.dispatch_study_command(args)

        elif cmd == "journal":
            journal.dispatch_journal_command(args)

        elif cmd == "trajectory":
            return handle_trajectory_cli(args)

        elif cmd == "compute":
            return compute.handle_compute_cli(args)

        elif cmd == "config":
            return compute.handle_config_cli(args)

        elif cmd == "deployment":
            if getattr(args, "deployment_action", "") == "inspect":
                import importlib.util
                base_dir = Path(__file__).resolve().parent.parent
                candidate_paths = (
                    base_dir / "scripts" / "tpu-model-onboarding" /
                    "inspect_deployment.py",
                    base_dir / "package" / "scripts" / "tpu-model-onboarding" /
                    "inspect_deployment.py",
                )
                script_path = next((p for p in candidate_paths if p.exists()),
                                   candidate_paths[0])
                if script_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "inspect_deployment", script_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        report = mod.inspect_deployment(
                            manifests=args.manifests,
                            namespace=args.namespace,
                            kubeconfig=args.kubeconfig,
                            context=args.context,
                            store_dir=args.store_dir,
                            capture_logs=args.capture_logs,
                        )
                        print(format_json(report))
                        return 0
                raise TpuAgentSuiteError(
                    f"inspect_deployment script not found at {script_path}")

        return 0
    except compute.ComputeConfigError as exc:
        # Machine-readable only when the caller asked for it; `args` may be
        # unbound if the failure happened before parsing completed.
        wants_json = getattr(locals().get("args"), "json", False)
        print(json.dumps(exc.to_dict())
              if wants_json else f"runtime compute: {exc.message}",
              file=sys.stderr)
        return 1
    except TpuAgentSuiteError as exc:
        print(f"runtime: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
