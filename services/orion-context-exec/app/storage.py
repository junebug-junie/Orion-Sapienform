from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .settings import settings

logger = logging.getLogger("orion-context-exec.storage")

_REDACT_KEY_MARKERS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
    "key",
)
_RUN_LEDGER_SCHEMA = "orion.context_exec.run_ledger.v1"


def configured_storage_paths() -> dict[str, Path]:
    return {
        "root": Path(settings.context_exec_storage_root),
        "runs": Path(settings.context_exec_run_root),
        "artifacts": Path(settings.context_exec_artifact_root),
        "ledger": Path(settings.context_exec_ledger_root),
        "workspaces": Path(settings.context_exec_workspace_root),
        "cache": Path(settings.context_exec_cache_root),
        "tmp": Path(settings.context_exec_tmp_root),
    }


def ensure_storage_dirs() -> None:
    for name, path in configured_storage_paths().items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("storage dir unavailable name=%s path=%s error=%s", name, path, exc)


def _path_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    writable = bool(exists and is_dir and os.access(path, os.W_OK))
    return {
        "path": str(path),
        "present": exists,
        "is_dir": is_dir,
        "writable": writable,
    }


def storage_health_block() -> dict[str, Any]:
    paths = configured_storage_paths()
    dirs = {name: _path_status(path) for name, path in paths.items()}

    configured = bool(str(settings.context_exec_storage_root).strip())
    ok = configured and all(
        block["present"] and block["is_dir"] and block["writable"]
        for block in dirs.values()
    )

    error = None
    if not configured:
        error = "CONTEXT_EXEC_STORAGE_ROOT is not configured"
    elif not ok:
        bad = [
            name
            for name, block in dirs.items()
            if not (block["present"] and block["is_dir"] and block["writable"])
        ]
        error = f"storage dirs unavailable or not writable: {', '.join(bad)}"

    return {
        "configured": configured,
        "ok": ok,
        "root": str(paths["root"]),
        "run_ledger_enabled": settings.context_exec_run_ledger_enabled,
        "dirs": dirs,
        "error": error,
    }


def run_dir(run_id: str) -> Path:
    return Path(settings.context_exec_run_root) / run_id


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to a JSON-serializable structure.

    Never raises; un-serializable objects fall back to ``repr``.
    """
    try:
        # Pydantic v2 first (a model may also expose ``.dict``).
        if hasattr(obj, "model_dump"):
            try:
                return _to_jsonable(obj.model_dump(mode="json"))
            except Exception:
                return repr(obj)
        # Pydantic v1 (has ``.dict`` but is not a plain dict).
        if hasattr(obj, "dict") and not isinstance(obj, dict):
            try:
                return _to_jsonable(obj.dict())
            except Exception:
                return repr(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            try:
                return _to_jsonable(dataclasses.asdict(obj))
            except Exception:
                return repr(obj)
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "<unrepresentable>"


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if any(marker in key_l for marker in _REDACT_KEY_MARKERS):
                out[str(k)] = "[REDACTED]"
            else:
                out[str(k)] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def write_json_atomic(path: Path, payload: Any) -> None:
    text = json.dumps(
        _to_jsonable(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
    )
    write_text_atomic(path, text)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def persist_context_exec_run(run: Any, *, request: Any | None = None) -> dict[str, Any]:
    """Persist an immutable forensic bundle for a completed context-exec run.

    Returns a small summary dict. Exceptions are allowed to propagate so the
    caller (runner) can apply its fail-open guard.
    """
    run_id = getattr(run, "run_id", None)
    if not run_id:
        # Collision-safe fallback: real runs always carry a run_id, but never
        # let two id-less runs silently merge into the same directory.
        run_id = f"ctxrun_unknown_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_id = str(run_id)

    target = run_dir(run_id)
    target.mkdir(parents=True, exist_ok=True)

    # Serialize once, then redact secret-named keys uniformly across every
    # persisted payload (not just request.json). Key-based redaction is
    # best-effort and harmless to the service's own structured output, which
    # carries no legitimately secret-named keys.
    run_data = _redact(_to_jsonable(run))
    if not isinstance(run_data, dict):
        run_data = {}

    persisted: list[str] = []

    # run.json — full serialized run.
    write_json_atomic(target / "run.json", run_data)
    persisted.append("run.json")

    # final.md — best-available final text (free text; not key-redactable).
    final_text = getattr(run, "final_text", None)
    if not final_text:
        final_text = run_data.get("final_text")
    write_text_atomic(target / "final.md", str(final_text) if final_text else "")
    persisted.append("final.md")

    # Optional payloads, only when present and non-empty.
    artifact = run_data.get("artifact")
    if artifact:
        write_json_atomic(target / "artifact.json", artifact)
        persisted.append("artifact.json")

    runtime_debug = run_data.get("runtime_debug")
    if runtime_debug:
        write_json_atomic(target / "runtime_debug.json", runtime_debug)
        persisted.append("runtime_debug.json")

    verb_trace = run_data.get("verb_trace")
    if verb_trace:
        write_json_atomic(target / "verb_trace.json", verb_trace)
        persisted.append("verb_trace.json")

    if request is not None:
        write_json_atomic(target / "request.json", _redact(_to_jsonable(request)))
        persisted.append("request.json")

    # manifest.json LAST.
    persisted.append("manifest.json")
    manifest = {
        "schema": _RUN_LEDGER_SCHEMA,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_root": str(settings.context_exec_storage_root),
        "run_root": str(settings.context_exec_run_root),
        "service": "orion-context-exec",
        "version": str(settings.service_version),
        "persisted_files": persisted,
    }
    write_json_atomic(target / "manifest.json", manifest)

    return {
        "run_id": run_id,
        "run_dir": str(target),
        "persisted_files": persisted,
        "ok": True,
    }
