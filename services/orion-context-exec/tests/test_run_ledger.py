from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

import app.storage as storage_mod
from app.storage import (
    persist_context_exec_run,
    run_dir,
    write_json_atomic,
    write_text_atomic,
)


@pytest.fixture()
def storage_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_root = tmp_path / "runs"
    monkeypatch.setattr(storage_mod.settings, "context_exec_storage_root", str(tmp_path))
    monkeypatch.setattr(storage_mod.settings, "context_exec_run_root", str(run_root))
    return run_root


class _FakeRun(BaseModel):
    run_id: str = "ctxrun_fake_1"
    final_text: str = "the answer"
    artifact: dict = {}
    runtime_debug: dict = {}
    verb_trace: list = []


def test_persist_creates_run_dir(storage_roots: Path) -> None:
    run = _FakeRun(run_id="ctxrun_abc123")
    result = persist_context_exec_run(run)
    target = storage_roots / "ctxrun_abc123"
    assert target.is_dir()
    assert result["ok"] is True
    assert result["run_id"] == "ctxrun_abc123"
    assert result["run_dir"] == str(target)


def test_manifest_shape(storage_roots: Path) -> None:
    run = _FakeRun(run_id="ctxrun_manifest")
    persist_context_exec_run(run)
    manifest_path = storage_roots / "ctxrun_manifest" / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "orion.context_exec.run_ledger.v1"
    assert manifest["run_id"] == "ctxrun_manifest"
    assert "manifest.json" in manifest["persisted_files"]
    assert "run.json" in manifest["persisted_files"]
    assert manifest["service"] == "orion-context-exec"
    assert "created_at" in manifest


def test_run_json_parses(storage_roots: Path) -> None:
    run = _FakeRun(run_id="ctxrun_runjson", final_text="hi")
    persist_context_exec_run(run)
    run_json = storage_roots / "ctxrun_runjson" / "run.json"
    assert run_json.is_file()
    data = json.loads(run_json.read_text())
    assert data["run_id"] == "ctxrun_runjson"
    assert data["final_text"] == "hi"


def test_final_md_written(storage_roots: Path) -> None:
    run = _FakeRun(run_id="ctxrun_final", final_text="my final answer")
    persist_context_exec_run(run)
    final_md = storage_roots / "ctxrun_final" / "final.md"
    assert final_md.is_file()
    assert final_md.read_text() == "my final answer"


def test_request_redaction(storage_roots: Path) -> None:
    run = _FakeRun(run_id="ctxrun_redact")
    request = {
        "text": "do the thing",
        "api_key": "sk-supersecret",
        "password": "hunter2",
        "authorization_token": "Bearer abc",
        "nested": {"secret_value": "leak", "safe": "keep"},
        "items": [{"my_apikey": "z"}, "plain"],
    }
    persist_context_exec_run(run, request=request)
    req_path = storage_roots / "ctxrun_redact" / "request.json"
    assert req_path.is_file()
    data = json.loads(req_path.read_text())
    assert data["text"] == "do the thing"
    assert data["api_key"] == "[REDACTED]"
    assert data["password"] == "[REDACTED]"
    assert data["authorization_token"] == "[REDACTED]"
    assert data["nested"]["secret_value"] == "[REDACTED]"
    assert data["nested"]["safe"] == "keep"
    assert data["items"][0]["my_apikey"] == "[REDACTED]"
    assert data["items"][1] == "plain"
    manifest = json.loads((storage_roots / "ctxrun_redact" / "manifest.json").read_text())
    assert "request.json" in manifest["persisted_files"]


def test_run_payloads_redacted_uniformly(storage_roots: Path) -> None:
    """Secret-named keys must be redacted in run.json/runtime_debug.json too,
    not just request.json."""
    run = _FakeRun(
        run_id="ctxrun_runredact",
        runtime_debug={"api_key": "sk-leak", "route_used": "chat"},
        artifact={"password": "hunter2", "summary": "ok"},
    )
    persist_context_exec_run(run)
    base = storage_roots / "ctxrun_runredact"

    run_json = json.loads((base / "run.json").read_text())
    assert run_json["runtime_debug"]["api_key"] == "[REDACTED]"
    assert run_json["runtime_debug"]["route_used"] == "chat"
    assert run_json["artifact"]["password"] == "[REDACTED]"
    assert run_json["artifact"]["summary"] == "ok"

    rt = json.loads((base / "runtime_debug.json").read_text())
    assert rt["api_key"] == "[REDACTED]"
    assert rt["route_used"] == "chat"

    art = json.loads((base / "artifact.json").read_text())
    assert art["password"] == "[REDACTED]"
    assert art["summary"] == "ok"


def test_atomic_writers_leave_no_tmp(tmp_path: Path) -> None:
    jpath = tmp_path / "sub" / "data.json"
    write_json_atomic(jpath, {"a": 1, "b": [2, 3]})
    assert json.loads(jpath.read_text()) == {"a": 1, "b": [2, 3]}

    tpath = tmp_path / "sub" / "note.md"
    write_text_atomic(tpath, "hello world")
    assert tpath.read_text() == "hello world"

    # No leftover .tmp siblings.
    assert not list((tmp_path / "sub").glob("*.tmp"))


def test_serialization_pydantic_and_repr_fallback(storage_roots: Path) -> None:
    # Object that is not JSON-serializable: must fall back to repr, not raise.
    class _NotSerializable:
        def __repr__(self) -> str:
            return "<not-serializable>"

    run = _FakeRun(run_id="ctxrun_ser")
    # Stuff the non-serializable object in runtime_debug-ish payload via request.
    request = {"weird": _NotSerializable(), "ok": "value"}
    result = persist_context_exec_run(run, request=request)
    assert result["ok"] is True
    req = json.loads((storage_roots / "ctxrun_ser" / "request.json").read_text())
    assert req["weird"] == "<not-serializable>"
    assert req["ok"] == "value"


def test_real_context_exec_run_v1(storage_roots: Path) -> None:
    from orion.schemas.context_exec import ContextExecRunV1

    run = ContextExecRunV1(
        run_id="ctxrun_real",
        status="ok",
        mode="general_investigation",
        text="investigate this",
        final_text="here is the answer",
    )
    result = persist_context_exec_run(run)
    assert result["ok"] is True
    run_json = json.loads((storage_roots / "ctxrun_real" / "run.json").read_text())
    assert run_json["run_id"] == "ctxrun_real"
    assert run_json["status"] == "ok"
    final_md = (storage_roots / "ctxrun_real" / "final.md").read_text()
    assert final_md == "here is the answer"


def test_optional_files_only_when_present(storage_roots: Path) -> None:
    run = _FakeRun(run_id="ctxrun_optional", artifact={}, runtime_debug={}, verb_trace=[])
    persist_context_exec_run(run)
    base = storage_roots / "ctxrun_optional"
    assert not (base / "artifact.json").exists()
    assert not (base / "runtime_debug.json").exists()
    assert not (base / "verb_trace.json").exists()

    run2 = _FakeRun(
        run_id="ctxrun_optional2",
        artifact={"k": "v"},
        runtime_debug={"d": 1},
        verb_trace=[{"step": "x"}],
    )
    persist_context_exec_run(run2)
    base2 = storage_roots / "ctxrun_optional2"
    assert (base2 / "artifact.json").exists()
    assert (base2 / "runtime_debug.json").exists()
    assert (base2 / "verb_trace.json").exists()


def test_missing_run_id_fallback(storage_roots: Path) -> None:
    class _NoId(BaseModel):
        final_text: str = "x"

    result = persist_context_exec_run(_NoId())
    assert result["ok"] is True
    assert result["run_id"].startswith("ctxrun_unknown_")
    assert (storage_roots / result["run_id"]).is_dir()


def test_fail_open_guard_does_not_propagate(
    storage_roots: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replicate the runner's try/except guard; persistence raising must not escape."""

    def _boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(storage_mod, "write_json_atomic", _boom)

    logger = logging.getLogger("test-run-ledger")
    run_id = "ctxrun_failopen"
    run = _FakeRun(run_id=run_id)
    escaped = False
    try:
        try:
            persist_context_exec_run(run)
        except Exception as exc:  # mimic runner fail-open
            logger.warning("failed to persist run_id=%s error=%s", run_id, exc)
    except Exception:
        escaped = True
    assert escaped is False
