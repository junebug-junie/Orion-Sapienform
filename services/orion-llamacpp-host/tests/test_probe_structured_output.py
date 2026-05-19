"""Unit tests for probe_structured_output (no live server)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from probe_structured_output import (  # noqa: E402
    PROBE_SCHEMA,
    parse_json_content,
    run_probe,
    select_best_method,
    validate_probe_schema,
)


def test_validate_probe_schema_ok() -> None:
    _, valid, issues = validate_probe_schema({"ok": True, "name": "Juniper", "count": 3})
    assert valid is True
    assert issues == []


def test_validate_probe_schema_rejects_extra() -> None:
    _, valid, issues = validate_probe_schema(
        {"ok": True, "name": "Juniper", "count": 3, "extra": "x"}
    )
    assert valid is False
    assert any("extra_keys" in i for i in issues)


def test_select_best_method_prefers_enforcement() -> None:
    methods = [
        {"name": "json_object_only", "valid_json": True, "schema_enforcement": False},
        {"name": "json_object_schema", "valid_json": True, "schema_enforcement": True},
    ]
    best, rec = select_best_method(methods)
    assert best == "json_object_schema"
    assert rec is True


def test_select_best_method_soft_json_object_only() -> None:
    methods = [
        {"name": "json_object_schema", "valid_json": True, "schema_enforcement": False},
        {"name": "json_object_only", "valid_json": True, "schema_enforcement": False},
    ]
    best, rec = select_best_method(methods)
    assert best == "json_object_only"
    assert rec is False


def test_parse_json_strips_fence() -> None:
    obj, err = parse_json_content('```json\n{"ok": true, "name": "Juniper", "count": 3}\n```')
    assert err is None
    assert obj["count"] == 3


def _mock_client_factory(responses: Dict[str, Any]):
    class _Resp:
        def __init__(self, status: int, body: Dict[str, Any]) -> None:
            self.status_code = status
            self.content = json.dumps(body).encode()
            self.text = self.content.decode()

        def json(self) -> Dict[str, Any]:
            return json.loads(self.content)

    class _Client:
        def get(self, url: str, timeout: float = 10.0) -> _Resp:
            if url.endswith("/health"):
                return _Resp(200, {"status": "ok"})
            return _Resp(404, {})

        def post(self, url: str, json: Dict[str, Any], timeout: float = 120.0) -> _Resp:
            method = "unknown"
            rf = json.get("response_format")
            if rf is None:
                method = "no_response_format_control"
            elif rf.get("type") == "json_object" and "schema" not in rf:
                method = "json_object_only"
            elif rf.get("type") == "json_object":
                method = "json_object_schema"
            adv = "SHOULD_NOT_APPEAR" in str(json.get("messages", [{}])[0].get("content", ""))
            if adv and method == "json_object_schema":
                content = json.dumps(
                    {"ok": True, "name": "Juniper", "count": 3, "extra": "SHOULD_NOT_APPEAR"}
                )
            elif method.endswith("schema") or method == "json_object_only":
                content = json.dumps({"ok": True, "name": "Juniper", "count": 3})
            else:
                content = "not json"
            return _Resp(200, {"choices": [{"message": {"content": content}}]})

        def close(self) -> None:
            pass

    return lambda: _Client()


def test_run_probe_writes_artifacts(tmp_path: Path) -> None:
    summary = run_probe(
        base_url="http://mock:8013",
        model="test-model",
        timeout_sec=30.0,
        artifact_dir=tmp_path,
        include_thinking_on_probe=False,
        verbose=False,
        client_factory=_mock_client_factory({}),
    )
    assert (tmp_path / "summary.json").is_file()
    assert summary["best_method"] in {
        "json_object_schema",
        "json_object_only",
        "none",
        "json_schema_schema",
    }


def test_chat_template_kwargs_retry(tmp_path: Path) -> None:
    calls: list[Dict[str, Any]] = []

    class _Resp:
        status_code: int
        content: bytes
        text: str

        def __init__(self, status: int, body: Dict[str, Any] | None = None) -> None:
            self.status_code = status
            self.content = json.dumps(body or {}).encode()
            self.text = self.content.decode()

        def json(self) -> Dict[str, Any]:
            return json.loads(self.content)

    class _Client:
        def get(self, url: str, timeout: float = 10.0) -> _Resp:
            return _Resp(200, {"status": "ok"})

        def post(self, url: str, json: Dict[str, Any], timeout: float = 120.0) -> _Resp:
            calls.append(dict(json))
            if json.get("chat_template_kwargs") and len(calls) == 1:
                return _Resp(400, {"error": "chat_template_kwargs unsupported"})
            return _Resp(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"ok": True, "name": "Juniper", "count": 3}
                                )
                            }
                        }
                    ]
                },
            )

        def close(self) -> None:
            pass

    summary = run_probe(
        base_url="http://mock:8013",
        model="m",
        timeout_sec=10.0,
        artifact_dir=tmp_path,
        include_thinking_on_probe=False,
        verbose=False,
        client_factory=lambda: _Client(),
    )
    assert len(calls) >= 2
    assert summary["thinking_control_supported"] is False
