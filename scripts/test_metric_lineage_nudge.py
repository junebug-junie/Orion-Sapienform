"""Tests for scripts/hooks/metric_lineage_nudge.py -- Phase 3 of the metric
semantic layer (docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md).

Exercises `build_nudge()` directly (pure function: payload + cache in, JSON
string or None out) rather than the process/stdin boundary, matching the
pattern in scripts/test_session_stop_agent_board.py.
"""
from __future__ import annotations

import importlib
import json
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/hooks")

hook = importlib.import_module("metric_lineage_nudge")


def _cache(nodes: list[dict], hits: list[dict] | None = None) -> dict:
    return {"generated_at": 0.0, "nodes": nodes, "hits": hits or [], "files_scanned": 1}


def _node(name: str, **overrides) -> dict:
    base = {
        "urn": f"metric://field_channel/test-producer/{name}",
        "surface": "field_channel",
        "producer_service": "test-producer",
        "name": name,
        "field": None,
        "registry_source": "config/field/test.yaml",
        "schema_id": None,
        "meaning": None,
        "upstream": [],
        "upstream_organs": [],
        "declared_consumers": [],
        "feeds_dimensions": [],
        "all_producers": [],
        "notes": None,
    }
    base.update(overrides)
    return base


def test_edit_touching_registered_metric_produces_a_card():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": 'vec["cpu_pressure"] = 0.5'}}
    cache = _cache([_node("cpu_pressure", meaning="How loaded a node's CPU is.")])
    out = hook.build_nudge(payload, cache)
    assert out is not None
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "cpu_pressure" in parsed["hookSpecificOutput"]["additionalContext"]
    assert "How loaded a node's CPU is." in parsed["hookSpecificOutput"]["additionalContext"]


def test_write_content_field_is_scanned_too():
    payload = {"tool_name": "Write", "tool_input": {"file_path": "foo.py", "content": "reasoning_load = 0.3"}}
    cache = _cache([_node("reasoning_load")])
    out = hook.build_nudge(payload, cache)
    assert out is not None
    assert "reasoning_load" in out


def test_no_metric_token_in_edit_is_silent():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "print('hello world')"}}
    cache = _cache([_node("cpu_pressure")])
    assert hook.build_nudge(payload, cache) is None


def test_missing_cache_is_silent_not_an_error():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": 'vec["cpu_pressure"]'}}
    assert hook.build_nudge(payload, None) is None


def test_malformed_payload_shapes_fail_open():
    cache = _cache([_node("cpu_pressure")])
    assert hook.build_nudge({}, cache) is None
    assert hook.build_nudge({"tool_input": "not a dict"}, cache) is None
    assert hook.build_nudge({"tool_input": {}}, cache) is None
    assert hook.build_nudge(None, cache) is None  # type: ignore[arg-type]


def test_blast_radius_counts_non_test_hits_and_excludes_test_hits():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": 'vec["cpu_pressure"]'}}
    cache = _cache(
        [_node("cpu_pressure")],
        hits=[
            {"token": "cpu_pressure", "path": "orion/a.py", "line": 1, "kind": "subscript", "is_test": False},
            {"token": "cpu_pressure", "path": "orion/b.py", "line": 2, "kind": "subscript", "is_test": False},
            {"token": "cpu_pressure", "path": "tests/test_a.py", "line": 3, "kind": "subscript", "is_test": True},
            {"token": "other_metric", "path": "orion/c.py", "line": 4, "kind": "subscript", "is_test": False},
        ],
    )
    out = hook.build_nudge(payload, cache)
    assert "blast radius (discovered, non-test): 2" in out
    assert "orion/a.py:1" in out
    assert "orion/b.py:2" in out
    assert "tests/test_a.py" not in out  # test hits excluded from the count and the list
    assert "orion/c.py" not in out  # different token's hit must not leak in


def test_multiple_matched_metrics_capped_with_remainder_note():
    payload = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "foo.py",
            "new_string": "a_metric = 1; b_metric = 2; c_metric = 3; d_metric = 4",
        },
    }
    cache = _cache([_node("a_metric"), _node("b_metric"), _node("c_metric"), _node("d_metric")])
    out = hook.build_nudge(payload, cache)
    parsed = json.loads(out)
    body = parsed["hookSpecificOutput"]["additionalContext"]
    assert body.count("metric lineage:") == hook.MAX_CARDS
    assert "1 more registered metric(s) touched, not shown" in body


def test_fcc_subprocess_env_skips_main_entirely(monkeypatch, capsys):
    monkeypatch.setenv("ORION_FCC_SUBPROCESS", "1")
    rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out == ""
