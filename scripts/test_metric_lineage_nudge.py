"""Tests for scripts/hooks/metric_lineage_nudge.py -- Phase 3 of the metric
semantic layer (docs/superpowers/specs/2026-08-12-metric-semantic-layer-design.md).

Exercises `build_nudge()` directly (pure function: payload + cache in, JSON
string or None out) rather than the process/stdin boundary, matching the
pattern in scripts/test_session_stop_agent_board.py.
"""
from __future__ import annotations

import importlib
import io
import json
import sys
import time
from unittest import mock

sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/hooks")

hook = importlib.import_module("metric_lineage_nudge")


def _set_stdin(monkeypatch, payload: dict) -> None:
    raw = json.dumps(payload).encode("utf-8")
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw)))


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


def test_more_than_max_consumers_shown_gets_a_remainder_line():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": 'vec["cpu_pressure"]'}}
    hits = [
        {"token": "cpu_pressure", "path": f"orion/f{i}.py", "line": i, "kind": "subscript", "is_test": False}
        for i in range(6)
    ]
    cache = _cache([_node("cpu_pressure")], hits=hits)
    out = hook.build_nudge(payload, cache)
    assert "blast radius (discovered, non-test): 6" in out
    assert out.count("orion/f") == hook.MAX_CONSUMERS_SHOWN
    assert "... and 2 more" in out


# --- Malformed-cache-shape regression coverage (review finding 2026-08-19:
# these four all raised an unhandled AttributeError/ValueError before the
# fix, on EVERY subsequent Edit/Write until the cache was fixed/deleted --
# the exact "never crash on a bad cache" contract this hook's own docstring
# claims). Exercised via build_nudge() directly (structural shape) and via
# the two dedicated main()-level tests below (the outer try/except).

def test_non_dict_node_in_cache_is_skipped_not_a_crash():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}}
    cache = _cache(["cpu_pressure"])  # nodes should be dicts, this one is a bare string
    assert hook.build_nudge(payload, cache) is None  # no valid node -> no token to key on, silent


def test_non_dict_hit_in_cache_is_skipped_not_a_crash():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}}
    cache = _cache([_node("cpu_pressure")], hits=["garbage", 42, None])
    out = hook.build_nudge(payload, cache)
    assert out is not None
    assert "blast radius (discovered, non-test): 0" in out


def test_cache_that_is_not_a_dict_at_all_is_treated_as_missing():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}}
    assert hook.build_nudge(payload, []) is None  # type: ignore[arg-type]
    assert hook.build_nudge(payload, "not even a container") is None  # type: ignore[arg-type]


def test_non_list_nodes_or_hits_in_cache_is_treated_as_missing():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}}
    assert hook.build_nudge(payload, {"generated_at": 0, "nodes": "not a list", "hits": []}) is None
    assert hook.build_nudge(payload, {"generated_at": 0, "nodes": [], "hits": "not a list"}) is None


# --- main()-level tests: staleness detection, background-refresh triggering
# and its cooldown lock, and generated_at-non-numeric fail-open -- none of
# this is reachable through build_nudge() alone (review finding 2026-08-19).

def test_main_missing_cache_triggers_background_refresh_and_stays_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(hook, "CACHE_PATH", tmp_path / "metric_lineage.json")
    monkeypatch.setattr(hook, "REFRESH_LOCK_PATH", tmp_path / "metric_lineage.refresh.lock")
    _set_stdin(monkeypatch, {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}})
    with mock.patch.object(hook.subprocess, "Popen") as popen:
        rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out == ""
    popen.assert_called_once()
    assert hook.REFRESH_LOCK_PATH.exists()  # cooldown lock actually written


def test_main_fresh_cache_does_not_trigger_refresh_and_prints_card(tmp_path, monkeypatch, capsys):
    cache_path = tmp_path / "metric_lineage.json"
    cache_path.write_text(json.dumps(_cache([_node("cpu_pressure")])).replace('"generated_at": 0.0', f'"generated_at": {time.time()}'))
    monkeypatch.setattr(hook, "CACHE_PATH", cache_path)
    monkeypatch.setattr(hook, "REFRESH_LOCK_PATH", tmp_path / "metric_lineage.refresh.lock")
    _set_stdin(monkeypatch, {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}})
    with mock.patch.object(hook.subprocess, "Popen") as popen:
        rc = hook.main()
    assert rc == 0
    popen.assert_not_called()
    assert "cpu_pressure" in capsys.readouterr().out


def test_main_stale_cache_triggers_refresh_but_still_uses_stale_data(tmp_path, monkeypatch, capsys):
    stale_ts = time.time() - hook.STALE_AFTER_SECONDS - 1
    cache_path = tmp_path / "metric_lineage.json"
    cache_path.write_text(json.dumps({**_cache([_node("cpu_pressure")]), "generated_at": stale_ts}))
    monkeypatch.setattr(hook, "CACHE_PATH", cache_path)
    monkeypatch.setattr(hook, "REFRESH_LOCK_PATH", tmp_path / "metric_lineage.refresh.lock")
    _set_stdin(monkeypatch, {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}})
    with mock.patch.object(hook.subprocess, "Popen") as popen:
        rc = hook.main()
    assert rc == 0
    popen.assert_called_once()  # stale -> self-heal triggered
    assert "cpu_pressure" in capsys.readouterr().out  # but still served from the stale cache this call


def test_main_recent_cooldown_lock_suppresses_a_second_refresh(tmp_path, monkeypatch, capsys):
    lock_path = tmp_path / "metric_lineage.refresh.lock"
    lock_path.touch()  # fresh lock, well within REFRESH_COOLDOWN_SECONDS
    monkeypatch.setattr(hook, "CACHE_PATH", tmp_path / "metric_lineage.json")  # missing
    monkeypatch.setattr(hook, "REFRESH_LOCK_PATH", lock_path)
    _set_stdin(monkeypatch, {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}})
    with mock.patch.object(hook.subprocess, "Popen") as popen:
        rc = hook.main()
    assert rc == 0
    popen.assert_not_called()  # cooldown active, no pile-up


def test_main_generated_at_non_numeric_is_treated_as_stale_not_a_crash(tmp_path, monkeypatch, capsys):
    cache_path = tmp_path / "metric_lineage.json"
    cache_path.write_text(json.dumps({"generated_at": "not-a-number", "nodes": [_node("cpu_pressure")], "hits": []}))
    monkeypatch.setattr(hook, "CACHE_PATH", cache_path)
    monkeypatch.setattr(hook, "REFRESH_LOCK_PATH", tmp_path / "metric_lineage.refresh.lock")
    _set_stdin(monkeypatch, {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "new_string": "cpu_pressure = 1"}})
    with mock.patch.object(hook.subprocess, "Popen") as popen:
        rc = hook.main()
    assert rc == 0
    popen.assert_called_once()  # unparseable timestamp -> treated as maximally stale, self-heals
    assert "cpu_pressure" in capsys.readouterr().out  # still serves the (otherwise valid) card
