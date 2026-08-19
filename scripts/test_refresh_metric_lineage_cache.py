"""Tests for scripts/refresh_metric_lineage_cache.py.

Mocks orion.metrics.{lineage,consumers} rather than running the real
~13-14s repo scan (measured live, see metric_lineage_nudge.py's docstring)
-- this must stay a fast gate test, not a periodic eval. The real scan is
exercised manually / via `make check-metric-lineage-cache-refresh`.
"""
from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, "scripts")

refresh_mod = importlib.import_module("refresh_metric_lineage_cache")


class _FakeNode:
    def __init__(self, name: str):
        self.name = name


class _FakeGraph:
    def __init__(self, names: list[str]):
        self.nodes = {f"metric://x/y/{n}": _FakeNode(n) for n in names}

    def scan_tokens(self):
        return {n.name: [n] for n in self.nodes.values()}


class _FakeHit:
    def __init__(self, token: str):
        self.token = token

    def to_dict(self):
        return {"token": self.token, "path": "orion/a.py", "line": 1, "kind": "subscript", "is_test": False}


def test_refresh_writes_atomic_and_well_shaped_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / ".cache" / "metric_lineage.json"
    monkeypatch.setattr(refresh_mod, "CACHE_PATH", cache_path)

    fake_graph = _FakeGraph(["cpu_pressure", "reasoning_load"])
    fake_scan = SimpleNamespace(hits=[_FakeHit("cpu_pressure")], files_scanned=42)

    def _fake_to_dict(node):
        return {"name": node.name, "urn": f"metric://x/y/{node.name}"}

    with mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.lineage.to_dict", side_effect=_fake_to_dict), \
         mock.patch("orion.metrics.consumers.scan_repo", return_value=fake_scan):
        result_path = refresh_mod.refresh()

    assert result_path == cache_path
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text())
    assert {n["name"] for n in payload["nodes"]} == {"cpu_pressure", "reasoning_load"}
    assert payload["hits"] == [
        {"token": "cpu_pressure", "path": "orion/a.py", "line": 1, "kind": "subscript", "is_test": False}
    ]
    assert payload["files_scanned"] == 42
    assert isinstance(payload["generated_at"], float)

    # No leftover temp files from the atomic write (write-to-temp + os.replace).
    leftovers = list(cache_path.parent.glob(".tmp.*")) + list(cache_path.parent.glob("*.tmp.*"))
    assert leftovers == []


def test_refresh_overwrites_stale_cache_cleanly(tmp_path, monkeypatch):
    cache_path = tmp_path / ".cache" / "metric_lineage.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"generated_at": 0.0, "nodes": [{"name": "stale_metric"}], "hits": [], "files_scanned": 1}))
    monkeypatch.setattr(refresh_mod, "CACHE_PATH", cache_path)

    fake_graph = _FakeGraph(["fresh_metric"])
    fake_scan = SimpleNamespace(hits=[], files_scanned=1)

    with mock.patch("orion.metrics.lineage.build_graph", return_value=fake_graph), \
         mock.patch("orion.metrics.lineage.to_dict", side_effect=lambda n: {"name": n.name}), \
         mock.patch("orion.metrics.consumers.scan_repo", return_value=fake_scan):
        refresh_mod.refresh()

    payload = json.loads(cache_path.read_text())
    assert {n["name"] for n in payload["nodes"]} == {"fresh_metric"}
