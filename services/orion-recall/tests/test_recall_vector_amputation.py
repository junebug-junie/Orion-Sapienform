"""Regression: orion-recall must not import or call vector_adapter."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def test_worker_and_recall_v2_import_without_vector_adapter() -> None:
    """Isolated subprocess so import hook does not disturb other tests."""
    import subprocess

    script = """
import importlib
import sys
from pathlib import Path
repo = Path(%r)
sys.path[:0] = [str(repo), str(repo / "services" / "orion-recall")]
real_import = importlib.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "app.storage.vector_adapter" or name.endswith("storage.vector_adapter"):
        raise ImportError("vector_adapter blocked")
    return real_import(name, globals, locals, fromlist, level)
importlib.__import__ = guarded
for mod in list(sys.modules):
    if mod.startswith("app."):
        del sys.modules[mod]
importlib.import_module("app.worker")
importlib.import_module("app.recall_v2")
importlib.import_module("app.collectors")
importlib.import_module("app.pipeline")
assert "app.storage.vector_adapter" not in sys.modules
print("ok")
""" % str(_REPO)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_RECALL_ROOT),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


def test_recall_vector_allowed_always_removed() -> None:
    from app.source_policy import build_vector_policy, recall_vector_allowed

    profile = {"profile": "reflect.v1", "vector_top_k": 99, "enable_vector": True}
    allowed, detail = recall_vector_allowed(profile, MagicMock(RECALL_ENABLE_VECTOR=True), path="main")
    assert allowed is False
    assert detail["reason"] == "removed_from_orion_recall"

    policy = build_vector_policy(profile, MagicMock(RECALL_ENABLE_VECTOR=True))
    for path, entry in policy.items():
        assert entry["allowed"] is False
        assert entry["reason"] == "removed_from_orion_recall"


def test_query_backends_and_v2_shadow_never_emit_vector(monkeypatch) -> None:
    from app import worker
    from app.profiles import get_profile
    from app.recall_v2 import run_recall_v2_shadow
    from orion.core.contracts.recall import RecallQueryV1

    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)

    profile = get_profile("reflect.v1")
    candidates, counts = asyncio.run(
        worker._query_backends(
            "hello",
            profile,
            session_id=None,
            node_id=None,
            entities=[],
            diagnostic=True,
            exclusion={},
        )
    )
    assert counts.get("vector", 0) == 0
    assert not any(c.get("source") == "vector" for c in candidates)

    async def _empty_exact(**kw):
        return []

    async def _empty_recent(*a, **k):
        return []

    monkeypatch.setattr("app.recall_v2.fetch_exact_fragments", _empty_exact)
    monkeypatch.setattr("app.recall_v2.fetch_recent_fragments", _empty_recent)
    monkeypatch.setattr("app.recall_v2.fetch_rdf_chatturn_exact_matches", lambda **kw: [])
    monkeypatch.setattr("app.recall_v2.fetch_rdf_fragments", lambda **kw: [])
    monkeypatch.setattr("app.recall_v2._pageindex_candidates", lambda *a, **k: [])

    q = RecallQueryV1(fragment="session abc123 token", profile="reflect.v1", verb="chat_general")
    bundle, debug = asyncio.run(run_recall_v2_shadow(q, profile=profile))

    assert not any(str(i.source) == "vector" for i in bundle.items)
    assert debug["backend_counts"].get("vector", 0) == 0
    assert debug["backend_counts"].get("vector_exact_anchor", 0) == 0
    for path, entry in debug["vector_policy"].items():
        assert entry["allowed"] is False
        assert entry["reason"] == "removed_from_orion_recall"


def test_fetch_anchor_candidates_vector_anchor_zero(monkeypatch) -> None:
    from app import worker

    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)

    candidates, counts = asyncio.run(
        worker._fetch_anchor_candidates(
            query_text="abc123 token",
            profile={"profile": "reflect.v1", "vector_top_k": 8},
            session_id=None,
            node_id=None,
            diagnostic=True,
            exclusion={},
        )
    )
    assert counts.get("vector_anchor", 0) == 0
    assert not any(c.get("source") == "vector" for c in candidates)
