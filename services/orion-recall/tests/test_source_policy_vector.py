"""Vector source policy — starvation and v2 shadow regression tests."""

from __future__ import annotations

import asyncio
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

from app import worker
from app.profiles import get_profile, load_profiles
from app.recall_v2 import run_recall_v2_shadow
from app.source_policy import build_vector_policy, recall_vector_allowed
from orion.core.contracts.recall import RecallQueryV1


def test_recall_vector_allowed_global_off() -> None:
    profile = {"profile": "reflect.v1", "vector_top_k": 8}
    allowed, detail = recall_vector_allowed(profile, MagicMock(RECALL_ENABLE_VECTOR=False), path="main")
    assert allowed is False
    assert detail["reason"] == "disabled_global"


def test_recall_vector_allowed_profile_top_k_zero() -> None:
    load_profiles.cache_clear()
    profile = get_profile("assist.light.v1")
    allowed, detail = recall_vector_allowed(profile, MagicMock(RECALL_ENABLE_VECTOR=True), path="main")
    assert allowed is False
    assert detail["reason"] == "disabled_profile_vector_top_k_zero"


def test_build_vector_policy_all_paths_disabled() -> None:
    load_profiles.cache_clear()
    profile = get_profile("assist.light.v1")
    policy = build_vector_policy(profile, MagicMock(RECALL_ENABLE_VECTOR=True))
    for path in ("main", "anchor", "graphtri", "v2_shadow_exact", "v2_shadow_semantic"):
        assert policy[path]["allowed"] is False
        assert policy[path]["reason"] == "disabled_profile_vector_top_k_zero"


@pytest.mark.parametrize("global_on,vector_top_k", [(False, 8), (True, 0)])
def test_query_backends_never_calls_vector_when_disabled(monkeypatch, global_on: bool, vector_top_k: int) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", global_on)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_SQL_TIMELINE", False)

    def _fail_vector(**kwargs):
        raise AssertionError("fetch_vector_fragments must not be called when vector disabled")

    monkeypatch.setattr(worker, "fetch_vector_fragments", _fail_vector)

    profile = {"profile": "reflect.v1", "vector_top_k": vector_top_k, "enable_sql_timeline": False, "rdf_top_k": 0}
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


@pytest.mark.asyncio
async def test_v2_shadow_no_vector_when_global_off(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", False)
    monkeypatch.setattr("app.recall_v2.settings", worker.settings)

    vec_frag_called = False
    vec_exact_called = False

    def _fail_frag(**kwargs):
        nonlocal vec_frag_called
        vec_frag_called = True
        raise AssertionError("fetch_vector_fragments must not run")

    def _fail_exact(**kwargs):
        nonlocal vec_exact_called
        vec_exact_called = True
        raise AssertionError("fetch_vector_exact_matches must not run")

    monkeypatch.setattr("app.recall_v2.fetch_vector_fragments", _fail_frag)
    monkeypatch.setattr("app.recall_v2.fetch_vector_exact_matches", _fail_exact)

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
    profile = get_profile("reflect.v1")
    bundle, debug = await run_recall_v2_shadow(q, profile=profile)

    assert not vec_frag_called
    assert not vec_exact_called
    assert not any(str(i.source) == "vector" for i in bundle.items)
    assert debug["backend_counts"].get("vector", 0) == 0
    assert debug["backend_counts"].get("vector_exact_anchor", 0) == 0
    for path in ("v2_shadow_exact", "v2_shadow_semantic"):
        assert debug["vector_policy"][path]["allowed"] is False


@pytest.mark.asyncio
async def test_v2_shadow_assist_light_no_vector(monkeypatch) -> None:
    load_profiles.cache_clear()
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    monkeypatch.setattr("app.recall_v2.settings", worker.settings)

    monkeypatch.setattr("app.recall_v2.fetch_vector_fragments", lambda **kw: (_ for _ in ()).throw(AssertionError("vector")))
    monkeypatch.setattr("app.recall_v2.fetch_vector_exact_matches", lambda **kw: (_ for _ in ()).throw(AssertionError("vector")))
    async def _empty_exact(**kw):
        return []

    async def _empty_recent(*a, **k):
        return []

    monkeypatch.setattr("app.recall_v2.fetch_exact_fragments", _empty_exact)
    monkeypatch.setattr("app.recall_v2.fetch_recent_fragments", _empty_recent)
    monkeypatch.setattr("app.recall_v2.fetch_rdf_chatturn_exact_matches", lambda **kw: [])
    monkeypatch.setattr("app.recall_v2.fetch_rdf_fragments", lambda **kw: [])
    monkeypatch.setattr("app.recall_v2._pageindex_candidates", lambda *a, **k: [])

    profile = get_profile("assist.light.v1")
    q = RecallQueryV1(fragment="hello", profile="assist.light.v1", verb="chat_general")
    bundle, debug = await run_recall_v2_shadow(q, profile=profile)

    assert not any(str(i.source) == "vector" for i in bundle.items)
    assert all(not debug["vector_policy"][p]["allowed"] for p in debug["vector_policy"])


def test_vector_enabled_sanity(monkeypatch) -> None:
    monkeypatch.setattr(worker.settings, "RECALL_ENABLE_VECTOR", True)
    profile = {"profile": "reflect.v1", "vector_top_k": 4}
    allowed, detail = recall_vector_allowed(profile, worker.settings, path="main")
    assert allowed is True
    assert detail["reason"] == "enabled"
