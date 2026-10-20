"""Regression tests: Hub's concept graph stores load in the background.

Live 2026-10-20: Hub took ~4 min after container start to serve :8080 because
``api_routes.py`` built three Falkor stores at import time and each one scans
its whole graph in its constructor. These tests pin the fix
(``scripts/substrate_store_warm.py``): startup never waits for a build, a store
is built once, an event-loop caller never builds inline, and a caller that
arrives early gets an explicit not-ready answer instead of an empty graph.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
hub_scripts_pkg = HUB_ROOT / "scripts" / "__init__.py"
if (
    "scripts" not in sys.modules
    or not str(getattr(sys.modules.get("scripts"), "__file__", "")).startswith(str(HUB_ROOT))
):
    spec = importlib.util.spec_from_file_location(
        "scripts",
        str(hub_scripts_pkg),
        submodule_search_locations=[str(HUB_ROOT / "scripts")],
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["scripts"] = module
        spec.loader.exec_module(module)

from scripts import api_routes, concept_atlas_routes  # noqa: E402
from scripts import substrate_store_warm as warm  # noqa: E402

from orion.substrate.store import InMemorySubstrateGraphStore  # noqa: E402


class _GatedBuilder:
    """Builder that blocks until released, and records each call's thread."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0
        self.threads: list[int] = []
        self._lock = threading.Lock()

    def __call__(self) -> InMemorySubstrateGraphStore:
        with self._lock:
            self.calls += 1
            self.threads.append(threading.get_ident())
        assert self.release.wait(5), "test never released the builder"
        return InMemorySubstrateGraphStore()


def _live_api_routes():
    """The api_routes module currently in sys.modules. conftest's per-test
    isolation purges ``scripts.*``, and concept_atlas_routes resolves stores via
    a call-time ``import scripts.api_routes`` -- patch that module, not the one
    this file imported at collection time."""
    import scripts.api_routes as live

    return live


def _live_warm():
    """Same reason as ``_live_api_routes``: the holder class must be the one
    concept_atlas_routes sees at call time, not a purged copy."""
    import scripts.substrate_store_warm as live

    return live


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_background_warm_returns_immediately_and_reports_not_ready_until_built() -> None:
    builder = _GatedBuilder()
    holder = warm.WarmingSubstrateStore("orion", builder)

    started = time.monotonic()
    thread = warm.start_background_warm([holder])
    assert time.monotonic() - started < 0.5, "starting the warm-up must not wait for the build"
    assert thread is not None
    assert _wait_until(lambda: builder.calls == 1)

    assert holder.get_ready() is None
    with pytest.raises(warm.SubstrateStoreNotReadyError) as excinfo:
        holder.snapshot()  # proxied attribute access while loading
    assert excinfo.value.reason == "substrate_store_warming"
    assert excinfo.value.as_payload()["available"] is False
    assert holder.warm_status()["state"] == warm.STATE_WARMING

    builder.release.set()
    thread.join(5)
    assert holder.is_ready()
    assert isinstance(holder.get_ready(), InMemorySubstrateGraphStore)
    # Once ready, the holder delegates to the real store.
    assert holder.snapshot() is not None
    assert holder.warm_status()["store_type"] == "InMemorySubstrateGraphStore"


def test_store_is_built_exactly_once_under_concurrent_callers() -> None:
    builder = _GatedBuilder()
    holder = warm.WarmingSubstrateStore("self", builder)

    threads = [threading.Thread(target=holder.start_warm) for _ in range(8)]
    threads += [threading.Thread(target=holder.build_blocking) for _ in range(4)]
    threads.append(threading.Thread(target=lambda: warm.start_background_warm([holder])))
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    builder.release.set()
    assert _wait_until(holder.is_ready)
    assert builder.calls == 1

    # Further calls after ready never rebuild.
    assert holder.start_warm() is False
    holder.build_blocking()
    warm.start_background_warm([holder])
    assert builder.calls == 1


def test_event_loop_caller_never_builds_inline() -> None:
    builder = _GatedBuilder()
    holder = warm.WarmingSubstrateStore("aitown", builder)
    loop_thread: dict[str, int] = {}

    async def _on_loop() -> None:
        loop_thread["id"] = threading.get_ident()
        with pytest.raises(warm.SubstrateStoreNotReadyError):
            holder.query_concept_region(limit_nodes=1, limit_edges=1)
        with pytest.raises(RuntimeError):
            holder.build_blocking()
        with pytest.raises(RuntimeError):
            holder.wait_ready(0.1)

    asyncio.run(_on_loop())
    builder.release.set()
    assert _wait_until(holder.is_ready)
    assert builder.calls == 1
    assert builder.threads[0] != loop_thread["id"], "the build ran on the event-loop thread"


def test_failed_build_reports_failure_and_retries_after_backoff(monkeypatch) -> None:
    attempts = {"n": 0}

    def _flaky() -> InMemorySubstrateGraphStore:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("falkor down")
        return InMemorySubstrateGraphStore()

    holder = warm.WarmingSubstrateStore("orion", _flaky)
    holder.build_blocking()
    assert not holder.is_ready()
    with pytest.raises(warm.SubstrateStoreNotReadyError) as excinfo:
        holder.require()
    assert excinfo.value.reason == "substrate_store_build_failed"
    assert "falkor down" in (excinfo.value.error or "")
    assert attempts["n"] == 1, "inside the backoff window a caller must not trigger a rebuild"

    monkeypatch.setattr(warm, "RETRY_AFTER_FAILURE_SEC", 0.0)
    with pytest.raises(warm.SubstrateStoreNotReadyError):
        holder.require()  # kicks a background retry, still not ready on this call
    assert _wait_until(holder.is_ready)
    assert attempts["n"] == 2


def test_durable_backend_holder_is_not_built_at_import(monkeypatch) -> None:
    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "falkor")
    calls = {"n": 0}

    def _builder() -> InMemorySubstrateGraphStore:
        calls["n"] += 1
        return InMemorySubstrateGraphStore()

    holder = api_routes._substrate_store_holder("orion", _builder)
    assert calls["n"] == 0, "a durable backend must not build on the importing thread"
    assert holder.get_ready() is None

    monkeypatch.setenv("SUBSTRATE_STORE_BACKEND", "")
    eager = api_routes._substrate_store_holder("orion", _builder)
    assert calls["n"] == 1 and eager.is_ready(), "in-memory backends stay synchronous (no network, no scan)"


def test_concept_atlas_gets_none_and_reason_while_store_loads(monkeypatch) -> None:
    builder = _GatedBuilder()
    holder = _live_warm().WarmingSubstrateStore("orion", builder)
    holder.start_warm()
    assert _wait_until(lambda: builder.calls == 1)
    monkeypatch.setattr(_live_api_routes(), "SUBSTRATE_SEMANTIC_STORE", holder)

    assert concept_atlas_routes._get_substrate_store() is None
    payload = concept_atlas_routes._unavailable("substrate_store_unavailable")
    assert payload["available"] is False
    assert [s["store"] for s in payload["store_warmup"]] == ["orion"]
    assert payload["store_warmup"][0]["state"] == "warming"

    builder.release.set()
    assert _wait_until(holder.is_ready)
    resolved = concept_atlas_routes._get_substrate_store()
    assert isinstance(resolved, InMemorySubstrateGraphStore), "ready -> the real store, not the holder"
    assert "store_warmup" not in concept_atlas_routes._unavailable("substrate_store_unavailable")


def test_concept_atlas_summary_route_answers_unavailable_while_loading(monkeypatch) -> None:
    builder = _GatedBuilder()
    holder = _live_warm().WarmingSubstrateStore("aitown", builder)
    holder.start_warm()
    assert _wait_until(lambda: builder.calls == 1)
    monkeypatch.setattr(_live_api_routes(), "SUBSTRATE_SEMANTIC_STORE_AITOWN", holder)

    result = asyncio.run(concept_atlas_routes.concept_atlas_summary(graph="aitown"))
    assert result["available"] is False
    assert result["reason"] == "substrate_store_unavailable"
    assert result["total_concepts"] == 0
    assert any(s["store"] == "aitown" for s in result["store_warmup"])
    builder.release.set()


def test_seed_skips_a_store_that_has_not_loaded(monkeypatch) -> None:
    builder = _GatedBuilder()
    holder = warm.WarmingSubstrateStore("orion", builder)
    holder.start_warm()
    assert _wait_until(lambda: builder.calls == 1)
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", holder)
    loaded: list[object] = []
    monkeypatch.setattr(
        "orion.substrate.seed.load_seed_concepts_into_store", lambda store: loaded.append(store) or 4
    )

    assert api_routes.seed_golden_concepts_at_startup(wait_timeout_sec=0.05) == 0
    assert loaded == [], "never seed into a store that has not loaded"

    threading.Timer(0.1, builder.release.set).start()
    assert api_routes.seed_golden_concepts_at_startup(wait_timeout_sec=5) == 4
    assert loaded == [holder.get_ready()]


def test_source_posture_reports_warming_not_healthy(monkeypatch) -> None:
    builder = _GatedBuilder()
    holder = warm.WarmingSubstrateStore("orion", builder)
    holder.start_warm()
    assert _wait_until(lambda: builder.calls == 1)
    monkeypatch.setattr(api_routes, "SUBSTRATE_SEMANTIC_STORE", holder)

    semantic = api_routes._substrate_source_posture()["semantic"]
    assert semantic["kind"] == "warming"
    assert semantic["degraded"] is True
    assert semantic["error"] == "substrate_store_warming"
    assert api_routes.store_type_name(holder) == "warming"
    builder.release.set()


def test_not_ready_error_maps_to_503_with_reason() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    builder = _GatedBuilder()
    holder = warm.WarmingSubstrateStore("orion", builder)
    app = FastAPI()
    app.add_exception_handler(warm.SubstrateStoreNotReadyError, warm.not_ready_exception_handler)

    @app.get("/probe")
    def _probe() -> dict:
        holder.query_hotspot_region(limit_nodes=1, limit_edges=1)
        return {"ok": True}

    response = TestClient(app).get("/probe")
    assert response.status_code == 503
    body = response.json()
    assert body["reason"] == "substrate_store_warming"
    assert body["store"] == "orion"
    builder.release.set()


def test_wait_all_ready_returns_names_still_loading() -> None:
    fast = warm.WarmingSubstrateStore("orion", InMemorySubstrateGraphStore)
    slow_builder = _GatedBuilder()
    slow = warm.WarmingSubstrateStore("aitown", slow_builder)
    warm.start_background_warm([fast])
    slow.start_warm()
    assert warm.wait_all_ready([fast, slow, object()], timeout=0.2) == ["aitown"]
    slow_builder.release.set()
    assert warm.wait_all_ready([fast, slow], timeout=5) == []
