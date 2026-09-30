"""PCR collectors under the recall deadline, timed, and a boot-time substrate
store warmup (fix/recall-pcr-block-timing, 2026-09-30).

Live finding this pins: the first purposeful recall after a restart took
9,542ms with timings_ms showing fetch 189ms + fusion 3ms. The missing ~9.3s
was the PCR block (active_packet + concept_region, the latter paying a cold
6.25s get_substrate_store() hydration), which ran after the deadline-bounded
fetch, outside the deadline, and untimed.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app import worker
from app.profiles import get_profile
from orion.core.contracts.recall import MemoryBundleStatsV1, MemoryBundleV1, RecallQueryV1

# Top-level, non-overlapping stages of process_recall. feeds/retrievers are
# sub-parts of fetch; pcr_active_packet/pcr_concept_region of pcr_collectors.
_TOP_LEVEL_STAGES = (
    "intake",
    "fetch",
    "windowing",
    "suppression",
    "pcr_collectors",
    "boost",
    "fusion",
    "eligible_count",
    "shadow_compare",
)


def _fetch_candidate() -> Dict[str, Any]:
    return {"id": "fetch-1", "source": "sql_timeline", "text": "gpu1 recovered", "ts": time.time(), "score": 0.6}


@pytest.fixture
def purposeful(monkeypatch: pytest.MonkeyPatch):
    s = worker.settings
    monkeypatch.setattr(s, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(s, "RECALL_ACTIVE_PACKET_ENABLED", True)
    monkeypatch.setattr(s, "RECALL_CONCEPT_REGION_ENABLED", True)
    monkeypatch.setattr(s, "RECALL_BELIEF_RENDER_BUDGET", 128)
    monkeypatch.setattr(s, "RECALL_ENABLE_SQL_CHAT", False)
    monkeypatch.setattr(s, "RECALL_ENABLE_SQL_TIMELINE", False)
    monkeypatch.setattr(s, "RECALL_ENABLE_RDF", False)
    monkeypatch.setattr(s, "RECALL_INTENT_ROUTING_ENABLED", False)
    monkeypatch.setattr(s, "RECALL_DEADLINE_MS_DEFAULT", 60000)

    belief_profile = get_profile("chat.belief.semantic.v1")
    monkeypatch.setattr(worker, "get_profile", lambda _name: dict(belief_profile))
    # Force both collectors into the plan regardless of the intent table.
    monkeypatch.setattr(worker, "collectors_for_intent", lambda _i: {"active_packet": True, "concept_region": True})

    async def _anchor(**kwargs):
        return [], {}

    monkeypatch.setattr(worker, "_fetch_anchor_candidates", _anchor)
    monkeypatch.setattr(
        worker, "_query_backends", AsyncMock(return_value=([_fetch_candidate()], {"sql_timeline": 1}))
    )

    fused: List[Dict[str, Any]] = []

    def _belief_fuse(**kwargs):
        fused.append(kwargs)
        return MemoryBundleV1(rendered="belief ok", stats=MemoryBundleStatsV1()), []

    monkeypatch.setattr(worker, "pcr_fuse_belief_candidates", _belief_fuse)
    monkeypatch.setattr(worker, "get_substrate_store", lambda: "fake-store")
    return fused


def _q(deadline_ms: int | None = None) -> RecallQueryV1:
    return RecallQueryV1(
        fragment="tell me about continuity and self-modeling",
        profile="chat.belief.semantic.v1",
        recall_phase="purposeful",
        retrieval_intent="semantic",
        session_id="sess-pcr-timing",
        deadline_ms=deadline_ms,
    )


def _blocking(seconds: float, result):
    """SYNC stub: blocks its thread. Freezes the event loop if called on it."""
    calls: List[Dict[str, Any]] = []

    def _fn(*args, **kwargs):
        calls.append({"on_loop_thread": threading.current_thread() is threading.main_thread()})
        time.sleep(seconds)
        return list(result)

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


async def _run_with_ticker(coro):
    ticks = 0
    stop = False

    async def _tick():
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.ensure_future(_tick())
    started = time.perf_counter()
    try:
        result = await coro
    finally:
        stop = True
        await t
    return result, time.perf_counter() - started, ticks


def _ap_frag() -> Dict[str, Any]:
    return {"id": "ap-1", "source": "active_packet", "snippet": "belief: continuity matters", "score": 0.8}


def _cr_frag() -> Dict[str, Any]:
    return {"id": "cr-1", "source": "concept_region", "snippet": "Orion: continuity", "score": 0.7}


# ── deadline ────────────────────────────────────────────────────────────────


def test_slow_concept_region_is_cut_at_the_deadline_and_other_candidates_kept(purposeful, monkeypatch) -> None:
    async def _ap(q, *, pool, settings):
        return [_ap_frag()]

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    blocking = _blocking(1.5, [_cr_frag()])
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", blocking)

    (bundle, decision), elapsed, ticks = asyncio.run(
        _run_with_ticker(worker.process_recall(_q(deadline_ms=500), corr_id="c-cr-slow"))
    )

    assert blocking.calls, "concept_region stub was not reached"
    assert not any(c["on_loop_thread"] for c in blocking.calls)
    assert elapsed < 1.0  # budget 400ms (80% of 500), not the 1.5s stub
    assert ticks >= 20
    assert decision.deadline_hit is True
    assert bundle.rendered == "belief ok"
    ids = [c.get("id") for c in purposeful[0]["candidates"]]
    assert "fetch-1" in ids
    assert "ap-1" in ids
    assert "cr-1" not in ids
    assert decision.backend_counts.get("active_packet") == 1
    assert "concept_region" not in decision.backend_counts


def test_slow_active_packet_is_cut_at_the_deadline_and_concept_region_kept(purposeful, monkeypatch) -> None:
    async def _slow_ap(q, *, pool, settings):
        await asyncio.sleep(10)
        return [_ap_frag()]

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _slow_ap)
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", lambda q, *, store, **_kw: [_cr_frag()])

    started = time.perf_counter()
    _bundle, decision = asyncio.run(worker.process_recall(_q(deadline_ms=500), corr_id="c-ap-slow"))
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert decision.deadline_hit is True
    ids = [c.get("id") for c in purposeful[0]["candidates"]]
    assert "fetch-1" in ids and "cr-1" in ids and "ap-1" not in ids


def test_collectors_skipped_when_deadline_already_passed(purposeful, monkeypatch) -> None:
    """Fetch used the whole budget: the collectors must not start at all."""

    async def _slow_backends(*a, **k):
        await asyncio.sleep(10)
        return [], {}

    monkeypatch.setattr(worker, "_query_backends", _slow_backends)
    ap_calls: list = []

    async def _ap(q, *, pool, settings):
        ap_calls.append(1)
        return [_ap_frag()]

    cr_calls: list = []
    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    monkeypatch.setattr(
        worker, "fetch_concept_region_fragment_and_reinforce", lambda q, *, store, **_kw: cr_calls.append(1) or []
    )

    _bundle, decision = asyncio.run(worker.process_recall(_q(deadline_ms=250), corr_id="c-pcr-skip"))
    assert decision.deadline_hit is True
    assert ap_calls == [] and cr_calls == []
    assert decision.timings_ms["pcr_active_packet"] == 0
    assert decision.timings_ms["pcr_concept_region"] == 0


def test_no_deadline_hit_when_collectors_finish(purposeful, monkeypatch) -> None:
    async def _ap(q, *, pool, settings):
        return [_ap_frag()]

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", lambda q, *, store, **_kw: [_cr_frag()])

    _bundle, decision = asyncio.run(worker.process_recall(_q(deadline_ms=5000), corr_id="c-pcr-ok"))
    assert decision.deadline_hit is False
    ids = [c.get("id") for c in purposeful[0]["candidates"]]
    assert ids.index("ap-1") < ids.index("cr-1")  # same order as before the fix
    assert decision.backend_counts.get("active_packet") == 1
    assert decision.backend_counts.get("concept_region") == 1


def test_a_failing_collector_does_not_fail_the_recall(purposeful, monkeypatch) -> None:
    async def _boom(q, *, pool, settings):
        raise RuntimeError("pg down")

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _boom)
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", lambda q, *, store, **_kw: [_cr_frag()])

    bundle, decision = asyncio.run(worker.process_recall(_q(deadline_ms=5000), corr_id="c-pcr-fail"))
    assert bundle.rendered == "belief ok"
    assert decision.deadline_hit is False
    ids = [c.get("id") for c in purposeful[0]["candidates"]]
    assert "cr-1" in ids and "ap-1" not in ids


# ── timings_ms ──────────────────────────────────────────────────────────────


def test_timings_ms_has_the_new_keys(purposeful, monkeypatch) -> None:
    async def _ap(q, *, pool, settings):
        return [_ap_frag()]

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", lambda q, *, store, **_kw: [_cr_frag()])

    _bundle, decision = asyncio.run(worker.process_recall(_q(), corr_id="c-pcr-keys"))
    for key in (*_TOP_LEVEL_STAGES, "pcr_active_packet", "pcr_concept_region", "total"):
        assert key in decision.timings_ms, key


def test_timed_stages_add_up_to_total(purposeful, monkeypatch) -> None:
    """The 2026-09-30 bug: ~9.3s of a 9.5s recall was in no timings_ms key.
    With slow-but-in-budget PCR collectors and eligible-count, the stages
    must now account for nearly all of total."""

    async def _ap(q, *, pool, settings):
        await asyncio.sleep(0.15)
        return [_ap_frag()]

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", _blocking(0.3, [_cr_frag()]))

    async def _count(_pool, **_k):
        await asyncio.sleep(0.2)
        return 3

    monkeypatch.setattr(worker, "count_eligible_active", _count)
    monkeypatch.setattr(worker, "_recall_pg_pool", object())

    _bundle, decision = asyncio.run(worker.process_recall(_q(deadline_ms=5000), corr_id="c-pcr-sum"))
    t = decision.timings_ms
    assert t["total"] >= 450
    assert t["pcr_collectors"] >= 280
    assert t["pcr_concept_region"] >= 280
    assert t["pcr_active_packet"] >= 140
    assert t["eligible_count"] >= 180
    # Relative bound: per-stage values are floored to whole ms and wall time
    # jitters on a loaded CI runner, so the remainder is not exactly 0. Before
    # the fix it was ~100% of total (every slow stage here was untimed).
    untimed = t["total"] - sum(t[k] for k in _TOP_LEVEL_STAGES)
    assert untimed >= -len(_TOP_LEVEL_STAGES), (untimed, t)  # rounding only
    assert untimed <= 0.2 * t["total"], (untimed, t)


# ── get_substrate_store concurrency ─────────────────────────────────────────


def test_get_substrate_store_builds_once_under_a_race(monkeypatch) -> None:
    import app.substrate_store as mod

    monkeypatch.setattr(mod, "_STORE", None)
    builds: list = []

    def _slow_build(**_kw):
        builds.append(1)
        time.sleep(0.2)
        return object()

    monkeypatch.setattr(mod, "build_substrate_store_from_env", _slow_build)
    results: list = []
    threads = [threading.Thread(target=lambda: results.append(mod.get_substrate_store())) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(builds) == 1
    assert len({id(r) for r in results}) == 1


# ── boot warmup ─────────────────────────────────────────────────────────────


def test_warmup_calls_get_substrate_store_once_off_the_event_loop(monkeypatch) -> None:
    import app.substrate_store as mod

    calls: list = []

    def _fake_get(**_kw):
        calls.append(threading.current_thread() is threading.main_thread())
        return object()

    monkeypatch.setattr(mod, "get_substrate_store", _fake_get)
    assert asyncio.run(mod.warm_substrate_store()) is True
    assert calls == [False]


def test_failing_warmup_returns_false_and_does_not_raise(monkeypatch) -> None:
    import app.substrate_store as mod

    def _boom(**_kw):
        raise RuntimeError("falkor unreachable")

    monkeypatch.setattr(mod, "get_substrate_store", _boom)
    assert asyncio.run(mod.warm_substrate_store()) is False

    monkeypatch.setattr(mod, "get_substrate_store", lambda **_kw: None)
    assert asyncio.run(mod.warm_substrate_store()) is False


def test_warmup_timeout_does_not_wait_for_the_thread(monkeypatch) -> None:
    import app.substrate_store as mod

    monkeypatch.setattr(mod, "get_substrate_store", _blocking(1.0, []))

    async def _go():
        started = time.perf_counter()
        ok = await mod.warm_substrate_store(timeout_s=0.1)
        return ok, time.perf_counter() - started

    ok, elapsed = asyncio.run(_go())
    assert ok is False
    assert elapsed < 0.5


class _FakeRabbit:
    def __init__(self, *a, **k):
        self.bus = object()
        self.handler = None

    async def start_background(self):
        return None

    async def stop(self):
        return None


def _patch_lifespan_deps(monkeypatch, main_mod) -> None:
    monkeypatch.setattr(main_mod, "Rabbit", _FakeRabbit)
    monkeypatch.setattr(main_mod, "chassis_cfg", lambda: None)
    monkeypatch.setattr(main_mod.settings, "RECALL_RDF_ENDPOINT_URL", "")
    monkeypatch.setattr(main_mod.settings, "RECALL_ENABLE_CARDS", False)


def test_lifespan_starts_warmup_once_and_boots_even_if_it_fails(monkeypatch) -> None:
    import app.main as main_mod

    _patch_lifespan_deps(monkeypatch, main_mod)
    monkeypatch.setattr(main_mod.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(main_mod.settings, "RECALL_CONCEPT_REGION_ENABLED", True)

    calls: list = []

    def _boom(**_kw):
        calls.append(threading.current_thread() is threading.main_thread())
        raise RuntimeError("falkor unreachable")

    import app.substrate_store as mod

    monkeypatch.setattr(mod, "get_substrate_store", _boom)

    async def _go():
        async with main_mod.lifespan(main_mod.app):
            task = main_mod.app.state.substrate_store_warmup
            assert task is not None
            return await task

    assert asyncio.run(_go()) is False
    assert calls == [False]


def test_lifespan_skips_warmup_when_concept_region_disabled(monkeypatch) -> None:
    import app.main as main_mod

    _patch_lifespan_deps(monkeypatch, main_mod)
    monkeypatch.setattr(main_mod.settings, "RECALL_PCR_ENABLED", True)
    monkeypatch.setattr(main_mod.settings, "RECALL_CONCEPT_REGION_ENABLED", False)

    import app.substrate_store as mod

    calls: list = []
    monkeypatch.setattr(mod, "get_substrate_store", lambda **_kw: calls.append(1))

    async def _go():
        async with main_mod.lifespan(main_mod.app):
            return main_mod.app.state.substrate_store_warmup

    assert asyncio.run(_go()) is None
    assert calls == []


# ── review round 1 (2026-09-30) ─────────────────────────────────────────────


class _HydratedStore:
    def __init__(self, ok: bool | None = True, nodes: int = 5):
        self.last_hydrate_ok = ok
        self.last_hydrate_node_count = nodes


def test_failed_hydrate_is_not_cached_and_warmup_does_not_log_warmed(monkeypatch, caplog) -> None:
    """Finding 1: Falkor down at boot -> hydrate swallowed the error and the
    empty store was cached for the process lifetime, logged as 'warmed'."""
    import app.substrate_store as mod

    builds: list = []
    monkeypatch.setattr(mod, "_build_store", lambda: builds.append(1) or _HydratedStore(ok=False, nodes=0))
    with caplog.at_level("INFO", logger=mod.logger.name):
        assert asyncio.run(mod.warm_substrate_store()) is False
    assert mod._STORE is None
    assert "recall_substrate_store_warmed" not in caplog.text
    assert "recall_substrate_store_warmup_failed reason=hydrate_failed" in caplog.text
    assert mod.last_failure_reason() == "hydrate_failed"


def test_empty_hydrate_is_not_cached(monkeypatch, caplog) -> None:
    import app.substrate_store as mod

    monkeypatch.setattr(mod, "_build_store", lambda: _HydratedStore(ok=True, nodes=0))
    with caplog.at_level("INFO", logger=mod.logger.name):
        assert asyncio.run(mod.warm_substrate_store()) is False
    assert mod._STORE is None
    assert "reason=hydrate_empty" in caplog.text
    assert "recall_substrate_store_warmed" not in caplog.text


def test_failed_build_backs_off_then_retries_and_caches(monkeypatch) -> None:
    import app.substrate_store as mod

    results = [_HydratedStore(ok=False, nodes=0), _HydratedStore(ok=True, nodes=7)]
    builds: list = []

    def _build():
        builds.append(1)
        return results[len(builds) - 1]

    monkeypatch.setattr(mod, "_build_store", _build)
    clock = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])

    assert mod.get_substrate_store() is None
    assert len(builds) == 1
    clock[0] += mod.RETRY_BACKOFF_BASE_S / 2
    assert mod.get_substrate_store() is None  # inside backoff: no rebuild
    assert len(builds) == 1
    clock[0] += mod.RETRY_BACKOFF_BASE_S
    store = mod.get_substrate_store()
    assert store is results[1] and len(builds) == 2
    assert mod.get_substrate_store() is store  # cached now


def test_backoff_doubles_and_caps(monkeypatch) -> None:
    import app.substrate_store as mod

    monkeypatch.setattr(mod, "_build_store", lambda: _HydratedStore(ok=False, nodes=0))
    clock = [0.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    waits = []
    for _ in range(12):
        mod.get_substrate_store()
        waits.append(mod._next_retry_at - clock[0])
        clock[0] = mod._next_retry_at
    assert waits[:3] == [5.0, 10.0, 20.0]
    assert max(waits) == mod.RETRY_BACKOFF_MAX_S


def test_store_without_hydrate_signal_is_accepted(monkeypatch) -> None:
    import app.substrate_store as mod

    plain = object()  # e.g. InMemorySubstrateGraphStore: no signal
    monkeypatch.setattr(mod, "_build_store", lambda: plain)
    assert mod.get_substrate_store() is plain


def test_request_path_gives_up_on_a_held_lock(monkeypatch) -> None:
    """Finding 2: a request thread must not wait indefinitely while another
    thread is mid-hydration."""
    import app.substrate_store as mod

    builds: list = []
    monkeypatch.setattr(mod, "_build_store", lambda: builds.append(1) or _HydratedStore())
    assert mod._STORE_LOCK.acquire()
    try:
        started = time.perf_counter()
        assert mod.get_substrate_store(lock_timeout_s=0.1) is None
        assert time.perf_counter() - started < 0.5
        assert builds == []
    finally:
        mod._STORE_LOCK.release()
    assert mod.REQUEST_LOCK_TIMEOUT_S <= 5.0


def test_recall_builder_passes_socket_timeouts(monkeypatch) -> None:
    import app.substrate_store as mod

    seen: list = []
    monkeypatch.setattr(mod, "build_substrate_store_from_env", lambda **kw: seen.append(kw) or _HydratedStore())
    mod.get_substrate_store()
    assert seen == [
        {
            "falkor_socket_timeout_s": mod.FALKOR_SOCKET_TIMEOUT_S,
            "falkor_socket_connect_timeout_s": mod.FALKOR_SOCKET_CONNECT_TIMEOUT_S,
        }
    ]


def _concept_region_split(monkeypatch, *, fetch_block_s: float):
    """Real fetch_concept_region_fragment_and_reinforce with its two halves
    stubbed: a blocking fetch that matches one node, and a recording
    reinforce."""
    import app.collectors.concept_region as cr

    reinforced: list = []
    fetch_done = threading.Event()

    def _fetch(query, *, store, limit_nodes, limit_edges):
        time.sleep(fetch_block_s)
        fetch_done.set()
        return [{"id": f"{cr._NODE_FRAGMENT_ID_PREFIX}concept-a", "source": "concept_region", "snippet": "x", "score": 0.7}]

    monkeypatch.setattr(cr, "fetch_concept_region_fragment", _fetch)
    monkeypatch.setattr(cr, "reinforce_matched_concepts", lambda ids, *, store: reinforced.append(list(ids)) or 1)
    monkeypatch.setattr(worker, "fetch_concept_region_fragment_and_reinforce", cr.fetch_concept_region_fragment_and_reinforce)
    return reinforced, fetch_done


def test_deadline_cut_concept_region_does_not_reinforce(purposeful, monkeypatch) -> None:
    """Finding 3: the abandoned thread used to write the activation bump for
    fragments the recall had already dropped."""

    async def _ap(q, *, pool, settings):
        return []

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    reinforced, fetch_done = _concept_region_split(monkeypatch, fetch_block_s=0.8)

    (_bundle, decision), elapsed, _ticks = asyncio.run(
        _run_with_ticker(worker.process_recall(_q(deadline_ms=500), corr_id="c-cr-abandon"))
    )
    # asyncio.run waited for the executor thread, so the fetch has finished.
    assert fetch_done.is_set()
    assert elapsed < 0.8
    assert decision.deadline_hit is True
    assert reinforced == []


def test_in_time_concept_region_still_reinforces(purposeful, monkeypatch) -> None:
    async def _ap(q, *, pool, settings):
        return []

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    reinforced, _fetch_done = _concept_region_split(monkeypatch, fetch_block_s=0.0)
    _bundle, decision = asyncio.run(worker.process_recall(_q(deadline_ms=5000), corr_id="c-cr-reinforce"))
    assert decision.deadline_hit is False
    assert reinforced == [["concept-a"]]


def test_recall_during_warmup_hydration_is_cut_at_the_deadline(purposeful, monkeypatch) -> None:
    """NIT 4: the real get_substrate_store with the real lock held by a
    mid-hydration warmup. The recall's concept_region thread waits on the
    lock; the recall itself returns at its deadline and keeps the rest."""
    import app.substrate_store as mod

    monkeypatch.setattr(worker, "get_substrate_store", mod.get_substrate_store)
    building = threading.Event()
    builds: list = []
    built_store = _HydratedStore(ok=True, nodes=3)

    def _slow_build():
        builds.append(1)
        building.set()
        time.sleep(1.0)
        return built_store

    monkeypatch.setattr(mod, "_build_store", _slow_build)

    async def _ap(q, *, pool, settings):
        return [_ap_frag()]

    monkeypatch.setattr(worker, "fetch_active_packet_fragments", _ap)
    cr_seen: list = []
    monkeypatch.setattr(
        worker,
        "fetch_concept_region_fragment_and_reinforce",
        lambda q, *, store, abandoned=None: cr_seen.append((store, abandoned.is_set())) or [_cr_frag()],
    )

    warm = threading.Thread(target=lambda: mod.get_substrate_store(lock_timeout_s=5.0))
    warm.start()
    assert building.wait(2.0)

    (bundle, decision), elapsed, ticks = asyncio.run(
        _run_with_ticker(worker.process_recall(_q(deadline_ms=500), corr_id="c-cr-warmup-race"))
    )
    warm.join(5.0)

    assert elapsed < 0.9
    assert ticks >= 20
    assert decision.deadline_hit is True
    ids = [c.get("id") for c in purposeful[0]["candidates"]]
    assert "fetch-1" in ids and "ap-1" in ids and "cr-1" not in ids
    assert builds == [1]  # the racing request did not build a second store
    # The recall's thread got the warmed store once the lock freed (1.0s <
    # REQUEST_LOCK_TIMEOUT_S), already marked abandoned.
    assert cr_seen == [(built_store, True)]
    assert mod._STORE is built_store
