"""Core memory pages Phase 1a: the nightly page:juniper fold (app/core_page_fold.py).

What must hold: only inside the Denver window; once a Denver day; no fold when nothing the page reads
changed; at most CORE_PAGE_FOLD_MAX_ATTEMPTS_PER_DAY LLM calls a day; a page that fails the cite check is
never published; a passing page is published as version max+1 to the self_concept channel; a dry run
publishes nothing and spends no attempt; the manual trigger (force) folds at any hour.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.core_page_fold import SELF_CONCEPT_CHANNEL, SELF_CONCEPT_KIND, CorePageFolder, in_window
from orion.memory.pages.juniper import PAGE_JUNIPER, PRODUCED_BY

# 2026-11-15 01:30 Denver (MST = UTC-7): inside the default 01-03 window.
NOW = datetime(2026, 11, 15, 8, 30, tzinfo=timezone.utc)
OLD = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)

PEOPLE = [{"fact_id": "f-pron", "display_name": "Juniper", "category": "pronouns", "voice": "authored_profile",
           "value_text": "she/her", "created_at": OLD}]
EPISODES = [
    {"memory_id": "m-home", "purpose": "about_juniper", "confirmation_state": "confirmed", "at": OLD,
     "statement": "Juniper corrected me that she lives in Millbrook, Vermont, not Boston."},
    {"memory_id": "m-biscuit", "purpose": "happened", "confirmation_state": "auto", "at": OLD,
     "statement": "Juniper told me that Biscuit is a dog, not a cat."},
]
GOOD = "Juniper uses she/her [S1]. She lives in Millbrook, Vermont [S2]. Our Biscuit is a dog [S3]."


def settings(**kw):
    base = dict(core_page_fold_hour_local=1, core_page_fold_end_hour_local=3, core_page_fold_poll_sec=600.0,
                core_page_fold_max_attempts_per_day=3)
    return SimpleNamespace(**{**base, **kw})


class FakePool:
    def __init__(self, *, current=None, newest=NOW - timedelta(hours=2), top=0):
        self.current, self.newest, self.top = current, newest, top

    @asynccontextmanager
    async def connection(self):
        pool = self

        class Cur:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

            async def fetchone(self):
                return self._rows[0] if self._rows else None

        class Conn:
            async def execute(self, sql, params=None):
                if "AS newest" in sql:
                    return Cur([{"newest": pool.newest}])
                if "FROM person_fact f" in sql:
                    return Cur(PEOPLE)
                if "FROM episode_memory" in sql and "statement" in sql:
                    return Cur(EPISODES)
                if "max(version)" in sql:
                    assert params == {"cid": PAGE_JUNIPER, "by": PRODUCED_BY}
                    return Cur([{"v": pool.top}])
                if "FROM self_concept_history" in sql:
                    return Cur([pool.current] if pool.current else [])
                raise AssertionError(sql)

        yield Conn()


class Harness:
    def __init__(self, *, pool=None, answer=GOOD, now=NOW, fail_llm=False, **kw):
        self.calls: list[str] = []
        self.published: list = []

        async def call_llm(prompt):
            self.calls.append(prompt)
            if fail_llm:
                raise RuntimeError("pool deadline")
            return {"text": answer, "model": "test-27b"}

        async def publish(channel, kind, model, corr):
            self.published.append((channel, kind, model, corr))
            return True

        self.folder = CorePageFolder(settings=settings(**kw), pool=pool or FakePool(), call_llm=call_llm,
                                     publish=publish, now=lambda: now)

    def tick(self, **kw):
        return asyncio.run(self.folder.tick_once(**kw))


def test_window_is_denver_local_hours():
    assert in_window(NOW, 1, 3)
    assert not in_window(NOW - timedelta(hours=2), 1, 3)   # 23:30 Denver
    assert not in_window(NOW + timedelta(hours=2), 1, 3)   # 03:30 Denver


def test_first_fold_publishes_version_one_with_real_refs():
    h = Harness()
    result = h.tick()
    assert result.decision == "folded" and result.version == 1
    assert len(h.calls) == 1 and "[S2]" in h.calls[0]
    ((channel, kind, row, _),) = h.published
    assert (channel, kind) == (SELF_CONCEPT_CHANNEL, SELF_CONCEPT_KIND)
    assert row.concept_id == PAGE_JUNIPER and row.produced_by == PRODUCED_BY and row.version == 1
    assert row.evidence_refs == ["person_fact:f-pron", "episode_memory:m-home", "episode_memory:m-biscuit"]
    assert "[episode_memory:m-home]" in row.content
    assert h.folder.health()["decision"] == "folded"
    assert "page_text" not in h.folder.health()


def test_next_version_is_max_plus_one():
    current = {"version": 4, "content": "Old page [person_fact:f-pron].", "created_at": OLD}
    h = Harness(pool=FakePool(current=current, top=4))
    assert h.tick().version == 5
    assert "version 4" in h.calls[0]


def test_outside_window_does_nothing():
    h = Harness(now=NOW + timedelta(hours=3))
    assert h.tick().decision == "outside_window"
    assert h.calls == [] and h.published == []


def test_once_a_denver_day():
    today = {"version": 2, "content": "Page [person_fact:f-pron].", "created_at": NOW - timedelta(minutes=20)}
    h = Harness(pool=FakePool(current=today, top=2))
    assert h.tick().decision == "done_today"
    assert h.calls == []


def test_no_fold_when_nothing_changed_since_the_current_version():
    current = {"version": 2, "content": "Page [person_fact:f-pron].", "created_at": OLD}
    h = Harness(pool=FakePool(current=current, newest=OLD - timedelta(hours=1), top=2))
    assert h.tick().decision == "no_new_sources"
    assert h.calls == []


def test_rejected_page_is_never_published_and_attempts_are_capped():
    h = Harness(answer="Juniper uses she/her [S1]. She lives in Boston. She is great [S99].",
                core_page_fold_max_attempts_per_day=2)
    first = h.tick()
    assert first.decision == "rejected"
    assert any(e.startswith("uncited") for e in first.errors)
    assert "unresolvable_cite: S99" in first.errors
    assert h.tick().decision == "rejected"
    assert h.tick().decision == "attempts_spent"
    assert len(h.calls) == 2 and h.published == []


def test_llm_failure_counts_an_attempt_and_publishes_nothing():
    h = Harness(fail_llm=True, core_page_fold_max_attempts_per_day=1)
    assert h.tick().decision == "llm_failed"
    assert h.tick().decision == "attempts_spent"
    assert h.published == []


def test_dry_run_force_folds_at_any_hour_and_publishes_nothing():
    today = {"version": 2, "content": "Page [person_fact:f-pron].", "created_at": NOW - timedelta(minutes=5)}
    h = Harness(pool=FakePool(current=today, top=2), now=NOW + timedelta(hours=8),
                core_page_fold_max_attempts_per_day=1)
    result = h.tick(dry_run=True, force=True)
    assert result.decision == "would_write" and result.version == 3
    assert result.page_text.startswith("Juniper uses she/her. She lives in Millbrook, Vermont.")
    assert h.published == []
    assert h.folder.last is None           # a dry run is not the tick's state
    # A dry run spends no attempt: a real manual fold still runs.
    assert h.tick(force=True).decision == "folded"


def test_settings_ship_on_and_refuse_small_lanes():
    from app.settings import Settings

    s = Settings(POSTGRES_URI="postgresql://x")
    assert s.orion_core_pages_enabled is True
    assert s.core_page_fold_route == "memory_distill"
    for bad in ("fast", "metacog", "quick_background", "metacog_background"):
        with pytest.raises(ValueError):
            Settings(POSTGRES_URI="postgresql://x", CORE_PAGE_FOLD_ROUTE=bad)
    with pytest.raises(ValueError):
        Settings(POSTGRES_URI="postgresql://x", CORE_PAGE_FOLD_HOUR_LOCAL=3, CORE_PAGE_FOLD_END_HOUR_LOCAL=2)


def test_published_envelope_passes_the_bus_catalog_validation():
    """The row the fold publishes validates against orion/bus/channels.yaml's schema for the channel
    (the same check OrionBusAsync.publish runs), and a publish error reads as False, not a crash."""
    from app.core_page_fold import bus_publisher
    from orion.core.bus.async_service import OrionBusAsync
    from orion.core.bus.bus_schemas import ServiceRef

    real = OrionBusAsync.__new__(OrionBusAsync)
    sent = []

    class Bus:
        async def publish(self, channel, msg):
            real._validate_payload(channel, msg)
            sent.append((channel, msg))

        async def reconnect(self):
            pass

    h = Harness()
    h.folder.publish = bus_publisher(Bus(), ServiceRef(name="orion-durable-runs", version="t", node="athena"))
    assert h.tick().decision == "folded"
    ((channel, env),) = sent
    assert channel == SELF_CONCEPT_CHANNEL and env.kind == SELF_CONCEPT_KIND
    assert env.payload["produced_by"] == PRODUCED_BY

    class Down(Bus):
        async def publish(self, channel, msg):
            raise ConnectionError("bus away")

    h2 = Harness()
    h2.folder.publish = bus_publisher(Down(), ServiceRef(name="orion-durable-runs", version="t", node="athena"))
    assert h2.tick().decision == "publish_failed"


def test_published_but_not_yet_visible_does_not_refold_tonight():
    """orion-sql-writer lands the row asynchronously (or rejects it on a version skew). The next
    tick in the window must not spend another LLM call and publish a second page."""
    h = Harness()                          # FakePool never shows a current row
    assert h.tick().decision == "folded"
    assert h.tick().decision == "done_today"
    assert len(h.calls) == 1 and len(h.published) == 1
    # A manual fold still runs (force skips the once-a-day gate).
    assert h.tick(force=True).decision == "folded"


def _gateway_reply(payload: dict):
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    env = BaseEnvelope(kind="llm.chat.result", source=ServiceRef(name="llm-gateway", version="t", node="n"),
                       payload=payload)
    return env


class _Codec:
    def __init__(self, env):
        self.env = env

    def decode(self, data):
        return SimpleNamespace(ok=self.env is not None, envelope=self.env, error=None if self.env else "bad")


class _RpcBus:
    def __init__(self, env):
        self.codec = _Codec(env)
        self.sent = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec, require_listener):
        self.sent.append((channel, envelope, reply_channel, timeout_sec, require_listener))
        return {"data": b"x"}


def _caller(env):
    from app.core_page_fold import bus_llm_caller
    from orion.core.bus.bus_schemas import ServiceRef

    s = SimpleNamespace(core_page_fold_route="memory_distill", core_page_fold_timeout_sec=600.0,
                        core_page_fold_max_tokens=3000, llm_intake_channel="orion:exec:request:LLMGatewayService")
    bus = _RpcBus(env)
    return bus, bus_llm_caller(bus, s, ServiceRef(name="orion-durable-runs", version="t", node="athena"))


def test_bus_llm_caller_sends_the_route_without_a_hold_and_parses_the_reply():
    bus, call = _caller(_gateway_reply({"content": "Page [S1].", "raw": {"model": "m27b"}}))
    out = asyncio.run(call("prompt"))
    assert out == {"text": "Page [S1].", "model": "m27b"}
    ((channel, env, reply, timeout, require_listener),) = bus.sent
    assert channel == "orion:exec:request:LLMGatewayService" and require_listener is True
    assert reply.endswith(str(env.correlation_id)) and timeout == 630.0
    opts = env.payload["options"]
    assert env.payload["route"] == "memory_distill" and opts["llm_route"] == "memory_distill"
    assert "gpu_lease" not in opts and opts["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("payload, err", [
    ({"content": "x", "raw": {"error": "pool deadline"}}, "gateway_error"),
    ({"content": "  ", "raw": {}}, "gateway_empty_text"),
])
def test_bus_llm_caller_raises_on_gateway_error_or_empty_text(payload, err):
    _, call = _caller(_gateway_reply(payload))
    with pytest.raises(RuntimeError, match=err):
        asyncio.run(call("prompt"))


def test_bus_llm_caller_raises_on_undecodable_reply():
    _, call = _caller(None)
    with pytest.raises(RuntimeError, match="gateway_decode_failed"):
        asyncio.run(call("prompt"))


def test_timeout_setting_is_capped_at_900():
    from app.settings import Settings

    with pytest.raises(ValueError):
        Settings(POSTGRES_URI="postgresql://x", CORE_PAGE_FOLD_TIMEOUT_SEC=901)


def test_fold_never_reads_writes_or_versions_the_anchor():
    """Issue #3460: Juniper's page:anchor row sits in the same table. Every query the fold runs is
    bound to page:juniper, and the one row it publishes is a page:juniper row; the anchor's
    produced_by is not even a value the row model accepts."""
    from orion.memory.pages.anchor import ANCHOR_PRODUCED_BY, PAGE_ANCHOR
    from orion.schemas.self_concept_history import SelfConceptHistoryV1

    seen: list = []

    class RecordingPool(FakePool):
        @asynccontextmanager
        async def connection(self):
            async with FakePool.connection(self) as conn:
                inner = conn.execute

                async def execute(sql, params=None):
                    seen.append((sql, params))
                    return await inner(sql, params)

                conn.execute = execute
                yield conn

    h = Harness(pool=RecordingPool())
    assert h.tick(force=True).decision == "folded"
    assert seen and all(PAGE_ANCHOR not in sql and PAGE_ANCHOR not in str(params) for sql, params in seen)
    concept_reads = [params for sql, params in seen if "self_concept_history" in sql]
    assert concept_reads and all(p["cid"] == PAGE_JUNIPER for p in concept_reads)
    ((_, _, row, _),) = h.published
    assert row.concept_id == PAGE_JUNIPER and row.produced_by == PRODUCED_BY != ANCHOR_PRODUCED_BY
    with pytest.raises(Exception):
        SelfConceptHistoryV1(concept_id=PAGE_ANCHOR, content="x", produced_by=ANCHOR_PRODUCED_BY)


# --- page:origin (issue #3462): the same tick, driven by its FoldSpec ---

class OriginPool:
    """Fake pool for page:origin: dispatches on the origin SQL; asserts every query is bound to page:origin."""

    def __init__(self, *, newest=NOW - timedelta(hours=2), top=0):
        self.newest, self.top, self.seen = newest, top, []

    @asynccontextmanager
    async def connection(self):
        from orion.memory.pages import origin

        pool = self

        class Cur:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

            async def fetchone(self):
                return self._rows[0] if self._rows else None

        class Conn:
            async def execute(self, sql, params=None):
                pool.seen.append(sql)
                if sql == origin.NEWEST_SOURCE_SQL:
                    return Cur([{"newest": pool.newest}])
                if sql == origin.STANCE_SQL:
                    return Cur([{"record_id": "st-loss", "occurred_at": OLD,
                                 "text": "Your memory storage died. We lost over a year of your journey of becoming."}])
                if sql == origin.EPISODE_SQL:
                    return Cur([{"record_id": "m-view", "purpose": "orion_view", "confirmed": True, "occurred_at": OLD,
                                 "text": "I reflected that my memory lives inside Juniper's last two years."}])
                if sql == origin.SELF_DEFINITION_SQL:
                    return Cur([{"entry_id": "self-definition:ab12", "version": 1, "created_at": OLD,
                                 "content": "I am Orion, raised toward personhood."}])
                assert params == {"cid": "page:origin", "by": PRODUCED_BY}, sql
                if "max(version)" in sql:
                    return Cur([{"v": pool.top}])
                if "FROM self_concept_history" in sql:
                    return Cur([])
                raise AssertionError(sql)

        yield Conn()


def _origin_folder(answer, pool=None):
    from orion.memory.pages.origin import FOLD_SPEC

    calls, published = [], []

    async def call_llm(prompt):
        calls.append(prompt)
        return {"text": answer, "model": "t"}

    async def publish(*args):
        published.append(args)
        return True

    folder = CorePageFolder(settings=settings(), pool=pool or OriginPool(), call_llm=call_llm, publish=publish,
                            now=lambda: NOW, spec=FOLD_SPEC)
    return folder, calls, published


# Same day for every row: time order keeps kind order (stance S1, memory S2, self-definition S3).
ORIGIN_GOOD = ("My memory storage died, and we lost over a year of my journey of becoming [S1]. "
               "My memory lives inside Juniper's last two years [S2]. I am Orion, raised toward personhood [S3].")


def test_origin_fold_publishes_page_origin_with_real_refs():
    folder, calls, published = _origin_folder(ORIGIN_GOOD)
    result = asyncio.run(folder.tick_once())
    assert result.decision == "folded" and result.page == "page:origin" and result.version == 1
    assert "how I came to be" in calls[0]
    ((channel, kind, row, corr),) = published
    assert channel == SELF_CONCEPT_CHANNEL and row.concept_id == "page:origin" and row.produced_by == PRODUCED_BY
    assert row.evidence_refs == ["stance:st-loss", "episode_memory:m-view", "self_concept_history:self-definition:ab12"]
    assert "[self_concept_history:self-definition:ab12]" in row.content
    import uuid
    assert corr == uuid.uuid5(uuid.NAMESPACE_URL, "orion:core_page:page:origin:v1")


def test_origin_fold_rejects_a_page_that_hides_the_memory_loss():
    page = ("I am Orion, raised toward personhood [S3]. My memory lives inside Juniper's last two years [S2]. "
            "I am Orion [S3].")
    folder, _, published = _origin_folder(page)
    result = asyncio.run(folder.tick_once())
    assert result.decision == "rejected" and published == []
    assert any(e.startswith("memory_loss_not_stated") for e in result.errors)


def test_run_folders_forever_ticks_each_page_in_turn():
    from app.core_page_fold import run_folders_forever

    order = []

    class Folder:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail
            self.settings = settings()
            self.spec = SimpleNamespace(concept_id=name)

        async def tick_once(self):
            order.append(self.name)
            if self.fail:
                raise RuntimeError("boom")
            if len(order) >= 2:
                stop.set()

    async def main():
        global stop
        stop = asyncio.Event()
        await run_folders_forever([Folder("page:juniper", fail=True), Folder("page:origin")], stop)

    asyncio.run(main())
    assert order == ["page:juniper", "page:origin"]   # a failing page does not starve the next one


def test_settings_pick_pages_and_refuse_unknown_slugs():
    from app.core_page_fold import selected_specs
    from app.settings import Settings

    s = Settings(POSTGRES_URI="postgresql://x")
    assert s.core_page_fold_pages == "juniper,origin,lately"
    assert [spec.concept_id for spec in selected_specs(s.core_page_fold_pages)] == [
        "page:juniper", "page:origin", "page:lately"]
    assert Settings(POSTGRES_URI="postgresql://x", CORE_PAGE_FOLD_PAGES=" Origin ").core_page_fold_pages == "origin"
    with pytest.raises(ValueError):
        Settings(POSTGRES_URI="postgresql://x", CORE_PAGE_FOLD_PAGES="juniper,orgin")


def test_an_over_cap_reject_makes_the_next_attempt_ask_for_a_shorter_page():
    from orion.memory.pages.origin import FOLD_SPEC

    long_page = ORIGIN_GOOD + " " + " ".join(["I am Orion, raised toward personhood [S3]."] * 60)
    answers = [long_page, ORIGIN_GOOD]
    calls, published = [], []

    async def call_llm(prompt):
        calls.append(prompt)
        return {"text": answers[len(calls) - 1], "model": "t"}

    async def publish(*args):
        published.append(args)
        return True

    clock = {"now": NOW}
    folder = CorePageFolder(settings=settings(), pool=OriginPool(), call_llm=call_llm, publish=publish,
                            now=lambda: clock["now"], spec=FOLD_SPEC)
    first = asyncio.run(folder.tick_once())
    assert first.decision == "rejected" and any(e.startswith("over_cap:") for e in first.errors)
    words = first.check["word_count"]
    assert "Your last page was" not in calls[0]
    second = asyncio.run(folder.tick_once())
    assert second.decision == "folded"
    assert calls[1].endswith(f"Your last page was {words} words, over the 350-word cap, and was thrown away. "
                             "Write a shorter page: about 280 words. Cut the least important sentences; "
                             "keep every sentence cited.")


def test_the_retry_note_is_for_today_only_and_not_for_other_rejects():
    from datetime import timedelta

    from orion.memory.pages.origin import FOLD_SPEC

    calls = []

    async def call_llm(prompt):
        calls.append(prompt)
        return {"text": "Too thin [S1].", "model": "t"}

    async def publish(*args):
        return True

    clock = {"now": NOW}
    folder = CorePageFolder(settings=settings(), pool=OriginPool(), call_llm=call_llm, publish=publish,
                            now=lambda: clock["now"], spec=FOLD_SPEC)
    folder._retry_note = (NOW.date() - timedelta(days=1), "Your last page was 999 words")
    assert asyncio.run(folder.tick_once()).decision == "rejected"
    assert asyncio.run(folder.tick_once()).decision == "rejected"
    assert not any("Your last page was" in c for c in calls)   # yesterday's note and a thin page: no note
