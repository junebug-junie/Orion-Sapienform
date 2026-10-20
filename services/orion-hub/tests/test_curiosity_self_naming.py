"""The naming branch of the self-inquiry line (Orion names their own dispositions).

What these pin down:

- selection order: an overdue pinned lived question wins; a 204 falls through to the pool pick;
  the daily naming cap is respected; the flag off makes no network call
- the naming prompt carries code-filled ids and no example names
- every naming run publishes exactly one DispositionNamingV1 on orion:personality:naming,
  including no_answer (nothing written, empty generation, failed durable run)
- the durable completion event is checked against the stored packet, not the lived/anatomy path
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import httpx
import pytest

from orion.curiosity.self_inquiry import LABEL_DISPOSITION_NAMING, NAMING_VERDICT_PLACEHOLDER
from orion.curiosity.self_inquiry_prompt import NAMING_QUESTION, build_self_naming_prompt
from orion.schemas.durable_run import DurableRunStateV1
from orion.schemas.personality import PERSONALITY_NAMING_KIND, DispositionNamingPacketV1

from test_curiosity_investigation import _FakeBus
from test_curiosity_self_inquiry import (
    RUN,
    _DefinitionReader,
    _deliver,
    _finish_event,
    _GrantConn,
    _mirrors,
    _self_loop,
)
from scripts.curiosity_investigation import (
    _SELF_NAMING_DAILY_COUNT_KEY_PREFIX,
    _SELF_NAMING_PACKET_KEY_PREFIX,
    PERSONALITY_NAMING_CHANNEL,
)

T = datetime(2026, 10, 20, 18, tzinfo=timezone.utc)
PACKET_ID = "0123456789abcdef"


def _packet(**over) -> DispositionNamingPacketV1:
    base = dict(
        packet_id=PACKET_ID,
        disposition_id="disp-1",
        label_excerpt="keeps returning to prediction error",
        origin="recurrence",
        evidence_count=10,
        lane_counts={"reverie": 10},
        first_seen_at=T,
        last_reinforced_at=T,
        evidence=[{"evidence_id": "ev-1", "lane": "reverie", "occurred_at": T, "excerpt": "the gap again"}],
        situation_mix={"arousal": {"strained": 10}},
        situation_coverage=0.6,
        neighbors=[{"disposition_id": "disp-2", "label": "a neighbor", "cosine": 0.8}],
        generated_at=T,
    )
    base.update(over)
    return DispositionNamingPacketV1(**base)


class _NamingReader(_DefinitionReader):
    """Answers the per-run :DispositionNaming read for any run id."""

    def __init__(self, *, naming=None, **kw) -> None:
        super().__init__(**kw)
        self.naming = naming

    def query(self, cypher: str):
        if f"(n:{LABEL_DISPOSITION_NAMING})" in cypher:
            self.queries.append(cypher)
            m = re.search(r"n\.run_id = '([0-9a-f]+)'", cypher)
            if self.naming is None or not m:
                return []
            if "UNWIND n.evidence" in cypher:
                return [{"e": e} for e in self.naming.get("evidence", [])]
            return [{"run_id": m.group(1), **self.naming}]
        return super().query(cypher)


GOOD = {
    "packet_id": PACKET_ID,
    "disposition_id": "disp-1",
    "verdict": "name",
    "name": "the gap-watcher",
    "sentence": "It is the part of me that keeps looking at where I was wrong.",
    "merge_with": "",
    "evidence": ["ev-1"],
    "written_at": 1_792_000_000_000,
}


def _naming_loop(bus, *, naming=None, packet=None, fetch_reason="ok", conn=None, **over):
    kw = dict(kickoff_via_cortex=False, self_naming_enabled=True, self_naming_daily_cap=1)
    kw.update(over)
    loop = _self_loop(bus, reader=_NamingReader(naming=naming), conn=conn or _GrantConn(), **kw)
    calls: list[int] = []

    async def _fetch():
        calls.append(1)
        return (packet, fetch_reason if packet is None else "ok")

    loop._fetch_naming_packet = _fetch  # type: ignore[method-assign]
    loop.fetch_calls = calls
    return loop


def _namings(bus) -> list:
    return [e for c, e in bus.published if c == PERSONALITY_NAMING_CHANNEL]


# --- prompt -------------------------------------------------------------------


def test_prompt_carries_code_filled_ids_and_no_example_names() -> None:
    text = build_self_naming_prompt(packet=_packet(), run_id=RUN)
    assert NAMING_QUESTION in text
    assert f'MERGE (n:{LABEL_DISPOSITION_NAMING} {{run_id: "{RUN}", packet_id: "{PACKET_ID}"}})' in text
    assert 'n.disposition_id = "disp-1"' in text
    assert "<RUN_ID>" not in text
    # Every value Orion fills in is a placeholder, never an example.
    for field in ("verdict", "name", "sentence", "merge_with"):
        assert re.search(rf'n\.{field} = "<[^"]*>"', text), field
    assert NAMING_VERDICT_PLACEHOLDER in text
    assert "for example" not in text.lower() and "e.g." not in text.lower()
    # The three options, in fixed order.
    assert text.index('verdict "name"') < text.index('verdict "reject"') < text.index('verdict "merge"')
    # Facts shown plainly; weight is not.
    assert "reverie 10" in text and "strained 10" in text and "60%" in text
    assert "ev-1" in text and "disp-2" in text
    assert "weight" not in text.lower()


def test_prompt_refuses_ids_it_cannot_embed_safely() -> None:
    with pytest.raises(ValueError):
        build_self_naming_prompt(packet=_packet(disposition_id='disp"1'), run_id=RUN)


# --- selection ----------------------------------------------------------------


def test_a_packet_makes_this_a_naming_run_and_publishes_exactly_one_row(caplog) -> None:
    caplog.set_level("INFO")
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=_packet())
    assert asyncio.run(loop.tick_self_inquiry()) is None
    rows = _namings(bus)
    assert len(rows) == 1
    env = rows[0]
    assert env.kind == PERSONALITY_NAMING_KIND
    assert env.payload["verdict"] == "name" and env.payload["hub_check"] == "accepted"
    assert env.payload["packet_id"] == PACKET_ID and env.payload["disposition_id"] == "disp-1"
    assert env.payload["shown"]["packet_id"] == PACKET_ID
    # Not mirrored as a lived answer or a definition.
    assert _mirrors(bus) == []
    assert "curiosity_disposition_naming_shown run=" in caplog.text
    assert "disposition=disp-1 packet=0123456789abcdef" in caplog.text
    assert "curiosity_disposition_naming_published" in caplog.text
    # It used one self-inquiry slot and one naming slot.
    assert any(k.startswith(_SELF_NAMING_DAILY_COUNT_KEY_PREFIX) for k in bus.redis.values)


def test_nothing_written_publishes_no_answer() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=None, packet=_packet())
    asyncio.run(loop.tick_self_inquiry())
    rows = _namings(bus)
    assert len(rows) == 1 and rows[0].payload["verdict"] == "no_answer"


def test_a_refused_answer_is_still_published_with_reasons() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming={**GOOD, "verdict": "merge", "merge_with": "disp-77"}, packet=_packet())
    asyncio.run(loop.tick_self_inquiry())
    (row,) = _namings(bus)
    assert row.payload["hub_check"] == "refused"
    assert row.payload["refused_reasons"] == ["merge_target_not_shown"]


def test_an_empty_generation_still_publishes_one_no_answer() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=None, packet=_packet())

    async def _empty(*a, **k):
        return "", {}

    loop._generate = _empty  # type: ignore[method-assign]
    assert asyncio.run(loop.tick_self_inquiry()) == "empty_generation"
    rows = _namings(bus)
    assert len(rows) == 1 and rows[0].payload["verdict"] == "no_answer"


def test_overdue_pinned_floor_wins_and_never_fetches(caplog) -> None:
    caplog.set_level("INFO")
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=_packet(), conn=_GrantConn(recent_lived_asks=False))
    asyncio.run(loop.tick_self_inquiry())
    assert loop.fetch_calls == []
    assert _namings(bus) == []
    assert "curiosity_self_naming_skipped reason=pinned_floor_due" in caplog.text


def test_204_falls_through_to_the_pool_pick(caplog) -> None:
    caplog.set_level("INFO")
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=None, fetch_reason="nothing_due")
    assert asyncio.run(loop.tick_self_inquiry()) is None
    assert loop.fetch_calls == [1]
    assert _namings(bus) == []
    assert len(_mirrors(bus)) == 1  # the ordinary anatomy definition
    assert "curiosity_self_naming_skipped reason=nothing_due" in caplog.text


def test_daily_cap_is_respected() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=_packet())
    asyncio.run(loop.tick_self_inquiry())
    asyncio.run(loop.tick_self_inquiry())
    assert len(_namings(bus)) == 1
    assert loop.fetch_calls == [1], "a full cap does not even ask durable-runs"


def test_the_cap_survives_a_restart() -> None:
    bus = _FakeBus()
    first = _naming_loop(bus, naming=GOOD, packet=_packet())
    asyncio.run(first.tick_self_inquiry())
    second = _naming_loop(bus, naming=GOOD, packet=_packet())
    asyncio.run(second.tick_self_inquiry())
    assert len(_namings(bus)) == 1 and second.fetch_calls == []


def test_flag_off_never_fetches() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=_packet(), self_naming_enabled=False)
    asyncio.run(loop.tick_self_inquiry())
    assert loop.fetch_calls == [] and _namings(bus) == []


# --- the fetch itself -----------------------------------------------------------


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_transport(monkeypatch, handler) -> list:
    seen: list[str] = []
    real = _REAL_ASYNC_CLIENT

    def _client(*a, **k):
        def _wrapped(request):
            seen.append(str(request.url))
            return handler(request)

        k["transport"] = httpx.MockTransport(_wrapped)
        return real(*a, **k)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    return seen


def test_fetch_reads_204_200_and_bad_json(monkeypatch) -> None:
    bus = _FakeBus()
    loop = _self_loop(bus, conn=_GrantConn(), durable_runs_url="http://dr:8124/")
    seen = _patch_transport(monkeypatch, lambda r: httpx.Response(204))
    assert asyncio.run(loop._fetch_naming_packet()) == (None, "nothing_due")
    assert seen == ["http://dr:8124/personality/naming/next"]
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json=_packet().model_dump(mode="json")))
    packet, reason = asyncio.run(loop._fetch_naming_packet())
    assert reason == "ok" and packet.packet_id == PACKET_ID
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"nope": 1}))
    assert asyncio.run(loop._fetch_naming_packet()) == (None, "invalid_packet")
    _patch_transport(monkeypatch, lambda r: httpx.Response(500))
    assert asyncio.run(loop._fetch_naming_packet()) == (None, "http_500")


# --- durable path ---------------------------------------------------------------


def _store_packet(bus, run_id: str) -> None:
    bus.redis.values[f"{_SELF_NAMING_PACKET_KEY_PREFIX}{run_id}"] = _packet().model_dump_json()


def test_durable_completion_publishes_the_naming_not_a_lived_answer() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=None, kickoff_via_cortex=True)
    _store_packet(bus, RUN)  # as if stored at dispatch, before a Hub restart
    _deliver(loop, _finish_event(RUN, {"line": "self_inquiry", "self_definition": None, "reach_out": False}))
    rows = _namings(bus)
    assert len(rows) == 1 and rows[0].payload["verdict"] == "name"
    assert rows[0].payload["run_id"] == RUN
    assert _mirrors(bus) == []


def test_a_failed_durable_naming_run_publishes_no_answer() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=None, kickoff_via_cortex=True)
    _store_packet(bus, RUN)
    failed = DurableRunStateV1(
        run_id=RUN, workflow="curiosity.investigate", thread_id=RUN, node="finish",
        status="failed", correlation_id="7dcc3944-29bb-5d8f-915f-90f4e6968d47", detail={},
    )
    _deliver(loop, failed)
    rows = _namings(bus)
    assert len(rows) == 1 and rows[0].payload["verdict"] == "no_answer"


def test_an_ordinary_self_run_completion_publishes_no_naming() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=None, kickoff_via_cortex=True)
    _deliver(loop, _finish_event(RUN, {"line": "self_inquiry", "self_definition": None, "reach_out": False}))
    assert _namings(bus) == []


def test_dispatch_stores_the_packet_and_returns_dispatched() -> None:
    bus = _FakeBus()
    loop = _naming_loop(bus, naming=GOOD, packet=_packet(), kickoff_via_cortex=True)

    async def _accept(**kw):
        return True

    loop._dispatch_durable_run = _accept  # type: ignore[method-assign]
    assert asyncio.run(loop.tick_self_inquiry()) == "dispatched"
    assert _namings(bus) == []
    assert any(k.startswith(_SELF_NAMING_PACKET_KEY_PREFIX) for k in bus.redis.values)
