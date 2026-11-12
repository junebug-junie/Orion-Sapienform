"""Prime's order for a day letter (living Prime design #3224, Part 3): Hub's status and order endpoints.

Hub never decides when a letter is written; it answers Prime's order: build the day's brief and submit
ONE orion_day.letter run under Prime's run id and correlation id. Fake Postgres and durable-runs."""

from __future__ import annotations

import ast
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.orion_day.store import SELECT_LETTER_SQL
from orion.orion_day.tests import fixtures as fx
from orion.schemas.orion_day import (
    ORION_DAY_ORDER_PATH, MissedLetterV1, OrionDayLetterOrderResultV1, OrionDayLetterOrderV1, OrionDayLetterV1,
    OrionDayLetterSourcesV1,
)
from scripts import orion_day_order as odo

DAY = fx.LETTER_DATE                 # 2026-09-29
NOW = datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)
RUN = f"orion-day-{DAY}-1"


def order(**kw) -> OrionDayLetterOrderV1:
    base = dict(letter_date=DAY, run_id=RUN, correlation_id="corr-letter-1", prime_act_id="act_make_1",
                deadline_at=NOW + timedelta(hours=12), timeout_sec=1500.0)
    base.update(kw)
    return OrionDayLetterOrderV1(**base)


def stored_letter(day: date) -> dict:
    material = asyncio.run(fx.gather.gather_orion_day(fx.FakeConn(), DAY, now=NOW))
    letter = OrionDayLetterV1(letter_date=DAY, run_id="orion-day-old-1", window_start=material.window_start,
                              window_end=material.window_end, note_md="A letter.", carry_forward_md="- a thread",
                              material=material, sources=OrionDayLetterSourcesV1(), created_at=NOW)
    return letter.model_dump(mode="json") | {"letter_date": day}


class Conn(fx.FakeConn):
    """The gather fixture day, plus the orion_day_letter reads the desk makes."""

    def __init__(self, letters: dict | None = None, *, empty=False, broken=False):
        super().__init__(juniper_chat=fx.JUNIPER_CHAT)
        self.letters = letters or {}
        self.empty = empty
        self.broken = broken
        if empty:
            self.fetch_map = {k: (src, []) for k, (src, _) in self.fetch_map.items()}
            self.chat = None

    async def fetchrow(self, sql, *args):
        if sql == SELECT_LETTER_SQL:
            if self.broken:
                raise RuntimeError('relation "orion_day_letter" does not exist')
            return self.letters.get(args[0])
        if self.empty and sql == fx.gather.WORLD_PULSE_DIGEST_SQL:
            return None
        return await super().fetchrow(sql, *args)

    async def fetchval(self, sql, *args):
        assert sql == odo.LETTER_EXISTS_SQL
        return args[0] in self.letters


class Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *a):
                return False

        return _Ctx()


class Durable:
    def __init__(self, *, runs=None, status=202, down=False):
        self.runs = runs or {}
        self.status = status
        self.down = down
        self.submitted = []

    async def get_run(self, run_id):
        if self.down:
            raise odo.DurableUnavailable("ConnectError")
        return self.runs.get(run_id)

    async def submit(self, request_json):
        self.submitted.append(request_json)
        return self.status, "nope" if self.status >= 400 else "{}"


def desk(conn=None, durable=None, *, enabled=True, pool=True):
    conn = conn or Conn({DAY - timedelta(days=1): stored_letter(DAY - timedelta(days=1))})
    return odo.OrionDayOrderDesk(enabled=enabled, pool_provider=(lambda: Pool(conn)) if pool else (lambda: None),
                                 durable=durable or Durable(), clock=lambda: NOW)


def run(d, o=None) -> OrionDayLetterOrderResultV1:
    return asyncio.run(d.order(o or order()))


def test_an_order_builds_the_brief_and_submits_one_run_under_prime_s_ids():
    durable = Durable()
    res = run(desk(durable=durable), order(missed_letters=[MissedLetterV1(letter_date=DAY - timedelta(days=3),
                                                                          reason="never funded")]))
    assert (res.status, res.run_id) == ("submitted", RUN)
    [req] = durable.submitted
    assert req["run_id"] == RUN and req["correlation_id"] == "corr-letter-1" and req["workflow"] == "orion_day.letter"
    brief = req["brief"]
    assert brief["prime_act_id"] == "act_make_1" and brief["timeout_sec"] == 1500.0
    assert [m["letter_date"] for m in brief["missed_letters"]] == [str(DAY - timedelta(days=3))]
    assert brief["llm_view"]["digest_md"] and brief["llm_view"]["source_shares"]
    assert req["admission"]["deadline_at"].startswith((NOW + timedelta(hours=12)).date().isoformat())


def test_the_day_before_with_no_row_is_named_even_when_prime_did_not():
    durable = Durable()
    run(desk(Conn({}), durable))
    [req] = durable.submitted
    [missed] = req["brief"]["missed_letters"]
    assert missed["letter_date"] == str(DAY - timedelta(days=1)) and missed["reason"] == odo.AUTO_MISSED_REASON


def test_an_existing_letter_is_already_written_and_nothing_is_submitted():
    durable = Durable()
    res = run(desk(Conn({DAY: stored_letter(DAY)}), durable))
    assert (res.status, res.run_id) == ("already_written", "orion-day-old-1") and durable.submitted == []


def test_a_run_already_in_the_registry_is_exists_and_nothing_is_resubmitted():
    durable = Durable(runs={RUN: {"status": "running"}})
    res = run(desk(durable=durable))
    assert res.status == "exists" and durable.submitted == []


def test_409_is_exists_other_refusals_are_refused_with_detail():
    assert run(desk(durable=Durable(status=409))).status == "exists"
    res = run(desk(durable=Durable(status=422)))
    assert res.status == "refused" and "http_422" in res.detail
    res = run(desk(durable=Durable(status=503)))
    assert res.status == "unavailable" and "http_503" in res.detail  # transient: Prime re-sends


def test_an_empty_day_writes_nothing():
    durable = Durable()
    res = run(desk(Conn({}, empty=True), durable))
    assert res.status == "empty_day" and res.run_id is None and durable.submitted == []


def test_store_pool_or_durable_trouble_is_unavailable_never_a_guess():
    assert run(desk(Conn(broken=True))).status == "unavailable"
    assert run(desk(pool=False)).status == "unavailable"
    assert run(desk(durable=Durable(down=True))).status == "unavailable"


def test_flag_off_refuses_and_submits_nothing():
    durable = Durable()
    res = run(desk(durable=durable, enabled=False))
    assert res.status == "refused" and "HUB_ORION_DAY_ENABLED" in res.detail and durable.submitted == []


def test_http_endpoints_answer_status_and_order():
    durable = Durable()
    d = desk(Conn({DAY: stored_letter(DAY)}), durable)
    app = FastAPI()
    app.include_router(odo.build_orion_day_order_router(lambda: d))
    client = TestClient(app)
    st = client.get(f"/api/orion-day/letters/{DAY}").json()
    assert st["letter_exists"] is True and st["run_id"] == "orion-day-old-1"
    other = client.get(f"/api/orion-day/letters/{DAY + timedelta(days=1)}").json()
    assert other["letter_exists"] is False
    res = client.post(ORION_DAY_ORDER_PATH, json=order().model_dump(mode="json"))
    assert res.status_code == 200 and res.json()["status"] == "already_written"
    assert client.post(ORION_DAY_ORDER_PATH, json={"letter_date": str(DAY)}).status_code == 422


def test_no_path_in_hub_submits_a_letter_run_without_an_order():
    """The desk's order() is the only caller of a durable submit for orion_day.letter in Hub."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    callers = []
    for path in scripts.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "build_orion_day_request":
                callers.append(path.name)
    assert callers == ["orion_day_order.py"]
