"""The walkway read must never cost the rest of the curiosity menu.

`vision_unresolved` is a manual migration; on a host without it the query
raises. That must leave the crystallizations/relations intact and the
material `available` -- an unapplied migration is not a broken store.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from scripts.curiosity_investigation import CuriosityInvestigation

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


class _Conn:
    def __init__(self, unresolved):
        self.unresolved = unresolved

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if "vision_unresolved" in sql:
            if isinstance(self.unresolved, Exception):
                raise self.unresolved
            return self.unresolved
        if "GROUP BY m.kind" in sql:
            return [{"kind": "semantic", "n": 1, "manual_n": 1}]
        if "FROM memory_crystallizations" in sql and "random()" in sql:
            return [{
                "crystallization_id": "c1", "kind": "semantic", "subject": "s",
                "summary": "s", "salience": 0.5, "created_at": NOW,
            }]
        return []

    async def fetchval(self, sql, *args):
        return 0


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def _loop(conn) -> CuriosityInvestigation:
    return CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=1500.0, session_id="orion_curiosity",
        crystallization_sample=12, relation_sample=6,
        pool_provider=lambda: _Pool(conn), source_ref="test",
    )


def test_missing_table_costs_only_the_walkway_section() -> None:
    loop = _loop(_Conn(RuntimeError('relation "vision_unresolved" does not exist')))
    material = asyncio.run(loop._read_study_material(NOW))
    assert not material.is_unavailable
    assert material.has_material
    assert material.unresolved == []


def test_rows_are_read_into_cards() -> None:
    row = {
        "unresolved_id": "u1", "stream_id": "walkway", "camera_id": "walkway",
        "observed_at": NOW, "reason": "surprise", "description": "something by the gate",
        "what_was_tried": [], "evidence_refs": [], "image_ref": None,
    }
    material = asyncio.run(_loop(_Conn([row]))._read_study_material(NOW))
    assert [c.unresolved_id for c in material.unresolved] == ["u1"]
