"""Dream hypotheses: kickoff section, offer claim, scorecard, contract pins."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from orion.curiosity import worldview
from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import StudyMaterial
from orion.curiosity.worldview import WorldviewSnapshot
from orion.dream import hypotheses as dh
from orion.dream.hypotheses import OfferedHypothesis, score_hypotheses, take_hypotheses_for_offer
from orion.schemas.dream_cycle import FORMED_FROM_PREFIX
from orion.schemas.registry import resolve

HYP = OfferedHypothesis("dh-aaa111", "Recall empties cluster right after GPU5 contention", "timing overlap")


def _material() -> StudyMaterial:
    return StudyMaterial(generated_at=datetime(2026, 9, 25, tzinfo=timezone.utc))


def test_status_literals_match_worldview():
    assert (dh._SUPPORTED, dh._REVISED, dh._REFUTED, dh._RETIRED, dh._OPEN) == (
        worldview.STATUS_SUPPORTED,
        worldview.STATUS_REVISED,
        worldview.STATUS_REFUTED,
        worldview.STATUS_RETIRED,
        worldview.STATUS_OPEN,
    )


def test_schema_registered():
    assert resolve("DreamCycleV1").__name__ == "DreamCycleV1"


def test_kickoff_offers_dream_section_only_when_writable():
    on = build_kickoff_prompt(_material(), run_id="abcd1234abcd", graph_enabled=True, dream_hypotheses=(HYP,))
    assert "WHILE YOU SLEPT" in on
    assert "hypothesis dh-aaa111:" in on
    assert f'formed_from = "{FORMED_FROM_PREFIX}<hypothesis id>"' in on
    assert "control" not in on.lower().split("while you slept")[1][:600]

    unreadable = build_kickoff_prompt(
        _material(),
        view=WorldviewSnapshot(unavailable_reason="down"),
        run_id="abcd1234abcd",
        graph_enabled=True,
        dream_hypotheses=(HYP,),
    )
    assert "WHILE YOU SLEPT" not in unreadable
    none = build_kickoff_prompt(_material(), run_id="abcd1234abcd", graph_enabled=True)
    assert "WHILE YOU SLEPT" not in none


class _Conn:
    def __init__(self, rows=None, exc=None):
        self.rows, self.exc, self.calls = rows or [], exc, []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if self.exc:
            raise self.exc
        return self.rows


class _Pool:
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


def test_take_for_offer_stamps_and_hides_order():
    conn = _Conn(rows=[
        {"hypothesis_id": "dh-zzz", "claim": "second claim long enough to be real", "why": ""},
        {"hypothesis_id": "dh-aaa", "claim": "first claim long enough to be real", "why": "w"},
        {"hypothesis_id": "", "claim": "dropped", "why": ""},
    ])
    got = asyncio.run(take_hypotheses_for_offer(_Pool(conn), run_id="run1", limit=3))
    assert [h.hypothesis_id for h in got] == ["dh-aaa", "dh-zzz"]
    sql, args = conn.calls[0]
    assert "offered_at IS NULL" in sql and "SET offered_at = now()" in sql
    assert "arm" not in sql.split("RETURNING")[1]
    assert args == ("run1", 3)


def test_take_for_offer_is_silent_on_failure():
    got = asyncio.run(take_hypotheses_for_offer(_Pool(_Conn(exc=RuntimeError("no table"))), run_id="r", limit=3))
    assert got == ()
    assert asyncio.run(take_hypotheses_for_offer(None, run_id="r", limit=3)) == ()


def test_scorecard_joins_priors_per_arm():
    offered = [
        {"hypothesis_id": "dh-1", "arm": "dream"},
        {"hypothesis_id": "dh-2", "arm": "dream"},
        {"hypothesis_id": "dh-3", "arm": "control"},
        {"hypothesis_id": "dh-4", "arm": "control"},
    ]
    priors = [
        {"prior_id": "p1", "formed_from": "dream_hypothesis:dh-1", "status": "supported", "times_tested": 2},
        # forked duplicate of the same adoption: counted once, most-tested wins
        {"prior_id": "p1", "formed_from": "dream_hypothesis:dh-1", "status": "open", "times_tested": 0},
        {"prior_id": "p3", "formed_from": "dream_hypothesis:dh-3 (tired)", "status": "refuted", "times_tested": 1},
        {"prior_id": "p9", "formed_from": "dream_hypothesis:dh-unknown", "status": "open"},
        {"prior_id": "px", "formed_from": "crystallization:abc", "status": "supported"},
    ]
    card = score_hypotheses(offered, priors)
    d, c = card.arms["dream"], card.arms["control"]
    assert (d.offered, d.adopted, d.tested, d.supported) == (2, 1, 1, 1)
    assert (c.offered, c.adopted, c.tested, c.refuted) == (2, 1, 1, 1)
    assert d.adoption_rate == 0.5 and d.support_rate == 1.0 and c.support_rate == 0.0
    assert card.unmatched_priors == 1
    assert card.verdict().startswith("too early")


@pytest.mark.parametrize(
    "d_adopt,c_adopt,expect",
    [(10, 1, "dream ahead of random"), (2, 2, "recombination not beating random")],
)
def test_scorecard_verdict(d_adopt, c_adopt, expect):
    offered = [{"hypothesis_id": f"d{i}", "arm": "dream"} for i in range(20)]
    offered += [{"hypothesis_id": f"c{i}", "arm": "control"} for i in range(5)]
    priors = [{"formed_from": f"{FORMED_FROM_PREFIX}d{i}", "status": "open"} for i in range(d_adopt)]
    priors += [{"formed_from": f"{FORMED_FROM_PREFIX}c{i}", "status": "open"} for i in range(c_adopt)]
    assert score_hypotheses(offered, priors).verdict().startswith(expect)
