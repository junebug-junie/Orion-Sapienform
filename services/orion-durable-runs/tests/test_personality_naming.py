"""Self-naming in the personality tick (fakes, no database): a naming row is read after the ``naming``
cursor, grounded, applied in the same committed tick, and never re-applied; a name never moves a
weight; the own-name echo gate holds end to end; PERSONALITY_NAMING_APPLY_ENABLED=false reads nothing;
the routes answer 204 when nothing is due or naming is off.

Design: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT), str(TESTS)]

from orion.personality.graph import Node, apply_fade  # noqa: E402

from app.personality_store import NAMING_CURSOR, PersonalityStore, node_key, split_nodes  # noqa: E402
from test_personality_loop import T0, item, utc, vec_for, world  # noqa: E402

NAME, SENTENCE = "The Long Watch", "I keep watching the field at night."


def active(did: str, text: str, **kw) -> Node:
    base = dict(disposition_id=did, origin="recurrence", status="active", label=f"label {did}", label_similarity=0.9,
                centroid=vec_for(text), centroid_n=5, weight=0.0, evidence_count=10,
                lane_counts={"dream": 5, "journal": 5}, chosen_count=0, first_seen_at=T0 - timedelta(days=3),
                last_reinforced_at=T0 - timedelta(hours=1), retired_at=None, updated_at=T0,
                lane_recent={"dream": 3.0, "journal": 3.0}, last_folded_at=T0)
    base.update(kw)
    n = Node(**base)
    from orion.personality.graph import DEFAULT_PARAMS, _derived_weight
    n.weight = n.history_weight = _derived_weight(n.lane_recent, DEFAULT_PARAMS)
    return n


def naming(nid: str, did: str, *, verdict="name", hub="accepted", created=None, **kw) -> dict:
    created = created or utc(minutes=1)
    return {"naming_id": nid, "disposition_id": did, "verdict": verdict, "hub_check": hub,
            "name": kw.get("name"), "sentence": kw.get("sentence"), "merge_with": kw.get("merge_with"),
            "written_at": created - timedelta(seconds=5), "created_at": created}


def booted():
    """A world after its first-boot tick (cursors at T0), with two active nodes stored."""
    drv, store, reader, emb, clock = world()
    asyncio.run(drv.tick())
    store.nodes = {"d1": active("d1", f"{NAME} {SENTENCE}"), "d2": active("d2", "something else entirely")}
    return drv, store, reader, emb, clock


def test_name_row_is_applied_in_the_tick_grounded_and_never_twice(caplog):
    drv, store, reader, emb, clock = booted()
    reader.namings = [naming("n1", "d1", name=NAME, sentence=SENTENCE)]
    clock.at = utc(minutes=10)
    fade_only = {n.disposition_id: n for n in apply_fade(list(store.nodes.values()), now=clock.at).nodes}
    with caplog.at_level(logging.INFO, logger="orion-durable-runs.personality_driver"):
        asyncio.run(drv.tick())
    d1 = store.nodes["d1"]
    assert d1.name == NAME and d1.name_sentence == SENTENCE and d1.orion_verdict == "named"
    assert d1.naming_id == "n1" and d1.shown_count == 1 and d1.verdict_evidence_count == 10
    assert d1.name_grounding_rank == 1 and d1.name_grounding_margin > 0
    # No weight jump at the apply tick: the weight is the fade-only value.
    assert d1.weight == pytest.approx(fade_only["d1"].weight)
    assert store.cursors[NAMING_CURSOR].ref == "n1"
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("personality_naming_applied")]
    assert lines == [f"personality_naming_applied naming=n1 verdict=name grounding_rank=1 "
                     f"margin={d1.name_grounding_margin:.4f} apply_status=applied"]
    assert drv.health()["namings_last_tick"][0]["apply_status"] == "applied"
    # Next tick: the cursor is past the row, nothing is applied again.
    clock.at = utc(minutes=20)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="orion-durable-runs.personality_driver"):
        asyncio.run(drv.tick())
    assert store.nodes["d1"].shown_count == 1
    assert not [r for r in caplog.records if r.getMessage().startswith("personality_naming_applied")]


def test_refused_and_reject_and_merge_too_far_end_to_end():
    drv, store, reader, emb, clock = booted()
    reader.namings = [naming("n1", "d1", hub="refused", name="x"),
                      naming("n2", "d2", verdict="reject", sentence="These are machine status lines, not me.",
                             created=utc(minutes=2)),
                      naming("n3", "d1", verdict="merge", merge_with="d2", created=utc(minutes=3))]
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    d1, d2 = store.nodes["d1"], store.nodes["d2"]
    assert d1.name is None and d1.shown_count == 2 and d1.status == "active"  # refused + merge too far
    assert d2.orion_verdict == "rejected" and d2.status == "active"
    assert [x["apply_status"] for x in drv.stats.namings_last_tick] == ["refused", "applied", "merge_too_far"]
    assert store.cursors[NAMING_CURSOR].ref == "n3"


def test_rows_inside_the_read_lag_wait_for_the_next_tick():
    drv, store, reader, emb, clock = booted()
    clock.at = utc(minutes=10)
    reader.namings = [naming("n1", "d1", name=NAME, sentence=SENTENCE, created=utc(minutes=9))]
    asyncio.run(drv.tick())
    assert store.nodes["d1"].name is None and NAMING_CURSOR not in store.cursors
    clock.at = utc(minutes=20)
    asyncio.run(drv.tick())
    assert store.nodes["d1"].name == NAME


def test_grounding_embed_failure_holds_then_applies_ungrounded():
    drv, store, reader, emb, clock = booted()
    reader.namings = [naming("n1", "d1", name=NAME, sentence=SENTENCE)]
    emb.fail = {f"{NAME} {SENTENCE}"}
    for i in (1, 2):
        clock.at = utc(minutes=10 * i)
        asyncio.run(drv.tick())
        assert store.nodes["d1"].name is None and NAMING_CURSOR not in store.cursors
        assert "naming_grounding_embed_failed" in drv.stats.warnings
    clock.at = utc(minutes=30)
    asyncio.run(drv.tick())
    d1 = store.nodes["d1"]
    assert d1.name == NAME and d1.name_grounding_rank is None and store.cursors[NAMING_CURSOR].ref == "n1"


def test_naming_disabled_reads_nothing_and_keeps_the_cursor():
    drv, store, reader, emb, clock = booted()
    drv.naming_enabled = False
    reader.namings = [naming("n1", "d1", name=NAME, sentence=SENTENCE)]
    clock.at = utc(minutes=10)
    asyncio.run(drv.tick())
    assert store.nodes["d1"].name is None and NAMING_CURSOR not in store.cursors


def test_own_name_echo_folds_with_no_weight_end_to_end():
    drv, store, reader, emb, clock = booted()
    text = f"{NAME} {SENTENCE}"  # embeds exactly onto d1's centroid
    store.nodes["d1"] = replace(store.nodes["d1"], name=NAME, lane_recent={"dream": 3.0})
    store.nodes["d1"].weight = store.nodes["d1"].history_weight = 1.386294361119890
    reader.add("chat", item("chat", "c1", utc(minutes=1), text))
    clock.at = utc(minutes=10)
    before = apply_fade([store.nodes["d1"]], now=clock.at).nodes[0].weight
    asyncio.run(drv.tick())
    d1 = store.nodes["d1"]
    ev = store.evidence["chat:c1"]
    assert ev.disposition_id == "d1" and ev.meta["cites_own_name"] is True and ev.weight_delta == 0.0
    assert d1.lane_counts["chat"] == 1 and "chat" not in d1.lane_recent
    assert d1.weight == pytest.approx(before)


def test_naming_columns_force_a_full_upsert():
    n = active("d1", "x")
    loaded = {"d1": node_key(n)}
    full, light = split_nodes([replace(n, shown_count=1)], loaded)
    assert [r["disposition_id"] for r in full] == ["d1"] and light == []
    full, _ = split_nodes([replace(n, name=NAME)], loaded)
    assert full[0]["name"] == NAME and "orion_verdict" in full[0]


# --- routes -----------------------------------------------------------------------------------


@pytest.fixture
def main_mod(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    from app import main

    return main


class Drv:
    naming_enabled = True


def test_naming_next_204_when_disabled_or_nothing_due(main_mod, monkeypatch):
    drv = Drv()
    monkeypatch.setattr(main_mod, "personality", drv)
    monkeypatch.setattr(main_mod, "_checkpointer_cm", object())

    async def none(self, now):
        return None

    monkeypatch.setattr(PersonalityStore, "naming_next", none)
    assert asyncio.run(main_mod.personality_naming_next()).status_code == 204
    drv.naming_enabled = False
    resp = asyncio.run(main_mod.personality_naming_next())
    assert resp.status_code == 204 and resp.headers["X-Personality-Naming"] == "disabled"


def test_naming_next_returns_the_packet_json(main_mod, monkeypatch):
    from orion.personality.naming import build_packet

    node = active("d1", "theme")
    rows = [{"evidence_id": "chat:1", "lane": "chat", "occurred_at": T0, "excerpt": "I kept looking at it.",
             "similarity": 0.9, "meta": {}}]
    packet = build_packet(node, rows, [node, active("d2", "other")], None, T0)

    async def nxt(self, now):
        return packet

    monkeypatch.setattr(main_mod, "personality", Drv())
    monkeypatch.setattr(main_mod, "_checkpointer_cm", object())
    monkeypatch.setattr(PersonalityStore, "naming_next", nxt)
    out = asyncio.run(main_mod.personality_naming_next())
    assert out["packet_id"] == packet.packet_id and out["neighbors"][0]["disposition_id"] == "d2"


def test_naming_routes_503_when_personality_disabled(main_mod, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(main_mod, "personality", None)
    for coro in (main_mod.personality_naming_next(), main_mod.personality_naming()):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(coro)
        assert exc.value.status_code == 503
