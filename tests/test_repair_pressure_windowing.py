"""Windowing tests for repair pressure appraisal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.mind.substrate_emit import emit_observation
from orion.substrate.appraisal.windowing import select_recent_chat_molecules
from orion.substrate.molecules import SubstrateMoleculeV1


def _obs(text: str, source_id: str, age_seconds: int) -> SubstrateMoleculeV1:
    mol = emit_observation(surface_text=text, source_id=source_id)
    when = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    mol.created_at = when
    mol.last_touched_at = when
    return mol


def test_windowing_prefers_source_id_match():
    a = _obs("topic A", "conv-1", age_seconds=10)
    b = _obs("topic B", "conv-2", age_seconds=10)
    out = select_recent_chat_molecules([a, b], source_id="conv-1")
    assert [m.molecule_id for m in out] == [a.molecule_id]


def test_windowing_drops_too_old():
    fresh = _obs("fresh", "conv-1", age_seconds=10)
    stale = _obs("stale", "conv-1", age_seconds=10_000)
    out = select_recent_chat_molecules([fresh, stale], max_age_seconds=300)
    assert [m.molecule_id for m in out] == [fresh.molecule_id]


def test_windowing_caps_count_and_sorts_desc():
    molecules = [_obs(f"t{i}", "conv-1", age_seconds=i) for i in range(50)]
    out = select_recent_chat_molecules(molecules, max_count=5)
    assert len(out) == 5
    ages = [m.created_at for m in out]
    assert ages == sorted(ages, reverse=True)


def test_windowing_empty_input():
    assert select_recent_chat_molecules([]) == []


def test_windowing_without_source_id_returns_all_fresh():
    a = _obs("a", "conv-1", age_seconds=10)
    b = _obs("b", "conv-2", age_seconds=20)
    out = select_recent_chat_molecules([a, b])
    assert {m.molecule_id for m in out} == {a.molecule_id, b.molecule_id}
