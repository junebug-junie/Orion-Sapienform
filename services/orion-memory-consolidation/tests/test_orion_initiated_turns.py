"""Orion's own messages reach episodes (2026-10-09).

Live: 53 of 101 chat turns over 14 days were Orion writing first (empty prompt). sql-writer
dropped them, so they never closed a stale episode (a goodnight turn waited 23 h for the next
Juniper message) and were never remembered. They now arrive with initiated_by="orion".
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path: str, name: str):
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    sys.path.insert(0, str(SERVICE_ROOT))
    spec = importlib.util.spec_from_file_location(name, SERVICE_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load("app/worker.py", "mc_worker_orion_initiated")
from app import episode_shadow  # noqa: E402  (worker load put the service root on sys.path)

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1  # noqa: E402

T0 = datetime(2026, 10, 7, 3, 45, tzinfo=timezone.utc)


def _env(turn: MemoryTurnPersistedV1) -> BaseEnvelope:
    return BaseEnvelope(kind="memory.turn.persisted.v1", correlation_id=turn.correlation_id,
                        source=ServiceRef(name="sql-writer", version="t", node="t"), payload=turn.model_dump(mode="json"))


def test_contract_defaults_to_juniper_and_accepts_orion():
    assert MemoryTurnPersistedV1(correlation_id="c", prompt="hi", response="hey").initiated_by == "juniper"
    t = MemoryTurnPersistedV1(correlation_id="c", prompt="", response="Morning! I kept thinking about the porch camera.",
                              initiated_by="orion")
    assert t.initiated_by == "orion"


@pytest.mark.asyncio
async def test_orion_turn_goes_only_to_the_episode_tracker(monkeypatch):
    classify = AsyncMock()
    monkeypatch.setattr(worker, "classify_turn", classify)
    window_store, episode_store = AsyncMock(), AsyncMock()
    episode_store.observe_turn = AsyncMock(return_value=None)
    episode_store.unpublished_closed = AsyncMock(return_value=[])
    turn = MemoryTurnPersistedV1(correlation_id="00000000-0000-0000-0000-000000000001", prompt="",
                                 response="Morning! I kept thinking about the porch camera.", initiated_by="orion")
    await worker.handle_memory_turn_persisted(_env(turn), bus=AsyncMock(), window_store=window_store,
                                              suggest_runner=AsyncMock(), episode_store=episode_store)
    classify.assert_not_awaited()
    window_store.append_turn.assert_not_awaited()
    episode_store.observe_turn.assert_awaited_once()
    args, kwargs = episode_store.observe_turn.call_args
    assert args[0].initiated_by == "orion" and args[1] == {} and kwargs["legacy_close_reason"] is None


def _entry(corr, minutes, initiated_by="juniper", is_command=False):
    return {"correlation_id": corr, "at": (T0 + timedelta(minutes=minutes)).isoformat(), "is_command": is_command,
            "initiated_by": initiated_by}


def _closing():
    return episode_shadow.ShadowTurn(correlation_id="close", at=T0 + timedelta(hours=10), phase_change=None,
                                     delta_user_seconds=None, phase_source=None, boundary_score=None,
                                     is_command=False, legacy_close_reason=None, initiated_by="orion")


def test_orion_turns_are_not_counted_as_juniper_turns():
    ev = episode_shadow.build_closed_event(
        episode_id="ep", source_platform=None, started_at=T0, closing=_closing(), close_reason="v2:no_phase_time_gap",
        turns=[_entry("j1", 0), _entry("o1", 50, "orion"), _entry("o2", 100, "orion")])
    assert (ev.episode_status, ev.juniper_turn_count, ev.turn_ids) == ("closed", 1, ["j1", "o1", "o2"])


def test_episode_of_only_orion_messages_is_skipped_with_its_own_reason():
    ev = episode_shadow.build_closed_event(
        episode_id="ep", source_platform=None, started_at=T0, closing=_closing(), close_reason="v2:no_phase_time_gap",
        turns=[_entry("o1", 0, "orion"), _entry("o2", 50, "orion")])
    assert (ev.episode_status, ev.skip_reason, ev.juniper_turn_count) == ("skipped", "no_juniper_turn", 0)


def test_rows_from_before_the_field_still_count_as_juniper():
    ev = episode_shadow.build_closed_event(
        episode_id="ep", source_platform=None, started_at=T0, closing=_closing(), close_reason="v2:no_phase_time_gap",
        turns=[{"correlation_id": "old", "at": T0.isoformat(), "is_command": False}])
    assert (ev.episode_status, ev.juniper_turn_count) == ("closed", 1)


def _shadow(initiated_by, phase=None):
    return episode_shadow.ShadowTurn(correlation_id="x", at=T0, phase_change=phase, delta_user_seconds=None,
                                     phase_source=None, boundary_score=None, is_command=False,
                                     legacy_close_reason=None, initiated_by=initiated_by)


def test_orion_message_after_a_long_silence_closes_the_stale_episode():
    """Live: goodnight at 03:45, Orion writes at 14:04 (10 h later) -> the goodnight episode
    closes then, instead of waiting 23 h for Juniper's next message."""
    turns = [_entry("j-goodnight", 0)]
    assert episode_shadow.episode_boundary(turns, _shadow("orion"), 10 * 3600, worker.settings) == (
        True, "v2:orion_after_silence")


def test_orion_message_soon_after_juniper_joins_her_conversation():
    """Review finding: at a 105-min gap the old 90-min fallback split a conversation she then
    resumed. Under 3 h an Orion turn joins."""
    turns = [_entry("j1", 0)]
    assert episode_shadow.episode_boundary(turns, _shadow("orion"), 105 * 60, worker.settings) == (
        False, "v2:orion_joins")


def test_juniper_reply_joins_the_outreach_it_answers_despite_her_long_gap_phase():
    """Review finding (25/25 live outreach): her reply is stamped long_gap/next_day from HER
    last turn, which used to close the Orion-only episode so outreach was never remembered with
    its reply. Orion at 14:00 and 14:50, Juniper next_day -> one episode."""
    turns = [_entry("o1", 0, "orion"), _entry("o2", 50, "orion")]
    for phase in ("next_day", "long_gap", "stale_thread"):
        assert episode_shadow.episode_boundary(turns, _shadow("juniper", phase), 6 * 3600, worker.settings) == (
            False, "v2:reply_to_orion")


def test_a_reply_days_after_unanswered_outreach_starts_fresh():
    turns = [_entry("o1", 0, "orion")]
    is_boundary, _ = episode_shadow.episode_boundary(turns, _shadow("juniper", "next_day"), 30 * 3600, worker.settings)
    assert is_boundary is True


def test_juniper_turns_still_follow_rule3():
    turns = [_entry("j1", 0)]
    assert episode_shadow.episode_boundary(turns, _shadow("juniper", "next_day"), 20 * 3600, worker.settings) == (
        True, "v2:phase_next_day")
