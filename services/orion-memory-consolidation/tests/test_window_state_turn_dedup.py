"""append_turn() must not accumulate a second window entry for a turn that gets
reclassified (fast pass then a deeper pass, both publishing turn:persisted for
the same correlation_id with updated scores).

Regression for the live bug confirmed 2026-08-20: a blind list.append() built up
two turn_correlation_ids entries for one real chat turn, which then made
fetch_grammar_evidence_for_window() query the same grammar_events trace_id
twice and build_crystallization_from_window() mint duplicate evidence rows --
571 duplicate memory_crystallization_sources groups in the live review queue,
and deleting one "duplicate" turn card in the crystallization-observatory UI
deleted every row sharing that source_id, i.e. every duplicate at once.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from orion.schemas.memory_consolidation import MemoryTurnPersistedV1  # noqa: E402


def _load_window_state():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    path = SERVICE_ROOT / "app" / "window_state.py"
    spec = importlib.util.spec_from_file_location("memory_consolidation_window_state_dedup", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


window_state = _load_window_state()
WindowStore = window_state.WindowStore


def _turn(corr: str, prompt: str = "hi") -> MemoryTurnPersistedV1:
    return MemoryTurnPersistedV1(correlation_id=corr, prompt=prompt, response="ok", source_platform=None)


def _written_turns(pool: AsyncMock) -> list[dict]:
    return json.loads(pool.execute.await_args.args[2])


@pytest.mark.asyncio
async def test_reclassified_turn_replaces_its_existing_entry_in_place():
    """Same correlation_id appended twice with different scores must not fan out
    into two window entries -- it must overwrite the first with the second."""
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps(
                [{"correlation_id": "c1", "memory_significance_score": 0.0}]
            ),
        }
    )
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("c1"), scores={"memory_significance_score": 0.15})

    written = _written_turns(pool)
    assert len(written) == 1
    assert written[0]["correlation_id"] == "c1"
    assert written[0]["memory_significance_score"] == 0.15


@pytest.mark.asyncio
async def test_reclassified_turn_keeps_its_original_position():
    """The replacement must not jump to the end of the list -- window ordering
    (used for the summary/excerpt fallback) must stay stable."""
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps(
                [
                    {"correlation_id": "c1"},
                    {"correlation_id": "c2"},
                ]
            ),
        }
    )
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("c1"), scores={})

    written = _written_turns(pool)
    assert [t["correlation_id"] for t in written] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_reclassified_turn_collapses_preexisting_legacy_duplicates():
    """Regression for a code-review finding on the first version of this fix:
    replacing only the FIRST matching entry left a second, still-stale
    duplicate in the list for any window that already had legacy corruption
    (the pre-fix append bug) when the fix shipped. build_crystallization_from_window's
    own dedup keeps the LAST occurrence it sees, so a stale entry surviving
    after the fresh one would silently win instead of the fresh one. A third,
    freshest publish must collapse every existing entry for the correlation_id
    down to one -- not just overwrite the first and leave the rest."""
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps(
                [
                    {"correlation_id": "c1", "memory_significance_score": 0.0},
                    {"correlation_id": "c2"},
                    {"correlation_id": "c1", "memory_significance_score": 0.05},
                ]
            ),
        }
    )
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("c1"), scores={"memory_significance_score": 0.9})

    written = _written_turns(pool)
    assert [t["correlation_id"] for t in written] == ["c1", "c2"]
    assert written[0]["memory_significance_score"] == 0.9


@pytest.mark.asyncio
async def test_genuinely_new_turn_still_appends():
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "memory_window_id": "win-1",
            "turn_correlation_ids": json.dumps([{"correlation_id": "c1"}]),
        }
    )
    pool.execute = AsyncMock()

    await WindowStore(pool).append_turn(_turn("c2"), scores={})

    written = _written_turns(pool)
    assert [t["correlation_id"] for t in written] == ["c1", "c2"]
