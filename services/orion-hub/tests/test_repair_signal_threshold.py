"""Stage 0A: the `repair_signal` grammar atom means real repair pressure.

It used to mean "a repair appraisal ran" (`has_repair_signal=repair_bundle is
not None`), which was true for ~96% of chat turns -- mostly at the
classifier's confident-NO floor of 0.087 -- and memory consolidation reads the
atom's presence as "a repair happened" and saves the window. The floor is the
repair contract's own `concrete_bias` threshold (0.45), the level at which the
reply actually changes.

The measured level must still reach the chat projection for sub-floor turns,
because chat_prediction_error diffs it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.substrate.appraisal.contract import REPAIR_SIGNAL_LEVEL_FLOOR, is_repair_signal
from orion.substrate.chat_loop.grammar_extract import extract_chat_turn_state


def _bundle(level: float, confidence: float = 0.65) -> TurnAppraisalBundleV1:
    return TurnAppraisalBundleV1(
        correlation_id="corr-1",
        grammar_scalars={"repair_pressure": {"level": level, "confidence": confidence}},
    )


async def _emitted_events(bundle):
    from orion.hub.turn_orchestrator import _publish_unified_turn_chat_grammar

    publish = AsyncMock()
    settings = SimpleNamespace(
        PUBLISH_HUB_CHAT_GRAMMAR=True, NODE_NAME="athena", GRAMMAR_EVENT_CHANNEL="orion:grammar:event"
    )
    with patch("scripts.grammar_publish.publish_hub_chat_grammar_trace", publish):
        await _publish_unified_turn_chat_grammar(
            bus=object(),
            correlation_id="corr-1",
            session_id="sess-1",
            user_message="sup",
            repair_bundle=bundle,
            stance_disposition="proceed",
            stance_disposition_reasons=[],
            stance_boundary_register=False,
            settings=settings,
        )
    publish.assert_awaited_once()
    return publish.await_args.args[1]


def _roles(events):
    return [e.atom.semantic_role for e in events if e.atom is not None]


def test_floor_is_the_repair_contract_concrete_bias_threshold():
    from orion.substrate.appraisal import contract

    assert REPAIR_SIGNAL_LEVEL_FLOOR == contract._LEVEL_MID == 0.45
    assert not is_repair_signal(0.08706577244027125)  # confident all-NO floor
    assert not is_repair_signal(0.34308499352730304)  # highest common sub-floor reading
    assert is_repair_signal(0.45)
    assert not is_repair_signal(None)


@pytest.mark.asyncio
async def test_appraisal_with_no_repair_gives_no_repair_signal():
    events = await _emitted_events(_bundle(0.08706577244027125))
    roles = _roles(events)
    assert "repair_signal" not in roles
    assert "repair_pressure_reading" in roles
    state = extract_chat_turn_state(events)
    assert state.has_repair_signal is False
    # Same level the projection read before the change.
    assert state.repair_pressure_level == pytest.approx(0.08706577244027125)
    assert state.repair_pressure_confidence == pytest.approx(0.65)


@pytest.mark.asyncio
async def test_real_repair_pressure_gives_repair_signal():
    events = await _emitted_events(_bundle(0.55, 0.7))
    roles = _roles(events)
    assert "repair_signal" in roles
    assert "repair_pressure_reading" not in roles
    state = extract_chat_turn_state(events)
    assert state.has_repair_signal is True
    assert state.repair_pressure_level == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_no_appraisal_emits_neither_atom():
    roles = _roles(await _emitted_events(None))
    assert "repair_signal" not in roles
    assert "repair_pressure_reading" not in roles


@pytest.mark.asyncio
async def test_consolidation_reads_sub_floor_turn_as_no_repair():
    from unittest.mock import MagicMock

    from orion.memory.consolidation_grammar import fetch_grammar_evidence_for_window

    events = await _emitted_events(_bundle(0.2))
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {"event_id": e.event_id, "event_json": e.model_dump(mode="json")} for e in events
        ]
    )
    repair, ids = await fetch_grammar_evidence_for_window(
        pool, turns=[{"correlation_id": "corr-1"}], node_id="athena", enabled=True
    )
    assert repair is False
    assert len(ids) == len(events)
