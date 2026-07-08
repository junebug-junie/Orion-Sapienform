from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RECALL_ROOT = _REPO / "services" / "orion-recall"
if str(_RECALL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECALL_ROOT))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from app.collectors.active_packet import fetch_active_packet_fragments
from orion.memory.crystallization.active_packet import build_active_packet
from orion.memory.crystallization.schemas import (
    CrystallizationDynamicsV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationV1,
)


def _crys(
    *,
    activation: float,
    salience: float = 0.5,
    crystallization_id: str = "crys_test",
    summary: str = "test belief",
) -> MemoryCrystallizationV1:
    now = datetime.now(timezone.utc)
    return MemoryCrystallizationV1(
        crystallization_id=crystallization_id,
        kind="semantic",
        subject="s",
        summary=summary,
        status="active",
        salience=salience,
        dynamics=CrystallizationDynamicsV1(activation=activation, formed_at=now),
        governance=CrystallizationGovernanceV1(proposed_by="t"),
        created_at=now,
        updated_at=now,
    )


def test_build_active_packet_ranks_by_activation_times_salience():
    low_score = _crys(activation=0.2, salience=0.5, crystallization_id="crys_low")
    high_score = _crys(activation=0.9, salience=0.3, crystallization_id="crys_high")

    packet = build_active_packet(query="move logistics", crystallizations=[low_score, high_score])

    assert packet.crystallization_refs[0] == "crys_high"
    assert packet.crystallization_refs[1] == "crys_low"


def test_build_active_packet_excludes_below_activation_floor():
    eligible = _crys(activation=0.2, crystallization_id="crys_eligible")
    ineligible = _crys(activation=0.01, crystallization_id="crys_ineligible")

    packet = build_active_packet(query="test", crystallizations=[eligible, ineligible])

    assert packet.crystallization_refs == ["crys_eligible"]


@pytest.mark.asyncio
async def test_active_packet_excludes_below_activation_floor():
    eligible = _crys(activation=0.2, crystallization_id="crys_eligible", summary="eligible belief")
    ineligible = _crys(activation=0.01, crystallization_id="crys_ineligible", summary="stale belief")
    captured: list[list[MemoryCrystallizationV1]] = []

    fake_packet = type("Pkt", (), {"items": []})()

    async def _capture_retrieve(**kwargs):
        captured.append(list(kwargs.get("crystallizations") or []))
        return fake_packet

    with patch(
        "app.collectors.active_packet.list_crystallizations",
        new=AsyncMock(return_value=[eligible, ineligible]),
    ), patch(
        "app.collectors.active_packet.retrieve_active_packet",
        new=AsyncMock(side_effect=_capture_retrieve),
    ):
        await fetch_active_packet_fragments(
            query=type(
                "Q",
                (),
                {
                    "session_id": "s1",
                    "node_id": "n1",
                    "retrieval_intent": "semantic",
                    "fragment": "eligible belief",
                },
            )(),
            pool=object(),
            settings=type("S", (), {"RECALL_ACTIVE_PACKET_ENABLED": True})(),
        )

    assert len(captured) == 1
    assert len(captured[0]) == 1
    assert captured[0][0].crystallization_id == "crys_eligible"
