from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

ROUTES = HUB_ROOT / "scripts" / "crystallization_routes.py"


def test_crystallization_api_surface_present() -> None:
    text = ROUTES.read_text(encoding="utf-8")
    for path in (
        "/api/memory/crystallizations/propose",
        "/api/memory/crystallizations/proposals/{crystallization_id}/approve",
        "/api/memory/crystallizations/{crystallization_id}/links",
        "/api/memory/crystallizations/{crystallization_id}/neighborhood",
        "/api/memory/crystallizations/projection/rebuild",
        "/api/memory/graphiti/sync/{crystallization_id}",
        "/api/memory/active-packet",
    ):
        assert path in text, f"missing route {path}"


def test_active_packet_route_filters_by_recall_eligibility() -> None:
    text = ROUTES.read_text(encoding="utf-8")
    start = text.index("async def memory_active_packet")
    block = text[start : start + 2500]
    assert "eligible_for_recall" in block


# --- Retirement surfacing (docs/superpowers/specs/2026-07-13-recall-followups-loop- ---
# --- retirement-saturation-gate-spec.md section 2): list endpoint computes            ---
# --- decayed_activation/retirement_candidate live, per active-status row.             ---


def _now() -> datetime:
    return datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


def _crys(
    *,
    crystallization_id: str,
    status: str = "active",
    activation: float = 0.5,
    formed_at: datetime | None = None,
    decay_half_life_days: float = 30.0,
):
    from orion.memory.crystallization.schemas import (
        CrystallizationDynamicsV1,
        CrystallizationGovernanceV1,
        MemoryCrystallizationV1,
    )

    now = _now()
    ref_time = formed_at if formed_at is not None else now
    return MemoryCrystallizationV1(
        crystallization_id=crystallization_id,
        kind="semantic",
        subject="test subject",
        summary="test summary",
        status=status,
        confidence="likely",
        salience=0.5,
        dynamics=CrystallizationDynamicsV1(
            activation=activation,
            formed_at=ref_time,
            decay_half_life_days=decay_half_life_days,
        ),
        governance=CrystallizationGovernanceV1(proposed_by="test"),
        created_at=ref_time,
        updated_at=ref_time,
    )


@pytest.fixture
def client(monkeypatch):
    from scripts.crystallization_routes import router

    app = FastAPI()
    app.include_router(router)
    app.state.memory_pg_pool = MagicMock()

    async def _need_session(_sid):
        return "sess-1"

    monkeypatch.setattr("scripts.crystallization_routes._need_session", _need_session)

    items = [
        # Long-decayed, low activation, 30-day half-life, ~10 half-lives elapsed:
        # decayed activation falls under should_retire()'s 0.05 default floor.
        _crys(
            crystallization_id="crys-stale",
            status="active",
            activation=0.1,
            formed_at=_now() - timedelta(days=300),
            decay_half_life_days=30.0,
        ),
        # Fresh, high activation: stays well above the retirement floor.
        _crys(
            crystallization_id="crys-fresh",
            status="active",
            activation=0.9,
            formed_at=_now(),
            decay_half_life_days=30.0,
        ),
        # Non-active status: retirement candidacy does not apply (out of the active pool,
        # same scoping as recall_eligibility.eligible_for_recall()).
        _crys(
            crystallization_id="crys-proposed",
            status="proposed",
            activation=0.1,
            formed_at=_now() - timedelta(days=300),
            decay_half_life_days=30.0,
        ),
    ]
    monkeypatch.setattr(
        "scripts.crystallization_routes.list_crystallizations",
        AsyncMock(return_value=items),
    )
    monkeypatch.setattr("scripts.crystallization_routes.datetime", _FixedDatetime)
    return TestClient(app)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102 - test shim
        return _now()


def test_list_endpoint_flags_stale_crystallization_as_retirement_candidate(client: TestClient) -> None:
    resp = client.get(
        "/api/memory/crystallizations",
        headers={"X-Orion-Session-Id": "test-session"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    by_id = {item["crystallization_id"]: item for item in body["items"]}

    stale = by_id["crys-stale"]
    assert stale["retirement_candidate"] is True
    assert stale["decayed_activation"] is not None
    assert stale["decayed_activation"] < 0.05

    fresh = by_id["crys-fresh"]
    assert fresh["retirement_candidate"] is False
    assert fresh["decayed_activation"] is not None
    assert fresh["decayed_activation"] >= 0.05

    # Non-active status: retirement candidacy is not computed at all.
    proposed = by_id["crys-proposed"]
    assert proposed["retirement_candidate"] is False
    assert proposed["decayed_activation"] is None


def test_list_endpoint_preserves_existing_response_shape(client: TestClient) -> None:
    """No regression: existing consumers reading pre-existing fields still work."""
    resp = client.get(
        "/api/memory/crystallizations",
        headers={"X-Orion-Session-Id": "test-session"},
    )
    body = resp.json()
    item = body["items"][0]
    for field in ("crystallization_id", "kind", "subject", "summary", "status", "salience", "dynamics"):
        assert field in item
