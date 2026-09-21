from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.harness.finalize import resolve_finalize_llm_lane
from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL
from orion.schemas.resource_admission import ResourceLeaseV1


def _lease(lane: str) -> ResourceLeaseV1:
    now = datetime.now(timezone.utc)
    return ResourceLeaseV1(
        run_id="r1",
        demand_id="d1",
        lease_id="L1",
        resource_key=f"llm.route.{lane}",
        lane=lane,
        backend_key="http://worker:8000",
        generation=1,
        granted_at=now,
        heartbeat_at=now,
        expires_at=now + timedelta(seconds=60),
    )


def test_no_lease_chat_owned_model_sonnet_resolves_chat() -> None:
    assert resolve_finalize_llm_lane(fcc_model_label="MODEL_SONNET") == "chat"


def test_no_lease_missing_label_defaults_chat() -> None:
    assert resolve_finalize_llm_lane(fcc_model_label=None) == "chat"


def test_no_lease_agent_fcc_label_resolves_agent() -> None:
    assert (
        resolve_finalize_llm_lane(fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL) == "agent"
    )


@pytest.mark.parametrize("lane", ["chat", "agent", "metacog"])
def test_lease_lane_wins_over_conflicting_model_label(lane: str) -> None:
    # Agent FCC label must not override an admitted chat (or other) lease.
    assert (
        resolve_finalize_llm_lane(
            resource_lease=_lease(lane),
            fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL,
        )
        == lane
    )
