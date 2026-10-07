from __future__ import annotations

import pytest

from orion.harness.finalize import resolve_finalize_llm_lane
from orion.llm.resource_lease import GPU_LEASE_ROUTE
from orion.llm.routes import AGENT_ROUTE_FCC_MODEL_LABEL
from orion.schemas.gpu_pool import GpuLeaseRefV1


def test_no_lease_chat_owned_model_sonnet_resolves_chat() -> None:
    assert resolve_finalize_llm_lane(fcc_model_label="MODEL_SONNET") == "chat"


def test_no_lease_missing_label_defaults_chat() -> None:
    assert resolve_finalize_llm_lane(fcc_model_label=None) == "chat"


def test_no_lease_agent_fcc_label_resolves_agent() -> None:
    assert (
        resolve_finalize_llm_lane(fcc_model_label=AGENT_ROUTE_FCC_MODEL_LABEL) == "agent"
    )


@pytest.mark.parametrize("label", ["MODEL_SONNET", None, AGENT_ROUTE_FCC_MODEL_LABEL])
def test_hold_wins_over_any_model_label(label) -> None:
    # A held turn's calls attach to the hold whatever route they name; the route names the hold's
    # work class, never the hold's role (agent-gpu2 is not a route) and never the chat label.
    hold = GpuLeaseRefV1(lease_id="L1", generation=1, role="agent-gpu2", holder="durable-runs:r1")
    assert resolve_finalize_llm_lane(gpu_lease=hold, fcc_model_label=label) == GPU_LEASE_ROUTE == "agent"
