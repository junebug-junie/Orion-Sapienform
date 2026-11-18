"""consult.orion_turn: one of Orion's consult turns, run as an admitted durable run.

Hub's consult loop (services/orion-hub/scripts/consult_loop.py) used to run Orion's turn
in-process (``EndogenousOutreach.compose``): no GPU pool lease, no retry. The first live consult
(2026-11-11) died on Orion's second turn when the shared agent lane was busy. Now each Orion turn is
a durable run with a deterministic ``run_id`` (``consult-<consult_id>-t<turn_no>``): the GPU pool
admits it (it queues, it never times out waiting), and durable-runs owns retries. The graph is the
single-step harness graph (services/orion-durable-runs/app/harness_smoke_graph.py); Hub reads the
step's full text back from ``GET /runs/{run_id}``.

ADDITIVE ``DurableWorkflowV1`` value on ``extra="forbid"`` / ``Literal`` models: deploy the state
consumers (orion-sql-writer, orion-hub, orion-actions) and orion-cortex-orch before orion-durable-runs
runs the first consult turn. Hub is the only producer and submits only after it is deployed.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CONSULT_ORION_TURN_WORKFLOW = "consult.orion_turn"
CONSULT_ORION_TURN_NODES: tuple[str, ...] = ("harness_step", "finish")
CONSULT_ORION_TURN_STEP_KEY = "turn"
# Hub's turn ``source`` for consult turns.
CONSULT_ORION_TURN_TAG = "consult_orion_turn"


def consult_turn_run_id(consult_id: str, turn_no: int) -> str:
    """Deterministic: one durable run per consult turn. A resubmit of the same turn is the same run."""
    return f"consult-{consult_id}-t{int(turn_no)}"


class ConsultOrionTurnBriefV1(BaseModel):
    """The turn's whole input. ``consult`` is a required literal so this brief can never be confused
    with another member of ``DurableRunRequestV1.brief``'s union (they all forbid unknown keys)."""

    model_config = ConfigDict(extra="forbid")

    consult: Literal["consult.orion_turn"]
    consult_id: str = Field(min_length=1, max_length=64)
    turn_no: int = Field(ge=1, le=99)
    prompt: str = Field(min_length=1, max_length=60000)
    # Orion's consult session (``orion.consult.core.orion_session_id``): none of Juniper's chats.
    session_id: str = Field(min_length=1, max_length=128)
    # What recall searches for: the consult question, not the whole prompt.
    retrieval_query: Optional[str] = Field(default=None, max_length=1000)
    # The step's budget per attempt (the attempt timer and Hub's turn limit derive from it).
    timeout_sec: float = Field(ge=60.0, le=3600.0)
