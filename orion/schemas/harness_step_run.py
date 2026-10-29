"""harness.echo_smoke: one durable-run step run as one harness turn, end to end.

The seam (services/orion-durable-runs/app/harness_step.py) lets any admitted durable workflow run a
step as a harness turn -- the same ``claude -p`` motor chat and curiosity use -- and checkpoint the
turn's final text as the step's result. It reuses the existing durable-run -> Hub turn RPC
(``orion:curiosity:turn:request``, ``CuriosityTurnRequestV1``, already used by
curiosity.investigate and self_sense_eval); no new channel. Hub runs the turn through
``execute_unified_turn`` under the run's GPU pool hold, so pool admission and the governor's lane
lock apply exactly as they do for curiosity.

This workflow is the smoke for that seam: one tiny prompt, one step, no persistence beyond the
durable-run checkpoint and its state rows. It is triggered by hand only
(services/orion-durable-runs/scripts/harness_echo_smoke.py). The later ``reminiscence.episode``
workflow (docs/superpowers/specs/2026-10-23-repo-chronicle-design.md) sits on the same seam.

ADDITIVE ``DurableWorkflowV1`` value on ``extra="forbid"`` / ``Literal`` models: deploy the state
consumers (orion-sql-writer, orion-hub, orion-actions) and orion-cortex-orch before orion-durable-runs
runs the first smoke, or an old validator drops this run's state rows.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HARNESS_ECHO_SMOKE_WORKFLOW = "harness.echo_smoke"
# The graph's work nodes, in order (services/orion-durable-runs/app/harness_smoke_graph.py).
HARNESS_ECHO_SMOKE_NODES: tuple[str, ...] = ("harness_step", "finish")
# The one step this smoke runs; its result is state["steps"][HARNESS_ECHO_SMOKE_STEP_KEY].
HARNESS_ECHO_SMOKE_STEP_KEY = "echo"
# A clean session of its own: the smoke never borrows curiosity's investigation session history.
HARNESS_ECHO_SMOKE_SESSION_ID = "harness-echo-smoke"
# Hub's turn ``source`` (and runtime-activity label) for this smoke's turn.
HARNESS_ECHO_SMOKE_TAG = "harness_echo_smoke"


class HarnessEchoSmokeBriefV1(BaseModel):
    """The smoke's whole input. ``smoke`` is a required literal so this brief can never be confused
    with another member of ``DurableRunRequestV1.brief``'s union (they all forbid unknown keys)."""

    model_config = ConfigDict(extra="forbid")

    smoke: Literal["harness.echo_smoke"]
    prompt: str = Field(min_length=1, max_length=2000)
    # The step's whole budget (the attempt timer and Hub's turn limit derive from it). At least 60 s:
    # Hub is told this minus a 30 s reply grace, so a smaller budget would send a ~1 s turn that can
    # only fail.
    timeout_sec: float = Field(ge=60.0, le=3600.0)
