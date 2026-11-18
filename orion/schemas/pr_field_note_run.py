"""pr.field_note: after a PR merges, one durable run has Orion write one field note on it.

Spec: docs/superpowers/specs/2026-11-04-pr-field-notes-design.md (approved by Juniper 2026-11-04),
Patch 2 ("Automatic"). orion-durable-runs polls GitHub's public API every 15 min
(services/orion-durable-runs/app/pr_field_note_poll.py) and submits one run per merged PR that has
no record yet. The run is one step on the harness step seam (app/harness_step.py):

    precheck -> resource_request -> resource_wait -> field_note (harness turn + verify_receipt) -> finish

The turn runs two dig commands (``field-note <n>``, then ``field-note <n> write ...``). The run is
done only when code finds the save receipt (``field-note #<n> saved fn:<n>:<hex>``) in the reply AND
that note id is in the ledger for this PR. A dependency-only or tiny docs-only PR never gets a run:
the poller writes its code record itself (Juniper's decision 2).

ADDITIVE ``DurableWorkflowV1`` value on ``extra="forbid"`` / ``Literal`` models: deploy the state
consumers (orion-sql-writer, orion-hub, orion-actions) and orion-cortex-orch before
orion-durable-runs submits the first run, or an old validator drops this run's state rows.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PR_FIELD_NOTE_WORKFLOW = "pr.field_note"
# The graph's work nodes, in order (services/orion-durable-runs/app/pr_field_note_graph.py).
PR_FIELD_NOTE_NODES: tuple[str, ...] = ("precheck", "field_note", "finish")
# The one step; its result is state["steps"][PR_FIELD_NOTE_STEP_KEY].
PR_FIELD_NOTE_STEP_KEY = "field_note"
# One clean Hub session for every note turn (Hub runs these turns with no_write, so nothing piles up
# in it); never curiosity's shared investigation session.
PR_FIELD_NOTE_SESSION_ID = "pr-field-notes"
# Hub's turn ``source`` (and runtime-activity label) for a note turn.
PR_FIELD_NOTE_TAG = "pr_field_note"
# Run ids: ``pr-field-note-<n>`` for the first run of a PR, ``pr-field-note-<n>-a<k>`` for a later one.
PR_FIELD_NOTE_RUN_PREFIX = "pr-field-note-"
# What `orion.dig field-note <n> write` prints on a save (orion/dig/__main__.py cmd_field_note_write);
# the note id shape is orion/body_atlas/field_notes.py ``_note_id``.
RECEIPT_RE = re.compile(r"field-note #(\d+) saved (fn:\d+:[0-9a-f]{10})")


def base_run_id(pr_number: int) -> str:
    return f"{PR_FIELD_NOTE_RUN_PREFIX}{int(pr_number)}"


def pr_number_of_run(run_id: str) -> int | None:
    """``pr-field-note-3105`` / ``pr-field-note-3105-a2`` -> 3105; anything else -> None."""
    m = re.fullmatch(re.escape(PR_FIELD_NOTE_RUN_PREFIX) + r"(\d+)(?:-a\d+)?", run_id or "")
    return int(m.group(1)) if m else None


class PrFieldNoteBriefV1(BaseModel):
    """The run's whole input. ``field_note`` is a required literal so this brief can never be confused
    with another member of ``DurableRunRequestV1.brief``'s union (they all forbid unknown keys)."""

    model_config = ConfigDict(extra="forbid")

    field_note: Literal["pr.field_note"]
    pr_number: int = Field(ge=1)
    merged_at: datetime
    title: str = Field(default="", max_length=300)
    # The step's whole budget (the attempt timer and Hub's turn limit derive from it). The time bound
    # is the hard bound on a note's agent-lane use; Hub's turn request has no step limit field.
    timeout_sec: float = Field(ge=60.0, le=3600.0)
