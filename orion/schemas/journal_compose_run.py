"""journal.compose: write one journal entry as an admitted durable run.

Why a durable run: the world-news journal (trigger_kind ``world_pulse_digest``) used to be composed
inside orion-actions with one direct cortex call. When the fast GPU lane was busy at 06:00 local
(``gpu_pool_unavailable:deadline``, live 2026-09-25..29) the compose failed and nothing retried it,
so the daily world-news email silently stopped. As a durable run the compose holds a GPU pool
hold: a busy pool means the run *waits* for its hold (never an attempt), a restart resumes it from
its checkpoint, and ``admission.deadline_at`` bounds how long it may wait.

The runner (services/orion-durable-runs/app/journal_compose_graph.py) sends the same
``journal.compose`` cortex verb orion-actions used (``orion.journaler.build_compose_request``),
attached to the hold via ``options.gpu_lease``, then publishes ``journal.entry.write.v1`` with the
brief's ``entry_id``. The entry id is fixed at submission, so a resumed/replayed publish upserts the
same row and the existing post-persist email path (orion-actions ``_handle_journal_created``, deduped
by entry_id, one email/day per cap scope) is unchanged.

Generic on purpose (any trigger_kind), because the upcoming daily letter reuses it.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orion.journaler.schemas import JournalTriggerV1

JOURNAL_COMPOSE_WORKFLOW = "journal.compose"
# The graph's node order, admitted shell included (for state events / Hub views).
JOURNAL_COMPOSE_NODES: tuple[str, ...] = ("resource_request", "resource_wait", "compose", "publish", "finish")


class JournalComposeRunBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: JournalTriggerV1
    # Fixed by the producer (deterministic per source run) so a replayed publish is an upsert.
    entry_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    recall_profile: str | None = None
    llm_route: str | None = None
    timeout_sec: float = Field(gt=0)
    # world_pulse_digest only: the WorldPulseRunResultV1 dump, merged into the draft
    # (orion.journaler.merge_world_pulse_curiosity_into_draft). Typed loosely so this contract
    # does not import the world-pulse schema; the runner validates it.
    world_pulse_result: dict[str, Any] | None = None
