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
brief's ``entry_id``. The entry id is fixed at submission. sql-writer's journal table is insert-only,
so a resumed/replayed publish with the same entry_id is dropped as a duplicate and
``journal.created`` is not re-emitted -- the existing post-persist email path (orion-actions
``_handle_journal_created``, one email/day per cap scope) fires once and is otherwise unchanged.

Generic on purpose (any trigger_kind), because the upcoming daily letter reuses it.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from orion.journaler.schemas import JournalTriggerV1

JOURNAL_COMPOSE_WORKFLOW = "journal.compose"


class JournalComposeRunBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: JournalTriggerV1
    # Fixed by the producer (deterministic per source run): a replayed publish is a duplicate
    # sql-writer drops, never a second entry.
    entry_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    recall_profile: str | None = None
    llm_route: str | None = None
    timeout_sec: float = Field(gt=0)
    # Deterministic text appended to the composed body unless the body already contains every
    # marker (orion.journaler.append_unless_present). Pre-rendered by the producer as plain text
    # (world_pulse_digest: orion.journaler.world_pulse_curiosity_appendix) so this contract never
    # carries -- or re-validates after the GPU call -- another service's schema.
    body_appendix: str | None = None
    body_appendix_markers: list[str] = Field(default_factory=list)
