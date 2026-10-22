"""The data a finished Retro play session hands to the memory export (Lane C).

Spec: docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md, sections E and F.

Call site (leisure): ``scripts.retro_leisure.run_leisure_session`` returns a
``LeisureOutcome``; when ``outcome.ended`` is set, the caller (the ``retro_leisure_session``
dispatch consumer) passes that ``RetroSessionEnded`` to the session summary / memory card
writer. This module defines only the data. There is deliberately no default hook and no
no-op writer: if nothing consumes it, nothing is written, and that absence is visible.

Every field is an observed fact from the runner or from Retro. Counts come from Retro's
``goal_finished`` records (prediction verdicts) as last read by the runner; the summary
builder should re-read Retro's journal and atlas for the authoritative deltas.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RetroSessionEnded:
    session_id: str
    lineage_id: Optional[str]          # from GET /sessions/{id}/atlas; None if it could not be read
    leisure_id: str
    game_id: str
    started_at: float                  # epoch seconds
    ended_at: float
    end_reason: str                    # e.g. session_cap, human_preempted, shed_x3, cabinet_hot
    controller: str = 'orion_leisure'  # who drove: the leisure sponsor (a human preempt ends it)
    act_id: Optional[str] = None       # the admitted act / dispatch id, for the proof trace
    loaded_save_id: Optional[str] = None
    end_save_id: Optional[str] = None
    end_save_error: Optional[str] = None
    decisions_planned: int = 0         # goals the model produced and Retro accepted
    planner_rejections: int = 0
    sheds: int = 0
    verdicts: dict = field(default_factory=lambda: {'met': 0, 'refuted': 0, 'unknown': 0})

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def as_dict(self) -> dict:
        return asdict(self) | {'duration_sec': round(self.duration_sec, 1)}
