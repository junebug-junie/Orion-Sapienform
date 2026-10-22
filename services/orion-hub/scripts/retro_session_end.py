"""The data a finished Retro play session hands to the memory export (Lane C).

Spec: docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md, sections E and F.

Two call sites:
- Leisure: ``scripts.retro_leisure.run_leisure_session`` returns a ``LeisureOutcome``; when
  ``outcome.ended`` is set, the caller (the ``retro_leisure_session`` dispatch consumer) passes
  it to the session summary / memory card writer.
- Co-play: ``scripts.retro_routes`` watches every browser cognition handoff on a Zelda
  session and, when it ends (stop, takeover, expiry, disconnect), passes a ``RetroSessionEnded``
  to ``app.state.retro_session_end_sink`` (sync or async callable, default None). With no sink
  registered nothing is written; that is the integration seam, not a fallback.

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
    leisure_id: Optional[str]          # None for a browser co-play session
    game_id: str
    started_at: float                  # epoch seconds
    ended_at: float
    end_reason: str                    # e.g. session_cap, human_preempted, shed_x3, cabinet_hot
    controller: str = 'orion_leisure'  # 'orion_leisure' or 'co_play' (a browser handed Orion the port)
    act_id: Optional[str] = None       # the admitted act / dispatch id, for the proof trace
    loaded_save_id: Optional[str] = None
    end_save_id: Optional[str] = None
    end_save_error: Optional[str] = None
    decisions_planned: int = 0         # goals the model produced and Retro accepted
    planner_rejections: int = 0
    sheds: int = 0
    # True when a human acted after the start, or presence could not be read: the runner then
    # did not save over or stop the game. The summary must not credit those frames to Orion.
    human_may_be_in_control: bool = False
    verdicts: dict = field(default_factory=lambda: {'met': 0, 'refuted': 0, 'unknown': 0})
    # From Retro's atlas session record (status ``zelda.atlas.session``). None when Retro did
    # not report the value -- never 0 -- so the summary cannot claim "0 new rooms" it never saw.
    rooms_new: Optional[int] = None
    deaths: Optional[int] = None
    controller_frames: Optional[dict] = None   # {controller tag: frames}, verbatim from Retro

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def as_dict(self) -> dict:
        return asdict(self) | {'duration_sec': round(self.duration_sec, 1)}


# Retro's per-controller frame counts inside the atlas session record. PROVISIONAL: Lane A is
# adding the tags now; this key and HUMAN_TAG_PREFIX change in one place when it lands.
CONTROLLER_FRAMES_KEY = 'controller_frames'
HUMAN_TAG_PREFIX = 'human'


def atlas_session_fields(session_status: object) -> dict:
    """rooms_new / deaths / controller_frames from a Retro status dict; None where absent."""
    zelda = session_status.get('zelda') if isinstance(session_status, dict) else None
    atlas = zelda.get('atlas') if isinstance(zelda, dict) else None
    rec = atlas.get('session') if isinstance(atlas, dict) else None
    rec = rec if isinstance(rec, dict) else {}
    def count(key):
        v = rec.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
    frames = rec.get(CONTROLLER_FRAMES_KEY)
    frames = ({str(k): v for k, v in frames.items() if isinstance(v, int) and not isinstance(v, bool)}
              if isinstance(frames, dict) else None)
    return dict(rooms_new=count('rooms_new'), deaths=count('deaths'), controller_frames=frames)


def human_held_frames(controller_frames: Optional[dict]) -> Optional[bool]:
    """True/False from Retro's counts; None when Retro sent none (callers treat None as 'maybe')."""
    if controller_frames is None:
        return None
    return any(v > 0 for k, v in controller_frames.items() if k.startswith(HUMAN_TAG_PREFIX))
