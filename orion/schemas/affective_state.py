"""Bus payload schema for the Juniper affective-state signal's
``orion:substrate:juniper_affective_state`` channel
(docs/superpowers/specs/2026-07-30-juniper-affective-state-signal-proposal.md).

Deliberately narrower than ``orion.cocreation.affective_signals.AffectiveWordScores``:
only ``swear_frequency`` is on this wire schema. ``typo_rate`` passed neither of
root ``CLAUDE.md``'s metric-quality-gate live-data checks in the real replay
(``docs/superpowers/pr-reports/2026-08-11-juniper-affective-state-signal-replay.md``)
-- it never reached a genuine rest state across 111 real sessions, a structural
finding about general-English spellcheck on a software-engineering chat corpus,
not a one-off bug. Per that gate, a metric that fails does not get wired in;
``typo_rate`` stays a tested library function, not a wire field, until a real
fix (a corpus-trained dictionary, or a different fatigue instrument) passes
its own replay.

Aggregate scalar only, per the proposal doc's privacy boundary -- never the
underlying message text or which words were flagged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class JuniperAffectiveStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["juniper_affective_state.v1"] = "juniper_affective_state.v1"
    observed_at: datetime
    # Half-open [window_since, window_until) tiling real transcript time,
    # same convention as CodebaseDeltaV1's pr_lifecycle domain -- this
    # tick's window_until becomes the next tick's window_since.
    window_since: datetime
    window_until: datetime
    message_count: int
    word_count: int
    swear_count: int
    # None (not a fabricated 0.0) when word_count == 0 -- an empty window
    # (no real typed messages fell in it) is a different fact from a calm
    # one (messages existed and none were swears).
    swear_frequency: float | None = None
