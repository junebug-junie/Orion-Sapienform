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

from pydantic import BaseModel, ConfigDict, model_validator


class JuniperAffectiveStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["juniper_affective_state.v1"] = "juniper_affective_state.v1"
    # Derived from the window bounds, not a uuid4, so a redelivered event
    # upserts onto the same row instead of inflating the sample count.
    #
    # Keyed on BOTH bounds rather than window_since alone: windows tile
    # contiguously tick-to-tick, but a container restart seeds
    # ``last_until = now - cold_start_lookback_sec``, which produces a window
    # that overlaps ones already published. Those are genuinely different
    # observations over different spans and must not collapse onto one row.
    event_id: str = ""
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
    # True on the FIRST tick after a producer (re)start, whose window is
    # cold_start_lookback_sec wide instead of one poll interval -- 3600s
    # against a 900s poll in the live config, so it re-covers up to four
    # windows a previous run already published.
    #
    # Ordinary ticks tile exactly and can be summed. Cold-start windows
    # OVERLAP them and cannot. Proven on the first two rows ever persisted
    # (2026-08-14): a 913s window sat entirely inside a 3600s one, and the
    # obvious SUM(word_count) over that hour returned 7669 against a true
    # 5913 -- +34.7%. Defaults False so an event predating this field is not
    # silently misread as a cold start.
    cold_start: bool = False

    @model_validator(mode="after")
    def _fill_event_id(self) -> "JuniperAffectiveStateV1":
        # Fills a blank only -- never clobbers a value that arrived on the
        # wire, or a replayed historical event would be rewritten under a
        # new key and duplicate the row it was meant to update.
        if not self.event_id:
            self.event_id = (
                "juniper-affective-state:"
                f"{self.window_since.isoformat()}:{self.window_until.isoformat()}"
            )
        return self
