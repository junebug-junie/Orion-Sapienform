"""Pure salience gate for town embodiment episodes.

Decides whether a world event is worth journaling as a "town episode". The gate
is intentionally conservative and deterministic: it fires on socially meaningful
events (a completed conversation, a first encounter with a new player) and stays
silent on ambient noise (bare proximity). Encounters are deduped per player via
``SalienceState`` so re-seeing the same person does not spam the journal.

No LLM, no keyword lists on user text — just structured event fields. The worker
owns the gating switch (``memory_enabled``) and the actual bus emit; this module
only answers "is this salient, and if so, what's the who/what summary?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SalienceState:
    """Cross-event memory for the salience gate.

    ``seen_players`` dedupes first-encounter salience. ``last_journaled`` records
    the last time we emitted for a given source ref, so callers can extend the
    gate with time-based cooldowns without changing the evaluate signature.
    """

    seen_players: set[str] = field(default_factory=set)
    last_journaled: dict[str, datetime] = field(default_factory=dict)


@dataclass(frozen=True)
class SalienceEvaluation:
    salient: bool
    summary: str
    source_ref: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def evaluate_salience(event: dict, state: SalienceState) -> SalienceEvaluation:
    """Decide whether ``event`` is worth journaling.

    Salient cases:
      - ``conversation_completed``: real utterances were exchanged.
      - first ``encounter`` with a player not yet in ``state.seen_players``.

    Not salient:
      - bare ``proximity`` (ambient co-location, no interaction).
      - a repeat ``encounter`` with an already-seen player (deduped).
      - anything else / malformed.
    """
    if not isinstance(event, dict):
        return SalienceEvaluation(salient=False, summary="", source_ref=None)

    kind = str(event.get("type") or "").strip()

    if kind == "conversation_completed":
        who = str(event.get("with") or event.get("player_id") or "someone").strip() or "someone"
        try:
            utterances = int(event.get("utterances") or 0)
        except (TypeError, ValueError):
            utterances = 0
        # A conversation with no utterances exchanged is not an episode.
        if utterances <= 0:
            return SalienceEvaluation(salient=False, summary="", source_ref=None)
        source_ref = str(event.get("conversation_id") or event.get("player_id") or who)
        summary = (
            f"Conversation with {who} in the town ({utterances} utterance"
            f"{'s' if utterances != 1 else ''} exchanged)."
        )
        state.last_journaled[source_ref] = _utcnow()
        return SalienceEvaluation(salient=True, summary=summary, source_ref=source_ref)

    if kind == "encounter":
        player_id = str(event.get("player_id") or "").strip()
        if not player_id or player_id in state.seen_players:
            return SalienceEvaluation(salient=False, summary="", source_ref=player_id or None)
        state.seen_players.add(player_id)
        name = str(event.get("name") or event.get("with") or player_id).strip() or player_id
        summary = f"First encounter with {name} in the town."
        state.last_journaled[player_id] = _utcnow()
        return SalienceEvaluation(salient=True, summary=summary, source_ref=player_id)

    # proximity and everything else: ambient, not an episode.
    return SalienceEvaluation(salient=False, summary="", source_ref=None)
