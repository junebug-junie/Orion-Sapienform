"""Shadow episode tracker: boundary Rule 3, recorded beside the live windows.

Spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md
("The episode-boundary contract"), Stage 1.

The live consolidation windows keep closing under the legacy rule. This module
runs Rule 3 on the same turns, in the same order, and keeps its own episodes in
``memory_episode_shadow``. Nothing reads these rows to change live behavior.
When a shadow episode closes it publishes ``memory.episode.closed.v1``, which
is what the Stage 1 distiller (PR 2) and the old-vs-new report consume.

Two deliberate differences from the legacy window, both from the spec:
- The turn that crosses a boundary OPENS the next episode; it is not also the
  last turn of the closed one (the legacy window carries it into both).
- Rule 3 never needs a second LLM call and never closes on a timer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.boundary import rule3_boundary, turn_phase
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1
from orion.schemas.memory_episode import MemoryEpisodeClosedV1

logger = logging.getLogger(__name__)

# A workflow command turn is identified by the Hub workflow runtime's own reply
# header ("Workflow: Journal Pass", "Workflow 'github_compactor_pass' ..."),
# per the spec's skip rule. This is a structural marker the runtime writes,
# not a judgment about what is worth remembering.
_WORKFLOW_REPLY = re.compile(r"^\s*Workflow\b")

SKIP_COMMAND_ONLY = "command_only"


def is_workflow_command_turn(response: str | None) -> bool:
    return bool(_WORKFLOW_REPLY.match(str(response or "")))


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str) and value:
        try:
            return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ShadowTurn:
    correlation_id: str
    at: datetime
    phase_change: Optional[str]
    delta_user_seconds: Optional[float]
    phase_source: Optional[str]
    boundary_score: Optional[float]
    is_command: bool
    legacy_close_reason: Optional[str]

    @classmethod
    def from_turn(
        cls,
        turn: MemoryTurnPersistedV1,
        scores: dict[str, Any],
        *,
        arrived_at: datetime,
        legacy_close_reason: Optional[str],
    ) -> "ShadowTurn":
        stamp = turn.spark_meta.get("conversation_phase")
        stamp = stamp if isinstance(stamp, dict) else {}
        score = scores.get("conversation_boundary_score")
        delta = stamp.get("delta_user_seconds")
        return cls(
            correlation_id=turn.correlation_id,
            at=_as_utc(turn.created_at) if turn.created_at else _as_utc(arrived_at),
            phase_change=turn_phase(turn),
            delta_user_seconds=float(delta) if isinstance(delta, (int, float)) else None,
            phase_source=str(stamp.get("source")) if stamp.get("source") else None,
            boundary_score=float(score) if isinstance(score, (int, float)) else None,
            is_command=is_workflow_command_turn(turn.response),
            legacy_close_reason=legacy_close_reason,
        )

    def entry(self, *, v2_boundary: bool, v2_reason: str) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "at": self.at.isoformat(),
            "phase_change": self.phase_change,
            "delta_user_seconds": self.delta_user_seconds,
            "phase_source": self.phase_source,
            "boundary_score": self.boundary_score,
            "is_command": self.is_command,
            "legacy_close_reason": self.legacy_close_reason,
            "v2_boundary": v2_boundary,
            "v2_reason": v2_reason,
        }


def build_closed_event(
    *,
    episode_id: str,
    source_platform: Optional[str],
    turns: list[dict[str, Any]],
    started_at: datetime,
    closing: ShadowTurn,
    close_reason: str,
) -> MemoryEpisodeClosedV1:
    """The close event for an episode, given its turns and the turn that crossed the boundary."""
    real_turns = [t for t in turns if isinstance(t, dict)]
    ended_at = max((_parse_ts(t.get("at")) for t in real_turns if _parse_ts(t.get("at"))), default=started_at)
    commands = sum(1 for t in real_turns if t.get("is_command"))
    juniper = len(real_turns) - commands
    status = "skipped" if real_turns and juniper == 0 else "closed"
    return MemoryEpisodeClosedV1(
        episode_id=episode_id,
        source_platform=source_platform,
        started_at=started_at,
        ended_at=ended_at,
        closed_at=closing.at,
        turn_ids=[str(t.get("correlation_id")) for t in real_turns],
        juniper_turn_count=juniper,
        command_turn_count=commands,
        close_reason=close_reason,
        phase_at_close=closing.phase_change,
        boundary_score_at_close=closing.boundary_score,
        close_lag_sec=max(0.0, (closing.at - ended_at).total_seconds()),
        closing_turn_id=closing.correlation_id,
        episode_status=status,  # type: ignore[arg-type]
        skip_reason=SKIP_COMMAND_ONLY if status == "skipped" else None,
    )


class EpisodeShadowStore:
    """Postgres-backed shadow episodes, one open episode per source_platform."""

    def __init__(self, pool, settings):
        self._pool = pool
        self._settings = settings

    async def observe_turn(
        self,
        turn: MemoryTurnPersistedV1,
        scores: dict[str, Any],
        *,
        legacy_close_reason: Optional[str],
        arrived_at: Optional[datetime] = None,
    ) -> Optional[MemoryEpisodeClosedV1]:
        """Apply Rule 3 to one arriving turn. Returns the close event if an episode closed."""
        shadow = ShadowTurn.from_turn(
            turn,
            scores,
            arrived_at=arrived_at or datetime.now(timezone.utc),
            legacy_close_reason=legacy_close_reason,
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT episode_id, turns, started_at, last_turn_at
                    FROM memory_episode_shadow
                    WHERE status = 'open' AND source_platform IS NOT DISTINCT FROM $1
                    ORDER BY started_at ASC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    turn.source_platform,
                )
                if row is None:
                    await self._open(conn, shadow, turn.source_platform, reason="v2:first_turn")
                    return None
                turns = json.loads(row["turns"]) if row["turns"] else []
                turns = turns if isinstance(turns, list) else []
                if any(isinstance(t, dict) and t.get("correlation_id") == shadow.correlation_id for t in turns):
                    return None  # duplicate publish of a turn already placed
                gap = (shadow.at - _as_utc(row["last_turn_at"])).total_seconds()
                is_boundary, reason = rule3_boundary(
                    phase=shadow.phase_change,
                    boundary_score=shadow.boundary_score,
                    gap_sec=gap,
                    settings=self._settings,
                )
                if not is_boundary:
                    turns.append(shadow.entry(v2_boundary=False, v2_reason=reason))
                    await conn.execute(
                        """
                        UPDATE memory_episode_shadow
                        SET turns = $2::jsonb, last_turn_at = GREATEST(last_turn_at, $3)
                        WHERE episode_id = $1
                        """,
                        row["episode_id"],
                        json.dumps(turns),
                        shadow.at,
                    )
                    return None
                event = build_closed_event(
                    episode_id=row["episode_id"],
                    source_platform=turn.source_platform,
                    turns=turns,
                    started_at=_as_utc(row["started_at"]),
                    closing=shadow,
                    close_reason=reason,
                )
                await conn.execute(
                    """
                    UPDATE memory_episode_shadow
                    SET status = 'closed', episode_status = $2, skip_reason = $3,
                        closed_at = $4, closing_correlation_id = $5, close_reason = $6,
                        phase_at_close = $7, boundary_score_at_close = $8, close_lag_sec = $9,
                        juniper_turn_count = $10, command_turn_count = $11
                    WHERE episode_id = $1
                    """,
                    event.episode_id,
                    event.episode_status,
                    event.skip_reason,
                    event.closed_at,
                    event.closing_turn_id,
                    event.close_reason,
                    event.phase_at_close,
                    event.boundary_score_at_close,
                    event.close_lag_sec,
                    event.juniper_turn_count,
                    event.command_turn_count,
                )
                await self._open(conn, shadow, turn.source_platform, reason=reason, boundary=True)
                return event

    async def _open(self, conn, shadow: ShadowTurn, platform: Optional[str], *, reason: str, boundary: bool = False) -> None:
        await conn.execute(
            """
            INSERT INTO memory_episode_shadow
              (episode_id, source_platform, status, boundary_rule, turns, started_at, last_turn_at)
            VALUES ($1, $2, 'open', 'v2', $3::jsonb, $4, $4)
            """,
            str(uuid4()),
            platform,
            json.dumps([shadow.entry(v2_boundary=boundary, v2_reason=reason)]),
            shadow.at,
        )

    async def mark_published(self, episode_id: str) -> None:
        await self._pool.execute(
            "UPDATE memory_episode_shadow SET closed_event_published_at = now() WHERE episode_id = $1",
            episode_id,
        )

    async def unpublished_closed(self, *, limit: int = 5) -> list[MemoryEpisodeClosedV1]:
        """Closed episodes whose event never went out (bus down at close time)."""
        rows = await self._pool.fetch(
            """
            SELECT * FROM memory_episode_shadow
            WHERE status = 'closed' AND closed_event_published_at IS NULL
            ORDER BY closed_at ASC
            LIMIT $1
            """,
            int(limit),
        )
        return [event_from_row(dict(r)) for r in rows]


def event_from_row(row: dict[str, Any]) -> MemoryEpisodeClosedV1:
    turns = row.get("turns")
    turns = json.loads(turns) if isinstance(turns, str) else (turns or [])
    ends = [_parse_ts(t.get("at")) for t in turns if isinstance(t, dict)]
    ends = [e for e in ends if e is not None]
    started = _as_utc(row["started_at"])
    return MemoryEpisodeClosedV1(
        episode_id=row["episode_id"],
        source_platform=row.get("source_platform"),
        started_at=started,
        ended_at=max(ends, default=started),
        closed_at=_as_utc(row["closed_at"]),
        turn_ids=[str(t.get("correlation_id")) for t in turns if isinstance(t, dict)],
        juniper_turn_count=int(row.get("juniper_turn_count") or 0),
        command_turn_count=int(row.get("command_turn_count") or 0),
        close_reason=str(row.get("close_reason") or ""),
        phase_at_close=row.get("phase_at_close"),
        boundary_score_at_close=row.get("boundary_score_at_close"),
        close_lag_sec=row.get("close_lag_sec"),
        closing_turn_id=row.get("closing_correlation_id"),
        episode_status=row.get("episode_status") or "closed",
        skip_reason=row.get("skip_reason"),
    )
