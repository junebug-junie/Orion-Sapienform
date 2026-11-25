"""Fold proposed goals into goal episodes (issue #3924).

A goal episode is a run of Orion wanting the same target. orion-attention-runtime
re-publishes the held goal on every qualifying field tick, so one goal message is
mostly a repeat of the last one. This module stores one row per run instead.

Rule, for one goal G:
  1. If G.artifact_id was already seen, skip it (re-delivery; no count bump).
  2. Take the most recently opened episode for ANY target.
  3. If it is for G's target and |G.ts - last_ts| <= GOAL_EPISODE_GAP, fold G in.
     "Most recent for any target" is the same test as "most recent for this target,
     with no episode for a different target opened after it".
  4. Otherwise open a new episode.

All of it runs in one transaction under one Postgres advisory lock, so two
concurrent handler calls cannot both open an episode or lose a count.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from app.models.field_goal_episode import FieldGoalEpisodeArtifactSQL, FieldGoalEpisodeSQL

# 10 minutes. Live (2026-11-25) a held goal is re-published about every 2 s (~420 in 15 min), and the
# producer stops only when another target wins or the field stops ticking. A silence of
# 10 min is far longer than any tick gap or a short writer/bus restart, so it is a new
# want, not the same one continuing; shorter would split one run on a restart, longer
# would merge separate wants. Code constant, not an env key, so every episode in the
# table uses the same definition.
GOAL_EPISODE_GAP = timedelta(minutes=10)

# How long a seen artifact_id is kept for dedup. Re-delivery happens within seconds
# (reconnect / replay); 48 h is a wide margin and keeps the table at ~3 days of ids.
ARTIFACT_DEDUP_WINDOW = timedelta(hours=48)

# One key for every goal write: the fold reads the newest episode across ALL targets,
# so the lock cannot be per target.
_ADVISORY_LOCK_KEY = "orion-sql-writer:field_goal_episode"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    # SQLite (tests) hands back naive datetimes; Postgres timestamptz is aware.
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def fold_goal(sess: Any, goal: Any, *, envelope_id: str | None, now: datetime | None = None) -> str:
    """Apply one validated FieldGoalProvenanceV1. Caller commits.

    Returns "duplicate", "folded", or "opened".
    """
    now = now or _utcnow()
    if sess.get_bind().dialect.name == "postgresql":
        sess.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:k)::bigint)"), {"k": _ADVISORY_LOCK_KEY}
        )

    if sess.get(FieldGoalEpisodeArtifactSQL, goal.artifact_id) is not None:
        return "duplicate"

    ts = _aware(goal.ts)
    latest = (
        sess.query(FieldGoalEpisodeSQL)
        .order_by(FieldGoalEpisodeSQL.first_ts.desc(), FieldGoalEpisodeSQL.received_at_first.desc())
        .first()
    )
    if (
        latest is not None
        and latest.field_target_id == goal.field_target_id
        and abs(ts - _aware(latest.last_ts)) <= GOAL_EPISODE_GAP
    ):
        latest.repeat_count += 1
        latest.last_ts = max(_aware(latest.last_ts), ts)
        latest.max_salience = max(latest.max_salience, goal.salience_score)
        latest.last_salience = goal.salience_score
        latest.updated_at = now
        episode_id, outcome = latest.episode_id, "folded"
    else:
        episode_id, outcome = f"episode-{goal.artifact_id}", "opened"
        sess.add(
            FieldGoalEpisodeSQL(
                episode_id=episode_id,
                field_target_id=goal.field_target_id,
                target_kind=goal.target_kind,
                first_artifact_id=goal.artifact_id,
                first_ts=ts,
                last_ts=ts,
                repeat_count=1,
                max_salience=goal.salience_score,
                last_salience=goal.salience_score,
                first_payload_json=goal.model_dump(mode="json"),
                first_envelope_id=envelope_id,
                received_at_first=now,
                updated_at=now,
            )
        )

    sess.add(FieldGoalEpisodeArtifactSQL(artifact_id=goal.artifact_id, episode_id=episode_id, seen_at=now))
    sess.query(FieldGoalEpisodeArtifactSQL).filter(
        FieldGoalEpisodeArtifactSQL.seen_at < now - ARTIFACT_DEDUP_WINDOW
    ).delete(synchronize_session=False)
    return outcome
