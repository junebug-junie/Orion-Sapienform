"""Goal episode write path (no Postgres, issue #3924).

``memory.field_goals.proposed.v1`` on ``orion:memory:goals:proposed`` -> ``field_goal_episode``,
one row per run of Orion wanting the same target. Routing (also under a stale operator env),
the migration matching both models and the post-pull allowlist, and the fold rule through the
real envelope handler: repeats fold, a gap or another target opens a new episode, a re-delivered
artifact does not bump the count, a malformed payload goes to the fallback log without raising.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.field_goal import FieldGoalProvenanceV1  # noqa: E402

import app.worker as worker  # noqa: E402
from app.field_goal_episode import GOAL_EPISODE_GAP  # noqa: E402
from app.models.field_goal_episode import FieldGoalEpisodeArtifactSQL, FieldGoalEpisodeSQL  # noqa: E402

CHANNEL = "orion:memory:goals:proposed"
KIND = "memory.field_goals.proposed.v1"
MIGRATION = REPO_ROOT / "services/orion-sql-db/manual_migration_field_goal_episode_v1.sql"
T0 = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _goal(target: str = "world:chat", *, ts: datetime = T0, salience: float = 0.9, **kw) -> FieldGoalProvenanceV1:
    # Same construction as services/orion-attention-runtime/app/worker.py.
    base = dict(
        subject="attention", model_layer="field_attention", entity_id=target, kind=KIND,
        field_target_id=target, target_kind="node", salience_score=salience,
        source_field_tick_id="tick-1", source_attention_frame_id="frame-1", priority=salience,
        ts=ts, provenance={"intake_channel": "internal.attention_runtime"},
    )
    base.update(kw)
    return FieldGoalProvenanceV1(**base)


def _envelope(payload: dict) -> BaseEnvelope:
    return BaseEnvelope(kind=KIND, source=ServiceRef(name="orion-attention-runtime", node="athena"),
                        correlation_id=uuid4(), payload=payload)


def _send(goal_or_payload) -> BaseEnvelope:
    payload = goal_or_payload if isinstance(goal_or_payload, dict) else goal_or_payload.model_dump(mode="json")
    env = _envelope(payload)
    asyncio.run(worker.handle_envelope(env))
    return env


def _session(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    FieldGoalEpisodeSQL.__table__.create(bind=engine)
    FieldGoalEpisodeArtifactSQL.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(worker, "get_session", lambda: session)
    monkeypatch.setattr(worker, "remove_session", lambda: None, raising=False)
    return session


def _fallbacks(monkeypatch) -> list[tuple]:
    out: list[tuple] = []
    monkeypatch.setattr(worker, "_write_fallback", lambda *a, **k: out.append(a))
    return out


def _episodes(session) -> list[FieldGoalEpisodeSQL]:
    session.expire_all()
    return session.query(FieldGoalEpisodeSQL).order_by(FieldGoalEpisodeSQL.first_ts).all()


def _env_example() -> dict[str, str]:
    return dict(line.split("=", 1) for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
                if line and not line.startswith("#") and "=" in line)


def test_subscribed_and_routed() -> None:
    from app.settings import DEFAULT_ROUTE_MAP, Settings

    env = _env_example()
    assert CHANNEL in Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]).effective_subscribe_channels
    assert CHANNEL in Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL in json.loads(env["SQL_WRITER_SUBSCRIBE_CHANNELS"])
    assert DEFAULT_ROUTE_MAP[KIND] == "FieldGoalEpisodeSQL"
    assert json.loads(env["SQL_WRITER_ROUTE_MAP_JSON"])[KIND] == "FieldGoalEpisodeSQL"
    assert worker.MODEL_MAP["FieldGoalEpisodeSQL"] == (FieldGoalEpisodeSQL, FieldGoalProvenanceV1)
    # Not a generic insert/merge model: the fold path owns every write.
    assert FieldGoalEpisodeSQL not in worker.INSERT_ONLY_MODELS


def test_registry_names_sql_writer_as_consumer() -> None:
    import yaml

    channels = yaml.safe_load((REPO_ROOT / "orion/bus/channels.yaml").read_text())["channels"]
    entry = next(c for c in channels if c["name"] == CHANNEL)
    assert entry["message_kind"] == KIND
    assert "orion-sql-writer" in entry["consumer_services"]


def test_migration_matches_models_and_fits_allowlist() -> None:
    import orion.sql_migration_drift as drift

    text = MIGRATION.read_text()
    assert text.splitlines()[0] == "-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer"
    for model in (FieldGoalEpisodeSQL, FieldGoalEpisodeArtifactSQL):
        table = model.__tablename__
        block = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", text, re.S).group(1)
        cols = {ln.split()[0] for ln in block.strip().splitlines() if ln.strip()}
        assert {c.name for c in inspect(model).columns} == cols
        # Same index names as the model, so create_all at boot and this file agree.
        for ix in model.__table__.indexes:
            on = ", ".join(c.name for c in ix.columns)
            assert f"CREATE INDEX IF NOT EXISTS {ix.name} ON {table} ({on});" in text
    assert drift.auto_apply_refusals(text, set()) == []
    rb = drift.parse_required_by(text)
    assert rb is not None and not rb.errors


def test_gap_is_ten_minutes() -> None:
    assert GOAL_EPISODE_GAP == timedelta(minutes=10)


def test_420_repeats_of_one_target_fold_into_one_episode(monkeypatch) -> None:
    session = _session(monkeypatch)
    fallbacks = _fallbacks(monkeypatch)
    goals = [_goal(ts=T0 + timedelta(seconds=2 * i), salience=0.5 + (0.4 if i == 7 else 0.0) + i * 1e-4)
             for i in range(420)]
    first_env = _send(goals[0])
    for g in goals[1:]:
        _send(g)
    (ep,) = _episodes(session)
    assert ep.repeat_count == 420
    assert ep.field_target_id == "world:chat" and ep.target_kind == "node"
    assert ep.first_artifact_id == goals[0].artifact_id
    assert ep.episode_id == f"episode-{goals[0].artifact_id}"
    assert ep.first_ts.replace(tzinfo=timezone.utc) == T0
    assert ep.last_ts.replace(tzinfo=timezone.utc) == goals[-1].ts
    assert ep.max_salience == goals[7].salience_score
    assert ep.last_salience == goals[-1].salience_score
    assert ep.first_payload_json == goals[0].model_dump(mode="json")
    assert ep.first_envelope_id == str(first_env.id)
    assert ep.received_at_first is not None and ep.updated_at >= ep.received_at_first
    assert session.query(FieldGoalEpisodeArtifactSQL).count() == 420
    assert fallbacks == []


def test_a_gap_longer_than_gap_opens_a_new_episode(monkeypatch) -> None:
    session = _session(monkeypatch)
    _send(_goal(ts=T0))
    _send(_goal(ts=T0 + GOAL_EPISODE_GAP))  # exactly GAP: still the same episode
    _send(_goal(ts=T0 + 2 * GOAL_EPISODE_GAP + timedelta(seconds=1)))  # > GAP after last_ts
    eps = _episodes(session)
    assert [e.repeat_count for e in eps] == [2, 1]


def test_a_b_a_opens_three_episodes(monkeypatch) -> None:
    session = _session(monkeypatch)
    _send(_goal("world:chat", ts=T0))
    _send(_goal("world:chat", ts=T0 + timedelta(seconds=2)))
    _send(_goal("capability:memory", ts=T0 + timedelta(seconds=4), target_kind="capability"))
    _send(_goal("world:chat", ts=T0 + timedelta(seconds=6)))
    eps = _episodes(session)
    assert [(e.field_target_id, e.repeat_count) for e in eps] == [
        ("world:chat", 2), ("capability:memory", 1), ("world:chat", 1)]
    assert eps[1].target_kind == "capability"


def test_duplicate_artifact_does_not_bump_the_count(monkeypatch) -> None:
    session = _session(monkeypatch)
    fallbacks = _fallbacks(monkeypatch)
    g = _goal()
    _send(g)
    _send(_goal(ts=T0 + timedelta(seconds=2)))
    _send(g)  # same artifact_id, new envelope
    (ep,) = _episodes(session)
    assert ep.repeat_count == 2
    assert fallbacks == []
    # The writer reports the repeat as a skip, not a write.
    assert worker._write_field_goal_episode(g, "env-x") is False
    assert _episodes(session)[0].repeat_count == 2


def test_duplicate_of_an_episode_opener_does_not_open_another(monkeypatch) -> None:
    session = _session(monkeypatch)
    a = _goal("world:chat", ts=T0)
    _send(a)
    _send(_goal("capability:memory", ts=T0 + timedelta(seconds=2)))
    _send(a)  # re-delivered after another target took over
    assert [e.repeat_count for e in _episodes(session)] == [1, 1]


def test_malformed_payload_is_rejected_logged_and_does_not_raise(monkeypatch, caplog) -> None:
    session = _session(monkeypatch)
    fallbacks = _fallbacks(monkeypatch)
    bad = _goal().model_dump(mode="json")
    bad["salience_score"] = 7.0
    del bad["field_target_id"]
    with caplog.at_level(logging.ERROR):
        _send(bad)
    assert _episodes(session) == []
    assert session.query(FieldGoalEpisodeArtifactSQL).count() == 0
    assert any("sql_writer_validation_failed" in r.getMessage() for r in caplog.records)
    assert len(fallbacks) == 1 and fallbacks[0][0] == KIND


def test_payload_without_artifact_id_is_refused(monkeypatch) -> None:
    session = _session(monkeypatch)
    fallbacks = _fallbacks(monkeypatch)
    payload = _goal().model_dump(mode="json")
    del payload["artifact_id"]
    _send(payload)
    _send(payload)
    assert _episodes(session) == []
    assert len(fallbacks) == 2


def test_seen_artifacts_older_than_the_window_are_pruned(monkeypatch) -> None:
    from app.field_goal_episode import ARTIFACT_DEDUP_WINDOW, fold_goal

    session = _session(monkeypatch)
    now = T0 + timedelta(hours=1)
    fold_goal(session, _goal(ts=T0), envelope_id=None, now=now)
    session.commit()
    later = now + ARTIFACT_DEDUP_WINDOW + timedelta(seconds=1)
    fold_goal(session, _goal(ts=T0 + timedelta(seconds=2)), envelope_id=None, now=later)
    session.commit()
    assert session.query(FieldGoalEpisodeArtifactSQL).count() == 1
