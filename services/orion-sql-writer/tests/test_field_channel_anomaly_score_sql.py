"""field_channel_anomaly_score write path (no Postgres): subscribed even under a
stale operator channel list, routed, every schema field lands on a column,
insert-only, and a real producer envelope reaches the table (sqlite)."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.telemetry.field_channel_anomaly_score import FieldChannelAnomalyScoreV1  # noqa: E402

from app.models.field_channel_anomaly_score import FieldChannelAnomalyScoreSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402

CHANNEL = "orion:field_channel:anomaly_score"
KIND = "field_channel_anomaly.score.v1"  # services/orion-field-digester/app/anomaly_bus_publish.py


def _env_example(key: str):
    raw = next(line.split("=", 1)[1].strip() for line in (SERVICE_ROOT / ".env_example").read_text().splitlines()
               if line.startswith(f"{key}="))
    return json.loads(raw)


def _score(**kw) -> FieldChannelAnomalyScoreV1:
    base = dict(
        correlation_id="score-1", encoder_id="mood-arc-encoder:v4", encoder_version="v4",
        recon_loss=0.006, recon_error_p95=0.004784, threshold_multiplier=3.0, threshold=0.014352,
        anomalous=False, window_start=datetime(2026, 10, 13, 22, 38, tzinfo=timezone.utc),
        window_end=datetime(2026, 10, 13, 22, 39, tzinfo=timezone.utc), window_size=30,
        top_channels=["egress_confidence_deficit=0.2"], mean_signed_deviation=0.01,
        deviation_direction="mixed", attribution_ok=True,
    )
    base.update(kw)
    return FieldChannelAnomalyScoreV1(**base)


def test_producer_kind_matches_route() -> None:
    src = (REPO_ROOT / "services/orion-field-digester/app/anomaly_bus_publish.py").read_text()
    assert f'FIELD_CHANNEL_ANOMALY_SCORE_KIND = "{KIND}"' in src


def test_channel_is_subscribed_even_with_stale_operator_list() -> None:
    assert CHANNEL in _env_example("SQL_WRITER_SUBSCRIBE_CHANNELS")
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP[KIND] == "FieldChannelAnomalyScoreSQL"
    assert _env_example("SQL_WRITER_ROUTE_MAP_JSON")[KIND] == "FieldChannelAnomalyScoreSQL"
    assert MODEL_MAP["FieldChannelAnomalyScoreSQL"] == (FieldChannelAnomalyScoreSQL, FieldChannelAnomalyScoreV1)
    assert FieldChannelAnomalyScoreSQL in INSERT_ONLY_MODELS


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(FieldChannelAnomalyScoreSQL).columns}
    for name in FieldChannelAnomalyScoreV1.model_fields:
        assert name in columns, name


def test_unflagged_and_flagged_scores_both_land(monkeypatch) -> None:
    import app.worker as worker
    from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    FieldChannelAnomalyScoreSQL.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    holder: dict = {}

    def _get_session():
        holder.setdefault("s", Session())
        return holder["s"]

    def _remove_session():
        s = holder.pop("s", None)
        if s is not None:
            s.close()

    monkeypatch.setattr(worker, "get_session", _get_session)
    monkeypatch.setattr(worker, "remove_session", _remove_session)

    calm_id, hot_id = str(uuid4()), str(uuid4())
    for score in (_score(correlation_id=calm_id),
                  _score(correlation_id=hot_id, recon_loss=0.03, anomalous=True)):
        # Producer stamps the envelope with the score's own correlation_id
        # (anomaly_bus_publish.py); sql-writer keys the row on it.
        env = BaseEnvelope(kind=KIND, source=ServiceRef(name="orion-field-digester"),
                           correlation_id=score.correlation_id, payload=score.model_dump(mode="json"))
        asyncio.run(worker.handle_envelope(env))

    s = Session()
    try:
        assert s.query(FieldChannelAnomalyScoreSQL).count() == 2
        calm = s.get(FieldChannelAnomalyScoreSQL, calm_id)
        hot = s.get(FieldChannelAnomalyScoreSQL, hot_id)
        assert calm is not None and calm.anomalous is False and calm.recon_loss == 0.006
        assert hot is not None and hot.anomalous is True
        assert calm.top_channels == ["egress_confidence_deficit=0.2"]
    finally:
        s.close()
