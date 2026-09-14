"""Compile-time shape checks for the curiosity_peer_brief write path
(no Postgres required) -- contractor peer briefs, 2026-09-14.

Real wiring: `CuriosityPeerBriefSQL` is registered in `MODEL_MAP` under its
own route key, keyed off kind `curiosity.peer.brief.v1`, the channel is
actually subscribed (both the .env_example list and a code-default-only
Settings), every field on `PeerBriefV1` maps onto a real column, and the
table has a bounded retention entry like AttentionSchema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import inspect

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.curiosity_peer import PEER_BRIEF_CHANNEL, PeerBriefV1  # noqa: E402

from app.grammar_retention_loop import retention_days_for  # noqa: E402
from app.grammar_truth import GRAMMAR_RETENTION_TABLES  # noqa: E402
from app.models.curiosity_peer_brief import CuriosityPeerBriefSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402


def test_the_channel_is_actually_subscribed() -> None:
    example = SERVICE_ROOT / ".env_example"
    raw = next(
        line.split("=", 1)[1].strip()
        for line in example.read_text().splitlines()
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS=")
    )
    assert PEER_BRIEF_CHANNEL in json.loads(raw)
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert PEER_BRIEF_CHANNEL in stale.effective_subscribe_channels


def test_route_map_and_model_map_agree() -> None:
    assert DEFAULT_ROUTE_MAP["curiosity.peer.brief.v1"] == "CuriosityPeerBriefSQL"
    assert MODEL_MAP["CuriosityPeerBriefSQL"] == (CuriosityPeerBriefSQL, PeerBriefV1)
    assert CuriosityPeerBriefSQL in INSERT_ONLY_MODELS


def test_every_schema_field_lands_on_a_column() -> None:
    columns = {c.key for c in inspect(CuriosityPeerBriefSQL).columns}
    for name in PeerBriefV1.model_fields:
        if name == "schema_version":
            continue
        assert name in columns, name


def test_the_table_has_bounded_retention_by_default() -> None:
    assert "curiosity_peer_brief" in dict(GRAMMAR_RETENTION_TABLES)
    days = retention_days_for(Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=[]))
    assert days["curiosity_peer_brief"] == 90
    example = (SERVICE_ROOT / ".env_example").read_text()
    assert "CURIOSITY_PEER_BRIEF_RETENTION_DAYS=90" in example
