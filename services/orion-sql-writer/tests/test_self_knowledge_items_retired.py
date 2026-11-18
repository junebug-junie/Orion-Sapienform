"""Self Atlas retirement (2026-11): self_knowledge_items is history only.

The daily scan that published self_study.items.write.v1 is deleted, so
sql-writer no longer routes or subscribes to it. The ORM model stays so the
table and its rows survive (no table drop, no row delete).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app.db import Base  # noqa: E402
from app.models.self_knowledge_item import SelfKnowledgeItemLogSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import MODEL_MAP  # noqa: E402

RETIRED_CHANNEL = "orion:self_study:items:write"
RETIRED_KIND = "self_study.items.write.v1"


def test_route_and_model_map_no_longer_carry_the_retired_write() -> None:
    assert RETIRED_KIND not in DEFAULT_ROUTE_MAP
    assert "SelfKnowledgeItemLogSQL" not in MODEL_MAP


def test_channel_is_not_subscribed_by_default_or_forced() -> None:
    stale = Settings(SQL_WRITER_SUBSCRIBE_CHANNELS=["orion:biometrics:summary"])
    assert RETIRED_CHANNEL not in stale.effective_subscribe_channels
    example = SERVICE_ROOT / ".env_example"
    text = example.read_text(encoding="utf-8")
    raw = next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="))
    assert RETIRED_CHANNEL not in json.loads(raw)
    route = next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="))
    assert RETIRED_KIND not in json.loads(route)


def test_table_is_kept_as_history() -> None:
    assert SelfKnowledgeItemLogSQL.__tablename__ == "self_knowledge_items"
    assert "self_knowledge_items" in Base.metadata.tables
