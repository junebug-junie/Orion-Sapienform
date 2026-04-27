from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
SQL_WRITER_ROOT = THIS_DIR.parent
REPO_ROOT = SQL_WRITER_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SQL_WRITER_ROOT))
sys.modules.pop("app", None)

from app.settings import DEFAULT_ROUTE_MAP
from app.worker import MODEL_MAP


def test_hub_message_route_maps_to_world_pulse_hub_table() -> None:
    assert DEFAULT_ROUTE_MAP["hub.messages.create.v1"] == "WorldPulseHubMessageSQL"
    assert "WorldPulseHubMessageSQL" in MODEL_MAP
    assert DEFAULT_ROUTE_MAP["hub.messages.create.v1"] != "WorldPulsePublishStatusSQL"


def test_world_pulse_extended_routes_present() -> None:
    assert DEFAULT_ROUTE_MAP["world.pulse.article.emit.v1"] == "WorldPulseArticleSQL"
    assert DEFAULT_ROUTE_MAP["world.pulse.cluster.emit.v1"] == "WorldPulseArticleClusterSQL"
    assert DEFAULT_ROUTE_MAP["world.pulse.digest.item.v1"] == "WorldPulseDigestItemSQL"
    assert DEFAULT_ROUTE_MAP["world.pulse.publish.status.v1"] == "WorldPulsePublishStatusSQL"
    assert DEFAULT_ROUTE_MAP["world.pulse.worth.reading.v1"] == "WorldPulseWorthReadingSQL"
    assert DEFAULT_ROUTE_MAP["world.pulse.worth.watching.v1"] == "WorldPulseWorthWatchingSQL"
    assert "WorldPulseArticleSQL" in MODEL_MAP
    assert "WorldPulseArticleClusterSQL" in MODEL_MAP


def test_hub_message_payload_shape_matches_hub_message_table_fields() -> None:
    payload = {
        "message_id": "msg-1",
        "run_id": "run-1",
        "title": "Daily World Pulse",
        "date": "2026-04-26",
        "executive_summary": "summary",
        "created_at": datetime.now(timezone.utc),
    }
    mapped = {
        "message_id": payload.get("message_id"),
        "run_id": payload.get("run_id") or "",
        "title": payload.get("title") or "Daily World Pulse",
        "date": payload.get("date") or "2026-04-26",
        "executive_summary": payload.get("executive_summary") or "",
        "payload_json": payload,
        "schema_version": "v1",
        "created_at": payload.get("created_at"),
    }
    assert mapped["message_id"] == "msg-1"
    assert mapped["run_id"] == "run-1"
    assert mapped["executive_summary"] == "summary"
