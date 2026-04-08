from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-sql-writer"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.models.chat_response_feedback import ChatResponseFeedbackSQL  # noqa: E402
from app.settings import DEFAULT_ROUTE_MAP, Settings  # noqa: E402
from app.worker import MODEL_MAP, _write_row, handle_envelope  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.schemas.chat_response_feedback import CHAT_RESPONSE_FEEDBACK_KIND  # noqa: E402


def test_route_map_and_channels_include_chat_response_feedback() -> None:
    settings = Settings()
    assert DEFAULT_ROUTE_MAP["chat.response.feedback.v1"] == "ChatResponseFeedbackSQL"
    assert "ChatResponseFeedbackSQL" in MODEL_MAP
    assert "orion:chat:response:feedback" in settings.effective_subscribe_channels


def test_channel_catalog_kind_and_schema_are_aligned_for_feedback() -> None:
    channels_doc = yaml.safe_load((REPO_ROOT / "orion" / "bus" / "channels.yaml").read_text(encoding="utf-8"))
    channels = channels_doc.get("channels") or []
    feedback_channel = next(item for item in channels if item.get("name") == "orion:chat:response:feedback")
    assert feedback_channel["schema_id"] == "ChatResponseFeedbackV1"
    assert feedback_channel["message_kind"] == CHAT_RESPONSE_FEEDBACK_KIND
    assert DEFAULT_ROUTE_MAP[feedback_channel["message_kind"]] == "ChatResponseFeedbackSQL"


class _FakeSession:
    def __init__(self) -> None:
        self.added = []
        self.commits = 0
        self.closed = False

    def add(self, obj):
        self.added.append(obj)

    def merge(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def rollback(self):
        return None

    def close(self):
        self.closed = True


def test_write_row_persists_feedback_id_and_linkage_fields(monkeypatch) -> None:
    import app.worker as worker_mod  # noqa: E402

    fake_session = _FakeSession()
    monkeypatch.setattr(worker_mod, "get_session", lambda: fake_session)
    monkeypatch.setattr(worker_mod, "remove_session", lambda: None)

    wrote = _write_row(
        ChatResponseFeedbackSQL,
        {
            "feedback_id": "feedback-write-1",
            "target_turn_id": "turn-1",
            "target_message_id": "turn-1:assistant",
            "target_correlation_id": "turn-1",
            "session_id": "sid-1",
            "user_id": "user-1",
            "feedback_value": "down",
            "categories": ["missed_relevant_context"],
            "free_text": "Needed more detail",
            "source": "hub_ui",
            "ui_context": {"mode": "brain"},
        },
    )
    assert wrote is True
    assert fake_session.commits == 1
    assert fake_session.added
    row = fake_session.added[0]
    assert row.feedback_id == "feedback-write-1"
    assert row.target_turn_id == "turn-1"


def test_handle_envelope_routes_feedback_payload_to_feedback_table(monkeypatch) -> None:
    import app.worker as worker_mod  # noqa: E402

    captured = []

    async def fake_write(sql_model, schema_model, payload, extra_fields, *, kind=None):
        captured.append((sql_model.__name__, getattr(schema_model, "__name__", None), payload, kind))
        return True

    monkeypatch.setattr(worker_mod, "_write", fake_write)

    env = BaseEnvelope(
        kind="chat.response.feedback.v1",
        source=ServiceRef(name="orion-hub", node="athena"),
        correlation_id="00000000-0000-0000-0000-000000000123",
        payload={
            "feedback_id": "fb-handle-1",
            "target_turn_id": "turn-abc",
            "target_message_id": "turn-abc:assistant",
            "target_correlation_id": "turn-abc",
            "session_id": "sid-2",
            "feedback_value": "up",
            "categories": ["helpful_actionable"],
            "free_text": "clear and useful",
            "source": "hub_ui",
            "created_at": "2026-04-08T00:00:00+00:00",
        },
    )
    asyncio.run(handle_envelope(env, bus=None))

    assert captured
    model_name, schema_name, payload, kind = captured[0]
    assert model_name == "ChatResponseFeedbackSQL"
    assert schema_name == "ChatResponseFeedbackV1"
    assert payload["feedback_id"] == "fb-handle-1"
    assert kind == "chat.response.feedback.v1"
