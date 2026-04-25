from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_notify_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_notify_request_normalization_hydrates_attention_fields_from_context() -> None:
    payload = {
        "event_kind": "orion.chat.attention",
        "context": {
            "attention_id": "attn-123",
            "require_ack": True,
            "ack_deadline_minutes": 15,
            "escalation_channels": ["email"],
            "expires_at": "2026-04-25T10:00:00Z",
        },
    }

    normalized = worker._normalize_notification_request_payload(payload)

    assert normalized["attention_id"] == "attn-123"
    assert normalized["attention_require_ack"] is True
    assert normalized["attention_ack_deadline_minutes"] == 15
    assert normalized["attention_escalation_channels"] == ["email"]
    assert isinstance(normalized["attention_expires_at"], datetime)


def test_notify_request_normalization_hydrates_chat_message_fields_from_context() -> None:
    payload = {
        "event_kind": "orion.chat.message",
        "session_id": "sess-abc",
        "context": {
            "message_id": "msg-777",
            "preview_text": "preview-text",
            "full_text": "full-text",
            "require_read_receipt": True,
            "expires_at": "2026-04-25T11:00:00Z",
        },
    }

    normalized = worker._normalize_notification_request_payload(payload)

    assert normalized["message_id"] == "msg-777"
    assert normalized["message_session_id"] == "sess-abc"
    assert normalized["message_preview_text"] == "preview-text"
    assert normalized["message_full_text"] == "full-text"
    assert normalized["message_require_read_receipt"] is True
    assert isinstance(normalized["message_expires_at"], datetime)


def test_notify_request_normalization_does_not_hydrate_generic_notify_event() -> None:
    payload = {
        "event_kind": "notify.manual.smtp",
        "context": {
            "attention_id": "attn-123",
            "message_id": "msg-123",
            "preview_text": "ignore-me",
        },
    }

    normalized = worker._normalize_notification_request_payload(payload)

    assert "attention_id" not in normalized
    assert "message_id" not in normalized
    assert "message_preview_text" not in normalized


def test_notify_request_normalization_handles_missing_or_invalid_context_without_crashing() -> None:
    payload = {
        "event_kind": "orion.chat.message",
        "context": "not-a-dict",
    }

    normalized = worker._normalize_notification_request_payload(payload)

    assert normalized["event_kind"] == "orion.chat.message"
    assert "message_id" not in normalized


def test_notify_request_normalization_preserves_existing_top_level_fields() -> None:
    payload = {
        "event_kind": "orion.chat.message",
        "message_id": "existing-message-id",
        "message_preview_text": "existing-preview",
        "context": {
            "message_id": "",
            "preview_text": "",
            "full_text": "new-full-text",
        },
    }

    normalized = worker._normalize_notification_request_payload(payload)

    assert normalized["message_id"] == "existing-message-id"
    assert normalized["message_preview_text"] == "existing-preview"
    assert normalized["message_full_text"] == "new-full-text"


def test_notify_config_models_are_wired_in_worker_model_map() -> None:
    assert "RecipientProfileDB" in worker.MODEL_MAP
    assert "NotificationPreferenceDB" in worker.MODEL_MAP
