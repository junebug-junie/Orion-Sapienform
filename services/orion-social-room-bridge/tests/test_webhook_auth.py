from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for module_name in [name for name in sys.modules if name == "app" or name.startswith("app.")]:
    sys.modules.pop(module_name)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SERVICE_ROOT))

from app import main


def _sign(secret: str, payload: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_webhook_rejects_invalid_hmac_when_secret_configured() -> None:
    payload = {"room_id": "orion-social", "message_id": "m-1", "sender_id": "peer-1", "text": "hello"}
    raw_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    expected = {"status": "ok"}
    original = main.service.process_callsyne_message
    original_token = main.settings.callsyne_webhook_token
    original_secret = main.settings.callsyne_webhook_hmac_secret
    try:
        main.settings.callsyne_webhook_token = "token-abc"
        main.settings.callsyne_webhook_hmac_secret = "hmac-secret-xyz"
        main.service.process_callsyne_message = _fake_process(expected)
        client = TestClient(main.app)
        response = client.post(
            "/webhooks/callsyne/room-message",
            content=raw_payload,
            headers={
                "Content-Type": "application/json",
                "X-Callsyne-Webhook-Token": "token-abc",
                "X-Callsyne-Signature": "sha256=deadbeef",
            },
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid_callsyne_webhook_signature"
    finally:
        main.service.process_callsyne_message = original
        main.settings.callsyne_webhook_token = original_token
        main.settings.callsyne_webhook_hmac_secret = original_secret


def test_webhook_accepts_valid_hmac_when_secret_configured() -> None:
    payload = {"room_id": "orion-social", "message_id": "m-2", "sender_id": "peer-2", "text": "hello"}
    raw_payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    expected = {"status": "ok"}
    original = main.service.process_callsyne_message
    original_token = main.settings.callsyne_webhook_token
    original_secret = main.settings.callsyne_webhook_hmac_secret
    try:
        main.settings.callsyne_webhook_token = "token-abc"
        main.settings.callsyne_webhook_hmac_secret = "hmac-secret-xyz"
        main.service.process_callsyne_message = _fake_process(expected)
        client = TestClient(main.app)
        response = client.post(
            "/webhooks/callsyne/room-message",
            content=raw_payload,
            headers={
                "Content-Type": "application/json",
                "X-Callsyne-Webhook-Token": "token-abc",
                "X-Callsyne-Signature": _sign(
                    "hmac-secret-xyz",
                    raw_payload,
                ),
            },
        )
        assert response.status_code == 200
        assert response.json() == expected
    finally:
        main.service.process_callsyne_message = original
        main.settings.callsyne_webhook_token = original_token
        main.settings.callsyne_webhook_hmac_secret = original_secret


def _fake_process(result):
    async def _inner(_payload):
        return result

    return _inner
