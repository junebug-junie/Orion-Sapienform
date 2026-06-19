from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.attention import AttentionPublisher
from app.settings import Settings


def test_attention_publisher_calls_notify_client() -> None:
    settings = Settings(
        NOTIFY_BASE_URL="http://notify:7140",
        NOTIFY_API_TOKEN="token",
    )
    with patch("app.attention.NotifyClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        publisher = AttentionPublisher(settings)
        publisher.publish_transition(
            service_id="landing-pad",
            heartbeat_name="landing-pad",
            event={"severity": "error", "message": "unhealthy", "correlation_id": "abc"},
        )
    mock_client.attention_request.assert_called_once()
    kwargs = mock_client.attention_request.call_args.kwargs
    assert kwargs["severity"] == "error"
    assert kwargs["context"]["event_kind"] == "orion.mesh.health.attention.v1"
