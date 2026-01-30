from app.topic_rail.http_server import build_health_payload


class DummyService:
    model_version = "v1"
    model_loaded = True
    last_fit_at = "2025-01-01T00:00:00Z"
    last_assign_at = "2025-01-01T01:00:00Z"
    last_summary_at = None
    last_drift_at = None
    last_error = None


def test_health_payload_contents():
    payload = build_health_payload(DummyService())
    assert payload["status"] == "ok"
    assert payload["model_version"] == "v1"
    assert payload["model_loaded"] is True
    assert payload["last_fit_at"] == "2025-01-01T00:00:00Z"
