from __future__ import annotations

from app.services.publish_email import publish_email_preview
from app.settings import Settings
from orion.schemas.world_pulse import EmailWorldPulseRenderV1


class _FakeNotifyClient:
    def __init__(self) -> None:
        self.calls = 0

    def send(self, req):  # noqa: ANN001
        self.calls += 1
        raise AssertionError("send should not be called when email is disabled")


def test_world_pulse_defaults_are_conservative() -> None:
    cfg = Settings()
    assert cfg.world_pulse_enabled is False
    assert cfg.world_pulse_dry_run is True
    assert cfg.world_pulse_graph_enabled is False
    assert cfg.world_pulse_graph_dry_run is True
    assert cfg.world_pulse_email_enabled is False
    assert cfg.world_pulse_stance_enabled is False


def test_email_publish_is_blocked_when_disabled() -> None:
    email = EmailWorldPulseRenderV1(
        run_id="run-1",
        subject="Daily World Pulse",
        opening="opening",
        plaintext_body="body",
        html_body=None,
        to=[],
        from_email=None,
        dry_run=True,
        created_at="2026-04-26T00:00:00+00:00",
    )
    fake = _FakeNotifyClient()
    assert publish_email_preview(notify_client=fake, email=email, enabled=False) is False
    assert fake.calls == 0
