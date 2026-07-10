from __future__ import annotations

from typing import Any

from orion.harness.fcc_motor import cancel_fcc_turn


def test_apply_harness_run_cancel_kills_registered(monkeypatch: Any) -> None:
    from app.cancel_listener import apply_harness_run_cancel
    from orion.schemas.harness_finalize import HarnessRunCancelV1

    calls: list[str] = []

    def fake_cancel(corr: str) -> bool:
        calls.append(corr)
        return True

    monkeypatch.setattr("orion.harness.fcc_motor.cancel_fcc_turn", fake_cancel)
    monkeypatch.setattr("app.cancel_listener.cancel_fcc_turn", fake_cancel)
    killed = apply_harness_run_cancel(
        HarnessRunCancelV1(correlation_id="corr-x", reason="client_disconnect")
    )
    assert killed is True
    assert calls == ["corr-x"]


def test_apply_harness_run_cancel_dict_payload(monkeypatch: Any) -> None:
    from app.cancel_listener import apply_harness_run_cancel

    monkeypatch.setattr(
        "app.cancel_listener.cancel_fcc_turn",
        lambda corr: corr == "corr-y",
    )
    assert apply_harness_run_cancel(
        {"correlation_id": "corr-y", "reason": "ws_disconnect"}
    ) is True


def test_cancel_fcc_turn_missing_arms_pending() -> None:
    from orion.harness import fcc_motor as motor

    try:
        assert cancel_fcc_turn("no-such-corr") is True
        assert "no-such-corr" in motor._PENDING_CANCEL
    finally:
        motor._unregister_process("no-such-corr")
