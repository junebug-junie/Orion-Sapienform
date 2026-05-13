from __future__ import annotations

from app.recall_utils import resolve_recall_bus_rpc_wait_sec


def test_resolve_recall_bus_wait_chat_quick_uses_quick_cap() -> None:
    assert (
        resolve_recall_bus_rpc_wait_sec(
            verb="chat_quick",
            step_timeout_ms=60000,
            rpc_timeout_override=None,
            recall_rpc_timeout_sec=90.0,
            chat_quick_recall_rpc_timeout_sec=37.5,
        )
        == 37.5
    )


def test_resolve_recall_bus_wait_non_quick_uses_recall_cap() -> None:
    assert (
        resolve_recall_bus_rpc_wait_sec(
            verb="chat_general",
            step_timeout_ms=60000,
            rpc_timeout_override=None,
            recall_rpc_timeout_sec=90.0,
            chat_quick_recall_rpc_timeout_sec=30.0,
        )
        == 60.0
    )


def test_resolve_recall_bus_wait_override_capped_by_step_budget() -> None:
    assert (
        resolve_recall_bus_rpc_wait_sec(
            verb="chat_general",
            step_timeout_ms=5000,
            rpc_timeout_override=999.0,
            recall_rpc_timeout_sec=90.0,
            chat_quick_recall_rpc_timeout_sec=30.0,
        )
        == 5.0
    )
