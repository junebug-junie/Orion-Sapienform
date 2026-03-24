from __future__ import annotations

import importlib.util
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from orion.schemas.cortex.contracts import AgentTraceSummaryV1
from orion.schemas.notify import ChatMessageNotification


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str, *, package_name: str, package_dir: str):
    package_root = REPO_ROOT / package_dir
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_root)]
        sys.modules[package_name] = package
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


notify_main = _load_module(
    "orion_notify_app.main",
    "services/orion-notify/app/main.py",
    package_name="orion_notify_app",
    package_dir="services/orion-notify/app",
)
sql_notify_api = _load_module(
    "app.api_notify",
    "services/orion-sql-writer/app/api_notify.py",
    package_name="app",
    package_dir="services/orion-sql-writer/app",
)


def _sample_trace() -> AgentTraceSummaryV1:
    return AgentTraceSummaryV1(
        corr_id="corr-agent-visible",
        message_id="msg-agent-visible",
        mode="agent",
        status="success",
        duration_ms=180,
        step_count=2,
        tool_call_count=2,
        unique_tool_count=2,
        unique_tool_families=["planning", "reasoning"],
        action_counts={"decide": 1, "analyze": 1},
        effect_counts={"read_only": 2},
        summary_text="Agent planned then analyzed before answering.",
        tools=[],
        steps=[],
        raw={"source": "test"},
    )


def test_notify_chat_message_roundtrip_preserves_agent_trace() -> None:
    payload = ChatMessageNotification(
        source_service="orion-hub",
        session_id="session-agent-visible",
        correlation_id="corr-agent-visible",
        preview_text="Agent-backed response.",
        full_text="Agent-backed response with trace metadata.",
        agent_trace=_sample_trace(),
        workflow={
            'id': 'dream_cycle',
            'display_name': 'Dream Cycle',
            'status': 'completed',
            'summary': 'Dream synthesis complete.',
            'user_invocable': True,
        },
    )

    notification = notify_main._chat_message_to_notification(payload)
    state = notify_main._chat_message_to_schema(notification)

    assert notification.context["agent_trace"]["corr_id"] == "corr-agent-visible"
    assert state.agent_trace is not None
    assert state.agent_trace.mode == "agent"
    assert state.agent_trace.summary_text.startswith("Agent planned then analyzed")
    assert notification.context['workflow']['id'] == 'dream_cycle'
    assert state.workflow['id'] == 'dream_cycle'


def test_sql_notify_reader_exposes_agent_trace_from_stored_context() -> None:
    row = SimpleNamespace(
        message_id=uuid4(),
        notification_id=uuid4(),
        created_at=datetime.utcnow(),
        source_service="orion-hub",
        message_session_id="session-agent-visible",
        session_id="session-agent-visible",
        correlation_id="corr-agent-visible",
        title="New message from Orion",
        message_preview_text="Agent-backed response.",
        body_text="Agent-backed response.",
        message_full_text="Agent-backed response with trace metadata.",
        body_md="Agent-backed response with trace metadata.",
        context={"agent_trace": _sample_trace().model_dump(mode="json"), "workflow": {'id': 'journal_pass', 'display_name': 'Journal Pass', 'status': 'completed'}},
        tags=["chat", "message"],
        severity="info",
        message_require_read_receipt=True,
        message_expires_at=None,
        message_first_seen_at=None,
        message_opened_at=None,
        message_dismissed_at=None,
        message_escalated_at=None,
    )

    state = sql_notify_api._chat_message_to_schema(row, receipts=None)

    assert state.agent_trace is not None
    assert state.agent_trace.corr_id == "corr-agent-visible"
    assert state.agent_trace.tool_call_count == 2
    assert state.workflow['id'] == 'journal_pass'
