from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_guard = Path(__file__).resolve().parent / "_exec_import_guard.py"
_spec = importlib.util.spec_from_file_location("_exec_guard_boot", _guard)
assert _spec and _spec.loader
_guard_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard_mod)

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def _exec_prep() -> None:
    _guard_mod.ensure_orion_cortex_exec_app()


from orion.schemas.cortex.schemas import ExecutionStep


def test_shortcut_returns_result_when_handoff_valid() -> None:
    _exec_prep()
    from app.executor import _attempt_mind_handoff_chat_stance_shortcut

    ctx = {
        "metadata": {
            "mind_skip_stance_synthesis": True,
            "mind_handoff": {
                "stance_payload": {
                    "conversation_frame": "technical",
                    "user_intent": "u",
                    "self_relevance": "s",
                    "juniper_relevance": "j",
                    "answer_strategy": "DirectAnswer",
                    "stance_summary": "st",
                }
            },
        }
    }
    merged: dict = {}
    logs: list[str] = []

    def record(*args, **kwargs):
        pass

    step = ExecutionStep(
        verb_name="chat_general",
        step_name="synthesize_chat_stance_brief",
        order=1,
        services=["LLMGatewayService"],
    )
    out = _attempt_mind_handoff_chat_stance_shortcut(
        step=step,
        service="LLMGatewayService",
        ctx=ctx,
        merged_result=merged,
        logs=logs,
        correlation_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        spark_vector=None,
        t0=0.0,
        record_scoped_step=record,
        node_name="test-node",
    )
    assert out is not None
    assert out.status == "success"
    assert merged.get("ChatStanceBrief")


def test_shortcut_returns_none_when_payload_invalid() -> None:
    _exec_prep()
    from app.executor import _attempt_mind_handoff_chat_stance_shortcut

    ctx = {
        "metadata": {
            "mind_skip_stance_synthesis": True,
            "mind_handoff": {"stance_payload": {"conversation_frame": "not-a-valid-frame"}},
        }
    }
    merged = {}
    step = ExecutionStep(
        verb_name="chat_general",
        step_name="synthesize_chat_stance_brief",
        order=1,
        services=["LLMGatewayService"],
    )
    out = _attempt_mind_handoff_chat_stance_shortcut(
        step=step,
        service="LLMGatewayService",
        ctx=ctx,
        merged_result=merged,
        logs=[],
        correlation_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        spark_vector=None,
        t0=0.0,
        record_scoped_step=lambda *a, **k: None,
        node_name="n",
    )
    assert out is None
