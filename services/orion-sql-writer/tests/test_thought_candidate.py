from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_thought_candidate_prefers_reasoning_trace_over_other_sources() -> None:
    payload = {
        "reasoning_trace": {"content": "trace-thought"},
        "reasoning_content": "reasoning-content",
        "metacog_traces": [{"trace_role": "reasoning", "content": "metacog-thought"}],
    }
    thought, source = worker._thought_candidate_and_reason(payload)
    assert thought == "trace-thought"
    assert source == "reasoning_trace.content(none)"


def test_thought_candidate_uses_metacog_when_reasoning_content_missing() -> None:
    payload = {
        "reasoning_trace": {},
        "metacog_traces": [{"trace_role": "reasoning", "trace_stage": "post_answer", "content": "metacog-only"}],
    }
    thought, source = worker._thought_candidate_and_reason(payload)
    assert thought == "metacog-only"
    assert source == "metacog_traces[0].content"


def test_thought_candidate_returns_none_when_no_reasoning_payload() -> None:
    payload = {
        "prompt": "hello",
        "response": "world",
    }
    thought, source = worker._thought_candidate_and_reason(payload)
    assert thought is None
    assert source == "none"


def test_thought_candidate_uses_inline_think_when_marked_as_inline_source() -> None:
    payload = {
        "reasoning_trace": {"content": "trace-thought"},
        "reasoning_content": "provider-structured",
        "inline_think_content": "inline-think",
        "thinking_source": "inline_think_close_tag_only",
    }
    thought, source = worker._thought_candidate_and_reason(payload)
    assert thought == "inline-think"
    assert source == "inline_think_content.inline_think_close_tag_only"


def test_thought_candidate_prefers_chat_general_llm_step_inline_think() -> None:
    payload = {
        "spark_meta": {
            "trace_verb": "chat_general",
            "thought_capture_step": "llm_chat_general",
        },
        "inline_think_content": "authoritative-thought",
        "reasoning_content": "non-authoritative",
        "thinking_source": "provider_reasoning",
    }
    thought, source = worker._thought_candidate_and_reason(payload)
    assert thought == "authoritative-thought"
    assert source == "inline_think_content.chat_general_llm_chat_general"


class _FakeRow:
    def __init__(self, thought_process: str | None) -> None:
        self.thought_process = thought_process


class _FakeQuery:
    def __init__(self, row: _FakeRow | None) -> None:
        self._row = row

    def filter(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def first(self):
        return self._row


class _FakeSession:
    def __init__(self, row: _FakeRow | None) -> None:
        self._row = row

    def query(self, _model):  # noqa: ANN001
        return _FakeQuery(self._row)


def test_chat_history_thought_for_merge_preserves_existing_non_empty_thought() -> None:
    sess = _FakeSession(_FakeRow("chat-thought"))
    resolved = worker._chat_history_thought_for_merge(
        sess,
        {"id": "corr-1", "thought_process": "follow-on-thought"},
        {"correlation_id": "corr-1"},
    )
    assert resolved == "chat-thought"


def test_chat_history_thought_for_merge_writes_insert_and_update_when_empty_existing() -> None:
    insert_sess = _FakeSession(None)
    insert_resolved = worker._chat_history_thought_for_merge(
        insert_sess,
        {"id": "corr-2", "thought_process": "chat-thought"},
        {"correlation_id": "corr-2"},
    )
    assert insert_resolved == "chat-thought"

    update_sess = _FakeSession(_FakeRow(""))
    update_resolved = worker._chat_history_thought_for_merge(
        update_sess,
        {"id": "corr-2", "thought_process": "chat-thought-refresh"},
        {"correlation_id": "corr-2"},
    )
    assert update_resolved == "chat-thought-refresh"
