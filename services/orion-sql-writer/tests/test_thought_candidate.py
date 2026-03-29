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
    assert source == "reasoning_trace.content"


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
