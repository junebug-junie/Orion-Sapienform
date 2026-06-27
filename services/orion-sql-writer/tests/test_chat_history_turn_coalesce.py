from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_coalesce_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))
SPEC.loader.exec_module(worker)


def test_coalesce_preserves_existing_response_when_incoming_empty():
    filtered = {"correlation_id": "abc", "prompt": "hi", "response": ""}
    existing = SimpleNamespace(prompt="hi", response="assistant reply")
    worker._coalesce_chat_history_turn_fields(filtered, existing)
    assert filtered["response"] == "assistant reply"


def test_merge_spark_meta_telemetry_does_not_clobber_classify_novelty():
    existing = {
        "turn_change_appraisal": {"turn_change_status": "ok", "novelty_score": 0.91},
        "novelty": 0.91,
        "turn_effect": {"turn": {"novelty": 0.91}},
    }
    telemetry = {"novelty": 0.0, "phi": 0.8}
    merged = worker._merge_spark_meta(existing, telemetry, source="telemetry")
    assert merged["novelty"] == 0.91
    assert merged["turn_effect"] == existing["turn_effect"]
