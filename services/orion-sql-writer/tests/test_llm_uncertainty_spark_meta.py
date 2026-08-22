from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_uncertainty_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

_UNC = {"schema_version": "v1", "available": True, "mean_logprob": -0.4}


class _FakeQuery:
    def filter(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def first(self):
        return None


class _FakeSession:
    """chat_history_log now goes through INSERT ... ON CONFLICT DO UPDATE, so the
    written values arrive via execute(stmt), not merge(orm_object)."""

    def __init__(self) -> None:
        self.row = None
        self.values: dict = {}

    def query(self, _model):  # noqa: ANN001
        return _FakeQuery()

    def merge(self, obj):
        self.row = obj

    def execute(self, stmt, params=None):
        # _lock_chat_history_row's advisory-lock text() statement passes a
        # params dict as a second positional arg -- not the real
        # chat_history_log write this fake exists to capture, so skip it.
        if params is not None:
            return
        # compile().params exposes the bound INSERT values without a live DB.
        self.values = dict(stmt.compile().params)

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


def _write_chat_history(monkeypatch, payload: dict) -> dict:
    sess = _FakeSession()
    monkeypatch.setattr(worker, "get_session", lambda: sess)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    assert worker._write_row(worker.ChatHistoryLogSQL, payload) is True
    spark_meta = sess.values.get("spark_meta")
    assert isinstance(spark_meta, dict), f"no spark_meta in written values: {sess.values}"
    return spark_meta


def _write_chat_history_values(monkeypatch, payload: dict) -> dict:
    """Full set of columns the upsert would write, for scalar assertions."""
    sess = _FakeSession()
    monkeypatch.setattr(worker, "get_session", lambda: sess)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    assert worker._write_row(worker.ChatHistoryLogSQL, payload) is True
    return sess.values


def test_chat_history_spark_meta_merges_llm_uncertainty_from_meta(monkeypatch) -> None:
    spark_meta = _write_chat_history(
        monkeypatch,
        {
            "correlation_id": "corr-meta-unc",
            "prompt": "hello",
            "response": "world",
            "meta": {"llm_uncertainty": _UNC},
        },
    )
    assert spark_meta["llm_uncertainty"] == _UNC


def test_chat_history_spark_meta_merges_llm_uncertainty_from_spark_meta(monkeypatch) -> None:
    spark_meta = _write_chat_history(
        monkeypatch,
        {
            "correlation_id": "corr-spark-unc",
            "prompt": "hello",
            "response": "world",
            "spark_meta": {"trace_verb": "chat_general", "llm_uncertainty": _UNC},
        },
    )
    assert spark_meta["llm_uncertainty"] == _UNC
    assert spark_meta["trace_verb"] == "chat_general"


_UNC_FULL = {
    "schema_version": "v1",
    "available": True,
    "source": "llamacpp_native_completion",
    "mean_logprob": -0.74,
    "min_logprob": -3.5,
    "mean_top1_margin": 1.2,
    "low_margin_token_count": 2,
    "low_logprob_token_count": 1,
    "unstable_span_count": 2,
}


def _write_chat_history_row(monkeypatch, payload: dict):
    """Column values the upsert writes, addressable by attribute so the
    assertions below read the same as when this was an ORM object."""
    return SimpleNamespace(**_write_chat_history_values(monkeypatch, payload))


def test_chat_history_log_scalar_columns_from_meta_llm_uncertainty(monkeypatch) -> None:
    row = _write_chat_history_row(
        monkeypatch,
        {
            "correlation_id": "corr-scalar-unc",
            "prompt": "hello",
            "response": "world",
            "meta": {"llm_uncertainty": _UNC_FULL},
        },
    )
    assert row.llm_uncertainty_source == "llamacpp_native_completion"
    assert row.llm_mean_logprob == pytest.approx(-0.74, rel=1e-3)
    assert row.llm_min_logprob == pytest.approx(-3.5, rel=1e-3)
    assert row.llm_mean_top1_margin == pytest.approx(1.2, rel=1e-3)
    assert row.llm_low_margin_token_count == 2
    assert row.llm_low_logprob_token_count == 1
    assert row.llm_unstable_span_count == 2
    assert row.llm_uncertainty_available is True
    assert row.spark_meta["llm_uncertainty"] == _UNC_FULL
