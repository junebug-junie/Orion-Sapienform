from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from orion.schemas.memory_consolidation import ChatHistorySparkMetaPatchV1

WORKER_PATH = SQL_WRITER_ROOT / "app" / "worker.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_worker_spark_patch_tests", WORKER_PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class _ExistingRow:
    def __init__(self, spark_meta: dict):
        self.spark_meta = spark_meta


class _FakeQuery:
    def __init__(self, row):
        self._row = row

    def filter(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def first(self):
        return self._row


class _FakeSession:
    def __init__(self, row):
        self._row = row
        self.updated = None

    def query(self, _model):  # noqa: ANN001
        return _FakeQuery(self._row)

    def execute(self, stmt):
        self.updated = stmt
        if self._row is not None:
            self._row.spark_meta = worker._merge_spark_meta(
                getattr(self._row, "spark_meta", None),
                {"memory_significance_score": 0.8},
            )

    def commit(self):
        return None

    def close(self):
        return None


def test_spark_meta_patch_merges_existing_row(monkeypatch):
    corr = "turn-corr-1"
    row = _ExistingRow({"foo": 1})
    sess = _FakeSession(row)
    monkeypatch.setattr(worker, "get_session", lambda: sess)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    patch = ChatHistorySparkMetaPatchV1(
        correlation_id=corr,
        spark_meta={"memory_significance_score": 0.8},
    )
    ok = worker._apply_spark_meta_patch(patch.model_dump(mode="json"))

    assert ok is True
    assert row.spark_meta["foo"] == 1
    assert row.spark_meta["memory_significance_score"] == pytest.approx(0.8)


def test_spark_meta_patch_missing_row(monkeypatch, caplog):
    sess = _FakeSession(None)
    monkeypatch.setattr(worker, "get_session", lambda: sess)
    monkeypatch.setattr(worker, "remove_session", lambda: None)

    patch = ChatHistorySparkMetaPatchV1(correlation_id="missing", spark_meta={"memory_significance_score": 0.5})
    ok = worker._apply_spark_meta_patch(patch.model_dump(mode="json"))

    assert ok is False
    assert any("spark_meta_patch_missing_row" in r.message for r in caplog.records)
