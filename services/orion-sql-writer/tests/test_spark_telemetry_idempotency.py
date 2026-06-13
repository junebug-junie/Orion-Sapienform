"""Tests for idempotent spark_telemetry persistence."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

from app.spark_telemetry_persist import upsert_spark_telemetry


def test_upsert_without_correlation_id_uses_plain_insert() -> None:
    sess = MagicMock()
    data = {"phi": 0.5, "novelty": 0.1, "correlation_id": None}

    assert upsert_spark_telemetry(sess, data) is True

    sess.add.assert_called_once()
    sess.commit.assert_called_once()
    sess.execute.assert_not_called()


def test_upsert_with_correlation_id_uses_on_conflict() -> None:
    sess = MagicMock()
    data = {
        "correlation_id": "corr-123",
        "phi": 0.7,
        "novelty": 0.2,
        "trace_mode": "observe",
        "metadata_": {"source": "test"},
    }

    assert upsert_spark_telemetry(sess, data) is True

    sess.add.assert_not_called()
    sess.execute.assert_called_once()
    sess.commit.assert_called_once()


def test_upsert_filters_unknown_keys() -> None:
    sess = MagicMock()
    data = {
        "correlation_id": "corr-456",
        "phi": 0.3,
        "not_a_column": "drop-me",
    }

    upsert_spark_telemetry(sess, data)

    sess.execute.assert_called_once()
