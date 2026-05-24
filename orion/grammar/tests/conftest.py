"""Pytest path setup: ledger/query import SQL models from orion-sql-writer ``app`` package."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SQL_WRITER_ROOT = _REPO_ROOT / "services" / "orion-sql-writer"


def pytest_configure() -> None:
    for root in (_REPO_ROOT, _SQL_WRITER_ROOT):
        candidate = str(root)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
