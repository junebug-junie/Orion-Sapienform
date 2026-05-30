from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/orion-substrate-runtime"))

from app.receipt_pruner import SAFE_PRUNE_DELETE_SQL_FRAGMENT


def test_safe_prune_sql_requires_expires_and_applied_or_error():
    assert "expires_at < now()" in SAFE_PRUNE_DELETE_SQL_FRAGMENT
    assert "substrate_field_applied_deltas" in SAFE_PRUNE_DELETE_SQL_FRAGMENT
