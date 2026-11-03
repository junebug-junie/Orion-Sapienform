"""Canonical result statuses for custom dispatch routes.

The theater tripwire consumes the same ``success|empty|failed`` vocabulary that
``substrate_dispatch_results`` documents.  Custom RPCs return a settlement state
instead of prose, so their useful result is the state itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CANONICAL_RESULT_STATUSES = frozenset({"success", "empty", "failed"})
PRODUCTIVE_SETTLEMENT_STATES = frozenset({"started", "control_recorded"})


def status_for_settlement(state: str | None, *, error: str | None = None) -> str:
    """Map a custom RPC settlement onto the existing result-status contract."""
    if error or state == "rpc_error":
        return "failed"
    if state in PRODUCTIVE_SETTLEMENT_STATES:
        return "success"
    # A refusal (or another non-productive reply) is a real send with no usable
    # result, which is exactly the documented meaning of ``empty``.
    return "empty"


def status_for_stored_settlement(row: Mapping[str, Any]) -> str:
    """Read canonical status from a result row, including legacy ``succeeded`` rows.

    Old custom-route rows flattened both ``started`` and ``refused`` to
    ``succeeded``.  Replays recover the actual meaning from the preserved
    settlement instead of teaching the tripwire that legacy spelling.
    """
    stored = str(row.get("status") or "")
    if stored in CANONICAL_RESULT_STATUSES:
        return stored
    result_json = row.get("result_json")
    if not isinstance(result_json, Mapping):
        return "empty"
    settlement = result_json.get("settlement")
    if not isinstance(settlement, Mapping):
        return "empty"
    error = result_json.get("error")
    return status_for_settlement(
        str(settlement.get("state")) if settlement.get("state") is not None else None,
        error=str(error) if error else None,
    )


__all__ = [
    "CANONICAL_RESULT_STATUSES",
    "PRODUCTIVE_SETTLEMENT_STATES",
    "status_for_settlement",
    "status_for_stored_settlement",
]
