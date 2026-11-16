"""Which substrate nodes are ALARMS, not prediction errors -- stdlib only, safe to import anywhere.

The full rationale lives in orion/substrate/prediction_error_freshness.py (which re-exports these). The
set is here so light modules (orion/field/pressure.py, imported by the field digester) can ask the
question without importing the heavy orion.substrate package. tests/test_pe_alarm_refile.py checks it
equals the glossary's trigger-kind prediction_error entries.
"""
from __future__ import annotations

from typing import Any

ALARM_NODE_IDS: frozenset[str] = frozenset(
    {"node:substrate.cabinet", "node:substrate.strain", "node:substrate.street"}
)


def is_alarm_node(node_id: Any) -> bool:
    """True when ``node_id`` is an alarm/news node, not a prediction error."""
    return str(node_id or "") in ALARM_NODE_IDS
