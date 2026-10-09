"""Read back the `:IncidentReport` an urgent curiosity run wrote to Orion's own graph.

The urgent prompt (`orion/curiosity/urgent_prompt.py`) asks Orion to write one
node with these exact property names; this module is the other half of that
contract. Same rules as `worldview.read_turn_outcome`: read-only, the run id is
validated before it reaches a query string, and it never raises -- a durable
run finishing must not fail because the report could not be read.

A report counts only when every verdict field is usable. A half-written node is
reported as `no_structured_verdict` rather than filled in with guesses; the
caller still delivers Orion's prose alongside that flag.

EVIDENCE IS UNWOUND, NOT READ AS A LIST. FalkorDB's default (non-compact) reply
renders a list property as one bracketed string, and entries contain commas, so
it cannot be split (see `self_inquiry.self_definition_evidence_cypher`). Checked
live 2026-09-28 against this deployment: `toJSON` does not escape inner quotes or
newlines and `reduce` errors on a string-valued property, while `UNWIND` yields
one row per list entry and one row for a plain string. A bare UNWIND yields no
rows for null or [], which would hide a newer evidence-less report behind an
older valid one; unwinding `coalesce(r.evidence, []) + [null]` keeps at least
one (null-evidence) row per node, also checked live.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, get_args

from orion.curiosity.worldview import _RUN_ID_RE, _as_float, _stamp_ms

logger = logging.getLogger("orion.curiosity.incident_report")

LABEL_INCIDENT_REPORT = "IncidentReport"
NO_STRUCTURED_VERDICT = "no_structured_verdict"

IsReal = Literal["real", "sensor_fault", "unclear"]
Severity = Literal["low", "high", "critical"]
IS_REAL_VALUES: tuple[str, ...] = get_args(IsReal)
SEVERITY_VALUES: tuple[str, ...] = get_args(Severity)

# Rows, not reports: one row per evidence entry. A safety cap, not a page size.
_ROW_LIMIT = 500


@dataclass(frozen=True)
class IncidentReport:
    incident_id: str
    is_real: IsReal
    likely_cause: str
    evidence: tuple[str, ...]
    severity: Severity
    operator_action: str
    confidence: float  # clamped to [0, 1]


def incident_report_for_run_cypher(run_id: str) -> str:
    """Every `:IncidentReport` THIS run wrote, one row per evidence entry.

    Newest-wins is decided in Python (`_stamp_ms` reads both `timestamp()` and
    ISO), so `ORDER BY` here only keeps the newest node's rows inside the cap.
    """
    if not _RUN_ID_RE.match(str(run_id or "")):
        raise ValueError(f"refusing to build Cypher for a non-hex run_id: {run_id!r}")
    return (
        f"MATCH (r:{LABEL_INCIDENT_REPORT}) WHERE r.run_id = '{run_id}' "
        "UNWIND coalesce(r.evidence, []) + [null] AS e "
        "RETURN id(r) AS node_id, r.run_id AS run_id, r.incident_id AS incident_id, "
        "r.is_real AS is_real, r.likely_cause AS likely_cause, e AS evidence, "
        "r.severity AS severity, r.operator_action AS operator_action, "
        "r.confidence AS confidence, r.written_at AS written_at "
        f"ORDER BY r.written_at DESC LIMIT {_ROW_LIMIT}"
    )


def _node_key(row: dict[str, Any]) -> Any:
    node_id = row.get("node_id")
    if node_id is not None:
        return ("id", str(node_id))
    return ("fields", str(row.get("written_at")), str(row.get("incident_id")))


def _evidence_items(value: Any) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in items if item is not None and str(item).strip()]


def build_incident_report(rows: list[dict[str, Any]]) -> Optional[IncidentReport]:
    """The report from one node's rows, or None if any verdict field is unusable."""
    if not rows:
        return None
    head = rows[0]
    is_real = str(head.get("is_real") or "").strip().lower()
    severity = str(head.get("severity") or "").strip().lower()
    likely_cause = str(head.get("likely_cause") or "").strip()
    operator_action = str(head.get("operator_action") or "").strip()
    confidence = _as_float(head.get("confidence"))
    evidence: list[str] = []
    for row in rows:
        for item in _evidence_items(row.get("evidence")):
            if item not in evidence:
                evidence.append(item)
    if (
        is_real not in IS_REAL_VALUES
        or severity not in SEVERITY_VALUES
        or not likely_cause
        or not operator_action
        or confidence is None
        or math.isnan(confidence)
        or not evidence
    ):
        return None
    return IncidentReport(
        incident_id=str(head.get("incident_id") or "").strip(),
        is_real=is_real,  # type: ignore[arg-type]
        likely_cause=likely_cause,
        evidence=tuple(evidence),
        severity=severity,  # type: ignore[arg-type]
        operator_action=operator_action,
        confidence=min(1.0, max(0.0, confidence)),
    )


def read_incident_report(reader: Any, run_id: str) -> tuple[Optional[IncidentReport], Optional[str]]:
    """`(report, None)` for a usable report; `(None, "no_structured_verdict")` otherwise.

    Otherwise covers: no node, a malformed or evidence-less newest node, a bad
    run id, and any reader failure. Never raises.
    """
    try:
        rows = list(reader.query(incident_report_for_run_cypher(run_id)) or [])
    except Exception as exc:  # noqa: BLE001 -- the caller's run must still finish
        logger.warning("curiosity_incident_report_read_failed run=%s err=%s: %s", run_id, type(exc).__name__, exc)
        return None, NO_STRUCTURED_VERDICT

    nodes: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict):
            nodes.setdefault(_node_key(row), []).append(row)
    if not nodes:
        return None, NO_STRUCTURED_VERDICT

    newest = max(nodes.values(), key=lambda group: _stamp_ms(group[0].get("written_at")) or 0)
    report = build_incident_report(newest)
    if report is None:
        logger.warning(
            "curiosity_incident_report_malformed run=%s nodes=%d row=%s",
            run_id,
            len(nodes),
            {k: v for k, v in newest[0].items() if k != "evidence"},
        )
        return None, NO_STRUCTURED_VERDICT
    return report, None
