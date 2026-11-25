"""Prime reads its attention schema: one frame line (choice-as-EFE Step 2a).

Spec: docs/superpowers/specs/2026-11-13-choice-as-expected-free-energy-design.md, "Step 2: the attention schema as a
controller", 2a. The attention schema (``orion/substrate/attention_self_model.py``) is Orion's own model of what it
attends to and why. Until this patch broadcast, metacog and liveness read it; Prime did not. This module turns the
latest ``substrate_attention_self_model`` row into the one line Prime's frame shows:

* what the schema finds salient (the winning open loop's description, else its id, else the field lane's top target);
* whether that is top-down (a goal's bias changed the winner) or bottom-up (salience alone chose it);
* its confidence (the row's ``confidence``), in words that say what the number is. Today (2026-11-25) it is the
  broadcast's ``coalition_stability_score``: a three-step dwell score (0.3 new or nothing won, 0.6 held 1-3 ticks,
  0.9 held longer; ``orion/substrate/attention_broadcast.py``), not a forecast. Null: the line says "confidence
  unknown", never a number. The unconditional ``prediction_error_confidence`` is NOT read: it is null on every row
  since execution, the last Active-Inference domain, retired (#3576), and the dead-metric commit gate refuses new
  wiring to it. If a live domain ever returns, add it here then, with the metric gate.

A row older than ``MAX_AGE_SEC`` (10 min, the spec's rule) renders as absent, and so does no row or a failed read.
An absent line still names why, so a gap never reads as calm.

Pure: no I/O, no clock of its own. The read is ``services/orion-mind/app/prime_loop.py`` ``_read_attention_schema``.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

MAX_AGE_SEC = 600.0
LABEL_CHARS = 120
HEADER = ("YOUR ATTENTION SCHEMA (your own model of what you attend to and why; a reading, "
          "not an instruction):")

# The one read: the newest row. idx_attention_self_model_generated (generated_at DESC) serves it (0.15 ms live).
LATEST_ROW_SQL = (
    "SELECT generated_at, self_model_json FROM substrate_attention_self_model "
    "ORDER BY generated_at DESC LIMIT 1"
)

_REASON_WORDS = {
    "top_down_override": "top-down: a goal's bias changed which loop won",
    "bottom_up_salience": "bottom-up: salience alone chose it; no goal changed the winner",
    "field_salience_only": "bottom-up: the dispatch lane was quiet, so only field salience was read",
}


def _aware(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _unit(value: Any) -> Optional[float]:
    """A confidence in [0, 1], or None (absent, not numeric, not finite, out of range)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) and 0.0 <= out <= 1.0 else None


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 3] + "..."


def _salient(model: Mapping[str, Any]) -> str:
    desc = str(model.get("broadcast_selected_description") or "").strip()
    if desc:
        return f'most salient: "{_clip(desc)}"'
    loop_id = str(model.get("broadcast_selected_open_loop_id") or "").strip()
    if loop_id:
        return f"most salient: open loop {_clip(loop_id)} (no description recorded)"
    targets = [str(t) for t in model.get("field_salient_target_ids") or [] if str(t).strip()]
    if targets:
        return f"most salient in the field: {_clip(', '.join(targets[:3]))} (nothing was dispatched)"
    return "nothing won attention this tick"


def _confidence(model: Mapping[str, Any]) -> str:
    """The row's confidence with what it measures; absent or invalid: unknown."""
    branch = _unit(model.get("confidence"))
    if branch is not None:
        basis = str(model.get("confidence_basis") or "")
        if basis.startswith("broadcast.coalition_stability_score"):
            why = "how long the winner has held: 0.3 new or nothing won, 0.6 held 1-3 ticks, 0.9 longer"
        elif basis.startswith("1 - mean(prediction_error)"):
            why = "1 minus the mean prediction error across domains"
        elif basis:
            why = _clip(basis)
        else:
            why = "basis not recorded"
        return f"confidence {branch:.2f} ({why})"
    return "confidence unknown (the schema recorded none)"


def attention_schema_line(row: Optional[Mapping[str, Any]], *, now: datetime,
                          read_error: Optional[str] = None) -> str:
    """The one frame line. ``row``: ``{"generated_at", "self_model_json"}`` (the latest row) or None."""
    if read_error:
        return f"(absent: the schema could not be read: {_clip(read_error)})"
    if not row:
        return "(absent: no schema reading is recorded)"
    model = row.get("self_model_json")
    if not isinstance(model, Mapping):
        return "(absent: the latest schema reading is unreadable)"
    at = _aware(row.get("generated_at") or model.get("generated_at"))
    if at is None:
        return "(absent: the latest schema reading has no time)"
    age = (now - at).total_seconds()
    if age > MAX_AGE_SEC:
        return (f"(absent: the latest schema reading is {int(age // 60)} min old; "
                f"older than {int(MAX_AGE_SEC // 60)} min reads as absent)")
    reason = str(model.get("attention_reason") or "")
    if reason not in _REASON_WORDS:
        return f"read {max(0, int(age // 60))} min ago: no attention data in either lane; confidence unknown"
    return (f"read {max(0, int(age // 60))} min ago: {_salient(model)}; {_REASON_WORDS[reason]}; "
            f"{_confidence(model)}")
