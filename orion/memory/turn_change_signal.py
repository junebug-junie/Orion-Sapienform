from __future__ import annotations

from datetime import datetime, timezone

from orion.signals.models import OrganClass, OrionSignalV1
from orion.signals.signal_ids import make_signal_id


def dimensions_for_shift(*, shift_kind: str, novelty_score: float) -> dict[str, float]:
    ns = max(0.0, min(1.0, float(novelty_score)))
    if shift_kind == "TOPIC":
        return {"novelty": ns, "salience": ns}
    if shift_kind == "STANCE":
        return {"contradiction": ns, "salience": ns}
    if shift_kind == "REPAIR":
        return {"contradiction": ns}
    return {"salience": min(0.15, ns * 0.2)}


def build_turn_change_signal(
    *,
    correlation_id: str,
    shift_kind: str,
    novelty_score: float,
    confidence: float,
) -> OrionSignalV1:
    now = datetime.now(timezone.utc)
    dims = dimensions_for_shift(shift_kind=shift_kind, novelty_score=novelty_score)
    dims["confidence"] = max(0.0, min(1.0, float(confidence)))
    hub_parent = make_signal_id("hub", correlation_id)
    return OrionSignalV1(
        signal_id=make_signal_id("memory_consolidation", correlation_id),
        organ_id="memory_consolidation",
        organ_class=OrganClass.endogenous,
        signal_kind="turn_change",
        dimensions=dims,
        causal_parents=[hub_parent],
        source_event_id=correlation_id,
        observed_at=now,
        emitted_at=now,
        summary=f"turn_change shift={shift_kind} novelty={novelty_score:.2f}",
        notes=[],
    )
