"""Compile turn-local AutonomyStateV2 evidence with proven-non-empty gates.

Must be called with explicit locals (social / social_bridge / reasoning flags).
Never reads ctx["chat_social_bridge_summary"] — that key is written after the
reducer runs on the live chat path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from orion.autonomy.models import AutonomyEvidenceRefV1

_INFRA_AVAIL = frozenset({"available", "degraded", "empty", "unavailable"})

# Kind-literal confidences (uncalibrated v1 constants).
_CONF_USER = 0.9
_CONF_INFRA = 1.0
_CONF_REASONING = 0.6
_CONF_RELATIONAL = 0.6


@dataclass
class AutonomyEvidenceCompileResult:
    evidence: list[AutonomyEvidenceRefV1] = field(default_factory=list)
    omitted: list[dict[str, str]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _unique_hazards(*groups: Any) -> list[str]:
    out: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        raw = group.get("hazards") or []
        if not isinstance(raw, (list, tuple)):
            continue
        for h in raw:
            s = str(h).strip()
            if s and s not in out:
                out.append(s)
    return out


def compile_autonomy_evidence(
    *,
    user_message: Any,
    social: Any,
    social_bridge: Any,
    reasoning_summary: Any,
    reasoning_upstream_nonempty: bool,
    autonomy_debug: Any,
    now: datetime,
) -> AutonomyEvidenceCompileResult:
    """Emit evidence only when upstream is proven non-empty. Never raises."""
    result = AutonomyEvidenceCompileResult()
    try:
        msg = ""
        try:
            msg = str(user_message or "").strip()
        except Exception:
            msg = ""
        if msg:
            digest = hashlib.sha256(msg[:200].encode()).hexdigest()[:16]
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id=f"user_turn:{digest}",
                    source="user_message",
                    kind="user_turn",
                    summary=msg[:200],
                    confidence=_CONF_USER,
                    observed_at=now,
                )
            )
        else:
            result.omitted.append({"kind": "user_turn", "reason": "empty_message"})

        debug = autonomy_debug if isinstance(autonomy_debug, dict) else {}
        orion_dbg = debug.get("orion") if isinstance(debug.get("orion"), dict) else {}
        avail = str(orion_dbg.get("availability") or "").strip()
        if avail in _INFRA_AVAIL:
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id=f"infra_health:autonomy_graph:{avail}",
                    source="infra",
                    kind="infra_health",
                    summary=f"autonomy graph availability={avail}",
                    confidence=_CONF_INFRA,
                    observed_at=now,
                )
            )
        elif avail:
            result.omitted.append(
                {"kind": "infra_health", "reason": "availability_not_recognized"}
            )
        else:
            result.omitted.append({"kind": "infra_health", "reason": "missing_availability"})

        rs = reasoning_summary if isinstance(reasoning_summary, dict) else {}
        fallback = bool(rs.get("fallback_recommended"))
        if not reasoning_upstream_nonempty:
            result.omitted.append({"kind": "reasoning_quality", "reason": "empty_upstream"})
        elif not fallback:
            result.omitted.append(
                {"kind": "reasoning_quality", "reason": "no_quality_signal"}
            )
        else:
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id="reasoning:fallback_recommended",
                    source="reasoning",
                    kind="reasoning_quality",
                    summary="reasoning fallback recommended",
                    confidence=_CONF_REASONING,
                    observed_at=now,
                    signal_kind="chat_reasoning_quality",
                    dimension="fallback",
                    value=1.0,
                )
            )

        hazards = _unique_hazards(social, social_bridge)
        if not hazards:
            result.omitted.append({"kind": "relational_signal", "reason": "no_hazards"})
        for hazard in hazards:
            hid = hashlib.sha256(hazard[:80].encode()).hexdigest()[:12]
            # Always stamp typed fields; SignalDriveMap.match is the sole pressure gate.
            result.evidence.append(
                AutonomyEvidenceRefV1(
                    evidence_id=f"social_bridge:{hid}",
                    source="social_bridge",
                    kind="relational_signal",
                    summary=hazard[:200],
                    confidence=_CONF_RELATIONAL,
                    observed_at=now,
                    signal_kind="chat_social_hazard",
                    dimension=hazard,
                    value=1.0,
                )
            )

        result.debug = {
            "emitted_kinds": [e.kind for e in result.evidence],
            "omitted": list(result.omitted),
            "hazard_count": len(hazards),
            "reasoning_upstream_nonempty": bool(reasoning_upstream_nonempty),
        }
    except Exception as exc:
        result.omitted.append({"kind": "_compiler", "reason": f"degraded:{type(exc).__name__}"})
        result.debug = {"error": type(exc).__name__}
    return result
