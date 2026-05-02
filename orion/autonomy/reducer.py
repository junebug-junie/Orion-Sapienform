from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable

from pydantic import BaseModel, Field

from orion.autonomy.models import (
    ActionOutcomeRefV1,
    AutonomyEvidenceRefV1,
    AutonomyStateDeltaV1,
    AutonomyStateV1,
    AutonomyStateV2,
    CandidateImpulseV1,
    InhibitedImpulseV1,
    AttentionItemV1,
    upgrade_autonomy_state_v1_to_v2,
)
from orion.autonomy.repository import SUBJECT_BINDINGS, SubjectBinding

_MAX_EVIDENCE = 20
_MAX_ATTENTION = 8
_MAX_CANDIDATE_IMPULSES = 8
_MAX_INHIBITED_IMPULSES = 8
_MAX_OUTCOMES = 12
_MAX_UNKNOWNS = 12

_DRIVE_KEYS = ("coherence", "continuity", "relational", "autonomy", "capability", "predictive")


def _utc_now(inp_now: datetime | None) -> datetime:
    if inp_now is not None:
        return inp_now
    return datetime.utcnow()


def _binding_for_subject(subject: str) -> SubjectBinding:
    return SUBJECT_BINDINGS.get(
        subject,
        SubjectBinding(model_layer="unknown", entity_id=subject),
    )


def _cold_state(subject: str, *, now: datetime) -> AutonomyStateV2:
    b = _binding_for_subject(subject)
    return AutonomyStateV2(
        subject=subject,
        model_layer=b.model_layer,
        entity_id=b.entity_id,
        source="reducer",
        generated_at=now,
        schema_version="autonomy.state.v2",
        confidence=0.25,
        unknowns=["no_previous_state"],
        evidence_refs=[],
        freshness={},
        attention_items=[],
        candidate_impulses=[],
        inhibited_impulses=[],
        last_action_outcomes=[],
        drive_pressures={k: 0.0 for k in _DRIVE_KEYS},
    )


def _normalize_pressures(raw: dict[str, float]) -> dict[str, float]:
    out = {k: float(raw.get(k, 0.0) or 0.0) for k in _DRIVE_KEYS}
    rs = float(raw.get("relational_stability") or 0.0)
    if rs:
        out["relational"] = max(out["relational"], rs)
    for k, v in raw.items():
        if k in _DRIVE_KEYS and k != "relational":
            out[k] = max(out[k], float(v or 0.0))
    return {k: min(1.0, max(0.0, out[k])) for k in _DRIVE_KEYS}


def _evidence_text(ev: AutonomyEvidenceRefV1) -> str:
    return f"{ev.kind} {(ev.summary or '')}".lower()


def _apply_single_evidence_pressures(pressures: dict[str, float], ev: AutonomyEvidenceRefV1) -> dict[str, float]:
    if ev.source == "user_message" or ev.kind == "infra_health":
        return pressures
    text = _evidence_text(ev)
    weight = 0.5 if ev.kind == "proxy_telemetry" else 1.0
    inc: dict[str, float] = {k: 0.0 for k in _DRIVE_KEYS}

    def hit(tokens: tuple[str, ...], drive: str, delta: float) -> None:
        if any(t in text for t in tokens):
            inc[drive] += delta * weight

    hit(("contradiction", "inconsistency", "failure", "bug", "broken", "drift", "confusion"), "coherence", 0.12)
    hit(("memory", "recall", "thread", "history", "stale", "missing context"), "continuity", 0.10)
    hit(("frustration", "repair", "trust", "relationship", "social", "apology"), "relational", 0.10)
    hit(("proposal", "self-modification", "workflow", "autonomous", "action"), "autonomy", 0.08)
    hit(("tool failure", "missing capability", "timeout", "unavailable", "error"), "capability", 0.10)
    hit(("surprise", "unexpected", "regression", "mismatch"), "predictive", 0.08)

    out = dict(pressures)
    for k in _DRIVE_KEYS:
        added = min(0.15, inc[k])
        if added:
            out[k] = min(1.0, out[k] + added)
    return out


def _dominant_and_active(pressures: dict[str, float], prev_dominant: str | None) -> tuple[str | None, list[str]]:
    ranked = sorted(_DRIVE_KEYS, key=lambda k: pressures.get(k, 0.0), reverse=True)
    top_p = pressures.get(ranked[0], 0.0) if ranked else 0.0
    dominant: str | None
    if top_p >= 0.15:
        dominant = ranked[0]
    else:
        dominant = (prev_dominant or "").strip() or None
    active = [k for k in ranked if pressures.get(k, 0.0) >= 0.12][:3]
    return dominant, active


def _combined_evidence_text(evs: Iterable[AutonomyEvidenceRefV1]) -> str:
    return " ".join(_evidence_text(e) for e in evs)


def _derive_tension_kinds(pressures: dict[str, float], evidence_blob: str) -> list[str]:
    kinds: list[str] = []
    coh = pressures.get("coherence", 0.0)
    cont = pressures.get("continuity", 0.0)
    cap = pressures.get("capability", 0.0)
    rel = pressures.get("relational", 0.0)

    def add(name: str) -> None:
        if name not in kinds:
            kinds.append(name)

    if coh >= 0.25 or any(
        x in evidence_blob for x in ("contradiction", "inconsistency", "bug", "broken", "confusion")
    ):
        add("tension.coherence_break.v1")
    if cont >= 0.25 or any(x in evidence_blob for x in ("stale", "missing context", "recall", "memory failure")):
        add("tension.continuity_gap.v1")
    if cap >= 0.25 or any(x in evidence_blob for x in ("unavailable", "timeout", "error", "tool failure")):
        add("tension.capability_gap.v1")
    if rel >= 0.25 or any(x in evidence_blob for x in ("frustration", "trust", "apology", "repair")):
        add("tension.relational_repair.v1")

    ranked = sorted(_DRIVE_KEYS, key=lambda k: pressures.get(k, 0.0), reverse=True)
    if len(ranked) >= 2:
        p1 = pressures.get(ranked[0], 0.0)
        p2 = pressures.get(ranked[1], 0.0)
        if p1 >= 0.25 and p2 >= 0.25 and abs(p1 - p2) < 0.08:
            add("tension.drive_competition.v1")

    return kinds


def _merge_evidence(
    existing: list[AutonomyEvidenceRefV1],
    incoming: list[AutonomyEvidenceRefV1],
) -> list[AutonomyEvidenceRefV1]:
    seq = 0
    bucket: dict[str, tuple[int, AutonomyEvidenceRefV1]] = {}
    for e in existing:
        bucket[e.evidence_id] = (seq, e)
        seq += 1
    for e in incoming:
        bucket[e.evidence_id] = (seq, e)
        seq += 1
    ordered = sorted(bucket.items(), key=lambda kv: kv[1][0])
    items = [t[1] for _, t in ordered]
    epoch = datetime(1970, 1, 1)
    stamped = [(i, ev.observed_at or epoch, ev) for i, ev in enumerate(items)]
    stamped.sort(key=lambda x: (x[1], x[0]))
    if len(stamped) > _MAX_EVIDENCE:
        stamped = stamped[-_MAX_EVIDENCE:]
    return [x[2] for x in sorted(stamped, key=lambda x: x[0])]


def _merge_outcomes(
    existing: list[ActionOutcomeRefV1],
    incoming: list[ActionOutcomeRefV1],
) -> list[ActionOutcomeRefV1]:
    seq = 0
    bucket: dict[str, tuple[int, ActionOutcomeRefV1]] = {}
    for o in existing:
        bucket[o.action_id] = (seq, o)
        seq += 1
    for o in incoming:
        bucket[o.action_id] = (seq, o)
        seq += 1
    ordered = sorted(bucket.values(), key=lambda t: t[0])
    out = [t[1] for t in ordered]
    return out[-_MAX_OUTCOMES:]


def _impulse_id(subject: str, kind: str) -> str:
    return hashlib.sha256(f"{subject}:impulse:{kind}".encode()).hexdigest()[:16]


def _maybe_attention_items(
    subject: str,
    dominant: str | None,
    tensions: list[str],
    evidence_ids: list[str],
) -> list[AttentionItemV1]:
    dom = (dominant or "").strip()
    if not dom and not tensions:
        return []
    seed_kind = "attention_seed"
    item_id = hashlib.sha256(f"{subject}:{seed_kind}:{dom or ''}".encode()).hexdigest()[:16]
    parts: list[str] = []
    if dom:
        parts.append(f"dominant_drive={dom}")
    if tensions:
        parts.append("tensions=" + ",".join(tensions[:6]))
    summary = "; ".join(parts) if parts else "attention"
    return [
        AttentionItemV1(
            item_id=item_id,
            summary=summary,
            source="reducer",
            salience=0.72,
            drive_links=[dom] if dom else [],
            tension_links=list(tensions)[:6],
            evidence_refs=list(evidence_ids)[:6],
        )
    ]


def _trim_bounded_unknowns(vals: list[str]) -> list[str]:
    out: list[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
        if len(out) >= _MAX_UNKNOWNS:
            break
    return out


class AutonomyReducerInputV1(BaseModel):
    subject: str = "orion"
    previous_state: AutonomyStateV1 | AutonomyStateV2 | None = None
    evidence: list[AutonomyEvidenceRefV1] = Field(default_factory=list)
    action_outcomes: list[ActionOutcomeRefV1] = Field(default_factory=list)
    now: datetime | None = None


class AutonomyReducerResultV1(BaseModel):
    state: AutonomyStateV2
    delta: AutonomyStateDeltaV1


def reduce_autonomy_state(inp: AutonomyReducerInputV1) -> AutonomyReducerResultV1:
    """Deterministic autonomy appraisal for one turn (no I/O)."""
    now = _utc_now(inp.now)
    subject = inp.subject or "orion"

    if inp.previous_state is None:
        working = _cold_state(subject, now=now)
    elif isinstance(inp.previous_state, AutonomyStateV2):
        working = inp.previous_state.model_copy(deep=True)
    else:
        working = upgrade_autonomy_state_v1_to_v2(inp.previous_state)

    baseline = working.model_copy(deep=True)
    baseline_conf = float(baseline.confidence)

    had_previous = inp.previous_state is not None

    merged_evidence = _merge_evidence(list(working.evidence_refs), list(inp.evidence))
    working.evidence_refs = merged_evidence
    working.last_action_outcomes = _merge_outcomes(list(working.last_action_outcomes), list(inp.action_outcomes))

    pressures = _normalize_pressures(dict(working.drive_pressures))
    prev_dom = working.dominant_drive

    for ev in inp.evidence:
        pressures = _apply_single_evidence_pressures(pressures, ev)

    working.drive_pressures = {k: pressures[k] for k in _DRIVE_KEYS}
    blob = _combined_evidence_text(working.evidence_refs)
    dominant, active_drives = _dominant_and_active(pressures, prev_dom)
    working.dominant_drive = dominant
    working.active_drives = active_drives
    working.tension_kinds = _derive_tension_kinds(pressures, blob)

    attn = _maybe_attention_items(
        subject,
        dominant,
        working.tension_kinds,
        [e.evidence_id for e in working.evidence_refs[:8]],
    )
    if len(attn) > _MAX_ATTENTION:
        attn = attn[-_MAX_ATTENTION:]
    working.attention_items = attn

    candidates: list[CandidateImpulseV1] = []
    prev_kinds = {c.kind for c in baseline.candidate_impulses}

    def push_impulse(kind: str, summary: str, drive_origin: str | None) -> None:
        if kind in prev_kinds:
            return
        candidates.append(
            CandidateImpulseV1(
                impulse_id=_impulse_id(subject, kind),
                kind=kind,
                summary=summary,
                drive_origin=drive_origin,
                confidence=min(1.0, pressures.get(drive_origin or "", 0.0)),
                evidence_refs=[],
            )
        )

    if pressures.get("coherence", 0.0) >= 0.35:
        push_impulse("synthesize_or_reduce", "reduce contradiction pressure", "coherence")
    if pressures.get("capability", 0.0) >= 0.35:
        push_impulse("triage_capability_gap", "address capability / dependency fault", "capability")
    if pressures.get("continuity", 0.0) >= 0.35:
        push_impulse("recover_context", "recover missing thread context", "continuity")
    if pressures.get("relational", 0.0) >= 0.35:
        push_impulse("repair_or_acknowledge", "relational repair signal", "relational")
    if pressures.get("autonomy", 0.0) >= 0.35:
        push_impulse("propose_bounded_action", "bounded autonomous action proposal", "autonomy")

    proxy_only = bool(working.evidence_refs) and all(e.kind == "proxy_telemetry" for e in working.evidence_refs)

    unknowns: list[str] = []
    if not had_previous:
        unknowns.append("no_previous_state")
    if not inp.evidence:
        unknowns.append("no_fresh_evidence")
    if proxy_only:
        unknowns.append("proxy_only_evidence")
    if not working.last_action_outcomes:
        unknowns.append("no_action_outcome_history")
    if working.latest_identity_snapshot_id is None:
        unknowns.append("no_identity_snapshot")
    if working.latest_drive_audit_id is None:
        unknowns.append("no_drive_audit")

    working.unknowns = _trim_bounded_unknowns(unknowns)

    conf = float(baseline_conf)
    high_direct = 0.0
    for ev in working.evidence_refs:
        if ev.kind != "proxy_telemetry" and float(ev.confidence) >= 0.7:
            high_direct += 1.0
    conf += min(0.10, 0.05 * min(high_direct, 2.0))

    if proxy_only:
        conf -= 0.10
    if "stale" in blob or "missing context" in blob:
        conf -= 0.05
    if "timeout" in blob or "unavailable" in blob:
        conf -= 0.05

    surprise_budget = 0.0
    for out in working.last_action_outcomes:
        if float(out.surprise) >= 0.7:
            surprise_budget += 0.08
    conf -= min(0.12, surprise_budget)

    working.confidence = min(1.0, max(0.0, conf))

    inhibited: list[InhibitedImpulseV1] = []
    final_candidates: list[CandidateImpulseV1] = []
    evid_ids = [e.evidence_id for e in working.evidence_refs][:6]

    if proxy_only and working.confidence < 0.6:
        for c in candidates:
            inhibited.append(
                InhibitedImpulseV1(
                    impulse_id=c.impulse_id,
                    kind=c.kind,
                    summary=c.summary,
                    inhibition_reason="proxy_signal_not_canonical_state",
                    evidence_refs=evid_ids,
                )
            )
        if not inhibited:
            inhibited.append(
                InhibitedImpulseV1(
                    impulse_id=_impulse_id(subject, "proxy_guard"),
                    kind="canonical_state_guard",
                    summary="proxy-only telemetry",
                    inhibition_reason="proxy_signal_not_canonical_state",
                    evidence_refs=evid_ids,
                )
            )
    else:
        for c in candidates:
            reason: str | None = None
            if c.kind == "propose_bounded_action" and pressures.get("autonomy", 0.0) >= 0.35 and working.confidence < 0.45:
                reason = "low_confidence_for_autonomous_action"
            elif c.kind == "triage_capability_gap" and pressures.get("capability", 0.0) >= 0.35:
                if "timeout" in blob or "unavailable" in blob:
                    reason = "dependency_unavailable"
            if reason:
                inhibited.append(
                    InhibitedImpulseV1(
                        impulse_id=c.impulse_id,
                        kind=c.kind,
                        summary=c.summary,
                        inhibition_reason=reason,
                        evidence_refs=evid_ids,
                    )
                )
            else:
                final_candidates.append(c)

    if len(final_candidates) > _MAX_CANDIDATE_IMPULSES:
        final_candidates = final_candidates[:_MAX_CANDIDATE_IMPULSES]
    if len(inhibited) > _MAX_INHIBITED_IMPULSES:
        inhibited = inhibited[-_MAX_INHIBITED_IMPULSES:]

    working.candidate_impulses = final_candidates
    working.inhibited_impulses = inhibited
    working.generated_at = now

    fresh: dict[str, str] = {"state_generated_at": now.isoformat()}
    direct_times = [e.observed_at for e in working.evidence_refs if e.kind != "proxy_telemetry" and e.observed_at]
    proxy_times = [e.observed_at for e in working.evidence_refs if e.kind == "proxy_telemetry" and e.observed_at]
    if direct_times:
        fresh["latest_direct_evidence_at"] = max(direct_times).isoformat()
    if proxy_times:
        fresh["latest_proxy_evidence_at"] = max(proxy_times).isoformat()
    working.freshness = fresh

    base_dump = baseline.model_dump(mode="json")
    final_dump = working.model_dump(mode="json")
    changed = [k for k in base_dump if base_dump.get(k) != final_dump.get(k)]
    drive_deltas: dict[str, float] = {}
    bp = _normalize_pressures({k: float(baseline.drive_pressures.get(k, 0.0) or 0.0) for k in _DRIVE_KEYS})
    fp = _normalize_pressures({k: float(working.drive_pressures.get(k, 0.0) or 0.0) for k in _DRIVE_KEYS})
    for k in _DRIVE_KEYS:
        d = fp[k] - bp[k]
        if abs(d) > 1e-9:
            drive_deltas[k] = round(d, 6)

    bt = set(baseline.tension_kinds or [])
    ft = set(working.tension_kinds or [])
    delta = AutonomyStateDeltaV1(
        subject=subject,
        changed_fields=changed,
        drive_deltas=drive_deltas,
        new_tensions=[x for x in working.tension_kinds if x not in bt],
        resolved_tensions=[x for x in baseline.tension_kinds if x not in ft],
        new_attention_items=[a.item_id for a in working.attention_items if a.item_id not in {x.item_id for x in baseline.attention_items}],
        new_impulses=[c.impulse_id for c in working.candidate_impulses if c.impulse_id not in {x.impulse_id for x in baseline.candidate_impulses}],
        new_inhibitions=[i.impulse_id for i in working.inhibited_impulses if i.impulse_id not in {x.impulse_id for x in baseline.inhibited_impulses}],
        confidence_delta=round(working.confidence - baseline_conf, 6),
        notes=(["compared vs upgrade baseline at turn start"][:1]),
    )

    return AutonomyReducerResultV1(state=working, delta=delta)

