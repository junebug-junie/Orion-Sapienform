from __future__ import annotations

from typing import Any

from orion.schemas.cockpit_sighting import CockpitHopStatusV1, CockpitHopV1, CockpitStageV1

_HUB_PRODUCER = "orion-hub"


def _base_hop(
    *,
    correlation_id: str,
    seq: int,
    stage: CockpitStageV1,
    visor_line: str,
    status: str,
    summary: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    producer: str = _HUB_PRODUCER,
) -> CockpitHopV1:
    return CockpitHopV1(
        correlation_id=correlation_id,
        seq=seq,
        stage=stage,
        visor_line=visor_line,
        status=status,  # type: ignore[arg-type]
        summary=summary or {},
        raw=raw or {},
        producer=producer,
    )


def hop_from_progress(
    *,
    correlation_id: str,
    seq: int,
    stage: CockpitStageV1,
    status: CockpitHopStatusV1,
    visor_line: str,
    summary: dict[str, Any] | None = None,
    raw: dict[str, Any] | None = None,
    producer: str = _HUB_PRODUCER,
) -> CockpitHopV1:
    """Honest pre-motor / boundary progress hop from a real Hub event."""
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage=stage,
        visor_line=visor_line,
        status=status,
        summary=summary,
        raw=raw,
        producer=producer,
    )


def _association_signal_count(association: dict[str, Any]) -> int | None:
    """Best-effort signal_count from broadcast.frame.debug, else open_loops len."""
    broadcast = association.get("broadcast")
    if not isinstance(broadcast, dict):
        return None
    frame = broadcast.get("frame")
    if not isinstance(frame, dict):
        return None
    debug = frame.get("debug")
    if isinstance(debug, dict) and "signal_count" in debug:
        try:
            return int(debug["signal_count"])
        except (TypeError, ValueError):
            pass
    open_loops = frame.get("open_loops")
    if isinstance(open_loops, list):
        return len(open_loops)
    return None


def hop_from_ingress(
    *,
    correlation_id: str,
    seq: int,
    ingress: dict[str, Any],
) -> CockpitHopV1:
    """User-text / turn-intake hop. Honest about observation publish when omitted."""
    raw = dict(ingress) if isinstance(ingress, dict) else {}
    user_message = str(raw.get("user_message") or "")
    msg_len = len(user_message)
    visor_line = f"ingress · {msg_len} chars" if msg_len else "ingress · empty"
    try:
        attachment_count = int(raw.get("attachment_count") or 0)
    except (TypeError, ValueError):
        attachment_count = 0
    summary: dict[str, Any] = {
        "user_message_len": msg_len,
        "attachment_count": attachment_count,
    }
    if "session_id" in raw:
        summary["session_id"] = raw.get("session_id")
    mode = raw.get("mode")
    if mode is None:
        mode = raw.get("client_mode")
    if mode is not None:
        summary["mode"] = mode
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="ingress",
        visor_line=visor_line,
        status="ok",
        summary=summary,
        raw=raw,
    )


def hop_from_association(
    *,
    correlation_id: str,
    seq: int,
    association: dict[str, Any],
) -> CockpitHopV1:
    stale = bool(association.get("broadcast_stale"))
    label = "stale" if stale else "fresh"
    signal_count = _association_signal_count(association)
    hollow = (not stale) and signal_count == 0
    if hollow:
        visor_line = f"association · {label} · empty"
    else:
        visor_line = f"association · {label}"
    summary: dict[str, Any] = {
        "broadcast_stale": stale,
        "read_source": association.get("read_source"),
    }
    if signal_count is not None:
        summary["signal_count"] = signal_count
        summary["hollow"] = hollow
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="association",
        visor_line=visor_line,
        status="ok",
        summary=summary,
        raw=dict(association),
    )


def hop_from_stance_inputs(
    *,
    correlation_id: str,
    seq: int,
    stance_inputs: dict[str, Any],
) -> CockpitHopV1:
    user_message = str(stance_inputs.get("user_message") or "")
    summary: dict[str, Any] = {"user_message_len": len(user_message)}
    # Ambient, not a status report (see recent_attention_cue.py's docstring):
    # surfaced here only as an inspectable count/staleness pair, same posture
    # as everything else in this summary -- Juniper can see what Oríon's
    # stance synthesis saw without it becoming a mandatory field to narrate.
    # NOTE: `stance_inputs` here is turn_orchestrator.py's outer wrapper
    # ({"user_message":..., "session_id":..., "llm_profile":...,
    # "stance_inputs": <the real per-turn dict build_chat_stance_inputs()
    # returned>}) -- the real dict `_inject_recent_attention_to_inputs`
    # writes into is one level deeper, at stance_inputs["stance_inputs"].
    nested_inputs = stance_inputs.get("stance_inputs")
    nested_inputs = nested_inputs if isinstance(nested_inputs, dict) else {}
    recent_attention = nested_inputs.get("recent_attention")
    if isinstance(recent_attention, dict) and recent_attention:
        items = recent_attention.get("items")
        summary["recent_attention_items"] = len(items) if isinstance(items, list) else 0
        summary["recent_attention_stale"] = bool(recent_attention.get("stale"))
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="stance_inputs",
        visor_line=f"stance inputs · {len(user_message)} chars",
        status="ok",
        summary=summary,
        raw=dict(stance_inputs),
    )


def hop_from_motor_boot(
    *,
    correlation_id: str,
    seq: int,
    prompt: str,
    producer: str = "orion-harness-governor",
) -> CockpitHopV1:
    text = prompt if isinstance(prompt, str) else ""
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="motor_boot",
        visor_line=f"motor_boot · {len(text)} chars",
        status="ok",
        summary={"prompt_char_len": len(text)},
        raw={"prompt": text, "prompt_char_len": len(text)},
        producer=producer,
    )


_SITUATION_VISOR_CHANNELS = (
    "cabinet",
    "weather",
    "time",
    "presence",
    "perception",
    "affect",
    "curiosity",
    "reverie",
    "runtime",
    "surface",
)


def _situation_visor_bits(
    *,
    compact_text: str,
    provider_status: dict[str, Any] | None,
    source_summary: dict[str, Any] | None,
) -> list[str]:
    """Name channels that honestly contributed, for the helmet line."""
    bits: list[str] = []
    status = provider_status if isinstance(provider_status, dict) else {}
    sources = source_summary if isinstance(source_summary, dict) else {}
    text_l = compact_text.lower()
    for name in _SITUATION_VISOR_CHANNELS:
        st = str(status.get(name) or "").lower()
        src = str(sources.get(name) or "").lower()
        if st in {"ok", "partial"} or (
            src
            and src
            not in {
                "disabled",
                "unavailable",
                "unconfigured",
                "empty",
                "error",
                "stub",
            }
        ):
            bits.append(name)
            continue
        # Heuristic fallback when diagnostics are thin but compact_text names it.
        if name in text_l or (name == "weather" and "weather" in text_l):
            bits.append(name)
    return bits


def hop_from_situation(
    *,
    correlation_id: str,
    seq: int,
    compact_text: str | None,
    status: CockpitHopStatusV1 = "ok",
    provider_status: dict[str, Any] | None = None,
    source_summary: dict[str, Any] | None = None,
    perception_enabled: bool | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> CockpitHopV1:
    """Situation fragment hop: exact compact_text that rides on the harness request."""
    text = compact_text if isinstance(compact_text, str) else ""
    has_fragment = bool(text.strip())
    cabinet_mentioned = "cabinet" in text.lower()
    if status == "failed":
        visor_line = "situation · failed"
        hop_status: CockpitHopStatusV1 = "failed"
    elif status == "skipped":
        visor_line = "situation · skipped"
        hop_status = "skipped"
    elif not has_fragment:
        visor_line = "situation · empty"
        hop_status = "ok" if status == "ok" else status
    else:
        bits = _situation_visor_bits(
            compact_text=text,
            provider_status=provider_status,
            source_summary=source_summary,
        )
        visor_line = (
            f"situation · {'+'.join(bits)}" if bits else f"situation · {len(text)} chars"
        )
        hop_status = "ok"

    summary: dict[str, Any] = {
        "compact_text_len": len(text),
        "has_fragment": has_fragment,
        "cabinet_mentioned": cabinet_mentioned,
    }
    if perception_enabled is not None:
        summary["perception_enabled"] = bool(perception_enabled)
    if isinstance(provider_status, dict) and provider_status:
        summary["provider_status"] = dict(provider_status)

    raw: dict[str, Any] = {
        "compact_text": text,
        "has_fragment": has_fragment,
        "cabinet_mentioned": cabinet_mentioned,
    }
    if perception_enabled is not None:
        raw["perception_enabled"] = bool(perception_enabled)
    if isinstance(provider_status, dict):
        raw["provider_status"] = dict(provider_status)
    if isinstance(source_summary, dict):
        raw["source_summary"] = dict(source_summary)
    if isinstance(diagnostics, dict) and diagnostics:
        raw["diagnostics"] = dict(diagnostics)

    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="situation",
        visor_line=visor_line,
        status=hop_status,
        summary=summary,
        raw=raw,
    )


def hop_from_thought(
    *,
    correlation_id: str,
    seq: int,
    thought: dict[str, Any],
) -> CockpitHopV1:
    disposition = str(thought.get("disposition", "unknown"))
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="stance_decision",
        visor_line=f"stance · {disposition}",
        status="ok",
        summary={"disposition": disposition},
        raw=dict(thought),
    )


def hop_from_motor_step(
    *,
    correlation_id: str,
    seq: int,
    step_index: int,
    step: dict[str, Any],
) -> CockpitHopV1:
    step_name = step.get("name") or step.get("type") or "step"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="motor_hop",
        visor_line=f"motor · {step_name}",
        status="ok",
        summary={"step_index": step_index},
        raw={"step_index": step_index, "step": step},
    )


def _has_draft_appraisal(run: dict[str, Any]) -> bool:
    return bool(run.get("draft_text") or run.get("substrate_appraisal"))


def _has_finalize(run: dict[str, Any]) -> bool:
    return bool(
        run.get("reflection")
        or run.get("final_text")
        or run.get("finalize_ran")
    )


def hop_from_run_artifact(
    *,
    correlation_id: str,
    seq: int,
    run: dict[str, Any],
) -> list[CockpitHopV1]:
    hops: list[CockpitHopV1] = []
    current_seq = seq

    if _has_draft_appraisal(run):
        hops.append(
            _base_hop(
                correlation_id=correlation_id,
                seq=current_seq,
                stage="draft_appraisal",
                visor_line="draft · substrate appraisal",
                status="ok",
                summary={
                    "draft_text_len": len(str(run.get("draft_text") or "")),
                },
                raw=dict(run),
            )
        )
        current_seq += 1

    if _has_finalize(run):
        compliance = run.get("compliance_verdict") or "completed"
        hops.append(
            _base_hop(
                correlation_id=correlation_id,
                seq=current_seq,
                stage="finalize",
                visor_line=f"finalize · {compliance}",
                status="ok",
                summary={"compliance_verdict": compliance},
                raw=dict(run),
            )
        )

    return hops


def hop_from_outcome(
    *,
    correlation_id: str,
    seq: int,
    outcome: dict[str, Any],
) -> CockpitHopV1:
    label = outcome.get("status") or outcome.get("kind") or "outcome"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="closure",
        visor_line=f"outcome · {label}",
        status="ok",
        summary={"outcome": label},
        raw=dict(outcome),
    )


def hop_from_closure(
    *,
    correlation_id: str,
    seq: int,
    closure: dict[str, Any],
) -> CockpitHopV1:
    label = closure.get("status") or closure.get("kind") or "closure"
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage="closure",
        visor_line=f"closure · {label}",
        status="ok",
        summary={"closure": label},
        raw=dict(closure),
    )


def gap_hop(
    *,
    correlation_id: str,
    seq: int,
    stage: CockpitStageV1,
    deferred_to: str = "slice_b",
) -> CockpitHopV1:
    return _base_hop(
        correlation_id=correlation_id,
        seq=seq,
        stage=stage,
        visor_line=f"gap · {stage} not recorded ({deferred_to})",
        status="gap",
        summary={"deferred_to": deferred_to},
        raw={},
    )


def extract_mind_quality_fields(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pull Mind quality flags from a Thought reply / nested artifact if present.

    ThoughtEventV1 does not currently carry these fields. Returns None when Hub
    has nothing honest to show (caller should emit mind_details_unavailable).
    """
    if not isinstance(payload, dict):
        return None
    candidates: list[dict[str, Any]] = [payload]
    for key in ("mind", "mind_run", "mind_brief", "brief", "mind_coloring", "metadata"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    keys = (
        "mind_quality",
        "fallback_contract_only",
        "authorized_for_stance_use",
        "coloring_skipped",
    )
    found: dict[str, Any] = {}
    for blob in candidates:
        for key in keys:
            if key in blob and key not in found:
                found[key] = blob[key]
        mq = blob.get("mind_quality")
        if mq == "fallback_contract_only" and "fallback_contract_only" not in found:
            found["fallback_contract_only"] = True
    return found or None
