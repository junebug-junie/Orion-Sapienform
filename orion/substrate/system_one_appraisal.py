from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import requests

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEventV1,
    GrammarProjectionV1,
    GrammarProvenanceV1,
)
from orion.schemas.system_one_appraisal import (
    SystemOneAnswerV1,
    SystemOneAppraisalFrameV1,
    SystemOneFieldTargetInputV1,
    SystemOneInputStateV1,
    SystemOneOpenLoopInputV1,
    SystemOneQuestionV1,
    SystemOneUsageV1,
)

# Bump this id whenever instructions, criteria, or question membership changes.
# It participates in frame identity, preventing a revised decision contract from
# silently colliding with frames produced by an older one.
QUESTION_SET_ID = "orion.system_one.shadow.v1"
MAX_TEXT_CHARS = 512
MAX_CHANNELS_PER_TARGET = 8

# These are shadow appraisals, not behavior thresholds. Their point is to
# collect a calibrated probability surface over real Orion state before any
# consumer is allowed to act on it.
SYSTEM_ONE_QUESTIONS: dict[str, SystemOneQuestionV1] = {
    "reverie_fit": SystemOneQuestionV1(
        type="score",
        instructions=(
            "Using only the supplied Orion attention state, rate how suitable it is "
            "for attention to drift into internally generated reverie right now. "
            "Do not invent facts that are not in the state."
        ),
        criteria=[
            "Stay externally engaged; reverie is a poor fit right now",
            "Either continued engagement or reverie is plausible",
            "Current state is a strong fit for internally generated reverie",
        ],
    ),
    "curiosity_pull": SystemOneQuestionV1(
        type="score",
        instructions=(
            "Using only the supplied state, rate the unresolved pull toward further "
            "investigation or inquiry. Treat absent evidence as uncertainty, not as "
            "evidence of curiosity."
        ),
        criteria=[
            "Little or no unresolved pull worth pursuing",
            "Some unresolved pull; observation or deferral is reasonable",
            "Strong unresolved pull worth an investigation",
        ],
    ),
    "deliberation_need": SystemOneQuestionV1(
        type="score",
        instructions=(
            "Using only the supplied state, rate whether the situation warrants "
            "expensive deliberate cognition instead of a cheap/reactive path."
        ),
        criteria=[
            "Cheap/reactive handling is sufficient",
            "Some deliberate cognition may help",
            "Expensive deliberate cognition is warranted",
        ],
    ),
    "attention_interrupt": SystemOneQuestionV1(
        type="score",
        instructions=(
            "Using only the supplied state, rate whether the currently represented "
            "matter should interrupt or supersede ongoing attention."
        ),
        criteria=[
            "Do not interrupt current focus",
            "May deserve attention without forcing an interruption",
            "Strong case to interrupt or supersede current focus",
        ],
    ),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _bounded(items: list[Any], limit: int) -> list[Any]:
    return list(items[: max(0, int(limit))])


def _clip_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(value or "")
    return text[: max(0, int(limit))]


def _bounded_text(items: list[Any], limit: int) -> list[str]:
    return [_clip_text(item) for item in _bounded(items, limit)]


def _bounded_mapping(values: dict[str, float], limit: int) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in list(values.items())[: max(0, int(limit))]
    }


def build_system_one_input_state(
    *,
    broadcast: AttentionBroadcastProjectionV1,
    field_frame: FieldAttentionFrameV1 | None,
    max_open_loops: int = 6,
    max_targets: int = 5,
) -> SystemOneInputStateV1:
    """Compile existing substrate artifacts into a bounded provider-neutral state.

    No new metric is created here. Values are copied from the two source
    projections with bounded text/evidence lists so the inference call remains
    inspectable and cheap.
    """

    open_loops = [
        SystemOneOpenLoopInputV1(
            loop_id=loop.id,
            target_type=loop.target_type,
            description=_clip_text(loop.description),
            why_it_matters=_clip_text(loop.why_it_matters),
            salience=loop.salience,
            combined_salience=loop.combined_salience,
            confidence=loop.confidence,
            source_refs=_bounded(loop.source_refs, 8),
        )
        for loop in _bounded(broadcast.frame.open_loops, max_open_loops)
    ]

    field_targets: list[SystemOneFieldTargetInputV1] = []
    if field_frame is not None:
        field_targets = [
            SystemOneFieldTargetInputV1(
                target_id=target.target_id,
                target_kind=target.target_kind,
                salience_score=target.salience_score,
                pressure_score=target.pressure_score,
                novelty_score=target.novelty_score,
                urgency_score=target.urgency_score,
                confidence_score=target.confidence_score,
                dominant_channels=_bounded_mapping(
                    target.dominant_channels, MAX_CHANNELS_PER_TARGET
                ),
                reasons=_bounded_text(target.reasons, 6),
                evidence_refs=_bounded(target.evidence_refs, 8),
            )
            for target in _bounded(field_frame.dominant_targets, max_targets)
        ]

    return SystemOneInputStateV1(
        source_broadcast_projection_id=broadcast.projection_id,
        source_broadcast_generated_at=broadcast.generated_at,
        source_field_attention_frame_id=field_frame.frame_id if field_frame else None,
        source_field_attention_generated_at=field_frame.generated_at if field_frame else None,
        selected_action_type=broadcast.selected_action_type,
        selected_open_loop_id=broadcast.selected_open_loop_id,
        selected_description=(
            _clip_text(broadcast.selected_description)
            if broadcast.selected_description is not None
            else None
        ),
        attended_node_ids=_bounded(broadcast.attended_node_ids, 12),
        dwell_ticks=broadcast.dwell_ticks,
        coalition_stability_score=broadcast.coalition_stability_score,
        effort_budget_used=broadcast.frame.effort_budget_used,
        voluntary_override_present=broadcast.frame.voluntary_override is not None,
        live_unknowns=_bounded_text(broadcast.frame.live_unknowns, 6),
        deferred_items=_bounded_text(broadcast.frame.deferred_items, 6),
        open_loops=open_loops,
        field_overall_salience=field_frame.overall_salience if field_frame else None,
        field_dominant_targets=field_targets,
    )


def _frame_id(state: SystemOneInputStateV1, *, provider: str, model: str) -> str:
    source = "|".join(
        [
            QUESTION_SET_ID,
            provider,
            model,
            state.source_broadcast_projection_id,
            state.source_broadcast_generated_at.isoformat(),
            state.source_field_attention_frame_id or "no-field-frame",
            (
                state.source_field_attention_generated_at.isoformat()
                if state.source_field_attention_generated_at
                else "no-field-time"
            ),
        ]
    )
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    return f"system-one-appraisal-{digest}"


def _answer_from_payload(question_id: str, raw: Any) -> SystemOneAnswerV1:
    if not isinstance(raw, dict):
        raise ValueError(f"System One answer {question_id!r} is not an object")
    answer_type = raw.get("type")
    if answer_type not in {"noul", "choice", "score"}:
        raise ValueError(f"System One answer {question_id!r} has invalid type {answer_type!r}")

    probabilities_raw = raw.get("probabilities")
    probabilities: dict[str, float] = {}
    if probabilities_raw is not None:
        if not isinstance(probabilities_raw, dict):
            raise ValueError(f"System One probabilities for {question_id!r} are not an object")
        probabilities = {str(k): float(v) for k, v in probabilities_raw.items()}

    legend_raw = raw.get("legend")
    legend: dict[str, str] = {}
    if legend_raw is not None:
        if not isinstance(legend_raw, dict):
            raise ValueError(f"System One legend for {question_id!r} is not an object")
        legend = {str(k): str(v) for k, v in legend_raw.items()}

    return SystemOneAnswerV1(
        question_id=question_id,
        type=answer_type,
        choice=str(raw["choice"]) if raw.get("choice") is not None else None,
        noul=float(raw["noul"]) if raw.get("noul") is not None else None,
        score=float(raw["score"]) if raw.get("score") is not None else None,
        confidence=float(raw["confidence"]) if raw.get("confidence") is not None else None,
        probabilities=probabilities,
        legend=legend,
    )


def run_system_one_appraisal(
    *,
    broadcast: AttentionBroadcastProjectionV1,
    field_frame: FieldAttentionFrameV1 | None,
    base_url: str,
    model: str,
    provider: str = "kev",
    api_key: str = "",
    timeout_sec: float = 3.0,
    ttl_sec: float = 90.0,
    max_open_loops: int = 6,
    max_targets: int = 5,
    post: Callable[..., Any] = requests.post,
    now: datetime | None = None,
) -> SystemOneAppraisalFrameV1:
    """Run one schema-bounded System One inference over current substrate state.

    Raises on transport/schema/incomplete-answer failures. The worker owns the
    fail-open boundary and therefore never persists an empty or partial
    cognition-shaped artifact.
    """

    if not base_url.strip():
        raise ValueError("System One base_url is required when appraisal is enabled")

    state = build_system_one_input_state(
        broadcast=broadcast,
        field_frame=field_frame,
        max_open_loops=max_open_loops,
        max_targets=max_targets,
    )
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/systemone"
    elif not endpoint.endswith("/v1/systemone"):
        endpoint += "/v1/systemone"

    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    request_payload = {
        "state": state.model_dump(mode="json"),
        "model": model,
        "questions": {
            key: question.model_dump(mode="json", exclude_none=True)
            for key, question in SYSTEM_ONE_QUESTIONS.items()
        },
    }
    response = post(
        endpoint,
        json=request_payload,
        headers=headers,
        timeout=float(timeout_sec),
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("System One response is not an object")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict) or not raw_answers:
        raise ValueError("System One response has no answers")

    answers = {
        key: _answer_from_payload(key, raw_answers.get(key))
        for key in SYSTEM_ONE_QUESTIONS
    }

    usage_raw = payload.get("usage")
    usage = SystemOneUsageV1()
    if isinstance(usage_raw, dict):
        usage = SystemOneUsageV1(
            input_tokens=(
                int(usage_raw["input_tokens"])
                if usage_raw.get("input_tokens") is not None
                else None
            ),
            output_tokens=(
                int(usage_raw["output_tokens"])
                if usage_raw.get("output_tokens") is not None
                else None
            ),
        )

    generated_at = now or _utc_now()
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    ttl = max(1.0, float(ttl_sec))

    source_refs = [
        (
            "attention.broadcast:"
            f"{state.source_broadcast_projection_id}@{state.source_broadcast_generated_at.isoformat()}"
        )
    ]
    if state.source_field_attention_frame_id:
        source_refs.append(f"field.attention:{state.source_field_attention_frame_id}")

    actual_model = str(payload.get("model") or model)

    request_id = None
    headers_obj = getattr(response, "headers", None)
    if headers_obj is not None:
        request_id = headers_obj.get("x-typesafe-request-id")

    return SystemOneAppraisalFrameV1(
        frame_id=_frame_id(state, provider=provider, model=actual_model),
        question_set_id=QUESTION_SET_ID,
        generated_at=generated_at,
        expires_at=generated_at + timedelta(seconds=ttl),
        provider=provider,
        model_id=actual_model,
        request_id=request_id,
        source_refs=source_refs,
        input_state=state,
        questions=SYSTEM_ONE_QUESTIONS,
        answers=answers,
        latency_ms=(
            float(payload["latency_ms"]) if payload.get("latency_ms") is not None else None
        ),
        usage=usage,
    )


def _answer_digest(frame: SystemOneAppraisalFrameV1) -> str:
    parts: list[str] = []
    for key, answer in frame.answers.items():
        if answer.type == "score":
            parts.append(f"{key}=score:{answer.score:.3f}")
        elif answer.type == "noul":
            parts.append(f"{key}=yes_p:{answer.noul:.3f}")
        else:
            parts.append(f"{key}=choice:{answer.choice}")
    return ", ".join(parts)


def build_system_one_grammar_events(
    frame: SystemOneAppraisalFrameV1,
) -> list[GrammarEventV1]:
    """Emit the causal shadow of the compiled frame, never the raw frame JSON."""

    trace_id = f"substrate.system_one:{frame.frame_id}"
    input_atom_id = f"{trace_id}:input"
    provenance = GrammarProvenanceV1(
        source_service="orion-substrate-runtime",
        source_component="system_one_appraisal",
        source_trace_id=trace_id,
        source_payload_ref=f"substrate_system_one_appraisal:{frame.frame_id}",
        model_id=frame.model_id,
    )
    common = {
        "trace_id": trace_id,
        "emitted_at": frame.generated_at,
        "observed_at": frame.generated_at,
        "layer": "substrate.system_one",
        "dimensions": list(frame.answers.keys()),
        "provenance": provenance,
    }

    return [
        GrammarEventV1(
            event_id=f"{trace_id}:start",
            event_kind="trace_started",
            **common,
        ),
        GrammarEventV1(
            event_id=f"{trace_id}:input-event",
            event_kind="atom_emitted",
            atom=GrammarAtomV1(
                atom_id=input_atom_id,
                trace_id=trace_id,
                atom_type="observation",
                semantic_role="system_one_bounded_input_state",
                layer="substrate.system_one",
                dimensions=["attention", "field_attention"],
                summary=(
                    "Bounded attention state compiled for shadow System One appraisal "
                    f"({len(frame.input_state.open_loops)} open loops, "
                    f"{len(frame.input_state.field_dominant_targets)} field targets)."
                ),
                time_range=None,
                payload_ref=f"substrate_system_one_appraisal:{frame.frame_id}",
            ),
            **common,
        ),
        GrammarEventV1(
            event_id=f"{trace_id}:projection-event",
            event_kind="projection_emitted",
            projection=GrammarProjectionV1(
                projection_id=frame.frame_id,
                trace_id=trace_id,
                source_atom_ids=[input_atom_id],
                projection_type="system_one_shadow_appraisal",
                summary=(
                    f"{frame.provider}/{frame.model_id} shadow appraisal: "
                    f"{_answer_digest(frame)}"
                ),
                confidence=None,
                expires_at=frame.expires_at,
                projected_atom_id=None,
            ),
            **common,
        ),
        GrammarEventV1(
            event_id=f"{trace_id}:end",
            event_kind="trace_ended",
            **common,
        ),
    ]
