"""
V2 scene interpretation: prompt building, LLM JSON parsing, and event projection.
Pure helpers — no bus or Redis dependencies.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pydantic import ValidationError

from orion.schemas.vision import (
    VisionEventBundleItem,
    VisionEventCandidateV1,
    VisionEventPayload,
    VisionSceneInterpretationV1,
    VisionWindowPayload,
)

from .evidence_grounding import enforce_evidence_grounding

ParseMode = Literal[
    "strict_v2",
    "salvaged_v2",
    "legacy_events_dict",
    "legacy_events_list",
    "parse_failed",
    "edge_fallback",
]


@dataclass(frozen=True)
class InterpretationParseOutcome:
    interpretation: VisionSceneInterpretationV1 | None
    parse_mode: ParseMode = "parse_failed"
    salvage_warnings: list[str] = field(default_factory=list)


def _strip_markdown_fences(content: str) -> str:
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def build_interpretation_prompt(window: VisionWindowPayload) -> str:
    summary = window.summary or {}
    context: dict[str, Any] = {
        "window_id": window.window_id,
        "stream_id": window.stream_id,
        "camera_id": window.camera_id,
        "start_ts": window.start_ts,
        "end_ts": window.end_ts,
        "summary": {
            "top_labels": summary.get("top_labels", []),
            "object_counts": summary.get("object_counts", {}),
            "captions": summary.get("captions", []),
            "detection_count": summary.get("detection_count", summary.get("item_count", 0)),
            "evidence": summary.get("evidence", {}),
        },
        "artifact_ids": window.artifact_ids or [],
    }
    if window.artifact_uris:
        context["artifact_uris"] = window.artifact_uris
    if window.freshness is not None:
        context["freshness"] = window.freshness
    if window.meta is not None:
        context["meta"] = window.meta

    schema_hint = {
        "schema_version": "1.0",
        "window_id": window.window_id,
        "stream_id": window.stream_id,
        "camera_id": window.camera_id,
        "scene_summary": "concise factual summary of what is visible",
        "scene_state": {},
        "entities": [],
        "relations": [],
        "salient_observations": [],
        "uncertainties": [],
        "task_relevance": [],
        "event_candidates": [
            {
                "event_type": "string",
                "narrative": "string",
                "entities": [],
                "tags": [],
                "confidence": 0.0,
                "salience": 0.0,
                "evidence_refs": [],
            }
        ],
        "memory_delta_candidates": [],
        "grammar_projection": None,
        "evidence_refs": [],
    }

    return (
        "Analyze this visual window and return exactly one JSON object matching "
        "VisionSceneInterpretationV1.\n\n"
        "Window context:\n"
        f"{json.dumps(context, indent=2, default=str)}\n\n"
        "Required output shape (use empty arrays instead of omitting list fields):\n"
        f"{json.dumps(schema_hint, indent=2)}\n\n"
        "Rules:\n"
        "- Output strict JSON only. No markdown fences, commentary, or prose.\n"
        "- Use empty arrays [] for missing list fields; do not omit required keys.\n"
        "- Do not invent identities, names, or facts not supported by the context.\n"
        "- Preserve uncertainty when visual evidence is weak; record it in uncertainties.\n"
        "- Prefer concrete observations over vague impressions.\n"
        "- Set evidence_refs on observations, entities, and event_candidates using "
        "artifact_ids from the window whenever possible.\n"
        "- Include scene_summary and event_candidates; other fields may be empty arrays.\n"
        "- Treat summary.evidence.hard_labels as factual detection evidence.\n"
        "- Treat summary.captions as soft hints only; never sole basis for activity claims.\n"
        "- Activity verbs require person in hard_labels.\n"
    )


def apply_evidence_pipeline(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1, list[str]]:
    return enforce_evidence_grounding(interpretation, window)


def _clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, coerced))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _default_evidence_refs(item: dict[str, Any], window: VisionWindowPayload | None) -> list[str]:
    refs = item.get("evidence_refs")
    if isinstance(refs, list) and refs:
        return [str(r) for r in refs]
    if window and window.artifact_ids:
        return list(window.artifact_ids)
    return []


def _coerce_salient_observation_item(
    item: Any,
    index: int,
    window: VisionWindowPayload | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    if isinstance(item, str):
        warnings.append(f"coerced salient_observations[{index}] string -> object")
        return {
            "observation": item,
            "confidence": 0.5,
            "salience": 0.5,
            "tags": [],
            "evidence_refs": _default_evidence_refs({}, window),
        }
    if not isinstance(item, dict):
        warnings.append(f"skipped salient_observations[{index}] uncoercible type")
        return None

    observation = (
        item.get("observation")
        or item.get("narrative")
        or item.get("summary")
        or item.get("description")
        or item.get("event_type")
    )
    if not observation:
        observation = str(item)
        warnings.append(f"coerced salient_observations[{index}] dict -> stringified observation")
    elif item.get("narrative") and not item.get("observation"):
        warnings.append(f"coerced salient_observations[{index}].narrative -> observation")
    elif item.get("event_type") and not item.get("observation") and not item.get("narrative"):
        warnings.append(f"coerced salient_observations[{index}].event_type -> observation")

    return {
        "observation": str(observation),
        "confidence": _clamp_float(item.get("confidence")),
        "salience": _clamp_float(item.get("salience")),
        "tags": _string_list(item.get("tags")),
        "evidence_refs": _default_evidence_refs(item, window),
    }


def _coerce_uncertainty_item(item: Any, index: int, warnings: list[str]) -> dict[str, Any] | None:
    if isinstance(item, str):
        warnings.append(f"coerced uncertainties[{index}] string -> object")
        return {"uncertainty": item}
    if not isinstance(item, dict):
        warnings.append(f"skipped uncertainties[{index}] uncoercible type")
        return None

    uncertainty = (
        item.get("uncertainty")
        or item.get("uncertainty_type")
        or item.get("label")
        or item.get("topic")
        or item.get("text")
        or item.get("reason")
    )
    if not uncertainty:
        uncertainty = str(item)
        warnings.append(f"coerced uncertainties[{index}] dict -> stringified uncertainty")
    elif item.get("uncertainty_type") and not item.get("uncertainty"):
        warnings.append(f"coerced uncertainties[{index}].uncertainty_type -> uncertainty")

    coerced: dict[str, Any] = {"uncertainty": str(uncertainty)}
    if item.get("reason") is not None:
        coerced["reason"] = str(item.get("reason"))
    if item.get("confidence") is not None:
        coerced["confidence"] = _clamp_float(item.get("confidence"))
    refs = item.get("evidence_refs")
    if isinstance(refs, list) and refs:
        coerced["evidence_refs"] = [str(r) for r in refs]
    return coerced


def _coerce_event_candidate_item(
    item: Any,
    index: int,
    window: VisionWindowPayload | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        warnings.append(f"skipped event_candidates[{index}] uncoercible type")
        return None

    event_type = item.get("event_type") or item.get("type") or item.get("label") or "observation"
    narrative = (
        item.get("narrative")
        or item.get("observation")
        or item.get("summary")
        or item.get("description")
        or event_type
    )
    return {
        "event_type": str(event_type),
        "narrative": str(narrative),
        "entities": _string_list(item.get("entities")),
        "tags": _string_list(item.get("tags")),
        "confidence": _clamp_float(item.get("confidence")),
        "salience": _clamp_float(item.get("salience")),
        "evidence_refs": _default_evidence_refs(item, window),
    }


def _synthesize_event_candidates_from_observations(
    observations: list[dict[str, Any]],
    window: VisionWindowPayload | None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not observations:
        return []
    warnings.append("synthesized event_candidates from salient_observations")
    synthesized: list[dict[str, Any]] = []
    for obs in observations:
        synthesized.append(
            {
                "event_type": "visual_observation",
                "narrative": obs.get("observation", ""),
                "entities": [],
                "tags": list(obs.get("tags") or []),
                "confidence": _clamp_float(obs.get("confidence")),
                "salience": _clamp_float(obs.get("salience")),
                "evidence_refs": list(obs.get("evidence_refs") or _default_evidence_refs({}, window)),
            }
        )
    return synthesized


def _derive_scene_summary(data: dict[str, Any], window: VisionWindowPayload | None) -> str:
    existing = data.get("scene_summary")
    if existing:
        return str(existing)
    if window:
        summary = window.summary or {}
        captions = summary.get("captions") or []
        if captions:
            return str(captions[0])
    return "Visual scene interpreted from window summary"


def _is_flat_legacy_event_dict(data: dict[str, Any]) -> bool:
    """Detect a lone event-shaped dict that is not V2 scene interpretation."""
    if "event_candidates" in data or "events" in data:
        return False
    v2_markers = {
        "scene_summary",
        "salient_observations",
        "uncertainties",
        "entities",
        "relations",
        "task_relevance",
        "memory_delta_candidates",
        "grammar_projection",
        "scene_state",
    }
    if v2_markers.intersection(data.keys()):
        return False
    return bool(data.get("event_type") or data.get("narrative") or data.get("type"))


def _legacy_outcome_from_data(
    data: Any,
    content: str,
    window: VisionWindowPayload,
    raw_model_output: dict[str, Any] | None,
) -> InterpretationParseOutcome | None:
    legacy = try_legacy_fallback(content, window)
    if legacy is None:
        return None
    if raw_model_output is not None:
        legacy = legacy.model_copy(update={"raw_model_output": raw_model_output})
    mode: ParseMode = "legacy_events_list" if isinstance(data, list) else "legacy_events_dict"
    return InterpretationParseOutcome(interpretation=legacy, parse_mode=mode)


def salvage_interpretation_dict(
    raw: Any,
    *,
    window: VisionWindowPayload | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize V2-ish model output into a dict suitable for VisionSceneInterpretationV1."""
    if not isinstance(raw, dict):
        return None, []

    warnings: list[str] = []
    data = _coerce_legacy_events_field(dict(raw))

    preserved_keys = (
        "schema_version",
        "scene_summary",
        "scene_state",
        "entities",
        "relations",
        "task_relevance",
        "memory_delta_candidates",
        "grammar_projection",
        "raw_model_output",
        "window_id",
        "stream_id",
        "camera_id",
        "evidence_refs",
    )
    salvaged: dict[str, Any] = {k: data[k] for k in preserved_keys if k in data}

    if window:
        salvaged.setdefault("window_id", window.window_id)
        salvaged.setdefault("stream_id", window.stream_id)
        salvaged.setdefault("camera_id", window.camera_id)
        if not salvaged.get("evidence_refs") and window.artifact_ids:
            salvaged["evidence_refs"] = list(window.artifact_ids)

    salvaged["scene_summary"] = _derive_scene_summary(data, window)

    raw_salient = data.get("salient_observations")
    coerced_salient: list[dict[str, Any]] = []
    if isinstance(raw_salient, list):
        for idx, item in enumerate(raw_salient):
            coerced = _coerce_salient_observation_item(item, idx, window, warnings)
            if coerced is not None:
                coerced_salient.append(coerced)
    elif raw_salient is not None:
        warnings.append("ignored salient_observations: expected list")
    salvaged["salient_observations"] = coerced_salient

    raw_uncertainties = data.get("uncertainties")
    coerced_uncertainties: list[dict[str, Any]] = []
    if isinstance(raw_uncertainties, list):
        for idx, item in enumerate(raw_uncertainties):
            coerced = _coerce_uncertainty_item(item, idx, warnings)
            if coerced is not None:
                coerced_uncertainties.append(coerced)
    elif raw_uncertainties is not None:
        warnings.append("ignored uncertainties: expected list")
    salvaged["uncertainties"] = coerced_uncertainties

    raw_events = data.get("event_candidates")
    coerced_events: list[dict[str, Any]] = []
    if isinstance(raw_events, list):
        for idx, item in enumerate(raw_events):
            coerced = _coerce_event_candidate_item(item, idx, window, warnings)
            if coerced is not None:
                coerced_events.append(coerced)

    if not coerced_events and coerced_salient:
        coerced_events = _synthesize_event_candidates_from_observations(
            coerced_salient, window, warnings
        )
    if not coerced_events and (
        data.get("event_type") or data.get("narrative") or data.get("type")
    ):
        top_level = {
            "event_type": data.get("event_type") or data.get("type"),
            "narrative": data.get("narrative") or data.get("observation") or data.get("summary"),
            "entities": data.get("entities"),
            "tags": data.get("tags"),
            "confidence": data.get("confidence"),
            "salience": data.get("salience"),
            "evidence_refs": data.get("evidence_refs"),
        }
        coerced = _coerce_event_candidate_item(top_level, 0, window, warnings)
        if coerced is not None:
            warnings.append("promoted top-level event fields -> event_candidates[0]")
            coerced_events = [coerced]
    salvaged["event_candidates"] = coerced_events

    if not salvaged.get("window_id"):
        return None, warnings

    return salvaged, warnings


def _dict_to_event_candidate(evt: dict[str, Any]) -> VisionEventCandidateV1:
    return VisionEventCandidateV1(
        event_type=str(evt.get("event_type", "unknown")),
        narrative=str(evt.get("narrative", "")),
        entities=list(evt.get("entities") or []),
        tags=list(evt.get("tags") or []),
        confidence=float(evt.get("confidence", 0.5)),
        salience=float(evt.get("salience", 0.5)),
        evidence_refs=list(evt.get("evidence_refs") or []),
    )


def _events_list_to_minimal_interpretation(
    events: list[dict[str, Any]],
    window: VisionWindowPayload,
) -> VisionSceneInterpretationV1:
    scene_summary = "Events from legacy format"
    if events:
        first_narrative = events[0].get("narrative") if isinstance(events[0], dict) else None
        if first_narrative:
            scene_summary = str(first_narrative)

    return VisionSceneInterpretationV1(
        window_id=window.window_id,
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        scene_summary=scene_summary,
        event_candidates=[_dict_to_event_candidate(evt) for evt in events if isinstance(evt, dict)],
        evidence_refs=list(window.artifact_ids or []),
    )


def _coerce_legacy_events_field(data: dict[str, Any]) -> dict[str, Any]:
    """Map legacy ``events`` arrays into ``event_candidates`` when the latter is absent."""
    events = data.get("events")
    if not isinstance(events, list) or data.get("event_candidates"):
        return data
    coerced = {k: v for k, v in data.items() if k != "events"}
    coerced["event_candidates"] = events
    if not coerced.get("scene_summary"):
        first_narrative = events[0].get("narrative") if events and isinstance(events[0], dict) else None
        coerced["scene_summary"] = str(first_narrative) if first_narrative else "Events from legacy format"
    return coerced


def _attach_raw_model_output(
    interpretation: VisionSceneInterpretationV1,
    raw_model_output: dict[str, Any] | None,
    data: dict[str, Any],
) -> VisionSceneInterpretationV1:
    if interpretation.raw_model_output is None and raw_model_output is not None:
        return interpretation.model_copy(update={"raw_model_output": raw_model_output})
    if interpretation.raw_model_output is None:
        return interpretation.model_copy(update={"raw_model_output": data})
    return interpretation


def _log_parse_outcome(
    outcome: InterpretationParseOutcome,
    window: VisionWindowPayload,
) -> None:
    if outcome.interpretation is None:
        return
    event_count = len(outcome.interpretation.event_candidates)
    logger.info(
        f"[COUNCIL] interpretation_parse mode={outcome.parse_mode} "
        f"window_id={window.window_id} events={event_count} "
        f"warnings={len(outcome.salvage_warnings)}"
    )


def parse_llm_content(content: str, window: VisionWindowPayload) -> InterpretationParseOutcome:
    try:
        text = _strip_markdown_fences(content)
        data = json.loads(text)
        raw_model_output: dict[str, Any] | None = None

        if isinstance(data, dict) and "interpretation" in data:
            raw_model_output = data
            data = data["interpretation"]

        if isinstance(data, list):
            interpretation = _events_list_to_minimal_interpretation(
                [evt for evt in data if isinstance(evt, dict)],
                window,
            )
            raw = raw_model_output if isinstance(raw_model_output, dict) else {"events": data}
            interpretation = interpretation.model_copy(update={"raw_model_output": raw})
            outcome = InterpretationParseOutcome(
                interpretation=interpretation,
                parse_mode="legacy_events_list",
            )
            _log_parse_outcome(outcome, window)
            return outcome

        if isinstance(data, dict):
            if set(data.keys()) == {"events"} and isinstance(data.get("events"), list):
                interpretation = _events_list_to_minimal_interpretation(data["events"], window)
                interpretation = interpretation.model_copy(update={"raw_model_output": data})
                outcome = InterpretationParseOutcome(
                    interpretation=interpretation,
                    parse_mode="legacy_events_dict",
                )
                _log_parse_outcome(outcome, window)
                return outcome

            if _is_flat_legacy_event_dict(data):
                outcome = _legacy_outcome_from_data(data, content, window, raw_model_output)
                if outcome is not None:
                    _log_parse_outcome(outcome, window)
                    return outcome

            strict_data = _coerce_legacy_events_field(dict(data))
            if not strict_data.get("window_id"):
                strict_data["window_id"] = window.window_id

            try:
                interpretation = VisionSceneInterpretationV1.model_validate(strict_data)
                interpretation = _attach_raw_model_output(interpretation, raw_model_output, data)
                outcome = InterpretationParseOutcome(
                    interpretation=interpretation,
                    parse_mode="strict_v2",
                )
                _log_parse_outcome(outcome, window)
                return outcome
            except ValidationError as exc:
                logger.warning(
                    f"[COUNCIL] interpretation_strict_failed window_id={window.window_id} "
                    f"error_count={len(exc.errors())}"
                )

            salvaged, warnings = salvage_interpretation_dict(data, window=window)
            if salvaged is not None:
                try:
                    interpretation = VisionSceneInterpretationV1.model_validate(salvaged)
                    interpretation = _attach_raw_model_output(
                        interpretation, raw_model_output, data
                    )
                    outcome = InterpretationParseOutcome(
                        interpretation=interpretation,
                        parse_mode="salvaged_v2",
                        salvage_warnings=warnings,
                    )
                    _log_parse_outcome(outcome, window)
                    return outcome
                except ValidationError as exc:
                    logger.warning(
                        f"[COUNCIL] interpretation_salvage_failed window_id={window.window_id} "
                        f"error_count={len(exc.errors())}"
                    )

            outcome = _legacy_outcome_from_data(data, content, window, raw_model_output)
            if outcome is not None:
                _log_parse_outcome(outcome, window)
                return outcome

        logger.error(f"[COUNCIL] Unexpected LLM JSON type: {type(data).__name__}")
        return InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed")
    except json.JSONDecodeError as exc:
        logger.error(f"[COUNCIL] Failed to parse interpretation JSON: {exc}")
        return InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed")
    except Exception as exc:
        logger.error(f"[COUNCIL] Failed to parse interpretation JSON: {exc}")
        return InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed")


def project_interpretation_to_events(
    interpretation: VisionSceneInterpretationV1,
    window: VisionWindowPayload,
) -> VisionEventPayload:
    fallback_refs = list(window.artifact_ids or [])
    bundle_items: list[VisionEventBundleItem] = []

    for candidate in interpretation.event_candidates:
        evidence_refs = list(candidate.evidence_refs) if candidate.evidence_refs else fallback_refs
        bundle_items.append(
            VisionEventBundleItem(
                event_id=str(uuid.uuid4()),
                event_type=candidate.event_type,
                narrative=candidate.narrative,
                entities=list(candidate.entities),
                tags=list(candidate.tags),
                confidence=candidate.confidence,
                salience=candidate.salience,
                evidence_refs=evidence_refs,
            )
        )

    return VisionEventPayload(events=bundle_items)


def try_legacy_fallback(
    content: str,
    window: VisionWindowPayload,
) -> VisionSceneInterpretationV1 | None:
    """Parse legacy flat event-list JSON when full interpretation parsing fails."""
    try:
        text = _strip_markdown_fences(content)
        data = json.loads(text)
        events: list[dict[str, Any]] | None = None
        if isinstance(data, list):
            events = [evt for evt in data if isinstance(evt, dict)]
        elif isinstance(data, dict) and isinstance(data.get("events"), list):
            events = [evt for evt in data["events"] if isinstance(evt, dict)]
        elif isinstance(data, dict):
            events = [data]
        if not events:
            return None
        return _events_list_to_minimal_interpretation(events, window)
    except Exception as exc:
        logger.error(f"[COUNCIL] Legacy fallback parse failed: {exc}")
        return None


def parse_and_project(
    content: str,
    window: VisionWindowPayload,
) -> tuple[VisionSceneInterpretationV1 | None, VisionEventPayload | None, InterpretationParseOutcome]:
    outcome = parse_llm_content(content, window)
    if outcome.interpretation is None:
        return None, None, outcome
    return (
        outcome.interpretation,
        project_interpretation_to_events(outcome.interpretation, window),
        outcome,
    )
