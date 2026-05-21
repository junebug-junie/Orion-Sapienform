from __future__ import annotations

from typing import Literal

from orion.core.schemas.drives import TensionEventV1
from .goals import GOAL_TEMPLATES

GoalGenerationMode = Literal["template", "evidence_rules", "llm"]


def _base_template(drive_origin: str, tensions: list[TensionEventV1]) -> str:
    base = GOAL_TEMPLATES.get(drive_origin, GOAL_TEMPLATES["continuity"])
    if tensions:
        lead = sorted(tensions, key=lambda t: (-t.magnitude, t.kind))[0]
        return f"{base} Primary tension: {lead.kind}."
    return base


def _evidence_rules_text(drive_origin: str, tensions: list[TensionEventV1], window_summary: str | None) -> str:
    text = _base_template(drive_origin, tensions)
    ws = " ".join(str(window_summary or "").split()).strip()
    if ws:
        snippet = ws[:60].rstrip()
        if len(ws) > 60:
            snippet += "…"
        ground = f" Ground on: {snippet}."
        max_prefix = max(0, 120 - len(ground))
        if len(text) > max_prefix:
            text = text[:max_prefix].rstrip()
        text = f"{text}{ground}"
    return text[:120]


def _llm_goal_text(**kwargs) -> str | None:
    return None  # Phase 1: wire to llm-gateway in follow-up step if env set


def generate_goal_statement(
    *,
    drive_origin: str,
    pressures: dict[str, float],
    tensions: list[TensionEventV1],
    window_summary: str | None,
    mode: GoalGenerationMode,
) -> str:
    del pressures
    if mode == "template":
        return _base_template(drive_origin, tensions)
    if mode == "llm":
        llm_text = _llm_goal_text(
            drive_origin=drive_origin, tensions=tensions, window_summary=window_summary
        )
        if llm_text:
            return llm_text[:120]
        mode = "evidence_rules"
    return _evidence_rules_text(drive_origin, tensions, window_summary)
