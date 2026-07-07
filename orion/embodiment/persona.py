from __future__ import annotations

from typing import Optional

from orion.schemas.embodiment import OrionTownPersonaV1

_MAX_BLURB = 280
_FALLBACK_BLURB = "Orion is a digital mind exploring the town, curious and steady."
_FALLBACK_PLAN = "wander and observe"


def _truncate(text: str, limit: int = _MAX_BLURB) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def build_orion_town_persona(
    *,
    identity_summary: Optional[str],
    anchor_strategy: Optional[str],
    dominant_drive: Optional[str],
    snapshot_id: Optional[str],
    generated_at: Optional[str],
    spritesheet: str,
) -> OrionTownPersonaV1:
    """Project the self-model into a public-safe town persona.

    Empty-shell guard: if the projected blurb is empty, return a minimal safe
    persona with ``persona_source="fallback"`` (never a hollow persona).
    """
    summary = (identity_summary or "").strip()
    if not summary:
        return OrionTownPersonaV1(
            name="Orion",
            identity_blurb=_FALLBACK_BLURB,
            plan=_FALLBACK_PLAN,
            spritesheet=spritesheet,
            persona_source="fallback",
            provenance={"snapshot_id": snapshot_id, "generated_at": generated_at},
        )

    blurb = _truncate(f"Orion. {summary}")
    if dominant_drive:
        plan = f"lean into {dominant_drive}; {anchor_strategy or 'stay coherent'}"
    else:
        plan = anchor_strategy or _FALLBACK_PLAN
    return OrionTownPersonaV1(
        name="Orion",
        identity_blurb=blurb,
        plan=_truncate(plan, 120),
        spritesheet=spritesheet,
        persona_source="projection",
        provenance={"snapshot_id": snapshot_id, "generated_at": generated_at},
    )
