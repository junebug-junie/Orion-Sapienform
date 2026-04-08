"""Runtime pack merging for agent-chain and orch.

Ensures delivery_pack is present when output mode or user text indicates
delivery-oriented work, so live tool resolution includes delivery verbs.
"""

from __future__ import annotations

from typing import Sequence

# Output modes that should expose delivery verbs in the planner toolset
DELIVERY_ORIENTED_OUTPUT_MODES: frozenset[str] = frozenset(
    {
        "implementation_guide",
        "tutorial",
        "code_delivery",
        "debug_diagnosis",
        "direct_answer",
        "comparative_analysis",
        "decision_support",
        "reflective_depth",
        "project_planning",
    }
)


def ensure_delivery_pack_in_packs(
    packs: Sequence[str] | None,
    *,
    output_mode: str | None,
    user_text: str = "",
) -> list[str]:
    """
    Return a new pack list with delivery_pack appended when appropriate.

    If output_mode is missing, callers should classify user_text first (orch does this).
    Agent-chain also classifies when output_mode is absent.
    """
    base = list(packs) if packs else ["executive_pack", "memory_pack"]
    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in base:
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    need_delivery = bool(output_mode and output_mode in DELIVERY_ORIENTED_OUTPUT_MODES)
    if not need_delivery and user_text:
        from orion.cognition.output_mode_classifier import classify_output_mode

        omd = classify_output_mode(user_text)
        if omd.output_mode in DELIVERY_ORIENTED_OUTPUT_MODES:
            need_delivery = True

    if need_delivery and "delivery_pack" not in seen:
        try:
            from pathlib import Path
            import orion
            from orion.cognition.packs_loader import PackManager

            pm = PackManager(Path(orion.__file__).resolve().parent / "cognition")
            pm.load_packs()
            if "delivery_pack" in pm.list_packs():
                out.append("delivery_pack")
        except Exception:
            pass
    return out
