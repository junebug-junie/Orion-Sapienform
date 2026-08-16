from __future__ import annotations

from enum import Enum

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

AUTO_ACTIVE_KINDS = frozenset({"semantic", "episode", "open_loop", "procedure"})
GATED_KINDS = frozenset({"stance", "decision", "contradiction", "attractor", "failure_mode"})
IDENTITY_SCOPE_PREFIX = "identity:"

# External worlds whose conversations Orion still remembers but never asks Juniper
# to review turn-by-turn. Read off provenance["source_platform"], which
# intake_consolidation_window sets only when every turn in the window agrees.
#
# Why this exists: on 2026-08-14 the live governor queue held 621 proposals, 610 of
# them ai-town NPC dialogue -- 98.2% noise against 11 real conversations. Every
# existing score was useless for separating them (salience is pinned at exactly 1.0
# for this whole path by construction: see salience.py's KIND_BASE["stance"]=0.85 +
# 0.075 evidence + 0.05 planning + confidence boost, which clamps to 1.0 for both
# "likely" and "certain"), so source is the only honest discriminator available.
DEFAULT_AUTO_ACTIVATE_PLATFORMS = frozenset({"aitown"})

# Which GATED_KINDS an allowlisted platform may bypass. Deliberately only
# "stance": that is the sole gated kind this producer can currently emit
# (_KIND_FOR_SHIFT maps STANCE->stance, REPAIR->open_loop, TOPIC->semantic, and
# the latter two are already in AUTO_ACTIVE_KINDS), so a broader bypass buys
# nothing today while creating a real latent hazard -- the moment a new shift
# mapping or another producer sets provenance["source_platform"], fictional NPC
# roleplay could write a "contradiction" or "decision" straight to active and
# project it into cards/chroma with nobody ever seeing it. Widening this should
# be a deliberate per-kind decision, not a side effect of adding a mapping.
EXTERNAL_PLATFORM_BYPASSABLE_KINDS = frozenset({"stance"})


class FormationPolicy(str, Enum):
    AUTO_ACTIVATE = "auto_activate"
    GOVERNOR_QUEUE = "governor_queue"
    REINFORCE_EXISTING = "reinforce_existing"


def _has_identity_scope(crystallization: MemoryCrystallizationV1, *, prefix: str = IDENTITY_SCOPE_PREFIX) -> bool:
    return any(str(s).startswith(prefix) for s in crystallization.scope)


def _source_platform(crystallization: MemoryCrystallizationV1) -> str | None:
    provenance = crystallization.provenance if isinstance(crystallization.provenance, dict) else {}
    platform = provenance.get("source_platform")
    return str(platform) if platform else None


def resolve_formation_policy(
    crystallization: MemoryCrystallizationV1,
    *,
    duplicate_id: str | None = None,
    identity_scope_prefix: str = IDENTITY_SCOPE_PREFIX,
    auto_activate_platforms: frozenset[str] = DEFAULT_AUTO_ACTIVATE_PLATFORMS,
) -> tuple[FormationPolicy, list[str]]:
    reasons: list[str] = []
    if duplicate_id:
        return FormationPolicy.REINFORCE_EXISTING, [f"duplicate:{duplicate_id}"]
    if crystallization.governance.sensitivity == "intimate":
        return FormationPolicy.GOVERNOR_QUEUE, ["intimate_sensitivity"]
    if _has_identity_scope(crystallization, prefix=identity_scope_prefix):
        return FormationPolicy.GOVERNOR_QUEUE, ["identity_scope"]
    # Ordered below the intimate/identity checks so those outrank a source-based
    # convenience gate for any producer that can set them -- but be honest about
    # today's reality: build_crystallization_from_window() hardcodes
    # sensitivity="private" and scope=["memory_window:<id>"], so NEITHER guard
    # above can fire for the only producer that currently sets source_platform.
    # They are defense-in-depth for a future producer, not a live rail, and the
    # tests covering them have to hand-mutate the object to reach them.
    #
    # Ordered above the GATED_KINDS check, since bypassing the stance gate for
    # external worlds is the entire point -- but scoped to
    # EXTERNAL_PLATFORM_BYPASSABLE_KINDS rather than all of GATED_KINDS.
    platform = _source_platform(crystallization)
    if (
        platform is not None
        and platform in auto_activate_platforms
        and crystallization.kind in EXTERNAL_PLATFORM_BYPASSABLE_KINDS
    ):
        return FormationPolicy.AUTO_ACTIVATE, [f"external_platform:{platform}"]
    if crystallization.kind in GATED_KINDS:
        return FormationPolicy.GOVERNOR_QUEUE, [f"gated_kind:{crystallization.kind}"]
    if crystallization.kind in AUTO_ACTIVE_KINDS:
        return FormationPolicy.AUTO_ACTIVATE, reasons
    return FormationPolicy.GOVERNOR_QUEUE, [f"unknown_kind:{crystallization.kind}"]
