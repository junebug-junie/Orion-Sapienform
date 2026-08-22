from __future__ import annotations

from enum import Enum

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

AUTO_ACTIVE_KINDS = frozenset({"semantic", "episode", "open_loop", "procedure"})
GATED_KINDS = frozenset({"stance", "decision", "contradiction", "attractor", "failure_mode"})
IDENTITY_SCOPE_PREFIX = "identity:"

# External worlds whose conversations Orion still stores as raw chat rows
# (chat_history_log, via orion-sql-writer -- untouched by this module) but never
# forms into a crystallization or projects into any graph/vector store. Read off
# provenance["source_platform"], which intake_consolidation_window sets only when
# every turn in the window agrees.
#
# Why this exists: on 2026-08-14 the live governor queue held 621 proposals, 610 of
# them ai-town NPC dialogue -- 98.2% noise against 11 real conversations. Every
# existing score was useless for separating them (salience is pinned at exactly 1.0
# for this whole path by construction: see salience.py's KIND_BASE["stance"]=0.85 +
# 0.075 evidence + 0.05 planning + confidence boost, which clamps to 1.0 for both
# "likely" and "certain"), so source is the only honest discriminator available.
#
# 2026-08-16 (Juniper, direct): "I do want it stored in whatever sql table, I just
# don't want it on the graphs and crystalizations and such." That superseded the
# first cut of this gate, which routed allowlisted platforms to AUTO_ACTIVATE (a
# stance still became a full active memory, still projected -- just skipped
# governor review). Renamed to DISCARD and widened from "stance only"
# (EXTERNAL_PLATFORM_BYPASSABLE_KINDS, now retired -- kill means kill, no partial
# exclusion left ticking) to every kind: an allowlisted platform now never gets a
# memory_crystallizations row at all, of any kind.
DEFAULT_DISCARD_PLATFORMS = frozenset({"aitown"})


class FormationPolicy(str, Enum):
    AUTO_ACTIVATE = "auto_activate"
    GOVERNOR_QUEUE = "governor_queue"
    REINFORCE_EXISTING = "reinforce_existing"
    DISCARD = "discard"


def _has_identity_scope(crystallization: MemoryCrystallizationV1, *, prefix: str = IDENTITY_SCOPE_PREFIX) -> bool:
    return any(str(s).startswith(prefix) for s in crystallization.scope)


def source_platform_of(crystallization: MemoryCrystallizationV1) -> str | None:
    """provenance["source_platform"], or None for a direct Juniper conversation
    (or any producer that predates the field). Public: also read by
    intake_pipeline.process_consolidation_crystallization() before duplicate
    detection even runs, so the two share one extraction, not two copies that
    could drift.
    """
    provenance = crystallization.provenance if isinstance(crystallization.provenance, dict) else {}
    platform = provenance.get("source_platform")
    return str(platform) if platform else None


def resolve_formation_policy(
    crystallization: MemoryCrystallizationV1,
    *,
    duplicate_id: str | None = None,
    identity_scope_prefix: str = IDENTITY_SCOPE_PREFIX,
    discard_platforms: frozenset[str] = DEFAULT_DISCARD_PLATFORMS,
) -> tuple[FormationPolicy, list[str]]:
    reasons: list[str] = []
    # Checked first, ahead of duplicate/intimate/identity: DISCARD is strictly
    # more restrictive than every other outcome (nothing gets persisted at all),
    # so there is no privacy regression in short-circuiting here. The opposite
    # order would have let an ai-town window that Jaccard-matches a REAL
    # crystallization reinforce it with NPC-dialogue evidence via
    # REINFORCE_EXISTING -- worse than a queued proposal, since reinforcement
    # writes straight into an already-active memory with no review step at all.
    platform = source_platform_of(crystallization)
    if platform is not None and platform in discard_platforms:
        return FormationPolicy.DISCARD, [f"discard_platform:{platform}"]
    if duplicate_id:
        return FormationPolicy.REINFORCE_EXISTING, [f"duplicate:{duplicate_id}"]
    if crystallization.governance.sensitivity == "intimate":
        return FormationPolicy.GOVERNOR_QUEUE, ["intimate_sensitivity"]
    if _has_identity_scope(crystallization, prefix=identity_scope_prefix):
        return FormationPolicy.GOVERNOR_QUEUE, ["identity_scope"]
    if crystallization.kind in GATED_KINDS:
        return FormationPolicy.GOVERNOR_QUEUE, [f"gated_kind:{crystallization.kind}"]
    if crystallization.kind in AUTO_ACTIVE_KINDS:
        return FormationPolicy.AUTO_ACTIVATE, reasons
    return FormationPolicy.GOVERNOR_QUEUE, [f"unknown_kind:{crystallization.kind}"]
