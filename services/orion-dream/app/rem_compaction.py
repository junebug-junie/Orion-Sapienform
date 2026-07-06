"""Phase F — REM compaction narration (staged, applies nothing).

REM sleep reads the awake reverie's Phase-E compaction *requests* and narrates
what sleep *would* do to memory as a `MemoryCompactionDeltaV1`. The delta is a
proposal: `proposal_marked` is hard-`True`, it is published on
`orion:dream:compaction-delta` and persisted to the `dream_compaction_delta`
staging table, and **nothing applies it**. The applier is Phase G, behind its own
hard gate (`ORION_DREAM_COMPACTION_APPLY_ENABLED`).

Discipline (§0A / hard constraints):
  - default-off (`ORION_DREAM_REM_ENABLED=false`) — a no-op unless enabled;
  - deterministic (§4): code owns the delta's *ops and metrics*; the narrator LLM
    (optional, injected) writes only the gist-card *text*. Absent a narrator, the
    gist card degrades to a deterministic summary — never a fabricated win;
  - degrades to None on any failure — a REM pass never raises;
  - staged: only downscale/prune the applier would ever run; Phase F emits
    consolidate previews (and passes through any request-hinted ops) but performs
    zero canonical writes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import uuid4

from orion.schemas.compaction import (
    MAX_CONSOLIDATE,
    MAX_OP_REFS,
    CompactionMetricsV1,
    ConsolidateEntryV1,
    MemoryCompactionDeltaV1,
)

from app.settings import settings

logger = logging.getLogger("orion-dream.rem_compaction")

# A narrator turns (theme, evidence_refs) into a short gist-card string. Injected;
# absent, `_default_gist` produces a deterministic (non-LLM) card.
Narrator = Callable[[str, list[str]], str]
RequestLoader = Callable[[int], list[dict]]


def _default_gist(theme: str, evidence_refs: list[str]) -> str:
    """Deterministic gist card — same request → same card, no LLM.

    Honest and boring on purpose: it names the theme and how many episodes fold
    in. A real narrator can be injected to make this prose; this floor guarantees
    we never emit an empty gist masquerading as cognition.
    """
    n = len(evidence_refs)
    return f"Consolidate theme {theme!r} from {n} settled episode(s)."


def _consolidate_from_request(req: dict, narrator: Narrator) -> ConsolidateEntryV1 | None:
    """Build one consolidate entry from a Phase-E request dict. None if unusable."""
    theme = str(req.get("theme") or "").strip()
    if not theme:
        return None
    evidence = req.get("evidence_refs") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(x) for x in evidence][:MAX_OP_REFS]
    try:
        gist = str(narrator(theme, evidence)).strip() or _default_gist(theme, evidence)
    except Exception:
        # A narrator failure must not sink the pass — fall back to the floor card.
        gist = _default_gist(theme, evidence)
    # `supersedes` stays empty in staged mode: nothing is superseded until the
    # gated applier actually writes a card. We only *evidence* what would fold in.
    return ConsolidateEntryV1(gist_card=gist, evidence_refs=evidence, supersedes=[])


def build_compaction_delta(
    requests: list[dict],
    *,
    dream_id: str | None = None,
    narrator: Narrator | None = None,
) -> MemoryCompactionDeltaV1:
    """Deterministically assemble the staged delta from Phase-E requests.

    Only `op_hint == "consolidate"` requests become entries in Phase F — downscale
    and prune are the applier's dangerous ops, never fabricated from the awake
    path here. Metrics are computed from the ops (code owns them, not the LLM).
    The returned delta is always `proposal_marked=True` and applies nothing.
    """
    narr = narrator or _default_gist
    consolidate: list[ConsolidateEntryV1] = []
    source_ids: list[str] = []
    for req in requests:
        if str(req.get("op_hint") or "consolidate") != "consolidate":
            continue
        entry = _consolidate_from_request(req, narr)
        if entry is None:
            continue
        consolidate.append(entry)
        rid = str(req.get("request_id") or "").strip()
        if rid:
            source_ids.append(rid)
        if len(consolidate) >= MAX_CONSOLIDATE:
            break

    return MemoryCompactionDeltaV1(
        delta_id=f"compaction-delta:{uuid4()}",
        dream_id=dream_id,
        consolidate=consolidate,
        downscale=[],  # staged: the applier's job, never the narrator's
        prune=[],
        metrics=CompactionMetricsV1(
            cards_out=len(consolidate),
            edges_downscaled=0,
            rows_pruned=0,
            bytes_reclaimed_est=0,
        ),
        source_request_ids=source_ids[:MAX_OP_REFS],
    )


async def run_rem_compaction_once(
    bus: Any,
    *,
    request_loader: RequestLoader | None = None,
    delta_persister: Callable[[MemoryCompactionDeltaV1], bool] | None = None,
    narrator: Narrator | None = None,
    dream_id: str | None = None,
) -> MemoryCompactionDeltaV1 | None:
    """One REM compaction pass. Returns the staged delta, or None. Never raises.

    Returns None (emits nothing) when REM is disabled or there is nothing settled
    to compact tonight — an honest empty night, not an empty-shell "success".
    """
    if not settings.ORION_DREAM_REM_ENABLED:
        logger.info("rem compaction disabled; pass skipped")
        return None

    loader = request_loader
    if loader is None:
        from app.rem_store import load_pending_requests as loader  # type: ignore[assignment]

    try:
        requests = list(loader(settings.DREAM_REM_MAX_REQUESTS))
    except Exception as exc:
        logger.warning("rem compaction request load failed: %s", exc)
        return None

    delta = build_compaction_delta(requests, dream_id=dream_id, narrator=narrator)
    if delta.is_empty():
        logger.info("rem compaction: nothing settled to compact tonight")
        return None

    try:
        from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

        envelope = BaseEnvelope(
            kind="dream.compaction.delta.v1",
            source=ServiceRef(
                name=settings.SERVICE_NAME,
                node=settings.NODE_NAME,
                version=settings.SERVICE_VERSION,
            ),
            payload=delta.model_dump(mode="json"),
        )
        if bus is not None:
            await bus.publish(settings.CHANNEL_DREAM_COMPACTION_DELTA, envelope)
    except Exception as exc:
        logger.warning("rem compaction publish failed id=%s err=%s", delta.delta_id, exc)
        # keep going — persistence still lets the hub preview it

    persist = delta_persister
    if persist is None:
        from app.rem_store import persist_compaction_delta as persist  # type: ignore[assignment]
    try:
        persist(delta)
    except Exception as exc:
        logger.warning("rem compaction persist failed id=%s err=%s", delta.delta_id, exc)

    logger.info(
        "rem compaction staged delta id=%s cards=%d (proposal only, applied nothing)",
        delta.delta_id,
        delta.metrics.cards_out,
    )
    return delta
