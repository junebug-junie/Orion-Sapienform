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
  - staged: Phase F emits consolidate previews and, since 2026-11-13, downscale
    proposals for the settled reverie thoughts those previews fold in
    (`downscale_entries`: salience x `DREAM_COMPACTION_DOWNSCALE_FACTOR`, once per
    thought, never a prune). This module itself performs zero canonical writes.

Consent ask and apply (attention patch 3d; apply wired 2026-11-13). After a delta is persisted, the pass
awaits an injected `consent_asker(delta)`. This module never imports the applier (Phase F stays out of
every mutation path). `POST /dreams/rem-preview` injects `ask_consent_and_apply(preview=True)` (asks,
never writes); the sleep cycle (app.main `_rem`, which persists after this pass returns) calls
`app.compaction_consent.ask_consent_and_apply` after its own persist. The ask goes to the want market's consent gate; on `allow` (Juniper's standing consent
`dream-compaction-downscale`) and `ORION_DREAM_COMPACTION_APPLY_ENABLED`, the applier snapshots and
downscales through `app.compaction_store`. `DREAM_COMPACTION_CONSENT_ASK_ENABLED=false` skips the ask
(and so the apply).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable
from uuid import uuid4

from orion.schemas.compaction import (
    MAX_CONSOLIDATE,
    MAX_DOWNSCALE,
    MAX_OP_REFS,
    CompactionMetricsV1,
    ConsolidateEntryV1,
    DownscaleEntryV1,
    MemoryCompactionDeltaV1,
)

from app.settings import settings

logger = logging.getLogger("orion-dream.rem_compaction")

# A narrator turns (theme, evidence_refs) into a short gist-card string. Injected;
# absent, `_default_gist` produces a deterministic (non-LLM) card.
Narrator = Callable[[str, list[str]], str]
RequestLoader = Callable[[int], list[dict]]
# thought ids -> {thought_id: (current salience column, original salience from thought_json or None)}.
SalienceLoader = Callable[[list[str]], dict[str, tuple[float, float | None]]]
# The applier's store addresses reverie thoughts by this prefix (app.compaction_store.TARGET_PREFIX).
THOUGHT_TARGET_PREFIX = "substrate_reverie_thought:"
_SAME = 1e-9


def consolidate_evidence_ids(requests: list[dict]) -> list[str]:
    """The evidence ids (reverie thought ids) of every request that becomes a consolidate entry, in order."""
    out: list[str] = []
    for req in requests:
        if str(req.get("op_hint") or "consolidate") != "consolidate" or not str(req.get("theme") or "").strip():
            continue
        ev = req.get("evidence_refs") or []
        if isinstance(ev, list):
            out.extend(str(x) for x in ev[:MAX_OP_REFS])
    return list(dict.fromkeys(out))


def downscale_entries(
    themed_evidence: list[tuple[str, list[str]]],
    salience: dict[str, tuple[float, float | None]],
    factor: float,
) -> list[DownscaleEntryV1]:
    """Deterministic downscale proposals: each settled thought's salience x ``factor``, once.

    Sleep downscales proportionally (synaptic homeostasis: relative order among the downscaled is kept),
    and only what was replayed and settled: the thoughts a consolidate entry folds in. A thought is
    proposed only when its salience column still equals the original in ``thought_json`` (never
    downscaled before), so a re-staged thought is not compounded. Unknown, already-lowered, zero or
    missing-original thoughts are skipped. Capped at ``MAX_DOWNSCALE``. A factor outside (0, 1) proposes
    nothing.
    """
    if not (0.0 < float(factor) < 1.0):
        return []
    out: list[DownscaleEntryV1] = []
    seen: set[str] = set()
    for theme, evidence in themed_evidence:
        for tid in evidence:
            if tid in seen or len(out) >= MAX_DOWNSCALE:
                continue
            seen.add(tid)
            cur, orig = salience.get(tid, (None, None))
            if cur is None or orig is None or abs(float(cur) - float(orig)) > _SAME:
                continue
            cur = float(cur)
            new = round(cur * float(factor), 6)
            if not (0.0 < cur <= 1.0) or not (0.0 < new < cur):  # too small to lower (or would zero): left alone
                continue
            out.append(DownscaleEntryV1(target_id=f"{THOUGHT_TARGET_PREFIX}{tid}", old_w=cur, new_w=new,
                                        reason=f"settled; folded into theme {theme!r}"[:200]))
    return out


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
    salience: dict[str, tuple[float, float | None]] | None = None,
    downscale_factor: float | None = None,
) -> MemoryCompactionDeltaV1:
    """Deterministically assemble the staged delta from Phase-E requests.

    Only `op_hint == "consolidate"` requests become entries — the awake path never
    hints downscale or prune into a delta. With ``salience`` (read by the REM pass),
    the thoughts those entries fold in get downscale proposals (`downscale_entries`).
    Prune is never proposed. Metrics are computed from the ops (code owns them, not
    the LLM). The returned delta is always `proposal_marked=True` and applies nothing.
    """
    narr = narrator or _default_gist
    consolidate: list[ConsolidateEntryV1] = []
    themed: list[tuple[str, list[str]]] = []
    source_ids: list[str] = []
    for req in requests:
        if str(req.get("op_hint") or "consolidate") != "consolidate":
            continue
        entry = _consolidate_from_request(req, narr)
        if entry is None:
            continue
        consolidate.append(entry)
        themed.append((str(req.get("theme") or "").strip(), list(entry.evidence_refs)))
        rid = str(req.get("request_id") or "").strip()
        if rid:
            source_ids.append(rid)
        if len(consolidate) >= MAX_CONSOLIDATE:
            break

    factor = settings.DREAM_COMPACTION_DOWNSCALE_FACTOR if downscale_factor is None else downscale_factor
    downscale = downscale_entries(themed, salience or {}, factor)
    return MemoryCompactionDeltaV1(
        delta_id=f"compaction-delta:{uuid4()}",
        dream_id=dream_id,
        consolidate=consolidate,
        downscale=downscale,  # deterministic, from the read salience; never the narrator's
        prune=[],  # never proposed
        metrics=CompactionMetricsV1(
            cards_out=len(consolidate),
            edges_downscaled=len(downscale),
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
    consent_asker: Callable[[MemoryCompactionDeltaV1], Awaitable[Any]] | None = None,
    salience_loader: SalienceLoader | None = None,
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

    sal_loader = salience_loader
    if sal_loader is None:
        from app.rem_store import load_thought_salience as sal_loader  # type: ignore[assignment]
    try:
        salience = dict(sal_loader(consolidate_evidence_ids(requests)))
    except Exception as exc:  # noqa: BLE001 -- no salience means no downscale proposals, not no delta
        logger.warning("rem compaction salience load failed: %s", exc)
        salience = {}

    delta = build_compaction_delta(requests, dream_id=dream_id, narrator=narrator, salience=salience)
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
    persisted = False
    try:
        persisted = bool(persist(delta))
    except Exception as exc:
        logger.warning("rem compaction persist failed id=%s err=%s", delta.delta_id, exc)

    logger.info(
        "rem compaction staged delta id=%s cards=%d downscale=%d (proposal; apply is the consent ask's)",
        delta.delta_id,
        delta.metrics.cards_out,
        delta.metrics.edges_downscaled,
    )
    # Only a delta that is really staged is asked about (the ledger row must name a stored delta).
    if consent_asker is not None and persisted and settings.DREAM_COMPACTION_CONSENT_ASK_ENABLED:
        try:
            await consent_asker(delta)
        except Exception as exc:  # noqa: BLE001 -- the ask is a trace; the staged delta stands
            logger.warning("rem compaction consent ask failed id=%s err=%s", delta.delta_id, exc)
    return delta
