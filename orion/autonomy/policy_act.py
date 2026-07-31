from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

from orion.autonomy.action_outcomes import append_action_outcome
from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, SurpriseSource, execute_readonly_fetch, resolve_surprise
from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend
from orion.autonomy.goal_state import get_active_goal
from orion.autonomy.models import ActionOutcomeRefV1, CapabilityDecisionV1, SubstrateActResultV1, SubstrateEpisodeIntentV1
from orion.autonomy.salience import gap_terms_from_signals, iter_gap_section_labels
from orion.cognition.recall_query import DEFAULT_RECALL_REPLY_PREFIX, build_recall_query_v1
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.contracts.recall import RecallReplyV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.field_goal import FieldGoalProvenanceV1

logger = logging.getLogger(__name__)


def _resolve_domain_surprise(surprise_source: SurpriseSource | None) -> float | None:
    """Fetch the real ambient bus_synaptic value for CapabilityEvaluationContext.
    domain_surprise_score -- reuses the same `surprise_source` callable already threaded
    through these functions for ActionOutcomeRefV1.surprise (one real read, two consumers).
    Unlike `resolve_surprise()`, there is no success/fail proxy to fall back to here --
    `None` means "not available," full stop, matching `_domain_surprise_gate()`'s own
    honest-absence handling in capability_policy.py.
    """
    if surprise_source is None:
        return None
    try:
        return surprise_source()
    except Exception:
        return None

_READONLY_CAPABILITY = "web.fetch.readonly"
_EPISODE_JOURNAL_CAPABILITY = "journal.compose.episode"
_RECALL_CAPABILITY = "recall.query.readonly"
_RECALL_REQUEST_CHANNEL = "orion:exec:request:RecallService"
_GAP_SIGNAL = "world_coverage_gap"


def curiosity_strength_from_signals(signals: Sequence[FrontierInvocationSignalV1]) -> float:
    strengths = [float(sig.signal_strength or 0.0) for sig in signals if sig.signal_type == _GAP_SIGNAL]
    return max(strengths) if strengths else 0.0


def signal_kinds_from_curiosity(signals: Sequence[FrontierInvocationSignalV1]) -> list[str]:
    return sorted({str(sig.signal_type) for sig in signals if str(sig.signal_type or "").strip()})


def build_readonly_fetch_query(signals: Sequence[FrontierInvocationSignalV1]) -> str:
    label = next(iter_gap_section_labels(signals), None)
    if label:
        return f"{label} recent news coverage"
    return "world coverage gap research"


async def maybe_execute_readonly_fetch_after_goal(
    *,
    goal: FieldGoalProvenanceV1 | None,
    subject: str,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
    budget_used: dict[str, int] | None = None,
    surprise_source: SurpriseSource | None = None,
) -> tuple[CapabilityDecisionV1, ActionOutcomeRefV1 | None]:
    """Layer C gate + Tier B readonly fetch for substrate-fed motivation bus tick.

    ``goal`` is the real, field-native active goal from
    ``orion.autonomy.goal_state.get_active_goal()`` (SSP §6 Objective 6, 2026-07-30) --
    None means honestly "no real goal currently dominant," which
    ``evaluate_capability()`` denies via ``missing_goal`` for any capability that
    ``requires_goal_status``. ``subject`` is passed separately (not read off
    ``goal.subject``): a real ``FieldGoalProvenanceV1``'s ``subject`` is always
    ``"attention"`` (its producer, per ``orion-attention-runtime``'s worker), not the
    episode's acting subject -- using it here would silently mislabel every fetch
    request.
    """
    if not curiosity_signals or _GAP_SIGNAL not in signal_kinds_from_curiosity(curiosity_signals):
        decision = CapabilityDecisionV1(
            capability_id=_READONLY_CAPABILITY,
            outcome="denied",
            reason_code="missing_signal_kinds",
            auto_execute=False,
        )
        return decision, None

    if not spawned_correlation_id:
        decision = CapabilityDecisionV1(
            capability_id=_READONLY_CAPABILITY,
            outcome="denied",
            reason_code="missing_spawned_correlation_id",
            auto_execute=False,
        )
        return decision, None

    _domain_surprise = _resolve_domain_surprise(surprise_source)
    ctx = CapabilityEvaluationContext(
        curiosity_strength=curiosity_strength_from_signals(curiosity_signals),
        signal_kinds=signal_kinds_from_curiosity(curiosity_signals),
        goal=goal,
        budget_used=budget_used or {},
        domain_surprise_score=_domain_surprise,
        domain_surprise_source="bus_synaptic" if _domain_surprise is not None else None,
    )
    decision = evaluate_capability(_READONLY_CAPABILITY, ctx)
    logger.info(
        "substrate_policy_act capability=%s outcome=%s reason=%s auto_execute=%s goal=%s spawned=%s",
        decision.capability_id,
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        goal.artifact_id if goal is not None else None,
        spawned_correlation_id,
    )
    if decision.outcome != "allowed" or not decision.auto_execute:
        return decision, None

    # requires_goal_status > 0 on this capability's rule is what makes goal real
    # here (evaluate_capability() denies via missing_goal above if it were None) --
    # that link lives in config/autonomy/capability_policy.v1.yaml, not in code, so
    # a defensive check here (not a bare assert) is the honest guard against that
    # config ever changing out from under this invariant.
    if goal is None:
        return decision, None
    query = build_readonly_fetch_query(curiosity_signals)
    gap_terms = gap_terms_from_signals(curiosity_signals, fallback_query=query)
    req = EpisodeFetchRequest(
        subject=subject,
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        query=query,
        gap_terms=tuple(sorted(gap_terms)),
    )
    if fetch_backend is None:
        fetch_backend = resolve_fetch_backend()
    # Reuse the value already resolved above for the gate decision -- one real read, two
    # consumers, not two reads that could observe the signal at slightly different
    # moments (found in review: passing the raw `surprise_source` through here would
    # call it again).
    outcome = await execute_readonly_fetch(
        req, fetch_backend=fetch_backend, surprise_source=lambda: _domain_surprise
    )
    if budget_used is not None:
        budget_used[_READONLY_CAPABILITY] = budget_used.get(_READONLY_CAPABILITY, 0) + 1
    return decision, outcome


async def _execute_readonly_recall(
    *,
    subject: str,
    query: str,
    bus: Any,
    source: ServiceRef | None,
    recall_channel: str,
    timeout_sec: float,
    spawned_correlation_id: str,
    surprise_source: SurpriseSource | None = None,
) -> ActionOutcomeRefV1:
    """Inline recall RPC (mirrors ``recall_prefetch.py:169-183``'s exact pattern).

    Degrades gracefully on any failure -- never raises. The caller falls through
    to the readonly-fetch path when the returned outcome is not a success.
    """
    action_id = f"recall-{spawned_correlation_id}-{uuid.uuid4().hex[:8]}"
    observed_at = datetime.now(timezone.utc)
    reply_channel = f"{DEFAULT_RECALL_REPLY_PREFIX}:{uuid.uuid4()}"
    envelope_source = source or ServiceRef(name="orion-spark-concept-induction")
    # BaseEnvelope.correlation_id requires a real UUID; autonomy run ids (e.g.
    # "wp-run-gap-gpu") are opaque strings, not UUIDs. Reuse spawned_correlation_id
    # verbatim when it happens to already be one; otherwise mint a fresh envelope
    # correlation id rather than raise.
    try:
        envelope_correlation_id = uuid.UUID(str(spawned_correlation_id))
    except (ValueError, AttributeError, TypeError):
        envelope_correlation_id = uuid.uuid4()

    try:
        req = build_recall_query_v1(
            {"raw_user_text": query, "verb": "autonomy_recall_check"},
            correlation_id=spawned_correlation_id,
            reply_to=reply_channel,
        )
        if req is None:
            outcome = ActionOutcomeRefV1(
                action_id=action_id,
                kind=_RECALL_CAPABILITY,
                summary="recall query build failed",
                success=False,
                surprise=resolve_surprise(surprise_source, success=False),
                observed_at=observed_at,
                query=query,
            )
        else:
            env = BaseEnvelope(
                kind="recall.query.v1",
                source=envelope_source,
                correlation_id=envelope_correlation_id,
                reply_to=reply_channel,
                payload=req.model_dump(mode="json"),
            )
            msg = await bus.rpc_request(
                recall_channel,
                env,
                reply_channel=reply_channel,
                timeout_sec=timeout_sec,
            )
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                outcome = ActionOutcomeRefV1(
                    action_id=action_id,
                    kind=_RECALL_CAPABILITY,
                    summary=f"recall decode failed: {decoded.error}",
                    success=False,
                    surprise=resolve_surprise(surprise_source, success=False),
                    observed_at=observed_at,
                    query=query,
                )
            else:
                payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
                if payload.get("error"):
                    outcome = ActionOutcomeRefV1(
                        action_id=action_id,
                        kind=_RECALL_CAPABILITY,
                        summary=f"recall service error: {payload.get('error')}",
                        success=False,
                        surprise=resolve_surprise(surprise_source, success=False),
                        observed_at=observed_at,
                        query=query,
                    )
                else:
                    reply = RecallReplyV1.model_validate(payload)
                    items = reply.bundle.items or []
                    found = bool(items)
                    outcome = ActionOutcomeRefV1(
                        action_id=action_id,
                        kind=_RECALL_CAPABILITY,
                        summary=f"recall found {len(items)} item(s)" if found else "recall found nothing",
                        success=found,
                        surprise=resolve_surprise(surprise_source, success=found),
                        observed_at=observed_at,
                        query=query,
                    )
    except Exception as exc:
        logger.warning(
            "autonomy_recall_rpc_failed subject=%s query=%r error=%s",
            subject,
            query,
            exc,
            exc_info=True,
        )
        outcome = ActionOutcomeRefV1(
            action_id=action_id,
            kind=_RECALL_CAPABILITY,
            summary=f"recall rpc failed: {exc}",
            success=False,
            surprise=resolve_surprise(surprise_source, success=False),
            observed_at=observed_at,
            query=query,
        )

    append_action_outcome(subject=subject, outcome=outcome)
    return outcome


async def maybe_execute_readonly_recall_after_goal(
    *,
    goal: FieldGoalProvenanceV1 | None,
    subject: str,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None,
    bus: Any = None,
    source: ServiceRef | None = None,
    recall_channel: str = _RECALL_REQUEST_CHANNEL,
    timeout_sec: float = 3.0,
    budget_used: dict[str, int] | None = None,
    surprise_source: SurpriseSource | None = None,
) -> tuple[CapabilityDecisionV1, ActionOutcomeRefV1 | None]:
    """Layer C gate + recall-first check, tried before the readonly web fetch.

    "Check what I already know first": mirrors
    ``maybe_execute_readonly_fetch_after_goal`` in shape exactly. On allow, issues
    an inline recall RPC and records the outcome via ``append_action_outcome``
    the same way the fetch path does. Never raises -- any RPC failure, timeout,
    or missing ``bus`` degrades to a ``None`` outcome so the caller falls through
    to the existing readonly-fetch path without consuming its budget.
    """
    if not curiosity_signals or _GAP_SIGNAL not in signal_kinds_from_curiosity(curiosity_signals):
        decision = CapabilityDecisionV1(
            capability_id=_RECALL_CAPABILITY,
            outcome="denied",
            reason_code="missing_signal_kinds",
            auto_execute=False,
        )
        return decision, None

    if not spawned_correlation_id:
        decision = CapabilityDecisionV1(
            capability_id=_RECALL_CAPABILITY,
            outcome="denied",
            reason_code="missing_spawned_correlation_id",
            auto_execute=False,
        )
        return decision, None

    _domain_surprise = _resolve_domain_surprise(surprise_source)
    ctx = CapabilityEvaluationContext(
        curiosity_strength=curiosity_strength_from_signals(curiosity_signals),
        signal_kinds=signal_kinds_from_curiosity(curiosity_signals),
        goal=goal,
        budget_used=budget_used or {},
        domain_surprise_score=_domain_surprise,
        domain_surprise_source="bus_synaptic" if _domain_surprise is not None else None,
    )
    decision = evaluate_capability(_RECALL_CAPABILITY, ctx)
    logger.info(
        "substrate_policy_act capability=%s outcome=%s reason=%s auto_execute=%s goal=%s spawned=%s",
        decision.capability_id,
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        goal.artifact_id if goal is not None else None,
        spawned_correlation_id,
    )
    if decision.outcome != "allowed" or not decision.auto_execute:
        return decision, None

    # requires_goal_status > 0 on this capability's rule is what makes goal real
    # here (evaluate_capability() denies via missing_goal above if it were None) --
    # that link lives in config/autonomy/capability_policy.v1.yaml, not in code, so
    # a defensive check here (not a bare assert) is the honest guard against that
    # config ever changing out from under this invariant.
    if goal is None:
        return decision, None

    if bus is None:
        # No bus wired for this call site -- degrade gracefully rather than raise;
        # the caller falls through to the fetch path with its budget untouched.
        return decision, None

    if budget_used is not None:
        budget_used[_RECALL_CAPABILITY] = budget_used.get(_RECALL_CAPABILITY, 0) + 1

    query = build_readonly_fetch_query(curiosity_signals)
    try:
        outcome = await _execute_readonly_recall(
            subject=subject,
            query=query,
            bus=bus,
            source=source,
            recall_channel=recall_channel,
            timeout_sec=timeout_sec,
            spawned_correlation_id=spawned_correlation_id,
            # Reuse the value already resolved above for the gate decision -- see the
            # matching comment in maybe_execute_readonly_fetch_after_goal.
            surprise_source=lambda: _domain_surprise,
        )
    except Exception:
        logger.warning(
            "autonomy_recall_execute_failed goal=%s spawned=%s",
            goal.artifact_id,
            spawned_correlation_id,
            exc_info=True,
        )
        return decision, None
    return decision, outcome


_MAX_SEED_DESC_CHARS = 300


def _gap_section_label(signals: Sequence[FrontierInvocationSignalV1]) -> str:
    return next(iter_gap_section_labels(signals), "")


def build_episode_narrative_seed(
    goal: FieldGoalProvenanceV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    fetch_outcome: ActionOutcomeRefV1,
) -> str:
    """Structured multi-line compose seed: why + what + salience + satiation ask.

    `goal` is part of the stable interface (spec contract + future goal_statement
    enrichment seam) even though the current body does not read it.
    """
    del goal  # part of stable interface; not read yet
    if not fetch_outcome.success:
        return f"fetch failed: {fetch_outcome.summary}"

    lines: list[str] = []
    strength = curiosity_strength_from_signals(curiosity_signals)
    section = _gap_section_label(curiosity_signals)
    if section:
        lines.append(f'Why: predictive coverage gap in "{section}" (strength {strength:.2f}).')
    else:
        lines.append(f"Why: predictive coverage gap (strength {strength:.2f}).")
    if fetch_outcome.query:
        lines.append(f'Query: "{fetch_outcome.query}"')

    articles = fetch_outcome.articles
    if articles:
        lines.append(f"Fetched {len(articles)} article(s):")
        # "scored" iff the fetch had gap terms to score against (mirrors the
        # gap_terms the fetch used). A genuine 0.0 overlap is honestly "salience
        # 0.00", not "unscored"; "unscored" means there was nothing to score by.
        scored = bool(
            gap_terms_from_signals(curiosity_signals, fallback_query=fetch_outcome.query or "")
        )
        for idx, art in enumerate(articles, start=1):
            marker = f"salience {art.salience:.2f}" if scored else "unscored"
            title = art.title or "(untitled)"
            lines.append(f"  {idx}. [{marker}] {title} — {art.url}")
            desc = (art.description or "").strip()
            if desc:
                if len(desc) > _MAX_SEED_DESC_CHARS:
                    desc = desc[:_MAX_SEED_DESC_CHARS].rstrip() + "…"
                lines.append(f"     {desc}")
    else:
        lines.append(f"fetch outcome: {fetch_outcome.summary}")

    lines.append(
        "Reflect: summarize each article and assess whether it closes the gap that "
        "drove this fetch. Name what is still missing. Do not invent sources."
    )
    return "\n".join(lines)


async def maybe_compose_autonomy_episode_after_fetch(
    *,
    goal: FieldGoalProvenanceV1 | None,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None,
    fetch_outcome: ActionOutcomeRefV1 | None,
    journal_dispatch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    budget_used: dict[str, int] | None = None,
    surprise_source: SurpriseSource | None = None,
) -> tuple[CapabilityDecisionV1, dict[str, Any] | None]:
    """Layer C gate + episode journal compose after successful readonly fetch."""
    if fetch_outcome is None:
        decision = CapabilityDecisionV1(
            capability_id=_EPISODE_JOURNAL_CAPABILITY,
            outcome="denied",
            reason_code="fetch_outcome_missing",
            auto_execute=False,
        )
        return decision, None

    if not spawned_correlation_id:
        decision = CapabilityDecisionV1(
            capability_id=_EPISODE_JOURNAL_CAPABILITY,
            outcome="denied",
            reason_code="missing_spawned_correlation_id",
            auto_execute=False,
        )
        return decision, None

    _domain_surprise = _resolve_domain_surprise(surprise_source)
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=goal,
        budget_used=budget_used or {},
        domain_surprise_score=_domain_surprise,
        domain_surprise_source="bus_synaptic" if _domain_surprise is not None else None,
    )
    decision = evaluate_capability(_EPISODE_JOURNAL_CAPABILITY, ctx)
    logger.info(
        "substrate_policy_act capability=%s outcome=%s reason=%s auto_execute=%s goal=%s spawned=%s",
        decision.capability_id,
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        goal.artifact_id if goal is not None else None,
        spawned_correlation_id,
    )
    if decision.outcome != "allowed" or not decision.auto_execute:
        return decision, None
    if journal_dispatch is None:
        return decision, None

    # requires_goal_status > 0 on this capability's rule is what makes goal real
    # here (evaluate_capability() denies via missing_goal above if it were None) --
    # that link lives in config/autonomy/capability_policy.v1.yaml, not in code, so
    # a defensive check here (not a bare assert) is the honest guard against that
    # config ever changing out from under this invariant.
    if goal is None:
        return decision, None
    narrative_seed = build_episode_narrative_seed(goal, curiosity_signals, fetch_outcome)
    result = await journal_dispatch(
        goal_artifact_id=goal.artifact_id,
        spawned_correlation_id=spawned_correlation_id,
        narrative_seed=narrative_seed,
    )
    if budget_used is not None:
        budget_used[_EPISODE_JOURNAL_CAPABILITY] = budget_used.get(_EPISODE_JOURNAL_CAPABILITY, 0) + 1
    return decision, result


def resolve_episode_intent(
    *,
    store,
    subject: str,
    run_id: str,
    drive_origin: str = "predictive",
) -> SubstrateEpisodeIntentV1:
    # `drive_origin` is now purely a store slot-key convention and a
    # historical label on the synthetic goal below -- it no longer drives any
    # capability_policy.py gating decision (that check was removed 2026-07-30,
    # chore/delete-orion-drives Wave 2a). Left as "predictive" so
    # store.load_goal_slot's existing key ("orion", "predictive") keeps
    # resolving the same historical goal-slot row it always has; renaming it
    # would just be cosmetic churn on a real storage key, not a behavior fix.
    slot = store.load_goal_slot(subject, drive_origin)
    artifact_id = slot.get("artifact_id") if isinstance(slot, dict) else None
    if isinstance(artifact_id, str) and artifact_id.strip():
        return SubstrateEpisodeIntentV1(
            goal_artifact_id=artifact_id.strip(),
            drive_origin=drive_origin,
            spawned_correlation_id=run_id,
            subject=subject,
        )
    return SubstrateEpisodeIntentV1(
        goal_artifact_id=f"episode-{run_id}",
        drive_origin="predictive",
        spawned_correlation_id=run_id,
        subject=subject,
    )


async def maybe_execute_substrate_act_after_metabolism(
    *,
    episode_intent: SubstrateEpisodeIntentV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None = None,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
    journal_dispatch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    budget_used: dict[str, int] | None = None,
    prefetched_outcome: ActionOutcomeRefV1 | None = None,
    episode_journal_enabled: bool = False,
    recall_bus: Any = None,
    recall_source: ServiceRef | None = None,
    recall_channel: str = _RECALL_REQUEST_CHANNEL,
    recall_timeout_sec: float = 3.0,
    surprise_source: SurpriseSource | None = None,
) -> SubstrateActResultV1:
    run_id = spawned_correlation_id or episode_intent.spawned_correlation_id
    # SSP §6 Objective 6 (2026-07-30): real, field-native active goal, not a
    # per-call synthetic fabrication -- see orion.autonomy.goal_state's module
    # docstring. None is an honest "no real goal currently dominant," which
    # every capability rule below already denies correctly via missing_goal.
    goal = get_active_goal()
    result = SubstrateActResultV1()

    if prefetched_outcome is not None:
        # Single shared fetch: world-pulse already fetched this gap section and put
        # the findings on the run result. Reuse them; do not call the backend again.
        fetch_outcome = prefetched_outcome
        result = result.model_copy(
            update={
                "fetch_attempted": True,
                "fetch_outcome_id": fetch_outcome.action_id,
                "fetch_outcome": fetch_outcome,
            }
        )
    else:
        # Recall-first: check what Orion already knows before spending a live
        # web fetch. Tried before the fetch call in the same tick; if recall
        # surfaces real content, the fetch capability's budget is left untouched
        # this cycle. Any recall failure/timeout/empty result degrades to None
        # and falls straight through to the existing fetch path below.
        try:
            _, recall_outcome = await maybe_execute_readonly_recall_after_goal(
                goal=goal,
                subject=episode_intent.subject,
                curiosity_signals=curiosity_signals,
                spawned_correlation_id=run_id,
                bus=recall_bus,
                source=recall_source,
                recall_channel=recall_channel,
                timeout_sec=recall_timeout_sec,
                budget_used=budget_used,
                surprise_source=surprise_source,
            )
        except Exception:
            logger.warning(
                "substrate_recall_check_failed goal=%s spawned=%s",
                goal.artifact_id if goal is not None else None,
                run_id,
                exc_info=True,
            )
            recall_outcome = None

        if recall_outcome is not None:
            # Recorded whenever an attempt happened (RPC issued, whatever the
            # result), mirroring fetch_attempted/fetch_outcome's own semantics
            # below -- "attempted" tracks that a real check happened, success
            # lives inside the outcome object itself. Without this, a recall
            # attempt was only ever visible via the local append_action_outcome
            # file-store fallback inside maybe_execute_readonly_recall_after_goal,
            # never reaching the durable bus-emit -> sql-writer -> SQL path a
            # fetch outcome does (see the publish site below this function).
            result = result.model_copy(
                update={
                    "recall_attempted": True,
                    "recall_outcome": recall_outcome,
                }
            )

        if recall_outcome is not None and recall_outcome.success:
            fetch_outcome = None
        else:
            fetch_decision, fetch_outcome = await maybe_execute_readonly_fetch_after_goal(
                goal=goal,
                subject=episode_intent.subject,
                curiosity_signals=curiosity_signals,
                spawned_correlation_id=run_id,
                fetch_backend=fetch_backend,
                budget_used=budget_used,
                surprise_source=surprise_source,
            )
            if fetch_decision.outcome == "allowed" and fetch_outcome is not None:
                result = result.model_copy(
                    update={
                        "fetch_attempted": True,
                        "fetch_outcome_id": fetch_outcome.action_id,
                        "fetch_outcome": fetch_outcome,
                    }
                )

    if not episode_journal_enabled or fetch_outcome is None:
        return result

    # The journal compose step issues an RPC (cortex-exec) that can time out. A journal
    # failure must NOT discard an already-successful fetch outcome: isolate it so the
    # caller still receives `result` (with fetch_outcome) and can persist the fetch.
    try:
        journal_decision, journal_payload = await maybe_compose_autonomy_episode_after_fetch(
            goal=goal,
            curiosity_signals=curiosity_signals,
            spawned_correlation_id=run_id,
            fetch_outcome=fetch_outcome,
            journal_dispatch=journal_dispatch,
            budget_used=budget_used,
            surprise_source=surprise_source,
        )
    except Exception:
        logger.warning(
            "substrate_episode_journal_failed goal=%s spawned=%s",
            goal.artifact_id if goal is not None else None,
            run_id,
            exc_info=True,
        )
        return result
    if journal_decision.outcome == "allowed" and journal_payload is not None:
        entry_id = None
        if isinstance(journal_payload.get("write"), dict):
            entry_id = journal_payload["write"].get("entry_id")
        result = result.model_copy(update={"journal_attempted": True, "journal_entry_id": entry_id})
    return result
