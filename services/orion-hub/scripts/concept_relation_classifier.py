"""Real LLM classifier for typed concept-relation edges (Phase 4 wiring).

Fills the injection seam left open by
``orion.substrate.relation_classification.classify_relation()`` (the
``RelationClassifier`` callable): given two co-occurring ``ConceptNodeV1``
nodes and the ``SubstrateEdgeV1`` connecting them, ask an LLM whether one
``supports``/``contradicts``/``refines`` the other, or ``None`` if the
relationship is not confidently one of those three.

Mechanism chosen: a direct, bounded, structured-output bus RPC to the LLM
gateway (``settings.CHANNEL_LLM_INTAKE``) -- the same shape already proven by
``orion/memory/crystallization/concept_relation.py::resolve_concept_relation()``
(same-shaped judgment: two candidates in, one of a small set of typed
relations or "none" out, never raises) and by this service's own existing
bus-RPC client convention (``pre_turn_appraisal_client.py``,
``thought_client.py``: a small class/function wrapping
``OrionBusAsync.rpc_request()`` directly against a bus channel). This is
deliberately NOT ``CortexGatewayClient.chat()``
(``services/orion-hub/scripts/bus_clients/cortex_client.py``): that sends a
``CortexChatRequest`` through the full cortex-orch cognition pipeline
(self-state, drives, memory recall, session context) -- the wrong weight
class for a bounded per-pair classification call, and reusing it here would
mean constructing a full chat request just to get a one-word structured
answer. The direct-RPC path requires no new plumbing: Hub already imports
``OrionBusAsync``/``bus_schemas`` and calls ``bus.rpc_request()`` directly in
multiple existing files.

``RelationClassifier`` is a plain **synchronous** 3-arg callable --
``classify_relation()`` calls it directly, not awaited (see
``orion/substrate/relation_classification.py``). The actual LLM judgment is
an async bus RPC, so ``build_llm_relation_classifier()`` below runs one
bounded async batch up front (a single ``asyncio.run()`` call, one shared bus
connection, sequential RPCs across the caller's already-capped pair list) and
returns a synchronous closure that reads from the precomputed result dict.
This avoids spinning up a fresh event loop + bus connection per pair, and
avoids the impossible alternative of reusing one ``OrionBusAsync`` connection
across multiple independent ``asyncio.run()`` calls (its Redis connection is
bound to the event loop it was created on).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.core.llm_json import parse_json_object
from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateEdgePredicateV1,
    SubstrateEdgeV1,
)
from orion.substrate.relation_classification import RelationClassifier

logger = logging.getLogger("hub.concept_relation_classifier")

RelationPair = tuple[ConceptNodeV1, ConceptNodeV1, SubstrateEdgeV1]

_SUMMARY_TRUNC_CHARS = 300
# "quick" mirrors services/orion-memory-consolidation's classify lane default use
# (small/fast route, not the full reasoning "metacog"/"brain" routes) -- this is a
# bounded per-pair structured judgment, not a conversational turn.
_DEFAULT_ROUTE = "quick"
_DEFAULT_MAX_TOKENS = 40


class _RelationJudgment(BaseModel):
    """Local to this seam -- not a bus-published event, no registry entry needed
    (same precedent as ``ConceptRelationDecision`` in
    ``orion/memory/crystallization/concept_relation.py``).

    ``extra="ignore"``, not ``"forbid"``: small instruct models frequently add a
    stray field (e.g. "reasoning") to structured output despite prompt
    instructions to emit only the bare object -- same rationale as
    ``ConceptRelationDecision``.
    """

    model_config = ConfigDict(extra="ignore")

    predicate: Literal["supports", "contradicts", "refines", "none"] = "none"


def _truncate(text: str, *, limit: int) -> str:
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _node_block(label: str, node: ConceptNodeV1) -> str:
    definition = _truncate(node.definition or "", limit=_SUMMARY_TRUNC_CHARS)
    line = f"{label}: {node.label}"
    if definition:
        line += f"\n  definition: {definition}"
    return line


def _build_relation_prompt(
    node_a: ConceptNodeV1, node_b: ConceptNodeV1, edge: SubstrateEdgeV1
) -> str:
    co_occurrence = None
    if isinstance(edge.metadata, dict):
        co_occurrence = edge.metadata.get("co_occurrence_count")
    evidence_line = f"co-occurrence count: {co_occurrence}\n" if co_occurrence is not None else ""

    return (
        "Judge the relationship between CONCEPT A and CONCEPT B below, which have "
        "been observed co-occurring in the same conversational context.\n\n"
        f"{_node_block('CONCEPT A', node_a)}\n"
        f"{_node_block('CONCEPT B', node_b)}\n"
        f"{evidence_line}\n"
        "Choose exactly one relation, judged from A's perspective toward B:\n"
        '  "supports"    = A provides evidence for, reinforces, or is consistent with B\n'
        '  "contradicts" = A conflicts with or undermines B\n'
        '  "refines"     = A is a more precise/updated/narrower version of B\n'
        '  "none"        = none of the above with real confidence; default when unsure\n\n'
        "Respond with ONLY a single JSON object, no prose, no markdown fences, shaped "
        "exactly like:\n"
        '{"predicate": "supports"|"contradicts"|"refines"|"none"}\n'
    )


async def _classify_pair_async(
    bus: OrionBusAsync,
    node_a: ConceptNodeV1,
    node_b: ConceptNodeV1,
    edge: SubstrateEdgeV1,
    *,
    settings: Any,
    timeout_sec: float,
    route: str,
) -> Optional[SubstrateEdgePredicateV1]:
    """One bounded, structured-output LLM call over an already-connected ``bus``.

    NEVER raises -- degrades to ``None`` on any failure (RPC timeout, decode
    failure, malformed/unparseable JSON, schema-invalid predicate, "none").
    """
    try:
        prompt = _build_relation_prompt(node_a, node_b, edge)
        rpc_corr = str(uuid4())
        gateway_service_name = str(getattr(settings, "LLM_GATEWAY_SERVICE_NAME", "LLMGatewayService"))
        reply_channel = f"orion:exec:result:{gateway_service_name}:{rpc_corr}"
        payload = ChatRequestPayload(
            messages=[LLMMessage(role="user", content=prompt)],
            route=route,
            options={
                "return_logprobs": False,
                "max_tokens": _DEFAULT_MAX_TOKENS,
                "llm_route": route,
                "purpose": "classify",
                "skip_spark_candidate_publish": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        env = BaseEnvelope(
            kind="llm.chat.request",
            source=ServiceRef(
                name=settings.SERVICE_NAME,
                version=settings.SERVICE_VERSION,
                node=settings.NODE_NAME,
            ),
            correlation_id=rpc_corr,
            reply_to=reply_channel,
            payload=payload.model_dump(mode="json"),
        )
        msg = await bus.rpc_request(
            settings.CHANNEL_LLM_INTAKE,
            env,
            reply_channel=reply_channel,
            timeout_sec=timeout_sec,
        )
        decoded = bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(decoded.error)
        content = str(
            decoded.envelope.payload.get("content") or decoded.envelope.payload.get("text") or ""
        )
        # Models sometimes wrap the JSON object in prose or markdown fences despite
        # instructions -- reuse the shared, already-battle-tested LLM JSON extraction
        # (handles ```json fences, trailing commas, Python True/False/None literals)
        # instead of a narrower hand-rolled parse.
        obj = parse_json_object(content)
        judgment = _RelationJudgment.model_validate(obj)
        if judgment.predicate == "none":
            return None
        return judgment.predicate  # type: ignore[return-value]
    except Exception as exc:
        logger.warning(
            "concept_relation_classify_pair_failed source=%s target=%s error=%s",
            node_a.node_id,
            node_b.node_id,
            exc,
        )
        return None


async def _classify_pairs_batch_async(
    pairs: list[RelationPair],
    *,
    settings: Any,
    timeout_sec: float,
    route: str,
) -> dict[str, Optional[SubstrateEdgePredicateV1]]:
    """One shared bus connection for the whole capped batch, sequential RPCs.

    NEVER raises -- a connect failure degrades every pair in the batch to
    ``None`` rather than propagating.
    """
    results: dict[str, Optional[SubstrateEdgePredicateV1]] = {}
    bus = OrionBusAsync(str(settings.ORION_BUS_URL))
    try:
        await bus.connect()
    except Exception as exc:
        logger.warning("concept_relation_classify_bus_connect_failed error=%s", exc)
        return {edge.edge_id: None for _, _, edge in pairs}
    try:
        for node_a, node_b, edge in pairs:
            results[edge.edge_id] = await _classify_pair_async(
                bus,
                node_a,
                node_b,
                edge,
                settings=settings,
                timeout_sec=timeout_sec,
                route=route,
            )
    finally:
        try:
            await bus.close()
        except Exception:
            pass
    return results


def build_llm_relation_classifier(
    pairs: list[RelationPair],
    *,
    settings: Any,
    timeout_sec: Optional[float] = None,
    route: Optional[str] = None,
) -> RelationClassifier:
    """Runs one bounded async batch of LLM classification calls up front, then
    returns a plain synchronous ``RelationClassifier`` closure over the
    precomputed results -- matching the sync-callable contract
    ``classify_relation()`` requires (it calls the injected classifier
    directly, not awaited).

    Callers are expected to have already capped ``pairs`` to a small bound
    (the ingestion route caps it before calling this) -- this function does
    not itself impose a limit, since the cap is a caller policy decision
    (how many pairs are worth the added route latency), not a classifier
    concern.

    NEVER raises -- any batch-level failure (bus connect, event loop) degrades
    every pair to ``None`` so the caller's per-pair ``classify_relation()``
    loop proceeds with "no typed edge for this pair" rather than failing the
    whole ingestion run.
    """
    resolved_timeout = float(
        timeout_sec if timeout_sec is not None else getattr(settings, "HUB_LLM_GATEWAY_TIMEOUT_SEC", 5.0)
    )
    resolved_route = str(route or _DEFAULT_ROUTE)
    try:
        judgments = asyncio.run(
            _classify_pairs_batch_async(
                pairs, settings=settings, timeout_sec=resolved_timeout, route=resolved_route
            )
        )
    except Exception as exc:
        logger.warning("concept_relation_classify_batch_failed error=%s", exc)
        judgments = {edge.edge_id: None for _, _, edge in pairs}

    def _classifier(
        node_a: ConceptNodeV1, node_b: ConceptNodeV1, edge: SubstrateEdgeV1
    ) -> Optional[SubstrateEdgePredicateV1]:
        return judgments.get(edge.edge_id)

    return _classifier
