"""One graph node per person, and an ``instance_of`` claim per camera cluster (2026-10-23).

The ``person`` / ``person_link`` tables (PR #2852, orion-sql-writer person_record_loop,
planner orion/person_model/record.py) already say which named camera clusters are one
person. This clocked pass puts that in the substrate graph without matching names again:

- one EntityNodeV1 per ``person`` row whose status is not ``not_a_person``. A row whose
  person_id is a chat referent (``referent-*``, from people.yaml) already has its node,
  written by the memory referent projector; only minted ``person-*`` rows get a node here;
- for each active ``vision_individual`` link (``revoked_at IS NULL``): one journalled
  claim "cluster instance_of person" (SubstrateGraphProposalV1 + a decision to
  ``provisional``), applied by AssertionProjector as a ``semantic_projection`` edge;
- a claim with no active link left (the link was revoked by a rename or a cleared name,
  or the person became ``not_a_person``) gets a decision to ``deprecated``: its edge is
  closed and stops walking. Nothing is deleted;
- a minted person node with no active link left (renamed, taken over by a people.yaml
  person, or ``not_a_person``) is re-written as ``deprecated``;
- a claim the journal says was applied but whose Assertion is missing from the graph (a
  wiped or rebuilt store) is replayed from the journal. A claim whose last attempt failed
  for a terminal reason is counted as ``stuck``: it needs a human, no retry will fix it.

The "different people" guard and the person-kinds-only rule live in record.py and are not
repeated here. Day sighting links stay on the clusters (app/walkway_seen_loop.py).
A claim waits until both endpoint nodes exist, so the projector never fails on a missing
endpoint. A cluster normally gets its node when the interlocutor names it
(app/ask_answered_listener.py, from the bus). That event is not replayed: 7 linked clusters
named 10-15..10-21 never got a node (the listener was not hearing at the time). So a linked
cluster with no node gets one here, built from its ``vision_individual`` row with the same
adapter the listener uses. An existing cluster node is never touched.

Trace: ``person_graph synced persons=... nodes_written=... nodes_retired=... clusters_written=... claims=...
retired=... waiting=... replayed=... stuck=... applied=... held=... failed=...``. Fail-open: an error is logged; the loop keeps going.
Kill switch: PERSON_GRAPH_ENABLED=false.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("orion.substrate.runtime.person_graph")

PRODUCER = "person.record"      # fenced in orion/substrate/reconcile.py
POLICY = "person_link_v1"
SOURCE_KIND = "person_record"
MINTED_PREFIX = "person-"       # orion/person_model/record.py minted_person_id
_NAMESPACE = uuid.UUID("2d8f5f0e-7c4b-5a39-9e61-4b3a8c1d7e25")

PERSONS_SQL = "SELECT person_id, display_name, status, created_at FROM person"
LINKS_SQL = (
    "SELECT l.link_id, l.person_id, l.source_ref, l.evidence_ref, l.decided_by, "
    "v.label, v.kind, v.label_ask_id, v.stream_id, o.answered_at "
    "FROM person_link l LEFT JOIN vision_individual v ON v.individual_id = l.source_ref "
    "LEFT JOIN orion_ask o ON o.ask_id = v.label_ask_id "
    "WHERE l.source_kind = 'vision_individual' AND l.revoked_at IS NULL"
)


@dataclass(frozen=True)
class PersonRow:
    person_id: str
    display_name: str
    status: str = "active"
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class LinkRow:
    link_id: str
    person_id: str
    individual_id: str
    evidence_ref: Optional[str] = None
    decided_by: str = "camera_name"
    # The cluster's own row, to rebuild its node if the ask listener missed the answer.
    label: str = ""
    kind: Optional[str] = None
    label_ask_id: Optional[str] = None
    stream_id: Optional[str] = None
    answered_at: Optional[datetime] = None


@dataclass
class SyncReport:
    persons: int = 0
    nodes_written: int = 0
    nodes_retired: int = 0
    clusters_written: int = 0
    claims: int = 0
    retired: int = 0
    waiting: int = 0
    replayed: int = 0
    stuck: int = 0
    applied: int = 0
    held: int = 0             # decisions waiting behind an earlier revision of their target
    failed: dict[str, str] = field(default_factory=dict)


def statement_key(cluster_node_id: str, person_id: str) -> str:
    return f"{cluster_node_id}|instance_of|{person_id}|"


def _decision_id(target: str, prior: int, state: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{target}|{prior}|{state}"))


def person_node(p: PersonRow, *, now: datetime, state: str = "provisional") -> Any:
    from orion.core.schemas.cognitive_substrate import EntityNodeV1, SubstrateProvenanceV1, SubstrateTemporalWindowV1

    return EntityNodeV1(
        node_id=p.person_id, label=" ".join(p.display_name.split()), entity_type="person", anchor_scope="world",
        subject_ref=f"person:{p.person_id}", promotion_state=state,
        temporal=SubstrateTemporalWindowV1(observed_at=now, valid_from=p.created_at or now),
        provenance=SubstrateProvenanceV1(authority="user_asserted", source_kind=SOURCE_KIND,
                                         source_channel="person", producer=PRODUCER,
                                         evidence_refs=[f"person:{p.person_id}"]),
    )


def plan_events(*, persons: list[PersonRow], links: list[LinkRow], latest: dict[str, Any],
                has_node: Callable[[str], bool], now: datetime) -> tuple[list[Any], int, int, int]:
    """Journal events that make the claims match the active links. Pure.

    ``latest``: target_id -> this producer's newest decision about it.
    Returns (events, claims, retired, waiting)."""
    from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1, SubstrateGraphProposalV1
    from orion.substrate.adapters.vision_individual import entity_node_id_for_individual
    from orion.substrate.assertion_projector import assertion_node_id

    live = {p.person_id: p for p in persons if p.status != "not_a_person"}
    events: list[Any] = []
    wanted: set[str] = set()
    claims = waiting = retired = 0
    for link in sorted(links, key=lambda x: x.individual_id):
        person = live.get(link.person_id)
        if person is None:
            continue
        cluster = entity_node_id_for_individual(link.individual_id)
        key = statement_key(cluster, link.person_id)
        target = assertion_node_id(key)
        wanted.add(target)
        prior = latest.get(target)
        if prior is not None and prior.resulting_state == "provisional":
            continue
        if not (has_node(cluster) and has_node(link.person_id)):
            waiting += 1
            continue
        revision = prior.resulting_revision if prior is not None else 0
        # One proposal per acceptance: a re-accepted claim gets its own valid_from and text.
        proposal_id = str(uuid.uuid5(_NAMESPACE, f"{key}|{revision}"))
        events.append(SubstrateGraphProposalV1(
            proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target, actor=PRODUCER,
            subject_node_id=cluster, subject_kind="entity", object_node_id=link.person_id, object_kind="entity",
            predicate="instance_of", statement_key=key,
            statement_text=f"a camera cluster named {person.display_name} is {person.display_name}",
            anchor_scope="world", subject_ref=f"vision_individual:{link.individual_id}",
            authority="user_asserted", recorded_at=now,
        ))
        events.append(SubstrateGraphDecisionV1(
            proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target, actor=PRODUCER,
            decision_id=_decision_id(target, revision, "provisional"), expected_prior_revision=revision,
            resulting_state="provisional", policy=POLICY, authority="user_asserted",
            rationale=f"person_link {link.decided_by}",
            evidence_refs=[f"person_link:{link.link_id}"] + ([f"orion_ask:{link.evidence_ref}"]
                                                            if link.evidence_ref else []),
            recorded_at=now,
        ))
        claims += 1
    for target, prior in sorted(latest.items()):
        if target in wanted or prior.resulting_state != "provisional":
            continue
        events.append(SubstrateGraphDecisionV1(
            proposal_id=prior.proposal_id, proposal_kind="relationship_assertion", target_id=target,
            actor=PRODUCER, decision_id=_decision_id(target, prior.resulting_revision, "deprecated"),
            expected_prior_revision=prior.resulting_revision, resulting_state="deprecated", policy=POLICY,
            authority="user_asserted", rationale="person_link revoked or person not_a_person", recorded_at=now,
        ))
        retired += 1
    return events, claims, retired, waiting


async def sync_person_graph(*, persons: list[PersonRow], links: list[LinkRow], journal: Any, store: Any,
                            projector: Any, now: Optional[datetime] = None) -> SyncReport:
    """One pass: person nodes, then journal events, then apply this producer's pending decisions."""
    from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
    from orion.substrate.materializer import SubstrateGraphMaterializer
    from orion.substrate.reconcile import SubstrateIdentityResolver

    ts = now or datetime.now(timezone.utc)
    report = SyncReport(persons=len(persons))
    materializer = SubstrateGraphMaterializer(store=store, identity_resolver=SubstrateIdentityResolver(store=store))
    linked = {link.person_id for link in links}
    for p in persons:
        if not p.person_id.startswith(MINTED_PREFIX):
            continue
        live = p.status != "not_a_person" and p.person_id in linked
        existing = await asyncio.to_thread(store.get_node_by_id, p.person_id)
        if existing is None and not live:
            continue
        node = person_node(p, now=ts, state="provisional" if live else "deprecated")
        if existing is not None and (getattr(existing, "label", None), existing.promotion_state) == (
                node.label, node.promotion_state):
            continue
        await asyncio.to_thread(materializer.apply_record,
                                SubstrateGraphRecordV1(anchor_scope="world", subject_ref=node.subject_ref, nodes=[node]))
        if live:
            report.nodes_written += 1
        else:
            report.nodes_retired += 1
    report.clusters_written = await _ensure_cluster_nodes(persons, links, store=store, materializer=materializer,
                                                         now=ts)
    latest = await journal.latest_decisions(PRODUCER)
    events, report.claims, report.retired, report.waiting = plan_events(
        persons=persons, links=links, latest=latest, has_node=lambda nid: store.get_node_by_id(nid) is not None,
        now=ts)
    for event in events:
        await journal.append(event)
    await _recover(latest, journal=journal, store=store, projector=projector, report=report)
    result = await projector.run_once()
    report.applied, report.held, report.failed = len(result.applied), len(result.waiting), dict(result.failed)
    if result.blocked is not None:
        report.failed["readers_not_ready"] = result.blocked.reason
    logger.info("person_graph synced persons=%d nodes_written=%d nodes_retired=%d clusters_written=%d claims=%d "
                "retired=%d waiting=%d replayed=%d stuck=%d applied=%d held=%d failed=%s",
                report.persons, report.nodes_written, report.nodes_retired, report.clusters_written,
                report.claims, report.retired,
                report.waiting, report.replayed, report.stuck, report.applied, report.held, report.failed)
    return report


async def _ensure_cluster_nodes(persons: list[PersonRow], links: list[LinkRow], *, store: Any, materializer: Any,
                                now: datetime) -> int:
    """A linked cluster of a live person with no node gets the node the ask listener would have
    written. Returns how many were written."""
    from orion.substrate.adapters.vision_individual import (
        entity_node_id_for_individual,
        map_vision_individual_label_to_substrate,
    )

    live = {p.person_id for p in persons if p.status != "not_a_person"}
    written = 0
    for link in links:
        if link.person_id not in live or not link.label.strip():
            continue
        if await asyncio.to_thread(store.get_node_by_id, entity_node_id_for_individual(link.individual_id)):
            continue
        record = map_vision_individual_label_to_substrate(
            individual_id=link.individual_id, label=link.label, kind=link.kind,
            ask_id=link.label_ask_id or link.evidence_ref or f"person_link:{link.link_id}",
            answered_at=link.answered_at or now, stream_id=link.stream_id)
        if record is None:
            continue
        await asyncio.to_thread(materializer.apply_record, record)
        written += 1
    return written


async def _recover(latest: dict[str, Any], *, journal: Any, store: Any, projector: Any, report: SyncReport) -> None:
    """For claims whose Assertion in the graph is not at the journal's latest revision: replay
    an applied one the graph lost, count a terminally failed one as stuck. A decision that is
    simply pending is left to the projector."""
    from orion.substrate.graph_journal import TERMINAL_FAILURE_REASONS

    for target, decision in sorted(latest.items()):
        node = store.get_node_by_id(target)
        if node is not None and getattr(node, "revision", None) == decision.resulting_revision:
            continue
        if await journal.latest_applied_revision(target) >= decision.resulting_revision:
            if await projector.replay(decision):
                report.replayed += 1
            continue
        last = await journal.last_materialization(decision.decision_id)
        if last is not None and last.outcome == "failed" and (
                (last.failure_reason or "").split(":")[0] in TERMINAL_FAILURE_REASONS):
            report.stuck += 1
            logger.warning("person_graph stuck target=%s decision=%s reason=%s",
                           target, decision.decision_id, last.failure_reason)


async def read_rows(pool: Any) -> tuple[list[PersonRow], list[LinkRow]]:
    async with pool.acquire() as conn:
        persons = [PersonRow(r["person_id"], r["display_name"], r["status"], r["created_at"])
                   for r in await conn.fetch(PERSONS_SQL)]
        links = [LinkRow(r["link_id"], r["person_id"], r["source_ref"], r["evidence_ref"], r["decided_by"],
                         r["label"] or "", r["kind"], r["label_ask_id"], r["stream_id"], r["answered_at"])
                 for r in await conn.fetch(LINKS_SQL)]
    return persons, links


def asyncpg_dsn(uri: str) -> str:
    """POSTGRES_URI is SQLAlchemy-shaped; asyncpg wants plain postgresql://."""
    scheme, sep, rest = uri.partition("://")
    return f"{scheme.split('+', 1)[0]}{sep}{rest}"


async def run_person_graph_loop(
    *,
    get_store: Callable[[], Any],
    settings: Any,
    stop_event: asyncio.Event | None = None,
) -> None:
    if not bool(getattr(settings, "person_graph_enabled", False)):
        logger.info("person_graph loop disabled by config (PERSON_GRAPH_ENABLED=false)")
        return
    import os

    import asyncpg

    from orion.substrate.assertion_projector import AssertionProjector
    from orion.substrate.graph_journal import SubstrateGraphJournal
    from orion.substrate.materializer import SubstrateGraphMaterializer
    from orion.substrate.reader_capability import readiness, required_readers_from_env

    interval = float(getattr(settings, "person_graph_interval_sec", 600.0) or 600.0)
    uri = str(os.getenv("FALKORDB_URI", "") or "").strip()
    required = required_readers_from_env()
    pool = None
    projector = None
    logger.info("person_graph loop starting interval_sec=%.0f", interval)
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                if pool is None:
                    pool = await asyncpg.create_pool(dsn=asyncpg_dsn(settings.postgres_uri), min_size=1, max_size=2)
                # Off the loop: the first get_store() hydrates the whole substrate graph.
                store = await asyncio.to_thread(get_store)
                if store is None:
                    logger.warning("person_graph skipped store=False")
                else:
                    journal = SubstrateGraphJournal(pool)
                    if projector is None:
                        projector = AssertionProjector(
                            journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                            readiness=lambda: readiness(uri, required), proposal_actors=(PRODUCER,))
                    persons, links = await read_rows(pool)
                    await sync_person_graph(persons=persons, links=links, journal=journal, store=store,
                                            projector=projector)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- fail-open; a missing person table lands here
                logger.warning("person_graph pass failed err=%s: %s", type(exc).__name__, exc)
            await asyncio.sleep(interval)
    finally:
        if pool is not None:
            await pool.close()
