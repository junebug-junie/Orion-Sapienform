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
  closed and stops walking. Nothing is deleted.

The "different people" guard and the person-kinds-only rule live in record.py and are not
repeated here. Day sighting links stay on the clusters (app/walkway_seen_loop.py).
A claim waits until both endpoint nodes exist (a cluster gets its node when the interlocutor
names it, app/ask_answered_listener.py), so the projector never fails on a missing endpoint.

Trace: ``person_graph synced persons=... nodes_written=... claims=... retired=...
waiting=... applied=... failed=...``. Fail-open: an error is logged; the loop keeps going.
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
    "SELECT link_id, person_id, source_ref, evidence_ref, decided_by FROM person_link "
    "WHERE source_kind = 'vision_individual' AND revoked_at IS NULL"
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


@dataclass
class SyncReport:
    persons: int = 0
    nodes_written: int = 0
    claims: int = 0
    retired: int = 0
    waiting: int = 0
    applied: int = 0
    failed: dict[str, str] = field(default_factory=dict)


def statement_key(cluster_node_id: str, person_id: str) -> str:
    return f"{cluster_node_id}|instance_of|{person_id}|"


def _decision_id(target: str, prior: int, state: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{target}|{prior}|{state}"))


def person_node(p: PersonRow, *, now: datetime) -> Any:
    from orion.core.schemas.cognitive_substrate import EntityNodeV1, SubstrateProvenanceV1, SubstrateTemporalWindowV1

    return EntityNodeV1(
        node_id=p.person_id, label=" ".join(p.display_name.split()), entity_type="person", anchor_scope="world",
        subject_ref=f"person:{p.person_id}", promotion_state="provisional",
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
        proposal_id = str(uuid.uuid5(_NAMESPACE, key))
        events.append(SubstrateGraphProposalV1(
            proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target, actor=PRODUCER,
            subject_node_id=cluster, subject_kind="entity", object_node_id=link.person_id, object_kind="entity",
            predicate="instance_of", statement_key=key,
            statement_text=f"a camera cluster named {person.display_name} is {person.display_name}",
            anchor_scope="world", subject_ref=f"vision_individual:{link.individual_id}",
            authority="user_asserted", recorded_at=now,
        ))
        revision = prior.resulting_revision if prior is not None else 0
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
    for p in persons:
        if p.status == "not_a_person" or not p.person_id.startswith(MINTED_PREFIX):
            continue
        existing = await asyncio.to_thread(store.get_node_by_id, p.person_id)
        node = person_node(p, now=ts)
        if existing is not None and getattr(existing, "label", None) == node.label:
            continue
        await asyncio.to_thread(materializer.apply_record,
                                SubstrateGraphRecordV1(anchor_scope="world", subject_ref=node.subject_ref, nodes=[node]))
        report.nodes_written += 1
    latest = await journal.latest_decisions(PRODUCER)
    events, report.claims, report.retired, report.waiting = plan_events(
        persons=persons, links=links, latest=latest, has_node=lambda nid: store.get_node_by_id(nid) is not None,
        now=ts)
    for event in events:
        await journal.append(event)
    result = await projector.run_once()
    report.applied, report.failed = len(result.applied), dict(result.failed)
    if result.blocked is not None:
        report.failed["readers_not_ready"] = result.blocked.reason
    logger.info("person_graph synced persons=%d nodes_written=%d claims=%d retired=%d waiting=%d applied=%d failed=%s",
                report.persons, report.nodes_written, report.claims, report.retired, report.waiting,
                report.applied, report.failed)
    return report


async def read_rows(pool: Any) -> tuple[list[PersonRow], list[LinkRow]]:
    async with pool.acquire() as conn:
        persons = [PersonRow(r["person_id"], r["display_name"], r["status"], r["created_at"])
                   for r in await conn.fetch(PERSONS_SQL)]
        links = [LinkRow(r["link_id"], r["person_id"], r["source_ref"], r["evidence_ref"], r["decided_by"])
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
