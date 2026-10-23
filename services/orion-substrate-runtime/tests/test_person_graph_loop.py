"""person / person_link rows -> person nodes and instance_of claims (app/person_graph_loop.py).

Real AssertionProjector and materializer over an in-memory store; the journal is an
in-memory stand-in with SubstrateGraphJournal's methods.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import EntityNodeV1, SubstrateGraphRecordV1
from orion.core.schemas.substrate_graph_journal import (
    SubstrateGraphDecisionV1,
    SubstrateGraphMaterializationV1,
    SubstrateGraphProposalV1,
)
from orion.substrate.adapters.vision_individual import (
    entity_node_id_for_individual,
    map_vision_individual_label_to_substrate,
)
from orion.substrate.assertion_projector import AssertionProjector, assertion_node_id
from orion.substrate.graph_journal import TERMINAL_FAILURE_REASONS, RevisionConflict
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.neighborhood import walkable_edge
from orion.substrate.reader_capability import ALWAYS_READY
from orion.substrate.store import InMemorySubstrateGraphStore

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
SAM = "person-0000aaaa-0000-5000-8000-000000000001"
ROBIN = "person-0000aaaa-0000-5000-8000-000000000002"
HOST = "referent-0000bbbb-0000-5000-8000-000000000003"


class FakeJournal:
    def __init__(self) -> None:
        self.events: dict[str, object] = {}
        self.order: list[str] = []
        self.slots: set[tuple[str, int]] = set()

    def _of(self, cls) -> list:
        return [self.events[k] for k in self.order if isinstance(self.events[k], cls)]

    async def append(self, event) -> bool:
        key = f"{event.event_kind}:{event.event_id}"
        if key in self.events:
            return False
        if isinstance(event, SubstrateGraphDecisionV1):
            slot = (event.target_id, event.resulting_revision)
            if slot in self.slots:
                raise RevisionConflict(str(slot))
            self.slots.add(slot)
        self.events[key] = event
        self.order.append(key)
        return True

    async def pending_decisions(self, *, limit: int = 100, proposal_actors=None) -> list:
        applied = {m.decision_id for m in self._of(SubstrateGraphMaterializationV1) if m.outcome == "applied"}
        out = []
        for d in self._of(SubstrateGraphDecisionV1):
            if d.decision_id in applied or d.expected_prior_revision > await self.latest_applied_revision(d.target_id):
                continue
            last = await self.last_materialization(d.decision_id)
            if last is not None and (last.failure_reason or "").split(":")[0] in TERMINAL_FAILURE_REASONS:
                continue  # same as the real _PENDING_DECISIONS: terminal failures are never retried
            proposal = await self.proposal(d.proposal_id)
            if proposal_actors is not None and (proposal is None or proposal.actor not in proposal_actors):
                continue
            out.append(d)
        return out[:limit]

    async def proposal(self, proposal_id: str):
        return next((p for p in self._of(SubstrateGraphProposalV1) if p.proposal_id == proposal_id), None)

    async def latest_applied_revision(self, target_id: str) -> int:
        return max([m.revision for m in self._of(SubstrateGraphMaterializationV1)
                    if m.outcome == "applied" and m.target_id == target_id] or [0])

    async def failed_attempts(self, decision_id: str) -> int:
        return sum(1 for m in self._of(SubstrateGraphMaterializationV1)
                   if m.outcome == "failed" and m.decision_id == decision_id)

    async def last_materialization(self, decision_id: str):
        rows = [m for m in self._of(SubstrateGraphMaterializationV1) if m.decision_id == decision_id]
        return rows[-1] if rows else None

    async def latest_decisions(self, actor: str) -> dict:
        out: dict = {}
        for d in self._of(SubstrateGraphDecisionV1):
            if d.actor == actor and (d.target_id not in out or d.resulting_revision > out[d.target_id].resulting_revision):
                out[d.target_id] = d
        return out


def _world(*clusters: str):
    from app.person_graph_loop import PRODUCER

    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    for iid in clusters:
        materializer.apply_record(map_vision_individual_label_to_substrate(
            individual_id=iid, label="Sam", kind="person", ask_id=f"ask-{iid}", answered_at=NOW))
    journal = FakeJournal()
    projector = AssertionProjector(journal=journal, materializer=materializer, readiness=ALWAYS_READY,
                                   proposal_actors=(PRODUCER,))
    return store, journal, projector


def _sync(store, journal, projector, persons, links, now=NOW):
    from app.person_graph_loop import sync_person_graph

    return asyncio.run(sync_person_graph(persons=persons, links=links, journal=journal, store=store,
                                         projector=projector, now=now))


def _walks(store, iid: str, person_id: str) -> bool:
    """Is there a walkable instance_of edge from the cluster's node to the person?"""
    from app.person_graph_loop import statement_key

    target = assertion_node_id(statement_key(entity_node_id_for_individual(iid), person_id))
    edges = [e for e in store.snapshot().edges.values()
             if e.edge_role == "semantic_projection" and e.assertion_id == target]
    return any(walkable_edge(e, store.get_node_by_id(target)) for e in edges)


def test_two_clusters_of_one_person_get_one_node_and_two_walkable_claims():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1", "c2")
    report = _sync(store, journal, projector, [PersonRow(SAM, "Sam")],
                   [LinkRow("l1", SAM, "c1", "ask-c1"), LinkRow("l2", SAM, "c2", "ask-c2")])
    node = store.get_node_by_id(SAM)
    assert isinstance(node, EntityNodeV1) and node.entity_type == "person" and node.label == "Sam"
    assert [n for n in store.snapshot().nodes.values() if getattr(n, "entity_type", "") == "person"
            and n.node_kind == "entity" and n.provenance.producer == "person.record"] == [node]
    assert (report.nodes_written, report.claims, report.applied, report.failed) == (1, 2, 2, {})
    assert _walks(store, "c1", SAM) and _walks(store, "c2", SAM)


def test_second_pass_writes_nothing():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1")
    args = ([PersonRow(SAM, "Sam")], [LinkRow("l1", SAM, "c1")])
    _sync(store, journal, projector, *args)
    before = len(journal.order)
    report = _sync(store, journal, projector, *args)
    assert (report.nodes_written, report.claims, report.retired, len(journal.order)) == (0, 0, 0, before)


def test_people_yaml_person_uses_the_referent_node_and_never_rewrites_it():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1")
    referent = EntityNodeV1.model_validate({**store.get_node_by_id(entity_node_id_for_individual("c1")).model_dump(),
                                            "node_id": HOST, "label": "host", "subject_ref": None})
    SubstrateGraphMaterializer(store=store).apply_record(SubstrateGraphRecordV1(anchor_scope="world",
                                                                                nodes=[referent]))
    report = _sync(store, journal, projector, [PersonRow(HOST, "Host")], [LinkRow("l1", HOST, "c1")])
    assert report.nodes_written == 0 and store.get_node_by_id(HOST).label == "host"
    assert _walks(store, "c1", HOST)


def test_rename_retires_the_old_claim_and_links_the_new_person():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1")
    persons = [PersonRow(SAM, "Sam"), PersonRow(ROBIN, "Robin")]
    _sync(store, journal, projector, persons, [LinkRow("l1", SAM, "c1")])
    report = _sync(store, journal, projector, persons, [LinkRow("l2", ROBIN, "c1")], now=NOW + timedelta(minutes=10))
    assert (report.claims, report.retired, report.failed) == (1, 1, {})
    assert not _walks(store, "c1", SAM) and _walks(store, "c1", ROBIN)
    # Review 1: the person left with no link stays in the graph, but deprecated.
    assert report.nodes_retired == 1 and store.get_node_by_id(SAM).promotion_state == "deprecated"
    assert store.get_node_by_id(ROBIN).promotion_state == "provisional"


def test_relink_after_retire_accepts_again_at_the_next_revision():
    from app.person_graph_loop import LinkRow, PersonRow, statement_key

    store, journal, projector = _world("c1")
    persons = [PersonRow(SAM, "Sam")]
    _sync(store, journal, projector, persons, [LinkRow("l1", SAM, "c1")])
    _sync(store, journal, projector, persons, [], now=NOW + timedelta(minutes=10))
    assert not _walks(store, "c1", SAM)
    _sync(store, journal, projector, persons, [LinkRow("l3", SAM, "c1")], now=NOW + timedelta(minutes=20))
    assert _walks(store, "c1", SAM)
    target = assertion_node_id(statement_key(entity_node_id_for_individual("c1"), SAM))
    assert store.get_node_by_id(target).revision == 3
    # Review 4: the re-accepted claim has its own proposal (own valid_from), not the first one.
    assert len({p.proposal_id for p in journal._of(SubstrateGraphProposalV1)}) == 2
    assert store.get_node_by_id(SAM).promotion_state == "provisional"


def test_not_a_person_gets_no_node_and_its_claims_retire():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1")
    links = [LinkRow("l1", SAM, "c1")]
    _sync(store, journal, projector, [PersonRow(SAM, "Sam")], links)
    report = _sync(store, journal, projector, [PersonRow(SAM, "Sam", "not_a_person")], links,
                   now=NOW + timedelta(minutes=10))
    assert report.retired == 1 and not _walks(store, "c1", SAM)
    assert store.get_node_by_id(SAM).promotion_state == "deprecated"

    fresh, j2, p2 = _world("c2")
    report = _sync(fresh, j2, p2, [PersonRow(ROBIN, "Robin", "not_a_person")], [LinkRow("l2", ROBIN, "c2")])
    assert fresh.get_node_by_id(ROBIN) is None and (report.claims, len(j2.order)) == (0, 0)


def test_cluster_without_a_node_waits_instead_of_failing_in_the_projector():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world()
    persons, links = [PersonRow(SAM, "Sam")], [LinkRow("l1", SAM, "c9")]
    report = _sync(store, journal, projector, persons, links)
    assert (report.waiting, report.claims, report.failed, len(journal.order)) == (1, 0, {}, 0)
    SubstrateGraphMaterializer(store=store).apply_record(map_vision_individual_label_to_substrate(
        individual_id="c9", label="Sam", kind="person", ask_id="ask-c9", answered_at=NOW))
    report = _sync(store, journal, projector, persons, links)
    assert (report.waiting, report.claims) == (0, 1) and _walks(store, "c9", SAM)


def test_display_name_change_relabels_the_person_node():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1")
    links = [LinkRow("l1", SAM, "c1")]
    _sync(store, journal, projector, [PersonRow(SAM, "Sam")], links)
    report = _sync(store, journal, projector, [PersonRow(SAM, "Samantha")], links)
    assert report.nodes_written == 1 and store.get_node_by_id(SAM).label == "Samantha"


def test_asyncpg_dsn_drops_the_sqlalchemy_driver():
    from app.person_graph_loop import asyncpg_dsn

    assert asyncpg_dsn("postgresql+psycopg2://u:p@h:5432/db") == "postgresql://u:p@h:5432/db"
    assert asyncpg_dsn("postgresql://u:p@h/db") == "postgresql://u:p@h/db"


def test_a_person_with_no_link_never_gets_a_node():
    from app.person_graph_loop import PersonRow

    store, journal, projector = _world()
    report = _sync(store, journal, projector, [PersonRow(SAM, "Sam")], [])
    assert report.nodes_written == 0 and store.get_node_by_id(SAM) is None


def test_claims_come_back_after_the_graph_is_wiped():
    """Review 2: the journal says applied, the graph lost it -> replayed, no new journal rows."""
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world("c1")
    persons, links = [PersonRow(SAM, "Sam")], [LinkRow("l1", SAM, "c1")]
    _sync(store, journal, projector, persons, links)
    rows = len(journal.order)
    wiped, _, _ = _world("c1")
    projector2 = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=wiped),
                                    readiness=ALWAYS_READY, proposal_actors=("person.record",))
    report = _sync(wiped, journal, projector2, persons, links)
    assert (report.replayed, report.claims, len(journal.order)) == (1, 0, rows)
    assert _walks(wiped, "c1", SAM)


def test_a_terminally_failed_claim_is_counted_as_stuck_not_done():
    """Review 3: a terminal failure is never retried; the trace must say so every pass."""
    from app.person_graph_loop import LinkRow, PersonRow, PRODUCER

    store, journal, projector = _world("c1")
    persons, links = [PersonRow(SAM, "Sam")], [LinkRow("l1", SAM, "c1")]

    class _Refusing:
        async def run_once(self):
            from orion.substrate.assertion_projector import ProjectionReportV1

            for d in await journal.pending_decisions(proposal_actors=(PRODUCER,)):
                await journal.append(SubstrateGraphMaterializationV1(
                    proposal_id=d.proposal_id, proposal_kind=d.proposal_kind, target_id=d.target_id, actor="t",
                    materialization_id=f"m-{d.decision_id}", decision_id=d.decision_id,
                    revision=d.resulting_revision, outcome="failed", failure_reason="canonical_id_mismatch"))
            return ProjectionReportV1()

        async def replay(self, decision):
            raise AssertionError("never replay an unapplied decision")

    _sync(store, journal, _Refusing(), persons, links)
    report = _sync(store, journal, _Refusing(), persons, links)
    assert (report.stuck, report.claims) == (1, 0) and not _walks(store, "c1", SAM)


def test_a_linked_cluster_the_ask_listener_missed_gets_its_node_and_claim():
    """Live 10-23: 7 typed clusters had no node because the bus event was missed."""
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world()
    links = [LinkRow("l1", SAM, "c7", "ask-c7", label="Sam", kind="person", label_ask_id="ask-c7",
                     stream_id="walkway", answered_at=NOW)]
    report = _sync(store, journal, projector, [PersonRow(SAM, "Sam")], links)
    node = store.get_node_by_id(entity_node_id_for_individual("c7"))
    assert isinstance(node, EntityNodeV1) and node.label == "Sam" and node.entity_type == "person"
    assert node.subject_ref == "vision_individual:c7"
    assert (report.clusters_written, report.waiting, report.claims) == (1, 0, 1) and _walks(store, "c7", SAM)
    assert _sync(store, journal, projector, [PersonRow(SAM, "Sam")], links).clusters_written == 0


def test_no_cluster_node_for_a_not_a_person_row_an_unnamed_or_a_copied_cluster():
    from app.person_graph_loop import LinkRow, PersonRow

    store, journal, projector = _world()
    report = _sync(store, journal, projector, [PersonRow(SAM, "Sam", "not_a_person"), PersonRow(ROBIN, "Robin")],
                   [LinkRow("l1", SAM, "c7", label="Sam", kind="person"), LinkRow("l2", ROBIN, "c8"),
                    LinkRow("l3", ROBIN, "c9", label="Robin", kind="person", label_copied=True)])
    assert report.clusters_written == 0 and report.waiting == 2
    assert store.get_node_by_id(entity_node_id_for_individual("c9")) is None
    assert store.get_node_by_id(entity_node_id_for_individual("c7")) is None
    assert store.get_node_by_id(entity_node_id_for_individual("c8")) is None
