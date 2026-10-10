"""In-process fakes for the reading-assertion path (tests + evals).

``FakeJournal`` mirrors ``orion.substrate.graph_journal.SubstrateGraphJournal``'s contract:
append is idempotent on event id, a second decision for the same (target, revision)
raises ``RevisionConflict``, ``pending_decisions`` lists unapplied decisions (optionally
only those whose proposal came from ``proposal_actors``). The real SQL is covered by
orion/substrate/tests/test_graph_journal_pg.py against a throwaway Postgres.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_graph_journal import (
    SubstrateGraphDecisionV1,
    SubstrateGraphMaterializationV1,
    SubstrateGraphProposalV1,
)
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
)
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate
from orion.substrate.graph_journal import TERMINAL_FAILURE_REASONS, RevisionConflict
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
SOURCE_URL = "https://example.org/heat-pumps"
FETCH_TEXT = (
    "Heat pumps explained. A heat pump transfers heat using a refrigeration cycle, "
    "moving thermal energy from a cold space to a warm one. Unlike a furnace, it does "
    "not burn fuel."
)
OBJECT_ID = "concept-refrigeration-cycle"
UNRELATED_ID = "concept-furnace"


class FakeJournal:
    def __init__(self) -> None:
        self.events: dict[str, Any] = {}
        self.order: list[str] = []
        self.decision_slots: set[tuple[str, int]] = set()

    async def append(self, event) -> bool:
        key = f"{event.event_kind}:{event.event_id}"
        if key in self.events:
            return False
        if isinstance(event, SubstrateGraphDecisionV1):
            slot = (event.target_id, event.resulting_revision)
            if slot in self.decision_slots:
                raise RevisionConflict(f"target {event.target_id} revision {event.resulting_revision}")
            self.decision_slots.add(slot)
        self.events[key] = event
        self.order.append(key)
        return True

    def _of(self, cls) -> list:
        return [self.events[k] for k in self.order if isinstance(self.events[k], cls)]

    def proposals(self) -> list[SubstrateGraphProposalV1]:
        return self._of(SubstrateGraphProposalV1)

    def decisions(self) -> list[SubstrateGraphDecisionV1]:
        return self._of(SubstrateGraphDecisionV1)

    def materializations(self) -> list[SubstrateGraphMaterializationV1]:
        return self._of(SubstrateGraphMaterializationV1)

    async def pending_decisions(self, *, limit: int = 100, proposal_actors=None) -> list:
        applied = {m.decision_id for m in self.materializations() if m.outcome == "applied"}
        out = []
        for d in self.decisions():
            if d.decision_id in applied:
                continue
            last = await self.last_materialization(d.decision_id)
            if last is not None and last.failure_reason and last.failure_reason.split(":")[0] in TERMINAL_FAILURE_REASONS:
                continue
            if d.expected_prior_revision > await self.latest_applied_revision(d.target_id):
                continue
            if proposal_actors is not None:
                proposal = await self.proposal(d.proposal_id)
                if proposal is None or proposal.actor not in proposal_actors:
                    continue
            out.append(d)
        return out[:limit]

    async def proposal(self, proposal_id: str):
        return next((p for p in self.proposals() if p.proposal_id == proposal_id), None)

    async def latest_applied_revision(self, target_id: str) -> int:
        return max([m.revision for m in self.materializations()
                    if m.outcome == "applied" and m.target_id == target_id] or [0])

    async def failed_attempts(self, decision_id: str) -> int:
        return sum(1 for m in self.materializations() if m.outcome == "failed" and m.decision_id == decision_id)

    async def last_materialization(self, decision_id: str):
        rows = [m for m in self.materializations() if m.decision_id == decision_id]
        return rows[-1] if rows else None


class FakeSnapshotConn:
    """``reading_document_snapshot`` as a dict: the two statements fetch_text.py issues."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, str]] = {}
        self.fail = False

    async def execute(self, sql: str, sha: str, text: str, chars: int, source: str) -> str:
        if self.fail:
            raise ConnectionError("down")
        assert "reading_document_snapshot" in sql and chars == len(text)
        self.rows.setdefault(sha, (text, source))
        return "INSERT 0 1"

    async def fetch(self, sql: str, shas: list[str]) -> list[dict]:
        return [{"sha256": s, "content": self.rows[s][0]} for s in shas if s in self.rows]


class FakePool:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def existing_concept(node_id: str, label: str) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id, anchor_scope="orion", subject_ref="world_pulse", label=label,
        temporal=SubstrateTemporalWindowV1(observed_at=NOW),
        provenance=SubstrateProvenanceV1(authority="local_inferred", source_kind="topic_foundry",
                                         source_channel="test", producer="topic_foundry_adapter"),
        signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.6),
    )


def handoff(*, sha: str | None, labels=("Heat pump",)) -> WorldPulseReadHandoffV1:
    evidence = SourceFetchEvidenceV1(url=SOURCE_URL, tool_name="WebFetch", content_chars=len(FETCH_TEXT),
                                     content_sha256=sha)
    return WorldPulseReadHandoffV1(
        seed_ref=WorldPulseReadSeedV1(seed_id="finding:heat-pump", kind="finding", run_id="run-1",
                                      url=SOURCE_URL, title="Heat pumps explained"),
        what_i_learned="Heat pumps move heat with a refrigeration cycle rather than burning fuel.",
        concept_candidates=[WorldPulseReadConceptCandidateV1(label=label) for label in labels],
        trace_id="trace-heat-pump", created_at=NOW, read_evidence=[evidence],
    )


def seeded_store(h: WorldPulseReadHandoffV1) -> tuple[InMemorySubstrateGraphStore, str]:
    """The atlas before Stage 2: two existing concepts, plus this read's Stage 1 concept
    written by the real Stage 1 path (adapter + materializer). Returns the stored subject id."""
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1

    materializer.apply_record(SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="world_pulse", nodes=[
        existing_concept(OBJECT_ID, "Refrigeration cycle"), existing_concept(UNRELATED_ID, "Furnace")]))
    result = materializer.apply_record(map_world_pulse_read_handoff_to_substrate(h))
    return store, result.node_decisions[0].canonical_node_id
