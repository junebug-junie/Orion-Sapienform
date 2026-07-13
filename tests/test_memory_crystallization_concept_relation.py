from __future__ import annotations

from uuid import uuid4

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from orion.memory.crystallization.concept_relation import (
    ConceptRelationDecision,
    maybe_resolve_concept_relation,
    merge_new_evidence,
    resolve_concept_relation,
)
from orion.memory.crystallization.governor import approve
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.repository import insert_concept_relation_decision
from orion.memory.crystallization.schemas import (
    CrystallizationEvidenceRefV1,
    MemoryCrystallizationProposeRequestV1,
    MemoryCrystallizationV1,
)


def _base_request(**kwargs) -> MemoryCrystallizationProposeRequestV1:
    defaults = {
        "kind": "semantic",
        "subject": "Test subject",
        "summary": "Test summary",
        "scope": ["project:orion"],
        "evidence": [
            CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="card_123", excerpt="fact")
        ],
        "proposed_by": "test",
    }
    defaults.update(kwargs)
    return MemoryCrystallizationProposeRequestV1(**defaults)


def _active_crystallization(**kwargs) -> MemoryCrystallizationV1:
    req = _base_request(**kwargs)
    crys = propose(req)
    crys.governance.approved_by = "operator"
    crys.governance.validation_status = "valid"
    updated, _ = approve(crys, actor="operator")
    return updated


class _Settings:
    SERVICE_NAME = "orion-memory-consolidation"
    SERVICE_VERSION = "0.1.0"
    NODE_NAME = "test"
    CHANNEL_LLM_INTAKE = "orion:exec:request:LLMGatewayService"
    TURN_CHANGE_CLASSIFY_ROUTE = "metacog"
    CONCEPT_RELATION_TIMEOUT_SEC = 8.0
    CONCEPT_RELATION_CONFIDENCE_FLOOR = 0.6
    CONCEPT_RELATION_CANDIDATE_LIMIT = 5
    CRYSTALLIZER_EMBED_HOST_URL = "http://embed.local"
    CHROMA_HOST = "chroma.local"
    CHROMA_PORT = 8000
    CRYSTALLIZER_VECTOR_COLLECTION = "orion_memory_crystallizations"
    CRYSTALLIZER_EMBED_TIMEOUT_MS = 8000


class TestConceptRelationDecisionSchema:
    def test_same_requires_target(self):
        with pytest.raises(ValidationError):
            ConceptRelationDecision(relation="same", confidence=0.7)

    def test_refines_requires_target(self):
        with pytest.raises(ValidationError):
            ConceptRelationDecision(relation="refines", confidence=0.7)

    def test_contradicts_requires_target(self):
        with pytest.raises(ValidationError):
            ConceptRelationDecision(relation="contradicts", confidence=0.7)

    def test_unrelated_does_not_require_target(self):
        decision = ConceptRelationDecision(relation="unrelated", confidence=0.0)
        assert decision.target_crystallization_id is None

    def test_valid_same_with_target(self):
        decision = ConceptRelationDecision(relation="same", target_crystallization_id="crys_x", confidence=0.8)
        assert decision.target_crystallization_id == "crys_x"


class TestResolveConceptRelation:
    @pytest.mark.asyncio
    async def test_empty_candidates_short_circuits(self):
        candidate = _active_crystallization()
        bus = MagicMock()
        bus.rpc_request = AsyncMock()

        decision = await resolve_concept_relation(
            bus, candidate=candidate, similar_existing=[], settings=_Settings()
        )

        assert decision.relation == "unrelated"
        assert decision.confidence == 0.0
        bus.rpc_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_parses_decision(self):
        candidate = _active_crystallization()
        neighbor = _active_crystallization()
        neighbor.crystallization_id = "crys_x"

        bus = MagicMock()
        bus.rpc_request = AsyncMock(return_value={"data": b"whatever"})
        decoded = MagicMock()
        decoded.ok = True
        decoded.envelope.payload = {
            "content": '{"relation": "same", "target_crystallization_id": "crys_x", "confidence": 0.8}'
        }
        bus.codec.decode = MagicMock(return_value=decoded)

        decision = await resolve_concept_relation(
            bus, candidate=candidate, similar_existing=[neighbor], settings=_Settings()
        )

        assert decision.relation == "same"
        assert decision.target_crystallization_id == "crys_x"
        assert decision.confidence == 0.8
        bus.rpc_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_malformed_json_degrades_to_unrelated(self):
        candidate = _active_crystallization()
        neighbor = _active_crystallization()
        neighbor.crystallization_id = "crys_x"

        bus = MagicMock()
        bus.rpc_request = AsyncMock(return_value={"data": b"whatever"})
        decoded = MagicMock()
        decoded.ok = True
        decoded.envelope.payload = {"content": "not json at all, no braces here"}
        bus.codec.decode = MagicMock(return_value=decoded)

        decision = await resolve_concept_relation(
            bus, candidate=candidate, similar_existing=[neighbor], settings=_Settings()
        )

        assert decision.relation == "unrelated"
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_rpc_request_raises_degrades_to_unrelated(self):
        candidate = _active_crystallization()
        neighbor = _active_crystallization()
        neighbor.crystallization_id = "crys_x"

        bus = MagicMock()
        bus.rpc_request = AsyncMock(side_effect=RuntimeError("timeout"))

        decision = await resolve_concept_relation(
            bus, candidate=candidate, similar_existing=[neighbor], settings=_Settings()
        )

        assert decision.relation == "unrelated"
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_extra_field_in_llm_response_is_ignored(self):
        # Regression: extra="forbid" used to reject the entire decision when a small
        # instruct model added a stray field (e.g. "reasoning") despite prompt
        # instructions to emit only the bare object -- extra="ignore" tolerates it.
        candidate = _active_crystallization()
        neighbor = _active_crystallization()
        neighbor.crystallization_id = "crys_x"

        bus = MagicMock()
        bus.rpc_request = AsyncMock(return_value={"data": b"whatever"})
        decoded = MagicMock()
        decoded.ok = True
        decoded.envelope.payload = {
            "content": (
                '{"reasoning": "because they match", "relation": "same", '
                '"target_crystallization_id": "crys_x", "confidence": 0.8}'
            )
        }
        bus.codec.decode = MagicMock(return_value=decoded)

        decision = await resolve_concept_relation(
            bus, candidate=candidate, similar_existing=[neighbor], settings=_Settings()
        )

        assert decision.relation == "same"
        assert decision.target_crystallization_id == "crys_x"
        assert decision.confidence == 0.8

    @pytest.mark.asyncio
    async def test_decode_not_ok_degrades_to_unrelated(self):
        candidate = _active_crystallization()
        neighbor = _active_crystallization()
        neighbor.crystallization_id = "crys_x"

        bus = MagicMock()
        bus.rpc_request = AsyncMock(return_value={"data": b"whatever"})
        decoded = MagicMock()
        decoded.ok = False
        decoded.error = "boom"
        bus.codec.decode = MagicMock(return_value=decoded)

        decision = await resolve_concept_relation(
            bus, candidate=candidate, similar_existing=[neighbor], settings=_Settings()
        )

        assert decision.relation == "unrelated"
        assert decision.confidence == 0.0


class TestMaybeResolveConceptRelation:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_none(self):
        candidate = _active_crystallization()
        pool = MagicMock()
        bus = MagicMock()

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[]),
        ) as mock_fetch, patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(),
        ) as mock_resolve:
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        mock_fetch.assert_called_once()
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_same_relation_reinforces_target(self):
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        decision = ConceptRelationDecision(
            relation="same", target_crystallization_id="crys_target", confidence=0.9
        )

        update_mock = AsyncMock()
        emit_mock = AsyncMock(return_value=True)

        decision_log_mock = AsyncMock(return_value="decision-1")

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.update_crystallization",
            new=update_mock,
        ), patch(
            "orion.memory.crystallization.concept_relation.emit_crystallization_lifecycle",
            new=emit_mock,
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=decision_log_mock,
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={"service_name": "x"}
            )

        assert result is not None
        cid, row, outcome = result
        assert cid == "crys_target"
        assert outcome == "reinforced_by_relation"
        update_mock.assert_called_once()
        emit_mock.assert_called_once()
        decision_log_mock.assert_called_once()
        assert decision_log_mock.call_args.kwargs["relation"] == "same"
        assert decision_log_mock.call_args.kwargs["floor_cleared"] is True

    @pytest.mark.asyncio
    async def test_refines_attaches_link_without_returning(self):
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        decision = ConceptRelationDecision(
            relation="refines", target_crystallization_id="crys_target", confidence=0.9
        )

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=AsyncMock(return_value="decision-1"),
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        assert len(candidate.links) == 1
        link = candidate.links[0]
        assert link.relation == "supersedes"
        assert link.target_crystallization_id == "crys_target"

    @pytest.mark.asyncio
    async def test_contradicts_attaches_link_with_contradicts_relation(self):
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        decision = ConceptRelationDecision(
            relation="contradicts", target_crystallization_id="crys_target", confidence=0.9
        )

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=AsyncMock(return_value="decision-1"),
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        assert len(candidate.links) == 1
        assert candidate.links[0].relation == "contradicts"

    @pytest.mark.asyncio
    async def test_confidence_below_floor_returns_none(self):
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        decision = ConceptRelationDecision(
            relation="same", target_crystallization_id="crys_target", confidence=0.3
        )

        decision_log_mock = AsyncMock(return_value="decision-1")

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=decision_log_mock,
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        assert candidate.links == []

        # Regression: previously this sub-floor "same" decision vanished silently --
        # only the decisive outcome (past the floor filter below) ever reached a log
        # line. It must still be recorded so scripts/concept_relation_digest.py's
        # threshold-tuning report can surface it as a near-miss.
        decision_log_mock.assert_called_once()
        assert decision_log_mock.call_args.kwargs["relation"] == "same"
        assert decision_log_mock.call_args.kwargs["confidence"] == 0.3
        assert decision_log_mock.call_args.kwargs["floor_cleared"] is False

    @pytest.mark.asyncio
    async def test_unrelated_decision_is_logged_even_though_it_returns_none(self):
        # Regression: an "unrelated" decision (the common case -- resolve_concept_relation
        # degrades to this on any LLM failure, and it's the model's own explicit choice
        # otherwise) previously never reached the logger.info line at all, since that line
        # sits after this exact filter. Every one of these vanished silently.
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        decision = ConceptRelationDecision(relation="unrelated", confidence=0.0)

        decision_log_mock = AsyncMock(return_value="decision-1")

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=decision_log_mock,
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        decision_log_mock.assert_called_once()
        assert decision_log_mock.call_args.kwargs["relation"] == "unrelated"
        assert decision_log_mock.call_args.kwargs["target_crystallization_id"] is None
        assert decision_log_mock.call_args.kwargs["floor_cleared"] is False

    @pytest.mark.asyncio
    async def test_subfloor_contradicts_is_logged_even_though_it_returns_none(self):
        # Regression: a "contradicts" decision below the confidence floor previously
        # vanished silently too (same filter, same missing log line) -- not just "same".
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        decision = ConceptRelationDecision(
            relation="contradicts", target_crystallization_id="crys_target", confidence=0.45
        )

        decision_log_mock = AsyncMock(return_value="decision-1")

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=decision_log_mock,
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        assert candidate.links == []
        decision_log_mock.assert_called_once()
        assert decision_log_mock.call_args.kwargs["relation"] == "contradicts"
        assert decision_log_mock.call_args.kwargs["target_crystallization_id"] == "crys_target"
        assert decision_log_mock.call_args.kwargs["floor_cleared"] is False

    @pytest.mark.asyncio
    async def test_confidence_floor_zero_is_respected_not_clobbered(self):
        # Regression: `getattr(settings, "CONCEPT_RELATION_CONFIDENCE_FLOOR", 0.6) or 0.6`
        # used to silently clobber an explicit 0.0 back to the 0.6 default (falsy-zero
        # bug). CONCEPT_RELATION_CONFIDENCE_FLOOR=0.0 means "accept every decision".
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        settings = _Settings()
        settings.CONCEPT_RELATION_CONFIDENCE_FLOOR = 0.0

        decision = ConceptRelationDecision(
            relation="same", target_crystallization_id="crys_target", confidence=0.01
        )

        update_mock = AsyncMock()
        emit_mock = AsyncMock(return_value=True)

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.update_crystallization",
            new=update_mock,
        ), patch(
            "orion.memory.crystallization.concept_relation.emit_crystallization_lifecycle",
            new=emit_mock,
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=AsyncMock(return_value="decision-1"),
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=settings, emit_kw={}
            )

        # confidence 0.01 >= floor 0.0 -- must be accepted, not silently re-floored to 0.6.
        assert result is not None

    @pytest.mark.asyncio
    async def test_unseen_target_id_returns_none(self):
        candidate = _active_crystallization()
        target = _active_crystallization()
        target.crystallization_id = "crys_target"

        pool = MagicMock()
        bus = MagicMock()

        # LLM referenced an id outside the candidate set it was actually given.
        decision = ConceptRelationDecision(
            relation="same", target_crystallization_id="crys_not_in_candidates", confidence=0.9
        )

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[target]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(return_value=decision),
        ), patch(
            "orion.memory.crystallization.concept_relation.insert_concept_relation_decision",
            new=AsyncMock(return_value="decision-1"),
        ):
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        assert candidate.links == []

    @pytest.mark.asyncio
    async def test_cross_kind_candidate_filtered_before_llm_call(self):
        # Candidate retrieval is scope-free by design, but must still respect the kind
        # boundary detect_duplicates() enforces -- a "stance" should never be judged
        # "same as" a "semantic" fact just because the text embeds nearby.
        candidate = _active_crystallization(kind="semantic")
        wrong_kind_neighbor = _active_crystallization(
            kind="stance", planning_effects=["x"], retrieval_affordances=["y"]
        )
        wrong_kind_neighbor.crystallization_id = "crys_wrong_kind"

        pool = MagicMock()
        bus = MagicMock()

        with patch(
            "orion.memory.crystallization.concept_relation.fetch_similar_candidates",
            new=AsyncMock(return_value=[wrong_kind_neighbor]),
        ), patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
            new=AsyncMock(),
        ) as mock_resolve:
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=_Settings(), emit_kw={}
            )

        assert result is None
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_embed_host_short_circuits(self):
        # fetch_similar_candidates() is the single source of truth for the empty-host
        # guard (candidate_retrieval.py:32) -- maybe_resolve_concept_relation no longer
        # duplicates that check, so this exercises the real function end-to-end rather
        # than asserting an internal call count.
        candidate = _active_crystallization()
        pool = MagicMock()
        bus = MagicMock()

        settings = _Settings()
        settings.CRYSTALLIZER_EMBED_HOST_URL = ""

        with patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
        ) as mock_resolve:
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=settings, emit_kw={}
            )

        assert result is None
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_chroma_host_short_circuits(self):
        candidate = _active_crystallization()
        pool = MagicMock()
        bus = MagicMock()

        settings = _Settings()
        settings.CHROMA_HOST = ""

        with patch(
            "orion.memory.crystallization.concept_relation.resolve_concept_relation",
        ) as mock_resolve:
            result = await maybe_resolve_concept_relation(
                pool, bus, candidate=candidate, settings=settings, emit_kw={}
            )

        assert result is None
        mock_resolve.assert_not_called()


class _FakeDecisionConn:
    """Minimal double for the single INSERT ... RETURNING decision_id query that
    insert_concept_relation_decision() issues -- proves the write round-trips (the
    params passed in come back out as the row read from the "database") without a
    live Postgres, matching this repo's existing convention of mocking at the
    pool/connection boundary."""

    def __init__(self) -> None:
        self.captured_sql: str | None = None
        self.captured_args: tuple | None = None

    async def fetchrow(self, sql, *args):
        self.captured_sql = sql
        self.captured_args = args
        return {"decision_id": str(uuid4())}


class _FakePool:
    def __init__(self, conn: _FakeDecisionConn) -> None:
        self._conn = conn

    def acquire(self) -> "_FakePool":
        return self

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


class TestInsertConceptRelationDecision:
    @pytest.mark.asyncio
    async def test_round_trips_all_fields(self):
        conn = _FakeDecisionConn()
        pool = _FakePool(conn)

        decision_id = await insert_concept_relation_decision(
            pool,
            candidate_crystallization_id="crys_candidate",
            target_crystallization_id="crys_target",
            relation="refines",
            confidence=0.73,
            floor_cleared=True,
        )

        assert decision_id  # a real id came back from the "INSERT ... RETURNING"
        assert conn.captured_args == ("crys_candidate", "crys_target", "refines", 0.73, True)
        assert "INSERT INTO memory_concept_relation_decisions" in conn.captured_sql
        assert "RETURNING decision_id" in conn.captured_sql

    @pytest.mark.asyncio
    async def test_round_trips_unrelated_with_null_target(self):
        # "unrelated" decisions never have a target_crystallization_id -- confirm the
        # write path tolerates None and it comes back through unchanged.
        conn = _FakeDecisionConn()
        pool = _FakePool(conn)

        await insert_concept_relation_decision(
            pool,
            candidate_crystallization_id="crys_candidate",
            target_crystallization_id=None,
            relation="unrelated",
            confidence=0.0,
            floor_cleared=False,
        )

        assert conn.captured_args == ("crys_candidate", None, "unrelated", 0.0, False)


class TestMergeNewEvidence:
    def test_merges_without_duplicating(self):
        target = _active_crystallization()
        candidate = _active_crystallization()
        candidate.evidence.append(
            CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-new", strength=0.6)
        )

        merged = merge_new_evidence(target, candidate)

        assert any(ev.source_id == "corr-new" for ev in merged.evidence)
        assert sum(1 for ev in merged.evidence if ev.source_id == "card_123") == 1
