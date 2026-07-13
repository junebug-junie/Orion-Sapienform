from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from orion.memory.crystallization.active_packet import build_active_packet
from orion.memory.crystallization.dynamics import (
    decay,
    decayed_activation,
    recall_boost,
    reinforce,
    seed_dynamics,
    seed_weak_dynamics,
    should_retire,
)
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.retriever import retrieve_active_packet
from orion.memory.crystallization.schemas import (
    CrystallizationDynamicsV1,
    CrystallizationEvidenceRefV1,
    CrystallizationGovernanceV1,
    MemoryCrystallizationProposeRequestV1,
    MemoryCrystallizationV1,
)


def _now() -> datetime:
    return datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)


def _crys() -> MemoryCrystallizationV1:
    req = MemoryCrystallizationProposeRequestV1(
        kind="semantic",
        subject="Test subject",
        summary="Test summary",
        scope=["project:orion"],
        evidence=[CrystallizationEvidenceRefV1(source_kind="memory_card", source_id="card_1", excerpt="fact")],
        proposed_by="test",
    )
    return propose(req)


def _active_crys(
    *,
    activation: float,
    salience: float = 0.5,
    crystallization_id: str,
    summary: str = "test memory",
    formed_at: datetime | None = None,
    decay_half_life_days: float = 30.0,
) -> MemoryCrystallizationV1:
    """Build an `active`-status crystallization with real dynamics, bypassing the full
    propose/governor pipeline -- mirrors the helper pattern used in
    services/orion-recall/tests/test_active_packet_activation_floor.py.
    """
    now = formed_at if formed_at is not None else _now()
    return MemoryCrystallizationV1(
        crystallization_id=crystallization_id,
        kind="semantic",
        subject="s",
        summary=summary,
        status="active",
        confidence="likely",
        salience=salience,
        dynamics=CrystallizationDynamicsV1(
            activation=activation,
            formed_at=now,
            decay_half_life_days=decay_half_life_days,
        ),
        governance=CrystallizationGovernanceV1(proposed_by="t"),
        created_at=now,
        updated_at=now,
    )


class TestDynamicsSchema:
    def test_defaults_are_inert(self):
        dyn = CrystallizationDynamicsV1()
        assert dyn.activation == 0.0
        assert dyn.reinforcement_count == 0
        assert dyn.formed_at is None
        assert dyn.last_reinforced_at is None
        assert dyn.decay_half_life_days == 30.0
        assert dyn.retired_at is None

    def test_crystallization_has_default_dynamics(self):
        crys = _crys()
        assert isinstance(crys.dynamics, CrystallizationDynamicsV1)
        assert crys.dynamics.activation == 0.0

    def test_roundtrip_preserves_dynamics(self):
        crys = seed_dynamics(_crys(), now=_now())
        crys = reinforce(crys, now=_now())
        dumped = crys.model_dump(mode="json")
        parsed = MemoryCrystallizationV1.model_validate(dumped)
        assert parsed.dynamics.activation == crys.dynamics.activation
        assert parsed.dynamics.reinforcement_count == 1


class TestSeed:
    def test_seed_sets_activation_to_salience(self):
        crys = _crys()
        assert crys.salience > 0.0
        seeded = seed_dynamics(crys, now=_now())
        assert seeded.dynamics.activation == crys.salience
        assert seeded.dynamics.formed_at == _now()


def test_seed_weak_dynamics_scales_salience():
    crys = _crys()
    crys.salience = 0.5
    seeded = seed_weak_dynamics(crys, now=_now(), ratio=0.4)
    assert seeded.dynamics.activation == 0.2
    assert seeded.dynamics.formed_at is not None


class TestReinforce:
    def test_reinforce_moves_toward_ceiling_and_counts(self):
        crys = _crys()  # activation starts at 0.0
        r1 = reinforce(crys, now=_now(), boost=0.2)
        assert abs(r1.dynamics.activation - 0.2) < 1e-9
        assert r1.dynamics.reinforcement_count == 1
        assert r1.dynamics.last_reinforced_at == _now()

        r2 = reinforce(r1, now=_now(), boost=0.2)
        # 0.2 + (1 - 0.2) * 0.2 = 0.36 — diminishing returns, never overshoots 1.0
        assert abs(r2.dynamics.activation - 0.36) < 1e-9
        assert r2.dynamics.reinforcement_count == 2

    def test_reinforce_saturates_below_one(self):
        crys = _crys()
        for _ in range(100):
            crys = reinforce(crys, now=_now(), boost=0.5)
        assert crys.dynamics.activation <= 1.0
        assert crys.dynamics.activation > 0.99


class TestReinforceConfidence:
    def test_reinforce_climbs_confidence_tiers_as_reinforcement_count_grows(self):
        # _crys() proposes with 1 moderate-strength evidence source (default strength
        # 0.5) and reinforcement_count starts at 0 -> apply_salience's infer_confidence
        # call lands on "possible" at formation.
        crys = _crys()
        assert crys.confidence == "possible"

        r1 = reinforce(crys, now=_now())
        assert r1.dynamics.reinforcement_count == 1
        assert r1.confidence == "likely"

        r2 = reinforce(r1, now=_now())
        assert r2.dynamics.reinforcement_count == 2
        assert r2.confidence == "likely"

        r3 = reinforce(r2, now=_now())
        assert r3.dynamics.reinforcement_count == 3
        assert r3.confidence == "certain"

    def test_reinforce_never_lowers_confidence_via_stale_recompute(self):
        crys = _crys()
        r1 = reinforce(crys, now=_now())
        r2 = reinforce(r1, now=_now())
        r3 = reinforce(r2, now=_now())
        tiers = ["uncertain", "possible", "likely", "certain"]
        ranks = [tiers.index(c.confidence) for c in (crys, r1, r2, r3)]
        assert ranks == sorted(ranks)


class TestRecallBoost:
    def test_recall_is_weaker_than_reinforce_and_marks_recall(self):
        crys = _crys()
        recalled = recall_boost(crys, now=_now(), boost=0.08)
        assert abs(recalled.dynamics.activation - 0.08) < 1e-9
        assert recalled.dynamics.last_recalled_at == _now()
        assert recalled.dynamics.reinforcement_count == 0

    def test_recall_boost_never_touches_confidence(self):
        # Hard invariant: being recalled is not evidence something is true, only
        # that it's relevant. recall_boost() must never move confidence.
        crys = _crys()
        before = crys.confidence
        recalled = recall_boost(crys, now=_now(), boost=0.08)
        assert recalled.confidence == before


class TestDecay:
    def test_one_half_life_halves_activation(self):
        crys = seed_dynamics(_crys(), now=_now())
        crys.dynamics.activation = 0.8
        crys.dynamics.decay_half_life_days = 30.0
        later = _now() + timedelta(days=30)
        assert abs(decayed_activation(crys, now=later) - 0.4) < 1e-3

    def test_decay_mutates_copy(self):
        crys = seed_dynamics(_crys(), now=_now())
        crys.dynamics.activation = 0.8
        later = _now() + timedelta(days=30)
        decayed = decay(crys, now=later)
        assert decayed.dynamics.activation < 0.8
        assert crys.dynamics.activation == 0.8  # original untouched

    def test_reinforcement_resets_the_decay_clock(self):
        crys = seed_dynamics(_crys(), now=_now())
        crys.dynamics.activation = 0.8
        # reinforced right at `later` → decay measured from `later`, not formation
        later = _now() + timedelta(days=30)
        crys = reinforce(crys, now=later, boost=0.0)  # boost 0 just refreshes recency
        assert abs(decayed_activation(crys, now=later) - 0.8) < 1e-9


class TestRetirement:
    def test_stale_low_weight_retires(self):
        crys = seed_dynamics(_crys(), now=_now())
        crys.dynamics.activation = 0.1
        crys.dynamics.decay_half_life_days = 30.0
        far_future = _now() + timedelta(days=300)  # ~10 half-lives
        assert should_retire(crys, now=far_future, floor=0.05) is True

    def test_recently_reinforced_survives(self):
        crys = seed_dynamics(_crys(), now=_now())
        crys = reinforce(crys, now=_now(), boost=0.5)
        crys = reinforce(crys, now=_now(), boost=0.5)
        assert should_retire(crys, now=_now(), floor=0.05) is False


# --- Phase 4+5 wiring: recall_boost + decay at the real call sites ------------------
#
# Acceptance checks from docs/superpowers/specs/2026-07-13-memory-recall-reinforcement-
# decay-wiring-spec.md. `retrieve_active_packet` persists via `update_crystallization`,
# which is patched here rather than hitting real Postgres -- the persistence *pattern*
# (mirroring `reinforce()`'s existing call sites) is exercised in orion/memory/crystallization/
# retriever.py itself; these tests exercise the ranking behavior it's supposed to produce.


class TestRecallBoostWiring:
    @pytest.mark.asyncio
    async def test_recall_boost_wins_contested_bucket_slot(self, monkeypatch):
        """Acceptance check 1: head-to-head recall competition.

        Two crystallizations seeded with equal activation/salience. One is recalled via
        repeated `retrieve_active_packet()` calls with distinct queries; the other is never
        touched. The reinforced one must win a subsequent contested ranking.
        """
        store: dict[str, MemoryCrystallizationV1] = {
            "crys_recalled": _active_crys(activation=0.3, salience=0.5, crystallization_id="crys_recalled"),
            "crys_untouched": _active_crys(activation=0.3, salience=0.5, crystallization_id="crys_untouched"),
        }

        async def _fetch(pool, cid):
            return store.get(cid)

        async def _persist(pool, crystallization):
            store[crystallization.crystallization_id] = crystallization

        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.get_crystallization",
            AsyncMock(side_effect=_fetch),
        )
        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.update_crystallization",
            AsyncMock(side_effect=_persist),
        )

        pool = object()  # non-None sentinel; the real call is mocked, so no live DB needed
        for query in ("first distinct query", "second distinct query", "third distinct query"):
            await retrieve_active_packet(
                query=query,
                crystallizations=[store["crys_recalled"]],
                pool=pool,
            )

        assert store["crys_recalled"].dynamics.activation > 0.3
        assert store["crys_untouched"].dynamics.activation == 0.3
        # Untouched item's other dynamics fields never moved.
        assert store["crys_untouched"].dynamics.last_recalled_at is None

        packet = await retrieve_active_packet(
            query="ambiguous query naming both",
            crystallizations=[store["crys_recalled"], store["crys_untouched"]],
            pool=pool,
        )
        assert packet.crystallization_refs[0] == "crys_recalled"

    @pytest.mark.asyncio
    async def test_recall_boost_only_applies_to_items_actually_in_refs(self, monkeypatch):
        """Only what made it into `packet.crystallization_refs` gets boosted -- not every
        candidate considered. An ineligible (below-floor) candidate must not be touched."""
        eligible = _active_crys(activation=0.3, salience=0.5, crystallization_id="crys_eligible")
        ineligible = _active_crys(activation=0.01, salience=0.5, crystallization_id="crys_ineligible")

        fetch_mock = AsyncMock(return_value=eligible)
        update_mock = AsyncMock()
        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.get_crystallization",
            fetch_mock,
        )
        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.update_crystallization",
            update_mock,
        )

        packet = await retrieve_active_packet(
            query="test",
            crystallizations=[eligible, ineligible],
            pool=object(),
        )

        assert packet.crystallization_refs == ["crys_eligible"]
        update_mock.assert_called_once()
        (_, persisted), _kwargs = update_mock.call_args
        assert persisted.crystallization_id == "crys_eligible"

    @pytest.mark.asyncio
    async def test_recall_boost_skips_persistence_when_pool_absent(self):
        """Pool defaults to None (offline/test callers) -- must degrade gracefully, never raise."""
        crys = _active_crys(activation=0.3, salience=0.5, crystallization_id="crys_no_pool")
        packet = await retrieve_active_packet(query="test", crystallizations=[crys])
        assert packet.crystallization_refs == ["crys_no_pool"]

    @pytest.mark.asyncio
    async def test_recall_boost_persistence_failure_does_not_raise(self, monkeypatch):
        """A DB error during persistence must not break the retrieval path."""
        crys = _active_crys(activation=0.3, salience=0.5, crystallization_id="crys_persist_fail")
        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.get_crystallization",
            AsyncMock(return_value=crys),
        )
        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.update_crystallization",
            AsyncMock(side_effect=RuntimeError("db down")),
        )
        packet = await retrieve_active_packet(query="test", crystallizations=[crys], pool=object())
        assert packet.crystallization_refs == ["crys_persist_fail"]

    @pytest.mark.asyncio
    async def test_recall_boost_refetches_fresh_row_before_persisting(self, monkeypatch):
        """The in-memory snapshot passed via `crystallizations=` can be stale by the time
        persistence runs (embed/chroma/graphiti round trips happen first). Persisting a
        stale snapshot through `update_crystallization`'s full-row UPDATE would clobber a
        concurrent governance write -- `_apply_recall_boost` must refetch the freshest row
        via `get_crystallization()` immediately before boosting+persisting instead."""
        stale_snapshot = _active_crys(activation=0.3, salience=0.5, crystallization_id="crys_race")
        fresh_row = stale_snapshot.model_copy(deep=True)
        fresh_row.confidence = "certain"  # simulates a concurrent governance write
        fresh_row.dynamics.activation = 0.7  # simulates a concurrent reinforcement

        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.get_crystallization",
            AsyncMock(return_value=fresh_row),
        )
        update_mock = AsyncMock()
        monkeypatch.setattr(
            "orion.memory.crystallization.retriever.update_crystallization",
            update_mock,
        )

        await retrieve_active_packet(query="test", crystallizations=[stale_snapshot], pool=object())

        update_mock.assert_called_once()
        (_, persisted), _kwargs = update_mock.call_args
        # Persisted row is boosted from the FRESH activation (0.7), not the stale
        # snapshot's (0.3), and preserves the concurrent confidence change.
        assert persisted.dynamics.activation > 0.7
        assert persisted.confidence == "certain"


class TestDecayAtRankingReadSite:
    def test_decayed_activation_measurably_lower_than_stored_value(self):
        formed = _now()
        stale = _active_crys(
            activation=0.6, salience=0.5, crystallization_id="crys_stale",
            formed_at=formed, decay_half_life_days=1.0,
        )
        later = formed + timedelta(days=5)  # 5 half-lives
        stored_value = stale.dynamics.activation
        decayed_value = decayed_activation(stale, now=later)
        assert decayed_value < stored_value - 0.3  # measurably lower, not a rounding wobble

    def test_disused_item_loses_contested_slot_to_recently_recalled_competitor(self):
        """Acceptance check 2: decay actually reduces rank at the ranking read site
        (active_packet.py), not just in isolation."""
        formed = _now()
        stale = _active_crys(
            activation=0.6, salience=0.5, crystallization_id="crys_stale",
            formed_at=formed, decay_half_life_days=1.0,
        )
        fresh = _active_crys(
            activation=0.6, salience=0.5, crystallization_id="crys_fresh",
            formed_at=formed, decay_half_life_days=1.0,
        )
        later = formed + timedelta(days=5)
        # Simulate `fresh` having been recalled moments before ranking time -- resets its
        # decay reference clock (same mechanic `test_reinforcement_resets_the_decay_clock`
        # already proves for reinforce()).
        fresh.dynamics.last_recalled_at = later - timedelta(minutes=1)

        packet = build_active_packet(
            query="test", crystallizations=[stale, fresh], now=later,
        )
        assert packet.crystallization_refs[0] == "crys_fresh"
        assert packet.crystallization_refs[1] == "crys_stale"

    def test_build_active_packet_without_now_still_ranks_correctly(self):
        """No-regression check: `now` is optional, defaults to current time, and freshly
        formed/equal-salience items with unequal activation still rank as before."""
        low = _active_crys(activation=0.2, salience=0.5, crystallization_id="crys_low")
        high = _active_crys(activation=0.9, salience=0.5, crystallization_id="crys_high")
        packet = build_active_packet(query="test", crystallizations=[low, high])
        assert packet.crystallization_refs[0] == "crys_high"


class TestRecallBoostDecayInvariant:
    """Acceptance check 3 (hard invariant): recall_boost()/decay() move `activation`
    (and recall bookkeeping fields) only. `confidence`, `salience`, and every other
    non-dynamics field must be provably unchanged, in any call order."""

    def test_confidence_and_salience_unchanged_across_all_orderings(self):
        base = _active_crys(activation=0.3, salience=0.37, crystallization_id="crys_invariant")
        base.confidence = "likely"
        base.dynamics.decay_half_life_days = 10.0

        now = _now()
        op_specs = [
            ("recall", now + timedelta(hours=1)),
            ("decay", now + timedelta(hours=2)),
            ("recall", now + timedelta(hours=3)),
            ("decay", now + timedelta(hours=4)),
        ]

        checked_any = False
        for order in itertools.permutations(range(len(op_specs))):
            state = base.model_copy(deep=True)
            for idx in order:
                kind, ts = op_specs[idx]
                state = recall_boost(state, now=ts) if kind == "recall" else decay(state, now=ts)
                assert state.confidence == base.confidence
                assert state.salience == base.salience
                assert state.summary == base.summary
                assert state.subject == base.subject
                assert state.status == base.status
                assert state.kind == base.kind
                checked_any = True
        assert checked_any
