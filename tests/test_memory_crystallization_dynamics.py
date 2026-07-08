from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from orion.memory.crystallization.schemas import (
    CrystallizationDynamicsV1,
    CrystallizationEvidenceRefV1,
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


class TestRecallBoost:
    def test_recall_is_weaker_than_reinforce_and_marks_recall(self):
        crys = _crys()
        recalled = recall_boost(crys, now=_now(), boost=0.08)
        assert abs(recalled.dynamics.activation - 0.08) < 1e-9
        assert recalled.dynamics.last_recalled_at == _now()
        assert recalled.dynamics.reinforcement_count == 0


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
