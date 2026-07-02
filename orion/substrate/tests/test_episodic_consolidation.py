from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.episodic_consolidation import (
    EpisodicConsolidationEvaluator,
    derive_episode_id,
)

_T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _receipt(i: int, *, offset_sec: int = 0, warnings: list[str] | None = None) -> ReductionReceiptV1:
    return ReductionReceiptV1(
        receipt_id=f"receipt:{i:04d}",
        organ_id="biometrics_pressure" if i % 2 == 0 else None,
        accepted_event_ids=[f"evt:{i}"],
        state_deltas=[
            StateDeltaV1(
                delta_id=f"delta:{i}",
                target_projection="substrate.biometrics.projection",
                target_kind="node",
                target_id=f"node:{i}",
                operation="update",
                caused_by_event_ids=[f"evt:{i}"],
                reducer_id="substrate.biometrics",
            )
        ],
        warnings=warnings or [],
        created_at=_T0 + timedelta(seconds=offset_sec),
    )


def test_window_of_receipts_yields_one_episode() -> None:
    evaluator = EpisodicConsolidationEvaluator(window_seconds=3600)
    receipts = [_receipt(i, offset_sec=i * 60) for i in range(5)]
    episode = evaluator.consolidate(receipts=receipts)
    assert episode is not None
    assert episode.status == "proposal"
    assert episode.receipt_count_total == 5
    assert episode.receipt_refs == [f"receipt:{i:04d}" for i in range(5)]
    assert episode.accepted_event_count == 5
    assert episode.state_delta_count == 5
    assert episode.reducer_counts == {"substrate.biometrics": 5}
    assert episode.organ_counts == {"biometrics_pressure": 3}
    assert episode.window_end == _T0 + timedelta(seconds=240)


def test_replay_is_idempotent() -> None:
    evaluator = EpisodicConsolidationEvaluator(window_seconds=3600)
    receipts = [_receipt(i, offset_sec=i * 60) for i in range(5)]
    first = evaluator.consolidate(receipts=receipts)
    second = evaluator.consolidate(receipts=list(reversed(receipts)))
    assert first is not None and second is not None
    assert first.episode_id == second.episode_id
    assert first == second


def test_receipt_cap_enforced() -> None:
    evaluator = EpisodicConsolidationEvaluator(window_seconds=3600, max_receipts_per_episode=3)
    receipts = [_receipt(i, offset_sec=i) for i in range(10)]
    episode = evaluator.consolidate(receipts=receipts)
    assert episode is not None
    assert len(episode.receipt_refs) == 3
    assert episode.receipt_count_total == 10
    assert episode.receipt_count_capped is True


def test_receipts_outside_window_excluded() -> None:
    evaluator = EpisodicConsolidationEvaluator(window_seconds=60)
    stale = _receipt(0, offset_sec=0)
    fresh = _receipt(1, offset_sec=3600)
    episode = evaluator.consolidate(receipts=[stale, fresh])
    assert episode is not None
    assert episode.receipt_refs == ["receipt:0001"]
    assert episode.receipt_count_total == 1


def test_empty_input_yields_none() -> None:
    evaluator = EpisodicConsolidationEvaluator()
    assert evaluator.consolidate(receipts=[]) is None


def test_warnings_sampled_and_counted() -> None:
    evaluator = EpisodicConsolidationEvaluator()
    receipts = [_receipt(i, offset_sec=i, warnings=[f"w{i}a", f"w{i}b"]) for i in range(6)]
    episode = evaluator.consolidate(receipts=receipts)
    assert episode is not None
    assert episode.warning_count == 12
    assert len(episode.sample_warnings) == 8


def test_episode_id_depends_on_inputs() -> None:
    base = derive_episode_id(receipt_ids=["a", "b"], window_start=_T0, window_end=_T0 + timedelta(hours=1))
    same = derive_episode_id(receipt_ids=["b", "a"], window_start=_T0, window_end=_T0 + timedelta(hours=1))
    other = derive_episode_id(receipt_ids=["a", "c"], window_start=_T0, window_end=_T0 + timedelta(hours=1))
    assert base == same
    assert base != other
