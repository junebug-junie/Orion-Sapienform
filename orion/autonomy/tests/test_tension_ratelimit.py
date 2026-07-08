"""Task 5: rate-limit / dedup / bounded state."""
from __future__ import annotations

from orion.autonomy.tension_ratelimit import TensionRateLimiter
from orion.core.schemas.drives import TensionEventV1


def _t(kind: str = "tension.signal.v1", drives=None) -> TensionEventV1:
    return TensionEventV1(
        subject="orion", model_layer="self-model", entity_id="self:orion",
        kind=kind, magnitude=0.5, drive_impacts=drives or {"capability": 0.4},
        provenance={"intake_channel": "test"},
    )


def test_storm_capped() -> None:
    rl = TensionRateLimiter(cap=3, window_sec=60.0)
    storm = [_t() for _ in range(100)]
    kept = rl.bounded(storm, now=1000.0)
    assert len(kept) == 3


def test_window_expiry_allows_again() -> None:
    rl = TensionRateLimiter(cap=2, window_sec=60.0)
    assert len(rl.bounded([_t(), _t()], now=0.0)) == 2
    assert len(rl.bounded([_t()], now=30.0)) == 0  # still within window, over cap
    assert len(rl.bounded([_t()], now=61.0)) == 1  # window slid, allowed


def test_distinct_signatures_have_own_budget() -> None:
    rl = TensionRateLimiter(cap=1, window_sec=60.0)
    kept = rl.bounded(
        [_t(drives={"capability": 0.4}), _t(drives={"coherence": 0.5}),
         _t(kind="tension.failure.v1", drives={"capability": 0.4})],
        now=5.0,
    )
    # Three different signatures (coherence vs capability vs failure-kind).
    assert len(kept) == 3


def test_state_bounded() -> None:
    rl = TensionRateLimiter(cap=3, window_sec=60.0, max_keys=10)
    # Distinct signatures (kind varies) so the key map would grow unbounded
    # without eviction.
    for i in range(100):
        rl.bounded([_t(kind=f"tension.k{i}.v1")], now=float(i))
    assert rl.key_count() <= 10
