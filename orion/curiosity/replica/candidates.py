"""The replica's candidate list: the fresh live priors Hub may put at the top of this run's offer.

Design section 2. The full live pool (246 of 277 priors live on 2026-11-04) does not fit a short frame, so Hub
sends at most ``MAX_CANDIDATES`` (60), de-duplicated, in this priority:

1. the ``UNCERTAIN_N`` (30) most uncertain -- the uncertainty arm's own sort (``Prior.uncertainty``, then
   times tested, then the per-run rotation);
2. the ``NATS_N`` (20) with the highest expected nats under the current yield model -- the value arm's sort;
3. every prior tested in the last ``RECENT_DAYS`` (7) days;
4. ``ROTATED_N`` (10) more, drawn from the rest by the per-run rotation.

Stale priors (tested ``stale_after`` times) are never candidates: the stale bucket is shown to the run anyway.

The list is then put in a NEUTRAL order, the per-run rotation, not either rule's order: the replica must
not be handed the uncertainty order as its starting point (it would then mostly copy the arm it has to beat).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Mapping, Optional, Sequence

from orion.curiosity.value import PriorTestRecord, valid_confidence
from orion.curiosity.worldview import EXPECTED_NATS_DECIMALS, Prior, _rotation_key, _stamp_ms
from orion.schemas.curiosity_replica import CLAIM_MAX, MAX_CANDIDATES, CuriosityOfferCandidateV1

UNCERTAIN_N = 30
NATS_N = 20
RECENT_DAYS = 7.0
ROTATED_N = 10
# A test stamp more than this far back (or in the future) is unreadable, not "very old": Orion hand-writes
# last_tested_at and the live graph holds ISO strings, epoch ints and partly typed values ("2026-11-01T19:4xZ").
_MAX_AGE_H = 24.0 * 3650


def last_tested_age_h(prior: Prior, now: datetime) -> Optional[float]:
    """Hours since the prior's last test; None when never tested or the stamp is unreadable."""
    if prior.times_tested <= 0:
        return None
    ms = _stamp_ms(prior.last_tested_at)
    if ms is None:
        return None
    if 0 < ms < 100_000_000_000:  # epoch seconds, not ms (a 2026 stamp in ms is ~1.79e12)
        ms *= 1000
    age = (now.timestamp() * 1000.0 - ms) / 3_600_000.0
    return age if 0.0 <= age <= _MAX_AGE_H else None


def last_moves(history: Iterable[PriorTestRecord]) -> dict[str, float]:
    """prior_id -> confidence change of its newest scored test (``history`` oldest first, as
    ``load_prior_test_history`` returns it)."""
    out: dict[str, float] = {}
    for rec in history:
        out[rec.prior_id] = max(-1.0, min(1.0, float(rec.after) - float(rec.before)))
    return out


def _clip_claim(text: str) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= CLAIM_MAX else value[: CLAIM_MAX - 1].rstrip() + "…"


def build_candidates(
    fresh: Sequence[Prior],
    *,
    now: datetime,
    rotate_seed: str,
    expected_nats_for: Callable[[Prior], float],
    moves: Mapping[str, float],
) -> list[CuriosityOfferCandidateV1]:
    """``fresh``: every fresh (not stale) live prior from the read Hub is about to show
    (``orion.curiosity.worldview.fresh_priors``)."""
    rot = {p.prior_id: _rotation_key(p.prior_id, rotate_seed) for p in fresh}
    by_uncertainty = sorted(fresh, key=lambda p: (p.uncertainty, p.times_tested, rot[p.prior_id]))
    by_nats = sorted(fresh, key=lambda p: (-round(expected_nats_for(p), EXPECTED_NATS_DECIMALS), p.uncertainty,
                                           p.times_tested, rot[p.prior_id]))
    ages = {p.prior_id: last_tested_age_h(p, now) for p in fresh}
    recent = [p for p in sorted(fresh, key=lambda p: rot[p.prior_id])
              if ages[p.prior_id] is not None and ages[p.prior_id] <= RECENT_DAYS * 24.0]
    chosen: list[Prior] = []
    seen: set[str] = set()

    def take(priors: Iterable[Prior], n: Optional[int] = None) -> None:
        added = 0
        for p in priors:
            if n is not None and added >= n:
                return
            if p.prior_id in seen:
                continue
            seen.add(p.prior_id)
            chosen.append(p)
            added += 1

    take(by_uncertainty, UNCERTAIN_N)
    take(by_nats, NATS_N)
    take(recent)
    take(sorted(fresh, key=lambda p: rot[p.prior_id]), ROTATED_N)
    chosen = chosen[:MAX_CANDIDATES]
    chosen.sort(key=lambda p: rot[p.prior_id])  # neutral presentation order (module docstring)
    return [
        CuriosityOfferCandidateV1(
            prior_id=p.prior_id,
            claim=_clip_claim(p.claim),
            confidence=valid_confidence(p.confidence),
            times_tested=max(0, p.times_tested),
            last_tested_age_h=ages[p.prior_id],
            last_move=moves.get(p.prior_id) if p.times_tested > 0 else None,
        )
        for p in chosen
    ]


__all__ = ["NATS_N", "RECENT_DAYS", "ROTATED_N", "UNCERTAIN_N", "build_candidates", "last_moves", "last_tested_age_h"]
