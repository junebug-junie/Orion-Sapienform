"""Organ 3b part 2: learned family weights inside each tier (body economy Patch C1, SHADOW).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, "Organ 3b".

    target_f = (kappa * prior_f + n_f * e_f) / (kappa + n_f)      kappa = 30 evidenced acts
    e_f      = mean of the evidence rates that EXIST for f (a missing source is skipped, never 0)

Evidence sources, each a rate per act in 0..1 (never a count, so running more does not win):

* (a) downstream use: ``act_downstream_use`` rows with ``link_status = 'linked'`` and
  ``window_closed``: the share whose output a decision site used within 72 h. ``no_link_path``
  rows are "cannot be measured" (used is NULL), so a family with only those has NO source (a),
  not a rate of 0. An open window is not a final "not used" and is not counted.
* (b) Juniper engagement: acts whose signal is ``juniper_replied``, closed by the horizon
  resolver: the share Juniper answered. Only families with Juniper-visible output have it.
* (c) Orion's endorsement (personality graph): EXCLUDED. The Patch A recon found it not yet
  independent of (a) (design "Producer split", item 7: "stays out until it records real choices").

Guards (the replay's R7 checks them):

* fewer than 10 evidenced acts: the target is the prior, labelled ``prior`` (ungraded);
* targets are normalized so each tier's mean is 1.0;
* the published weight moves at most 0.1 per day toward its target (bounded drift), replayed day
  by day from the window start, so the cap holds without any stored state;
* rest: with no new evidence the evidence ages out of the 30-day window and the weight relaxes
  to its (normalized) prior, never toward 0 or 1.

The 5% exploration floor is applied in Choice (it needs the funded shares), not here.

Pure: the caller passes the act and use history and the day range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Mapping, Optional, Sequence

from orion.physiology.appetite import ActRecord
from orion.physiology.families import FamiliesConfig

KAPPA = 30.0
MIN_EVIDENCED = 10
MAX_DRIFT_PER_DAY = 0.1
EVIDENCE_WINDOW = timedelta(days=30)
ENGAGEMENT_SIGNAL = "juniper_replied"


@dataclass(frozen=True)
class UseRecord:
    """One ``act_downstream_use`` row."""
    act_id: str
    family: str
    started_at: Optional[datetime]
    link_status: str            # linked | no_output | no_link_path
    used: Optional[bool]
    window_closed: bool


@dataclass(frozen=True)
class Evidence:
    use_rate: Optional[float]
    use_n: int
    engagement_rate: Optional[float]
    engagement_n: int

    @property
    def n(self) -> int:
        # Distinct acts are counted once: the two sources never share an act today ((a) has no link
        # path for outreach, the only family with (b) that bids). A shared act would count twice.
        return self.use_n + self.engagement_n

    @property
    def mean_rate(self) -> Optional[float]:
        rates = [r for r in (self.use_rate, self.engagement_rate) if r is not None]
        return sum(rates) / len(rates) if rates else None


def evidence_for(family: str, acts: Sequence[ActRecord], uses: Sequence[UseRecord], *, since: datetime,
                 until: datetime) -> Evidence:
    u = [x for x in uses if x.family == family and x.link_status == "linked" and x.window_closed
         and x.used is not None and x.started_at is not None and since <= x.started_at < until]
    e = [a for a in acts if a.family == family and a.signal_id == ENGAGEMENT_SIGNAL and a.outcome_source == "resolver"
         and a.outcome_value is not None and a.closed_at is not None and since <= a.closed_at < until]
    return Evidence(use_rate=(sum(1 for x in u if x.used) / len(u)) if u else None, use_n=len(u),
                    engagement_rate=(sum(min(1.0, max(0.0, a.outcome_value)) for a in e) / len(e)) if e else None,
                    engagement_n=len(e))


def shrunk_target(prior: float, ev: Evidence) -> tuple[float, str]:
    """(target, basis). Under MIN_EVIDENCED acts, or no source at all: the prior."""
    m = ev.mean_rate
    if m is None or ev.n < MIN_EVIDENCED:
        return prior, "prior"
    return (KAPPA * prior + ev.n * m) / (KAPPA + ev.n), "evidence"


@dataclass(frozen=True)
class FamilyWeight:
    family: str
    tier: int
    weight: float
    target: float
    basis: str                 # prior | evidence
    evidence: Evidence


def _day_end(d: date) -> datetime:
    return datetime.combine(d + timedelta(days=1), time(0), tzinfo=timezone.utc)


def weight_series(families: FamiliesConfig, acts: Sequence[ActRecord], uses: Sequence[UseRecord], *,
                  start: date, end: date) -> dict[date, dict[str, FamilyWeight]]:
    """Daily weights for every bidding family, start..end inclusive (UTC days, evidence up to each
    day's end). Day before ``start``: every family at its normalized prior."""
    bidders = {f: c for f, c in families.families.items() if c.bids}
    tiers = sorted({c.tier for c in bidders.values()})

    def normalize(vals: Mapping[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in tiers:
            fs = [f for f in vals if bidders[f].tier == t]
            if not fs:
                continue
            mean = sum(vals[f] for f in fs) / len(fs)
            for f in fs:
                out[f] = vals[f] / mean if mean > 0 else 1.0
        return out

    prev = normalize({f: c.prior_weight for f, c in bidders.items()})
    series: dict[date, dict[str, FamilyWeight]] = {}
    d = start
    while d <= end:
        until = _day_end(d)
        since = until - EVIDENCE_WINDOW
        raw: dict[str, tuple[float, str, Evidence]] = {}
        for f, c in bidders.items():
            ev = evidence_for(f, acts, uses, since=since, until=until)
            tgt, basis = shrunk_target(c.prior_weight, ev)
            raw[f] = (tgt, basis, ev)
        norm = normalize({f: v[0] for f, v in raw.items()})
        today: dict[str, FamilyWeight] = {}
        for f in bidders:
            w = min(prev[f] + MAX_DRIFT_PER_DAY, max(prev[f] - MAX_DRIFT_PER_DAY, norm[f]))
            today[f] = FamilyWeight(family=f, tier=bidders[f].tier, weight=w, target=norm[f], basis=raw[f][1],
                                    evidence=raw[f][2])
        series[d] = today
        prev = {f: today[f].weight for f in bidders}
        d += timedelta(days=1)
    return series


def weights_now(families: FamiliesConfig, acts: Sequence[ActRecord], uses: Sequence[UseRecord],
                now: datetime) -> dict[str, FamilyWeight]:
    """Today's weights: the daily replay over the evidence window ending today."""
    end = now.astimezone(timezone.utc).date()
    start = end - timedelta(days=int(EVIDENCE_WINDOW.days))
    series = weight_series(families, acts, uses, start=start, end=end)
    return series[end]


def max_daily_move(series: Mapping[date, Mapping[str, FamilyWeight]]) -> float:
    days = sorted(series)
    worst = 0.0
    for a, b in zip(days, days[1:]):
        for f, w in series[b].items():
            if f in series[a]:
                worst = max(worst, abs(w.weight - series[a][f].weight))
    return worst


def funded_shares(families: FamiliesConfig, acts: Iterable[ActRecord], *, since: datetime, until: datetime
                  ) -> dict[str, Optional[float]]:
    """Each bidding family's share of its tier's funded acts (acts opened in [since, until)). In
    shadow every real act was funded by today's gates, so act counts are the funded counts. None:
    the tier funded nothing in the window (no share to defend)."""
    counts: dict[str, int] = {f: 0 for f, c in families.families.items() if c.bids}
    for a in acts:
        t = a.opened_at or a.closed_at
        if a.family in counts and t is not None and since <= t < until:
            counts[a.family] += 1
    out: dict[str, Optional[float]] = {}
    for f, n in counts.items():
        tier = families.families[f].tier
        total = sum(m for g, m in counts.items() if families.families[g].tier == tier)
        out[f] = (n / total) if total > 0 else None
    return out


__all__ = [
    "Evidence",
    "FamilyWeight",
    "KAPPA",
    "MAX_DRIFT_PER_DAY",
    "MIN_EVIDENCED",
    "UseRecord",
    "evidence_for",
    "funded_shares",
    "max_daily_move",
    "shrunk_target",
    "weight_series",
    "weights_now",
]
