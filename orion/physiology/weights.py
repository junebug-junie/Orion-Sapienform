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
* (c) Orion's endorsement (personality rev 4, Move 2; ``ChoiceRecord`` from ``personality_choice``): the
  share of choice points where Orion picked an option from family f when one was offered, a rate per
  offer. Only a REAL choice counts: one that could be declined ("no move", "say nothing":
  ``DECLINABLE_LANES``, or any row that did decline) or that offered options from two or more families.
  A curiosity prior menu (one family, no "none") is never a point for (c): picking from it always reads
  1.0, which measures nothing. A ``care_promoted`` choice counts half (runaway guard). Families with no
  choice point have NO source (c), never 0. Off unless the caller passes ``choices``.

  Which family an option belongs to (``option_family``): an option key ``<family>::<id>`` names it
  (``endorsement_option_key``, for a producer whose options span families). Prime uses it since
  2026-11-13: a concern that came from a funded event want is keyed ``<want family>::<concern_id>``, so a
  Prime pick of a reading concern counts for ``hub.world_pulse_read`` and a camera ask for ``vision.ask``.
  A Prime "no move" (``declined_all``) is then a decline of every shown concern's family: Orion passed on
  those events. Only a concern with no event (a turn promise, arc mode) keeps a plain key and counts for
  ``prime``. Otherwise the lane's family (``LANE_FAMILIES``): outreach -> hub.outreach,
  reading -> hub.world_pulse_read, curiosity -> durable.curiosity_investigate, prime -> prime. The chat
  lane has none (Juniper's turns are Tier 0 and never bid).

  Independence (spec gate step 2): (c) is measured at the choice, (b) engagement after the send, and for
  outreach Juniper replies to what Orion chose to send. ``endorsement_exclusions`` correlates the two per
  family on non-overlapping daily rates; above ``ENDORSEMENT_CORR_EXCLUDE`` (0.8) with at least
  ``ENDORSEMENT_CORR_MIN_DAYS`` paired days, (c) is left out for that family (the caller logs it).
* (d) reliability, the self-monitor's track record (attention as the want competition, patch 1):
  the share of the family's world-settled closes that the self-monitor grades ``upheld``
  (``orion.self_monitor.adapters.act_outcome``, the same function the self-monitor uses, so
  there is no second track record). A close that never ran (``failed_to_execute``) and one with no
  outcome are not grades. Resolver closes (Juniper's turn did or did not start) are left out: they
  ARE source (b), and counting them twice would double Juniper's vote. Off unless the caller asks
  (``reliability=True``) and reads ``grammar_acts.outcome_note`` into ``ActRecord``; without the
  note a failed run would read as a refuted one.

  **Self-grades (patch 2, Juniper decides).** Most closes are graded by the doer itself
  (``outcome_source = producer``), which saturates (d) near 1.0. ``self_grades`` chooses how a
  producer self-grade counts; the default is today's behaviour, and the live market passes nothing:

  * ``all`` (default): every grade counts fully.
  * ``world_only``: only grades the world made count. Resolver closes are already source (b), so
    what is left is the ``expired`` close (the deadline passed with no artifact).
  * ``half``: a producer self-grade counts as half a grade, both in the rate and in ``n``.

Each act counts once toward ``n`` however many sources measured it.

Guards (the replay's R7 checks them):

* fewer than 10 evidenced acts: the target is the prior, labelled ``prior`` (ungraded);
* targets are normalized so each tier's mean is 1.0;
* the published weight moves at most 0.1 per day toward its target (bounded drift), replayed day
  by day from the window start, so the cap holds without any stored state;
* rest: with no new evidence the evidence ages out of the 30-day window and the weight relaxes
  to its (normalized) prior, never toward 0 or 1.

The 5% exploration floor is applied in Choice, on want_ledger's funded wants (Patch C3a), not here.

Pure: the caller passes the act and use history and the day range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Mapping, Optional, Sequence

from orion.physiology.appetite import ActRecord
from orion.physiology.families import FamiliesConfig
from orion.self_monitor.adapters import act_outcome

KAPPA = 30.0
MIN_EVIDENCED = 10
MAX_DRIFT_PER_DAY = 0.1
EVIDENCE_WINDOW = timedelta(days=30)
ENGAGEMENT_SIGNAL = "juniper_replied"
RESOLVER_SOURCE = "resolver"
PRODUCER_SOURCE = "producer"
SELF_GRADE_MODES = ("all", "world_only", "half")
SELF_GRADE_HALF = 0.5

# Source (c): choice lanes -> want families (module docstring). ``chat`` has none (Tier 0).
LANE_FAMILIES: Mapping[str, str] = {
    "outreach": "hub.outreach",
    "reading": "hub.world_pulse_read",
    "curiosity": "durable.curiosity_investigate",
    "prime": "prime",
}
# Lanes whose choice point shows a "none" option (orion.personality.graph.DECLINE_TEXT's lanes).
DECLINABLE_LANES = frozenset({"prime", "outreach"})
OPTION_FAMILY_SEP = "::"
CARE_PROMOTED_WEIGHT = 0.5
ENDORSEMENT_CORR_EXCLUDE = 0.8
ENDORSEMENT_CORR_MIN_DAYS = 7


@dataclass(frozen=True)
class UseRecord:
    """One ``act_downstream_use`` row."""
    act_id: str
    family: str
    started_at: Optional[datetime]
    link_status: str            # linked | no_output | no_link_path
    used: Optional[bool]
    window_closed: bool


def endorsement_option_key(family: str, option_id: str) -> str:
    """A ``ChoiceOptionV1.key`` that names the option's want family, for a producer whose options span
    families (``option_family`` reads it back)."""
    return f"{family}{OPTION_FAMILY_SEP}{option_id}"


def option_family(lane: str, key: Optional[str]) -> Optional[str]:
    """The want family an option belongs to: ``<family>::<id>`` in its key, else the lane's family."""
    if isinstance(key, str) and OPTION_FAMILY_SEP in key:
        fam = key.split(OPTION_FAMILY_SEP, 1)[0].strip()
        if fam:
            return fam
    return LANE_FAMILIES.get(lane)


@dataclass(frozen=True)
class ChoiceRecord:
    """One ``personality_choice`` row, as much as source (c) reads of it."""
    choice_id: str
    lane: str
    occurred_at: datetime
    chosen_family: Optional[str]          # None when declined_all
    offered_families: frozenset           # the family of every shown option (chosen + alternatives)
    declined_all: bool = False
    care_promoted: bool = False
    act_id: Optional[str] = None

    @property
    def real_choice(self) -> bool:
        """Could Orion have done otherwise? A decline was on offer, or the options span families."""
        return self.lane in DECLINABLE_LANES or self.declined_all or len(self.offered_families) >= 2


def choice_record(choice_id: str, lane: str, occurred_at: datetime, chosen_key: Optional[str],
                  alternative_keys: Sequence[str], *, declined_all: bool = False, care_promoted: bool = False,
                  act_id: Optional[str] = None) -> ChoiceRecord:
    chosen_f = None if declined_all or chosen_key is None else option_family(lane, chosen_key)
    offered = {option_family(lane, k) for k in alternative_keys}
    if chosen_f is not None:
        offered.add(chosen_f)
    offered.discard(None)
    return ChoiceRecord(choice_id=choice_id, lane=lane, occurred_at=occurred_at, chosen_family=chosen_f,
                        offered_families=frozenset(offered), declined_all=declined_all,
                        care_promoted=care_promoted, act_id=act_id)


def endorsement_for(family: str, choices: Sequence[ChoiceRecord], *, since: datetime, until: datetime
                    ) -> tuple[Optional[float], int, frozenset]:
    """(rate, n, ids) of source (c) for ``family`` in [since, until). No real choice point: (None, 0)."""
    rate, n, weighted = _endorsement(family, choices, since=since, until=until)
    return rate, n, frozenset(weighted)


def _endorsement(family: str, choices: Sequence[ChoiceRecord], *, since: datetime, until: datetime
                 ) -> tuple[Optional[float], int, dict[str, float]]:
    """(rate, n, id -> weight): a ``care_promoted`` point weighs ``CARE_PROMOTED_WEIGHT`` in the rate AND in n."""
    num = den = 0.0
    n = 0
    ids: dict[str, float] = {}
    for c in choices:
        if family not in c.offered_families or not c.real_choice or not since <= c.occurred_at < until:
            continue
        w = CARE_PROMOTED_WEIGHT if c.care_promoted else 1.0
        den += w
        num += w if c.chosen_family == family else 0.0
        n += 1
        key = c.act_id or f"choice:{c.choice_id}"
        ids[key] = max(ids.get(key, 0.0), w)
    return (num / den if den > 0 else None), n, ids


@dataclass(frozen=True)
class Evidence:
    use_rate: Optional[float]
    use_n: int
    engagement_rate: Optional[float]
    engagement_n: int
    reliability_rate: Optional[float] = None
    reliability_n: int = 0
    # Every act any source measured, so an act measured by two sources counts once in ``n``.
    act_ids: frozenset = frozenset()
    # self_grades="half": how much less than one full act each act measured ONLY by a producer
    # self-grade counts toward ``n`` (0.5 per such act). 0 in every other mode.
    n_discount: float = 0.0
    # Source (c): Orion's endorsement (personality rev 4), a rate per real choice point.
    endorsement_rate: Optional[float] = None
    endorsement_n: int = 0

    @property
    def n(self) -> float:
        base = len(self.act_ids) if self.act_ids else self.use_n + self.engagement_n + self.reliability_n
        if not self.act_ids:
            base += self.endorsement_n
        return base - self.n_discount if self.n_discount else base

    @property
    def mean_rate(self) -> Optional[float]:
        rates = [r for r in (self.use_rate, self.engagement_rate, self.reliability_rate, self.endorsement_rate)
                 if r is not None]
        return sum(rates) / len(rates) if rates else None


def reliability_grades(family: str, acts: Sequence[ActRecord], *, since: datetime, until: datetime,
                       self_grades: str = "all") -> list[tuple[str, bool, float]]:
    """Source (d): (act_id, upheld, weight) for each graded close of ``family`` in [since, until).
    ``self_grades`` sets how a producer self-grade counts (module docstring); weight is 1 except a
    producer grade under ``half``."""
    if self_grades not in SELF_GRADE_MODES:
        raise ValueError(f"self_grades must be one of {SELF_GRADE_MODES}, not {self_grades!r}")
    out: list[tuple[str, bool, float]] = []
    for a in acts:
        if (a.family != family or a.closed_at is None or not since <= a.closed_at < until
                or a.outcome_source == RESOLVER_SOURCE):
            continue
        self_graded = a.outcome_source == PRODUCER_SOURCE
        if self_graded and self_grades == "world_only":
            continue
        grade = act_outcome({"outcome_source": a.outcome_source, "outcome_note": a.outcome_note,
                             "outcome_value": a.outcome_value})
        if grade in ("upheld", "refuted"):
            out.append((a.act_id, grade == "upheld", SELF_GRADE_HALF if self_graded and self_grades == "half" else 1.0))
    return out


def evidence_for(family: str, acts: Sequence[ActRecord], uses: Sequence[UseRecord], *, since: datetime,
                 until: datetime, reliability: bool = False, self_grades: str = "all",
                 choices: Optional[Sequence[ChoiceRecord]] = None) -> Evidence:
    u = [x for x in uses if x.family == family and x.link_status == "linked" and x.window_closed
         and x.used is not None and x.started_at is not None and since <= x.started_at < until]
    e = [a for a in acts if a.family == family and a.signal_id == ENGAGEMENT_SIGNAL and a.outcome_source == RESOLVER_SOURCE
         and a.outcome_value is not None and a.closed_at is not None and since <= a.closed_at < until]
    if reliability and any(a.closed_at is not None for a in acts) and all(a.outcome_note is None for a in acts):
        # Without grammar_acts.outcome_note a run that never happened reads as a refuted one.
        raise ValueError("reliability=True needs ActRecord.outcome_note read from grammar_acts")
    r = reliability_grades(family, acts, since=since, until=until, self_grades=self_grades) if reliability else []
    other_ids = {x.act_id for x in u} | {a.act_id for a in e}
    c_rate, c_n, c_w = _endorsement(family, choices, since=since, until=until) if choices else (None, 0, {})
    measured = set(other_ids) | {i for i, _, _ in r}
    other_ids |= set(c_w)
    ids = frozenset(other_ids | {i for i, _, _ in r})
    r_weight = sum(w for _, _, w in r)
    # An act measured only by a half-weight self-grade counts half toward n (one grade per act).
    discount = sum(1.0 - w for i, _, w in r if i not in other_ids)
    # A care_promoted choice point measured by nothing else counts half toward n, as it does in the rate.
    discount += sum(1.0 - w for i, w in c_w.items() if i not in measured)
    return Evidence(use_rate=(sum(1 for x in u if x.used) / len(u)) if u else None, use_n=len(u),
                    engagement_rate=(sum(min(1.0, max(0.0, a.outcome_value)) for a in e) / len(e)) if e else None,
                    engagement_n=len(e),
                    reliability_rate=(sum(w for _, ok, w in r if ok) / r_weight) if r_weight > 0 else None,
                    reliability_n=len(r), act_ids=ids, n_discount=discount,
                    endorsement_rate=c_rate, endorsement_n=c_n)


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
                  start: date, end: date, reliability: bool = False, self_grades: str = "all",
                  choices: Optional[Sequence[ChoiceRecord]] = None,
                  endorsement_excluded: frozenset = frozenset()) -> dict[date, dict[str, FamilyWeight]]:
    """Daily weights for every bidding family, start..end inclusive (UTC days, evidence up to each
    day's end). Day before ``start``: every family at its normalized prior. ``reliability``: also
    read source (d) (module docstring). ``choices``: also read source (c), except for the families in
    ``endorsement_excluded`` (the independence check)."""
    # ``weighted: false`` families (families.yaml) stay out: Choice falls back to their prior, and they do not
    # move the tier's mean, so adding one leaves every other family's weight unchanged.
    bidders = {f: c for f, c in families.families.items() if c.bids and c.weighted}
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
            ev = evidence_for(f, acts, uses, since=since, until=until, reliability=reliability,
                              self_grades=self_grades,
                              choices=None if f in endorsement_excluded else choices)
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
                now: datetime, *, reliability: bool = False, self_grades: str = "all",
                choices: Optional[Sequence[ChoiceRecord]] = None,
                endorsement_excluded: frozenset = frozenset()) -> dict[str, FamilyWeight]:
    """Today's weights: the daily replay over the evidence window ending today."""
    end = now.astimezone(timezone.utc).date()
    start = end - timedelta(days=int(EVIDENCE_WINDOW.days))
    series = weight_series(families, acts, uses, start=start, end=end, reliability=reliability,
                           self_grades=self_grades, choices=choices, endorsement_excluded=endorsement_excluded)
    return series[end]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


@dataclass(frozen=True)
class EndorsementCheck:
    """Gate step 2 for one family: the correlation of daily (c) with daily (b) engagement."""
    family: str
    correlation: Optional[float]   # None: fewer than 2 paired days, or a constant series
    paired_days: int
    excluded: bool

    @property
    def reason(self) -> str:
        return f"corr_with_engagement:{self.correlation:.2f}" if self.excluded and self.correlation is not None else ""


def endorsement_exclusions(families: FamiliesConfig, acts: Sequence[ActRecord], choices: Sequence[ChoiceRecord],
                           now: datetime) -> dict[str, EndorsementCheck]:
    """Per bidding family with both sources: Pearson r of source (c) against source (b) over
    non-overlapping UTC days of the evidence window (a rolling-window series would be autocorrelated
    and read high). ``excluded`` when r > ``ENDORSEMENT_CORR_EXCLUDE`` on at least
    ``ENDORSEMENT_CORR_MIN_DAYS`` paired days."""
    end = now.astimezone(timezone.utc).date()
    out: dict[str, EndorsementCheck] = {}
    for f, c in families.families.items():
        if not (c.bids and c.weighted):
            continue
        xs: list[float] = []
        ys: list[float] = []
        for i in range(int(EVIDENCE_WINDOW.days)):
            d = end - timedelta(days=i)
            since = datetime.combine(d, time(0), tzinfo=timezone.utc)
            until = since + timedelta(days=1)
            rate, n, _ids = endorsement_for(f, choices, since=since, until=until)
            if rate is None:
                continue
            ev = evidence_for(f, acts, [], since=since, until=until)
            if ev.engagement_rate is None:
                continue
            xs.append(rate)
            ys.append(ev.engagement_rate)
        if not xs:
            continue
        r = _pearson(xs, ys)
        out[f] = EndorsementCheck(family=f, correlation=round(r, 4) if r is not None else None, paired_days=len(xs),
                                  excluded=r is not None and len(xs) >= ENDORSEMENT_CORR_MIN_DAYS
                                  and r > ENDORSEMENT_CORR_EXCLUDE)
    return out


def max_daily_move(series: Mapping[date, Mapping[str, FamilyWeight]]) -> float:
    days = sorted(series)
    worst = 0.0
    for a, b in zip(days, days[1:]):
        for f, w in series[b].items():
            if f in series[a]:
                worst = max(worst, abs(w.weight - series[a][f].weight))
    return worst


__all__ = [
    "ChoiceRecord",
    "DECLINABLE_LANES",
    "ENDORSEMENT_CORR_EXCLUDE",
    "ENDORSEMENT_CORR_MIN_DAYS",
    "EndorsementCheck",
    "LANE_FAMILIES",
    "choice_record",
    "endorsement_exclusions",
    "endorsement_for",
    "endorsement_option_key",
    "option_family",
    "Evidence",
    "FamilyWeight",
    "KAPPA",
    "MAX_DRIFT_PER_DAY",
    "MIN_EVIDENCED",
    "SELF_GRADE_MODES",
    "UseRecord",
    "evidence_for",
    "reliability_grades",
    "max_daily_move",
    "shrunk_target",
    "weight_series",
    "weights_now",
]
