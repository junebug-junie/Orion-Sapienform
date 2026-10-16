"""Rung 6 producer: turns recorded raw readings into governed mutation proposals.

Runs inside the Hub's existing mutation scheduler cycle and feeds the existing
governed pipeline (``SubstrateAdaptationWorker``): trial -> decision -> apply ->
post-adoption monitor -> rollback or settle. It owns three things:

1. ``propose()``: per domain, at most once per ``min_interval``, assess the
   recent window (``prediction_expectation.assess``). If a candidate beats the
   live setting with confidence on the older part of the window, record one
   signal and enqueue one proposal whose patch IS that candidate -- derived
   from evidence, not a constant.
2. ``trial_metrics()``: the trial scores the candidate against the setting it
   would replace, on readings after the fit span only.
3. ``post_adoption_deltas()``: after adoption, replay both settings over the
   readings that arrived since, and report the upper confidence bound of
   (adopted - previous). The worker's ``PostAdoptionMonitor`` rolls back when
   that bound is at or below its regression threshold (Orion is confident the
   new setting is clearly worse); otherwise the change settles when its window
   closes.

It bypasses ``PressureAccumulator`` on purpose: that score converges to a
fixed point set by signal strength, not rate, so it can't say "how much
evidence" (see memory: mutation-loop signal/action mismatch). The evidence
strength here is the paired confidence bound itself.

I/O is injected (``read_readings``, ``read_live``) so this module stays
testable without Postgres; ``postgres_reading_reader`` is the production
reader.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from orion.core.schemas.substrate_mutation import (
    MutationPatchV1,
    MutationProposalV1,
    MutationSignalV1,
)
from orion.substrate.mutation_contracts import CONTRACTS
from orion.substrate.prediction_expectation import (
    MIN_HOLDOUT_READINGS,
    MIN_POST_ADOPTION_READINGS,
    MUTATION_CLASS,
    Expectation,
    assess,
    domain_for_surface,
    domain_specs,
    expectation_from_payload,
    holdout_gain,
)

logger = logging.getLogger(__name__)

RAW_READING_TABLE = "substrate_prediction_raw_reading"
REPLAY_CORPUS_ID = "replay-prediction-raw-reading-v1"
BASELINE_METRIC_REF = "baseline-prediction-expectation-live-v1"
_FIT_START_NOTE = "fit_window_start:"
_FIT_END_NOTE = "fit_window_end:"

Readings = list[tuple[datetime, float]]
ReadingReader = Callable[[str, datetime], Readings]


def _note_time(notes: list[str], prefix: str) -> datetime | None:
    for note in notes:
        if note.startswith(prefix):
            try:
                return datetime.fromisoformat(note[len(prefix):])
            except ValueError:
                return None
    return None


@dataclass
class PredictionExpectationTuner:
    read_readings: ReadingReader
    read_live: Callable[[str], Expectation]
    window: timedelta = timedelta(hours=72)
    min_interval: timedelta = timedelta(hours=1)
    _last_assessed: dict[str, datetime] = field(default_factory=dict)

    def propose(self, *, store, now: datetime | None = None) -> list[MutationProposalV1]:
        t = now or datetime.now(timezone.utc)
        proposals: list[MutationProposalV1] = []
        for domain, spec in domain_specs().items():
            surface = spec.target_surface
            last = self._last_assessed.get(domain)
            if last is not None and t - last < self.min_interval:
                continue
            if store.active_surface(surface) is not None or store.has_queued_proposal_for_surface(surface):
                continue
            cooldown = store.rollback_cooldown_until(surface)
            if cooldown is not None and t < cooldown:
                continue
            self._last_assessed[domain] = t
            current = self.read_live(domain)
            readings = self.read_readings(domain, t - self.window)
            assessment = assess(spec, readings, current=current)
            if assessment is None:
                continue
            gain = assessment.fit_gain
            evidence_refs = [
                f"prediction_raw_reading:{domain}:{assessment.fit_window_start.isoformat()}..{assessment.fit_window_end.isoformat()}",
                f"fit_readings:{assessment.reading_count}",
                f"fit_log_score_gain_mean:{gain.mean:.6g}",
                f"fit_log_score_gain_lower_bound:{gain.lower_bound:.6g}",
            ]
            signal = MutationSignalV1(
                event_kind="prediction_expectation_miscalibrated",
                anchor_scope="orion",
                subject_ref="entity:orion",
                target_surface=surface,
                target_zone="prediction_error",
                strength=1.0,
                evidence_refs=evidence_refs,
                source_ref=f"{RAW_READING_TABLE}:{domain}",
                metadata={
                    "source_kind": "prediction_raw_reading_replay",
                    "domain": domain,
                    "current": current.as_patch(),
                    "candidate": assessment.candidate.as_patch(),
                    "fit_gain_mean": gain.mean,
                    "fit_gain_lower_bound": gain.lower_bound,
                    "fit_readings": gain.count,
                },
                detected_at=t,
            )
            contract = CONTRACTS[MUTATION_CLASS]
            proposal = MutationProposalV1(
                mutation_class=MUTATION_CLASS,
                risk_tier=contract.risk_tier,
                target_surface=surface,
                anchor_scope="orion",
                subject_ref="entity:orion",
                rationale=(
                    f"{domain} surprise is better calibrated with alpha={assessment.candidate.alpha:g}, "
                    f"min_variance={assessment.candidate.min_variance:g} than the live "
                    f"alpha={current.alpha:g}, min_variance={current.min_variance:g}: "
                    f"+{gain.mean:.4g} nats/reading over {gain.count} readings (lower bound {gain.lower_bound:.4g})"
                ),
                expected_effect="raise_prediction_log_score",
                evidence_refs=evidence_refs,
                source_signal_ids=[signal.signal_id],
                source_pressure_id=f"direct-evidence:{signal.signal_id}",
                patch=MutationPatchV1(
                    mutation_class=MUTATION_CLASS,
                    target_surface=surface,
                    target_ref=domain,
                    patch=assessment.candidate.as_patch(),
                    rollback_payload=current.as_patch(),
                ),
                notes=[
                    _FIT_START_NOTE + assessment.fit_window_start.isoformat(),
                    _FIT_END_NOTE + assessment.fit_window_end.isoformat(),
                ],
                created_at=t,
            )
            store.record_signal(signal)
            store.add_proposal(proposal, priority=60, due_at=t)
            proposals.append(proposal)
        return proposals

    def trial_metrics(self, proposal: MutationProposalV1) -> dict[str, float] | None:
        """Holdout gain of the candidate over the setting it would replace.

        Scored only on readings after the fit span, so the trial never grades
        the candidate on the data that picked it. ``None`` (-> inconclusive)
        until the holdout holds ``MIN_HOLDOUT_READINGS``."""
        spec = domain_for_surface(proposal.target_surface)
        start = _note_time(proposal.notes, _FIT_START_NOTE)
        end = _note_time(proposal.notes, _FIT_END_NOTE)
        if spec is None or start is None or end is None:
            return None
        reference = expectation_from_payload(proposal.patch.rollback_payload, default=spec.default)
        candidate = expectation_from_payload(proposal.patch.patch, default=reference)
        gain = holdout_gain(
            self.read_readings(spec.domain, start),
            candidate=candidate,
            reference=reference,
            after=end,
            min_readings=MIN_HOLDOUT_READINGS,
        )
        if gain is None:
            return None
        return {
            "holdout_log_score_gain_lower_bound": gain.lower_bound,
            "holdout_log_score_gain_mean": gain.mean,
            "holdout_reading_count": float(gain.count),
        }

    def post_adoption_deltas(self, *, store) -> dict[str, float]:
        """proposal_id -> upper bound of (adopted - previous) log-score gain on
        readings since adoption, for every live adoption of this class with
        enough of them. Missing = not enough evidence yet (the worker then
        settles on the window instead of guessing)."""
        deltas: dict[str, float] = {}
        for adoption in list(store._adoptions.values()):  # bounded store state, as the worker reads it
            if adoption.status != "applied":
                continue
            spec = domain_for_surface(adoption.target_surface)
            if spec is None:
                continue
            adopted = expectation_from_payload(adoption.applied_patch, default=spec.default)
            previous = expectation_from_payload(adoption.rollback_payload, default=spec.default)
            gain = holdout_gain(
                self.read_readings(spec.domain, adoption.applied_at - self.window),
                candidate=adopted,
                reference=previous,
                after=adoption.applied_at,
                min_readings=MIN_POST_ADOPTION_READINGS,
            )
            if gain is not None:
                deltas[adoption.proposal_id] = gain.upper_bound
        return deltas


def postgres_reading_reader(postgres_url: str, *, limit: int = 20000) -> ReadingReader:
    """Production reader over ``substrate_prediction_raw_reading``."""
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_url, pool_pre_ping=True, pool_size=1, max_overflow=0)

    def read(domain: str, since: datetime) -> Readings:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT observed_at, raw_value FROM {RAW_READING_TABLE} "
                    "WHERE domain = :domain AND observed_at >= :since "
                    "ORDER BY observed_at DESC LIMIT :limit"
                ),
                {"domain": domain, "since": since, "limit": limit},
            ).fetchall()
        return [(row[0], float(row[1])) for row in reversed(rows)]

    return read
