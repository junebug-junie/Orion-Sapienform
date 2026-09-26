"""Bounded retry for world-pulse reading turns.

A failed Stage 1 / Stage 2 turn used to be terminal: ``status='failed'`` with
no path back to ``pending``. Live on 2026-09-19 that left Stage 2 with nothing
to do for four days (done/failed 12, done/pending 0) and 54 Stage 1 seeds dead,
almost all from one transient GPU-capacity refusal
(``turn_deferred:stance_react_failed``). The seeds were fine; the
infrastructure was not.

``is_transient_failure`` is the whole policy: a pure prefix match on the short,
grep-friendly ``fail_reason`` labels the two loops already write into
``last_error`` / ``stage2_error``. Anything it does not recognise stays
terminal on the first failure -- a bad seed (invalid handoff, unparseable
JSON, schema rejection, bad URL) must not burn two more wallet slots.
"""

from __future__ import annotations

from dataclasses import dataclass

# Prefixes of failure reasons that describe the *turn* failing, not the seed.
# Each of these is a label the loops' `_generate` / `_reason_from_non_final_frame`
# emit (services/orion-hub/scripts/world_pulse_read_pipeline.py and
# world_pulse_read_stage2.py) or a downstream rail being unavailable. A retry
# is a real turn: it is claimed, debited, and cooldown-spaced like any other.
TRANSIENT_FAILURE_PREFIXES: tuple[str, ...] = (
    "turn_deferred:",          # harness deferred the turn (GPU capacity, stance refusal)
    "turn_deferred",
    "turn_error:",             # harness error frame (fcc turn timed out, stream stalled)
    "turn_error",
    "turn_exception:",         # execute_unified_turn raised (governor/bus unreachable)
    "turn_exception",
    "stage1_turn_timeout",     # Hub-side asyncio.wait_for ceiling
    "stage2_turn_timeout",
    "empty_generation",
    "blank_final_response",
    "looks_like_error_text",
    "no_final_frame",
    "non_final_frame:",
    "bus_unavailable",
    "journal_bus_unavailable",
    "concept_atlas_store_unavailable",
    # The governor that ran the turn predates HarnessRunV1.source_fetches, so
    # there is no way to tell whether the source was read. Infra, not the seed.
    # Plain `no_read_evidence` (orion/world_pulse_read/read_evidence.py) is
    # deliberately NOT here: the model ran and did not read the source.
    "no_read_evidence:harness_unreported",
)


def is_transient_failure(reason: str | None) -> bool:
    """True when ``reason`` names an infrastructure/turn failure worth retrying.

    Prefix match only, on the exact labels the loops write. Validation errors,
    JSON parse failures, ``handoff_invalid:*``, ``stage2_result_not_object``,
    bad URLs and reclaim markers (``interrupted:*``) are all non-transient.
    """
    if not reason:
        return False
    text = str(reason).strip()
    return any(text.startswith(prefix) for prefix in TRANSIENT_FAILURE_PREFIXES)


@dataclass(frozen=True)
class FailureOutcome:
    """What ``mark_seed_failed`` / ``mark_stage2_failed`` actually did.

    ``status`` is the row's new stage status: ``'pending'`` means a retry was
    scheduled, ``'failed'`` means terminal. ``attempts`` is the number of
    failed turns so far, including this one.
    """

    status: str
    attempts: int

    @property
    def retry_scheduled(self) -> bool:
        return self.status == "pending"


# Failure reasons that mean the turn stopped *before any reading happened*, so
# the wallet slot it was debited is refunded (orion/world_pulse_read/
# wallet_refund.py). `turn_deferred` frames are only built by the stance phase
# of execute_unified_turn (orion/hub/turn_orchestrator.py: stance timeout /
# missing thought, stance defer/refuse, and stance_react_failed -- which is
# where a GPU capacity refusal lands, today as `gpu_pool_unavailable:<reason>`,
# before 2026-09-25 as `gateway_capacity_rejected:<stage>`), all of which return
# before the harness/FCC reader is dispatched. Everything else -- `turn_error:*`
# (the reader ran and failed), timeouts, exceptions, parse failures -- still
# costs its slot.


def is_refused_before_work(reason: str | None) -> bool:
    """True when ``reason`` names a turn that did no reading (refund its slot)."""
    if not reason:
        return False
    text = str(reason).strip()
    return text == "turn_deferred" or text.startswith("turn_deferred:")


def is_capacity_deferral(reason: str | None) -> bool:
    """Known pre-reader admission failures do not spend a reading attempt.

    Actual stance refusals and unknown failures keep the bounded retry policy.
    The workers already apply wallet refund backoff to these deferred turns.
    """
    text = str(reason or "").strip()
    if text == "turn_deferred:stance_react_timeout":
        return True
    prefix = "turn_deferred:stance_react_failed:"
    if not text.startswith(prefix):
        return False
    detail = text[len(prefix):].strip()
    # Older Thought deployments combined agent/chat capacity failures.
    parts = [part.strip().removeprefix("agent=").removeprefix("chat=") for part in detail.split(";")]
    return bool(parts) and all(
        part.startswith(("gpu_pool_unavailable:", "gateway_capacity_rejected:"))
        or part == "timeout:caller_budget_exhausted"
        for part in parts
    )
