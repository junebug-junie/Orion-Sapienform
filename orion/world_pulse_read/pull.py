"""Reading pull: one extra world-pulse read the motor allocator admitted (thinking rung 3 follow-up).

The self-scheduled lane (world_pulse_read_pipeline.tick) keeps draining the queue exactly as before,
up to HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT reads at once. A pull starts ONE more read now, on the seed
that lane would claim next (CLAIM_SQL's own order: priority, then age), so it adds a read rather
than reordering one. That is why it needs gates the lane does not have: it is new GPU work.

Gates, in order (the first that fails is the refusal), all reused, none duplicated:

1. ``disabled``: HUB_WORLD_PULSE_READ_PULL_ENABLED (kill switch, shipped on).
2. the reading pipeline's own Wallet A gate (``outside_window`` / ``refund_backoff``):
   orion/world_pulse_read/wallet_a.py ``wallet_a_block_reason``, as computed by tick().
3. ``daily_cap``: started pulls in the UTC day >= HUB_WORLD_PULSE_READ_PULL_DAILY_CAP. The default
   is derived from data (``derive_daily_cap``), not picked.
4. ``arousal_strained``: Orion's arousal (orion/regulation/arousal_read.py), the same reader and the
   same rule as the waking baseline painting (orion/reverie/baseline.py: strained defers it).
5. ``thermal_hot``: orion/autonomy/thermal_gate.py ``thermal_state`` with its default thresholds and
   hysteresis, on Hub's own cabinet sensor -- the gate orion-thought runs before a painting.
6. ``no_candidate``: no pending seed that is not already in flight and not already pulled.

Pure functions and SQL only; the I/O lives in services/orion-hub/scripts/world_pulse_read_pipeline.py.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from orion.autonomy.thermal_gate import DEFAULT_MAX_READING_AGE_SEC, ThermalVerdict
from orion.regulation.arousal_view import AROUSAL_STRAINED_REASON, DEFAULT_MAX_AGE_SEC

# The `look` act's ledger key (services/orion-hub/scripts/world_pulse_read_pipeline.py
# _open_read_act: test_before = read_seed:<seed_id>). An episode's object_ref is this key, so the
# outcome resolver joins grammar_acts on it with no reading-specific code.
READ_ACT_KEY_PREFIX = "read_seed:"

# Derived 2026-10-21 by derive_daily_cap over the 30 days ending 2026-10-21 05:00Z: distinct seeds
# whose Stage 1 durable turn started, per UTC day (reading_durable_turn, stage 1), 25 days with
# reads, median 8. Re-derive with orion/autonomy/evals/run_rung3_reading_replay_eval.py --live.
DERIVED_DAILY_CAP = 8

# Retry hints (seconds the same refusal will hold), each from the gate's own clock:
# arousal: the regulation state's Redis TTL (orion-durable-runs writes it every 120 s, 360 s TTL);
# thermal: the oldest reading the thermal gate trusts; no candidate: the reading lane's tick.
AROUSAL_RETRY_SEC = DEFAULT_MAX_AGE_SEC
THERMAL_RETRY_SEC = DEFAULT_MAX_READING_AGE_SEC

ENSURE_PULL_COLUMNS_SQL = """
ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS pull_dispatch_id text,
    ADD COLUMN IF NOT EXISTS pull_arm text,
    ADD COLUMN IF NOT EXISTS pulled_at timestamptz
"""

# CLAIM_SQL's order, restricted to seeds a pull may take: pending, no Stage 1 durable turn already
# submitted (that read is in flight and the lane polls it), and never pulled before (a control seed
# chosen again would turn its control into a delay).
SELECT_PULL_CANDIDATE_SQL = """
SELECT seed_id, kind, run_id, url, title, section, item_id, request_json
FROM world_pulse_read_seed s
WHERE s.status = 'pending'
  AND s.pull_dispatch_id IS NULL
  AND NOT EXISTS (SELECT 1 FROM reading_durable_turn d
                  WHERE d.seed_id = s.seed_id AND d.stage = 1 AND d.consumed_at IS NULL)
ORDER BY s.priority ASC, s.created_at ASC, s.seed_id ASC
LIMIT 1
"""

# Marks the seed for this decision. A treated pull also claims it (status claimed), so the lane's
# CLAIM_SQL cannot take the same row while the pull starts it. Returns nothing if the row moved.
MARK_PULL_SQL = """
UPDATE world_pulse_read_seed
SET pull_dispatch_id = $2,
    pull_arm = $3,
    pulled_at = now(),
    status = CASE WHEN $3 = 'treated' THEN 'claimed' ELSE status END,
    claimed_at = CASE WHEN $3 = 'treated' THEN now() ELSE claimed_at END
WHERE seed_id = $1 AND status = 'pending' AND pull_dispatch_id IS NULL
RETURNING seed_id, kind, run_id, url, title, section, item_id, request_json
"""

COUNT_STARTED_PULLS_SQL = """
SELECT count(*)::int FROM world_pulse_read_seed
WHERE pull_arm = 'treated' AND pulled_at >= $1 AND pulled_at < $2
"""

# The derivation input: distinct seeds whose Stage 1 read started, per UTC day.
DAILY_READS_STARTED_SQL = """
SELECT date_trunc('day', created_at AT TIME ZONE 'UTC') AS day, count(DISTINCT seed_id)::int AS n
FROM reading_durable_turn
WHERE stage = 1 AND created_at > now() - make_interval(days => $1)
GROUP BY 1 ORDER BY 1
"""


def derive_daily_cap(daily_reads: Iterable[int]) -> int:
    """The cap is the reading lane's own median day: pulls may add at most one more typical day of
    reading on top of what the lane already does. The median, not the mean or max, because the
    series has burst days (46-49 reads on 10-14..10-16, a backlog drain) that would otherwise set
    the bound. Days with no reads are not in the series (no row), so this is the median READING day."""
    values = [int(v) for v in daily_reads if int(v) > 0]
    if not values:
        raise ValueError("no reading days to derive a cap from")
    return max(1, int(round(statistics.median(values))))


def utc_day_bounds(now: datetime) -> tuple[datetime, datetime]:
    start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


@dataclass(frozen=True)
class PullGateInputs:
    enabled: bool
    wallet_reason: Optional[str]
    wallet_retry_sec: Optional[float]
    pulled_today: int
    daily_cap: int
    arousal_level: Optional[str]
    thermal: Optional[ThermalVerdict]
    now: datetime
    # Hold for a refusal whose gate has no clock of its own (kill switch): the lane's tick.
    default_retry_sec: float = 300.0


@dataclass(frozen=True)
class PullGateVerdict:
    refusal: Optional[str]
    retry_after_sec: Optional[float]

    @property
    def open(self) -> bool:
        return self.refusal is None


def pull_block_reason(inp: PullGateInputs) -> PullGateVerdict:
    """The first closed gate, or open. Unknown arousal / thermal changes nothing (both readers'
    documented rule: unknown is never strained, a dead sensor does not silently remove a capability)."""
    if not inp.enabled:
        return PullGateVerdict("disabled", inp.default_retry_sec)
    if inp.wallet_reason is not None:
        wait = inp.wallet_retry_sec if inp.wallet_retry_sec and inp.wallet_retry_sec > 0 else inp.default_retry_sec
        return PullGateVerdict(inp.wallet_reason, wait)
    if inp.pulled_today >= max(0, int(inp.daily_cap)):
        _, end = utc_day_bounds(inp.now)
        return PullGateVerdict("daily_cap", max(0.0, (end - inp.now).total_seconds()))
    if inp.arousal_level == "strained":
        return PullGateVerdict(AROUSAL_STRAINED_REASON, AROUSAL_RETRY_SEC)
    if inp.thermal is not None and not inp.thermal.allows_gpu_work:
        return PullGateVerdict("thermal_hot", THERMAL_RETRY_SEC)
    return PullGateVerdict(None, None)


def read_act_key(seed_id: str) -> str:
    return f"{READ_ACT_KEY_PREFIX}{seed_id}"


__all__ = [
    "AROUSAL_RETRY_SEC",
    "COUNT_STARTED_PULLS_SQL",
    "DAILY_READS_STARTED_SQL",
    "DERIVED_DAILY_CAP",
    "ENSURE_PULL_COLUMNS_SQL",
    "MARK_PULL_SQL",
    "PullGateInputs",
    "PullGateVerdict",
    "READ_ACT_KEY_PREFIX",
    "SELECT_PULL_CANDIDATE_SQL",
    "THERMAL_RETRY_SEC",
    "derive_daily_cap",
    "pull_block_reason",
    "read_act_key",
    "utc_day_bounds",
]
