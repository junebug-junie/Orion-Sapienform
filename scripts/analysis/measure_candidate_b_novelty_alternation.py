"""Read-only: did Candidate B novelty stop inventing news from steady inputs?

Until 2026-09-25, Candidate B (host/capability targets) diffed this tick's
pressure proxy against the previous frame's `salience_score` -- which for
these targets IS the previous novelty. A steady non-zero input therefore
scored p, 0, p, 0 forever. Active targets past a per-kind cap were also left
out of the frame entirely, so they read as brand new on the next tick.
(D1 in docs/superpowers/specs/2026-09-25-attention-with-stakes-design.md.)

This reads persisted `substrate_attention_frames` rows and answers, from the
stored numbers alone:

- Of consecutive-frame pairs where a Candidate B target's pressure did not
  move, how many still carried non-zero novelty? About half before the fix,
  about none after it.
- Which formula produced each pair? Novelty that matches
  |pressure_t - novelty_{t-1}| but not |pressure_t - pressure_{t-1}| is the
  old formula's fingerprint. This is how you confirm the fix is what is live.
- How often did a target vanish for exactly one frame and come back?
  That is the over-cap drop's fingerprint.

Writes nothing, emits nothing, flips nothing.

    POSTGRES_URI=... python scripts/analysis/measure_candidate_b_novelty_alternation.py --hours 24
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pg_readonly import open_readonly_connection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.attention.field_attention.selectors import PREDICTION_ERROR_NATIVE_TARGETS  # noqa: E402

logger = logging.getLogger("measure_candidate_b_novelty_alternation")

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
BUCKETS = (
    "dominant_targets",
    "node_targets",
    "capability_targets",
    "system_targets",
    "suppressed_targets",
)
# Pressure values are max() over stored channel floats, so a truly steady
# input repeats exactly; the tolerance only absorbs JSON float round-trips.
STEADY_EPS = 1e-6
# Novelty at or below this counts as "no news".
NOVELTY_EPS = 1e-3


@dataclass(frozen=True)
class TargetReading:
    pressure: float
    novelty: float


@dataclass
class AlternationReport:
    frames: int = 0
    targets: int = 0
    consecutive_pairs: int = 0
    steady_pairs: int = 0
    steady_pairs_with_novelty: int = 0
    old_formula_pairs: int = 0
    new_formula_pairs: int = 0
    ambiguous_pairs: int = 0
    one_frame_gaps: int = 0
    per_target_steady_with_novelty: dict[str, int] = field(default_factory=dict)

    @property
    def steady_novelty_fraction(self) -> Optional[float]:
        if self.steady_pairs == 0:
            return None
        return self.steady_pairs_with_novelty / self.steady_pairs


def is_candidate_b(target: dict[str, Any]) -> bool:
    kind = target.get("target_kind")
    target_id = str(target.get("target_id") or "")
    if kind == "capability":
        return True
    return kind == "node" and target_id not in PREDICTION_ERROR_NATIVE_TARGETS


def candidate_b_readings(frame_json: dict[str, Any]) -> dict[str, TargetReading]:
    """Every Candidate B target in one frame, first bucket wins (the same
    five-bucket order `scoring._find_prior_target` searches)."""
    readings: dict[str, TargetReading] = {}
    for bucket in BUCKETS:
        for target in frame_json.get(bucket) or []:
            if not isinstance(target, dict) or not is_candidate_b(target):
                continue
            target_id = str(target.get("target_id"))
            if target_id in readings:
                continue
            try:
                readings[target_id] = TargetReading(
                    pressure=float(target.get("pressure_score") or 0.0),
                    novelty=float(target.get("novelty_score") or 0.0),
                )
            except (TypeError, ValueError):
                continue
    return readings


def _close(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps


def analyze(frames: Iterable[dict[str, Any]]) -> AlternationReport:
    """`frames` oldest first. Pure: no I/O."""
    report = AlternationReport()
    history: list[dict[str, TargetReading]] = [candidate_b_readings(f) for f in frames]
    report.frames = len(history)
    report.targets = len({tid for readings in history for tid in readings})
    for i in range(1, len(history)):
        prev, cur = history[i - 1], history[i]
        for target_id, now in cur.items():
            before = prev.get(target_id)
            if before is None:
                if i >= 2 and target_id in history[i - 2]:
                    report.one_frame_gaps += 1
                continue
            report.consecutive_pairs += 1
            new_formula = abs(now.pressure - before.pressure)
            old_formula = abs(now.pressure - before.novelty)
            matches_new = _close(now.novelty, min(1.0, new_formula), NOVELTY_EPS)
            matches_old = _close(now.novelty, min(1.0, old_formula), NOVELTY_EPS)
            if matches_old and not matches_new:
                report.old_formula_pairs += 1
            elif matches_new and not matches_old:
                report.new_formula_pairs += 1
            else:
                report.ambiguous_pairs += 1
            if _close(now.pressure, before.pressure, STEADY_EPS):
                report.steady_pairs += 1
                if now.novelty > NOVELTY_EPS:
                    report.steady_pairs_with_novelty += 1
                    report.per_target_steady_with_novelty[target_id] = (
                        report.per_target_steady_with_novelty.get(target_id, 0) + 1
                    )
    return report


def render(report: AlternationReport) -> str:
    fraction = report.steady_novelty_fraction
    lines = [
        f"frames read: {report.frames}; Candidate B targets seen: {report.targets}",
        f"consecutive pairs: {report.consecutive_pairs}",
        f"steady pairs (pressure unchanged): {report.steady_pairs}",
        (
            "steady pairs that still carried novelty: "
            f"{report.steady_pairs_with_novelty}"
            + (f" ({fraction:.1%})" if fraction is not None else " (no steady pairs)")
        ),
        "  expect ~50% on frames written before the 2026-09-25 fix, ~0% after",
        (
            "formula fingerprint: "
            f"old={report.old_formula_pairs} new={report.new_formula_pairs} "
            f"ambiguous={report.ambiguous_pairs}"
        ),
        "  (ambiguous = both formulas give the same number, e.g. a first tick or zero pressure)",
        f"one-frame gaps (target vanished for one frame, then returned): {report.one_frame_gaps}",
    ]
    if report.per_target_steady_with_novelty:
        worst = sorted(report.per_target_steady_with_novelty.items(), key=lambda kv: -kv[1])[:10]
        lines.append("most affected targets: " + ", ".join(f"{t}={n}" for t, n in worst))
    return "\n".join(lines)


FRAMES_SQL = """
SELECT frame_json
FROM substrate_attention_frames
WHERE generated_at >= now() - (%s * interval '1 hour')
ORDER BY generated_at ASC
LIMIT %s
"""


def fetch_frames(dsn: str, *, hours: float, max_frames: int) -> Optional[list[dict[str, Any]]]:
    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(FRAMES_SQL, (float(hours), int(max_frames)))
            rows = cur.fetchall()
    finally:
        conn.close()
    frames: list[dict[str, Any]] = []
    for (raw,) in rows:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                continue
        if isinstance(raw, dict):
            frames.append(raw)
    return frames


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--max-frames", type=int, default=50_000)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    frames = fetch_frames(
        os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI),
        hours=args.hours,
        max_frames=args.max_frames,
    )
    if frames is None:
        print("UNKNOWN: could not open a read-only Postgres session", file=sys.stderr)
        return 2
    report = analyze(frames)
    if args.json:
        payload = asdict(report)
        payload["steady_novelty_fraction"] = report.steady_novelty_fraction
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
