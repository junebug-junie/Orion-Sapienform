#!/usr/bin/env python3
"""Wire the missing half of the attention-loop label stream: implicit decay.

Context: `orion/substrate/attention/implicit_outcome.py::derive_implicit_verdicts()`
has existed since 2026-08-14 -- tested, correct, `decayed_unattended` documented as
one of `AttentionLoopOutcomeV1`'s three verdicts since the schema was written -- but
had zero live callers. `attention_loop_outcome` was 100% hand-clicked
(`resolved`/`dismissed` from Juniper's own Resolve/Dismiss in the Hub), which cannot
scale to judging a machine-generated candidate stream, and the Hub's pending-attention
panel (`services/orion-hub/scripts/attention_loops_store.py::load_pending_loops`) had
no expiry at all -- a loop left the panel only via that same hand click. This script
is the missing producer: it reads `attention_salience_trace`, finds loops that were
scored, never explicitly closed by a human, and then stopped being re-scored, and
labels them `decayed_unattended`.

`decayed_unattended` is NOT a suppression signal for live coalition SELECTION --
`orion/substrate/attention/verdicts.py::TERMINAL_VERDICTS` deliberately excludes it
("implicit non-engagement signal, not an explicit closure -- left eligible to
compete"), so a decayed loop can still be re-selected by a live reverie tick if the
underlying pressure is real.

**Scoped to `scope='chat'` only -- this is deliberate, not an oversight.** This
script ALSO writes a `substrate_reverie_refractory` suppression (same table/
mechanism as a human Dismiss, 24h cooldown). Code review (2026-08-21) caught
that an earlier version of this docstring claimed that table "is read only by
the Hub panel query, never by the live coalition" -- FALSE:
`services/orion-thought/app/chain.py::theme_key_for()`'s own docstring says
this is the SAME table a human's Resolve/Dismiss already suppresses real
reverie-chain reignition through, by deliberate pre-existing design ("A chain
must see the refractory suppression a human's Resolve/Dismiss action in the
Hub just wrote, or a closed loop keeps re-igniting chains indefinitely"). A
HUMAN'S explicit Resolve/Dismiss intentionally carrying that consequence is a
reasonable design (a person said "done with this"). This script's IMPLICIT,
machine-driven decay is not the same kind of act -- `decayed_unattended` means
"nobody engaged," not "a human confirmed this is closed," and chronic_pressure
(scope='reverie') loops are exactly the ones the pending-attention panel
already renders as ongoing system state, never as something needing
resolution (see services/orion-hub/README.md's `card_kind` section) -- auto-
suppressing THEIR chain reignition would be the false-closure-of-live-pressure
failure this whole feature-arc exists to prevent, just moved one layer down
into a different table's cross-service reader instead of the Hub API's 409
guard. Chat-scope loops carry no such consequence today (nothing in
`orion-thought` spawns a chain keyed to a chat-derived theme_key in the
ordinary case), so this script only ever touches `scope='chat'` rows.

Residual, pre-existing risk this narrowing does NOT eliminate (disclosed, not
fixed here -- a larger change than this script owns): chat and reverie loop
ids share the same `theme_key` namespace by design (`chain.py::theme_key_for`'s
own docstring) and both use the identical `stable_id("open-loop", <normalized
phrase>)` formula (`orion/substrate/attention/scoring.py`), so a
chat turn and a reverie signal that normalize to the exact same phrase collide
on theme_key. Scoping this script's SELECT to `scope='chat'` means it only
ever WRITES a decay-driven refractory suppression when the row it read was
itself chat-scoped, which is the fix that matters -- but does not change that
`attention_loops_store.py::latest_trace_for_theme`'s per-click lookup (used by
the Hub's Resolve/Dismiss guard) has no scope filter and could, on an exact
phrase collision, read whichever scope's row is most recent.

Deliberately does NOT publish to the `orion:attention:loop_outcome` bus channel the
way a human Resolve/Dismiss does (`services/orion-hub/scripts/bus_publish.py`) --
that channel's `consumer_services` list in `orion/bus/channels.yaml` is empty (no
live consumer), so publishing would only mean registering a new bus producer for an
event nothing reads.

Not a service loop -- a standalone script run on demand or via cron, same category as
`scripts/concept_relation_digest.py` (its structure mirrors this one).

Usage:
    POSTGRES_URI=postgresql://user:pass@host:port/db python scripts/attention_loop_decay_digest.py
    python scripts/attention_loop_decay_digest.py --postgres-uri postgresql://... [--dry-run] [--json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
# Running as `python scripts/attention_loop_decay_digest.py` puts scripts/ on
# sys.path[0], which shadows stdlib modules (same issue documented in
# scripts/check_inner_state_registry.py / scripts/concept_relation_digest.py).
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orion.substrate.attention.implicit_outcome import (  # noqa: E402
    DEFAULT_MIN_SILENCE,
    LoopObservation,
    derive_implicit_verdicts,
)

# Same 24h cooldown attention_loops_store.py::suppress_loop() already applies for a
# human Dismiss -- see that function's own docstring for why this constant is
# borrowed rather than reinvented.
_REFRACTORY_COOLDOWN = timedelta(hours=24)

_SELECT_TRACES_SQL = """
    SELECT theme_key, loop_id, salience, features, created_at
    FROM attention_salience_trace
    WHERE scope = 'chat'
    ORDER BY loop_id, created_at
"""

_SELECT_LATEST_VERDICTS_SQL = """
    SELECT DISTINCT ON (loop_id) loop_id, verdict
    FROM attention_loop_outcome
    ORDER BY loop_id, created_at DESC
"""

_INSERT_OUTCOME_SQL = """
    INSERT INTO attention_loop_outcome
        (outcome_id, loop_id, theme_key, verdict, actor, note,
         salience_at_close, weights_version, features_at_close, created_at)
    VALUES
        ($1, $2, $3, 'decayed_unattended', 'system:implicit_decay', $4, $5, $6, $7, $8)
    ON CONFLICT (outcome_id) DO NOTHING
"""

_UPSERT_REFRACTORY_SQL = """
    INSERT INTO substrate_reverie_refractory (theme_key, suppressed_until)
    VALUES ($1, $2)
    ON CONFLICT (theme_key)
    DO UPDATE SET suppressed_until = EXCLUDED.suppressed_until, updated_at = now()
"""


@dataclass
class DigestReport:
    themes_scanned: int
    decayed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"themes_scanned": self.themes_scanned, "decayed": self.decayed}


def _outcome_id(loop_id: str, last_seen: datetime) -> str:
    from orion.core.ids import stable_hash_id

    # Includes last_seen so a loop that decays, gets reused by a later candidate,
    # and decays again gets a distinct outcome_id per episode -- ON CONFLICT DO
    # NOTHING should only make a RE-RUN against the SAME episode idempotent, not
    # collapse two genuinely different decay events into one row.
    return stable_hash_id(
        "loopoutcome", ["decayed_unattended", "system:implicit_decay", loop_id, last_seen.isoformat()]
    )


async def _run_digest(
    postgres_uri: str, *, now: datetime, min_silence: timedelta, dry_run: bool
) -> DigestReport:
    import asyncpg

    conn = await asyncpg.connect(postgres_uri)
    try:
        trace_rows = [dict(r) for r in await conn.fetch(_SELECT_TRACES_SQL)]
        verdict_rows = [dict(r) for r in await conn.fetch(_SELECT_LATEST_VERDICTS_SQL)]
        report = await _apply_decisions(
            conn, trace_rows, verdict_rows, now=now, min_silence=min_silence, dry_run=dry_run
        )
    finally:
        await conn.close()
    return report


def build_observations(
    trace_rows: list[dict[str, Any]], verdict_rows: list[dict[str, Any]]
) -> list[LoopObservation]:
    """Shared by this digest AND check_attention_loop_decay_liveness.py -- the
    liveness gate needs the identical grouping to measure the same eligibility
    this script acts on, not a hand-reimplemented copy. Review caught a real bug
    from the first hand-reimplemented copy: it built `theme_key=loop_id` instead
    of reading the row's actual `theme_key` column (dormant today only because
    every current producer happens to set them equal)."""
    existing_verdict: dict[str, str] = {r["loop_id"]: r["verdict"] for r in verdict_rows}

    by_loop: dict[str, dict[str, Any]] = {}
    for row in trace_rows:
        loop_id = row["loop_id"]
        entry = by_loop.setdefault(
            loop_id,
            {"theme_key": row["theme_key"], "times": [], "last_salience": 0.0, "last_features": {}},
        )
        entry["times"].append(row["created_at"])
        # _SELECT_TRACES_SQL orders by (loop_id, created_at) -- matching the
        # grouping key used here (loop_id), not theme_key -- so the last row
        # seen for a given loop_id is genuinely its most recent, even in the
        # (currently unobserved) case where one loop_id's rows span more than
        # one theme_key. Keep its salience/features.
        entry["last_salience"] = float(row["salience"] or 0.0)
        features = row["features"]
        if isinstance(features, str):
            try:
                features = json.loads(features or "{}")
            except Exception:
                features = {}
        entry["last_features"] = dict(features or {})

    return [
        LoopObservation(
            loop_id=loop_id,
            theme_key=entry["theme_key"],
            trace_times=entry["times"],
            last_salience=entry["last_salience"],
            last_features=entry["last_features"],
            existing_verdict=existing_verdict.get(loop_id),
        )
        for loop_id, entry in by_loop.items()
    ]


def eligible_verdicts(
    observations: list[LoopObservation], *, now: datetime, min_silence: timedelta
) -> list:
    """derive_implicit_verdicts() results, further restricted to loops that have
    NEVER received any outcome at all -- shared by this digest and the liveness
    gate so both agree on "eligible" the same way.

    derive_implicit_verdicts() deliberately keeps decayed_unattended out of its
    own TERMINAL_VERDICTS -- a real design choice for the label stream (a loop
    CAN be re-derived as decayed_unattended indefinitely if it stays silent; see
    that function's own docstring). Left unfiltered here, a repeat digest run
    would re-report (though not re-INSERT -- outcome_id is idempotent, see
    _outcome_id) every already-decayed loop as "decayed" forever, which is
    misleading in a cron log and is exactly what made
    check_attention_loop_decay_liveness.py falsely report STALE right after a
    successful run (confirmed live 2026-08-21). A loop only needs writing once
    per silence episode -- skip anything that already has any outcome.
    """
    existing = {o.loop_id: o.existing_verdict for o in observations}
    verdicts = derive_implicit_verdicts(observations, now=now, min_silence=min_silence)
    return [v for v in verdicts if existing.get(v.loop_id) is None]


async def _apply_decisions(
    conn, trace_rows: list[dict[str, Any]], verdict_rows: list[dict[str, Any]], *,
    now: datetime, min_silence: timedelta, dry_run: bool,
) -> DigestReport:
    observations = build_observations(trace_rows, verdict_rows)
    verdicts = eligible_verdicts(observations, now=now, min_silence=min_silence)
    last_seen_by_loop = {o.loop_id: max(o.trace_times) for o in observations}

    decayed: list[dict[str, Any]] = []
    if not dry_run:
        async with conn.transaction():
            for v in verdicts:
                until = now + _REFRACTORY_COOLDOWN
                await conn.execute(
                    _INSERT_OUTCOME_SQL,
                    _outcome_id(v.loop_id, last_seen_by_loop[v.loop_id]),
                    v.loop_id,
                    v.theme_key,
                    v.reason,
                    v.salience_at_close,
                    "gwt-coalition-v1",
                    json.dumps(v.features_at_close),
                    now,
                )
                await conn.execute(_UPSERT_REFRACTORY_SQL, v.theme_key, until)

    for v in verdicts:
        decayed.append(
            {
                "loop_id": v.loop_id,
                "theme_key": v.theme_key,
                "silence_hours": round(v.silence.total_seconds() / 3600, 1),
                "reason": v.reason,
            }
        )

    return DigestReport(themes_scanned=len(observations), decayed=decayed)


def _print_report(report: DigestReport, *, dry_run: bool) -> None:
    verb = "would decay" if dry_run else "decayed"
    print(
        f"attention_loop_decay_digest: {report.themes_scanned} chat-scope loop(s) scanned, "
        f"{len(report.decayed)} {verb}"
    )
    for item in report.decayed:
        print(f"  [{item['theme_key']}] silent {item['silence_hours']}h -- {item['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument(
        "--min-silence-hours",
        type=float,
        default=DEFAULT_MIN_SILENCE.total_seconds() / 3600.0,
        help="floor on silence before a loop is called decayed (default matches suppress_loop's 24h cooldown).",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would decay; write nothing.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of prose.")
    args = parser.parse_args(argv)

    if not args.postgres_uri.strip():
        print(
            "attention_loop_decay_digest: no --postgres-uri given and $POSTGRES_URI is unset. "
            "Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    now = datetime.now(timezone.utc)
    try:
        report = asyncio.run(
            _run_digest(
                args.postgres_uri,
                now=now,
                min_silence=timedelta(hours=args.min_silence_hours),
                dry_run=args.dry_run,
            )
        )
    except Exception as exc:
        print(f"attention_loop_decay_digest: run failed -- {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict()))
    else:
        _print_report(report, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
