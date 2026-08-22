#!/usr/bin/env python3
"""Clear the accumulated ai-town backlog out of the crystallization governor queue.

One-shot backlog cleanup for the 2026-08-14 finding. The formation gate
(orion/memory/crystallization/formation_policy.py) stops NEW ai-town windows
from ever reaching the queue; this handles what already piled up behind it.

Targets exactly the crystallizations the runtime gate itself would classify as
external -- same build_crystallization_from_window() + resolve_formation_policy()
predicate, resolved against live chat_history_log platforms. It deliberately
does not use a looser "window contains any ai-town turn" rule: 11 of the live
proposals mix NPC dialogue with Juniper's own words, and those must survive for
review rather than be swept up with the noise.

Follows CLAUDE.md section 14 (backfill protocol): snapshots every affected row
before writing, logs progress, and writes a report plus a before/after CSV.

    # See what would happen. Writes the snapshot, touches no rows.
    python3 scripts/bulk_reject_aitown_proposals.py

    # Actually apply.
    python3 scripts/bulk_reject_aitown_proposals.py --apply

Job directory: /tmp/aitown-crystallization-purge/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.memory.consolidation_gate import ConsolidationGateResult  # noqa: E402
from orion.memory.crystallization.formation_policy import (  # noqa: E402
    DEFAULT_DISCARD_PLATFORMS,
    FormationPolicy,
    resolve_formation_policy,
)
from orion.memory.crystallization.intake_consolidation_window import (  # noqa: E402
    build_crystallization_from_window,
)
from scripts.smoke_aitown_crystallization_gate import QUERY  # noqa: E402

JOB_DIR = Path("/tmp/aitown-crystallization-purge")
DEFAULT_DSN = os.environ.get(
    "ORION_SQL_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
)
ACTOR = "bulk_aitown_purge"
REASON = (
    "ai-town NPC dialogue: window is unanimously external-platform. Backlog cleanup for the "
    "2026-08-14 governor-queue pollution finding; new windows are gated at formation."
)
# Snapshot-size ceiling from CLAUDE.md section 14. The live backlog is ~621 rows,
# three orders of magnitude under this -- the guard exists so a future run against
# an unexpectedly huge queue stops and asks instead of silently proceeding.
MAX_ROWS_WITHOUT_APPROVAL = 100_000


def _log(handle, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line)
    handle.write(line + "\n")
    handle.flush()


def _platform_allowlist() -> frozenset[str]:
    """The SAME allowlist the running service uses, not the module default.

    resolve_formation_policy() falls back to DEFAULT_DISCARD_PLATFORMS when
    given nothing, so calling it bare silently ignores
    MEMORY_FORMATION_DISCARD_PLATFORMS (renamed 2026-08-16 from
    MEMORY_FORMATION_AUTO_ACTIVATE_PLATFORMS). An operator who had set that key to
    the empty string -- documented in .env_example as disabling the gate -- would
    still have had 599 rows mass-rejected by a script whose docstring promised it
    mirrored the runtime predicate.
    """
    raw = os.environ.get("MEMORY_FORMATION_DISCARD_PLATFORMS")
    if raw is None:
        return DEFAULT_DISCARD_PLATFORMS
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def _classify(rows, *, platforms: frozenset[str]) -> tuple[list[dict], list[dict]]:
    """(external, keep) -- external is what the runtime gate would discard."""
    grouped: dict[str, dict] = {}
    for cid, window_id, subject, created_at, kind, corr, prompt, response, platform in rows:
        entry = grouped.setdefault(
            cid,
            {
                "crystallization_id": cid,
                "window_id": window_id,
                "subject": subject,
                "created_at": created_at,
                "kind": kind,
                "turns": [],
            },
        )
        if corr is None:
            continue
        entry["turns"].append(
            {
                "correlation_id": corr,
                "prompt": prompt or "",
                "response": response or "",
                "spark_meta": {},
                "source_platform": platform,
            }
        )

    external, keep = [], []
    for entry in grouped.values():
        crys = build_crystallization_from_window(
            memory_window_id=entry["window_id"] or "unknown",
            turns=entry["turns"],
            gate=ConsolidationGateResult(
                action="propose", dominant_shift="STANCE", grammar_event_ids=[]
            ),
        )
        # The row's REAL kind, not whatever dominant_shift="STANCE" reconstructs.
        # DISCARD now applies to every kind (2026-08-16), so this no longer
        # matters for the discard decision itself, but the replay should still
        # reflect what was actually stored, not the STANCE reconstruction.
        if entry.get("kind"):
            crys.kind = entry["kind"]
        entry["resolved_platform"] = crys.provenance.get("source_platform")
        policy, _ = resolve_formation_policy(crys, discard_platforms=platforms)
        (external if policy == FormationPolicy.DISCARD else keep).append(entry)
    return external, keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--apply", action="store_true", help="actually write; default is dry-run")
    args = ap.parse_args()

    import psycopg2

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with (JOB_DIR / "progress.log").open("a") as progress:
        _log(progress, f"start apply={args.apply} dsn={args.dsn.rsplit('@', 1)[-1]}")

        conn = psycopg2.connect(args.dsn)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                cur.execute(QUERY)
                rows = cur.fetchall()
            platforms = _platform_allowlist()
            _log(progress, f"platform allowlist={sorted(platforms) or '(empty -- gate disabled)'}")
            external, keep = _classify(rows, platforms=platforms)
            total = len(external) + len(keep)
            _log(progress, f"proposed={total} external={len(external)} keep={len(keep)}")

            if len(external) > MAX_ROWS_WITHOUT_APPROVAL:
                _log(progress, f"ABORT: {len(external)} rows exceeds the {MAX_ROWS_WITHOUT_APPROVAL} ceiling")
                return 2
            if not external:
                _log(progress, "nothing to do")
                return 0

            # 1. SNAPSHOT -- full current row state for everything about to change,
            #    written and flushed before any UPDATE is issued.
            ids = [e["crystallization_id"] for e in external]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT row_to_json(t) FROM (SELECT * FROM memory_crystallizations "
                    "WHERE crystallization_id = ANY(%s::uuid[])) t",
                    (ids,),
                )
                snapshot = [r[0] for r in cur.fetchall()]
            snap_path = JOB_DIR / "snapshot.json"
            snap_path.write_text(json.dumps(snapshot, indent=2, default=str))
            _log(progress, f"snapshot rows={len(snapshot)} -> {snap_path}")

            with (JOB_DIR / "before_after.csv").open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["crystallization_id", "window_id", "platform", "status_before", "status_after", "subject"]
                )
                for e in external:
                    writer.writerow(
                        [
                            e["crystallization_id"],
                            e["window_id"],
                            e["resolved_platform"],
                            "proposed",
                            "rejected" if args.apply else "proposed (dry-run)",
                            " ".join((e["subject"] or "").split())[:160],
                        ]
                    )

            if not args.apply:
                _log(progress, "DRY RUN -- no rows written. Re-run with --apply.")
                errors = 0
            else:
                errors = 0
                with conn.cursor() as cur:
                    for i, cid in enumerate(ids, start=1):
                        # SAVEPOINT per row, not a bare try/except. psycopg2 puts
                        # the whole transaction into INERROR on the first failing
                        # statement: without this, every later execute would raise
                        # InFailedSqlTransaction (caught, counted, logged as a
                        # per-row error) and the final commit() would be silently
                        # converted to a ROLLBACK by PostgreSQL *without raising*.
                        # The run would report "committed", write verdict: APPLIED,
                        # and have changed nothing. Rolling back to the savepoint
                        # confines a bad row to itself and keeps the transaction
                        # usable, so a partial-error run really does commit the
                        # rows the report claims.
                        cur.execute("SAVEPOINT row_sp")
                        try:
                            cur.execute(
                                "UPDATE memory_crystallizations SET status='rejected', "
                                "updated_at=now() WHERE crystallization_id=%s::uuid "
                                "AND status='proposed'",
                                (cid,),
                            )
                            # Audit trail: the queue's own history table, same op
                            # a human Reject writes, with a non-human actor so the
                            # two are never confused in a later review-behavior
                            # analysis.
                            cur.execute(
                                "INSERT INTO memory_crystallization_history "
                                "(crystallization_id, op, actor, before, after, reason) "
                                "VALUES (%s::uuid, 'reject', %s, %s::jsonb, %s::jsonb, %s)",
                                (
                                    cid,
                                    ACTOR,
                                    json.dumps({"status": "proposed"}),
                                    json.dumps({"status": "rejected"}),
                                    REASON,
                                ),
                            )
                        except Exception as exc:  # noqa: BLE001
                            errors += 1
                            cur.execute("ROLLBACK TO SAVEPOINT row_sp")
                            _log(progress, f"ERROR cid={cid} {exc}")
                        else:
                            cur.execute("RELEASE SAVEPOINT row_sp")
                        if i % 100 == 0 or i == len(ids):
                            pct = 100.0 * i / len(ids)
                            _log(
                                progress,
                                f"reject {i}/{len(ids)} ({pct:.1f}%) errors={errors}",
                            )
                conn.commit()
                _log(progress, "committed")

            with conn.cursor() as cur:
                cur.execute("SELECT status, count(*) FROM memory_crystallizations GROUP BY 1")
                after_counts = dict(cur.fetchall())

            # Read-back check, not a restatement of what we intended. The report
            # must not be able to say APPLIED while the database disagrees --
            # that is exactly the failure the savepoints above prevent, so prove
            # it rather than trusting it.
            #
            # Scoped to the ids we touched, NOT a global `proposed` count. The
            # consolidation worker keeps producing proposals while this runs, so
            # a global comparison against a pre-run count fails the moment one
            # real conversation lands mid-run -- reporting APPLY FAILED
            # VERIFICATION and exit 1 over a run where every intended row
            # committed correctly. That already happened once in this queue's
            # history ("23, not 22, because a real conversation landed").
            verified = True
            if args.apply:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT count(*) FROM memory_crystallizations "
                        "WHERE crystallization_id = ANY(%s::uuid[]) AND status <> 'rejected'",
                        (ids,),
                    )
                    not_rejected = cur.fetchone()[0]
                verified = not_rejected == 0
                if not verified:
                    _log(
                        progress,
                        f"VERIFY FAILED: {not_rejected} of the {len(ids)} targeted rows "
                        "are not 'rejected' after commit",
                    )
                else:
                    _log(progress, f"verified all {len(ids)} targeted rows are rejected")

            report = JOB_DIR / "report.md"
            report.write_text(
                f"""# ai-town crystallization purge

- verdict: {("APPLIED" if verified else "APPLY FAILED VERIFICATION") if args.apply else "DRY RUN"}
- proposed before: {total}
- classified external (unanimous ai-town window): {len(external)}
- kept for review: {len(keep)}
- errors: {errors}
- post-apply read-back verified: {verified if args.apply else "n/a (dry run)"}
- needs another pass: {"yes" if (errors or not verified) else "no"}

## Outcome

Governor queue goes from {total} to {len(keep)} items. Everything removed had a
window in which *every* turn came from ai-town; the {len(keep)} survivors each
contain at least one real turn with Juniper and were deliberately left alone.

## Status counts after

{json.dumps(after_counts, indent=2)}

## Kept for review

{chr(10).join(f"- `{e['crystallization_id'][:8]}` {' '.join((e['subject'] or '').split())[:110]}" for e in keep)}

## Files

- `{JOB_DIR}/snapshot.json` — full pre-change row state for every affected row
- `{JOB_DIR}/before_after.csv`
- `{JOB_DIR}/progress.log`

## Reversal

Rows were status-flipped, not deleted; nothing cascaded. To undo:

```sql
UPDATE memory_crystallizations SET status='proposed'
WHERE status='rejected' AND crystallization_id IN (
  SELECT crystallization_id FROM memory_crystallization_history
  WHERE actor = '{ACTOR}');
```
"""
            )
            _log(progress, f"report -> {report}")
            _log(progress, f"after={after_counts}")
            return 1 if (errors or not verified) else 0
        finally:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
