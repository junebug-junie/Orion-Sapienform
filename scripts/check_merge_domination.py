#!/usr/bin/env python3
"""Ratchet gate for merge domination: has a NEW low-resolution input started
masking a high-resolution one?

    POSTGRES_URI=postgresql://user:pass@host:port/db \
        python scripts/check_merge_domination.py --gate

Runs `orion.field.commensurability.analyse_field_history()` over recent real
`substrate_field_state` ticks and compares the findings to a committed
baseline. A finding on a merge point that is NOT in the baseline fails the
gate. Findings that disappear are reported as improvements and never fail.

WHY THIS EXISTS
---------------
Two instances of this defect are on record, and BOTH were caught by a person
reading a diff:

- `thermal_pressure` (39 distinct values) beat capability `pressure` (1,895
  distinct) on 91.76% of ticks until 2026-08-11. Deleting one routing entry
  took `resource_pressure` from 128 to 1,895 distinct values.
- `node:substrate.codebase` won the `prediction_error` merge on 68.7% of
  6,000 ticks on 6 informative distinct values, masking three continuously
  varying signals. Found 2026-08-14, fixed by the merge staleness rule.

Human review is a latent check that has already failed twice by not existing
in time. CLAUDE.md 4: when a latent failure repeats, turn it into
deterministic code.

NOT IN CI, AND WHY
------------------
This needs live Postgres, so it cannot join `.github/workflows/
orion-static-gates.yml` -- that workflow deliberately installs only
pydantic/pydantic-settings/PyYAML and already excludes four checks for exactly
this reason. It belongs to the same family as `check_activation_saturation.py`
and `check_concept_relation_digest_liveness.py`: a `make` target run against
the real host.

Run it on a schedule the same way `concept-relation-digest` is run -- a host
crontab entry. Suggested:

    17 * * * * cd /mnt/scripts/Orion-Sapienform && \
        POSTGRES_URI=... .venv/bin/python scripts/check_merge_domination.py \
        --gate >> /tmp/merge-domination/gate.log 2>&1

Until that entry exists this gate only runs when someone types it, which is
weaker than the failure it replaces deserves. Stated plainly rather than
implied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BASELINE_PATH = REPO_ROOT / "config" / "metrics" / "merge_domination_baseline.json"

DEFAULT_TICKS = 6000


def _load_states(postgres_uri: str, limit: int):
    import asyncio

    import asyncpg

    from orion.schemas.field_state import FieldStateV1

    async def _run():
        conn = await asyncpg.connect(postgres_uri)
        try:
            rows = await conn.fetch(
                "select field_json from substrate_field_state "
                "order by created_at desc limit $1",
                limit,
            )
        finally:
            await conn.close()
        return rows

    rows = asyncio.run(_run())
    states = []
    for row in rows:
        payload = row["field_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        try:
            states.append(FieldStateV1.model_validate(payload))
        except Exception:
            # A tick that will not validate is a schema problem, not a merge
            # problem -- counted and reported, never silently dropped.
            continue
    states.reverse()  # oldest first; refresh/decay detection is order-sensitive
    return states, len(rows) - len(states)


def _key(analysis) -> str:
    return f"{analysis.layer}::{analysis.merge_id}"


def _finding_payload(analysis) -> dict:
    return {
        "winner": analysis.winner,
        "winner_share": round(analysis.winner_share, 4),
        "winner_distinct_values": analysis.winner_distinct_values,
        "masked": sorted(m.name for m in analysis.masked),
    }


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text())
    return data.get("known_dominated_merges", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postgres-uri",
        default=os.getenv("POSTGRES_URI", ""),
        help="Postgres DSN. Defaults to $POSTGRES_URI (e.g. services/orion-hub/.env).",
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                        help=f"How many recent ticks to analyse (default {DEFAULT_TICKS}).")
    parser.add_argument("--gate", action="store_true",
                        help="Exit 1 if a merge not in the baseline is dominated.")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Rewrite the baseline from the current findings.")
    args = parser.parse_args(argv)

    if not args.postgres_uri:
        print(
            "check_merge_domination: no --postgres-uri given and $POSTGRES_URI is unset. "
            "Check services/orion-hub/.env for POSTGRES_URI.",
            file=sys.stderr,
        )
        return 2

    from orion.field.commensurability import analyse_field_history
    from orion.field.credit_integrity import (
        analyse_credit_integrity,
        load_credited_dimensions,
    )

    states, unparsed = _load_states(args.postgres_uri, args.ticks)
    if len(states) < 100:
        print(
            f"check_merge_domination: only {len(states)} usable ticks "
            f"(asked for {args.ticks}). Too few to judge a win share; not gating.",
            file=sys.stderr,
        )
        return 2

    # Second, unrelated question on the same expensive state load: could a
    # feedback-credited dimension currently be fooled by a producer outage?
    # See orion/field/credit_integrity.py. Reported alongside rather than in a
    # separate cron entry, because loading the states is the costly part and
    # both questions are "is a number about to mislead a consumer".
    credited, feedback_window = load_credited_dimensions()
    credit = analyse_credit_integrity(states, credited, feedback_window)

    results = analyse_field_history(states)
    findings = {_key(r): r for r in results if r.is_finding}
    baseline = _load_baseline()

    new = {k: v for k, v in findings.items() if k not in baseline}
    resolved = [k for k in baseline if k not in findings]
    # A known merge whose WINNER changed is not the same finding any more.
    changed = {
        k: v
        for k, v in findings.items()
        if k in baseline and baseline[k].get("winner") != v.winner
    }

    if args.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Ratchet baseline for scripts/check_merge_domination.py --gate. "
                        "May shrink, never grow. Regenerate deliberately with "
                        "--update-baseline when a fix lands or a finding is accepted."
                    ),
                    "_finding_comment": (
                        "A merge where one source wins the majority of non-tied merges "
                        "while carrying an order of magnitude fewer informative distinct "
                        "values than a contender that never wins. See "
                        "orion/field/commensurability.py."
                    ),
                    "ticks_analysed": len(states),
                    "known_dominated_merges": {
                        k: _finding_payload(v) for k, v in sorted(findings.items())
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"baseline rewritten: {len(findings)} known dominated merge(s)")
        return 0

    if args.json:
        print(json.dumps({
            "ticks_analysed": len(states),
            "unparsed_ticks": unparsed,
            "merge_points": len(results),
            "feedback_credit": {
                "policy_window_sec": credit.window_seconds,
                "samples": credit.channels,
                "source_kind": credit.source_kind,
                "longest_unbacked_run": credit.longest_unbacked_run,
                "findings": [f.describe() for f in credit.findings],
            },
            "findings": {k: _finding_payload(v) for k, v in sorted(findings.items())},
            "new": sorted(new),
            "resolved": sorted(resolved),
            "winner_changed": sorted(changed),
        }, indent=2))
    else:
        print(f"analysed {len(states)} ticks across {len(results)} merge points"
              + (f" ({unparsed} unparseable)" if unparsed else ""))
        for key, analysis in sorted(findings.items()):
            mark = "NEW " if key in new else ("CHANGED " if key in changed else "known ")
            print(f"  {mark}{key}")
            print(f"      winner {analysis.winner!r} takes {analysis.winner_share:.1%} of "
                  f"{analysis.sample_count - analysis.tied_count} decided merges on "
                  f"{analysis.winner_distinct_values} informative distinct values")
            for m in sorted(analysis.masked, key=lambda c: -c.distinct_values):
                print(f"        masked: {m.name} ({m.distinct_values} distinct, "
                      f"wins {m.win_share:.1%})")
        for key in sorted(resolved):
            print(f"  RESOLVED {key} -- no longer dominated. "
                  f"Run --update-baseline to bank it.")

    if not args.json:
        print(
            f"\nfeedback-credit watch: {len(credit.findings)} finding(s) "
            f"(policy window {credit.window_seconds:.0f}s)"
        )
        for dim in sorted(credit.channels):
            kinds = credit.source_kind.get(dim, {})
            print(
                f"    {dim:22} samples={credit.channels[dim]:>6} "
                f"node={kinds.get('node', 0):>6} capability={kinds.get('capability', 0):>6} "
                f"longest_unbacked={credit.longest_unbacked_run.get(dim, 0)}"
            )
        for finding in credit.findings:
            print(f"    FINDING {finding.describe()}")
        if credit.findings:
            print(
                "    REPORT-ONLY, same as the merge-domination gate above: this is\n"
                "    the precondition for a real feedback-loop guard (CLAUDE.md 0A\n"
                "    proposal mode), not the guard itself. Each finding names its\n"
                "    evidence ([timestamp] = a real node write, [contribution] = a\n"
                "    real diffusion contribution) so it can be checked by hand."
            )

    if args.gate and (new or changed):
        print("\nmerge domination gate: FAIL", file=sys.stderr)
        for key in sorted(new):
            print(f"  new dominated merge: {key}", file=sys.stderr)
        for key in sorted(changed):
            print(
                f"  winner changed on a known merge: {key} "
                f"({baseline[key].get('winner')} -> {findings[key].winner})",
                file=sys.stderr,
            )
        return 1
    if args.gate:
        print("\nmerge domination gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
