## Summary

- Design doc for #3442: curiosity keeps **lines of inquiry** across runs (question, runs, what was learned with refs, what is open, status active/parked/resumed/closed, park-until condition).
- Before a run, code builds a short brief from the ledger. After a run, the run writes its own update into its turn outcome, and code checks every ref.
- Park and resume use the existing kickoff and the existing event-want market. No new scheduler.
- Inquiry changes become experiences in Orion's now (#3453), register with the integration gate (#3454), feed projects (#3428) and the roundup (#3426).
- Two build paths with a decision rule for the 11-18 review: (b) a shared, arm-blind layer if the blinded three-arm test continues (self-inquiry line first); (a) the replica as PM if the test stops or the replica wins.
- Live 30-day baselines for re-investigation, building on earlier runs, tokens per run and dropped threads.

## Outcome moved

Design only. It gives Juniper measured baselines and a decision rule for 11-18. It also found and filed a live blinding risk (#3458).

## Current architecture

Curiosity runs get six separate hints from the past (continuation note, last-runs claims, priors, peer briefs, lived questions, revisions). Nothing holds a question across runs with refs, and nothing can wait for an event and resume.

## Architecture touched

None (doc only). Proposed: two Postgres tables, one pure module in `orion/curiosity/`, optional TurnOutcome fields, one Temporal Self source kind, Hub kickoff and turn-end reader.

## Files changed

- `docs/superpowers/specs/2026-11-15-curiosity-inquiry-manager-design.md`: the design.
- `docs/superpowers/pr-reports/2026-11-15-curiosity-inquiry-manager-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none (proposed only, see the doc).
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
Doc only. git diff --check: clean.
```

## Evals run

```text
Baseline measurement (read-only, 2026-11-15, window 10-16 09:43Z .. 11-15 09:43Z, 329 runs; queries in the doc appendix):
- blind repeat (near-same finding, cos >= 0.90, earlier run not named): investigate 5.4%, self-inquiry 16.0%
- builds on an earlier run: 59.6%
- continue_line threads picked up by the next run: 17.4% (236 threads)
- formed beliefs never tested later: 63.9%
- per run: median 29 calls, 1.63M prompt tokens (90.4% cached), 163k uncached, 41k completion
No metric was split by experiment arm.
```

## Docker/build/smoke checks

```text
Not applicable (doc only).
```

## Review findings fixed

- Design doc; no code review run. Self-check: every live number traced to a query in the appendix; threshold hand-checked on 5 real pairs.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: high. Concern: Temporal Self `curiosity_run` events store the arm next to the outcome, and PR #3453 prints both in the now. Mitigation: filed #3458 (opened), commented on #3453; recommended as P0.
- Severity: medium. Concern: the token saving from a brief is a hypothesis (UNVERIFIED); 90% of prompt tokens are already cached. Mitigation: target is measured, with a "learned per run not lower" guard.
- Severity: medium. Concern: whether curiosity admission reads event-want salience is UNVERIFIED. Mitigation: acceptance check P3-2; resume still works at the next normal run.
- Issues opened: #3458. Issues closed: none. Touched: #3375, #3396, #3438 (see doc section 8).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3459

🤖 Generated with [Claude Code](https://claude.com/claude-code)
