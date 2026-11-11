## Summary

- Design only (proposal mode) for step D of the unified-turn self-prediction arc (PR #2869): the intention ledger.
- Spec: `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`.
- Finding: "meant" and "did" are already recorded for every turn, and "left open" has two existing records. Only **promises** have no record.
- Proposal: the 5b finalize judge names promises by exact quote. Each promise opens an act on the existing act ledger with a new `promise_kept` signal and a 72 h horizon. A later judge marks it kept, or it expires as 0. Stance sees up to 3 open promises.
- No new table. The self-monitor organ grades promise keeping with no new grader.

## Outcome moved

None yet. This is a design. It names the eval (replay precision ≥ 0.8) and the settled outcome (keep rate, Fisher exact test, read on 2026-12-16) that decide whether the ledger helps.

## Current architecture

- Meant: `thought_decision.imperative` / `self_prediction`. 3,022 of 3,031 rows in 14 days have an imperative.
- Did: `harness_turn_trace.run_artifact`, settled into `prediction_turn_self_bet` and graded by the self-monitor organ.
- Left open: attention open loops are body-signal themes. All five loops stance cites most are "... prediction error". `closure.surprise_unresolved` is read only by reverie.
- Promised: nothing on the unified turn. The social-memory word-list detector has made 1 commitment in 67 rooms.

## Live data (14 days)

- 95 of the 446 chat-caller turns are Juniper's. 5 of those 95 held a forward promise (6 statements, 2 of them offers).
- Each promise left the 8-turn window within hours. A text search found a later reference for at most 1 of the 6.

## Architecture touched

None (docs only).

## Files changed

- `docs/superpowers/specs/2026-11-11-intention-ledger-design.md`: the design
- `docs/superpowers/pr-reports/2026-11-11-intention-ledger-design-pr.md`: this report

## Schema / bus / API changes

- Proposed only: `PromiseV1` plus `promises_made` / `promises_kept` on `FinalizeReflectionV1`, and `ActSignal += "promise_kept"`. Nothing is changed in this PR.

## Env/config changes

- None in this PR. Proposed: `ORION_PROMISE_LEDGER_ENABLED`, shipped ON.

## Tests run

```text
python -m pytest tests/test_no_personal_terms.py -q   (passed)
```

## Evals run

```text
None: design only. The spec defines replay eval D0 as the next patch.
```

## Docker/build/smoke checks

```text
Not applicable (docs only).
```

## Review findings fixed

- No code review: docs only. Live counts were checked twice against Postgres. The first draft counted a non-Juniper turn among Juniper's promises, and that count is now corrected.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium. Concern: the sample is small (about 3 promises a week), so the "does it help" test can stay inconclusive. Mitigation: the read date is pre-registered, and an inconclusive result is not a pass.
- Severity: medium. Concern: the same judge model makes and closes promises, so the keep rate could saturate at 100%. Mitigation: a substring gate, open-act-only closes, and a hand-checked close precision with an off switch for the "kept" mark.

## PR link

(added after the PR opens)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
