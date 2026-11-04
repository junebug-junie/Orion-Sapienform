# PR: give the field-note turn's harness motor a real budget (420 -> 900 s)

## Summary

- The first live field-note run, `pr-field-note-3104`, failed. The harness motor stopped after 59.99 s.
- Cause: Hub keeps a finalize reserve (`HUB_CURIOSITY_HELD_TURN_FINALIZE_RESERVE_SEC=330`) out of every
  held turn. A 420 s note turn reaches Hub as 390 s, so the motor got 390 - 330 = 60 s. The cold
  claude spawn alone took 58 s to its first event.
- Fix: `PR_FIELD_NOTES_TURN_TIMEOUT_SEC` goes from 420 to 900, so the motor gets 540 s. The value is a
  cap, not a cost: a note that finishes early uses only what it uses.
- A new test reads both services' shipped defaults and fails if the motor budget falls under 300 s.

## Outcome moved

Field-note turns can now finish. Before this fix, every attempt of every note ran into the same 60 s wall.

## Current architecture

`app/pr_field_note_graph.py` sends `timeout_sec - 30` (`TURN_REPLY_GRACE_SEC`) to Hub. Hub's
`held_turn_fcc_budget_sec` gives the motor that limit minus the finalize reserve
(`services/orion-hub/scripts/curiosity_investigation.py`). The motor's timer starts when the motor
starts, after the 100 s thought/stance pass. So the stance time did not cause this. The reserve did.

## Architecture touched

Only the orion-durable-runs config default. No code path changed. No schema, bus or Hub change.
Skipping the stance pass for this mechanical turn was considered and rejected: `execute_unified_turn`
has no clean flag for it, and stance owns the defer/refuse gates. That would be a Hub design change.

## Files changed

- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: 420 -> 900, with the reason.
- `services/orion-durable-runs/tests/test_pr_field_note_budget.py` (new): the regression gate.
- `services/orion-durable-runs/app/run_acts.py`: the horizon basis text (3 x 900 s + backoff still fits).
- `services/orion-durable-runs/README.md`, the spec's "Patch 2 as built": the new number and why.

## Schema / bus / API changes

- None.

## Env/config changes

- Changed default: `PR_FIELD_NOTES_TURN_TIMEOUT_SEC` 420 -> 900.
- local `.env` synced: the sync script reported the key as diverged (local 420), so I set that one key
  to 900 by hand in `services/orion-durable-runs/.env`.
- Skipped keys: none.

## Tests run

```text
pytest tests/test_pr_field_note_budget.py tests/test_pr_field_note_poll.py tests/test_pr_field_note_graph.py tests/test_run_acts.py
  65 passed, 8 skipped
test_the_old_default_is_exactly_the_live_failure: motor_budget(420, 330) == 60
```

## Evals run

```text
None: this is a config bound. The live check below is the proof.
```

## Docker/build/smoke checks

```text
No build needed for the default. The live .env now has 900.
Live evidence of the failure: governor log corr 6dc7f360-...: spawn_or_acquire_ms=4196 first_event_ms=57922,
"fcc turn timed out after 59.99s", steps=7, draft_len=0.
```

## Review findings fixed

REVIEW_TBD

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: the worst case for one note's agent-lane time rises from about 3 x 7 min to about 3 x 15 min.
  - Mitigation: one note runs at a time, and the measured cost (`scripts/pr_field_note_cost.py`) will show
    the real time per note.
- Severity: low.
  - Concern: a run submitted before the restart keeps 420 in its brief.
  - Mitigation: none is in flight now. `pr-field-note-3104` gets its second run from the poller about an
    hour after it failed, with the new value.

## PR link

PR_TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
