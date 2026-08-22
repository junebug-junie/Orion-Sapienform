## Summary

Live-caught follow-up to #1817 (merged, deployed). Checking the deployed panel found that ~22 of ~32 showing cards already had a real `juniper` Resolve/Dismiss verdict — one from 11 days ago. `suppress_loop()`'s refractory suppression is only a 24h cooldown; `load_pending_loops()` never checked `attention_loop_outcome` directly, so once the cooldown lapsed the exact same already-judged trace row just qualified again. The docstring literally claimed the theme "won't re-ignite" — true for 24 hours only.

## Outcome moved

`load_pending_loops()` now excludes a trace row if the loop's most recent verdict is at least as new as that row. Ran the exact new query against production data before committing: **32 → 12 showing cards**, all 22 stale reappearances gone, the 2 genuine reopens (new activity *after* a prior close) correctly still present.

## Current architecture

`suppress_loop()` writes a 24h `substrate_reverie_refractory` row on Resolve/Dismiss. `load_pending_loops()`'s query only ever checked that refractory table for the panel's own exclusion — never `attention_loop_outcome` directly.

## Architecture touched

`services/orion-hub/scripts/attention_loops_store.py` (`load_pending_loops` query, `suppress_loop` docstring correction), `services/orion-hub/tests/test_attention_loops_reader.py` (SQL-text regression guard + `_Conn`/`_Engine` execute-capture).

## Files changed

- `services/orion-hub/scripts/attention_loops_store.py`: added `NOT EXISTS (... attention_loop_outcome ... o.created_at >= t.created_at)` to `load_pending_loops`'s query; corrected `suppress_loop`'s docstring.
- `services/orion-hub/tests/test_attention_loops_reader.py`: capture executed SQL for assertion; new test asserting the exclusion clause is present.

## Schema / bus / API changes

None (query-only change, no schema/contract touched).

## Env/config changes

None.

## Tests run

```
services/orion-hub/tests/test_attention_loops_{api,reader}.py
  + test_attention_{card_legibility,closure_e2e,loop_closure}.py    25 passed
```

Mock harness can't exercise real SQL WHERE-clause semantics (Postgres evaluates that, not Python) — the behavioral proof is the live query run against production data before commit (see below), not the mock. This is disclosed in the test file's own docstring, not silently assumed.

## Evals run

N/A — no eval harness applicable to a single SQL clause fix.

## Docker/build/smoke checks

Live query verification (before committing):

```sql
-- old query: 32 rows (22 already-verdicted stale reappearances included)
-- new query (with the fix's WHERE clause): 12 rows
--   -> all 22 stale reappearances gone
--   -> LLMGatewayService, Compact correctly still present (real new trace
--      activity postdates their prior verdict -- genuine reopen)
```

## Review findings fixed

- Finding: docstring's "mirrors `verdicts.py::load_terminal_verdict_loop_ids`" was imprecise — that function deliberately keeps `decayed_unattended` eligible for live coalition re-selection; this exclusion deliberately does not (different question: stale panel evidence vs. loop re-selection).
  - Fix: reworded to state the relationship precisely rather than claim equivalence.
  - Evidence: `d65b345bc`.
- One other candidate (loop_id/theme_key correlation mismatch across the `DISTINCT ON` grouping) was investigated by the reviewer and refuted — `stable_id`-based ids are stable across the query, no mismatch.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Concern: a loop that recurs frequently (like the "Compact" burst pattern from the original arc) will keep reopening after every Resolve/Dismiss as long as it keeps generating new trace rows — this is correct/intended (real new evidence should surface), but an operator clicking Dismiss repeatedly on a genuinely-recurring-but-low-value theme won't get lasting relief from this fix alone. Mitigation: that's what the implicit-decay digest (#1817) is for on the chat side, and `chronic_pressure` framing on the reverie side — orthogonal fixes, working together.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1819
