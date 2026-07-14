# PR report — reverie verdict-aware narration + refractory theme-key fix

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1055
Branch: `fix/reverie-verdict-aware-narration`
Status: **DONE**

## Summary

Two compounding bugs meant reverie could confidently narrate urgency about an
open loop a human had already resolved or dismissed via the Hub:

1. `chain.py`'s `theme_key_for()` keyed refractory suppression as
   `"loop:<id>"` — a format only that function ever wrote or read.
   `attention_loops_store.py`'s `suppress_loop()` (called on a human's
   Resolve/Dismiss action) and `attention_salience_trace.theme_key` both use
   the bare loop id. The two never matched, so a closed loop's
   chain-refractory suppression silently never took effect. **Live-broken in
   prod** — `ORION_REVERIE_CHAIN_ENABLED=true` is the live config.
2. Even with suppression working, reverie's narration prompt had no
   visibility into `attention_loop_outcome` verdicts at all — `OpenLoopV1`
   carries no outcome field, and that table was read only by the offline
   weight-refit script and Hub display routes.

This is changeset 1 of 2 from `docs/superpowers/specs/2026-07-14-reverie-narration-continuity-design.md`
(chain narration continuity — prior-step memory, next_focus wiring,
confidence/dwell framing — is changeset 2, not in this patch).

## Outcome moved

A human resolving/dismissing an attention loop in the Hub now actually
suppresses reverie chains from re-igniting on it. Reverie's narration is
grounded in recorded human verdicts instead of confabulating urgency about
already-closed loops.

## Root-cause discovery

Found while implementing the spec's changeset 1 (narration-only
verdict-awareness): reading `attention_loops_store.py` to design the outcome
lookup surfaced that a `suppress_loop()` mechanism already existed and was
*intended* to solve this exact problem end-to-end (Resolve → refractory
suppression → chain skips it) — it was just silently broken by a key-format
mismatch, not missing as a feature. Fixing that mismatch was folded into this
changeset since it's the same root problem class, thin, and directly in scope.

## Review findings fixed

Ran the `code-review` skill (level: high, 8 finder angles via subagents,
1-vote verify) against the diff. Fixed:

- **Row-shaping outside try/except**: `load_recent_loop_outcomes`'s docstring
  promised "never raises," but only the DB round-trip was guarded — a
  malformed row shape would propagate. Widened the try to cover shaping.
- **`age_days: None` could reach the prompt**: when `created_at` wasn't a
  usable datetime, the LLM could see a literal null and narrate "closed None
  days ago." Now the key is omitted rather than null.
- **Blocking sync DB call in the async event loop**: the new
  `load_recent_loop_outcomes` call had no `asyncio.to_thread` offload, unlike
  `chain.py`'s established convention for blocking DB/HTTP calls on this
  shared event loop. Wrapped it.
- **Prompt block rendered unconditionally**: the `ALREADY-SETTLED LOOPS`
  instruction appeared on every tick with any open loops, not just when a
  loop actually had a verdict. Gated on a new deterministic
  `has_settled_loops` flag.
- **No coverage of actual narration behavior**: original tests only checked
  context-dict shape, not what the model is actually told — a misworded or
  dropped instruction would have passed every test. Added a golden-prompt
  test rendering the real Jinja template.
- **Deploy-boundary orphaned rows** (found independently by 3 of the 8
  review angles): the theme-key format change has no backfill for rows
  already written under the old prefixed key. Added an idempotent one-time
  SQL backfill.

Not fixed, accepted and documented in the PR: `resonance_monitor.py`'s
in-memory tracked-theme set (built from orion-notify's pending-reason
strings) can emit one spurious "recovered" notification for a loop actively
paging at the exact deploy instant — self-heals on the next tick, judged not
worth code for a single cosmetic notification.

## Tests run

```text
pytest services/orion-thought/tests services/orion-thought/evals -q   → 162 passed
pytest orion/reverie/ -q                                              → 46 passed
pytest services/orion-hub/tests/test_reverie_observability_section.py -q → 8 passed
pytest services/orion-cortex-exec/tests/test_chat_stance_reverie_glimpse_projection.py -q → 11 passed
pytest services/orion-dream/tests/test_rem_compaction.py -q           → 15 passed
git diff --check                                                      → clean
```

## Restart required

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_reverie_theme_key_bare_id_backfill.sql

docker compose \
  --env-file .env \
  --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml \
  up -d --build
```

## Next

Changeset 2 (`feat/reverie-chain-continuity`: #2 prior-step memory, #3
next_focus, #6 salience margin, #7 dwell framing) is spec'd and pinned for
after this lands and is verified live.
