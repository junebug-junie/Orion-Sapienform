## Summary

Full fix for the Hub's "Cognitive Loops" / pending-attention panel, which was showing every card — real substrate signals and LLM-probe garbage alike — with the identical boilerplate "This other has stayed active without resolution," and never expiring anything.

- Persist `why_it_matters`/`target_type` from `OpenLoopV1` into `attention_salience_trace` (both producers computed these and dropped them before storage — that's why every card showed the same sentence).
- Split `PendingAttentionCardV1.card_kind` into `resolvable` (chat, human-actionable) vs `chronic_pressure` (reverie/substrate-broadcast, re-selected every tick by design) — chronic cards render without Resolve/Dismiss; the API rejects those verbs on one with `409`.
- Wire `orion/substrate/attention/implicit_outcome.py::derive_implicit_verdicts()` (tested, correct, zero live callers since 2026-08-14) into a real cron digest + liveness gate — **scoped to `scope='chat'` only**, after live-verifying that `substrate_reverie_refractory` also gates real reverie-chain reignition in `orion-thought`, not just this panel.
- Add a structural floor to `current_turn_llm_signals.py`'s LLM-probe parser: confirmed live, hours after the old regex detector was already replaced, the model still returned bare single words ("bus", "Glad", "Compact", "Interesting") as candidates — the same failure mode one step removed.

## Outcome moved

- Every new `attention_salience_trace` row (both scopes) now carries real, distinct `why_it_matters`/`target_type` instead of always falling back to the same sentence.
- Live-verified: digest run against real data decayed 1 genuinely stale chat-scope loop and correctly leaves 27 chat-scope + all reverie-scope loops untouched; liveness gate reports OK.
- A chronic reverie loop my own earlier (pre-fix) digest run had wrongly suppressed was found and remediated live (see "Live verification" below) — its 24h reverie-chain-reignition block was deleted.

## Current architecture

`services/orion-hub/scripts/attention_loops_store.py::load_pending_loops()` read `attention_salience_trace`, reconstructed `OpenLoopV1` with `why_it_matters`/`target_type` always empty/default (never persisted), and had no expiry — a loop left the panel only via a human's Resolve/Dismiss click, forever. `orion/substrate/attention/implicit_outcome.py::derive_implicit_verdicts()` existed, tested, with zero live callers. `current_turn_llm_signals.py` replaced a deleted regex detector (`_PROPER_RE` matching any capitalized word) with an LLM probe, but nothing floored its output beyond a length check.

## Architecture touched

- `orion/schemas/attention_salience.py` — `AttentionSalienceTraceV1` gets `why_it_matters`/`target_type`; `PendingAttentionCardV1` gets `card_kind` (`PendingCardKindV1`).
- `services/orion-sql-db/manual_migration_attention_salience_trace.sql` — additive `ALTER TABLE ADD COLUMN IF NOT EXISTS`.
- `services/orion-cortex-exec/app/{chat_attention_salience_trace,current_turn_llm_signals}.py` — chat-scope producer + LLM-probe floor.
- `services/orion-thought/app/{reverie,store}.py` — reverie-scope producer.
- `services/orion-hub/scripts/{attention_loops_store,attention_loops_routes}.py` — consumer, `card_kind_for_scope`, `latest_trace_for_theme`, 409 guard.
- `services/orion-hub/static/js/{app,cognitive-loop-card}.js` + `templates/index.html` — chronic vs resolvable rendering, extracted to a testable pure module.
- `scripts/attention_loop_decay_digest.py` + `scripts/check_attention_loop_decay_liveness.py` — new cron digest + liveness gate, chat-scope only.
- `Makefile` — `attention-loop-decay-digest`, `check-attention-loop-decay-liveness` targets.

## Files changed

See commit messages (4 commits) — each documents its own rationale in detail, including two rounds of code-review findings and fixes.

## Schema / bus / API changes

- Added: `attention_salience_trace.why_it_matters`, `.target_type` (additive, defaulted, backward-compatible). `PendingAttentionCardV1.card_kind` (additive, defaulted `"resolvable"`).
- Removed: `attention_loops_store.py::latest_scope_for_theme` (folded into new `latest_trace_for_theme`, which also replaces the old `latest_salience_for_theme`'s second query — `latest_salience_for_theme` itself kept as a thin wrapper for backward compatibility).
- Behavior changed: `POST /api/attention/loops/{id}/resolve|dismiss` now returns `409` for a `chronic_pressure` (reverie-scope) loop.
- Compatibility notes: no breaking changes to existing consumers; all new fields are additive with safe defaults.

## Env/config changes

None. No new env keys.

## Tests run

```
# Isolated per-service runs (this repo's cross-service `scripts`/`app` package
# collision means a combined run across services is unreliable — documented,
# pre-existing, not something this PR introduces or fixes)
orion/substrate/tests/                                                    592 passed
services/orion-hub/tests/test_attention_loops_{api,reader}.py
  + test_attention_{card_legibility,closure_e2e,loop_closure}.py           24 passed
services/orion-cortex-exec/tests/test_current_turn_llm_signals.py
  + test_chat_attention_salience_trace.py
  + test_attention_frame_integration.py                                   42 passed
services/orion-thought/tests/test_reverie_salience_trace.py                3 passed
tests/test_attention_loop_decay_digest.py
  + test_check_attention_loop_decay_liveness.py
  + test_check_concept_relation_digest_liveness.py
  + test_top_down.py + test_voluntary_attention_wiring.py                 47 passed
node --test services/orion-hub/static/js/*.test.js                        43 passed, 22 skipped (pre-existing)
```

All against the real project venv (`/mnt/scripts/Orion-Sapienform/.venv`), not a synthetic minimal one.

**Full-suite investigation:** `services/orion-cortex-exec/tests` run as one combined pytest invocation shows 2 pre-existing `test_attention_frame_integration.py` failures. Ran an A/B comparison (real venv, full suite) against unmodified `main` — the identical 2 tests fail there too. Pre-existing test-order pollution in a 700+ file suite, not a regression from this branch (both pass 100% in isolation and paired with the changed files).

## Evals run

```
python services/orion-cortex-exec/evals/run_current_turn_signal_eval.py
10/10 fixtures correct (includes the exact live-garbage strings — "bus",
"Glad", "Compact", "Interesting" — as regression cases)
```

## Docker/build/smoke checks

Docker not exercised directly (no compose/runtime changes); DB migration and both new scripts verified live against the real running Postgres (see below).

```
POSTGRES_URI=... python scripts/check_sql_migrations_applied.py --file manual_migration_attention_salience_trace.sql
  -> ok, applied

python scripts/attention_loop_decay_digest.py --dry-run
  -> attention_loop_decay_digest: 27 chat-scope loop(s) scanned, 0 would decay

python scripts/check_attention_loop_decay_liveness.py
  -> OK -- no chat-scope loop is currently decay-eligible.
```

## Review findings fixed

Two full code-review passes (background subagent, medium effort), all findings addressed:

**Round 1:**
- `card_kind_for_scope` denylisted `'reverie'` instead of allowlisting `'chat'` → a future `'broadcast'` scope would silently render as resolvable. Fixed: inverted to allowlist.
- `load_pending_loops`/`latest_scope_for_theme` disagreed on their null-scope default (`'reverie'` vs `'chat'`) → same loop could show as chronic in the panel but be closable via direct API call. Fixed: consolidated into one `latest_trace_for_theme` call.
- `check_attention_loop_decay_liveness.py` hand-reimplemented the digest's grouping logic with a real bug (`theme_key=loop_id` instead of the row's real `theme_key`). Fixed: extracted shared `build_observations`/`eligible_verdicts`.
- No JS test coverage for the new chronic/resolvable branching. Fixed: extracted `cognitive-loop-card.js` (pure, Node-testable) + `cognitive-loop-card.test.js`.
- Two sequential DB round-trips in `_close()`. Fixed: one `latest_trace_for_theme` call.
- Migration deploy-order hazard (silent fail-open if service deploys before migration). Fixed: loud comments on both producers' INSERT paths + README.
- `--min-silence-hours` drift risk between digest/liveness gate. Fixed: README note.

**Round 2 (deeper, on the round-1 fixes):**
- **`substrate_reverie_refractory` also gates real reverie-chain reignition** (`orion-thought/app/chain.py`), contradicting this digest's own docs. Verified live via direct code read. Fixed: digest now scoped to `scope='chat'` only, never touches chronic_pressure/reverie loops. **Live-remediated**: deleted an active 24h suppression row my own earlier (pre-fix) digest run had wrongly written for a reverie-scope loop.
- Round-1's `latest_trace_for_theme` failure-path default (`'unknown'` → chronic/blocked) was itself a regression — would block a legitimate human Resolve/Dismiss click on a transient DB hiccup. Fixed: reverted failure-path default to `'chat'` (permissive); only a successfully-read non-chat scope routes to chronic now.
- `current_turn_llm_signals.py`'s floor used a literal `" " not in phrase` check, missing Unicode whitespace (e.g. NBSP) and giving no observability into what it drops. Fixed: `.split()` + debug log line.
- SQL `ORDER BY theme_key` vs the actual `loop_id` grouping key (latent, unreachable today). Fixed: `ORDER BY loop_id`.
- Disclosed (not fixed, pre-existing, out of scope): chat/reverie loop ids share a namespace by design (`chain.py::theme_key_for`), so an exact-phrase collision between a chat turn and a reverie signal is a residual, low-probability race in `latest_trace_for_theme`.

## Live verification

Applied the migration live, ran the digest live (dry-run then real), found and fixed the cross-service refractory bug live, then re-verified end to end post-fix — see "Tests run"/"Docker" sections above. This is not a claim resting on code review alone; every consequential piece was checked against the real running database.

## Restart required

```bash
# orion-cortex-exec and orion-thought must be redeployed to pick up
# why_it_matters/target_type persistence and the LLM-probe floor. The
# migration is ALREADY APPLIED live — do not redeploy before confirming that
# (it already is, per this PR's own live verification), or every
# attention_salience_trace insert will silently fail-open until it is.
docker compose --env-file .env --env-file services/orion-cortex-exec/.env -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build

# Install the new cron entry (see services/orion-hub/README.md's "Scheduled
# maintenance" section for the exact line) -- not automatic, host crontab only.
```

## Risks / concerns

- **Severity: low.** `chronic_pressure` loops never expire from the panel (by design — see README) — if the underlying reverie signal genuinely goes permanently quiet, its card stays framed as "sustained pressure" indefinitely rather than eventually clearing. Disclosed as an intended trade-off (safety over tidiness), not silently accepted.
- **Severity: low.** Residual chat/reverie theme_key collision risk in `latest_trace_for_theme` (disclosed above) — pre-existing shared-namespace design, not introduced by this PR, worth a follow-up if it's ever observed live.
- **Severity: none (remediated).** The live cross-service refractory bug found in round-2 review was fixed in code and its one live consequence was found and reverted before this PR was opened.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1817
