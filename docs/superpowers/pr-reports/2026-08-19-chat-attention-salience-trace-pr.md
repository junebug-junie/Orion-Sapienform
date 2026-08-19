# PR report: chat-scoped attention salience trace (scope="chat")

## Summary

- Sentience Striving Program item 1 (of the 5-item punch list): the live-chat-scoped attention/curiosity frame (`chat_stance.py:build_chat_stance_inputs` → `build_attention_frame`) runs the same Borda `score_loop`/`select_actions` scoring reverie's tick already persists — but nothing persisted the chat-turn side. Confirmed live 2026-08-19: `SELECT count(*) FROM cognition_traces WHERE metadata::text LIKE '%chat_attention_frame%'` → 0, always.
- Added `services/orion-cortex-exec/app/chat_attention_salience_trace.py`: a bounded, fail-open Postgres writer that persists the selected loop's score into the existing `attention_salience_trace` table with `scope="chat"` (already documented as a valid value on `AttentionSalienceTraceV1.scope`, never previously populated) — no new table, no new schema.
- Wired into `chat_stance.py` in its own `try/except`, separate from the existing attention-frame-build block, so a persist failure can never fall through and wrongly clear the already-successful `chat_attention_frame`/`chat_attention_frame_debug` ctx keys.
- Documented the pre-existing, previously-undocumented `ORION_CURIOSITY_FRAME_ENABLED` key in `.env_example` at its real live default (`false`) — found during this investigation, not flipped on by this patch.
- Review finding fixed: `trace_id` now hashes `turn_id` alongside `correlation_id`/`loop_id` — `correlation_id` can genuinely be `None` for a real chat turn (unlike reverie's tick), and with `ON CONFLICT (trace_id) DO NOTHING` that could otherwise silently collapse distinct turns selecting the same loop into one write.

## Outcome moved

Before this patch, `orion/substrate/attention/policy.py`'s `select_actions()` thresholds (`min_ask`, inline `0.48`/`0.35` cutoffs — disclosed 2026-07-31 as uncalibrated against the Borda rank-aggregated score, see that file's own comment and `orion/sentience_striving_program/README.md`'s 2026-07-31 entry) had zero real chat-traffic data points to recalibrate against — only reverie's system-substrate-node ticks (100% `substrate:node:*` descriptions, a much narrower population). Once `ORION_CURIOSITY_FRAME_ENABLED=true` is flipped on, this patch closes that observability gap: every chat turn's selected loop now writes a real, queryable row. Recalibration itself is future work once enough real rows accumulate (CLAUDE.md's metric-quality-gate step 4 — "live-data sanity check" — needs real rows before it can even start).

## Current architecture

`services/orion-thought/app/reverie.py::run_reverie_once()` already calls `build_salience_trace()` → `persist_salience_trace()` (`services/orion-thought/app/store.py`) after each reverie tick, writing `scope="reverie"` rows into `attention_salience_trace`. `services/orion-cortex-exec/app/chat_stance.py:2364-2382` runs the same underlying scoring (`orion/substrate/attention_frame.py::build_attention_frame`) on every real chat turn but had no persistence at all — confirmed via direct Postgres query, not inference.

## Architecture touched

- `services/orion-cortex-exec` only. No bus/schema/contract changes (the `attention_salience_trace` table and `AttentionSalienceTraceV1.scope` already existed and already documented `"chat"` as a valid value — this patch is the first real producer for it).

## Files changed

- `services/orion-cortex-exec/app/chat_attention_salience_trace.py` (new): bounded fail-open writer, mirrors `metacog_trend_reader.py`/`perception_reader.py`'s established reader conventions (module-level cached engine, shared felt-state/endogenous-runtime DSN fallback chain, per-connection `statement_timeout`, `asyncio.wait_for(asyncio.to_thread(...))`).
- `services/orion-cortex-exec/app/chat_stance.py`: import + call site, own `try/except` after the existing attention-frame-build block.
- `services/orion-cortex-exec/.env_example`: new keys `ENABLE_CHAT_ATTENTION_SALIENCE_TRACE` (default `true`), `CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC` (default `0.8`); documents pre-existing `ORION_CURIOSITY_FRAME_ENABLED` at its real default (`false`).
- `services/orion-cortex-exec/tests/test_chat_attention_salience_trace.py` (new): 12 tests — row shape/scope, None-on-no-selection, None-on-missing-loop, trace_id stability, trace_id turn-disambiguation regression test, success/disabled/no-selection/DSN-unset/timeout/exception fail-open paths, DSN fallback.
- `services/orion-cortex-exec/tests/test_attention_frame_integration.py`: 3 new tests — persist called when flag on, never called when off, persist failure doesn't clear ctx.

## Schema / bus / API changes

- Added: none (reuses `attention_salience_trace`'s existing `scope` column, already documented as `reverie | chat | broadcast`).
- Removed: none.
- Renamed: none.
- Behavior changed: `attention_salience_trace` will start receiving `scope="chat"` rows once `ORION_CURIOSITY_FRAME_ENABLED=true` is set (currently `false`, unchanged by this patch).
- Compatibility notes: none — additive only, existing `scope="reverie"` consumers unaffected.

## Env/config changes

- Added keys: `ENABLE_CHAT_ATTENTION_SALIENCE_TRACE=true`, `CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC=0.8`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (also newly documents pre-existing `ORION_CURIOSITY_FRAME_ENABLED=false`).
- local `.env` synced: hand-edited directly (`services/orion-cortex-exec/.env` in the primary checkout) per the established "env sync is mandatory, hand-edit the live file" preference — not run through `sync_local_env_from_example.py` (known bug reading from primary checkout regardless of invoking worktree).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-cortex-exec
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest tests/test_chat_attention_salience_trace.py tests/test_attention_frame_integration.py -q
→ 21 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/pytest tests/test_chat_attention_salience_trace.py tests/test_attention_frame_integration.py tests/test_attention_frame.py -q
→ 26 passed, 1 failed (test_open_loop_without_autonomy_boost_defers_below_ask_threshold)
```

The one failure (`test_attention_frame.py::test_open_loop_without_autonomy_boost_defers_below_ask_threshold`, `assert 'watch' == 'defer'`) is **pre-existing and unrelated to this patch** — confirmed identical on unmodified `origin/main` (56a5402e9) run from the primary checkout. It is, incidentally, corroborating evidence for the exact threshold-miscalibration this instrumentation exists to eventually let someone fix — not touched here, out of scope.

Also confirmed pre-existing and unrelated: running the full `services/orion-cortex-exec/tests/` suite in one process produces 13 `ValueError: Verb already registered` collection errors in unrelated files (documented pattern, several prior PR reports reference it) — reproduces identically on unmodified `origin/main`.

## Evals run

No dedicated eval harness exists for this service's attention-frame scoring. Flagging as a known gap (per CLAUDE.md §11): a future eval would replay real `attention_salience_trace(scope="chat")` rows once accumulated and check `select_actions()`'s threshold behavior against the real score distribution — exactly the recalibration item 1 exists to enable, deferred until real rows exist.

## Docker/build/smoke checks

Not deployed live this patch — `ORION_CURIOSITY_FRAME_ENABLED=false` means this code path does not run in the current deployment regardless of build/deploy state, so a live docker smoke would exercise nothing new. Flagged for the operator: flipping `ORION_CURIOSITY_FRAME_ENABLED=true` (a separate, pre-existing, undocumented-until-now decision, not part of this patch) is what actually activates both the chat attention frame and this new tracer.

## Review findings fixed

- Finding: `trace_id` hashed only `[correlation_id, loop_id]`; `correlation_id` can genuinely be `None` for a real chat turn (`orion/substrate/attention_frame.py`'s own fallback-to-`None`), and with `ON CONFLICT (trace_id) DO NOTHING` this could silently collapse distinct turns selecting the same loop into a single write.
  - Fix: hash now includes `turn_id` (`frame.turn_id or ""`) alongside `correlation_id` and `loop_id`.
  - Evidence: new regression test `test_trace_id_does_not_collapse_across_distinct_turns_when_correlation_id_is_absent` (`tests/test_chat_attention_salience_trace.py`), passes.
- Finding (informational, not fixed — correctly out of scope): the instrumentation is currently a no-op end-to-end since `ORION_CURIOSITY_FRAME_ENABLED` defaults `false` in the real deployment.
  - Disposition: disclosed plainly in `.env_example`'s new comment block and in this report; flipping that flag is an operator decision, not something this patch should make unilaterally.
- Everything else reviewed came back clean: fail-open contract, DSN fallback chain, engine caching, `statement_timeout`, idempotency/row shape (byte-identical INSERT to `services/orion-thought/app/store.py::persist_salience_trace`), env parity, and CLAUDE.md §0A keyword-cathedral check (real producer + existing consumer + existing schema + real tests).

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build
```

Not run this patch, since `ORION_CURIOSITY_FRAME_ENABLED=false` means the new code path is inert either way. Deploy whenever convenient; a restart is only functionally meaningful once the operator also flips `ORION_CURIOSITY_FRAME_ENABLED=true`.

## Risks / concerns

- Severity: Low
- Concern: Merge-order risk with the in-progress `Orion-Sapienform-kill-dead-phi-hint-fallback` worktree, which has a large uncommitted/unmerged diff touching the same files this patch touches (`chat_stance.py`, `settings.py`) and — notably — appears to **delete** `metacog_trend_reader.py` and `perception_reader.py` entirely, the two sibling modules this patch's DSN/engine-caching conventions were mirrored from.
- Mitigation: Not a blocker for this patch (that worktree is not merged, and this PR is a small, additive, independently-mergeable diff). Whichever branch merges second will need conflict resolution in `chat_stance.py`, and if `metacog_trend_reader.py`/`perception_reader.py` really are being deleted there, that context is useful for whoever reviews that branch — flagging it here for visibility, not resolving it in this PR.
- Severity: Low
- Concern: The whole feature (chat attention frame + this new tracer) is currently disabled in production (`ORION_CURIOSITY_FRAME_ENABLED=false`), so this patch produces zero real rows until an operator makes a separate, deliberate decision to turn it on.
- Mitigation: None needed from this patch — disclosed plainly rather than silently shipped as if it were already active. Turning it on is a follow-up decision for Juniper, not assumed here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1753
