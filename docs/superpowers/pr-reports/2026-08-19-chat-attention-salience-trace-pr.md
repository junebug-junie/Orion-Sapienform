# PR report: chat-scoped attention salience trace (scope="chat")

## Summary

- Sentience Striving Program item 1 (of the 5-item punch list): the live-chat-scoped attention/curiosity frame (`chat_stance.py:build_chat_stance_inputs` → `build_attention_frame`) runs the same Borda `score_loop`/`select_actions` scoring reverie's tick already persists — but nothing persisted the chat-turn side. Confirmed live 2026-08-19: `SELECT count(*) FROM cognition_traces WHERE metadata::text LIKE '%chat_attention_frame%'` → 0, always.
- Added `services/orion-cortex-exec/app/chat_attention_salience_trace.py`: a bounded, fail-open Postgres writer that persists the selected loop's score into the existing `attention_salience_trace` table with `scope="chat"` (already documented as a valid value on `AttentionSalienceTraceV1.scope`, never previously populated) — no new table, no new schema.
- Wired into `chat_stance.py` in its own `try/except`, separate from the existing attention-frame-build block, so a persist failure can never fall through and wrongly clear the already-successful `chat_attention_frame`/`chat_attention_frame_debug` ctx keys.
- Documented the pre-existing, previously-undocumented `ORION_CURIOSITY_FRAME_ENABLED` key in `.env_example`; originally left at its real live default (`false`, disclosed-not-changed). Juniper then explicitly authorized flipping it on ("turn on flag in env and env example") now that this patch gives it real observability — `.env_example` and the live `.env` both now read `ORION_CURIOSITY_FRAME_ENABLED=true`.
- Review finding fixed: `trace_id` now hashes `turn_id` alongside `correlation_id`/`loop_id` — `correlation_id` can genuinely be `None` for a real chat turn (unlike reverie's tick), and with `ON CONFLICT (trace_id) DO NOTHING` that could otherwise silently collapse distinct turns selecting the same loop into one write.

## Outcome moved

Before this patch, `orion/substrate/attention/policy.py`'s `select_actions()` thresholds (`min_ask`, inline `0.48`/`0.35` cutoffs — disclosed 2026-07-31 as uncalibrated against the Borda rank-aggregated score, see that file's own comment and `orion/sentience_striving_program/README.md`'s 2026-07-31 entry) had zero real chat-traffic data points to recalibrate against — only reverie's system-substrate-node ticks (100% `substrate:node:*` descriptions, a much narrower population). With `ORION_CURIOSITY_FRAME_ENABLED=true` now flipped on, this patch closes that observability gap: every chat turn's selected loop writes a real, queryable row. Recalibration itself is future work once enough real rows accumulate (CLAUDE.md's metric-quality-gate step 4 — "live-data sanity check" — needs real rows before it can even start).

## Current architecture

`services/orion-thought/app/reverie.py::run_reverie_once()` already calls `build_salience_trace()` → `persist_salience_trace()` (`services/orion-thought/app/store.py`) after each reverie tick, writing `scope="reverie"` rows into `attention_salience_trace`. `services/orion-cortex-exec/app/chat_stance.py:2364-2382` runs the same underlying scoring (`orion/substrate/attention_frame.py::build_attention_frame`) on every real chat turn but had no persistence at all — confirmed via direct Postgres query, not inference.

## Architecture touched

- `services/orion-cortex-exec` only. No bus/schema/contract changes (the `attention_salience_trace` table and `AttentionSalienceTraceV1.scope` already existed and already documented `"chat"` as a valid value — this patch is the first real producer for it).

## Files changed

- `services/orion-cortex-exec/app/chat_attention_salience_trace.py` (new): bounded fail-open writer, mirrors `metacog_trend_reader.py`/`perception_reader.py`'s established reader conventions (module-level cached engine, shared felt-state/endogenous-runtime DSN fallback chain, per-connection `statement_timeout`, `asyncio.wait_for(asyncio.to_thread(...))`).
- `services/orion-cortex-exec/app/chat_stance.py`: import + call site, own `try/except` after the existing attention-frame-build block.
- `services/orion-cortex-exec/.env_example`: new keys `ENABLE_CHAT_ATTENTION_SALIENCE_TRACE` (default `true`), `CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC` (default `0.8`); documents pre-existing `ORION_CURIOSITY_FRAME_ENABLED`, flipped `true` per Juniper's explicit authorization.
- `services/orion-cortex-exec/tests/test_chat_attention_salience_trace.py` (new): 12 tests — row shape/scope, None-on-no-selection, None-on-missing-loop, trace_id stability, trace_id turn-disambiguation regression test, success/disabled/no-selection/DSN-unset/timeout/exception fail-open paths, DSN fallback.
- `services/orion-cortex-exec/tests/test_attention_frame_integration.py`: 3 new tests — persist called when flag on, never called when off, persist failure doesn't clear ctx.

## Schema / bus / API changes

- Added: none (reuses `attention_salience_trace`'s existing `scope` column, already documented as `reverie | chat | broadcast`).
- Removed: none.
- Renamed: none.
- Behavior changed: `attention_salience_trace` starts receiving `scope="chat"` rows now that `ORION_CURIOSITY_FRAME_ENABLED=true`. This also activates the chat attention/curiosity frame itself (`inputs["attention_frame"]`, `ctx["chat_attention_frame"]`) inside real chat-turn prompts for the first time — the frame computation and its prompt-contract injection were part of an already-merged, already-tested prior patch; this PR only adds the persistence side and flips the pre-existing flag on.
- Compatibility notes: none — additive only, existing `scope="reverie"` consumers unaffected.

## Env/config changes

- Added keys: `ENABLE_CHAT_ATTENTION_SALIENCE_TRACE=true`, `CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC=0.8`.
- Removed keys: none.
- Renamed keys: none.
- Value changed (Juniper-authorized, 2026-08-19): `ORION_CURIOSITY_FRAME_ENABLED` `false` → `true`. Pre-existing key (predates this PR), initially just documented at its real default; flipped on in a follow-up to this same branch once Juniper explicitly asked for it.
- `.env_example` updated: yes.
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

Deployed live (worktree `.env`/`services/orion-cortex-exec/.env` symlinked to the primary checkout, confirmed gitignored via `git check-ignore -v`):

```text
./scripts/safe_docker_build.sh orion-cortex-exec build
→ 4/4 images built (cortex-exec, cortex-exec-chat, cortex-exec-spark, cortex-exec-background)

./scripts/safe_docker_build.sh orion-cortex-exec up -d --build
→ all 4 containers recreated and started

docker exec orion-athena-cortex-exec-chat printenv ORION_CURIOSITY_FRAME_ENABLED ENABLE_CHAT_ATTENTION_SALIENCE_TRACE CHAT_ATTENTION_SALIENCE_TRACE_TIMEOUT_SEC
→ true / true / 0.8   (confirmed live inside the container, not just in the .env file)

docker logs orion-athena-cortex-exec-chat --tail 40 | grep -iE "error|traceback|attention_frame_build_failed|chat_attention_salience"
→ no matches
```

**Runtime-truth verification (not just config/container-health):** queried Postgres directly seconds after deploy —

```text
SELECT scope, count(*) FROM attention_salience_trace GROUP BY scope;
 scope  | count
--------+-------
 chat   |     1
 reverie|   915

SELECT * FROM attention_salience_trace WHERE scope='chat';
 trace_id: saltrace_89345dba95975d41cb8d4af4
 loop_id: open-loop-3be13c7644a0
 description: substrate:node:substrate.biometrics
 correlation_id: 6fb4f8f2-bb13-40a7-bf23-d9de519523ed
 salience: 0.5
 weights_version: gwt-coalition-v1
 scope: chat
 features: {"evidence_breadth": 0.5, "evidence_strength": 0.34}
 created_at: 2026-08-19 23:16:18+00
```

A real `scope="chat"` row landed from genuine production chat traffic within seconds of the container coming up — not a synthetic/test insert. Early-data caveat (n=1, not a claim): the first row's `description` is a `substrate:node:*` label, the same category reverie's 915 rows are 100% drawn from, not a purely conversational loop about the user's own turn content — worth watching as more rows accumulate before assuming chat-scope data looks meaningfully different from reverie-scope data.

## Review findings fixed

- Finding: `trace_id` hashed only `[correlation_id, loop_id]`; `correlation_id` can genuinely be `None` for a real chat turn (`orion/substrate/attention_frame.py`'s own fallback-to-`None`), and with `ON CONFLICT (trace_id) DO NOTHING` this could silently collapse distinct turns selecting the same loop into a single write.
  - Fix: hash now includes `turn_id` (`frame.turn_id or ""`) alongside `correlation_id` and `loop_id`.
  - Evidence: new regression test `test_trace_id_does_not_collapse_across_distinct_turns_when_correlation_id_is_absent` (`tests/test_chat_attention_salience_trace.py`), passes.
- Finding (informational, not fixed — correctly out of scope): the instrumentation is currently a no-op end-to-end since `ORION_CURIOSITY_FRAME_ENABLED` defaults `false` in the real deployment.
  - Disposition: disclosed plainly in `.env_example`'s new comment block and in this report; flipping that flag is an operator decision, not something this patch should make unilaterally.
- Everything else reviewed came back clean: fail-open contract, DSN fallback chain, engine caching, `statement_timeout`, idempotency/row shape (byte-identical INSERT to `services/orion-thought/app/store.py::persist_salience_trace`), env parity, and CLAUDE.md §0A keyword-cathedral check (real producer + existing consumer + existing schema + real tests).

## Restart required

Already done — deployed and live-verified above (`orion-athena-cortex-exec`, `-chat`, `-spark`, `-background` all recreated and running with `ORION_CURIOSITY_FRAME_ENABLED=true`). For reference, the exact command used:

```bash
./scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

## Risks / concerns

- Severity: Low
- Concern: Merge-order risk with the in-progress `Orion-Sapienform-kill-dead-phi-hint-fallback` worktree, which has a large uncommitted/unmerged diff touching the same files this patch touches (`chat_stance.py`, `settings.py`) and — notably — appears to **delete** `metacog_trend_reader.py` and `perception_reader.py` entirely, the two sibling modules this patch's DSN/engine-caching conventions were mirrored from.
- Mitigation: Not a blocker for this patch (that worktree is not merged, and this PR is a small, additive, independently-mergeable diff). Whichever branch merges second will need conflict resolution in `chat_stance.py`, and if `metacog_trend_reader.py`/`perception_reader.py` really are being deleted there, that context is useful for whoever reviews that branch — flagging it here for visibility, not resolving it in this PR.
- Severity: Medium (was Low-disabled, now live)
- Concern: `ORION_CURIOSITY_FRAME_ENABLED=true` doesn't just activate this patch's tracer — it activates the chat attention/curiosity frame *itself* in real chat-turn prompts for the first time in production (`inputs["attention_frame"]`, `ctx["chat_attention_frame"]` injected into `chat_stance_brief.j2`/`chat_general.j2`, per `test_prompt_contracts_include_attention_policy`). That means Orion can now select and ask a real curiosity question (`selected_action.action_type == "ask"`) on a live chat turn where it never could before. The frame-build/prompt-injection code itself is from an already-merged, already-tested prior patch (not new in this PR) — but this is the first time it has ever run against real traffic with the flag on.
- Mitigation: Explicitly authorized by Juniper ("turn on flag in env and env example"). Deployed and live-verified (see Docker/build/smoke checks above) — no errors in logs, one real `scope="chat"` row landed cleanly. Given `min_ask`/`0.48`/`0.35` are disclosed-uncalibrated thresholds, worth watching real chat behavior over the next few turns/days for an unexpectedly frequent or out-of-place curiosity question; this PR's own tracer is exactly the tool for checking that once enough rows accumulate.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1753
