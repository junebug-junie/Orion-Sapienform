## Summary

- Orion's world-pulse reading used to spend a daily reading slot even when the turn was turned away before reading anything (usually because the GPU was full). Six refusals used up the whole day, and reading went silent until the next day. Those refusals now give their slot back.
- Both loops are covered. Stage 1 uses Wallet A and Stage 2 uses Wallet B. `debit_wallet_a/b` now return a receipt, and `refund_wallet_a/b` undo that one debit, once.
- The slot comes back on the day it was charged, even if the refusal lands after midnight. The count never goes below zero. The cooldown timestamp goes back to the last debit that actually counted, so the dashboard's "last read" stays true.
- Without the daily cap as a brake, a capacity outage could retry every 5-minute tick. Each retry is a stance call on a saturated GPU, and it spends one of the seed's 3 attempts. So each refusal now also sets a separate retry-not-before time that doubles per consecutive refusal: 30m, 1h, 2h, then 4h max at live config. It resets once a turn reaches the reader.
- Stage 2 now charges its wallet only after the stored handoff validates, so an invalid handoff costs nothing.

## Outcome moved

The failure mode, observed live 2026-09-23 and 09-24: `orion:wp_read:wallet_a:count:<day>` hit 6 on refused turns, then `world_pulse_read_blocked reason=daily_cap` logged every tick for the rest of the day, and no seed reached `done` from 09-15 to 09-24.

After this patch, a turn refused before reading costs no daily slot. A full-day capacity outage costs about 8 backed-off stance attempts (close to the old 6), not a whole day of silence once capacity returns.

## Current architecture

- Both loops live in `services/orion-hub/scripts/world_pulse_read_pipeline.py` (Stage 1) and `world_pulse_read_stage2.py` (Stage 2). Each claims a seed from Postgres `world_pulse_read_seed`, debits Redis (`orion:wp_read:wallet_{a,b}:last_at` and `:count:<local day>`), then runs `execute_unified_turn`.
- A failed turn's reason becomes `last_error`: `turn_deferred:<stance reason>`, `turn_error:<code>`, and so on. `retry.is_transient_failure` sends the seed back to `pending` until `max_attempts` (3).
- `turn_deferred` frames are only built in the stance phase of `orion/hub/turn_orchestrator.py`: stance timeout or missing thought, stance defer/refuse, and `stance_react_failed`. That last one is where GPU-capacity refusals land. All of them return before `HarnessGovernorClient.run`, so the harness/FCC reader never started.

## Architecture touched

- Hub reading loops only. No bus, schema, or env changes.
- New Redis keys, owned by the wallet modules:
  - `orion:wp_read:wallet_{a,b}:retry_not_before`
  - `orion:wp_read:wallet_{a,b}:refund_streak`
  - Both have a 48h TTL, the same as the existing wallet keys.

## Files changed

- `orion/world_pulse_read/wallet_refund.py`: new. Holds the receipt, the refund (slot, cooldown restore, backoff), the streak settle, and the retry-wait read.
- `orion/world_pulse_read/wallet_a.py`, `wallet_b.py`: debit returns a receipt; adds refund/settle/retry-wait wrappers, the new key constants, and the `refund_backoff` block reason.
- `orion/world_pulse_read/retry.py`: adds `is_refused_before_work`, which matches exactly `turn_deferred` or `turn_deferred:*`.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`, `world_pulse_read_stage2.py`: add the backoff gate, `_settle_wallet` on every turn outcome, and `refund_backoff` in `_FORCE_OVERRIDE`. Stage 2 debits after handoff validation.
- `services/orion-hub/scripts/world_pulse_read_routes.py`: `/api/status` exposes `wallet_{a,b}.retry_not_before`.
- `services/orion-hub/README.md`: documents the refund behavior.
- Tests:
  - `services/orion-hub/tests/test_world_pulse_read_wallet_refund.py` is new.
  - `test_world_pulse_read_pipeline.py`, `test_world_pulse_read_stage2.py`, and `test_world_pulse_read_routes.py` gain new cases.

## Schema / bus / API changes

- Added: `wallet_a.retry_not_before` and `wallet_b.retry_not_before` fields in the world-pulse-read `/api/status` JSON.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - A `turn_deferred*` failure refunds its wallet debit.
  - New block reason `refund_backoff`, which `force=True` overrides.
- Compatibility notes: `debit_wallet_a/b` used to return `None` and now return a receipt. Its only callers are the two loops.

## Env/config changes

- Added keys: none. The backoff reuses `HUB_WORLD_PULSE_READ_MIN_COOLDOWN_SEC` and `HUB_WORLD_PULSE_READ_STAGE2_MIN_COOLDOWN_SEC` as its base. The cap is a code constant, `_REFUND_BACKOFF_CAP_MULTIPLIER = 8`.
- Removed/renamed keys: none.
- `.env_example` updated: no.
- Local `.env` synced: not needed, since no templates changed.
- Skipped keys: none.

## Tests run

```text
pytest services/orion-hub/tests -k world_pulse                       -> 140 passed
CI orion-reading test set (workflow command, run locally)            -> 501 passed, 9 skipped
New tests against origin/main sources                                -> 9 failed (as intended)
```

The key tests feed a real `{"type":"turn_deferred","reason":"stance_react_failed: agent=gpu_pool_unavailable:deadline"}` frame through `_generate`, then `_stage1_read`, then the refund:

- Deferred turn: the count goes back to its prior value.
- `turn_error:fcc_stream_stalled`: the count stays charged.
- Backoff doubles and resets.
- A refund after midnight hits the debited day.
- The count never goes below 0.
- A refund happens only once.
- A later debit's cooldown is never overwritten.

## Evals run

```text
No eval added. This is wallet accounting, not generation quality. services/orion-hub/evals/test_reading_handoff_eval.py is unaffected.
```

## Docker/build/smoke checks

```text
Not deployed (per task). No Docker build run: no dependency or compose change.
Live read-only observations (2026-09-25, UTC):
- Hub log 03:56-06:00: Stage 1 `world_pulse_read_blocked reason=daily_cap` every tick (count:2026-09-24 = 6).
- 06:00 UTC (local-day reset, America/Denver): 06:01:32 seed finding:60d59b10…:9b084fc0f1583da0 claimed,
  06:10:11 marked done -> first `done` since 2026-09-15. The turn ran through the GPU pool end to end.
  BUT its handoff says "Metadata-only extraction; I did not fetch or read the article body this turn" — see Risks.
- 06:15, 06:20: reason=cooldown (1800s), as expected.
- world_pulse_read_seed: pending 143, done 16, failed 3, skipped 134.
```

## Review findings fixed

- Finding: the refund removed the daily cap as the only brake during a capacity outage. That meant a retry every 30 minutes (up to about 48 a day on live 24h Stage 1), each burning a seed attempt.
  - Fix: exponential retry-not-before over consecutive refusals (30m, 1h, 2h, 4h cap). The streak resets when a turn reaches the reader.
  - Evidence: `test_refund_backoff_blocks_the_next_tick_then_doubles` and `test_retry_not_before_backs_off_over_consecutive_refunds_and_resets`.
- Finding: the first version rewrote `last_at` to a synthetic "anchor" time, so the dashboard showed a `last_read_at` that never happened.
  - Fix: restore the real prior value, or delete it if there was none. The spacing lives in its own key.
  - Evidence: `test_refund_restores_prior_cooldown_or_clears_it`.
- Finding: the tests replaced `_stage1_read` with a raised label and never exercised the real frame path.
  - Fix: tests now drive real turn frames through `_generate` for both stages.
  - Evidence: `test_real_deferred_frame_refunds_wallet_a_slot` and `test_stage2_real_deferred_frame_refunds_wallet_b`.
- Finding: `bus_unavailable` could never refund, because there is no receipt without a bus. The bare `turn_deferred` prefix also over-matched.
  - Fix: the classifier is now exactly `turn_deferred` or `turn_deferred:*`.
  - Evidence: classifier test, including `turn_deferredX` and `turn_error:bus_unavailable` as charged.
- Finding: the compensating `incr` after a racing expiry left a count key with no TTL.
  - Fix: re-apply `expire`.
- Finding: the refund unit tests sat in `orion/world_pulse_read/tests`, which CI does not run.
  - Fix: moved to `services/orion-hub/tests/test_world_pulse_read_wallet_refund.py`, which the CI glob covers.

## Restart required

```bash
./scripts/safe_docker_build.sh orion-hub up -d --build   # from a worktree at merged main
```

## Risks / concerns

- Severity: medium (not introduced here; found while verifying).
  - Concern: the only successful read today is hollow. The model returned a handoff saying it never fetched the article body. That is metadata only, yet the seed was marked `done` and spent a slot. The URL (`…/nvidia-latest-news-and-insights.html`) is also a roundup/hub page that `url_looks_like_section_index` did not catch.
  - Mitigation: follow-up. Gate `done` on evidence that a fetch happened (a tool_use/WebFetch trace on the turn), and widen the index-URL filter.
- Severity: low.
  - Concern: Orion's own stance defer/refuse of a seed (not capacity), e.g. `turn_deferred:empty_imperative`, also refunds and retries under backoff. The seed still spends one of its 3 attempts per refusal, so it cannot loop forever.
  - Mitigation: none needed. This is intended, since nothing was read.
- Severity: low.
  - Concern: after a Stage 1 refund, Wallet A `last_at` goes back to the prior real debit. Stage 2's re-entry gate (`_reenter_stage1`, which reads Wallet A cooldown) can therefore reopen sooner.
  - Mitigation: none needed. That is correct, since the refused turn read nothing. Re-entry only enqueues.
- Severity: info (stale backlog, report only, no data mutated).
  - Concern: the 143 pending Stage 1 seeds are not stuck by the attempts logic. All have `attempts` ≤ 1 against max 3, so they remain claimable. They are stuck by throughput and priority. Capacity is 6 reads a day. Claim order is `priority, attempts, created_at`. 39 priority-0 `finding`/`reading` seeds (32 of them already-tried retries, since 2026-09-07) sit ahead of 104 priority-10 `digest_item` seeds, and roughly 5 non-index digest items a day keep arriving. The digest backlog therefore never drains, and news from 2026-09-07 is stale.
  - Mitigation (proposal): age out pending `digest_item` seeds older than N days (e.g. 5) to `skipped` with `last_error='stale_digest_item'`, via a scheduled query in the enqueue path. Separately decide whether retries should outrank fresh seeds of the same priority.
