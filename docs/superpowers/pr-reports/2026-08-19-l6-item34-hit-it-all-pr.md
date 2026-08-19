# L6 self-model: chat_prediction_error EWMA fix + TEST-validated predicted_shift window=2

## Summary

- Fixed `chat_prediction_error`'s fixed `_THRESHOLD=0.30` divisor bug — the exact defect
  `execution_prediction_error` was fixed for on 2026-07-28 — by moving it to the same
  self-calibrating EWMA z-score baseline pattern.
- Live-confirmed root cause: `predicted_shift`'s cross-domain argmax had never once named `chat`
  across 19,426 real ticks over a 7-day window, despite chat having genuine, non-flat deltas.
- Ran item 4 sub-idea #3's long-paused TEST-set validation (window=2 vs window=10 vs window=30)
  against real, now-available `substrate_attention_self_model` biometrics history. window=2 wins
  on held-out TEST (61.9% reversion accuracy, not overfit) — shipped as the new
  `SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS` default (was 10, itself never calibrated).
- Audited item 3 (confidence formula)'s independence question — already deliberately gated in the
  existing code's own docstring; no change needed.
- Corrected a stale memory pointer (`substrate_field_state` no longer holds the per-domain
  prediction-error series item 4's investigation depended on — it's been repurposed for Causal
  Geometry capability-edge snapshots).

## Outcome moved

`predicted_shift` ("what am I predicted to shift toward next") previously could only ever name
execution, biometrics, or bus_synaptic — chat and route were structurally invisible regardless of
real internal state. Chat is now a real, competitive candidate. Separately, the trend-window
constant driving *when* a shift counts as real went from an admittedly-uncalibrated placeholder to
a value backed by a real, held-out-TEST-validated accuracy improvement (61.9% vs 56.4%).

## Current architecture

L6's self-model (`orion/substrate/attention_self_model.py`) computes `predicted_shift` by taking
the argmax of `|trend|` across `ACTIVE_INFERENCE_DOMAINS` (execution/biometrics/chat/route/
bus_synaptic), where each domain's raw `prediction_error` comes from `orion/substrate/
prediction_error.py`. `execution_prediction_error` and `bus_synaptic_prediction_error` already use
self-calibrating EWMA/z-score baselines (fixed 2026-07-28 and 2026-07-26 respectively).
`chat_prediction_error`, `biometrics_prediction_error`, and `route_prediction_error` did not.
`SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS` (live default 10, `services/
orion-substrate-runtime/app/settings.py`) sizes the rolling window `compute_prediction_error_trend()`
(`orion/substrate/prediction_error_trend.py`) uses — its own comment already flagged it as "not
independently calibrated."

## Architecture touched

- `orion/substrate/prediction_error.py`: `chat_prediction_error` rewritten to use
  `compute_ewma_update`; new domain-specific constants.
- `orion/schemas/chat_projection.py`: `ChatSessionProjectionV1` gets 3 new EWMA baseline fields
  (mirrors `ExecutionTrajectoryProjectionV1`'s existing pattern — same generic
  `_load_projection`/`_save_projection` store helpers already round-trip them, no store-layer
  change needed).
- `services/orion-substrate-runtime/app/settings.py`,`.env_example`, `docker-compose.yml`:
  `attention_self_model_trend_window_ticks` default 10 → 2.

## Files changed

- `orion/substrate/prediction_error.py`: `chat_prediction_error` EWMA rewrite + 3 new constants
  (`_CHAT_PREDICTION_ERROR_EWMA_ALPHA`, `_ZSCORE_SATURATION`, `_MIN_VARIANCE`).
- `orion/schemas/chat_projection.py`: 3 new fields on `ChatSessionProjectionV1`.
- `orion/substrate/tests/test_prediction_error.py`: chat test block rewritten for the new EWMA
  shape (cold-start seeding, established-baseline scoring, clamp-to-zero, domain variance floor,
  saturation) — mirrors `execution_prediction_error`'s own test suite exactly. 65/65 pass.
- `services/orion-substrate-runtime/app/settings.py`: `attention_self_model_trend_window_ticks`
  default 10 → 2, docstring records the full TEST-validation numbers.
- `services/orion-substrate-runtime/.env_example`: matching default + comment update.
- `services/orion-substrate-runtime/docker-compose.yml`: passthrough fallback default 10 → 2.
- `services/orion-substrate-runtime/tests/test_worker_attention_self_model_tick.py`: one test
  (`test_trend_buffer_accumulates_across_ticks`) was silently coupled to the old default=10 (it
  fired 4 ticks and asserted a buffer length of 4); now sets its window explicitly via
  `monkeypatch.setenv`, decoupling it from whatever the default happens to be.

## Schema / bus / API changes

- Added: `ChatSessionProjectionV1.prediction_error_baseline_ewma` / `_var` / `_n` (floats/int,
  all default to `0.0`/`0.0`/`0` — backward-compatible with every already-persisted row).
- Removed: none.
- Renamed: none.
- Behavior changed: `chat_prediction_error()`'s return value scale/meaning is unchanged (still a
  0-1 surprise score) but its *actual observed values* will shift upward materially once live,
  since it's no longer artificially suppressed by the `_THRESHOLD=0.30` divisor.
  `SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS`'s default behavior changes on next restart
  (see below).
- Compatibility notes: no migration needed — new fields default correctly for both a fresh
  projection and an upgrade from an older persisted row (Pydantic fills the default when absent
  from stored JSON, same mechanism already proven live for execution's identical fields).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none (value-only change to `SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS`).
- `.env_example` updated: yes (`services/orion-substrate-runtime/.env_example`).
- local `.env` synced: **not** via `python scripts/sync_local_env_from_example.py` — that script
  resolves both `.env` and `.env_example` from the **primary checkout**, so a worktree-side
  `.env_example` value change is invisible to it (confirmed, matches
  [[feedback_env_sync_reads_example_from_primary_checkout]]). Hand-edited
  `/mnt/scripts/Orion-Sapienform/services/orion-substrate-runtime/.env` directly instead, and
  confirmed the key is passed through in `docker-compose.yml`.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests/test_prediction_error.py -q
65 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest orion/substrate/tests/ -q
546 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-substrate-runtime/tests -q \
  --ignore=services/orion-substrate-runtime/tests/test_grammar_consumer_integration.py
249 passed, 16 failed — confirmed pre-existing via `git stash -u` + re-run (identical 16 failures,
identical pass count, with this patch's changes stashed out). All 16 are local-Postgres-at-
127.0.0.1:5432 / other-infra dependencies unrelated to this patch (2 of the 16,
test_quarantine_truth / test_worker_independent_reducers, already documented as pre-existing in
[[project_self_modeling_ladder]]).

test_grammar_consumer_integration.py: excluded — needs a local Postgres server on 127.0.0.1:5432
not reachable in this environment (unrelated to this patch; the live conjourney DB used for the
TEST validation below is reached via `docker exec orion-athena-sql-db`, a different path).
```

## Evals run

Not a unit-test-shaped eval — this patch's real validation is the live-data TEST-set backtest
below, run directly against real production history via the actual shipped
`compute_prediction_error_trend()` function (not a reimplementation):

```text
Source: substrate_attention_self_model.self_model_json->'prediction_error_by_domain'->>'biometrics'
        (Postgres, conjourney db, docker exec orion-athena-sql-db)
19,425 ticks, 2026-08-12 -> 2026-08-19 (~7 days, ~30s cadence)
Chronological 70/30 TRAIN/TEST split (no shuffle -- avoids leakage)
Horizon: 2 ticks (~60s ahead), matching the original PR #1304 methodology

           TRAIN reversion acc.   TEST reversion acc. (n, z vs 50%)
window=2   59.9%                  61.9% (n=4885, z=+16.6)
window=10  57.3%                  56.4% (n=5300, z=+9.3)   <- old live default
window=30  53.4%                  54.2% (n=5293, z=+6.1)   <- offline replay script's own default

window=2 wins on held-out TEST, not just TRAIN -- TEST outperforms TRAIN, not an overfit result.
```

Script preserved at `/tmp/claude-1000/.../scratchpad/item4_window_test_validation.py` (session
scratchpad, not committed — reproducible from this report's methodology description plus the
already-shipped `compute_prediction_error_trend()`).

## Docker/build/smoke checks

```text
bash scripts/safe_docker_build.sh orion-substrate-runtime config
-- Could not run from the worktree (no root .env there by design -- .env is gitignored,
   per-checkout). Verified instead by grep: docker-compose.yml's environment passthrough
   (`SUBSTRATE_ATTENTION_SELF_MODEL_TREND_WINDOW_TICKS=${...:-2}`) already lists the key
   (pre-existing, only its fallback default changed) -- no deploy-gap risk of the PR #1378 class.
```

Not deployed/restarted as part of this patch — see Restart required below.

## Review findings fixed

See code-review skill run against this branch; findings and fixes recorded here once the run
completes.

## Restart required

```bash
bash scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

This restarts a live cognition-loop tick (`SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED` is already
`true` in production) — the chat fix and window=2 change only take effect after this restart. Low
blast radius (both changes are to an already-shadow-measured, non-dispatching self-model field;
neither adds a new consumer or dispatch path), but flagging explicitly per CLAUDE.md's "runtime
truth beats config truth" — the code being merged is not the same as the change being live.

## Risks / concerns

- Severity: low. Concern: window=2's trend estimate is a single-prior-sample vs
  single-recent-sample comparison — much more reactive/noisy than window=10 or 30, by design (the
  TEST data says this reactivity is net-positive for biometrics, but it's a real trade-off, not a
  free win). Mitigation: revisit if `predicted_shift`'s live text starts flapping direction every
  tick in a way that reads as noise rather than signal — the fix is a bigger window, not reverting
  this one, given the TEST evidence.
- Severity: low. Concern: the window=2 validation is biometrics-only (same domain the original
  reversion-sign fix and item 4's own docstring already validated on, for the same "only domain
  with enough real variance" reason) and applied uniformly to all 5 domains — a reasoned
  extrapolation, not independently confirmed for execution/chat/route/bus_synaptic. Mitigation:
  same accepted-risk shape as the original reversion-sign fix; a future pass should back-test the
  other domains once they have independently-confirmable variance (chat's own EWMA fix in this
  patch may eventually enable that).
- Severity: none — route was investigated and deliberately NOT touched. Its near-zero
  `predicted_shift` win-rate may be a real reflection of route arbitration decisions genuinely
  rarely flipping, not the same bug class as chat's fixed-divisor problem — flagged as an open
  question, not assumed broken.

## PR link

<to be filled in after push>
