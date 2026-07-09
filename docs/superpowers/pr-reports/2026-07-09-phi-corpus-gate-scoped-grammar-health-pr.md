# fix(spark-introspector): scope grammar-degraded check to execution_trajectory lane

**Status:** IMPLEMENTED, tested, reviewed. Fixes a live production incident
discovered ~45 minutes after deploying specs 1-3 + the enable-switches PR
(#922) — corpus accrual was completely stalled.

## Summary

`orion-spark-introspector`'s `handle_self_state` computed
`grammar_degraded = gt.degraded or cognitive_lane_dark(gt)` — the blanket
`gt.degraded` flag trips whenever **any** substrate reducer's cursor lags,
including `chat_grammar_consumer`, which has nothing to do with φ's
execution/reasoning signals. In production this froze `phi_health` on every
single row (`chat_grammar_consumer` was ~19.4h stale simply because no chat
had occurred), and spec 3's corpus-hygiene gate correctly rejected every
frozen row at write time — the net effect was **zero seed-v4 rows accruing
at all**, silently, with the service otherwise running and ticking normally.

`cognitive_lane_dark(gt)` already existed as the properly-scoped check (only
cares about `execution_trajectory`, the reducer φ's cognitive features
actually depend on) — it just wasn't being trusted on its own; the blanket
flag was still OR'd in alongside it, defeating the point.

## Fix

- `worker.py`: `grammar_degraded = cognitive_lane_dark(gt)`. Dropped the
  blanket `gt.degraded` OR entirely. `cognitive_lane_dark` already fails
  closed on total fetch failure (`enabled_reducers={}`, from
  `_grammar_http_error`) and on `execution_trajectory` being disabled — so
  nothing is lost by removing the blanket flag.
- `substrate_reads.py`: `cognitive_lane_dark()` needed a second check added.
  It originally only looked at `reducer_health_by_name["execution_trajectory"]
  ["classification"]`, but that field — computed server-side by
  `ReducerHealthSnapshot.classify()` — only reflects heartbeat/stream-backlog
  state and **never reflects wall-clock cursor lag** (`cursor_wall_lag_sec`)
  at all. That means `execution_trajectory`'s own cursor going genuinely
  stale would have been invisible to the old scoped check, and was only ever
  caught (incidentally) via the blanket `gt.degraded` OR. Added a targeted
  read of `degraded_reasons` for an entry starting with
  `"cursor_lag:execution_grammar_reducer"` (its own cursor name), so
  execution_trajectory's own lag still freezes correctly, without pulling in
  unrelated reducers.

Confirmed this preserves the existing
`test_handle_self_state_grammar_truth_freeze` test's original intent
byte-for-byte, without modifying that test at all.

## Files changed

- `services/orion-spark-introspector/app/substrate_reads.py` —
  `cognitive_lane_dark()` gains the cursor-lag check; new
  `_EXECUTION_TRAJECTORY_CURSOR_NAME` local constant (duplicated as a stable
  literal rather than importing `orion.substrate.*` into this service).
- `services/orion-spark-introspector/app/worker.py` — one-line change plus
  comment explaining why.
- `services/orion-spark-introspector/tests/test_substrate_reads.py` — 6 new
  unit tests directly on `cognitive_lane_dark`: reproduces the exact bug
  (unrelated lag doesn't trip it), confirms execution_trajectory's own lag
  still does, multiple-reducers-lagging-including-own, dark classification
  still works, reducer-disabled still works, fully-healthy stays clean.
- `services/orion-spark-introspector/tests/test_inner_state_emit.py` — 1 new
  end-to-end test reproducing the live incident exactly through
  `handle_self_state`: `chat_grammar_consumer` lagging + `execution_trajectory`
  healthy → `phi_health == "ok"`, `grammar_truth_degraded is False`, corpus
  sink `.append()` called.

## Tests run

```text
pytest services/orion-spark-introspector/tests -q
  → 104 passed, 1 pre-existing unrelated failure (test_phi_reward_emitted_when_encoder_ok)
```

Every new test verified to fail without the fix and pass with it (reverted
the fix, confirmed the exact expected failures, reapplied).

## Review findings fixed

Self-reviewed (medium effort). No findings — the fix reuses the existing,
already-scoped `cognitive_lane_dark` mechanism rather than inventing a new
health model; confirmed no accidental widening (biometrics/transport_bus
reducer lag still correctly ignored, matching φ's actual dependency surface).

## Docker/build/smoke checks

Not run against containers in this environment. Live-verified the *bug*
directly against the running deployment (`docker logs`, `/grammar/truth`
endpoint) before writing the fix; the fix itself is unit/integration tested
only pending redeploy.

## Restart required

**This is blocking live corpus accrual right now.** Restart
`orion-spark-introspector` to pick up the fix:
```bash
docker compose --env-file .env --env-file services/orion-spark-introspector/.env \
  -f services/orion-spark-introspector/docker-compose.yml up -d --build
```
After restart, confirm with:
```bash
docker logs orion-athena-spark-introspector --since 2m | grep inner_state_corpus_row_rejected
python scripts/diag.py --corpus /mnt/telemetry/phi/corpus/inner_state.jsonl
```
Rejection log lines should stop appearing (or only appear when
execution_trajectory is genuinely down), and `diag.py`'s `rows_matching_version_and_healthy`
should start climbing from 0.

## Risks / concerns

- Severity: low. Narrows a health check, doesn't widen anything — the only
  behavior change is that an unrelated reducer's lag no longer blocks phi
  corpus writes. execution_trajectory's own degradation (both
  classification-based and wall-clock-lag-based) still freezes correctly.

## PR link

Branch pushed: `fix/phi-corpus-gate-scoped-grammar-health`.
Compare: https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/phi-corpus-gate-scoped-grammar-health
