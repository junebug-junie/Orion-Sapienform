# Hub Attention Organ operator tab — PR report

Branch: `feat/hub-attention-organ-tab`
Date: 2026-07-30
Status: **DONE_WITH_CONCERNS** (see Risks / concerns — the concerns are live findings this tab
surfaced about *existing* signals, not defects in the tab itself)

## Summary

- New top-level Hub tab, **Attention Organ**, visualizing in near-realtime the three live pieces of
  `docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-and-heartbeat-discrimination-design.md`,
  which had no human-visible surface anywhere: the heartbeat ensemble, the AST/HOT row that consumes
  it, and each Active-Inference domain's raw prediction error.
- Two panels are the design doc's own **acceptance checks computed live** rather than by eyeballing
  container logs: verdict-band distribution over a window (Acceptance Check 1) and `predicted_shift`
  domain dominance (Acceptance Check 3).
- New read-only Hub API (`/api/attention-organ/snapshot` + `/history`) that fans out to three
  independent backends and degrades each one separately.
- Additive `orion-heartbeat` inspectability so the tab reports **values a real producer produced**
  rather than mirroring another service's constants: per-trajectory `ratios` on `/h1`, and `config`
  (live tuning + verdict band edges), `last_reheat`, `organ_site_map`, `absorb_queue_maxsize` on
  `/health`.
- Read-only throughout: no writes, no bus publish, no new channel, no new schema.

## Outcome moved

The design spec's central question — *does the attention organ actually discriminate, and is its
signal reaching AST/HOT* — was previously answerable only by `docker logs` plus manual Postgres and
FalkorDB queries. It is now a tab. Concretely, within minutes of the tab existing it surfaced two
live problems that were not previously visible (see Risks / concerns).

## Current architecture (before this patch)

- `orion-heartbeat` ran an 8-trajectory quimb MPS dissipation ensemble, exposing `/h1` (mean ratio,
  std, verdict, tick_count, seeds) and `/health` (event counters). Registered in
  `orion/bus/channels.yaml` as a read-only research consumer that "publishes nothing back."
- `orion-substrate-runtime`'s `_attention_self_model_tick()` fetched that `/h1` and threaded it into
  `AttentionSelfModelV1` as `heartbeat_mean_ratio`/`heartbeat_verdict`/`heartbeat_basis`, persisting
  to `substrate_attention_self_model` on a ~30s tick (PR #1459 + the 2026-07-29 heartbeat-into-AST/HOT
  patch).
- Per-domain `prediction_error` lived as a flat property on `node:substrate.<domain>` in the
  `orion_substrate` FalkorDB graph, read back by `_brain_frame_prediction_error_by_domain()`.
- None of these three had any UI. `AttentionSelfModelV1` still has no bus channel and no downstream
  consumer — this patch does not change that.

## Architecture touched

- **services/orion-hub** — new read-only route module + new tab panel + tab-router registration.
- **services/orion-heartbeat** — additive fields on the existing debug HTTP surface only. No change
  to the substrate, the ensemble mathematics, the dissipation loop's behavior, or what it consumes.
- No contract surfaces touched: `orion/bus/channels.yaml`, `orion/schemas/registry.py`, and
  `orion/schemas/` are all unchanged.

## Files changed

- `services/orion-hub/scripts/attention_organ_routes.py` (new): the read-only API. Pure helpers
  (`parse_predicted_shift_domain`, `summarize_history`, `build_domain_rows`, `reconcile_confidence`)
  are separated from I/O so the aggregation logic is testable against fixed rows.
- `services/orion-hub/static/js/attention-organ.js` (new): rendering + poll lifecycle. Vanilla,
  inline SVG, no external deps, `textContent` only.
- `services/orion-hub/tests/test_attention_organ_page.py` (new): 34 tests.
- `services/orion-hub/scripts/api_routes.py`: register the router.
- `services/orion-hub/templates/index.html`: nav anchor, panel section with nine mount points,
  script tag.
- `services/orion-hub/static/js/app.js`: the six registration points `setActiveTab` needs, plus the
  `activate()`/`deactivate()` poll lifecycle.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: three new keys.
- `services/orion-heartbeat/app/substrate/ensemble.py`: `EnsembleH1ResultV1.ratios`.
- `services/orion-heartbeat/app/substrate/reconstruction.py`: populate `ratios`; new
  `verdict_thresholds()`.
- `services/orion-heartbeat/app/service.py`: record `last_reheat`; expose `config`,
  `organ_site_map`, `absorb_queue_maxsize` on `/health`.
- `services/orion-heartbeat/tests/{test_reconstruction_h1,test_http_endpoints}.py`: 4 new tests.

## Schema / bus / API changes

- **Added** (HTTP debug surfaces only, no bus/schema registry impact):
  - `orion-heartbeat` `GET /h1` → `h1.ratios: list[float]`, index-aligned with `seeds`.
  - `orion-heartbeat` `GET /health` → `config` (n_trajectories, gamma, base_decay_prob,
    decay_spread_sensitivity, reheat_strength, reheat_prob_scale, decay_reheat_interval_sec,
    h1_interval_sec, high_ratio, low_ratio), `last_reheat`, `organ_site_map`,
    `absorb_queue_maxsize`.
  - Hub `GET /api/attention-organ/snapshot`, `GET /api/attention-organ/history?minutes=`.
- **Removed / renamed**: none.
- **Behavior changed**: none. Every heartbeat addition is a new key on an existing response;
  existing keys are untouched.
- **Compatibility**: the Hub frontend treats every new heartbeat field as optional and renders an
  explicit "not reported by this build" note when absent, so a Hub running against an older
  heartbeat degrades honestly rather than breaking.

## Env/config changes

- Added keys (services/orion-hub): `FALKORDB_SUBSTRATE_GRAPH` (default `orion_substrate` — same env
  key and default `build_falkor_substrate_store_from_env()` already uses, deliberately not a second
  name for the same graph), `HUB_HEARTBEAT_BASE_URL` (default `http://localhost:7251`),
  `HUB_ATTENTION_ORGAN_TIMEOUT_SEC` (default `3.0`).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes. `docker-compose.yml` inline `${VAR:-default}` fallbacks also set to
  the real working values, because a worktree with no Hub `.env` falls through to those, not to
  `.env_example` — the exact gap that silently killed the AST/HOT tick for an hour on 2026-07-29.
- local `.env` synced: yes. `scripts/sync_local_env_from_example.py` skips this service in a fresh
  worktree ("no .env"), so the live `/mnt/scripts/Orion-Sapienform/services/orion-hub/.env` was
  updated directly. `FALKORDB_SUBSTRATE_GRAPH` was already present there (a pre-existing
  `.env`/`.env_example` drift this patch also closes); the two `HUB_*` keys were appended.
- Skipped keys requiring operator action: none.

**`HUB_HEARTBEAT_BASE_URL` is the published port, not the compose DNS name, on purpose.** Verified
live: Hub runs on the host network in this deployment, and `orion-athena-heartbeat` does not resolve
from inside the Hub container while `http://localhost:7251` does.

## Tests run

```text
services/orion-heartbeat$ ORION_BUS_ENABLED=false pytest tests -q
57 passed, 14 warnings in 18.62s

services/orion-hub$ pytest tests/test_attention_organ_page.py -q
34 passed, 15 warnings in 5.33s

services/orion-hub$ pytest tests -q            # full suite
35 failed, 1068 passed, 5 skipped in 171.01s
```

The 35 hub failures are **pre-existing and not caused by this patch**, established by running the
same suite on a scratch worktree at `origin/main`:

```text
origin/main baseline:  34 failures
this branch:           35 failures
symmetric difference:  1 test, and it is a swap WITHIN the same file
  only on branch:   test_substrate_mutation_manual_route_routing.py::test_routing_apply_succeeds_for_auto_promote_and_can_rollback
  only on baseline: test_substrate_mutation_manual_route_routing.py::test_routing_dry_run_produces_trial_and_decision_without_side_effects
```

That file is flaky on both, confirmed by running it in isolation 3× on each:

```text
this branch:  7 passed / 1 failed / 7 passed
origin/main:  1 failed / 1 failed / 1 failed
```

Two of the pre-existing failures assert on `app.js` strings, so they were checked directly rather
than assumed: `openResponseFeedbackModal('up', meta, text)` and
`const substrateTabButton = document.getElementById("substratePageLink");` are absent from
`origin/main`'s `app.js` as well (`grep -cF` → 0 on both). This patch only ever inserted into
`app.js`; every edit was an exact-match replacement asserted `count == 1` and replaced with a
superset.

## Evals run

```text
No eval harness exists for services/orion-hub or services/orion-heartbeat.
```

Neither service has an `evals/` directory. Not created here: the meaningful eval for this surface
would be "does the organ discriminate," which is the *subject* of the design spec's own acceptance
checks and belongs to `orion-heartbeat`'s calibration harness
(`scripts/analysis/measure_heartbeat_ensemble_calibration.py`), not to a read-only viewer of it.
Flagged as a follow-up rather than silently claimed.

## Docker/build/smoke checks

```text
$ scripts/safe_docker_build.sh orion-heartbeat up -d --build
Image orion-heartbeat-heartbeat Built
Container orion-athena-heartbeat Recreated / Started

$ curl -fsS http://localhost:7251/h1
{"ok":true,"h1":{"mean_ratio":0.9041836228299582,"std_ratio":0.03225184412851619,
 "verdict":"redundant","tick_count":90,"seeds":[42,...,49],
 "ratios":[0.94515,0.91386,0.88107,0.90808,0.93307,0.83436,0.92000,0.89787],...}}

$ curl -fsS http://localhost:7251/health
... "last_reheat":{"raw_mean_abs_gap_zscore":29.27,"zscore_saturation":3.0,
     "signal":1.0,"reheat_prob":0.02,"at":"2026-07-30T20:11:50Z"},
    "config":{...,"high_ratio":0.6,"low_ratio":0.2}, "organ_site_map":{...}
```

Hub endpoints exercised through the real `api_routes.router` against live Postgres + FalkorDB +
heartbeat (Hub itself not redeployed — see Restart required):

```text
GET /api/attention-organ/snapshot        -> 200 in 167ms
  heartbeat.ok      True
  reconciliation    verdict=within_drift  delta=0.0022  row_age=0.3s
  link              self_model_mean_ratio 0.7711 == live_mean_ratio 0.7711,
                    basis tick_count 6184 == live tick_count 6184
GET /api/attention-organ/history?minutes=60 -> 200
  sample_count 119, verdict_counts {'redundant': 116}, null_verdict 3,
  domain_counts {'execution': 68, 'bus_synaptic': 40, 'biometrics': 9}
GET /api/attention-organ/history?minutes=7  -> normalized to 60
```

**Metric quality gate** (CLAUDE.md §0A) for the one derived number this tab computes,
`reconciliation.recomputed`:

1. *Provenance*: `1 - mean(prediction_error over ACTIVE_INFERENCE_DOMAINS)`, the literal formula in
   `_unconditional_prediction_error_confidence()`. Inputs are the flat `prediction_error` property
   on `node:substrate.<domain>` — the same values `_brain_frame_prediction_error_by_domain()` feeds
   the reducer.
2. *Independence*: it is not independent, and is not claimed to be — it is deliberately a
   **redundant recomputation** of a value the reducer already stored, whose entire purpose is to
   diff the two.
3. *Theory anchor*: none needed; it is an identity check, not a new measurement.
4. *Live-data sanity*: verified by hand against real data — five live domain values
   (execution 1.0, bus_synaptic 1.0, biometrics 0.102157, chat 0.013468, route 0.00025) give
   `1 - mean = 0.5768`, exactly the persisted `prediction_error_confidence` of the row written at
   the same time.
5. *Existing mechanism*: none — no existing surface compared these two.
6. *Reversibility*: trivially removable; nothing consumes it.

## Review findings fixed

Code review ran in a subagent per CLAUDE.md §12. All three must-fix and all seven should-fix
findings were fixed.

- **M1 — poll timer never stops when leaving via the "Self" tab.**
  `self_observability.js` hides every `section[data-panel]` directly and `preventDefault()`s its own
  click, and `app.js` has zero references to it — so `setActiveTab`, the only caller of
  `deactivate()`, never runs on that path. The timer would have kept firing forever into a hidden
  panel: 2 HTTP calls + a Postgres query + a FalkorDB query every 2–5s.
  - Fix: the interval body checks real DOM visibility and self-deactivates, so the contract holds
    against any panel that hides it, not just the current router.
  - Evidence: `test_attention_organ_js_guards_polling_on_real_visibility` inspects the
    `startTimer`→`activate` region for both the visibility check and the `deactivate()` call.

- **M2 — fabricated `tick lag` when the organ is offline.**
  Global `isFinite(null)` is `true`, so an unreachable heartbeat (`live_tick_count: null`) rendered
  a confident negative lag in ordinary gray, next to two rows honestly showing `—` for the same
  null. Reviewer reproduced it: `lag rendered: -1614`.
  - Fix: `Number.isFinite`. The `lag > 5000` amber threshold was also removed — `tick_count` counts
    absorbed events at a rate set entirely by live organ traffic, so no fixed threshold means
    anything (and 5000 was larger than the metric had ever been). Staleness is now keyed on row age,
    which is a real duration.
  - Evidence: `test_attention_organ_js_uses_strict_number_isfinite_for_nullable_readings`.

- **M3 — blocking I/O on the event loop at a 2s cadence.**
  Both handlers were `async def` with blocking `requests.get` ×2, blocking SQLAlchemy, and blocking
  redis — up to ~6s of a frozen Hub (chat, websockets, every route) per poll. The pattern exists in
  older read-only route modules, but those are user-initiated one-shots, not an always-open tab.
  - Fix: dropped `async`; FastAPI runs sync handlers in its threadpool.
  - Evidence: `test_attention_organ_routes_are_sync_handlers` asserts via `inspect.iscoroutinefunction`.

- **S1 — history failure masked, stale history rendered as current.** The "History unavailable"
  status was immediately overwritten by "Updated <now>", while `lastHistory` kept rendering an old
  window under its own `window_minutes`/`sample_count` labels.
  - Fix: `state.historyError`, `"history"` pushed into the status line's degraded list, and a
    banner on both history-backed panels stating how old the data actually is.

- **S2 — hardcoded `"0.2"` band edge**, the exact mirrored constant `verdict_thresholds()` was added
  to eliminate. Fix: with no live `config`, the sentence omits the number entirely.

- **S3 — `last_reheat` staleness invisible.** `service.py` only overwrites it after a *successful*
  tick, so a persistently failing FalkorDB query serves the last good block forever. Fix: aged
  against the loop's own `decay_reheat_interval_sec` and flagged amber past 5×.

- **S4 — new Redis client + connection pool per request**, never closed (the class has no
  `close()`). Not an unbounded leak, but ~30 TCP connect/teardowns per minute for a 7-row query.
  Fix: module-level cache, matching `_engine()` directly above it.

- **S5 — unlabeled `MATCH (n)`** on the substrate concept graph, with no index on `node_id`
  (confirmed: `CALL db.indexes()` is empty), scanned every poll. Fix: `MATCH (n:SubstrateNode)`.
  `STARTS WITH` kept over an `IN $ids` form so an unknown future domain node is still discovered.

- **S6 — `heartbeat.ok` true when there is no H1 at all.** `/h1` answers HTTP 200 with
  `{"ok": false, "reason": "no_h1_computed_yet"}` before the first ensemble tick; the transport-level
  flag reported healthy while the panel showed a red "no H1" block. Fix: `and live_h1 is not None`.
  Evidence: `test_snapshot_is_not_ok_when_heartbeat_is_reachable_but_has_no_h1_yet`.

- **S7 — no PR report.** This file.

- **T2/T3 — tests were string-matching, not behavior.** Added 9 tests that execute the handlers with
  stubbed backends, covering snapshot's independent-degradation contract (all three failure
  branches), the `/history` SQL, and the `truncated` flag. Added `malformed_row_count` so a window
  where every row failed to parse cannot look identical to a window with no activity.

- **N1/N2/N4/N5/N6 fixed** (time-proportional x-axis so an outage draws as a gap rather than an
  unbroken line; lag threshold removed; trace relabeled "ensemble ticks" since it deduplicates on
  `tick_count`; a comment that asserted the opposite of the real load order corrected;
  `absorb_queue_maxsize` given the consumer it was shipped without). **N7 fixed** by dropping two
  unrendered fields from the history series payload. N3 left as-is — it degrades honestly.

### Found by live verification after the review, not by the review

The reviewer confirmed `reconcile_confidence`'s `abs(delta) <= 0.05` tolerance matched the reducer
exactly on the data it was tested against. A later live poll returned `delta = -0.0908` →
`reconciles: False`, and the cause was not a defect in either signal: the row is written on
substrate-runtime's ~30s tick while the graph read happens now, and `execution`/`bus_synaptic`
genuinely swing across their whole range between ticks. A flat tolerance therefore reported
"divergent" **as a function of when the operator happened to poll** — a check firing on its own
sampling schedule, which is precisely the kind of instrument this tab exists to expose.

Replaced with a stated model rather than a tuned constant: one of five domains traversing its full
range can move the mean by at most `1/5 = 0.2`, so a delta within that is `within_drift`; beyond it
would need two or more domains to have moved fully, which points at a source mismatch. Past a 60s
row age the comparison carries no information either way and the verdict is withheld (`unknown`)
rather than manufactured. `basis` states which rule fired. Covered by
`test_reconcile_confidence_does_not_cry_divergence_on_ordinary_sampling_skew` and
`test_reconcile_confidence_withholds_a_verdict_on_a_stale_row`.

## Restart required

`orion-heartbeat` is already rebuilt and running with this branch's code (see Docker checks). Note
its ensemble restarts from a fresh random state on every deploy — v0 has no crash-safe persistence
by design, so `tick_count` reset from ~35,700 to 0.

Hub still needs a redeploy to pick up the new route module and settings. `templates/` and `static/`
are volume-mounted **relative to the checkout the deploy runs from**, so this must be run from the
main checkout after merge, not from the worktree — deploying Hub from a worktree would repoint those
mounts at a directory that disappears when the worktree is pruned:

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
scripts/safe_docker_build.sh orion-hub up -d --build
curl -fsS http://localhost:8081/api/attention-organ/snapshot | head -c 400
```

(That wrapper refuses to run from the shared checkout by design. Deploying Hub is the documented
exception that needs `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1`, set consciously for this one command.)

## Risks / concerns

Both concerns are **live findings about existing signals that this tab made visible**. Neither is a
defect in this patch, and neither is fixed here — fixing either is a heartbeat/substrate calibration
change that needs its own metric-quality-gate pass and its own proposal.

- **Severity: should-fix. The `bus_synaptic` reheat driver is fully saturated, so it is not gating
  anything.** `last_reheat.raw_mean_abs_gap_zscore` reads **~29.3** against a
  `BUS_SYNAPTIC_ZSCORE_SATURATION` of **3.0**, making `signal = min(1, z/3.0)` a constant **1.0**
  and `reheat_prob` a constant `0.02` regardless of how bus activity moves. Verified stable across
  three direct FalkorDB samples (29.30 / 29.31 / 29.27 over 436 edges, max |gap_zscore| 7087.8).
  The design spec's own calibration states this raw value "sits around 1.0-1.1 during normal
  operation" — it is ~30× that. The whole point of the reheat term is to be a *dynamic* driver;
  clipped at its ceiling it is a constant. The tab flags this explicitly with a red "saturated"
  badge and an explanation.
  *Mitigation*: none applied. Recommend re-deriving the saturation constant against real
  `orion_bus_synapse` data before the reheat term is trusted as dynamic — and checking whether the
  underlying `gap_zscore` distribution itself is healthy, given a max of 7087.

- **Severity: note. Acceptance Check 1 is still not met live, and the tab says so.** 24h of persisted
  rows carry exactly one distinct `heartbeat_verdict` (`redundant`). This is already documented in
  the spec as a structural gap deferred by explicit decision (2026-07-29), not a new finding — but
  the tab now states it as a live verdict rather than leaving it to a doc.

- **Severity: note. Two Active-Inference domains sit pinned at exactly 1.0.** `execution` and
  `bus_synaptic` both read a saturated `prediction_error = 1.0` on most ticks, dragging
  `prediction_error_confidence` into a bimodal 0.5768 ↔ ~0.96 pattern. `chat` and `transport` nodes
  are ~6 days stale, and `chat` is nonetheless still averaged into the aggregate by the upstream
  reducer. All are flagged in the tab (red at-ceiling badges, staleness ages) rather than rendered as
  ordinary values. Directly relevant to the spec's still-open Missing Question 3.

- **Severity: note. No eval harness** for either touched service (see Evals run).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1504
