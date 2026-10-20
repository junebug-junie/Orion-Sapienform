# PR: street events compete for attention (`node:substrate.street`)

Spec: `docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md` (PR #2803),
section 6 "How loud", plus Juniper's rule in PR #2763 (answer 7): the street may win
attention with events, not raw pixel surprise, with habituation, a camera-down check, an
outside-world label and a measured cap with a kill switch. Proposal mode: the spec's
"Privacy, failure modes, rollback" section.

Depends on #2816 (`StreetEventV1`, `orion:vision:street:events`), which is already merged to
`main`; this branch is merged with `origin/main`.

## Summary

- `orion-substrate-runtime` now listens to `orion:vision:street:events` with the hardened
  `run_subscription` listener (resubscribes after a drop; listed in `/health` `listeners`).
- Each `StreetEventV1` goes through one gate with a stated reason: a doorbell ring always
  passes; every other kind needs `p_value < 0.05`, every camera in `stream_ids` up, and fewer
  than `SUBSTRATE_STREET_ATTENTION_MAX_PER_HOUR` (4) passes in the trailing hour.
- New attention node `node:substrate.street`: written every 30 s with the held salience of a
  passed event (90 s), else a real 0.0. It is an outside-world source (`is_world_source_id`),
  admitted to attention, fenced from endogenous curiosity.
- New Rung 6 domain `street_handoff`: one raw reading `-ln(p_value)` per graded handoff.
- Trace: one `substrate_street_event` log line per event; `substrate_street_attention_won`
  when the broadcast picks it; counters on `/health` -> `street_attention`.

## Outcome moved

Before: nothing outside the house could take Orion's attention except raw walkway pixel
surprise, which is fenced out because it cannot tell events from noise. After: a doorbell ring
(and, once the handoff reducer ships, a surprising handoff) reaches the world-first contest as
a world source, and the correlation id from the retina rides on the node into the broadcast's
`attended_node_ids`. Baseline (live, 24 h to 2026-10-20 22:51 UTC): 2,018 broadcast cycles
(~43 s apart), 1,566 (78%) with no winner; winners perception 244, chat 64, execution 62.

## Current architecture

- Attention broadcast (`_attention_broadcast_tick`, every ~30-43 s) ranks every cognitive
  `node:substrate.*` node with a `prediction_error` by its percentile against its own 7-day
  history (`orion/attention/world_first.py`); external sources are eligible at band
  `high`/`unusual` (>= 0.9 strict-below percentile) and absent when the reading is > 180 s old.
- `StreetEventV1` had a producer (retina, rings) and no consumer.
- Rung 6 had two domains (execution, chat).

## Architecture touched

- `orion-substrate-runtime`: new listener + tick + `/health` block; the perception listener
  also records "last frame seen" per stream (any artifact, before the embedding check).
- Contracts: `orion/schemas/attention_candidate.py` (`STREET_NODE_ID`, world label, curiosity
  fence), `orion/substrate/prediction_expectation.py` (domain), glossary entry, metric lock.
- No schema change to `StreetEventV1`.

## Files changed

- `orion/substrate/street_attention.py`: pure gate, held value, counters, Rung 6 reading.
- `orion/schemas/attention_candidate.py`: `STREET_NODE_ID`; world source; curiosity fence.
- `orion/substrate/prediction_expectation.py`: `street_handoff` DomainSpec.
- `services/orion-substrate-runtime/app/worker.py`: listener, handler, tick, win counter,
  frame-seen hook.
- `services/orion-substrate-runtime/app/main.py`: `/health` `street_attention`.
- `services/orion-substrate-runtime/app/settings.py`, `.env_example`, `docker-compose.yml`,
  `README.md`: six new keys.
- `config/field/field_channel_glossary.v1.yaml`, `config/metrics/metric_definitions.lock.json`:
  the node's semantics (trigger, rest 0.0, per-tick writer). The lock also picks up
  `orion:personality:naming`, which `main` had left unlocked.
- `orion/bus/channels.yaml`: consumer comment (consumer list already named this service).
- Tests: `orion/substrate/tests/test_street_attention.py`,
  `services/orion-substrate-runtime/tests/test_worker_street_attention.py`,
  `tests/test_metric_rest_range.py` (count 11 -> 12).
- Evals: `services/orion-substrate-runtime/evals/run_street_attention_eval.py`,
  `test_street_attention_eval.py`, `run_street_gate_mutation_check.py`.
- `.github/workflows/street-attention-tests.yml`: runs all of the above in CI.

## Schema / bus / API changes

- Added: consumer of `orion:vision:street:events`; node `node:substrate.street`; Rung 6
  surface `prediction_expectation:street_handoff`; `/health` key `street_attention`.
- Removed / renamed: none.
- Behavior changed: the street can now win the substrate attention broadcast.
- Compatibility: `StreetEventV1` is read as-is (extra forbidden, no new fields).

## Env/config changes

- Added keys: `SUBSTRATE_STREET_ATTENTION_ENABLED=true`, `SUBSTRATE_STREET_EVENTS_CHANNEL`,
  `SUBSTRATE_STREET_ATTENTION_MAX_PER_HOUR=4`, `SUBSTRATE_STREET_ATTENTION_HOLD_SEC=90.0`,
  `SUBSTRATE_STREET_ATTENTION_TICK_INTERVAL_SEC=30.0`, `SUBSTRATE_STREET_CAMERA_UP_MAX_AGE_SEC=600.0`.
- `.env_example` updated: yes. Local `.env` synced: yes, `python scripts/sync_local_env_from_example.py --all-keys orion-substrate-runtime` (default mode skipped them; all six added to the primary checkout's `.env`).
- Skipped keys requiring operator action: none.
- Rung 6 has no per-domain flag; `street_handoff` rides `SUBSTRATE_PREDICTION_RAW_READINGS_ENABLED` and `SUBSTRATE_PREDICTION_EXPECTATION_LIVE_ENABLED` like the others.

## Gate as built

| kind | rule |
|---|---|
| `doorbell_ring` | always passes, even over the cap and with cameras dark; it still uses up the hourly budget |
| everything else | `p_value` present, `p_value < 0.05`, every stream in `stream_ids` sent a frame within 600 s, and fewer than 4 passes in the trailing hour |

"Unusual for the hour": no event-level reading exists that this runtime can read cheaply.
The rhythm learner grades per subject and window (`vision_percept_expectation`) and
`vision_rollup` hour rows exist only for closed hours. So the event's own `p_value` stands in
for the hour clause and the hourly cap does the habituation. A known person
(`person_guess`) is named in the reason but cannot change the verdict yet.

## Metric gate: `node:substrate.street.prediction_error`

1. **Provenance:** `StreetAttentionState.held_reading()` (`orion/substrate/street_attention.py`)
   -> `_write_street_node()` -> `_write_prediction_error_node()` in
   `services/orion-substrate-runtime/app/worker.py`, from `_street_attention_tick` (30 s) and
   `_process_street_event` (on pass). The value is the producer's `StreetEventV1.salience`
   (retina: 1.0 per ring, `ring_events.py`).
2. **Independence:** fed only by `orion:vision:street:events`. The ring comes from Ring's
   push API, not frames, so it is independent of `perception_walkway` (embedding drift). The
   camera-up check reads the same artifact stream as perception, but only as a yes/no
   precondition, never as magnitude. Future handoff events share the detector with every
   vision signal (spec metric gate, item 2).
3. **Theory anchor:** a trigger. A ring is a direct social bid (someone pressed the button).
   A handoff passes only on a calibrated tail probability (`p < 0.05` under split-conformal
   transit times, spec item 3). The node does not measure "how unusual"; the world-first
   contest does that against its own 7 days.
4. **Live sanity:** UNVERIFIED for real values: no `node:substrate.street` rows exist yet
   (`substrate_node_prediction_error_history`: 0 rows), and 0 `doorbell_ring` rows in
   `vision_events` over 14 days. Rest state is checked two ways. **Calm is genuine:** the
   value is rewritten as a real 0.0 every tick; nothing decays it (the field digester's
   `NODE_DECAY_CHANNELS` only touches FieldStateV1 node vectors, and this node is not one).
   **Silence is not calm:** with the listener down or no street camera frame for 600 s the
   tick writes nothing, so after 180 s attention reads the node absent, not quiet. Live inputs
   checked: the runtime scores ~14 walkway frames/min and ~14 doorbell frames/day (log,
   2026-10-20), so walkway reads "up" and the doorbell reads "up" only around its bursts.
   Synthetic week (eval): nonzero on 1.7% of ticks, quiet hour exactly 0.0, ring wins, 0.0
   never wins; under a flooding model nonzero 8.4%, still capped at 4/hour.
5. **Existing mechanism:** none for street events (searched `street`, `doorbell`,
   `StreetEventV1`, `vision:street`). Reuses `_write_prediction_error_node`, world-first
   ranking, `run_subscription`, the Rung 6 tuner.
6. **Reversibility:** cheap. `SUBSTRATE_STREET_ATTENTION_ENABLED=false`, or add the id to
   `NO_ATTENTION_CONSUMER_NODE_IDS`. No migration, no schema change; the Rung 6 corpus rows are
   pruned at 192 h.

## Rung 6 `street_handoff`

- Reading: `-ln(p_value)` nats per `handoff_met|late|missed`, keyed on `occurred_at` (a
  redelivered event hits the primary key, not a second row). p floored at 1e-6 (max 13.8 nats).
- Floor stated: under a calibrated model p is uniform, so `-ln p` is Exponential(1):
  E = 1 nat, Var = 1. A reading near 1 nat is the rest point, not signal.
- Default expectation alpha 0.2, min_variance 0.01; rollback window 24 h. The tuner needs
  480 readings before it can propose anything, so it stays silent for weeks.
- Consumer of the tuned expectation: `handoff_surprise_z` on the trace line.
- **Producer field needed:** the spec's reading is the arrival log-score, which
  `StreetEventV1` cannot carry. The producer must add `arrival_log_score: float | None`
  (nats: log density of the observed transit time under the predictive distribution; None
  for `handoff_missed`). Consumer-first: this runtime must accept the field before the
  producer sends it (extra is forbidden).

## Tests run

```text
pytest orion/substrate/tests/test_street_attention.py                       26 passed
pytest services/orion-substrate-runtime/tests/test_worker_street_attention.py \
       services/orion-substrate-runtime/evals/test_street_attention_eval.py  15 passed
pytest tests/test_attention_world_first*.py tests/test_attention_sustained_body.py \
       tests/test_attention_self_readings.py tests/test_metric_*.py \
       tests/test_street_event_bus_catalog.py orion/substrate/tests/test_prediction_expectation.py \
       orion/substrate/tests/test_endogenous_curiosity_walkway_fence.py orion/reverie/tests   279 passed
pytest services/orion-substrate-runtime/tests (whole dir)   505 passed, 13 failed -- the same
       13 fail on a clean origin/main checkout (cursor/quarantine/self-model/falkor tests,
       not in CI); test_grammar_consumer_integration.py does not import on main either
check_metric_lineage --gate / --prompt-semantics: PASS; check_definition_drift --gate: PASS
check_env_template_parity: PASS; check_inner_state_registry: OK
check_control_surface_store_parity: OK; check_async_routes_not_blocking: OK
```

## Evals run

```text
python services/orion-substrate-runtime/evals/run_street_attention_eval.py
  calibrated week: graded pass rate 0.0476, worst trailing hour 4 graded (5 with a ring),
  quiet hour max 0.0, nonzero ticks 1.7%, ring_wins true, calm_wins false, problems []
python services/orion-substrate-runtime/evals/run_street_gate_mutation_check.py
  11 mutations (ring bypass, p threshold, missing p, camera check, camera age, cap
  off-by-one, cap window, hold edge, refused event held, p floor) -- all killed
--live mode: UNVERIFIED until street events reach the bus (0 node rows today)
```

The first eval run found a real rule interaction: rings bypass the cap, so a trailing hour can
hold 5 passes (4 graded + 1 ring). That matches the spec (a ring always passes); the eval now
checks the cap on graded passes and reports the ring total next to it.

## Docker/build/smoke checks

```text
Not built here (no deploy from a worktree). The change is Python-only inside the existing
image; no new dependency.
```

## Review findings fixed

- Finding (self): the bus handler, the tick and the perception listener touch the street state
  from different threads; iterating the held deque while another thread appends can raise.
  - Fix: one `RLock` around every read and write; counters go through `bump()`.
  - Evidence: tests and mutation check re-run green.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```

Then check: `curl -fsS http://localhost:8115/health | jq .street_attention`
and `docker logs orion-athena-substrate-runtime 2>&1 | grep substrate_street_`.

## Risks / concerns

- Severity: medium. Concern: the node cannot win for its first ~200 history readings (~2.4 h
  at the live ~43 s broadcast cadence): band `insufficient_history`. Mitigation: none needed;
  stated so a ring in that window is not read as a failure.
- Severity: medium. Concern: the doorbell camera only sends frames around Ring bursts, so a
  handoff whose `stream_ids` include `doorbell` passes the camera check only within 600 s of a
  burst. That is the intended "camera down -> never surprising" rule, but it will refuse
  walkway->door arrivals when the doorbell was not recording. Mitigation: the producer already
  marks low coverage `unscorable`; revisit with live handoff data.
- Severity: low. Concern: the hourly cap counts gate passes, not actual broadcast wins.
  Mitigation: wins are counted separately (`attention_wins_total`) so the two can be compared.
- Severity: low. Concern: in-process state (cap budget, held events, camera times) resets on
  restart. Mitigation: a restart can at most allow one extra hour's budget; the camera check
  fails closed after a restart until frames arrive.

## PR link

TBD
