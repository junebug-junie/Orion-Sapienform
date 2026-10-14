# feat(metacog): self_signal_episode rows from the attention seam's calibration

## Summary

- Metacog now writes an entry when one of Orion's own body signals stays unusual **for itself** for longer than a single event can explain, and a short record when it settles back. This is spec section D ("Metacog becomes a reader", PR #2528, approved; "hit it" 2026-10-14).
- The judgement "is this signal high/unusual in its bad direction" is not re-derived: every poll runs the attention seam's own `world_first.node_candidate` + `judge_candidate` (PR #2583) over the same `substrate_node_prediction_error_history` window the attention contests read.
- One open row (drafted into prose by the existing metacog writer) and one close row (no prose) per episode. cortex-orch and orion-actions skip the close row; sql-writer persists both to `metacog_trigger` unchanged.
- Silence is not calm: an unreadable history, a stale node, too little history or a placeholder never opens or closes an episode as if calm; a long silence closes as `went_silent`.
- Flag `EQUILIBRIUM_METACOG_SELF_SIGNAL_EPISODES_ENABLED` ships on (env_example, settings default, compose default, primary-checkout local `.env`).

## Outcome moved

Metacog gains a reader of calibrated body signals. On 7 days of real stored history the gate would have written **~9.5 open rows/day** (67 episodes over 7.03 days, max 3 in any hour). Opening on the first hot poll instead would have written ~110/day, because "high" is the top decile of each signal's own week and so is hot ~10% of the time by construction.

## Current architecture

- `orion-equilibrium-service` owns every metacog trigger kind and publishes `MetacogTriggerV1` on `orion:equilibrium:metacog:trigger` → cortex-orch dispatches `log_orion_metacognition` (LLM) for every trigger; orion-actions composes a `metacog_digest` journal for every trigger; sql-writer persists each trigger to `metacog_trigger`.
- Calibrated per-node magnitudes (`PredictionErrorMagnitudeV1`: 7-day percentile/band) existed only inside the attention contests (substrate-runtime broadcast, attention-runtime field contest). No metacog path read them.
- Existing episode machinery to mirror: `orion/metacog/anomaly_episode.py` (Redis-persisted state, calm gap, publish-then-mark), transport baseline gate (open/close rows, hourly budget).

## Architecture touched

- **New pure module** `orion/metacog/self_signal_episode.py`: observation classification from a seam verdict, per-signal lifecycle (pending → open → close), payload, persistence, open budget, `trigger_wants_prose`.
- **Equilibrium**: `app/self_signal_episode_gate.py` (adapter + gate), `app/pe_history_reader.py` (read-only Postgres reader, raises on failure so the gate reads "absent"), a 30 s poll loop in `app/service.py`, Redis state key.
- **Consumers**: cortex-orch `dispatch_metacog_trigger` and orion-actions `_handle_journal_metacog` return early for a `self_signal_episode` close row.
- **evidence_map**: `map_self_signal_episode` (open: high → degraded, unusual → critical, same cut points as the seam's bands; close: nominal).

## How the lifecycle works (and why each number)

- **hot** = the seam's verdict is eligible (internal, band high/unusual in its bad direction). **calm** = a real reading that is not eligible. **faded** = an event-written node whose last event is past the 300 s orienting window (treated as calm; close row says `faded`). **unknown** = absent / stale / insufficient history / placeholder / no polarity.
- **Open (dwell):** hot for MORE than 300 s within the trailing 600 s. Anchor: the seam's `EVENT_ORIENTING_WINDOW_SEC` (measured 2026-10-10: one event's consequence on the body is gone by 300 s). An episode is the body staying unusual longer than one event can explain. A test asserts the two constants are equal.
- **Close (calm gap):** 600 s of continuous calm/faded. On live data 47-64% of the calm runs between two hot polls of biometrics, bus_synaptic and execution are shorter than 600 s, i.e. flicker inside one episode. Pending runs that calm down write nothing.
- **Unknown** resets the calm clock and never opens; 3600 s of it closes an open episode as `went_silent`.
- **Restart:** state in Redis (`equilibrium:self_signal_episodes:v1`) → open episodes resume, no burst. No persisted state at all → a signal already hot on the first poll is adopted silently (no open, no close).
- **Budget:** 6 opens/hour across signals (replay max is 3/hour, so it does not bind); extra episodes are tracked silently.
- **Caps:** evidence refs ≤ 200 per episode (onset ref always kept); remembered hot-poll times limited to the trailing window; close-row outbox ≤ 50.

## Signals that can open episodes (7-day replay, 2026-10-07 02:45Z → 2026-10-14 03:26Z)

| Signal | Episodes (per day) | Duration p50 / max | Hot share of polls | Notes |
|---|---|---|---|---|
| `bus_synaptic` | 22 (3.1) | 52 min / 2.1 h | 8.5% | per-tick level |
| `execution` | 17 (2.4) | 23 min / 36 min | 5.7% | event-written |
| `cabinet` | 12 (1.7) | 21 min / 36 min | 1.6% | trigger (rise ≥ 0.5 °C/15 min) |
| `route` | 9 (1.3) | 21 min / 54 min | 1.7% | event-written count |
| `codebase` | 5 (0.7) | 45 min / 60 min | 2.9% | event-written |
| `biometrics` | 2 (0.3) | 4.2 h / 5.2 h | 6.7% | per-tick level; long sustained runs |
| `chat` (node) | 0 | — | 1.4% | eligible, but one turn's reading is news for ≤ 300 s, so it never out-lasts the dwell this week. Not excluded. |

**Never open (by the seam's own rules, not a list):**
- `harness_closure`: glossary `value_kind: placeholder` (a hard-coded 0.65) — the seam never treats it as a measurement. Live today it is also `insufficient_history` (183 rows).
- `perception`: an external (world) source; section D is internal only.
- `vision_organ`: has no rows in the history table.

**Known real body spikes:**
- **10-08 06:00Z GPU shed** (923 gpu_pool `unavailable` refusals that hour): a `bus_synaptic` episode opens at 06:05Z (onset) and lasts 2 h. Pinned by the eval.
- **10-07 cabinet heating run** (06:00-15:00Z, ahead of the first GPU-pool refusal at 14:05Z): 6 cabinet episodes. Pinned by the eval. Caveat: the history table starts 10-07 02:45Z, so on 10-07 the percentiles are against hours, not 7 days.
- **10-08 14:00-19:00Z shed:** not distinguished — opens during those hours (route 19:53Z) run at the baseline rate. The PE nodes measure surprise, and a sustained thermal plateau stops events rather than surprising them; the cabinet trigger only fires on a rise. Reported, not tuned around.
- **09-28 gateway timeout storm:** cannot be replayed — the PE history table is pruned at 168 h and starts 10-07.

## Metric quality gate (each signal allowed to open episodes)

1. **Provenance.** Values are each node's `prediction_error` as written by orion-substrate-runtime: `orion/substrate/prediction_error.py` (`codebase_prediction_error` L293, `execution_prediction_error` L456, `biometrics_prediction_error` L567, `chat_prediction_error` L607, `route_prediction_error` L717, `bus_synaptic_prediction_error` L826) and the cabinet warming error (`services/orion-substrate-runtime/app/worker.py:1425`), persisted one row per real `observed_at` advance by `store.save_prediction_error_history_samples` (`store.py:775`, called from the broadcast tick ~`worker.py:3028`). Lineage card: `check_metric_lineage.py --metric prediction_error` (producer orion-field-digester / substrate-runtime, polarity higher_is_worse); node-qualified semantics (value_kind, rest, sparsity, absent_means) come from `config/field/field_channel_glossary.v1.yaml` via `world_first.node_prediction_error_semantics`.
2. **Independence.** Episodes are per signal, never summed into one score. Some share causes (a chat turn moves execution, route and chat); in replay only 2 of 67 episodes opened within 10 min of another signal's.
3. **Theory anchor.** "Unusual for itself" = percentile against the signal's own 7 days (`PredictionErrorMagnitudeV1`, Juniper-approved seam, 2026-10-10). Dwell = the seam's measured 300 s orienting window. No new threshold is invented.
4. **Live data.** Pulled all 61,734 stored rows (the whole table) and replayed them. Every signal spends most polls calm (rest is reachable), every close in replay is `calm`/`faded` (none `went_silent`). Not a decay artifact: `prediction_error` was removed from every decay set 2026-07-26, and history rows are only written when `observed_at` moves. Live smoke from the built image against production Postgres (read-only): all 8 internal nodes read; at 03:5xZ none hot (codebase `faded`, harness_closure `insufficient_history`).
5. **Existing mechanism.** No PE-episode path existed. `orion/substrate/metacog_trend_signals.py` renders the latest execution/biometrics PE into the metacog prompt as a cue; it is not a trigger.
6. **Reversibility.** One flag; no schema/table change; trigger_kind is a free string and upstream a free dict; one Redis key.

## Files changed

- `orion/metacog/self_signal_episode.py`: pure lifecycle, payload, persistence, budget, `trigger_wants_prose`.
- `orion/metacog/evidence_map.py`: `map_self_signal_episode`.
- `orion/schemas/telemetry/metacog_trigger.py`: trigger_kind description documents the new kind and `upstream.phase`.
- `services/orion-equilibrium-service/app/self_signal_episode_gate.py`: seam-judged observations, gate, outbox, budget.
- `services/orion-equilibrium-service/app/pe_history_reader.py`: read-only PE history reader.
- `services/orion-equilibrium-service/app/service.py`: poll loop, Redis load/persist, shutdown.
- `services/orion-equilibrium-service/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: 8 new keys, docs.
- `services/orion-cortex-orch/app/orchestrator.py`: close row not dispatched.
- `services/orion-actions/app/main.py`: close row not journaled (`should_journal_metacog_trigger`).
- `scripts/analysis/replay_self_signal_episodes.py`: replay over DB/fixture through the live code path.
- Tests: `orion/metacog/tests/test_self_signal_episode.py`, `orion/metacog/tests/test_evidence_map_producer_contract.py`, `services/orion-equilibrium-service/tests/test_self_signal_episode_gate.py`, `services/orion-cortex-orch/tests/test_metacog_self_signal_episode_prose.py`, `services/orion-actions/tests/test_metacog_self_signal_episode_journal.py`.
- Eval: `services/orion-equilibrium-service/evals/test_self_signal_episode_replay_eval.py` + fixture `self_signal_pe_history_2026-10-07_2026-10-14.csv.gz` (internal nodes, 397 KB).
- CI: `.github/workflows/self-signal-episode-tests.yml`.

## Schema / bus / API changes

- Added: `trigger_kind="self_signal_episode"` on the existing `orion:equilibrium:metacog:trigger` channel; `upstream` keys `phase`, `prose`, `signal_id`, `episode_id`, `opened_at`, `confirmed_at`, `band_at_open`, `percentile_at_open`, `raw_at_open`, `peak_*`, `duration_sec`, `close_reason`, `evidence_refs` (≤ 200, each `substrate_node_prediction_error_history:<node>@<observed_at>`, the table's primary key), etc.
- Removed / renamed: none.
- Behavior changed: cortex-orch and orion-actions skip a `self_signal_episode` row with `upstream.phase == "close"`.
- Compatibility: `MetacogTriggerV1` is not `extra="forbid"` and `upstream` is a free dict — no model change. Consumer-first deploy still matters: an old cortex-orch/actions would draft and journal close rows too (an extra prose per episode until they are rebuilt).
- Evidence refs resolve while the history row is retained (168 h); the row itself carries the open/peak raw values and percentiles, so it stays self-describing after pruning.

## Env/config changes

- Added keys (orion-equilibrium-service): `EQUILIBRIUM_METACOG_SELF_SIGNAL_EPISODES_{ENABLED=true, POLL_INTERVAL_SEC=30, OPEN_DWELL_SEC=300, OPEN_WINDOW_SEC=600, CALM_GAP_SEC=600, SILENT_CLOSE_SEC=3600, MAX_OPENS_PER_HOUR=6, STATE_KEY=equilibrium:self_signal_episodes:v1}`. Postgres DSN reuses `EQUILIBRIUM_METACOG_GENERATIVE_POSTGRES_URI`.
- Flag name: the spec says `METACOG_SELF_SIGNAL_EPISODES_ENABLED`; it is namespaced `EQUILIBRIUM_…` like every other key of this service.
- `.env_example` updated: yes. Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-equilibrium-service` (wrote all 8 keys into the primary checkout's `services/orion-equilibrium-service/.env`). Skipped keys: none.

## Tests run

```text
orion/metacog/tests + orion/attention + orion/schemas/tests: 443 passed, 1 failed
  (test_context_provenance::test_static_ctx_assignments_covered — fails identically on unmodified HEAD)
services/orion-equilibrium-service tests + evals: 241 passed
services/orion-cortex-orch tests: 224 passed, 31 failed (the same 31 fail on unmodified HEAD: test_workflow_lane x27, test_verb_runtime_rpc x2, test_ready, test_mind_skip_logging)
services/orion-actions tests: 252 passed
python scripts/check_env_template_parity.py: PASS
python scripts/check_definition_drift.py: no definition changes
orion/metacog/evals/run_capture_eval.py: PASS
```

## Evals run

```text
services/orion-equilibrium-service/evals/test_self_signal_episode_replay_eval.py: 8 passed (21 s)
per-day opens: 10-07 12, 10-08 9, 10-09 13, 10-10 11, 10-11 9, 10-12 5, 10-13 6, 10-14 2 (partial day)
- replay magnitude == compute_prediction_error_magnitude on sampled points
- no-dwell control: > 100/day
- 10-08 06:00Z GPU shed opens bus_synaptic; 10-07 cabinet heating opens cabinet
- harness_closure never opens; closes are calm/faded only
```

## Docker/build/smoke checks

```text
docker build -f services/orion-equilibrium-service/Dockerfile .  -> OK
in-image import of app.service / gate; glossary resolves (execution=level, harness_closure=placeholder); flag default True
live read-only smoke (image on app-net, production Postgres, SET default_transaction_read_only):
  61,449 rows read; 8 internal nodes observed; none hot at smoke time
scripts/safe_docker_build.sh orion-equilibrium-service build: refused before compose — the worktree has no root .env
  (the wrapper's --env-file .env); not deployed, per instructions
```

## Review findings fixed

(filled in after review)

## Restart required

Not deployed. Consumer-first, from the primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-cortex-orch up -d --build && scripts/safe_docker_build.sh orion-actions up -d --build && scripts/safe_docker_build.sh orion-equilibrium-service up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the 10-08 afternoon GPU shed is not distinguished by these signals; PE nodes measure surprise, not sustained load. Mitigation: reported; no tuning around it. A load/thermal level signal would be a separate seam source.
- Severity: low. Concern: each open row still gets both a metacog draft and an orion-actions `metacog_digest` journal (existing behavior for every trigger kind). Mitigation: unchanged here; the close row gets neither.
- Severity: low. Concern: the replay's first day is scored against hours, not 7 days, of history (table began 10-07). Mitigation: stated; in production the window is full.
- Severity: low. Concern: a flickering pending run can stretch onset-to-open latency (replay p50 8 min, max 70 min). Mitigation: `opened_at` is the onset, `confirmed_at` the open row time; both are in the row.
- Follow-up (not done here, per instructions): no remaining bespoke trigger is clearly superseded. `telemetry_anomaly` (autoencoder), `transport` (per-hop latency) and `insight` (confidence recovery) read different evidence. The `metacog_trend_signals` prompt cue (latest raw execution/biometrics PE) could later read open self-signal episodes instead of raw values.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
