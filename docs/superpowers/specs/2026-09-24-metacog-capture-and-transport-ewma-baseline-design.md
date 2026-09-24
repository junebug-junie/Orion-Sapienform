# Metacog capture fix: honest rows first, transport on self-learning baselines

Status: **design mode, not implemented.** Juniper approved the direction on 2026-09-24:
fix per-channel expected speed using an EWMA that cannot learn "busy" as normal, with the four
transport sub-fixes. Touches a trigger definition (a metric-meaning change), so the numbers
below are proposals to be confirmed by the log-only phase, not settled constants.

## Arsonist summary

Every metacog row costs an LLM call (about 2,400 a day), and almost none of them tell you
anything. The triggers underneath are mostly honest: each one carries its real numbers
(`metacog_trigger.upstream`). Everything after the trigger throws those numbers away:

- **Severity** is scored on how sure the *writing model* felt about its own paraphrase. The
  event's own numbers never touch it.
- **The density score** is a dead flag that only ever reads 0 or 0.25.
- **The prose** keeps saying "zen persists", because the model is handed a zen score that sits
  flat at 0.965.
- **Nothing reads the table.**

The single biggest producer, transport, is mostly measuring metacog itself. Each metacog
write-up is an LLM call that goes over the orchestrator's background channel and takes 10 to 20
seconds. The transport trigger compares that against a flat 5-second limit, fires, and starts
another metacog write-up. Seen live: in a 150-second sample, cortex-orch's *entire* RPC traffic was
one state call plus one ~17-second background call every 30 seconds. That is exactly the
30-second transport cooldown.

The fix is not more infrastructure:

1. Transport learns what normal looks like for each channel, and cannot quietly learn "busy" as
   normal.
2. Each trigger's own numbers are mapped straight into its row, deterministically.
3. A replay check over stored triggers proves it.

A stream of consciousness over metacog stays out of scope until the rows are worth reading. The
vehicle for that already exists as a design:
`docs/superpowers/specs/2026-07-29-stream-of-consciousness-hop-chain-design.md`.

## Current architecture (review, verified live 2026-09-24)

### Volumes: `orion_metacog`, 114,437 rows, 2026-07-23 → now

| trigger_kind | rows | last 24h | what it really measures |
|---|---|---|---|
| transport | 76,058 | 2,325 | RPC latency and timeouts, plus the bus_synaptic prediction-error edge |
| telemetry_anomaly | 36,530 | 103 | mood-arc encoder reconstruction loss against 3× its own p95 |
| baseline | 741 | 23 | scheduled heartbeat, always `scheduled_check` |
| chat_turn | 573 | 25 | real turn failures: alignment, compliance, strain, timeouts |
| flow / insight | 448 / 58 | 16 / 1 | attention self-model confidence patterns |
| relational, repair_pressure_trend, llm_surface_instability, manual | 25 total | 0 | rare |

Transport over the last 24 hours, by reason: `cortex-orch:success_latency_ms_p95` 1,903;
`cortex-exec:success_latency_ms_p95` 284; `cortex-exec:timeout_count` 75; `bus_synaptic` 57;
`rpc_timeout:*` 9.

### Pipeline

1. **Producers.** `orion-equilibrium-service` publishes `orion.metacog.trigger.v1` on
   `orion:equilibrium:metacog:trigger`. `orion-mind` publishes `llm_surface_instability` directly on
   the same channel and bypasses equilibrium's cooldown. Gates:
   - `transport_metacog_gate.py` (3 sources)
   - `telemetry_anomaly_metacog_gate.py`
   - `chat_turn_metacog_gate.py`
   - `repair_pressure_*_gate.py`
   - `_generative_metacog_poll_loop` (insight, flow)
   - `_maybe_emit_baseline_metacog_trigger`
2. **Raw sink.** sql-writer stores every trigger in `metacog_trigger`. **This table is honest data.**
3. **Draft.** cortex-orch `dispatch_metacog_trigger` (`orchestrator.py:891`, background lane) runs
   the `log_orion_metacognition` verb:
   - Context step: global spark snapshot, `trigger_upstream_json`, the eventfulness score (≤ 0.25).
   - Draft step: `log_orion_metacognition_draft.j2`, temperature 0.8. It writes summary, mantra,
     `what_changed`, and tags.
   - Uncertainty probe: a second LLM call.
   - Publish step: deterministic.
4. **Row.** `metacog.entry.v1` goes over `orion:metacog:sql-write` into `orion_metacog`.
5. **Consumers of `orion_metacog`: none in production.** One eval script reads it. The only live
   consumer of metacog at all is `orion-actions` `_handle_journal_metacog`, and it reads the
   *trigger*, not the entry.

### Where capture breaks (live evidence)

- **Severity ignores the event.** `compute_severity` (`orion/metacog/service.py:128-160`) uses the
  metacog pipeline's own failed-step count and the logprob margin of a temperature-0.8 paraphrase.
  Transport p95 by severity over the last 7 days:

  | severity | median p95 |
  |---|---|
  | critical | 10.2 s |
  | degraded | 10.1 s |
  | nominal | 15.8 s |

  Baseline rows with identical inputs split 110 critical and 43 degraded.
- **`causal_density`** has taken exactly two values in its whole history, 0 and 0.25 (see memory
  `project_metacog_causal_density_degenerate_todo`).
- **`state` describes the world, not the event.** It carries the global biometrics snapshot, not
  the trigger's evidence.
- **"zen" is a flat input the model narrates.** `metacognition_ticks.zen_score` averaged 0.965 with
  a standard deviation of 0.010 over the last 7 days. The words "zen" or "stable" appear in 57–97%
  of each kind's summaries.
- **Transport measures the wrong population.** `RpcHealthSnapshotV1` has **one pooled** p50/p95/max
  per service per 30-second window, plus call counts per channel. There is no latency per channel.
  - Windows average about 2 successful calls, so "p95" is really "the slowest call".
  - Last 7 days of rpc_health transport triggers, by channel mix:

    | service | channels in window | rows | avg successes | timeouts |
    |---|---|---|---|---|
    | cortex-orch | background exec + state | 6,349 | 2.1 | 0 |
    | cortex-exec | LLMGateway + state | 3,798 | 2.2 | 1.01 avg |
    | cortex-exec | LLMGateway + Recall + state | 3,671 | 7.4 | 0.75 avg |
    | cortex-orch | state only | 296 | 0 | 1.97 |

  - The real signal is the timeouts: state-service RPC timeouts (the same failure behind
    biometrics `NO_SIGNAL`) and LLMGateway timeouts. The latency rows bury them.
- **The self-loop.** Each metacog draft is a background LLM call measured by cortex-orch's
  rpc_health. It trips the 5-second p95 limit, which fires transport, which dispatches another draft.
- **Telemetry anomaly is roughly honest** at the trigger: recon_loss, threshold, direction and top
  channels are all present. Over the last 7 days, 82% of rows were 1–2× threshold, mostly driven by
  `failure_pressure`. None of those numbers reach severity, density, or `what_changed`.

### Comparison: what makes reverie and curiosity stream-like

| loop | outer loop | continuity |
|---|---|---|
| reverie text | has the right outer loop (`chain.py`): tick, theme refractory, salience EMA, hollow-drop | none between steps: each step re-reads the coalition, never its own last thought |
| visual chain | — | the only true self-feed: each re-captioned image becomes the next `prior_description`, with a forced reset after N runs |
| curiosity | — | carries a `TurnOutcome` continuation note into the next run |

Metacog has none of these, and nothing to carry yet, because its rows don't hold the event.

## Proposed design

### A. Transport: per-channel EWMA baselines that cannot normalize "busy"

**A0. Per-channel latency in the snapshot (contract change, consumer-first).**
- Add an optional `channel_latency: Dict[str, RpcChannelLatencyV1]` to `RpcHealthSnapshotV1`.
  `RpcChannelLatencyV1` fields:
  - `success_count`
  - `timeout_count`
  - `log_ms_sum`, `log_ms_sumsq`: sufficient statistics for folding the log-latency mean and
    variance exactly, without trusting a 2-sample percentile
  - `max_ms`
- The accumulator in `orion/core/bus/rpc_health.py` already sees every call's channel and elapsed
  time, so this is extending an existing producer, not a new one.
- The model is `extra="forbid"`, so **every consumer ships first**: equilibrium, the signal-gateway
  `rpc_health` adapter, and anything else validating it. Then the producer (memory:
  `feedback_additive_schema_fields_are_a_consumer_first_migration_on_forbid_models`).

**A1. Per-(service, channel) baseline in log space, with two horizons.**
Latency is right-skewed and roughly log-normal, so fold `ln(ms)`. Per key, keep:

- **`fast`**: an EWMA mean and variance of log-latency, updated per window from the window's
  sufficient statistics using `orion/bus/ewma.py::compute_ewma_update`. It gives a z-score for
  spikes.
- **`floor`**: an *asymmetric* EWMA of the window's log-mean. It follows improvements quickly
  (`alpha_down` ≈ 0.2) and follows degradations very slowly (`alpha_up` ≈ 0.002 per window,
  a half-life of about 3 days at 30-second windows). This is the "best recent normal".

**A2. Anti-normalization guards: "busy never silently becomes normal".**

1. **No learning from flagged windows.** A window that fires (spike or timeout) does not update
   `fast`. Borderline windows update it with the value clipped at `mean + 2σ`. This is the
   Phase I / Phase II separation from statistical process control: an out-of-control sample never
   recalibrates the control limits.
2. **Saturation is measured against the floor, not the fast mean.**
   `saturation_ratio = exp(fast.mean − floor)`. If `fast` drifts upward during a long busy stretch
   (guard 1 only protects against *flagged* samples, and slow creep never flags), the floor has
   barely moved. So the ratio rises and a `saturation` condition opens.
3. **Drift is reported, never absorbed.** If `saturation_ratio ≥ R` (about 2.0) lasts longer than
   `T_regime` (about 6 hours), write a `regime_shift` row once: "normal for cortex-exec→LLMGateway
   moved from 9 s to 17 s over 3 days". Only after that row is written may `floor` be re-seeded to
   the new level. The mesh can change (a new model, a new lane), but the change is always stated,
   never silent.
4. **Load is carried as context.** Per key, also track an EWMA of calls per window.
   - The row says "slow while busy (40/min vs usual 6/min)" or "slow while idle". Slow while idle
     is the more worrying one.
   - Evidence only, not a gate. Load and latency are not independent (queueing), so throughput is
     never a second firing signal.
5. **Warm-up and persistence.**
   - A key cannot fire until `fast.count ≥ N_warm` windows with at least `min_calls` calls each.
   - State is persisted to Redis the same way `repair_pressure_trend` already persists its state
     (`service.py:208`).
   - The alpha, N and threshold config is stored alongside the state. A resume under different
     config is refused and cold-starts, per `trend_reducer`'s own disclosed caveat.

**A3. The four transport sub-fixes.**

1. **Expected speed per channel.** Covered by A1 and A2. Nothing is hand-rolled; each channel
   learns its own normal.
2. **Minimum sample count.**
   - Latency can fire only when the window has at least `min_calls` calls on that key (proposed 5).
   - Or across aggregated consecutive windows: spike needs `z ≥ 3` sustained over 2 windows.
   - Timeouts need no minimum.
3. **Metacog's own traffic can't fire transport.**
   - The metacog dispatch call site passes a `health_label` (the verb).
   - The accumulator keys background-lane calls by `(channel, verb)`.
   - Keys named in `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS` (default: `log_orion_metacognition`)
     are measured and baselined but never trigger. That keeps them visible without the loop.
4. **Timeouts and zero-success lead, latency follows.**
   - Primary conditions:
     - any `timeout_count > 0`
     - zero successes on a key that had traffic in its EWMA
   - Latency only opens `spike` or `saturation`.

**A4. Episodes, not windows.** Each transport condition — per key: `timeout`, `spike`,
`saturation`, `regime_shift` — is a small state machine: **open, escalate, close**. That gives one
trigger on open, one on a severity escalation, and one on close with duration and peak. The
per-30-second firing is gone, and so is the per-lane cooldown's job of papering over it.

### B. Deterministic mapping from each trigger's upstream into the row

A per-kind mapper, `orion/metacog/evidence_map.py`, turns `trigger.upstream` into the fields below.
Each kind gets its own function, all pure and unit-tested.

- **`severity`**: from the event's own magnitude, never from the pipeline or the LLM.

  | kind | severity rule (proposed, thresholds provisional) |
  |---|---|
  | transport | `timeout` count and `z`; `saturation_ratio`; `regime_shift` is degraded unless the new normal is a timeout |
  | telemetry_anomaly | `recon_loss / threshold`: <1.5 nominal, <2.5 degraded, ≥2.5 critical |
  | chat_turn | a failed compliance, timeout, or exit≠0 is critical; misalignment or strain alone is degraded |
  | relational, repair_trend | the gate's own z / level, against the gate's own thresholds |

- **`causal_density.score`**: the same normalized magnitude, from 0 to 1. This is the fix
  the 07-18 redesign already named, now built.
- **`what_changed.evidence`**: the upstream fields themselves, e.g. `p95 17.4s vs normal 9.1s
  (z=3.4)`, `timeouts 2/3 calls on orion:state:request`. The 20 pipeline step-log lines move out of
  evidence into `provenance`.
- **`touches`**: the services and channels named in upstream.
- **`state`**: the trigger's evidence block, with the global biometrics reduced to a short
  reference.
- **LLM**: writes the one-line summary only.
  - The `zen_state`/`pressure` inputs are removed from the draft prompt, because they are flat
    and they cause the "zen persists" narration.
  - The uncertainty-probe LLM call is removed, since severity no longer uses it. That saves one of
    two LLM calls per row.
  - If the draft fails, the row still publishes with a deterministic summary built from evidence.

### C. Replay proof

`scripts/analysis/replay_metacog_capture.py` replays the last 7 days of `metacog_trigger` rows
through B, and the stored rpc_health snapshots through A where they exist (see acceptance checks).
It writes `/tmp/metacog-capture-replay/report.md`.

## Mesh transport coverage: yes, there is a gap

Cortex used to be the unit all work passed through, so measuring cortex's RPC calls was close to
measuring the mesh. That stopped being true. Work now moves through the FCC motor (HTTP from a
subprocess), durable-runs, hub-hosted turns, the harness-governor, the substrate dispatch runtime,
and orion-mind. **Transport health still only sees cortex.**

**How measurement works today.** `OrionBusAsync.rpc_request()` (`orion/core/bus/async_service.py:459`)
records the success or timeout of each call into an aggregator that lives on that bus object. Only
cortex-orch (`main.py:721`) and cortex-exec (`main.py:1127`) run `rpc_health_publish_loop`, behind
`RPC_HEALTH_PUBLISH_ENABLED`, which defaults to false. The signal-gateway adapter additionally
whitelists just those two (`orion/signals/adapters/rpc_health.py:51-52`). Live 2026-09-24: 25
snapshots in 150 s, all from those two services, all `instance=None`.

| execution path | how it calls | seen by transport health? | its own telemetry |
|---|---|---|---|
| cortex-orch, non-chat lanes | `rpc_request` | yes | route grammar |
| cortex-orch chat lane | hand-rolled `orion:verb:request` publish+subscribe (`orchestrator.py:675,747`) | **no** (the bulk of chat traffic) | — |
| cortex-exec (4 lane containers) | `rpc_request` | **blurred**: all four report `service=cortex-exec, instance=None`, so a `:background` backlog is indistinguishable from `:chat` | traces |
| hub → harness-governor (Orion-mode and curiosity turns) | hand-rolled `_pending_rpc` future (`harness_governor_client.py:186`) | **no**, and no timeout atom either. This is the longest RPC in the system | cockpit frames |
| FCC motor | `claude` CLI subprocess → HTTP → orion-fcc proxy → llm-gateway | **no** (no bus at all) | `HarnessRunV1.fcc_elapsed_sec`, `harness_turn_trace`, which are per-run and not health windows |
| durable-runs | `rpc_request` + HTTP to gateway, cabinet, elastic | **no** (counted, never published) | durable admission, lease and permit tables, `orion:durable:run:state` |
| substrate dispatch runtime | new bus **per tick** → exec `:background` | **no**; the aggregator is thrown away each tick | dispatch frames and results |
| orion-mind | new bus **per LLM call** → gateway | **no**; the aggregator is thrown away each call | — |
| orion-thought (reverie, visual, stance) | `rpc_request` + HTTP to mind | **no** | reverie health monitors |
| hub, actions, embodiment, others | `rpc_request` | **no** | varies |

**Nesting worth knowing.**
- exec → orch → exec happens in `bound_capability_exec.py:197` and `self_study.py:1516/1661`. When
  the inner hop is the chat lane, the round trip is unmeasured.
- The durable chain is orch → durable-runs → hub (in-process turn) → governor → FCC subprocess →
  exec `:background`. **Not one hop of it reaches transport health.**

**The only mesh-wide fallback** is the `rpc_transport_timeout` grammar atom, fired from every
`rpc_request` timeout in every service. It sees timeouts only: no latency, no hand-rolled RPC, no
HTTP.

**What the existing durable/harness telemetry is and isn't.** It is outcome bookkeeping per run.
It answers "did this run finish". It does not answer "is this hop slower than its normal right now".
It complements transport health; it cannot replace it.

**Proposed direction: extend the existing aggregator, don't build a new one.**
1. **"Hop", not "channel", is the key.** `channel_latency` (A0) is keyed by a hop label:
   - a bus channel for `rpc_request`
   - `verb:<name>` for the orch chat lane
   - `governor:<mode>` for hub → governor
   - `http:<host><path>` for HTTP
   - `fcc:<served_model>` for the motor subprocess wall time

   The A1–A4 EWMA machinery is identical for every key.
2. **Record into the aggregator from the hand-rolled paths.** Hub → governor and the orch chat-lane
   verb path each call the aggregator's `record_success`/`record_timeout`. They already hold the bus
   object.
3. **One thin HTTP timing helper** (an `httpx` event hook) records into the same aggregator. Users:
   thought → mind, durable-runs → gateway, orion-fcc proxy → gateway. The governor records FCC
   subprocess wall time, which it already computes.
4. **Long-lived aggregators.** Dispatch runtime and mind keep one process-level aggregator instead
   of discarding it per tick or per call.
5. **Identity.**
   - Every publisher sets `instance`; exec uses its lane.
   - `RPC_HEALTH_PUBLISH_ENABLED` defaults on in every service's `.env_example`.
   - The signal-gateway whitelist becomes a pass-through keyed by `(service, instance)`.

Because the EWMA baselines learn each key's own normal, adding a slow hop (the FCC motor, the
governor) brings no hand-tuned thresholds with it. That is the whole point of A1.

## Metric quality gate: per-channel log-latency EWMA (A1/A2)

1. **Provenance.**
   - `orion/core/bus/rpc_health.py` records the elapsed time of every `OrionBusAsync.rpc_request()`
     call.
   - `orion/core/bus/rpc_health_publish.py` drains it every 30 s onto `orion:rpc_health:snapshot`.
   - Live 2026-09-24: only cortex-exec and cortex-orch published in a 150-second sample.
2. **Independence.**
   - Timeouts are the censored tail of the same latency sensor, so they are *one* condition with
     two fields, not two signals.
   - Throughput is causally upstream of latency (queueing), so it is evidence only, never a trigger.
   - `bus_synaptic` prediction error comes from bus-mirror inter-arrival gaps, a different sensor.
     It stays a separate transport source.
3. **Theory anchor.**
   - EWMA control chart (Roberts 1959; Lucas & Saccucci 1990) on log-transformed service times.
     Service latencies are right-skewed and multiplicative, which is why the log.
   - Phase I/II separation (never recalibrate limits on out-of-control samples) is the named guard
     against baseline contamination.
   - The asymmetric floor is a slow-drift detector against a best-known reference. Its job is to
     make slow creep, which a CUSUM would catch and a plain EWMA absorbs, visible.
4. **Live-data sanity.**
   - **UNVERIFIED for the per-channel series itself**, because it doesn't exist yet: the snapshot
     is pooled.
   - What is verified: the pooled series is degenerate for this purpose. About 2 calls per window,
     and a mix of ~300 ms state calls with 10–20 s LLM calls, so p95 is "whichever LLM call
     happened".
   - The rest state is representable: `z ≈ 0` when calm, and `saturation_ratio ≈ 1`.
   - This is *not* a `mean(|z|)` aggregate, so there is no `sqrt(2/π)` floor. Firing uses signed z
     per key.
   - The log-only phase (acceptance check 1) must confirm per key that `z` returns to about 0 and
     that the ratio returns to about 1 during quiet hours, before anything fires.
5. **Existing mechanism.**
   - `orion/bus/ewma.py::compute_ewma_update`: reused, with a per-domain `min_variance`. Log-ms
     variance is on the order of 0.1–1, so the default 1e-6 floor is irrelevant, but the floor is
     passed explicitly anyway.
   - Redis state persistence pattern from `repair_pressure_trend_gate` / `service.py:208`.
   - No existing per-channel latency baseline was found.
6. **Reversibility.**
   - The schema field is optional and additive.
   - The gate is behind a new flag, off by default, with log-only mode first.
   - State lives in one Redis key per service.
   - Removing it means deleting the flag and the key. Nothing gets baked into training defaults.

## Missing questions

1. `min_calls`, `N_warm`, `R` and `T_regime` are proposals. They are set from the log-only week,
   not before.
2. Should `regime_shift` rows also notify Juniper, as orion-notify, or stay metacog-only?
3. Is `orion_metacog` kept as the per-trigger table at all once B lands? Alternative: rows are
   written only on episode open and close, and the draft LLM runs only for severity ≥ degraded.

## Proposed schema / API changes

- `RpcHealthSnapshotV1.channel_latency: Optional[Dict[str, RpcChannelLatencyV1]]`, additive.
  Consumers ship first.
- `MetacogTriggerV1.upstream` for transport gains these fields (upstream is a free-form dict, so
  this is not a schema change):
  - `condition`: `timeout | spike | saturation | regime_shift | zero_success`
  - `phase`: `open | escalate | close`
  - `key`
  - `z`
  - `saturation_ratio`
  - `baseline_ms`
  - `floor_ms`
  - `calls_per_min`, `calls_per_min_usual`
  - `duration_s`, `peak_ms`
- `orion_metacog` columns are unchanged. Severity, `causal_density` and `what_changed` change
  *meaning* to event-derived, recorded in the PR as a metric-definition change.

## Files likely to touch

- `orion/core/bus/rpc_health.py`, `rpc_health_publish.py`, `orion/schemas/telemetry/rpc_health.py`,
  `orion/schemas/registry.py`
- `orion/signals/adapters/rpc_health.py` (consumer-first)
- `services/orion-equilibrium-service/app/transport_metacog_gate.py`, `service.py`, `settings.py`,
  `.env_example`
- new `orion/metacog/transport_baseline.py`, a pure reducer holding the A1/A2 state
- new `orion/metacog/evidence_map.py`
- `orion/metacog/service.py`, whose `compute_severity` and `causal_density` sources are replaced
- `services/orion-cortex-exec/app/executor.py`: MetacogContext/Draft/Publish steps, probe removed
- `orion/cognition/prompts/log_orion_metacognition_draft.j2`. cortex-orch must be rebuilt too; it
  reads this template itself (memory: PR #2065 deploy gotcha).
- `services/orion-cortex-orch/app/orchestrator.py`, to pass the `health_label` on metacog dispatch
- tests alongside each, plus `services/orion-equilibrium-service/evals/`

## Non-goals

- The metacog stream of consciousness. Deferred to the hop-chain spec, once rows are honest.
- A new metacog taxonomy or new trigger kinds.
- Rewriting the 114k historical rows.
- Retuning telemetry_anomaly's trigger. Its capture is fine; only its row mapping changes.
- Measuring serving-side queueing (time a request waits inside exec, gateway or hub before work
  starts). The caller-side measurement above includes it but cannot separate it out.
- Replacing durable/harness per-run telemetry. It stays; transport health complements it.

## Acceptance checks

1. **Log-only week.** The A gate runs with `emit=false`, logging per-key `z`, ratio, and calls per
   window.
   - Pass when every key with traffic has `z` median within ±0.5 and `saturation_ratio` median
     within [0.8, 1.3] during 01:00–06:00 MDT quiet hours.
   - Pass when no key's floor moved more than 1.5× without a logged `regime_shift` candidate.
2. **Self-loop gone.** With emit on, 24 hours later: zero transport triggers whose only slow key
   is `log_orion_metacognition`.
3. **Volume.** Transport rows per day drop from about 2,300 to under 100, and every row is a
   condition open, escalate, or close.
4. **Severity monotonic.** On the replay:
   - Transport and telemetry severity rank-correlate with magnitude (Spearman ρ ≥ 0.6).
   - No "nominal" row has a larger magnitude than the median "critical" row of the same kind.
5. **Density not degenerate.** `causal_density.score` takes more than 10 distinct values across a
   day, and is 0 only when the magnitude is 0.
6. **No zen narration.** Under 5% of new summaries contain "zen", and none contain it when
   severity is critical.
7. **Anti-normalization.** A unit test replays a synthetic 12-hour busy plateau at 2.5× latency,
   with no spikes. It must open `saturation`, then write exactly one `regime_shift` after
   `T_regime`, and must never report nominal during the plateau.

## Recommended next patch

1. **B alone first.** Map each trigger's numbers into its row, drop the zen inputs and the probe
   call. It needs no new data, and it immediately makes telemetry_anomaly and chat_turn rows
   honest. It also cuts the per-row LLM calls from 2 to 1.
2. **A0 plus coverage, consumer-first.**
   - The hop-keyed `channel_latency` field.
   - Every consumer ships before any producer.
   - Then publishers: set `instance`, enable publishing everywhere, record from the hand-rolled
     paths, add the HTTP hook, make the aggregators long-lived.
3. **A1–A4 in log-only mode**, then the log-only week.
4. **Turn on emit**, set thresholds from the data, run the replay.
