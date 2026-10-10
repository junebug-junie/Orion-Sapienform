# orion-equillibrium-service

System health + (currently) baseline Collapse Mirror metacognition ticks.

> Spelling matches repo (`equillibrium`).

---

## The Metacognition (double duty--refactor me into a new service!)
Equilibrium is doing **two** intentional jobs in your current implementation:

1) **Health aggregator**: “Which services are healthy/missing?”
2) **Metacognition tick emitter**: a periodic **Collapse Mirror baseline snapshot** with:
   - `trigger = "equilibrium.metacognition_tick"`
   - `snapshot_kind = "baseline"`
   - summary like “Periodic metacognition snapshot emitted by equilibrium monitor.”
   - published to `orion:event:equilibrium:snapshot`
   - envelope kind: `equilibrium.collapse.snapshot`

This is controlled by:
- `EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC` (default ~15s)

### The metacog trigger family — big picture

This section exists because design decisions for this system have historically only lived in one-off docs under `docs/superpowers/specs/`/`docs/superpowers/design/` that nobody goes back and reads once the branch merges. This README is meant to be the thing that stays current — the design docs below are the deep-dive/forensic record of *how* each decision got made, not the place to look first.

**The pipeline, same for every trigger kind:** a gate module in this service (`app/<kind>_metacog_gate.py`) evaluates real evidence and, if a real condition holds, builds a `MetacogTriggerV1` (`orion/schemas/telemetry/metacog_trigger.py`). `_publish_metacog_trigger()` (`app/service.py`) checks that kind's cooldown lane, then publishes to `CHANNEL_EQUILIBRIUM_METACOG_TRIGGER` (`orion:equilibrium:metacog:trigger`). `orion-cortex-orch`'s `dispatch_metacog_trigger()` picks it up and runs a fresh, independent `log_orion_metacognition` plan (`MetacogContextService` → `MetacogDraftService` → `MetacogPublishService`, all in `orion-cortex-exec`), which writes a real `MetacogEntryV1` row into the `orion_metacog` Postgres table via `orion-sql-writer`. Every trigger kind below reuses this exact mechanism unmodified — a new kind only ever adds a gate module + a dispatch branch here, never touches the draft/publish machinery. (A separate `MetacogEnrichService` step used to sit between draft and publish; removed 2026-07-28 after an audit found zero of its 7 output fields ever survived into the published `MetacogEntryV1`.)

**The design rule every trigger kind here follows, learned the hard way across several rounds:**
- **Ground every gate condition in a real, already-computed field.** Never invent a derived signal or guess a boolean from vibes — if the field doesn't exist yet, the condition doesn't ship yet (see `chat_turn`'s own history below, where two originally-proposed conditions got dropped for exactly this reason).
- **One gate module per trigger kind** (`app/chat_turn_metacog_gate.py`, `app/telemetry_anomaly_metacog_gate.py`, `app/transport_metacog_gate.py`, etc.) — never one shared mega-function.
- **Give a new kind its own cooldown lane if it fires on a fundamentally different cadence than the shared periodic/rare pattern.** `chat_turn` (fires on essentially every remarkable turn) originally shared the global `EQUILIBRIUM_METACOG_COOLDOWN_SEC` lane with `baseline`/`manual`/`pulse`/`relational`/`telemetry_anomaly` — a real bug, since a burst of `chat_turn` fires could silently starve the others. Fixed 2026-07-23 (`_publish_metacog_trigger`'s `_PER_KIND_COOLDOWN_SETTINGS_ATTR` dict, generalized from a hardcoded if/else the same day `transport` needed its own lane too). Any future kind that fires often should get its own `EQUILIBRIUM_METACOG_<KIND>_COOLDOWN_SEC` from day one, not retrofit it after shipping.
- **Ships disabled by default.** Every kind flips on only after a real, non-degenerate `orion_metacog` row has been observed post-deploy — "the flag is on" is not the same claim as "it's verified," and this repo's own history (`telemetry_anomaly`'s 2026-07-21 arsonist audit, in the design doc below) is a documented case of that distinction being missed for a while.
- **`orion_metacog` currently has no confirmed real consumer.** This is a standing open question, not resolved by adding more trigger kinds — see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s "Missing questions" for the full framing (`orion_metacog` vs. the separately-fed `orion_metacognitive_trace` table). Shipping a new trigger kind is real, verifiable progress on *evidence quality* regardless — but it is not, by itself, progress toward that open question, and shouldn't be reported as if it were.

**Current trigger kinds:**

| `trigger_kind` | Evidence source | Cooldown lane | Status |
|---|---|---|---|
| `baseline` | Scheduled tick | — | **retired 2026-09-29** (see below) |
| `manual` | User-triggered Collapse Mirror event | shared | live |
| `dense` / `pulse` | Substrate self-state eventfulness score | — | **retired 2026-09-29** (never fired; see below) |
| `relational` | Real `repair_pressure_v2` appraisal | shared | live |
| `telemetry_anomaly` | Trained autoencoder reconstruction-loss anomaly | shared | live (2026-07-21) |
| `chat_turn` | Correlated `ThoughtEventV1` + `HarnessRunV1` (or a governor/stance-react timeout) | own (`EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC`) | live (2026-07-23) |
| `transport` | per-hop baseline gate episodes (timeout / zero_success / spike / saturation / regime_shift) + real per-call RPC timeout grammar events for timeouts no gate window claims (pooled timeouts retired 2026-09-29; bus_synaptic retired 2026-09-30) | own (`EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC`) | live; gate emitting since 2026-10-01 |
| `insight` | Sustained low→high recovery in `AttentionSelfModelV1.prediction_error_confidence` (`substrate_attention_self_model`) | own (`EQUILIBRIUM_METACOG_INSIGHT_COOLDOWN_SEC`) | **ships disabled**, not yet live-verified |
| `flow` | Sustained high-confidence plateau in the *same* field | — | **retired 2026-10-10** (its plateau was the idle rest state; see below) |

**A separate, parallel system this table's `transport` row borrows from, not the same pipeline:** `rpc_health` (`orion/core/bus/rpc_health.py` → `orion/core/bus/rpc_health_publish.py` → `orion:rpc_health:snapshot` → `orion-signal-gateway`'s `RpcHealthAdapter` → `OrionSignalV1`) is the `orion-signal-gateway` **organ-signal** pipeline (see that service's own README), completely independent of `orion_metacog`/`MetacogTriggerV1`. `transport`'s per-hop baseline gate subscribes to the same `orion:rpc_health:snapshot` channel as a *second* consumer, reading the same real data into a different destination table. Don't confuse the two pipelines when reading logs — a `rpc_health` organ signal in `orion-signal-gateway` and a `transport` metacog trigger in `orion_metacog` can both exist (or not) independently of each other.

**Deep-dive / forensic history**, if you need the "how did we get here" story behind any of the above (each is long — read the README first, these are for when you need the receipts):
- `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` — the original redesign (why `collapse_mirror`/`numeric_sisters` got replaced, the `relational`/`telemetry_anomaly`/`chat_turn` build history, the open `orion_metacog` consumer question).
- `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md` + `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md` — why the old `transport_pressure`/`bus_health` family was found narrowly-scoped/misleading, and the real `rpc_health` signal built to replace it.
- `docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md` — the `transport` trigger kind's own design (why it doesn't build on the old `bus.transport` grammar lane, Options A/B/C).

### Retired: baseline heartbeat and substrate dense/pulse triggers (2026-09-29)

`_metacog_baseline_loop()` used to publish `trigger_kind=baseline` (`reason="scheduled_check"`, empty `upstream`) on a timer, first trying the substrate `dense`/`pulse` gate. Both are gone -- loop, gate module, settings and compose keys:

- **baseline** carried no evidence, so once PR #2393 let its rows publish again they were the same sentence every hour ("Baseline check triggered with no active alerts, indicating stable system state." / "Stability is the foundation of progress.") -- an LLM call per hour to say nothing happened. `orion_metacog` now only holds rows where something happened.
- **dense/pulse** were structurally unreachable since the 2026-07-22 SelfStateV1 removal: `compute_substrate_eventfulness()` maxes at `0.25`, below both thresholds (`0.30`/`0.55`).

A real "nothing is happening" signal, if wanted later, should be derived from the absence of the event-driven triggers, not from a timer. `orion/metacog/evidence_map.py` keeps its baseline/dense/pulse mappers only so historical rows still replay.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_ENABLE` | `true` | Master gate for every trigger type below |
| `EQUILIBRIUM_METACOG_COOLDOWN_SEC` | `30` | Global cooldown in `_publish_metacog_trigger()` for kinds without their own lane; a trigger firing during cooldown is dropped (logged, not queued) |

### Manual metacog trigger

Fires `trigger_kind=manual` (`reason="user_collapse_event"`) whenever a real user (not Orion itself) manually triggers a Collapse Mirror snapshot from the Hub UI, published on `CHANNEL_COLLAPSE_MIRROR_USER_EVENT` (`orion:collapse:intake`). Guarded against feedback loops: a payload with `observer=orion` is skipped outright (`elif channel == settings.channel_collapse_mirror_user_event` branch in `app/service.py`), so Orion's own collapse-mirror activity can never re-trigger itself through this path. No dedicated enable flag -- gated only by `EQUILIBRIUM_METACOG_ENABLE` above, same as every trigger type in this section.

| Env | Default | Purpose |
|-----|---------|---------|
| `CHANNEL_COLLAPSE_MIRROR_USER_EVENT` | `orion:collapse:intake` | Source channel (single consumer: this service) |

### Relational metacog trigger

When `EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE=true`, equilibrium subscribes to `orion:repair_pressure:appraisal`, published by `orion-hub`'s `services/orion-hub/scripts/pre_turn_appraisal_wiring.py` whenever the `repair_pressure` paradigm actually runs for a turn (real repair_pressure_v2 evidence: rupture/repair detectors over the live turn window, not a self-report). `level >= EQUILIBRIUM_METACOG_RELATIONAL_LEVEL_THRESHOLD` and `confidence >= EQUILIBRIUM_METACOG_RELATIONAL_CONFIDENCE_THRESHOLD` fires `trigger_kind=relational`, carrying the full evidence breakdown (`evidence_kind`/`score`/`confidence` per detector) and `behavior_applied` in the trigger's `upstream` field.

As of 2026-07-18 this replaced the previous source, `orion/memory/turn_change_classify.py`'s SHIFT appraisal (NONE/TOPIC/STANCE/REPAIR) consumed off `orion:chat:history:spark_meta:patch` — see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` for the swap rationale. `trigger_kind=relational` is kept: same conceptual trigger category, different evidence source.

**Confidence threshold lowered 0.7 → 0.65 on 2026-08-11.** This trigger had never fired even once, and not because it's rare: live rows pulled from `repair_pressure_appraisal_log` (from before an 11-day mesh outage, so genuinely representative of real traffic) show `confidence` pinned at exactly `0.65` on essentially every real appraisal — the "confidently calm" text-fallback constant `reduce_repair_level()` documents, not a value this gate could ever clear at the old `0.7` floor. `0.65` makes the fallback case itself reachable; still excludes genuine zero-evidence reads below it.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_RELATIONAL_TRIGGER_ENABLE` | `true` | Master gate for the relational trigger |
| `EQUILIBRIUM_METACOG_RELATIONAL_CONFIDENCE_THRESHOLD` | `0.65` | Minimum appraisal confidence to fire |
| `EQUILIBRIUM_METACOG_RELATIONAL_LEVEL_THRESHOLD` | `0.5` | Minimum repair_pressure level to fire |
| `CHANNEL_REPAIR_PRESSURE_APPRAISAL` | `orion:repair_pressure:appraisal` | Source channel (single consumer: this service) |

### repair_pressure_trend metacog trigger

Shipped 2026-07-30 (hop 0 of the stream-of-consciousness hop-chain design) but undocumented here until now. Folds every real, confidence-gated `repair_pressure_v2` appraisal into a persisted EWMA baseline (`orion/metacog/trend_reducer.py`) and fires `trigger_kind=repair_pressure_trend` only on a *sustained* multi-appraisal elevated run — distinct from the single-appraisal `relational` trigger above, which reacts to one reading. State checkpoints to Redis under `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_STATE_KEY`.

**Flipped on 2026-08-11**, alongside `insight`/`flow` below, after the same 11-day mesh outage left no live window to judge it against since it shipped. Watch for the first real fire before trusting it.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_TRIGGER_ENABLE` | `true` | Master gate |
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_CONFIDENCE_FLOOR` | `0.3` | Minimum appraisal confidence to fold into the EWMA at all (separate from whether the resulting trend fires) |
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_MIN_SAMPLES` | `20` | Cold-start floor before a z-score is trusted |
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_ELEVATED_ZSCORE` | `1.0` | Z-score above which a reading counts as elevated |
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_SUSTAINED_HITS` | `3` | Consecutive elevated readings required to fire |
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_STATE_KEY` | `equilibrium:metacog_trend_state:repair_pressure` | Redis key for the checkpointed EWMA state |
| `EQUILIBRIUM_METACOG_REPAIR_PRESSURE_TREND_COOLDOWN_SEC` | `1800` | Own cooldown lane |

### Telemetry-anomaly metacog trigger

When `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE=true`, equilibrium subscribes to `orion:field_channel:anomaly_score`, published by `orion-field-digester`'s periodic anomaly-scoring loop (`app/anomaly_scorer.py`) whenever `FIELD_CHANNEL_ANOMALY_ENABLED=true` there. The score is reconstruction loss from a trained `orion/mood_arc/fit_encoder.py` autoencoder against the most recent rolling window of live `field_channel_corpus.v1` pressures.

Same design as the relational trigger above: the producer publishes the raw measurement (`recon_loss`, plus its own train-time `recon_error_p95` reference), this service applies its OWN `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER` rather than trusting the producer's embedded `anomalous` flag -- so trigger sensitivity is tunable here without redeploying `orion-field-digester`. Fires `trigger_kind=telemetry_anomaly` when `recon_loss > recon_error_p95 * threshold_multiplier`, carrying `recon_loss`/`recon_error_p95`/`threshold`/`window_start`/`window_end`/`encoder_id`/`encoder_version` in the trigger's `upstream` field.

Added 2026-07-21 -- see `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s trigger taxonomy.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_TRIGGER_ENABLE` | `true` | Master gate for the telemetry-anomaly trigger |
| `EQUILIBRIUM_METACOG_TELEMETRY_ANOMALY_THRESHOLD_MULTIPLIER` | `3.0` | Multiplier applied to the encoder's own `recon_error_p95` |
| `CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE` | `orion:field_channel:anomaly_score` | Source channel (single consumer: this service) |

### chat_turn metacog trigger

When `EQUILIBRIUM_METACOG_CHAT_TURN_TRIGGER_ENABLE=true`, equilibrium subscribes to `orion:thought:artifact` (`ThoughtEventV1`, published by `orion-thought` for every chat turn), `orion:harness:run:artifact` (`HarnessRunV1`, published by `orion-harness-governor` on every real `handle_harness_run_request` exit path), and `orion:grammar:event` filtered to two timeout atoms. A Redis-backed correlator (`app/chat_turn_metacog_gate.py::ChatTurnCorrelator`, key `orion:equilibrium:chat_turn_corr:<correlation_id>`, TTL `EQUILIBRIUM_METACOG_CHAT_TURN_CORRELATOR_TTL_SEC`) accumulates evidence per turn and fires once it's terminal.

Unlike the other triggers, this one reuses two already-registered schemas instead of a purpose-built payload -- no new bus contract, since the accumulator is ephemeral internal state, not a durable artifact. Fires `trigger_kind=chat_turn` when any of these real conditions hold (see the gate-condition table in `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md`'s chat_turn spec section): `thought_event.disposition != "proceed"`, `thought_event.boundary_register is True`, `run_artifact.reflection.alignment_verdict != "aligned"`, `run_artifact.reflection.strain_unresolved is True`, `run_artifact.substrate_appraisal.surprise_level >= EQUILIBRIUM_METACOG_CHAT_TURN_SURPRISE_THRESHOLD`, `run_artifact.compliance_verdict != "completed"`, `run_artifact.exit_code not in (0, None)`, `run_artifact.finalize_degraded_reason is not None`, or a timeout (see below).

**Terminal evidence, four cases** (no more evidence will ever arrive for that `correlation_id`, so the correlator evaluates and clears immediately instead of waiting out the TTL):
- `run_artifact` arrived -- the turn ran to completion.
- `exec_turn_timeout` -- the harness-governor RPC never returned (`orion/hub/turn_orchestrator.py`'s `if run is None:` branch, Patch B / PR #1287).
- `stance_react_timeout` -- the *earlier* `ThoughtClient.react()` RPC itself never returned to Hub (`orion/hub/turn_orchestrator.py`'s `if thought is None:` branch); Hub never calls the harness governor on this path either.
- `thought_event.disposition in ("defer", "refuse")` -- Hub short-circuits before calling the harness governor.

Added 2026-07-22 (PR #1291), shipped disabled by default; **enabled 2026-07-23**. Live-data verification (real `orion_metacog` rows, non-degenerate `upstream`) is the acceptance check named in `docs/superpowers/design/2026-07-18-collapse-mirror-metacog-redesign.md` and still needs to happen post-deploy -- watch for it, don't assume it's done just because the flag is on.

**Operational note (resolved 2026-07-23)**: unlike the other trigger kinds here (rare/periodic), `chat_turn` is designed to fire on essentially every remarkable chat turn. Sharing `EQUILIBRIUM_METACOG_COOLDOWN_SEC`'s single global cooldown timestamp with baseline/manual/pulse/relational/telemetry_anomaly would have let a burst of `chat_turn` fires silently starve those other trigger kinds too, not just drop `chat_turn`'s own excess -- so `chat_turn` now has its own separate cooldown lane (`EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC`, own timestamp, own setting). `_publish_metacog_trigger` still silently drops (logs only, does not queue) anything that fires within *its own kind's* cooldown window -- that part of the behavior is unchanged and intentional.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_CHAT_TURN_TRIGGER_ENABLE` | `true` | Master gate for the chat_turn trigger |
| `EQUILIBRIUM_METACOG_CHAT_TURN_CORRELATOR_TTL_SEC` | `600` | Correlator entry TTL (leak backstop for non-terminal evidence) |
| `EQUILIBRIUM_METACOG_CHAT_TURN_SURPRISE_THRESHOLD` | `0.7` | Minimum `substrate_appraisal.surprise_level` to fire |
| `EQUILIBRIUM_METACOG_CHAT_TURN_COOLDOWN_SEC` | `30` | chat_turn's own cooldown window, separate from `EQUILIBRIUM_METACOG_COOLDOWN_SEC` |
| `CHANNEL_THOUGHT_ARTIFACT` | `orion:thought:artifact` | Source channel (wildcard consumers) |
| `CHANNEL_HARNESS_RUN_ARTIFACT` | `orion:harness:run:artifact` | Source channel (wildcard consumers) |
| `CHANNEL_GRAMMAR_EVENT` | `orion:grammar:event` | Source channel, filtered to `semantic_role in ("exec_turn_timeout", "stance_disposition")` |

### transport metacog trigger

Full design: `docs/superpowers/specs/2026-07-24-transport-metacog-trigger-design.md`. Three independent evidence sources, all feeding `trigger_kind=transport` (`app/transport_metacog_gate.py`), no correlator needed -- each source fires on its own real evidence directly:

- **(A) pooled `RpcHealthSnapshotV1` timeouts: retired 2026-09-29.** It fired on a window's pooled `timeout_count` for cortex-exec/cortex-orch only. Every timeout it counted is an `rpc_request()` timeout, and every one of those also emits the (C) atom below, so it was a second, coarser copy of (C): live over 48 h, 675 of 677 timeouts it counted had a matching atom in the same 30 s window, and the two together fired ~500 transport rows a day for mostly the same LLM-gateway timeouts. Removed outright (no fallback). Its pooled-p95 latency branch had already gone on 2026-09-24 (the metacog self-loop). Per-hop latency and timeouts now live in the baseline gate below.
- **(C) `rpc_transport_timeout` grammar atoms** on `orion:grammar:event` (published by `orion/core/bus/async_service.py::_emit_rpc_timeout_grammar`, fired from both of `rpc_request()`'s real timeout branches -- generalizes `chat_turn`'s own `exec_turn_timeout`/`stance_timeout` markers, scoped to one harness/thought RPC each, to every one of the 37+ real `rpc_request()` call sites sharing that one client). Terminal by construction -- a real RPC already timed out by the time this atom exists, no threshold to evaluate. **The single owner of timeouts while the baseline gate is log-only.** When `EQUILIBRIUM_TRANSPORT_BASELINE_EMIT` is effective, `app/transport_timeout_owner.py` decides ownership per timeout: a gate-folded snapshot window that saw a timeout on the same request channel (hop key before `#`) within its `[window_start, window_end]` owns it (its `timeout`/`zero_success` episode), and the atom is dropped; an atom no window claims within `EQUILIBRIUM_TRANSPORT_TIMEOUT_ATOM_GRACE_SEC` fires as before. The atom does not say which service emitted it, so this is decided by evidence, count for count, never by a list of 'covered' services -- a caller that does not publish rpc_health, a skipped snapshot, or a cold-started gate just leaves the atom unclaimed, and it fires. Log lines: `transport_timeout_owner owner=baseline_gate|atom`. The match window reaches 60 s before a snapshot window's start, because short-lived buses (orion-mind, execution-dispatch, orion-thought) fold their timeouts into whatever window is open when they are absorbed. At most 5,000 markers are held; on overflow the oldest fires early rather than being dropped. A timeout the gate owns is subject to the gate's hourly publish budget: past `EQUILIBRIUM_TRANSPORT_BASELINE_MAX_TRIGGERS_PER_HOUR`, the gate row is dropped (`transport_baseline_suppressed`) and the marker deliberately does not fall back, because that budget is the mesh-wide-outage cap.
- **(bus_synaptic) -- RETIRED 2026-09-30.** A FalkorDB poll of `node:substrate.bus_synaptic`'s `prediction_error` (fraction of bus-synaptic edges at `|z| >= 3`). Its metric quality gate failed on live data: the edge z-scores are stamped at `orion-bus-mirror`'s dequeue time by a single consumer Redis disconnects for output-buffer overflow roughly every 20 minutes, it fired 50-155 episodes/day evenly across the clock without tracking real RPC-timeout storms, and its stated purpose (one bespoke organ) sits below its own noise band by design. Builder, poll loop, settings and env keys were removed, not disabled. Evidence: `docs/superpowers/pr-reports/2026-09-30-retire-bus-synaptic-transport-trigger-pr.md`.

Own cooldown lane from day one (`EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC`) -- not sharing the global lane, avoiding the exact bug `chat_turn` had to fix after the fact (see above). Both remaining evidence sources (A/C) share this one lane.

`EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE` is live (`true`) as of 2026-07-24 (flipped shortly after shipping, commit `40cd21f80` -- correcting a stale claim this paragraph carried before). The bus_synaptic source was live 2026-07-26 to 2026-09-30 and is now retired (see above).

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_TRANSPORT_TRIGGER_ENABLE` | `true` | Master gate for the transport trigger |
| `EQUILIBRIUM_METACOG_TRANSPORT_COOLDOWN_SEC` | `30` | transport's own cooldown window, separate from `EQUILIBRIUM_METACOG_COOLDOWN_SEC` |
| `CHANNEL_RPC_HEALTH_SNAPSHOT` | `orion:rpc_health:snapshot` | Baseline gate's source channel |
| `CHANNEL_GRAMMAR_EVENT` | `orion:grammar:event` | Option C's source channel, filtered to `semantic_role=="rpc_transport_timeout"` |

### transport baseline gate (per-hop EWMA, 2026-09-24)

Spec: `docs/superpowers/specs/2026-09-24-metacog-capture-and-transport-ewma-baseline-design.md` (A1-A4). Reducer: `orion/metacog/transport_baseline.py` (pure). Glue: `app/transport_baseline_gate.py`.

What it does, plainly: every rpc_health window now carries per-hop latency sums (`channel_latency`). For each `(service, instance, hop)` this learns what "normal speed" is, in log space, and reports when a hop is suddenly slow (`spike`), has been slow for a while compared with its best recent normal (`saturation`), has been slow so long that normal has really moved (`regime_shift`, stated once, then the baseline re-seeds), or is timing out (`timeout`, `zero_success`). Each is an episode: one row on open, one per severity doubling, one on close with duration and peak -- not one per 30-second window.

Guards against learning "busy" as normal: windows that look like incidents never teach the baseline; slow drift is measured against a floor that barely moves up and does not move up at all while saturated; the floor only jumps after a `regime_shift` row has said so. Load (calls per minute) is carried as evidence, never as a trigger.

Old producers without `channel_latency` are skipped (nothing is guessed from the pooled p95). Hops labelled with anything in `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS` (default metacog's own `log_orion_metacognition` dispatch) are measured and logged, never triggered. Episode triggers bypass the 30 s transport cooldown lane and do not consume it; they have their own hourly budget instead. One outage on a hop is one `zero_success` row (it subsumes `timeout`). Episodes close after 15 minutes quiet. Hop identity is `instance`, falling back to `node`. Snapshots skipped for lack of `channel_latency` are logged as `transport_baseline_skip` (first and every 100th), so 'no data' never reads as 'calm'.

Log-only by default. Per-window lines: `transport_baseline_obs` (per-key z, ratio, calls, open conditions) and `transport_baseline_event emit=False`. Those die with the container, so the gate also keeps **one durable summary per (service, instance, hop, UTC hour)**: `TransportBaselineHourlyV1` on `CHANNEL_TRANSPORT_BASELINE_HOURLY`, persisted by orion-sql-writer into `transport_baseline_hourly` (windows seen/evaluated, z p50/p90, saturation-ratio p50, baseline and floor ms, calls/min, conditions opened, conditions open at hour end, would-emit counts per `condition:phase`, excluded, warm, config fingerprint). Flushed 90 s after each hour ends, and on shutdown from the service's `_shutdown` override, before the chassis cancels tasks and closes the bus (`flush_reason=shutdown`; a restart mid-hour gives two rows for that hour). A snapshot that arrives after its hour was flushed becomes a separate `flush_reason=late` row. `warm_at_start` says whether the hop had finished learning when the hour began; the grader ignores warm-up hours. A row that fails to reach Redis is kept in a bounded outbox and retried every 10 s; a row the bus rejects as invalid is dropped with an error. Delivery past Redis is pub/sub, so a row published while sql-writer is down is lost. Grade acceptance check 1 from it: `python scripts/analysis/grade_transport_baseline.py`. **Deploy orion-sql-writer before this service** so the route exists before the first row is published.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_TRANSPORT_BASELINE_ENABLE` | `true` | Fold + persist + log. Publishes nothing on its own |
| `EQUILIBRIUM_TRANSPORT_BASELINE_EMIT` | `true` (on since 2026-10-01) | Publish episode triggers; the gate then owns every timeout it saw and the matching atom is dropped |
| `EQUILIBRIUM_TRANSPORT_EXCLUDE_LABELS` | `log_orion_metacognition,gpu_pool_wait,current_turn_probe` | Baselined but never trigger |
| `EQUILIBRIUM_TRANSPORT_BASELINE_STATE_KEY` | `equilibrium:transport_baseline_state:v1` | Redis key for reducer state + config fingerprint |
| `EQUILIBRIUM_TRANSPORT_BASELINE_MIN_CALLS` | `5` | Calls needed (pooled across windows if sparse) before latency is judged |
| `EQUILIBRIUM_TRANSPORT_BASELINE_N_WARM` | `10` | Judged windows before any latency condition may fire |
| `EQUILIBRIUM_TRANSPORT_BASELINE_SPIKE_Z` | `3.0` | Spike z, sustained 2 judged windows |
| `EQUILIBRIUM_TRANSPORT_BASELINE_MIN_EXCESS_MS` | `250` | Spike/saturation also need this absolute excess over normal (materiality) |
| `EQUILIBRIUM_TRANSPORT_BASELINE_SATURATION_RATIO` | `2.0` | Saturation opens at recent level / floor >= this (closes below 1.5) |
| `EQUILIBRIUM_TRANSPORT_BASELINE_REGIME_AFTER_SEC` | `21600` | Saturation or spike observed this long becomes one `regime_shift` |
| `EQUILIBRIUM_TRANSPORT_BASELINE_MAX_TRIGGERS_PER_HOUR` | `30` | Hourly publish budget for baseline triggers; over budget is logged `transport_baseline_suppressed`. `0` = no cap |
| `EQUILIBRIUM_TRANSPORT_TIMEOUT_ATOM_GRACE_SEC` | `75` | EMIT only: how long a timeout atom waits for a snapshot window that saw it before firing on its own |
| `EQUILIBRIUM_TRANSPORT_BASELINE_HOURLY_PUBLISH_ENABLE` | `true` | Publish hourly per-hop summaries (durable log-only-week evidence) |
| `CHANNEL_TRANSPORT_BASELINE_HOURLY` | `orion:equilibrium:transport_baseline:hourly` | Hourly summary channel (sql-writer -> `transport_baseline_hourly`) |

Changing any tunable changes the state fingerprint: the next boot logs `transport_baseline cold_start reason=config_fingerprint_mismatch` and re-learns. Rollback: set `EQUILIBRIUM_TRANSPORT_BASELINE_ENABLE=false` and delete the Redis key.


### insight metacog trigger (generative, non-rupture; sibling `flow` retired 2026-10-10)

Full design: `docs/superpowers/specs/2026-07-28-collapse-mirror-generative-triggers-design.md`.

**`flow` was retired 2026-10-10** (gate module, detector, settings, env and compose keys, cooldown lane, and cortex-exec's `trigger_kind=="flow"` type override all removed). It fired when the last 20 ticks of `prediction_error_confidence` all sat at or above 0.90 with low variance. It was calibrated 2026-07-30 to fire on 3.2% of windows; by 2026-10-10 (upstream definition changes: z-scored per-domain error, touched-only chat/route, stale-domain omission) **38% of 20-tick windows qualified** and it published ~30 metacog entries a day, with a median gap between publishes of **1800.6 s — exactly its cooldown**, so the cooldown, not the data, set the rate. Recalibrating was checked and rejected: the confidence value is `1 - mean` over five domains, three of which (chat, execution, route) read exactly 0 whenever nothing is happening, so a high calm plateau is what *idleness* looks like. 76% of qualifying windows, and 93% of the calmest 3.2%, had zero chat/execution/route activity in the whole window; a percentile threshold would just pick the quietest idle stretches. No threshold on this field separates "flow" from "nothing happened". Full numbers: `docs/superpowers/pr-reports/2026-10-10-metacog-flow-trigger-calibration-pr.md`. Historical `flow` rows still render through `orion/metacog/evidence_map.py`'s `map_flow` (replay only; nothing produces new ones).

`insight` is the **trigger kind in this service that fires on a positive or neutral state rather than a failure.** Every kind above it is rupture-shaped: a timeout, an anomaly, a repair need. `orion/schemas/collapse_mirror.py`'s own entry-type taxonomy was never error-only (`flow→stabilizing` and `epiphany→reorientation` have always sat alongside `turbulence→escalating`), and the live `pulse` kind already proved a positive-valence trigger works in this exact pipeline.

It reads **one already-live field**, `AttentionSelfModelV1.prediction_error_confidence`, from the `substrate_attention_self_model` Postgres table (written every ~30s by `orion-substrate-runtime`'s `_attention_self_model_tick()`, PR #1459 — this service only ever *reads* that table, and enforces `default_transaction_read_only` on its own session). No new producer, reducer, schema field, or bus channel. The trailing row window is fetched once per poll (`_generative_metacog_poll_loop()`):

- **`insight`** — a *sustained low→high transition*: some tick dropped to/below `EQUILIBRIUM_METACOG_INSIGHT_LOW_THRESHOLD`, and confidence has since climbed to/above `EQUILIBRIUM_METACOG_INSIGHT_HIGH_THRESHOLD` and **held there for `EQUILIBRIUM_METACOG_INSIGHT_CONFIRM_TICKS` consecutive newest ticks**. A surprise got resolved.

**`insight` is deliberately the one gate in this service that is *not* a single-tick threshold crossing.** Every other gate here (`chat_turn`/`transport`/`relational`) fires on a point condition. That would be wrong for this signal, and the reason is measured, not stylistic: PR #1463's baseline pass over real history found confidence recoveries unfold over a **median 3 ticks (~90s), max 12** — a gradual climb. A single-tick crossing gate would fire on noise partway up it. Hence the multi-tick confirm requirement.

**Calibration (real data, not guessed).** Thresholds are **provisional pending a longer-window re-run scheduled 2026-08-02**. Measured 2026-07-30 over 2265 real ticks / ~20.6h:

| Check | Result |
|---|---|
| `prediction_error_confidence` range | 0.597 – 0.977, mean 0.893 |
| Ticks at/below `0.70` (insight's low band) | 14 — rare but real, so arming genuinely happens |
| Ticks at/above `0.90` (high band) | 1544; 1229 consecutive ≥0.90 pairs, so a 2-tick confirm is reachable |
| Rolling 20-tick stdev | p10 0.016 / p50 0.037 / p90 0.059 |

(Rows for the retired `flow` gate's floor calibration removed; see the retirement note above.)

**Freshness and contiguity are enforced, not assumed.** Two guards exist because row adjacency is *not* tick adjacency, and both bugs they close were reproduced against the real detectors during review:

- **Staleness.** The tick writing these rows is itself flag-gated and can simply stop, and a frozen window keeps satisfying both conditions forever ("reducers alive but cursors stale", CLAUDE.md §0A). Pre-fix, a window of rows **3 days old** fired the (since retired) flow gate. Now the poll loop rejects any window whose newest row is older than `EQUILIBRIUM_METACOG_GENERATIVE_MAX_AGE_SEC` and logs a warning naming the writing flag.
- **Gaps.** The reader drops rows whose `prediction_error_confidence` is missing/non-finite, so 20 "consecutive" rows can span hours. Pre-fix, a 20-row window covering **6.08h** fired the (since retired) flow gate as though it were 10 minutes of calm, and a low **5 hours** before the high run fired insight reporting `ticks_to_cross=1`. The detector now also bounds the window in *wall-clock seconds* (derived from `EQUILIBRIUM_METACOG_GENERATIVE_EXPECTED_TICK_SEC` × `..._SPAN_TOLERANCE`), and records the real span (`cross_span_sec`) in `upstream` so a stored row is self-auditing rather than requiring a reader to trust that ticks equal time.

**De-dupe.** `insight` anchors on `low_at`, the tick that armed the recovery — real episode identity, since a genuinely new recovery requires a new low crossing. It deliberately does **not** anchor on `high_at`, which looks stable but is not: when a high run breaks on a single sub-threshold tick and re-forms, `high_at` re-anchors to the new run. Review reproduced that publishing one real recovery **twice, 390s apart** — clearing the 300s cooldown entirely. Replaying 21h of real history confirms the fix: 5 fires before, 4 after, with `low_at` identical across the collapsed pair.

The de-dupe key is recorded **only after a real publish**, never on a cooldown-suppressed one — otherwise an episode would be marked seen while never having been emitted and never retried.

**Downstream type mapping (this is new behavior for the whole family).** Before this, `CollapseMirrorEntryV2.type` was guessed *only* from phi bands in `orion-cortex-exec`'s `_fallback_metacog_draft()` — which is not fallback-only, since the successful-LLM-draft path seeds its `base_entry` from that same function and the draft prompt forbids the LLM from choosing `type` itself. So `trigger_kind` drove `type` in **no** path at all, and `"epiphany"` was unreachable dead code. That heuristic now consults `trigger_kind` **first**: `insight → type="epiphany"` (`change_type=reorientation`); everything else (including a replayed historical `flow`, whose override branch was retired with the gate) falls through to the unchanged phi-band guess.

Shipped **disabled**, same standard as `transport`'s (since retired) bus_synaptic option: it dispatches a real `MetacogTriggerV1` into `orion_metacog`, so flipping it on is a human decision made after a post-merge live-data check. That check never happened — this service's mesh dependency went down for 11 days right after these shipped, so there was no live window to watch. **Flipped on 2026-08-11**, now that the mesh is back. Watch for the first real fire (`orion_metacog` row, `trigger_kind=insight`, `upstream.evidence_source=attention_self_model_prediction_error_confidence`) the same way you would for a fresh flip.

| Env | Default | Purpose |
|-----|---------|---------|
| `EQUILIBRIUM_METACOG_INSIGHT_TRIGGER_ENABLE` | `true` | Master gate for the insight trigger |
| `EQUILIBRIUM_METACOG_GENERATIVE_POLL_INTERVAL_SEC` | `30` | Shared poll cadence; matches the ~30s tick that writes the rows |
| `EQUILIBRIUM_METACOG_GENERATIVE_POSTGRES_URI` | `postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney` | Read-only connection for `substrate_attention_self_model` |
| `EQUILIBRIUM_METACOG_GENERATIVE_WINDOW_TICKS` | `20` | Trailing rows read per poll; the service takes the max of this and what the insight detector needs, so setting it too low cannot silently disable the gate |
| `EQUILIBRIUM_METACOG_GENERATIVE_MAX_AGE_SEC` | `120` | Staleness guard: reject a window whose newest row is older than this |
| `EQUILIBRIUM_METACOG_GENERATIVE_EXPECTED_TICK_SEC` | `30` | Real cadence of the writing tick, used to derive wall-clock span bounds |
| `EQUILIBRIUM_METACOG_GENERATIVE_SPAN_TOLERANCE` | `2.0` | Slack on those derived bounds; absorbs tick jitter without allowing a gappy window |
| `EQUILIBRIUM_METACOG_INSIGHT_COOLDOWN_SEC` | `300` | insight's own cooldown lane |
| `EQUILIBRIUM_METACOG_INSIGHT_LOW_THRESHOLD` | `0.70` | Provisional; low band that arms a recovery |
| `EQUILIBRIUM_METACOG_INSIGHT_HIGH_THRESHOLD` | `0.90` | Provisional; high band that resolves it |
| `EQUILIBRIUM_METACOG_INSIGHT_MAX_TICKS_TO_CROSS` | `15` | Rejects an hours-old low being called a recovery (observed max was 12) |
| `EQUILIBRIUM_METACOG_INSIGHT_CONFIRM_TICKS` | `2` | Consecutive newest ticks that must hold the high band |

---

## Quick start (copy/paste)

### Bus URL
```bash
BUS=redis://100.92.216.81:6379/0
```

### Run
```bash
docker compose up -d orion-equillibrium-service
docker logs -f orion-equillibrium-service
```

### Watch health snapshots
```bash
redis-cli -u "$BUS" SUBSCRIBE "orion:equilibrium:snapshot"
```

### Watch baseline Collapse Mirror snapshots (the metacognition tick)
```bash
redis-cli -u "$BUS" SUBSCRIBE "orion:event:equilibrium:snapshot"
```

---

## What it does
### A) Health aggregation
- Tracks expected services
- Computes healthy/degraded/missing over a time window
- Emits `orion:equilibrium:snapshot`

### B) Baseline Collapse Mirror tick (currently embedded here)
- Constructs `CollapseMirrorStateSnapshot` + `CollapseMirrorEntryV2`
- Emits it as a system “self-awareness” baseline snapshot

---

## Testing

### Unit tests: Transport metacog trigger logic

```bash
# From repo root
python3 -m pytest services/orion-equilibrium-service/tests/test_transport_metacog_gate.py -v
```

`tests/test_bus_synaptic_transport_retired.py` keeps the retired bus_synaptic transport source retired (no builder, poll loop, settings or env keys).

---

## How to use it (practical)
1) Start equilibrium
2) Start a handful of services
3) Stop one service and watch equilibrium mark it missing
4) If collapse ticks are enabled, confirm `orion:event:equilibrium:snapshot` emits every interval

---

## Architectural note (placement)
Two valid stances:

### Clean
- Equilibrium stays only health.
- A dedicated `baseline-snapshotter` / `state-service` emits Collapse Mirror baseline snapshots.

### Pragmatic / evolved
- Keep baseline tick inside Equilibrium.
- Make it opt-in and quiet by default.

---

## Preferred workflow (no channel memorization)
Use logs first:

```bash
docker logs -f orion-equillibrium-service
```

Future stub: equilibrium should print on startup:
- expected services list
- window + publish interval
- collapse tick interval + output channel

---

## Future stubs we should add
- `GET /healthz`, `/readyz`, `/stats`
- A “disable collapse tick” mode via `EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC=0`
- Standard bus summary logging flags (`ORION_LOG_BUS_IN/OUT`)
- Clear schema separation:
  - `EquilibriumSnapshotV1` for health
  - `CollapseMirrorEntryV2` for metacognition tick

---

## Common failure modes
- Publishing to a channel not registered in Titanium channel catalog (enforcer error)
- Expected services list out of sync with actual compose deploy
- Window/grace tuning too aggressive → false missing
