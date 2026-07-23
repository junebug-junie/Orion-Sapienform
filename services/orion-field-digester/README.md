# orion-field-digester

Substrate field digestion worker that consumes committed reduction receipts (biometrics + cortex-exec execution trajectories) and compiles lattice field state:

```text
substrate_reduction_receipts → delta dedupe → perturb/decay/diffuse/suppress → substrate_field_state
```

## Setup

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_field_digester_v1.sql
cp services/orion-field-digester/.env_example services/orion-field-digester/.env
```

## Run

```bash
cd services/orion-field-digester
docker compose up --build
```

Health: `GET http://localhost:8116/health`

## Idle tick (pacemaker)

`FIELD_DIGESTER_IDLE_TICK_ENABLED` (**default `true`**) keeps `tick_id` advancing on every poll even when there are no new receipts to consume. On a quiet poll the worker still loads/reconciles field state, runs decay + diffusion + suppression with an empty perturbation set, mints a new `tick_id`, and persists it via `save_field` — but it does not advance the receipt cursor or write pending deltas (that only happens via `commit_digest_tick` when receipts were actually consumed). This lets downstream free-running consumers (`orion-attention-runtime`, `orion-self-state-runtime`) keep advancing off the latest tick during quiet periods.

**Retention, the applied-deltas pruner, and the health monitor are all prerequisites, and are all live.** `save_field` mints a fresh `tick_id` every call, so every idle tick is a genuine new row in `substrate_field_state` — at the default 2s poll interval that's ~43k rows/day. This is bounded by an hourly batched pruner (`FIELD_STATE_RETENTION_HOURS`, default 72h) that never deletes the newest row. The same cascade happens downstream in `orion-attention-runtime`/`orion-self-state-runtime`, each of which independently prunes its own tables on the same 72h/hourly pattern.

`substrate_field_applied_deltas` (the delta dedup ledger) has no natural "latest row" reader, so instead of a time-based cutoff it is pruned by receipt existence: a dedup row is deleted only once its source receipt is confirmed gone from `substrate_reduction_receipts`, at which point that `delta_id` can structurally never be redelivered by `fetch_new_receipts()`. `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS` is only a small safety margin against racing the receipt pruner's own transaction, not a correctness bound.

A background health monitor (`FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC`, default 900s) watches for: `substrate_field_state`'s oldest row exceeding `FIELD_STATE_STALL_MULTIPLIER` × the retention window (pruner stalled), `substrate_field_applied_deltas` row count exceeding `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT`, and the `conjourney` database exceeding `FIELD_DIGESTER_DB_SIZE_ALERT_GB`. Checks are edge-triggered — an alert (via `orion-notify`'s `POST /attention/request`, surfacing in Hub's existing Pending Attention panel) fires only on a healthy→unhealthy transition, plus a lower-severity recovery note on the way back, so a persisting condition does not spam a fresh attention item every check.

This whole chain was originally left disabled after a prior unbounded-Postgres-growth incident on this host; the guardrails above (mirroring the existing `receipt_pruner.py` pattern in `orion-substrate-runtime`) are what made re-enabling it safe.

## Decay vs. injection-interval mismatch — confirmed root cause of the "accumulator-oscillation artifact" (2026-07-17)

The "Field channel glossary" section below (written 2026-07-16) flagged `cpu_pressure`/`gpu_pressure`
as *"a known accumulator-oscillation artifact (~60s beat); not confirmed whether that beat
reflects real hardware load or a polling-architecture artifact."* This section resolves that
question, found while tracing a separate, unrelated-looking symptom (a drive-pressure economy
saturation bug in `orion/spark/concept_induction/drives.py`, unpacked in full in
[`orion/autonomy/drives_and_autonomy_retrospective.md` §5b](../../orion/autonomy/drives_and_autonomy_retrospective.md)) all the way back to this service.

**It is a polling-architecture artifact, not real hardware load**, and it is a straightforward
consequence of two mechanisms in this pipeline working against each other:

- `apply_decay()` (`app/digestion/decay.py`) multiplies every channel in `NODE_DECAY_CHANNELS`
  (`cpu_pressure`, `memory_pressure`, `gpu_pressure`, `thermal_pressure`, `disk_pressure`,
  `staleness`, `failure_pressure`, `execution_friction`, `reliability_pressure`, and more) by
  `BIOMETRICS_FIELD_DECAY_RATE` (default `0.92`) on **every single `RECEIPT_POLL_INTERVAL_SEC`
  tick (default 2.0s), unconditionally** — whether or not fresh biometrics data arrived that
  tick.
- `apply_perturbations()` (`app/digestion/perturbation.py`) applies a fresh `node_biometrics`/
  `active_node_pressure` reading via `mode="replace"` (the default mode for most hardware
  channels — see the node-channel entries below): `node_vec[channel] = max(0.0, min(1.0,
  p.intensity))`, a **full overwrite**, not a blend, discarding whatever the decayed value had
  drifted to.

`orion-biometrics` only publishes fresh readings every `CLUSTER_PUBLISH_INTERVAL` (default 15s,
a re-broadcast of the latest received summary) or `TELEMETRY_INTERVAL` (default 30s, the actual
host re-measurement) — see `services/orion-biometrics/.env_example`. Between those publishes,
`apply_decay` erodes the channel continuously: `0.92^7 ≈ 0.56`, i.e. a ~44% loss over the ~15s
gap between publishes (7-8 ticks at 2s each). The next publish then snaps the value straight back
up via the full-overwrite `mode="replace"` path, with zero memory of the decayed trajectory.
**This produces a mechanical sawtooth regardless of whether the real underlying host metric is
stable or bursty** — it is an artifact of decay-every-tick-unconditionally plus
reset-via-full-replace, not a reflection of genuine host volatility.

Downstream, this channel-level sawtooth propagates into `SelfStateV1`'s `coherence` dimension
(`coherence_score()` in `orion/self_state/scoring.py` subtracts a penalty over the diffused
generic `pressure` channel, which is fed by `cpu_pressure`/`gpu_pressure`/`memory_pressure`/
`disk_pressure` via the topology edges in `config/field/orion_field_topology.v1.yaml`), producing
a ~16-second-period sawtooth in `coherence` (observed live: smooth rise 0.47→0.84, then a hard
snap back, repeating), which in turn drives `uncertainty` (derived directly from `coherence`) and
`agency_readiness` (a weighted composite of `coherence` and other dimensions) into their own
synchronized sawtooths. This is *not* organic cognitive signal — it mechanically fires
`tension.distress.v1` in `orion/spark/concept_induction/tensions.py` on a schedule tied to this
service's own publish/decay cadence, feeding artificially elevated tension volume into the
drive-pressure economy. See the autonomy retrospective §5b linked above for the full downstream
trace (self-state → tensions → `DriveEngine`) and why this is the same bug *class* — a decay
mechanism whose injection cadence isn't reconciled against its own decay rate — as two other bugs
found and fixed in the same investigation, just in the opposite parameter regime (decay here is
*strong enough* relative to the injection interval to swing dramatically, rather than pinning
flat).

**Blast radius is wider than the drive economy.** `NODE_DECAY_CHANNELS` feeds every consumer of
`FieldStateV1` node/capability vectors, not just self-state's `coherence`/`agency_readiness`/
`resource_pressure` — attention scoring and any other capability-vector consumer reads the same
channels and would show the same artifact.

**Audited 2026-07-22: `"prediction_error"` (one of `NODE_DECAY_CHANNELS`) confirmed working
correctly, not another instance of this bug class.** `node:substrate.route`'s prediction_error
was observed as a subnormal float (~3e-323) and initially suspected as a decay-mechanism defect.
Traced instead to a real quiet period — route arbitration hadn't produced a `lane`/`mind_
requested` mismatch recently, and the write path (`app/ingest/state_deltas.py`'s
`prediction_signal` branch, `mode="replace"`) correctly stamps `node_vector_updated_at` through
the normal `apply_perturbations()` path, so the hold-then-decay logic above applies to it exactly
as designed. See the "fourth training-data quality cutoff" section below for the related (and
real) finding: three of the five nodes contributing to this channel's `max()`-merge were
themselves broken upstream (in `orion-substrate-runtime`, not here) until today.

**Fixed 2026-07-17** (`docs/superpowers/specs/2026-07-17-field-digester-decay-hold-fix-design.md`).
`FieldStateV1.node_vector_updated_at` (new field) tracks the wall-clock timestamp of each
`(node_id, channel)`'s last real write from `apply_perturbations()`. `apply_decay()` now holds a
`NODE_DECAY_CHANNELS` entry flat instead of decaying it, as long as it is within
`FIELD_DECAY_STALENESS_THRESHOLD_SEC` (default 90s — ~3x `TELEMETRY_INTERVAL`/~6x
`CLUSTER_PUBLISH_INTERVAL` of real margin for jitter/backlog) of its last real perturbation, and
only decays it once genuinely stale. A channel that has never been perturbed (or was persisted
from before this fix) keeps decaying every tick unconditionally — the same safe default as
before, so there is no special migration step. `capability_vectors`/`CAPABILITY_DECAY_CHANNELS`
is untouched (already flagged dead under the live diffusion model, out of scope here). This closes
the mechanical-sawtooth mechanism described above; it does **not** by itself address `DriveEngine`'s
separate fold-batch clamp collapse (see the autonomy retrospective §5b/§6 item 5), which is
explicitly deferred pending live data on whether tension volume drops enough after this fix alone.

## `capability_provenance` vs. attention salience (Phase 4, 2026-07-12)

`FieldStateV1.capability_provenance` records which edge source contributed the largest weighted amount to each capability channel this tick (`app/digestion/diffusion.py`) — a magnitude-comparison primitive, distinct from `orion-attention-runtime`'s salience scoring (which factors in novelty/urgency/confidence too, threshold-gated). Cross-checked against live data
(`docs/notes/2026-07-12-phase4-attention-provenance-crosscheck.md`): they corroborate on structurally-obvious nodes (100% agreement for the two single-source edges into `capability:transport`/`capability:graph`) and diverge in an explainable way on contested ones (`node:atlas` wins `capability:llm_inference`'s provenance 81.7% of ticks but only clears attention's top-5 salience list 52.9% of the time — expected, since salience is a stricter, multi-factor, capped signal, not the same question as "who contributed most"). Verdict: keep both, do not unify — they answer different questions by design.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_URI` | (required) | Postgres connection string |
| `LATTICE_PATH` | `config/field/orion_field_topology.v1.yaml` | Node/capability lattice YAML (canonical) |

`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
| `RECEIPT_POLL_INTERVAL_SEC` | `2.0` | Receipt poll interval |
| `BIOMETRICS_FIELD_DECAY_RATE` | `0.92` | Per-tick pressure decay multiplier |
| `BIOMETRICS_FIELD_DIFFUSION_RATE` | `1.0` | Node→capability diffusion strength |
| `FIELD_DIGESTER_IDLE_TICK_ENABLED` | `true` | Keep minting ticks (decay/diffusion only) on quiet polls with no new receipts — see section above |
| `FIELD_STATE_RETENTION_HOURS` | `72.0` | `substrate_field_state` retention window (hourly batched prune) |
| `FIELD_STATE_PRUNE_INTERVAL_SEC` | `3600.0` | `substrate_field_state` prune cadence |
| `FIELD_APPLIED_DELTAS_PRUNE_MIN_AGE_HOURS` | `1.0` | Safety margin before pruning an applied-delta row whose receipt is already gone |
| `FIELD_DIGESTER_HEALTH_CHECK_INTERVAL_SEC` | `900.0` | Health-monitor check cadence |
| `FIELD_STATE_STALL_MULTIPLIER` | `1.5` | Alert if `substrate_field_state`'s oldest row exceeds this × retention hours |
| `FIELD_APPLIED_DELTAS_ALERT_ROW_COUNT` | `5000000` | Alert if `substrate_field_applied_deltas` row count exceeds this |
| `FIELD_DIGESTER_DB_SIZE_ALERT_GB` | `60.0` | Alert if the `conjourney` database exceeds this size (observed baseline ~37.5GB as of 2026-07-12; default leaves real headroom, not a round guess) |
| `NOTIFY_BASE_URL` | `http://orion-athena-notify:7140` | `orion-notify` base URL for health-monitor attention alerts |
| `NOTIFY_API_TOKEN` | (empty) | `orion-notify` auth token, if configured |
| `LOG_LEVEL` | `INFO` | Python log level |
| `FIELD_DIGESTER_PORT` | `8116` | Host port for `docker compose` (compose-only) |
| `FIELD_CHANNEL_CORPUS_PATH` | (empty) | Field-channel raw-substrate corpus sink path — see section below. Empty/unset = disabled |
| `CORPUS_SINK_MAX_BYTES` | `200000000` | Corpus sink rotation threshold (bytes) |
| `CORPUS_SINK_ROTATED_KEEP` | `5` | Corpus sink rotated-file retention count |

v1 persists projections to Postgres only; bus emit is deferred (`orion/bus/channels.yaml` unchanged).

## Field-channel raw-substrate corpus collector (2026-07-13, roadmap item 1 v2)

`_tick()` optionally appends one `FieldChannelCorpusRowV1` row per tick to a
JSONL sink (`FIELD_CHANNEL_CORPUS_PATH`) — the flat, channel-name-keyed
output of `orion.self_state.scoring.collect_field_channel_pressures(state)`
(the merged `node_vectors`/`capability_vectors` pressure dict, e.g.
`cpu_pressure`/`gpu_pressure`/`memory_pressure`/`execution_load`/etc.,
typically 10-20 channels, not a fixed set). This is Item 1 v2 of
`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md` — the
corrected replacement target for `orion-spark-introspector`'s
`mood_arc_corpus.v1` sink (that sink is unaffected, still running; see the
spec doc's "Correction, 2026-07-13" note for why the original 4-scalar
post-composite design was found insufficient for detecting genuine
emergent structure).

**Off by default** (`FIELD_CHANNEL_CORPUS_PATH` empty = no-op, no file
written) — same convention as `MOOD_ARC_CORPUS_PATH`, and for the same
reason: a sibling corpus sink was found to grow unbounded (~104MB/36.8k
rows over 5 days) before rotation was added, so anything new stays inert
until an operator opts in.

The append happens unconditionally on every tick that produces a `state`
(right after the coherence-warning loop, before the `if not fetched: ...`
branch that only decides which store-write path runs) — it does not depend
on whether new receipts were actually fetched this tick.

**Rotation policy** (`CORPUS_SINK_MAX_BYTES`/`CORPUS_SINK_ROTATED_KEEP`) is
the same shared class (`orion.telemetry.corpus_sink.InnerStateCorpusSink`,
promoted out of `orion-spark-introspector`'s `app/` folder in this same
patch since a second service now needs it) and the same default values as
`orion-spark-introspector`'s corpus sinks — each service reads its own env
file independently.

**Dark by design (`REHEARSAL`)**: no bus publish, no cognition consumer —
see `field_channel_corpus.v1` in `orion/inner_state_registry.py`.
A future rework of `orion/mood_arc/fit_encoder.py` to train against this
dict-shaped corpus instead of `mood_arc_corpus.v1` is separate, not-yet-
built work.

## Telemetry-anomaly metacog trigger (2026-07-21)

`FIELD_CHANNEL_ANOMALY_ENABLED` (default `false`) turns on a periodic in-process rescoring loop (`app/anomaly_scorer.py`, `_anomaly_loop()` in `app/worker.py`) against a trained `orion/mood_arc/fit_encoder.py` encoder. Independent of `FIELD_CHANNEL_CORPUS_PATH` above: the scorer maintains its own small in-memory rolling buffer of the same per-tick `FieldChannelCorpusRowV1` rows (not the JSONL sink), so live rescoring works even with the JSONL corpus collector off.

Requires a real trained artifact (`manifest.json` + `weights.npz`, written by `orion/mood_arc/fit_encoder.py train`) at `FIELD_CHANNEL_ANOMALY_ENCODER_DIR` -- this service never trains one itself, and a missing/malformed directory fails open (scoring silently disabled, logged once) rather than crashing the tick loop. Training example (using the corpus-quality cutoff below):

```bash
python orion/mood_arc/fit_encoder.py train \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --min-generated-at 2026-07-17T04:32:14Z \
  --out /mnt/telemetry/models/field_channel_anomaly/v1
```

Every `FIELD_CHANNEL_ANOMALY_CHECK_INTERVAL_SEC` (default 60s -- the encoder's own `window_size` is ~30 rows / ~60s at the default 2s tick cadence, so this scores a genuinely new window each check), the most recent complete window's reconstruction loss is published on `CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE` (`orion:field_channel:anomaly_score`) alongside the encoder's own train-time `recon_error_p95` reference. `orion-equilibrium-service`'s `telemetry_anomaly_metacog_gate.py` is the consumer -- it applies its own threshold multiplier rather than trusting this service's `FIELD_CHANNEL_ANOMALY_THRESHOLD_MULTIPLIER` (informational only here), so trigger sensitivity is tunable on the equilibrium side without redeploying this service. See `services/orion-equilibrium-service/README.md`'s matching section.

| Env | Default | Purpose |
|-----|---------|---------|
| `FIELD_CHANNEL_ANOMALY_ENABLED` | `false` | Master gate |
| `FIELD_CHANNEL_ANOMALY_ENCODER_DIR` | (empty) | Directory with `manifest.json` + `weights.npz` from a prior `train` run |
| `FIELD_CHANNEL_ANOMALY_CHECK_INTERVAL_SEC` | `60` | Rescoring cadence |
| `FIELD_CHANNEL_ANOMALY_THRESHOLD_MULTIPLIER` | `3.0` | Informational only -- see above |
| `CHANNEL_FIELD_CHANNEL_ANOMALY_SCORE` | `orion:field_channel:anomaly_score` | Publish channel |

**Deployed model history** (live `FIELD_CHANNEL_ANOMALY_ENCODER_DIR`, disk artifacts kept, not
deleted, on supersession):

| Version | Trained against | Rows | `floor_ratio` | `ceiling_ratio` | Why superseded |
|---|---|---|---|---|---|
| `v1` | full corpus, no cutoff | — | — | 0.240 | Contaminated by the 2026-07-17 channel-behavior fix sprint (PRs #1108-#1113/#1115) |
| `v2` | `--min-generated-at 2026-07-17T04:32:14Z` | 207,415 / 5 days | 0.282 (CI 0.266-0.304) | 0.189 | `catalog_drift_pressure` stuck the *entire* corpus span (PR #1248's mode=`add`→`replace` fix, second cutoff below) |
| `v3` (current) | `--min-generated-at 2026-07-22T08:29:48Z` | 18,377 / 10.3h | 0.210 (CI 0.174-0.231) | 0.190 | n/a — current live model, deployed after PR #1262's availability-ratchet + merge-window-dedup fix (third cutoff below) |

`v3` trained on a much smaller corpus than `v2` (10.3h of clean data vs. 5 days) — it's a
calibration-quality run on what was available at the time, not a full-corpus retrain. Notably its
`ceiling_ratio` (0.190) landed within 0.001 of `v2`'s (0.189) despite the different corpus size and
composition — an early (n=2) signal this number may be more stable than the "one uncalibrated run"
framing this doc used to carry, though the roadmap's full multi-seed calibration (5+ seeds against
one fixed corpus) hasn't been run yet. Field selection also picked up `availability` for the first
time (`std=0.0398`) — excluded from `v1`/`v2` as degenerate, which in hindsight was itself a symptom
of the ratchet bug PR #1262 fixed, not a real absence of signal.

## `field_channel_corpus.v1` training-data quality cutoff (2026-07-17)

`field_channel_corpus.v1` rows generated **before `2026-07-17T04:32:14Z`**
reflect known-broken channel behavior from before a 7-PR fix sprint
(PRs #1108-#1113, #1115 — #1115 was the last to merge, at the timestamp
above per `gh pr view 1115 --json mergedAt`) that addressed several of the
verdict categories described in the glossary below: one-way ratchets,
saturating counters, merge-polarity masking, and folded-away channels.
Confirmed at full-corpus scale by `orion/mood_arc/fit_encoder.py train`
against the complete `/mnt/telemetry/field_channels/corpus/
field_channels.jsonl` corpus (161,795 rows spanning back to
`2026-07-13T23:46Z`, ~76% pre-cutoff): the run's own
`prune_correlated_fields()` log showed `availability`/`bus_health`/
`delivery_confidence` collapsing pairwise at `r=1.0000` — not real
redundancy, but the "several ratcheted/saturated channels move in lockstep
because they're all frozen or capped the same way" contamination artifact
from the pre-fix period. That run's two-tier gate failed
(`floor_ratio=0.624`, gate requires `<0.5`), worse than validated
pre-full-corpus spike slices scoped entirely to post-fix data
(`floor_ratio` 0.35-0.39).

**Any training run against this corpus should exclude pre-cutoff rows**:

```bash
python orion/mood_arc/fit_encoder.py train \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --min-generated-at 2026-07-17T04:32:14Z \
  --out <out-dir>
```

This is the git **merge** timestamp for PR #1115, not a precisely-confirmed
container-restart/deploy timestamp — actual `orion-field-digester` restart
may have lagged briefly behind the merge. Treat this cutoff as
conservative-but-not-perfectly-precise, not a scientifically exact boundary;
a training run right at the edge of it may still carry a small amount of
pre-fix contamination.

## `field_channel_corpus.v1` second training-data quality cutoff (2026-07-22, PR #1248)

A second, independent contamination window: `transport_pressure`/
`catalog_drift_pressure`/`observer_failure_pressure`/`reliability_pressure`/
`contract_pressure` were injected into the field via the default
`Perturbation` `mode="add"` (`app/ingest/state_deltas.py`'s `transport_bus`
branch) instead of `mode="replace"`, even though `transport_loop_reducer`
recomputes all five fresh every reduction (a "current reading," not an
incremental delta). Combined with `apply_perturbations()` unconditionally
stamping `node_vector_updated_at` regardless of mode, and `apply_decay()`
holding a channel flat while "fresh," any of these five channels that ever
picked up a nonzero value from a real event could get permanently stuck —
immune to both correction (an `add`-mode `0.0` "no drift" report is a no-op)
and decay (perpetually re-marked "fresh"). Confirmed live: `catalog_drift_pressure`
was frozen at exactly `0.13517857261119032` for 10+ minutes across a service
restart while the real value (read directly from `transport_bus_reducer`'s
live Postgres receipts) was `0.0` the entire time — traced end to end via
the `telemetry_anomaly` metacog trigger firing on nearly every tick because
of it (this one channel accounted for ~66% of the average reconstruction
error against the trained `field_channel_anomaly.v2` encoder).

Fixed by PR #1248 (`mode="replace"` for all five channels, matching
`bus_health`/`delivery_confidence`'s existing correct handling), merged
2026-07-22T04:32:27Z, `orion-field-digester` restarted 2026-07-22T04:35:01Z
(`docker inspect orion-athena-field-digester --format
'{{.State.StartedAt}}'`). **Rows generated before `2026-07-22T04:35:01Z`
may carry a stuck value for any of the five channels above** — not
necessarily contaminated at every timestamp (a channel only gets stuck once
it has picked up a nonzero value from a real event and then failed to
decay/correct), but unlike the 2026-07-17 cutoff above, this window has no
known start — a channel could have been stuck for an arbitrarily long time
before anyone noticed, and `catalog_drift_pressure` was in fact stuck for
the entire span of `field_channel_anomaly.v2`'s training corpus
(2026-07-17T04:32:14Z-2026-07-22T01:30:24Z per that manifest), confirmed
live post-fix: with the pipeline bug fixed and the real value now correctly
reading `0.0`, `v2` (trained almost entirely on the stuck ~`0.135` reading)
started flagging the *correct* value as anomalous instead —
`telemetry_anomaly` still fired 20 times in the 41 minutes after the
restart, same channel, opposite direction. **Resolved by `v3`** (see
"Deployed model history" above and the third cutoff below) — `v2` is no
longer deployed.

```bash
python orion/mood_arc/fit_encoder.py train \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --min-generated-at 2026-07-22T04:35:01Z \
  --out <out-dir>
```

If both cutoffs apply, use whichever is later. **Superseded by the third
cutoff below as of `v3`** — the retrain against this cutoff alone was
skipped; by the time enough clean data existed to retrain at all, PR #1262
had also merged, so `v3` trained directly against the third (later)
cutoff instead. Kept here for the historical record and in case a future
retrain ever needs to reason about this cutoff specifically (e.g. scoring
a slice that only needs to exclude the second contamination window, not
the third).

## `field_channel_corpus.v1` third training-data quality cutoff (2026-07-22, PR #1262)

A third, independent contamination window, found in `orion/substrate/
biometrics_loop` (upstream of this service, not a field-digester bug
itself) while re-auditing channels for the work above:

1. **`availability` one-way ratchet.** A transient biometrics staleness
   blip permanently flagged a node's `availability` pressure — the only
   removal rule was hardcoded to clear a different pressure kind
   (`"strain"`), so `availability` could never recover. Confirmed live:
   `node:atlas` reported `availability=0.0` for hours after its biometrics
   resumed reporting fresh. `availability` itself is **not** one of `v2`'s
   16 trained channels (excluded already by field selection, presumably as
   a near-constant/degenerate signal) — but that exclusion was itself an
   artifact of this bug: a channel that's permanently stuck at one value
   looks exactly like a channel with no real signal. Once nodes can
   actually recover, `availability` may show real variance in future
   corpus data and could be selected into a future retrain where it
   previously wasn't.
2. **Biometrics-pressure merge-window dedup was a no-op across ticks**
   (a separate bug in the same reducer, unrelated to the ratchet above) —
   `node:atlas` alone accepted 767 "reinforce" deltas in 2 hours instead of
   the ~24 a working 5-minute window should have allowed. This inflated
   `pressure_score`, which feeds `cpu_pressure`/`gpu_pressure` (via
   `active_node_pressure` deltas' `"strain"` pressure kind, `mode="add"`,
   `state_deltas.py:36`) — `cpu_pressure` **is** one of `v2`'s 16 trained
   channels. Corpus rows recorded before this fix reflect artificially
   inflated reinforcement-flood contamination on `cpu_pressure` (and, for
   `atlas`/`circe`, `gpu_pressure`); rows after it reflect the
   `merge_window_sec=300` dedup actually working as designed.

Both fixed by PR #1262 (`orion/substrate/biometrics_loop/pressure_organ.py`
Rule B' + `pressure_reducer.py`'s `last_accepted_at`-based dedup, plus a
companion fix in this service's `state_deltas.py` restoring `availability`
to `1.0` on the recovery transition), merged 2026-07-22T08:06:41Z.
`orion-field-digester` restarted 2026-07-22T08:29:00Z,
`orion-substrate-runtime` (which runs the biometrics_loop reducer)
restarted 2026-07-22T08:29:48Z — the later of the two,
**`2026-07-22T08:29:48Z`, is the binding cutoff** (corpus contamination
could come from either service, so the cutoff is only safe once both have
picked up the fix). Confirmed live immediately post-restart:
`node:atlas`'s `availability` read `1.0` (was permanently `0.0`), and its
`node_pressure_reducer` receipts dropped to 2 in the first 40 seconds then
went quiet, consistent with the 5-minute merge window actually holding
instead of firing every ~9 seconds:

```bash
python orion/mood_arc/fit_encoder.py train \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --min-generated-at 2026-07-22T08:29:48Z \
  --out <out-dir>
```

If multiple cutoffs apply, use whichever is latest (as of this writing,
that's this third cutoff, `2026-07-22T08:29:48Z`). Same reasoning as the
second cutoff still applies to any future retrain: skip pre-cutoff rows,
or the run will reproduce a `cpu_pressure`-shaped version of the same
contaminated-baseline problem `v2` already hit for `catalog_drift_pressure`.

**`v3` trained against exactly this cutoff** (18,377 rows / 10.3h clean
data — much smaller than `v2`'s 207K-row/5-day corpus, since this cutoff
had only just landed) and is the currently deployed
`FIELD_CHANNEL_ANOMALY_ENCODER_DIR`. See "Deployed model history" above
for the real gate results. A future retrain against a fuller corpus once
more clean data accumulates past this cutoff would still be worthwhile
(more data, tighter confidence intervals) but is not blocking — `v3` is a
real, gate-passing, currently-serving model, not a placeholder. **See the
fourth cutoff below, though** — `v3`'s training window predates that fix,
so it inherited that contamination on one input channel.

## `field_channel_corpus.v1` fourth training-data quality cutoff (2026-07-22)

A fourth contamination window, this one on `prediction_error` — one of the
15 channels `v2` actually trained on (`orion/mood_arc/docs/DESIGN.md`), and
by field-selection carryover presumably one of `v3`'s too (`v3`'s own
per-channel selection results aren't independently confirmed in this pass —
see the caveat at the end of this section). **`v3`, the currently deployed
encoder, is also affected**, since its training window (the third cutoff,
2026-07-22T08:29:48Z onward) predates the fix below.
`prediction_error` is a `max()`-merge across five nodes (`orion/field/
pressure.py::collect_field_channel_pressures()`, `PRESSURE_CHANNELS`) —
`node:substrate.{biometrics,execution,transport,chat,route}`, each written
by `services/orion-substrate-runtime`'s five shadow prediction-error
instruments (`orion/substrate/prediction_error.py`). Three of those five
were broken until today, upstream of this service, not a field-digester bug:

1. `execution_prediction_error()`/`route_prediction_error()` diffed a
   `curr` run against `prev`'s exact `trace_id` match, which structurally
   never occurred (real runs are single-shot creates) — permanently `0.0`
   regardless of activity. Fixed (commit `a98854a2`).
2. `chat_prediction_error()` skipped every brand-new `turn_id` (the only
   kind that ever appears in a fresh tick), also permanently `0.0` in
   production — confirmed live, `node:substrate.chat` had never been
   written despite 241 real accumulated chat turns. Fixed (PR #1267).
3. `transport_prediction_error()` was **not** broken — its low reading is a
   real quiet bus, not an instrument defect (see the earlier fix note in
   this same file). **Correction (2026-07-22, later same day): "quiet bus"
   undersold this.** `transport_prediction_error()` is fed entirely by
   `orion-bus`'s bus-observer role, which structurally can only ever watch
   `orion:stream:world_pulse:run:result` and its DLQ — the only two real
   Redis Streams anywhere in the architecture (everything else is pub/sub,
   with no depth/backlog concept). So this instrument doesn't sample "the
   bus" and find it quiet; it can only ever see one service's result queue,
   full stop. Confirmed live: `XLEN orion:stream:world_pulse:run:result` =
   `91` against `BUS_STREAM_DEPTH_CRITICAL=100000` (ratio ~0.00091), flat
   with zero movement for the entire ~18h window since the fix above — the
   contributing `transport` node to this merged channel is not "currently
   calm," it is structurally incapable of reflecting broader bus/transport
   stress at all. Full trace: `services/orion-bus/README.md`'s "Naming
   caveat for downstream consumers," `services/orion-substrate-runtime/
   README.md`'s "transport domain scope" note, `orion/mood_arc/README.md`'s
   matching caveat, and `orion/sentience_striving_program/README.md` §9b
   item 3's matching correction.

Net effect: for as long as those two instruments were broken, the merged
`prediction_error` channel could only ever reflect `biometrics_prediction_
error()` (introduced 2026-07-21) or `transport_prediction_error()` (always
near-zero) — confirmed directly against the training corpus file itself
(`/mnt/telemetry/field_channels/corpus/field_channels.jsonl`): its earliest
available rows (2026-07-18T20:41Z, predating even the biometrics/chat/
route instruments' 2026-07-21 introduction) already read `prediction_error
= 3e-323` — the decay floor `apply_decay()` (`NODE_DECAY_CHANNELS` includes
`"prediction_error"`) settles a stale channel to. This is consistent with
`prediction_error` having been at or near that floor for its entire history
in this corpus, not a channel with real learnable variance — the opposite
of what `v2`'s field-selection pass presumably assumed when it kept this
channel.

Fixed by `a98854a2` (merged earlier 2026-07-22) and PR #1267 (merged
2026-07-22T19:16:01Z); `orion-substrate-runtime` restarted
2026-07-22T19:18:31Z (`docker inspect orion-athena-substrate-runtime
--format '{{.State.StartedAt}}'`) — the later of the two, and the binding
cutoff since both fixes live in that one service. Confirmed live
post-restart directly against the corpus file: rows minutes after the
restart read `prediction_error = 0.0671`-`0.1171` (varying across
consecutive ticks, not stuck) — a qualitatively different, real signal
compared to the pre-cutoff decay-floor constant. Exact per-node
attribution of this specific post-restart value has not been traced
further in this pass (worth a follow-up if a `v4` retrain's field-selection
behaves unexpectedly for this channel) — the corpus-level before/after
contrast is the confirmed finding here, not a claim about which of the
three now-real instruments produced it.

```bash
python orion/mood_arc/fit_encoder.py train \
  --corpus /mnt/telemetry/field_channels/corpus/field_channels.jsonl \
  --min-generated-at 2026-07-22T19:18:31Z \
  --out <out-dir>
```

If multiple cutoffs apply, use whichever is latest (as of this writing,
that's this fourth cutoff, `2026-07-22T19:18:31Z`). Same reasoning as the
second/third cutoffs: **do not retrain yet** — there are only minutes of
clean post-cutoff data as of this writing, nowhere near `v2`'s 207K-row/
5-day corpus. Let clean data accumulate first; see the agent board for the
tracked follow-up. **`v3` remains the right model to keep serving in the
meantime** — it is not invalidated on the metrics it actually gates on
(`floor_ratio`/`ceiling_ratio` don't single out `prediction_error`), just
carrying one contaminated-but-not-dominant input channel, the same way
`v2` carried `catalog_drift_pressure` before the second cutoff.

## Field channel glossary

This is the consolidated reference for all 29 channels in
`field_channel_corpus.v1` (`orion.schemas.telemetry.field_channel_corpus.FieldChannelCorpusRowV1`),
sourced from `app/tensor/channels.py`'s `NODE_CHANNELS` (23) + `CAPABILITY_CHANNELS`
(8), overlapping on `transport_pressure`/`contract_pressure` (23+8-2=29). This
corpus is the raw input for a future windowed autoencoder (roadmap item 2,
not yet trained) — but several of these same channels also feed
`SelfStateV1`'s live, cognition-facing dimensions today, via
`orion.self_state.scoring.collect_field_channel_pressures()` and
`config/self_state/self_state_policy.v1.yaml`. Written 2026-07-16 because no
centralized doc previously existed for what each channel means, how it's
calculated, or which self-state dimension (if any) it feeds.

**Live observability (2026-07-21)**: the per-channel category/meaning/
self-state-dimension metadata below is now also structured, machine-readable
data at `config/field/field_channel_glossary.v1.yaml`
(`orion.self_state.field_channel_glossary`), read by Hub's "Field Channels"
tab (`services/orion-hub/static/field-channel-glossary.html`,
`GET /api/field-channel-glossary/{channels,health}`). That panel does
**not** read the verdicts below -- it computes a live clean/dead verdict
per channel from real `substrate_field_state` data over a rolling window
using the same subnormal-cutoff + variance heuristic
`scripts/analysis/measure_capability_channel_health.py` validated, because
the verdicts in this section are a point-in-time write-up and can (and,
per the "Update" notes throughout this section, already have) go stale as
fixes land. Treat this section as the prose mechanism reference and the
Hub panel as the current-truth check.

**Pipeline order** (`app/tensor/update_rules.py::run_digestion_tick`, called
once per tick from `app/worker.py::_tick`): reconcile (seed any
lattice-declared node/capability not yet present with
`DEFAULT_NODE_VECTOR`/`DEFAULT_CAPABILITY_VECTOR`, `app/tensor/reconcile.py`)
→ `apply_perturbations` → `apply_decay` → `apply_diffusion` →
`apply_suppression` → `check_field_coherence` (sets `field_coherence_warning`
directly, outside the perturbation system) → `collect_field_channel_pressures`
(merges `node_vectors`+`capability_vectors` into the flat corpus row).

**Producer/calculation column key**: "target_kind" refers to
`StateDeltaV1.target_kind` as consumed by
`app/ingest/state_deltas.py::delta_to_perturbations()`, which has six blocks:
`active_node_pressure`, `node_biometrics`, `execution_run`, `chat_turn`,
`transport_bus`, `prediction_signal`. "mode" refers to `Perturbation.mode`
(`app/digestion/perturbation.py`): `replace` sets the channel directly
(clamped 0-1); the `availability` special case floors via `min(current,
intensity)` (can only decrease); every other default (`add`) accumulates via
`min(1.0, current + intensity)` (only increases within a tick, relies on
decay to come back down). `NODE_DECAY_CHANNELS`/`CAPABILITY_DECAY_CHANNELS`
are `app/digestion/decay.py`'s two decay sets — the capability set is
currently dead weight for every capability in the live topology, since
`apply_diffusion()` runs immediately after in the same tick and
unconditionally overwrites every diffusion-target channel with a fresh
memoryless recompute (`decay.py` lines 46-64, dated 2026-07-12). Topology
edges are `config/field/orion_field_topology.v1.yaml`. Note: that yaml file
also declares its own `node_channels`/`capability_channels` lists (lines
19-49) that look authoritative but are **not parsed by any code** (no
reference to either key anywhere under `app/`) — they are stale, incomplete
documentation inside the yaml itself (missing `egress_confidence_deficit`,
`prediction_error`, `field_coherence_warning` from the node list), not a
second source of truth. `channels.py`'s `NODE_CHANNELS`/`CAPABILITY_CHANNELS`
are the only enforced channel set.

**SelfState dimension column key**: from
`config/self_state/self_state_policy.v1.yaml`. `channel_dimension_map`
entries score a dimension directly. `evidence_channel_map` entries are
evidence/transparency only (surfaced in `dominant_evidence`/reasons) and do
**not** score anything — this exists specifically so
`orion-spark-introspector`'s tissue-viz bypass can read raw hardware-channel
names without reintroducing the double-counting bug that removing them from
`channel_dimension_map` fixed (2026-07-12). `stabilizing_channels` entries
carry an explicit weight used inside `scoring.py::coherence_score()`.
Verified against the full 138-line policy file, not a partial read:
`contract_pressure` and `catalog_drift_pressure` appear in **none** of
`channel_dimension_map`, `evidence_channel_map`, `stabilizing_channels`, or
`pressure_channels` — both are calculated and corpus-collected but feed no
live `SelfStateV1` dimension at all today.

**Live-data verdict column key**: verdicts below were re-verified against
the live corpus at `/mnt/telemetry/field_channels/corpus/field_channels.jsonl`
(123,332+ rows / 69h+ as of 2026-07-16) rather than taken on faith. Verdict
categories used: **real signal** (continuously or genuinely varying, no
known bug), **one-way ratchet** (mode=`add` + absent from
`NODE_DECAY_CHANNELS`, so it can only climb, never come back down),
**collateral damage** (a healthy-looking channel whose observed flatness is
actually caused by a *different* channel's bug), **masked by merge-polarity
bug** (real variance exists somewhere in the mesh but is hidden by
`collect_field_channel_pressures()`'s max()-merge, which is correct for
pressure-type channels but backwards for "higher-is-better" channels),
**saturating counter** (pinned by a fixed-size cap, not real activity
running out), **folded-away** (computed individually upstream but only ever
reaches the field as part of a different composite scalar, never perturbed
by its own name), **fully unproduced** (more extreme than folded-away: no
perturbation source and no diffusion source ever populates it, so it never
even reaches the merged corpus as a key, not even at 0.0), **exact
duplicate** (two channels observed byte-identical across every live row
checked), and **quiet/genuinely-wired** (correctly implemented, healthy
decay mechanism, simply has not received a real nonzero input in the
observed window).

Two mechanics worth knowing before reading the verdicts: (1)
`collect_field_channel_pressures()` (`orion/self_state/scoring.py:70,76`)
only adds a channel to the merged output when `channel in PRESSURE_CHANNELS
or value > 0` — `PRESSURE_CHANNELS` (scoring.py:7-27) is a 19-channel subset
of the 29; a channel outside that set only ever appears in the corpus once
it has received at least one genuinely nonzero perturbation, which is why
some channels (e.g. `transport_pressure`) never appear at all while others
with near-zero float noise (e.g. `observer_failure_pressure`, max observed
value `3e-323` — a floating-point subnormal, not a real signal) do appear.
(2) `DEFAULT_NODE_VECTOR`/`DEFAULT_CAPABILITY_VECTOR` (`channels.py:37-42`,
`availability`/`confidence`/`available_capacity` seeded to `1.0`) are
actively applied every tick via `reconcile_field_state_with_lattice()`
(`app/tensor/reconcile.py`, called from `app/worker.py:132`, before
`run_digestion_tick` at `app/worker.py:156`) — they are not vestigial.
`app/digestion/suppression.py:9`'s own `vec.get("availability", 1.0)`
default is redundant defensive coding on top of that, not the actual source
of the observed 1.0 baseline.

### Semantic categories

The per-channel entries below are organized mechanically (by `NODE_CHANNELS`
vs `CAPABILITY_CHANNELS`, roughly alphabetical within each). That's the right
order for looking up a specific channel, but it doesn't show what the 29
channels actually *mean* as a set. Grouping by real-world meaning instead
(not by producer mechanism, decay membership, or any other code-structural
property) surfaces 7 categories — written up here because no such taxonomy
existed anywhere in the repo before this section (checked: neither this
glossary, `orion/inner_state_registry.md` (relocated 2026-07-22 from
`orion/self_state/README.md`, SelfStateV1 burn), nor any `docs/superpowers/specs/`
felt-state-arc doc groups channels by meaning rather than by mechanism or
producer). This grouping was originally derived while selecting a feature
subset for the mood-arc windowed-autoencoder spike (see
`docs/superpowers/specs/2026-07-13-felt-state-arc-roadmap-spec.md` and
`project_mood_arc_roadmap_status` in memory) but applies to the full
29-channel set, not just that spike's pruned subset.

1. **Physical substrate / embodiment** — `cpu_pressure`, `gpu_pressure`,
   `thermal_pressure`, `memory_pressure`, `disk_pressure`. Pure hardware
   sensor readings. Nothing cognitive here at all.
2. **Task-execution domain** — `execution_load`, `execution_friction`,
   `execution_pressure`, `reasoning_load`, `reasoning_pressure`,
   `failure_pressure`, `egress_confidence_deficit`. What Orion is actively
   *doing* computationally right now (all sourced from `execution_run`
   deltas) and how well it's going.
3. **Social / conversational domain** — `conversation_load`,
   `repair_pressure`. The texture of the current chat interaction (sourced
   from `chat_turn` deltas) — a genuinely separate axis from execution: a
   conversation can be smooth while execution struggles, or vice versa.
4. **Infrastructure / transport workload** — `catalog_drift_pressure`,
   `transport_pressure`, `contract_pressure`, `observer_failure_pressure`.
   The bus's own workload/health as infrastructure (sourced from
   `transport_bus` deltas), not a compute resource — a distinct axis from
   both hardware and task execution.
5. **Sensor trust / liveness** — `availability`, `staleness`,
   `expected_offline_suppression` at the node level; `bus_health`,
   `delivery_confidence` are the direct transport-layer parallel to the
   same question applied to the bus rather than a node. None of these five
   say what a node (or the bus) is *doing* — all five say whether that
   source's *other* readings can be trusted right now. This is a meta-layer
   over categories 1-4, not a peer to them.
6. **Synthesized / aggregate rollups** — `pressure`, `available_capacity`,
   `confidence`, `reliability_pressure`. Capability-level composites that
   already blend multiple node-level signals (from categories 1-2) into one
   summary per functional capability (orchestration, transport, etc) — no
   direct sensor among them.
7. **Self-monitoring / introspection** — `prediction_error`,
   `field_coherence_warning`. The most distinct category: neither measures
   the world or the hardware body — both measure whether Orion's own
   internal model is internally consistent (`field_coherence_warning`: do
   two reducers that should agree actually disagree) or accurate
   (`prediction_error`: did the prediction hold). No physical-sensor analog.
   This is the one category in the raw channel set that is already closer
   to genuine introspection than external-world sensing — directly relevant
   to the felt-state-arc roadmap's actual premise (testing "whether Orion
   has recurring felt-state trajectory structure beyond single-tick phi"):
   categories 1-6 are all substrate/task/social/infra telemetry: prediction
   error and cross-reducer coherence are the one place self-referential
   signal already exists in this corpus. Worth deliberately checking
   whether a trained encoder's latent dimensions correlate more with
   category 7 than with the others (mirrors
   `orion/mood_arc/fit_encoder.py::compute_window_probes()`'s existing
   valence-correlation-probe pattern, generalized to this channel set).

### Node channels (`NODE_CHANNELS`, 23)

#### `availability`
- **Meaning**: how available/reachable a compute node currently is.
- **Producer**: `active_node_pressure` delta, `"availability"` in
  `active_pressures` → special-cased in `perturbation.py`'s `apply_perturbations`
  (not through the generic add/replace path): `min(current, intensity)`,
  i.e. floor-only, can only decrease within a tick. Node-only; no topology
  edge diffuses `availability` to any capability. Not in `NODE_DECAY_CHANNELS`.
- **SelfState dimension fed**: `channel_dimension_map`: `availability` →
  `coherence`. Also `stabilizing_channels`: `availability` weight `0.50`
  (used inside `coherence_score()`).
- **Live-data verdict**: collateral damage. Floored to `≥0.85` on every tick
  by `apply_suppression()` (`suppression.py:8-9`) because
  `expected_offline_suppression` is latched at `1.0` essentially always;
  separately, its own perturbation path has never fired in the live corpus.
  Confirmed live 2026-07-16.

#### `staleness`
- **Meaning**: whether a node's most recent biometrics sample is too old to
  trust.
- **Producer**: `node_biometrics` delta, set to `0.5` (mode=`add`, default)
  when `availability_status == "stale"`. In `NODE_DECAY_CHANNELS`. Not a
  diffusion source/target anywhere in the topology.
- **SelfState dimension fed**: `channel_dimension_map`: `staleness` →
  `continuity_pressure`. Also referenced directly inside
  `coherence_score()`'s penalty loop (`("failure_pressure",
  "execution_friction", "staleness", "pressure")`, `0.25` weight each).
- **Live-data verdict**: collateral damage. Unconditionally zeroed every
  tick by `apply_suppression()` (`suppression.py:10`) whenever
  `expected_offline_suppression >= 1.0` — not natural rarity, actively
  suppressed. Confirmed live 2026-07-16.

#### `cpu_pressure`
- **Meaning**: how loaded a node's CPU currently is.
- **Producer**: `active_node_pressure` delta (`"strain"` in
  `active_pressures`, non-GPU nodes) and `node_biometrics` delta
  (`hints["strain"]`), both mode=`add`. In `NODE_DECAY_CHANNELS`. Diffuses
  into `capability:orchestration` (`pressure`, weight `0.90`),
  `capability:graph` (`pressure`, weight `0.70`), `capability:memory` via
  `node:prometheus` (`pressure`, weight `0.60`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly
  (removed 2026-07-12 — its diffused capability value is what's mapped
  instead). `evidence_channel_map`: `cpu_pressure` → `resource_pressure`
  (evidence/transparency only).
- **Live-data verdict**: real signal, continuous. The "known
  accumulator-oscillation artifact" flagged 2026-07-16 was confirmed
  2026-07-17 as a polling-architecture artifact, not real hardware load
  (`apply_decay()`'s unconditional `0.92`-per-2s decay against
  `orion-biometrics`' ~15-30s publish cadence produced a mechanical sawtooth,
  ~16s period observed downstream in `coherence`, independent of whether the
  underlying CPU load was actually bursty) — **and fixed the same day**. See
  "Decay vs. injection-interval mismatch" below for the mechanism and the
  fix (hold-flat-until-stale via `node_vector_updated_at`).

#### `memory_pressure`
- **Meaning**: node RAM pressure.
- **Producer**: declared in `NODE_CHANNELS`, `NODE_DECAY_CHANNELS`, and as a
  diffusion target for `capability:storage` (`memory_pressure` → `pressure`,
  weight `0.75`, alongside `disk_pressure`) — but no `target_kind` block in
  `state_deltas.py` ever perturbs it by name, so its value never leaves its
  reconcile-seeded `0.0` default.
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `memory_pressure` → `resource_pressure`
  (evidence-only; in practice inert since the channel never carries a real
  value).
- **Live-data verdict**: folded-away, never produced. Computed individually
  as `mem_pressure` in `orion/telemetry/biometrics_pipeline.py:103` but only
  reaches the field as part of the composite `"strain"` scalar
  (`services/orion-biometrics/app/grammar_emit.py:159`); `state_deltas.py`'s
  `node_biometrics` block only reads `hints["gpu"]`/`hints["strain"]`, never
  a memory-named hint key.

#### `gpu_pressure`
- **Meaning**: how loaded a node's GPU currently is.
- **Producer**: `active_node_pressure` delta (`"strain"`, GPU nodes
  `{atlas, circe}`) and `node_biometrics` delta (`hints["gpu"]`), both
  mode=`add`. In `NODE_DECAY_CHANNELS`. Diffuses into
  `capability:llm_inference` from `node:atlas` (`pressure`, weight `0.85`,
  alongside `memory_pressure`) and `node:circe` (`pressure`, weight `0.50`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `gpu_pressure` → `resource_pressure`
  (evidence-only).
- **Live-data verdict**: real signal, continuous — same accumulator-
  oscillation mechanism as `cpu_pressure`, confirmed as the
  decay/injection-interval mismatch described below, not real hardware load,
  and fixed by the same 2026-07-17 patch.

#### `thermal_pressure`
- **Meaning**: node thermal/temperature pressure.
- **Producer**: computed in `biometrics_pipeline.py:117` but no
  `target_kind` block perturbs it by name. In `NODE_DECAY_CHANNELS`. Not a
  diffusion source/target anywhere in the topology.
- **SelfState dimension fed**: `channel_dimension_map`: `thermal_pressure` →
  `resource_pressure` — this one **is** a direct entry (unlike
  `cpu_pressure`/`gpu_pressure`/`memory_pressure`/`disk_pressure`, which were
  demoted to evidence-only in the 2026-07-12 dedup pass), but since the
  channel is never produced this mapping is currently a dead entry in
  practice.
- **Live-data verdict**: folded-away, never produced — same mechanism as
  `memory_pressure`, composited into `"strain"` only.

#### `disk_pressure`
- **Meaning**: node disk I/O pressure.
- **Producer**: computed in `biometrics_pipeline.py:110` but no
  `target_kind` block perturbs it by name. In `NODE_DECAY_CHANNELS`.
  Diffusion target for `capability:storage` (`disk_pressure` → `pressure`,
  weight `0.75`, alongside `memory_pressure`) — inert since the source value
  is never nonzero.
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `disk_pressure` → `resource_pressure`
  (evidence-only).
- **Live-data verdict**: folded-away, never produced.

#### `expected_offline_suppression`
- **Meaning**: signals a node is expected to be offline (e.g. a scheduled or
  known-suppressed state), so its absence shouldn't be read as a failure.
- **Producer**: `active_node_pressure` delta sets `1.0` (mode=`add`) when
  `delta.operation == "suppress"`; `node_biometrics` delta sets `1.0`
  (mode=`add`) when `expected_online is False`. **Not** in
  `NODE_DECAY_CHANNELS` — mode=`add` with no decay means it can only
  increase, never come back down.
- **SelfState dimension fed**: `channel_dimension_map`:
  `expected_offline_suppression` → `coherence`. Also `stabilizing_channels`:
  weight `0.30`.
- **Live-data verdict**: one-way ratchet — latched at `1.0` for the entire
  observed corpus span. Confirmed live 2026-07-16.

#### `execution_load`
- **Meaning**: how much active execution work (agent runs, tool calls) is in
  flight on a node.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:orchestration`
  (`execution_pressure`, weight `0.90`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly
  (removed 2026-07-12). `evidence_channel_map`: `execution_load` →
  `execution_pressure` (evidence-only).
- **Live-data verdict**: real signal, continuous. Confirmed live 2026-07-16.

#### `execution_friction`
- **Meaning**: how much resistance (retries/backoff) execution is
  encountering on a node.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:orchestration`
  (`reliability_pressure`, weight `0.90`, `max()`'d against
  `failure_pressure`'s contribution to the same target).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `execution_friction` → `reliability_pressure`
  (evidence-only). Also referenced directly in `coherence_score()`'s
  penalty loop (`0.25` weight).
- **Live-data verdict**: real signal — part of a correlated sparse-event
  trio with `failure_pressure`/`reliability_pressure`; one real spike traced
  to `2026-07-16T02:15:08Z`, decaying cleanly afterward at the `0.92`/tick
  rate — genuine rare-event signal, not dead.

#### `reasoning_load`
- **Meaning**: how much active LLM-reasoning work a node is doing.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:llm_inference`
  (`reasoning_pressure`, weight `0.85`, from `node:atlas`) and
  `capability:orchestration` (`reasoning_pressure`, weight `0.90`, from
  `node:athena`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `reasoning_load` → `reasoning_pressure`
  (evidence-only).
- **Live-data verdict**: real signal — the cleanest channel in the corpus,
  continuously varying, small amplitude. Confirmed live 2026-07-16.
  **Update 2026-07-18**: this verdict was `node:athena`-only. `node:atlas`'s
  `reasoning_load` was permanently `0.0` (48h live-confirmed) because
  `execution_run` deltas always carried `node_id=athena` (cortex-exec's own
  static `NODE_NAME`), regardless of which physical node actually served the
  LLM call. Fixed: `execution_run`'s `pressure_hints` gained a companion
  `llm_serving_node` field (sourced from llm-gateway's response
  `meta.served_by`, already live in `LLM_GATEWAY_ROUTE_TABLE_JSON`) that
  routes this specific channel's perturbation to the real serving node —
  see `app/ingest/state_deltas.py`'s `execution_run` branch and
  `orion/substrate/execution_loop/`. `node:athena`'s reading is unaffected
  (still the orchestrating-node value); `node:atlas` should now show real
  variation once live LLM traffic accumulates post-deploy. Live-verified
  post-deploy (PR #1177, merged 2026-07-18T04:10:01Z): `node:atlas`'s
  `reasoning_load` mean/std/max went from exactly `0`/`0`/`0` (767 rows,
  pre-merge) to `0.003`/`0.012`/`0.05` (115 rows immediately post-merge),
  climbing to `0.032`/`0.024`/`0.05` a few minutes later as more traffic
  accumulated. `node:circe`'s `reasoning_load` is **still** permanently
  `0.0` -- this is a separate, operational (not code) gap: circe is never
  `served_by` in the live `LLM_GATEWAY_ROUTE_TABLE_JSON` today, so it never
  produces an `execution_run` at all. See
  `services/orion-llm-gateway/README.md`'s route-table section for the
  known-gap note and what's needed to bring it online -- this fix is
  already ready for circe (`circe` is in cortex-exec's
  `_KNOWN_FIELD_NODES`), it just has no live traffic to attribute yet.

#### `failure_pressure`
- **Meaning**: recent execution failure rate/severity on a node.
- **Producer**: `execution_run` delta, mode=`replace`. In
  `NODE_DECAY_CHANNELS`. Diffuses into `capability:orchestration`
  (`reliability_pressure`, weight `0.90`).
- **SelfState dimension fed**: not in `channel_dimension_map` directly.
  `evidence_channel_map`: `failure_pressure` → `reliability_pressure`
  (evidence-only). Also referenced directly in `coherence_score()`'s
  penalty loop (`0.25` weight).
- **Live-data verdict**: real signal — part of the same correlated
  sparse-event trio as `execution_friction`, same `2026-07-16T02:15:08Z`
  spike, genuine rare-event signal.

**Cross-lane overwrite fix (live-confirmed 2026-07-23, same day as initial deploy):**
`harness_step_load`, `tool_failure_streak_pressure`, `avg_step_chars_pressure`,
`compliance_deficit`, and `turn_incompletion` (the 5 entries below) are now gated in
`app/ingest/state_deltas.py` to only ever produce a perturbation from their own
specific trace lane (`:harness_motor` for the first 4, `:hub_turn_timeout` for the
last). Before this fix, `compute_pressure_hints()` unconditionally included these
keys in every `execution_run`'s `pressure_hints` dict regardless of source (defaulting
to `0.0`/`"unknown"` for cortex-exec-only runs), so `key in hints` was always true and
every execution_run delta targeting the same node -- including the very same turn's
own `:harness_finalize_reflect`/`:orion_voice_finalize` cortex-exec sub-lanes, and
every later unrelated cortex-exec-only turn -- emitted a `mode="replace"` perturbation
that reset the real value. Live-reproduced within hours of the initial deploy: a real
`harness_step_load=0.6892` was overwritten back to `0.0` within 15 seconds by the same
turn's own `:harness_finalize_reflect` delta. See
`docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md`'s
follow-up note, and `test_execution_run_fcc_channels_ignored_off_lane` /
`test_execution_run_off_lane_delta_does_not_reset_prior_harness_value`
(`tests/test_field_execution_run_perturbations.py`) for the regression coverage.

#### `harness_step_load`
- **Meaning**: FCC-motor-only step-load proxy for one unified Hub turn, split out of
  `execution_load` (which blends cortex-exec + harness-governor step counts and hard-caps
  at 8). A compute/thermal/power cost proxy for how much processing one turn actually cost
  on the expensive GPU node running the motor.
- **Producer**: `execution_run` delta, `min(1.0, log1p(harness_started_step_count) /
  log1p(60))`, mode=`replace`. In `NODE_DECAY_CHANNELS`.
  **Correction (found in code review, before this patch ever shipped)**: the first version
  of this channel used bare `log1p(harness_started_step_count)`, reasoned to be safely
  unbounded because the one consumer checked at design time
  (`orion/substrate/relational/adapters/execution_ctx.py`'s `max(hints.values())` →
  `_clamp()`) already clamps at read time. That reasoning missed the actual primary
  consumer: `apply_perturbations()` (`app/digestion/perturbation.py`) hard-clamps every
  `mode="replace"` channel to `[0,1]` at write time, and bare `log1p(n)` exceeds `1.0` at
  `n=2` — saturating this channel for virtually every real harness-governor run and
  defeating its own purpose (an 8-step and a 40-step run both landing on the materialized
  value `1.0`). The log-ratio form above keeps the channel genuinely continuous instead.
  `60` is a starting reference denominator, not yet calibrated against live step-count
  distributions. Not yet wired into `CAPABILITY_CHANNELS`/diffusion — an explicit follow-up
  decision, not an oversight.
- **SelfState dimension fed**: none yet — new channel, no downstream mapping wired in this
  patch.
- **Live-data verdict**: not yet live-verified, pending deploy. See
  `docs/superpowers/specs/2026-07-23-fcc-motor-field-digester-signals-design.md` for the
  full design and the `execution_load` split rationale (also flagged there as needing its
  own follow-up: `execution_load` itself was left unnarrowed in this patch, so it still
  double-counts harness-governor steps that this channel now also measures separately).

#### `tool_failure_streak_pressure`
- **Meaning**: is the FCC motor stuck repeating the same tool failure (e.g. a denied
  permission) instead of adapting? `min(1.0, longest_consecutive_same_error_kind_streak / 3.0)`
  for one turn.
- **Producer**: `execution_run` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`. Sourced
  from `is_error` flags on `tool_result` content blocks inside each FCC step (NOT the
  once-per-turn fcc-subprocess `error` event, which is structurally separate and doesn't
  fire on individual tool-call failures) — see `orion/harness/fcc_motor.py`'s
  `_extract_tool_result_errors()`.
- **SelfState dimension fed**: none yet.
- **Live-data verdict**: not yet live-verified, pending deploy. The `3`-repeat saturation
  threshold and `short_error_kind()`'s bucketing granularity against realistic tool-result
  text are both starting anchors, not yet sampled against live data — see the spec doc's
  Missing Questions.

#### `avg_step_chars_pressure`
- **Meaning**: per-step verbosity of the FCC motor's tool output for one turn —
  `min(1.0, (total_step_chars / completed_step_count) / 4000.0)`.
- **Producer**: `execution_run` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`. Sourced
  from `measure_step_payload_chars()` (`orion/fcc/context_budget.py`, already computed
  per-step for context-overflow budgeting, now also summed/max-tracked across a turn).
- **SelfState dimension fed**: none yet.
- **Live-data verdict**: not yet live-verified, pending deploy. The `4000`-char saturation
  anchor is a starting guess, not yet sampled against live data — see the spec doc's
  Missing Questions.

#### `compliance_deficit`
- **Meaning**: did the FCC motor actually produce a compliant, delivered turn?
  `HarnessRunV1.compliance_verdict` rank (`completed`=0, `partial`=1, `failed`=2,
  `refused`=3) `/ 3.0`. Deliberately separate from `failure_pressure` — a turn can resolve
  `partial` (context overflow, empty draft) with zero individual tool-step failures, so the
  two measure genuinely different things.
- **Producer**: `execution_run` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`. Sourced
  from `HarnessRunV1.compliance_verdict`, threaded through as a new
  `exec_result_assembled` grammar kv (previously absent from the grammar stream entirely).
  `"refused"` is set on the pre-motor refusal path
  (`services/orion-harness-governor/app/bus_listener.py`'s `_emit_refused_lifecycle_grammar`)
  and now reaches this channel too — previously a documented coverage gap, closed in this
  same patch.
- **SelfState dimension fed**: none yet.
- **Live-data verdict**: not yet live-verified, pending deploy.

#### `turn_incompletion`
- **Meaning**: did Hub fail to get a usable result back from the harness-governor RPC
  for this turn? Most commonly a true timeout; occasionally a reply that arrived but
  failed to decode (see Producer note). A distinct, worse severity than
  `compliance_deficit`'s worst rank (`refused`/`failed`) — those still require the
  governor to have replied with *something* Hub could use. `1.0` if the turn hit Hub's
  `run is None` path, `0.0` otherwise.
- **Producer**: `execution_run` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`. Sourced
  from a NEW `orion-hub`-published grammar event (`semantic_role="exec_turn_timeout"`,
  `services/orion-hub/scripts/grammar_emit.py`'s `build_turn_timeout_grammar_events()`),
  fired from `orion/hub/turn_orchestrator.py`'s `run is None` branch. For a true timeout
  this is the ONE unified-turn failure mode where no governor-side grammar event exists
  at all — `HarnessGrammarCollector` only flushes its buffered lifecycle atoms once, at
  the end of a run, and a true timeout never reaches that point. For the rarer
  decode-failure sub-case, the governor's own reply did arrive and may already have
  flushed real lifecycle grammar on its own trace lane — this channel doesn't
  distinguish the two sub-cases (both read `run is None` to Hub). Required widening
  `EXECUTION_SOURCE_SERVICES` (`orion/substrate/execution_loop/constants.py`) to
  include `orion-hub` — the one real contract-surface touch in this signal's design,
  since without it the reducer silently no-ops the event.
- **Node attribution**: Hub's own node (parsed from this event's `hub_turn_timeout`
  trace lane, under Hub's `NODE_NAME` — Hub cannot reliably know which physical node
  the governor's own `HarnessGrammarCollector` would have used, since the RPC it was
  waiting on never returned). This measures **Hub's view of governor
  unresponsiveness**, not the governor's own node health — the two may run on
  different physical nodes.
- **SelfState dimension fed**: none yet.
- **Live-data verdict**: not yet live-verified, pending deploy.

#### `egress_confidence_deficit`
- **Meaning**: `1 - confidence` that an execution's output actually reached
  its destination.
- **Producer**: `execution_run` delta, `max(0.0, min(1.0, 1.0 -
  egress_raw))`, mode=`replace`. In `NODE_DECAY_CHANNELS`. Not a diffusion
  source/target anywhere in the topology.
- **SelfState dimension fed**: `channel_dimension_map`:
  `egress_confidence_deficit` → `introspection_pressure` (direct).
- **Live-data verdict**: real signal, sparse/low-duty-cycle but genuinely
  varying. Confirmed live 2026-07-16.

#### `repair_pressure`
- **Meaning**: how much conversational "repair" (corrections,
  re-explaining) is happening in chat.
- **Producer**: `chat_turn` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`.
  Not a diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`: `repair_pressure` →
  `social_pressure` (direct).
- **Live-data verdict**: real signal, sparse but genuinely varying.

#### `conversation_load`
- **Meaning**: how much active conversational load (turn volume/complexity)
  is occurring.
- **Producer**: `chat_turn` delta, mode=`replace`. In `NODE_DECAY_CHANNELS`.
  Not a diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`: `conversation_load`
  → `social_pressure` (direct).
- **Live-data verdict**: real signal, sparse but genuinely varying.

#### `transport_pressure`
- **Meaning**: bus/transport-layer backpressure/congestion (a node channel
  and a capability channel, one of the two overlapping names between
  `NODE_CHANNELS` and `CAPABILITY_CHANNELS`).
- **Producer**: `transport_bus` delta, via `hints["transport_pressure"]`,
  `hints["stream_depth_pressure"]`, or `hints["backpressure"]` — all
  mode=`add` (default), targeting a **node** vector. In
  `NODE_DECAY_CHANNELS` and `CAPABILITY_DECAY_CHANNELS`. As a node channel
  it is a diffusion source for `capability:orchestration` (`pressure`,
  weight `0.90`) and `capability:transport` (`pressure`, weight `0.85`), and
  for the `capability:transport → capability:orchestration` cap-cap edge
  (`transport_pressure` → `transport_pressure`, weight `0.70`) — but that
  cap-cap edge's own source value (`capability:transport`'s own
  `transport_pressure` key) is only ever seeded `0.0` by
  `DEFAULT_CAPABILITY_VECTOR`, since the `node:athena → capability:transport`
  edge maps `transport_pressure` → `"pressure"`, not `"transport_pressure"`
  — no edge ever writes a channel literally named `transport_pressure`
  directly onto `capability:transport`.
- **SelfState dimension fed**: not in `channel_dimension_map` directly
  (removed 2026-07-12). `evidence_channel_map`: `transport_pressure` →
  `resource_pressure` (evidence-only).
- **Live-data verdict, corrected 2026-07-23 (the 2026-07-16 "fully
  unproduced" verdict below is stale — do not carry it forward):**
  `transport_pressure` is not in `PRESSURE_CHANNELS`, so
  `collect_field_channel_pressures()` (`scoring.py:70,76`) only includes it
  when a real nonzero perturbation lands — same gate the 2026-07-16 check
  found nothing passing. Since then its live history has moved through three
  distinct phases, all traced to the same producer
  (`compute_transport_pressures()`'s `transport_pressure = max(stream_depth_
  pressure, backpressure)`, `orion/substrate/transport_loop/extract.py:105`):
  (1) **before 2026-07-18** (`BUS_OBSERVER_STREAMS` still pointed at
  placeholder/never-wired stream names — see `services/orion-bus/README.md`)
  — genuinely `MISSING`, matching the 2026-07-16 check exactly; (2)
  **2026-07-18 through the second training-data cutoff (2026-07-22T04:35Z,
  PR #1248)** — pinned at `1.0` (ceiling-saturated; confirmed directly
  against the corpus at rows dated 2026-07-18T20:41Z and
  2026-07-20T17:42Z), the same `mode="add"` accumulation bug documented
  above for `catalog_drift_pressure`/`observer_failure_pressure`/
  `reliability_pressure`; (3) **since the fix** — flat and constant at
  `0.00091` across every row checked (20,000-row window, 2026-07-22T12:50Z
  through 2026-07-23T00:04Z, zero variance) — the same value, and the same
  root cause, as `prediction_error`'s transport contribution: `XLEN
  orion:stream:world_pulse:run:result` (`91`) divided by
  `BUS_STREAM_DEPTH_CRITICAL` (`100000`). See
  `services/orion-substrate-runtime/README.md`'s "transport domain is one
  queue, not the bus" note — this raw channel and `prediction_error`'s
  transport contribution are two different computations (current level vs.
  tick-over-tick delta) over the exact same narrowly-scoped input, so both
  inherit the identical structural ceiling on what they can ever measure.
  **`stream_depth_pressure` and `backpressure` are not independent corpus
  channels** — per the Producer note above, all three hint keys
  (`transport_pressure`, `stream_depth_pressure`, `backpressure`) target the
  same single node channel, `transport_pressure`; `stream_depth_pressure`
  never appears under its own name in the corpus (confirmed `MISSING` in
  every row checked), it is folded into `transport_pressure` at the source.
  **This is not a documented-and-abandoned problem** — a real replacement is
  in progress, not a rename: see `services/orion-substrate-runtime/README.md`'s
  "transport domain is one queue, not the bus" note for the correction to an
  earlier (wrong) "dead consumer" theory, and
  `docs/superpowers/specs/2026-07-23-transport-domain-rpc-health-redesign.md`
  (PR #1290 shipped the first measurement step) for the redesign itself. Not
  yet wired to replace this channel or feed the field-digester corpus.

#### `contract_pressure`
- **Meaning**: intended to represent pressure from bus/schema "contract"
  mismatches (the precise real-world condition isn't otherwise documented
  in code — it's perturbed from the same `transport_bus` hint dict as
  `catalog_drift_pressure`). A node channel and a capability channel (the
  other overlapping name).
- **Producer**: `transport_bus` delta, `hints["contract_pressure"]`,
  mode=`add` (default), onto a **node** vector. In `NODE_DECAY_CHANNELS` and
  `CAPABILITY_DECAY_CHANNELS`. At the capability level, `contract_pressure`
  is actually populated by a *different* node-level channel:
  `node:athena → capability:transport`'s edge maps `catalog_drift_pressure`
  → `"contract_pressure"` (weight `0.85`) — no edge maps the node-level
  `contract_pressure` channel itself into any capability field.
- **SelfState dimension fed**: **none.** Verified against the full 138-line
  `self_state_policy.v1.yaml` (not a partial read): `contract_pressure`
  appears in none of `channel_dimension_map`, `evidence_channel_map`,
  `stabilizing_channels`, or `pressure_channels`. Calculated and
  corpus-collected, feeds no live `SelfStateV1` dimension.
- **Live-data verdict**: exact duplicate of `catalog_drift_pressure` —
  byte-identical across every row checked (123,246 pairs, 0 mismatches,
  re-verified 2026-07-16). Root cause not yet determined by this patch
  (documentation-only): structurally, the field-digester code treats them
  as two independent perturbation targets from two separate hint keys, and
  the one diffusion path linking them (`catalog_drift_pressure` →
  capability-level `contract_pressure`, weight `0.85`) would scale the
  value down, not reproduce it byte-for-byte — which points more toward an
  upstream reducer setting both `hints["contract_pressure"]` and
  `hints["catalog_drift_pressure"]` to the same value in the same
  `transport_bus` delta, rather than a field-digester-side convergence. Not
  confirmed; a separate investigation is tracking the actual root cause.

#### `catalog_drift_pressure`
- **Meaning**: intended to represent drift/staleness in the bus event
  "catalog" (schema registry) relative to what's actually flowing.
- **Producer**: `transport_bus` delta, `hints["catalog_drift_pressure"]`,
  mode=`add` (default), onto a node vector. In `NODE_DECAY_CHANNELS`. This
  is the channel that actually feeds the capability-level `contract_pressure`
  field via the `node:athena → capability:transport` diffusion edge (weight
  `0.85`) — see `contract_pressure` above.
- **SelfState dimension fed**: **none** — same verification as
  `contract_pressure` above; absent from all four policy maps.
- **Live-data verdict**: exact duplicate of `contract_pressure` (see above);
  open root-cause question, not fixed by this patch. **Correction
  (2026-07-23): this channel is one of `v2`'s (and, by field-selection
  carryover, presumably `v3`'s) 15 trained mood-arc channels
  (`orion/mood_arc/docs/DESIGN.md`) — worth flagging beyond the byte-
  identical-duplicate finding above.** Producer:
  `catalog_drift_pressure = min(state.uncataloged_stream_count / denom,
  1.0)` (`orion/substrate/transport_loop/extract.py:88`) —
  `uncataloged_stream_count` counts streams the bus-observer watches
  (`BUS_OBSERVER_STREAMS`) that are *not* in `orion/bus/channels.yaml`'s
  catalog. Since the 2026-07-18 fix (`services/orion-bus/README.md`),
  `BUS_OBSERVER_STREAMS` only ever names the two streams that *are*
  cataloged (`orion:stream:world_pulse:run:result` + its DLQ) — so
  `uncataloged_stream_count` is now structurally always `0`, and
  `catalog_drift_pressure` is now structurally always `0.0` too, going
  forward, by construction, not by observation. Confirmed against the
  live corpus: `MISSING` in every one of the last 5,000 rows checked
  (2026-07-23) — same `PRESSURE_CHANNELS`/`value > 0` gate as
  `transport_pressure` above, and this channel never clears it anymore.
  Practically: a channel `v2`/`v3` were trained to expect some real signal
  from is now permanently silent under the current bus-observer config —
  worth accounting for if a future retrain's field-selection drops it or
  treats it as dead. Not a bug to fix (the 2026-07-18 change was correct
  for its own stated purpose); a downstream side effect worth having on
  record.

#### `delivery_confidence`
- **Meaning**: confidence that bus messages are actually being delivered
  end-to-end.
- **Producer**: `transport_bus` delta, `hints["delivery_confidence"]`,
  mode=`replace` (fixed prior to this session — the mode=`add` one-way
  ratchet described below in the 2026-07-16 verdict no longer applies; see
  `bus_health`/`delivery_confidence`'s existing correct handling" in the
  cutoff section above). **Not** in `NODE_DECAY_CHANNELS` — a "current
  reading" score with no decay story, same as `availability`. Diffuses into
  `capability:transport`'s `"confidence"` field (weight `0.85`, from
  `node:athena`) — the only capability with a direct diffusion edge into
  `confidence`.
- **Single-observer node channel (2026-07-22)**: only `node:athena`'s
  bus-observer can ever produce a real reading for this channel —
  atlas/circe/prometheus run llamacpp-host + biometrics only, with no code
  path that could ever legitimately report bus/delivery state.
  `collect_field_channel_pressures()` merges `delivery_confidence` across
  all 4 lattice nodes via `min()` (it's a `HIGHER_IS_BETTER_CHANNELS`
  entry), and a stale pre-2026-07-17 `0.0` already persisted on a
  non-reporting node could never self-correct (reconcile only fills in
  *missing* channels, it doesn't overwrite an already-present stale value).
  Confirmed live: `node:athena` reported a real, fresh `1.0` continuously
  while the merged corpus/`SelfStateV1` `coherence` dimension read `0.0`,
  masked by `node:atlas`'s stale, never-updated entry.
  `SINGLE_OBSERVER_NODE_CHANNELS` (`app/tensor/channels.py`) now makes this
  explicit: `reconcile.py`'s `_ensure_node_vector()` prunes this channel
  from every node except its owner on every tick — self-healing, no manual
  data migration needed. `orion/self_state/transport.py`'s
  `transport_channel_hints()` already read this channel from `node:athena`
  directly (never merged) and was unaffected by the bug.
- **SelfState dimension fed**: not in `channel_dimension_map` directly (the
  capability-level `confidence` field it feeds is the one mapped — see
  `confidence` below). `evidence_channel_map`: `delivery_confidence` →
  `coherence` (evidence-only) — this is the dimension the stale-mask bug
  above was corrupting.

#### `bus_health`
- **Meaning**: overall bus/transport subsystem health signal.
- **Producer**: `transport_bus` delta, `hints["bus_health"]`, mode=`replace`
  (fixed prior to this session — see `delivery_confidence` above). **Not**
  in `NODE_DECAY_CHANNELS` — same "current reading, no decay" reasoning.
  Diffuses into `capability:transport`'s `"available_capacity"` field
  (weight `0.85`, from `node:athena`) — the only capability with a direct
  diffusion edge into `available_capacity`.
- **Single-observer node channel (2026-07-22)**: identical fix and
  rationale as `delivery_confidence` above — see that entry.
  `SINGLE_OBSERVER_NODE_CHANNELS` covers both channels together.
- **SelfState dimension fed**: not in `channel_dimension_map` directly (the
  capability-level `available_capacity` field it feeds is the one mapped —
  see `available_capacity` below). `evidence_channel_map`: `bus_health` →
  `coherence` (evidence-only) — this is the dimension the stale-mask bug
  above was corrupting.

#### `observer_failure_pressure`
- **Meaning**: pressure from failures of the bus "observer" role
  (monitoring/subscriber-side failures).
- **Producer**: `transport_bus` delta, `hints["observer_failure_pressure"]`,
  mode=`add` (default). **Is** in `NODE_DECAY_CHANNELS` — a healthy decay
  mechanism, same as `cpu_pressure`/`gpu_pressure`. Diffuses into
  `capability:transport`'s `"reliability_pressure"` field (weight `0.85`,
  from `node:athena`).
- **SelfState dimension fed**: not in `channel_dimension_map` or
  `evidence_channel_map` by its own name — only its diffusion target
  (`reliability_pressure`) is mapped (see `reliability_pressure` below).
- **Live-data verdict**: genuinely quiet, correctly wired, no bug. Uses
  add-mode and is in the healthy decay set; simply has never received a
  real nonzero perturbation. Confirmed live 2026-07-16: present in 123,258
  rows, max observed value `3e-323` (a floating-point subnormal — decay
  noise, not a real signal).

#### `field_coherence_warning`
- **Meaning**: per-node incoherence score — fires when one channel is high
  and a paired channel that should track it is simultaneously low,
  suggesting two different reducers disagree about the node's actual
  state (rules: `execution_load`/`cpu_pressure`,
  `execution_load`/`gpu_pressure`, `failure_pressure`/`availability`,
  `transport_pressure`/`bus_health`, `reasoning_load`/`cpu_pressure` —
  `orion/field_coherence.py`).
- **Producer**: **not** one of `state_deltas.py`'s six `target_kind`
  blocks. Computed by `check_field_coherence(state)`
  (`orion/field_coherence.py:37`) and written directly onto
  `state.node_vectors[node_id]["field_coherence_warning"]`
  (`app/worker.py:163-164`) *after* `run_digestion_tick` (perturb/decay/
  diffuse/suppress) completes each tick — it bypasses the `Perturbation`
  add/replace mechanism entirely (direct dict assignment, effectively
  memoryless per tick). In `NODE_DECAY_CHANNELS` (largely moot since it's
  overwritten fresh every tick anyway). Not a diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`:
  `field_coherence_warning` → `coherence` (direct).
- **Live-data verdict**: real signal — present in all 123,332 rows checked,
  nonzero in all of them, max observed `0.4`. Sparse/low-duty-cycle in
  magnitude but genuinely, continuously computed. Confirmed live
  2026-07-16.

#### `prediction_error`
- **Meaning**: how much a recent prediction (from the prediction/surprise
  subsystem) missed reality.
- **Producer**: `prediction_signal` delta, `hints["prediction_error"]`,
  `max(0.0, min(1.0, ...))`, mode=`replace`. In `NODE_DECAY_CHANNELS`. Not a
  diffusion source/target.
- **SelfState dimension fed**: `channel_dimension_map`: `prediction_error` →
  `uncertainty` (direct).
- **Live-data verdict**: quiet-so-far-but-correctly-wired, not dead. Flat
  (float noise on the order of `1e-18`, an artifact of repeated `0.92`
  decay applied to something already effectively zero) for the original
  69h scan window. A live re-check mid-session (2026-07-16) found real
  values starting to appear: first meaningful (`>1e-4`) value at row
  122,564, `2026-07-16T20:33:59.918142Z`, subsequently reaching `0.92`. This
  is a live-growing corpus; the channel was quiet during the scan window,
  not broken.

### Capability-only channels (`CAPABILITY_CHANNELS` not already listed above, 6)

#### `pressure`
- **Meaning**: overall load/pressure for a given capability
  (`llm_inference`, `orchestration`, `storage`, `graph`, `memory`,
  `transport`).
- **Producer**: capability-level only, populated purely by diffusion — most
  node→capability edges target `"pressure"` (see the per-channel entries
  above for the specific node-level sources feeding each capability). In
  `CAPABILITY_DECAY_CHANNELS`, but per `decay.py`'s 2026-07-12 comment this
  is dead weight: `apply_diffusion()` runs immediately after in the same
  tick and unconditionally overwrites it with a fresh memoryless recompute.
- **SelfState dimension fed**: `channel_dimension_map`: `pressure` →
  `resource_pressure` (direct).
- **Live-data verdict**: real signal, no bug — continuously varying (e.g.
  `capability:orchestration` observed mean `0.55`, std `0.20`). Confirmed
  live 2026-07-16.

#### `confidence`
- **Meaning**: how confident the system is that a given capability is
  currently reliable/available.
- **Producer**: capability-level. `capability:transport` has the only
  direct diffusion edge into `confidence` (from `delivery_confidence`,
  weight `0.85`). Every other capability
  (`orchestration`/`llm_inference`/`storage`/`graph`/`memory`) uses
  `apply_diffusion()`'s fallback formula when no direct edge fed it a real
  value this tick: `confidence = max(0.0, 1.0 - 0.5 * pressure)`
  (`diffusion.py:216-229`). Seeded `1.0` by `DEFAULT_CAPABILITY_VECTOR` via
  reconcile.
- **SelfState dimension fed**: `channel_dimension_map`: `confidence` →
  `coherence` (direct). Also `stabilizing_channels`: weight `0.50`.
- **Live-data verdict**: masked by merge-polarity bug. Real variance exists
  at `capability:orchestration` (derived from its own genuinely-varying
  `pressure`, mean `0.55`/std `0.20`) but is masked because
  `collect_field_channel_pressures()`'s max()-merge
  (`scoring.py:70,76`, `v >= out.get(channel, 0.0)`) is correct for
  pressure-type channels (higher = worse, worst-contributor wins) but
  backwards for a "goodness" channel like `confidence` — `capability:
  transport`'s near-1.0 constant value (itself a symptom of the
  `delivery_confidence` one-way ratchet) always wins the max(), permanently
  hiding `orchestration`'s real variation from the merged corpus. Confirmed
  live 2026-07-16.

#### `available_capacity`
- **Meaning**: how much spare operating capacity a capability has right
  now.
- **Producer**: capability-level. `capability:transport` has the only
  direct diffusion edge into `available_capacity` (from `bus_health`,
  weight `0.85`). Every other capability falls back to
  `available_capacity = max(0.0, 1.0 - pressure)` (`diffusion.py:216-227`).
  Seeded `1.0` by `DEFAULT_CAPABILITY_VECTOR` via reconcile.
- **SelfState dimension fed**: `channel_dimension_map`: `available_capacity`
  → `coherence` (direct). Also `stabilizing_channels`: weight `0.50`.
- **Live-data verdict**: masked by the same merge-polarity bug as
  `confidence` — `capability:transport`'s ratcheted near-1.0 constant
  always wins the max()-merge over `orchestration`'s real variation.

#### `execution_pressure`
- **Meaning**: capability-level execution-load pressure (currently only
  populated for `orchestration`).
- **Producer**: capability-level, diffused from `node:athena →
  capability:orchestration` (`execution_load` → `execution_pressure`,
  weight `0.90`). In `CAPABILITY_DECAY_CHANNELS`, same dead-weight caveat as
  `pressure` (overwritten by `apply_diffusion()` immediately after decay,
  every tick).
- **SelfState dimension fed**: `channel_dimension_map`: `execution_pressure`
  → `execution_pressure` (direct, 1:1 dimension name).
- **Live-data verdict**: real signal, continuous — same underlying trace as
  `execution_load`. Confirmed live 2026-07-16.

#### `reasoning_pressure`
- **Meaning**: capability-level reasoning-load pressure. Nominally
  `llm_inference`/`orchestration`'s reasoning burden, but see live-data
  verdict below — in practice this is `orchestration`-only right now.
- **Producer**: capability-level, diffused from `node:atlas →
  capability:llm_inference` (`reasoning_load` → `reasoning_pressure`,
  weight `0.85`) and `node:athena → capability:orchestration`
  (`reasoning_load` → `reasoning_pressure`, weight `0.90`). In
  `CAPABILITY_DECAY_CHANNELS`, same dead-weight caveat as `pressure`.
- **SelfState dimension fed**: `channel_dimension_map`: `reasoning_pressure`
  → `reasoning_pressure` (direct, 1:1).
- **Live-data verdict**: real signal, but single-source, not dual-source as
  the topology implies. Confirmed live 2026-07-18 against
  `substrate_field_state.field_json` (48h, 85,347 rows):
  `capability:llm_inference`'s `reasoning_pressure` is exactly `0.0` for the
  entire window (std=0, min=max=0), because its diffusion source,
  `node:atlas`'s own `reasoning_load`, is itself permanently `0.0` over the
  same window — never instrumented/populated, not decayed. `capability:
  orchestration`'s `reasoning_pressure` is the only real contributor (mean
  `0.018`, std `0.022`, range `0.0`–`0.315`).

  The merged corpus channel is genuinely real and continuously varying (this
  investigation's "cleanest channel" verdict from 2026-07-17 still holds as
  a description of the *merged output*) — but that's because
  `collect_field_channel_pressures()`'s max()-merge correctly picks the one
  live source over a dead zero, not because both capabilities are
  contributing. This is a different mechanism from the `confidence`/
  `available_capacity` masking bug above: there, max()-merge had the wrong
  *polarity* and let a broken value beat a real one; here the polarity is
  correct (pressure-type, worst-contributor-wins) and the merge is doing
  exactly what it should — no code fix needed. `reasoning_pressure` is one
  of `capability:llm_inference`'s 5/8 dead channels found independently by
  `scripts/analysis/measure_capability_channel_health.py` (PR #1171,
  `feat/capability-salience-channel-audit`), consistent with that
  investigation's broader finding that most of `llm_inference`'s and
  `transport`'s channels are structurally never populated upstream. Treat
  any future "is reasoning_pressure real" question as answered for
  `orchestration`; `llm_inference`'s side needs a `node:atlas` instrumentation
  fix, not a merge-logic fix, before it contributes anything.

  **Update 2026-07-18**: the `node:atlas` instrumentation fix above shipped.
  Root cause was narrower than "never instrumented" implied: `node:atlas`
  never received *any* `execution_run`-derived channels (not just
  `reasoning_load`) because `execution_run`'s `node_id` always came from
  cortex-exec's own static `NODE_NAME` (`athena`), never from which physical
  node actually served the LLM call. Fixed by threading llm-gateway's
  `meta.served_by` (already live) through as a new `llm_serving_node` field
  on the run, and routing `reasoning_load`'s perturbation to that node
  specifically instead of the orchestrating node — see `reasoning_load`'s
  entry above and `app/ingest/state_deltas.py`. Deliberately scoped to
  `reasoning_load` only; `execution_load`/`execution_friction`/
  `failure_pressure` are legitimately orchestrator-node concerns and stay on
  `node:athena`. Re-verify `capability:llm_inference.reasoning_pressure`'s
  liveness against `substrate_field_state` once post-deploy LLM traffic has
  accumulated — not yet re-measured as of this fix landing.

#### `reliability_pressure`
- **Meaning**: capability-level pressure representing risk to reliability
  (from execution friction, failures, or observer failures).
- **Producer**: capability-level only per `channels.py`'s declared lists
  (not in `NODE_CHANNELS`), diffused from `node:athena →
  capability:orchestration` (`execution_friction` → `reliability_pressure`
  and `failure_pressure` → `reliability_pressure`, both weight `0.90`,
  `max()`'d together per the 2026-07-12 diffusion fix) and `node:athena →
  capability:transport` (`observer_failure_pressure` →
  `reliability_pressure`, weight `0.85`). Note: `decay.py`'s
  `NODE_DECAY_CHANNELS` set *also* lists `reliability_pressure` (line 26)
  even though it's absent from `channels.py`'s `NODE_CHANNELS` — currently
  inert for `node_vectors` specifically, since no producer ever writes
  `reliability_pressure` onto a node vector by name; it's meaningful for
  `capability_vectors` via `CAPABILITY_DECAY_CHANNELS` (same dead-weight
  caveat as `pressure`, though, per the diffusion overwrite).
- **SelfState dimension fed**: `channel_dimension_map`: `reliability_pressure`
  → `reliability_pressure` (direct, 1:1).
- **Live-data verdict**: real signal — the third leg of the correlated
  sparse-event trio with `execution_friction`/`failure_pressure`, same
  `2026-07-16T02:15:08Z` spike, genuine rare-event signal, decaying cleanly
  afterward.
