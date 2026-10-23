# Orion SQL Writer

The **SQL Writer** service is a durable consumer that subscribes to various bus channels and persists structured payloads into a relational database (PostgreSQL). It uses a configurable routing map to determine which SQLAlchemy model to use for each message kind.

## Contracts

### Consumed Channels
Configured via `SQL_WRITER_SUBSCRIBE_CHANNELS` (JSON list).

| Default Channel | Kind(s) | Target Table |
| :--- | :--- | :--- |
| `orion:tags:enriched` | `tags.enriched`, `collapse.enrichment` | `CollapseEnrichment` |
| `orion:collapse:sql-write` | `collapse.mirror` | `CollapseMirror` |
| `orion:chat:history:log` | `chat.history.message.v1` | `ChatMessageSQL` |
| `orion:chat:history:turn` | `chat.history`, `chat.log` | `ChatHistoryLogSQL` |
| `orion:chat:gpt:log` | `chat.gpt.message.v1` | `ChatGptMessageSQL` |
| `orion:chat:gpt:turn` | `chat.gpt.log.v1`, `chat.gpt.turn.v1` | `ChatGptLogSQL` |
| `orion:chat:gpt:message:log` | `chat.gpt.message.v1` | `ChatGptMessageSQL` |
| `orion:dream:log` | `dream.result.v1` (canonical), `dream.log` (legacy) | `Dream` |

**Dream persistence:** `dream.result.v1` payloads are validated as `DreamResultV1` and projected into `dreams`. Legacy `dream.log` + `DreamRequest` is still accepted and mapped into the same table (narrative from `context_text`). Extended telemetry lives under `metrics._dream_audit`.
| `orion:telemetry:biometrics` | `biometrics.telemetry` | `BiometricsTelemetry` |
| `orion:biometrics:summary` | `biometrics.summary.v1` | `BiometricsSummarySQL` |
| `orion:biometrics:induction` | `biometrics.induction.v1` | `BiometricsInductionSQL` |
| `orion:spark:telemetry` | `spark.telemetry` | `SparkTelemetrySQL` |
| `orion:vision:events:sql-write` | `vision.event.v1` | `VisionEventSQL` |
| `orion:autonomy:action:outcome` | `action.outcome.emit.v1` | `ActionOutcomeSQL` |
| `orion:vision:crops:sql-write` | `vision.crop.observation.v1` | `vision_crop_observation` (one row per crop, dedicated handler) |
| `orion:vision:unresolved:sql-write` | `vision.unresolved.v1` | `VisionUnresolvedSQL` (`vision_unresolved`) |

**Action outcome persistence:** `action.outcome.emit.v1` (produced by `orion-spark-concept-induction` after an autonomous readonly fetch) is projected into `action_outcomes` (PK `action_id`, idempotent upsert). `orion-cortex-exec` reads it back per-subject for chat-stance action feedback. DDL is applied on boot (`app/main.py` lifespan) and also lives in `services/orion-sql-db/manual_migration_action_outcomes_v1.sql`.


**Drive audit persistence — REMOVED 2026-08-13:** `orion-spark-concept-induction`'s `DriveEngine` (the sole producer of `memory.drives.audit.v1`) was deleted outright 2026-07-30 (`chore/delete-orion-drives`, PR #1486). The `drive_audits` table was dropped 2026-08-13 (snapshotted first, `/tmp/drive_audits_drop_2026-08-13/`) alongside the Hub Drives Analytics tab that read it (docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-analytics-tab-pr.md, which also removed the boot DDL). This service's write-path wiring for it (`DriveAuditSQL` model, `MODEL_MAP`/`INSERT_ONLY_MODELS`/route-map entries, the channel subscription, `DRIVE_AUDITS_RETENTION_DAYS` and its startup prune job) was fully untangled the same day (docs/superpowers/pr-reports/2026-08-13-untangle-drive-audit-sql-writer-pr.md) — an earlier scope note claiming `DriveAuditSQL` shared a `_JSONB` type declaration with other live models was checked and found wrong (every model declares its own private copy), so nothing blocked full removal. `scripts/drive_history_reflection_synthesis.py` and `scripts/analysis/measure_autonomy_gate.py` both still reference the concept but already degrade safely to "insufficient/missing data" — neither is on any cron/scheduler in this repo.

**Body organ history (2026-10-23, body economy Patch A3):** `physiology.state.v1` on `orion:physiology:state` (produced every ~120 s by orion-durable-runs' `physiology` node) -> `physiology_state` (`PhysiologyStateSQL`, PK `tick_id`, insert-only). One row per tick; `constraints` (jsonb) keeps every reading verbatim, `no_reading` ones included, so a missing reading stays missing. Table: `services/orion-sql-db/manual_migration_physiology_state_v1.sql` (create_all makes it too). The channel is force-subscribed in code (`effective_subscribe_channels`). ~720 rows/day; no retention policy yet.

**Field-channel anomaly scores (2026-10-13):** `field_channel_anomaly.score.v1` on `orion:field_channel:anomaly_score` (produced every 60s by `orion-field-digester`'s mood-arc anomaly scorer, flagged or not) is appended to `field_channel_anomaly_score` (`FieldChannelAnomalyScoreSQL`, PK `correlation_id`, insert-only). Created by `Base.metadata.create_all` at boot. Calibration data for the telemetry_anomaly trigger threshold: before this table only flagged windows survived. ~1,440 rows/day; no retention policy yet. The channel is force-subscribed in code (`effective_subscribe_channels`), so a stale `SQL_WRITER_SUBSCRIBE_CHANNELS` in a local `.env` still receives it.

**Transport baseline hourly readings (2026-09-29):** `transport_baseline.hourly.v1` on `orion:equilibrium:transport_baseline:hourly` (produced by `orion-equilibrium-service`'s per-hop transport baseline gate, one row per (service, instance, hop, UTC hour)) is appended to `transport_baseline_hourly` (`TransportBaselineHourlySQL`, PK `summary_id`, insert-only). Created by `Base.metadata.create_all` at boot (new table, no manual migration). This service must be deployed **before** equilibrium publishes, or rows land in the fallback log. Read by `scripts/analysis/grade_transport_baseline.py` (spec 2026-09-24 acceptance check 1). ~1,500 rows/day; no retention policy yet.

**Self-signal outcome association (2026-10-14):** `self_signal.hour.v1` on `orion:equilibrium:self_signal:hour` -> `self_signal_hour` (`SelfSignalHourSQL`, PK `hour_id`, upsert: a retried publish rewrites the same hour) and `self_signal.outcome_association.v1` on `orion:equilibrium:self_signal:outcome_association` -> `self_signal_outcome_association` (`SelfSignalOutcomeAssociationSQL`, PK `association_id`, insert-only), both produced by `orion-equilibrium-service`'s hourly rollup (spec 2026-10-07 section E). Created by `Base.metadata.create_all` at boot (new tables, no manual migration). Deploy this service **before** equilibrium. ~24 hour rows and ~500 association rows per day (`self_signal_hour`: no retention, tiny; associations: `SELF_SIGNAL_OUTCOME_ASSOCIATION_RETENTION_DAYS`, 30); read by orion-thought and equilibrium (each pair's last few hourly rows, consecutive-holds rule) and `scripts/analysis/self_signal_outcome_truth.py --persisted`.

**Juniper multimodal affect persistence (2026-08-25):** `orion:affectgpt:assessment` (produced by `orion-juniper-affective-state` after wrapping a real `orion-affectgpt-worker` AffectGPT read) is projected into `juniper_multimodal_affect_log` (`JuniperMultimodalAffectSQL`, PK `event_id` derived from the envelope `correlation_id` -- upserts on redelivery). First real persistence consumer of that channel; previously the only reader was the producer's own `scripts/tap_assessments.py` debug tap, and the live cognition path (`orion/situational/juniper_affect_state.py`, PR #1865) reads a separate 1h-TTL Redis SETEX mirror -- once that key expires, nothing durable recorded a capture ever happened. Deliberately does NOT persist `transcript` (Whisper's verbatim transcription of Juniper's spoken words) -- a privacy-boundary decision documented on the model class, mechanically enforced by `_write_row()`'s column-filter rather than bespoke redaction code. `raw_response` (the model's own generated affect read, already surfaced in the Hub UI and on the bus) is kept in full. **2026-08-26:** added `chat_correlation_id` (boot-time `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, same convention as the `chat_*` statements) for Hub's per-chat-turn affect bracket -- the `chat_turn_pre`/`chat_turn_post` pair fired around one Orion-mode turn. It is a *separate join axis* from `correlation_id`: that one joins a single capture's retina/worker/event legs, this one joins a capture to the conversation turn that caused it and joins a turn's pre/post pair to each other. Indexed, NULL for `manual`/`ambient` captures. Without the column, `_write_row`'s column-filter would have silently dropped the key and the join would have existed only on the bus and in the 1h Redis mirror -- never durably.

### Environment Variables
Provenance: repo root `.env` (mesh globals: `ORION_BUS_URL`, `PROJECT`, `NET`, …) → service `.env_example` → `docker-compose.yml` → `settings.py`

**Compose (from repo root):**
```bash
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build
```

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `SQL_WRITER_SUBSCRIBE_CHANNELS` | (See above) | List of channels to subscribe to. |
| `SQL_WRITER_ROUTE_MAP_JSON` | (See above) | JSON mapping of `kind` → `ModelName`. |
| `SPARK_LEGACY_MODE` | `accept` | Legacy Spark handling: `accept`, `warn`, `drop`. |
| `SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL` | `false` | If `true`, append `orion:spark:state:snapshot` to subscriptions. |
| `POSTGRES_URI` | ... | Database connection string. |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |

### Deprecation Controls
`SPARK_LEGACY_MODE` controls how legacy Spark kinds are handled:
- `accept`: write legacy kinds normally (default).
- `warn`: write legacy kinds and emit a deprecation warning log.
- `drop`: skip legacy writes and emit a warning log.

Example logs:
- `SPARK_LEGACY_DEPRECATED kind=spark.introspection.log mode=warn action=accept_write`
- `SPARK_LEGACY_DEPRECATED kind=spark.introspection.log mode=drop action=skip_write`

`SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL` (default `false`) appends the snapshot channel
`orion:spark:state:snapshot` to subscriptions without altering existing lists.

Legacy spark introspection channels/kinds are disabled by default; you can re-add them via
`SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` if needed.

## Grammar/substrate table retention

`grammar_events`, `grammar_edges`, `grammar_atoms`, `substrate_organ_emissions`,
`grammar_traces`, `substrate_proposal_frames`, `orion_biometrics_cluster`,
`power_intent_settled`, `substrate_attention_schema` and `substrate_durable_run_state` all get bounded retention
(`*_RETENTION_DAYS` env keys, **default 3** for the grammar tables, **10** for
`substrate_proposal_frames`, **30/90/90** for the last three). Each is a batched `DELETE ... LIMIT batch_size` loop
(`GRAMMAR_EVENTS_RETENTION_BATCH_SIZE`/`_MAX_BATCHES_PER_STARTUP`/`_MAX_ELAPSED_SEC`,
shared across all seven tables), capped so a huge backlog can't turn a pass into an
unbounded operation. See `app/grammar_truth.py`'s `_apply_bounded_table_retention()`.

Retention runs on a **60-second timer** (`GRAMMAR_RETENTION_INTERVAL_SEC`,
`app/grammar_retention_loop.py`), and **that is now the only retention path**. It used to be
startup-only, which deleted ~365,000 rows per process start against 1,117,440 rows/day of
arrival -- it could not converge against ANY window, and logged a growing `remaining_debt`
nobody read. The timer was added first and the startup pass left in place alongside it;
**the startup pass was removed 2026-08-20**, for three measured reasons:

1. It ran synchronously on the event loop, **ahead of `svc.start()`**, so it delayed not just
   readiness but the bus subscription. Retention starting a minute late is a far cheaper
   failure. The ~260s figure is carried forward from the 2026-08-20 periodic-retention PR
   report ("observed ~260s across four tables"), not re-measured. What is measured on this
   branch: boot is now **4.6s**, container start to "Application startup complete".
2. It could not converge anyway. That is the whole reason the timer exists.
3. It had drifted out of step: `main.py` hand-listed four tables while
   `GRAMMAR_RETENTION_TABLES` had six, so `grammar_traces` and `substrate_proposal_frames`
   were startup-exempt purely by omission, and the divergence had to be pinned by a test
   rather than being impossible. A test now asserts **no** table has a startup block.

Three caps bound a cycle. `GRAMMAR_RETENTION_PERIODIC_MAX_BATCHES` (3) and
`_MAX_ELAPSED_SEC` (20) bound **one table**; `_MAX_CYCLE_SEC` (45) bounds **the whole
cycle**. That third one was a genuinely missing bound -- seven tables x a 20s per-table cap is
a 140s cycle on a 60s timer, with nothing anywhere saying so.

The cycle budget is split as a **fair share** (remaining budget / remaining eligible tables),
not first-come, because `GRAMMAR_RETENTION_TABLES` has a fixed order: whichever table is
earliest **and** has debt would eat a first-come budget every cycle while the ones behind it
silently never ran -- which reads as healthy in the logs, because the first table's own
numbers look fine. Not hypothetical: `substrate_proposal_frames` is **last** in the tuple and
was measured at **8.21s with 109,585 rows of debt** during its own backfill, in a cycle where
`grammar_events` took 1.08–3.33s and the other four took 0.01–0.30s. (Do not restate this as
"only table X has debt" -- debt is transient, and an earlier draft of this paragraph said
exactly that and was wrong within the hour.) Fair share rather than rotation
because the order is a correctness constraint: `grammar_traces` must be pruned after the
children that reference it. A table skipped for lack of budget logs at WARNING; a table
dropped silently is indistinguishable from a table with no debt.

**The batch cap was deliberately not raised.** Measured live 2026-08-20, `grammar_events`
stops on the *batch* cap, using **1.08–3.33s** of its per-table budget across three
consecutive cycles, and drains its 4.13M-row backlog in **~31 hours** net of arrival. (A
naive "4.13M ÷ 3,000 rows ÷ 1,440 cycles" gives ~26h; the loop sleeps 60s *after* each cycle,
so the real period is ~68s, and ~795k rows/day arrive into that table meanwhile.) Raising the cap to 10 would cut that at 3.3x the delete I/O -- on a
host already I/O-stalled ~22% of wall time, and where this same pass was measured driving
stall to ~21%. Converging in a day is not worth buying with that. Raise it only with an I/O
measurement in hand -- **and raise `_MAX_CYCLE_SEC` with it**: at ~1.11s/batch, 10 batches
needs ~11.1s, which is above the ~6.43s fair share a 45s budget gives the first of seven
tables (it was 7.5s at six; adding orion_biometrics_cluster on 2026-08-28 cut every
table's share by 14%),
so the batch increase would be silently clipped. `effective_max_elapsed_sec` on
`/grammar/truth` reports the cap a run was actually handed, which is the field to read next
to `capped_by_elapsed_limit`.

**One boot-time caveat:** for the first cycle interval after a restart (~68s measured) all
seven tables report `*_retention_not_run` and `/health` reads `degraded`. That is expected and
was widened from two tables to six by removing the startup pass, then to seven when
orion_biometrics_cluster was added. There is no `healthcheck`
block in this service's compose file, so nothing restarts on it.

`grammar_events` and `grammar_traces` additionally respect a **cursor floor**: retention
refuses to delete rows any of the five reducer lanes still has unconsumed below the cutoff
(`_grammar_events_cursor_floor`). When it binds, `/health` reports
`cursor_floor_applied` and `remaining_debt` is measured against the retention window
rather than the clamped cutoff -- so "a reducer is stuck" and "retention is caught up"
cannot look the same.

`grammar_traces` is the parent row the Grammar Atlas lists, and is deliberately covered by
the periodic loop only, not by `main.py`'s startup pass. It was added last (2026-08-20):
until then its children were pruned at 3 days while the trace rows lived forever, so 42%
of traces expanded into empty graphs.

`substrate_proposal_frames` (2026-08-20) is the first stage of the substrate pipeline
(proposal -> policy decision -> execution dispatch -> feedback) and had **no bound at all**:
474,230 rows / 1,758 MB live, oldest 2026-07-23, growing ~27k rows and ~105 MB a day. It
uses its own floor, `_substrate_chain_floor`, not the grammar cursor floor -- three later
stages can reach back into it by `frame_id`, so the floor asks each stage's **pending
marker** (`policy_pending` / `dispatch_pending` / `feedback_pending`, from
`manual_migration_substrate_pending_markers.sql`) for the oldest row still owed work.
Anything at or above that timestamp survives regardless of age.

Two details in that floor are load-bearing and were both wrong in the first draft:

* **Every probe resolves back to the PROPOSAL row**, by joining through
  `source_proposal_frame_id` -- not to the pending row's own timestamp. A pending row in a
  downstream table is always *newer* than the proposal it needs (it is a child, written
  later), so flooring at the child's timestamp leaves the parent below the floor and
  deletable, which is backwards from the safety being claimed. Live: dispatch rows are
  created a mean 123.2s and max 920.2s after their parent proposal; 0 of 55,573 before it.
* **`OFFSET 0` fences the min-aggregate.** Written plainly, Postgres rewrites
  `SELECT MIN(created_at) ... WHERE dispatch_pending` into `ORDER BY created_at LIMIT 1`
  over the *full* `created_at` index with the marker as a filter. Pending rows are always
  the newest rows, so the ascending scan discards the whole table first: measured live at
  490 ms / 102,343 buffers, every 60 seconds. With the fence: 0.18 / 5.4 / 1.6 ms. Note the
  cost was *highest when the pipeline was caught up* and would have fallen as a backlog
  developed, so it would never have looked like a backlog problem.

The floor's distinction from the grammar cursor floor is deliberate and the two are not
interchangeable. A reduction cursor stops moving when its *source* goes quiet, which is
indistinguishable from a stall, so the grammar floor asks "are there unconsumed rows below
the cutoff" instead. A pending marker has much less ambiguity -- it is set at insert and
cleared in the same transaction as the downstream write. The same migration's header records
that a *time* bound was tried for this pipeline and reverted: the dispatch->feedback hop
legitimately ran p50 34.6 hours and max 11.3 days behind. Like every other managed table, this one is
pruned only by the periodic loop; `main.py` has no startup retention pass any more (see
above).

**The window is 10 days, and the reason is consumer windows, not disk.** Two live readers
reach back further than a week: `orion/autonomy/evals/run_attention_bound_proposal_eval.py`
uses `WINDOW_DAYS = 7`, and `scripts/analysis/measure_proposal_feedback_correlation.py`
defaults to 200 hours (8.33 days). A 7-day retention window would tie exactly to the first
and sit *inside* the second, racing both at their oldest edge -- and both degrade quietly to
"insufficient data" rather than failing loudly. 10 days clears the longer of the two with a
~1.7-day margin. Live: 10 days keeps 202,276 of 474,861 rows; 7 days would have kept 113,060.

`scripts/analysis/measure_proposal_feedback_correlation.py`'s chain-completeness check was
bounded to its window in the same patch. It previously joined **every** feedback frame ever
written against the upstream tables with no `WHERE` clause at all, which was correct only
while those tables were unbounded; under retention it would have reported `INCOMPLETE`
forever for the entirely expected reason that pruned rows do not resolve. It now reports the
excluded count explicitly, so a shrinking denominator cannot be mistaken for a clean result.

**Two hazards worth knowing about before extending any of this:**

1. **`_verify_delete_safe` does not mean what it says here.** It checks `pg_constraint` for
   incoming foreign keys and reports
   `substrate_proposal_frames_delete_is_child_safe; no incoming FK constraints found`. That
   check was written for the grammar tables, which genuinely have no children. This table
   has three referencing tables -- they store `source_proposal_frame_id` as a plain `text`
   column with **no FK declared** -- so "no incoming FK" here means the database will not
   *tell* you when you orphan something, not that nothing can be orphaned. The chain floor
   is what provides the actual safety; the FK check is not a second opinion.
2. **`reconcile_policy_pending` is a third writer of the marker.**
   `services/orion-policy-runtime/app/store.py` re-sets `policy_pending = true`, on a 900s
   timer, for any proposal with no matching policy decision frame, with **no time bound**.
   It can only ever add work, so today it is harmless. It becomes a trap the moment
   `substrate_policy_decision_frames` gets retention of its own: pruning a decision frame
   makes its proposal look unprocessed, reconcile re-flags it, the chain floor pins at the
   oldest re-flagged row, and proposal retention stops permanently while policy-runtime
   reprocesses hundreds of thousands of ancient proposals. Bound reconcile's anti-join
   first.

**Retention does not reclaim disk.** `DELETE` returns space for reuse inside the relation,
not to the OS. `substrate_proposal_frames` stays ~1.76 GB on disk after pruning; 84% of that
is TOAST (1,471 MB of 1,760 MB), with only 145 MB of heap and 144 MB of indexes. Recovering
it would need `VACUUM FULL` (exclusive lock) or `pg_repack` (not in the image), and neither
is worth it -- the volume has 619 GB free. **The win here is bounding future growth, not
reclaiming what is already allocated.**

**Autovacuum:** these tables now carry per-table autovacuum settings
(`services/orion-sql-db/manual_migration_autovacuum_high_churn_tables.sql`). The cluster
default `autovacuum_vacuum_scale_factor = 0.2` is a proportional trigger, which on a large
churning table means an enormous absolute one -- `grammar_events` had to reach ~1.18M dead
tuples before autovacuum would look at it. Continuous 60-second retention DELETEs make that
worse, not better. The override is `scale_factor 0.05 + threshold 3000`, a **2.3x-4.0x**
increase in vacuum frequency measured against live row counts.

An earlier draft used `0.01 + 10000` (~17x on `grammar_events`) and justified it as "same
total work, spread out". That was wrong: heap work scales with dead tuples, but **index**
vacuum cost scales with index *size*, not dead-tuple count, and these tables carry
3442 MB / 2756 MB / 1184 MB of indexes. 17x passes means ~17x the index-scan I/O, on a host
already I/O-stalled ~22% of wall time. ~3.5x is a deliberate compromise, not a measured
optimum -- if dead tuples still climb, measure the I/O cost before lowering it further.
Vacuum cost limits are deliberately left alone for the same reason.

The four substrate tables additionally get `toast.autovacuum_*` settings. TOAST relations
have their own autovacuum parameters and inherit **nothing** from the main table, and all
four were on `reloptions = NULL`. Since 84% of `substrate_proposal_frames`' bytes are TOAST,
omitting these would have skipped the very relation the file exists to fix. (TOAST relations
are never ANALYZEd, so only the vacuum parameters apply. `toast.*` settings also do not show
up in the *main* table's `reloptions` -- verify them by joining to `reltoastrelid`.)

This depends entirely on each table having a `(created_at, <id>)` index --
`services/orion-sql-db/manual_migration_grammar_atlas.sql` declares
`idx_grammar_events_created_at` / `idx_grammar_edges_created_at` /
`idx_grammar_atoms_created_at` (`substrate_organ_emissions` already had
`idx_substrate_organ_emissions_created` from its original table definition). Without
these, the retention DELETE's `ORDER BY created_at ... LIMIT` has no efficient access
path and forces a full scan on every run -- confirmed live 2026-08-19:
`grammar_events` retention had been silently failing on every single startup with
`psycopg2.errors.QueryCanceled: canceling statement due to statement timeout` since
the table grew past whatever size made that scan exceed
`SQL_WRITER_GRAMMAR_STATEMENT_TIMEOUT_MS` (10s) -- `n_tup_del` on that table was 585
lifetime against 8.58M inserts, i.e. retention had essentially never actually pruned
anything despite being "enabled" the whole time.

Applying the new indexes against a small/fresh database, the plain
`create index if not exists` in the migration file (applied normally) is fine. Against
a large, already-live database, add `CONCURRENTLY` and apply each index as its own
statement (not inside the batch file, since `CONCURRENTLY` cannot run inside an
implicit/explicit multi-statement transaction):

```bash
docker exec -e PGPASSWORD=postgres <sql-db-container> psql -U postgres -d <db> -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_grammar_events_created_at ON grammar_events(created_at, event_id);"
docker exec -e PGPASSWORD=postgres <sql-db-container> psql -U postgres -d <db> -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_grammar_edges_created_at ON grammar_edges(created_at, edge_id);"
docker exec -e PGPASSWORD=postgres <sql-db-container> psql -U postgres -d <db> -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_grammar_atoms_created_at ON grammar_atoms(created_at, atom_id);"
```

`build_grammar_truth_snapshot()` (`app/grammar_truth.py`) reports live status for all
seven tables' retention under `grammar_retention`/`other_table_retention`, and flags
`degraded_reasons` if an index is missing or a retention run failed/never ran.

**Known gap (updated 2026-08-20):** `grammar_traces` is now covered -- see above. Still
unbounded: `grammar_temporal_hops`, `grammar_compactions`, `grammar_projections`, and the
rest of the substrate frame family. Deleting a `grammar_events`/`grammar_atoms`/
`grammar_edges` row doesn't touch these, so they keep accumulating.

Measured live 2026-08-20, the substrate frame family alone is ~15 GB and only two of its
tables have any bound at all:

```text
substrate_organ_emissions              5,248 MB   bounded
substrate_execution_dispatch_frames    2,126 MB   UNBOUNDED
substrate_proposal_frames              1,758 MB   bounded (this patch)
substrate_field_state                  1,745 MB   UNBOUNDED
substrate_feedback_frames              1,641 MB   UNBOUNDED
substrate_policy_decision_frames       1,485 MB   UNBOUNDED
substrate_attention_frames             1,055 MB   UNBOUNDED
substrate_perception_embedding_baseline  477 MB   UNBOUNDED
```

Extending retention to the rest is mostly mechanical now that `_substrate_chain_floor`
exists -- the same pending markers cover the policy and dispatch stages. It is deliberately
NOT done here: that is ~8.3 GB of substrate history, which is the cognition substrate's own
record of what it proposed, decided and did, and deleting it is Juniper's call to make
explicitly rather than a side effect of a retention patch.

## Walkway camera reducers

### Deploy checklist (walkway camera)

The one place the vision-host, vision-council and this README point to.

1. **Apply the migration first:**
   `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_walkway_camera_v1.sql`.
   It is idempotent and upgrades tables made from an earlier draft. It also
   adds `vision_events.stream_id`; this writer re-tries that at boot in its
   own transaction, and until the column exists it writes vision events
   without it (logged at ERROR) -- room readers then ignore those rows.
2. **Rebuild together** (they share `orion/schemas/vision.py`): orion-vision-
   host, orion-vision-window, orion-vision-council, **orion-vision-scribe**,
   orion-sql-writer, orion-hub, orion-thought, orion-cortex-exec. A scribe on
   old code re-parses `VisionEventPayload` and drops `stream_id`; the room
   readers refuse NULL-stream rows written after
   `ORION_VISION_EVENTS_LEGACY_CUTOFF` (default 2026-09-24T00:00:00Z), so that
   fails closed (no percept) rather than leaking the walkway.
3. **Changing `config/vision_zones.yaml` requires rebuilding all three images
   that bake it in:** orion-vision-host (which boxes may be embedded and
   thumbnailed), orion-vision-council (what an unresolved percept may say),
   and orion-sql-writer (defense-in-depth recompute, `zone_no_embed`, presence
   rows). A partial rebuild leaves them disagreeing about where the patio is.
4. orion-hub mounts `/mnt/telemetry/orion-vision-host/crop_thumbs` read-only.
5. **v2 (2026-10-15):** apply
   `services/orion-sql-db/manual_migration_walkway_privacy_v2.sql` (drops the
   no-embed CHECKs, adds face + stationary columns) BEFORE deploying this
   orion-sql-writer (its crop INSERT names the new columns), and deploy
   orion-sql-writer before orion-vision-window: `VisionCropV1` is
   `extra="forbid"`, so an old writer rejects any crop observation that holds a
   face (the window leaves out face keys that are empty).

Spec: `docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md`
(ideas 1, 2, 3-opener, 6-score, 8-writer, 9). Tables come from
`services/orion-sql-db/manual_migration_walkway_camera_v1.sql`, which must be
applied by hand; until it is, both loops log `migration not applied` and back
off (doubling up to an hour) instead of crash-looping, and crop writes land
in `bus_fallback_log`. The ORM for `vision_unresolved` sits on its own
declarative base so boot-time `create_all` can never create these tables
without the migration's constraints.

**Crop persistence.** `vision.crop.observation.v1` fans out to one
`vision_crop_observation` row per crop (`crop_id = <observation_id>:<index>`,
idempotent). Before insert, each box's zone is recomputed from
`config/vision_zones.yaml` (copied into the image), and any box whose declared
OR recomputed zone is a no-embed zone loses its embedding, thumbnail and face
fields, and the row gets `zone_no_embed = true` (from the config, never from
the zone's name). Since 2026-10-15 no zone is no-embed (the patio is
fingerprinted) and v2 dropped the table CHECKs that enforced it. Face fields
(`face_embedding`, `face_score`, `face_thumb_ref`) are stored as sent.
Since 2026-10-16 (`manual_migration_vision_osnet_shadow_v1.sql`) each row also
stores `osnet_embedding` (shadow OSNet person re-identification vector, person
crops only), `osnet_model_id` and `embedding_model_id` (the body model that
actually made `embedding`; SigLIP v1 today). The no-embed strip clears the
OSNet vector too. **Nothing in the individuals loop or the split reads the
OSNet columns** (`tests/test_vision_osnet_shadow.py` checks). They are scored
only by `evals/run_osnet_vs_siglip_eval.py --live <dsn>` (READ ONLY): same-frame
different-people pairs vs next-frame same-person pairs (from box geometry, not
from either model) and Juniper's picture verdicts, per model: cosine
quantiles, AUC, and how many different people pass at the line that keeps 90%
of same-person pairs. A winner is named only when a paired bootstrap puts the
95% interval of the let-through difference on one side of zero. The geometry
pairs are seconds apart, so they test short-term consistency and telling
side-by-side people apart, not matching one person across visits; only the
verdict pairs speak to that. It reports `INSUFFICIENT DATA` (exit 2) until about
a day of crops carry both vectors. Retention follows the crop rows (7 days).

**Individuals loop** (`app/vision_individuals.py`, every
`VISION_INDIVIDUALS_INTERVAL_SEC`, default 60 s). Reads new crops per stream
since `vision_individuals_cursor`, which tracks LANDING time
(`vision_crop_observation.created_at`), so a crop that arrives late (old
`observed_at`) is still processed; within a batch crops are clustered in
`observed_at` order, and a late crop outside the latest sighting becomes its
own earlier sighting; a crop with a face first tries the nearest same-kind
FACE centroid (`VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD`, 0.60; that match wins
over appearance). Otherwise the body has two lanes (2026-10-16,
`docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md`):
**continuation** -- among individuals whose last box is within
`VISION_INDIVIDUALS_CONTINUATION_MAX_DISP` (1.5) box heights and was seen
within `VISION_INDIVIDUALS_CONTINUATION_GAP_SEC` (10 s), the best joins at
body cosine >= `VISION_INDIVIDUALS_CONTINUATION_THRESHOLD` (0.88); then
**re-identification** -- the nearest same-kind centroid anywhere at cosine >=
`VISION_INDIVIDUALS_MATCH_THRESHOLD` (0.90); else a new `vision_individual`
opens (centroid = renormalized running mean; two crops from one frame never
join the same individual). Each batch logs
`[VISION_MATCH] stream=<s> face=<n> continuation=<n> reid=<n> new=<n>`, and
warns `fragmenting` above 1,500 new person individuals in 24 h (about 500 a
day is expected). Rollback to the old single body rule, exactly:
`VISION_INDIVIDUALS_CONTINUATION_ENABLED=false` and
`VISION_INDIVIDUALS_MATCH_THRESHOLD=0.80`. Replay eval:
`evals/run_vision_body_match_replay_eval.py`. Named individuals built under
the old rule carry a mixed body centroid; `scripts/reanchor_vision_individuals.py`
rebuilds it from their face crops (dry run by default). Observations no more than
`VISION_INDIVIDUALS_MERGE_GAP_SEC` apart form one `vision_individual_sighting`
(zone = most observations, tracked in `zone_counts`; dwell = ended - started).
Every touched sighting is scored (`app/vision_attention_score.py`: unknown,
unusual time from its own hour history, long dwell vs the zone's
`dwell_rare_sec`, few prior sightings; weights in one dict; null components
count 0 and are stored as null). At or above `VISION_ATTENTION_THRESHOLD` it
writes one `vision_events` row `attention_worthy` per sighting. Every zone with
`presence: true` (or `embed: false`) keeps one `substrate_embodied_presence`
row, `<stream>:<zone name>` (e.g. `walkway:patio`; `subject = {"count": n}`,
`head_count_zone: true` and, for the situation brief which still filters on
it, `no_embed_zone: true`), only while the camera is producing census rows.
**Stationary fixtures:** an individual whose box keeps IoU >=
`VISION_STATIONARY_IOU` (0.5) with its run's first box, gaps <=
`VISION_STATIONARY_MAX_GAP_SEC` (300), for `VISION_STATIONARY_MIN_SEC` (1800)
is `stationary` (individual + current sighting): a prop such as the patio
skeleton seen live 2026-10-15. Stationary sightings are kept but get no
attention score/event, do not count toward asks or rhythm, and are left out
of the zone head count. The first box that moves clears it; while stationary
a gap does not. Then: Juniper's answers are applied (`orion_ask`
answered -> `vision_individual.label`, `applied_at` set), open asks past
`expires_at` expire, and new asks open for unlabeled individuals with
>= `VISION_ASK_MIN_SIGHTINGS` sightings over >= `VISION_ASK_MIN_DAYS` local
days, under `ORION_ASK_DAILY_CAP`, which counts only `vision_individual` asks
(since 2026-10-06; memory confirmation cards in the same table have their own
caps in orion-memory-consolidation) from `orion_ask.created_at` since local
midnight (restart-proof). Opened asks are rows only (no bus event);
the Hub's ask card reads `orion_ask`.

**Visual confirmation asks** (2026-10-15, Juniper: "have Orion ask to confirm
visual things to help them learn"). Two more kinds of ask, opened first each
tick under their own cap `VISION_CONFIRM_ASK_DAILY_CAP` (6; counted by
`source_kind='vision_same_individual'` or the `rule:face` evidence ref, so
`ORION_ASK_DAILY_CAP` stays the 10-sightings rule's budget). No ask of any
kind opens without a picture: a candidate with no thumbnail is skipped, stays
a candidate, and does not use up a slot. The kinds:

1. *"Are these the same person?"* (`source_kind=vision_same_individual`,
   `source_ref=<id_a>|<id_b>`, sorted). Two unlabeled-or-half-labeled people,
   each with >= `VISION_ASK_FACE_MIN_FACES` (5) face crops, whose face
   centroids are >= `VISION_SAME_ASK_MIN_FACE_COSINE` (0.40) alike. Opened only
   with both faces (`image_refs`, two thumbs). People Juniper already said are
   one person count as one (no "X vs A" and "X vs B"). Answer is `yes` / `no`
   (the Hub refuses anything else; Not sure = dismiss, and the pair may come
   back after `VISION_ASK_COOLDOWN_DAYS`). Each answer writes one
   `vision_individual_pair_verdict` row with the face similarity at that
   moment: `same` links the two (no merge, nothing deleted) and carries a name
   to every unnamed member of the confirmed group (transitively) when exactly one has one; `different` keeps the pair from being asked
   again. Those rows are the labeled data for calibrating
   `VISION_INDIVIDUALS_FACE_MATCH_THRESHOLD` (`scripts/report_vision_individuals.py`
   prints them). The 0.40 band comes from live faces on 2026-10-15: two faces
   in one frame (known different people) median 0.27, p75 0.41; the same face
   a few seconds apart median 0.65, p10 0.39.
2. *"Who is this?" from a face* (`source_kind=vision_individual`, evidence
   ref `rule:face`). A person with >= `VISION_ASK_FACE_MIN_FACES` face crops
   is asked about the same day (no 5-day rule), with up to three pictures,
   face first. Skipped while the person is in an open "same person?" ask, and
   once per group of people Juniper said are the same (her answer names them
   all). `VISION_ASK_FACE_MIN_FACES=0` turns both kinds off.

**Names on the card** (2026-10-15, Juniper). A "same person?" answer can carry
names (`orion_ask.answer_detail = {"names": {id: name}}`: one name for both on
Same person, one per face on Different people). Each is applied exactly like a
"who is this?" answer for that individual: a *typed* label
(`vision_individual.label_copied=false`), then copied to everyone confirmed the
same person (`label_copied=true`), then any still-open "who is this?" card for
those people is closed (`answer_detail={"resolved_by": <ask_id>}`). Rules: the
last typed name wins (replacing a typed name is logged as
`[VISION_LABEL_REPLACED]`); a typed name replaces a copied one; a copied name
never replaces a typed one. Answers apply oldest first, so "last" means answer
time. A multi-picture "who is this?" answer can carry one verdict per picture
(`{"pictures": [{image_ref, index, verdict: same|different|unsure, name}]}`);
the individual still gets the first (face) picture's name, and every picture's
verdict lands in `vision_ask_picture_verdict` (with the `crop_id` behind the
thumbnail when it still exists). A 'different' picture splits the individual
(next section). A "who is this?" card closed because its person only
got a *copied* name is closed here (no bus event), so the substrate gets no
entity for that person until Juniper types a name for them. If a sql-writer
built before these names marked a named pair answer applied (Hub deployed
first), the next tick finds names whose individual's label is older than the
answer and applies them (`[VISION_PAIR_NAMES_HEALED]`).

**Same-name link** (2026-10-20; story cards only before that). Every typed
name -- a "who is this?" answer, a walkway story name, a name box on a "same
person?" card -- also adds a `'same'` pair verdict to an existing individual
(same stream and kind) Juniper already TYPED that name for (a copied name does
not count), case-insensitive, earliest `labeled_at` first. It is skipped when
the two are already linked, have any pair verdict, or when linking would join
two groups she said are 'different' (a cannot-link through anyone already
confirmed the same). Idempotent (`ON CONFLICT DO NOTHING`), logged as
`[VISION_SAME_NAME_LINK]`. Correcting a typed name to a different one (not
just its case) first drops the 'same' links the old name made -- rows involving
that individual written by a `vision_individual` ask, never her own "same
person?" answers (`[VISION_SAME_NAME_UNLINK]`); there is no card to undo an
automatic link, so the rename is the undo. Applies to answers applied from 2026-10-20 on; older
applied answers are not re-linked.

**Answers from a walkway story card** (2026-10-18). The Hub's "Walkway stories"
card lets Juniper name, or correct, each person in a story. It writes the
`orion_ask` row already answered (`applied_at` NULL), and this loop applies it:
a typed name is a `vision_individual` ask with `answer_detail.story_id`, applied
exactly as above (typed label, `label_ask_id` = that ask, last typed name wins),
including the same-name link below. "Someone else" / "not sure" is a
`vision_story_person` ask: only its `answer_detail.pictures` (one per story
crop, with `crop_id`) land in `vision_ask_picture_verdict`; 'different' ones are
split by the pass below (a split that would move most of the individual's
crops is refused, `majority`, as for any card). It never names the individual
and never stops the opener from asking "who is this?" about them. Story names
are not counted against the opener's daily cap (`ASKED_TODAY_SQL` skips rows
with `answer_detail.story_id`).

**Splits on "someone else" pictures** (`app/vision_split.py`, 2026-10-16;
design `docs/superpowers/specs/2026-10-16-vision-individual-split-proposal.md`).
Every crop the clusterer assigns now stores its individual
(`vision_crop_observation.individual_id`, same transaction as the batch; older
crops stay NULL, no backfill). After answers are applied, each 'different'
picture group (one per typed name, case-insensitive; unnamed ones together) of
an answered card becomes one split, automatically, since Juniper's answer is
the confirmation:

- The named crops move to a NEW individual, plus member FACE crops whose face
  cosine to the moved picture's face is >= `VISION_SPLIT_FOLLOW_MIN_FACE_COSINE`
  (0.60) and beats the confirmed picture's face by `VISION_SPLIT_FOLLOW_MARGIN`
  (0.10). Body-only crops never follow. A crop with no stored membership
  (older than the migration) still moves when a verdict names it, but never
  follows. Moving more than `VISION_SPLIT_MAX_MOVE_FRACTION` (0.5) of the
  individual is refused (`[VISION_SPLIT_REFUSED] reason=majority`).
- The asked individual keeps its id, label and substrate entity node. Its
  centroid/counts are recomputed (exactly when all its crops are stored, else
  the moved crops are subtracted from the running mean); its sightings lose the
  moved crops' counts, zones and pictures, so an ask card never shows the other
  person under its name. The new individual gets its own sightings (no
  attention score: `_basis: split`).
- Links: always a 'different' pair verdict (cannot-link) between the two; with a
  typed name, the new individual gets it as a typed label plus a 'same' link to
  an existing individual Juniper typed that name for (no crop merge). The new
  individual gets no substrate entity node from the split.
- The matcher honours cannot-links: a crop whose best face match X is within
  `VISION_INDIVIDUALS_CANNOT_LINK_MARGIN` (0.10) of an individual Juniper said
  is different from X does not join X by face; it falls through to the body rule
  (`face_blocked_by_cannot_link` in the tick summary).
- Every split, refusal and skip (crop pruned or already moved) is one
  `vision_individual_split` row (idempotent per individual + ask + name), with
  the exact before/after. Trace: `[VISION_SPLIT]`. Undo:
  `python3 scripts/undo_vision_split.py <split_id>` (dry run, rolled back) then
  `--apply`; it refuses when crops joined the new individual since.
- `VISION_SPLIT_ON_VERDICT_ENABLED=false` turns splits and the cannot-link rule
  off (verdicts are recorded only, as before). Replay eval (read-only on live):
  `evals/run_vision_split_replay_eval.py --live <dsn>`.

Needs `services/orion-sql-db/manual_migration_vision_confirm_asks_v1.sql`
(`orion_ask.image_refs`, `vision_individual_pair_verdict`),
`manual_migration_vision_ask_names_v1.sql` (`orion_ask.answer_detail`,
`vision_individual.label_copied`, `vision_ask_picture_verdict`) and
`manual_migration_vision_split_v1.sql` (`vision_crop_observation.individual_id`,
`vision_individual_split`); without them
the individuals loop backs off with a clear error. Replay eval (read-only on live):
`evals/run_vision_confirm_asks_replay_eval.py`. Retention: crops `VISION_CROP_RETENTION_DAYS` (7) except labels in
`VISION_CROP_RETENTION_KEEP_LABELS` (`person`: every person crop is kept, so Juniper's labels keep their examples),
sightings `VISION_SIGHTING_RETENTION_DAYS` (90); unlabeled individuals unseen
for the sighting retention are deleted, labeled ones stay.

**Person learning** (`app/vision_person_model.py`, inside the individuals
tick at most every `VISION_PERSON_MODEL_INTERVAL_SEC` (600), flag
`VISION_PERSON_MODEL_ENABLED`, on). Orion learns who people are from Juniper's
answers (design: `docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md`).
Each tick it:

1. Reconciles the album (`vision_person_exemplar`). A cluster Juniper named
   (typed: `cluster_label`; carried by a "same person" answer: `pair_same`)
   adds its crops as +1 for that name. A "different" picture adds its crop as
   -1 for the cluster's name (`picture_different`). A named cluster whose best
   two-way split of body vectors is less alike than the 10th percentile of
   same-sighting crop pairs (learned each tick) looks like two people and adds
   nothing until she answers. Rows whose answer no longer holds (a rename, a
   split) get `revoked_at`.
2. Refits the bars (`vision_person_bar`, per person and channel). `accept_at`
   is a conformal false-accept bound at `VISION_PERSON_FALSE_ACCEPT_ALPHA`
   (0.05) over the CLUSTER scores of everyone who is not that person: the
   street (every unnamed recent person cluster), other named people's
   clusters, and clusters she said are not them. Calibrating on the other
   named people alone let ~9% of street clusters through on live data
   (2026-10-20). The street is fitted on and guessed on, so about alpha of it
   clears any armed bar: alpha is split across people (alpha / people), and a
   bar is armed only when at least 3 of the person's own clusters, each held
   out with every album crop from its days, exist and half of them clear it.
   Below about 2 x people / alpha impostor clusters (~280 at 7 people) the
   bound is the street maximum: no unnamed cluster can be accepted yet.
   `reject_below` is the same rule on cross-day genuine crop scores (>= 19).
   NULL = ask only. On 2026-10-20 no person had an accept bar: everything is
   `unsure` until her answers grow the albums.
3. Guesses (`vision_person_attribution`): one row per unnamed, recently seen
   person cluster, its best person, band `accept` ("probably X") or `unsure`.
   Score = median over its newest 20 crops of the mean top-`VISION_PERSON_TOP_K`
   cosines to the album. Face when both sides have faces, else
   `VISION_PERSON_FALLBACK_CHANNEL` (body). Accepted for two people = unsure;
   a person ruled out by a -1 crop in the cluster is never guessed. A guess
   never writes `vision_individual.label`.

4. Asks *"Is this <name>?"* (`source_kind=vision_person_confirm`,
   `source_ref=<individual_id>|<name>`) about the strongest unsure guesses of
   individuals seen at least twice, under their own cap
   `VISION_PERSON_CONFIRM_ASK_DAILY_CAP` (6). Until 2026-10-22 they shared
   `ORION_ASK_DAILY_CAP` with the slow rule, which opens first and used both
   slots every day, so none were ever asked. Same rules as
   the other openers: no picture, no ask; a skipped candidate never uses up a
   slot; one open question per individual, both ways (the face and slow
   openers skip an individual with an open "Is this <name>?" card, and a name
   given anywhere closes it); a pair already answered or dismissed, in any
   casing, is not asked again. `yes` names the individual through the usual
   typed-name path (the album takes it as a named cluster on the next cycle;
   orion-substrate-runtime writes the entity); `no` adds its newest 20 crops
   (the ones guessed on) as `confirm_no` exemplars, kept across
   reconciliation, and revokes that guess; Can't tell = dismiss.
   **A card with several pictures is answered picture by picture**
   (2026-10-23: a cluster can hold more than one person, and "one of the three
   is Emerald" could not be said). `answer_detail.pictures` holds one
   `same` (is <name>) / `different` (is not) / `unsure` per picture. All
   `same` = the old yes, all `different` = the old no. Anything else never
   names the cluster: each marked picture's crop becomes its own exemplar
   (`confirm_pick_yes` +1 / `confirm_pick_no` -1, kept across
   reconciliation), "not sure" adds nothing, and the guess is revoked.

Needs `manual_migration_vision_person_album_v1.sql`; without it the step logs
`person_model: skipped` and the rest of the tick runs. Eval (read-only on
live; `--write` stores a `vision_person_model_eval` row):
`evals/run_person_recognition_eval.py --live <dsn>` -- leave-one-cluster-out,
gate = zero wrong accepts; a new model version must beat the last on an exact
McNemar test (p < 0.05).

**One record per person** (`app/person_record.py`, 2026-10-22, spec
`docs/superpowers/specs/2026-10-22-people-one-record-design.md`). Runs inside
the individuals tick, after answers apply, while `PERSON_RECORD_ENABLED` (true):

1. `person` / `person_link` follow the named clusters
   (`orion/person_model/record.py`): one person per folded name, joined to the
   `config/person_model/people.yaml` referent when the name matches. A
   "different people" verdict blocks a merge (logged `person_link_conflict`).
   Named bicycles and cars link as that person's belonging; other named things
   are ignored. Renames and cleared names revoke links.
2. Stated pronouns from people.yaml become `person_fact` rows.
3. Answered `person_fact` asks become `person_fact` rows (her words verbatim).
4. New `person_fact` asks ("How do you know X? For example: ...", then
   pronouns, age, gender, sexuality, ...; `orion/person_model/categories.yaml`)
   open under their own cap `PERSON_FACT_ASK_DAILY_CAP` (3), one open per
   person, least-asked person first. Pictures (2026-10-23): only up to 3 that
   Juniper herself tied to that person (`orion/person_model/pictures.py`: the
   picture she named on a "who is this?" card, ones she marked "the same
   person", a face she named on a "same person?" card, a picture she picked
   on an "Is this <name>?" card). Never a cluster's newest pictures: a cluster
   can hold more than one person, and the first live cards showed three
   different people. No confirmed picture, no picture. Every pass also
   corrects the pictures of questions already open.

Needs `manual_migration_person_record_v1.sql`; without it the step reports
`person_record: migration_missing`. Dry run on live data:
`orion/person_model/evals/run_person_record_backfill_eval.py --live <dsn>`.

**Rhythm loop** (`app/vision_rhythm.py`, every `VISION_RHYTHM_INTERVAL_SEC`,
default 900 s). Subjects are individuals and the labels in
`VISION_RHYTHM_LABELS` (arrivals from `vision_scene_inventory`: first positive
window after `VISION_RHYTHM_ARRIVAL_GAP_SEC` without one). Per
(stream, subject, weekday/weekend/any) it fits a circular kernel density over
local minute-of-day and emits `vision_percept_expectation` rows for the next
24 h only with >= 5 occurrences over >= 5 distinct days, each window itself
hit on >= 5 days. Closed windows are graded `met` / `missed` / `unscorable`
(no census frames in the window) and `met`/`missed` write `vision_events`
`arrived_as_expected` / `expected_absent` citing `expectation:<id>`. While a
window is open and unmet it sets Redis `orion:vision:expect:<stream>` (on
`ORION_BUS_URL`) with TTL = seconds to window end. That key is refreshed by
its own cheap loop every `VISION_EXPECT_REFRESH_INTERVAL_SEC` (default 60 s;
only the open-window query, no fitting), so steering starts within a minute
of a window opening rather than up to one rhythm tick late. Rhythm surprise stays in
this table; it is not wired to the substrate graph until live data passes the
metric gate.

**Street continuity loop** (`app/vision_handoff.py` rules, `app/vision_handoff_loop.py`
I/O, every `VISION_HANDOFF_INTERVAL_SEC`, default 15 s; design:
`docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md`). One passage
across the door and walkway cameras. Portals live in `config/vision_topology.yaml`
(override with env `VISION_TOPOLOGY_PATH`, read by the loader `orion/vision/topology.py`, not Settings): `central_walk`
(overlap, the cameras see the central walkway together), `approach_door` (a walkway
person moving toward the house expects a Ring motion / ring event or a doorbell story
3-30 s later), `door_to_walkway` / `walkway_to_door` (the sidewalk gap). Tables:
`manual_migration_vision_handoff_v1.sql` (`vision_handoff`, `vision_handoff_hypothesis`,
`vision_handoff_street_outbox`, `vision_handoff_cursor`); without it the loop backs off with a log line.
Each row is processed in its own savepoint: a row that raises is rolled back alone,
counted (`row_failed_<step>`), and settled (`unscorable`, or its open hypotheses `false`).
Companion / standing / parked / out-of-scope / prop stories never depart; props and
companions are never arrival candidates.

- **Open** from new motion stories (cursor on `created_at`). Most walkway stories end
  `inside` (the detector looks every ~5 s), so a departure is projected: moving toward
  the portal edge, last foot within 0.6 of it, a measured exit speed (frame widths/s).
  Window = `(exit_dist + g) / exit_speed` with `g` the prior x0.4..x3 until
  `VISION_HANDOFF_MIN_MET` met for that portal and kind, then the split-conformal 10-90%
  band of learned `g = transit_sec x exit_speed - exit_dist`. While a window is ahead it
  sets `orion:vision:expect:walkway` (extend-only; the rhythm loop keeps the key until
  `handoff_until`) or `orion:vision:look:doorbell` (= handoff id, TTL = window; the
  retina opens a capped live view). Stories land ~50 s after the walker left, so about
  half the windows are already past (`steer_too_late` in the tick summary).
- **Grade** once the target camera's motion cursor passed the late margin: `met` /
  `early` / `late` / `ambiguous` (two candidates, no clear appearance winner; never
  trains), `missed` only if the target camera delivered census frames for at least
  `VISION_HANDOFF_MIN_COVERAGE` of the window, else `unscorable` (the doorbell only
  looks during Ring bursts and self-looks). Appearance: story centroid cosine, face then
  body then OSNet. Writes `vision_events` `handoff_met` / `handoff_early` /
  `handoff_late` / `handoff_missed` / `overlap_pair`.
- **Misses** open five hypotheses (`detector_miss`, `late`, `turned_back`,
  `entered_place`, `unexplained`) with counted-outcome priors (start at 1 each), settled
  from data by deadline; `unexplained` writes a `vision_unresolved` row (reason
  `handoff_missed`, the curiosity seam) and the `handoff_missed` street event. A ring
  with no approach seen first is an `unannounced` row (walkway `detector_miss` or
  `out_of_view`); every ring gets a row (`announced` / `unannounced` / `unscorable` when
  the walkway was not looking). Hypotheses whose stories never caught up settle `false`
  with `never_caught_up`, never `unexplained`.
- **Street events** (`StreetEventV1` on `orion:vision:street:events`, outbox columns on
  `vision_handoff`, `VISION_HANDOFF_STREET_EVENTS_ENABLED`): `approach_door`,
  `handoff_met` / `handoff_late` for every arrival (with `arrival_log_score`, nats/s of the
  arrival time under the predicted distribution, for the Rung 6 `street_handoff` domain),
  `handoff_missed` once unexplained, `overlap_pair` (low salience). One outbox row per
  event; facts older than 1 h (approach: 5 min) are recorded but not published. The prior
  band's p is floored at `2/(VISION_HANDOFF_MIN_MET+1)`, so a hand-set prior never opens
  the attention gate. Salience = who x what x surprise (`SALIENCE_*` in
  `vision_handoff.py`). p-value calm state and the conformal floor `2/(n+1)` are
  documented in the module.
- **Asks**: a strong met / overlap pair opens a `vision_same_individual` ask
  (`VISION_HANDOFF_SAME_ASK_DAILY_CAP`, evidence `rule:handoff`). The handoff code never
  writes a label, a pair verdict or a merge. No "where did they go?" ask: `OrionAskV1`
  has no choices field.
- Replay (read-only, the metric gate): `PYTHONPATH=. python
  services/orion-sql-writer/evals/run_vision_handoff_replay_eval.py [--since ...]`.

Safety rails added after review: each cycle holds a Postgres advisory lock
(a second instance skips the tick); crops newer than
`VISION_INDIVIDUALS_SETTLE_SEC` wait a tick so out-of-order writes are not
skipped by the cursor; match candidates are individuals seen in
`VISION_INDIVIDUALS_CANDIDATE_DAYS` plus every labeled one (numpy matmul);
a failed crop write logs to `bus_fallback_log` with embeddings and boxes
redacted; if the zones file cannot load, every embedding is stripped (fail
closed); ask images are crop thumbnails (`thumb:<sha256>`, written by
orion-vision-host for embedded crops only and served by the Hub), never whole
frames. `VISION_CROP_THUMB_RETENTION_DAYS=0` (default since 2026-10-20; the
same key the host reads) keeps thumbnails forever, so any sighting's thumbnail
may be offered; a positive value limits asks to sightings recent enough that
the thumbnail outlives the ask (asks expire after 7 days); an
unanswered expired ask is not repeated for `VISION_ASK_COOLDOWN_DAYS`; a window
is `missed` only if census coverage (5 s windows bridged across gaps up to
30 s) is at least `VISION_RHYTHM_MIN_COVERAGE`, and an individual's window is
not graded until the individuals cursor has passed its end; the Redis expect
key is deleted as soon as everything open on the stream has arrived.

Every knob (all `VISION_*`, `ORION_ASK_*` keys) is listed with a comment in
`.env_example`. The zones file path can be overridden with `VISION_ZONES_PATH`
(read by `orion/vision/zones.py`, not Settings).

Report: `python3 scripts/report_vision_individuals.py [--stream walkway] [--json]`.

### Walkway motion stories (2026-10-16, Hub only)

`app/vision_motion.py` (pure) + `app/vision_motion_loop.py` (clocked, every
`VISION_MOTION_STORY_INTERVAL_SEC`, default 15 s). Turns each passing thing on
the walkway into one sentence from box maths -- no model, no embeddings, no
new identity. Design: `docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md`.

- Links `vision_crop_observation` boxes into passages by geometry (velocity
  prediction), not by `individual_id`; ids only supply a name, and only under
  the strict rule (every id on the passage carries the same name, that name is on
  >= 2/3 of the crops, no crop touching the frame edge).
- Boxes touching the left, right or bottom edge give no zone and no travel.
- Published = a moving story with >= `VISION_MOTION_STORY_MIN_DETECTIONS` (3)
  detections, enough of them unclipped. Everything else is stored with
  `published=false` and a `drop_reason` (glimpse, standing, parked, prop,
  edge_clipped, too_long). Fast classes (ran / ridden / drove) need >= 3
  unclipped detections.
- Props: stationary individuals, `VISION_MOTION_PROP_INDIVIDUAL_IDS` (default:
  the patio skeleton), and boxes overlapping their still/last boxes.
- Table `vision_motion_story` (+ `_cursor`, `_verdict`), migration
  `services/orion-sql-db/manual_migration_vision_motion_story_v1.sql`
  (auto-applies on pull). Stories are kept `VISION_MOTION_STORY_RETENTION_DAYS` (90),
  except a story that still has its picture while strips are kept forever (below).
- Late individual links (2026-10-20): a story freezes each crop's
  `individual_id` in `evidence` when it closes, often seconds before the
  individuals reducer assigns the crops (live: 101 of 147 person stories over
  5 days had no link). Each tick, `relink_stories` fills those nulls from
  `vision_crop_observation` for stories that ended in the last 6 h (null-only,
  never changes an id). The link then lives in the story row, so it outlives
  crop retention (7 days). The Hub resolves the same link at read time for
  the seconds before that. Trace: `[MOTION_STORY] relinked stories=.. crops=..`.
  One-time backfill of older stories: `scripts/backfill_story_evidence_links.py`
  (dry run by default; snapshots to `/tmp/` before `--apply`).
- Trace: `[MOTION_STORY] stream=walkway crops=.. closed=.. published=.. glimpses=..
  dropped_props=.. cursor=..` once per tick.
- **Hub only.** Nothing in Orion's cognition reads the table. The Hub's
  "Walkway stories" card (Vision panel, "Walkway (live)") shows them and
  records Juniper's verdicts; the gate to switch them on for Orion is >= 50
  checked, >= 80% fully right, zero wrong names.
- Slice 1 publishes walking-pace stories of people, dogs, strollers and walked
  bicycles only; vehicles and fast classes are stored as `out_of_scope`, and a
  dog/bicycle/stroller walking with a person is stored as `companion` (the
  person's story tells it), so one walk-by is one story to check.
- Known limits: a crop that lands more than `VISION_MOTION_STORY_SETTLE_SEC`
  after it was seen (a writer restart, a bus backlog) is behind the cursor and
  never joins a story. `name_label` and `sentence` are kept as written; the Hub
  re-derives the name when it shows a story (same rule,
  `orion/vision/story_naming.py`, on today's names), and Juniper's own answer
  on a story wins for that story. Retention never
  deletes a story Juniper has given a verdict.
- Kill switch: `VISION_MOTION_STORIES_ENABLED=false`.
- Replay (read only): `PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_motion_story_replay_eval.py`
  (clean frames from `VISION_MOTION_CLEAN_SINCE` by default; `--include-corrupt` for earlier;
  `--stream doorbell --frame-size WxH` to replay old crops under a frame size).
- Frame size and edges (2026-10-20, migration
  `services/orion-sql-db/manual_migration_vision_motion_story_edges_v1.sql`, auto-applies on pull;
  for the street-continuity handoff, `docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md`).
  The writer now keeps the detector's frame size on each crop
  (`vision_crop_observation.frame_width/height`). Every box is measured against its own frame:
  that size, else `VISION_MOTION_FRAME_SIZE_BY_STREAM` (`stream=WxH,...`), else
  `VISION_MOTION_FRAME_WIDTH/HEIGHT`. Before this, the doorbell was measured as a 2560x1920 frame,
  so its lateral direction was under-read and its right/bottom clipping never fired.
  Each story also records (NULL on older rows):
  - `entry_edge`, `exit_edge`: `left|right|top|bottom|inside`, from the first/last box. Left/right/bottom:
    the box side within `VISION_MOTION_EDGE_ZONE_FRAC` (0.05) of the frame (nearest wins). Top: only when
    the foot point is that close to the top (a head cut by the top edge is not an exit). Else `inside`.
  - `exit_foot_x`, `exit_foot_y`: last box's foot point, 0..1 of its frame.
  - `exit_speed_fw_per_sec`: lateral foot speed over the last <= 4 unclipped boxes, frame widths per second,
    unsigned (direction: `exit_edge` / `dir_lateral`); NULL with < 2 unclipped boxes.
  - `frame_w`, `frame_h`, `frame_size_source` (`detection|stream_setting|default`).
  `rules_version` stays 1: walkway stories (all verdicts so far) read exactly as before. Without the
  migration the loop logs `[MOTION_STORY] missing columns ...` each tick and writes the older columns.

### Walkway story pictures of the moment (2026-10-17)

`app/vision_story_strip.py`, run inside the motion loop when
`VISION_STORY_STRIPS_ENABLED=true` (default). Why here: the walkway's frames
live in `/mnt/telemetry/vision/frames` for only ~60 s (the edge prunes the
shared dir), and a story is written >= 41 s after its last box. sql-writer is
the one service that knows which crops become stories, and it ticks every
15 s, so it can grab the pictures before they go; the frames dir is mounted
read-only.

- Capture (every tick, first): each new walker-kind crop (by `created_at`)
  gets a padded crop (256 px tall) and its frame a 640 px copy in
  `<VISION_STORY_STRIP_DIR>/_pending/`. The frame is found by name
  (`<camera>_<stream>_<epoch_ms>_*.jpg`, +-2 ms of `observed_at`). A crop
  whose frame is already gone gets a `.miss` marker. Pending files older than
  `VISION_STORY_STRIP_PENDING_SEC` are deleted.
- Assemble (every published story without a strip): up to 4 crops spread
  over the passage plus one 640 px frame with all boxes drawn (1 = green
  first, last = red), in `<VISION_STORY_STRIP_DIR>/<story_id>/`. The row gets
  `evidence_strip` (each crop's file or `missing: frame_gone|not_captured`),
  `frame_luminance` (whole-frame mean luma), `scene_luminance` and
  `scene_contrast` (median over the crops of each crop's median luma / RMS
  contrast = std of luma, 0..1) and `dark`.
- `dark = scene_contrast < VISION_STORY_STRIP_DARK_CONTRAST` (0.10). Why
  contrast, not brightness (measured live 2026-10-17, 20:12-20:37 MDT): the
  camera was still in colour (no IR) and the porch light held the whole-frame
  mean at 0.255, so the frame mean never reads dark. Worse, the 00:47 story's
  place (porch-lit lawn, nothing visible) has crop median luma 0.11-0.125 --
  the same as a dark-clothed person in daylight (0.114 at 13:33). RMS
  contrast separates them: all 10 clean non-prop story places on the night
  frame read 0.012-0.066 through this code; the 31 clean daytime/dusk crop
  thumbnails (13:09-18:47 MDT) read 0.133-0.3 (p10 0.187). Dark stories are
  still published (`dark=true`); the Hub's "only checkable" filter hides
  them by default.
- Retention: `VISION_STORY_STRIP_RETENTION_DAYS=0` (default since 2026-10-20,
  Juniper: keep photos indefinitely) keeps every strip forever: no age rule
  deletes one, story retention skips a story that still has a saved picture,
  and the orphan-directory sweep is off (only stale pending files go). With a
  positive value a strip is kept until that many days after its story ended
  AND the story has a verdict ("Can't tell" counts); `evidence_strip.pruned_at`
  is then set. An unchecked story keeps its strip until story retention
  (`VISION_MOTION_STORY_RETENTION_DAYS`, 90) deletes the story row itself; the
  hourly sweep then removes the orphan directory.
- Robustness: the capture read re-covers 10 s behind its cursor (a crop row's
  `created_at` is its transaction start, so it can commit late) and skips the
  newest second; prop boxes (stationary individuals, the patio skeleton) are
  matched by box overlap and recorded as `prop_overlap`, not decoded; capture
  runs again just before strips are built; a strip that keeps failing gets a
  final `{"error": ...}` record after half the pending window.
- `frame_luminance` is a diagnostic only (stored as asked, nothing reads it).
- Trace: `[STORY_STRIP] story_id=... crops_saved= crops_missing= frame= frame_lum= scene_lum= scene_contrast= dark=`.
- Migration `services/orion-sql-db/manual_migration_vision_motion_story_strip_v2.sql`
  (additive). Without it the loop logs `[STORY_STRIP] missing columns` and
  stories keep running without pictures.
- Eval (read only): `evals/run_story_strip_darkness_eval.py` -- coverage
  (saved vs frame_gone) and scene contrast by verdict, to re-check 0.10
  once Juniper has used "Can't tell".

### Walkway rollups by hour, day, week and month (2026-10-18)

`orion/vision/walkway_rollup.py` (pure) + `app/vision_rollup.py` (reads, build
order, writes) + `app/vision_rollup_loop.py` (clocked, every
`VISION_ROLLUP_INTERVAL_SEC`, default 300 s; work first, then sleep). Deterministic,
no model. Design: `docs/superpowers/plans/2026-10-18-walkway-rollups-design.md`.

- One `vision_rollup` row per (stream, grain, period): `rollup` (JSON,
  `vision_rollup.v1`) + `line` (one plain sentence). Hours close as they end;
  days are built from their hour rows, weeks (Mon-Sun) and months from day rows,
  so totals always agree. Local time `VISION_LOCAL_TZ`; DST days have 23/25 hours.
- A **passing** is one individual seen in one clock hour (sightings, not crops).
  Not counted, tallied under `excluded`: stationary sightings/individuals,
  `VISION_MOTION_PROP_INDIVIDUAL_IDS`, no-embed zones, and vehicles seen in
  `VISION_ROLLUP_PARKED_MIN_HOURS` (4) or more hours of the trailing 24 h.
- **People are a range** (`rules_version` 2, 2026-10-20). Person clusters
  over-count (one person splits into several), so `kinds.person.count_range =
  {low, high, method_low, method_high}`: low = most people in one vision window
  (`peak_at_once.person`, less a person-kind prop sighted that hour; a parent takes
  its largest child low), high = person clusters (passings). Lines say "people: at
  least 5, at most 79 (the camera splits one person into several)" and other
  totals read "at most N passings". Vehicles keep their count. A version bump
  rebuilds the hours inside `VISION_ROLLUP_CATCHUP_HOURS`; older rows keep v1 and
  every reader re-renders them (`walkway_rollup.display_line`), or run the backfill
  with `--apply` to rebuild them for good.
- **Camera minutes** = minutes with a `vision_scene_inventory` row. An hour with
  fewer than 15 seen minutes reads "camera was down", never "nobody passed"; a
  child period with no row reads "No record".
- An hour is built only once the individuals reducer's cursor is past its end
  (+60 s) and it has been closed for 120 s. The newest 3 closed hours are rebuilt
  every tick (sightings keep growing); missing hours are filled back
  `VISION_ROLLUP_CATCHUP_HOURS` (48).
- Story sentences appear only when Juniper's motion-story spot-check gate passed
  (`orion/vision/motion_story_gate.py`, shared with the Hub card).
- Migration `services/orion-sql-db/manual_migration_vision_rollup_v1.sql`
  (auto-applies on pull). Kill switch: `VISION_ROLLUP_ENABLED=false`.
- Trace: `[VISION_ROLLUP] tick_complete {... hours, days, weeks, months, ready_until, story_gate_passed}`.
- Readers: orion-actions walkway journal, day letter, curiosity study material,
  Hub "Walkway history" panel, orion-substrate-runtime "seen on <day>" links.
- Backfill (dry run by default): `PYTHONPATH=. python scripts/backfill_vision_rollups.py --days 7 [--apply]`.
- Replay (read only): `PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_rollup_replay_eval.py --days 3`.
- **Which people count is closest to the truth** (spec PR #2763, Item 2 patch B):
  `PYTHONPATH=. python services/orion-sql-writer/evals/run_walkway_count_estimator_eval.py --days 7`
  (read only). It scores the estimators in `orion/vision/walkway_count.py` (clusters,
  most at once, motion stories + clear glimpses, presence runs) against hand counts in
  `evals/fixtures/walkway_count_labels.csv` (format in the file). With fewer than 10
  hand-counted hours it says INSUFFICIENT LABELS and lists the 10 hours most worth
  counting. Gates: an empty hour reads 0, a person standing 2 minutes stays 1.
- **A glimpse counts only when it is clearly a person** (decision 6, Juniper 2026-10-23):
  a one- or two-frame person story counts when one detection has score >= 0.44, a box at
  least 6.25 % of the frame tall (120 px of 1920), is not edge-clipped, lies in a zone, and
  is not a prop; and no other counted person story lies within 11 s of it
  (`walkway_count.clear_glimpse`, `story_count`). The thresholds come from the stories
  Juniper checked in the Hub. This feeds only the story-based estimator. The Orion-facing
  range does not change, and a story-based count may only be shown held inside
  [most at once, clusters] (`within_range`).

## Storage-write organ (2026-10-02)

The writer reports on its own writes (`app/write_health.py`). Every incoming write ends in one
outcome per envelope: `committed`, `duplicate` (idempotent skip), `skipped` (writer chose not to
write), `unrouted` (no route; counted, never part of the reading), or a failure class --
`validation`, `constraint`, `serialization`, `db_unavailable`, `timeout`, `db_error`,
`backpressure` (grammar queue full), `other`. The class comes from the exception or from the
error text already written to `bus_fallback_log` (only the head of the message is matched, so
payload text in `[parameters: ...]` cannot pick a class). Grammar events are counted in the
persist helpers, one count per event, and the organ never counts its own `sql_writer.storage:`
events.

Once per `SQL_WRITER_WRITE_HEALTH_WINDOW_SEC` (60) it publishes one trace on
`orion:grammar:event`: one `storage_write_window_observed` atom per table family that saw traffic
(at most 32, overflow folded into `_other`) with counts by class and write wall-time p50/p95, and
one `storage_writer_window_completed` atom with totals and the grammar queue high-water mark.
Counts only, never payloads or error messages. A publish failure drops that window with a warning
and never touches the write path. substrate-runtime's storage_write reducer turns it into
`write_failure_pressure` on `node:substrate.storage_write` -> `capability:storage`
`reliability_pressure` (see the field-digester README).

Flag: `SQL_WRITER_WRITE_HEALTH_ENABLED` (off in code, on in `.env_example`). Replay real history
through the emitter and reducer: `python services/orion-sql-writer/evals/storage_write_replay.py --live`.

## Orion energy hourly rollup (2026-10-22)

`orion_energy_hourly` holds Orion's own kWh per UTC hour per source, written only by
`app/orion_energy_rollup.py` and read by the Hub Orion Energy tab. Raw cluster watts are pruned
after `ORION_BIOMETRICS_CLUSTER_RETENTION_DAYS` (30); this table has no retention, so months and
years survive.

- Sources: every node in `orion_biometrics_cluster.measurements_by_node` with `chassis_watts`
  (BMC or PDU wall draw), plus `cooling` from `home_cooling_sample` (non-stale, device online).
- kWh = each sample's watts held until the next sample, capped at 60 s (nodes) / 30 s (cooling),
  so a gap is left uncovered rather than filled in. `covered_sec` says how much of the hour was
  measured. No samples, no row.
- Every `ORION_ENERGY_ROLLUP_INTERVAL_SEC` (600) it recomputes the last 6 hours; an empty table
  backfills from every raw row still on disk (~3 s for 30 days). An upsert never lowers
  `covered_sec`, so a half-pruned hour cannot overwrite a complete one.
- Log: `[ORION_ENERGY_ROLLUP] tick_complete rows_upserted=N`.

```sql
SELECT source, date_trunc('day', hour_start) d, round(sum(kwh)::numeric, 1) kwh
FROM orion_energy_hourly GROUP BY 1, 2 ORDER BY 2 DESC, 1 LIMIT 12;
```

## Act ledger (rung 2, 2026-10-16)

"Doing on one track" from `docs/superpowers/specs/2026-10-14-orion-thinking-architecture-design.md`.
Producers publish `act_opened` / `act_closed` grammar events (`ActV1` on `GrammarEventV1.act`,
`orion/schemas/grammar.py`) on `orion:grammar:event`. This writer:

- materialises them into `grammar_acts` (one row per act, closed by `act_id`, first close wins);
- stamps `predicted` at open from the current `substrate_action_effect_posterior` mean for
  (`act:<verb>`, actor, signal) -- producers never type a magnitude;
- grades every close through the existing L10 path: one `substrate_action_outcomes` row
  (`dispatch_kind = act:<verb>`, `target_id = actor`) plus the posterior upsert, with the same
  two double-count guards as orion-feedback-runtime (`orion/grammar/act_ledger.py`);
- runs a horizon resolver every `SQL_WRITER_ACT_RESOLVER_INTERVAL_SEC` (60 s): `juniper_replied`
  acts close 1 when Juniper's next chat turn STARTS (its hub ingress hop in
  `cockpit_turn_sighting`, so a slow or errored turn still counts; a logged `chat_history_log`
  turn without one falls back to its `created_at`; canonical `orion.regulation.juniper_turns`
  predicates; the act's own turn excluded by correlation id) or 0 at the horizon; producer-reported `artifact_produced` acts unclosed 120 s past their horizon
  close 0 as `expired`.

Producers today: orion-thought paintings (`make`), hub chat reply (`say`), hub outreach
(`say`/`ask`), hub world-pulse reading Stage 1 (`look`). Debug surface: `GET /act/ledger?days=7`.
Kill switch: `SQL_WRITER_ACT_LEDGER_ENABLED=false` (events still land in `grammar_events`).
Eval: `python evals/act_ledger_calibration.py [--live]`.

**Cost filled on close (body economy Patch A2, 2026-10-23).** `ActV1.cost` accepts `ActCostV1` or
`ActCostV2` (V1 plus `gpu_slot_s_by_lane`, `wh_by_host`, `juniper_msgs`, `claude_limit_events`; all
optional, None = not measured, never 0). On every close the ledger fills what it can measure and
the producer left unset, into `grammar_acts.cost_json`: `seconds` = closed_at - opened_at (needs
the open), and `gpu_slot_s_by_lane` = lease time per pool role from `gpu_pool_events` whose
`turn_correlation_id` is the act's correlation_id (window: open - 1 h .. close + 120 s; each lease
once; the read runs in its own savepoint). Producer values win. `substrate_action_outcomes.latency_ms`
still comes only from a producer-reported `seconds`. Rollout is consumer-first: producers keep
sending V1 until every reader of `orion:grammar:event` / `grammar_events` runs V2-aware code.

**Rung 3: acts bound to the workspace winner (2026-10-21).** At open, an act whose actor answers a
world-first workspace source is stamped with the winning loop it was opened under
(`grammar_acts.workspace_ref` = the broadcast's selected open loop id), through the same binding
rule the proposal layer uses (`orion.autonomy.self_shed.bind_workspace_winner`: action selected,
<= 90 s old, held >= 2 ticks). Today one pairing: `hub.chat_reply` answers `world:chat` /
`node:substrate.chat` (`ACT_ANSWERS_WORKSPACE_REFS`). When the act closes, the writer adds Orion's own
non-final `acted` verdict on that loop to `attention_loop_outcome` (`actor='orion'`, outcome in
`features_at_close`), in the same transaction under its own savepoint. Never `resolved` or
`dismissed`. Kill switch: `SQL_WRITER_ACT_WORKSPACE_BINDING_ENABLED=false`.

## Running & Testing


### Run via Docker
```bash
docker-compose up -d orion-sql-writer
```

### Smoke Test
Validate GPT turn ingest end-to-end (bus -> sql-writer -> Postgres):

```bash
python services/orion-sql-writer/scripts/smoke_chatgpt_turn_sql.py
```

Expected output includes `found_in_chat_gpt_log: True`; sql-writer logs should include:
`Written ChatGptLogTurnV1 -> chat_gpt_log`.

Visual execution adds nullable `visual_outcome` to `action.outcome.emit.v1` and
`action_outcomes`. The shared schema restricts it to `produced`,
`deferred_thermal`, `deferred_busy`, `already_satisfied`, `failed`, or `unknown`;
nonvisual and historical rows may omit it. The normal writer row mapper persists
it and repeated action IDs upsert the same row. Boot DDL adds the column on
existing databases; the standalone additive migration is
`services/orion-sql-db/manual_migration_action_outcomes_visual_outcome.sql`.
Apply that migration before deploying producers/writers that use this field.
This patch does not run the migration or rewrite historical rows.


## GPU state history for offline replay (2026-10-09)

`orion:gpu_pool:state` / `gpu_pool.state.v1` now routes through the existing
`GpuPoolStateV1` validator into `gpu_pool_state_history`. This records the
broadcast's source time, host, configuration digest, mode, backlog and queue
counts. Cards, lease bodies and prompts are not stored. No GPU producer or
scheduler changes. Missing source time/host/depths are rejected rather than
filled with schema defaults. Host + source time + configuration digest form a
stable snapshot id, so redelivery is insert-only and idempotent.

`GPU_POOL_STATE_HISTORY_RETENTION_DAYS=30` bounds roughly 518,400 small rows per
pool at the current five-second cadence. Retention health includes this table.
`GRAMMAR_RETENTION_PERIODIC_MAX_CYCLE_SEC=75` preserves the existing minimum
five seconds per managed table now that there are fifteen (2026-10-20 added
`self_signal_outcome_association`, `SELF_SIGNAL_OUTCOME_ASSOCIATION_RETENTION_DAYS=30`). Both keys are passed
through compose; the subscription and route templates include the state channel.

Apply `services/orion-sql-db/manual_migration_regulation_history.sql` before
restarting the writer (normal SQL model bootstrap can also create the GPU table).
The dream observation table in that migration is owned by dream, not this writer.
Use the actual snapshot host, which currently reads `circe`, for offline replay:

```bash
python scripts/analysis/measure_arousal_replay.py --print-sql --gpu-host circe \
  --start 2026-10-09T06:00:00Z --end 2026-10-10T06:00:00Z
```

Export with `psql -XqAt -v ON_ERROR_STOP=1`; give the resulting JSONL back to the
same report. The report rejects mixed-host histories. A snapshot older than
15 seconds remains unknown, even if lease events continue arriving. Existing
lease-event-only exports still produce an explicitly provisional replay.
