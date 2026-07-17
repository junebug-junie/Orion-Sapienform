# Orion Spark Concept Induction

Bus-native Spark capability that consolidates recent Orion experience into concept profiles and deltas, then publishes them on the Titanium bus.

## Run

```bash
docker compose -f services/orion-spark-concept-induction/docker-compose.yml --env-file .env up -d orion-spark-concept-induction
```

Health check: http://localhost:8510/health

## Env lineage

- `.env_example` → `docker-compose.yml` environment → `orion.spark.concept_induction.settings.ConceptSettings`

Key knobs:

- `BUS_INTAKE_CHANNELS`: JSON list of channels to watch (defaults: chat history, collapse mirror, memory episodes). **This is a full replace, not a merge**: `ConceptSettings` reads it via a `validation_alias`, so setting the env var overrides the Python default list entirely rather than adding to it. Adding a new intake channel to the code default alone does nothing in any environment where `BUS_INTAKE_CHANNELS` is set (`.env_example` and every deployed `.env` set it) -- the new channel must also be added to every `.env`/`.env_example` that sets this key, or the subscription is a silent no-op.
- `BUS_PROFILE_OUT`: profile publish channel (kind `memory.concepts.profile.v1`)
- `BUS_DELTA_OUT`: delta publish channel (kind `memory.concepts.delta.v1`)
- `SPACY_MODEL`: spaCy model name (default `en_core_web_sm`)
- `EMBEDDINGS_BASE_URL`: vector host base URL; concept induction calls `POST /embedding` with `EmbeddingGenerateV1` payloads and degrades gracefully if unavailable
- `USE_CORTEX_ORCH`: enable LLM refinement via Cortex-Orch verb `concept_induction`

## Local test

```bash
python -m scripts.test_concept_induction_publish
```

This publishes a fake chat event and waits for a profile on `BUS_PROFILE_OUT`.

## Drive-state divergence audit

`drive_state.v1` (this service's `DriveEngine`, persisted to the local
`LocalProfileStore` JSON file at `CONCEPT_STORE_PATH`) and `autonomy_state_v2`
(`orion.autonomy.reducer`, persisted to Postgres via
`ORION_AUTONOMY_STATE_DB_URL`) independently compute pressures over the same
6-key drive taxonomy. `orion/self_state/inner_state_registry.py` marks both
`DUPLICATE` of each other and defers the merge-or-keep-separate decision to a
later phase. `scripts/drive_state_divergence_audit.py` (repo root) is a
report-only diagnostic that loads the current value of each and prints
per-drive pressure divergence and activation-flag agreement -- it never
merges or picks a winner between them, and it always exits 0.

```bash
python scripts/drive_state_divergence_audit.py
python scripts/drive_state_divergence_audit.py --json
```

### This service is the producer of `DriveAuditV1` (Hub Drives Analytics)

This service is the sole **producer** of `DriveAuditV1` audit artifacts (schema
`orion/core/schemas/drives.py`, `kind="memory.drives.audit.v1"`) — including the
`tick_attribution` (per-drive float weights) and `tension_kinds` (short list of
contributing tension kinds) fields. `orion/spark/concept_induction/audit.py::build_drive_audit`
builds each audit from the current `DriveStateV1` plus the tick's tensions and attribution, then
`ConceptWorker._publish_artifact` (`bus_worker.py`) publishes it on `self.cfg.drive_audit_channel`
(`orion/spark/concept_induction/settings.py`). `tick_attribution` itself comes from
`compute_tick_attribution()` in `orion/spark/concept_induction/drive_attribution.py`, which also
derives `dominant_drive` when it isn't passed explicitly.

**This service does not serve the Hub UI.** The Hub **Drives** tab (`#drives`,
`/drives-analytics`) never talks to concept-induction directly. Instead: `orion-sql-writer`
subscribes to the audit channel and persists each audit (including, as of
`services/orion-sql-db/manual_migration_drive_audits_v4_tick_attribution.sql`, the
`tick_attribution`/`tension_kinds` columns — previously dropped by the sql-writer column filter) to
the Postgres `drive_audits` table (`services/orion-sql-writer/app/models/drive_audit.py`). Hub then
reads that Postgres history via its own `RECALL_PG_DSN`-backed pool
(`services/orion-hub/scripts/drives_analytics_queries.py`). This service is upstream producer only —
it has no dependency on Hub, and a Hub outage does not affect audit production.

See also: [Hub Drives tab operator docs](../orion-hub/README.md) (`#drives`) and
[orion/autonomy/README.md § Hub Drives Analytics](../../orion/autonomy/README.md#hub-drives-analytics)
for what the persisted audits mean to an operator.

## Satisfaction tensions (drive relief)

`DriveEngine.update()` supports signed drive impacts: a `TensionEventV1` with a
negative `drive_impacts` weight *relieves* pressure instead of raising it
(`_clamp_signed` to `[-1, 1]`, then the leaky-math formula branches on impulse
sign so relief is bounded to `[0, base]` and growth to `[base, 1]` -- relief can
never push a drive negative or growth push it past 1). Prior to this, negative
weights were silently double-clamped to zero and had no effect, so drives could
only ever accumulate pressure, never discharge it from a successful action.

`orion:autonomy:action:outcome` (published by Layer-9 dispatch on every
autonomous action, see `services/orion-execution-dispatch-runtime`) is
subscribed here specifically to close that loop:
`extract_tensions_from_action_outcome` (`orion/spark/concept_induction/tensions.py`)
mints a relief tension when `outcome.success is True` for a closed set of
dispatch kinds (`inspect` relieves `coherence`, `summarize` relieves
`predictive`, `observe` relieves `continuity` -- unmapped kinds mint nothing).
`bus_worker.py` filters out envelopes this service itself published
(`source.name == self.cfg.service_name`) before parsing, since this service is
also a downstream consumer of channels it doesn't produce here but could in a
replay/backfill scenario.

Known accepted risk: `extract_tensions_from_feedback` (fires on Postgres-polled
failure) and `extract_tensions_from_action_outcome` (fires on bus-emitted
success) are independently-computed classifications of the same Layer-9
dispatch from two separate pipelines. A dispatch that the feedback path scores
as failed/mixed while the outcome-emit reports `success=True` can fire both a
growth and a relief tension for the same event. Not coordinated across
pipelines in this patch -- named here as a follow-up, not fixed.

## Readonly capabilities (recall.query.readonly, P4)

`ConceptWorker` (`bus_worker.py`) is the sole production call site for
`orion.autonomy.policy_act.maybe_execute_substrate_act_after_metabolism`. That
function gates two readonly capabilities under `config/autonomy/capability_policy.v1.yaml`
per cycle -- a Firecrawl fetch and (new) an inline `RecallService` RPC
(`_execute_readonly_recall` / `maybe_execute_readonly_recall_after_goal` in
`orion/autonomy/policy_act.py`) -- and tries recall first: a successful recall
leaves that cycle's fetch budget unconsumed. Both capabilities require the
caller to pass `recall_bus`/`recall_source` (this worker's own `self.bus` and
service identity); the function degrades to a no-op recall attempt if either
is `None`, so wiring the kwargs at the call site is load-bearing, not
cosmetic. A successful recall populates `SubstrateActResultV1.recall_outcome`
(mirroring `fetch_outcome`) and is published through the same
`ActionOutcomeEmitV1` → sql-writer path a fetch success uses, so it reaches
durable SQL storage rather than only the local file-store fallback inside
`_execute_readonly_recall`.
