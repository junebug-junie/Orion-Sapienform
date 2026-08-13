# Orion Spark Concept Induction

Bus-native Spark capability that consolidates recent Orion experience into concept profiles and deltas, then publishes them on the Titanium bus.

## Run

```bash
docker compose -f services/orion-spark-concept-induction/docker-compose.yml --env-file .env up -d orion-spark-concept-induction
```

Health check: http://localhost:8510/health

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `ConceptWorker`'s own bus connection.

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

## Drive-pressure engine, goal generation, drive-audit production — DELETED 2026-07-30

Everything this section used to describe — `DriveEngine`, the drive-state
divergence audit (`scripts/drive_state_divergence_audit.py`, which no longer
exists anywhere in the repo — do not try to run the commands a prior version
of this README had here), `DriveAuditV1` production (`audit.py`,
`drive_attribution.py`), and drive-relief "satisfaction tensions"
(`tensions.py`'s signed-impact handling) — was deleted outright 2026-07-30
(`chore/delete-orion-drives`, PR #1486), following through on
`orion/sentience_striving_program/README.md` §8's 2026-07-18 halt. Concept
extraction/clustering/embedding/dossier/identity/profile production (the rest
of this README, and the actual reason `ConceptWorker` exists) is untouched and
runs unconditionally on every intake event, independent of this deletion.

**What this means for Postgres `drive_audits` and the (now-removed) Hub Drives
tab:** this service is no longer a producer of anything drive-shaped.
`drive_audits` was a frozen, finite historical table for a couple of weeks
after this deletion — until it, and the Hub **Drives** tab that read it, were
both removed outright 2026-08-13 (table dropped, snapshotted first; tab and
its backend deleted). See
[orion/autonomy/README.md § Hub Drives Analytics](../../orion/autonomy/README.md)
(now a removal note, not a live description) and the PR reports
(`docs/superpowers/pr-reports/2026-07-30-delete-orion-drives-pr.md` for this
service's own producer deletion,
`docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-analytics-tab-pr.md`
for the tab/table removal) for the full picture, including the accepted
consequence that this service no longer proposes goals at all.

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

## Real surprise signal for ActionOutcomeEmitV1 (2026-07-28)

Every `ActionOutcomeRefV1`/`ActionOutcomeEmitV1` this worker emits
(`episode_fetch.py`'s readonly fetch, `policy_act.py`'s readonly recall,
`curiosity_reuse.py`'s world-pulse-followup reuse) previously hardcoded
`surprise` as a binary success/fail proxy (`0.0`/`1.0`) — see
`orion/autonomy/models.py::ActionOutcomeRefV1`'s docstring. `ConceptWorker`
now supplies a real `surprise_source` callable
(`self._bus_synaptic_surprise_source`) to all three emit paths, backed by
`orion.substrate.bus_synaptic_surprise.latest_bus_synaptic_prediction_error()`
— a real, generic, already-live-validated ambient mesh-wide surprise signal
(PR #1377, calm-floor fixed PR #1391), the same instrument
`services/orion-execution-dispatch-runtime` uses for the same field.

This is optional and fail-open: `ORION_ACTION_OUTCOME_DB_URL` (unset by
default) gates whether `_get_surprise_engine()` ever builds a real Postgres
connection — this service has no other Postgres dependency. Unset, absent,
stale (see that function's staleness-guard docstring), or erroring reads all
degrade to `None`, and every emitter's own `resolve_surprise()` falls back to
the pre-2026-07-28 success/fail-proxy exactly as before. Set it to the same
`conjourney` DSN `services/orion-cortex-exec` already uses for this class of
read to turn the real signal on.
