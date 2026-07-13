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

- `BUS_INTAKE_CHANNELS`: JSON list of channels to watch (defaults: chat history, collapse mirror, memory episodes)
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
