# Orion Spark Concept Induction

Bus-native Spark capability that consolidates recent Orion experience into concept profiles and deltas, then publishes them on the Titanium bus.

## Run

```bash
docker compose -f services/orion-spark-concept-induction/docker-compose.yml --env-file .env up -d orion-spark-concept-induction
```

Health check: http://localhost:8500/health

## Env lineage

- `.env_example` → `docker-compose.yml` environment → `orion.spark.concept_induction.settings.ConceptSettings`

Key knobs:

- `BUS_INTAKE_CHANNELS`: JSON list of channels to watch (defaults: chat history, collapse mirror, memory episodes)
- `BUS_PROFILE_OUT`: profile publish channel (kind `memory.concepts.profile.v1`)
- `BUS_DELTA_OUT`: delta publish channel (kind `memory.concepts.delta.v1`)
- `SPACY_MODEL`: spaCy model name (default `en_core_web_sm`)
- `EMBEDDINGS_BASE_URL`: external embedding host, POST /embed with `{"items": [...]}`; gracefully degrades if missing
- `USE_CORTEX_ORCH`: enable LLM refinement via Cortex-Orch verb `concept_induction`

## Local test

```bash
python -m scripts.test_concept_induction_publish
```

This publishes a fake chat event and waits for a profile on `BUS_PROFILE_OUT`.
