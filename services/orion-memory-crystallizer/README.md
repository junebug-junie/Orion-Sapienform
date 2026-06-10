# orion-memory-crystallizer

Governed cognitive memory crystallization worker.

## Role

- Proposes `MemoryCrystallizationV1` artifacts (never canonical without governor)
- Validates proposals (schema, evidence, stance/contradiction rules)
- Projects approved crystallizations to Chroma (`orion:memory:vector:upsert`) and optional Graphiti/FalkorDB
- Does **not** replace `MemoryCardV1` or RDF `memory_graph`

## Invariants

```text
MemoryCardV1 = turn-facing recall surface
MemoryCrystallizationV1 = governed cognitive memory
Chroma = semantic projection (Postgres wins on drift)
Graphiti/FalkorDB = additive temporal projection
RDF memory_graph = existing approved graph path (unchanged)
```

## Hub API

Primary operator surface is Hub:

- `POST /api/memory/crystallizations/propose`
- `POST /api/memory/crystallizations/proposals/{id}/approve`
- `POST /api/memory/crystallizations/{id}/project/chroma`
- `POST /api/memory/active-packet`

## Bus channels

Registered in `orion/bus/channels.yaml`:

- `orion:memory:crystallization:proposed`
- `orion:memory:crystallization:validated`
- `orion:memory:crystallization:approved`
- `orion:memory:crystallization:rejected`
- `orion:memory:crystallization:quarantined`
- `orion:memory:crystallization:project`
- `orion:memory:crystallization:retrieved`

Vector projection uses existing `orion:memory:vector:upsert`.

## Run locally

```bash
cp .env_example .env
docker compose up --build
```

Schema auto-applies when `CRYSTALLIZER_AUTO_APPLY_SCHEMA=true` and `POSTGRES_URI` is set.
