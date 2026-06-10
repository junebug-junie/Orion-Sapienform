# orion-memory-crystallizer

Governed cognitive memory service for `MemoryCrystallizationV1`.

Core invariant:

```text
MemoryCardV1 is the turn-facing recall artifact.
GrammarEventV1 is the substrate trace artifact.
MemoryCrystallizationV1 is the governed cognitive memory artifact.
Chroma is semantic recall projection.
Existing RDF memory_graph remains the approved graph path.
Graphiti/FalkorDB is an additive temporal graph projection.
Postgres preserves canonical crystallizations.
The governor decides.
The user inspects.
```

## What it does

- Ingests `MemoryCrystallizationV1` proposals via HTTP
  (`POST /api/memory/crystallizations/propose`) and via the bus channel
  `orion:memory:crystallization:proposed`.
- Validates proposals (schema, evidence, scope, kind rules — see
  `orion/memory/crystallization/validator.py`).
- Owns the governor path: approve / reject / quarantine / supersede /
  deprecate / archive. Proposals never become `active` without it.
- Persists canonical crystallizations in Postgres
  (`memory_crystallizations*` tables, idempotent DDL applied at boot from
  `orion/memory/crystallization/sql/memory_crystallizations.sql`).
- Projects active crystallizations:
  - `MemoryCardV1` recall surface (with `subschema.crystallization_ref`)
  - Chroma via the existing `memory.vector.upsert.v1` →
    `orion-vector-writer` path (collection `orion_memory_crystallizations`)
  - Graphiti/FalkorDB temporal episodes (additive, disabled by default,
    routes under `/api/memory/graphiti/*` — never `/api/memory/graph/*`)
- Builds `ActiveMemoryPacketV1` retrieval packets
  (`POST /api/memory/active-packet`) and records retrieval events.

## What it deliberately does not do

- Does not replace `MemoryCardV1` (no `MemoryCardV2`).
- Does not redefine grammar law; it only references existing
  `GrammarEventV1` / `GrammarAtomV1` / `GrammarEdgeV1` ids via the memory
  grammar envelope.
- Does not touch the existing RDF memory_graph routes or approval flow.
- Does not let local models, Graphiti, or Chroma canonize memory.

## API surface

```http
POST  /api/memory/crystallizations/propose
GET   /api/memory/crystallizations/proposals
GET   /api/memory/crystallizations/proposals/{id}
POST  /api/memory/crystallizations/proposals/{id}/validate
POST  /api/memory/crystallizations/proposals/{id}/approve
POST  /api/memory/crystallizations/proposals/{id}/reject
POST  /api/memory/crystallizations/proposals/{id}/quarantine
GET   /api/memory/crystallizations
GET   /api/memory/crystallizations/{id}
GET   /api/memory/crystallizations/{id}/history
POST  /api/memory/crystallizations/{id}/status
POST  /api/memory/crystallizations/{id}/supersede
POST  /api/memory/crystallizations/{id}/links
GET   /api/memory/crystallizations/{id}/links
POST  /api/memory/crystallizations/{id}/project/card
POST  /api/memory/crystallizations/{id}/project/chroma
POST  /api/memory/crystallizations/{id}/project/graphiti
GET   /api/memory/graphiti/health
POST  /api/memory/active-packet
GET   /api/memory/retrieval-events/{id}
GET   /health
```

## Run

```bash
cd services/orion-memory-crystallizer
cp .env_example .env   # then adjust for your host
docker compose up -d --build
```

## Tests

```bash
./scripts/test_service.sh orion-memory-crystallizer
```
