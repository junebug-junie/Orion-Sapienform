# Bus observer: retire the XLEN stream-depth family (2026-09-25)

Branch: `fix/bus-observer-scope`. Answers open question 4 of
`docs/superpowers/specs/2026-09-22-substrate-lattice-audit.md` ("widen
`BUS_OBSERVER_STREAMS` or retire"). Decision: **retire**.

## What the bus observer was measuring

`services/orion-bus/app/bus_observer.py` ran `XLEN` on the two keys in
`BUS_OBSERVER_STREAMS` (`orion:stream:world_pulse:run:result` and its `:dlq`)
every 10 s. `orion/substrate/transport_loop/extract.py` turned that into:

| Field | Real input |
|---|---|
| `stream_depth_pressure` | `max XLEN / 100_000` |
| `backpressure` | count of streams past `BUS_STREAM_DEPTH_WARNING` |
| `stream_backlog_pressure` | `max(stream_depth_pressure, backpressure)` |
| `stream_backlog_health` | the observer's own Redis `PING` (1.0 / 0.5 / 0.0) |
| `delivery_confidence` | `PING` again, forced to 0.0 on observer failure |

The last two never touched a stream: they were the observer's `PING`.

## Live stream inventory (read-only, 2026-09-25 ~06:00Z, `ORION_BUS_URL`)

`SCAN 0 TYPE stream` over the whole bus, then `XINFO STREAM` / `XINFO GROUPS`:

| Stream | XLEN | Last write | Groups | Group | Consumers | Pending | Lag |
|---|---|---|---|---|---|---|---|
| `orion:stream:world_pulse:run:result` | 160 | ~18 h ago | 1 | `cg:concept-induction` | 1 | 0 | 0 |
| `orion:queue:spark:introspection` | 22,165 | ~61 days ago | 1 | `spark-introspector-workers` | 343 | 0 | 0 |
| `orion:pad:events` | 500 | ~59 days ago | 0 | — | — | — | — |
| `orion:pad:frames` | 501 | ~59 days ago | 0 | — | — | — | — |
| `orion:curiosity:help:request` | 1 | ~10 days ago | 0 | — | — | — | — |

`orion:stream:world_pulse:run:result:dlq` does not exist; `XLEN` of a missing
key returns 0, which is why it always read 0.

Stored history (retained `grammar_events`, 2026-09-22 → 09-25, 24,633 observer
ticks): world_pulse `stream_length` 157–160, DLQ 0 on every tick, zero
`bus_backpressure_observed` atoms, zero `redis_ping_ok=false`.

## Why not widen

Option (a) was to auto-discover every stream with a consumer group and read
group lag / pending instead of `XLEN`. Live, that set is two streams: one live
consumer at lag 0 / pending 0, and one queue nobody has written to in 61 days.
The only way such a signal leaves calm is one consumer (concept-induction)
dying while world_pulse keeps publishing, which service heartbeats already
cover. Everything else on the bus is Pub/Sub, which has no backlog. A widened
instrument would read 0 by construction, the same failure as the one it would
replace. It fails gate step 4 (can it ever read non-calm?) before it is built.

The `PING`-derived pair fails step 4 by construction: the observer publishes
its atoms on the same Redis it pings, so a failed `PING` means the atom
reporting it cannot be delivered. `delivery_confidence` is also not
independent (step 2): it is exactly `1 - reliability_pressure`.

## What was removed (kill means kill)

- Producer: the `XLEN` loop, `bus_stream_depth_observed` / `bus_backpressure_observed`
  atoms, `uncertainty_from_backpressure`, `BUS_STREAM_DEPTH_WARNING` /
  `BUS_STREAM_DEPTH_CRITICAL` (orion-bus and orion-substrate-runtime settings,
  `.env_example`, compose).
- Reducer: `TransportBusStateV1.{total_stream_depth, max_stream_depth,
  backpressure_count, stream_backlog_health, delivery_confidence,
  stream_depth_pressure, backpressure, stream_backlog_pressure}`; the matching
  `pressure_hints`; `DEFAULT_STREAM_DEPTH_CRITICAL`; the `stream_depth_critical`
  parameter chain.
- Field: node channels `stream_backlog_pressure`, `stream_backlog_health`,
  `delivery_confidence`; capability channel `stream_backlog_pressure`; their
  decay entries, glossary entries, polarity entries, the
  `field_coherence` rule, both lattice YAMLs, the `node:athena ->
  capability:orchestration` `stream_backlog_pressure` mapping, and the
  `capability:transport -> capability:orchestration` edge (source channel was
  never written; 2026-09-22 audit).
- Consumers: relational transport adapter, hub M3 card, consolidation
  `transport_healthy_idle` motif (now `max_pressure` on
  `capability:transport.pressure`), incident-signal log, smoke script.

Kept on purpose: `mood_arc`'s encoder exclusion list (historical corpora still
contain these channels); `scripts/analysis/*` history tools; historical prose.

## What was not changed

- `reliability_pressure` keeps identical values in every ping / failure case
  (`ping_pressure` = 0.0 ok / 0.5 unknown / 1.0 failed, folded in directly;
  parametrized equivalence test against the old formula).
- `BUS_OBSERVER_STREAMS` still drives catalog membership and the bounded
  schema sample behind `contract_pressure`. That is the same narrow
  two-stream scope; it is flagged as the next instrument to judge, not fixed
  here.

## Migration safety

- `TransportBusStateV1` is `extra="forbid"`. Its `mode="before"` validator drops
  exactly the eight retired names, so the live persisted row (verbatim copy in
  `tests/test_transport_projection_schemas.py`) still loads; any other unknown
  key still fails. `scripts/check_substrate_projection_schema_drift.py` against
  live Postgres: OK.
- Field reconcile prunes the retired node channels from every node
  (`RETIRED_NODE_CHANNELS`, `None` = no successor) and, via a new
  `RETIRED_CAPABILITY_CHANNELS`, from every capability vector and its
  provenance. That also clears the capability-level `transport_pressure` left
  over from the 2026-07-24 rename, which was still on every live capability
  vector.
- A pre-deploy receipt still carrying the retired hints produces no
  perturbation.

## Where transport health lives now

`capability:transport.pressure` (fed by `node:substrate.bus_synaptic`),
`catalog_drift_pressure` (mesh-wide census, `bus_census_computed`),
`observer_failure_pressure` / `reliability_pressure`, and RPC health
(`orion:rpc_health:snapshot`).

## Acceptance checks

1. After deploy, `bus_stream_depth_observed` / `bus_backpressure_observed` rows
   stop appearing in `grammar_events`.
2. The live `substrate_transport_bus_projection` row no longer carries any of
   the eight retired keys after one reducer tick.
3. No `node_vectors` entry carries `stream_backlog_*` / `delivery_confidence`,
   and no `capability_vectors` entry carries `stream_backlog_pressure` /
   `transport_pressure`, after one field-digester tick.
