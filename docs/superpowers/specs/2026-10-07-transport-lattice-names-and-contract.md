# Transport lattice: one name per signal, and retiring the two-stream contract check (2026-10-07)

Branch: `fix/transport-lattice-names-and-contract`. Closes two open items from
`2026-09-22-substrate-lattice-audit.md` ("three vocabularies for one signal",
and "M3 observer covers only two world_pulse streams" for `contract_pressure`)
and the "next instrument to judge" left by
`2026-09-25-bus-observer-stream-depth-retirement.md`.

All numbers below were pulled live on 2026-10-07 between 01:10 and 02:00 UTC
(Postgres `conjourney`, Redis at `ORION_BUS_URL`, container logs).

## A. Two names for one signal

### What was true

- The field calls it `pressure` on `capability:transport`. The topology fills
  it as 0.85 x `node:substrate.bus_synaptic` `prediction_error` (the fraction
  of live bus-synaptic edges with |z| >= 3).
- The lattice policy calls it `bus_synaptic_pressure`. The Hub turned one into
  the other with a hand-kept table (`_CHANNEL_VALUE_SOURCES`); the mind recall
  resolver had its own mapping; `orion/field/transport_thresholds.py` hand-codes
  the 0.85.
- `make check-metric-lineage METRIC=bus_synaptic_pressure` -> `UNREGISTERED: no
  URN in any registry`. The semantic layer only knows
  `metric://field_channel/orion-field-digester/pressure` and the node-qualified
  `prediction_error` entry.
- `bus_synaptic_pressure` is also a live Redis key: the hash field holding the
  EWMA threshold state (`orion:lattice:transport_thresholds:v1`). Live: n =
  18,851 samples over 160.2 h, slow mean 0.0301, slow sd 0.0235, so the learned
  watch threshold (~0.148) is active and below the static 0.25.
- Real bug found on the way: the mind recall line printed raw
  `prediction_error` ("31% of live bus channels running anomalous") next to
  rungs on the 0.85-scaled `pressure` scale. The threshold it named sat ~15%
  below the point where it actually trips on the stated scale.

### Decision D1: keep the row id, make the policy carry the address

The honest name of the signal is `capability:transport.pressure` (registered
URN `metric://field_channel/orion-field-digester/pressure`). `bus_synaptic_pressure`
is kept as the lattice **row id**, not a metric. Every policy row now has a
`source:` block that says which reading it is, and readers resolve through it:

```yaml
bus_synaptic_pressure:
  source: {layer: m4, vector: capability:transport, channel: pressure}
catalog_drift_pressure:
  source: {layer: m3, field: catalog_drift_pressure}
```

Why not rename:

- Renaming the field channel `pressure` is a definition change on a generic
  capability channel with 19 generic whole-vector consumers. No.
- Renaming the row id changes the Redis state key. The new key cold-starts for
  24 h (`min_samples` 2880), thresholds fall back to static 0.25 instead of the
  learned ~0.148, and the Hub pressure gate and the mind recall line both
  decide differently for a day. That is a live behavior change for a cosmetic
  gain.
- Registering `bus_synaptic_pressure` as its own metric would hand-author a URN
  for an alias, which the lineage tool explicitly refuses.

What enforces it: `tests/test_transport_lattice_policy_sources.py` (new, in
`orion-static-gates` CI) fails if a row has no `source`, points at a capability
channel nothing in the topology writes, at a channel with no glossary entry, or
at a `TransportBusStateV1` field that does not exist or is retired. It also
pins `bus_synaptic_pressure -> capability:transport.pressure`, fed only by
`node:substrate.bus_synaptic` `prediction_error`. The existing
`test_scale_weight_matches_topology_edge` keeps pinning the 0.85.

### Decision D2: fix the mind recall line's scale

The ladder is divided by the shared 0.85 (`DERIVED_CHANNELS`) before printing,
so the sentence compares like with like ("against a 0.29 watch threshold" for
the static 0.25). The render gate itself (0.15 on raw `prediction_error`) is
unchanged, so whether the line appears does not change; only the threshold
numbers inside it do.

## B. `contract_pressure` sampled two world_pulse streams

### Metric gate (CLAUDE.md 0A)

1. **Provenance.** `services/orion-bus/app/bus_observer.py` took an
   `XREVRANGE COUNT 5` sample of each `BUS_OBSERVER_STREAMS` key that is in the
   catalog with a `schema_id`, validated it, and emitted
   `bus_schema_validation_failed`. `orion/substrate/transport_loop/extract.py`
   turned that into `contract_pressure = mismatched_streams / streams_observed`,
   per bus (M3), injected into `node:athena.contract_pressure` (M4 node level).
2. **Independence.** Independent of catalog drift. But note
   `capability:transport.contract_pressure` is **not** this value: the topology
   fills it from `catalog_drift_pressure` (x 0.85).
3. **Theory anchor.** "Producers send payloads that break their declared
   contract" is a real failure. The anchor is fine; the instrument is not.
4. **Live data (can it read calm and non-calm?).**
   - Stored grammar events, 72 h: 24,570 observer ticks, **zero**
     `bus_schema_validation_failed` atoms.
   - Field, 72 h: `node:athena.contract_pressure` = 0.0 on **123,412 of
     123,412** ticks (1 distinct value).
   - Every retained `orion:stream:world_pulse:run:result` entry (172): 172
     valid, 0 invalid.
   It has never left calm.
5. **Existing mechanism.** `OrionBusAsync.publish()` already validates every
   payload against its catalog `schema_id` and raises before sending
   (`_validate_payload`). So:
   - A mesh-wide receiver-side sample of Pub/Sub traffic reads calm by
     construction. Measured: 120 s `PSUBSCRIBE *`, 9,498 cataloged messages
     on 125 channels, **0 mismatches, 0 decode failures** (24 uncataloged,
     already counted by catalog drift).
   - The publish-side reject itself is the real event. Container logs, 72 h,
     every running container: **0** "Payload validation failed" lines.
   - Streams (the one path `publish()` does not cover): 5 on the whole bus,
     one written in the last 21 days (world_pulse, 13 h ago).
6. **Reversibility.** Cheap: one sampler, one atom, two projection fields, one
   node channel, one policy row, one Hub gate.

### Decision D3: retire it end to end; keep the capability channel

No honest wider version exists today: the widened check is calm by
construction, and the honest event (publish-time rejects) happened zero times
in 72 h. Kill means kill:

| Layer | Removed |
|---|---|
| Producer (orion-bus) | `count_schema_mismatches`, `load_channel_catalog_schema_ids`, the XREVRANGE loop, `record_schema_mismatch` / `bus_schema_validation_failed` atom, `uncertainty_from_sample_mismatch`, `BUS_OBSERVER_SCHEMA_SAMPLE_COUNT` (settings, `.env_example`, compose, README, AGENT_CONTEXT, SERVICE_PORTS, trace map) |
| Reducer (M3) | `TransportBusStateV1.contract_pressure`, `.schema_mismatch_stream_count` (dropped on read via `RETIRED_TRANSPORT_BUS_STATE_FIELDS`); `contract_pressure` pressure_hint; atom role now ignored |
| Field (M4 node) | `node:athena.contract_pressure`: removed from `NODE_CHANNELS`, node decay, topology `node_channels`, ingest; pruned from every node via `RETIRED_NODE_CHANNELS` |
| Consumers | relational transport adapter salience/metadata; substrate-runtime incident log; Hub M3 card |
| Lattice policy + Hub | `contract_pressure` row; Hub contract gate |
| Glossary | entry narrowed to capability level with the truthful meaning |

**Kept on purpose:** `capability:transport.contract_pressure`. It is catalog
drift (0.85 x `node:athena.catalog_drift_pressure`) under a misleading name.
Removing or renaming it is a live decision change, so it is not done here:

- Live 72 h (10% sample, 13,134 ticks): it is the largest entry of
  capability:transport's attention pressure proxy (`_current_pressure_proxy`,
  a max over channels) on **314 ticks (2.4%)**, by a mean of 0.0018 (max
  0.0136). Dropping it lowers that proxy on those ticks; novelty for
  capability:transport is computed from the proxy.
- Its range: 7 distinct values, 0.0027 to 0.0191 (catalog_size 312,
  undeclared 1 to 7).

Options for Juniper (not implemented):

- (a) Leave as is (current). Mislabel documented in the glossary.
- (b) Rename to `catalog_drift_pressure` at capability level: same values,
  same attention effect, honest name; definition change plus field reconcile
  (`RETIRED_CAPABILITY_CHANNELS` with a successor).
- (c) Drop the `catalog_drift_pressure -> contract_pressure` map: catalog drift
  stays on node:athena only; capability:transport attention proxy loses its
  floor on ~2.4% of ticks.

Recommendation: (b). It is the only option that ends the vocabulary bug
without changing any value.

## Consumer impact

| Consumer | Change | Over/under-weights transport? |
|---|---|---|
| Field attention (host targets, `max` proxy) on node:athena | loses a channel that was always 0.0 | No change (max unaffected) |
| Field attention (capability:transport) | untouched (capability channel kept) | No change |
| `collect_field_channel_pressures` / commensurability | node entry 0.0 gone, capability entry still wins | No change |
| Endogenous curiosity, field coherence, decay, feedback `pressure_delta` | 0.0 key gone | No change |
| mood_arc encoder fit | zero-variance channel already dropped by `select_fields`; absent = 0.0 by convention | No change |
| Relational transport adapter salience | `max(reliability, contract)` -> `reliability` | No change (contract was 0.0) |
| Substrate-runtime incident log | field gone | No change (never logged; always 0) |
| Proposals (`dimension_confidence`) | none (capability channel never routed to a dimension) | No change |
| Hub pressure gate | reads M4 via the policy `source` | No change in value |
| Hub contract gate | deleted | n/a (display only; read catalog drift) |
| Hub Lattice Values / simulator | `contract_pressure` row gone (always read M3 0.0) | Simulator can no longer promote on it (it never did live) |
| Mind recall line | ladder numbers converted to the stated scale | Render decision unchanged; stated thresholds now ~1.18x the old printed numbers (correct) |
| EWMA thresholds (Redis) | key unchanged | No change; no cold start |

## Semantic layer

- `config/field/field_channel_glossary.v1.yaml`: `contract_pressure` now
  `level: [capability]` with the real meaning. Definition lock re-generated
  (1 `semantics_changed`, 1 `annotation_changed`, both this entry).
- `make check-metric-lineage-gate`: PASS (no new orphan).
- No new URN registered for `bus_synaptic_pressure` (it is a row id).

## Acceptance checks (after deploy)

1. No new `bus_schema_validation_failed` rows in `grammar_events`.
2. The live `substrate_transport_bus_projection` row has no
   `contract_pressure` / `schema_mismatch_stream_count` after one reducer tick.
3. No `node_vectors` entry carries `contract_pressure` after one digester
   tick; `capability_vectors['capability:transport'].contract_pressure` still
   present and nonzero.
4. Hub `/api/substrate-lattice/transport/gates` has no `contract` gate;
   `/transport/latest` lattice rows show `value_source` "M4
   capability:transport.pressure" for `bus_synaptic_pressure`.
5. Redis `orion:lattice:transport_thresholds:v1` still has the
   `bus_synaptic_pressure` field with n still growing.

## Follow-ups (not done here)

- `BUS_OBSERVER_STREAMS` now only feeds `bus_configured_stream_uncataloged`
  and the census-off fallback of `catalog_drift_pressure`. The census ran on
  24,570 of 24,570 ticks in 72 h, so the fallback is dormant; it is the next
  instrument to judge.
- If contract violations need watching, count `publish()` rejects at the
  source (where they actually happen), not a receiver sample. Zero in 72 h
  today, so not built.
- Hub pressure gate still compares M4 `reliability_pressure` against the
  `observer_failure_pressure` row's threshold (labelled in its reason).
