# orion/spark — Spark: Orion's felt-state tissue layer

> This README was rewritten 2026-07-10 after most of it (Spark verbs, `OrionState`,
> knobs & policies) turned out to describe a system that was never built. What
> follows is grounded in the actual code, traced live.

## What's actually here

`orion/spark/` has two genuinely separate things living under one name:

1. **The tissue/φ layer** (`orion_tissue.py`, `signal_mapper.py`,
   `surface_encoding.py`, `introspection_metadata.py`) — computes and
   broadcasts Orion's felt-state stats (`valence`, `energy`/arousal,
   `coherence`, `novelty`) consumed by Hub's tissue-viz display. This
   README covers that layer.
2. **`orion/spark/concept_induction/`** — a separate, actively-developed
   subsystem (drives, goals, concept graph materialization) wired into
   `orion-cortex-orch`, `orion-cortex-exec`, and `orion/autonomy/`. Not
   covered here — see its own phase notes in that directory.

## Canonical φ: self-state-derived, not the tissue tensor

The live φ (`valence`/`energy`/`coherence`/`novelty`) broadcast to Hub's
tissue-viz comes from `_phi_from_self_state()` in
`services/orion-spark-introspector/app/worker.py`, computed from the
`SelfStateV1` payload published by `orion-self-state-runtime`
(`orion/self_state/builder.py` + `orion/self_state/scoring.py`).

This is the **canonical** source. It's read via
`_get_phi_stats()` in `worker.py`, which every `tissue.update` broadcast
site uses.

**Known limitation, disclosed not solved**: several `SelfStateV1` dimensions
this depends on have their own liveness problems (structurally pinned,
aggregation-saturated, or hardcoded). `energy`/arousal and `novelty` were
fixed in 2026-07-10 to bypass the worst of these (see the fix history in
`services/orion-spark-introspector/tests/test_tissue_viz_arousal.py` and
`test_tissue_viz_novelty.py`), but `coherence`, `policy_pressure`, and
others are still theater at the `SelfStateV1` level. A full accounting is
in `docs/superpowers/specs/2026-07-10-cognition-metric-lineage-registry-design.md`.

## `OrionTissue`: fallback only, not canonical

`orion_tissue.py::OrionTissue` is a persistent 16×16×8-ish tensor with real
decay+diffusion physics. It's genuinely live-fed: every chat embedding
(`handle_semantic_upsert` in `worker.py`) calls `TISSUE.propagate()`, which
updates the tensor and periodically snapshots it to disk
(`ORION_TISSUE_SNAPSHOT_PATH`, defaults to
`/mnt/graphdb/orion/spark/tissue-brain.npz` in production).

But `TISSUE.phi()` — its own valence/energy/coherence/novelty computation —
is **only read** by `_get_phi_stats()` when `_LATEST_SELF_STATE is None`,
i.e. cold start or a self-state-runtime outage. In steady-state operation
this is essentially never. The tensor gets real physics updates from real
chat activity; almost nothing reads its output. This was the system
Juniper's earlier "was theater, mostly phased out" memory refers to — it's
accurate. It's not dead code (still feeding a real resilience path), but
it is not where the canonical numbers come from.

**If you're adding a new consumer of φ, use `_phi_from_self_state()` /
`_get_phi_stats()`, not `OrionTissue.phi()` directly.**

## The "heartbeat" is a different system, and it's real

There's a periodic cadence that triggers φ recomputation even without new
chat activity. It does **not** come from `OrionTissue` or anything in this
directory — it's `services/orion-equilibrium-service`, which:

1. Reads `substrate_self_state` / `substrate_execution_trajectory_projection`
   directly from Postgres (via `orion/substrate/felt_state_reader.py`,
   written by `orion-self-state-runtime/app/store.py`) to score
   "eventfulness."
2. Publishes `CognitionTracePayload(verb="equilibrium_heartbeat")` on
   `orion:cognition:trace` roughly every `EQUILIBRIUM_COLLAPSE_MIRROR_INTERVAL_SEC`
   (~15s), or sooner if substrate activity is scored "dense."

`worker.py::handle_trace()` consumes that event (`_is_heartbeat_trace()`),
and on a heartbeat calls the same `_get_phi_stats()` used everywhere else
to compute and broadcast φ. The heartbeat is a **trigger for when to
recompute/broadcast**, not an alternate **source** of φ — it still goes
through the same self-state-first, tissue-fallback path described above.

## Removed 2026-07-10: `spark_engine.py`, `integration.py`, `strategies.py`

These formed a third, fully independent φ implementation
("Spark Engine Facade," meant to be "the API that Hub, Brain, Cortex, Dream
Engine, etc. can call"). Grepped the whole repo on `main`: zero production
consumers. Two abandoned worktrees showed a partial, never-merged attempt
to wire it into `orion-llm-gateway`. Deleted rather than left to rot
further or accidentally get wired in as a second source of truth.

## Contract (channels / schemas — verified against `orion/bus/channels.yaml`)

The canonical Spark telemetry and snapshot schemas live in
`orion/schemas/telemetry/spark.py`. Services should import Spark schema
classes from that module (or a compatibility shim that re-exports it) to
avoid drift across duplicate definitions.

Spark contract surfaces in use:

* **Telemetry**: channel `orion:spark:telemetry`, kind `spark.telemetry`, payload `SparkTelemetryPayload`.
* **State snapshot**: channel `orion:spark:state:snapshot`, kind `spark.state.snapshot.v1`, payload `SparkStateSnapshotV1`.
* **Snapshot ACK (reply-only)**: kind `spark.state.snapshot.ack.v1`, payload `SparkStateSnapshotAckV1`.
* **Candidate**: channels matching `orion:spark:introspect:candidate*`, kind `spark.candidate`, payload `SparkCandidateV1`.
* **Signal**: channel `orion:spark:signal`, kind `spark.signal.v1`, payload `SparkSignalV1`.
* **Concept induction outputs**: `orion:spark:concepts:profile` (`memory.concepts.profile.v1`) and
  `orion:spark:concepts:delta` (`memory.concepts.delta.v1`).

Legacy kinds `spark.introspection.log` and `spark.introspection` are deprecated and should not be emitted.

## What this README no longer claims

Prior drafts described `spark.introspect` / `spark.debug` /
`spark.theme_weaver` / `spark.weekly_digest` verbs, an `OrionState` JSON
blob (`mood.curiosity`/`strain`/`uncertainty`), and a "knobs & policies"
config surface. Grepped for all of these across the repo: no hits outside
this file. None of it was ever implemented. If this direction is still
wanted, it needs a design spec with real schema/producer/consumer seams
(per `AGENTS.md` 0A) — not a doc describing it as if it already exists.
