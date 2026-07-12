# Metrics swamp arsonist review — 2026-07-12

Context: deciding whether to delete `orion-self-state-runtime`, which surfaced a
bigger problem — multiple competing/overlapping "metrics" systems (self-state,
AutonomyV2, phi/EKG, drives) with unclear ownership of what's real vs. garbage-in.
This is a snapshot from live `.env` files and git log, not `.env_example` defaults
or stale memory — some conclusions from earlier in this investigation were already
out of date by the time this was written.

## Arsonist summary

There are **two independently-computed drive-pressure vectors over the same 6-key
taxonomy** (`coherence, continuity, capability, relational, predictive, autonomy`),
a **5-service dry-run action ladder** that has never executed a real action, **two
unrelated systems both named "endogenous"**, and at least one decision already made
(kill concept-clustering) that was never actually applied to the live `.env`. This
is a legibility problem as much as a correctness one.

## Current architecture (verdict table)

| System | File(s) | Live? | Fed by | Reaches real behavior? | Verdict |
|---|---|---|---|---|---|
| **SelfStateV1** | `orion-self-state-runtime` | ✅ `ENABLE_SELF_STATE_RUNTIME=true` | attention frames + field state | Yes — root sensor for everything below | **KEEP** (foundation, not a metric to prune) |
| **Substrate ladder L7–L11** (proposal→policy→execution-dispatch→feedback→consolidation) | `orion-{proposal,policy,execution-dispatch,feedback,consolidation}-runtime` | ✅ all `ENABLE_*=true`, polling every 2–60s | self_state | `EXECUTION_DISPATCH_MODE=dry_run` — **no layer has ever mutated the real world**. Only confirmed consumer past L9 is `orion-dream/compaction_applier.py` and Hub debug tiles. | **UNVERIFIED whether consolidation output changes anything real** — flag for a live check before deciding; strong burn/merge candidate if it's rehearsal-only |
| **AutonomyStateV2** | `orion/autonomy/reducer.py`, `evidence_compiler.py` | ✅ persisted, wired into `stance_react.j2` + `harness/prefix.py` | chat-turn evidence only (user_message, social hazard, reasoning quality) — **test `0d3b21d5` explicitly bans importing self_state/phi** | **Yes, proven** — reaches the LLM prompt and harness prefix every turn | **KEEP** — this is the one system with proven behavioral reach |
| **DriveEngine** | `orion/spark/concept_induction/drives.py` | ✅ live, real bus traffic | self_state tensions + biometrics/mesh/spark_signal/failure_event via `signal_drive_map.yaml` + homeostatic leaky-decay consumer | Produces persisted `GoalProposalV1`, but the only two real gating consumers (`capability_policy.py`, `world-pulse curiosity.py`) **hardcode `drive_origin="predictive"`**, discarding the per-drive computation entirely | **MERGE candidate** — output nuance is thrown away downstream; unification already scoped |
| **`signal_drive_map.py`** (shared taxonomy) | `orion/autonomy/signal_drive_map.py`, `config/autonomy/signal_drive_map.yaml` | ✅ imported by *both* `reducer.py` and `bus_worker.py` | — | This is the seam where the AutonomyStateV2/DriveEngine merge has **already started** (commits `10fc1a12`, `c250d53a` — not "design only" anymore) | Each still keeps its own local `drive_pressures` dict — **not yet one shared store.** Needs a side-by-side numeric comparison before calling it done |
| **Concept extraction/clustering** | `bus_worker.py` clustering path | ⚠️ `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=true` right now | noun-chunk salience, no stopword/POS filter | Already decided to kill this (2026-07-11) — **decision was never applied to the live `.env`** | **BURN — just execute the decision already made** |
| **phi encoder** | `orion-spark-introspector`, `ORION_PHI_ENCODER_ENABLED=true` | ✅ live, seedv4 weights | self_state | Feeds Hub EKG | **KEEP**, but "garbage-in" fix was relaxed-threshold, not re-verified live since 2026-07-11 |
| **Tissue-viz novelty/arousal** | Hub EKG | ✅ | hardware evidence extraction | Feeds Hub UI | **KEEP**, recently fixed, disclosed limitation on arousal |
| **Homeostatic leaky-decay drives** | `signal_tension.py`, `ORION_HOMEOSTATIC_DRIVES_ENABLED=true`, `ORION_DRIVE_LEAKY_MATH_ENABLED=true` | ✅ live | biometrics/mesh/failure signals | Feeds DriveEngine pressures | **KEEP** — legitimate sensor-to-drive translation, recently repaired |
| **Scarcity economy v2 (`resource_pressure`)** | per 2026-07-11 findings | ⚠️ unverified today | infrastructure contention | Saturated at 1.0 constantly per last measurement = zero information content | **Likely BURN or redesign** — needs a live re-check, not a code check |
| **`endogenous_origination.py`** | `orion/autonomy/` | `ORION_ENDOGENOUS_ORIGINATION_ENABLED` unset anywhere live (defaults off) | self_state | 2026-07-08 gate measurement returned **NO-GO on both required verdicts**, never committed to docs | **BURN** — proven not to work by its own data gate; the verdict is sitting uncommitted in `/tmp/autonomy-gate/report.md` |
| **`ENDOGENOUS_RUNTIME`** (different thing, same word) | `services/orion-cortex-exec/app/endogenous_runtime.py` | ✅ `ENDOGENOUS_RUNTIME_ENABLED=true`, surfaces to chat + operator | — | Unrelated to the NO-GO system above, "phase 8 adoption" doc suggests it's actually shipped | **Rename one of these two "endogenous" things before doing anything else** — this exact naming collision is a likely source of confusion |

## Missing questions (need live data, not more grep)

1. Does `orion-dream/compaction_applier.py` actually change anything downstream
   when it reads policy/execution frames, or does the whole L7–L11 ladder
   terminate in a table nobody outside Hub debug ever looks at? This is the
   single biggest "burn 5 services" candidate and was not confirmed either way.
2. Is `resource_pressure` still pinned at 1.0 today, or did the homeostatic-drive
   fixes touch it incidentally?
3. Are DriveEngine's and AutonomyStateV2's independently-computed
   `drive_pressures` numerically converging or diverging on the same live
   traffic? If they already roughly agree, the merge is cheap. If they diverge,
   need to decide which one is *right* before merging.
4. Two `identity_snapshot` producers exist (self-state-runtime's
   `_maybe_emit_identity_snapshot` every 10 ticks, and concept-induction's
   `BUS_IDENTITY_SNAPSHOT_OUT`) — not yet traced whether these are duplicates or
   serve different consumers.

## Recommended next patch (smallest useful slice, not a cathedral)

Don't try to resolve the whole table at once. In order:

1. **Apply decisions already made** (zero new design risk): flip
   `CONCEPT_AUTONOMOUS_TRIGGER_ENABLED=false` live, commit the
   endogenous-origination NO-GO verdict into its spec, delete or clearly
   deprecate `endogenous_origination.py` with the verdict as the commit message.
2. **Rename the endogenous collision** — `endogenous_runtime.py` vs
   `endogenous_origination.py` is exactly a keyword-cathedral trap; costs
   nothing, removes a recurring confusion.
3. **Answer missing question #1 (dream consumption)** with a live trace before
   deciding whether L7–L11 is theater — this determines whether the actual
   burn scope is 5 services or 0.
4. Only after that: decide the DriveEngine ↔ AutonomyStateV2 merge direction
   using missing question #3's numeric comparison.

## Addendum: mesh/multi-node weighting swamp (found 2026-07-12, same day)

Same pattern, different layer. The mesh (Athena = orchestration/hub/postgres/self-state
host, Atlas = small inference node, Circe = monster burst-inference node, both Atlas/Circe
running only `llamacpp-host`) already has **three independent, disconnected weighting
schemes** for "how much should each node's health count," plus a fourth service nobody
in this investigation had previously touched:

| System | Weights | Live? | Connected to Layer 6? |
|---|---|---|---|
| Field-topology edge weights (`config/field/orion_field_topology.v1.yaml`) | `atlas→llm_inference: 0.85`, `circe→llm_inference: 0.50`, `athena→orchestration: 0.90`, `athena→transport: 0.85`, `athena→storage: 0.75`, `athena→graph: 0.70` | ✅ genuinely applied in `apply_diffusion` (`services/orion-field-digester/app/digestion/diffusion.py:12`), not just declared | **No** — Layer 6's `collect_field_channel_pressures` (`orion/self_state/scoring.py:54`) reads raw `node_vectors` *and* already-weighted `capability_vectors` together via `.values()`, discarding node identity and double-counting |
| `CLUSTER_ROLE_WEIGHTS={"atlas":0.7,"athena":0.3,"other":0.5}` (`services/orion-biometrics/.env`) | Opposite emphasis from the topology edges, undocumented reasoning | ✅ live, feeds `orion:biometrics:cluster` | No — feeds `orion-hub`/`orion-state-service` only |
| `orion-state-service`'s own aggregation ("per-node / global — most-recent wins, or authoritative node if configured") | Undefined/config-dependent | Unverified how live | No |
| `node_catalog.yaml` roles | Athena: hub/redis_bus/graphdb/postgres/orchestration all `true`. Atlas: `inference_gpu`. Circe: `burst_gpu`, `expected_online: false` | ✅ | Partially — `expected_offline_suppression` is already a stabilizing channel in `self_state_policy.v1.yaml` (weight 0.30), so Circe's normal-off state is already handled gracefully at L6 |

**Key finding:** the topology edges already encode the right idea (Athena = seat of
cognition/orchestration, Atlas/Circe = inference limbs, weighted accordingly) and the
diffusion math genuinely uses it. The gap isn't "we need to invent node weighting" — it's
that Layer 6 currently can't see it, because it flattens `node_vectors` and
`capability_vectors` into one anonymous pool. Turning on biometrics for Atlas/Circe today
would make the existing `max()`-saturation problem worse (raw node pressure competing
unweighted against its own already-weighted diffusion output for the same dimension).

**New verdict entries:**

| System | Verdict |
|---|---|
| Field-topology edge weights + diffusion | **KEEP — this is the real mechanism.** Layer 6 should read `capability_vectors` only, never raw `node_vectors`, to stop double-counting and to inherit correct per-node weighting for free. |
| `CLUSTER_ROLE_WEIGHTS` / `orion-state-service` aggregation | **Needs the same live-consumption trace as L9-L11** — determine whether this is a legitimate separate "ops health for Hub" concern or a fourth duplicate of the same question, before deciding merge vs. keep-separate. |
| Node-attributed provenance in `dominant_evidence`/`reasons` | **Missing, worth building** — currently a channel's source node is available at the diffusion step (`edge.source_id`) but not threaded through to the self-state evidence fields, so nothing downstream can say "reasoning_pressure is high because Circe is down." |
| Capacity-pressure vs. continuity-threat distinction | **Genuinely open design question, not a wiring gap.** Athena going down threatens Orion's ability to exist (hub/bus/postgres/self-state itself live there); Circe going down (or being asleep, which is normal) just means less burst inference. Nothing in the current 0-1 pressure scale distinguishes a difference in *kind* from a difference in *degree*.
