# Self Tab: Substrate Brain-EKG — Design

- Date: 2026-07-07
- Status: Approved for planning (revised after arsonist pass)
- Service (primary): `orion-hub` (viz) + `orion-substrate-runtime` (producer)
- Contract touch: `orion/schemas/registry.py`, `orion/bus/channels.yaml`

## Arsonist summary

The Hub "Self" tab is currently a static, 30s-polled, four-card snapshot served by `GET /api/substrate/observability/summary`. It names Orion's self-model but shows almost none of it and cannot show motion over time. Meanwhile the substrate is a live, ticking cognitive system — a graph of activation/pressure nodes, an attention-broadcast coalition selector, reducer ingestion lanes with health, a 13-dimension self-state vector, and an M3→L11 processing lattice — none of which the Self tab renders as the living organ it is.

This design turns the Self tab into a **realtime diagnostic instrument for runtime intuition, shared by Orion and the operator**: a brain-shaped visualization where substrate "centers" fire and starve, with historical playback, driven by one compact per-tick contract. It is deliberately not a re-skin of the spark "Cognitive EKG" (`/spark/ui`) and not a scatter of pre-existing embeddings.

## Purpose and payoff

The tab is a **sensory instrument for runtime**, not an action console. The concrete payoff (operator-stated): drive a known load — e.g. a deep, tool-heavy chat turn — and *watch the right centers light up*, building an intuitive mental model of how the substrate responds. It serves two viewers through one representation:

- **Operator instrument (primary):** Juniper triggers activity and reads the response — "I sent a heavy tool turn and the execution lane + reasoning/execution pressure lit while concept stayed dim." That legible stimulus→response is the point.
- **Orion's introspective mirror:** the same surface is what Orion's mind looks like to itself.
- **Felt-sense layer:** self-state vitals give a slow read of load/preoccupation over time.

**Explicitly not optimizing for "action levers" yet.** We do not know which knobs the operator will pull in response, and that is acceptable. The instrument earns its keep by making runtime *legible*, not by wiring decisions. This is a deliberate YAGNI boundary, not an oversight.

### Named stimulus→response cases (payoff, and the basis for the live acceptance check)

1. **Heavy tool-using chat turn** → `execution_trajectory` lane firing, `execution_pressure` / `reasoning_pressure` self-state dims rising, transport lane active; `concept`/`ontology_branch` node-kinds stay dim.
2. **Idle / no input** → lanes quiet, dwell rising on a stable coalition, most node-kinds decaying toward `starving`.
3. **Contradiction/tension injected** → `tension`/`contradiction` node-kinds fire, coherence dim drops, spotlight shifts onto the tension cluster.

If, watching a live turn, these do not visibly happen, the instrument has failed regardless of whether it renders. That is the real bar (see Acceptance checks §6).

## Owned scope decision: pre-named regions before emergent decomposition

The original ask included "decomposed like PCA… labeled regions." This design ships **pre-named parcellations (node-kinds, lanes, self-state, lattice) in Phase 1** and defers **emergent PCA/cluster decomposition to Phase 2**. This is a deliberate reorder, called out so it can be vetoed:

- The stated purpose is *recognition* ("execution lit up"), which pre-named regions serve directly. Emergent clusters produce unnamed blobs that need interpretation — worse for building intuition early.
- PCA needs a corpus of retained frames to be meaningful; Phase 1's log *is* that corpus. So Phase 1 is a prerequisite for a good Phase 2 decomposition anyway.
- Emergent mode returns in Phase 2 as an additional toggle, not a replacement.

## Current architecture

- **Self tab (inline pattern):** `services/orion-hub/templates/index.html` §`#self-observability` (~L1345) + `static/js/self_observability.js` (polls every 30s). Backed by `scripts/substrate_observability_routes.py` → `GET /api/substrate/observability/summary` (already returns self_state, attention_broadcast, curiosity, reverie, hub_presence; UI renders only 4).
- **Standalone/iframe pattern (precedent to follow):** `#substrate-lattice` iframes `/static/substrate-lattice.html?v={{HUB_UI_ASSET_VERSION}}`; logic in `static/js/substrate-lattice.js`; API in `scripts/substrate_lattice_routes.py`, registered via `router.include_router(...)` in `scripts/api_routes.py`. Static mount in `scripts/main.py`.
- **Substrate runtime cadences** (`services/orion-substrate-runtime/app/settings.py`): `grammar_poll_interval_sec=5`, `dynamics_tick_interval_sec=30`, `attention_broadcast_interval_sec=30`, `endogenous_curiosity_tick_interval_sec=60`, `episodic_tick_interval_sec=300`. Loops in `app/worker.py`; dynamics in `orion/substrate/dynamics.py`; coalition in `orion/substrate/attention_broadcast.py`; grammar/reducer health in `app/grammar_truth.py`.
- **Self-state runtime** (`services/orion-self-state-runtime`): polls every 2s but each tick is gated on a new attention frame (`source_field_tick_id`), so self-state effectively moves on the ~30s attention clock. Produces `SelfStateV1` (`orion/schemas/self_state.py`), latest-only.
- **Signal families:** node kinds (11, activation/pressure/dormancy via `orion/core/schemas/cognitive_substrate.py`); reducer lanes (`biometrics`, `chat_grammar`, `execution_trajectory`, `transport_bus`, plus `node_pressure`/`biometrics_node` reducers) with cursor lag/backlog/quarantine/heartbeat; self-state (13 dims); lattice M3→L11 (transport lane today); attention coalition (`AttentionBroadcastProjectionV1`).
- **History retention:** most projections are **latest-only** (`ON CONFLICT DO UPDATE`). Per-tick series exist only for `substrate_coalition_dwell_log` (24h) and `substrate_endogenous_curiosity_candidates` (24h). Multi-dimension playback has **no backing store today**.

### Current gap

Realtime is cheap (slow cadence). Historical playback across the wanted dimensions is impossible today (projections overwrite each tick). The tab shows a fraction of an available payload and no motion.

## Chosen approach

**One typed frame as the backbone.** On a dedicated cadence, `orion-substrate-runtime` assembles a compact `SubstrateBrainFrameV1`, publishes it to a new bus channel, and appends it to a bounded history log. The Hub reads the log for realtime (tail) and playback (range) through one path.

Rejected: read-only over latest-only projections (playback half-broken); full analytical warehouse with per-node history + server-side UMAP (cathedral).

### Design spine (post-arsonist)

- **Regions are the contract's spine.** `intensity`/`state` per region are the stable, continuous signals that drive the EKG and playback. They are computed from real substrate activity, never fabricated.
- **Node/edge samples are best-effort decoration.** The sampled top-K nodes/edges make the interior feel alive but carry **no continuity guarantee** — a node may leave the sample when it cools. Nothing (EKG, playback, tracking) depends on a node persisting across frames. The frontend treats them as ambient, not trackable series.
- **Frame cadence is decoupled from source cadence.** Fast signals (lanes) move; slow signals (attention/self-state) are shown as held steps via per-region `as_of` + `stale`, never fake-jittered.
- **Cold start is explicit.** Before the graph has real activation, the producer emits frames with `phase="warming"` (or none), never an all-zero frame masquerading as a live dead brain.

## Proposed schema / API changes

### New schema: `SubstrateBrainFrameV1`

Location: `orion/schemas/brain_frame.py`, registered in `orion/schemas/registry.py`. Compact/bounded.

```
SubstrateBrainFrameV1:
  frame_id: str                 # deterministic: hash(generated_at + tick_seq)
  generated_at: datetime
  tick_seq: int                 # monotonic per-run frame counter
  phase: Literal["warming","live"]
  source: str = "orion-substrate-runtime"

  regions: list[BrainRegionV1]     # the spine — all parcellation dimensions
  spotlight: BrainSpotlightV1 | None
  nodes: list[BrainNodeSampleV1]   # decoration, no continuity guarantee (default K=40)
  edges: list[BrainEdgeSampleV1]   # decoration (default K=60)

BrainRegionV1:
  dimension: Literal["node_kind","lane","self_state","lattice_layer"]
  region_id: str                # "node_kind:tension", "lane:execution_trajectory"
  label: str
  intensity: float              # 0..1 normalized activity
  state: Literal["firing","steady","starving"]
  node_count: int
  as_of: datetime               # when this dimension's source last actually moved
  stale: bool                   # true = held value (source slower than frame cadence)
  detail: dict[str, float]      # dim-specific extras (lag_sec, score, confidence)

BrainSpotlightV1:
  attended_node_ids: list[str]
  dwell_ticks: int
  coalition_stability: float
  description: str | None
  as_of: datetime
  stale: bool

BrainNodeSampleV1:
  node_id: str
  node_kind: str
  activation: float
  pressure: float
  dormant: bool
  label: str                    # non-private display label only

BrainEdgeSampleV1:
  src: str
  dst: str
  weight: float
```

Normalization/state thresholds live in producer settings, not hard-coded. `stale` is set when `frame.generated_at - region.as_of` exceeds that dimension's known cadence.

### New bus channel

`orion/bus/channels.yaml`: `orion:substrate:brain_frame` → payload `SubstrateBrainFrameV1`, producer `orion-substrate-runtime`, consumers `orion-hub` (+ future).

### New storage: `substrate_brain_frame_log`

Append-per-frame, bounded (mirror `save_coalition_dwell`):

```
substrate_brain_frame_log(
  frame_id text primary key,
  tick_seq bigint,
  generated_at timestamptz,
  phase text,
  frame_json jsonb,
  created_at timestamptz
)
-- prune: DELETE WHERE generated_at < now() - interval '<retention>'
```

**Cadence & volume (pinned):** frame emission on dedicated `BRAIN_FRAME_INTERVAL_SEC` (default **5s**, matching the fastest meaningful signal / diagnostic responsiveness). Retention default **24h** (`BRAIN_FRAME_RETENTION_HOURS=24`) → ~17,280 rows/day; at a few KB of `frame_json` each, ~50–150MB. Both cadence and retention are env levers. `warming` frames are not retained beyond the live window.

### New Hub API: `scripts/self_brain_routes.py` (prefix `/api/self-brain`)

- `GET /frames/tail?limit=N` (default 1, max ~120) → most-recent N ascending. Realtime feed (browser polls ~3s).
- `GET /frames/range?from=<iso>&to=<iso>&max=<n>` → playback window, downsampled to `max` (default 240).
- `GET /window` → retention bounds + earliest/latest frame timestamps + current `phase` (for the scrubber + a "warming" banner).
- Phase 2: `GET /clusters?from=&to=` → on-read decomposition over the frame window.

Registered in `scripts/api_routes.py`. Read-only; degrades to empty-with-200 when no frames.

## Files likely to touch

- `orion/schemas/brain_frame.py` (new); `orion/schemas/registry.py` (register); `orion/bus/channels.yaml` (channel).
- `services/orion-substrate-runtime/app/brain_frame_producer.py` (new); `app/worker.py` (dedicated `_brain_frame_loop`; assemble + publish + append); `app/store.py` (`save_brain_frame`, `load_brain_frames_tail`, `load_brain_frames_range`, prune); `app/settings.py` + `.env_example` (cadence/retention/sample-K/threshold + `warming` gate).
- `services/orion-hub/scripts/self_brain_routes.py` (new) + `scripts/api_routes.py` (register).
- `services/orion-hub/static/self-brain.html` (new); `static/js/self-brain.js` (new).
- `services/orion-hub/templates/index.html`: replace `#self-observability` body with a slim iframe to `/static/self-brain.html`; keep the four readouts as a side rail fed by frame; retire old inline card JS as needed.
- Tests + evals in both services.

## Data flow

```
brain-frame loop (runtime, every BRAIN_FRAME_INTERVAL_SEC=5s)
  -> assemble SubstrateBrainFrameV1
       regions: node_kind + lane (from live dynamics + grammar health, as_of=now)
                self_state (latest substrate_self_state row, as_of=its generated_at, stale if old)
       spotlight: latest attention broadcast (as_of=its generated_at, stale if old)
       nodes/edges: sampled top-K (decoration)
       phase: "warming" until graph has real activation, else "live"
  -> publish orion:substrate:brain_frame
  -> append substrate_brain_frame_log (+ prune)

hub browser (self-brain.js in iframe)
  realtime: poll GET /api/self-brain/frames/tail?limit=1 every ~3s
  playback: GET /api/self-brain/frames/range?from&to on scrub
  both read the same log -> one render path
  -> render brain (Phase 1: hybrid C) + dimension toggles + EKG + scrubber
     stale dims render as held/hatched, never as movement
```

## Frontend design

- **Embedding:** slim `#self-observability` section iframes `/static/self-brain.html?v={{HUB_UI_ASSET_VERSION}}`. Heavy DOM/JS stays in the standalone file.
- **Phase 1 view — Hybrid (C) only:** fixed, always-labeled anatomical zones (stable/trackable across playback) with living, pulsing nodes + edges inside; coalition spotlight as a dashed hull that can span zones. (Organic constellation **B** moves to Phase 2 — it is a second layout engine, not "just a toggle.")
- **Dimension toggle rail:** `node-kinds | lanes | self-state | spotlight` re-parcellate the same frame (`spotlight` is an overlay). `layers M3–L11` toggle arrives with Phase 2.
- **EKG strip:** per-selected-**region** intensity over the loaded window (regions are the stable series; never node-level). Realtime cursor on the right edge; multiple region traces overlaid.
- **Staleness rendering:** regions with `stale=true` draw as held/hatched with an `as_of` age, so a 30s self-state dim reads as a step, not a wiggle.
- **Scrubber:** `−<retention> ⟶ LIVE`. Drag loads `frames/range` (freeze); LIVE resumes tail polling. A `warming` banner shows when `phase="warming"`.
- **Privacy:** node/region rendering uses kind + activation + display label only. No raw private node payloads, journals, mirrors, or blocked material.

## Phasing

**Phase 1 (thin working slice):**
- `SubstrateBrainFrameV1` + registry + channel.
- Producer for `node_kind`, `lane`, `self_state` regions + `spotlight` + sampled nodes/edges; dedicated 5s loop; `warming` gate; per-dimension `as_of`/`stale`.
- Log table + 24h retention + tail/range/window endpoints.
- `self-brain.html`/`self-brain.js`: **hybrid (C) only**, dimension toggles (phase-1 dims), EKG (region series), poll-realtime + scrub, staleness + warming rendering.
- Tests + frame-shape eval + a live stimulus→response smoke.

**Phase 2:**
- Organic constellation (**B**) view toggle.
- `lattice_layer` dimension (generalize beyond transport lane).
- **Emergent-cluster / PCA mode** on-read over a window of retained frames (`GET /clusters`); label clusters by dominant node/coalition.
- SSE/WS streaming only if polling proves laggy; configurable/longer retention.

## Non-goals

- No per-node full-history warehouse (log is bounded + sampled).
- **No node-level continuity guarantee** — samples are decoration; only regions are trackable series.
- No server-side UMAP/t-SNE batch jobs.
- No millisecond/cardiac EKG — cadence is 5s; the trace is a slow polysomnograph.
- No new service; producer rides `orion-substrate-runtime`.
- No action/decision wiring ("levers") in this scope — instrument only, by intent.
- No exposure of private/blocked node contents.
- Not replacing the spark `/spark/ui` Cognitive EKG.

## Acceptance checks

1. **Contract:** `SubstrateBrainFrameV1` registered; channel in `channels.yaml`; `python scripts/check_schema_registry.py` + `python scripts/check_bus_channels.py` pass.
2. **Producer substance (anti empty-shell):** given a fixture graph with active + dormant nodes, the producer yields ≥1 `firing` and ≥1 `starving` region and non-empty `nodes`; region counts > 0.
3. **Cold start:** before graph activation, frames carry `phase="warming"` and the UI shows the warming banner — never an all-zero "live" brain.
4. **Store:** append writes a row; tail returns most-recent ascending; range returns a bounded/downsampled window; prune deletes beyond retention.
5. **Endpoints:** tail/range/window return valid shapes and degrade to empty-with-200 when the log is empty.
6. **Live stimulus→response (the real bar):** with the mesh running, send a deep tool-using chat turn; within a few frames the `execution_trajectory` lane and `execution_pressure`/`reasoning_pressure` dims visibly rise while `concept` stays dim (case §1). Evidence: log rows + screenshot/log line with `frame_id`. Until verified, mark `UNVERIFIED`.
7. **Staleness honesty:** a dimension slower than frame cadence renders `stale=true`/held for the intervening frames, not as movement.
8. **Frontend:** iframe loads; hybrid view renders; dimension rail re-parcellates; EKG renders a region series from ≥2 frames; scrubber loads a range and returns to LIVE. Interaction check, not just page load.
9. **Env parity:** new keys in `.env_example`, synced via `python scripts/sync_local_env_from_example.py`.

## Recommended next patch

Phase 1, contract-first: (1) `SubstrateBrainFrameV1` + registry + channel + parity checks; (2) producer + store + settings + `warming`/`as_of`/`stale` with tests; (3) hub endpoints; (4) `self-brain.html`/`self-brain.js` (hybrid only) + iframe wiring; (5) frontend interaction check + frame-shape eval; (6) live stimulus→response smoke against the mesh.
