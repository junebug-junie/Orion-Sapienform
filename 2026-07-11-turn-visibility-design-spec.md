# Design spec: end-to-end turn visibility in Hub

Status: design only, nothing implemented. Written 2026-07-11 from a brainstorming session.
Owner: Juniper. No idea below has been picked yet — this is the reference doc to revisit before starting a patch.

## Arsonist summary

Juniper wants to watch an `orion-unified` turn move through the whole system live in Hub: an animated pipeline (not a force-directed node graph), cards you click to inspect the payload at each step, and a bottom timeline with timestamps you can scrub, rewind, and fast-forward through.

Nothing that does this exists today. The pieces that are closest are either pull-only and unconsumed, push-only-but-plain-text, or solving an adjacent problem (grammar-atom graph, not turn pipeline). The real gap is upstream of UI: there is no bus-native, cross-service, timestamped "turn hop" event. The turn's stage sequence currently exists only as regex patterns matching container log lines in a debug script. Build the event spine first; the animation is the easy part once live data exists.

## Current architecture

Grounded by reading the repo on 2026-07-11. Nothing here is aspirational.

**`orion:cognition:trace` (built, wired, unconsumed by any UI)**
- Schema: `CognitionTracePayload` (`orion/schemas/telemetry/cognition_trace.py`) — `correlation_id`, `mode`, `verb`, `steps: List[StepExecutionResult]`, `final_text`, `recall_debug`, `metadata`.
- Channel: `orion:cognition:trace`, producer `orion-cortex-exec`, consumers `orion-spark-introspector`, `orion-landing-pad`, `orion-bus-tap` (`orion/bus/channels.yaml:1505`).
- Hub side: `services/orion-hub/scripts/cognition_trace_cache.py` subscribes and caches per `correlation_id` (OrderedDict + TTL + max-entries eviction, same pattern used elsewhere in this service).
- Exposed via `GET /api/cognition/trace/{correlation_id}` (`services/orion-hub/scripts/api_routes.py:4886`) with a redacted (`get_redacted`) and a debug (`get_debug`) view, the latter gated by `api_debug`.
- **Gap: no template or static JS in Hub calls this endpoint.** It has test coverage (`tests/test_cognition_trace_api.py`) but no UI consumer. It only captures steps generated *inside cortex-exec's own run*, not the full hub → thought → harness-governor → substrate-runtime → cortex-exec → hub loop.

**`orion:harness:run:step` (built, wired, live-pushed, consumed by a flat text log)**
- Schema: `HarnessRunStepV1` (`orion/schemas/harness_finalize.py`) — `correlation_id`, `step_index`, `step: dict`.
- `services/orion-hub/scripts/harness_step_relay.py` (`HarnessStepRelay`) subscribes on the bus and fans steps out per-`correlation_id` to `asyncio.Queue`s registered by the open turn's WebSocket connection (`services/orion-hub/scripts/websocket_handler.py:1260`, `relay.register_queue`).
- This is the **only real live push-to-browser channel that exists today** for turn-in-progress data.
- Sole frontend consumer: `services/orion-hub/static/js/agent-trace.js` — appends one plain-text row per step to a scrolling `<div>` ("Reasoning steps (live)"). No visual pipeline, no layers, no clickable cards, no timestamps-as-scrubber, no replay.

**Substrate Atlas (built, wired, solving an adjacent problem)**
- `services/orion-hub/templates/substrate_atlas.html` + `services/orion-hub/static/js/substrate-atlas.js`: a Cytoscape.js force-directed graph of **grammar atoms and dimensions**, polling `GET` every `POLL_MS` (default 3000ms — not push).
- Has a "Timeline" panel (`#atlasTimeline`) that is a single flat text string of hop names concatenated with `|` — not an interactive scrubber, no replay.
- This is almost certainly the "shitty node chart, hard to navigate" Juniper is rejecting as a UI pattern. It is also a different concept: grammar-atom provenance, not turn-pipeline stages. Do not extend it for this purpose — a new, purpose-built view is warranted (see Ideas §3).
- `substrate-lattice.js` (510 lines) is a sibling visualization, not yet inspected in depth; check before assuming it's unrelated.

**The turn's actual stage sequence already exists — as log-scraping regex**
- `scripts/trace_unified_turn.py` (646 lines) defines `_HOP_PATTERNS`: 15 regexes matching container log lines for hops:
  `ingress_pre_turn → stance_react → grammar_step (×N) → substrate_5a → cortex_5b → cortex_5c → verdict → outcome → harness_complete → closure_publish → closure_received → prediction_error`, plus `system_error` / `context_overflow` failure detectors.
- It has two modes: `dump <correlation_id>` (reads container logs after the fact) and `live --bus` (subscribes to 8 bus channels: `orion:thought:artifact`, `orion:harness:run:step`, `orion:grammar:event`, `orion:harness:verdict:artifact`, `orion:substrate:turn_outcome`, `orion:substrate:post_turn_closure`, `orion:harness:run:artifact`, `orion:cognition:trace`).
- This script is genuinely useful and is the best existing map of what a turn's hops actually are — but per CLAUDE.md's "no regex swamp" rule, 15 regexes over log text is exactly the fragile pattern this project explicitly warns against for anything that needs to grow into real cognition/telemetry infrastructure.

**No replay/scrub pattern exists anywhere in the repo.** Verified by grep across `static/js` and `templates` for scrub/replay/rewind-shaped code — zero hits.

## Core question

The sharpest framing: every piece needed for "watch a turn move through the system live, click any step, scrub back through it" already half-exists, scattered across three unconnected subsystems (cognition-trace cache, harness-step relay, substrate atlas), none of which span the full loop with timestamps. The bottleneck is not the UI — it's that there is no bus-native, cross-service "this turn moved from stage X to stage Y at time T" event. Build that first.

## Proposed schema / API changes

**New schema** — `orion/schemas/telemetry/turn_hop.py`:

```python
class TurnHopV1(BaseModel):
    schema_version: Literal["turn.hop.v1"] = "turn.hop.v1"
    correlation_id: str
    hop_name: str            # e.g. "ingress_pre_turn", "stance_react", "grammar_step", "substrate_5a", ...
    service: str              # producing service name
    ts: float                 # unix timestamp, hop emission time
    step_index: int | None = None   # for repeating hops like grammar_step
    status: Literal["started", "completed", "failed"] = "completed"
    duration_ms: float | None = None
    payload_summary: dict[str, Any] = Field(default_factory=dict)  # redacted, small
```

**New channel** — `orion:turn:hop` in `orion/bus/channels.yaml`, kind `event`, schema_id `TurnHopV1`. Producers: whichever services own each hop (hub, orion-thought, orion-harness-governor, orion-substrate-runtime, orion-cortex-exec). Consumers: `orion-hub` initially.

**New Hub API**:
- `GET /api/turn/{correlation_id}` — ordered list of buffered `TurnHopV1` for a correlation_id, for scrubber hydration and late-connect catch-up.
- Reuse existing `GET /api/cognition/trace/{correlation_id}` for payload-card detail rather than duplicating redaction logic, unless a hop's payload isn't representable there (see Missing questions on schema overlap).

**Registry**: `orion/schemas/registry.py` gets `"TurnHopV1": TurnHopV1`.

## Ideas (not yet chosen — pick one to start)

### 1. `TurnHopV1` schema + `orion:turn:hop` channel
- **What**: canonical event emitted at the points `trace_unified_turn.py` already regex-matches in logs.
- **Why it matters**: prerequisite for everything else. Converts tribal log-grep knowledge into an inspectable, queryable, replayable bus fact.
- **Smallest buildable version**: schema + channel + registry entry, instrument 2 hops in 2 services (e.g. hub ingress, harness-governor's `harness_run_complete`), confirm one real turn produces both events in order with matching `correlation_id`.
- **Files**: `orion/schemas/telemetry/turn_hop.py` (new), `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `services/orion-hub/app/settings.py`, `services/orion-harness-governor/app/...` (wherever "harness run complete" is logged).

### 2. Hub turn-hop WebSocket relay
- **What**: sibling to `HarnessStepRelay` that subscribes to `orion:turn:hop`, keeps an ordered per-correlation-id ring buffer (same OrderedDict + TTL + max-entries pattern as `cognition_trace_cache.py`), fans hops to the browser live, hydrates late-connecting clients from the buffer.
- **Why it matters**: the actual live-stream gap — only harness FCC steps push live today; nothing spans ingress → closure.
- **Smallest buildable version**: copy `harness_step_relay.py`'s `_queues` / `register_queue` fan-out pattern, subscribe to the new channel.
- **Files**: `services/orion-hub/scripts/turn_hop_relay.py` (new), `services/orion-hub/scripts/websocket_handler.py`.

### 3. Swimlane pipeline view (explicitly not a node graph)
- **What**: fixed lanes, one per service/stage, in canonical hop order. A hop lights its lane; a marker animates lane-to-lane as hops arrive.
- **Why it matters**: matches the actual topology — the turn is staged/linear, not graph-shaped, so a force layout fights reality. This directly answers "not a shitty node chart, hard to navigate."
- **Smallest buildable version**: static HTML/CSS lanes + JS toggling a "lit" class with CSS transitions. No charting library needed for v1.
- **Files**: `services/orion-hub/templates/turn_pipeline.html` (new), `services/orion-hub/static/js/turn-pipeline.js` (new).

### 4. Click-through payload cards
- **What**: clicking a lit hop opens a side panel with that hop's redacted payload, reusing `cognition_trace_cache._redacted_step`'s shape (services touched, latency, artifact keys, log tail — not raw dumps).
- **Why it matters**: the literal "total inspectability" ask, and it finally gives the already-built `/api/cognition/trace/{correlation_id}` endpoint a UI consumer.
- **Smallest buildable version**: on click, fetch the existing cognition-trace endpoint, or the new `/api/turn/{correlation_id}` hop detail if payload isn't already captured there.
- **Files**: `services/orion-hub/static/js/turn-pipeline.js`, `services/orion-hub/scripts/api_routes.py`.

### 5. Timestamp scrubber with rewind / replay / fast-forward
- **What**: bottom timeline bar over the ordered hop buffer from idea 2. Scrubbing replays lane activations at recorded relative offsets. Pinned to "now" for a live turn until dragged back.
- **Why it matters**: the literal ask, and nothing like it exists anywhere in this repo today (verified). Becomes a reusable primitive beyond turns — dream cycles, drive shifts, anything timestamped.
- **Smallest buildable version**: purely client-side. Once idea 2 exposes `GET /api/turn/{correlation_id}`, replay is `setTimeout`-driven playback of that array at a speed multiplier — no new backend beyond ideas 1+2.
- **Files**: `services/orion-hub/static/js/turn-pipeline.js`, `services/orion-hub/scripts/api_routes.py`.

### 6. Retire the log-regex hop detector once the bus event is proven
- **What**: `trace_unified_turn.py`'s regex patterns become a cross-check/fallback, not the primary source, once `orion:turn:hop` is reliable.
- **Why it matters**: CLAUDE.md explicitly bans regex-as-architecture for anything that needs to extend. This is that pattern, applied to ops tooling instead of cognition — same fix applies.
- **Smallest buildable version**: add a `live --bus-hops` mode reading the new channel; run side-by-side with the regex mode on a few real turns to check parity before deprecating anything.
- **Files**: `scripts/trace_unified_turn.py`, `tests/test_trace_unified_turn.py`.

### 7. Latency / anomaly highlighting on the lanes
- **What**: color a lane red/amber if `duration_ms` exceeds a rolling baseline for that hop name, or flag known failure hops (`harness_finalize_system_error_published`, `context_overflow` — both already detected by the script).
- **Why it matters**: passive visibility is weaker than actionable visibility. A stuck/slow turn should be visible while running, not discovered later from a frozen host (cf. the earlier Athena host-freeze incident — a silent per-event save loop, exactly the class of failure a live pipeline view would surface faster).
- **Smallest buildable version**: client-side only, naive threshold (>2× median of last 20 same-named hops this session).
- **Files**: `services/orion-hub/static/js/turn-pipeline.js`.

### 8. Feed the same stream into Orion's own self-observability (explicitly deferred — do not build yet)
- **What**: eventually let `orion:turn:hop` feed `self-state-runtime` / `self_observability.js` so Orion has durable material about its own turn structure, not just an operator dashboard.
- **Why it matters for Orion's development toward sentience**: this is the kind of raw self-model material the mission statement asks for (continuity, self-modeling, error correction) — right now a turn's own shape doesn't durably exist anywhere for Orion to reflect on, only in ephemeral logs.
- **Smallest buildable version**: none for v1. Flag as a later consumer once idea 1's schema is stable.

## Non-goals

- Not replacing or extending `substrate-atlas.js` / the grammar-atom Cytoscape graph — that's a different concept (grammar dimensions/provenance), left as-is.
- Not building a general-purpose DAG/graph visualization library. The turn pipeline is staged, not arbitrary-graph-shaped; if a real branch/parallel case turns up (see Missing questions), handle it as a narrow exception, not by generalizing to a graph engine.
- Not wiring instrumentation into all 5 services in one patch. Stage it: schema + channel → 2 hops in 2 services → prove on one real turn → expand.
- Not designing the self-observability consumer (idea 8) yet — explicitly deferred until the schema is stable and has at least one real UI consumer.
- Not dumping raw, unredacted payloads into click-through cards — must reuse existing redaction discipline.

## Tensions and risks

- **Cathedral risk**: instrumenting every service at once violates CLAUDE.md's thin-seam mandate directly. Stage the rollout.
- **Trace-concept fork**: `CognitionTracePayload.steps` already has real consumers (spark-introspector, sql-writer, rdf-writer, vector-writer). A new `TurnHopV1` risks becoming a second, overlapping trace lineage that drifts from the first. Must decide explicitly — does turn-hop subsume cognition:trace, or does cognition:trace nest inside one hop's payload — **before** wiring more than the first 2 hops. (This is the same category of concern flagged in the "Cognition metric lineage registry ideas" memory — worth reading before deciding.)
- **Privacy**: click-through payload cards collide with CLAUDE.md's privacy-boundary rule. Raw `recall_debug`, user message content, memory digests need the same redaction `cognition_trace_cache.py` already applies. A naive raw-dump inspector would leak more than intended.
- **Unbounded buffers**: live rewind/ffwd implies buffering in-flight-turn payloads in memory. This codebase has already been bitten twice by unbounded per-event growth (pressure-reducer evidence list, execution-merge evidence list — both had to be retrofitted with caps). The ring buffer needs cap + TTL discipline from the first commit, not as a follow-up fix.
- **Animation vs. signal**: grammar steps can fire sub-100ms apart. Literal real-time playback may blur into noise; may need a minimum per-hop display time rather than true wall-clock animation.

## Missing questions

Answering these would most change which idea to start with:

1. Is the hop sequence strictly linear, or does it branch/parallelize (e.g. concurrent `RecallService` + `VisionWindowService` calls within one turn)? Determines whether swimlanes are sufficient or a light DAG is unavoidable somewhere.
2. Is `trace_unified_turn.py`'s hop list complete, or are there hops missing from it (recall, vision, MetaTags, agent-council) that should be in `TurnHopV1` from day one?
3. Scope: the full "unified turn" (chat, ingress→closure) only, or also agent-mode turns (which already have a separate trace shape via `agent-trace.js` / `agent-claude-trace.js`)?
4. Persistence: does replay need to survive past the in-memory TTL window (days later — piggyback on sql-writer's existing cognition:trace persistence), or is replay scoped to the current session/cache window only?
5. Audience: Juniper-only debug tool, or eventually something Orion queries as self-model input (idea 8)? Changes whether the schema should be designed for machine-reflection from day one vs. pure human debug convenience.

## Recommended next patch

Start with **idea 1**: define `TurnHopV1` + `orion:turn:hop`, instrument exactly 2 hops in 2 services (hub ingress, harness-governor `harness_run_complete`). Everything else here — lanes, cards, scrubber — is UI over data that doesn't durably/live-stream today.

First concrete step: add the schema and channel, wire the two emit points, run one real chat turn, and confirm `redis-cli SUBSCRIBE orion:turn:hop` shows both hops in order with the matching `correlation_id`. That proves the spine before any animation work starts, and it's the right moment to force the explicit call on the cognition-trace overlap question (Tensions §2) before more hops get wired on top of an unresolved fork.

## Acceptance checks (for whichever patch starts this)

- `orion:turn:hop` channel registered in `orion/bus/channels.yaml` and `TurnHopV1` in `orion/schemas/registry.py` (passes `scripts/check_schema_registry.py` / `scripts/check_bus_channels.py`).
- One real turn through the running stack produces at least 2 ordered `TurnHopV1` events on the bus with a shared `correlation_id`, observable via `redis-cli SUBSCRIBE orion:turn:hop`.
- No change to `CognitionTracePayload` consumers' existing behavior (spark-introspector, sql-writer, rdf-writer, vector-writer keep working unmodified).
- Redaction of any payload summary field matches or reuses `cognition_trace_cache._redacted_step`'s discipline — no raw user message / recall_debug content in `payload_summary`.
