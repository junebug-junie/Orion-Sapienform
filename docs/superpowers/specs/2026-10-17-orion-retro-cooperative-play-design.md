# Orion Retro Play — real-time NES/SNES embodiment and cooperative play

**Date:** 2026-10-17  
**Status:** PROPOSAL / design-only; no emulator, game-control, or streaming runtime implemented by this PR.  
**Owner:** Orion; human operator: Juniper.  
**Target:** Existing Orion mesh; CPU-first live path on Athena (dual Xeon Gold 6138 candidate), optional existing V100 capacity for training/semantic vision.  
**Implementation posture:** Small contracts and one authoritative game loop; measure before allocating new hardware.  
**Decisions requiring implementation validation:** libretro host packaging/audio, browser A/V transport, real input latency, specific ROM/core compatibility, controller policy competence.

## 0. Executive decision

**Goal:** Let Orion play ordinary, unmodified NES/SNES ROMs in real time while Juniper watches; let Juniper and Orion play together with independent controllers or alternate control of a single-player game. The result is a genuine external, repeatable action environment for Orion, not a demo where an LLM describes game frames and never acts.

**Recommended design:** One authoritative emulator on Athena; hard real-time-ish frame/input scheduling local to the emulator; fast, bounded CPU motor policy; an asynchronous Orion cognition bridge for goals, reflection, and social commentary; a low-latency browser stream and gamepad channel; separate offline training in Stable-Retro. Begin with deterministic or human-demonstration baselines. Do **not** require a V100, an LLM, Redis RPC, or a GPU-pool lease for any per-frame input decision.

Why this shape:
- At 60 Hz, a frame is ~16.67 ms. A queued LLM turn cannot close a per-frame control loop.
- A 10–30 Hz motor policy can emit 1–6-frame button batches, while the emulator applies controller states at 60 Hz. The motor must safely handle missed deadlines.
- Game-playing **competence**, not emulator FPS, is the main open research problem. Having the hardware to evaluate a neural policy does not produce a policy capable of playing an arbitrary ROM.
- Existing Orion primitives can receive *meaningful game experiences*, not raw frame spam; the cognition stack should never become the controller's availability dependency.
- The user already has substantial CPU/GPU resources; new purchases are prohibited until measurements prove a specific bottleneck.

**Two independently useful products:** (1) reliable live shared gaming even with a simple controller policy; (2) a reproducible agent-learning bench that records observations, actions, and measured outcomes. Neither should wait for broad ROM-general intelligence.

### Success definition

A qualifying session can be launched from Hub; Orion controls an assigned port at normal game speed; Juniper sees and hears gameplay, can take over immediately, and can join in a genuinely two-player ROM. The system retains attributable traces of Orion's attempts, goals, interventions, and outcomes. If cognition or a GPU is unavailable, gameplay stays safe and responsive, with no zombie button holds.

A Twitch-like spectator UI without controller agency is *not* success. A tool-driven game that pauses between every LLM call is *not* success. A scripted agent that always presses right must be visibly identified as a baseline, not presented as learned competence.

## 1. Sources and repo reconnaissance (read-only, 2026-10-17)

This design uses the current default-branch files below as integration evidence; code and live hosts were not executed or benchmarked in this session. Some previously dated docs discuss earlier production placement; placement here remains proposed, not a claim of current free capacity.

| Existing seam | Verified file | Decision |
|---|---|---|
| Prior game embodiment | `orion/embodiment/aitown_client.py`; `docs/superpowers/specs/2026-07-06-fcc-claude-mcp-aitown-design.md`; `services/orion-hub/static/js/aitown-panel.js` | Reuse interaction patterns (embodied intent + operator-visible game UI), **not** AI Town/Convex as the emulator's motor transport. The client itself documents 15–25 s mutation contention in a dated incident. |
| Bus and schema convention | `services/orion-bus/AGENT_CONTEXT.md`; `orion/schemas/system_one_appraisal.py` | Add explicit typed game events; keep bus outside the frame loop. For subscribers that also perform RPC, use separate client via `fork_rpc_client`. |
| Attention / System One | `orion/substrate/system_one_appraisal.py`; `orion/substrate/system_one_access.py` | Use optionally to select *when to deliberate*; **not** for 60 Hz directional commands. Do not overload its existing curiosity decision semantics. |
| Temporal continuity | `orion/schemas/temporal_self.py`; `services/orion-durable-runs/app/temporal_self_driver.py` | Emit bounded session/attempt/outcome evidence first; integrate into strict temporal enums only after source + consumer-first rollout. |
| Visual intake | `services/orion-vision-retina/README.md`; `orion/schemas/vision.py` | Retina is capture and publishes pointers, not a 60 Hz object tracker. Publish sampled game evidence into vision only when requested; no JPEG/Redis round-trip in motor path. |
| GPU governance | `services/orion-gpu-pool/README.md`; `config/gpu_pool.yaml`; `services/orion-llm-gateway/README.md` | Optional semantic-vision/LLM jobs use existing GPU-pool lease and background priorities. Motor remains CPU-resident and independent. Never silently usurp Juniper's chat lane. |
| Motor vs game motor | `docs/superpowers/specs/2026-09-10-fcc-motor-step-trace-design.md` | FCC motor is a tool/turn harness. The new game motor is a deterministic, deadline-bound controller; avoid confusing the two or routing per-frame actions through the FCC subprocess. |
| Trace discipline | `CLAUDE.md` | Each new signal needs provenance, nonduplication, calibration, observed values, reversibility and explicit eval; no decorative cognitive taxonomies. |

**Upstream references to pin and test, not assumed dependencies:** [libretro core API](https://docs.libretro.com/), [RetroArch netplay](https://docs.libretro.com/development/retroarch/netplay/), [Stable-Retro](https://github.com/Farama-Foundation/Stable-Retro), [Stable-Retro supported emulators](https://github.com/Farama-Foundation/Stable-Retro/blob/main/docs/supported_emulators.md), and experimental [RetroArchAI](https://github.com/MatPoliquin/RetroArchAI). Stable-Retro's documented Linux cores include FCEUmm (NES) and Snes9x (SNES). RetroArchAI's P1/P2 learned-input override is useful prior art, **not** a required production fork. RetroArch netplay is a fallback for geographically separated participants, not the first local implementation.

No existing stable-retro/libretro/retroarch integration was found in this repo's indexed code search. This is a new *environment interface*, not a replacement for AI Town.

## 2. Modes, user stories, and hard scope

### 2.1 Modes

| Mode | Controller authority | Behavior |
|---|---|---|
| `spectate_orion` | Orion on P1; human spectator | Orion plays; human watches, asks questions, may take over. Streaming and game logic do not block on commentary. |
| `co_op` | Human P1 and Orion P2 (configurable) | Two independent ports on ROM/core supporting two local players; both view one game timeline. Human wins arbitration for the human port. |
| `shared_single_player` | Explicit P1 ownership lease, human or Orion | Human and Orion alternate, hand off by UI/gamepad, or ask Orion to attempt a segment. **Never** OR together conflicting directional/button states by default. |
| `assisted` (later) | Human P1; Orion suggests only | Orion proposes moves and plans but cannot actuate unless human grants control. |

A game that exposes only one emulated player cannot be called cooperative P2. Some RPGs allow multiple-character control or alternating turns, but the interface must honestly report the game/core's actual capabilities. No hack to force a second controller into a single-player ROM.

### 2.2 MVP games

- **Deterministic smoke:** freely distributable/open homebrew NES/SNES test ROM with clearly permitted redistribution, or a self-authored test ROM. Repository includes **no copyrighted game ROMs, BIOS, proprietary game art, or save files**.
- **Autonomous evaluation:** one legally supplied NES or SNES platformer with a curated integration and reproducible checkpoint. Candidate examples can be explored with user-owned copies, but are not bundled.
- **Co-op acceptance:** a user-supplied two-controller-capable SNES game, tested with P2 buttons independently. Secret of Mana is an *illustrative* co-op use case, not a distribution or compatibility guarantee.
- **Turn-based/shared control:** user-supplied RPG with stable save/load and explicit P1 handoff.

### 2.3 Goals / non-goals

**Goals:** 60 Hz deterministic core advancement; low-latency remote video/audio; two logical ports; explicit control ownership; agent observability; reproducible outcome data; CPU-first motor policy; transparent skill level; user-supplied ROM validation; no impact on existing Orion production traffic.

**Non-goals for v1:** universal competence across arbitrary games; autonomous ROM downloading; 3D conversion; AI-generated replacement graphics; frame-by-frame LLM control; humanlike voice conversation at 60 Hz; cloud-hosted multiplayer; hardware-console passthrough; public game streaming; ROM patching/cheating as a default; online competitive matchmaking; self-updating policies in production without review; replacing Orion's current temporal-self or memory layers.

## 3. Deployment and top-level boundaries

~~~text
  Carbon X1 / Hub browser                  ATHENA: LOCAL REAL-TIME ISLAND
┌────────────────────────┐              ┌────────────────────────────────┐
│ view video + hear audio│◄── WebRTC ───│ encoder / audio packetizer     │
│ local gamepad/keyboard │── inputs ───►│ session auth + input ingress   │
│ takeover / P1 ↔ P2 UI  │              │ ┌────────────────────────────┐ │
└────────────────────────┘              │ │ authoritative libretro core│ │
                                        │ │ frame clock 60Hz, A/V, RAM │ │
                                        │ └──────────┬─────────────────┘ │
                                        │   observation ring (in memory) │
                                        │ ┌──────────▼─────────────────┐ │
                                        │ │ CPU motor policy + watchdog│ │
                                        │ │ no RPC or LLM on this path │ │
                                        │ └──────────┬─────────────────┘ │
                                        └────────────┼───────────────────┘
                                                     │ bounded summaries
                                                    ▼
                                       event spool / adapter / bus
                                                    │
                          ┌─────────────────────────┼───────────────────┐
                          ▼                         ▼                   ▼
                  Orion cognition            Orion memory         operator UI
                  existing LLM/GPU pool      episodic outcomes    traces/controls
                          │
                          └── goals/intent ─────────► bounded mailbox
                                                         │
                                    OFFLINE TRAINING     ▼
                                    Stable-Retro      motor planner
                                    CPU rollouts + V100 optional
~~~

**Authoritative state:** a single emulator instance is the sole owner of game clock, current state, controller port states, and save-state writes. Nothing external steps it directly. A separate offline simulator may replay compatible inputs against a pinned core to validate runs; cross-backend save-state binary compatibility is **not assumed**.

**No Redis RPC inside frame deadlines.** The local motor consumes the latest consistent observation snapshot and outputs bounded actions to a mailbox; a local arbiter translates them to controller states before the next emulation step. Disk, database, LLM, Graphiti, vision-host, WebRTC signaling, and operator notifications are all asynchronous.

**Service topology (first iteration):** one deployable `services/orion-retro/` containing the emulator host, session manager, control arbiter, motor interface, event outbox, and streaming endpoints as separable modules/processes. Do *not* preemptively create 4–6 microservices. The only new stable package-level API is `orion/retro/` for contracts/logic reusable by evals; optional cognition adapter may run in the service's non-real-time worker.

### 3.1 Placement / resource priority

- **Athena** is the first candidate (known Xeon Gold 6138 host). Pin emulator frame loop to a lightly loaded physical core or reserved CPU set **only if measurements show contention**. Dual socket affinity/NUMA locality matter more than aggregate 40-core count for one game.
- Start motor inference on **CPU** using rules or ONNX Runtime CPU (one small quantized CNN/GRU if trained). Set explicit inference thread cap; no OpenMP oversubscription. If empirical p95 violates the deadline, benchmark *one* optional V100 placement under gpu-pool control. Do not reserve VRAM merely because GPUs exist.
- CPU software H.264 `libx264` at low resolution and `ultrafast`/`veryfast` is the **baseline** streaming encoder, subject to benchmark. Prefer native low-res output without expensive arbitrary scaling. NVENC on a particular V100 **must be probed** rather than assumed from the family name; do not make hardware video encoding a boot prerequisite.
- Existing `orion-gpu-pool` exclusively governs optional vision, planning, or training inference GPU use. A background vision request queues or skips; never blocks emulator or compels GPU swap. Training gets a cancelable, low-priority allocation and operator gate.
- Existing power/thermal telemetry should attribute incremental CPU/GPU energy; do not invent precise watts based on TDP. No new purchases until the measurement gate in §12 fails with a reproducible profile.

## 4. Runtime ownership and clock correctness

**One master clock:** emulator thread runs `step(frame_id, p1_state, p2_state)` at nominal game-system cadence (NTSC target ~60/60.1 depending on core). Use the core's actual reported timing, not an assumed fixed 60.000 Hz. Audio callback uses a bounded PCM ring and timestamp derived from the same frame timeline.

All game inputs carry `session_id`, monotonically increasing `seq`, intended `frame_id` or `valid_from_frame`, source ID, port, controller mask, TTL, and a control-lease token. Validate/reject stale, duplicate, cross-session, unauthorized, impossible ports, and out-of-order proposals. Ack includes actual applied frame and reason; clients never infer that an HTTP 200 equals actuation.

**Live-action flow:**
1. Emulator publishes immutable `FrameObservation` to a single-producer bounded ring: frame_id, monotonic timestamp, downsampled pixels, optional raw RAM digest/read-only adapters, current inputs, and recent reward/outcome hints.
2. Motor policy reads the *latest* eligible snapshot, not an unbounded queue of old frames; it may skip observations rather than accumulate latency.
3. Fast controller returns `ControllerIntent`: button mask and hold duration (clamped 1–6 frames initially), goal source, uncertainty, model/policy version, originating frame, expiration.
4. Arbiter chooses exactly one authority per port and frame. Input edges are applied atomically at frame boundaries; overdue intents become neutral/no-op or bounded last-known-safe action (configurable per policy, neutral default).
5. Emulator advances; adapter emits a short action acknowledgment + sparse outcome deltas.
6. Cognition can set **goals** (e.g. `reach_exit`, `protect_player_1`) asynchronously; a mailbox with generation ID replaces stale goals. The goal planner cannot bypass port ownership or duration limits.

### 4.1 Input arbitration precedence (locked)

`emergency_stop` > `operator_takeover` > `assigned_human_port` > `operator-authorized_agent_port` > `demo/scripted_policy` > `neutral`.

- Human handoff is explicit, atomic at next frame, and acknowledged. The old controller mask releases **before** new owner applies an action; ownership never silently merges.
- On disconnect: human-owned port becomes neutral immediately/within one frame; Orion-owned port may continue **only** if operator chose autonomous-after-disconnect; default **pause all** on last authorized viewer disconnect.
- On motor deadline miss (>100 ms without valid intent by default) drop held buttons, record a watchdog trip, and optionally pause if repeated. No indefinite RIGHT/B holds, turbo-spam, or unbounded text-driven actions.
- Controller state cannot operate the host OS, emulator config/menu, file system, or other game/session. The ROM sandbox exposes controller bits, approved save/load, and pause; not shell execution.
- Manual pause/resume always wins. Pause means emulator clock stops; freeze timeout clock where appropriate, but reset all pending controller leases on resume.
- The controller/ROM port map is validated against actual core-reported device support at launch.

### 4.2 Real time and time semantics

- Emulation FPS and motor policy Hz are **independent**. Start at 15 Hz policy / 60 Hz emulation; A/B test 10, 20, 30 Hz.
- Target local motor policy `p95 <= 20 ms` and controller arbitration `p99 <= 2 ms` measured under CPU load; these are **gates**, not measured facts.
- Human input takes priority. For client gamepad, capture events/poll local controller state at animation-frame cadence; send sequence-numbered state changes plus periodic snapshots. Server sanitizes held-button expiry.
- A/V frames are timestamped using emulation time with capture monotonic clock; client instrumentation can calculate capture→encode→transport→decode→display latency. Correlate controller input sequence with applied frame to quantify input-to-photon latency.
- CPU scheduler jitter matters. Affinity, nice/real-time scheduling, and clock sources are optimization experiments, not defaults; do not deploy SCHED_FIFO privileges until starvation/liveness tests pass.

## 5. Emulator backend and media implementation

### 5.1 Driver interface

Define `EmulatorBackend` methods:

~~~python
load_rom(path, core_id, profile) -> RomIdentity
get_capabilities() -> CoreCapabilities
step(frame_inputs: dict[int, ButtonMask]) -> FrameResult  # sole owner/thread
snapshot() -> OpaqueState
restore(opaque_state) -> None
reset() -> None
close() -> None
~~~

`FrameResult` contains frame id, native-resolution RGB/RGBA pixels (when emitted), timestamped PCM, core timing, and a limited game-memory reader through a validated integration map. Never expose arbitrary process memory to the remote UI. Core's AV details, framerate, sample rate, pixel format, overscan, aspect ratio, and controller topology must be discovered, recorded, and pinned.

**Primary interactive backend:** minimal in-process libretro host with pinned core build hashes (NES FCEUmm, SNES Snes9x), packaged under `services/orion-retro/app/emulator/`. Prefer reusing a hardened binding/runner if licensing and operational probes demonstrate it can access frame/audio/controller callbacks *and* correct save states; do not reimplement CPU emulation. Capture SDL/FFmpeg in a detached codec worker, not inside step's deadline.

**Alternative benchmark lane:** RetroArch/RetroArchAI adapter for interactive GUI if custom libretro host proves expensive; performance/latency/controls must meet the **same** contract. Do not merge incompatible session semantics or simultaneously step two emulator authorities.

**Offline training backend:** Stable-Retro with its supported FCEUmm/Snes9x cores and per-game integrations. Data compatibility lives in a versioned `ObservationAdapter` and button-map adapter, not an assumption that RetroArch and Stable-Retro serialization bytes are interchangeable.

### 5.2 Audio/video

- WebRTC A/V stream to Hub browser: 60-ish FPS at native/pixel-doubled resolution; audio sourced from emulator PCM and synchronized to frame clock. Never publish every decoded frame to Redis.
- Implement signaling under authenticated same-origin Hub proxy routes; WebRTC transport/session may be housed within `orion-retro` and reverse-proxied by Hub. ICE for LAN/Tailscale first. Disclose LAN-only status on failure; no default public TURN server.
- Streaming encoder selection: verified NVENC if available *and faster/cheaper*, else CPU x264; startup benchmark logs codec, encoder, dimensions, frame latency, dropped frames. V100 has no normal display output; no monitor/X server is required for libretro's framebuffer.
- Audio-only failure degrades to muted but visually playable with warning; video failure triggers explicit session pause (unless operator opts into headless autonomous mode).
- Input transport: WebRTC DataChannel **or** WebSocket; whichever meets LAN p95 and reliable ordering needs. Never rely on unordered packets without seq/ack. Send controller state, not raw HID access outside browser.
- UI: Hub tab `Retro Play` with aspect-correct canvas/video, game name & ROM hash prefix, state, P1/P2 owner, headset/audio state, controller map, per-port latency, policy type & version, pause/resume, takeover, emergency stop, save/load (operator only), and latest Orion goal. Keyboard/gamepad bindings must respect focus and not intercept global Hub shortcuts.
- Spectator cannot actuate. Sharing stream access does not grant controller authority; one authenticated operator owns the session. Reject hidden-tab input resumption until resynchronized.

### 5.3 Content integrity

- Operator mounts legally acquired ROMs read-only into private configured path (e.g. `/mnt/retro/roms`), not Git or web URL. Allowlist NES/SNES extensions and validate core acceptance/magic where feasible.
- Compute SHA-256, file size, core ID, core binary digest, emulator build, controller topology, integration version; attach to session manifest. Same title with different ROM hashes is *not* automatically the same training environment.
- No remote arbitrary path/open/URL in game-start API; caller selects a server-discovered ROM catalog ID. No uploads by spectators. Explicit BIOS allowance checks if future consoles added.
- Operator-only save states in bounded private paths and atomic replace; tag with ROM/core build and frame. Refuse incompatible restores; never overwrite a prior slot without confirmation/backups.
- Keep ROM data and frame captures out of LLM prompts and public logs unless explicitly sampled with operator permission. Do not place game audio in chat transcript storage by default.

## 6. Canonical contracts (proposal; names/version subject to implementation)

Contract types under `orion/schemas/retro.py` with strict validation, `schema_version`, UTC timestamps for persisted events, monotonic/frame timestamps for low-latency alignment. Distinguish *command*, *acknowledged actuation*, *observation*, and *outcome*.

### 6.1 Shared data shapes

~~~json
{
  "schema_version": "retro.session.v1",
  "session_id": "uuid",
  "rom": {"sha256": "hex", "name": "user-supplied-title", "system": "snes"},
  "core": {"id": "snes9x", "build_sha256": "hex", "nominal_fps": 60.0988},
  "mode": "co_op",
  "controller_owners": {"1": "human:operator", "2": "orion:policy"},
  "policy": {"id": "baseline", "version": "script-v0", "competence": "untrained"},
  "state": "running",
  "started_at": "UTC timestamp",
  "max_duration_minutes": 120
}
~~~

~~~json
{
  "schema_version": "retro.controller.intent.v1",
  "session_id": "uuid",
  "intent_id": "uuid",
  "seq": 142,
  "source": "orion:policy",
  "controller_port": 2,
  "observed_frame_id": 19421,
  "valid_from_frame": 19422,
  "buttons": ["LEFT", "B"],
  "duration_frames": 3,
  "expires_monotonic_ns": 1234567890123,
  "lease_id": "opaque-owner-lease",
  "policy_id": "cnn-gru-v1",
  "goal_id": "goal-17"
}
~~~

~~~json
{
  "schema_version": "retro.action.applied.v1",
  "session_id": "uuid",
  "intent_id": "uuid",
  "frame_id": 19422,
  "controller_port": 2,
  "applied_buttons": ["LEFT", "B"],
  "decision": "applied",
  "reason": null,
  "owner": "orion:policy"
}
~~~

~~~json
{
  "schema_version": "retro.outcome.v1",
  "session_id": "uuid",
  "attempt_id": "uuid",
  "frame_window": [18800, 19480],
  "source": "integration:level1-v1",
  "observed_delta": {"lives": -1, "progress_tiles": 13},
  "outcome": "death",
  "confidence": 1.0,
  "objective_id": "reach_exit",
  "evidence_ref": "retro-replay:<sha256>"
}
~~~

**Do not fabricate semantic outcomes:** a ROM without a verified memory map can expose action traces, frame digests and opaque transitions, but `outcome=death` and `progress_tiles` must remain unavailable/unknown until backed by game-specific observations or labeled evidence. A value being zero is different from missing.

### 6.2 Observation contract and rate limits

`RetroObservationV1` includes session/frame/capture times, image ring reference (in-process), previous N button states, optional signed integration-derived RAM features with producer/version, audio event hints if implemented, policy goal, observation freshness. No bus payload carries the raw pixel array. Vision consumers receive **sampled image pointer** (e.g. ≤1–2 FPS on request or event) through the existing retina/vision-host pattern, with `source=retro` and privacy/scope tags. That is an adapter into vision, not a second retina instance.

Do not start an object detector on every gameplay frame. For any externally readable RAM metric, document its address map, decoding, emulator/core conditions, sampling interval, and known failure modes. Profile models are per-game/ROM revision; no fake universal "player_x" pointer.

### 6.3 Bus channel proposal

Following schema registry/channel catalog conventions, and only after reading their current implementation in the build PR:

| Channel | Kind | Producer → consumer | Cadence |
|---|---|---|---|
| `orion:retro:session:event` | `retro.session.event.v1` | retro → sql-writer/Hub/cognition adapter | State transitions only |
| `orion:retro:action:summary` | `retro.action.summary.v1` | retro → telemetry/agent evaluation | Aggregate 1–5 s, **not 60/s** |
| `orion:retro:outcome` | `retro.outcome.v1` | retro → episodic memory/learning | Validated outcome only |
| `orion:retro:goal:request` | `retro.goal.request.v1` | cognition adapter → retro | Event-driven, TTL/generation constrained |
| `orion:retro:goal:ack` | `retro.goal.ack.v1` | retro → cognition adapter | Every accepted/rejected goal |

Do not duplicate existing grammar/system health channels merely to get another dashboard counter. If registry rules prefer existing action/event channel, document mapping and use it. Use `BaseEnvelope` correlation UUIDs (deterministic idempotency where retried) and schema registry registration; avoid raw ad-hoc dicts on global bus. At-least-once delivery is acceptable off-path; de-duplicate by IDs. Never replay expired control commands from Redis.

### 6.4 Persistence and event sourcing

- **Local durable session journal**: append compact frame ranges/action changes, authoritative applied inputs, goal revisions, save/load/reset, sparse outcomes, time/clock discontinuities, policy/core digests, intervention events. Write asynchronously via bounded queue, fsync/rotate based on explicit policy; on storage pressure discard optional pixel capture before critical input trace and stop safely if cannot preserve minimal audit.
- **Session projection in Postgres** (new tables only if no existing generic run artifact is adequate): `retro_sessions`, `retro_attempts`, `retro_action_ranges`, `retro_outcomes`, `retro_interventions`. Keep large movie/video binaries in file storage, not Postgres.
- **Replay bundle:** manifest + initial state/reference + ordered input deltas + emulator/core/ROM digests + checkpoints and checksums. Replaying a complete trace must match known checkpoints using the **same pinned core** or explicitly report nondeterminism; no claims of portability across backends.
- **Retention defaults proposed**: per-frame raw image in RAM only, no durable raw-frame log; lossless input/summary trace 30 d, optional compressed video 7 d, save state until operator deletes, learned policy training datasets operator-approved with manifest and finite quota. Storage ceilings configurable; no infinite Convex-style revision accumulation.
- **Capacity estimate (planning only):** encoded 2 Mbit/s ≈0.9 GB/hour before container overhead; 7 days of continuous 24/7 recording would be ~151 GB, so default record-on-request and max quota rather than continuous capture. Actual codec bitrate and usage measured.
- **Idempotency:** one event ID per transition or outcome; writer retries cannot duplicate episodic facts. Outbox poison events are visible and bounded.

## 7. Orion cognition: goal choice, learning, relationship to the player

The system should produce *action with consequences*, not anthropomorphic narration. Avoid adding new generic cognition boxes because gaming makes them sound evocative.

### 7.1 Action hierarchy

- **Level 0 frame arbiter (60 Hz):** authoritative, local, deterministic; applies buttons and safety rules.
- **Level 1 motor reflex (10–30 Hz target):** bounded visual/RAM observations → button candidates/durations, policies plus stop/recover; CPU preferred.
- **Level 2 tactical planner (event-driven):** coarse goal selected from verified game state + recent failed attempts, can request exploration/combat/traversal/assistance, no hard latency guarantee.
- **Level 3 Orion deliberation (as available):** choose whether to play, shared goals with Juniper, ask for help, explain recent decisions, consolidate experiences, remember what has/hasn't worked. GPU pool / LLM queue is permitted *only* here.
- A failed or stale Level 3 response never makes Level 0 hang; last acknowledged goal may continue until TTL or a safe motor state is reached, then pause/neutral.

### 7.2 System One and salience

System One is an *attention/admission primitive* for whether to spend expensive cognition on new uncertainty, failure streak, human request, or unrecognized scene; it should not be retrained to predict B-button presses. Start with explicit deterministic deliberation triggers (level ended, user spoke, death, repeated lack of progress, new mechanic evidence) and optionally compare against an existing System One question format in shadow. Gate any new question/feature through `CLAUDE.md` metric quality requirements before live behavior.

### 7.3 Temporal self and episodic integration

Initial durable evidence types: `session_started`, `player_joined`, `ownership_changed`, `goal_committed`, `attempt_finished`, `outcome_validated`, `session_ended`. These are **retro-domain facts**, not modifications to `TemporalSourceKind` on day one.

Later: a bounded retro-game session summary becomes a candidate chronological event/arc in `orion/schemas/temporal_self.py` only when (1) source exists, (2) strict consumers have been migrated before producer, (3) temporality is grounded in start/end times, and (4) the reducer can replay exact source rows. Do not inject generic game dopamine/reward directly into φ, arousal, intrinsic value, or system self-report. Evidence must demonstrate a meaningful link first.

### 7.4 Memory and conversation

- Episodic: game/ROM identity, teammate, key episodes, selected goals, what Orion actually tried, outcome evidence and uncertain interpretations, operator corrections. Record *who controlled each port* for each interval. A failure under Juniper's hand is not Orion's motor-policy failure.
- Graph/semantic: *candidate* relationships (e.g. mechanic → outcome) require repeated or validated observations and provenance; avoid auto-asserting "fire damages enemy X" from one caption.
- Social co-presence: Orion can speak via existing TTS/chat paths about game plans or acknowledge user input, but spoken turns must be rate-limited and never needed to continue controls. User mute/quiet mode wins. Do not impersonate an independently sentient agent.
- Human coaching: operator can label a frame/action window ("jump earlier", "stay close"), override goal, tag a demonstration and choose to opt that experience into training. Training labels must preserve precise source and controller owner.

## 8. Motor-policy competency ladder and data strategy

**P0 / deterministic stub:** movement patterns and manually authored game-specific triggers for test ROM only. Useful to prove control, not intelligence.

**P1 / demonstration replay:** collect synchronized (frame, pixels/features, applied buttons, port owner) sequences from consented human play. A replay policy can reproduce a *known state* or trajectory; it does not generalize to unseen states. Measure replay agreement separately from gameplay progress.

**P2 / imitation learning:** supervised behavior cloning with small CPU-friendly CNN+GRU or transformer-lite; sample 4–8 consecutive frames at cropped/scaled resolution (example 84×84 luminance plus validated RAM features), previous actions, optional current goal. Output button distribution and bounded hold horizon. Quantize/export ONNX; benchmark on Xeon before giving GPU runtime lease. Output an uncertainty flag and fail safe.

**P3 / game-specific RL:** Stable-Retro vectorized headless rollouts (PPO or similarly validated baseline) with *verified* reward/integration. Simulated fast-forward only offline. Replay buffers/trajectories tagged with emulator/core/ROM and policy version. Stop on plateau, resource budget or operator cancel; checkpoint best policy, require offline eval before promotion.

**P4 / novel game transfer:** later research. Evaluate across unseen ROMs and random starts, without assuming trained policy generalized because model inference was fast. Query memory for mechanic hypotheses; use operator-approved exploration where safe. Explicitly distinguish unknown reward from poor reward.

**Policy model contract:** `policy_id`, model digest, compatible system/core/game feature profile, input preprocessing version, train dataset SHA/consent scope, button mapping, max inference runtime, offline eval scores, runtime fallback. No hot reload mid-episode; atomic swap only at pause/checkpoint and explicit approval.

**Safety for training:** no user ROM or video leaves home mesh by default; training excludes user audio/chat unless separately consented; GPU pool reservation cancelable; cap CPU rollout workers and storage budget; training never preempts interactive chat workload.

## 9. APIs and UI contracts

Hub should proxy only authenticated control/status/WebRTC signaling; retro process runs private on mesh. No arbitrary Internet-exposed emulator port.

| Route (proposed) | Auth / purpose |
|---|---|
| `GET /api/retro/roms` | operator: server-discovered allowlisted catalog ID + compatible core, no raw server paths |
| `POST /api/retro/sessions` | operator: create session with rom_id, mode, core, controller mapping, policy |
| `GET /api/retro/sessions/{id}` | authorized viewer: state, owners, health, age, policy, recent goals |
| `POST /api/retro/sessions/{id}/pause` / `resume` / `stop` | operator |
| `POST /api/retro/sessions/{id}/claim` / `release` | operator: ownership lease and atomic handoff |
| `POST /api/retro/sessions/{id}/save` / `load` | operator with confirmation and compatibility checks |
| `POST /api/retro/sessions/{id}/goal` | privileged Orion service/operator; bounded generation/TTL |
| `POST /api/retro/sessions/{id}/webrtc-offer` | authenticated WebRTC signaling |
| `WS /api/retro/sessions/{id}/control` | owner-only input state + seq/ack, ping/clock sync |
| `GET /api/retro/sessions/{id}/events` | bounded, filtered recent events, no raw ROM/video binary |
| `GET /api/retro/health` / `metrics` | read-only service health/operational counters with secure exposure |

HTTP statuses and typed error codes: `ROM_NOT_FOUND`, `ROM_HASH_CHANGED`, `CORE_UNSUPPORTED`, `PORT_NOT_PRESENT`, `PORT_NOT_OWNED`, `LEASE_EXPIRED`, `STALE_FRAME`, `SESSION_PAUSED`, `EMULATOR_CRASHED`, `POLICY_UNAVAILABLE`, `SAVE_INCOMPATIBLE`, `STREAM_UNAVAILABLE`. Avoid generic 500s for expected conflicts.

**Session state machine:**

~~~text
CREATED --validated--> READY --start--> RUNNING <--> PAUSED
                            |             |
                            v             v
                         FAILED         STOPPING --> STOPPED
                            ^             |
                            +-- crash ----+
~~~

Guard transitions and enforce idempotent start/stop. Never leave multiple active processes owning the same session/core. One active session system-wide for MVP; additional sessions require quota and CPU evidence.

## 10. Observability and metric quality gates

Do **not** claim performance from Xeon model numbers. Instrument the causal path and gather evidence under actual Orion load. For every new metric below, its producer is the emulator host, motor, encoder or browser explicitly named; source timestamps and numerator/denominator must be stored. All are **engineering telemetry** until separately validated as cognitive signals.

| Signal | Producer | Calculation / provenance | Initial target or alert |
|---|---|---|---|
| `retro_emulated_fps` | emulator step loop | advanced frames / monotonic active seconds (exclude pause) | within 1% of core nominal over 60s |
| `retro_frame_deadline_miss_total` | scheduler | actual step > deadline in each 60s window | <1% frames, evaluate p99 |
| `retro_motor_inference_ms` | motor | observation read → decision committed, histogram | p95 ≤20 ms target |
| `retro_intent_age_frames` | arbiter | applied frame - source observed frame | histogram and age budget |
| `retro_control_handoff_ms` | session arbiter | operator claim accepted → first authorized frame | one frame when local; client RTT reported separately |
| `retro_stream_glass_ms` | browser / encoder | capture ts → displayed ts, synchronized clocks with uncertainty | measure median/p95, don't fake cross-clock deltas |
| `retro_action_reject_total` | arbiter | typed reject reason | alert on unauthorized/stale bursts |
| `retro_motor_watchdog_total` | watchdog | TTL expired / inference stale | zero in nominal soak |
| `retro_outcome_evidence_coverage` | integration grader | validated outcomes / candidate outcomes | report denominator and unavailable |
| `retro_goal_execution_latency_s` | goal adapter | goal ack → first policy action or safe decline | monitor distribution |
| `retro_energy_increment_wh` | existing power evidence | measured incremental power over control baseline, if attributable | **unknown** unless properly measured |

**Metric quality gate before wiring to cognition (per repo mandate):**
1. Point to code producer and real clock/step source; verify capture vs emulation time differences.
2. Independence: distinguish emulator FPS from stream FPS; stream delay from motor latency; outcome score from action frequency; avoid counting correlated proxies as three independent motivations.
3. Theory: identify which actual system property each metric measures (deadline scheduling, queueing latency, episodic outcome) and explicitly disallow claims about felt agency or sentience.
4. Reproduce with live/synthetic telemetry including rest state: paused FPS does not become "broken"; zero actions with spectator mode is normal; unchanged reward may be unknown, not zero.
5. Existing-mechanism search: reuse Orion health/bus/energy telemetry rather than create duplicate counters.
6. Rollback: feature gate, versioned schema, bounded retention and no unconditional insertion into φ/attention models.

Dashboards: session header; separate panels for A/V delay, frame/motor deadline, ownership timeline, model/policy identity, goal vs outcome, event log, actual watts *where attributable*. Empty evidence says "not measured", never a green badge.

## 11. Test matrix, evals, and acceptance gates

### 11.1 Determinism / API

- Unit: port arbitration precedence, sequence & lease validation, 1–6-frame TTL clamp, P2 not present, pause/resume resets stale actions, explicit release of held buttons, stale goal generation, duplicate command ack.
- Unit: ROM hash/size/catalog validation, path traversal prevention, malicious/malformed ROM rejected without host privilege, no cross-session save restore, replay schema compatibility.
- Unit: simulated inference latency and queue overload always yield safe neutral/pause without stuck input.
- Unit: typed envelope registry + mocked bus/writer failure; no action replay from retries and no duplicate episodic outcomes.
- Integration: known open test ROM, 10,000 stepped frames, save then restore then replay fixed 1,000-frame input sequence; checkpoint hashes equal on *same pinned core*.
- Integration: P1 human + P2 scripted independent button signals lead to **distinct** state changes observable in appropriate co-op test ROM.
- Integration: an orphan browser closes; lease drops and no input remains held. Emulator crash transitions session FAILED, terminates media and exposes reason.

### 11.2 Performance

Run on **actual Athena** under (a) idle, (b) normal Orion production services, (c) synthetic CPU contention, (d) simultaneous ordinary chat turn, (e) GPU pool lease queue/swap. Capture host CPU pinning, NUMA, clock governor, audio underruns, thermals, encoder and network path.

| Gate | Proposed pass |
|---|---|
| No-agent emulation | core nominal FPS ±1%, <1% frame deadline misses across 30 min |
| Baseline scripted CPU agent | same FPS target for 30 min; no watchdog trips |
| Learned CPU policy (when one exists) | p95 motor ≤20 ms; misses/rate reported (reject model if not met) |
| Co-op human P1 + Orion P2 | stable 60-ish FPS, independently controlled buttons, 30-min interactive smoke |
| Human takeover | ownership changed by next emulation frame after server receives authorized claim; late Orion actions rejected |
| User-visible latency | record input→applied-frame and capture→display p50/p95 on LAN; **set final thresholds from baseline and player evaluation** |
| LLM/GPU service loss | gameplay continues or cleanly pauses per mode; no stuck inputs; no forced GPU reservation |
| 2-hour endurance | bounded memory/disk queues, no port-hold leaks; recovery from browser reconnect |

Add a **failure injection bench**: disconnect Redis; poison LLM route; timeout model; exhaust disk quota; drop stream packets; reload Hub; reboot motor process; conflicting human/agent claim; suspend browser; change controller; corrupt save state; send future-frame input; pause/unpause during held jump. Assert no unsafe or unauditable behavior.

### 11.3 Behavioral learning evaluation

No qualitative "Orion seemed good" gate. Define per-game validated success with 3 seed categories: memorized replay start, perturbed checkpoint, held-out level/starting state. At least 20 trials per category where meaningful; compare (1) neutral/random legal inputs, (2) scripted baseline, (3) human demonstration/replay, (4) learned policy. Track success, completion/progress from validated source, deaths/restarts, stuck time, intervention frequency, inferred-action latency, and policy confidence calibration. Report median and confidence intervals; stratify by ROM hash, integration version, and player port.

For shared co-op, additionally measure control interference (ports owned incorrectly = **zero tolerated**), assistance requests acknowledged, and player feedback. Do not optimize an unvalidated "cooperation score" built from button proximity.

## 12. Hardware forcing-function decision tree

**Initial assumption:** existing Athena + current mesh can run one NES/SNES emulator and a tiny controller without new equipment. This is a plausible sizing hypothesis, not a measured guarantee.

1. **Profile emulator + encoder with no AI.** If frame-time p99 fails nominal cadence on a Xeon 6138, profile one-thread throughput, audio callbacks, encoder, host contention and NUMA before blaming GPU count. Try core choice, native resolution, CPU affinity. Extra GPU does not fix a single-thread timing miss automatically.
2. **Add deterministic motor at 15 Hz.** If local p95 inference/arbiter violates limits, slim model, reduce preprocessing/copies, quantize model, cap threads. The CPU-based path remains the reliability floor.
3. **Try existing V100 only if CPU cannot meet acceptable control quality at the required policy size.** Request GPU-pool role/service via proper config; first measure acquire latency, preemption/cold-start/eviction behavior and total draw. If frequent swaps prevent a latency SLA, keep that policy offline and revert to CPU live.
4. **Measure A/V end-to-end.** If CPU encoding dominates host time, probe existing available video encoder capabilities (including exact V100 card/driver via FFmpeg test), else use CPU x264 at low resolution; spending on an encoder is last resort, not default.
5. **Measure training throughput separately.** Multiple CPU simulation workers can generate experiences while a V100 trains minibatches. GPU training speed and gameplay control runtime are different budgets. Set concurrency so interactive Orion cognition remains responsive.
6. **Only then justify hardware purchases** with reproducible before/after plots and an explicit blocker: CPU single-core bottleneck, training throughput, encoder capability, memory, network, or power/thermal headroom. No shopping list based on TFLOPS.

## 13. Implementation map (deliverable seams)

| Change | First implementation work |
|---|---|
| `services/orion-retro/` | Dockerfile, dependency/core pin, single-host compose profile disabled by default, health, runbook; no public ports without Hub auth |
| `services/orion-retro/app/emulator/` | LibretroAdapter, frame scheduler, ROM fingerprint, read-only integration profiles, replay/save handling |
| `services/orion-retro/app/control/` | bounded observation ring, authoritative port arbiter, lease/TTL, watchdog, baseline CPU policy, action ack |
| `services/orion-retro/app/media/` | PCM/frame ring, encoder abstraction, WebRTC audio/video and signaling; independent CPU media worker |
| `services/orion-retro/app/cognition/` | goals inbox/outbox, event summary producer, optional vision sample bridge, low-priority Orion planner client |
| `orion/schemas/retro.py` | contracts with strict Pydantic validation; register only deployed event producers/consumers |
| `config/bus/channels.yaml` or current registry path | channel catalog update after repo validation; do not invent an alternative registry |
| `services/orion-hub/scripts/` | Retro API proxy/auth, session lifecycle routes, WS/signaling authorization |
| `services/orion-hub/static/js/retro-panel.js` | Game tab, A/V, gamepad, ownership handoff, visible policy status and latencies |
| `services/orion-sql-db/` and writer | migrations/projection for durable, bounded session/outcome evidence, following current migration pattern |
| `tests/` + service-local tests/evals | deterministic ROM harness, policy harness, frame-time perf CLI, co-op and failure injection |
| `docs/runbooks/` | ROM catalog import, exact core fingerprints, operator start/stop, rollback, storage quotas, safety procedures |

File destinations are recommended; existing naming conventions and registry import cycle must be inspected before implementing. No code is included in this design PR.

## 14. Implementation sequence and go/no-go criteria

**P0 — Host/evidence preflight** (no new runtime dependency): enumerate Athena free physical cores and host load; confirm ROM legality and the first two games; pin exact core/Stable-Retro versions, validate audio requirements; inventory Hub auth/proxy and existing metrics. Deliver benchmark harness and issue list. Gate: measurable no-agent 60-Hz baseline, actual build hashes.

**P1 — Headless game authority**: one libretro backend, controller ports, in-memory frames/audio, deterministic step + save/replay, minimal local health API, test ROM. Gate: stable gameplay speed, determinism, safe stop, no Linux host escape.

**P2 — Human view/control**: same-origin Hub tab, WebRTC A/V, keyboard/gamepad, port ownership/leases, save/load, takeover and disconnect watchdog. Gate: interactive 30-min local session, operator takeover and P2 tested. **This already makes playing together technically possible** once Orion has a policy.

**P3 — Autonomous baseline**: CPU resident rule/script motor, 15 Hz observation-policy loop, goals inbox, sparse journal and outcome markers. Gate: independent 30-min session; no ghost buttons after failure; user can watch and interrupt; model identity says "scripted".

**P4 — Orion cognitive seam**: bounded game session facts/outcomes, goal selection and conversational summaries via existing Orion routes (never hot path), optional sampled visual observations; typed bus registration. Gate: an end-to-end trace proves observed state → selected goal → applied actions → validated or explicitly unknown outcome; LLM crash does not stop gameplay unexpectedly.

**P5 — Imitation/RL**: operator-owned demonstrations; Stable-Retro integration and training harness; reproducible experiments; export quantized CPU model and offline promotion gate. Gate: model at least outperforms random on held-out perturbed starts **and** meets runtime latency/compatibility. It is acceptable to ship without a learned policy if the data doesn't pass.

**P6 — Temporal integration / longitudinal evaluation**: replayable day/session arcs and tested effect on future gameplay; only after the strict consumer/producers are staged. Gate: evidence from two separate sessions that Orion recalled an actual mechanic/failed strategy with accurate source references, then behaviorally changed a decision; no unsupported self-claims.

**Feature gates/rollback:** `RETRO_ENABLED=false`, `RETRO_AGENT_CONTROL_ENABLED=false`, `RETRO_COGNITION_ENABLED=false`, `RETRO_TRAINING_ENABLED=false`, `RETRO_MEDIA_ENABLED=false`. Disabling agent control neutralizes Orion-owned ports at next frame while human session continues. Rollback must not delete saves or training evidence. All defaults off until local smoke succeeds.

## 15. Failure modes / risk register

| Risk | Severity | Countermeasure |
|---|---|---|
| LLM pause blocks game | Critical | no LLM/RPC on frame/motor deadline |
| Stuck button / ambiguous control ownership | Critical | authoritative arbiter, lease and TTL, neutral watchdog, deterministic handoff |
| Port swap P1/P2 during coop | High | validate core topology, display ownership, acceptance smoke |
| CPU overload / noisy Athena neighbors | High | benchmark busy host; bounded queues, affinity only if verified |
| Policy learned one ROM, applied to another | High | ROM+core+preprocessing compatibility manifest, default reject |
| Save-state incompatibility / corruption | High | hash-tagged states, atomically written, version gate and backups |
| Video stream disconnected; agent continues unseen | High | default pause unless operator opted autonomous-after-disconnect |
| Excessive storage from always-on capture | High | no raw frame persistence, record-on-request, hard quota, retention |
| GPU trainer steals chat capacity | High | existing GPU-pool low-priority lease, cancel/preempt, operator budget |
| Telemetry invents success/reward | High | verified RAM integration/evidence, unknown is not zero |
| Misleading AI competence in UI | Medium | always show policy type, version, success evidence and handoff |
| Unlicensed ROM distribution | High | operator-supplied read-only catalog, no ROMs in repo, no automated download |
| Unauthorized remote controller | Critical | Hub auth/session owner, ephemeral leases, private backend, audit and rate limits |
| Agent's speech floods conversation | Medium | explicit rate limit/mute, goal summaries not per-frame chatter |
| Strict temporal schema breaks readers | High | no new enum until consumer-first migration and migration tests |

## 16. Decisions and questions for implementation kickoff

**Proposed locked for v1:** no new hardware; one authoritative local emulator; NES/SNES first; operator-supplied ROMs; CPU-first fast controller; default no recording; human-priority lease arbitration; Hub as user surface; game state/events only into Orion's bus; existing GPU pool for expensive requests; feature gates default off.

**Must be measured before narrowing design:**
1. Which pinned libretro runner exposes all needed audio/video/controller/serialization callbacks with minimal code and appropriate license obligations? Benchmark libretro direct vs a RetroArchAI prototype; do not pick the latter from README alone.
2. Which first legal ROM and first *genuinely two-player* ROM provide deterministic smoke paths, expected port input effects and machine-checkable outcomes?
3. Browser input transport: DataChannel vs WebSocket on Carbon X1 over LAN/Tailscale, with measured input-to-photon latency and reconnection semantics.
4. Exact motor input features: raw pixels alone vs verified per-game RAM decoder; prove benefits and independence before adding a feature.
5. Actual Athena CPU contention/thermal headroom and whether another existing host offers lower effective latency.
6. Whether current Hub authentication/proxy supports WebRTC offer/ICE and WebSocket upgrade with correct origin/CSRF/session-owner enforcement. Treat this as a prerequisite, not a hand wave.

## 17. Definition of done for *this* PR

Documentation-only review can approve the intended *interfaces and build order*, not assert functionality. Done means:
- [x] Existing Orion seams and their boundaries called out with actual paths.
- [x] Functional modes, per-port authority, goals/non-goals and legal content policy specified.
- [x] Hard timing constraints separated from eventual cognitive behavior.
- [x] Wire/event/storage schemas and operator/API surfaces proposed with versioning.
- [x] CPU/V100 deployment hypotheses stated without invented measurements.
- [x] Failure, rollback, telemetry-quality, evals and acceptance gates defined.
- [ ] Implementation PRs, tests and live host measurements — **not performed**.

**Final recommendation:** build P0–P3 first. Until live latency and benchmark results say otherwise, spend **zero dollars** on hardware for this project.
