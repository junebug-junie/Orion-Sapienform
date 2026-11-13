# Retro continuation: scripted baseline and remaining gates

## Arsonist summary

This is the first P3 infrastructure slice, not completed P3 or demonstrated learning.
The roadmap remains [original proposal #2751](https://github.com/junebug-junie/Orion-Sapienform/pull/2751)
(still open at this audit). Its spec is on that PR's branch, not on main.
The playable-authority follow-up narrows delivery; neither it nor this patch
waives the original acceptance criteria.

The old 502 report is superseded: the public Retro page returned 200 and a real
browser loaded all six catalog games without a code. The live paused game was
preserved. Subsequent merged work includes #2771 (input pipeline), #2772 (grant
grace/self-heal), #2774 (lease TTL), #2787 (dedicated relay/play harness), and
#2793 (tap latch/core startup). Do not reimplement those patches.

## Current architecture

- Services: native `orion-retro`; Hub page plus its dedicated Retro relay.
- Entry points: Retro `app/main.py`, spawned `app/worker.py`; Hub
  `scripts/retro_routes.py` and `scripts/retro_gateway.py`.
- Config: Retro `settings.py` and `.env_example`; Hub `app/settings.py` and
  `.env_example`. Both have `docker-compose.yml`, Dockerfile and requirements.
- Transport: private Unix socket; emulator network disabled; same-origin
  browser WebSockets. No additional Retro access code.
- Bus channels/schema registry: none for Retro; no Retro cognition consumer.
- Tests: `tests/retro`, `services/orion-retro/tests`, Hub `tests/retro`.
- Evals: native/container smokes and Hub `evals/retro_browser.py` / `retro_play.mjs`.
- Current gap: human input infrastructure exists, but no trained policy,
  supported cognition goal inbox, validated game outcome, or learning loop.

## Missing questions

These are unresolved research/qualification gates, not configuration defaults:

1. Actual-client 30-minute latency/audio-drift, physical controller and speaker
   output remain UNVERIFIED. Server acknowledgments and scheduled PCM cannot
   replace those measurements.
2. Which pinned game mechanic supplies a validated failure/outcome observation?
   Pad Lab establishes independent buttons, not platforming competence.
3. Which live planner path should consume a game attempt and return a bounded
   supported goal? Inspect its current producer/consumer before wiring it.

## Proposed schema / API changes

Implemented in this slice: service-local strict `ScriptControl`, routed through
the existing owner/origin boundary as `POST /sessions/{id}/scripted`.
Start requires a live input connection, current generation, port, exact expected
human lease epoch, and a 5–1800 second duration (UI: 60 seconds). It transfers
only that connection's already-owned port; a delayed handoff cannot consume a
newer human lease. Stop checks the expected script epoch.

The fixed `scripted-right-jump-v1` policy emits at most 15 decisions/second,
holds each decision at most 100 ms without refresh, and never catches up after
a stall. It does not inspect images/RAM or select a goal. The native arbiter
remains authoritative. Human takeover inserts a neutral frame and permanently
ends that script grant. Pause, sponsor disconnect, deadline failure and expiry
also cancel it. Resume never restarts it. Keep the sponsoring browser connected;
independent leisure without a browser is outside this slice.

Session status adds `scripted` summaries (port, policy ID, epoch, duration and
start frame); journal adds `policy_started` / `policy_stopped`. Applied script
actions carry `policy_id` plus the existing epoch/sequence/frame fields.
Sparse lifecycle records and the existing bounded applied-action journal are
local evidence; nothing is exported to bus, memory, training, or model metrics.
No token or connection identifier is exposed in the policy summary/journal.

### Cognition seam: proposal only

Inspected `orion/schemas/field_goal.py`, `orion/schemas/action_prediction.py`,
`orion/autonomy/prediction.py`, `orion/feedback/outcome_resolution.py`,
`orion/schemas/self_experiments.py`, and the Temporal Self source contract.
Field goals describe attention competition; action predictions score a closed
set of field signals; self-experiment dispatch has been retired. None supplies
a ready-made Retro mechanic/outcome loop. Do not manufacture game rewards into
those schemas or resurrect their retired dispatchers.

Next experiment should add a service-local attempt/goal/outcome contract only
alongside its actual consumers:

- Attempt: session/generation, ROM/core fingerprint, port/controller epoch,
  observed frame window, evidence reference, validated failure or `unknown`.
- Goal request/ack: attempt ID, expected controller epoch, supported mechanic
  goal, expiry; explicit accepted/rejected result before changed input.
- Outcome: actual applied-action window and independently checked observation,
  including `unknown` and human-intervention exclusion.

Capability change: an observed mistake may cause an existing live planner to
request a different supported action, instead of replaying a fixed script.
Data: local attempt windows and bounded game observations; no chat/private
recall or automatic memory writes. Privacy boundary: explicit session opt-in,
no recording/export by default. Proof: failed attempt → evidence-linked planner
revision → acknowledged goal → different applied action → checked outcome.
Dangerous failure: a stale planner response overrides a human or fabricated
success becomes memory. Disable: cancel the experiment, revoke its lease and
reject further goals; human input keeps working. Metric quality gates apply
before any observation is promoted into a cognition signal.

## Files likely to touch

This patch: Retro contracts, API, worker, session projection and `app/scripted.py`;
Hub proxy, page and client; existing tests/browser eval; runbook.

Next cognition patch (after choosing/validating the mechanic and live planner):
service-local attempt contracts and goal reducer, a bounded observation adapter,
an isolated outcome eval, and the selected planner's documented API consumer.
No global schema/channel entries until real producers and consumers exist.

## Non-goals

Learned competence, autonomous leisure admission, memory/identity changes,
RAM-derived rewards, world-model revival, arbitrary uploaded scripts, save/load,
WebRTC migration, ROM installation, or replacing an existing user game.

## Acceptance checks

| Original gate | Current verdict |
|---|---|
| No-code page and catalog | Live verified; six games visible |
| Human start/input/pause/reconnect, synthetic P2 | Isolated native browser eval; live paused game preserved |
| WebRTC transport | DEVIATION: WebSockets remain; documented in browser runbook |
| Save/load and strict replay qualification | Save/load implemented (#2825, hotkeys #2832); strict replay qualification remains failed/UNVERIFIED |
| Actual-client 30-minute latency/audio drift | UNVERIFIED |
| Physical gamepad and speaker output | UNVERIFIED |
| Mario browser/audio qualification | UNVERIFIED; prior native/input evidence is not full qualification |
| Explicit scripted identity, bounded handoff, takeover | This patch; deterministic tests and native browser eval |
| P3 independent 30-minute policy run | UNVERIFIED; this slice requires connected sponsor |
| P3 observation/goal inbox and sparse outcomes | Implemented for Nova v1.0.6a only (#2801): RAM-assisted attempts, native goal ack, five-second outcomes |
| Mistake → revised goal → changed action → checked outcome | BLOCKED: `cognitive_loop.py` real-planner run refused `gpu_pool_unavailable` on 2026-10-20 and again 2026-10-21 05:17 UTC (cabinet 33.6 °C, `cabinet_hot` shed). Mechanic-only fixture passes |
| Shared-play social behavior, transfer, independent leisure | NOT IMPLEMENTED |

## Status update 2026-10-21

#2801 implemented the attempt/goal/outcome seam proposed above
([implementation record](2026-10-20-retro-goal-loop-implementation.md)); #2825
added save/load. Both cognition gates now default on. A per-session handoff is
still required. The open cognition gate is the real-planner eval, which waits
on cabinet temperature. Do not bypass the thermal shed. Rerun when
`orion:regulation:latest` no longer lists `cabinet_hot`:

```bash
python services/orion-retro/evals/cognitive_loop.py \
  --core ~/.local/share/orion/retro/games/nova-the-squirrel-v1.0.6a/cores/fceumm-ac0d00a7.so \
  --rom ~/.local/share/orion/retro/games/nova-the-squirrel-v1.0.6a/roms/nova.nes \
  --bus-url "$ORION_BUS_URL" --output /tmp/retro-cognition/real-loop.json
```

## Recommended next patch

Validate one actual game-mechanic failure/outcome reader in an isolated native
eval, with unknown and human-takeover cases. Then wire one bounded goal through
an inspected live planner. A hardcoded script change must not pass the cognition
gate. Preserve the original client/audio and 30-minute gates as separate work.
