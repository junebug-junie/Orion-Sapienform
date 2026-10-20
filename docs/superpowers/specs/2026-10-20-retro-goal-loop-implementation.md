# Retro observed attempts and bounded goals

Implements the next P3/P4 seam in the original Retro cooperative-play design
(#2751), on top of merged #2796. This is an implementation record, not a claim
that the entire six-phase research roadmap is complete.

## Current architecture and capability change

The native worker owns observation, attempts, goal acceptance, applied input and
outcome. Hub's optional planner requests a supported goal through the existing
registered `llm.chat.request` contract on the `metacog_background` route. It uses
a fresh RPC client and never blocks the native frame/input path. No new global
bus shape, schema enum, memory writer or scheduler is introduced.

The operator explicitly hands off a live owned port. `goal-motor-v1` begins at
wait; the browser requests at most one plan at a time. A native acknowledgment
is required before changed inputs. Accepted goals last five seconds and can
only be wait, walk left/right or pulsed jump left/right. This is a rule motor
with optional LLM goal choice, not a learned policy. Unknown game observations
permit only wait in the planner. Direct operator goal requests are explicit
motor commands, not evidence of an LLM decision.

## Contracts

- `ScriptControl.cognition`: optional boolean, default false; both service gate
  and explicit handoff are required. Existing fixed baseline remains available.
- `POST /sessions/{id}/goal`: live sponsoring connection, generation, port,
  exact controller epoch, current attempt ID, supported goal, bounded rationale
  and planner identity. Native authority rejects stale/replaced/expired grants.
- `POST /api/retro/sessions/{id}/plan`: same-origin Hub route. Reads native facts,
  invokes the existing gateway, then submits the returned goal through the same
  native validation. A slow result cannot reacquire a controller or start a game.
- Native status exposes the latest offered attempt and last accepted goal.
  These are bounded projections, not an assertion the last goal is still active.
- Journal: attempt_finished, goal_committed, goal_action_started, actual applied
  controls and policy lifecycle. Every native row includes session/generation/frame.
  The session's start record contains ROM/core/runtime digests. Applied inputs
  reference the accepted attempt. Goal execution baseline is captured at the
  first applied frame, not before asynchronous planning. Outcomes are observed
  after the five-second action window; unaccepted offers expire after 30 seconds.

Pause, takeover, disconnect, budget expiry and deadline miss still cancel the
controller. Planner timeout, invalid JSON, unsupported output or GPU refusal
produces no fallback goal. The motor returns to wait at the accepted deadline.
No automatic script restart. A pending planner cannot override a human.

## Observation quality gate

1. **Provenance:** `app/observation.py:Observer.read` copies exactly 2048 bytes of
   libretro SYSTEM_RAM, only for the qualified ROM and two verified FCEUmm binary
   fingerprints. FCEUmm source `ac0d00a7`,
   `src/drivers/libretro/libretro.c:retro_get_memory_data/size`, maps type 2 to
   NES RAM. Nova source `02a3913f72713ec2e3d1c34fbd39541cd45179a0`,
   `src/memory.s` and `src/nova.x`, compiled with ca65/ld65, gives health $4b,
   X low/high $25/$26, Y high/low $27/$28, level $a7, ground $46.
   `global.s:HurtPlayer` decrements health; `main.s` reads it for its death path.
   The adapter does **not** label death or level completion.
2. **Independence:** health and coordinates share the same RAM observation.
   Outcome labels are deterministic comparisons of those fields, not additional
   independent signals. They are not summed into any cognitive metric.
3. **Theory:** direct instrument reading of program state, fixed-point 12.4
   coordinates and half-heart health; finite differences over an attributed
   action window. No-horizontal-progress means coordinate equality, not a
   universal failure detector; health-decreased means measured health loss,
   not proof of enemy damage. A reset/level change is unknown.
4. **Live data/rest:** native installed Nova v1.0.6a starts with unqualified RAM
   (255, then zero), correctly unknown. After menu inputs health=4 and X=52px.
   Walking right reaches X=136.9375px and stays there across further walking.
   Neutral is not classified as failure. Pulsed jump-right moves beyond the
   obstacle; the observed test ended at X=687.9375px with health=3. This is
   position change **and damage**, not success. The independent source-probe
   run also observed health 4→3→2 and stable rest values between changes.
   Raw runs/artifacts remain local under `/tmp/retro-cognition/`.
5. **Existing mechanism:** prior Retro code only exposed frame hashes and
   applied masks. Field goals, action-prediction and retired self-experiment
   dispatchers do not supply game state. No duplicate generic telemetry system.
6. **Reversibility:** two default-off gates; remove the service-local adapter
   and goal path without changing stored cognition schemas, training defaults,
   attention, reward, self-model, identity or memory. Unknown binary fingerprints
   disable interpretation. No host addresses or arbitrary RAM access API.

The finite-difference verdict is intentionally narrow. RAM-assisted observations
are explicit. Titles/menus and transition states can remain unknown; there is
no universal game-state detector or advertised pixels-only competence.

## Privacy and rollback

Only small game-state facts and local action provenance reach the planner. No
raw RAM, ROM, frames, chat, private recall, recordings or training export. New
gates `RETRO_COGNITION_ENABLED` and `HUB_RETRO_COGNITION_ENABLED` default false;
local ignored env files received these keys. Stop the session before disabling
and recreating services. Nothing deletes evidence or saves. No live deployment
is part of this patch; restarting Retro can destroy the existing paused game.

## Acceptance and limits

- Deterministic tests cover explicit handoff, current epoch/attempt, idempotency,
  expiry, takeover, sponsor mismatch, unsupported goals, unknown states, delayed
  planning and correct action-window attribution.
- Native mechanic eval uses the installed pinned ROM, real menu inputs, real
  core stepping and authoritative arbiter. `--mechanic-only` uses a clearly
  named fixture decision and **cannot** pass the real cognition gate.
- Browser eval uses a synthetic wait planner and real native goal acceptance,
  applied evidence and human takeover. It is separately labeled in its artifact.
- `cognitive_loop.py` without `--mechanic-only` requires an actual gateway choice;
  missing/invalid output fails and saves a failure artifact. Current live proof
  is BLOCKED: GPU pool returns `gpu_pool_unavailable` while shedding work for
  `cabinet_hot`. The thermal admission rule is respected, not bypassed.
- Thus observed mistake → actual Orion revision → changed input → checked outcome
  remains **UNVERIFIED** until that real-planner eval passes. Infrastructure and
  the game mechanic are testable independently; no fabricated cognition result.
- Original open gates remain: WebRTC, save/load, SNES qualification, actual-client
  30-minute A/V/physical-controller tests, independent 30-minute policy run,
  shared-evening social acceptance, leisure admission, training/promotion,
  world-model experiment, cross-session recall and transfer. Those are not
  represented as completed by this bounded goal-loop patch.
