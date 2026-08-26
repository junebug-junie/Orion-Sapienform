# Orion hears Juniper, and each spoken turn is bracketed by an affect record

## Summary

- **Orion now knows a turn was spoken.** Hub already transcribed Juniper's
  microphone into the unified turn — that was verified live before anything
  was written — but the turn had no way to tell a dictated sentence from a
  typed one. `surface_context` is now carried into the situation builder and
  rendered as a real prompt line.
- **New per-turn affect bracket** (`services/orion-hub/scripts/chat_turn_affect.py`):
  one AffectGPT capture as an Orion-mode turn starts, one as it finishes.
- **`AFFECT_CHAT_TURN_SCOPE`** (`off` | `voice` | `all`, default `voice`)
  gates it. Fails closed on any unrecognized value.
- **Contract**: `JuniperMultimodalAffectV1.trigger` gains `chat_turn_pre` /
  `chat_turn_post`; new `chat_correlation_id` field joins a capture to the
  turn that caused it, and a turn's pre/post pair to each other.
- **Fixed a latent crash** introduced mid-change and caught by the existing
  suite: the new `_normalize_trigger` set-membership test raised `TypeError`
  on unhashable input.

## Outcome moved

Before: an affect capture could only be produced by Juniper pressing "Check
now" or leaving the ambient toggle on — both untethered from any particular
conversation. There was no way to ask *"how did Juniper's affect move across
this exchange?"* from stored events, because nothing produced a matched pair
around a known stimulus, and `observed_at` proximity cannot separate a
turn-adjacent capture from a concurrent ambient tick.

After: a spoken Orion-mode turn emits two `orion:affectgpt:assessment` events
sharing one `chat_correlation_id`, joinable to the turn and to each other.

Separately: `SurfaceContextV1.input_modality` went from a schema field that
reached **no prompt at all** to one that changes what Orion is told.

## Current architecture (before this patch)

Verified live against the running system on 2026-08-25 rather than read off
the code:

- **The microphone already worked into the unified turn.** A real
  WebSocket probe in `mode: "orion"` carrying real audio produced, in the
  live `orion-athena-hub` log:
  `voice.ws.audio_received` → `voice.stt.done transcript_len=50` →
  `orion:cortex:pre_turn_appraisal:request` → `run_unified_turn`.
  `websocket_handler.py`'s STT block sits *ahead* of the Orion-mode branch,
  so it was never bypassed.
- **The Whisper rail itself was healthy.** A TTS→STT round-trip on the real
  bus (`orion:tts:intake` → `orion:stt:intake`) returned
  `'Hello Orion. Can you hear my microphone now?'` with
  `silence_gate: 'passed'`.
- **The real gap was one level in.**
  `orion/hub/turn_orchestrator.py::_build_situation_prompt_fragment` built
  its `situation_ctx` from only `session_id` / `raw_user_text` /
  `presence_context`. `_build_surface_context` reads
  `ctx["metadata"]["surface_context"]`, found nothing, and fell through to
  its `"typed"` default on **every** unified turn.
- **And even if it had been carried,** `_build_prompt_fragment` never
  rendered `input_modality`, so the value reached no prompt either way.
- **Affect captures existed but were conversation-blind**: the manual route
  and `vision_affect_ambient`'s loop both call
  `POST /v1/juniper/affect/capture_and_assess`, sharing one exclusive
  capture slot.

## Architecture touched

| Seam | Change |
| --- | --- |
| `orion/hub/turn_orchestrator.py` | passes `payload["surface_context"]` into the situation ctx under `metadata` |
| `orion/situational/context.py` | renders a spoken-input line in the prompt fragment |
| `orion/schemas/affectgpt.py` | `trigger` widened; `chat_correlation_id` added |
| `services/orion-juniper-affective-state` | threads both through `capture_and_assess` → `trigger_assessment` → `_wrap_event` |
| `services/orion-hub/scripts/chat_turn_affect.py` | **new** — the bracket itself |
| `services/orion-hub/scripts/websocket_handler.py` | two fire points inside the Orion-mode branch |

## Files changed

- `orion/hub/turn_orchestrator.py`: carry `surface_context` into the situation builder (nested under `metadata`, copied not aliased).
- `orion/situational/context.py`: render the spoken-modality line; stay silent for `typed`/`unknown`.
- `orion/schemas/affectgpt.py`: two new `trigger` values + `chat_correlation_id`.
- `orion/bus/channels.yaml`: document the additive, backward-compatible change on `orion:affectgpt:assessment`.
- `services/orion-juniper-affective-state/app/main.py`: `_normalize_trigger` rewritten around `_VALID_TRIGGERS`; `chat_correlation_id` threaded through every path **including both failure paths**; missing `Field` import added.
- `services/orion-juniper-affective-state/README.md`: the bracket, the two join axes, and why `subtitle` is deliberately empty.
- `services/orion-hub/scripts/chat_turn_affect.py`: **new**.
- `services/orion-hub/scripts/vision_affect_ambient.py`: the single shared call site gains an optional `chat_correlation_id`, omitted from the body when unset so manual/ambient requests stay byte-identical on the wire.
- `services/orion-hub/scripts/websocket_handler.py`: pre fire before `run_unified_turn`, post fire in its `finally`.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: `AFFECT_CHAT_TURN_SCOPE`.
- 4 test files (3 new, 1 renamed-and-annotated).

## Schema / bus / API changes

- **Added**: `JuniperMultimodalAffectV1.chat_correlation_id` (optional);
  `CaptureAndAssessRequest.chat_correlation_id`.
- **Behavior changed**: `trigger` Literal widened from `manual|ambient` to
  include `chat_turn_pre|chat_turn_post`.
- **Compatibility**: additive. Existing consumers matching `manual`/`ambient`
  are unaffected; `chat_correlation_id` is absent on those two triggers.
  `_normalize_trigger` still clamps anything unrecognized to `"manual"`
  rather than raising.

### Two join axes, deliberately not one

| field | joins |
| --- | --- |
| `correlation_id` (existing) | the three legs of ONE capture attempt (retina RPC, worker RPC, event) |
| `chat_correlation_id` (new) | a capture to the turn that caused it; a turn's pre/post pair to each other |

Reusing `correlation_id` would have destroyed the first meaning to get the
second.

## Env/config changes

- Added key: `AFFECT_CHAT_TURN_SCOPE=voice` (orion-hub).
- `.env_example` updated: yes.
- Local `.env` synced: **by hand, deliberately.**
  `scripts/sync_local_env_from_example.py` logged
  `reading live .env from primary checkout: /mnt/scripts/Orion-Sapienform`
  — it reads `.env_example` from the **primary checkout**, so a key added in
  a worktree is invisible to it and it reported no change. Confirmed the key
  was genuinely absent from the live `.env`, then added it there directly and
  re-verified. This is a known trap, not a new one.
- Also added to `docker-compose.yml`'s `environment:` block. Note the comment
  there is accurate about *why*: orion-hub declares `env_file: .env`, so the
  key would have reached the container anyway
  (`check_service_env_compose_parity.py orion-hub` reports N/A for this
  service for that reason). The entry is for visibility and robustness, not
  necessity — an earlier draft of that comment claimed otherwise and was
  wrong.

## Tests run

```text
# new + directly affected
services/orion-cortex-exec/tests/test_situation_input_modality.py    5 passed
services/orion-hub/tests/test_chat_turn_affect.py                   21 passed
services/orion-hub/tests/test_unified_turn_surface_context.py        8 passed
services/orion-juniper-affective-state/tests/                       44 passed

# whole hub suite, branch vs origin/main, identical subset comparison
branch:        13 failed, 55 passed, 1 skipped   (the 7 files that fail)
origin/main:   13 failed, 55 passed, 1 skipped   (same 7 files, same tests)
=> all failures pre-existing; none introduced here.

# also confirmed pre-existing on origin/main:
services/orion-cortex-exec/tests/test_situation_prompt_integration.py
  2 failed (FileNotFoundError on a .j2 path) — fails identically at base.
```

## Evals run

```text
No eval harness exists for orion-hub or orion-juniper-affective-state.
Not added here — the behaviour this patch adds is covered by deterministic
gate tests, and inventing an eval harness for it would be scope creep.
Flagged as a real gap: there is no quality measure of whether the affect
bracket's readings are USEFUL, only that they are produced and joinable.
```

## Docker/build/smoke checks

```text
docker compose --env-file <primary>/.env \
  --env-file <primary>/services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml config
  => AFFECT_CHAT_TURN_SCOPE: voice     (resolves correctly from live .env)

Note: scripts/safe_docker_build.sh could not be used for this check from the
worktree — it passes --env-file .env relative to the worktree root, and the
gitignored root .env does not exist there ("couldn't find env file"). Ran the
equivalent read-only `config` directly against the primary checkout's env
files instead. No build, no deploy, nothing brought up.
```

## Live verification performed (pre-change)

```text
TTS->STT round-trip on the real bus (redis://100.92.216.81:6379/0):
  STT TEXT = 'Hello Orion. Can you hear my microphone now?'
  STT META = {... 'silence_gate': 'passed'}

Real WebSocket probe, mode=orion, real audio, live orion-athena-hub:8080:
  voice.ws.audio_received session_id=voice-probe-0d3db222
  voice.stt.done session_id=voice-probe-0d3db222 transcript_len=50
  -> orion:cortex:pre_turn_appraisal:request -> run_unified_turn
```

## NOT verified live

The bracket itself has **not** run against the real webcam — that requires
deploying Hub, which would also start recording Juniper on every spoken turn.
Held deliberately for Juniper's call. Status is `DONE_WITH_CONCERNS`, not
`DONE`, for exactly this reason.

## Restart required

```bash
# Hub (worktree only; the wrapper refuses the shared checkout)
cd /mnt/scripts/Orion-Sapienform-orion-hears-juniper-affect-bracket
bash scripts/safe_docker_build.sh orion-hub up -d --build

# orion-juniper-affective-state runs on circe:
ssh circe@circe
git pull
docker compose --env-file .env \
  --env-file services/orion-juniper-affective-state/.env \
  -f services/orion-juniper-affective-state/docker-compose.yml up -d --build
```

Both must be redeployed. Hub alone would send `chat_turn_pre` to an affect
service whose `CaptureAndAssessRequest` still rejects it — `extra="ignore"`
would drop `chat_correlation_id` silently and the Literal would 422 the
trigger.

## Risks / concerns

- **Severity: medium — privacy surface widens.** A spoken turn now triggers
  two webcam+mic recordings of Juniper that no one pressed a button for.
  Mitigation: default scope is `voice` (never typed turns), the mic press is
  itself an explicit per-turn physical action, `off` is one env change, and
  the setting fails closed on any typo.
- **Severity: low — the pre-capture does not colour its own turn.** It
  cannot: captures take up to ~195s and are detached. It lands in the 300s
  situational mirror for the *next* turn. Stated in code, README and
  `.env_example` rather than left to be discovered.
- **Severity: low — a pair can be half-present.** All callers share one
  capture slot; a losing leg is dropped and logged, never queued. Consumers
  must treat a lone `chat_turn_pre` as an explainable gap.
- **Severity: low — deploy ordering.** See "Restart required".

## PR link

<filled in after push>
