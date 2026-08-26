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
- **Made the join durable.** `JuniperMultimodalAffectSQL` had no column for
  `chat_correlation_id`, and `_write_row` filters payload keys against the
  mapper's columns — the key would have been silently dropped. Added the
  column plus boot-time DDL.

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
- `services/orion-sql-writer/app/models/juniper_multimodal_affect.py`: `chat_correlation_id` column (indexed, nullable).
- `services/orion-sql-writer/app/main.py`: boot-time `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + index, following the existing `chat_*` convention (`create_all` does not alter existing tables).
- `services/orion-sql-writer/README.md`: the new column and why it is a separate join axis.
- 5 test files (4 new, 1 renamed-and-annotated; plus 3 cases added to the sql-writer shape test).

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

services/orion-cortex-exec/tests/test_situation_*.py (5 files)       85 passed
services/orion-hub/tests/test_chat_turn_affect.py                   29 passed
services/orion-hub/tests/test_unified_turn_surface_context.py        8 passed
services/orion-juniper-affective-state/tests/                       44 passed

services/orion-sql-writer/tests/  branch:  11 failed, 446 passed, 3 skipped
services/orion-sql-writer/tests/  main:    11 failed, 443 passed, 3 skipped
=> same 11 pre-existing failures; +3 are this patch's new tests.

# whole hub suite, branch vs origin/main, identical subset comparison
branch:        13 failed, 55 passed, 1 skipped   (the 7 files that fail)
origin/main:   13 failed, 55 passed, 1 skipped   (same 7 files, same tests)
=> all failures pre-existing; none introduced here.

# whole hub suite, full run, both branches (post-review-fix)
branch:        33 failed, 1521 passed, 5 skipped
origin/main:   33 failed, 1487 passed, 5 skipped

The +34 passed are this patch's new tests. The two FAILED sets are
byte-identical -- `comm` in both directions returns empty. No test fails on
this branch that does not also fail on origin/main.

(An earlier pre-fix run showed 34 vs 33 with sets that differed in BOTH
directions -- 20 baseline-only, 1 branch-only. That was test-ordering
pollution in this suite, confirmed by running the one branch-only failure,
test_substrate_mutation_manual_route_routing.py::test_routing_manual_apply_changes_real_live_routing_surface,
in isolation on both: passed on both.)

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

## Review findings fixed

High-effort review over `main...HEAD`. Seven findings, all material, all
fixed. Each was independently reproduced before fixing — none were taken on
the reviewer's word.

- **Finding: situation-brief cache key omitted `input_modality`.** The brief
  is cached per session for 300s, so a spoken turn's brief was replayed on
  the next *typed* turn — telling Orion "Juniper SPOKE this turn aloud" about
  something she typed (and suppressing it on a real spoken turn in the
  reverse order). Invisible before this PR only because the value was a
  constant.
  - Fix: `input_modality` is part of `_situation_cache_key`.
  - Evidence: `test_cache_key_separates_spoken_from_typed`.

- **Finding: the new spoken line pushed the fragment past the 1200 cap and
  cut the affect privacy caution mid-sentence.** Reproduced exactly:
  typed = 1021 chars (cautions intact), spoken = 1200 truncated, with
  `"…don't announce it unprompted."` sliced in half — i.e. the guard on how
  to handle a webcam reading of Juniper was dropped from precisely the turn
  that fired one. My own truncation test asserted truncation *occurred* and
  never checked what was lost.
  - Fix: two parts. (1) The spoken line was shortened from 246 to ~140
    chars, so at the production cap nothing truncates at all (spoken now
    1175/1200 with a maximal 300-char affect summary). (2) Structurally,
    cautions are now appended **whole or not at all**, never sliced, and are
    ordered most-important-first so the affect guard is the last to go.
  - Evidence: `test_nothing_is_truncated_at_the_live_cap_even_with_a_max_affect_summary`,
    `test_a_caution_is_never_emitted_half_written` (six caps, 200→4000),
    `test_the_affect_guard_is_the_last_caution_to_be_dropped`.
  - Note: an intermediate fix reserved the *entire* caution block up front.
    That was an over-correction — it starved the body to boilerplate at the
    400-char cap `test_situation_provider`'s fixture uses, breaking 4
    pre-existing tests. Body keeps first claim on the budget in the shipped
    version.

- **Finding: the post leg usually lost the capture slot to its own pre leg.**
  `_capture_blocking` holds the lock for the whole round trip (~8s clip +
  ~20s warm inference). The post leg fired the instant the turn returned, so
  any turn shorter than that was dropped — meaning the matched pair the
  feature exists to produce mostly would not exist.
  - Fix: the post leg now *awaits* this turn's own pre leg (via
    `_PENDING_PRE`, self-cleaning, bounded by the pre leg's own timeout)
    rather than racing it. A failed pre leg does not skip the post leg.
  - Evidence: `test_post_leg_waits_for_its_own_pre_leg_instead_of_racing_it`
    asserts strict ordering; `test_post_leg_still_runs_when_the_pre_leg_failed`.

- **Finding: the `finally` fired a webcam capture after the client
  disconnected.** Closing the tab mid-turn cancels the turn, the
  cancellation lands in the `finally`, and a live recording of Juniper
  started *after she left* — the same objection the pre leg's own comment
  makes.
  - Fix: post leg is skipped (and logged) unless
    `websocket.client_state == CONNECTED`. A disconnect is the one case
    where the missing half of the pair is the correct outcome.

- **Finding: the module docstring claimed "off by default at the service
  boundary" while the code shipped `voice`.** A plain contradiction I wrote.
  - Fix: docstring rewritten to state the truth and make the actual consent
    argument — the right analogue is the manual "Check now" button (each
    capture preceded by a deliberate human act: the mic press), not the
    ambient loop (which resets to off on restart precisely because it runs
    with no human in the loop). It also now names what that argument does
    *not* cover: `all` fires on typed turns where no such act exists, and
    there is no UI toggle.
  - Default remains `voice` — Juniper asked for this directly.

- **Finding: the compose `environment:` entry could *override* the kill
  switch.** Compose gives `environment:` precedence over `env_file:`, and
  `${AFFECT_CHAT_TURN_SCOPE:-voice}` interpolates from the compose
  invocation's env-file context (the repo-root `.env`, which lacks the key),
  not from `services/orion-hub/.env`. An operator setting `off` in the
  service `.env` would have been silently overridden back to `voice`. The
  comment I wrote claimed the opposite.
  - Fix: the entry is removed entirely. `env_file: .env` already delivers
    the key (`check_service_env_compose_parity.py orion-hub` → N/A). A
    comment in its place explains why it must stay absent.

- **Finding: chat-turn captures mutated the ambient loop's scheduling state
  and the Vision panel.** `try_begin_capture` bumps `last_attempt_at`, from
  which `affect_ambient_loop` computes whether a tick is due — so a
  conversation with a spoken turn more often than every 5 minutes would
  reset the ambient due-clock every turn, leaving the toggle reading
  "enabled" while never firing. It also overwrote the panel's
  `last_trigger`/`last_result_ok`/`last_raw_response`, which exist to report
  what the *operator* started.
  - Fix: `try_begin_capture(..., record_state=False)` / `end_capture(...,
    record_state=False)` take the shared mutex without touching observable
    state. The mutex is still shared — one physical camera.
  - Evidence: `test_chat_capture_does_not_disturb_ambient_scheduling_or_the_panel`
    (asserts the full state tuple is unchanged, then proves the lock was
    genuinely taken and released) and `test_manual_capture_still_records_state`
    (the default path is untouched).

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

```bash
# orion-sql-writer (applies the boot-time ALTER on startup)
cd /mnt/scripts/Orion-Sapienform-orion-hears-juniper-affect-bracket
bash scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

**Order matters.** Deploy `orion-juniper-affective-state` and
`orion-sql-writer` BEFORE Hub:

- Hub alone would send `chat_turn_pre` to an affect service whose
  `CaptureAndAssessRequest` still rejects it — `extra="ignore"` drops
  `chat_correlation_id` silently and the `Literal` 422s the trigger.
- The affect service alone, publishing a `chat_correlation_id` an
  un-migrated `orion-sql-writer` has no column for, is safe in that
  direction (`_write_row` filters it out) — but the *reverse* is not: a
  sql-writer running the new model against a table lacking the column
  raises `UndefinedColumn`, which its handlers do not catch, halting ALL
  `juniper_multimodal_affect_log` writes. The boot-time DDL is what closes
  that, so sql-writer must be restarted (not just rebuilt) for the ALTER to
  run.

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
