# Ask "who is this?" when Orion cannot see, not only when it misrecognizes

Juniper, 2026-08-29: *"identity_recognition we never got this to work. orion
never bites when they can't recognize me (eg I close the camera lid). they are
supposed to ask who dis"*

## Summary

- The identity-ask feature was fully built, fully tested, and **structurally
  incapable of firing in the case it was built for.** Four independent
  blockers, each fatal on its own.
- `identity_confirmed` added to the presence snapshot: the positive fact.
  Absence of `identity_uncertain` conflated "matched" with "never looked".
- Presence reads resolve across cameras and now carry the row's write time, so
  a camera that goes dark can be told from one that is watching an empty room.
- The ask decision moved ahead of the percept staleness gate a dark camera
  trips, and split into two reasons with separate cooldowns and separate
  prompt text.
- `carbon` (Juniper's laptop webcam) opted in to `identity_face` dispatch.
- Latent bug fixed in passing: the presence read propagated database errors out
  of turn assembly while its own docstring claimed it failed open.

## Outcome moved

Orion can now ask "is that you, Juniper?" when it has **no fresh confirmed
visual read** — lid closed, camera off, empty frame, vision stack down — not
only when a face was detected and failed to match. Live-verified against
production Postgres: the resolver picks `carbon` (present, row age 3.1s) over
`cam0`, and the decision comes out `no_visual_confirmation`.

## Current architecture (before this patch)

```
vision-window/presence.py
  identity_uncertain = present_now AND identity_confidence == "uncertain"
        |                                    ^ requires a face DETECTED and unmatched
        v  substrate_embodied_presence (JSONB, one row per stream)
cortex-exec  orion/situational/context.py::_build_perception_context
  percept stale? -> return early            <-- ask died here
  presence = fetch_presence("cam0")          <-- one hardcoded camera
  if identity_uncertain and claim_cooldown: presence_identity_uncertain = True
        v
  prompt caution: "the person currently in view..."
```

### The four blockers, all confirmed live 2026-08-29

1. **The lid-closed case had no code path — by design.** `"uncertain"` means a
   face was *detected and did not match*. A closed lid emits no frames, so no
   face, so `identity_confidence` is `None`, not `"uncertain"`. The signal
   fired on *"a stranger is looking at the camera"*; Juniper described *"I
   can't see anyone"*. Opposite conditions; only the first existed.
2. **`identity_face` was opted out on `carbon`.**
   `config/vision_frame_router.yaml` enabled `identity_dispatch` only under
   `streams.cam0`; `carbon` inherited `defaults.triggered`, which has no such
   key, so `policy.py::decide_identity` short-circuited on
   `cfg.get("enabled", False)`. The face check had **never once run on the
   camera pointed at Juniper's face.** Evidence: 2 identity dispatches in 24h,
   both `stream_id=cam0`. `carbon` does reach the `triggered` tier (6 hits in
   6h), so it was not inert for want of traffic.
3. **The chat turn read the wrong camera.** `ORION_SITUATION_PERCEPTION_STREAM_ID=cam0`,
   verified inside `orion-athena-cortex-exec-chat`. Live at the time:

   | stream | state | last seen |
   |---|---|---|
   | `carbon` | **present** | 0.0s |
   | `cam0` | absent | 4170s (70 min) |

   The prompt was narrating an empty room at someone sitting at her desk.
4. **The decision lived behind the staleness gate.** Even had the signal
   existed, it was computed inside the `available=True` branch. A dark camera
   produces no fresh percepts, so `_build_perception_context` returned
   `source="stale"` long before reaching it.

Consequently `identity_uncertain` had never been `true` in any presence row.

## Architecture touched

```
vision-window/presence.py    + identity_confirmed (the positive fact)
frame-router yaml            + carbon identity_dispatch (explicit opt-in)
perception_reader.py         + fetch_presence_resolved, row_updated_at,
                               presence_row_age_seconds
context.py                   + _resolve_presence_and_identity_ask, ahead of
                               the percept gate, on every return path
identity_ask_cooldown.py     key: (reason, stream) instead of stream
schemas/situation.py         + presence_identity_ask (the decision)
```

## Files changed

- `services/orion-vision-window/app/presence.py`: add `identity_confirmed`.
  Consumers cannot ask "do I have a confirmed read" from a field that only says
  "I explicitly don't".
- `config/vision_frame_router.yaml`: opt `carbon` in. Still per-stream opt-in,
  not a `defaults` change — the design doc's §9B concern was a permissive
  global default reaching every camera, and that reasoning survives intact.
  `min_seconds_between_dispatch: 60` vs cam0's 30: a webcam on one seated
  person re-matches cheaply; a room camera can gain a genuinely new person.
- `orion/situational/perception_reader.py`: `fetch_presence_resolved` (one
  query, prefers fresh+present, then fresh+recent, then configured order);
  rows carry `row_updated_at`; `presence_row_age_seconds`.
- `orion/situational/context.py`: `_resolve_presence_and_identity_ask` runs
  before the percept fetch and its result rides every return path. Presence
  *prose* stays gated on a valid percept; the *ask* deliberately does not.
- `orion/situational/identity_ask_cooldown.py`: reason-scoped keys.
- `orion/schemas/situation.py`: `presence_identity_ask`. Added rather than
  widening `presence_identity_uncertain`, which is registered in
  `orion/schemas/registry.py` and read as a structured debug signal — changing
  its meaning in place would silently alter what an existing reader sees.
- `services/orion-cortex-exec/app/settings.py`, `.env_example`: three keys.

## Schema / bus / API changes

- **Added:** `PerceptionContextV1.presence_identity_ask`
  (`"unmatched_face" | "no_visual_confirmation" | None`). Additive on a model
  with `extra="forbid"`, so old payloads still validate.
- **Added:** `identity_confirmed` in the `substrate_embodied_presence` JSONB
  blob. Consumers using `.get()` are unaffected.
- **Behavior changed:** the identity-ask cooldown Redis key gains a reason
  segment. Retires in-flight claims under the old key once — at most one extra
  ask per camera on the first turn after deploy.
- **Not changed:** `presence_identity_uncertain` keeps its exact prior meaning
  and cooldown.

## Env/config changes

- Added keys (`services/orion-cortex-exec`):
  `ORION_SITUATION_PERCEPTION_STREAM_IDS=carbon,cam0`,
  `ORION_SITUATION_IDENTITY_ASK_UNCONFIRMED_COOLDOWN_SECONDS=21600`,
  `ORION_SITUATION_IDENTITY_ASK_MAX_PRESENCE_AGE_SECONDS=120`
- `.env_example` updated: yes
- local `.env` synced: **by hand.** `scripts/sync_local_env_from_example.py`
  would have missed these keys for **two independent reasons**, either one
  sufficient, and would have printed a success-shaped message either way:

  1. It logs `reading live .env from primary checkout` and reads
     `.env_example` from there, so keys added in a worktree are invisible to
     it. Observed: it reported only unrelated "Diverged" entries and added
     nothing.
  2. Even run from the primary checkout with the keys present, a default
     invocation skips them. `orion-cortex-exec` *is* in `DEFAULT_SERVICES`,
     but `should_sync_key` allowlists by prefix and `ORION_SITUATION_` is not
     in `SYNC_PREFIXES`. Verified as a positive control:

     ```
     ORION_SITUATION_PERCEPTION_STREAM_IDS                  -> False
     ORION_SITUATION_IDENTITY_ASK_UNCONFIRMED_COOLDOWN_SECONDS -> False
     ORION_SITUATION_IDENTITY_ASK_MAX_PRESENCE_AGE_SECONDS   -> False
     ```

     `--all-keys` returns True for all three. This independently corroborates
     an existing agent-board finding (`191c0c08`, 2026-08-29) that the
     CLAUDE.md-prescribed bare command silently checks nothing for most
     services and keys.

  Keys written directly into
  `/mnt/scripts/Orion-Sapienform/services/orion-cortex-exec/.env` and verified
  present at lines 403-405, then confirmed readable by the running container's
  env after restart is required (see Restart section).
- No compose change needed: the three replicas `extends` the base service and
  inherit its `env_file`, which is how the existing
  `ORION_SITUATION_PERCEPTION_STREAM_ID` already reaches them (confirmed by
  `docker exec ... env`).

### Cooldown ratio

`no_visual_confirmation` is a *condition* that holds for hours, not an event.
At the 20-minute mismatch cadence it would produce roughly nine questions
across one lid-closed evening. 6h makes it about once per sitting. A test
locks the ratio rather than just the values.

## Tests run

```text
# the two suites this touches, from the worktree
pytest services/orion-cortex-exec/tests orion/situational -q --continue-on-collection-errors
  92 failed, 840 passed, 14 errors

# same command, unmodified main (baseline)
  92 failed, 810 passed, 14 errors
  => +30 passing, zero regressions. The 92 failures and 14 collection errors
     are pre-existing (e.g. "ValueError: Verb already registered: legacy.plan",
     reproduced on main; test_situation_prompt_integration's jinja
     'metadata' is undefined likewise reproduces standalone on main).

cd services/orion-vision-window && pytest tests -q     -> 92 passed
cd services/orion-vision-frame-router && pytest tests/test_decide_identity.py -q -> 10 passed
pytest services/orion-cortex-exec/tests/test_presence_stream_resolution.py -q -> 16 passed
python scripts/check_env_key_single_source.py -> OK
git diff --check -> clean
```

### Mutation tests (the green had to be earned)

| Mutation | Result |
|---|---|
| `reason = "no_visual_confirmation"` -> `None` (pre-patch behavior) | 6 failed |
| `fresh = age <= max` -> `fresh = True` (ignore row age) | 1 failed (`test_camera_that_went_dark_mid_session_asks`) |
| delete the `carbon:` block from the shipped yaml | 2 failed; `cam0` still passed |

The third matters most: every pre-existing router test builds its own synthetic
policy, which is exactly how this feature stayed "fully tested" while never
running on Juniper's webcam. The new tests read
`config/vision_frame_router.yaml` itself.

## Evals run

```text
No eval harness exists for orion-vision-window, orion-vision-frame-router, or
the orion/situational package (no evals/ directory in any of the three).
Not created here. Follow-up below.
```

## Docker/build/smoke checks

```text
# Live resolver against production Postgres, from the worktree:
resolved stream_id : carbon        <-- was cam0, the empty room
state              : present
subject            : unknown
identity_confirmed : None          <-- field not yet in the deployed writer
row age (sec)      : 3.11533
=> ask reason      : no_visual_confirmation
```

`identity_confirmed: None` is expected and correct pre-deploy: `carbon` has a
person present but `identity_face` has never run there (blocker 2), so Orion
has no confirmation and asks. Once the router redeploy lets `identity_face`
run on `carbon` and it matches, this goes quiet on its own.

Not yet exercised live: the rebuilt `presence.py` writing `identity_confirmed`,
and an actual `identity_face` dispatch on `carbon`. Both need the deploy below.

## Restart required

The router mounts `config/` and `app/` read-only from the checkout, so it needs
a restart, not a rebuild. `orion-vision-window` and `orion-cortex-exec` mount
no code and must be rebuilt. All paths are the **primary** checkout after merge.

```bash
# 1. after merge
git switch main && git pull --ff-only

# 2. frame-router -- config + code are volume-mounted, restart only
docker restart orion-orion-athena-vision-frame-router

# 3. vision-window -- presence.py is baked into the image
bash scripts/safe_docker_build.sh orion-vision-window up -d --build

# 4. cortex-exec (all four replicas: base, -chat, -spark, -background)
bash scripts/safe_docker_build.sh orion-cortex-exec up -d --build

# 5. verify
docker exec orion-athena-cortex-exec-chat env | grep ORION_SITUATION_PERCEPTION_STREAM_IDS
docker logs orion-orion-athena-vision-frame-router --since 10m | grep identity_dispatch
# expect: at least one line with stream_id=carbon
```

## Risks / concerns

- **Severity: medium.** Ask frequency is now driven by a condition rather than
  an event. If Juniper works with the lid shut all day, she gets the question
  about once per 6h per camera. The cooldown is the only brake, and 21600s is
  a judgement call, not a measurement. It is a single env key to retune.
- **Severity: low.** A total vision outage (no readable presence row) now
  reads as `no_visual_confirmation` and asks. Arguably correct — Orion
  genuinely cannot see — but it means an infra failure becomes conversational.
  Bounded to ~4 asks/day by the same cooldown.
- **Severity: low.** `identity_face` on `carbon` is new GPU load: one extra
  dispatch per 60s per triggered frame window. `cam0` has carried the same
  work at 30s since 2026-08-26.
- **Follow-up, not fixed here:** `ENDOGENOUS_OUTREACH_PERCEPTION_STREAM_ID=cam0`
  in orion-hub has the identical wrong-camera bug for the outreach prompt. Left
  alone deliberately — different feature, different blast radius.
- **Follow-up:** no eval harness exists for any of the three touched services.
