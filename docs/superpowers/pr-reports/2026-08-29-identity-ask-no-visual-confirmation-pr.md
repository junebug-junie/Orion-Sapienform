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
  92 failed, 849 passed, 14 errors     (after review fixes)

# same command, unmodified main (baseline)
  92 failed, 810 passed, 14 errors
  => +39 passing, zero regressions. The 92 failures and 14 collection errors
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
| remove the `identity_unread` branch (review fix 1) | 1 failed, the right one |
| ignore `read_ok` (review fix 2) | 1 failed, the right one |

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

## Review findings fixed

Code review (high effort) returned 12 findings; all 12 addressed. The two
MAJOR ones were real defects that would have shipped a lying prompt.

- **Finding 1 (MAJOR): the catch-all asserted a physical fact that is often
  false.** `no_visual_confirmation` said *"your camera is closed, off, or
  showing an empty frame"*, but a live camera with a person in frame and no
  identity reading lands there routinely — `identity_confidence_from_artifact`
  returns `None` when no face is in the sampled frame and deliberately returns
  `None` for an unenrolled gallery. The brief would then carry *"Someone has
  been in view for 15 minutes"* and *"your camera is closed"* simultaneously.
  - Fix: a third reason, `identity_unread` ("someone is in view, you just
    haven't verified who"), and **no wording anywhere asserts a physical
    camera state** — Orion now says only what it has read, not why.
  - Evidence: `test_person_in_view_without_an_identity_read_does_not_claim_a_dark_camera`
    asserts the scene fragment and the caution coexist without contradiction;
    `test_no_wording_anywhere_asserts_why_orion_cannot_see` bans the phrases.
    Mutation: removing the `identity_unread` branch fails exactly that test.

- **Finding 2 (MAJOR): a database outage produced a false "my camera is
  closed" claim.** `fetch_presence_resolved` swallowed its own DB errors and an
  unset DSN, returning `(None, None)` — indistinguishable from "answered, and
  nobody is there". Only a *raise* reached the `except`, so a Postgres blip
  fell through to the catch-all.
  - Fix: it now returns a `PresenceResolution(stream_id, presence, read_ok)`
    NamedTuple; `read_ok=False` yields no ask at all. An infrastructure fault
    must never be laundered into a claim about the physical world.
  - Evidence: `test_a_failed_presence_read_never_becomes_a_claim_about_the_room`
    (also asserts no cooldown is burned). Mutation: ignoring `read_ok` fails
    exactly that test.

- **Finding 3 (MEDIUM): cross-service deploy coupling.** `identity_confirmed`
  is written only by the new vision-window build, so until it is redeployed
  `confirmed` was permanently `False` and Orion would doubt Juniper while
  looking straight at her.
  - Fix: a named `subject` also counts as confirmation — the **currently
    deployed** build already writes a real name there on a probable/possible
    match. The deploy order no longer matters.
  - Evidence: `test_a_named_subject_counts_as_confirmation_on_a_pre_deploy_row`.

- **Finding 4 (MEDIUM): the 120s freshness bound rested on an unvalidated
  write cadence.** Reviewer sampled the live table and found it frozen. I
  reproduced and went further: **both `carbon` and `cam0` sat frozen at the
  same instant for 16+ minutes** while `orion-athena-vision-window` was
  `Up (healthy)`, `vision-edge` was publishing (31,500 frames) and the router
  dispatched 283 frames in 15 minutes — with **zero presence log lines and
  zero errors** in 60 minutes of window logs. `cam0` is RTSP and cannot
  "close", so this is a silently stalled writer, not dark cameras.
  - This is a **pre-existing live bug, not introduced here**, and it is
    decisive for Finding 1: a patch that concluded "the camera is off" from
    that state would have been confidently wrong. Logged on the agent board
    (`2998eba2`). Not fixed in this PR — different service, different cause.
  - The 120s bound is retained but is now purely a *"do not describe this as
    current"* gate, no longer a claim about why.

- **Finding 5 (MEDIUM/LOW): a global condition was rate-limited per camera.**
  The resolved stream flips between `carbon` and `cam0`, and each flip granted
  a fresh 6h slot — two cameras bought two asks, a third would raise the
  ceiling again.
  - Fix: `no_visual_confirmation` keys on a constant `_global` scope; the other
    two reasons stay genuinely per-camera.
  - Evidence: `test_the_global_reason_is_not_rate_limited_per_camera`.

- **Finding 6 (MEDIUM): the prose path consumed an unbounded-age row.** The
  tier-3 fallback returns the last known row at any age; it was fed straight
  into present-tense prose. The live frozen row would have rendered "Someone
  has been in view for about 27 minutes."
  - Fix: prose is gated on `row_fresh`; the identity path deliberately is not.
  - Evidence: `test_a_frozen_row_never_renders_as_present_tense_prose`.

- **Finding 7 (LOW/MEDIUM): the one-shot ask was baked into a 300s cached
  brief**, so a single Redis claim was replayed to every turn within the TTL
  and only the model's own compliance prevented repetition.
  - Fix: the cached copy is stored with the ask cleared — this turn asks,
    later cache hits do not.

- **Finding 8 (LOW): a comment contradicted the shipped default.** It claimed
  unsetting the multi-stream key restores cam0-only behavior; cortex-exec's
  `Settings` default is `"carbon,cam0"`, so it does not. Comment corrected to
  say so explicitly.

- **Finding 9 (LOW): reported a stream it did not read.** On the
  nothing-readable path `stream_id` fell back to `cfg.perception_stream_id` —
  reproducing the exact dishonesty the adjacent comment cites. Now `None`.
  - Evidence: `test_nothing_readable_reports_no_stream_id_rather_than_a_guess`.

- **Finding 10 (LOW): a router test could pass vacuously.** If no frame was
  ever sampled, `decide_identity` returned `False` because `should_dispatch`
  was `False`, not because `kitchen` was opted out. Added the
  `assert dispatched is not None` guard its sibling already had.

- **Finding 11 (LOW): `dict()` sat outside the fail-open try** in
  `fetch_presence`, so a driver returning a JSON string would raise out of a
  function documented never to. Moved inside.

- **Finding 12 (LOW): a `datetime` was injected into a cross-service dict.**
  `row_updated_at` flowed into orion-hub's `OutreachContext.embodied_presence`,
  inert today but a trap for the first caller to serialize it. `fetch_presence`
  now returns its original shape; only the resolved path carries the timestamp.


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
