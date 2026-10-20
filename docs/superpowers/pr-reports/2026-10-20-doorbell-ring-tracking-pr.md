## Summary

- A doorbell press is no longer lost. Before, a ring that came while the
  motion-triggered live view was open (or in its 30 s cooldown) was dropped
  with "ignored: burst running or in cooldown". That is the usual case: you
  walk up (motion), then press (ding). Now every Ring push is counted, logged
  with a correlation id, and recorded. A ring extends the open live view, or
  opens a new one even in the cooldown.
- Every Ring push becomes one `vision_events` row (`doorbell_ring`, salience
  1.0, or `doorbell_motion`, salience 0.2, `stream_id=doorbell`). It uses the
  same rail as the window's `camera_caption` rows (`orion:vision:events` ->
  scribe -> sql-writer).
- New contract `StreetEventV1` on new channel `orion:vision:street:events`.
  The retina publishes one per ring (salience 1.0). The consumer
  (orion-substrate-runtime) comes in a later PR.
- Active looking: when Redis key `orion:vision:look:doorbell` appears (the
  handoff reducer will set it in a later PR), the doorbell opens a live view on
  its own. It is capped at 8 per UTC day and spaced 300 s apart.
- `GET /health` now returns the SystemHealthV1 payload, with `dings_total`,
  `motions_total`, `last_ding_ts`, `self_looks_today` and `self_looks_total`.

## Outcome moved

A ring at the door always reaches Orion's record and the attention rail.
Before, a ring during a motion burst left no trace except an "ignored" log line.

## Current architecture

`ring_source.RingEventFrameSource.on_event` filtered pushes by kind. It opened
one 15 s WebRTC live view per push, but only when no burst was running and the
cooldown had passed. Anything else only bumped `events_ignored`. No Ring push
was written anywhere except the container log.

## Ding investigation (ring_doorbell 0.9.14, read in the live image)

- The legacy (`gcmData`) push path has a library bug.
  `listen/eventlistener.py:228` compares `action.lower == PUSH_ACTION_DING.lower()`.
  That compares the method object, never its result, so a legacy ding never
  matches. Its `kind` becomes the raw action string
  `com.ring.push.HANDLE_NEW_DING`. Legacy motion uses `.lower()` correctly and
  maps to `motion`.
- The new-format path maps the category through `PUSH_NOTIFICATION_KINDS`
  (`com.ring.pn.live-event.ding` -> `ding`). An unmapped category becomes
  `"Unknown"`.
- `ring_source.KIND_ALIASES` now maps every ding and motion spelling the
  library can produce. A kind that is still unknown is logged with its raw
  spelling.
- Live logs: the container restarted 2026-10-20 03:55 UTC, so `docker logs`
  only covers the time since then. That window has 20 `ring motion` lines and
  zero ding lines, including zero "not in event_kinds; ignored" lines. The old
  code would have logged a mis-spelled legacy ding with exactly that line. So
  in this window, no ding push reached the listener under any spelling. One
  of these happened, and the data cannot tell which: nobody rang, or Ring
  does not deliver dings to this FCM client. UNVERIFIED until a real press.
  The new `ring ding recorded ... raw_kind=` line and `dings_total` will show it.

## Active looking: can Ring open a live view with no event?

The burst uses `doorbot.generate_async_webrtc_stream(sdp_offer, session_id, cb)`.
That call takes only the device and an SDP offer. It needs no ding id or event
context, which is the same path the Ring app's "Live View" button uses. So the
code path works without an event. Ring's servers accepting a live view with no
preceding event is UNVERIFIED live. I did not open one, because each one wakes
the battery camera. The feature is on (`RETINA_RING_SELF_LOOK_ENABLED=true`),
but nothing sets the key until the handoff reducer PR, so no self-look can
happen yet.

## Architecture touched

- orion-vision-retina: ring source, a new event builder module, service
  wiring, health.
- Contracts: `orion/schemas/vision.py`, `orion/schemas/registry.py` (both
  `_REGISTRY` and `SCHEMA_REGISTRY`, checked with `resolve()`),
  `orion/bus/channels.yaml`.
- No change to `VisionFramePointerPayload`, the scribe or the sql-writer. The
  scribe forwards every event_type, and `vision_events.event_type` is free
  text with no allow-list (checked in `orion-vision-scribe/app/main.py
  _write_to_sinks` and `orion-sql-writer/app/models/vision_event.py`).

## Files changed

- `services/orion-vision-retina/app/ring_source.py`: records every push,
  `KIND_ALIASES`, `RingPush`, burst extension (`MAX_BURST_FACTOR=4`),
  `self_look()`.
- `services/orion-vision-retina/app/ring_events.py` (new): builds the push
  vision event and street event, plus their envelopes.
- `services/orion-vision-retina/app/main.py`: `event_sink` -> publish,
  `_self_look_loop`, `GET /health`, `ring_corr=` on frame log lines.
- `services/orion-vision-retina/app/health.py`: new counters.
- `services/orion-vision-retina/app/settings.py`, `sources.py`: new settings
  and pass-through.
- `services/orion-vision-retina/.env_example`, `docker-compose.yml`,
  `README.md`: env parity and documentation.
- `services/orion-vision-retina/tests/test_ring_source.py`: regression and
  new tests.
- `orion/schemas/vision.py`, `orion/schemas/registry.py`,
  `orion/bus/channels.yaml`: the StreetEventV1 contract.
- `tests/test_street_event_bus_catalog.py` (new): contract test.
- `.github/workflows/orion-vision-retina-tests.yml`: runs on contract paths
  and runs the contract test.

## Schema / bus / API changes

- Added: `StreetEventV1` (`vision.street.event.v1`). Fields: `schema_version`,
  `event_id`, `kind` (doorbell_ring | handoff_met | handoff_late |
  handoff_missed | overlap_pair | known_person_arrived | approach_door),
  `stream_ids` (min 1), `occurred_at`, `subject_kind` (person | dog | vehicle
  | unknown, default unknown), `individual_id?`, `person_guess?`, `salience`
  (0..1), `p_value?` (0..1), `evidence_refs`, `correlation_id`.
  `extra="forbid"`.
- Added: channel `orion:vision:street:events`. Producer orion-vision-retina.
  Consumer orion-substrate-runtime (pending PR; nothing subscribes yet).
- Behavior changed: orion-vision-retina is now a producer on
  `orion:vision:events`, with new `vision_events.event_type` values
  `doorbell_ring` and `doorbell_motion`.
- Added: `GET /health` on the retina.
- Compatibility notes: additive only. The vision event `event_id` is a uuid5
  of the Ring push id, so a re-delivered push cannot write a second row. Burst
  frames keep per-frame envelope correlation ids, because the frame router
  keys pending tasks by them. The push's correlation id is in the
  vision_events row and in the frame log lines instead.

## Env/config changes

- Added keys: `DOORBELL_RING_SELF_LOOK_ENABLED=true`,
  `DOORBELL_RING_SELF_LOOK_MAX_PER_DAY=8`,
  `DOORBELL_RING_SELF_LOOK_MIN_SPACING_SECONDS=300` (compose maps them to
  `RETINA_RING_SELF_LOOK_*`). The settings-only defaults are
  `RETINA_RING_SELF_LOOK_KEY=orion:vision:look:doorbell`,
  `RETINA_RING_SELF_LOOK_POLL_SECONDS=2`,
  `CHANNEL_VISION_EVENTS_PUB`, and `CHANNEL_STREET_EVENTS_PUB`.
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes.
  `--all-keys orion-vision-retina` added the three keys to the primary
  checkout's `services/orion-vision-retina/.env`. A plain run only covers its
  fixed feature-key list and reported no changes.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-vision-retina/tests (venv from requirements.txt, as CI): 148 passed
tests/test_street_event_bus_catalog.py: 9 passed
tests/test_agent_trace_schema_registry.py + catalog/channel/registry tests: pass,
  except 5 failures that fail identically on main (exec_result wildcards,
  field glossary count, single-consumer annotation, autonomy goals registry)
services/orion-vision-scribe/tests: 12 passed
Regression: the new ding tests run against origin/main's ring_source.py -> 7 failed
scripts/check_env_template_parity.py: PASS
bus publish validation (_validate_payload, enforce_catalog=True) on both new envelopes: ok
git diff --check: clean
```

scripts/check_schema_registry.py, scripts/check_bus_channels.py and
`make agent-check` do not exist in this repo. The registry and catalog tests
above cover them.

## Evals run

```text
None. The retina's evals/ covers RTSP frame smear, not Ring. The Ring path
has no eval harness. Follow-up: once real presses exist, compare
dings_total against Ring app history to measure delivered vs pressed.
```

## Docker/build/smoke checks

```text
No build needed: the doorbell container bind-mounts app/ and orion/ from the
primary checkout, so a restart after merge picks the code up.
Live: not deployed (deploys run from primary main after merge). UNVERIFIED live.
```

## Review findings fixed

A review subagent (orion-repo-agent) checked `origin/main...HEAD`.

- Finding (must): a ding that came after the frame loop ended, while
  `close_webrtc_stream` was still awaiting, or one that "extended" a view that
  then failed to connect, was marked `extended`, but no live view ever covered it.
  - Fix: a ding now extends only while the frame loop runs (`_frames_open`).
    Otherwise it is `queued`. A connecting view takes it over when frames
    start. A closing or failed view reopens for it in `finally`.
  - Evidence: `test_ding_while_closing_reopens_a_live_view` and
    `test_ding_while_connecting_is_queued_and_served_by_frames_start`.
- Finding (should): back-to-back dings skip the cooldown, so they could keep
  the battery camera streaming.
  - Fix: at most 3 ding reopens that skip the cooldown per 10 min
    (`ding_capped`, `ring_ding_reopens_refused` on health). Every press is
    still recorded.
  - Evidence: `test_dings_in_cooldown_are_capped`.
- Finding (should): the frame-loop `wait_for` timeout could end an extended
  burst during a frame gap.
  - Fix: on a timeout, the loop re-reads the extension before it ends.
  - Evidence: `test_extension_survives_a_frame_gap`.
- Finding (should): a push with no Ring id fell back to a random id, so
  dedupe could not work.
  - Fix: fall back to kind plus Ring's created_at second.
  - Evidence: `test_ring_event_id_is_stable_per_push`.
- Finding (should): the extension test stubbed `_burst`.
  - Fix: real `_burst` tests now cover connecting, closing and frame-gap cases.
- Finding (nit): a self-look key value went into logs and health unbounded.
  - Fix: it is truncated to 80 chars.
  - Evidence: `test_self_look_value_is_bounded`.
- Finding (nit): StreetEventV1 had no consumer-first note, and settings-only
  keys were missing from `.env_example`.
  - Fix: added a docstring note and commented lines in `.env_example`.

## Restart required

After merge, from the clean primary `main` checkout:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-vision-retina up -d --force-recreate orion-vision-retina-doorbell
```

Check: `curl -s 127.0.0.1:8029/health | jq '.details | {dings_total, motions_total, self_looks_today}'`

## Risks / concerns

- Severity: medium. Concern: a ring always opens or extends a live view
  (ignores the cooldown), and self-looks add up to 8 views a day. Both cost
  battery. Mitigation: extension is capped at 4 bursts, self-look has a cap
  and spacing, and `DOORBELL_RING_SELF_LOOK_ENABLED=false` turns self-look off.
- Severity: medium. Concern: no ding has been seen live, so delivery of
  dings to this push client is unproven. Mitigation: `dings_total` and the
  `raw_kind=` log line make the first real press visible.
- Severity: low. Concern: `StreetEventV1` is `extra="forbid"`, so adding a
  field later needs consumers updated first. Mitigation: there is no consumer
  yet, so the shape can still change freely.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2816

🤖 Generated with [Claude Code](https://claude.com/claude-code)
