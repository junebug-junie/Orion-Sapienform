# PR: the doorbell has been deaf since 10-28 -- fix it so its faces reach person learning again

## Summary

- The question was why the doorbell camera's close-up faces never reach Orion's person albums. The answer is not a missing pipeline. The doorbell already gets crop vectors, face vectors and clusters, and its crops already sit in the albums (374 `+1` exemplars). The doorbell has simply stopped hearing the Ring since 2026-10-28 21:15 UTC.
- Cause: Google's push service (FCM) began to send two message headers without base64 padding. The push library (`firebase-messaging` 0.4.5) decodes them strictly and crashes, and it crashes before it acknowledges the message. So the same message comes back on every reconnect and kills every new listener within a second. The listener died about 1,440 times on 11-03. Health still read `ok`.
- Fix (`services/orion-vision-retina/app/ring_source.py`): pad both headers before decoding. If one message still cannot be handled, log it, count it and acknowledge it. Do not let it kill the listener. Same fix as another Ring/FCM project ([fermax-blue-hass v0.16.6](https://github.com/bvis/fermax-blue-hass/releases/tag/v0.16.6), [v0.16.7](https://github.com/bvis/fermax-blue-hass/releases/tag/v0.16.7)).
- Health now goes `degraded` with `ring_deaf: true` when 3 supervisor checks in a row find a dead listener. A deaf doorbell no longer reads as a quiet one.
- No flag: this is a bug fix with no off switch. No env keys changed.

## Outcome moved

- Before: 0 Ring events, 0 doorbell crops outside self-looks for 6 days. Health `ok`.
- Expected after deploy (based on the last healthy week, 10-21..10-28): about 48 motion events and 1 ring a day. About 600 doorbell person crops, about 44 face vectors and about 19 face clusters a week. About 17 face exemplars a week land in clusters that Juniper names (at the current naming rate). UNVERIFIED until deployed.
- This does **not** make "probably X" appear on its own. See "Risks / concerns". The face bar cannot arm until it has about 320 impostor clusters with faces. The doorbell adds about 19 a week, and the 30-day street window caps the pool near 150.

## Current architecture

Step-1 investigation, read-only, live DB and logs on 2026-11-03.

- **Router policy** (`config/vision_frame_router.yaml` `streams.doorbell`): both tiers already send `want_crop_embeddings: true` and `want_crop_faces: true`. `every_n_frames: 1`.
- **Zones** (`config/vision_zones.yaml`): doorbell has one whole-frame `porch` zone with `embed: true`, so the fail-closed zone rule does not block it.
- **vision-host** (`app/crop_embeddings.py` `attach_crop_embeddings`/`_attach_faces`): this code does not special-case any stream. Faces come from MTCNN, need at least 40 px, and are embedded by InceptionResnetV1.
- **sql-writer**: the crop writer stores doorbell crops, and the individuals reducer clusters them **per stream** (doorbell individuals are separate from walkway individuals). The person model (`app/vision_person_model.py`) reads crops from every stream. Doorbell clusters are named through the same asks.
- **Cross-camera handoff** (`app/vision_handoff.py`): it links door and walkway passages for arrival timing. By design it "never writes a label or merges individuals". I did not change this.

Live numbers (all time unless noted):

| | doorbell | walkway |
|---|---|---|
| person crops | 936 | 12,476 |
| person crops with a face vector | 68 (7 %) | 354 (3 %) |
| person clusters / named | 201 / 42 | 1,292 / 78 |
| active `+1` album exemplars | 374 (32 with face) | 1,233 |
| unnamed person clusters with a face, last 30 d | 23 | 30 |

- Doorbell face hit rate by person box height: below 400 px, 4 of 681 crops get a face. At 400 px and above, 66 of 255 (26 %). So faces are big enough when people come up to the door.
- Doorbell `vision_events`: about 48 `doorbell_motion` a day from 10-21 to 10-28, then 0. The last push was at 2026-10-28 21:15:36 UTC. Crops after that come only from self-looks (2 to 8 a day).
- Retina log (journald, from 11-03 03:07): 1,114 `Incorrect padding, shutting down FcmPushClient`. Docker log since the 09:47 restart: 1,440. A restart does not fix it.
- Health at 21:59 UTC: `status: ok`, `ring_events_seen: 0`, `last_error: null`.
- GPU: no new embedding work. The fix restores the load the doorbell already had before 10-28. athena GPU0 (P4) 251/7680 MiB, GPU1 (T10) 15341/16384 MiB at check time.

## Architecture touched

`orion-vision-retina` only (the Ring source + its health). No contracts.

## Files changed

- `services/orion-vision-retina/app/ring_source.py`: adds `ensure_fcm_push_tolerance()`, which pads the headers and skips plus acknowledges a bad message. It runs before every listener start. Also adds `consecutive_listener_deaths`, `listener_restarts`, and the `deaf`, `push_messages_skipped` and `push_last_skip_error` values.
- `services/orion-vision-retina/app/health.py`: new metrics, `ring_deaf` degrades status.
- `services/orion-vision-retina/app/main.py`: copies the new source counters into metrics.
- `services/orion-vision-retina/tests/test_ring_source.py`: the regression tests use a real FcmPushClient, real `DataMessageStanza` objects and real aesgcm encryption with unpadded headers. They cover: the unpatched library raises; the patched client decrypts and delivers; padded headers still work; an undecryptable message is skipped and acknowledged; the supervisor goes deaf after 3 deaths and clears when a listener survives; health degrades.
- `services/orion-vision-retina/README.md`: the incident, the new health fields, and the live check.

## Schema / bus / API changes

- Added: health `details` keys `ring_consecutive_listener_deaths`, `ring_listener_restarts`, `ring_deaf`, `ring_push_messages_skipped`, `ring_push_last_skip_error`.
- Removed / Renamed: none.
- Behavior changed: health `status` becomes `degraded` when the doorbell is deaf.
- Compatibility notes: additive health details only.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest services/orion-vision-retina/tests -q            -> 161 passed (py3.12 venv from requirements.txt)
docker build (retina Dockerfile, py3.10, pinned firebase-messaging 0.4.5) + the 8 new tests inside it
                                                         -> 7 passed; 1 failed only on a test-path
                                                            assumption (ROOT.parents[1] outside the repo
                                                            layout), passes in the repo layout
```

## Evals run

```text
pytest services/orion-vision-retina/evals -q   -> no tests collected (frame_smear.py is a script, unrelated)
```

There is no eval for "is the doorbell hearing Ring". The health `ring_deaf` gate plus the one-week check below is the measurement.

## Docker/build/smoke checks

```text
docker build -f services/orion-vision-retina/Dockerfile .   -> built (throwaway tag, removed)
```

The live smoke is UNVERIFIED. Not deployed (no deploy from agent sessions).

## Review findings fixed

The review ran in a subagent. It found no must-fix issues. It confirmed three things: `ring_doorbell` creates the same `FcmPushClient` class that is patched, a swallowed message still gets its selective ack, and callback bugs stay in the library's own handler.

- Finding: if every message fails to decode, the listener stays up and drops all of them, and health still reads `ok`. This is the same silent deafness, one layer down.
  - Fix: `_FcmSkips.consecutive` counts failures in a row and resets on the first good message. `deaf` is now also true at `DEAF_AFTER_SKIPPED_MESSAGES` (3).
  - Evidence: `test_every_message_failing_reads_deaf_and_one_good_message_clears_it`.
- Finding: the wrappers depend on the library's exact argument list.
  - Fix: the wrappers pass `*args/**kwargs` through. If the internals are missing, the code logs a warning and listens unpatched.
  - Evidence: the existing tests pass.
- Finding: during a long live view, the supervisor counted the same dead listener as several deaths and several restarts.
  - Fix: a dead listener is counted once (by object reference, not `id()`, because ids are reused after garbage collection). A restart is counted only when `_reconnect_when_idle` really reconnects.
  - Evidence: `test_a_dead_listener_during_a_live_view_counts_once_and_is_not_a_restart`.
- Finding: test hygiene. The supervisor test depended on timing, one test used `/tmp`, and the fixture could save an already-patched class.
  - Fix: the supervisor test now polls with `wait_for`, `/tmp` is replaced by `tmp_path`, and the fixture asserts the class is unpatched.
  - Evidence: 161 passed.
- Not changed (nit): a reconnect that keeps failing does not move the death counter. In that case `last_error` is set and `_source_ok()` already reports a failure, so it does not stay silent.

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-retina up -d --build orion-vision-retina-doorbell
```

Run after merge, from the clean primary `main` checkout. Live check within an hour of the next person at the door:

```bash
curl -s localhost:8029/health | python3 -c "import sys,json;d=json.load(sys.stdin)['details'];print(d['ring_events_seen'],d['ring_deaf'],d['ring_listener_restarts'],d['ring_push_messages_skipped'])"
journalctl CONTAINER_NAME=orion-vision-retina-doorbell -o cat --since "-1 hour" | grep -c "Incorrect padding"   # want 0
```

## The check to run in a week (2026-11-10)

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select count(*), count(face_embedding), count(distinct individual_id) filter (where face_embedding is not null) from vision_crop_observation where stream_id='doorbell' and label='person' and observed_at > now()-interval '7 days'"
# expect roughly 600 | 44 | 19 (last healthy week: 695 | 50 | 22 over 8 days)
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select person_label, n_impostor, accept_at from vision_person_bar where channel='face' order by 1"
# expect n_impostor up from ~70 toward ~90; accept_at still empty (see concern 1)
PYTHONWARNINGS=ignore .venv/bin/python services/orion-sql-writer/evals/run_person_recognition_eval.py --live postgresql://postgres:postgres@localhost:55432/conjourney
# read-only (no --write). Compare face AUC (0.614 on 11-03) and correct/wrong accepts (0/0).
```

## Risks / concerns

- Severity: high (for the goal, not for this patch).
  - Concern: "probably X" still cannot appear through the face channel. `conformal_accept_at` returns a bar only when there are at least `1/a - 1` impostor scores, where `a = 0.05 / people`. With 16 album labels that is 319 face impostor clusters. Today there are about 70. The doorbell adds about 19 a week, and `candidate_days = 30` keeps the street pool near 150 at best. Removing the 4 non-person labels (skeleton, Mail carrier, UPS delivery, "Addy and Esme") lowers the need to 239. That is still out of reach.
  - Mitigation: needs a proposal-mode change to the person model, so it is not done here. The options are: a longer street window for the face channel only; counting only real people in the Bonferroni split (the spec already flags this); or a doorbell-to-walkway face link that names doorbell clusters through the handoff (needs a design: handoff today never writes labels). Recommended next step: count only real people, and use a 90-day face street window. Then re-run the eval and check that wrong accepts stay at 0.
- Severity: medium.
  - Concern: this patches a third-party library at runtime. A future `firebase-messaging` upgrade could rename `_decrypt_raw_data` or `_handle_data_message`.
  - Mitigation: the version is pinned in requirements. The tests use the real library, so an upgrade that breaks the patch fails CI. If the attribute is missing, the patch fails at listener start inside `_connect_once`'s try, so it shows in `last_error` and is not silent.
- Severity: low.
  - Concern: a message that still cannot be handled is dropped (acknowledged). That one Ring event is lost.
  - Mitigation: the alternative is losing every event. Skips are counted in health.

## PR link

(pending)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
