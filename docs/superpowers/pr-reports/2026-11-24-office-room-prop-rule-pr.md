## Summary

- The office camera (cam0) now has the "a person box that never moves is an object" rule, as a **room** camera.
- A room camera has its own hold: 2 h. The walkway keeps 30 min.
- On a room camera, a gap of more than 5 min always ends a still run, even after it was marked. An outdoor prop stays marked across gaps, so the patio skeleton survives a night of missed detections.
- On a room camera, a box with a face starts a new run.
- Office visits now ignore the person record's own "object" flag and use only each sighting's flag. A stale flag can no longer drop Juniper's visit after she sat still and left.
- The env sync script now syncs `VISION_STATIONARY_*` keys. New keys with that prefix were never synced before.

## Outcome moved

- Background: a pile of boxes read as a person for 4–5 h on 2026-11-15..16, and Orion sent check-ins with nobody in the room. The 0.55 cut-off (#3616) stopped most of it. One pile box still scored 0.554 on 11-22.
- Now, a pile that scores above the cut-off becomes an object after 2 h, in the eval replay.
- Real people never do. The longest still run of a real person is now 13.7 min, because faces restart runs.

## Current architecture

The stationary rule existed only for walkway and doorbell (`VISION_STATIONARY_STREAMS`, `WINDOW_STATIONARY_STREAMS`). It used one hold for all of them, and a marked prop was sticky (a 24 h forget timer in the window, and gaps never broke a marked run).

The office was left out on purpose. A sticky rule would hide Juniper's seat on her later visits (review of #3616).

## Architecture touched

- `orion/vision/stationary.py`: `StreamRule(min_sec, room)`, `stream_rules()`, and `advance(restart=)`.
- orion-vision-window: `FixtureTracker` and `CaptionGate` take `room_streams`/`room_min_sec` and a has-face flag per box. `main._person_boxes` passes faces.
- orion-sql-writer: `apply_batch` uses a rule per stream, and `advance_stationary(room=, face=)`. `vision_office_sessions._records` reads only `s.stationary`.

## Files changed

- `orion/vision/stationary.py`, `orion/vision/tests/test_stationary.py`: the room rule and its tests.
- `services/orion-vision-window/app/{fixtures,caption_gate,main,settings}.py`, `.env_example`, `docker-compose.yml`, tests: the room rule in presence and the caption gate.
- `services/orion-sql-writer/app/{vision_individuals,vision_individuals_loop,vision_office_sessions,settings}.py`, `.env_example`, tests: the room rule in the people tracker, and the stale-flag fix in office visits.
- `scripts/sync_local_env_from_example.py`: adds the `VISION_STATIONARY_` prefix.
- `services/orion-vision-host/evals/office_person_cutoff_eval.py` and its fixture: the eval checks the cut-off and the room rule. The data now runs through 11-23.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: cam0 sightings can now be `stationary`. Office visits read only the sighting flag.
- Compatibility notes: no migration.

## Env/config changes

- Added keys: `VISION_STATIONARY_ROOM_STREAMS=cam0` and `VISION_STATIONARY_ROOM_MIN_SEC=7200`, in orion-sql-writer and orion-vision-window.
- `.env_example` updated: yes.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py orion-sql-writer orion-vision-window`: yes. Both keys were added in both services.
- Skipped keys: none.
- Off switch for the office: `VISION_STATIONARY_ROOM_STREAMS=` (empty), or `VISION_STATIONARY_ROOM_MIN_SEC=0`.

## Tests run

```text
orion/vision/tests                                   56 passed
orion-vision-window tests                           187 passed
orion-sql-writer vision unit tests                   97 passed (PG tests skipped locally)
orion-sql-writer PG: office sessions + person model  81 passed (throwaway postgres:16, CI schema setup)
Mutation checks: sticky gaps, face restart, and the stale individual flag each make a new test fail.
```

## Evals run

```text
.venv/bin/python services/orion-vision-host/evals/office_person_cutoff_eval.py
cut-off cam0 person = 0.55
  prop boxes kept: 1/2397
  real boxes kept: 1194/1246 (96%)
  real boxes with a face kept: 68/69
room prop rule: VISION_STATIONARY_ROOM_STREAMS=cam0 hold=7200 s
  real: longest still run 13.7 min, first marked a prop: never
  prop: longest still run 286.4 min, first marked a prop: 2026-11-15T21:46+00:00
PASS
```

## Docker/build/smoke checks

```text
No image build is needed. The change is Python only, plus env.
Live behavior: UNVERIFIED until deploy. Tracked in #3770.
```

## Review findings fixed

- Finding: office visits dropped any person whose record carried a stale "object" flag (`i.stationary`), which brought back "the sticky prop hides Juniper".
  - Fix: `_records` uses only the sighting flag. `became_stationary` already back-flags the sightings a prop run covers.
  - Evidence: a new PG test fails against the old code and passes now.
- Finding: with `stationary_streams=None`, the sql-writer gave cam0 the sticky outdoor rule.
  - Fix: room streams win on that path too.
- Finding: `VISION_STATIONARY_MIN_SEC=0` no longer turns cam0 off.
  - Fix: `.env_example` documents the room keys as cam0's own off switch.
- Finding: the face check was `is not None`, but the rest of the file uses a truthy check.
  - Fix: changed to `bool(...)`.
- Not changed (low): `vision_person_model.unnamed_clusters` skips an individual whose flag is stale until that person is seen again. The next crop clears the flag. The impact is a short delay in attribution only.

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-window up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

Run both from the clean primary main checkout after the merge.

## Risks / concerns

- Severity: medium
  - Concern: few real office visits exist (about 2.6 h of presence, and only 69 crops with a face). A person who sits still for over 2 h with their back to the camera would be marked an object until they move or leave. That visit's sightings would stay flagged.
  - Mitigation: a gap or a face clears the flag. #3770 checks it live.
- Severity: low
  - Concern: the pile now mostly scores under 0.55, so the room rule may never fire live.
  - Mitigation: #3770 closes on "real visits unaffected" if so.
- Issues opened: #3770. Fixes: #3614.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
