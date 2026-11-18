## Summary

- **Orion can now tell Juniper is home from evidence,** not only infer it because a trip's end date passed. A corroborated face match on a home camera becomes "I saw Juniper at home on camera cam0" in the situation graph's whereabouts.
- **The room camera already recognized her.** Live on 10-09 between 01:56 and 02:00 it made 6 probable and 14 possible matches. But the match lived only in the camera's presence row while she was in view, and was rewritten to `subject: unknown` once she stepped away.
- **New contract:** `IdentitySightingV1` on `orion:vision:identity:sighting`.
  - orion-vision-window publishes it for home streams only (`cam0`), never the laptop webcam.
  - It needs two "probable" matches within 10 minutes, and publishes at most once per 30 minutes per camera.
- **In the situation graph:**
  - the sighting is whereabouts for 2 hours, unless something Juniper said is newer;
  - it never erases her words: a stated stay it outranks stays current in `doing`, and the Hub shows both;
  - a refreshed or aged-out sighting is never recorded as "no longer true".
- **Hub situation block** renders it as "(seen N minutes ago)" and lists the stays she stated.

## Outcome moved

Orion has positive evidence for "Juniper is home" for the first time. Before this, after the Chicago trip, "not in Chicago" rested only on her stated flight time passing.

Live identity checks on cam0 (vision-window log, 10-09 01:56–02:00):

```text
probable 6 (best similarity 0.70), possible 14, no_face 8
```

## Current architecture

1. vision-host matches faces against a one-person gallery (Juniper, enrolled 08-26) whenever the frame router sees a person on cam0.
2. vision-window keeps the latest match per stream for the presence row and the council.
3. Nothing else consumed it, and nothing persisted it.

## Architecture touched

- **Contract:** `orion/schemas/vision_sighting.py`, the registry, `channels.yaml`, the metric lock (+1 channel).
- **Producer:** orion-vision-window `_maybe_publish_sighting`.
- **Consumer:** orion-durable-runs, through the situation driver intake and `apply_sighting` in the graph.
- **Situation schema:** `SituationStateV1` gains `last_sighting`; fact and lapsed `memory_id` become optional; `until_source` and event kind gain `"sighting"`.
- **orion-hub:** the `situation_lines` wording.

## Files changed

- `orion/schemas/vision_sighting.py` (new), `orion/schemas/registry.py`, `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`
- `orion/schemas/situation_state.py`
- `services/orion-vision-window/app/main.py`, `app/settings.py`, `.env_example`, `tests/test_identity_sighting.py` (new, 7 tests)
- `services/orion-durable-runs/app/situation_graph.py`, `app/situation_driver.py`, `app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `tests/test_situation_graph.py` (+11 tests)
- `services/orion-hub/scripts/endogenous_outreach.py`, `tests/test_endogenous_outreach.py` (+2 tests)

## Schema / bus / API changes

- **Added:** `IdentitySightingV1`, `vision.identity.sighting.v1`, channel `orion:vision:identity:sighting`; `SituationStateV1.last_sighting`; `SituationSightingV1`.
- **Behavior changed:** `SituationFactV1.memory_id` and `SituationLapsedV1.memory_id` are optional; `until_source="sighting"`; `EventKind="sighting"`.
- **Compatibility:** these models are `extra="forbid"`. An old Hub reading the new Redis payload fails validation and silently drops the whole situation block. **Deploy Hub first.**

## Env/config changes

- **orion-vision-window:** `WINDOW_SIGHTING_ENABLED=true`, `WINDOW_SIGHTING_HOME_STREAMS=cam0`, `WINDOW_SIGHTING_MIN_INTERVAL_SEC=1800.0`, `WINDOW_SIGHTING_MIN_MATCHES=2`, `WINDOW_SIGHTING_MATCH_WINDOW_SEC=600.0`, `CHANNEL_IDENTITY_SIGHTING_PUB`.
- **orion-durable-runs:** `SITUATION_SIGHTING_HOLD_HOURS=2.0`, `CHANNEL_IDENTITY_SIGHTING`.
- **Local `.env`:** synced (`--all-keys` for both services). The hold value was corrected from an earlier 18.0 to 2.0 by hand, for that one key only.

## Tests run

```text
orion-durable-runs  pytest tests -q                310 passed, 72 skipped
orion-vision-window pytest tests -q                113 passed
orion-hub           pytest tests/*outreach*.py     318 passed
metric drift + registry + bus catalog tests        59 passed
check_service_env_compose_parity orion-durable-runs  OK
```

## Evals run

```text
None beyond tests. Live proof is the first corroborated cam0 match after deploy (see Restart).
```

## Docker/build/smoke checks

```text
Not built from this branch.
```

## Review findings fixed

- Finding (HIGH): each new sighting marked the previous one "No longer true", so the prompt contradicted itself, and it used up a lapsed slot.
  - Fix: one fixed key for the sighting; a sighting never lapses.
  - Evidence: `test_a_refreshed_sighting_is_not_a_lapse`, `test_an_aged_out_sighting_leaves_no_lapsed_entry`.
- Finding (HIGH): one borderline match could wipe a stay Juniper stated, and the stay returned 18 hours later.
  - Fix:
    - two probable matches are needed within 10 minutes;
    - the sighting leads only while fresh (2 h);
    - her stated stay stays current in `doing` and is shown next to the sighting.
  - Evidence: `test_one_probable_frame_is_not_enough_two_within_ten_minutes_are`, `test_a_fresh_sighting_outranks_a_stated_trip_without_erasing_it`, `test_once_the_sighting_is_stale_her_stated_trip_leads_again`, `test_a_stated_stay_outranked_by_a_sighting_is_still_shown`.
- Finding (MEDIUM): an 18-hour hold claimed she was home long after she might have left.
  - Fix: the hold is now 2 hours. After that, nothing is claimed.
- Finding (MEDIUM): deploy order.
  - Fix: Hub, then durable-runs, then vision-window (below).
  - Rollback: roll durable-runs back only after any sighting has aged out, because old code can't read a sighting checkpoint.
- Finding (LOW): `seen_at` used wall-clock time.
  - Fix: it now uses the frame's `frame_ts` when present.
  - Evidence: `test_seen_at_is_the_frame_time_when_known`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-hub orion-durable-runs orion-vision-window; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done
```

To prove it live, stand in front of cam0 for a minute. Then:
- the vision-window log shows `identity_sighting stream=cam0`;
- `orion:situation:latest` shows whereabouts "I saw Juniper at home on camera cam0."

## Risks / concerns

- **Medium: false-positive rate unknown.**
  - Concern: the gallery holds one person, so "probable" means cosine ≥ 0.55 against Juniper alone. Nobody has measured how other household members score.
  - Mitigation: two matches are required, the claim lasts only 2 hours, and her stated words are never erased.
  - Follow-up: log the matches for a week and check them.
- **Low: the rate limit resets on restart.**
  - Concern: it's in-memory, so the first corroborated match after a restart publishes again.
  - Mitigation: harmless; it just refreshes the sighting.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
