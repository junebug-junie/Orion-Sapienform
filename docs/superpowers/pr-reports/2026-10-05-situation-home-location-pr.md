## Summary

- The unified-turn "Situation" block had a timezone but no place: the Hub adapter hardcoded `location_label="Unknown"`, `locality=None`.
- Hub now reads `ORION_SITUATION_LOCATION_LABEL/LOCALITY/REGION/COUNTRY/LOCATION_PRECISION`.
- New fixed facts `ORION_SITUATION_HOME_LOCATION` ("Ogden, Utah") and `ORION_SITUATION_PHYSICAL_LOCATION` ("Server cabinet in basement, in office, laundry closet") render as a `Place:` line right after `Local context:`.
- The line says Juniper's travel city is where *she* is, not where Orion's body, cameras or sensors are.
- Same two keys added to cortex-exec (legacy chat-verb lane) and its stance place compaction.

## Outcome moved

2026-10-05, correlation `5063fb71-471c-46e2-860d-8415f9253e14`: Juniper was in a Chicago hotel and said a camera would soon watch the road at home. Orion replied "whatever Chicago looks like at 2 AM from your hotel window". Recall had Chicago 4x and Ogden 0x; the Situation block had no place at all.

## Current architecture

Hub builds the fragment (`orion/hub/turn_orchestrator.py::_build_situation_prompt_fragment`) via `hub_settings_to_runtime_namespace` -> `settings_from_runtime` -> `_build_place_context` -> `_build_prompt_fragment`. cortex-exec had real place config (Ogden) but only for its own legacy lane.

## Architecture touched

`orion/schemas/situation.py` (`PlaceContextV1` +2 optional fields), `orion/situational/context.py`, hub + cortex-exec settings/`.env_example`, `chat_stance.py` place compaction.

## Files changed

- `orion/schemas/situation.py`: `home_location`, `physical_location` on `PlaceContextV1`.
- `orion/situational/context.py`: settings fields, hub adapter reads place keys, `Place:` line.
- `services/orion-hub/app/settings.py`, `.env_example`: new place keys.
- `services/orion-cortex-exec/app/settings.py`, `.env_example`, `app/chat_stance.py`: new keys, compaction.
- `orion/situational/tests/test_situation_home_location.py`: regression tests.

## Schema / bus / API changes

- Added: two optional `PlaceContextV1` fields (default None). No bus/channel change.
- Compatibility: unconfigured services render nothing new.

## Env/config changes

- Added keys: `ORION_SITUATION_HOME_LOCATION`, `ORION_SITUATION_PHYSICAL_LOCATION` (hub + cortex-exec); hub also gets `LOCATION_LABEL/LOCALITY/REGION/COUNTRY/LOCATION_PRECISION`.
- `.env_example` updated: yes. Local `.env` for hub and cortex-exec: keys appended directly, because `sync_local_env_from_example.py` reads the primary checkout's example, not this worktree's.
- Skipped keys: none.

## Tests run

```text
orion/situational/tests: 98 passed (4 new)
cortex-exec situation + relational stance (from repo root): 59 passed
orion-hub -k "situation or settings": 35 passed
env template parity: PASS; git diff --check: clean
```

## Evals run

No eval harness for the Situation block; the rendered-line test is the check. Follow-up: live smoke after Hub restart (below).

## Docker/build/smoke checks

Not run. UNVERIFIED live until Hub restarts and a turn's prompt is inspected.

## Review findings fixed

- Finding: Place line too long / could crowd cautions (reviewer assumed a 1200 cap; live cap is 7200 per `_DEFAULT_PROMPT_MAX_CHARS`).
  - Fix: shortened to ~150 chars; test asserts it stays under 260.
  - Evidence: `orion/situational/tests` 99 passed.
- Finding: travel hint is emitted every turn, even when Juniper is home.
  - Fix: facts only, "(fixed; independent of where Juniper is right now)".
  - Evidence: `test_rendered_block_states_home_and_body_as_fixed`.
- Finding: blank `ORION_SITUATION_LOCATION_LABEL` became `configured_home` with empty label.
  - Fix: blank falls back to "Unknown".
  - Evidence: `test_blank_location_label_falls_back_to_unknown`.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --force-recreate orion-hub
```
(and cortex-exec containers pick up the two new keys on their next recreate; not required for the Hub fix.)

## Risks / concerns

- Low: the `Place:` line adds ~150 chars to every turn; cap is 7200.

## PR link

TBD
