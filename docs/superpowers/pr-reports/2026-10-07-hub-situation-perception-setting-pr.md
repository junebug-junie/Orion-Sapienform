**Privacy change, read first.** After this merges and Hub is recreated, every Hub chat prompt carries what the room camera last saw, in plain text. Real narratives from the live `vision_events` table (2026-10-06/07, cam0), shown in the exact prompt-line format. Nothing needed redacting: the narratives contain no names or faces. Only the first line's "8 min ago" age was observed at build time; the second line's age is illustrative.

- `Room (seen 8 min ago): Multiple chairs, doors, tables, and desks are visible in the scene. One item of clothing is also present.`
- `Room (seen 3 min ago): A table with four legs is present, with two chairs, a desk, a door, and a box nearby. One person is visible.`
- `A person was detected on camera.` (real narrative, 2026-10-06 16:25 UTC). When the presence row is under 2 minutes old and reads "present", the line is prefixed with a coarse duration clause, for example `Someone has been in view for 20 minutes.`

When the walkway camera has something to say, a `Street (walkway camera): ...` line is added too. Today the walkway tables are empty, so that line is absent. About once every 6 hours, one of Juniper's own turns can also get a one-off instruction to ask "is that you, Juniper?". That cooldown is shared with cortex-exec through the same Redis key, so Hub and cortex-exec cannot both ask. Turns Orion starts on its own (outreach) never use up that ask.

**Kill switch:** set `ORION_SITUATION_PERCEPTION_ENABLED=false` in `services/orion-hub/.env`, then recreate orion-hub.

## Summary

- Hub's chat Situation block now includes camera perception. A new flag, `ORION_SITUATION_PERCEPTION_ENABLED`, defaults to on (per Juniper's standing flag rule) and replaces the literal `False` in `hub_settings_to_runtime_namespace`.
- Hub uses the same perception builder as cortex-exec (`_build_perception_context`). It is not a fork. Room cameras (`carbon,cam0`), the street camera (`walkway`) and the 900 s staleness limit all match cortex-exec's defaults.
- The room-percept SQL read now runs in a worker thread (`asyncio.to_thread`), so a slow database cannot freeze Hub's event loop. The presence and street reads already did this; the room read did not. cortex-exec gets the same fix.
- `ORION_SITUATION_PERCEPTION_` and `ORION_SITUATION_STREET_` are added to the env-sync prefixes, so the local `.env` actually receives the new keys.
- The identity ask is now limited to Juniper's own turns, the database connect has a 3 s timeout, and a cached brief is rebuilt once its room view passes the 15-minute (900 s) limit. All three came from review.
- New tests cover: flag on/off, Orion-authored turns not spending the ask, cache vs staleness, stale/absent/erroring camera, a stale presence row, the worst-case budget, a tight cap, and the read running off the event loop. A live eval measures what perception adds on real data.

## Outcome moved

Hub (unified-turn) chat used to say `Room: haven't seen anything recently; do not infer.` on every turn, even when the camera had a fresh view. cortex-exec's chat, meanwhile, got the real scene. With this change Orion in Hub chat can ground "what's around me" in the same live percept.

## Current architecture

- `orion/hub/turn_orchestrator.py::_build_situation_prompt_fragment` calls `hub_settings_to_runtime_namespace(settings)` and then `build_situation_for_ctx`. The resulting `compact_text` becomes `HarnessRunRequestV1.situation_prompt_fragment`, and is also emitted as a cockpit `situation` hop.
- Perception was hardcoded `False` in that adapter. The original reason was that no vetted DB path existed for Hub's event loop.
- cortex-exec has run perception for weeks with `ORION_SITUATION_PERCEPTION_ENABLED=true`.

## Architecture touched

- Shared: `orion/situational/context.py` (the adapter, plus the room read moving to a worker thread).
- Hub: settings, `.env_example`, eval.
- No bus, schema, or channel changes.

## Files changed

- `orion/situational/context.py`: the adapter reads Hub's perception/street keys; `fetch_latest_percept` is awaited via `asyncio.to_thread`; the identity ask only fires when `record_user_turn` is true; a cache hit is refused once the cached percept's age plus the time it has sat in the cache passes `perception_max_age_seconds`; stale "perception stays off" docstrings are corrected.
- `orion/situational/perception_reader.py`: `connect_timeout=3` on the engine.
- `services/orion-hub/app/settings.py`: 4 new fields (default on, cortex-exec parity).
- `services/orion-hub/.env_example`: 4 new keys with a privacy note and the kill switch.
- `scripts/sync_local_env_from_example.py`: sync prefixes for the new keys. Without them the sync silently skipped them.
- `orion/situational/tests/test_hub_situation_perception.py`: new tests.
- `orion/situational/tests/test_hub_settings_adapter.py`: perception default flipped to on.
- `services/orion-hub/evals/run_situation_perception_eval.py`: live eval comparing perception off vs on.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: Hub's Situation block now carries Room/Street perception lines and can carry the identity-ask caution. The cockpit `situation` hop's `perception_enabled` is now `true`, and its `compact_text` (stored in `cockpit_turn_sighting.raw`) now contains the room line.
- Compatibility notes: same `PerceptionContextV1` and the same builder as cortex-exec.

## Semantic-layer impact (over/under-indexing)

- **Where the text goes:** only into the harness prompt and the cockpit `situation` hop. The hop is persisted to `cockpit_turn_sighting` (about 580 rows/week), and the only reader of that table is Hub's own cockpit UI (`chat_cockpit_routes.py`). No recall, embedding, memory or journal path reads either the hop or `situation_prompt_fragment`. So the room narrative is not indexed into memory, and recall gets no repetitive "chairs, doors, desks" text.
- **Signals and metrics:** none. Nothing in metric lineage reads Hub's `provider_status`/`source_summary` perception keys (world-pulse `situation_state` is a separate producer). No new metric, so the metric gate does not apply.
- **Under-indexing guard:** the "do not infer" line stays in place whenever the camera is off, stale, or unread, so the absence of a view never turns into a claim that the room is empty.

## Staleness behaviour

All of this is existing shared-builder logic, now pinned by tests on the Hub path:

| Camera state | What the prompt says |
|---|---|
| fresh percept (<= 900 s) | `Room (seen N min ago): <narrative>` |
| percept older than 900 s | `Room: haven't seen anything recently; do not infer.`; scene text is dropped from the payload, not just from the prompt |
| no row / empty narrative | same "do not infer" line |
| database error / no DSN | same "do not infer" line; status `error` |
| cached brief whose percept has aged past 900 s while in the cache | rebuilt, never served (the effective limit was 900 s + the 300 s cache lifetime before this) |
| presence row older than 120 s | no "Someone has been in view" clause, even when the scene is fresh |
| street quiet or unread | no Street line at all |

## Prompt budget

Hub's cap is `ORION_SITUATION_PROMPT_MAX_CHARS=7200`.

- **Live (2026-10-07, real vision_events):** perception off gives 733 chars, on gives 810-811, so it adds **77-78 chars**. That is the room line replacing the 51-char "do not infer" line.
- **Over 7 days of cam0 narratives:** median 65 chars, p95 101, max 159.
- **Synthetic worst case** (longest real scene, a presence clause, a 4-line street summary, and the longest identity-ask caution): adds 1063 chars (about 15% of the cap). Nothing is truncated and every caution survives (test-pinned).
- Under a tight cap, facts are shortened first; cautions are kept whole or dropped, never cut mid-sentence (test-pinned at 600).

## Env/config changes

- Added keys (orion-hub): `ORION_SITUATION_PERCEPTION_ENABLED=true`, `ORION_SITUATION_PERCEPTION_MAX_AGE_SECONDS=900`, `ORION_SITUATION_PERCEPTION_STREAM_IDS=carbon,cam0`, `ORION_SITUATION_STREET_STREAM_IDS=walkway`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (`services/orion-hub/.env_example`)
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. All 4 keys were added to the primary checkout's `services/orion-hub/.env` (lines 849-852). cortex-exec's existing values already match, so nothing diverged.
- Skipped keys requiring operator action: none
- Compose: orion-hub uses `env_file: .env`, so all keys reach the container (`check_service_env_compose_parity.py orion-hub`: N/A, env_file).
- DSN: Hub already has `POSTGRES_URI`. Verified that the running Hub container reads `vision_events` (`fetch_latest_percept` returned a row observed at 2026-10-07 01:22 UTC).

## Tests run

```text
pytest orion/situational/tests -q                                   121 passed
pytest services/orion-cortex-exec/tests/{situation,perception,street,identity}*.py   183 passed, 2 failed (pre-existing cwd-relative template path; pass from repo root, fail identically on main)
pytest services/orion-hub/tests/{situation,turn_orchestrator,unified_turn,settings}*.py  142 passed, 1 failed (test_execute_unified_turn_uses_mind_appraisal_text_for_stance_not_harness -- fails identically on main, unrelated); after review fixes: situation/unified_turn/cockpit_hops/ws_frames 119 passed
mutation check: reverting the identity-ask gate or the cache-staleness guard each fails its new test
pytest tests/test_env_key_single_source.py tests/test_check_settings_defaults.py tests/test_check_service_env_compose_parity.py  37 passed, 1 skipped
all 16 scripts/check_*.py gates referenced in .github/workflows   pass
```

## Evals run

```text
POSTGRES_URI=<hub> PYTHONPATH=. python services/orion-hub/evals/run_situation_perception_eval.py
perception_source=live, age=177s, provider_status={perception: ok, perception_street: quiet}
off_chars=733 on_chars=811 added_chars=78 budget=7200   rc=0
(after review, run with Hub's real .env sourced) added_chars=68, rc=0; with no DSN -> UNVERIFIED, rc=2
```

## Docker/build/smoke checks

```text
No image build needed: shared-package and settings change only.
docker exec orion-athena-hub python3 -c "fetch_latest_percept(stream_ids=['carbon','cam0'])"
-> True 2026-10-07 01:22:47 UTC  (the Hub container can read vision_events)
UNVERIFIED: a live Hub turn carrying the Room line. That needs a deploy, which this PR does not do.
```

## Review findings fixed

- Finding (material): Orion-authored Hub turns (endogenous outreach, `record_user_turn=False`) could claim the shared identity-ask cooldown. The result would be an unprompted "is that you?" and a used-up slot for Juniper's next real turn.
  - Fix: `allow_identity_ask=_records_user_turn(ctx)` is threaded through `_build_perception_context`, `_build_room_perception_context` and `_resolve_presence_and_identity_ask`. The default is True, so cortex-exec is unchanged.
  - Evidence: `test_orion_authored_turn_never_spends_the_identity_ask` (the claim stub raises if called) and `test_juniper_turn_still_gets_the_identity_ask`. Removing the gate fails the first test.
- Finding (minor): the perception engine had no `connect_timeout`. A blackholed database host would hold each uncached turn for the OS TCP timeout.
  - Fix: `connect_timeout=3` in `connect_args`.
  - Evidence: the existing reader tests still pass. This is a static change and was not exercised against a blackholed host.
- Finding (minor): the eval returned 0 on "no data" and ignored Hub's configured values.
  - Fix: it now exits 2 with `UNVERIFIED` when no DSN is set or the read is not ok/stale, and reads Hub's perception keys from the environment.
  - Evidence: no DSN gives rc=2; Hub's `.env` sourced gives rc=0 with a live Room line.
- Finding (minor): the cache let a percept outlive the 900 s gate by up to 300 s.
  - Fix: `_cached_percept_outlived_gate` turns that case into a cache miss.
  - Evidence: `test_cached_brief_is_rebuilt_once_its_percept_passes_the_gate`, which fails with the guard removed, and `test_fresh_cached_brief_is_still_served_from_cache`.
- Findings (nits): the outage test pinned a path production never takes; the flag-off and empty-street tests were too weak; the tight-cap prefix check was loose.
  - Fix: added an outage test through the real reader with a dead engine (`unavailable`); flag-off now fails if presence or street are read; a builder-level empty-street test; any cut-off prefix of a caution now fails.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --force-recreate orion-hub
```

cortex-exec also picks up the off-loop read on its next rebuild/recreate. That is not urgent: its behaviour is unchanged apart from no longer blocking.

## Risks / concerns

- Severity: medium (privacy, by design). Concern: camera-derived home text now reaches Hub chat prompts and is stored in `cockpit_turn_sighting.raw`. Mitigation: narrative-only reader (no identity columns), staleness gate, one-line kill switch.
- Severity: low. Concern: the identity-ask caution can now fire on a Hub turn instead of a cortex-exec turn. Mitigation: shared Redis cooldown, so it is still at most once per window across both.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
