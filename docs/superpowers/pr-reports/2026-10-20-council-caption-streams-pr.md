## Summary

- The vision council no longer sees walkway caption text. Only cameras listed in the new setting `COUNCIL_CAPTION_STREAMS` (default `cam0`, the room camera) keep their captions.
- For every other camera the council gets a copy of the window with the captions removed, and with the words the window service derived from captions (`soft_labels`, `caption_count`) removed too. No "soft hint" fallback.
- The same caption-free copy feeds the prompt, the scene-summary fallback (`_derive_scene_summary` can no longer return a walkway caption), grounding, and the published events. Walkway activity claims keep the 0.4 confidence cap. A walkway window that held only a caption skips the LLM.
- The unresolved-percept row ("I could not name this") still gets the full window. It quotes the caption as "caption model said: ...", which names the source. Juniper decided on 2026-10-15 that walkway unresolved rows carry the caption.
- Implements spec PR #2763, item 3 patch A (Juniper decision 4: room camera keeps captions).

## Outcome moved

Failure mode closed at the source: the council copied walkway caption guesses ("A person is pushing a lawn mower...", "A vehicle is parked...") into unlabelled `visual_observation` rows, and readers showed those rows as fact. Live copy rate before: about 8 council rows in 7,585 walkway rows. Target after deploy: 0 on walkway (spec appendix query A3). The caption itself still reaches Orion as its own labelled `camera_caption` row from orion-vision-window.

## Current architecture

orion-vision-window puts accepted captions into `window.summary.captions` and caption stoplist tokens into `summary.evidence.soft_labels`. The council put up to 3 captions into every prompt with a rule "soft hints only", which the 8B model did not always follow. `_derive_scene_summary` fell back to `captions[0]` when the model gave no summary.

## Architecture touched

- orion-vision-council only. One new setting. No bus, schema, or registry change.
- Seam: `app/caption_policy.py` `window_for_council`, applied once at intake in `_process_window` and `_handle_rpc`.
- Stream key: `stream_key_from_window` (window `stream_id`, then `camera_id`), the same key the evidence-transition gate uses. Live check 2026-10-20: council rows in the last 72 h carry `stream_id` `walkway` and `cam0` only.

## Files changed

- `services/orion-vision-council/app/caption_policy.py`: new. Parses the setting and builds the caption-free window copy.
- `services/orion-vision-council/app/main.py`: applies the policy at both intake paths; logs `[COUNCIL] caption_streams=...` at startup.
- `services/orion-vision-council/app/settings.py`: `COUNCIL_CAPTION_STREAMS: str = "cam0"`.
- `services/orion-vision-council/.env_example`, `docker-compose.yml`: the new key, default `cam0`.
- `services/orion-vision-council/README.md`: how captions are handled per camera; grounding table note; test list; eval gap.
- `services/orion-vision-council/tests/test_caption_policy.py`: new gate tests.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: for streams not in `COUNCIL_CAPTION_STREAMS`, the council prompt has `"captions": []` and no `soft_labels`/`caption_count`; activity claims there are capped at 0.4 but no longer tagged `caption_inferred` (no caption was in the input); caption-only windows there skip the LLM.
- Compatibility notes: cam0 behavior unchanged. Set `COUNCIL_CAPTION_STREAMS=cam0,walkway` to restore the old walkway behavior.

## Env/config changes

- Added keys: `COUNCIL_CAPTION_STREAMS` (default `cam0`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`+COUNCIL_CAPTION_STREAMS='cam0'` in `services/orion-vision-council/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-vision-council && PYTHONPATH=<worktree> python -m pytest tests -q
  124 passed
python -m pytest tests/test_no_personal_terms.py tests/test_compose_logging_journald.py \
  tests/test_ci_service_image_sources.py tests/test_check_settings_defaults.py \
  tests/scripts/test_sync_local_env_from_example.py -q
  56 passed
scripts/check_service_env_compose_parity.py orion-vision-council   N/A (env_file reaches container)
scripts/check_env_key_single_source.py                            OK
scripts/check_metric_lineage.py --gate                            PASS
scripts/check_definition_drift.py --gate                          PASS
compose / hostname / health-producer / async-route static gates  PASS
git diff --check                                                  clean
```

No GitHub workflow runs the council's own tests; they were run locally.

## Evals run

```text
None. orion-vision-council has no evals/ harness.
```

Follow-up: the spec's live acceptance check (query A3, copy rate on walkway = 0 over 24 h after deploy) is the eval for this patch. It needs the deploy first.

## Docker/build/smoke checks

```text
Not run. No dependency or Dockerfile change; the compose file only gains one env line.
Live behavior after deploy is UNVERIFIED.
```

## Review findings fixed

Review ran in a subagent on commit `cac500c75`. No must-fix items.

- Finding (should-fix): walkway activity claims lost the 0.4 confidence cap, because the cap fired only when captions were in the window. Less support, more confidence.
  - Fix: the stripped window carries a non-text marker `captions_withheld`. Grounding caps activity claims at 0.4 on such windows, with no caption tag (log note `capped_confidence:<type>:captions_withheld`).
  - Evidence: `test_grounding_caps_withheld_activity_without_caption_tag`, `test_process_window_walkway_end_to_end`.
- Finding (should-fix): caption-only walkway windows (a late caption artifact, no detector pass) would reach the LLM with an empty summary. The model could invent content.
  - Fix: intake skips the LLM when captions were withheld and `item_count` is 0. It logs `[COUNCIL] caption_only_window skip` and still runs the unresolved check. cam0 caption-only windows still interpret.
  - Evidence: `test_caption_only_walkway_window_skips_the_llm`, `test_caption_only_cam0_window_still_interprets`. How often this happens live is UNVERIFIED until the log line runs after deploy.
- Finding (should-fix): the tests did not catch removal of most of the wiring (no RPC test; the person fallback has fixed text, so the published check passed anyway).
  - Fix: the end-to-end tests now patch only the LLM call. They check the prompt, the scene summary, the published confidence and tags, and the unresolved window. There is also a new RPC test.
  - Evidence: mutation check. Reverting the RPC wiring and the finalize wiring to the raw window fails `test_handle_rpc_uses_caption_free_window` and `test_process_window_walkway_end_to_end`.
- Finding (nit): stream ids were compared case-sensitively.
  - Fix: both sides are lowercased. Evidence: `test_stream_match_ignores_case`.
- Finding (nit, not changed): the prompt still says "treat summary.captions as soft hints" when captions are always `[]`. This is harmless. Changing it would also change the cam0 prompt.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-vision-council up -d --build
```

Proof after restart: `docker logs orion-athena-vision-council 2>&1 | grep caption_streams` shows `caption_streams=cam0`.

## Risks / concerns

- Severity: low
  - Concern: motion verbs (driving, walking, pushing) are still not checked by grounding. The council can still guess motion from labels alone.
  - Mitigation: that is item 3 patch C (motion cross-check), out of scope here.
- Severity: low
  - Concern: a window with no `stream_id` and no `camera_id` gets no captions (it is not on the list).
  - Mitigation: live windows carry `stream_id`; fail-closed is the intended direction.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2777

🤖 Generated with [Claude Code](https://claude.com/claude-code)
