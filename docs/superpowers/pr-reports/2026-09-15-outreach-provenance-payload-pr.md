# Outreach provenance payload — PR report

Branch: `feat/outreach-provenance-payload`

## Summary

- Endogenous outreach now keeps a full `outreach_provenance.v1` capsule (generation prompt + lane summary + shared `decision_id`) instead of dropping context at deliver time.
- Capsule is written to chat `client_meta.outreach_provenance`, `endogenous_outreach_decisions.result_json.provenance`, and the live WS `orion_outreach` payload.
- Unified turns reinject the latest still-relevant capsule into the situation prompt so follow-ups can cite the stored prompt instead of inventing a collapse-mirror frame.
- Hub bubbles render a collapsible “why I spoke” panel from live WS provenance.
- Collapse-mirror and non-outreach `_deliver` paths stay fail-open with `provenance=None`.

## Outcome moved

When Orion speaks unsolicited, both Orion and Juniper can answer “where did that come from?” from the real generation prompt. The seed incident (`correlation_id=bfda9f83-…`) was confabulation after provenance was discarded; that path now has durable evidence.

## Current architecture

Before this patch:

- `build_outreach_prompt` existed only in memory for generation.
- `endogenous_outreach_decisions` stored lane booleans/counts (`grounding`), not the prompt.
- Chat history got `client_meta.unsolicited=true` only.
- Follow-up unified turns had no outreach provenance; Hub showed no “why I spoke”.

## Architecture touched

- `services/orion-hub` — capsule builders, deliver write path, WS + UI, decision PK reuse
- `orion/hub/turn_orchestrator.py` — async fail-open situation injection
- Design + plan docs under `docs/superpowers/`

## Files changed

- `docs/superpowers/specs/2026-09-15-outreach-provenance-payload-design.md`: approved design
- `docs/superpowers/plans/2026-09-15-outreach-provenance-payload.md`: implementation plan
- `services/orion-hub/scripts/endogenous_outreach.py`: `summarize_outreach_lanes` / `build_outreach_provenance`; mint capsule on successful deliver; thread through `_deliver` / sockets / history
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: prefer caller `decision_id` as PK
- `services/orion-hub/scripts/outreach_provenance.py`: fetch / format / merge / capsule validation
- `orion/hub/turn_orchestrator.py`: `_situation_with_outreach_provenance` via `asyncio.to_thread`
- `services/orion-hub/static/js/app.js`: forward WS provenance; render `<details class="om-outreach-why">`
- `services/orion-hub/tests/test_endogenous_outreach.py`: capsule + deliver threading tests
- `services/orion-hub/tests/test_outreach_provenance.py`: format/merge/fetch validation + async helper
- `services/orion-hub/tests/test_hub_ui_layout_pass.py`: static UI wiring asserts

## Schema / bus / API changes

- Added: additive JSON shape `outreach_provenance.v1` inside existing `client_meta` / `result_json` / WS payload (no new bus channel, no registry event)
- Removed: none
- Renamed: none
- Behavior changed: successful endogenous outreach ships provenance; follow-up turns may append a provenance block to `situation_prompt_fragment`
- Compatibility notes: readers that ignore unknown keys are fine; missing/malformed capsules fail open (no injection, no UI panel)

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: n/a
- skipped keys requiring operator action: none

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-outreach-provenance-payload
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest \
  services/orion-hub/tests/test_endogenous_outreach.py \
  services/orion-hub/tests/test_outreach_provenance.py \
  services/orion-hub/tests/test_hub_ui_layout_pass.py \
  services/orion-hub/tests/test_collapse_mirror_chat_reply.py -q

249 passed, 18 warnings in 9.37s
```

Warnings are pre-existing pydantic `model_*` protected-namespace noise, not from this patch.

## Evals run

```text
No new eval harness for this change.
services/orion-hub/evals/ exists for other hub concerns; this patch is
deterministic capsule write/inject/UI wiring covered by gate tests.
Live quality smoke (force outreach → “where did that come from?”)
is listed under Restart / acceptance and remains UNVERIFIED until hub rebuild.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub config
→ env template parity PASS; hostname refs OK
→ FAILED: couldn't find env file:
   /mnt/scripts/Orion-Sapienform-outreach-provenance-payload/.env
Compose config itself: UNVERIFIED in this worktree (no local .env).
No shared-checkout deploy attempted.
```

## Review findings fixed

- Finding: sync SQLAlchemy fetch inside async unified-turn path could block Hub’s event loop
  - Fix: `await asyncio.to_thread(fetch_latest_outreach_provenance, …)` in Task 3 follow-up
  - Evidence: `40d0e0f6a`; `test_outreach_provenance.py` async helper coverage
- Finding: malformed capsules could still format/inject
  - Fix: `_valid_outreach_capsule` requires `schema == outreach_provenance.v1` and non-empty `prompt_text`
  - Evidence: bad-schema / non-string prompt tests in `test_outreach_provenance.py`

### Minor findings left as optional follow-ups (SDD reviews)

- `summary_line` content assert is presence/non-empty only; no fixture that pins exact lane prose.
- No direct unit test that `record_decision` uses caller `decision_id` as the DB PK (threading covered at deliver/WS/history level).
- UI fallback can produce a duplicate “why I spoke” label when `summary_line` is empty (`why I spoke — why I spoke` style path).
- History rehydrate does not restore “why I spoke” on past outreach after page reload; live WS coverage satisfies acceptance.

## Restart required

```bash
# From this worktree (not the shared checkout), after merge or for live verify:
scripts/safe_docker_build.sh orion-hub up -d --build
```

Acceptance smoke after restart:

1. `POST /api/debug/endogenous-outreach/trigger` (or wait for organic send)
2. Confirm Hub bubble has “why I spoke”
3. Confirm Postgres `chat_history_log.client_meta->'outreach_provenance'` and `endogenous_outreach_decisions.result_json->'provenance'` for that corr
4. Ask “where did that come from?” → answer cites stored prompt, not collapse-mirror

Hard-reload the Hub browser tab so `app.js` picks up the UI.

## Risks / concerns

- Severity: low
- Concern: Compose config check and live acceptance smoke are UNVERIFIED in this session (missing worktree `.env`; no hub rebuild run here).
- Mitigation: restart commands above; smoke checklist is explicit.
- Severity: low
- Concern: Page reload drops the collapsible panel for historical outreach until a history-rehydrate path forwards `client_meta.outreach_provenance`.
- Mitigation: DB still holds the capsule for follow-up injection; UI gap documented.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2235
