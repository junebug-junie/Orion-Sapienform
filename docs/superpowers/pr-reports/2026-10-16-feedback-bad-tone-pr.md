## Summary

- Thumbs down on a chat turn now offers **"Bad tone"** (`bad_tone`). It pairs with thumbs up "Good tone" (`good_tone`).
- It replaces "Incorrect tone" (`incorrect_tone`). That option was never used: 0 rows in live `chat_response_feedback` (checked 2026-10-16).
- New test: it fails if the Hub's fallback option list in `app.js` drifts from the schema. Without it, a tag the modal offers can be rejected on submit.
- Fixed a stale UI test assertion (`text` -> `displayText`). It was already failing on main.
- New CI workflow `chat-response-feedback.yml`. Before this, no workflow ran any feedback test.

## Outcome moved

Juniper can tag generic LLM voice per turn. `good_tone` and `bad_tone` become the vibe-check labels for the personality graph's tone dials (#2726).

## Current architecture

Feedback categories live in `orion/schemas/chat_response_feedback.py`. The Hub serves them from `/api/chat/response-feedback/options` and keeps a hand-copied fallback list in `app.js`. The Hub validates submissions (`api_routes.py:2543`). The sql-writer validates them again before writing (`worker.py:505/581`). Nothing tested that the fallback matched the schema.

## Architecture touched

Shared feedback schema (category list), Hub `app.js` fallback, tests, one CI workflow.

## Files changed

- `orion/schemas/chat_response_feedback.py`: `incorrect_tone` -> `bad_tone`, with a dated comment
- `services/orion-hub/static/js/app.js`: same change in the fallback list
- `tests/test_chat_response_feedback_schema.py`: `bad_tone` is accepted for thumbs down, rejected for thumbs up; `incorrect_tone` is rejected
- `services/orion-hub/tests/test_response_feedback_ui.py`: fallback-matches-schema test; stale assertion fixed
- `.github/workflows/chat-response-feedback.yml`: runs both tests

## Schema / bus / API changes

- Added: category `bad_tone` (thumbs down)
- Removed: category `incorrect_tone` (0 historical rows)
- Renamed: none
- Behavior changed: a submission carrying `incorrect_tone` now gets a 422
- Compatibility notes: the Hub and the sql-writer both validate. Rebuild the sql-writer first, or the first `bad_tone` rows are dropped by the old writer. A modal already open in a browser from before the deploy that sends `incorrect_tone` gets a visible 422. Retry after a page reload.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest tests/test_chat_response_feedback_schema.py tests/test_artifact_rating.py tests/test_sql_writer_chat_feedback.py services/orion-hub/tests/test_response_feedback_ui.py -> 51 passed
pytest services/orion-hub/tests/test_response_feedback_api.py (run alone; a `scripts` package clash breaks mixed collection, which predates this change) -> 5 passed
Clean venv with only the workflow's deps (pytest, pydantic==2.11.5, PyYAML) -> 11 passed
Mutation check: putting incorrect_tone back in app.js makes the drift test fail
```

## Evals run

```text
None. This is a label change. The tags themselves feed the #2726 P0 tone baseline.
```

## Docker/build/smoke checks

```text
Not run from the worktree. Deploy happens from main after merge (below).
```

## Review findings fixed

- Finding: deploy skew between the Hub and the sql-writer could drop the first `bad_tone` rows.
  - Fix: the deploy order is written into this report and the restart commands.
  - Evidence: both validate against the shared schema (`api_routes.py:2543`, `sql-writer worker.py:505/581`).
- Finding: no other consumer of `incorrect_tone` (graphify plus rg across the repo). No change needed.

## Restart required

From the clean primary `main` checkout after merge, sql-writer first:

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: an old open modal sends `incorrect_tone` and gets a 422. Mitigation: visible error, reload fixes it, 0 historical use.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2727

🤖 Generated with [Claude Code](https://claude.com/claude-code)
