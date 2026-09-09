# PR report — a human-visible panel for Orion's self-definition

**Branch:** `feat/self-panel`
**Date:** 2026-09-09

## Summary

- Juniper asked "where do I see this?" after the self-inquiry line (#2158/#2165/#2169) started producing real, revised self-definitions with nowhere to look at them except raw Postgres/graph queries.
- New "Self" section on the existing Curiosity Atlas page (`/curiosity`): current definition + evidence, version history, latest self-sense eval scores, recent self-inquiry journal entries, and an "Ask self-inquiry now" button — the self-inquiry line's run-now endpoint has existed since #2158 with no UI trigger anywhere.
- New Postgres-only reader (`orion/curiosity/self_panel.py`) that reads `self_concept_history` and `journal_entries` directly, deliberately bypassing the existing `:TurnOutcome`-keyed run list — that list only shows runs that wrote an outcome node, which is optional, and the run that produced the *first* self-definition never wrote one.
- Reviewed (`/code-review low`) before commit; the standout finding — the panel would have ordered "current" by `version` while every other reader in the codebase treats `version` as non-authoritative and orders by `created_at` — is fixed.

## Outcome moved

Before: the only way to see Orion's current self-definition was a direct SQL query or `GRAPH.RO_QUERY`. After: a page Juniper already has open.

## Current architecture (before)

- `services/orion-hub/scripts/curiosity_routes.py`'s `/api/atlas` builds one payload from `orion/curiosity/atlas.py`'s `read_atlas()` — graph-only, keyed on `:TurnOutcome` run nodes.
- `self_concept_history` and `self_sense_eval_log` (Postgres) had zero UI readers.
- `POST /curiosity/api/self-inquiry/run-now` existed with no button anywhere.

## Architecture touched

- New Postgres reader module (`orion/curiosity/self_panel.py`), wired into the existing single-read `/api/atlas` endpoint as `payload["self"]` — not a second endpoint, so the panels cannot disagree about the same run (the endpoint's own stated design principle).
- Template: `services/orion-hub/templates/curiosity_atlas.html` — new "Self" section, `renderSelf()`, a second run-now button.
- `pyproject.toml` — `orion/curiosity/tests` added to `testpaths` (review finding: the new test directory was invisible to a bare `pytest` run without this).

## Files changed

- `orion/curiosity/self_panel.py`: new. Dataclasses + `read_self_panel(pool)` + `to_payload(view)`.
- `orion/curiosity/tests/test_self_panel.py`: new, 11 tests.
- `services/orion-hub/tests/test_curiosity_self_panel_route.py`: new, 3 tests.
- `services/orion-hub/scripts/curiosity_routes.py`: `_read_self_panel_payload`, `_get_memory_pg_pool` (deduped from two inline copies).
- `services/orion-hub/templates/curiosity_atlas.html`: Self section, CSS, `renderSelf`, self-inquiry run-now handler.
- `orion/curiosity/README.md`: documents the panel.
- `pyproject.toml`: testpaths.

## Schema / bus / API changes

- Added: `GET /curiosity/api/atlas` response gains a `self` key. No new endpoint, no new channel, no schema change.
- Compatibility notes: purely additive to an existing JSON payload; nothing reads that key today except the new template code.

## Env/config changes

None.

## Tests run

```text
orion/curiosity/tests/test_self_panel.py                    11 passed
services/orion-hub/tests/test_curiosity_self_panel_route.py   3 passed
services/orion-hub/tests/test_curiosity_self_inquiry.py     167 passed (full curiosity suite, no regressions)
python -m pytest --collect-only -q (repo root)               confirms test_self_panel.py now collected
```

## Evals run

None applicable — this is a read-only dashboard panel.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub up -d --build
curl -s http://localhost:8080/curiosity/api/atlas | jq .self
# -> available: true, current.version: 3, current.created_at matches the live
#    self_concept_history row, history versions in created_at order [3, 2, 1]
```

**The visible panel itself could not be screenshotted from this worktree.** `services/orion-hub/docker-compose.yml` bind-mounts `templates/` from `${ORION_HOST_REPO_ROOT:-/mnt/scripts/Orion-Sapienform}` — the primary checkout — by design (this is the fix for the "worktree deploys can pin a worktree as production" incident: host mount sources must be absolute-rooted at the primary checkout, never a disposable worktree path). The backend route change is baked into the container image and is live now, verified against real data above; the HTML/JS only takes effect once this file exists at that path, i.e. after merge. No redeploy is needed post-merge — it is a live bind mount; a page refresh is enough.

## Review findings fixed

- Finding: the panel picked "current" via `ORDER BY version DESC`, while `self_study.py`'s version bump is a non-transactional `MAX+1` read, the table has no unique constraint on `(concept_id, version)`, and the felt-state lane that actually feeds every chat turn orders by `created_at`. A divergence between the two orderings would show Juniper a different "current" definition than the one Orion is using in chat.
  - Fix: `ORDER BY created_at DESC`, documented inline with the reasoning. Test asserts the SQL literally contains that clause and that `current` does not re-sort by version.
- Finding: the history query had no `LIMIT`, unlike every other query in the same function, and the front end rendered the whole list into the DOM on every 60s poll.
  - Fix: `_HISTORY_LIMIT = 50` (self-inquiry caps at 3 runs/day, so this covers several weeks).
- Finding: `orion/curiosity/tests/` was not in `pyproject.toml`'s `testpaths` allowlist — confirmed live that a bare `pytest` run from repo root collected zero tests from the new file.
  - Fix: added to `testpaths`. Reconfirmed collection.
- Finding: the `hub_main.app.state.memory_pg_pool` lookup was duplicated inline in two readers in the same file.
  - Fix: one `_get_memory_pg_pool()` helper, used by both. (Six other route files in this service have the same inline pattern independently of this PR — out of scope here, named so it isn't silently re-discovered.)

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Already done from this worktree and verified live above; nothing further needed after merge (bind-mounted template, live refresh).

## Risks / concerns

- Severity: low. The dedup helper (`_get_memory_pg_pool`) only covers this one file; five more route files in `services/orion-hub/scripts/` have the identical inline pattern, disclosed above as a named non-goal.
- Severity: low. Two durable runs were mid-retry (a pre-existing GPU-contention pattern, unrelated to this change — see the self-inquiry line's own incident history) when this was deployed; the Hub redeploy cost them one more retry cycle each, which the runner's resume mechanism already recovers from.

## PR link

(filled after `gh pr create`)
