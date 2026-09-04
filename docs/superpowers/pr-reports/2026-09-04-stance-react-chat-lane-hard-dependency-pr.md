# stance_react's hardcoded chat-lane dependency (Flash-Next test fallout)

Branch: `chore/qwen3.8-flash-next-iq4xs-test`
Status: **DONE_WITH_CONCERNS** (docs + operator workaround; no code fix)

## Summary

- Every Hub turn -- regardless of which Mode/Compute the user picks --
  runs an internal `stance_react` step that is hardcoded to the "chat"
  gateway route (`services/orion-cortex-exec/app/executor.py`'s
  `_default_llm_route_for_step`). It has no fallback and no health check.
- Evicting circe:8011 (the chat/harness 35B) to free GPUs 0+1+3 for the
  Qwen3.8-Flash-Next IQ4_XS live test (`config/llm_profiles.yaml`'s
  `qwen3.8-flash-next-udiq4xs-3xv100-32gb-circe-flagship`) took that route
  down, which broke `stance_react` on every turn -- Agent mode, Orion mode,
  all of them.
- Live-confirmed failure: gateway logs `llamacpp error: [Errno 111]
  Connection refused` for `route=chat served_by=circe-worker-1`; orion-thought's
  `bus_listener.py` surfaces that as `"stance_react exec result missing
  thought payload"`, shown to the user as a deferred turn.
- Applied Juniper's chosen workaround (operator config only, no code
  change): `services/orion-llm-gateway/.env`'s `LLM_GATEWAY_ROUTE_TABLE_JSON`
  temporarily points `chat` and `harness` at circe:8015 (the URL `agent`
  already uses) instead of the dead `:8011`, so `stance_react` gets a real
  answer -- from the lite model, not the 35B -- instead of a connection error.
- Recorded both the finding and the workaround in-repo so this isn't
  tribal/verbal knowledge: a docstring addition next to the 2026-08-20
  incident this pattern already documents, and a comment in `.env_example`
  next to the route table.

## Outcome moved

Hub turns work again during the live-model comparison (verified via `GET
/routes` once the restart below is applied). The underlying fragility --
`stance_react` cannot tell "chat" is down, and fails deterministically
rather than degrading -- is now documented in two places instead of
living only in this session's chat log.

## Current architecture

- `orion-llm-gateway`'s route table is one JSON blob
  (`LLM_GATEWAY_ROUTE_TABLE_JSON`) read at container start; each logical
  route (`chat`, `agent`, `quick`, `metacog`, `harness`, ...) maps to a
  URL/`served_by`/`backend`.
- `services/orion-cortex-exec/app/executor.py`'s
  `_default_llm_route_for_step` picks the route per verb. `stance_react`
  -> `"chat"` unconditionally; this mapping predates and is unrelated to
  which Mode/Compute lane the Hub UI is set to.
- No component in this path checks whether the resolved route's worker is
  actually up before dispatching -- `GET /routes`' health cache exists
  (`services/orion-llm-gateway/app/route_catalog.py`) but nothing in the
  stance_react dispatch path consults it first.

## Architecture touched

- `services/orion-llm-gateway/.env` (local, not committed): route table
  values changed, not keys. `.env.bak-prechatreroute-1788564376` holds the
  pre-change copy for revert.
- `services/orion-llm-gateway/.env_example`: comment only, documenting the
  fragile point and the workaround. Example values unchanged (the standing
  default is still `:8011`; this is a documented *temporary* override, not
  a new default).
- `services/orion-cortex-exec/app/executor.py`: comment only, extending the
  existing `_default_llm_route_for_step` docstring with this second
  occurrence.

## Files changed

- `services/orion-cortex-exec/app/executor.py`: documents the 2026-09-04
  recurrence of the "chat route down -> every stance_react call fails"
  failure shape, and names what a real fix would need (route-health check
  or a distinct failure mode).
- `services/orion-llm-gateway/.env_example`: documents the fragile
  dependency and the temporary-override pattern next to the route table.
- `docs/superpowers/pr-reports/2026-09-04-stance-react-chat-lane-hard-dependency-pr.md`:
  this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: none in code. Live routing behavior changes only via
  the operator's own `.env` (not part of this diff).
- Compatibility notes: none.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes (comment only, values unchanged)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not applicable -- no key added/removed/renamed, only a documented live-test value override on the operator's own `services/orion-llm-gateway/.env`
- skipped keys requiring operator action: none

## Tests run

```text
None -- comment-only changes, no executable logic touched.
```

## Evals run

```text
Not applicable to this patch.
```

## Docker/build/smoke checks

```text
Dry-run compose diff (this worktree, PROJECT=orion forced to match the live
container name):
  PROJECT=orion docker compose --env-file .env --env-file services/orion-llm-gateway/.env \
    -f services/orion-llm-gateway/docker-compose.yml config
Confirmed the only rendered difference vs the live container's actual
environment is the intended LLM_GATEWAY_ROUTE_TABLE_JSON chat/harness url
change (8011 -> 8015). No other drift.

Restart itself NOT run by the agent -- blocked by the permission classifier
as a live production container recreate. See "Restart required" below.
```

## Review findings fixed

- Not run -- comment/docs-only change, no executable logic to review.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform/.claude/worktrees/feat+qwen3.8-flash-next-model-card
PROJECT=orion scripts/safe_docker_build.sh orion-llm-gateway up -d
```

`PROJECT=orion` must be passed explicitly: the checked-in root `.env`
currently says `PROJECT=orion-athena`, but the live container is named
`orion-llm-gateway` (older naming). Omitting the override would create a
second, differently-named gateway instead of updating the real one --
that naming drift is a separate, pre-existing loose end, not something
this patch fixes.

After restart, verify:

```bash
curl -fsS http://localhost:8210/routes   # chat + harness should report status "up"
```

**Revert once circe:8011 (the 35B) is back up:** restore
`services/orion-llm-gateway/.env` from
`.env.bak-prechatreroute-1788564376` (or just change `chat`'s and
`harness`'s `url` back to `http://100.112.254.99:8011`), then re-run the
same restart command.

## Risks / concerns

- Severity: should
- Concern: `stance_react` still has no route-health check or fallback --
  this exact failure shape will recur any time the chat lane goes down for
  any reason, test or real outage.
- Mitigation: none shipped in this patch (docs-only, per explicit
  instruction not to make code changes without approval). Follow-up: either
  a pre-dispatch health check for the resolved route, or a distinct
  "chat lane unavailable" turn-level failure the UI can show instead of the
  generic "missing thought payload" deferred-turn text.
- Severity: note
- Concern: `PROJECT=orion` vs `orion-athena` drift on the root `.env` --
  unrelated to this task, discovered while checking this deploy is safe.
- Mitigation: flagged here for Juniper; not fixed in this patch.

## PR link

<pending -- opened after push, see below>
