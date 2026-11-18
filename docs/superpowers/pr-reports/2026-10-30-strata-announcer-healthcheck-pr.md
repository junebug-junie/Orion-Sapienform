## Summary

- `atlas-chat-announce` (the Strata chat lane's announcer) read **unhealthy** live on circe after the 2026-10-30 cutover.
- Cause: it inherits the llama.cpp base image's healthcheck, which calls llama-server on `localhost:8080`. Announce-only mode never starts llama-server.
- Fix: the announcer is healthy when the engine it announces answers. The new check curls `http://atlas-chat:8080/health` over app-net. Strata answers 403 for unknown Host names, so the check sends `Host: 127.0.0.1:8080`.

## Outcome moved

The announcer's health becomes a real signal, "the chat engine is reachable", instead of a permanent false alarm.

## Current architecture

PR #3016. `atlas-chat` is the Strata engine; `atlas-chat-announce` is orion-llamacpp-host in announce-only mode.

## Architecture touched

`services/orion-llamacpp-host/docker-compose.atlas-workers.yml` (the healthcheck only).

## Files changed

- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`: healthcheck on `atlas-chat-announce`.

## Schema / bus / API changes

- Added / Removed / Renamed / Behavior changed: none (healthcheck only).
- Compatibility notes: none.

## Env/config changes

- None.

## Tests run

```text
docker compose config: healthcheck renders as intended
tests/test_compose_logging_journald.py  2 passed
llamacpp-host test_worker_announce + test_prism_seat  23 passed
```

## Evals run

```text
n/a (healthcheck only)
```

## Docker/build/smoke checks

```text
Live, inside the running announcer on circe:
  curl -fsS -H 'Host: 127.0.0.1:8080' http://atlas-chat:8080/health
  -> {"status": "ok", "max_context": 262144, "loaded": true, "service": "strata"}
  without the Host header -> HTTP 403
```

## Review findings fixed

- Trivial compose fix; no separate review.

## Restart required

```bash
# circe, primary main after merge
scripts/safe_docker_build.sh orion-llamacpp-host --compose-file docker-compose.atlas-workers.yml up -d atlas-chat-announce
```

## Risks / concerns

- Severity: low. Concern: the check depends on Strata's Host rule. Mitigation: the header names the engine as local, which Strata always accepts.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3017

🤖 Generated with [Claude Code](https://claude.com/claude-code)
