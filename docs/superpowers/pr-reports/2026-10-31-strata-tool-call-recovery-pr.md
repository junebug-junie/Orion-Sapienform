## Summary

- Turn on Strata's `tool_call_recovery` for both Strata lanes (`strata/chat-defaults.json`, `strata/agent-defaults.json`).
- In our version, Strata 0.2.7 (engine 0.1.52), it is opt-in and **off** (`serve/server.py`: `self.tool_call_recovery = False`). A tool call the model writes as `<tool_call><function=NAME><parameter=...>` is left as plain text.

## Outcome moved

- **Live evidence (2026-10-31).** Since chat moved to Strata (2026-10-30 23:38Z), 6 durable runs finished on chat: 5 completed and 1 failed. The failure was `reading-75e32b76`, `turn_error: Could not parse JSON object from LLM text: '<tool_call>\n<function=Bash>\n<parameter=command>\nls -...'`.
- **Before the switch:** 0 of the 60 latest runs failed this way.
- **Upstream measurement** (Strata DETAILS.md, `tool_call_recovery`): 98.3% of 1,462 agent turns read as intended with it on, 92.9% with it off. Only a declared tool's call is recovered.

## Current architecture

Each Strata lane's run config gets Orion's defaults merged once by `scripts/strata_apply_defaults.py`. The keys survive setup reruns.

## Architecture touched

`services/orion-llamacpp-host/strata/*.json`, `services/orion-llamacpp-host/tests/test_strata_apply_defaults.py`.

## Files changed

- `services/orion-llamacpp-host/strata/chat-defaults.json`: `"tool_call_recovery": true`.
- `services/orion-llamacpp-host/strata/agent-defaults.json`: `"tool_call_recovery": true`.
- `services/orion-llamacpp-host/tests/test_strata_apply_defaults.py`: both lanes keep it on.

## Schema / bus / API changes

- None.

## Env/config changes

- None (Strata run config only).

## Tests run

```text
services/orion-llamacpp-host/tests/test_strata_apply_defaults.py  6 passed
```

## Evals run

```text
Live check after apply: the failure count for '<tool_call>' parse errors on chat/agent runs should go to 0. UNVERIFIED until applied.
```

## Docker/build/smoke checks

```text
Applied on circe by merging the defaults into each lane's run config and restarting each engine (README "Chat lane on Strata").
```

## Review findings fixed

- Config-only change with live evidence and the upstream measurement; no separate review.

## Restart required

```bash
# circe, primary main after merge: merge defaults into both run configs, then restart each engine
set -a; . services/orion-llamacpp-host/.env; set +a
for lane in chat agent; do d=$([ $lane = chat ] && echo "$ATLAS_CHAT_STRATA_DATA_DIR" || echo "$ATLAS_AGENT_STRATA_DATA_DIR"); docker run --rm --entrypoint /opt/strata/terminal/python/bin/python3 -v "$d:/data" -v "$PWD/services/orion-llamacpp-host:/svc:ro" "$ATLAS_CHAT_STRATA_IMAGE" /svc/scripts/strata_apply_defaults.py /data/install/strata-iq3_s.json /svc/strata/$lane-defaults.json; done
docker restart orion-circe-atlas-strata-agent orion-circe-atlas-strata-chat
```

## Risks / concerns

- Severity: low. Concern: recovery can read a declared tool's call out of text that was not meant as one. Mitigation: only declared tools are recovered; upstream made it the default in 0.2.8.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3021

🤖 Generated with [Claude Code](https://claude.com/claude-code)
