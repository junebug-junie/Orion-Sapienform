## Summary

- A freshly granted controller lease now waits 1 s for its first input (was 250 ms).
- Later inputs keep the 250 ms expiry.
- Regression test, plus README/runbook updates.

## Outcome moved

After #2771 was deployed, every Connect / resume from the operator laptop lost control with zero applied inputs (`control` → `no_controller` about 0.27 s later, three times in a row). The lease clock started before the grant reply reached the browser.

## Current architecture

`Arbiter.control` set `expires=now+.25` at the grant. The browser can send only after the HTTP grant reply crosses Hub and Tailscale.

## Architecture touched

`services/orion-retro/app/control.py` only. No schema, bus, env or API shape change.

## Files changed

- `services/orion-retro/app/control.py`: `GRANT_GRACE = 1.0` for fresh leases.
- `services/orion-retro/tests/test_authority.py`: regression test.
- `services/orion-retro/README.md`, `docs/runbooks/retro-browser.md`: contract text.

## Schema / bus / API changes

- Behavior changed: a fresh lease lasts 1 s without input. A granted lease holds the neutral mask, so no button can be held down.

## Env/config changes

None.

## Tests run

```text
services/orion-retro/tests: 34 passed; new test fails with GRANT_GRACE=.25
```

## Evals run

```text
Not re-run: one-constant change covered by the deterministic arbiter test.
```

## Docker/build/smoke checks

```text
Live smoke after deploy: watch evidence for control -> applied rows (no no_controller).
```

## Review findings fixed

- Self-review only (urgent: the operator could not start play). Risk surface: a silent client keeps a neutral lease 0.75 s longer.

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
```

## Risks / concerns

- Severity: low. Concern: browser main-thread jank in the Hub tab is still UNVERIFIED and may still cause mid-play expiry. Mitigation: monitor evidence; next step is moving input/decode to a Web Worker.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
