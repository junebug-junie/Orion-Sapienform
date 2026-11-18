## Summary

- One-button play: New game connects and lands at the title screen. Secondary controls sit under More. A lost transport reconnects itself (at most 3 times per 10 s).
- A freshly granted controller lease now waits 1 s for its first input (was 250 ms).
- Later inputs keep the 250 ms expiry.
- Regression test, plus README/runbook updates.

## Outcome moved

After #2771 was deployed, every Connect / resume from the operator laptop lost control with zero applied inputs (`control` → `no_controller` about 0.27 s later, three times in a row). The lease clock started before the grant reply reached the browser.

## Current architecture

`Arbiter.control` set `expires=now+.25` at the grant. The browser can send only after the HTTP grant reply crosses Hub and Tailscale.

## Architecture touched

`services/orion-retro/app/control.py`, Hub `retro.html` and `retro-panel.js`. No schema, bus, env or API shape change.

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
services/orion-hub/tests/retro: 1 passed
```

## Evals run

```text
retro_browser.py (worktree image, Pad Lab): 4/4 PASS, delay 0 x2 and 350 x2
```

## Docker/build/smoke checks

```text
Live smoke after deploy: watch evidence for control -> applied rows (no no_controller).
```

## Review findings fixed

- Finding: self-heal resumed a Pause made by another window or the CLI (`control_revoked` / `session_not_running` were treated as lease loss).
  - Fix: only transport losses heal. A server-side pause clears `wanted`.
- Finding: a Pause during an in-flight connect was undone (connect set `wanted=true` after returning).
  - Fix: an `intent` counter. Only the latest connect may set `wanted`.
- Finding: New game raced a pending heal timer, which could connect to the old session.
  - Fix: New bumps `intent`, and the heal timer aborts when `intent` has changed.
- Minor fixed: input-socket close now heals like media close. Visibility plus focus no longer spends 2 heals. The port_owned message names the More toggle. Pause/Stop always release, even when their API call fails.
- Evidence: `retro_browser.py`, 4/4 PASS (2 with --ack-delay-ms 350; play resumes with no click).
- Not changed: a silent fresh lease blocks another window's `acquire` on that port for 1 s (takeover still works).

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: browser main-thread jank in the Hub tab is still UNVERIFIED and may still cause mid-play expiry. Mitigation: monitor evidence; next step is moving input/decode to a Web Worker.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
