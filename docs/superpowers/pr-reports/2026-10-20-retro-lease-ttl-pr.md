## Summary

- A controller lease survives 2 s without input (was 250 ms after each input, 1 s after a grant).
- One constant, `LEASE_TTL`, covers both the grant and every input.
- Regression test plus README/runbook contract updates.

## Outcome moved

After #2772 was deployed, the operator still could not play. Evidence from session 5234d2af: on each of 3 grants, input seq 1 arrived about 500 ms after the grant, then no second input arrived, and `no_controller` followed 250 ms later. The browser's 50 ms repeat timer ran at about 1 Hz (background-style timer throttling in the Hub page).

## Current architecture

`Arbiter` expired a lease 250 ms after its last input. That assumed the browser keeps a 50 ms heartbeat.

## Architecture touched

`services/orion-retro/app/control.py` only. No schema, bus, env or API shape change.

## Files changed

- `services/orion-retro/app/control.py`: `LEASE_TTL = 2.0` replaces `GRANT_GRACE` and the 0.25 s input expiry.
- `services/orion-retro/tests/test_authority.py`: expiry test parameterized on `LEASE_TTL`; new throttled-heartbeat regression test.
- `services/orion-retro/evals/container_smoke.py`: expiry wait reads `LEASE_TTL` instead of a hard-coded 0.4 s.
- `services/orion-retro/README.md`, `docs/runbooks/retro-browser.md`: contract text.

## Schema / bus / API changes

- Behavior changed: a lease expires after 2 s of silence. Button changes are still sent at once. Socket close, blur and a hidden tab still release control immediately.

## Env/config changes

None.

## Tests run

```text
services/orion-retro/tests: 34 passed; new test fails with LEASE_TTL=.25
```

## Evals run

```text
retro_browser.py (worktree image, Pad Lab): delay 0 PASS; delay 350: 2 PASS, 1 FAIL
(the fail was at mjs:69, the post-heal 'running' check; it did not reproduce
on 2 reruns, so it looks like a timing race in the eval, UNVERIFIED)
```

## Docker/build/smoke checks

```text
docker build orion-retro (worktree): OK
container_smoke.py --image orion-retro-eval:ttl: PASS (expiry wait now LEASE_TTL+.15, read from app/control.py)
```

## Review findings fixed

- Self-review: worker `expired_in_queue` (0.25 s queue age) is unchanged and correct. Takeover still emits a neutral frame. `port_owned` now holds for up to 2 s after a silent window. Worst case: a browser that freezes with its socket open holds its last buttons for up to 2 s.

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build
```

## Risks / concerns

- Severity: low. Concern: the root cause of timer throttling on the operator browser (Hub-tab iframe) is UNVERIFIED. This fix tolerates it. A Web Worker input timer would remove the dependency.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
