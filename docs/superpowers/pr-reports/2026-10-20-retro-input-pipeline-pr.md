## Summary

- Retro's input socket no longer waits for one input's game-frame confirmation before reading the next input.
- The browser treats `superseded` / `expired_in_queue` replies as normal, not as lost control.
- Command-timeout cleanup is shielded, so a closed socket cannot leave a session wedged.
- Two regression tests; the runbook records the cause and the trade-offs.

## Outcome moved

Live play paused mid-game every 10–25 s (`no_controller` exactly 0.25 s after the last applied input) and showed "Connection stalled". The cause: one slow confirmation (217 ms measured on the live path, with a perfect 50 ms sender) held newer inputs past the 250 ms controller lease.

## Current architecture

Browser → Hub WebSocket relay → Retro API input socket → worker queue → per-frame apply → ack. The API awaited each ack serially.

## Architecture touched

`services/orion-retro/app/main.py` (pipelined input tasks, serialized sends), `services/orion-retro/app/session.py` (shielded stop), `services/orion-hub/static/js/retro-panel.js` (benign rejections). No schema, bus, or env changes.

## Files changed

- `services/orion-retro/app/main.py`: inputs run as ordered tasks; a send lock; a send failure ends the loop.
- `services/orion-retro/app/session.py`: `asyncio.shield` around timeout cleanup.
- `services/orion-hub/static/js/retro-panel.js`: `superseded` / `expired_in_queue` are cleared from outstanding.
- `services/orion-retro/tests/test_authority.py`: two regression tests.
- `docs/runbooks/retro-browser.md`: cause, fix, trade-offs.

## Schema / bus / API changes

- Behavior changed: input acks can now arrive out of order, and older inputs on a port get `rejected: superseded`. The client matches acks by seq.

## Env/config changes

None.

## Tests run

```text
services/orion-retro/tests: 33 passed (new tests fail on old code)
services/orion-hub/tests/retro: 1 passed (run separately; combined run has a pre-existing `app` package collision, same on main)
```

## Evals run

```text
retro_browser.py on worktree image, --ack-delay-ms 350: Pad Lab PASS x2, Mario PASS x1
(one Mario run hit a headless screenshot timeout, a harness flake;
 one earlier delayed-ack Mario run failed with a blank screen, unexplained, then passed on rerun;
 Mario --ack-delay-ms 0 fails the audio-peak assertion on BOTH old and new images; game-specific)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-retro/Dockerfile -t orion-retro-eval:input-pipeline . : OK
```

## Review findings fixed

- Finding: cancelling an input task during timeout cleanup could leave `stopping=True` and the worker alive.
  - Fix: `asyncio.shield(self.stop(...))`.
  - Evidence: `test_cancelled_input_cannot_abandon_timeout_cleanup` fails without the shield and passes with it.
- Minor (documented, not changed): a stall longer than about 0.5 s can hit the shared 24-command cap. During a stall, a tap shorter than one frame collapses to its release.

## Restart required

```bash
scripts/safe_docker_build.sh orion-retro up -d --build && scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low. Concern: UNVERIFIED whether the user's laptop browser adds its own timer jitter. Mitigation: if stalls persist after deploy, inspect evidence gaps again; next suspect is main-thread frame decoding.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
