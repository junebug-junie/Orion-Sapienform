## Summary

- Fix a regression in #3196: a second click during a slow controller grant invalidated the successful response, leaving a server lease that the page never used.
- Bound controller acquisition to five seconds. Pause/Stop still interrupt it; late old responses cannot replace a newer takeover, and timeout recovery runs after cleanup.
- Bound the entire media path to three frames awaiting browser presentation. The previous one-packet application queue did not bound frames accepted into socket/proxy buffers.
- Negotiate presentation acknowledgments with the gateway; preserve legacy behavior when either side lacks the capability.
- Add delayed-control, repeated-click, out-of-order response, hung-request and presentation-credit regressions. Bump the page asset to v15.

## Outcome moved

Controls remain attached to a successful delayed acquisition. Media recovery skips to the latest native frame instead of draining a hidden backlog of old pictures. Actual improvement on Carbon remains **UNVERIFIED** until this patch is deployed and played.

## Current architecture

Hub's dedicated Retro gateway and Hub main share a private-socket relay. Native Retro owns the emulator and leases. The browser presents compressed video/PCM and sends controller input separately. #3196 bounded the application queue but did not bound transport buffers or protect the whole acquisition phase.

## Architecture touched

Hub Retro relay, page, tests and docs only. No native Retro code, planner, memory export, bus, database, chat orchestration, credentials or service config changes.

## Files changed

- `retro-panel.js`: scope/bound pending acquisition; negotiate and send presentation credits.
- `retro_transport.py`, `retro_routes.py`: advertise/validate credits and cap unpresented media.
- `retro.html`: asset v15.
- Retro client/relay tests and browser evals: reproduce both failure mechanisms and verify the negotiated path.
- Hub README and Retro runbook: protocol, compatibility and rollout.

## Schema / bus / API changes

- Added: proxied `GET /api/retro/health/ready` capability `retro_media_ack_v1`.
- Opt-in media query: `?flow=ack-v1`; browser sends cumulative `{"ack": N}` after presenting message N on that socket.
- Gateway allows at most three sent/unacknowledged frames and retains the latest packet while waiting. ACKs are validated and never forwarded into native controls.
- Removed/Renamed: none. Native media and bus schemas are unchanged.
- Compatibility: old pages/new gateways and new pages/old gateways use legacy media. Refresh the page before downgrading a gateway that advertised credits.

## Env/config changes

None. No env templates or dependencies changed; no env sync or operator key handling required.

## Tests run

- Hub Retro gate: **239 passed, 2 optional skips**.
- Actual page-asset regression suite: **18 scenarios passed**.
- Regression evidence: repeated-click case failed the deployed asset with changed intent/discarded grant; the corrected implementation preserves the grant.
- Relay test models an immediately accepting socket with a non-presenting viewer: exactly three frames are accepted; after cumulative ACK, the next packet is the latest.
- JS syntax and whitespace checks passed.

## Evals run

- Real native/browser: **PASS**, including 650 ms delayed control response plus repeated clicks.
- Real native/browser with 350 ms delayed input acknowledgments: **PASS**.
- Browser eval explicitly asserts negotiated ack-v1 sockets and presentation ACKs.
- Media failure/co-play browser fixture: **PASS**.

## Docker/build/smoke checks

Browser smokes use the existing pinned native container plus the changed Hub router/page, in an isolated disposable environment. No live deployment or native image change was performed.

## Review findings fixed

Independent review found and verified fixes for stale acquisitions overwriting newer takeovers, hung acquisition blocking recovery, and timeout retry occurring before connect cleanup. The presentation-credit protocol and cancellation/resource bounds were reviewed. No material review blocker remains.

## Restart required

After merge, from clean synchronized primary main on Athena:

```sh
scripts/safe_docker_build.sh orion-hub up -d --build hub-retro-gateway hub-app
```

Reload /retro afterward (v15). Native Retro needs no rebuild. This also restarts Hub chat because both Hub processes share the router.

## Risks / concerns

- This bounds buffering; it does not remove the measured Wi-Fi jitter/loss.
- Slow clients may see fewer frames and PCM gaps rather than a growing playback delay.
- A five-second acquisition timeout closes that connection; stale responses cannot restore it.
- User-device input-to-photon latency and deployment acceptance remain UNVERIFIED.
- CI status is tracked on this PR; all checks must pass before merge.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3197
