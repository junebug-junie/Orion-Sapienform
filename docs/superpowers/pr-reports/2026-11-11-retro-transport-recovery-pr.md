## Summary

- Recover a dropped Retro media socket without closing or reacquiring the controller. The relay retains one latest complete video/PCM packet per viewer and closes a stalled send after 750 ms.
- Send neutral human input after one second without a presented picture; close the controller/co-play sponsor after three seconds. Keep retries bounded.
- Preserve explicit Pause/Stop, another controller's ownership, and session/transport generations. Pause/Stop close local controls before waiting for HTTP; a deliberate game click explicitly resumes.
- Add conditional native resume so a delayed recovery request cannot undo a newer Pause. Older native services are detected and handled conservatively.
- Log relay-close reasons under Uvicorn and add deterministic regressions, browser failure injection, and CI coverage.

## Outcome moved

Previously, a media failure tore down both sockets and could strand play during reconnect. The new path reconnects media alone, releases held input when the picture is unavailable, and preserves controller ownership and deliberate pause intent.

## Current architecture

Tailscale routes `/api/retro` to the dedicated Hub gateway on loopback 8092. Hub main also mounts the same router. Both relay over a private Unix socket to Retro, whose spawned native worker owns controller leases and emulator frames. The browser owns input/media connections.

## Architecture touched

Only Retro's relay/page and its native resume contract change. Chat orchestration, other Hub routes, planner calls, memory export, Redis channels, credentials, native lease expiry, and service resource limits retain their existing contracts.

## Files changed

- `services/orion-hub/scripts/retro_transport.py`, `retro_routes.py`: bounded media relay, ordered input replies, cancellation cleanup, close diagnostics.
- `services/orion-hub/static/js/retro-panel.js`, `templates/retro.html`: independent media recovery, stale-picture input handling, intent/session/socket guards, asset version bump.
- `services/orion-retro/app/{contracts,main,worker}.py`: validated conditional resume, checked atomically by the worker.
- Hub/native regression tests and browser/native evals: reproduce ownership, pause, stale-callback, slow-viewer and recovery cases.
- `.github/workflows/retro-preflight.yml`: run client regressions and browser media-failure/co-play eval.
- Service READMEs and `docs/runbooks/retro-browser.md`: compatibility, failure behavior, deployment and rollback.

## Schema / bus / API changes

- Added: native `/health/ready` advertises `capabilities.conditional_resume=true`; `POST /sessions/{id}/resume` optionally accepts `expected_pause_reason` (`disconnect|no_controller|deadline_overrun`).
- Removed/Renamed: none.
- Behavior changed: a mismatched current pause reason returns 409; invalid conditional-resume fields return 422. Empty-body/`{}` explicit resumes retain their behavior.
- Compatibility: the page only sends conditional automatic resumes when the capability is advertised. Refresh/roll back the page before downgrading a native service that advertised it. No bus events or shared schema registry changes.

## Env/config changes

No env keys, templates, dependencies, ports, mounts or resource limits changed. No template sync was needed. Disposable build-only env files are ignored; no credentials/config are committed. The local wrapper build used its documented env-drift override because its parity check resolves to the primary checkout, which has no Retro env on this host. No live deployment used that override.

## Tests run

- Native/preflight: **167 passed**.
- Focused Hub + memory export + leisure startup + UI layout + caller routing: **336 passed, 2 optional skips**.
- Relay regressions include a genuinely blocked media write while unrelated HTTP GET/POST routes succeed.
- Actual page-asset tests cover delayed grants, another owner, explicit Pause/Stop, session replacement, old native capability, hung HTTP, and stale decoder errors.
- JavaScript syntax and diff whitespace checks passed.

## Evals run

- Real-browser fixture: **PASS**, including dropped media, silent pictures, co-play handoff/renewal/take-back.
- Disposable native container smoke: **PASS**, including conditional-resume/Pause race, controller expiry, child failure, quota failure and denied egress.
- Native browser/preflight initially failed the deliberate click-after-Pause scenario. That exposed a real regression in how explicit clicks were classified; the fix and regression test are included. Both normal and 350 ms delayed-ack native browser reruns now **PASS**. Final CI checks are tracked on this PR and must pass before merge.

## Docker/build/smoke checks

Pinned FCEUmm Retro image built successfully through `safe_docker_build.sh`. Native smoke passed against the built image. No live service was restarted. Athena play after deployment remains **UNVERIFIED**; this patch does not repair Carbon's network jitter/loss.

## Review findings fixed

Independent adversarial review found and fixed:

- A delayed automatic resume could override a newer Pause → atomic worker comparison + capability negotiation.
- Hung Pause/Stop could leave input active and accept a late grant → immediate local cleanup + intent guard.
- INFO diagnostics were disabled in the gateway → configured Uvicorn logger + fresh-process test.
- An old asynchronous decode failure could close a new transport → per-packet epoch/socket checks and continued draining.

No material review findings remain in the reviewed transport changes. Native-browser reruns pass; the final commit's CI checks are linked on this PR.

## Restart required

After merge, save the current game, then deploy from clean primary main synchronized with origin/main on Athena:

```sh
scripts/safe_docker_build.sh orion-retro up -d --build retro
scripts/safe_docker_build.sh orion-hub up -d --build hub-retro-gateway hub-app
```

Recreating Retro ends the active session. The Hub command also restarts chat because both processes import the shared router. Reload the page afterward. Static assets are host-mounted, so capability negotiation protects the interval between pulling the page and rebuilding native Retro.

## Risks / concerns

- Severe network stalls still pause play by design; input cannot remain held indefinitely without a usable picture.
- A media packet drop also drops its PCM; existing sample-discontinuity handling flushes audio.
- Native capability is cached for the page lifetime; refresh/roll back frontend before a native downgrade.
- Runtime acceptance on Athena requires deployment and actual play; production has not been changed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3196
