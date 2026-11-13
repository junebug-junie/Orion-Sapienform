## Summary

- Present decoded Retro video immediately and return presentation credit without waiting for scheduled audio.
- Bound queued PCM start to 50 ms ahead of the audio clock; preserve latest-packet buffering, the three-frame transport window, and controller safety.
- Ship asset v16, a failing-before PCM regression, and a native browser eval with delayed short timers.
- Isolate browser evals from the operator's physical gamepad, which was contaminating scripted input during this investigation.

## Outcome moved

Live read-only observation on Carbon measured arrival-to-canvas latency of 29.2 ms median / 63.8 ms p95 on v15, versus 2.7 ms / 4.9 ms with this asset. These are sequential observations on variable Wi-Fi, not a controlled network benchmark or input-to-photon measurement. The deterministic regression independently establishes removal of the audio wait.

The prior renderer awaited a 25–120 ms audio-start timer before drawing and returning `ack-v1` credit. That also stopped the decoder from handling its latest pending packet. A delayed timer therefore stalled both local presentation and the relay's three-frame window.

## Current architecture

- Service: Hub browser adapter; native Retro remains the emulator/controller authority.
- Entry points: `/retro`, `retro-panel.js`, `/api/retro` through Tailscale to the dedicated gateway on 8092.
- Config: Hub `app/settings.py`, `.env_example`, and `docker-compose.yml`. There is no root `services/orion-hub/settings.py`.
- Bus channels/schema registry: none in the touched media path; no cognition signal added.
- Docker: Hub binds its `static/` and `templates/` directories from primary main. Native Retro runs separately over a private Unix socket.
- Tests: `services/orion-hub/tests/retro/`; evals: `retro_browser.py/.mjs`, `retro_play_panel.py/.mjs`.
- Gap: presentation waited for audio even when a fresh picture was decoded; previous browser gates did not exercise late audio timers.

## Architecture touched

Browser rendering/audio scheduling and test harnesses only. The wire packet, relay credits, leases, reconnection rules, native worker, planner, memory, bus, and database contracts are unchanged.

## Files changed

- `services/orion-hub/static/js/retro-panel.js`: remove the audio wait; bound scheduled audio start.
- `services/orion-hub/templates/retro.html`: load v16.
- `services/orion-hub/tests/retro/{transport_client.cjs,test_play_panel_static.py}`: PCM timing regression and asset check.
- `services/orion-hub/evals/{retro_browser.mjs,retro_play_panel.mjs}`: late-timer native scenario, hardware input isolation, current asset check.
- Hub README, Retro runbook, and this report: behavior, evidence, rollout, limits.

## Schema / bus / API changes

- Added/Removed/Renamed: none.
- Behavior changed: video/credit no longer waits for PCM start; excessive queued audio is dropped sooner.
- Compatibility notes: existing native service and `ack-v1` gateway are compatible. Legacy media also benefits.

## Env/config changes

- Added/Removed/Renamed keys: none.
- `.env_example` updated: no.
- local `.env` sync: not required; no template changed.
- skipped keys requiring operator action: none.
- Operator-approved live action: reconnected Carbon's existing `bearpeople-5G-2` Wi-Fi profile. No saved settings, credentials, or service env changed.

## Tests run

```text
python3 -m pytest services/orion-hub/tests/retro -q
239 passed, 2 optional skips
node --test services/orion-hub/tests/retro/transport_client.cjs
19 scenarios pass; the added PCM scenario fails on v15 with "picture waited for audio".
python3 scripts/check_metric_lineage.py --gate
PASS
JavaScript syntax and git diff --check: PASS
```

## Evals run

```text
retro_browser.py --output /tmp/retro-immediate-native-browser-isolated: PASS (Chromium)
retro_browser.py --ack-delay-ms 350 --output /tmp/retro-immediate-native-delayed: PASS
retro_browser.py --output /tmp/retro-immediate-firefox-snap: PASS (Firefox)
retro_play_panel.py --output /tmp/retro-immediate-panel-isolated: PASS
```

The native browser runs include the late-short-timer presentation scenario, real
PCM/pixels, save/load, fullscreen, input, takeover, disconnect and reconnect.
The first native run exposed real gamepad input contaminating its scripted key masks;
the harness now substitutes empty gamepads except in its deliberate gamepad scenarios.

## Docker/build/smoke checks

Real native browser smokes use the existing pinned Retro image in disposable offline
containers; no native source/image changes are needed. Live observations acquire only
a media viewer and do not replace a session, acquire control, or send game input.

Athena's local live path delivered 752 media packets in 25 s (~30 fps), p95 inter-packet
gap 40.3 ms. Carbon's link had large timing swings and packet loss before any dependency
download. A later clean-browser sample presented 343 frames in 25 s before the renderer
change, versus 678 in 25 s with it. Network variation prevents attributing all that
frame-rate difference to this patch. Arrival-to-draw numbers above isolate client delay
more directly. A Wi-Fi reconnect reduced a router-ping sample mean from 138 ms to 44 ms,
but spikes remained; reconnecting alone did not solve the problem.

Local diagnostic artifacts: `/tmp/retro-carbon-before.json`,
`/tmp/retro-browser-before-clean.json`, `/tmp/retro-browser-after-reconnect.json`,
`/tmp/retro-browser-before-age.json`, `/tmp/retro-browser-after-age.json`.
Athena local capture: gateway container `/tmp/retro-athena-before.json`.
These contain timings/frame counters, not raw media or credentials.

## Review findings fixed

- Independent review-agent review found no actionable introduced defect in the renderer/credit change.
- Review limit: 50 ms is a scheduled PCM start bound, not measured sound at speakers.
  - Fix: document that distinction and the audio-drop tradeoff.
  - Evidence: this report, README, and runbook.

## Restart required

No service restart is required for these bind-mounted static/template changes.
After merge, update clean primary main on Athena, then reload the Retro page (v16).
Use the repository's normal post-pull deployment flow if it schedules a Hub rebuild.
Never copy a worktree asset into the running service or deploy from a feature branch.

## Risks / concerns

- Carbon's Wi-Fi still has jitter; this removes application-added delay, not network stalls.
- Audio/video output-device alignment and subjective play quality remain UNVERIFIED until operator play after rollout. More queued audio may be dropped after delivery bursts.
- No new telemetry is wired into a cognition pipeline. Diagnostic timings are test evidence only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3198
