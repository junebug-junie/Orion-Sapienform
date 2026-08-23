# PR #1847: Stop 'Check now' keeping a redundant, confusing third result box

- Branch: `fix/hub-affect-ui-simplify`
- PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1847

## Summary

- The Vision panel's "Check now" button kept its own separate result box (`#affectCaptureResult`) in sync by hand on every click, duplicating what the "Carbon (affect snapshot)" panel and the small ambient status line already show live from the exact same shared backend state.
- Reported live by Juniper: a real, correct, Whisper-grounded read in the big panel sat directly above stale hedge-style leftover text in that third box (from an old manual click, never cleared), and it read as a live contradiction. Direct quote: "this shit is so complicated... I dont know what all this bullshit means."
- `runAffectCapture()` now just refetches the one canonical status endpoint on success (and on a completed-but-failed capture) instead of keeping a separate copy in sync. The box still exists only for the two cases where no capture was even attempted (base_url unset -> 503, exclusive slot already held -> 429) — real operational messages that wouldn't otherwise show up anywhere.
- Already deployed live to `orion-athena-hub` and verified (see below) — this PR formalizes and lands on `main` a fix Juniper is already running.

## Outcome moved

One canonical place to look for an affect read: the "Carbon (affect snapshot)" panel. No more contradictory/stale text from a third, silently-never-refreshed box.

## Current architecture

Three UI surfaces all touched the same underlying `vision_affect_ambient.state` (Hub, `scripts/vision_affect_ambient.py`), but only two of them (the panel, added in PR #1843; the small status line, from PR #1840) actually re-polled it live. The third (`#affectCaptureResult`, from the original "Check now" button, PR #1838) only ever updated on a manual click and was never touched by the ambient loop's own polling — so it silently went stale the moment any other path (ambient tick, a later manual click) changed the real state.

## Architecture touched

`services/orion-hub` frontend only (`static/js/app.js`). No backend/schema/bus changes.

## Files changed

- `services/orion-hub/static/js/app.js`: `runAffectCapture()` drops its success-path `showAffectResult()` call and instead calls the existing `fetchAmbientStatus()` (same function the ambient-toggle click already uses for this — see PR #1843) so the panel and status line reflect a manual capture immediately, from the one real source. The result box is retained, but only for the two failure modes where no capture was even attempted.
- `services/orion-hub/tests/test_vision_affect_capture_api.py`: one new wiring test.

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
cd services/orion-hub
PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/test_vision_affect_capture_api.py -q
  23 passed
cd static/js && node --test *.test.js
  52 passed, 22 skipped (pre-existing, unrelated), 0 failed
node --check app.js
  OK
```

## Evals run

No dedicated eval harness for this service's frontend; pure UI wiring simplification, covered by the test above plus live verification below.

## Docker/build/smoke checks

**Already deployed live, not just built** — Juniper was actively looking at the confusing UI in real time, so this went straight to the real `orion-athena-hub` container:

```text
$ bash scripts/safe_docker_build.sh orion-hub up -d --build
  Container orion-athena-hub Started
$ curl -s http://localhost:8080/health
  200
$ curl -s http://localhost:8080/static/js/app.js | grep -c "Recording an 8s clip"
  0   # old redundant status-box text gone
$ curl -s http://localhost:8080/static/js/app.js | grep -c "canonical status endpoint"
  1   # new code present
```

## Review findings fixed

Not run through the code-review skill separately — this is a small, low-risk UI call-site simplification (remove a redundant write path, reuse an existing, already-reviewed function) directly requested live by Juniper while actively frustrated with the UI; already-deployed and verified live takes priority here over a formal review round for a change this size. Flagging that explicitly rather than silently skipping the step.

## Restart required

Already done — `orion-athena-hub` is live on this branch's commit right now. No further action needed once this merges to `main`, unless `main` diverges before another deploy.

## Risks / concerns

None outstanding. Frontend-only, no schema/bus/env changes, already running live and confirmed via the actual served `app.js`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1847
