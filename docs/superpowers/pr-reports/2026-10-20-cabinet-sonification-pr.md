# Cabinet sonification: live instrument

## Summary

- Add an explicit-Listen, read-only Cabinet instrument with magnetic pitch, LiDAR
  pulses and vibration onset paths. Magnetic is available; LiDAR/vibration are
  calibration-gated off, visibly, until controlled movement evidence exists.
- Reuse the existing poll, resolve per-channel source freshness, preserve valid
  zero, and suppress repeated/recovery percussion. No API or cognition writes.
- Add mute/solo, Stop, source/target inspection, bounded opt-in local trace capture,
  automatic shutdown, native Web Audio eval, real-data replay and CI provenance parity.

## Outcome moved

Cabinet measurements can now control bounded browser audio following an explicit
operator gesture. Missing/frozen inputs cannot borrow another Nano's timestamp.
User mute/solo leaves playback armed; Stop/hide/failure requires a new Listen.

## Current architecture

`orion-hub` FastAPI projects host Nano snapshots through existing Cabinet latest /
history APIs. `cabinet-sensors.js` owns polling in the Biometrics modal's Cabinet
subview. Shared `cabinet_snapshot_merge.py` selects newest non-stale channels and
`cabinet_sensors.py` extracts measurements. No initial sonification consumer.

## Architecture touched

One browser-local source adapter, pure mapper, onset state and dedicated audio
context; existing Cabinet poll/hide integration; existing Hub template. No new
service. TTS remains separate; no dependable public TTS lifecycle hook was found,
so manual Stop during TTS is documented and tested against an unrelated context.

## Files changed

- `services/orion-hub/static/js/cabinet-sonification.js`: pure mapping and bounded synthesis/control lifecycle.
- `services/orion-hub/static/js/cabinet-sonification.test.js`: mappings, freshness, onset and lifecycle regression tests.
- `services/orion-hub/static/js/cabinet-sensors.js`: forward poll success/failure and deactivation.
- `services/orion-hub/templates/index.html`: controls, inspector and versioned asset.
- `services/orion-hub/tests/test_cabinet_sensors_panel.py`: rendered controls/asset linkage.
- `services/orion-hub/tests/test_cabinet_sonification_provenance.py`: parity with shared merger/extractor on real and partial/tied-source fixtures.
- `services/orion-hub/evals/cabinet_sonification_{server.py,browser.cjs}`: isolated rendered Hub/native audio smoke.
- `services/orion-hub/evals/fixtures/cabinet-sonification-live.json`: sanitized 120-frame local measurement capture.
- `services/orion-hub/README.md`: controls, limitations, rollback, eval and listening procedure.
- `.github/workflows/orion-static-gates.yml`: source parity gate; existing Node discovery automatically includes new tests.

## Schema / bus / API changes

- Added/removed/renamed: none.
- Behavior changed: browser-only measurement-to-sound display.
- Compatibility: no server fields, bus shapes, registry, storage or firmware changes.

## Env/config changes

- Added/removed/renamed keys: none.
- `.env_example` updated: no.
- local `.env` sync: not required; templates unchanged. Ignored root/Hub env copies
  were copied from the primary checkout solely for the mandatory build wrapper.
- Skipped keys requiring operator action: none.

## Six-part metric quality gate

These are display transforms, not new independent cognition metrics.

1. **Provenance:** firmware `read_magnetic()` at
   `firmware/athena-cabinet-nano/athena-cabinet-nano.ino:554` serializes magnitude, LiDAR
   `distance_mm/status`, and BNO085 `accel_x/y/z` pass through host reader snapshots.
   `extract_cabinet_measurements()` in `orion/telemetry/cabinet_sensors.py:118–137`
   reads magnitude, accepts status-zero distance and computes
   `abs(norm(accel)/9.80665 - 1)`. Browser uses actual source snapshots, not equality
   guesses against the merged value. Cross-language parity covers the shared
   `merge_cabinet_frame_channels()` newest-source rule and all three native values.
2. **Independence:** magnetic and IMU share Nano b, timing and potentially machinery
   causes; LiDAR comes from Nano a and shares physical context. Vibration is a
   deterministic transform of acceleration; audio is a deterministic transform of
   raw readings. Neither audio targets nor existing activity scores add independent
   evidence. No combined cognitive model is introduced.
3. **Theory:** parameter-mapping sonification (Grond/Berger, *The Sonification
   Handbook*, ch. 15) supports an audible data display. The mapping says nothing
   about affect, identity, safety or the cause of a measurement.
4. **Live sanity:** 120 GET responses from the running Hub, 2026-10-20
   17:57:02.599Z–17:59:02.600Z, retained in the sanitized fixture. Magnetic:
   21.79–51.86 µT, 117 distinct levels, so fixed 20–60 µT bounds contain this capture.
   LiDAR: 89–100 mm, 12 distinct levels; vibration: 0.0382827–0.0441612 g, only
   six distinct levels. The ~0.04 g rest proxy must not be called zero or impact.
   Magnetic is an absolute-level tone: steady input is steady pitch, not an anomaly
   or an instrument expected to decay to silence. Missing/stale data mutes it.
   Controlled quiet/movement/return-to-rest labels were unavailable. LiDAR and
   vibration therefore fail intervention suitability and remain disabled; their
   0–4000 mm and 0.10/0.06 g settings are testable proposals only. Human audibility,
   intelligibility and blinded source/direction trials remain **UNVERIFIED**.
5. **Existing mechanism:** reuse existing Hub poll/API and source snapshot contract;
   no parallel poller, new event, activity channel or duplicate backend producer.
6. **Reversibility:** remove the module/controls/three integration calls. No stored
   Orion state, schema, training default or migration depends on this instrument.

## Tests run

```text
node --test services/orion-hub/static/js/cabinet-sonification.test.js services/orion-hub/static/js/cabinet-sensors.test.js
12 passed.
python -m pytest services/orion-hub/tests/test_cabinet_sonification_provenance.py services/orion-hub/tests/test_cabinet_sensors_panel.py services/orion-hub/tests/test_cabinet_sensors_api.py tests/test_cabinet_sensors.py -q
72 passed (22 existing dependency warnings).
git diff --check: passed.
```

## Evals run

```text
python services/orion-hub/evals/cabinet_sonification_server.py --port 18097
PUPPETEER_MODULE=<primary Hub>/node_modules/puppeteer node services/orion-hub/evals/cabinet_sonification_browser.cjs http://127.0.0.1:18097
PASS: real rendered controls and linked asset, gesture, mute/solo across scheduler
steps, trace, Stop preserves separate audio, hide/reopen, suspended recovery,
pagehide, poll failure; replayed 120 recorded frames.
Native OfflineAudioContext mix: finite=true, peak=0.1097215935587883 (<1).
```

Browser interaction fixture timestamps are refreshed intentionally; original
source times are retained in the separate replay. Replay time is the newest raw
source time plus one second because the API reads raw snapshots after the merged
snapshot (one real capture contained a newer raw source). No fetching timestamp
replaces source time in production. The eval isolates the real rendered Cabinet
panel from unrelated Hub overlays; it does not claim full modal-navigation coverage.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-hub build
PASS: Hub app and retro gateway images built; no up/deploy.
Running Hub GET /api/cabinet/sensors/latest: fresh two-source measurements captured.
Isolated worktree template/static HTTP smoke and native browser audio: PASS.
```

The deployed data-to-audio path is **UNVERIFIED**; no merge/deploy was performed.

## Review findings fixed

- Finding: all-muted or soloing a disabled voice stopped the context permanently.
  - Fix: scheduler checks unfiltered source availability separately from user mix.
  - Evidence: Node test ticks while muted/soloed; browser holds mute 100 ms before
    unmuting. Review subagent rechecked and found no further material issues.
- Review used `requesting-code-review` SKILL.md under the local superpowers 7.0.0
  cache. Human calibration/listening and production runtime were explicitly excluded
  from the review verdict, not silently treated as passed.

## Restart required

After a separately authorized merge/deploy, from clean primary main synchronized
with origin/main:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Medium: LiDAR/vibration calibration and human listening acceptance incomplete.
  Their voices stay gated off. Capture controlled movement/rest and submit a reviewed
  preset change before enabling; run the README headphone/blinded listening procedure.
- Low: no TTS lifecycle hook; manual Stop during TTS. Separate context prevents Stop
  interrupting speech. Speaker output can affect existing ambient observations.
- Scope: recommended first live slice only; history and optional voices deferred as
  specified. Production audio path remains UNVERIFIED until separately deployed.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2799
