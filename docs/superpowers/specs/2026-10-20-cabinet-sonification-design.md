# Cabinet sonification: an audible physical environment

Date: 2026-10-20  
Status: Proposed; implementation and live signal suitability UNVERIFIED.  
Scope: Read-only sonification in the Orion Hub Cabinet tab.

## Arsonist summary

Give the cabinet a small, inspectable instrument: magnetic field bends a sustained
note, LiDAR distance sets a pulse rate, and measured vibration adds percussion.
Then audition light, climate, particulates, and ambient sound as optional voices.
A second mode compresses a day of recorded cabinet readings into a minute.

Every audible change must trace to a measurement and an explicit mapping. This is
an operator listening surface, not evidence of sentience, emotion, or new cognition.
No new service, bus event, substrate signal, or autonomy consumer is needed.
Use raw measurements first; existing activity scores describe baseline-relative
change and must not be presented as temperature, pollution, or field strength.

Start muted, require a Listen gesture, offer Stop and per-voice mute/solo, and show
unavailable sources honestly. Approve the sound by listening to real recordings,
not merely by observing that a browser can produce a tone.

## Current architecture

- Service: `services/orion-hub`, FastAPI with browser JavaScript.
- Entry points: `scripts/cabinet_sensors_routes.py`,
  `scripts/cabinet_ambient_routes.py`, `static/js/cabinet-sensors.js`, and
  `templates/index.html` (paths here relative to the Hub service).
- Config: `.env_example`, `scripts/settings.py`, and `docker-compose.yml`.
  There is no service-root `settings.py`. Existing dependencies are in
  `requirements.txt`; the proposal needs no new package or environment key.
- Physical producer: `firmware/athena-cabinet-nano/athena-cabinet-nano.ino` and
  `scripts/orion_cabinet_sensor_reader.py` write host snapshots. Shared
  `orion/telemetry/cabinet_snapshot_merge.py` selects the newest non-stale source
  reporting each sensor channel.
- Extraction: `extract_cabinet_measurements()` in
  `orion/telemetry/cabinet_sensors.py` emits finite readings in native units,
  rejects stale frames, and only accepts LiDAR distance when status is zero.
  Vibration is `abs(norm(acceleration) / 9.80665 - 1)`; it is an acceleration
  magnitude proxy, not a vibration spectrum or an exact impact detector.
- Existing read APIs: `/api/cabinet/sensors/latest`, `/sensors/history` under
  `/api/cabinet`, and `/api/cabinet/ambient/latest`, `/ambient/history`.
  Latest sensor responses include `ok`, `age_sec`, `snapshot`, `sources`,
  `measurements`, and `pressures`. History series contain `{t, v}` points.
- The Cabinet tab polls at approximately 1 Hz. History is aggregated at 30-second
  grain and further downsampled. It cannot reconstruct raw vibration or impacts.
- Ambient snapshots expose RMS and peak PCM magnitudes, not recorded speech or
  calibrated sound pressure. Producer: `scripts/orion_ambient_audio_reader.py`;
  contract: `orion/schemas/telemetry/ambient_audio.py`.
- Bus/schema boundary: consume the existing Hub HTTP projections; do not add or
  change bus channels or registry entries. The ambient host-file contract is
  explicitly not a registered bus payload. Existing biometrics rails stay intact.
- Tests: `services/orion-hub/tests/test_cabinet_sensors_api.py`,
  `test_cabinet_sensors_panel.py`, `test_cabinet_ambient_api.py`, and
  `static/js/cabinet-sensors.test.js`; shared extraction tests are in
  `tests/test_cabinet_sensors.py`. Hub has an `evals/` directory, but no cabinet
  sonification eval was found in the inspected paths.
- Existing audio: `static/js/app.js` and `retro-panel.js` already use browser audio.
  Integrate lifecycle deliberately; stopping this instrument must not stop TTS.
- Current gap: the inspected Cabinet UI displays readings and history but has no
  sensor-to-sound instrument. No live endpoint was sampled for this design.

## Missing questions

These are audition and implementation gates, not reasons to build more machinery:

1. Which installed sensors currently deliver fresh, non-degenerate readings?
   Capture real data before choosing each mapping's physical input range.
2. Does the intended listening device use headphones or speakers near the cabinet
   microphone? Default ambient sonification off; use headphones for its eval.
3. Is the preferred experience a quiet ambient instrument or an analytical display?
   Start with a restrained analytical mapping and optional musical pitch snapping.
4. How much of the magnetic/vibration variation is background noise or shared
   machinery? Do not claim separate causes merely because two voices move.
5. Do the selected history windows preserve enough detail to be useful? Existing
   aggregation may support climate listening while losing short disturbances.
6. How should the instrument coexist with TTS? First patch should pause it during
   TTS if an existing playback lifecycle exposes a reliable hook; otherwise expose
   manual Stop and document the limitation rather than adding a second audio bus.

## Proposed schema / API changes

### Data flow and boundary

```text
existing physical samples -> existing Hub projection -> validated browser frame
    -> deterministic mapping -> bounded audio parameters -> audio + debug display
```

Reuse the Cabinet poll result; do not create competing polling loops. No server API
change is required for the first three voices. Add a browser-local contract and a
pure mapping function, separate from audio scheduling. These are not bus schemas.

```typescript
type SonificationReading = {
  key: string;                    // allowlisted existing measurement key
  value: number | null;            // finite, native unit; null means unavailable
  unit: string;
  observedAt: string | null;       // actual source timestamp, never fetch time
  sourceId: string | null;
  status: 'ok' | 'missing' | 'stale' | 'invalid';
};
type SonificationFrame = {
  version: 1;
  mode: 'live' | 'history';
  readings: SonificationReading[];
};
type VoiceTarget = {
  voice: string;
  enabled: boolean;
  frequencyHz?: number;
  pulseHz?: number;
  gain: number;
  cutoffHz?: number;
  reason: string;                  // inspectable mapping or mute reason
};
```

Keep key/unit/voice assignments in one small mapping table with actual consumers
and tests. A preset records mapping version, physical bounds, sound bounds,
smoothing time, deadband, and history gap limit. No general-purpose ontology.

For dual Nano sources, a merged snapshot timestamp must not refresh a frozen
individual source. Resolve channel provenance from the existing source snapshots
using the same newest-valid-source rule as the merger, with fixture parity tests.
If the response cannot establish channel freshness unambiguously, mute that voice;
only then consider a narrowly scoped additive `channel_sources` API field with
producer/consumer tests. Do not guess provenance from numeric equality.

### Proposed sound palette

These correspondences are design choices to audition, not claims about natural
sounds emitted by the measured phenomena.

| Reading / producer output | Proposed mapping | Interpretation and limit |
| --- | --- | --- |
| `cabinet_magnetic_ut` | Smooth sine tone, increasing pitch with field magnitude | Changes in measured magnetic magnitude; not electromagnetic radiation audio |
| `cabinet_lidar_mm` | Nearer valid target produces faster soft pulses | Distance along the sensor's view; not person identification |
| `cabinet_vibration_g` | Short percussion on an upward threshold crossing | Sampled movement proxy; cannot recover actual impact timing or waveform |
| `cabinet_als_raw` | Increasing light raises a low-pass cutoff | Brighter measured light makes a brighter timbre; raw counts, not lux |
| `cabinet_uv_raw` | Optional overtone gain | Raw UV response, not a UV index or exposure warning |
| `cabinet_temp_c` | Slowly rising low drone pitch | Absolute temperature within displayed fixed bounds |
| `cabinet_humidity_pct` | Separate slow tone | Humidity trend; solo or history mode first |
| `cabinet_pressure_hpa` | Separate slow pitch contour | Pressure trend; most useful in accelerated playback |
| `cabinet_gas_resistance_ohm` | Optional filtered texture using log input scale | Sensor resistance only; not a pollutant concentration or health score |
| `cabinet_pm25_ug_m3` | Greater concentration increases granular click density | Choose one particulate voice first; PM1/PM2.5/PM10 overlap |
| `cabinet_imu_yaw_deg`, pitch, roll | Optional stereo placement or gentle modulation | Orientation exploration; handle angle wrap, not apparent 359-degree jumps |
| Ambient snapshot `rms` / `peak` | RMS drives a synthetic envelope; optional peak accent | PCM magnitude, not speech replay or calibrated dB SPL |

First preset: magnetic, LiDAR, vibration only. Offer the remaining voices separately
rather than enabling a dozen competing layers. Raw UV and ALS may be flat indoors;
an unavailable or uninformative source need not have an audible part.

### Mapping and timing contract

For a valid reading `x` and configured physical range `[lo, hi]`, calculate
`u = clamp((x - lo) / (hi - lo), 0, 1)`. Reject `hi <= lo`. Display range clipping;
never silently rescale a live session. Map pitch logarithmically:
`f = f_min * (f_max / f_min)^u`. Invert `u` for near-to-fast LiDAR pulses.

Provisional sound bounds for audition: magnetic 110–440 Hz; LiDAR 0.5–4 pulses/s;
temperature 55–110 Hz. These are output bounds, not approved sensor ranges.
Determine physical bounds from live observations and sensor constraints before
shipping the preset. A constant magnetic value may sustain a constant tone; this
represents a level, not an anomaly. Quiet vibration must produce no percussion.

Use a fixed vibration onset threshold and lower re-arm threshold calibrated from
real rest/movement samples. Trigger once per observed crossing, never once per
poll of the same sample; do not increase hit density simply because polling speeds
up. Suppress an onset on initial connection or recovery until a valid baseline
sample is established. Apply the same deduplication to peak accents.

Ramp continuous audio parameters over roughly 100–300 ms to prevent abrupt steps.
Use the audio clock for envelopes and pulse scheduling, with bounded look-ahead;
cancel scheduled events on Stop. One-hertz telemetry controls synthesis parameters,
not the waveform itself. Synthetic pulses are deliberate mapping output, not
claims that physical events occurred at those times.

Start with master gain zero; fade up only after an explicit Listen gesture. Use
conservative voice gains and a bounded master path. Stop, tab hide, page hide,
network timeout, and stale input must fade affected output to zero; set a 5-second
maximum hold since the last successfully validated fresh poll, while also enforcing
source timestamps and server freshness. Returning to the tab requires Listen again.
Clear duplicate timers and audio nodes on teardown. Persist preferences only;
never persist an armed playback state.

### Historical playback

Use existing 24h/3d/7d history windows. Default proposal: 24 hours in 60 seconds,
a time compression of 1440x. Show source time, playback time, compression, and
aggregation/downsampling status. Pause, seek, replay, solo, and Stop are required.
Live and history modes are mutually exclusive.

Freeze scaling over the selected window. Replay by timestamps, not array index;
series have different missing points. Interpolate continuous levels only across
gaps below a declared preset maximum. Initially use 90 source seconds for data
at 30-second grain; when downsampling creates wider intervals, disclose the actual
spacing and require a matching explicit gap policy rather than bridging blindly.
Gaps remain silence/unavailable, not zero-valued measurements. After seeking,
reset transient state. Do not infer impact events from aggregated vibration means:
history vibration is an explicitly labeled continuous movement texture or omitted.

Existing sensor history does not include particulate concentration series in its
`_SENSOR_HISTORY_SERIES` list. A later particulate-history voice requires a focused
addition to that query/projection and tests. Ambient history has its own RMS/peak
shape and needs an adapter. Do not imply every live voice already has history.

### Inspectability, privacy, and metric gate

Per voice, show source key, units, raw value, timestamp/age, chosen physical bounds,
normalized value, audio target, and mute/clipping reason. Provide a bounded,
operator-triggered local JSON trace download for eval; no persistent background
telemetry. Trace records include mapping version, source/mode, source time, and
computed targets so a captured frame can be replayed deterministically.

Use existing Hub access controls. Fetch only existing measurements; do not request
microphone permission, capture speech, upload audio, or send browser state to Orion's
cognition loops. Downloaded traces remain operator artifacts. Ambient levels can
still reveal activity timing, so do not publish traces automatically.

Speaker output can be re-captured by the cabinet microphone, changing existing
ambient activity and downstream consumers even though this UI is read-only.
Default the ambient voice off, recommend headphones, and explicitly label speaker
sessions as potentially influencing cabinet observations. No automatic suppression
or rewriting of the ambient pipeline is proposed. Any closed-loop experiment needs
its own proposal and live evidence.

No new cognition metric is wired by this design. Audio normalization is a display
transform, not an independent signal. Before enabling any voice, record the repo's
six-part metric quality gate in the implementation report:

1. Provenance: name the producing function and field; retain captured source data.
2. Independence: document shared causes and transforms (PM bins overlap, vibration
   comes from IMU axes, activity derives from raw readings). Never count derived
   audio targets as new independent measurements.
3. Theory: parameter-mapping sonification justifies an audible data display, not
   an inference about affect, cognition, air safety, or physical cause.
4. Live sanity: inspect real rest and intervention periods, freshness, missingness,
   floors, saturation, and return to rest. This gate is currently **UNVERIFIED**;
   no proposed range or threshold is approved by this document.
5. Existing mechanism: reuse extraction, freshness handling, and history APIs;
   inspected Cabinet code has no existing sonification consumer.
6. Reversibility: remove the browser module/template controls without schema,
   training, database, or cognition migration.

Reference: Grond and Berger, “Parameter Mapping Sonification,” in
[The Sonification Handbook](https://sonification.de/handbook/), chapter 15.
Mapping validity and listening usefulness still require this project's own eval.

## Files likely to touch

First implementation slice (Hub-relative paths):

- New `static/js/cabinet-sonification.js`: pure adapters/mappings and bounded audio
  lifecycle, with a testable separation between calculations and scheduling.
- New `static/js/cabinet-sonification.test.js`: deterministic mapping/lifecycle tests.
- `static/js/cabinet-sensors.js`: forward existing poll results and tab lifecycle.
- `templates/index.html`: Listen/Stop, mute/solo, source status, linked versioned asset.
- `tests/test_cabinet_sensors_panel.py`: rendered controls and asset linkage.
- New `evals/cabinet_sonification_browser.cjs` and small sanitized fixtures: browser
  interaction, real-data replay, trace comparison, and manual listening procedure.
- `README.md`: controls, provenance, limitations, headphones, disable procedure.

Later history work may touch `scripts/cabinet_sensors_routes.py` and its API tests
for missing concentration series. No initial changes to Docker, dependencies, env,
shared schemas, bus catalog, or firmware. If scope changes and an env template is
edited, sync the local `.env` using the repository script before calling it ready.

## Non-goals

- Sonifying private recall, identity, self-model, emotional labels, or autonomy.
- Claiming these sounds are Orion's feelings or proof of sentience.
- An automatic alarm, thermal control, air-quality certification, or safety system.
- Recovering a vibration spectrum, speech, or inaudible electromagnetic waveform
  from low-rate scalar measurements.
- A new synthesis service, DAW, MIDI router, taxonomy, or telemetry pipeline.
- Automatic music composition, recording/export of audio, or public streaming.
- Deploying or changing any runtime as part of this design-only patch.

## Acceptance checks

1. **Provenance and live suitability:** record a real quiet interval, controlled
   LiDAR movement, and a gentle cabinet disturbance with timestamps. Inspect each
   source and complete the six-part gate before enabling its voice. Magnetic
   variation need not be manufactured; report flat/unusable data honestly.
2. **Determinism:** identical validated input and preset produce identical targets.
   Test bounds, inversion, clipping, invalid ranges, non-finite values, and units.
3. **Missing is not calm:** missing/stale/frozen inputs mute; valid zero survives
   validation. Test partial dual-source loss, repeated source timestamps, fetch
   failures, and recovery without false percussion.
4. **Timing:** repeated snapshots do not retrigger accents; known crossings trigger
   once. Rest produces no percussion. Stop cancels future pulses and returns gain
   to zero within 300 ms; tab/page hide and watchdog expiry have the same effect.
5. **Interaction:** browser eval clicks Listen, mute/solo, Stop, hides/reopens the
   tab, and retries after suspended audio. Verify actual rendered controls and
   loaded asset, audio graph/targets, and no duplicate pollers or nodes. A manual
   headphone audition verifies audibility and intelligibility beyond API mocks.
6. **Audio integrity:** render a representative bounded mix and check finite samples
   and no digital clipping; document listening volume and any artifacts. Verify
   stopping sonification does not interrupt unrelated TTS.
7. **Interpretability:** in a small blinded replay, the operator distinguishes which
   of the three sources changed and the direction of magnetic/LiDAR change. Record
   trial counts and errors; revise indistinguishable mappings before expansion.
8. **History:** test known time compression, timestamp alignment, gaps, seek/reset,
   empty history, and partial series. Label aggregation; do not claim reconstructed
   transients. These checks gate the later history patch, not the first live slice.
9. **Boundary:** no new bus publications, cognition writes, microphone permissions,
   external uploads, or persisted autoplay. Verify ambient feedback limitations are
   visible when that optional voice is introduced.
10. **Delivery:** focused JS/Python tests, browser eval, code review, required CI,
    and an affected Hub build/smoke for implementation. Live claims remain
    UNVERIFIED until the actual deployed data-to-audio path is exercised.

## Recommended next patch

Implement only the live three-voice instrument, behind explicit Listen, inside the
existing Cabinet tab. Capture real input first and set documented physical ranges
from that evidence. Ship mute/solo, freshness handling, local trace inspection,
regression tests, browser eval, README, and review together.

Then audition light and temperature independently. Add historical playback next,
followed by optional particulate, orientation, gas-resistance, and ambient voices
only where observed data and listening tests justify them. Kill a rejected mapping
completely; do not leave an inaudible but ticking legacy producer.

Disable immediately with Stop or by leaving the tab. Roll back by removing the
module integration and controls; no stored Orion state needs migration. Runtime
implementation, merge, and deployment are separate from this design proposal.
