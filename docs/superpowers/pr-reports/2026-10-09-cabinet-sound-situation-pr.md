## Summary

- Orion now hears the cabinet mic in chat. The unified-chat situation brief gets a line with the live mic level in dBFS, how it compares to the last 24 h (quieter / about usual / louder), the 24 h range and median, and the last-10-minute median.
- "Usual" comes from the readings biometrics already stores every ~30 s (`orion_biometrics_summary`), not an in-process EWMA. The brief only runs on chat turns, so an EWMA fed there would sample whenever someone talks and reset on every Hub restart.
- The mic reading stands independently of the Nano board. A stale Nano frame no longer hides it, and a mic or database fault no longer wipes the Nano reading.
- The snapshot loader moved from orion-biometrics into `orion/telemetry/ambient_audio.py`, so biometrics and the brief validate the mic file the same way. Biometrics keeps a re-export.

## Outcome moved

Orion said their cabinet was "quiet as a basement" (2026-10-09) while the server fans roar. The mic had been captured since 2026-08-24, but chat never saw it.

## Current architecture

```
USB mic -> host reader -> /run/orion-audio/latest.json -> biometrics -> orion_biometrics_summary
```

The situation brief's `CabinetContextV1` carried only Nano sensors (temperature, humidity, particulates, and so on). There was no audio.

## Architecture touched

- `orion/schemas/situation.py`: new `sound_*` fields on `CabinetContextV1`. They are additive. The brief is not re-parsed by any other process, which was checked.
- `orion/situational/cabinet_sound_reader.py` (new): one bounded query that is fail-open. It uses a text compare on the varchar timestamp so the `(node, timestamp)` index is used: 36 ms vs 405 ms when cast.
- `orion/situational/context.py`:
  - settings wiring: reuses Hub's existing `AMBIENT_AUDIO_PATH`, `AMBIENT_AUDIO_STALE_AFTER_SEC` and `CABINET_AMBIENT_HISTORY_NODE`
  - fetch split
  - render line
  - brief-cache staleness gate now covers the mic
  - `cabinet_sound` provider diagnostics
- `orion/telemetry/ambient_audio.py`: shared snapshot loader, plus `pcm16_to_dbfs`.

## Schema / bus / API changes

- Added: `CabinetContextV1.sound_available`, `sound_age_seconds`, `sound_dbfs`, `sound_peak_dbfs`, `sound_recent_dbfs`, `sound_usual_low_dbfs`, `sound_usual_dbfs`, `sound_usual_high_dbfs`, `sound_vs_usual`.
- No bus or channel changes.

## Env/config changes

None. Only existing Hub keys are reused. Processes without a mic path, such as cortex-exec, stay off.

## Metric gate

1. **Provenance:** `scripts/orion_ambient_audio_reader.py` computes RMS and peak over 0.5 s of S16 PCM at 16 kHz. Biometrics stores them as `measurements.cabinet_ambient_rms` on node athena.
2. **Independence:** this is a new physical sense. It correlates 0.47 with total GPU watts per minute over 48 h (about 920 W in loud stretches vs about 640 W in quiet ones), so fans following load is plausible but not established. It is not a transform of any other brief field.
3. **Theory:**
   - dBFS is level relative to 16-bit full scale.
   - The mic is uncalibrated. No absolute anchors such as "quiet room" or "speech" are claimed (review finding).
   - The 24 h p10/p90 band is the comparison.
4. **Live data:**
   - Over 72 h the readings are bimodal: about −17 dBFS (RMS around 4,400) and about −12.6 dBFS (around 7,700).
   - The level never drops below about −18.4 dBFS, so nothing in this data shows the cabinet ever being quiet.
   - The data is not flat and not saturated.
   - Live end-to-end against the real mic file and Postgres rendered: "-16 dBFS, about usual (last 24h ranged -17 to -11, median -12; last 10 min -16)".
5. **Existing mechanism:**
   - The biometrics `cabinet_ambient_audio_activity` (EWMA volatility) is reused only as context. It averages about 0.2 at every loudness level, so it measures change, not loudness, and is not rendered.
   - No dB conversion or baseline existed anywhere.
6. **Reversible:** additive optional fields and one prompt line. Remove the render block to retire it.

## Tests run

```
orion/situational/tests                    152 passed
services/orion-biometrics/tests -k ambient 9 passed
reviewer: biometrics full 168 passed / 2 pre-existing circe failures (fail on main too)
reviewer: check_definition_drift --gate, check_metric_lineage --gate, async-routes, sentience-instruments, inner-state-registry: pass
```

CI: `session-scope-tests.yml` runs `orion/situational/tests` on this path.

## Evals run

No eval harness exists for the situation brief line. The live smoke above is the evidence. Follow-up: once this is deployed, check whether Orion still calls the cabinet quiet in chat.

## Review findings fixed

- **Finding:** the 300 s brief cache replayed "heard just now" with a stale verdict.
  - Fix: `_cached_percept_outlived_gate` also expires the cache on mic age.
  - Evidence: `test_brief_cache_hit_expires_once_mic_reading_is_stale`.
- **Finding:** generic "quiet room −50 / speech −30" anchors were unverified on this mic.
  - Fix: removed them. The line now says to judge against the cabinet's own range.
  - Evidence: render test asserts no "-50".
- **Finding:** a single 0.5 s window drove the verdict.
  - Fix: the verdict now uses the 10-minute median and falls back to the live window.
  - Evidence: two parametrized tests.
- **Finding:** a mic or database exception wiped the Nano reading.
  - Fix: try/except around the sound fields.
  - Evidence: `test_mic_failure_does_not_wipe_nano_read`.
- **Finding:** diagnostics were missing on cabinet cache hits, and the cache key omitted the history node.
  - Fix: both fixed.
- **Not fixed:** the reviewer said no CI covers `orion/situational/tests`. That is wrong: `session-scope-tests.yml` runs it.

## Restart required

Hub only (one line, from primary checkout on main after merge):

```bash
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: low.** The mic gain is uncalibrated, so Orion gets relative loudness, not absolute loudness. A one-time silent reading and a speech reference on this mic would allow honest absolute anchors.
- **Severity: low.** The line adds about 250 characters to the brief. That fits well within the 7,200-character budget.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
