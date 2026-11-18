## Summary

- **Cabinet audio button.** The chat card has a new "Cabinet audio" button at the far left of the vision chip / Copy chat row. Click it to hear the cabinet mic live. Click again to stop.
- **One waveform for both sounds.** The cabinet mic and Orion's TTS voice draw on the same "Oríon's Voice" canvas: cabinet in teal underneath, voice in violet on top. The label follows what is playing.
- **The reader now streams.** The host mic reader used to record a fresh 0.5s clip each loop and keep only loudness. It now runs one continuous `arecord` stream. It still writes the same `latest.json` levels. It also writes each 0.5s window to a 16-slot ring on tmpfs (~8s, overwritten, never saved).
- **Hub endpoint.** New `GET /api/cabinet/ambient/live` tails that ring and streams raw PCM to the browser.
- **Recall Mode moved.** The Recall Mode card is off the main page. It is now the first row in the Debug Panel, with its own Modal. The modal borrows the one live set of selects and gives them back, so a pick there is what the next chat turn uses.

## Outcome moved

Before, nobody could hear the cabinet: the reader threw the sound away. Now Juniper can listen live, about 0.5–1s behind real time. The main page also loses one control card.

## Current architecture

`orion-ambient-audio.service` (host) → spawned `arecord --samples 8000` once per loop → RMS/peak → `/run/orion-audio/latest.json` → biometrics and the Hub chips. The PCM was discarded. The Hub already bind-mounted `/run/orion-audio:ro`. The Recall Mode/Profile selects sat in their own card above Operator tools.

## Architecture touched

- Host reader `scripts/orion_ambient_audio_reader.py`: continuous stream plus the `LiveRing`.
- Host-local contract `orion/schemas/telemetry/ambient_audio.py`: ring slot layout. This is not a bus payload and is not registered, same as the snapshot.
- Hub: `scripts/cabinet_ambient_routes.py` (`/live`), settings/env/compose (`AMBIENT_AUDIO_LIVE_DIR`), `static/js/cabinet-live-audio.js` (new), `static/js/app.js`, `templates/index.html`.

## Files changed

- `scripts/orion_ambient_audio_reader.py`: one long-lived arecord. stderr goes to a temp file, never an undrained pipe. A `select` watchdog fires after 3 windows with no audio. `LiveRing` clears the previous run's slots on start. Kill switch `ORION_AMBIENT_AUDIO_LIVE_ENABLED`.
- `orion/schemas/telemetry/ambient_audio.py`: slot header, slot count and slot naming, shared by the reader and the Hub.
- `scripts/smoke_athena_ambient_audio.sh`, `scripts/setup_athena_ambient_audio.sh`: use the stream; the default-env template gets the kill switch.
- `services/orion-hub/scripts/cabinet_ambient_routes.py`: `/live` StreamingResponse. Returns 503 when the ring is empty or the newest slot is older than 5s. Follows seq order, skips a missing seq, resyncs after a restart or lag, and ends the stream after 5s with no new chunk.
- `services/orion-hub/static/js/cabinet-live-audio.js`: fetch stream → int16→float → gapless `AudioBuffer` scheduling with a 0.35s jitter head start → gain → analyser → speakers.
- `services/orion-hub/static/js/app.js`: shared visualizer (`drawAnalyserBars`, a single rAF loop); voice box visibility and label; button wiring; Recall Mode toggle, modal, Escape and scroll lock.
- `services/orion-hub/templates/index.html`: button, label id, Recall Mode row and modal, script tag.
- `services/orion-hub/{.env_example,app/settings.py,docker-compose.yml,README.md}`, `services/orion-biometrics/README.md`.
- Tests: `tests/test_ambient_audio_reader.py`, `services/orion-hub/tests/test_cabinet_ambient_api.py`, `services/orion-hub/tests/test_hub_cabinet_live_audio_ui.py` (new, includes a node harness for the player), `test_hub_ui_layout_pass.py`, `test_llm_route_selector.py`.

## Schema / bus / API changes

- Added: `GET /api/cabinet/ambient/live`, which streams `application/octet-stream` S16_LE with the headers `X-Orion-Sample-Rate`, `X-Orion-Channels` and `X-Orion-Sample-Format`. Also added: the host-local ring layout at `/run/orion-audio/live/slot-NN.pcm`.
- Removed: `capture_pcm_via_arecord`, the one-shot capture. It is replaced by `stream_pcm_via_arecord`.
- Behavior changed: `latest.json` now refreshes every 0.5s instead of ~1.5s, and it covers 100% of the time instead of ~33%. `received_at` is the end of the window. The schema is unchanged. No bus changes.

## Env/config changes

- Added keys: Hub `AMBIENT_AUDIO_LIVE_DIR=/run/orion-audio/live`. Host reader `ORION_AMBIENT_AUDIO_LIVE_ENABLED` (default on) and `ORION_AMBIENT_AUDIO_LIVE_DIR`, both read from `/etc/default/orion-ambient-audio`.
- `.env_example` updated: yes (hub).
- local `.env` synced: yes. `sync_local_env_from_example.py --all-keys orion-hub` added `AMBIENT_AUDIO_LIVE_DIR`. The default (feature-key) mode reported "no changes" for this key.
- skipped keys requiring operator action: none.

## Tests run

```text
tests/test_ambient_audio_reader.py                                   18 passed
hub: test_cabinet_ambient_api + test_hub_cabinet_live_audio_ui
     + test_hub_ui_layout_pass + test_llm_route_selector             81 passed
full services/orion-hub/tests: 48 failed / 3957 passed. The same 48 fail on an
unmodified 95422b47d worktree (diff of failing ids is empty). They are
pre-existing: browser smokes, live e2e and unrelated mocks.
check_env_template_parity.py: PASS
```

Real `arecord` check (`-D null`, because the service holds the mic): 3 × 16000-byte windows, and no orphan arecord after the consumer breaks.

## Evals run

```text
None. The hub has no eval harness for cabinet audio. This is a playback
surface, not a cognition path; the node harness covers decode and scheduling.
```

## Docker/build/smoke checks

```text
Not run. Live deploy happens from primary main after merge.
Live listen in a browser: UNVERIFIED until the reader restarts and the hub redeploys.
```

## Review findings fixed

- Finding (high): the arecord stderr pipe was never drained. Overrun lines would fill it, arecord would block, and the reader would freeze with no restart.
  - Fix: stderr goes to a `TemporaryFile`; a `select` watchdog raises after 3 windows with no audio.
  - Evidence: `test_arecord_stream_is_one_continuous_process_cut_into_windows` asserts stderr is not a PIPE; `test_arecord_stream_raises_when_device_hangs_silently`.
- Finding (medium): seq reset on restart while old slots survived (`RuntimeDirectoryPreserve`), so old chunks shadowed new ones.
  - Fix: `LiveRing` deletes old slots on start.
  - Evidence: `test_live_ring_drops_previous_runs_slots_on_start`.
- Finding (medium): one failed slot write stalled the tail for 8s.
  - Fix: skip to the next present seq.
  - Evidence: `test_next_live_chunks_skips_a_seq_the_reader_failed_to_write`.
- Finding (low): 200 + a stale chunk when the reader was dead but files remained.
  - Fix: 503 when the newest slot is older than 5s.
  - Evidence: `test_live_endpoint_503_when_reader_stopped_but_slots_remain`.
- Finding (low): the TTS analyser stayed set after watchdog/error/interrupt, so the rAF loop could run forever.
  - Fix: draw the voice only while `isPlayingAudio`; the loop stops when nothing is live.
  - Evidence: UI test assertion.
- Finding (low): a stereo frame split across chunks would swap channels.
  - Fix: carry the partial frame.
  - Evidence: node harness stereo case.
- Finding (low): feedback risk.
  - Fix: README note (headphones).
- Not changed (low): synchronous 16-file tmpfs read per 100ms poll. Fine for one operator.

## Restart required

Run each command after merge, from the primary `main` checkout, in this order:

1. Pull main:

```bash
git -C /mnt/scripts/Orion-Sapienform pull --ff-only
```

2. Restart the host reader:

```bash
sudo systemctl restart orion-ambient-audio
```

3. Rebuild and bring up the hub:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Medium. Feedback.**
  - Concern: if the speakers are within reach of the cabinet mic, playback can howl, and it inflates `cabinet_ambient_*` levels.
  - Mitigation: README note. Use headphones.
- **Low. Level cadence.**
  - Concern: biometrics sees 0.5s-fresh levels with full time coverage instead of ~1/3 coverage. Peak/RMS per window is unchanged, but catching more loud moments can move `cabinet_ambient_audio_activity` baselines slightly.
- **Low. Privacy.**
  - Concern: a live room mic is reachable from the Hub (tailnet only). Nothing is persisted.
  - Mitigation: kill switch `ORION_AMBIENT_AUDIO_LIVE_ENABLED=0`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2835

🤖 Generated with [Claude Code](https://claude.com/claude-code)
