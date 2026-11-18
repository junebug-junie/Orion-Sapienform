## Summary

- The frame check passed visibly broken walkway frames. It now catches them.
- Bug: on the 2560x1920 walkway frames, every 16-row strip fell under the "too flat to judge" floor, so streaked frames read `smear_fraction=0.00`. The check now scales each frame to 640 px wide first, so the floor means the same thing at every resolution.
- New `gray_fraction`: the share of 16x16 macroblocks filled with the flat mid-gray that FFmpeg paints when it loses a reference frame. Combined `corrupt` verdict per frame, printed one line per frame.
- The CLI exits 1 when any frame is corrupt (`--max-corrupt-fraction` allows a share), so it can serve as a post-deploy check. Half-written newest JPEGs are skipped, not scored.
- README: a post-deploy check section. If the check fails and transport is already TCP, reboot the camera first.

## Outcome moved

The post-deploy check now fails on the failure it was meant to catch. Before: 27 of 50 streaked walkway frames read 0.00 and the rest read about 0.5. After: all 75 streaked frames are corrupt (smear 0.45-0.83). All 131 clean post-reboot walkway frames and all 155 cam0 frames are clean.

## Current architecture

`services/orion-vision-retina/evals/frame_smear.py` was a smear-only CLI with a fixed per-pixel texture floor (`dx_min=2.0`). The floor was calibrated on 640x480 cam0 frames only. There was no gray detection and no per-frame output.

## Architecture touched

Only the offline eval (it reads saved JPEGs from disk). The running service, bus, schemas, and env are unchanged.

## Files changed

- `services/orion-vision-retina/evals/frame_smear.py`: resolution-normalized smear, `gray_fraction`, `frame_report`, `load_frame` (skips half-written JPEGs), corrupt-share exit code, `--max-gray`, `--max-corrupt-fraction`, `--quiet`.
- `services/orion-vision-retina/tests/test_frame_smear_eval.py`: synthetic fixtures for streak smear, high-res streak regression, decoder gray (exact and shifted), clean textured scene, clear blue sky, night, overcast/fog, gray walls, grayscale input, and the CLI gate (exit codes, max share, truncated JPEG, `--last 0`).
- `services/orion-vision-retina/README.md`: post-deploy check, false-positive risks, camera-reboot guidance.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: eval CLI only. It now prints per-frame lines, exits 1 on corrupt-share above the limit, and exits 2 when no frame is readable. The default smear limit is still 0.05.
- Compatibility notes: `smear_fraction()` gains a `width` keyword. Frames 640 px wide or narrower score exactly as before.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Measured calibration (2026-10-15, frames snapshotted to a scratch dir; none committed)

| Set | n | smear_fraction | gray_fraction | corrupt |
|---|---|---|---|---|
| walkway, streaked (18:50-18:53 UTC, TCP, `bad cseq`) | 75 | 0.448 min / 0.690 median / 0.828 max | 0.000 max | 75 |
| walkway, clean after camera reboot (18:55-18:56 UTC) | 131 | 0.000 max | 0.000 max | 0 |
| cam0, indoor clean | 155 | 0.000 max | 0.000 max | 0 |
| FFmpeg concealment (local H.264 test stream, keyframe cut out, JPEG q88) | 30 | 0.000 max | 0.677 min / 0.883 median / 0.912 max | 30 |

- Old code on the streaked walkway frames: 0.47-0.60 on the first 23 frames and **0.00** on the remaining 27.
- Lowest strip ratio on clean frames (the smear line is 0.3): walkway 0.82, cam0 0.57.
- FFmpeg's concealment color, measured: YCrCb (130, 128, 128) after OpenCV conversion, flat (block luma std 0 at the 90th percentile).
- Synthetic overcast sky (neutral, Y 125-145 ramp, noise sigma 0.8): 0.134 over the top 40% and 0.204 over the top 60%. Gray limit is 0.4.

**Honest gap:** the walkway frames on disk were streaked, not gray. The "~85% mid-gray" frames had already rotated out (the frame store keeps about one minute). The gray threshold was **not measured on real corrupted camera frames**. Its positive side comes from FFmpeg's own concealment output, decoded locally from a synthetic stream. No camera was contacted. The "gray" seen by eye on the live frames was most likely the neutral-gray building columns smeared down the frame. The fixed smear check catches those.

Root cause of the live corruption (per Juniper/coordinator): the camera's own encoder state, not the network. A camera reboot at about 18:55 UTC cleared it (0 `bad cseq`, 0 decode errors).

## Tests run

```text
.venv/bin/python -m pytest services/orion-vision-retina/tests -q -p no:cacheprovider
42 passed in 3.76s
New tests run against the pre-patch eval: the high-res regression test fails (reads 0.0, needs > 0.5); gray/CLI tests cannot pass without the new metric.
```

## Evals run

```text
Live post-deploy check, read-only on saved frames:
frame_smear.py /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 20  -> corrupt=0, exit 0
frame_smear.py /mnt/telemetry/vision/frames --glob 'frame_*.jpg'   --last 20  -> corrupt=0, exit 0
Same CLI on the snapshotted streaked walkway frames -> corrupt=20/20, worst_smear 0.76, exit 1
```

## Docker/build/smoke checks

```text
None needed: the eval is not part of the image and the service runtime is unchanged.
```

## Review findings fixed

- Finding (must): the retina writes JPEGs in place, so the newest frame can be half-written, and libjpeg pads a cut-off JPEG with flat gray (measured 0.67 gray, corrupt). The gate could trip on its own read timing.
  - Fix: `load_frame` skips JPEGs with no end marker and counts them as `skipped_unreadable`.
  - Evidence: `test_cli_skips_half_written_jpeg`.
- Finding (should): overcast sky sat just under the gray limit (0.227 vs 0.25).
  - Fix: tightened to the measured concealment signature (std < 0.6, |Y-130| < 4) and set the limit to 0.4. Overcast now reads 0.134 (40%) / 0.204 (60%). Concealment still reads 0.68 or more.
  - Evidence: `test_overcast_or_fog_is_not_decoder_gray`, `test_decoder_gray_a_few_levels_off_is_still_caught`.
- Finding (should): infrared black-and-white night mode passes the color test by construction.
  - Fix: covered by the tighter window. The README and docstring now name it as a residual risk.
  - Evidence: README section.
- Finding (should): a frame deleted between glob and `stat()` crashed the CLI.
  - Fix: `_newest` tolerates `FileNotFoundError`.
  - Evidence: code path.
- Finding (nit): `--last 0` checked every frame.
  - Fix: must be 1 or more.
  - Evidence: `test_cli_rejects_last_zero`.
- Finding (nit): grayscale input crashed the library functions.
  - Fix: grayscale and BGRA input are accepted.
  - Evidence: `test_grayscale_input_is_accepted`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium.
  - Concern: the gray threshold is not calibrated on real corrupted camera frames.
  - Mitigation: the limit sits between measured concealment (0.68 or more) and synthetic overcast (0.20 or less). The README says to look at the frame before acting on a gray-only failure. Re-calibrate the next time gray corruption is seen live.
- Severity: low.
  - Concern: an all-overcast or foggy frame at exactly mid-gray, or a flat infrared night scene, could still trip the gray check.
  - Mitigation: documented. Use `--max-corrupt-fraction` to tolerate a share.
- Severity: low (follow-up, out of scope).
  - Concern: `app/frame_store.py` writes frames in place (no temp file + rename). Any reader can see a half-written JPEG, including the code that turns frames into Orion's perception.
  - Mitigation: the eval skips these. A follow-up should switch the store to an atomic write.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2693

🤖 Generated with [Claude Code](https://claude.com/claude-code)
