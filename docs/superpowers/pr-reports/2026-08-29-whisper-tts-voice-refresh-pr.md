# Whisper-TTS: new reference voice (`orion_reference_v2.wav`)

Branch: `chore/tts-voice-refresh`

## Summary

- Repointed `TTS_DEFAULT_SPEAKER_WAV` at a new reference voice built from a
  source recording Juniper supplied, and recreated the container.
- **There is no retraining involved and none was performed.** Coqui XTTS-v2
  is zero-shot: no fine-tuning step, no checkpoint to retrain. The voice is
  determined entirely by one reference `.wav`. Reported this before acting
  rather than implying a training run had happened.
- Built the reference with a deliberately minimal chain (single-speaker
  window, downmix/resample, static gain) and matched its integrated
  loudness to the outgoing reference exactly, so timbre is the only
  variable that changed.
- Verified on the live bus, including a matched same-text control
  synthesized from the OLD reference, so the similarity numbers have a
  scale instead of being an uninterpretable single cosine.
- Documented the whole procedure in the service README so the next voice
  change is a recipe, not a re-derivation.
- Drive-by, called out: repaired two pre-existing test failures in this
  service that were failing identically on `main`.

## Outcome moved

Orion speaks in the requested voice on the live `orion:tts:intake` path.

Measured, not asserted -- XTTS speaker-embedding cosine, against a matched
same-text control synthesized from the old reference:

|             | `ref_new`   | `ref_old`   |
| :---------- | :---------- | :---------- |
| `synth_new` | **+0.6364** | +0.4608     |
| `synth_old` | +0.3337     | **+0.5515** |

Each synthesis is closest to its own reference. `synth_new`/`ref_new`
(+0.6364) also exceeds `synth_old`/`ref_old` (+0.5515), i.e. the new
reference clones *better* than the one it replaced.

The control row is what makes this readable. `ref_new`/`ref_old` is itself
+0.6498 -- the two references are not far apart in this space -- so
"synth scored +0.64 against the new reference" on its own would have proven
nothing.

## Current architecture

`services/orion-whisper-tts` runs Coqui XTTS-v2 behind a bus worker on
`orion:tts:intake`. `app/tts.py: resolve_synthesis_plan()` picks a speaker
in a fixed precedence order: per-request `options.speaker_wav` ->
`TTS_DEFAULT_SPEAKER_WAV` -> `voice_id` -> `TTS_DEFAULT_SPEAKER`. Reference
wavs live on the host under `TTS_VOICE_PROFILE_HOST_DIR`
(`/mnt/telemetry/models/coqui/voices`), bind-mounted read-only at
`/models/voices`. The wavs are host assets and are not in git.

Before this patch the live default was `orion_reference.wav` (29.97s,
-29.1 LUFS), in place since 2026-08-27.

## Architecture touched

Config only. No Python application code, no bus channel, no schema, no
compose change. `app/tts.py`, `orion/bus/channels.yaml`, and
`orion/schemas/` are all untouched.

## Files changed

- `services/orion-whisper-tts/.env_example`: `TTS_DEFAULT_SPEAKER_WAV` ->
  `/models/voices/orion_reference_v2.wav`, plus the build chain and the
  reason level is matched rather than normalized to a generic target.
- `services/orion-whisper-tts/README.md`: new "Changing the voice" and
  "Verifying a voice change" sections; updated the env table default and
  the example payloads that named the old file.
- `services/orion-whisper-tts/tests/test_tts_worker_replies.py`: drive-by
  fixture repair (see below).

Host assets, correctly NOT in git:

- `/mnt/telemetry/models/coqui/voices/orion_reference_v2.wav` (new live
  reference, 27.8s, 24kHz mono `pcm_s16le`, -29.1 LUFS)
- `/mnt/telemetry/models/coqui/voices/orion_reference.wav` (retained,
  untouched, rollback target)
- `/mnt/telemetry/models/coqui/voices/_sources/` (source recording +
  `orion_reference_v2.provenance.txt` recording the exact chain, so the
  reference is reproducible rather than a mystery binary)

## How the reference was built

Source: 109.0s, 48kHz stereo mp3.

1. **Window selection: 8.55s -> 36.35s (27.80s).** The longest stretch
   containing exactly one speaker with no silence gap over 0.6s. This is
   the step that mattered most: **the source's first 8.24s contain a second
   speaker** (an interviewer asking a question). Including any of it would
   have baked that voice into the clone. Found by transcribing the source
   and reading the segment table, not by trusting the crop.
2. **Cut / downmix / resample** to 24kHz mono `pcm_s16le`, matching the
   outgoing reference's container format.
3. **Static gain of -7.5 dB**, chosen so integrated loudness lands on
   -29.1 LUFS -- the outgoing reference's exact measured value. No
   compression, EQ, or de-noise: those alter the timbre the speaker encoder
   is being asked to copy. Level is matched on purpose so that timbre is
   the only thing that differs between reference versions.

Checked and deliberately skipped: sub-60Hz energy measured -50.5 dB against
an overall -22 dB, ~28 dB down, so no high-pass filter was warranted and
none was applied.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: the default synthesis voice. Per-request
  `options.speaker_wav` and `voice_id` overrides are unaffected -- exercised
  live in this session, since the control sample was produced by overriding
  back to the old reference through the normal bus path.
- Compatibility notes: none. Reply envelope shape is unchanged; only
  `metadata.speaker_wav_basename` reads differently.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- Value changed: `TTS_DEFAULT_SPEAKER_WAV`
  `/models/voices/orion_reference.wav` ->
  `/models/voices/orion_reference_v2.wav`
- `.env_example` updated: yes
- local `.env` synced: yes -- hand-edited (a value change on an existing
  key, not a key add), backup kept, then confirmed by explicit check
- skipped keys requiring operator action: none

### Env-parity gate does not cover this key by default

Worth recording, because the silence reads as a pass:

`python3 scripts/sync_local_env_from_example.py` as prescribed in CLAUDE.md
**cannot see this divergence at all.** Two independent reasons:

1. `orion-whisper-tts` is not in the script's hand-maintained
   `DEFAULT_SERVICES` list (12 of the 92 services on disk).
2. `TTS_*` is outside `should_sync_key()`'s default prefix allowlist, so
   even naming the service explicitly reports "No changes needed (feature
   keys already match)".

Established with a positive control rather than assumed: with
`.env_example` at v1 and `.env` at v2 -- a guaranteed divergence -- the
default invocation still printed "No changes needed". Adding `--all-keys`
made it report the divergence correctly. The passing check was therefore
re-run in the form that can actually fail:

```
cd <worktree> && python3 scripts/sync_local_env_from_example.py \
    orion-whisper-tts --all-keys
```

Not fixed here (widening `DEFAULT_SERVICES` / the prefix set is a
repo-wide gate change, not a voice patch), but flagged in the concerns
below.

## Tests run

The suite only runs from the service directory with `PYTHONPATH` set; from
the repo root it fails at collection with
`ModuleNotFoundError: No module named 'app'` (also true on `main`).

```text
cd services/orion-whisper-tts && PYTHONPATH=. python -m pytest tests -q

  baseline on main : 2 failed, 53 passed
  this branch      : 55 passed
```

The 2 baseline failures were pre-existing and unrelated: `BaseEnvelope.
correlation_id` is now a strict UUID, but `test_tts_worker_replies.py`
still passed `"cid-1"` / `"cid-legacy"`. Confirmed failing identically on
`main` with the same invocation before touching them. Neither test asserts
on the correlation_id value, so valid UUIDs change nothing about coverage.

Env-parity and compose-parity:

```text
python3 scripts/sync_local_env_from_example.py orion-whisper-tts --all-keys
  -> reading live .env from primary checkout: /mnt/scripts/Orion-Sapienform
  -> No changes needed.        (instrument first proven able to fail; see above)

python3 scripts/check_service_env_compose_parity.py orion-whisper-tts
  -> declares env_file: -- all 26 .env_example keys reach the container. N/A.
```

## Evals run

```text
none -- services/orion-whisper-tts has no evals/ directory.
```

Not claiming eval coverage. The service has never had an eval harness. The
live speaker-similarity 2x2 below is the closest thing this change has to
one, and it is a manual measurement, not a harness.

**Follow-up worth filing:** the similarity matrix in this PR is exactly the
shape of a real eval -- synthesize fixed text from each candidate
reference, embed, assert the diagonal dominates. That would turn "the voice
sounds right" from a one-off manual check into a gate that fires whenever
a reference is swapped or the model version moves.

## Docker/build/smoke checks

```text
cd <worktree> && scripts/safe_docker_build.sh orion-whisper-tts \
    up -d --force-recreate --no-build
  -> Container orion-athena-whisper-tts Recreated / Started

curl -fsS http://localhost:7800/health
  -> {"status":"ok","service":"whisper-tts","bus":"connected",
      "cuda_available":true,"cuda_watchdog_enabled":true}

docker inspect ... | grep TTS_DEFAULT_SPEAKER
  -> TTS_DEFAULT_SPEAKER_WAV=/models/voices/orion_reference_v2.wav
```

Live bus smoke on `orion:tts:intake` (real path, already-loaded model -- not
a second in-container model load):

```text
kind=tts.synthesize.result
bytes=284780 duration_sec=5.932
metadata={... 'speaker_wav_basename': 'orion_reference_v2.wav',
          'speaker_wav_used': True, 'synthesis_ms': 3801,
          'gpu_enabled': True}
```

Round-tripped the returned audio back through Whisper, because
schema-valid bytes are not evidence of intelligible speech:

```text
TRANSCRIPT>>> Hello, Juniper. This is Orion speaking with the new reference voice.
```

Matched control, same text, overriding `options.speaker_wav` back to the
old reference:

```text
metadata={... 'speaker_wav_basename': 'orion_reference.wav', ...}
```

Speaker-embedding cosine over all four files (`Xtts.
get_conditioning_latents`, run on CPU so the live GPU model was left
alone):

```text
cos(ref_new  , ref_old  ) = +0.6498
cos(ref_new  , synth_new) = +0.6364
cos(ref_new  , synth_old) = +0.3337
cos(ref_old  , synth_new) = +0.4608
cos(ref_old  , synth_old) = +0.5515
cos(synth_new, synth_old) = +0.5293
```

Graph:

```text
scripts/safe_graphify_update.sh
  -> REFUSED: node count dropped 28306 -> 2485 (~91.2%, threshold 10%).
     Restored. Nothing to commit.
```

That is the known, still-unroot-caused destructive-update bug firing again
and the wrapper containing it exactly as designed. Not caused by this
change and not investigated here.

## Review findings fixed

Code review dispatched in a subagent against this branch's diff. See the
follow-up commits on this branch for anything it surfaced.

## Restart required

Already performed in this session. To reproduce, or after a rollback:

```bash
cd <a worktree, not the shared checkout>
scripts/safe_docker_build.sh orion-whisper-tts up -d --force-recreate --no-build
curl -fsS http://localhost:7800/health
```

## Rollback

The previous reference is untouched on the host. To revert the voice:

```bash
# services/orion-whisper-tts/.env and .env_example
TTS_DEFAULT_SPEAKER_WAV=/models/voices/orion_reference.wav

scripts/safe_docker_build.sh orion-whisper-tts up -d --force-recreate --no-build
```

## Risks / concerns

- **Severity: low.** Concern: the reference `.wav` is a host asset outside
  git, so a fresh host or a rebuilt telemetry mount has no
  `orion_reference_v2.wav` and XTTS start-up validation
  (`_validate_xtts_defaults`) will fail on a missing speaker wav.
  Mitigation: unchanged from the pre-existing design and already documented
  in `.env_example` -- supply a reference at that path, or comment the key
  out to fall back to the built-in `TTS_DEFAULT_SPEAKER`. The source
  recording and exact build chain are now stored beside the asset, so the
  file is reproducible rather than irreplaceable.

- **Severity: low.** Concern: `services/orion-whisper-tts` has no `evals/`
  harness at all, so voice quality has no automated gate and nothing would
  catch a future reference swap that degrades cloning. Mitigation: none in
  this patch; proposed eval sketched in "Evals run" above.

- **Severity: medium (repo-wide, pre-existing).** Concern: the CLAUDE.md-
  prescribed env-parity command silently covers neither this service nor
  `TTS_*` keys, and prints a success-shaped message while checking nothing
  relevant. Any future agent following the contract literally would get a
  false green on this service's env parity. Mitigation here was to run the
  form that can actually fail, and to prove it can fail first. A real fix
  means widening `DEFAULT_SERVICES` (or deriving it from disk) and is out
  of scope for a voice change.

- **Severity: low.** Concern: two reference wavs now sit side by side and
  `_sources/` was added as a subdirectory of the read-only voice mount.
  Mitigation: `app/tts.py` only ever resolves explicitly-named paths and
  never enumerates the directory, so extra entries are inert.

## Status

DONE_WITH_CONCERNS -- see above; all concerns are pre-existing repo/harness
gaps rather than defects introduced by this change.
