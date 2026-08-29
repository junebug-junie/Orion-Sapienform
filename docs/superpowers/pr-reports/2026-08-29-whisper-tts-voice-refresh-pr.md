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

Each synthesis is closest to its own reference: the config change genuinely
moved the output, and each synthesis tracks the reference it was given.
**That is the whole of what this supports.**

The control row is what makes even that readable. `ref_new`/`ref_old` is
itself +0.6498 -- the two references are not far apart in this space -- so
"synth scored +0.64 against the new reference" on its own would have proven
nothing.

**Retracted (review finding S2).** An earlier revision of this report and of
the README claimed the new reference "clones better" than the old, on the
grounds that +0.6364 > +0.5515. That claim is not supported and has been
removed everywhere it appeared, including the host provenance file: each
cell is **n=1** against a stochastic decoder (`full_inference(...,
do_sample=True)`) with no spread reported; it compares magnitudes *across*
references, which vary with intrinsic properties of a reference
independently of clone quality; and the control row actively undercuts it,
since `ref_new`/`ref_old` (+0.6498) is *higher* than `synth_new`/`ref_new`
(+0.6364). Earning a quality comparison needs k draws per cell and a
reported spread. I attempted exactly that during review follow-up and
aborted it -- see the GPU incident below.

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
- `services/orion-whisper-tts/tests/test_tts_voice_resolution.py`: two new
  precedence tests (review finding S1).
- `scripts/sync_local_env_from_example.py`: cover this service and its key
  prefixes (review finding S5). The only file outside the service.

Host assets, correctly NOT in git:

- `/mnt/telemetry/models/coqui/voices/orion_reference_v2.wav` (new live
  reference, 27.8s, 24kHz mono `pcm_s16le`, -29.1 LUFS)
- `/mnt/telemetry/models/coqui/voices/orion_reference.wav` (retained,
  untouched, rollback target)
- `/mnt/telemetry/models/coqui/voice_sources/` (source recording +
  `orion_reference_v2.provenance.txt` recording the exact chain, so the
  reference is reproducible rather than a mystery binary). A **sibling** of
  the voice dir, not a child: anything under the bind mount is reachable as
  a caller-supplied `options.speaker_wav` (review finding N1).

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

1. `orion-whisper-tts` was not in the script's hand-maintained
   `DEFAULT_SERVICES` list (24 of the 93 service dirs on disk -- an earlier
   revision of this report said "12 of 92" from a truncated read of the
   tuple; corrected).
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

**Now fixed in this patch** (review finding S5). CLAUDE.md's "deterministic
gates over repeated yelling" rule applies squarely: the right fix for a
gate that cannot see a service is not a louder PR note. Two changes to
`scripts/sync_local_env_from_example.py` -- add `orion-whisper-tts` to
`DEFAULT_SERVICES`, and add `TTS_` / `STT_` / `WHISPER_TTS_` /
`CUDA_WATCHDOG_` to `SYNC_PREFIXES`.

Blast radius checked before keeping it: only `orion-hub` and
`orion-whisper-tts` carry any of those prefixes, and a full `--dry-run`
across every default service produces output **identical** to the
pre-change run (no new divergences, no `Would update` lines).

Verified end-to-end rather than by inspection -- temporarily pointed the
worktree's `.env_example` at a deliberately wrong path and confirmed the
**bare** invocation now reports it:

```text
orion-whisper-tts: TTS_DEFAULT_SPEAKER_WAV
  local='/models/voices/orion_reference_v2.wav'
  example='/models/voices/DELIBERATELY_WRONG.wav'
```

(reverted immediately). Before this change the same command printed "No
changes needed". The follow-on improvement -- making "no changes needed"
distinguish "0 keys matched the filter" from "all keys in sync" -- is a
larger fix and stays a follow-up.

## Tests run

The suite only runs from the service directory with `PYTHONPATH` set; from
the repo root it fails at collection with
`ModuleNotFoundError: No module named 'app'` (also true on `main`).

```text
cd services/orion-whisper-tts && PYTHONPATH=. python -m pytest tests -q

  baseline on main       : 2 failed, 53 passed
  this branch            : 57 passed
```

Mutation check on the two tests added for review finding S1 -- a green from
a new test is not evidence until it has been seen to fail. Disabling the
precedence branch (`elif cfg.tts_default_speaker_wav:` -> `elif False and
...`) in `app/tts.py`:

```text
FAILED tests/test_tts_voice_resolution.py::test_default_speaker_wav_from_settings
FAILED tests/test_tts_voice_resolution.py::test_default_speaker_wav_beats_voice_id
2 failed, 7 passed
```

`app/tts.py` restored and confirmed clean (`git status` shows no change
under `app/`).

The 2 baseline failures were pre-existing and unrelated: `BaseEnvelope.
correlation_id` is now a strict UUID, but `test_tts_worker_replies.py`
still passed `"cid-1"` / `"cid-legacy"`. Confirmed failing identically on
`main` with the same invocation before touching them. Neither test asserts
on the correlation_id value, so valid UUIDs change nothing about coverage.

Env-parity and compose-parity:

```text
# after the S5 gate fix, the BARE invocation now covers this service:
python3 scripts/sync_local_env_from_example.py orion-whisper-tts
  -> reading live .env from primary checkout: /mnt/scripts/Orion-Sapienform
  -> No changes needed.        (instrument proven able to fail; see above)

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

Subagent code review against this branch's diff. It verified the load-bearing
technical claims independently (zero-shot; the 30s conditioning window; that
truncation is from the *start* of the file, which is what makes step 1's
single-speaker window advice correct; that `sound_norm_refs=false` in the
shipped checkpoint means absolute level really does reach the speaker
encoder, so the loudness-matching step is load-bearing rather than
ceremony). All findings below are fixed.

- **S1 — `voice_id` is inert under the config this PR installs, and two
  README examples documented the impossible.** The resolve order is
  `options.speaker_wav` > `TTS_DEFAULT_SPEAKER_WAV` > `voice_id` >
  `TTS_DEFAULT_SPEAKER`, so the env key outranks a per-request `voice_id`,
  not merely `TTS_DEFAULT_SPEAKER` as the env table said. Payload examples
  #2 and #4 could not do what they claimed.
  - Fix: env table now states the full order; both examples carry an
    explicit "inert under the current live config" note. Added
    `test_default_speaker_wav_beats_voice_id` (the shipping combination was
    covered by nothing — every existing case set exactly one of the two) and
    `test_request_speaker_wav_beats_default_speaker_wav`.
  - Evidence: 57 passed. Mutation-checked rather than trusted — disabling
    the precedence branch in `app/tts.py` fails the new test; `app/tts.py`
    reverted clean.

- **S2 — "clones better" was an over-claim.** Fixed by retraction, in the
  README, this report, and the host provenance file. Details under "Outcome
  moved".

- **S3 — `.env_example` told the operator to "restart" for an `env_file`
  change.** `docker compose restart` reuses the baked environment, so a
  rollback would silently keep playing the old voice.
  - Fix: `.env_example` and the provenance file now say RECREATE, with the
    exact command and an explicit "restart will NOT work".

- **S4 — "env changes are read at boot" invited reading a green `up -d` as
  proof.** `TTSEngine` is built lazily (`app/tts_worker.py:22-25`), so
  `_validate_xtts_defaults` and its `is_file()` check never run at startup,
  and `/health` never touches the engine. A typo'd path yields a container
  that is Up and healthy and fails on Orion's first real turn.
  - Fix: explicit callout in README and `.env_example`. Confirmed against
    the source, and then observed live during the GPU incident below, where
    the first post-recreate request paid the model load.

- **S5 — the env-sync gate was a structural false green for all 26 keys of
  this service.** Fixed in this patch; see "Env/config changes".

- **S6 — the metadata key list omitted `speaker_wav_basename`,** the very
  field the new verification step tells an operator to check (and
  `gpu_enabled`). Fix: both added, cited to `app/tts.py:112-121` / `:245-249`.

- **N1 (privacy) — `_sources/` was placed INSIDE the read-only voice
  mount,** which put the raw source recording in the container and made it
  reachable as a caller-supplied `options.speaker_wav` (it passes
  `_resolve_speaker_wav_path`'s containment and `is_file()` checks). This is
  the "blocked material stays blocked" rule: the caller-reachable file set
  should be sanctioned references, not raw source material.
  - Fix: moved to `/mnt/telemetry/models/coqui/voice_sources/`, a sibling of
    the mount rather than a child. `voices/` now contains exactly the two
    reference wavs.

- **N2 — the `~30s` figure comes from the shipped checkpoint's
  `config.json`, not the library.** Installed TTS 0.22.0's class defaults
  are `gpt_cond_len=12` / `max_ref_len=10`, so a future reader checking the
  library would conclude the doc was wrong. Fix: stated explicitly, together
  with the fact that truncation is from the start of the file.

- **N3 — the build recipe existed in three places and had already drifted**
  (S3 is that drift). Fix: `.env_example` trimmed from 34 to 24 lines,
  keeping only zero-shot / ~30s / current file / rollback / lazy-init, and
  pointing at the README as the single source.

- **N4 — no "never overwrite in place" rule.** Latents are recomputed per
  request with no cache, so overwriting an installed `.wav` takes effect
  immediately with no restart, no config change and no trace, silently
  destroying the rollback target this procedure promises. Fix: step 4 now
  says install under a NEW filename and states why.

- **N6 — "fallback only while `TTS_DEFAULT_SPEAKER_WAV` is set" scans
  backwards.** Fix: "used only when `TTS_DEFAULT_SPEAKER_WAV` is unset".

Accepted, not actioned:

- **N5** — the rollback target is unversioned host state no gate can see.
  Said out loud in Risks below; there is no in-repo fix.
- **N8** — `check_service_env_compose_parity.py` answers "N/A" for any
  service declaring `env_file:`, but `environment:` *overrides* `env_file`,
  and `docker-compose.yml:58` has
  `TTS_DEFAULT_SPEAKER_WAV=${TTS_DEFAULT_SPEAKER_WAV:-}` — exactly the trap
  that compose file's own comment (`:66-76`) says it avoided for
  `CUDA_WATCHDOG_*`. Benign today because `safe_docker_build.sh` passes both
  `--env-file`s and the root `.env` has no `TTS_` keys, but a repo-root
  invocation with only `--env-file .env` would blank the key and silently
  fall back to `Ana Florence`. A separate gate bug, not this PR's to fix,
  and the more valuable half of it is that the gate *declares* N/A here.
- CLAUDE.md §17's `make agent-check` chain is partly fictional —
  `scripts/check_env_template_parity.py` does not exist, and `Makefile:46-52`
  already documents this. Not in scope.

## Incident caused during review follow-up (self-inflicted, resolved)

Recorded because it was mine and it took Orion's voice down for ~25 minutes.

Trying to answer S2 properly, I fired **10 synthesis requests in parallel**
over the bus to get k draws per cell. GPU 0 is a 7.68 GB Tesla P4 shared by
three services (`vision-host` 4892 MiB, `whisper-tts` ~2454 MiB,
`vision-edge` 248 MiB) — roughly 86 MiB of headroom. The concurrent
allocations OOM'd:

```text
RuntimeError: Coqui synthesis failed ... CUDA out of memory.
Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 7.42 GiB
of which 3.25 MiB is free.
```

Afterwards even single requests would not complete, because the service's
allocator state stayed wedged against a full device. Resolved by recreating
the container to release its ~2.4 GB:

```text
scripts/safe_docker_build.sh orion-whisper-tts up -d --force-recreate --no-build
  -> GPU 7597 MiB -> 5143 MiB used
  -> synthesis OK: synthesis_ms=3956, speaker_wav_basename=orion_reference_v2.wav
```

Voice is confirmed working. Two real findings fall out, neither introduced
by this patch:

- **The service has no concurrency guard.** `process_tts_request` synthesizes
  on demand with nothing serialising GPU work, so N concurrent bus requests
  contend for a device with ~86 MiB spare and each fails with a hard
  `RuntimeError`. If Hub ever issues parallel TTS, this is reachable in
  production without an agent doing anything unusual.
- **GPU 0 is effectively fully subscribed.** ~86 MiB free across three
  services means TTS has no headroom for a transient. Related history:
  `docs/superpowers/pr-reports/` on the 2026-08-21 vision-host VRAM outage.

Also worth naming: the CUDA watchdog did **not** fire, correctly —
`torch.cuda.is_available()` stayed true throughout. It watches for device
staleness, not exhaustion, so an OOM-wedged TTS is invisible to it.

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

- **Severity: medium (repo-wide, pre-existing, now partly fixed).**
  Concern: the CLAUDE.md-prescribed env-parity command printed a
  success-shaped message while considering zero of this service's 26 keys.
  Fixed for this service (see Env/config changes) and verified end-to-end.
  **The general defect remains:** `DEFAULT_SERVICES` is still hand-maintained
  at 25 of 93 service dirs, and `should_sync_key`'s prefix allowlist is still
  opt-in, so 68 services keep the same silent-false-green behaviour. Two
  durable fixes worth doing: derive the service list from disk, and make the
  final message distinguish "0 keys matched the filter" from "all keys in
  sync". Not attempted here — that is a gate rewrite, not a voice patch.

- **Severity: medium (pre-existing, found the hard way).** Concern: TTS has
  no concurrency guard and GPU 0 has ~86 MiB of headroom across three
  services, so concurrent synthesis requests OOM and leave the service
  unable to synthesize until the container is recreated. I triggered this
  myself during review follow-up (see the incident section). The CUDA
  watchdog does not cover it — it detects device staleness, not exhaustion.
  Mitigation: none in this patch. Worth a follow-up: serialise synthesis
  behind a semaphore, and/or register whisper-tts with
  `orion-mesh-guardian` (the README already names this gap for the
  restart-loop case).

- **Severity: low.** Concern: two reference wavs now sit side by side and
  `_sources/` was added as a subdirectory of the read-only voice mount.
  Mitigation: `app/tts.py` only ever resolves explicitly-named paths and
  never enumerates the directory, so extra entries are inert.

## Status

DONE_WITH_CONCERNS. All review findings are fixed. The remaining concerns
are pre-existing repo/harness gaps (the env-sync gate's other 68 services,
the absent eval harness, TTS concurrency and GPU 0 headroom) rather than
defects introduced by this change -- with the honest exception of the
self-inflicted GPU incident above, which was caused by my own verification
attempt, is resolved, and left the service confirmed working.
