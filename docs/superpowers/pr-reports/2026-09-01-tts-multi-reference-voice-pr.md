# Multi-reference speaker voice for whisper-tts

## Summary

- `TTS_DEFAULT_SPEAKER_WAV` (and `options.speaker_wav`, and `voice_id`) may now
  point at a **directory**; every `*.wav` inside is passed to XTTS together.
- This is the only way to use more than ~30s of a recording, because
  `max_ref_len` truncates **each reference independently**. Orion's voice has
  been running on 27.8s of a 100.8s clean take.
- XTTS **means the per-file speaker embeddings**, which averages out the
  per-clip codec artifacts a single clip bakes into the clone — the reported
  "roboty tinges" from an ~64-73 kbps source MP3.
- Single-reference behaviour is byte-identical: a bare string, not a
  one-element list.
- Natural sort order, and three new refusals (empty dir, the profile root
  itself, a symlink escaping the profile dir).

## Outcome moved

Orion's speaker embedding is built from 99.1s of reference audio across seven
loudness-matched chunks instead of a single 27.8s window. Prosody still comes
from the first ~30s (`gpt_cond_len`), so the change is isolated to the
embedding — which is the part that carries timbre.

## Current architecture

`services/orion-whisper-tts` runs Coqui XTTS-v2, zero-shot: the voice IS the
reference audio. `_resolve_speaker_wav_path()` resolved exactly one file and
`resolve_synthesis_plan()` passed a single string as `speaker_wav`, so no
config surface could hand XTTS more than one reference. Verified against the
installed TTS 0.22.0 source in the live container:
`Xtts.get_conditioning_latents` takes `audio_path` as `str | List[str]`,
truncates each entry to `max_ref_length`, `torch.cat`s them for
`get_gpt_cond_latents` (itself capped at `gpt_cond_len`), and
`torch.stack(speaker_embeddings).mean(dim=0)`.

## Architecture touched

`services/orion-whisper-tts/` only. No bus channel or schema change — the new
metadata keys ride in `TTSResultPayload.metadata`, which is `Optional[dict]`
(`orion/schemas/tts.py:22`).

## Files changed

- `services/orion-whisper-tts/app/tts.py`: `_resolve_speaker_wav_refs()`
  returning `list[Path]`; `_natural_key()`; `_speaker_wav_refs_exist()`;
  plan passes a list only when N > 1; new metadata `speaker_wav_count`,
  `speaker_wav_source`, `speaker_wav_basenames`.
- `services/orion-whisper-tts/tests/test_tts_multi_reference_voice.py`: new.
- `services/orion-whisper-tts/.env_example`: default flipped to the chunk
  directory; the Mode 2 preamble rewritten rather than appended beside.
- `services/orion-whisper-tts/README.md`: "Multi-reference" section.

## Schema / bus / API changes

- Added metadata keys: `speaker_wav_count` (int), `speaker_wav_source`
  (str|None), `speaker_wav_basenames` (list|None).
- Behavior changed: `speaker_wav_basename` names the **directory** under
  multi-reference rather than the first chunk; a directory candidate resolves
  instead of raising; `options.speaker_wav: "."` now raises instead of
  resolving to the profile root.
- Compatibility: single-file configs are unaffected in both call shape and
  metadata.

## Env/config changes

- Added keys: none.
- Changed: `TTS_DEFAULT_SPEAKER_WAV` `/models/voices/orion_reference_v2.wav`
  -> `/models/voices/orion_v3_chunks`.
- `.env_example` updated: yes.
- local `.env` synced: yes, with `--force` (the divergence was the intended
  change). Chunk directory staged on **both** hosts, md5-identical, so the
  contract is satisfiable where it is read.

## Tests run

```text
cd services/orion-whisper-tts && PYTHONPATH=. python -m pytest tests -q
82 passed      (was 59 on main)
```

Mutations against the real file, each reverted after:

```text
if resolved.is_dir()            -> if False and ...    6 failed
sorted(...)                     -> list(...)           1 failed (order test)
len(refs) > 1                   -> len(refs) > 0       7 failed, 3 of them PRE-EXISTING
p.resolve().relative_to(root)   -> pass                1 failed (symlink escape)
if resolved == root             -> if False            1 failed (profile root)
if not p.is_file(): continue    -> if False: continue  1 failed (dir named *.wav)
key=_natural_key                -> (removed)           1 failed (chunk_2 vs chunk_10)
_speaker_wav_refs_exist         -> old inline form     2 failed
```

The `len(refs) > 1` mutation is the important one: it was caught by three
**pre-existing** tests, so the single-reference call shape is pinned
independently of anything added here.

## Evals run

`services/orion-whisper-tts` has no `evals/` directory. Not added here.
The natural shape of one is the speaker-embedding similarity matrix used to
verify a voice change (README, "Verifying a voice change"); flagged as a
pre-existing gap, unchanged by this patch.

## Docker/build/smoke checks

```text
UNVERIFIED on the live path.
```

There is no whisper-tts container on athena — the service runs on circe, whose
image was built 2026-08-30T01:27, i.e. before #1981 merged. This patch is
source-verified and unit-tested only. Per the service's own README the engine
is constructed lazily and `/health` never touches it, so a green `up -d` would
prove nothing either. After merge, on circe:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
scripts/safe_docker_build.sh orion-whisper-tts up -d --build
```

then round-trip a synthesis over `orion:tts:intake` and confirm the reply
metadata reports `speaker_wav_count: 7` and `speaker_wav_basename:
orion_v3_chunks`, and record single-vs-multi synthesis latency (see Risks).

## Review findings fixed

Ran the code-review skill in a subagent. Two must-fix, both reproduced before
fixing:

- Finding: a **symlink inside** the reference directory escaped
  `tts_voice_profile_dir`. `glob` does not resolve symlinks and `is_file()`
  follows them, so the candidate-level `relative_to(root)` check did not cover
  children. A containment regression introduced by this patch's own branch.
  - Fix: per-file `p.resolve().relative_to(root)`.
  - Evidence: reproduced against the real function — a planted link returned
    `/tmp/.../secret/private.wav` from a call rooted at `/tmp/.../voices`.
    Now `test_symlink_inside_the_directory_cannot_escape`.

- Finding: the diagnostic log read `Path(str(speaker_wav)).is_file()`, which
  under multi-reference is `Path(str(<list>))` — always `False`, so every
  multi-reference synthesis logged `speaker_wav_exists=False` while the files
  were present. That line is also the first thing an operator reads in the
  failure handler. Above ~85 refs it was worse: the stringified list exceeds
  `NAME_MAX` and `is_file()` **raises** `ENAMETOOLONG` (not in pathlib's
  ignored errnos), which inside the `except` would replace the real synthesis
  error.
  - Fix: `_speaker_wav_refs_exist()`, list-aware and non-raising.
  - Evidence: measured — `Path(str(list)).is_file() -> False` at n=7, and
    `OSError errno=36` at n=90. Both now covered.

Should-fix findings also addressed: `options.speaker_wav: "."` resolved to the
profile root and blended every voice on the host (now refused); lexicographic
sort put `chunk_10` second and the README taught exactly that naming (now
natural sort, plus tests for both padded and unpadded); `speaker_wav_basename`
reported `chunk_1.wav` so a 7-file voice read as single-file (now the
directory, with `speaker_wav_basenames` alongside); `.env_example`'s Mode 2
preamble still said "this one .wav IS the voice" and named a stale rollback
target three lines above the new directory default (rewritten); and a
directory literally named `something.wav` was filtered at runtime but untested
— that mutation survived the first suite, and now does not.

## Restart required

```bash
# on circe, after this merges
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
scripts/safe_docker_build.sh orion-whisper-tts up -d --build
```

A rebuild, not a restart: this is a code change, and `docker compose restart`
would not pick up the `.env` change either. The same rebuild finally lands
#1981 on circe, whose image predates that merge.

## Risks / concerns

- Severity: medium. Concern: **unmeasured per-request cost.** Conditioning
  latents are recomputed on every request and nothing caches them (README),
  and the directory is re-globbed per request, so this multiplies reference
  decode + speaker-encoder work by 7 on every turn Orion speaks.
  `WHISPER_TTS_SYNTH_TIMEOUT_SEC=120` gives headroom against a ~2.9s baseline,
  but the real number is not yet known. Mitigation: measure on the first live
  round-trip and record it; rollback is one env value.
- Severity: low. Concern: re-globbing per request means the voice can change
  the moment a file is added to the directory, with no log line at that
  instant — only the per-request `speaker_wav_count` records it after the
  fact. Mitigation: the directory is a `:ro` mount inside the container;
  changing it requires host access.
- Severity: low. Concern: a fresh clone has no `orion_v3_chunks/`. No worse
  than before — the previous default file was equally absent — and
  `.env_example` documents supplying your own asset or commenting the line
  out.

## PR link

<filled in on push>
