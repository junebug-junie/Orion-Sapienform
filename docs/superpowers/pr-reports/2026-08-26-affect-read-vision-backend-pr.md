# feat(affect): replace AffectGPT with a VL read of the clip's own frames

## Summary

- Juniper reported that the affect read in her chat turns was "bullshit not
  able to infer garbage". Traced correlation_id `ddddfe40-da7d-4348-8df7-9a2d3f5ed4f6`
  end to end and found three independent defects, all confirmed live.
- Replaced the AffectGPT inference backend with a VL read of the clip's own
  frames through `orion-llm-gateway`'s `chat` route (circe:8011,
  Qwen3.6-35B-A3B), selected by a new `AFFECT_BACKEND` key.
- Output is now a structured `AffectReadV1`, rendered into a short prompt line,
  instead of the model's raw prose.
- Added two structural trust gates between "a read happened" and "a read may
  colour a chat turn". Every read is still published; only trusted ones mirror.
- Removed audio from the path entirely — the blob is no longer even fetched.
- Added the five telemetry columns whose absence made this undiagnosable
  without pointing Juniper's webcam at her twice.

## Outcome moved

**Before**, Orion's prompt for turn `ddddfe40` literally contained:

```
Juniper's affect (captured just now (no speech detected)): In the text, based on
the provided information, it is not possible to infer the character's emotional
state from the subtitle content. The subtitle content only describes the
character's facial expressions and body movements without providing any clues
related to emotions...
```

**After** (live run, same clip, real gateway, real model):

```
neutral and contemplative | valence -0.1, arousal 0.2 | cues: The person's gaze is
directed downwards and slightly away from the camera in all frames; The eyes are
partially closed or looking down... | confidence 0.85
```

## Investigation: what was actually wrong

Three things had to line up, and all three were true.

**1. AffectGPT answered from its text branch and ignored the face.** Its only
obtainable checkpoint is `multiface_audio_face_text`. Handed an empty subtitle
it refused — while holding a face crop with a **100% detection rate** (231/231
frames, `frames_carried_forward=0`, `frames_no_face_fallback_full_frame=0`).
Proven by A/B on two live captures: identical pipeline, only `user_message`
changed to "ignore the subtitle, use video and audio", and the same model
returned a grounded read instead of the refusal.

**2. The subtitle is empty by design on every chat-turn capture.** The clip is
recorded *after* Juniper stops speaking (`chat_turn_affect.py`'s own docstring),
so `subtitle_source="none"` is the normal case. Whisper was added specifically
to fix this and structurally cannot: there is nothing to transcribe.

**3. `ok=True` over meaningless content, with no input-quality gate.** The
second live capture scored `detection_rate=0.052` — 170 of 231 frames had no
detectable face — and AffectGPT still returned a confident "anger, frustration,
or sadness", citing "the acoustic characteristics of the individual's voice"
from an audio track measured (by pulling the wav out of percept-store and
running `volumedetect` on it) at:

```
max_volume: -49.2 dB    mean_volume: -71.6 dB    first-second RMS: -113.9 dB
```

That is a dead channel, not a quiet room.

**And the finding that settled the direction.** Every AffectGPT read in the
stored log that committed to anything called the subject **"the man" — 3 of 3**.
The refusals say "the character"; the moment it commits, it commits to male.
Juniper is female. A model asserting the wrong sex from a clean frontal crop is
not a cosmetic annoyance — it is evidence the face branch was not reading her at
all, and it is why this patch replaces the instrument instead of tuning its
prompt.

Fixing only the prompt would have been worse than doing nothing: the A/B capture
with the prompt fixed cheerfully narrated a voice from the silent track.

## Current architecture (before this patch)

```
carbon retina (8s clip: video+audio) -> percept-store
  -> orion-juniper-affective-state (circe): fetch BOTH blobs to shared scratch
  -> bus RPC -> orion-affectgpt-worker (circe GPU1, AffectGPT 7B + Whisper)
  -> JuniperMultimodalAffectV1 -> orion:affectgpt:assessment
  -> Redis mirror orion:juniper_affect:latest (any non-empty raw_response)
  -> orion/situational/context.py -> Orion's chat prompt
```

## Architecture touched

Only the inference hop moved. Capture, transport, schema, channel, mirror and
consumer are unchanged:

```
carbon retina -> percept-store
  -> orion-juniper-affective-state: fetch VIDEO ONLY
  -> sample 5 frames (cv2) + Haar detection_rate
  -> upload frames to percept-store -> AttachmentRefV1(kind="percept")
  -> bus RPC -> orion-llm-gateway (vision probe, route "chat", structured output)
  -> validated AffectReadV1
  -> JuniperMultimodalAffectV1 -> orion:affectgpt:assessment   [unchanged]
  -> mirror ONLY if confidence >= X AND detection_rate >= Y
  -> context.py renders a short line from the structured fields
```

Deliberately routed through the gateway rather than straight at circe:8011
(which would be a localhost hop from this service) because the gateway owns the
live `/props` vision-capability probe, the route table, and the "bytes enter at
the last possible moment" attachment resolution.

## Files changed

- `orion/schemas/affectgpt.py`: new `AffectReadV1`; `JuniperMultimodalAffectV1`
  gains `backend`/`affect`/`frames_used`, `source` widened to include `vision`.
- `orion/situational/juniper_affect_state.py`: mirror carries `confidence` and
  `backend`.
- `orion/schemas/situation.py`: `AffectContextV1` gains the same two.
- `orion/situational/context.py`: prompt line says "visual only" instead of "no
  speech detected" on the vision path, and hedges low-confidence reads.
- `services/orion-juniper-affective-state/app/vision_backend.py`: **new** —
  frames → percept-store → gateway RPC → validated `AffectReadV1`.
- `services/orion-juniper-affective-state/app/frame_sample.py`: **new** — frame
  selection plus the `detection_rate` quality signal.
- `services/orion-juniper-affective-state/app/main.py`: backend dispatch,
  `_mirror_decision`, `_render_affect_summary`, conditional audio fetch.
- `services/orion-juniper-affective-state/app/settings.py`: 9 new keys.
- `services/orion-sql-writer/app/models/juniper_multimodal_affect.py` +
  `app/main.py`: 5 new columns + idempotent DDL.
- `orion/bus/channels.yaml`: `orion-juniper-affective-state` declared as a
  producer on `orion:exec:request:LLMGatewayService`; provenance note updated.

## Schema / bus / API changes

- **Added:** `AffectReadV1`; `JuniperMultimodalAffectV1.{backend,affect,frames_used}`;
  `AffectContextV1.{confidence,backend}`; 5 SQL columns.
- **Renamed / removed:** none.
- **Behavior changed:** `source` Literal widened `["affectgpt"]` →
  `["affectgpt","vision"]`, default unchanged. The mirror write is now gated.
- **Compatibility:** additive. Every field defaults, so already-stored rows and
  in-flight events validate unchanged; a pre-existing Redis mirror payload
  (no `confidence`/`backend` keys) reads back with both `None`, and `None`
  confidence is explicitly *not* treated as low (pinned by a test).

## Env/config changes

- **Added keys** (9, all in `services/orion-juniper-affective-state/`):
  `AFFECT_BACKEND`, `CHANNEL_LLM_INTAKE`, `AFFECT_VISION_LLM_ROUTE`,
  `AFFECT_VISION_RPC_TIMEOUT_S`, `AFFECT_VISION_MAX_FRAMES`,
  `AFFECT_VISION_JPEG_QUALITY`, `AFFECT_VISION_MAX_TOKENS`,
  `AFFECT_MIRROR_MIN_CONFIDENCE`, `AFFECT_MIRROR_MIN_DETECTION_RATE`.
- **Removed / renamed:** none.
- `.env_example` updated: yes.
- `python scripts/sync_local_env_from_example.py`: ran, and **reported no
  changes for this service** — it reads `.env_example` from the *primary*
  checkout, so keys added in a worktree are invisible to it (known issue).
  The 9 keys were therefore appended to the live
  `services/orion-juniper-affective-state/.env` **by hand**, verified present.
- **Skipped keys requiring operator action:** none, but see Restart below —
  circe has its own clone and its own `.env`, and this session has no SSH
  access to circe (tailnet policy). Those 9 keys must be added there too.

## Metric quality gate (`detection_rate`, `confidence`)

1. **Provenance.** `detection_rate` = `frames_detected / frames_total` from
   `frame_sample.sample_frames()`, Haar cascade per decoded frame — same
   parameters as the retired `face_extract.py`, deliberately, so the number
   means the same thing across the cutover. `confidence` is the model's own
   field, validated by `AffectReadV1`.
2. **Independence.** They are causally unrelated: `detection_rate` is measured
   from pixels before any model call; `confidence` is generated after. Neither
   is a transform of the other. Both gate the same decision, which is the point
   — they fail in different ways (bad input vs uncertain model).
3. **Theory anchor.** Not a vibe: both thresholds exist to separate the two
   observed live failure modes. `detection_rate` catches "the frames had no
   face in them"; `confidence` catches "the model was guessing".
4. **Live-data sanity.** Real captures produced `1.0` and `0.052` (not
   degenerate, and it genuinely reaches both ends); a synthetic noise clip
   produces exactly `0.0`, and the eval pins that. `confidence` came back
   `0.85` on a real clip. **Caveat stated plainly: n=2 for detection_rate and
   n=1 for confidence.** The defaults (0.15 / 0.35) are starting points, which
   is exactly why they are env keys and not constants.
5. **Existing mechanism.** `face_detection` telemetry already existed on the
   bus — it was simply never persisted and never gated on. This uses it rather
   than inventing a new signal.
6. **Reversibility.** Cheap. Both are env keys read live (pinned by
   `test_thresholds_are_read_live_not_captured_at_import`); the whole backend
   reverts with one key.

## Tests run

```
pytest services/orion-juniper-affective-state/tests -q
  -> 81 passed

pytest services/orion-cortex-exec/tests/test_situation_affect_context.py -q
  -> 21 passed

pytest orion/situational/tests services/orion-sql-writer/tests -q
  -> 11 failed, 448 passed, 3 skipped
     ALL 11 failures are pre-existing on main (verified by running the same
     suite on the primary checkout: identical 11). No new failures.
```

New coverage worth naming:

- `test_prompt_never_mentions_a_subtitle_when_there_is_no_transcript` — the
  direct regression test for turn `ddddfe40`.
- `test_backend_selection_fails_closed_to_vision` — a typo must not select the
  backend that misgendered her.
- `test_low_detection_rate_read_is_not_mirrored` — the `0.052` capture.
- `test_write_row_keeps_the_quality_telemetry_it_used_to_drop` — exercises the
  actual column-filter that ate `face_detection`.
- `test_vision_backend_says_visual_only_never_no_speech_detected` — "no speech
  detected" claims we listened; on this path we never did.

## Evals run

```
ORION_BUS_URL=redis://100.92.216.81:6379/0 \
  pytest services/orion-juniper-affective-state/evals -q
  -> 6 passed, 1 skipped

  (skip = the pre-existing affectgpt eval; no live worker reachable from athena)
```

Against the **live model**, not a mock. Including
`test_model_never_asserts_gender_or_identity`, which is the check that would
have caught the replaced backend and which a prompt-only assertion cannot make.

## Docker/build/smoke checks

Live end-to-end smoke of the **shipped module** (not a re-implementation) —
real bus, real gateway, real clip, only the retina capture leg substituted:

```
OK=True
affect      : {"valence": -0.1, "arousal": 0.2,
               "primary_affect": "neutral and contemplative",
               "cues": [4 specific visual observations],
               "confidence": 0.85, "cannot_tell": []}
frames_used : 5
face_detect : {'frames_total': 231, 'frames_detected': 231,
               'detection_rate': 1.0, 'frames_sampled': 5}
timings     : {'sample_s': 4.31, 'upload_s': 0.166,
               'generate_s': 2.661, 'total_s': 7.139}
model       : /models/gguf/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf
```

Docker build/deploy **not run**: this service is deployed on circe and this
session has no SSH access there (tailnet policy blocks it).

## Review findings fixed

Code review ran in a subagent and returned 14 findings. All material ones are
fixed in the follow-up commit; the two nits left standing are recorded below
with reasons.

- **3.1 (should-fix, most serious): unbounded frame retention in
  `frame_sample.py`.** Pass 1 retained every decoded frame so the chosen
  indices could be picked at the end. Neither term is bounded —
  `RETINA_CLIP_WIDTH`/`HEIGHT` default to the device default, so a 1080p webcam
  is ~6.2MB/frame, and the live capture had 231 frames: **~1.4GB peak RSS on a
  path that fires every spoken chat turn.**
  - Fix: split into `_scan` (Haar, retains indices only) and `_collect`
    (re-decodes, retains only the chosen frames). Peak retention is now
    `max_frames` regardless of clip length or resolution. Sequential re-read
    rather than `CAP_PROP_POS_FRAMES` seeking, which lands on the nearest
    keyframe on some backends and would silently return a different frame than
    the one Haar judged.
  - Evidence: 10 frame tests pass; live smoke unchanged at **7.08s** vs 7.14s
    before, `detection_rate` still 1.0 (231/231).

- **2.6 (should-fix): an empty `primary_affect` reached Orion's prompt.**
  `Field(..., max_length=64)` accepted `""`, and `_render_affect_summary`
  filtered the empty part out, so a read could render as
  `Juniper's affect (read just now, visual only): valence +0.0, arousal 0.0 |
  confidence 0.90` — label-less, cue-less, exactly neutral, high confidence,
  and indistinguishable from a genuine calm read. Exactly the empty-shell
  failure this patch exists to stop, reintroduced through a side door.
  - Fix: `min_length=1` plus a strip-validator (min_length alone accepts
    `"   "`). Evidence: `test_blank_primary_affect_cannot_reach_the_prompt`.

- **2.1 (should-fix): failure rows mis-attributed to the wrong backend.** Both
  early-return branches in `capture_and_assess` fell through to
  `_wrap_event`'s `backend="affectgpt"` default, so a retina outage under
  `AFFECT_BACKEND=vision` persisted a row blaming a backend that was never
  invoked — defeating the stated reason the column exists.
  - Fix: `backend_name` bound once and threaded into both branches. Evidence:
    `test_capture_failure_is_attributed_to_the_selected_backend`.

- **2.2 (should-fix): the record contradicted the privacy claim.** `input_ref`
  wrote an `audio_path` for a file that was never downloaded, written, or read
  on the vision branch.
  - Fix: `audio_path` omitted entirely on that branch. Evidence:
    `test_vision_backend_records_no_audio_path_in_the_event`.

- **2.4 (should-fix): the kill switch was partially inert.**
  `POST /v1/juniper/affect/trigger` called `trigger_assessment()`
  unconditionally, so with `AFFECT_BACKEND=vision` set it *still ran AffectGPT*
  — the backend that misgendered her.
  - Fix: the route now dispatches on the same selector as `/capture_and_assess`.

- **2.5 (should-fix): the detection-rate gate failed open.** A missing, `None`,
  string, or bool `detection_rate` skipped the gate entirely and the read was
  mirrored.
  - Fix: resolved per backend. `vision` always supplies a float, so its absence
    means something broke → **fail closed**; `affectgpt` rows legitimately
    predate the field → fail open, as before. Evidence: three new tests,
    including bool/string/None cases.

- **1.1 (should-fix): `decoded.envelope` unchecked**, so an ok-but-envelope-less
  decode raised `AttributeError` and got reported as `vision_unexpected`,
  destroying the stable error code on exactly the failure an operator triages.
  Fix: check `decoded.envelope` and that `msg` is a dict, matching the sibling
  RPC path in the same service.

- **1.2 (nit, taken anyway): no non-empty guard on `attachments`.** Unreachable
  today, but would have sent a request with zero images and a prompt reading
  "0 webcam stills… read their affect". Fix: explicit `no_frames` error, so the
  empty-shell rule is structural here rather than a consequence of a guard in a
  different module.

- **2.3 (nit): the backend flag was read twice** per capture. Fix: bound once.

- **4.1 (nit): `isinstance(x, (int, float))` accepts `bool`,** so
  `"confidence": true` read back as 1.0 — maximum confidence from a value that
  expressed none. Fix: explicit bool exclusion.

- **4.2 (nit): `", visual only"` was asserted even when a caller supplied a real
  transcript,** which the vision backend *does* pass to the model. Fix:
  condition now excludes `subtitle_source == "caller"`.

- **7.1 (should-fix): no coverage of `capture_and_assess` on the vision path.**
  Notably, **deleting the conditional that skips the audio fetch broke no test**
  — the headline privacy claim was unpinned. Fix: 5 new tests, including
  `test_vision_backend_never_fetches_the_audio_blob`.

- **7.2 (nit): `"tone"` in the eval's audio-word list** would fail on correct
  reads ("muscle tone", "skin tone"). Fix: removed, along with `"pitch"` (a head
  can pitch forward); `"audible"`/`"vocal"` added instead.

**Left standing, deliberately:**

- **5.1 — deploy ordering, not a code defect.** `JuniperMultimodalAffectV1` is
  `extra="forbid"`, so a *new* producer against an *old* `orion-sql-writer`
  fails validation on the whole event and diverts it to `bus_fallback_log` —
  rows lost while both services look healthy. Not fixable in code without
  weakening the schema; handled as an explicit **deploy-order requirement**
  below. Good catch, and it changes the restart section.
- **The `app.main` import collision** between
  `services/orion-juniper-affective-state/tests` and
  `services/orion-cortex-exec/tests` is pre-existing and unrelated; the two
  suites must be run in separate pytest processes. Noted, not fixed here.

## Restart required

> **ORDER MATTERS — `orion-sql-writer` FIRST, producer second.** Review finding
> 5.1: `JuniperMultimodalAffectV1` is `extra="forbid"`, so a new producer
> emitting `backend`/`affect`/`frames_used` at an old sql-writer fails
> validation on the *entire* event and diverts every affect row to
> `bus_fallback_log` — silently, with both services reporting healthy.

**1. athena — sql-writer first** (the new-column DDL runs at its startup):

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
scripts/safe_docker_build.sh orion-sql-writer up -d --build
```

**2. circe — the producer**, after adding the 9 new keys to that host's
`services/orion-juniper-affective-state/.env`:

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
scripts/safe_docker_build.sh orion-juniper-affective-state build
scripts/safe_docker_build.sh orion-juniper-affective-state up -d
curl -fsS http://100.112.254.99:32799/health
```

**3. carbon — retina**, for `want_audio` (without this the mic still arms for
every affect capture, and the vision read simply gets no transcript):

```bash
cd /mnt/scripts/Orion-Sapienform
git pull --ff-only
scripts/safe_docker_build.sh orion-vision-retina up -d --build
```

**4.** `orion-cortex-exec` and `orion-hub` also carry the changed
`orion/situational/context.py` and must be rebuilt for the new prompt line:

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- **Severity: medium.** *Thresholds are calibrated on n=2.* `0.15` /
  `0.35` are honest starting points, not tuned values. Too high and Orion goes
  quiet about affect; too low and a bad read gets through. Both are live-read
  env keys; revisit once real rows accumulate — which is now possible, because
  `face_detection` is finally persisted.
- **Severity: medium.** *Carbon's clip audio is still broken and this patch
  does not fix it* — it removes the dependency. Carbon's default PulseAudio
  source is the built-in Digital Microphone (`hw:sofhdadsp,6`) at `vol: 0.50`,
  measuring ~21 dB colder than the headset mic on identical ambient captures.
  **Outstanding test:** record from both sources while Juniper speaks a known
  sentence. Until then "is the mic wrong, or was she just quiet" is unresolved.
- **Severity: low.** *`sample_s` is ~4.3s* — Haar over all 231 frames dominates
  the non-generation cost. It is off the critical path (the capture is
  fire-and-forget and never blocks a turn), and scanning every frame is what
  makes `detection_rate` a real measurement rather than an estimate. Noted
  rather than optimised.
- **Severity: low.** *The affectgpt rollback path is retained.* A bounded
  exception to "kill means kill" — bounded in the way that rule cares about:
  nothing falls back automatically, a vision failure publishes `ok=False`, and
  only a human editing `AFFECT_BACKEND` selects it. Should be deleted outright
  once the vision path has a week of real rows.
- **Unrelated, flagging not fixing.** During this session
  `services/orion-cortex-exec/.env_example` appeared modified in the *shared*
  checkout, adding `ORION_SITUATION_IDENTITY_ASK_COOLDOWN_SECONDS` and
  referencing `orion/situational/identity_ask_cooldown.py`, which does not
  exist on main. Not mine and not in this branch — looks like a concurrent
  session's in-flight work. Left untouched.

## Follow-up: one recording, not two (2026-08-26, same branch)

Juniper, after watching a live turn: *"the mic record button should be merged
with the affect recording so you don't get two divorced audio recordings where
I have to repeat myself."*

She was right, and turn `7dc1bab2` proved how bad the second recording is. She
said, through the browser mic:

> **"I'm feeling really tired."**

The affect clip's own Whisper pass, on carbon's DMIC, produced:

| leg | transcript |
| --- | --- |
| pre | `"Tired, tired, tired"` |
| post | `"Thanks for the light, Egyptians. Thanks for the eyesight, thanks for the thanks, this was a long time ago."` |

The post-leg text is fabricated outright — Whisper's notorious "thanks for
watching" family, emitted from a noise floor. The pre-leg text is one real word
degraded into a repetition loop. **AffectGPT then anchored on the fabrication**
("the subtitle content 'Tired, tired, tired' is likely the man expressing his
emotional state"), so the deployed system has been producing affect reads
grounded in sentences she never said — a worse failure than the refusal that
started this investigation.

This also refines the earlier "dead channel" call in this report: carbon's DMIC
is **not** electrically dead — it does pick up faint speech. It is ~21 dB too
quiet at `vol: 0.50` for Whisper to decode reliably. A gain/device problem, and
more fixable than "-49.2 dB is a dead channel" implied.

**The fix: the microphone is never armed for an affect capture, and the
transcript Hub already holds is threaded in instead.**

- `RetinaClipCaptureRequestPayload.want_audio` (default `True`, for the
  affectgpt rollback which still Whispers its own wav). `False` opens no pulse
  stream at all — "never armed", not "armed and discarded", so the OS recording
  indicator does not light.
- retina skips the audio percept upload rather than POSTing `b""`
  (percept-store returns 400 on an empty body, which would turn a correct
  video-only capture into a hard failure). `audio_sha256` comes back `None`, so
  a caller can distinguish "no mic opened" from "mic opened, got something".
- Hub passes the real transcript on the **PRE leg only**. POST gets none
  deliberately: she is not speaking then, and reusing the pre leg's text would
  present her opening words as her reaction to Orion's reply.

Net effect: exactly one recording of her voice exists in the system, it is the
good one, and she never repeats herself.

### Two real bugs found while building it

- `websocket_handler`'s call site read `subtitle=user_text`, but that name is
  **not bound** in `websocket_endpoint` — a `NameError` in the live turn path.
  The variable is `transcript`. Caught by walking the function's AST for actual
  bindings rather than trusting a grep that matched a different function's
  parameter.
- Six monkeypatched test fakes did not match the widened signatures. One of
  them **hung the entire hub suite instead of failing**: the fake raised
  `TypeError` before appending to `order`, so the test's
  `while "start:chat_turn_pre" not in order` spun forever. Worth naming because
  a hang reads as infrastructure flakiness, not as a broken contract.

### Verification

```text
services/orion-juniper-affective-state/tests                  92 passed
tests/test_vision_retina_clip_{rpc,capture,cooldown}.py       27 passed
services/orion-cortex-exec/.../test_situation_affect_context.py
  + orion/situational/tests                                   39 passed
services/orion-hub/tests/test_{chat_turn_affect,
  vision_affect_ambient,vision_affect_capture_api}.py         67 passed
                                                             ---
                                                             225 passed

all 8 static gates: PASS
```

Mutation-checked, per this branch's own earlier lesson: forcing
`want_audio=True` on the vision path turns
`test_vision_backend_asks_retina_not_to_arm_the_microphone` red.

**Not live-verified.** `want_audio` runs on carbon's retina, which serves the
currently-deployed image. Unlike circe, carbon IS reachable over SSH this
session, but deploying an unmerged branch to Juniper's laptop is her call, not
mine. The restart section below covers it.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1903
