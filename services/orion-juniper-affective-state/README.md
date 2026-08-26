# orion-juniper-affective-state

Thin CPU orchestrator in front of whichever affect backend is selected. Two
ways in:

- `POST /v1/juniper/affect/trigger` — given an already-written video+audio
  pair, does a real bus RPC round-trip to the worker.
- `POST /v1/juniper/affect/capture_and_assess` (2026-08-22) — the live
  path: bus RPC to `orion-vision-retina` (carbon) for a fresh clip, fetch
  both blobs from `orion-percept-store`, then the same worker round trip.
  This is what Hub's "Check now" button AND its ambient toggle both call
  (`{"trigger": "manual"}` vs `{"trigger": "ambient"}` in the request body).

Both wrap the result as `JuniperMultimodalAffectV1` and publish it to
`orion:affectgpt:assessment` — one event stream regardless of entry point.

**Deployed on circe, not athena.** `video_path`/`audio_path` fed to the
worker are resolved on the *worker's* filesystem, and circe/athena share no
filesystem (`/mnt/telemetry` is athena-local ext4, no NFS/exports;
`/mnt/scripts` is a separate per-host clone, not synced — see
`reference_circe_gpu_inventory_and_lane_map`). Co-locating sidesteps that
gap for the worker call; see `app/settings.py`'s `NODE_NAME` comment.

## The AffectGPT backend was replaced (2026-08-26)

`AFFECT_BACKEND` selects the inference path. Default `vision`.

**What broke.** Chat turn `ddddfe40` put this into Juniper's own prompt:

> Juniper's affect (captured just now (no speech detected)): In the text,
> based on the provided information, it is not possible to infer the
> character's emotional state from the subtitle content...

Three independent defects had to line up for that, all confirmed live against
`juniper_multimodal_affect_log` and two instrumented captures:

1. **AffectGPT answered from its text branch and ignored the face.** Its only
   obtainable checkpoint is `multiface_audio_face_text`. Handed an empty
   subtitle it refused — while holding a face crop with a **100% detection
   rate** (231/231 frames). The face was right there and it never looked.
2. **The subtitle is empty by design on every chat-turn capture.** The clip is
   recorded *after* Juniper stops speaking (see `chat_turn_affect.py`), so
   `subtitle_source="none"` is the normal case, not an edge case. Whisper was
   added to fix exactly this and cannot: there is nothing to transcribe.
3. **`ok=True` over meaningless content, and no input-quality gate.** A second
   capture scored `detection_rate=0.052` — 170 of 231 frames contained no
   detectable face — and AffectGPT still returned a confident "anger,
   frustration, or sadness", citing "the acoustic characteristics of the
   voice" from an audio track measured at **-49.2 dB peak**, i.e. silence.

And the finding that settles it: **every AffectGPT read that committed to
anything called the subject "the man" — 3 of 3 in the stored log.** Juniper is
female. A model asserting the wrong sex from a clean frontal crop is not a
cosmetic quirk; it is evidence the face branch was not reading her at all.

**What replaced it.** A VL read of the clip's own frames through
`orion-llm-gateway`'s `chat` route (circe:8011, Qwen3.6-35B-A3B). Same clip,
same frame:

> Eyes slightly narrowed or half-lidded... mouth closed in a straight line...
> best described as neutral-to-mildly-negative, with cues pointing toward
> fatigue, boredom, or emotional restraint.

Specific, hedged, and it asserted no gender at all. ~7-10s end to end.

**Five design calls, and why:**

1. **Five frames, not one.** Affect is temporal — a still cannot distinguish
   an expression settling from one tightening.
2. **Full frames, not face crops.** Reverses AffectGPT's approach, which
   cropped only because its checkpoint was trained on OpenFace crops. A
   general VL model cites posture, gaze and head tilt, none of which survive a
   tight crop. Haar still runs — it gates trust, it no longer picks pixels.
3. **Structured output** (`AffectReadV1`), not prose. The prompt line is now
   rendered from fields. Raw prose in a prompt is what broke this.
4. **Identity inference is forbidden in the system prompt**, first rule, not a
   trailing caveat.
5. **Two trust gates on the mirror write.** Every read is still *published*;
   only reads clearing `AFFECT_MIRROR_MIN_CONFIDENCE` and
   `AFFECT_MIRROR_MIN_DETECTION_RATE` reach the Redis key
   `orion/situational/context.py` polls. Below either, Orion gets "no recent
   capture; do not infer" — the honest line. Structural gates, not a regex
   hunting for hedging phrases.

**Audio is gone from this path entirely.** Not optional — gone. The audio blob
is no longer even fetched from percept-store on the vision path, so Juniper's
recorded voice never crosses a host boundary. `subtitle_source` can only be
`caller` or `none` here, never `transcribed`.

**Rollback.** `AFFECT_BACKEND=affectgpt` restores the old path. Nothing falls
back to it automatically — a vision failure publishes `ok=False`; only a human
editing that key selects it. Backend selection otherwise fails closed *to*
vision, including on a typo.

**Still open:** the clip's audio capture on carbon is genuinely broken and this
patch does not fix it (it removes the dependency on it). Carbon's default
PulseAudio source is the built-in Digital Microphone at `vol: 0.50`, which
measures ~21 dB colder than the headset mic on identical ambient captures. See
the PR report for the outstanding test.

## Ambient mode exists now, but it does not live here (2026-08-22)

Both entry points above are still single, explicit calls with no scheduling
logic of their own inside THIS service — every request here is still one
caller asking for one attempt. Recurring capture is real, though: Hub owns
that loop (`services/orion-hub/scripts/vision_affect_ambient.py`) and just
calls `/capture_and_assess` repeatedly with `trigger="ambient"` while its
toggle is on. This corrects an earlier version of this README, which
described Hub's button as "a manual turn-scoped trigger, not a toggle" --
that was true of the button that shipped first (2026-08-22, PR #1838), not
of the toggle that replaced it as the primary control the same day.

`trigger` (`"manual"` | `"ambient"` | `"chat_turn_pre"` | `"chat_turn_post"`)
and `correlation_id` on `JuniperMultimodalAffectV1`
(`orion/schemas/affectgpt.py`) exist so a consumer can tell them apart and,
via `correlation_id`, join one attempt's retina-RPC/worker-RPC/event legs
together -- `capture_and_assess()` generates ONE id per attempt and threads
it through all three, rather than each leg getting its own
independently-generated one.

### The chat-turn bracket (2026-08-25)

`chat_turn_pre` / `chat_turn_post` are a *pair*, fired by Hub around a single
Orion-mode chat turn (`services/orion-hub/scripts/chat_turn_affect.py`, gated
by `AFFECT_CHAT_TURN_SCOPE`, default `voice` = spoken turns only). They are
what makes "how did Juniper's affect move across this exchange" answerable
from stored events: manual and ambient captures are both untethered from any
particular conversation, so neither can produce a matched pair around a known
stimulus.

Those two triggers additionally carry **`chat_correlation_id`** — a
*different join axis* from `correlation_id` above, and deliberately not a
reuse of it:

| field | joins |
| --- | --- |
| `correlation_id` | the three legs of ONE capture attempt (retina RPC, worker RPC, event) |
| `chat_correlation_id` | a capture to the conversation turn that caused it, and a turn's pre/post pair to each other |

`observed_at`-proximity cannot substitute for the second one: a concurrent
ambient tick lands in the same time window and is indistinguishable by
timestamp alone.

Two properties worth knowing before consuming these:

- **Neither capture blocks the turn.** Both are detached; a capture can take
  up to ~195s. So the `chat_turn_pre` read does **not** colour the turn that
  fired it — it lands in the 300s situational mirror in time for the *next*
  turn, and gives `chat_turn_post` something to be compared against.
- **A pair can legitimately be half-present.** All callers share one
  exclusive capture slot; a leg that loses it is dropped (logged, never
  queued, never retried — same no-retry policy the ambient loop already
  has). Treat a lone `chat_turn_pre` as a real, explainable gap rather than
  corrupt data.

`subtitle` is deliberately sent EMPTY by both legs, even though Hub already
holds the microphone transcript. The clip retina records is captured live at
request time — *after* Juniper finished speaking that sentence — so its audio
is not that transcript, and supplying it would ground the model in text that
does not belong to the footage. Empty means the worker Whisper-transcribes
the clip's own audio (`subtitle_source="transcribed"`), which is the honest
read.

## The cross-host bridge (built 2026-08-22)

`capture_and_assess()` is the answer to the "Cross-host capture" gap this
README used to flag as future work: it bus-RPCs `orion-vision-retina`
(`orion:exec:request:RetinaClipCaptureService`, see that service's README)
for a live clip, fetches the resulting `video_sha256`/`audio_sha256` blobs
from percept-store with **hash verification on the fetched bytes**
(`_fetch_percept` — never trusts a reported hash without recomputing it),
and writes them to `AFFECTGPT_SCRATCH_DIR`
(`/mnt/scripts/orion-affectgpt-scratch` by default) — the SAME shared volume
`orion-affectgpt-worker` already mounts read-only at the identical
container path. That's the whole trick: a plain `tempfile.TemporaryDirectory()`
default would write somewhere private to *this* container and the worker
container could never see it. The temp dir (and its fetched bytes) is
removed once the worker call returns, success or failure.

A capture or fetch failure never reaches the worker at all — it's wrapped
straight into a failed `AffectGptAssessResultPayload`
(`error_code` in `{"capture_failed", "fetch_failed"}`) and published like
any other failed assessment.

## Bus contract

- Calls `orion:exec:request:AffectGptWorkerService` (RPC, via
  `OrionBusAsync.rpc_request`), reply on a per-request `orion:affectgpt:reply:<corr_id>`.
- Calls `orion:exec:request:RetinaClipCaptureService` (same RPC pattern),
  reply on `orion:retina:clip:reply:<corr_id>` — only from
  `capture_and_assess()`, not from `/trigger`.
- Publishes `orion:affectgpt:assessment` (`JuniperMultimodalAffectV1`) after
  every trigger, success or failure (the event's `ok`/`error` fields carry
  that — a failed assessment is still a real event, not a silent drop).

## Closing the loop into Orion's own chat turns (2026-08-25)

Before this date, `orion:affectgpt:assessment` had exactly one consumer:
`scripts/tap_assessments.py`, a manual debug CLI. Orion's own chat turns
never found out about a capture — only the Hub UI panel showed it to
Juniper. `_publish_event` now ALSO mirrors every successful capture (a
truncated excerpt of `raw_response`, capped at `_AFFECT_SUMMARY_MAX_CHARS`
= 300 chars — never the verbatim `transcript`) into a single Redis key
(`orion:juniper_affect:latest`, `orion/situational/juniper_affect_state.py`)
that `orion/situational/context.py` polls for every "orion" mode chat turn's
situation brief, gated on a configurable max-age (default 300s,
`ORION_SITUATION_AFFECT_MAX_AGE_SECONDS` in orion-hub/orion-cortex-exec).
Failed/empty captures are not mirrored — a failure should not overwrite a
real prior read, and the reader's own age gate ages that prior read out on
its own schedule. The mirror write is additive and fail-open: it runs after
the real `orion:affectgpt:assessment` publish already succeeded, and never
raises, so a Redis hiccup here cannot break the real event stream.

## Durable persistence (2026-08-25)

`orion:affectgpt:assessment` now has a second real consumer:
`orion-sql-writer` projects every event into `juniper_multimodal_affect_log`
(see that service's README). The Redis mirror above has a 1h TTL and is
the live-read path for chat turns; this table is the durable history for
any capture published while `orion-sql-writer` was actually connected to
the bus. Review finding, 2026-08-25: `OrionBusAsync.publish()` is plain
Redis pub/sub with no redelivery -- a capture published while
`orion-sql-writer` itself is disconnected (restart, DB pool exhaustion)
is dropped before it ever reaches this table too, same as any other
consumer on this bus. Once both the TTL key and that window have passed
with nothing durable written, the capture leaves no trace anywhere. Same
privacy boundary as the mirror: `transcript` is never persisted here
either (including on error -- see `JuniperMultimodalAffectSQL`'s
docstring for the fallback-path fix).

## Operator checklist

1. `GET /health`
2. `POST /v1/juniper/affect/trigger` — `{"video_path": "...", "audio_path": "...", "subtitle": "..."}` (paths must be readable inside the *worker's* container).
3. `POST /v1/juniper/affect/capture_and_assess` — optional `{"subtitle": "...", "user_message": "...", "trigger": "manual"|"ambient"|"chat_turn_pre"|"chat_turn_post", "chat_correlation_id": "..."}` (trigger defaults to "manual" if omitted; `chat_correlation_id` is only meaningful with the two `chat_turn_*` triggers). Synchronous, typically well under a minute but up to ~195s worst case (real capture + real GPU inference) — use a generous client timeout, not a quick one.

## Tests

```bash
pytest services/orion-juniper-affective-state/tests -q
```

Bus-free by design (mocked `OrionBusAsync`) — no live worker or Redis
required.

## Evals

```bash
pytest services/orion-juniper-affective-state/evals -q
```

Requires a live bus + live `orion-affectgpt-worker`. Round-trips a real
trigger and checks the published `orion:affectgpt:assessment` event actually
lands.
