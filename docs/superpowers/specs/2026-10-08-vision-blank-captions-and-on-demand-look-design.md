# Orion can tell someone is in the basement but can't describe them: blank captions and no on-demand look

Date: 2026-10-08 · Status: design, no code changed · Author: investigation session with Juniper

## Arsonist summary

Juniper asked Orion, through Hub chat, "any chance you see me?" Orion said yes. Juniper then
asked "what am I wearing?" Orion couldn't answer, and filled the reply with basement-cabinet
temperature and humidity readings instead.

The camera works. The weak part is the step that describes the picture:

1. **Every room-camera caption is blank.** The basement camera (cam0) uses a small, old
   captioning model (BLIP-base). It doesn't follow instructions; it continues whatever text it
   is given. Since 2026-07-02 it has been handed an instruction ("List visible objects and
   people..."). It repeats that sentence back, the echo filter correctly throws the repeat
   away, and the caption that gets published is `""`. Measured 2026-10-08: about 600 captioned
   tasks per 30 minutes, every one with empty `caption.text`.
2. **The blank is hidden.** The rejection reason goes into a local `warnings` list that the
   published artifact never carries. Downstream sees `{"text": "", "confidence": 1.0}`, which
   looks like a normal result: the caption is empty but marked fully confident.
3. **Chat gets labels, never a look.** What reaches the chat turn is a narrative built from
   object-detector counts ("Two chairs, two tables... one person and one piece of clothing").
   Nothing in the pipeline can answer an appearance question such as clothing, colour, or
   posture. The pieces that could already exist but are not used: an idle Qwen2-VL host and a
   "foveal probe" that sends a frame to circe's vision-capable chat model. Today the probe is
   reachable only through a debug endpoint.
4. **The person is "unknown".** cam0 presence says `subject: unknown, identity_confirmed:
   false`, so Orion knows someone is there but not that it's Juniper.

## Evidence (live, 2026-10-08 ~02:45 UTC)

| Claim | Evidence |
|---|---|
| cam0 is producing scenes | `vision_scene_inventory` cam0: 10,388 rows / 24h, newest 02:45:15; latest counts `person:1, clothing:1, shoe:2, desk, chair, door, table` |
| Chat receives the perception brief | Hub `orion/situational/context.py` ~L549-562: perception was a literal `False` until 2026-10-07, now `ORION_SITUATION_PERCEPTION_ENABLED=true`, `STREAM_IDS=carbon,cam0` (confirmed in the running `orion-athena-hub` env) |
| Narrative is label-only | `vision_events` cam0 newest: "Two chairs, two tables, and two desks are present in the scene along with one person and one piece of clothing." |
| Presence is anonymous | `substrate_embodied_presence` cam0: `state: present, subject: unknown, since_sec: 3034, identity_confirmed: false` (carbon row stale since 2026-09-09) |
| Captions are requested | frame-router log, 30 min: `612 task_type=retina_fast want_caption=True` |
| Captions are blank | `redis-cli subscribe orion:vision:artifacts`: every artifact `outputs.caption = {"text":"","confidence":1.0}`, `model_fingerprints.vlm_caption = Salesforce/blip-image-captioning-base` |
| Cause | Ran BLIP-base by hand inside `orion-athena-vision-host` on the newest cam0 frame. **With `CAPTION_PROMPT`:** returns the prompt itself. **No prompt:** `"a woman sitting at a desk in an office"`. **`"a photo of"`:** `"a photo of a woman sitting at a desk"` |
| Better model idle | `orion-athena-vision-host-qwen`: `Qwen/Qwen2-VL-2B-Instruct`, GPU 1, `VISION_ENABLED_PROFILES=vlm_caption`; **0** `vision_task_complete` in the last 60 min |

Reproduce the hand caption test:

```bash
F=$(docker exec orion-athena-vision-host sh -c 'ls -t /mnt/telemetry/vision/frames/ | head -1')
docker exec orion-athena-vision-host python3 -c "
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
m='Salesforce/blip-image-captioning-base'
p=BlipProcessor.from_pretrained(m); mod=BlipForConditionalGeneration.from_pretrained(m).to('cuda:0')
img=Image.open('/mnt/telemetry/vision/frames/$F').convert('RGB')
for pr in ['List visible objects and people. State only what is directly visible. No guesses about activity.', None]:
  i=p(img, pr, return_tensors='pt').to('cuda:0') if pr else p(img, return_tensors='pt').to('cuda:0')
  print(repr(pr)[:20], '->', p.decode(mod.generate(**i, max_new_tokens=40)[0], skip_special_tokens=True))"
```

**How long has it been broken?** Probably since `03fbd9c42` (2026-07-02, "factual caption
prompt and garbage sanitizer"), which introduced both the instruction prompt and the BLIP-base
default. This is **UNVERIFIED**: rejection reasons are never persisted, so no history exists.
The foveal probe's docstring records the same symptom on 2026-08-25 (`caption_rejected:too_short`
via the placeholder `vlm_caption` model_id). It was routed around rather than fixed.

## Current architecture

```text
cam0 RTSP -> orion-vision-edge (frames to /mnt/telemetry/vision/frames/)
  -> orion-vision-frame-router (config/vision_frame_router.yaml: baseline + triggered[person] want_caption=true)
  -> orion:exec:request:VisionHostService -> orion-vision-host (athena GPU0)
       pipeline_retina_fast (config/vision_profiles.yaml): grounding-dino detect, siglip2 embed, vlm_caption (BLIP-base)
       runner._run_caption_frame -> sanitize_caption -> rejected -> text=""   <-- BREAK
  -> orion:vision:artifacts -> orion-vision-window (projection.py: summary.captions) -> orion-vision-council
       (narrative from labels; evidence_grounding.py uses captions when present) -> vision_events.narrative
  -> Hub chat turn: orion/situational/context.py -> perception_reader.fetch_latest_percept (vision_events)
                    + fetch_presence_resolved (substrate_embodied_presence)
```

Unused vision capacity:
- `orion-athena-vision-host-qwen` (Qwen2-VL-2B, circe lane), intake
  `orion:exec:request:VisionHostService:circe-vl`. The only producer is orion-thought's visual
  chain (`CHANNEL_VISION_HOST_REQUEST`).
- `orion-vision-council/app/foveal_probe.py`: sends the newest frame as a `kind="percept"`
  attachment through the LLM gateway to circe's vision-capable chat lane. Trigger:
  `POST /debug/foveal-probe?question=...` (`main.py:524`) only. `settings.py:51` says "not yet
  wired to any" automatic trigger.

## Missing questions

1. Should the always-on (peripheral) caption be Qwen2-VL-2B (better, but a GPU-1 cost every ~3s
   while a person is present), or a fixed BLIP without a prompt plus Qwen/foveal only on demand?
   **Recommendation:** fix BLIP now (cheap), use the foveal probe for questions.
2. Should "what am I wearing" in chat trigger a live look (foveal probe with Juniper's question)?
   Or should a periodic rich caption sit in the situation brief? On-demand answers the actual
   question, and only costs GPU when someone asks.
3. Which GPU lease does a chat-triggered foveal call take? It goes through the gateway, so it
   should take the normal chat lease (see the PINNED note about outreach bypassing leases via the
   FCC (agent harness) proxy). Don't add another path that skips the lease.
4. Is the cam0 `subject: unknown` a face-match threshold, an angle/lighting problem, or an
   enrolment gap? Needs its own look at `identity_face` results on cam0.

## Proposed schema / API changes

- **Artifact honesty (no new schema):** when a caption is rejected, publish
  `outputs.caption = {"text": "", "confidence": 0.0, "rejected_reason": "<reason>"}` or carry the
  existing `warnings` through. Check `VisionCaption` / `VisionArtifactPayload` in
  `orion/schemas` (and `services/orion-vision-host/app/artifacts.py`) for an `extra="forbid"`
  model. If one exists, adding a field is a consumer-first migration. **Hard rule:** never
  publish `confidence: 1.0` on an empty caption.
- **Prefix-style captioning:** add a "captioner family" decision next to `vlm_family.py`.
  BLIP/BLIP-2 get no prompt (or a short prefix such as `"a photo of"`, stripped afterwards).
  Chat-template VLMs keep `CAPTION_PROMPT`. Any instruction prompt then has to work for the
  model family it is sent to.
- **On-demand look (contract reuse, no new channel):** Hub chat turn -> council foveal probe
  (existing gateway path, `kind="percept"` attachment) with the user's question. This needs a
  bus-callable trigger in place of the debug HTTP endpoint. Reuse an existing registered council
  request channel if there is one; otherwise this is the one contract addition, and it must be
  registered in `orion/bus/channels.yaml`.

## Files likely to touch

| Purpose | Path |
|---|---|
| Caption call + rejection | `services/orion-vision-host/app/runner.py` (`_run_caption_frame` ~L960-1012, `_generate_vlm_text`) |
| Echo/degeneracy filter | `services/orion-vision-host/app/caption_sanitize.py`, `orion/vision/caption_echo.py` (`CAPTION_PROMPT`) |
| Model family switch | `services/orion-vision-host/app/vlm_family.py`, `app/model_manager.py` (`load_vlm_captioner` L213) |
| Artifact shape | `services/orion-vision-host/app/artifacts.py`, `orion/schemas` vision artifact models |
| Model config | `services/orion-vision-host/.env(_example)` `VISION_VLM_MODEL_ID`; `config/vision_profiles.yaml` `vlm_caption` placeholder `REPLACE_ME/...` |
| Dispatch policy | `config/vision_frame_router.yaml` (cam0 `triggered.request.want_caption`) |
| Qwen lane | `services/orion-vision-host/docker-compose.circe-qwen.yml`, README §"Circe Qwen2-VL lane" |
| Caption consumers | `services/orion-vision-window/app/projection.py`, `services/orion-vision-council/app/evidence_grounding.py` (L146-155, caption-sourced confidence 0.4) |
| On-demand look | `services/orion-vision-council/app/foveal_probe.py`, `app/main.py:524`, `app/settings.py:51` |
| Chat side | `orion/situational/context.py` (~L549), `orion/situational/perception_reader.py` (`fetch_latest_percept`, `fetch_presence_resolved`) |
| Live checks | `docker exec orion-athena-sql-db psql -U postgres -d conjourney`; tables `vision_scene_inventory`, `vision_events`, `substrate_embodied_presence`; channel `orion:vision:artifacts` on the bus at `redis://100.92.216.81:6379/0` |

Prior context: `docs/superpowers/specs/2026-08-12-perception-frontier-design.md` (foveal tier),
`docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md`, PR #1960 (identity on
carbon/cam0), `docs/vision_services.md`.

## Non-goals

- No new vision service and no second captioning pipeline: the Qwen host and the foveal probe
  already exist.
- No keyword lists of clothing words or appearance categories.
- No face/appearance persistence change for anyone other than Juniper (walkway privacy stance
  stands).
- No change to how Orion talks about the cabinet sensors, even though the reply leaned on them.
  That's a symptom of having nothing else to say.

## Acceptance checks

1. **Regression test:** a fake BLIP-family generator that echoes its prompt must produce a
   non-empty caption under the new path (no prompt sent). It must not pass by loosening the
   echo filter.
2. **Live:** 5 min of `orion:vision:artifacts` for cam0 with a person present has more than
   80% of `want_caption` artifacts with non-empty `caption.text`. Script:
   `timeout 300 redis-cli -u $ORION_BUS_URL subscribe orion:vision:artifacts | grep -o '"caption":{"text":"[^"]*"' | sort | uniq -c`.
3. **Honesty:** a forced rejection publishes `confidence` 0 and a reason; no artifact ever has
   empty text with `confidence: 1.0`.
4. **Downstream:** a cam0 `vision_events.narrative` in the 30 min after deploy contains caption
   content (e.g. "woman sitting at a desk"), not only counts.
5. **Chat end to end:** with Juniper in the basement, Hub chat "what am I wearing?" returns a
   description grounded in a foveal-probe result. The turn trace has to show the probe's
   correlation id. Otherwise, Orion says plainly that it couldn't look, rather than narrating
   sensors.
6. **Identity:** a separate check of why cam0 presence reads `subject: unknown`.

## Recommended next patch

**Patch 1 (small, ship first):** in `orion-vision-host`, choose the caption prompt by model
family (BLIP: none; chat VLM: `CAPTION_PROMPT`), and publish rejection reasons with confidence
0. Include the regression test from check 1, then do the live check 2 after rebuilding
`orion-vision-host` via `scripts/safe_docker_build.sh orion-vision-host up -d --build` from
the primary checkout on main (prod deploy rule).

**Patch 2:** wire the foveal probe as a bus-triggered tool the chat turn can call for appearance
or "look at me" questions, under the chat GPU lease. Proposal mode applies (this touches the
perception and cognition loop).

**Patch 3:** cam0 identity: `subject: unknown` investigation.
