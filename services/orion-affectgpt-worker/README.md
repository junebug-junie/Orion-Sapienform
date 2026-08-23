# orion-affectgpt-worker

Real-time facial+vocal affect assessment for Juniper, via
[AffectGPT](https://github.com/zeroQiaoba/AffectGPT) (Qwen2.5-7B-Instruct +
CLIP ViT-L + HuBERT-L, LoRA fine-tuned). Runs on **circe GPU2** (V100-32GB,
the "agent / affect / testing" lane, :8014).

Bus: consumes `orion:exec:request:AffectGptWorkerService`
(`AffectGptAssessRequestPayload`), replies on `orion:affectgpt:reply:*`
(`AffectGptAssessResultPayload`). HTTP: `POST /v1/affect/assess` mirrors the
bus handler directly. See `orion/bus/channels.yaml` for the full contract and
why this is deliberately NOT the same channel/schema family as the existing
`orion:substrate:juniper_affective_state` (text-only) signal.

## Non-goals (read before extending)

- **No emotion taxonomy/classifier.** `raw_response` (the model's own
  free-text reasoning) is the only signal shipped. Inventing a fixed label
  set (`Literal["sad","anxious",...]`) without real accumulated data to
  justify the categories would be exactly the "keyword cathedral" CLAUDE.md
  bans. Add one later, backed by real examples, not now.
- **No live capture pipeline.** This worker takes a `video_path`/`audio_path`
  pair that must already exist on disk. Nothing in this repo currently
  captures Juniper's webcam/mic — that is a separate, unbuilt piece.
- **No concurrency/scheduling.** One 7B model, one GPU, a single asyncio lock
  serializes requests. A queue/scheduler (like `orion-vision-host`'s) is
  premature for a single-user (Juniper) signal.
- **No real OpenFace.** See "Face crops" below.

## Provenance (read before trusting the numbers)

Everything below was proven live on circe GPU2, 2026-08-22, not assumed from
docs. Summarized here because none of it is obvious from the code alone.

**Face crops.** AffectGPT's only released checkpoint
(`MERChallenge/AffectGPT` epoch 60, downloaded via HuggingFace) is trained on
OpenFace-extracted, similarity-aligned face crops. Real OpenFace could not be
stood up on this hardware: the only reachable Docker image
(`algebr/openface`, built 2018 against OpenCV 3.4) segfaults during actual
per-frame detection even after its MTCNN→HOG-SVM fallback engages cleanly,
and upstream (`TadasBaltrusaitis/OpenFace`) ships no Dockerfile to build from
source (confirmed 404 on `master`). `app/face_extract.py` is a documented
substitute: Haar-cascade (OpenCV, bundled, zero extra deps) per-frame
detection + fixed-margin crop + resize, carrying the last real detection
forward across any gap. On the repo's own bundled demo clip this got 88/88
real detections, zero fallback frames. It is **not** bit-identical to
OpenFace's landmark-based alignment (no eye/rotation normalization) — every
result carries `face_detection` telemetry (`frames_total`, `frames_detected`,
`frames_carried_forward`, `frames_no_face_fallback_full_frame`,
`detection_rate`) so a caller can see when this approximation was actually
exercised vs. degraded to full-frame fallback.

**No frame-mode checkpoint exists** (reachably). The frame-mode training
config's `ckpt`/`ckpt_2` fields are empty on HuggingFace; a frame-mode
checkpoint is referenced only via Baidu Netdisk in upstream's README, which
was not used (untrusted/inaccessible source for this deployment). This is
why `AFFECTGPT_FACE_OR_FRAME` is hardcoded to `multiface_audio_face_text`,
not a request-time option — there is nothing else to run.

**Determinism ceiling.** `do_sample=False` + `torch.use_deterministic_algorithms`
+ cudnn deterministic flags (all applied in `model_runtime.py`) reduce but do
**not** eliminate run-to-run variance on identical input. 5 repeat runs on
the same clip: `sad, contemplative` / `sad, depressed` / `sad, depressed` /
`sad, frustrated, anxious, overwhelmed` / `sad, depressed` — "sad" every
time, second descriptor varies. No PyTorch nondeterminism warnings were ever
raised, meaning the residual source is outside torch's tracked-op list —
almost certainly the fused SDPA attention kernel, which is a documented gap
in `use_deterministic_algorithms`'s coverage. **Forcing
`attn_implementation="eager"` to close that gap was tried and rejected**: it
corrupted output into incoherent repeating garbage on this checkpoint. sdpa
(the default) stays. Do not "fix" this by trying eager again without
re-verifying against real output.

**Subtitle text matters a lot.** Early testing ran with `subtitle=""` and got
terse output ("The character's emotional state is sad, depressed."). Passing
the clip's real subtitle text produced grounded, reasoned output that
correctly paraphrased the transcript, identified the speaker's gender, and
cited specific audio/video cues — and began with **"In the text..."**, which
upstream's own README documents as the sanity check that inference is
actually working ("otherwise your inference code or downloaded model may
contain errors"). Always pass real subtitle text when available;
`subtitle=""` is a degraded mode, not a neutral default.

**Whisper auto-transcription (2026-08-22, Juniper's own ask -- "HIT IT")
closes this gap when the caller doesn't supply one.** Confirmed live 2026-08-22:
neither Hub's ambient loop nor its manual "Check now" route ever sent real
subtitle text, so every real capture was running in the degraded mode above
by default. `app/transcribe.py` now runs Whisper ("base" model,
`AFFECTGPT_WHISPER_MODEL`) on `audio_path` whenever the request's `subtitle`
is empty, and uses the result as the prompt's subtitle if non-empty. Purely
additive and fails open: an explicit caller-supplied subtitle always wins
(never overwritten), Whisper never blocks or crashes an assessment on
failure (falls back to empty, exactly today's behavior), and it's disabled
entirely via `AFFECTGPT_TRANSCRIBE_ENABLED=false` if needed. Loads in-process
on the same GPU as the AffectGPT model rather than calling the repo's other
Whisper deployment (`orion-whisper-tts`) over the bus -- see
`app/transcribe.py`'s module docstring for why. `AffectGptAssessResultPayload.
subtitle_source` (`"caller" | "transcribed" | "none"`) and `.transcript`
report which case actually happened and, when transcribed, the real text --
so a caller reading a generic hedge in `raw_response` can tell whether that
came from real detected speech or from silence/no-subtitle, instead of
guessing.

**VRAM / timing** (cold start, one-off subprocess run during benchmarking,
not yet re-measured against this warm-loaded service): ~18.4GB peak / 32GB,
~20.7s wall time including ~30s of checkpoint-shard loading. This service
warm-loads the model once at startup specifically to avoid paying that cost
per request — the per-request cost should be closer to `timings.generate_s`
alone; re-measure and update this line once live.

## Model weights

Not baked into the image (~33GB: CLIP 6.4G, HuBERT 4.8G, Qwen2.5-7B 15G,
AffectGPT checkpoint 6.9G). Bind-mounted read-only from
`/mnt/scripts/orion-affectgpt-models` on circe (circe-local — `/mnt/telemetry`
is athena-local ext4 with no NFS/exports, per
`reference_circe_gpu_inventory_and_lane_map`, so athena's model-cache
convention does not apply here).

## Operator checklist

1. `GET /health` — model_loaded, bus_enabled.
2. `GET /ready` — 503 until the model has finished loading (~30-60s cold).
3. `POST /v1/affect/assess` — `{"video_path": "...", "audio_path": "...", "subtitle": "..."}`.
4. Watch `orion:system:health` for this service's heartbeat (`HeartbeatOnly`
   chassis, independent bus connection from the request-handling one).

## Tests

```bash
pytest services/orion-affectgpt-worker/tests -q
```

GPU/model-free by design (settings defaults, face-crop Haar-cascade logic
against a synthetic clip, request/response schema validation). No CI runner
here has the 33GB of weights or a V100 — full-model tests live in `evals/`
and are run manually against the live circe deployment.

## Evals

```bash
pytest services/orion-affectgpt-worker/evals -q
```

Requires a live worker + real weights (`AFFECTGPT_WORKER_URL` env var,
default `http://localhost:32798`). Checks the upstream "In the text" sanity
signal against the repo's own bundled demo clip + real subtitle — this is
the same live check that validated the pipeline on 2026-08-22.
