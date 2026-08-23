# PR #1846: Whisper subtitle transcription for orion-affectgpt-worker

- Branch: `feat/affectgpt-whisper-subtitle`
- PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1846

## Summary

- `orion-affectgpt-worker` now runs Whisper transcription on `audio_path` whenever a request's `subtitle` is empty, instead of always running in the degraded empty-subtitle mode this service's own README already documented as materially worse-grounded.
- Juniper's own ask, live in this session ("HIT IT"), after testing the real ambient-capture pipeline (PR #1840/#1841) and getting a generic hedge response — investigation traced it to `subtitle=""` being the effective default on every real capture (neither Hub's ambient loop nor its manual "Check now" route ever sent one).
- `AffectGptAssessResultPayload` gained `subtitle_source` (`"caller" | "transcribed" | "none"`) and `transcript`, threaded through to `JuniperMultimodalAffectV1` too — so a future generic-hedge response can be told apart from "the model actually had a real transcript and still hedged."
- Fully additive, fails open: an explicit caller-supplied subtitle always wins (now with a real test proving it); Whisper failure never blocks or crashes an assessment.
- Live-verified twice on circe GPU2 against the real running production containers, including catching and fixing a real live deploy-ordering bug along the way (see below).

## Outcome moved

Every real ambient/manual capture now gets Whisper-grounded model output instead of running in the documented-degraded empty-subtitle mode by default. Confirmed live against the bundled AffectGPT demo clip: `raw_response` went from generic hedging to explicitly quoting and reasoning from the real transcript.

## Current architecture

`orion-juniper-affective-state` (thin CPU orchestrator) fetches `video_path`/`audio_path` from percept-store and calls `orion-affectgpt-worker`'s `assess()` with whatever `subtitle` the caller supplied — always `""` in practice, since Hub never populates it. The worker's `model_runtime.py` fed that string straight into `get_prompt_for_multimodal` with no transcription step of any kind.

## Architecture touched

`services/orion-affectgpt-worker` (new `app/transcribe.py`, changes to `model_runtime.py`/`main.py`/`settings.py`), `orion/schemas/affectgpt.py` (two new optional fields, backward/forward compatible), `services/orion-juniper-affective-state/app/main.py` (threads the two fields through `_wrap_event()`). No bus channel/contract changes.

## Files changed

- `services/orion-affectgpt-worker/app/transcribe.py` (new): silence-gate + Whisper call (mirrors `orion-whisper-tts/app/stt.py`'s proven technique) plus `resolve_subtitle()`, a pure function encapsulating the full "what does the model see" decision — extracted specifically so the "caller always wins" guarantee is unit-testable without needing the vendored AffectGPT/GPU state.
- `services/orion-affectgpt-worker/app/model_runtime.py`: warm-loads Whisper in `load()` (advisory — a load failure never blocks core readiness), `assess()` calls `resolve_subtitle()`, threads `subtitle_source`/`transcript`/`meta` into the result, fixed timing so `data_load_s` no longer silently absorbs transcription time.
- `services/orion-affectgpt-worker/app/main.py`: threads the three new fields into the response payload.
- `services/orion-affectgpt-worker/app/settings.py` + `.env_example`: `AFFECTGPT_TRANSCRIBE_ENABLED` / `AFFECTGPT_WHISPER_MODEL` / `AFFECTGPT_TRANSCRIBE_NEAR_SILENT_PEAK_INT16` / `AFFECTGPT_WHISPER_LANGUAGE`.
- `services/orion-affectgpt-worker/requirements.txt`: `openai-whisper` (unpinned, matching `orion-whisper-tts`'s own convention; reuses this image's already-present torch — no separate install).
- `orion/schemas/affectgpt.py`: `subtitle_source`/`transcript` on `AffectGptAssessResultPayload` and `JuniperMultimodalAffectV1` — both already registered by class reference in the schema registry, no registry change needed.
- `services/orion-juniper-affective-state/app/main.py`: `_wrap_event()` threads the two fields through.
- Tests: `test_transcribe.py` (16 real executing tests — silence gate, error-vs-silence distinction, whitespace-only subtitle, caller-always-wins with an exploding fake model), `test_schemas.py` extended, one new orchestrator round-trip test.

## Schema / bus / API changes

- Added: `AffectGptAssessResultPayload.subtitle_source`, `.transcript`; `JuniperMultimodalAffectV1.subtitle_source`, `.transcript`. Both `Optional`, default `None` — old producers omit them, old consumers ignore them.
- Removed: none.
- Renamed: none.
- Behavior changed: a request with an empty `subtitle` now gets a real Whisper transcript instead of running the documented-degraded empty-subtitle path, whenever `AFFECTGPT_TRANSCRIBE_ENABLED=true` (default) and the clip isn't judged near-silent.
- Compatibility notes: **real deploy-ordering hazard, caught live** — both `AffectGptAssessResultPayload` and `JuniperMultimodalAffectV1` keep `extra="forbid"`, and each service bakes its own copy of `orion/schemas/` into its Docker image at build time with no shared version pin. Redeploying the worker before the consumer means the consumer's still-running old schema rejects every real reply with a `ValidationError` (I hit this directly on circe mid-session — see "Docker/build/smoke checks" below). **Redeploy `orion-affectgpt-worker` and `orion-juniper-affective-state` together, not staggered.**

## Env/config changes

- Added keys: `AFFECTGPT_TRANSCRIBE_ENABLED=true`, `AFFECTGPT_WHISPER_MODEL=base`, `AFFECTGPT_TRANSCRIBE_NEAR_SILENT_PEAK_INT16=50`, `AFFECTGPT_WHISPER_LANGUAGE=en` (all in `services/orion-affectgpt-worker/.env_example` and `.env`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced: hand-edited (the sync script has a known bug — reads the primary checkout's live `.env` and silently skips genuinely new keys rather than adding or reporting them; confirmed live, matches a known prior finding). Appended by hand to athena's primary checkout, this worktree, and circe's primary checkout + deploy worktree.
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-affectgpt-worker
PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/ -q
  33 passed

cd services/orion-juniper-affective-state
PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/ evals/ -q
  23 passed, 1 skipped
```

(Both services use a top-level `app.` package name and collide when run in one pytest invocation from repo root — a pre-existing cross-service test-isolation gap, not something this PR introduces; run separately as above.)

## Evals run

No dedicated eval harness exists for either service. In its place: two full live end-to-end verifications against the real running production containers on circe GPU2 (see below) — real HTTP requests, real GPU inference, real Whisper transcription, not mocks.

## Docker/build/smoke checks

**This is the actual verification, not a formality.**

Round 1 (before the review-fix commit):
```text
$ ssh circe@circe  # via a dedicated worktree, not the shared checkout
$ bash scripts/safe_docker_build.sh orion-affectgpt-worker build
  Image orion-affectgpt-worker-affectgpt-worker Built
$ docker run --rm orion-affectgpt-worker-affectgpt-worker:latest python -c 'import whisper; print(whisper.__file__)'
  /opt/conda/lib/python3.11/site-packages/whisper/__init__.py
$ bash scripts/safe_docker_build.sh orion-affectgpt-worker up -d --build
  Container orion-circe-affectgpt-worker Started
$ curl -X POST http://localhost:32798/v1/affect/assess ... (bundled AffectGPT demo clip, subtitle="")
  HTTP 200, subtitle_source="transcribed", transcript="I don't know. I don't know how to explain this."
  raw_response explicitly quotes and reasons from the transcript
  face_detection: 88/88 (matches README's own documented number, no regression)
```

**Real bug #1, caught live:** my first draft's new logging used stdlib `logging.getLogger(__name__)`. This service only configures a loguru sink; stdlib log calls were completely silent in `docker logs` despite the feature genuinely running (proven by the HTTP response itself). Fixed by switching to `from loguru import logger`; re-verified the log lines appear.

**Real bug #2, caught live, active in production at the time:** after redeploying only the worker, I checked the still-running (old) consumer container directly:
```text
$ docker exec orion-circe-juniper-affective-state python -c \
  "from orion.schemas.affectgpt import AffectGptAssessResultPayload; \
   print('subtitle_source' in AffectGptAssessResultPayload.model_fields)"
  False
```
Every real reply from the (already redeployed) worker would have failed the consumer's `extra="forbid"` validation. Redeployed the consumer immediately, re-checked: `True`.

Round 2 (after the review-fix commit, both services rebuilt and redeployed together):
```text
$ bash scripts/safe_docker_build.sh orion-affectgpt-worker up -d --build
$ bash scripts/safe_docker_build.sh orion-juniper-affective-state up -d --build
$ curl -X POST http://localhost:32798/v1/affect/assess ... (same demo clip)
  HTTP 200
  subtitle_source: "transcribed"
  transcript: "I don't know. I don't know how to explain this."
  timings: {transcribe_s: 0.912, data_load_s: 0.064, encode_s: 0.56, generate_s: 14.801, total_s: 16.337}
  meta: {transcribe: {peak: 13370, rms: 2147.96, peak_threshold: 50, silence_gate: "passed", text_len: 47}}
  face_detection: 88/88 (still no regression)
$ docker exec orion-circe-juniper-affective-state python -c "...subtitle_source..."
  True  # consumer back in schema sync
```
`data_load_s` (0.064s) is no longer inflated by transcription time, `total_s` (16.337) now correctly sums every stage including `transcribe_s`, and the full silence-gate telemetry (`meta.transcribe`) is now actually reachable in the response — all three were review findings, all three independently confirmed fixed against the real running system, not just unit tests.

## Review findings fixed

Real code-review skill, run twice (initial pass, then a re-review of the fix commit).

- Finding: **[LIVE, ACTIVE] deploy-ordering hazard** — `extra="forbid"` on both schemas + no shared version pin between services' baked-in `orion/schemas/` copies.
  - Fix: caught mid-session as an actual live issue (see Docker/build/smoke checks above), redeployed the consumer immediately. Documented the required together-not-staggered redeploy order in this report's "Compatibility notes."
  - Evidence: `docker exec ... model_fields` check, `False` → redeploy → `True`.
- Finding: whitespace-only subtitle (e.g. `" "`) passed plain truthiness as real caller text, skipping Whisper and silently reproducing the degraded-mode failure this feature exists to fix.
  - Fix: `resolve_subtitle()` strips before the check.
  - Evidence: `test_resolve_subtitle_whitespace_only_is_treated_as_empty`.
- Finding: the "caller subtitle always wins" guarantee had no test.
  - Fix: extracted the decision into `resolve_subtitle()`, a pure function with no GPU/vendored-AffectGPT dependency; new test uses a fake Whisper model that raises `AssertionError` if ever called, proving it isn't when real subtitle text is present.
  - Evidence: `test_resolve_subtitle_caller_text_always_wins_whisper_never_called`.
- Finding: `measure_wav_peak`'s error path (missing/corrupt file) returned the identical `(0.0, 0)` as genuine silence — a real upstream bug (e.g. a truncated percept-store fetch) would look identical to routine silence-gating in every log line and response field.
  - Fix: now returns `(rms, peak, error)`; `transcribe_audio` runs Whisper anyway on a measurement failure instead of silently rejecting as near-silent.
  - Evidence: `test_measure_wav_peak_distinguishes_unreadable_from_real_silence`, `test_transcribe_audio_runs_whisper_anyway_when_peak_cannot_be_measured`.
- Finding: `timings["data_load_s"]` silently absorbed the new `transcribe_s` duration.
  - Fix: moved `t0` to right before the dataset-load work it's meant to measure; `total_s` now uses a separate `t_request_start` covering the whole request.
  - Evidence: live response above — `data_load_s=0.064`, `total_s` exactly sums all four stages.
- Finding: `transcribe_audio`'s diagnostic `meta` (silence_gate/peak/rms) was computed but structurally unreachable — `AssessmentResult` had no `meta` field.
  - Fix: added one, threaded through to `AffectGptAssessResultPayload`'s existing generic `meta` field.
  - Evidence: live response above — `meta.transcribe` present with real values.
- Finding: Whisper's language hardcoded to `"en"`, no config knob, unlike the sibling service this diff compares itself to.
  - Fix: added `AFFECTGPT_WHISPER_LANGUAGE` (default `"en"`).
- Finding: silence-gate loop bound diverged from `orion-whisper-tts`'s proven code (`range(0, len(frames)-1, 2)` vs `range(0, len(frames), 2)`) despite the module docstring claiming the technique was "deliberately copied."
  - Fix: matched exactly (functionally identical for the even-length buffers always produced here, but removed the discrepancy).
- Finding: `transcript` is verbatim transcribed speech on the wire — a real widening of this module's stated "paths only, never raw bytes" privacy principle.
  - Not changed — documented as a deliberate, Juniper-approved exception in the schema field's own docstring. Mitigating factor noted: `raw_response` already routinely leaks the same content indirectly (confirmed live in this session's own testing).
- Finding: silence-gate/whisper-load code duplicates `orion-whisper-tts`'s own technique with no shared module.
  - Declined — genuinely separate services/dependency footprints; extracting shared `orion/` audio-utils infra now is scope creep beyond this feature (same call made on a similar finding earlier this session, PR #1843).
- Finding (plausible, not confirmed): Whisper now runs inside the same `_busy_lock`/`AFFECTGPT_REQUEST_TIMEOUT_S` budget as AffectGPT inference, extending lock-hold time with no timeout increase.
  - Not changed — measured live at `transcribe_s=0.912-1.07s` against a 120s budget on an 8s-capped clip (retina's own duration cap bounds this); real but low-impact, not worth inflating the timeout without a demonstrated need.

## Restart required

Already done as part of live verification — both `orion-circe-affectgpt-worker` and `orion-circe-juniper-affective-state` are live on this branch's final commit right now. Once merged to `main`, no further restart needed unless `main` diverges further before another deploy. If redeploying manually later:

```bash
# Together, not staggered -- see "Compatibility notes" above.
bash scripts/safe_docker_build.sh orion-affectgpt-worker up -d --build
bash scripts/safe_docker_build.sh orion-juniper-affective-state up -d --build
```

## Risks / concerns

- Severity: low. Concern: `transcript` widens this module's privacy surface (verbatim transcribed speech on the wire). Mitigation: explicit, Juniper-approved (this whole feature was Juniper's own ask, "HIT IT"); documented in the schema field's own docstring; `raw_response` already routinely carries equivalent content indirectly.
- Severity: low. Concern: added GPU/VRAM footprint (Whisper "base", ~1GB) on circe GPU2 alongside AffectGPT's ~18.4GB peak on a 32GB card. Mitigation: confirmed real headroom in the worker's own README; live-verified no OOM/contention across two full redeploy-and-request cycles this session.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/affectgpt-whisper-subtitle
