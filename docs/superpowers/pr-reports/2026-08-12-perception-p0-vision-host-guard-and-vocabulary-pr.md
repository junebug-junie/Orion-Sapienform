# PR report — perception P0: pipeline guard fix + detector vocabulary

Implements **P0** of the perception frontier design (PR #1590,
`docs/superpowers/specs/2026-08-12-perception-frontier-design.md`), and one
finding P0's investigation turned up that the design doc did not anticipate.

## Summary

- Fixed the live `_safe_when` bug: pipeline `when:` guards raised
  `AttributeError` on any request flag the caller omitted, logging a warning on
  every single vision task (347 in 30 minutes on `orion-athena-vision-host`).
- Extracted guard evaluation into `app/when_guard.py`, free of torch/PIL/numpy,
  so guard semantics are testable outside the container.
- **Reviewed real cam0 frames** — the second half of P0, which the doc calls the
  gate on everything else. The camera watches an information-rich home office,
  not a low-information wall.
- **Found the real information ceiling:** `retina_detect_open_vocab` shipped
  with no `default_prompts`, so the runner fell back to a hardcoded six-word
  list. An open-vocabulary detector can only name what it is asked about.
- Gave the detector a 26-term vocabulary. Live narratives went from *"Two doors
  and three screens are present"* to *"Multiple tables, chairs, desks, doors,
  storage units, monitors, and a guitar are present in the scene."*
- Corrected two claims in the design doc that live inspection contradicted.

## Outcome moved

Perceptual content, measured on live `vision_events`:

| | Before | After (14 min post-deploy) |
| --- | --- | --- |
| Distinct narratives | 211 / 1117 (18.9%) over 19 days | 16 / 18 (89%) |
| Nouns ever produced | `door`, `screen`, `package`, `person` | + `desk`, `table`, `chair`, `monitor`, `storage unit`, `cardboard box`, `guitar` |
| `[PIPE] when eval failed` | 347 / 30 min | 0 |

The post-deploy sample is **14 minutes, 18 events**. It is enough to show the
vocabulary reaches the narrative, and explicitly not enough to state a settled
distinct-narrative rate — see Risks.

## Current architecture

`orion-vision-edge` → `orion-vision-frame-router` → `orion-vision-host`
(GroundingDINO + BLIP + CLIP) → `orion-vision-window` (label habituation) →
`orion-vision-council` (text-model interpretation) → `orion-vision-scribe` →
Postgres `vision_events`. Seven services, all up, and no cognition surface
reads the terminal table.

## Architecture touched

`services/orion-vision-host` only. No contract, bus, schema, or env change.
`config/vision_profiles.yaml` is `COPY`d into the image at build time
(`Dockerfile:30`), so one rebuild carries both changes.

## Files changed

- `services/orion-vision-host/app/when_guard.py`: new. `RequestView` +
  `safe_when`, dependency-light so it is testable without torch.
- `services/orion-vision-host/app/runner.py`: delegates to `when_guard`;
  drops the now-unused `SimpleNamespace` import.
- `config/vision_profiles.yaml`: `default_prompts` on
  `retina_detect_open_vocab`.
- `services/orion-vision-host/tests/test_when_guard.py`: new, 9 tests.
- `services/orion-vision-host/tests/test_detector_vocabulary.py`: new, 5 tests.

## The two bugs

### 1. Guards raised on absent flags

`_safe_when` wrapped the request in `SimpleNamespace(**request)`, which raises
`AttributeError` for any key the caller omitted. Every optional step in every
pipeline is guarded on exactly such a key — `request.is_video`,
`request.want_masks`, `request.want_embeddings`. The exception was swallowed as
a `False`, so behaviour was accidentally almost-correct while logging:

```
[PIPE] when eval failed expr=request.is_video == True
       err='types.SimpleNamespace' object has no attribute 'is_video'
```

Absent now resolves to `None` (falsy). The `true`/`false` normalisation is also
whole-word now; a bare `str.replace` would have corrupted a key like
`want_true_color`.

### 2. The detector had a six-word vocabulary

`runner.py:393`, reached because neither the request nor the profile supplied
prompts:

```python
prompts = ["person", "face", "phone", "screen", "door", "package"]
```

The router's baseline tier for `cam0` sends no `prompts`, so this was the live
path. Of those six words, only `screen` and `door` exist in that room. The
degenerate narratives were not a weak captioner — they were an exhaustive,
accurate report of a six-word world:

```
Two doors and three screens are present in the visual frame.
Three doors and three screens are visible, along with one package.
Multiple doors and screens are visible in the frame with one person detected.
```

Every noun in all 1117 rows comes from that list.

## Corrections to the design doc

Recorded here rather than edited into #1590, since that PR is merged.

1. **P0 will not make `retina_track` run.** The doc reads as though the guard
   is the blocker. The path is dead three layers deep: `_safe_when` raised
   (fixed here); *nothing anywhere in the repo sets `is_video`*, so the guard
   would still be false; and `kind: tracking` is not implemented in
   `_run_profile`, which falls through to `"kind not implemented yet"`.
   ByteTrack is config-only — there is no ByteTrack code. Object continuity
   remains absent, and delivering it is a real implementation task, not a
   guard fix. This answers the doc's open question "is the guard dead by
   construction?" — yes.
2. **BLIP-base is not the information ceiling.** P1 attributes the degenerate
   percepts to the captioner and the paraphrase chain. The ceiling was one
   hardcoded list upstream of the captioner, and moving it cost a config block
   and no measurable latency. The paraphrase-chain critique still stands on its
   own merits; it was just not what was capping this stack.
3. **The camera is aimed at something worth seeing.** The doc's gating
   question, resolved: a desk, two office chairs, two monitors, a whiteboard, a
   guitar case, storage totes, a table, a doorway to a hall, and clutter.
   Easily 15+ nameable objects.

## Also found, not fixed

`retina_detect_open_vocab.params` declares `score_threshold` and `nms_iou`, but
`_run_detect_grounding_dino` reads `box_threshold` and `text_threshold` and
never looks at either declared key. Both silently fall back to the code default
of `0.25`. An operator tuning `score_threshold` — the obvious knob for the
detection flicker discussed under Risks — would change nothing and have no
indication why. Left alone deliberately: making it live changes detection
thresholds, which needs measurement first rather than a drive-by edit.

## Schema / bus / API changes

None. `VisionEventPayload`, the scribe contract, and the `orion:vision:*`
channel set are untouched. `vision_events` rows carry richer narrative text in
the same shape.

## Env/config changes

No env keys added, removed, or renamed. No `.env_example` touched, so no sync
was required. `config/vision_profiles.yaml` gained `default_prompts` under
`profiles[retina_detect_open_vocab].params`.

## Tests run

```text
$ PYTHONPATH=. pytest services/orion-vision-host/tests -q \
    --ignore=services/orion-vision-host/tests/test_heartbeat_chassis.py
29 passed, 1 warning in 0.52s
```

`test_heartbeat_chassis.py` cannot collect outside the container
(`ModuleNotFoundError: No module named 'PIL'`). Confirmed pre-existing by
running it on unmodified `main` — same failure, unrelated to this branch.

## Evals run

```text
None. services/orion-vision-host has no evals/ directory.
```

The acceptance checks below are live runtime measurements, not an eval harness.
The service still lacks one; the natural first eval is the P1 check from the
design doc (distinct-narrative rate on comparable scenes), which now has a real
before/after to anchor it.

## Docker/build/smoke checks

```text
$ bash scripts/safe_docker_build.sh orion-vision-host config
  -> valid

$ bash scripts/safe_docker_build.sh orion-vision-host build
  -> Image orion-vision-host-vision-host Built

$ bash scripts/safe_docker_build.sh orion-vision-host up -d
  -> Container orion-athena-vision-host Started

$ curl -fsS http://localhost:32797/health
  {"ok":true,"service":"vision-host",...,"bus_enabled":true}

29 tasks completed, all "ok": true, "error": null
0 tracebacks, 0 "kind not implemented" warnings
0 "when eval failed" warnings  (was 347 per 30 min)
inference_s ~0.63s  (was ~0.60s pre-deploy, 26 prompts vs 6)
```

## Acceptance checks vs the design doc

P0's stated checks:

- *"zero `[PIPE] when eval failed` warnings"* — **met**, 0 since restart.
- *"at least one artifact with non-empty `tracks`"* — **not met, and not
  achievable by this patch.** See correction 1: tracking is unimplemented.
- *"the camera's actual view described in the PR from reviewed frames"* —
  **met**, above.

## Restart required

Already applied. For reference, or to roll back by rebuilding from `main`:

```bash
cd /mnt/scripts/Orion-Sapienform-vision-host-track-guard
bash scripts/safe_docker_build.sh orion-vision-host build
bash scripts/safe_docker_build.sh orion-vision-host up -d
curl -fsS http://localhost:32797/health
```

## Risks / concerns

- **Severity: medium.** Event rate rose from 2.3/hour (19-day baseline) to a
  post-deploy burst of 95/hour. Each `vision_event` is a council LLM call, so
  a sustained rate at that level is a real cost increase. The per-minute series
  is decaying (9, 4, 2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 3), which is what
  re-habituation to a larger label vocabulary should look like, and the council
  gate still reports `stable_scene` 165× against `salient_labels_changed` 7×.
  *Mitigation:* 14 minutes is too short to call the steady state — this needs a
  re-measure over 24h. If it settles high, the cause is most likely borderline
  detections flickering across 26 classes at a 0.25 threshold, and the fix is
  the threshold knob described under "Also found" (which must be made live
  first). Fully reversible: delete the `default_prompts` block and rebuild.
- **Severity: low.** 26 prompts is a judgement call, not a measured optimum.
  Near-synonyms make GroundingDINO emit duplicate boxes for one object; the
  list was curated for visual distinctness but not tuned against data.
- **Severity: low.** The vocabulary is global to the profile, so a second
  camera pointed somewhere else inherits a living-room word list. The
  per-camera seam already exists (`config/vision_frame_router.yaml` supports a
  per-camera `request.prompts`, as `porch_eye` demonstrates) and should be used
  when a second stream goes live.
- **Privacy.** `face` was dropped rather than carried forward from the fallback
  list; identity/face/re-ID remains a non-goal. The vocabulary is furniture and
  objects. `person` is retained, as it already was. No frames are attached to
  this report.

## What P0 does not deliver

Object continuity. `retina_track` still never runs, and this branch does not
change that. Episodes (Movement II), and anything that needs to know an object
persisted across frames, remain blocked on a real tracking implementation.
