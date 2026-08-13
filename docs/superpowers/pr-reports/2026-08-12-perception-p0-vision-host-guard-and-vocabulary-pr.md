# PR report — perception P0: pipeline guard fix + detector vocabulary

Implements **P0** of the perception frontier design (PR #1590,
`docs/superpowers/specs/2026-08-12-perception-frontier-design.md`), plus the
finding P0's investigation turned up that the design doc did not anticipate.

## Summary

- Fixed the live `_safe_when` bug: pipeline `when:` guards raised
  `AttributeError` on any request flag the caller omitted, logging a warning on
  every vision task (347 in 30 minutes on `orion-athena-vision-host`).
- **Reviewed real cam0 frames** — the second half of P0, which the doc calls the
  gate on everything below it. The camera watches an information-rich home
  office, not a low-information wall.
- **Found the real information ceiling:** `retina_detect_open_vocab` shipped
  with no `default_prompts`, so the runner fell back to a hardcoded six-word
  list. An open-vocabulary detector can only name what it is asked about.
- Shipped a 25-term vocabulary, then tuned it twice against live output —
  once for token-span bleed, once for false-positive counts.
- Wired two config knobs that were declared and never read (`nms_iou`,
  `score_threshold`), added per-label NMS and score-ordered capping.
- Deleted a stale shadow config that shipped in the image carrying a `face`
  prompt against the identity/re-ID non-goal.
- Corrected three claims in the design doc that live inspection contradicted,
  and one of my own that live measurement contradicted.

## Outcome moved

Live `vision_events`, same camera, same room:

| | Before | After |
| --- | --- | --- |
| Narrative | *"Two doors and three screens are present in the visual frame."* | *"Three chairs, two tables, one door, one box, and one screen are visible in the scene."* |
| Nouns the stack can produce | `door`, `screen`, `package`, `person` | + `chair`, `table`, `desk`, `box`, `book`, `guitar`, `laptop`, `lamp`, … (25 terms) |
| Distinct narratives | 211 / 1117 (19%) over 19 days | 16 / 18 in the first post-deploy window |
| Objects per frame | 26, incl. one desk counted 6× | 8, against a true 2 chairs / 1 desk / 1 table / 2 doors / 2 monitors |
| `[PIPE] when eval failed` | 347 / 30 min | 0 |

Object counts are now within roughly ±1 of ground truth per class instead of
inflated 3–6×. They are still approximate — see Risks.

## Current architecture

`orion-vision-edge` → `orion-vision-frame-router` → `orion-vision-host`
(GroundingDINO + BLIP + CLIP) → `orion-vision-window` (label habituation) →
`orion-vision-council` (text-model interpretation) → `orion-vision-scribe` →
Postgres `vision_events`. Seven services, all up, and no cognition surface
reads the terminal table.

## Architecture touched

`services/orion-vision-host` only. No contract, bus, schema, or env change.
`config/vision_profiles.yaml` is `COPY`d into the image at build time
(`Dockerfile:30`), so one rebuild carries code and config together.

## Files changed

- `services/orion-vision-host/app/when_guard.py`: new. AST-validated guard
  evaluation, dependency-light so it is testable without torch.
- `services/orion-vision-host/app/detections.py`: new. Per-label NMS and
  score-ordered capping, pure Python.
- `services/orion-vision-host/app/runner.py`: delegates to both; honours
  `nms_iou` and `score_threshold`; records them in the stored artifact.
- `config/vision_profiles.yaml`: `default_prompts`, tuned `score_threshold`.
- `services/orion-vision-host/app/config/vision_profiles.yaml`: **deleted.**
- `services/orion-vision-host/tests/test_when_guard.py`: new, 11 tests.
- `services/orion-vision-host/tests/test_detections.py`: new, 12 tests.
- `services/orion-vision-host/tests/test_detector_vocabulary.py`: new, 6 tests.

## The two bugs

### 1. Guards raised on absent flags

`_safe_when` wrapped the request in `SimpleNamespace(**request)`, which raises
`AttributeError` for any key the caller omitted. Every optional step in every
pipeline is guarded on exactly such a key — `request.is_video`,
`request.want_masks`, `request.want_embeddings`:

```
[PIPE] when eval failed expr=request.is_video == True
       err='types.SimpleNamespace' object has no attribute 'is_video'
```

Absent now resolves to `None` (falsy). **Honest scope:** every shipped guard
uses `== true`, for which the old code also returned `False` — by raising and
swallowing. So this fixes a per-task warning and an incorrect failure mode; it
does **not** un-skip any pipeline step. Nothing that was skipped now runs.

### 2. The detector had a six-word vocabulary

`runner.py:393`, reached because neither the request nor the profile supplied
prompts, and the router's baseline tier for `cam0` sends none:

```python
prompts = ["person", "face", "phone", "screen", "door", "package"]
```

Of those six words, only `screen` and `door` exist in that room. The degenerate
narratives were not a weak captioner — they were an accurate report of a
six-word world. Every noun in all 1117 rows comes from that list.

## Two rounds of tuning, both driven by live output

Neither was predictable from code; both came from watching what the running
stack actually emitted.

**Round 1 — token-span bleed.** `cardboard box . storage bin` produced the
believed label set `cardboard box storage, …, storage` and never `storage bin`.
GroundingDINO resolves detections to token spans in the dot-joined caption, and
adjacent multi-word phrases bleed together. Switched every term to a single
word and added a test so a multi-word term cannot reintroduce it.

**Round 2 — false positives, and a correction to my own diagnosis.** One desk
was being reported as six. I assumed overlapping duplicate boxes and wired NMS.
Measurement said otherwise: pairwise IoU between same-label `desk` boxes was
**~0.000** — spatially disjoint low-confidence false positives scattered across
the frame, which NMS cannot touch. Only the confidence threshold can. Swept it
over 6 consecutive frames of a verified-static room holding one desk, one table
and two chairs:

| `score_threshold` | detections | `desk` | `table` | notes |
| --- | --- | --- | --- | --- |
| 0.25 | 119 | 23 (~3.8/frame) | 18 | phantom objects everywhere |
| 0.35 | 57 | 5 (~0.8/frame) | 13 | object classes kept, count roughly halved |
| 0.50 | 20 | 0 | 0 | only `chair` and `door` survive; real objects vanish |

Set **0.35**. NMS is kept because it is correct for the genuinely-overlapping
case and the config already promised it — not because it fixed this.

## Corrections to the design doc

Recorded here rather than edited into #1590, since that PR is merged.

1. **P0 will not make `retina_track` run.** The doc reads as though the guard is
   the blocker. The path is dead three layers deep: `_safe_when` raised (fixed
   here); *nothing anywhere in the repo sets `is_video`*, so the guard would
   still be false; and `kind: tracking` is not implemented in `_run_profile`,
   which falls through to `"kind not implemented yet"`. ByteTrack is
   config-only — there is no ByteTrack code. This answers the doc's open
   question "is the guard dead by construction?": **yes**.
2. **BLIP-base is not the information ceiling.** P1 attributes the degenerate
   percepts to the captioner and the paraphrase chain. The ceiling was one
   hardcoded list upstream of the captioner, and moving it cost a config block
   and no measurable latency. The paraphrase-chain critique still stands on its
   own merits; it was not what was capping this stack.
3. **The camera is aimed at something worth seeing.** The doc's gating question,
   resolved: a desk, two office chairs, two monitors, a whiteboard, a guitar
   case, storage totes, a table, an open doorway to a hall, a second closed
   door, and clutter. Easily 15+ nameable objects.

## Schema / bus / API changes

None. `VisionEventPayload`, the scribe contract, and the `orion:vision:*`
channel set are untouched. `vision_events` rows carry richer narrative text in
the same shape. Stored detection artifacts gain `nms_iou` and `max_detections`
alongside the existing `box_threshold`/`text_threshold` provenance.

## Env/config changes

No env keys added, removed, or renamed. No `.env_example` touched, so no sync
was required. Changes are inside
`config/vision_profiles.yaml` → `profiles[retina_detect_open_vocab].params`:
`default_prompts` added, `score_threshold` 0.25 → 0.35.

## Tests run

```text
$ PYTHONPATH=. pytest services/orion-vision-host/tests -q \
    --ignore=services/orion-vision-host/tests/test_heartbeat_chassis.py
44 passed, 1 warning in 0.87s
```

`test_heartbeat_chassis.py` cannot collect outside the container
(`ModuleNotFoundError: No module named 'PIL'`). Confirmed pre-existing by
running it on unmodified `main` — same failure, unrelated to this branch.

## Evals run

```text
None. services/orion-vision-host has no evals/ directory.
```

The acceptance checks below are live runtime measurements, not an eval harness.
The service still lacks one; the natural first eval is the design doc's P1
check (distinct-narrative rate on comparable scenes), which now has a real
before/after and a ground-truth frame description to anchor it.

## Docker/build/smoke checks

```text
$ bash scripts/safe_docker_build.sh orion-vision-host config   -> valid
$ bash scripts/safe_docker_build.sh orion-vision-host build    -> Image Built
$ bash scripts/safe_docker_build.sh orion-vision-host up -d    -> Container Started
$ curl -fsS http://localhost:32797/health
  {"ok":true,"service":"vision-host",...,"bus_enabled":true}

29 tasks completed, all "ok": true, "error": null
0 tracebacks, 0 "kind not implemented" warnings
0 "when eval failed" / "unknown request flag" warnings   (was 347 per 30 min)
inference_s ~0.63s   (was ~0.60s pre-deploy, 25 prompts vs 6)

Direct probe, POST /v1/vision/task on a live frame:
  before: 26 objects  desk x6, box x5, table x4, chair x3, door x3, screen x3, monitor x2
  after :  8 objects  chair x3, table x2, door x1, box x1, screen x1
```

## Acceptance checks vs the design doc

P0's stated checks:

- *"zero `[PIPE] when eval failed` warnings over 10 minutes"* — **met**, 0 since
  restart.
- *"at least one artifact with non-empty `tracks`"* — **not met, and not
  achievable by this patch.** See correction 1: tracking is unimplemented.
- *"the camera's actual view described in the PR from reviewed frames"* —
  **met**, above.

## Review findings fixed

Code review returned 9 findings. All were either fixed or verified against live
data. The first review run targeted the wrong tree (uncommitted `graphify-out/`
changes in the shared checkout) and was re-run against this branch.

- **Finding: `__getattr__` returns `None` for any unknown name, so a renamed or
  mistyped guard flag becomes a permanent silent no-op.**
  - Fix: unrecognised names warn once per process against a `KNOWN_GUARD_FLAGS`
    set; known flags stay quiet, so this cannot regress into per-task spam.
  - Evidence: `test_unknown_flag_warns_once_so_a_rename_is_not_silent`,
    `test_known_flags_do_not_warn`.
- **Finding: the regression test cannot distinguish "evaluated cleanly to
  False" from "raised and was swallowed to False" — the reverted bug still
  passes it.** Correct, and the most important finding of the review.
  - Fix: the test now asserts on captured loguru output, not the return value.
  - Evidence: re-ran the rewritten test against the original `SimpleNamespace`
    implementation — *"RESULT: test FAILED against old code → test is a real
    regression guard"*. The old implementation returns the same `False`.
- **Finding: detections truncated at `max_detections` in query order, not by
  score.**
  - Fix: `cap_by_score`. Evidence: `test_cap_keeps_highest_scores_not_query_order`.
    Note the cap was not binding in practice (26 detections against a cap of
    60); this is a latent-bug fix, not an observed one.
- **Finding: vocabulary contains near-synonyms and the runner applies no NMS —
  `nms_iou` is declared and never read.**
  - Fix: per-label NMS wired to `nms_iou`; `monitor` dropped as a cross-label
    duplicate of `screen`. Evidence: 12 tests in `test_detections.py`; live
    probe 26 → 8 objects. **Partial credit:** measurement showed the inflation
    was disjoint false positives (IoU ~0.000), not overlap, so the threshold did
    the work — recorded above.
- **Finding: `{"__builtins__": {}}` is not a sandbox; the test asserted a
  guarantee the module did not have.**
  - Fix: expressions validated against an AST allowlist.
  - Evidence: `test_guard_language_rejects_calls_and_arbitrary_names` covers
    `().__class__.__bases__[0].__subclasses__()`, which the reviewer verified
    escaped the old evaluator.
- **Finding: the `re.sub` rewrite corrupts `true`/`false` inside string
  literals; an attribute named `true` becomes a `SyntaxError`.**
  - Fix: removed textual substitution entirely; `true`/`false` are bound as
    names at eval time. Evidence: `test_string_literals_are_not_rewritten`.
- **Finding: shadow config `app/config/vision_profiles.yaml` still ships a
  `face` prompt, and the identity-term test only checks the root file.**
  - Fix: deleted. Verified it ships in the image (`/app/app/config/`, 1450 B)
    but is never loaded (`VISION_PROFILES_PATH=/app/config/…`) and self-declares
    as non-canonical.
- **Finding: `score_threshold`/`nms_iou`/`labels_hint` are dead knobs.**
  - Fix: `score_threshold` honoured as an alias for `box_threshold`, `nms_iou`
    honoured. `labels_hint` left alone — it is genuinely superseded by
    `default_prompts` and removing it is a config-contract change.
  - Evidence: artifact JSON now records `box_threshold: 0.35`, `nms_iou: 0.6`.
- **Finding: the PR report should not imply the guard fix un-skips pipeline
  steps.**
  - Fix: stated explicitly under "The two bugs" above.

## Restart required

Already applied. To rebuild or roll back:

```bash
cd /mnt/scripts/Orion-Sapienform-vision-host-track-guard
bash scripts/safe_docker_build.sh orion-vision-host build
bash scripts/safe_docker_build.sh orion-vision-host up -d
curl -fsS http://localhost:32797/health
```

## Risks / concerns

- **Severity: medium. Event-rate blast radius is not settled.** Each
  `vision_event` is a council LLM call. The 19-day baseline was 2.3/hour; the
  first post-deploy window burst to 95/hour, then decayed (9, 4, 2, 1, 1, 0, 0,
  0, 0, 0, 1, 0, 3 per minute), which is what re-habituation to a larger label
  vocabulary should look like. The council gate held `stable_scene` 214× against
  `salient_labels_changed` 13×, and the tightened threshold has since cut label
  churn further. **This needs a 24h re-measure before the steady state is
  known** — the windows here are minutes long, and short-window rates in this
  repo have been wrong before. Fully reversible: revert the `default_prompts`
  block and rebuild.
- **Severity: medium. Object counts are approximate and should not be read as
  inventory.** At 0.35 the detector still reported 3 chairs for 2, and 1 screen
  for 2. No threshold in the sweep gives correct counts: by 0.50 real objects
  disappear entirely. This is a detector-capability limit on this scene, not a
  config bug, and the narratives inherit it — "three chairs" is a claim the
  stack cannot currently justify. Worth knowing before any consumer treats
  `objects` as a count rather than a presence signal.
- **Severity: low.** 25 prompts is a curated judgement, not a measured optimum.
  Remaining plausible cross-label overlaps (`box`/`bin`, `bag`/`backpack`) were
  left in; per-label NMS does not merge them.
- **Severity: low.** The vocabulary is global to the profile, so a second camera
  pointed elsewhere inherits a home-office word list. The per-camera seam
  already exists (`config/vision_frame_router.yaml` supports per-camera
  `request.prompts`, as `porch_eye` shows) and should be used when a second
  stream goes live.
- **Privacy.** `face` was dropped from the vocabulary rather than carried
  forward, and the shadow config that reintroduced it was deleted.
  Identity/face/re-ID remains a non-goal. The vocabulary is furniture and
  objects; `person` is retained, as it already was. No frames are attached to
  this report.

## What P0 does not deliver

Object continuity. `retina_track` still never runs, and this branch does not
change that. Episodes (Movement II), and anything needing an object to persist
across frames, remain blocked on a real tracking implementation — which is a
build, not a config fix.
