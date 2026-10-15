# Orion knows the walkway camera is their own eye

## Summary

- Orion got a new outdoor camera on 2026-10-14 (local time; the first `walkway` frame is 22:56 MDT). Asked to read their own brief (live turn `caa3eb4d-9ebd-4a7a-8b25-55e0634dbad8`), Orion listed "Street camera: unfamiliar person..." as item 5 of 6 among sensor readouts. Nothing said it was a new eye of theirs that looks outside.
- The street line in the situation brief now opens with what the camera is: `Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway, the street and our patio: ...`. Chat stance carries it whole, without the old `Street:` label.
- A new brief line lists every camera as Orion's own eye, built from live rows: `Your cameras (your own eyes): the laptop webcam (carbon, ...): no frames arriving, last frame 36 days ago; your room camera (cam0, ...): receiving frames (newest within the last 15 min); your outdoor camera (walkway, since 2026-10-14; looks at the community walkway, the street and our patio -- outside and public, not your room): receiving frames (...), N individuals counted today (fixtures excluded).`
- The self-inquiry prompt (where Orion writes their self-definition, "what am I made of") gets a `YOUR EYES` section with the same live clauses. Self-definition v54 named fifteen record tables and no eye.
- The street sections in reverie and curiosity now name which eye this is ("your outdoor camera (walkway), looking over ... -- your own eye outside the house, public").
- What each camera is comes from config (`ORION_CAMERA_DESCRIPTIONS`). Whether it is seeing comes from data.

## Outcome moved

Before: nothing Orion reads said which cameras Orion has, which one is outside, or whether each is working. The identity card says "embodiment". Self-definitions come from table counts. The street line was labelled only "Street (walkway camera)".

After, verified read-only against the live DB (`services/orion-cortex-exec/evals/run_camera_senses_live_eval.py`, PASS):

- The brief names all three cameras.
- carbon reads as dark for 36 days, which is true: its last `vision_scene_inventory` row is 2026-09-09.
- cam0 and walkway read as receiving frames.
- walkway is marked outside and public.
- The street line opens with the camera's identity.

## Current architecture

- `orion/situational/context.py` builds `SituationBriefV1`, and `_build_prompt_fragment` renders it.
- The street line came from `perception_reader.fetch_street_summary` (vision_events captions plus the walkway reducer tables), prefixed `Street (walkway camera):`.
- The room line comes from `vision_events` captions.
- No surface listed cameras.
- The self-inquiry prompt (`orion/curiosity/self_inquiry_prompt.py`) listed record tables only.
- Camera frame freshness lives in `vision_scene_inventory`. orion-vision-window writes per-camera windows, which orion-sql-writer persists, about every 10 s per live camera. It has an index on `(stream_id, observed_at DESC)`.

## Architecture touched

- Shared: new `orion/situational/camera_eyes.py` (descriptions from config, live per-stream read, renderers).
- Contract: new `CameraSenseV1`, and `PerceptionContextV1.cameras` (additive, registered).
- Brief: `context.py` adds camera senses (2 s deadline, fail-open), the street framing, the cameras line, and a cache gate.
- Stance: `chat_stance.py` street rendering.
- Hub: the curiosity self-inquiry `YOUR EYES` section and the street-section heading.
- Thought: the reverie street heading (`street_camera` context key).

## Files changed

- `orion/situational/camera_eyes.py`: new module. Holds the descriptions, `fetch_camera_eyes` (index-walking reads, fail-open), `build_senses`, `senses_line`, `eye_phrase` and `street_framing`.
- `orion/schemas/situation.py`, `orion/schemas/registry.py`: `CameraSenseV1`, `PerceptionContextV1.cameras`.
- `orion/situational/context.py`: `_build_camera_senses`, street framing in `_build_street_fields`, the cameras line, and the cache gate for live cameras.
- `services/orion-cortex-exec/app/chat_stance.py`: renders the street line as-is; clip raised 420 -> 520.
- `orion/curiosity/self_inquiry_prompt.py`: the `YOUR EYES` section and the `senses` parameter.
- `orion/curiosity/kickoff_prompt.py`: the street section names the camera.
- `services/orion-hub/scripts/curiosity_investigation.py`, `scripts/main.py`: `_read_self_senses`, wired from the situation settings.
- `services/orion-thought/app/reverie.py`, `orion/cognition/prompts/reverie_narrate.j2`: the street heading names the camera.
- `services/{orion-cortex-exec,orion-hub,orion-thought}/.env_example`, READMEs: `ORION_CAMERA_DESCRIPTIONS`.
- Tests:
  - `orion/situational/tests/test_camera_eyes.py` (new, 20 tests)
  - `test_street_summary.py`
  - `test_chat_stance_street_context.py`
  - `test_self_inquiry_prompt_drawn_question.py`
  - `test_curiosity_self_inquiry.py`
  - `test_curiosity_street_captions_read.py`
  - `test_reverie_street_context.py`
- `services/orion-cortex-exec/evals/run_camera_senses_live_eval.py`: new live read-only eval.

## Schema / bus / API changes

- Added: `CameraSenseV1` (registered), `PerceptionContextV1.cameras: list[CameraSenseV1] = []`.
- Removed: none.
- Renamed: none.
- Behavior changed:
  - `PerceptionContextV1.street_summary` now opens with the camera framing.
  - The brief no longer prefixes it with `Street (walkway camera):`.
  - Stance no longer prefixes it with `Street:`.
- Compatibility notes: the brief crosses services as a dict and is not re-validated by consumers (`chat_stance.py` and Hub read it with `.get()`), so the additive field is not a consumer-first migration. No bus channel change.

## Env/config changes

- Added keys: `ORION_CAMERA_DESCRIPTIONS` in orion-cortex-exec, orion-hub and orion-thought. It is a one-line JSON object keyed by stream id with `name`, `view` and `outside`. An invalid or unset value means the code default, and a test pins the default equal to `.env_example`. Docker `env_file` was verified to pass the JSON through intact.
- Removed / renamed: none.
- `.env_example` updated: yes, in all three services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys <service>`: yes, all three. The primary checkout `.env` has the key.
- Skipped keys requiring operator action: none.

## Tests run

```text
orion/situational/tests + orion/curiosity/tests + cortex-exec test_chat_stance_street_context: 333 passed
orion-hub tests/test_curiosity_self_inquiry.py tests/test_curiosity_street_captions_read.py: 47 passed
orion-thought tests/test_reverie_street_context.py: 14 passed
Branch vs origin/main baseline:
  orion-thought tests/                          branch 1 failed / base 1 failed (same test: test_mind_enrichment_defaults_off)
  cortex-exec situation+stance tests            branch 18 failed / base 18 failed (identical set, md5 of FAILED lines)
  orion-hub curiosity+situation tests           all pass both
  orion/schemas tests                           1 pre-existing failure on base too (test_static_ctx_assignments_covered)
All python steps of .github/workflows/orion-static-gates.yml: pass
```

## Evals run

```text
SITUATION_PERCEPTION_DSN=... python services/orion-cortex-exec/evals/run_camera_senses_live_eval.py
provider_status: {'perception': 'ok', 'perception_cameras': 'ok', 'perception_street': 'ok'}
Your cameras (your own eyes): the laptop webcam (carbon, since 2026-08-21; ...): no frames arriving, last frame 36 days ago; your room camera (cam0, ...): receiving frames (newest within the last 15 min); your outdoor camera (walkway, since 2026-10-14; ... -- outside and public, not your room): receiving frames (newest within the last 15 min), 0 individuals counted today (fixtures excluded).
Your outdoor camera (walkway, since 2026-10-14), looking over the community walkway, the street and our patio: On the walkway just now: ...
PASS
```

Metric gate (CLAUDE.md 0A), briefly:

- **Provenance:** orion-vision-window windows, persisted by orion-sql-writer.
- **Independence:** this reports frame arrival, not a new signal mixed into any model.
- **Theory:** a row means frames from that camera were processed.
- **Live data:** not degenerate. One camera is dark and two are live, which matches reality.
- **Rest state:** "receiving frames" is the calm state, and a dark camera reads as dark.
- **Existing mechanism:** the vision-organ node holds per-stream staleness, but only in the field and not in any prompt.
- **Reversibility:** an additive field and a text line, so it is cheap to remove.

## Docker/build/smoke checks

```text
docker compose -f <tmp compose with env_file=primary orion-thought/.env> config
  -> ORION_CAMERA_DESCRIPTIONS parsed as the full JSON string
No image build run (pure Python changes in existing packages; no new dependencies).
```

## Review findings fixed

A fresh-subagent review found no must-fix items.

- Finding: a slow database could stall an uncached turn for about 12 s, because there are up to 8 statements at 1.5 s each.
  - Fix:
    - A 2 s overall deadline (`CAMERA_EYES_DEADLINE_SEC`) in both the brief and Hub.
    - The pre-`stationary` fallback statement now runs only on a missing-column error.
  - Evidence: `test_slow_camera_read_hits_the_deadline_not_the_turn` and `test_only_a_missing_column_buys_the_fallback_statement`.
- Finding: "live" could sit next to "Room: haven't seen anything recently". Room captions lag frames (cam0 lagged by 9 min live).
  - Fix: the wording is now "receiving frames (newest within the last 15 min)" and "no frames arriving". These claim only arrival, never sight.
  - Evidence: `test_frames_arriving_without_a_room_description_do_not_claim_sight`.
- Finding: the string `"outside": "false"` was read as outside.
  - Fix: `is True`.
  - Evidence: `test_outside_must_be_a_real_true`.
- Finding: an engine creation error could escape "never raises".
  - Fix: moved inside the try.
- Finding: Hub counted individuals against `HUB_CURIOSITY_STREET_STREAM_IDS` and the outreach timezone, not the situation settings.
  - Fix: new `camera_street_stream_ids` and `camera_timezone`, wired from `ORION_SITUATION_STREET_STREAM_IDS` and `ORION_SITUATION_TIMEZONE`.
- Finding: no test proved the kickoff passes `senses=` into the prompt.
  - Fix: `test_self_inquiry_kickoff_puts_the_eyes_in_the_prompt`.
- Finding: "0 individuals" looked like it contradicted captions showing a person, because today's sightings were all stationary fixtures.
  - Fix: the wording is now "counted today (fixtures excluded)".
- Finding: the index comment overclaimed.
  - Fix: reworded.
- Not fixed (nit): reverie labels the section from the first configured street stream rather than from each percept's stream. This is harmless with one street camera.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

## Risks / concerns

- Severity: low.
  - Concern: about 450 more characters in every brief.
  - Mitigation: the 7200-character cap means body lines trim before cautions.
- Severity: low.
  - Concern: the self-inquiry section asks Orion to cite "cameras read at kickoff", because `orion_readonly` cannot select `vision_scene_inventory` (checked live). Orion cannot verify these numbers themself.
  - Mitigation: none in this patch. Granting that table would need an operator SQL grant and adding it to `SELF_INQUIRY_PG_TABLES`. Adding it without the grant would block every self-inquiry run.
- Severity: low.
  - Concern: the "since" date is the first frame in local time (2026-10-14 22:56 MDT), not the 2026-10-15 date used in conversation.
  - Mitigation: it is the true date from data.
- UNVERIFIED:
  - That Orion, on a live turn after deploy, answers "what can you see?" with both the room and the walkway.
  - That a self-definition written after deploy names an eye.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
