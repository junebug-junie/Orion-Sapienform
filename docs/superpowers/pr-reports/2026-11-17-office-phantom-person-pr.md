## Summary

- A pile of boxes on the office floor read as a person all day on Sunday and Monday (2026-11-15..16). Orion sent Juniper check-ins about "her" packing boxes, with nobody in the room.
- The office camera (cam0) now needs a score of 0.55 before it keeps a `person` box. It was 0.35, the shared default.
- New eval: `services/orion-vision-host/evals/office_person_cutoff_eval.py`, with labelled boxes in `evals/fixtures/office_person_boxes.csv`. There are no images in it.
- The "never moves = prop" rule is NOT turned on for the office. Review showed that a marked prop stays marked for 24 h, even across gaps. One long desk session would then hide Juniper at her desk on every later visit. Tracked in #3614.

## Outcome moved

- Before: all 2,396 box-pile detections were kept. They opened office visits, the captioner described a person, and Orion reached out to Juniper.
- After, replayed on the same data: 0 of the 2,396 box-pile detections are kept.
- Real-person detections kept: 1,085 of 1,137 (95%).
- Real-person detections with a face kept: 61 of 62.

## Current architecture

All cameras use one GroundingDINO profile (`retina_detect_open_vocab`). The walkway has per-label cut-offs that were tuned against Juniper's verdicts. The office was left on the 0.35 default and had only a size floor.

The pile's detections had these traits:

- All were the same box, at about x 1085–1305 and y 798–918 on a 2560×1920 frame.
- Scores were 0.35–0.52, with a mean of 0.41.
- None had a face.

Real office people, from the 479 crops before 11-14, score:

- p10 0.61 and median 0.68.
- p10 0.68 when a face is present.

## Architecture touched

- `config/vision_profiles.yaml`: `box_threshold_by_stream.cam0.person: 0.55`. Other office labels stay on `score_threshold`.

## Files changed

- `config/vision_profiles.yaml`: sets the office person cut-off and records the evidence.
- `services/orion-vision-host/tests/test_box_thresholds.py`: adds a shipped-config test. The runner test now expects the office to drop a 0.5 `person`.
- `services/orion-vision-host/evals/office_person_cutoff_eval.py` and `evals/fixtures/office_person_boxes.csv`: the eval and its labelled data.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: office `person` boxes under 0.55 are dropped at the detector.
- Compatibility notes: none

## Env/config changes

- No env keys changed. The change is to the profile YAML only.
- `.env_example` updated: no
- local `.env` synced: n/a

## Tests run

```text
vision-host image: pytest tests/test_box_thresholds.py -> 14 passed
vision-host image: pytest tests/ -> 4 failures (test_heartbeat_chassis x2, test_identity_face_broadcast_suppression x2); the same 4 fail on main
```

## Evals run

```text
.venv/bin/python services/orion-vision-host/evals/office_person_cutoff_eval.py
cut-off cam0 person = 0.55
  prop boxes kept: 0/2396
  real boxes kept: 1085/1137 (95%)
  real boxes with a face kept: 61/62
PASS
```

## Docker/build/smoke checks

```text
The tests ran inside the existing orion-vision-host-vision-host image. No rebuild was needed for that.
Live behaviour: UNVERIFIED until vision-host is rebuilt from main.
```

## Review findings fixed

- Finding: a 3 h office hold for the stationary rule would make Juniper's seat a sticky prop. A marked prop is kept 24 h, and gaps never break a marked run.
  - Fix: the hold is dropped and the office stays out of the stationary rule. The cut-off alone removes every box-pile detection. #3614 tracks a room-safe rule.
  - Evidence: the eval PASS above.
- Finding: the eval docstring had a wrong count (2,397 instead of 2,396) and described a hold check it no longer has.
  - Fix: the eval is now cut-off only, with the right count.
- Finding: no CI covers the vision-host tests or `vision_profiles.yaml`.
  - Fix: filed #3615.
- Finding: the change is not retroactive. The 11-15/16 office sessions and the box-pile individual stay as stored.
  - Fix: none. This is noted here.

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
```

Run it from the clean primary main checkout after the merge.

## Risks / concerns

- Severity: low
  - Concern: about 5% of real office person boxes score 0.55–0.61, so some real visits in poor light may be missed.
  - Mitigation: re-run the eval with fresh crops after a week.
- Severity: low
  - Concern: a prop that scores above 0.55 would still read as a person, because the office has no prop rule.
  - Mitigation: #3614.
- Issues opened: #3614, #3615. Closed: none.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
