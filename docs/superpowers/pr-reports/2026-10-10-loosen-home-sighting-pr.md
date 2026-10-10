## Summary

- **Juniper waved at the room camera (cam0) at 6:32 pm MDT and wasn't picked up.** She was in view about 45 s, and the identity check runs once per 30 s, so she got exactly one check. It scored 0.20 ("unsure").
- **Face checks on cam0 now run every 5 s instead of 30 s** (`config/vision_frame_router.yaml`). Each check is 0.06–0.14 s of GPU inference, and a short visit now gets about 9 tries.
- **New sighting rule.** **One "probable" match (≥0.55) is enough on its own,** or **two "possible"-or-better (≥0.35) within 10 min.** It was two probable matches before.
- **`IdentitySightingV1.outcome` records which rule qualified:** `probable` or `corroborated`.

## Outcome moved

A brief, deliberate visit to cam0 can now become a home sighting. Before, a visit shorter than about a minute could never qualify.

## Not addressed: the 0.20 score

No threshold fixes a 0.20 score. That is the same range as a stranger's face. The enrolled gallery is 6 photos averaged together (08-26), probably not taken from the ceiling angle cam0 sees. The follow-up is to re-enroll from cam0 frames: frames are saved under `/mnt/telemetry/vision/frames`, and enrollment is done with the human-run `services/orion-vision-host/scripts/enroll_identity_face.py`.

## Files changed

- `config/vision_frame_router.yaml`: cam0 `identity_dispatch.min_seconds_between_dispatch` 30 → 5; `global.max_inflight_total` 2 → 3.
- `orion/schemas/vision_sighting.py`: `outcome: "probable" | "corroborated"`.
- `services/orion-vision-window/app/main.py`, `app/settings.py`, `.env_example`: the rule.
- Tests: `services/orion-vision-window/tests/test_identity_sighting.py`, `services/orion-durable-runs/tests/test_situation_graph.py`.

## Schema / bus / API changes

- **Behavior changed:** `IdentitySightingV1.outcome` gains `"corroborated"`.
- **Compatibility:** `extra="forbid"` with a `Literal`, so the consumer deploys first. Deploy **orion-durable-runs before orion-vision-window**; an old durable-runs would drop corroborated sightings.

## Env/config changes

- No new keys.
- `WINDOW_SIGHTING_MIN_MATCHES=2` now counts possible-or-better matches; the comment is updated.
- The router config is mounted from the main checkout, so it needs a restart, not a rebuild.

## Tests run

```text
orion-vision-window        pytest tests -q                       116 passed
orion-durable-runs         pytest tests/test_situation_graph.py   33 passed
orion-vision-frame-router  pytest tests -q                       81 passed
```

## Review findings fixed

- Finding (medium): an identity job counts toward the router's global cap of 2 jobs. With cam0 checking faces every 5 s, cam0's detection plus its face check would fill both slots, and carbon and walkway frames would drop (`global_inflight_limit`) about half the time cam0 sees a person.
  - Fix: `max_inflight_total` 2 → 3.
  - Watch after deploy: router skip counts for carbon and walkway.
- Finding (low): matches were keyed by camera only, so a near-match for another person could corroborate one for Juniper.
  - Fix: matches are keyed by (camera, person).
  - Evidence: `test_matches_for_different_people_do_not_corroborate`.
- Finding (low): the outcome wasn't logged anywhere.
  - Fix: the `identity_sighting` log now carries the outcome and the number of matches in the window.
- Finding (low): untested edges.
  - Fix: added `test_after_the_hold_the_next_probable_publishes_again` and `test_possible_matches_further_apart_than_the_window_do_not_corroborate`.
- Finding (low): stale "30s" comments in the router config and the vision-window settings.
  - Fix: updated.

## Risks / concerns

- **Low: more load on vision-host.**
  - Concern: with 3 jobs allowed at once instead of 2, vision-host can be asked for more work at the same time. Each identity job is 0.06–0.14 s of inference, and queue wait is already about 2.4 s.
  - Mitigation: watch `queue_wait_est_s` after deploy.


- **Medium: more chance of mistaking someone else for Juniper.**
  - Concern: a single 0.55 match, or two 0.35+ matches, could come from another household member. Nobody has measured how others score against Juniper's gallery.
  - Mitigation: a sighting holds only 2 h and never erases what Juniper said. Each sighting logs its similarity and its `outcome`, so a week of data shows the false-positive rate.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-durable-runs orion-vision-window; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done && docker restart orion-orion-athena-vision-frame-router
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
