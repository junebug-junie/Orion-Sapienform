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

- `config/vision_frame_router.yaml`: cam0 `identity_dispatch.min_seconds_between_dispatch` 30 → 5.
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
orion-vision-window        pytest tests -q                       113 passed
orion-durable-runs         pytest tests/test_situation_graph.py   33 passed
orion-vision-frame-router  pytest tests -q                       81 passed
```

## Risks / concerns

- **Medium: more chance of mistaking someone else for Juniper.**
  - Concern: a single 0.55 match, or two 0.35+ matches, could come from another household member. Nobody has measured how others score against Juniper's gallery.
  - Mitigation: a sighting holds only 2 h and never erases what Juniper said. Each sighting logs its similarity and its `outcome`, so a week of data shows the false-positive rate.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-durable-runs orion-vision-window; do docker compose --env-file .env --env-file services/$s/.env -f services/$s/docker-compose.yml up -d --build; done && docker restart orion-orion-athena-vision-frame-router
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
