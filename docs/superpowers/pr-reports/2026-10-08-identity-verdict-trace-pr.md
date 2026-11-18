## Summary

- `orion-vision-window` now logs every `identity_face` verdict at INFO (`[WINDOW] identity_check ...`): outcome (`no_face` / `not_enrolled` / `unsure` / `possible` / `probable`), face count, best similarity, detector confidence, correlation id, running totals.
- New pure helper `projection.identity_verdict_summary()`; 4 regression tests.
- No schema, bus, env, or storage change. Numbers only -- no embeddings, no pixels.

## Outcome moved

2026-10-08: Juniper sat in the cam0 office for ~13 min; the matcher ran 26 times (host log) and the system could not say whether it saw a face, matched, or failed. "No face" and weak matches were dropped by `continue` with no log, and real verdicts were DEBUG only. Now every check leaves a trace, so the next step (tune or re-enroll) is driven by data.

## Current architecture

host `identity_face` (MTCNN full-frame + InceptionResnetV1 vs 1 enrolled gallery) -> `orion:vision:artifacts:identity` -> window `_consume_identity` -> `presence.subject` (latest row per camera, overwritten). Frames live 1 h in percept-store (deliberate).

## Files changed

- `services/orion-vision-window/app/projection.py`: `identity_verdict_summary`
- `services/orion-vision-window/app/main.py`: INFO log + `_identity_checks` counter
- `services/orion-vision-window/tests/test_identity_verdict_summary.py`: new

## Tests run

`.venv/bin/python -m pytest services/orion-vision-window/tests/test_identity_verdict_summary.py services/orion-vision-window/tests/test_identity_evidence_wiring.py` -> 21 passed.

## Evals run

None; no eval harness for window identity. Live smoke is the eval: after deploy, stand in front of cam0 and read `docker logs orion-athena-vision-window | grep identity_check`.

## Restart required

Deploy from primary checkout on main after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-vision-window up -d --build
```

## Risks / concerns

- Low: one INFO line per identity check (~2/min while someone is in view).
- Not fixed here (needs the data this PR produces): why recognition may fail (face too small at 640x480, enrollment from a different camera, thresholds 0.35/0.55). Sightings table empty and 15:00-16:00 / 17:00-19:00 UTC event gaps are separate and unexamined.
