# Social shakedown workflow

The social shakedown loop turns live-room weirdness into repeatable regressions without introducing a new runtime.

## Workflow

1. notice a live or replayed behavior that feels off
2. record it as a `SocialShakedownIssueV1`
3. link it to a replay scenario (or add a new one)
4. tune the bounded heuristics or prompt grounding that caused the issue
5. record the applied change as a `SocialShakedownFixV1`
6. re-run the linked replay scenario through the existing harness and mark the issue verified once it passes

## Issue categories

The first-pass categories are intentionally compact:

- repair tone
- bridge summary behavior
- clarifying-question behavior
- handoff / closure behavior
- stale-context behavior
- calibration / freshness phrasing
- re-entry / snapshot overreach
- safety-boundary handling

## Live finding -> replay regression

A live finding should become a replay regression when it can be described with:

- the room/thread setup that produced the weirdness
- the observed behavior
- the expected behavior
- the narrow seam that should change (routing, deliberation, context selection, prompt wording, repair tone, floor preference)

That replay regression then becomes the durable reference point for future tuning.

## Expected operator / developer loop

- operators or developers add/update issue and fix entries in `tests/fixtures/social_room/shakedown_issues.json`
- replay scenarios live in `tests/fixtures/social_room/scenario_replay.json`
- the harness plus shakedown workflow are exercised with pytest:

```bash
pytest -q tests/test_social_room_scenario_replay.py tests/test_social_room_shakedown.py
```

This keeps the loop small: live weirdness, capture it, tighten the bounded heuristic, and keep the regression around.
