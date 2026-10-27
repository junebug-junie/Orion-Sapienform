## Summary

- Removes the "lab" situation stub. It never had a provider; every unified-chat turn carried the line "Lab: unavailable/stub; do not infer."
- Adds "Outside temperature now: NF (feels like NF)." to the situation block, from the Open-Meteo reading Orion already fetched.
- Fixes the forecast windows. Open-Meteo hourly data starts at local midnight, so "next 6h" read 00:00-05:00 of today. Windows now start at the current hour.

## Outcome moved

Orion's situation block now says how warm it is outside, and the rain/icy-road/hot-car flags describe the next hours instead of last night. One dead placeholder line per turn is gone.

## Current architecture

`orion/situational/context.py` builds `SituationBriefV1` and renders the prompt lines. `LabContextV1` was always `available=False`; the Hub adapter even forced it off, but the renderer still printed the placeholder.

## Architecture touched

- `orion/schemas/situation.py`, `orion/schemas/registry.py`: `LabContextV1`, `SituationBriefV1.lab`, `lab_thermal_risk`/`lab_power_risk` affordance kinds removed.
- `orion/situational/context.py`: lab builder/settings removed; temperature line; `_current_hour_index` window alignment.
- `services/orion-cortex-exec`: stance-brief lab section and lab-derived `operational_context` fallback removed; lab settings + env keys removed.
- `services/orion-hub`: diagnostics `"lab": "stub"` and the debug "Lab context" panel removed.
- `orion/cognition/prompts/chat_stance_brief.j2`: dropped the "lab-risk affordance" mention.

## Schema / bus / API changes

- Removed: `LabContextV1`, `SituationBriefV1.lab`, two affordance kinds.
- Compatibility: the brief travels as a dict read with `.get()`; nothing re-validates stored briefs (checked in review).

## Env/config changes

- Removed keys: `ORION_SITUATION_LAB_CONTEXT_ENABLED`, `ORION_SITUATION_LAB_PROVIDER` (orion-cortex-exec).
- Local `services/orion-cortex-exec/.env`: both keys removed by hand (the sync script only adds keys).

## Tests run

```text
orion/situational/tests: 261 passed
cortex-exec test_chat_stance_brief + test_situation_provider + test_chat_relational_stance: 79 passed (from repo root)
hub test_situation_settings_env: 7 passed
check_env_template_parity: PASS
New: forecast window starts at current hour (fails on old code: high=70 vs 30), temp line rendered, no Lab line, no lab key in brief.
```

## Evals run

None; no situation eval harness exists. Live check: Open-Meteo at 21:00 local returned hourly[0]=00:00, confirming the window bug.

## Docker/build/smoke checks

Not run; pure Python/JS change.

## Review findings fixed

- Finding: stale docstring/README wording about `LabContextV1`.
  - Fix: reworded as removed.
- No correctness findings.

## Restart required

Rebuild orion-hub and orion-cortex-exec after merge (post-pull rebuild covers both).

## Risks / concerns

- Severity: low. Concern: `_current_hour_index` falls back to 0 if the current hour is past the array end. Mitigation: cannot happen with `forecast_days=2`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2908
