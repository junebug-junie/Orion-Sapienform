## Summary

- The first live consult failed on 2026-11-11. It started at 19:51:48 and failed at 19:56:48.
- Cause 1: bus envelopes require a UUID correlation id, and the consult loop passed `consult-<id>-t1`. Orion's agent-lane turn failed validation at once (`pre_turn_appraisal raised ... Input should be a valid UUID`).
- Cause 2: the chat-lane fallback then ran Orion's real turn, which recalled, entered the harness and started digging. It hit outreach's 300 s timeout (`consult_orion_turn_empty debug={'error': 'timeout', 'lane': 'chat_fallback'}`).
- Fix 1: every correlation id the consult loop creates is now a UUID: uuid4 for each turn, uuid5 for the brief and the email.
- Fix 2: `EndogenousOutreach.compose()` takes an optional per-attempt `timeout_sec`. Consults pass the new `HUB_CONSULT_ORION_TIMEOUT_SEC=600`. Other callers are unchanged.

## Outcome moved

- Orion's consult turn now reaches the agent lane. It has 10 minutes per attempt to run digs.

## Files changed

- `services/orion-hub/scripts/consult_loop.py`: UUID ids (`consult_uuid`), `orion_timeout_sec`
- `services/orion-hub/scripts/endogenous_outreach.py`: `compose`/`_generate` `timeout_sec` on both attempts, and the log line now shows the real budget
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`, `scripts/main.py`, `README.md`: `HUB_CONSULT_ORION_TIMEOUT_SEC`
- `services/orion-hub/tests/test_consult_loop.py`: every bus correlation id is a UUID, and Orion gets its budget
- `services/orion-hub/tests/test_office_reach_out.py`: the override reaches both lanes, and the defaults are unchanged

## Env/config changes

- Added: `HUB_CONSULT_ORION_TIMEOUT_SEC=600`. Local `.env` synced.

## Tests run

```text
services/orion-hub/tests full suite: 1 failed, 4675 passed, 147 skipped. The 1 failure is the same one main has (test_fcc_model_labels_disabled_by_default).
check_settings_defaults.py orion-hub --example-drift: OK
```

## Evals run

None new. Evidence is the journald trace of consult `9f138a849520` (lines quoted above).

## Review findings fixed

- Review subagent: nothing material.
- Nit: the fallback log printed 210 s while a consult had 600 s.
  - Fix: it now logs the budget that actually ran.

## Restart required

The post-pull rebuild handles `orion-hub`.

## Risks / concerns

- Severity: low.
  - Concern: the worst case for one Orion consult turn is 600 + 600 s. It holds the consult lock, so no other consult advances meanwhile. Ticks skip, and Claude replies are spawned tasks.
  - Mitigation: one consult at a time is the design.
- Severity: low.
  - Concern: consult `9f138a849520` (the stuck belief) stays `failed`. A source starts one consult ever.
  - Mitigation: to retry it, set it back to pending. This is a one-row production write on the consult table, and Juniper's call: `UPDATE orion_consult SET status='pending', status_reason=NULL, created_at=now() WHERE consult_id='9f138a849520';`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
