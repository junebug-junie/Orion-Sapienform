## Summary

- **Root cause of the missing "Where Juniper is right now" block (#2551):** `apply_content_novelty` rebuilt the outreach context field by field and never copied `situation`. Every non-forced outreach runs that filter before the prompt is built, so 5 of 5 outreaches on 10-09 went out without the block (grounding `situation: false`), whatever Redis held. It now uses `dataclasses.replace`, so no field can be dropped this way again. Found in code review.
- **`_read_situation` now returns a reason with every result,** and never raises. The reasons are:
  - `ok`;
  - `no_bus`;
  - `bus_not_connected`;
  - `no_key`;
  - `error:<Type>: <message>`.
- **The reason is stored on every outreach decision** as the grounding field `situation_status` (decision record, plus `client_meta.outreach_provenance.lanes` on the delivered message). Anything other than `ok` is also logged as `endogenous_outreach_situation_unavailable`.

## Outcome moved

The next outreach either carries the block or states exactly why it didn't.

Live check against production Redis:
- a connected bus returns `ok` (revision 13);
- an unconnected `OrionBusAsync` returns `bus_not_connected`.

An unconnected bus would also have raised in the old code, but Hub connects its bus before outreach starts, so that wasn't the 10-09 cause.

Regression proof: re-introducing a field-by-field rebuild fails both new tests (`test_novelty_filter_keeps_every_other_field`, `test_a_normal_outreach_cycle_carries_the_situation_block`). With the fix, all 322 outreach tests pass.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `apply_content_novelty` uses `dataclasses.replace` (the fix); `_read_situation` returns `(data, reason)`; `OutreachContext.situation_status`; the grounding lane `situation_status`; the warning log.
- `services/orion-hub/tests/test_endogenous_outreach.py`: one test per reason, a grounding test, and the exact-dict asserts updated.

## Schema / bus / API changes

- Behavior changed: the outreach grounding record gains `situation_status`. It is JSON in `result_json` and `client_meta`, not a model.

## Env/config changes

None.

## Tests run

```text
services/orion-hub: pytest tests/*outreach*.py   322 passed
```

## Review findings fixed

- Finding (BLOCKER, the real cause): `apply_content_novelty` dropped `situation` and `situation_status`, so the block, and this PR's own diagnostic, never reached a non-forced outreach.
  - Fix: `dataclasses.replace`.
  - Evidence: the two regression tests above, which fail on the old rebuild.

- Self-review: appended test stubs named `_FakeBus` shadowed an existing helper of the same name and broke `test_notification_payload_validates_against_the_real_schema`.
  - Fix: renamed them to `_SituationBusStub` / `_SituationRedisStub`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- **Low: dependency on the situation graph.**
  - Concern: the block now depends on the situation graph's Redis key existing.
  - Mitigation: if the key is missing, the outreach records `no_key` instead of silently omitting the block.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
