## Summary

- **Five outreaches on 10-09 (03:56–16:41 UTC) went out without the "Where Juniper is right now" block from #2551,** with grounding `situation: false`. Nothing recorded why, and Hub's logs from that time are gone after a restart, so the cause is **UNVERIFIED**.
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

The old code would have raised on an unconnected bus (`OrionBusAsync.redis` raises `RuntimeError`, which `getattr(..., None)` does not catch). That exception was swallowed upstream into a log line that has since rotated away, so it's one plausible explanation for 10-09.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `_read_situation` returns `(data, reason)`; `OutreachContext.situation_status`; the grounding lane `situation_status`; the warning log.
- `services/orion-hub/tests/test_endogenous_outreach.py`: one test per reason, a grounding test, and the exact-dict asserts updated.

## Schema / bus / API changes

- Behavior changed: the outreach grounding record gains `situation_status`. It is JSON in `result_json` and `client_meta`, not a model.

## Env/config changes

None.

## Tests run

```text
services/orion-hub: pytest tests/*outreach*.py   320 passed
```

## Review findings fixed

- Self-review: appended test stubs named `_FakeBus` shadowed an existing helper of the same name and broke `test_notification_payload_validates_against_the_real_schema`.
  - Fix: renamed them to `_SituationBusStub` / `_SituationRedisStub`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

## Risks / concerns

- **Low: diagnostic only.**
  - Concern: if the 10-09 cause was a disconnected bus, outreach still won't get the block. This PR makes that visible; it doesn't fix it.
  - Mitigation: the fix follows from whichever reason the next outreach records.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
