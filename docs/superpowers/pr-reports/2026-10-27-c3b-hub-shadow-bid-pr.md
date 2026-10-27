# Body economy Patch C3b (hub): shadow bids for outreach and retro play

## Summary

- Hub now asks the body economy "would you fund this?" at three points, and ignores the answer. The answer is only logged, next to what Hub really did.
- Endogenous outreach bids at compose time, and on a tick that strain alone holds (the gate the economy is meant to replace).
- Retro leisure bids when a dispatch asks to start a session. Retro co-play bids when a handoff opens its play act.
- Each bid's `want_id` is the correlation id of the act the producer opens, so a report can join the shadow decision (`want_ledger`) to the real act (`grammar_acts`).
- A funded shadow decision whose real act was blocked releases its reservation, so a want that never ran does not hold shadow capacity.
- Switch `HUB_PHYSIOLOGY_SHADOW_BID_ENABLED`, shipped on.

## Outcome moved

Before Patch D can let the economy decide anything, someone has to see where the economy and the real gates disagree. Hub's producers now leave that evidence: one `physiology_shadow_bid ... shadow=<funded|deferred|none> real=<ran|blocked:reason>` line per want, plus the decision row in `want_ledger`.

## Current architecture

Patch C1 built the market (`POST /physiology/bid` on durable-runs) and the client (`orion.physiology.client`, 0.5 s timeout, `None` on any failure). No producer called it. Hub's outreach, leisure and co-play acts were gated only by their own rules (quiet hours, cooldown, strain, cabinet heat, ...).

## Architecture touched

- orion-hub only, plus one contract registration and the env sync list.
- No new bus channel. durable-runs writes the decision to `want_ledger` as before.

## Files changed

- `services/orion-hub/scripts/physiology_shadow.py`: new. `ShadowBid` places the bid as a background task, then logs it beside the real outcome once the bid answers. It releases a funded-but-blocked reservation. Every failure is swallowed.
- `services/orion-hub/scripts/endogenous_outreach.py`: `maybe_outreach` creates the shadow and settles it with the tick's result. `_outreach_once` places it at compose (the compose correlation id is the shadow's `want_id`) and on a strain-only hold.
- `services/orion-hub/scripts/retro_leisure_listener.py`: `start` bids on the dispatch id and settles with started or the refusal.
- `services/orion-hub/scripts/retro_memory_export.py`: the co-play start sink bids on the handoff row key and settles with whether the act opened.
- `services/orion-hub/app/settings.py`, `.env_example`: the switch, and `ORION_PHYSIOLOGY_BID_URL=http://127.0.0.1:8124/physiology/bid`. Hub runs on the host network, so the client's app-net default name does not resolve there.
- `scripts/sync_local_env_from_example.py`: `HUB_PHYSIOLOGY_SHADOW_BID_ENABLED` added to `SYNC_EXACT`. No `HUB_PHYSIOLOGY_` prefix is synced, so without this the env sync silently skipped the key.
- `config/physiology/contract.yaml`: the helper registered as `display` (it reads price state and gates nothing).
- `services/orion-hub/tests/conftest.py`: autouse fixture so unit tests never POST to the live market.
- `services/orion-hub/tests/test_physiology_shadow_bid.py`: new (23 tests).
- `services/orion-hub/README.md`: section, with the known gaps.

## Correlation join, per producer

| producer | family | want_id | act correlation id | verified by |
|---|---|---|---|---|
| outreach | `hub.outreach` | compose correlation id | `publish_outreach_act(correlation_id=...)` in `_deliver` | `test_outreach_sent_path_is_identical_for_every_answer_and_joins_the_act` asserts want_id == the published act's correlation id |
| retro leisure | `hub.retro_leisure` | `dispatch_id` | `publish_play_opened(correlation_id=dispatch_id)` | `test_leisure_start_logs_ran_on_the_dispatch_id`, refusal test |
| retro co-play | `hub.retro_coplay` | `coplay_key` (`<session>:<start>`) | `publish_coplay_opened(correlation_id=key)` | `test_coplay_open_is_unchanged_and_bids_on_the_row_key` |

Exception: when outreach's agent lane times out and the chat lane retries, the act opens on the retry's new id. The log line then carries `act_correlation_id=<retry id>`. That line is the only link.

## Real-gate field, per producer

- outreach: `real=ran` when sent, else `blocked:<the tick's reason>` (`orion_passed`, `empty_generation`, `named_ungrounded_signal`, `<gate>_after_generation`, ...). A strain-only hold logs `blocked:arousal_strained`, whatever the strain-report door does.
- retro leisure: `ran` when the session started, else `blocked:<refusal>` (`cabinet_hot`, `arousal_strained`, `human_recent`, `leisure_already_running`, ...).
- retro co-play: `ran` when the play act opened, `blocked:no_bus` / `blocked:act_open_failed` otherwise.

## Not bid here

- Reading pull. A read runs as a `reading.turn` durable run, and that bid is left to durable-runs admission (a separate patch), so a read is never bid twice. The Hub pull gate itself (arousal, thermal, daily cap) is not compared by this patch.
- `offer_message` (curiosity Door-A) and reading-finding outreach also open `hub.outreach` acts. They place no bid.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: none (shadow only). The outreach compose correlation id is still a fresh `uuid4`. It is now created by `ShadowBid`.
- Compatibility notes: none.

## Env/config changes

- Added keys: `HUB_PHYSIOLOGY_SHADOW_BID_ENABLED=true`, `ORION_PHYSIOLOGY_BID_URL=http://127.0.0.1:8124/physiology/bid` (orion-hub).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. Both keys are in the primary checkout's `services/orion-hub/.env`.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub: pytest tests/test_physiology_shadow_bid.py -> 23 passed
services/orion-hub: pytest test_*outreach*.py test_*strain_report*.py test_retro_leisure*.py test_retro_memory_export.py + tests/test_physiology_shadow_bid.py -> 471 passed (after the review fixes)
orion-static-gates.yml python steps (38 gates, incl. check_body_economy_contract, settings defaults, env single source) -> all OK
```

## Evals run

```text
None. The shadow-vs-real comparison is the eval, and it needs live decisions: scripts/analysis/body_economy_shadow_vs_real.py (another agent) joins on correlation_id after deploy.
```

## Docker/build/smoke checks

```text
curl -XPOST http://127.0.0.1:8124/physiology/bid (from the host network, which Hub uses) -> 200 with a WantDecisionV1 (shadow=true).
The live path from inside Hub is UNVERIFIED until deploy.
```

## Review findings fixed

- Finding: on a strain hold, when the strain-report door sent its own message, the log line named the report's act id as this want's act.
  - Fix: a producer-named outcome (`shadow.real`) drops `act_correlation_id`.
  - Evidence: `test_strain_report_door_never_claims_its_report_as_this_want`.
- Finding: co-play logged `real=ran` before it knew whether the act opened.
  - Fix: settle after `publish_coplay_opened`; `blocked:no_bus` / `blocked:act_open_failed` otherwise.
  - Evidence: `test_coplay_without_a_bus_is_logged_blocked`.
- Finding: tests did not prove "never blocks", did not cover leisure `ran`, and compared answers only with each other.
  - Fix: a slow-economy test, a leisure-started test, and a flag-off baseline in the invariance test.
  - Evidence: the 23 tests above.
- Finding: the outreach retry id, the unbid outreach doors, and leisure re-bids on the same want_id were undocumented.
  - Fix: README "Known gaps".

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run it from the clean primary `main` checkout after merge. The env sync has already written the keys.

## Risks / concerns

- Severity: low. Concern: each funded shadow bid reserves shadow capacity in durable-runs' market until the act closes or the reservation expires, and Hub's bids now count toward it. Mitigation: blocked wants release at once. No real gate reads the book.
- Severity: low. Concern: a strain-only hold bids every tick while strain lasts. Mitigation: it only fires once every other gate (cooldown, quiet hours, daily cap) has passed.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
