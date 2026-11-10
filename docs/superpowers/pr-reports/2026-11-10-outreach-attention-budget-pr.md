# One attention budget for every door

## Summary

Every Hub door that messages Juniper now spends from one daily budget, and the price of the next message grows with the day's pace. Stacked on the door-policy PR (https://github.com/junebug-junie/Orion-Sapienform/pull/3177).

- No exemptions: the background tick, curiosity, readings and office visits all count on the one daily cap and on the body's `attention:juniper` line. Not counted: the strain report and forced (operator) sends. Collapse-mirror replies and the Claude room relay answer Juniper's own input and never wrote outreach rows.
- The cap is 6 (was 4), because every door now counts. It has one owner: `JUNIPER_DAILY_CAP` in `orion/physiology/attention_pacing.py`, imported by the Hub gate and by the body. The env key `HUB_ENDOGENOUS_OUTREACH_DAILY_CAP` is retired, and the single-source gate refuses it if it comes back.
- The `attention:juniper` price is now a pacing threshold: the first message of the day is cheap, and each later one costs more when the day runs ahead of its budget. It can now beat a typical message's value (~0.38). The old price never could (most it could charge: 0.25).
- A message held on that price waits for the day's pace (the market now computes when it will be let through), at most until Juniper's waking day ends. Office visits wait at most 5 minutes, then go anyway.
- Rollback: `ORION_PHYSIOLOGY_ATTENTION_PACING=shadow` on durable-runs brings back the old price (the pacing price still rides in the reading's detail for comparison). Shipped `on`.

## Outcome moved

Replay of 14 whole days (10-27 .. 11-09, `evals/run_outreach_attention_budget_replay.py`, read-only), messages per day:

| | before PR A | after PR A | after PR B |
|---|---|---|---|
| reading findings | 7.86 | 7.86 | 1.86 |
| curiosity | 3.50 | 1.43 | 1.43 |
| background tick | 3.93 | 2.21 | 1.29 |
| office visits | 0.29 | 0.29 | 0.29 |
| self findings | 0.14 | 0 | 0 |
| strain report | 0.07 | 0.07 | 0.07 |
| total | 15.79 | 11.86 | 4.93 |

- Days: 5 on 12 days, 6 on 2 days (both had office visits), 2 on 10-29 (the strain-locked day). The cap was the limit on only 2 days; the price did the rest.
- Every office visit message still went (4 of 4), 2 of them at the end of their 5-minute hold.
- Spread (local hour of sends): 8:00 12, 9:00 6, 11:00 6, 12:00 9, 15:00 7, 19:00 12, plus a few curiosity sends at night (Door-A ignores quiet hours, as before). The 08:00 pile of readings (live 10-28: 14 messages in an hour) became one message at 08:00, then the rest spaced through the day.
- Honest caveat on "not a schedule": in the replay the sends bunch near the hours the allowance lets the next message through (08, 12, 15, 19), because the replay's want strength is drive alone. Live, doors offer at different moments, Prime's regulate charge moves the value, and Orion's PASS / `Why Juniper` decisions drop some messages, so the hours should vary more; this is UNVERIFIED until a week of live rows. Door-specific event strength (a reading's salience) is not wired into the outreach bid yet.
- Held then sent: 35 messages, median hold 3.2 h. Never sent: 73 reading findings and 11 ticks waited all day without becoming worth the price; 13 more hit the cap.

A strong message still goes late, a weak one waits (from the replay):

- 11-01 19:18, tick: wanted 0.646 against a price of 0.646 (4 sent that day): sent.
- 10-30 19:59, reading: wanted 0.629 against 0.598: sent.
- 10-27 08:18, reading, right after the day's first message (drive 0.24 against 0.39): held 70 min, sent 09:28 at 0.43 against 0.39.
- 10-27 08:24, reading: held 7.5 h, sent 15:51.

What the replay cannot reproduce: Orion's own PASS and `Why Juniper` decisions, the GPU lanes' small charges, Prime's regulate charge, and readings carried to the next day (a held reading finding the replay marks expired is in fact retried the next day, inside its 48 h lookback). It replays each message at the moment its door really wanted to speak.

How it was tuned: pacing to the full cap of 6 gave 5.71/day but hit the cap on 13 of 14 days and refused 2 of 4 office visits; pacing to 4 gave 4.07/day. Pacing to 5 (one slot of headroom under the hard cap of 6) gave 4.93/day, hit the cap twice, and refused no office visit. That is the shipped setting (`PACE_HEADROOM = 1`).

## Metric gate: the pacing price

1. **Provenance.** `orion/physiology/budget.py` `juniper_pacing()` -> `orion/physiology/attention_pacing.py` `message_price(pace(sent, waking_fraction(now), pace_budget(cap)))`. `sent` is the `attention:juniper` level: `services/orion-durable-runs/app/physiology_store.py` `JUNIPER_SENT_SQL`, today's `sent` rows in `endogenous_outreach_decisions` except `CAP_EXEMPT_SOURCES` and forced rows. The clock is the tick's `now`.
2. **Independence.** Not new signal: a monotone function of the same count the old price used, plus the clock. It replaces the old price on that line (in `on`), it is not added beside it. It shares its count with the Hub's daily cap gate on purpose: the price is the soft limit, the cap the hard one. It is not in the strain dial (`attention:juniper` is self-spent).
3. **Theory anchor.** The online knapsack threshold of Zhou, Chakrabarty and Lukose (2008): accept an item when its value per unit beats psi(z) = (L/e)(Ue/L)^z, z = fraction of the budget used; (ln(U/L)+1)-competitive, the best any online rule can do against adversarial arrivals. The budget accrues over the waking day (budget pacing, Balseiro and Gur 2019). L = 0.30 (lowest hub.outreach want value in 14 days, 0.306 over 1,797 bids), U = 0.65 (the family weight 0.648; value = weight x rank and rank <= 1).
4. **Live data.** Rest point: every local day starts at z = 0, price L/e = 0.110: a real, reachable calm, reset daily, not a floor artifact. It reaches U (0.65) when the day runs ahead (z >= 1). On the replay the price at send ranged 0.11 .. 0.65. The old price is still computed and logged in shadow.
5. **Existing mechanism.** The dual-ascent price on this line already existed and could not decide (#3165 report: ceiling 1.0 x 1/cap 4 = 0.25 < ~0.38). This changes that price, it does not add a second one.
6. **Reversibility.** Cheap: one flag (`shadow`), one module. No schema field (the pacing inputs ride in the reading's free-form `detail`). The cap is one constant.

## Current architecture

- Hub's `daily_cap` (env, 4) counted only the background tick (and the strain report). Readings, office visits and curiosity Door-A skipped it and did not count.
- The body counted the same: `CAP_EXEMPT_SOURCES` in durable-runs listed four cap-free doors, kept in step with Hub by a regex parity test.
- `attention:juniper`'s price was dual ascent on the day-end pace; max charge per message 0.25 (#3165).
- Cap-free doors bid with `juniper_msgs=0`. Every held want composed anyway after 1 h.

## Architecture touched

- `orion/physiology`: new `attention_pacing.py` (cap, exempt list, pacing threshold, relief time); `budget.py` posts the pacing price; `choice.py` computes a pacing deferral's re-bid time; `body.py` takes the cap from the module.
- durable-runs: flag, waking-day hours (from the Hub's quiet hours, single-source gated), the shared exempt list, forced rows excluded.
- Hub: the cap gate holds for every door; every door's send counts; the strain report and forced sends do not; doors bid cap-counted; hold windows (day end, office 5 min).

## Files changed

- `orion/physiology/attention_pacing.py`: new. The cap (one owner), the exempt list, the pacing threshold, the waking-day clock, the relief time.
- `orion/physiology/budget.py`: posts the pacing price on `attention:juniper` (flag), the pacing inputs in the reading's detail.
- `orion/physiology/choice.py`: a pacing deferral's `retry_after_sec` is the relief time.
- `orion/physiology/body.py`: the cap from the module.
- `services/orion-durable-runs/app/{settings,main,physiology_store}.py`, `.env_example`, `docker-compose.yml`: flag, waking-day hours, shared exempt list, forced rows excluded; the cap env key removed.
- `services/orion-hub/scripts/endogenous_outreach.py`: the cap gate for every door; every door's send counts; strain report and forced sends do not; day-bounded holds.
- `services/orion-hub/scripts/outreach_bind.py`: per-want hold window (day end, or a door window); `OFFICE_WINDOW_S`.
- `services/orion-hub/scripts/{reading_finding_outreach,office_reach_out}.py`: bid cap-counted; office with its 5-minute window.
- `services/orion-hub/scripts/endogenous_outreach_decisions.py`: `count_sent_on` from the shared exempt list, forced rows excluded.
- `services/orion-hub/scripts/main.py`, `app/settings.py`, `.env_example`: the cap from the module; the env key removed.
- `scripts/check_env_key_single_source.py`: the cap env key retired; the quiet hours single-sourced.
- `config/physiology/contract.yaml`: the Hub cap count registered as a reader.
- READMEs (Hub, durable-runs): the one budget.
- Tests: `orion/physiology/tests/test_attention_pacing.py` (new), `test_budget.py`, `test_choice.py`, `test_body.py`; durable-runs `test_physiology_store_cap_exempt_parity.py`; Hub `test_endogenous_outreach.py`, `test_office_reach_out.py`, `test_outreach_bind.py`, `test_reading_finding_outreach.py`, `test_strain_report_outreach.py`.
- Evals: `services/orion-hub/evals/run_outreach_attention_budget_replay.py` + `test_outreach_attention_budget_replay_eval.py`.

## Schema / bus / API changes

- Added: none. New keys inside `PhysiologyConstraintReadingV1.detail` for `attention:juniper` (`pacing_mode`, `pacing_price_per_msg`, `pacing_allowance`, `pacing_rate_per_hour`, `pacing_pace`, `pacing_waking_fraction`, `pacing_cap`, `pacing_pace_budget`, `pacing_span_h`). `detail` is a free dict, so no consumer-first order is needed.
- Behavior changed: under `on` the `attention:juniper` price's `value` = psi x cap, `ceiling` = 0.65 x cap (`sized`), `use`/`budget` = sent / allowance.
- Compatibility notes: none on the bus.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_ATTENTION_PACING=on`, `HUB_ENDOGENOUS_OUTREACH_QUIET_START_HOUR=23`, `HUB_ENDOGENOUS_OUTREACH_QUIET_END_HOUR=8` (durable-runs; the two hours are owned by `services/orion-hub/.env_example`).
- Removed keys: `HUB_ENDOGENOUS_OUTREACH_DAILY_CAP` (Hub and durable-runs).
- `.env_example` updated: both services; compose updated (durable-runs).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (3 keys added to durable-runs). The sync never prunes, so the retired key was removed from both local `.env` files by hand.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest -q orion/physiology                                     -> 265 passed
cd services/orion-durable-runs && pytest -q tests/                           -> 829 passed, 130 skipped
cd services/orion-hub && pytest -q tests/test_outreach_door_policy.py tests/test_endogenous_outreach.py \
  tests/test_curiosity_investigation.py tests/test_reading_finding_outreach.py tests/test_office_reach_out.py \
  tests/test_outreach_bind.py tests/test_strain_report_outreach.py tests/test_endogenous_outreach_decisions.py \
  tests/test_physiology_shadow_bid.py evals/                                  -> 623 passed, 3 skipped
Full Hub tests/: only the 4 failures that also fail on clean main (ui smoke x2, async-routes repo scan, fcc labels).
Repo root: tests/test_check_body_economy_contract.py tests/test_check_verb_physiology.py tests/test_env_key_single_source.py
  tests/test_check_settings_defaults.py tests/test_report_dead_env_keys.py tests/test_no_personal_terms.py
  tests/test_metric_prompt_semantics.py                                       -> all passed
scripts/check_env_key_single_source.py -> OK; check_settings_defaults.py orion-hub --example-drift -> OK;
check_body_economy_contract.py -> OK (after registering the Hub cap count); check_verb_physiology.py -> OK;
check_metric_lineage.py --gate / --prompt-semantics -> PASS; check_definition_drift.py --gate -> PASS;
check_inner_state_registry.py -> OK
```

## Evals run

```text
evals/run_outreach_attention_budget_replay.py --days 14   (read-only, live DB)
  before PR A 15.79/day -> after PR A 11.86/day -> after PR B 4.93/day; cap reached on 2 of 14 days;
  4 of 4 office visits sent; held then sent 35 (median 3.2 h)
evals/test_outreach_attention_budget_replay_eval.py (offline: morning pile spread, office window,
  strong-late vs weak-waits, Door-A needs only cap room, strain report exempt)      -> 4 passed
```

## Docker/build/smoke checks

```text
No Dockerfile or requirements change; no build run.
docker compose --env-file <primary>/.env --env-file <primary>/services/orion-durable-runs/.env -f services/orion-durable-runs/docker-compose.yml config
  -> renders ORION_PHYSIOLOGY_ATTENTION_PACING: "on", HUB_ENDOGENOUS_OUTREACH_QUIET_START_HOUR: "23", _END_HOUR: "8"; no DAILY_CAP
(scripts/safe_docker_build.sh config refuses in a linked worktree: it wants a .env there.)
```

## Review findings fixed

REVIEW_BELOW

## Restart required

No schema field was added to an `extra=forbid` model, so there is no consumer-first order. Merge the door-policy PR first. From the clean primary `main` checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

durable-runs first, so the body counts every door before Hub starts letting them spend on it.

## Risks / concerns

- Severity: medium. Concern: reading findings lose most of their sends (7.9 -> 1.9 a day in the replay). Mitigation: that is the budget Juniper set; a held reading finding stays pending for 48 h and can win a later slot.
- Severity: medium. Concern: curiosity Door-A does not bid, so it can use slots at night (it ignores quiet hours, as before) and make the morning's first message cost more. Mitigation: it is 1.4/day after the door policy; the cap still holds it.
- Severity: low. Concern: a held tick or reading now waits up to the end of the day instead of 1 h. Mitigation: the re-bid time is computed (no blind waiting); the door's own gates and the 15-minute lapse rule still apply.
- Severity: low. Concern: L and U are constants from 14 days of data; the family weight moves. Mitigation: documented in the module; the replay is re-runnable.

## PR link

PR_LINK
