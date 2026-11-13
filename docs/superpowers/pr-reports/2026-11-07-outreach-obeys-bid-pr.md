## Summary

- Hub outreach now obeys its body-economy bid (Patch D for `hub.outreach`). Before this, only the background tick placed a SHADOW bid, and nothing obeyed it. Now every Hub door that composes a message asks the market just before it composes, and does what the market says.
- The rule is the one curiosity investigate (#3010) and the dream cycle obey (`orion.physiology.binding.bind_verdict`). Funded: compose. A full GPU lane, but the message is still worth its price: compose (the GPU pool's queue is the queue). Price, no hunger yet (`sated`), or net <= 0: hold, keep the item pending, bid again at the market's retry time (clamped to 2 min .. 1 h), and compose anyway at the end of the 1 h window. No answer (market down, stale body state, 5 s timeout, any error): compose, exactly as today.
- Forced sends, curiosity Door-A and the strain report never bid (reasons below). The daily cap (4) and the cooldowns are unchanged.
- The `arousal_strained` outreach veto STAYS in this PR. Deleting it is the next PR, after this reader has run `on` for 7 days with its checks green.
- Fixed on the way: the body's `attention:juniper` line counted office-visit messages as cap-counted, while the Hub's own cap does not. Office messages have been cap-exempt in Hub since 11-05 (`office_reach_out` was missing from durable-runs `CAP_EXEMPT_SOURCES`). A parity test now holds the two lists together.
- Flag `ORION_PHYSIOLOGY_BID_HUB_OUTREACH=on` (shipped on). Set `shadow` and restart Hub for an instant rollback.

## Outcome moved

The gate that the strained veto stands in for now exists as a real, obeyed bid on every outreach door that composes. The next PR can delete the veto on evidence instead of on faith.

### Replay: what the new rule would have done (14 days to 2026-11-07 05:32 UTC)

Every real compose attempt of a binding door (`endogenous_outreach_decisions`), re-bid with the live market code against the `physiology_state` row in force, obeyed, re-bid on holds (`scripts/analysis/outreach_binding_replay.py`):

| door | compose attempts | really sent | let through at the first bid | held | funded | full lane, worth it |
|---|---|---|---|---|---|---|
| background tick | 102 | 54 | 102 | 0 | 22 | 80 |
| reading findings | 103 | 97 | 103 | 0 | 38 | 65 |
| self findings | 2 | 2 | 2 | 0 | 1 | 1 |
| office visits | 3 | 3 | 3 | 0 | 2 | 1 |
| **all** | **210** | **156** | **210** | **0** | **63** | **147** |

Last 7 days alone: 93 attempts, 0 held (tick 35: 7 funded, 28 lane-full-but-worth-it).

So, today, binding changes nothing: no message would have been held or delayed. Every deferral was `capacity_reserved` on `lane:agent` with a positive net, which the rule treats as queueing. The lowest net seen was +0.037. This matches the live shadow data the task cited (7 days: 11 funded, 34 deferred, every deferral `capacity_reserved` on `lane:agent`). The live shadow count is higher than the replay's 35 tick attempts because the shadow bids also include the strain-held bids (#2988), which are not compose attempts.

What changes it: a lane price or Prime's slow charge rising far enough to beat the outreach value, or a send right after another one closes (no hunger: `sated`). Then a held message waits and records why.

### Why the attention price never deferred an outreach bid

It is not unpriced, and it is not a bug in the outreach bid's request. Every hub.outreach decision charges it (`cost_frac["attention:juniper"] = 0.25`, i.e. 1 message over the cap of 4). Two things keep it from ever deciding:

1. **The price cannot get big enough.** The `attention:juniper` price has a ceiling of 1.0 (7 days of `physiology_state`: max 1.000). One message costs 1/4 of the cap, so the most this line can ever charge one outreach bid is 1.0 x 0.25 = 0.25. The outreach want is worth weight ~0.68 x rank ~0.55 = ~0.38 (the lowest real net in 14 days was +0.037, and that was after lane charges). A charge of at most 0.25 can never beat a value of ~0.38 on its own. This is a design limit of the unit (ceiling x 1/cap), not a sampling accident.
2. **When the line binds, nobody bids.** Of 2,782 `binding` ticks in 7 days, 2,372 (85%) had the cap already full (level >= 4). At those moments the Hub's own `daily_cap` gate refuses the tick before the compose point, so no bid is placed. The capacity line on `attention:juniper` only bites at level + 1 > 4, which is exactly the moment the Hub gate already holds. The ~51% "bound" figure is real, but it describes the hours after the cap is spent.

What would make attention decide: a sized (not provisional) ceiling, or a price per message that grows with the day's pace. That is a market change, not an outreach change, so it is not in this PR.

Related, fixed here: the body counted office-visit messages on this line (missing from `CAP_EXEMPT_SOURCES`), so the line read slightly higher than the Hub's own cap since 11-05 (3 office messages in 14 days).

## Current architecture

- `EndogenousOutreach.maybe_outreach` placed a C3b SHADOW bid (`scripts/physiology_shadow.py`) at the tick's compose point, and once per window on a tick held by strain alone (#2988). It was logged next to the real outcome and never obeyed.
- Reading findings, self findings and office visits composed through `EndogenousOutreach.compose` and delivered through `offer_message`. They placed no outreach bid at all.
- Curiosity Door-A delivered a message composed inside the curiosity run. The strain report door spoke only when `arousal_strained` was the one gate holding.

## Door inventory

| door | code | composes in Hub? | bid before | bids now | why |
|---|---|---|---|---|---|
| background tick | `endogenous_outreach.py` `_outreach_once` | yes | shadow at compose; shadow once per window when strain holds | **binding**, cap-counted (1 message on `attention:juniper`) | the want the strained veto stands in for |
| reading findings | `reading_finding_outreach.py` `_tell` | yes (`compose`) | none | **binding**, cap-exempt (`juniper_msgs=0`) | composes a message; Juniper made it cap-free (10-13) |
| self findings (#3061) | `self_finding_outreach.py` `_tell` | yes (`compose`) | none | **binding**, cap-exempt | composes a message; cap-free (11-03) |
| office visits | `office_reach_out.py` `_tell` | yes (`compose`, chat lane) | none (it posts a separate `perception.office` ATTENTION want, unchanged) | **binding**, cap-exempt | composes a message; cap-free (11-05) |
| curiosity Door-A | `curiosity_investigation.py` `_maybe_reach_out_inner` -> `offer_message` | composed under the curiosity run's own GPU hold | none | **none** | the compose was priced and obeyed by the run's own binding bid (#3010); the send is cap-exempt; a blocked Door-A is final (no retry path), so a hold would drop the want |
| strain report | `endogenous_outreach.py` `_strain_report_once` | yes | the strain-held shadow bid (as the tick's real outcome) | **none** (strain-held shadow bid unchanged) | it exists to speak when strain holds everything else; a market that is pricing strain would hold exactly the report about strain. Its "only when `arousal_strained` is the one gate" logic is untouched |
| forced debug trigger | `POST /api/debug/endogenous-outreach/trigger` | yes | shadow | **none** (`forced`, logged) | the operator's call: never held |
| collapse-mirror reply, Claude room relay | `collapse_mirror_chat_reply.py`, `room_claude_relay.py` | replies to Juniper's own input / relays | none | none | Juniper's turn (Tier 0), not outreach |

## Architecture touched

- New `services/orion-hub/scripts/outreach_bind.py`: `OutreachBinder` (one open want per door key, hold / re-bid / window end, duplicate guard, release) and `bind_mode`.
- `EndogenousOutreach`: binds at the tick's compose point (the want id is the compose correlation id, so the decision joins the act); a tick inside a hold stops before any content read, without a bid. New `economy_admit` / `economy_done` for the doors. The verdict lands in `result_json.economy` on every decision row after the compose point. `status()` shows the binder.
- Doors: one `economy_admit` call after their own gates and before `compose`; held -> reason `economy_held`, the item stays pending and the door retries on its next tick.
- `orion/physiology/binding.py`: `orion-hub.outreach.binding` added to `BINDING_PRODUCERS` (shadow-vs-real keeps obeyed bids out of the agreement table; the market applies Prime's charge only to binding producers, and `hub.outreach` has no receptors, so nothing changes there).
- durable-runs `physiology_store.CAP_EXEMPT_SOURCES`: + `office_reach_out`.

## Files changed

- `services/orion-hub/scripts/outreach_bind.py`: the binding bid (new).
- `services/orion-hub/scripts/endogenous_outreach.py`: tick binding, door helpers, decision-row economy, status.
- `services/orion-hub/scripts/reading_finding_outreach.py`, `self_finding_outreach.py`, `office_reach_out.py`: bind before composing.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: the flag, shipped `on`.
- `orion/physiology/binding.py`: binding producer registered.
- `services/orion-durable-runs/app/physiology_store.py`: `office_reach_out` cap-exempt.
- `config/physiology/contract.yaml`: `outreach_bind.py` protected path + bidder row; outreach row note; replay registered as display.
- `orion/physiology/report_text.py`, `services/orion-hub/static/body-economy.html`: "Hub outreach obeys its bid" in Orion's body read and the Body Economy page.
- `services/orion-hub/README.md`: the door table and the rule.
- `scripts/analysis/outreach_binding_replay.py`: read-only replay (new).
- `services/orion-hub/evals/run_outreach_binding_replay.py`: the eval (fixture lane in CI, `--live` lane on the live DB).
- Tests: `services/orion-hub/tests/test_outreach_bind.py` (new), door tests in `test_reading_finding_outreach.py`, `test_office_reach_out.py`, `test_self_finding_outreach.py`; `test_physiology_shadow_bid.py` pinned to `shadow` (it tests the C3b seam); `services/orion-durable-runs/tests/test_physiology_store_cap_exempt_parity.py`; `scripts/analysis/tests/test_outreach_binding_replay.py`.
- `.github/workflows/orion-durable-runs-tests.yml`: runs the eval and the replay tests.

## Schema / bus / API changes

- Added: decision-row key `result_json.economy` (verdict, want_id, door, holds, bids, held_sec, and the market's decision_id, outcome, reason, binding_constraint, net, charge, retry_at, why). New decision reasons `economy_held` (and `economy_duplicate` on the tick if two checks race). `GET /api/debug/endogenous-outreach` gains `status.economy`.
- Added: want_ledger producer `orion-hub.outreach.binding` (bids still publish with `shadow=true`, as #3010's and the dream's do: `WantDecisionV1.shadow` is `Literal[True]`; the producer is the record of obedience).
- Removed / renamed: none.
- Behavior changed: binding doors may hold a message up to 1 h when the market defers on price / sated / net <= 0.
- Compatibility notes: `count_sent_on` counts only `reason='sent'`, so `economy_held` rows change no cap count. No bus channel or schema model changes.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_BID_HUB_OUTREACH=on` (orion-hub).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-hub: +ORION_PHYSIOLOGY_BID_HUB_OUTREACH='on'` in the primary checkout's `services/orion-hub/.env`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub: pytest tests  (full folder)
  4672 passed, 146 skipped, 4 failed. The 4 failures fail identically on main (617d2e9e5) in the
  primary checkout: test_attention_loops_ui_smoke (2), test_check_async_routes_not_blocking::
  test_the_real_repo_is_clean, test_fcc_model_labels_api::test_fcc_model_labels_disabled_by_default.
  Not touched by this patch.
services/orion-hub: tests/test_outreach_bind.py -> 30 passed (mutation check: forcing every verdict to
  admit failed 16 of the first 25)
after review fixes: test_outreach_bind + the three door files + test_physiology_shadow_bid +
  test_endogenous_outreach -> 390 passed
services/orion-durable-runs: pytest tests -k "physiology or report or cap_exempt or bid" -> 117 passed, 11 skipped
orion/physiology/tests scripts/analysis/tests tests/test_check_body_economy_contract.py -> 823 passed
scripts/check_body_economy_contract.py -> OK (104 state-reading modules, all registered)
scripts/check_env_template_parity.py -> PASS; check_env_key_single_source -> OK
scripts/check_definition_drift.py --gate -> PASS; check_metric_lineage --gate -> PASS;
check_inner_state_registry -> OK; tests/test_no_personal_terms.py etc. -> 18 passed
```

## Evals run

```text
PYTHONPATH=. python services/orion-hub/evals/run_outreach_binding_replay.py --live --days 14
fixture: PASS (0 failure(s))
live: 210 attempts; let through at the first bid 210; held 0 (median None s, max None s)
PASS   (gates: nothing held past the 1 h window, every attempt decided, bids per attempt bounded)
```

## Docker/build/smoke checks

```text
No image build: Python-only change; the Hub image already copies scripts/ and the orion package.
Not deployed (task rule). Live behaviour UNVERIFIED until Hub restarts.
```

## Review findings fixed

A code-review subagent read commit 3b5106c6a. It confirmed as correct: the exempt doors' `juniper_msgs=0` really zeroes the attention charge in `decide`; the `_TICK_ECONOMY` ContextVar cannot leak into a door's or a later tick's rows; the strained veto still runs first; no door drops a held item; the new reasons do not change any consumer's counts (`count_sent_on`, reading/office SQL); one bid per relief window; the replay arithmetic. Its findings, all fixed in the second commit:

- Finding (must-fix): a held want that nobody re-checked (the tick stopped at `no_tension_trigger`, or a door sat in quiet hours) got a free `window_end` pass hours later: no bid, and a decision row claiming a 5 h hold.
  - Fix: a hold not re-checked within 15 min after its retry time LAPSES (`LAPSE_AFTER_SEC`); the next check opens a new want with a fresh bid and window.
  - Evidence: `test_a_lapsed_hold_bids_afresh_instead_of_a_free_window_end`.
- Finding (should-fix): a forced send during a hold borrowed the held want's id as its correlation id (two acts on one id).
  - Fix: a `forced` verdict owns no want; the compose gets a fresh id; the held want stays held.
  - Evidence: `test_tick_forced_never_borrows_a_held_wants_id`.
- Finding (should-fix): every tick inside a hold recorded `economy_held`, even when Orion had nothing to say, overwriting `no_tension_trigger` / `content_already_used`.
  - Fix: the early hold check is gone; the tick checks the hold at its compose point (no bid inside a hold).
  - Evidence: `test_tick_inside_a_hold_with_nothing_to_say_records_no_trigger_not_economy_held`.
- Finding (should-fix): a timed-out bid may still reserve in the market, and nothing released it.
  - Fix: release whenever the bid gave no answer (as `run_bids.py` does).
  - Evidence: `test_a_timed_out_bid_is_released`.
- Finding (should-fix): a funded reservation was kept after a successful send (the dream and run_bids release at the act's end).
  - Fix: `close` releases a funded want sent or not.
  - Evidence: `test_close_releases_a_funded_reservation_sent_or_not`.
- Finding (should-fix): the CAP_EXEMPT parity test did not run when Hub's list changed.
  - Fix: `endogenous_outreach_decisions.py` (and the new replay / binder) added to both path lists of `orion-durable-runs-tests.yml`.
- Finding (nit): the tick's `finally` closed whatever tick want was admitted, even one it did not own.
  - Fix: `close(..., want_id=)` closes only the want this attempt was admitted on.
- Finding (nit): the tick's bind call was not wrapped; one lock serializes all doors' bids.
  - Fix: wrapped (a failure composes as `no_answer:bind_failed`); the shared lock is documented (a bid waits at most ~6 s behind another door's).
- Finding (nit): the replay's reservation book omits other families' reservations.
  - Fix: stated under the replay's approximations (it cannot hide a hold: a capacity deferral worth its price is admitted).

## Restart required

Hub (code + new env key). durable-runs too, for the `CAP_EXEMPT_SOURCES` fix (one line; without it the body keeps counting office messages on `attention:juniper`). From the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

### Live check after deploy

```sql
-- The first obeyed outreach verdicts (want_ledger, binding producer)
SELECT decided_at, want_id, outcome, reason, binding_constraint, round(net::numeric,3) AS net, decision->>'why' AS why
FROM want_ledger WHERE family = 'hub.outreach' AND producer = 'orion-hub.outreach.binding'
ORDER BY decided_at LIMIT 10;

-- The same verdicts as the outreach decision rows saw them (door, verdict, hold)
SELECT decided_at, reason, coalesce(result_json->>'source', 'endogenous_tick') AS door,
       result_json->'economy'->>'verdict' AS verdict, result_json->'economy'->>'holds' AS holds,
       result_json->'economy'->>'held_sec' AS held_sec, result_json->'economy'->>'why' AS why
FROM endogenous_outreach_decisions WHERE result_json ? 'economy' ORDER BY decided_at DESC LIMIT 20;
```

Log: `docker logs orion-athena-hub 2>&1 | grep 'physiology_bind family=hub.outreach'`. Flag: `docker exec orion-athena-hub env | grep ORION_PHYSIOLOGY_BID_HUB_OUTREACH` should print `on`.

## Risks / concerns

- Severity: medium. Concern: **while the `arousal_strained` veto stays, the bid never decides during strain.** A strained tick is refused before its compose point, so the binding bid never sees it; the only record of what the market would say then is the strain-held shadow bid (#2988, once per window). The same caveat #3159's reviewer raised for curiosity. Mitigation: the veto's deletion is the follow-up PR, after 7 days of this reader `on` with checks green; that PR must look at the strain-held shadow decisions to see what the market would have done under strain.
- Severity: low. Concern: today the rule holds nothing (replay: 0 of 210). Its value is the channel, not today's effect. The attention price structurally cannot defer an outreach bid (ceiling 1.0 x 1/4), so the daily cap stays the Hub's own gate. Mitigation: reported above; a sized attention ceiling is a market change for a later patch.
- Severity: low. Concern: held wants live in Hub memory. A Hub restart opens a new want and a new 1 h window. Mitigation: the doors' own items persist (pending bundles, Redis, the office claim), so a restart only re-asks; it never drops.
- Severity: low. Concern: the office door holds a "what are you up to" question for up to 1 h if the market says so, which could land after Juniper leaves the room. Mitigation: zero such holds in the replay; the reach-out still needs a funded or lane-queued answer only on a price / sated deferral.
- Severity: low. Concern: replay approximations (attempt time = the shadow bid time when known, else the decision row's time; real act history after a hypothetical hold). Mitigation: stated in the script docstring; the result (0 holds) does not hinge on minutes.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3165

🤖 Generated with [Claude Code](https://claude.com/claude-code)
