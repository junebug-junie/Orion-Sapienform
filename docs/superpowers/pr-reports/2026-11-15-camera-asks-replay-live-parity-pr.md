# Camera asks: the replay matches the live market, and vehicle asks get held (#3364)

Fixes #3364. Follows attention patch 4b (#3351) and the family-share prior (#3353). Design: #3213.

## Summary

The 4b replay said 95.5% of vehicle asks would be held. Live, the market funded all 16 bids, both vehicles included. The two disagreed because they ranked each ask against different history.

- **Root cause.** The market ranks a camera ask against the family's past event wants in `want_ledger`. That ledger still held 120 old rows from 11-04 to 11-13. They were posted before camera asks had acts or grades, so their intensity was mostly just salience, and three were near 0. A vehicle ask (expected answer 0.135) beat those three: rank 3/120 = 0.025. Camera asks carry no cost, so any rank above 0 funds. The replay never read the ledger. It ranked against its own bids, all graded and all at intensity 1.0, so a vehicle ranked 0.
- **Market fix (smallest rule change).** A want now ranks only against past events decided with the family's current grading (graded with graded, ungraded with ungraded). The ledger already records `graded` on every decision. No vision.ask exception, no cost for asks, no deadline, ranking still hunger x expected answer (#3285, #3286 unchanged).
- **Replay fix.** The replay now ranks against the market's own ledger, read through the market's 300 s history cache. Its own bids stand in only for the time before the ledger existed. Over the whole 30-day window it re-decides every live 4b bid: **16 of 16 match** exactly (value_rank to 6 places, and the reference size: 120 rows for the 07:00 batch) under the rule the market ran then.
- **Live effect after deploy.** The next vehicle ask ranks 0 and is held. Confirm, same and office asks are still funded. Face (expected answer 0.725) and sighting (0.804) are now below the family mean (0.817), so they are held too: see Risks and #3381.
- **No change for other families.** Every reading and cooling event-want decision of the last 30 days gets the same rank before and after (474 and 11 decisions, 0 changed).

## Outcome moved

- Replay vs live: 0 of 16 live bids reproduced before (the replay held both vehicles, the market funded them) -> 16 of 16 now.
- Vehicle asks: funded at rank 0.025 -> held at rank 0 against today's ledger. Without this patch they stay funded until the old rows age out on 2026-12-12 (about 40 more unanswerable cards).
- The 4b gate on live 30-day data, corrected replay (2026-11-15). Three views, vehicle asks:

  | View | Held at first bid | Never opened in 7 d | Gate |
  |---|---|---|---|
  | Rule the market ran, history as it really was | 52.2% | 23.9% | FAILS (limit 70%) |
  | Same-grading rule, history as it really was | 43.5% | 15.2% | fails: no graded event existed before 11-14, so those bids ranked cold |
  | **Same-grading rule, as if 4b had bid every ask (the gate)** | **87.0%** | **87.0%** | **passes** |

  - The old gate passed only because its history was wrong. With the real history, the rule the market ran fails.
  - The gate view is the counterfactual the gate is meant to answer: 4b's own graded bids replace the shadow era's rows. From today the live ledger is exactly that: 16 graded rows, all intensity 1.0.
  - Gate view, all rules: confirm 0% held (161 asks), same 0% (45), face 0% (3), office 0% (1), sighting 0% (4), vehicle 87.0% (46). Answered asks per day 7.033 -> 7.000 (loss 0.47%, limit 10%; 1 lost vehicle answer).
  - Next bid of each rule against today's real ledger (`next_bid_by_rule`, at salience 0 and 1): vehicle held at any salience (rank 0); confirm, same and office funded at any salience; face (0.725) and sighting (0.804) held.

## Current architecture

- Camera asks bid at the body-economy market before the card opens (`services/orion-sql-writer/app/ask_bind.py`). Held: no card, re-bid 600 s doubling to 3 h.
- The market (`orion/physiology/choice.py` `decide_priced`) ranks an event want with `appetite.rank_value`: the share of the family's past event wants (30 days of `want_ledger`, `physiology_market.EVENT_HISTORY_SQL`) whose intensity x family mean answer is strictly below this want's intensity x expected answer. A want with no charge is funded when its rank is above 0.
- Camera-ask hunger is always saturated (tau 60 s), so every current bid has intensity 1.0.
- The replay (`services/orion-durable-runs/evals/event_wants_replay.py --camera-asks`) rebuilt the history from its own first bids.

## Architecture touched

- `orion/physiology/appetite.py`: new `same_grading_events`; `build_family_appetite` keeps only same-grading `(intensity, graded)` pairs. Plain floats are kept as before (callers that do not know the grading). A pair with unknown grading is dropped.
- `services/orion-durable-runs/app/physiology_market.py`: `EVENT_HISTORY_SQL` returns each row's `graded`; `read_event_history` returns `(intensity, graded)` pairs.
- `orion/physiology/choice.py`: type hint and docstring only.
- The replay: ledger reference, 300 s snapshot lag, `live_parity`, `next_bid_by_rule`, `--export-parity-fixture`, new live checks.

## Files changed

- `orion/physiology/appetite.py`: same-grading reference.
- `orion/physiology/choice.py`: `build_context` accepts pairs (hint + docstring).
- `services/orion-durable-runs/app/physiology_market.py`: grading in the event-history read.
- `services/orion-durable-runs/evals/event_wants_replay.py`: replay reads the market's reference; live parity; next-bid probe; fixture export.
- `services/orion-durable-runs/evals/fixtures/camera_asks_live_parity_2026-11-15.json`: the live inputs, anonymised (ask ids hashed; no names, no individual ids, no evidence refs; answer detail reduced to the `dont_know`/`resolved_by` flags).
- `services/orion-durable-runs/evals/event_rank_grading_mutations.py`: mutation check, 10 mutants.
- `orion/physiology/tests/test_event_sources.py`: same-grading unit tests, live shape through `decide`, cold start after a grading flip.
- `services/orion-durable-runs/tests/test_camera_asks_replay_parity.py`: 16/16 parity (short and full window, reference size pinned), old replay did not match, next-bid probe, held-candidate lifecycle, snapshot lag, gate counterfactual vs as-ledger view.
- `services/orion-durable-runs/tests/test_event_wants_menu.py`: fixture rows carry grading; new market tests.
- `.github/workflows/orion-durable-runs-tests.yml`: mutation step (`orion/physiology/**` already triggers it).
- `docs/superpowers/specs/2026-11-12-attention-camera-asks-bind-design.md`: acceptance check 3 records the parity requirement.

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: an event want ranks only against past events decided with the family's current grading. Today this changes only `vision.ask` (its 120 ungraded rows leave its reference).
- Compatibility notes: `want_ledger` already stores `decision.graded` on every decision with an intensity (checked: 0 rows in 30 days without it). No migration.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change).
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. pytest -q orion/physiology/tests                                  350 passed
cd services/orion-durable-runs && PYTHONPATH=<repo> pytest tests -q            941 passed, 144 skipped
PYTHONPATH=. python services/orion-durable-runs/evals/event_rank_grading_mutations.py
  baseline 37 passed; 10/10 mutants KILLED (pairs of the other grading kept; unknown grading kept;
  appetite strips the grading; JSON list pairs dropped; market read drops the grading; market SQL
  drops the grading; replay ignores the ledger; replay has no snapshot lag; replay ignores
  same_grading; replay counts stand-ins after the ledger starts)
```

## Evals run

```text
event_wants_replay.py --camera-asks                       (fixture lane) rc=0
event_wants_replay.py                                     (reading fixture lane) rc=0
cooling_act_replay.py                                     rc=0
event_wants_replay.py --camera-asks --live --days 30      (read-only, live DB, 2026-11-15 00:26Z) rc=0
  checks: vehicle_held_at_least_70pct, vehicle_never_opened_in_horizon_at_least_70pct,
          confirm_and_same_held_at_most_10pct, answered_per_day_falls_at_most_10pct,
          replay_reproduces_live_bids (16/16), next_vehicle_bid_held,
          next_confirm_and_same_bids_funded: all true
Other-family impact (read-only scratch script, every event-want decision of 30 days, rank before vs after):
  hub.world_pulse_read 474 decisions, rank recomputed = logged for 471, 0 changed by the rule
  self_regulate.shed    11 decisions, rank recomputed = logged for 11,  0 changed by the rule
```

## Docker/build/smoke checks

```text
Not run: no dependency, Dockerfile or compose change. Runtime change is pure Python in orion/physiology and
orion-durable-runs, covered by the tests above. Live effect UNVERIFIED until durable-runs is rebuilt.
```

## Review findings fixed

Review ran in a subagent (by hand, read-only, against 224f3fc22).

- Finding (blocker): over the full 30-day window, stand-in bids stayed in the reference after the ledger started, so the 11-14 vehicle bids saw 239 rows, not the market's 120 (14 of 16 matched; the parity test only passed because it started 10 minutes before the live bids).
  - Fix: stand-ins count only while the market had no ledger; the gate's counterfactual is now an explicit `counterfactual=True` (stand-ins for asks with no 4b bid). `live_parity` takes `start` and the live lane runs it over the whole window.
  - Evidence: `test_the_full_window_replay_reproduces_them_too` (16/16, reference sizes pinned); mutant "replay counts stand-ins after the ledger starts" KILLED; live lane 16/16.
- Finding (should-fix): the "next vehicle bid held" check probed at salience 0, the rank's floor, which cannot prove "held".
  - Fix: probe at salience 0 and 1; "funded" is checked at the floor, "held" at the ceiling.
  - Evidence: `next_bid_by_rule.vehicle.held_at_any_salience = true` live; test updated.
- Finding (should-fix): the fixture export dropped a `resolved_by` key whose value is falsy, but `ask_grade` reads its presence.
  - Fix: export `resolved_by` on presence, `dont_know` on truth.
  - Evidence: `export_parity_fixture.detail`.
- Finding (should-fix): a `[intensity, graded]` list (a pair after JSON) was silently dropped.
  - Fix: lists are pairs too.
  - Evidence: new assertion in `test_same_grading_events_keeps_floats_and_only_matching_pairs`; mutant KILLED.
- Finding (nit): the replay built the appetite at the snapshot time; the market builds it at the bid time over the cached acts.
  - Fix: appetite at `t`, acts as of `t - 300 s`; the docstring calls 300 s an upper bound. Parity unchanged (16/16).
- Finding (nit): the parity test pinned ranks only. Fix: it now pins each bid's reference size too.
- Finding (nit): redundant CI path entry. Fix: removed.
- Finding (nit, not changed): the mutation harness rewrites files in place. Same pattern and same documented caveat as `ask_bind_mutations.py`; fine in CI.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The market lives in orion-durable-runs. sql-writer is unchanged (it reads no changed code at bid time).

## The 3 "no_answer:no_decision" bids

- 2026-11-14 21:13:37-38Z, three `same` asks. The sql-writer client waited its 0.5 s and gave up (`physiology_bid_unavailable ... TimeoutError`), so `ask_bind` opened each card without a stamp, as designed.
- The durable-runs process logged nothing from 21:13:32.6Z to 21:13:44.7Z: its event loop was blocked. It then decided the three queued bids (funded, `decided_at` 21:13:43.99Z) after the client was gone. That is why the ledger says funded.
- The shadow loop already treats a late-decided bid as the card's one want (`BID_AT_OPEN_SQL`), so there is no duplicate want.
- Not fixed here: the blocking call is unknown and not straightforward to find from logs. Filed as #3382.

## Risks / concerns

- Severity: medium. Concern: face (0.725) and sighting (0.804) asks are below the family mean and will now be held at every bid, so they get no new grade and stay held (#3350's starvation trap). Without this patch the same trap starts when the old rows age out on 2026-12-12. Mitigation: #3381 (exploration share per rule, #3350 option c) is the recommended next patch.
- Severity: low. Concern: a family whose grading flips ranks cold until it has 10 same-grading events (`EVENT_HISTORY_MIN`). vision.ask has 16 graded rows today, so it is not cold. Mitigation: tested (`test_a_grading_flip_ranks_cold_until_enough_same_grading_events`).
- Severity: low. Concern: right after deploy the market has 16 graded camera-ask events, above the cold-start minimum of 10. If the family had fewer, vehicles would rank cold and be funded (the as-ledger view above shows that for 11-04..11-13). Mitigation: checked live (`reference_n` 16).
- Severity: low. Concern: the parity model assumes the market's 300 s cache. The exact cache age at each bid is unknown; 300 s reproduces all 16. Mitigation: test pins it; a different lag fails one bid (`test_the_snapshot_lag_matters_for_parity`).
- Issues opened: #3381 (rule below the family mean held forever), #3382 (durable-runs event-loop stall, 3 bids timed out). Issues closed: #3364 (on merge). #3352 (verify-live) stays open until the deploy proves vehicle bids held.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
