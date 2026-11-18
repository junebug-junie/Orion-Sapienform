## Summary

Orion's own chosen work now asks the body economy "may I?" at the two places it really starts, and the answer is written down next to what really happened. Nothing obeys the answer yet (SHADOW).

- **Durable runs bid at admission.** When orion-durable-runs accepts a curiosity investigate, self-inquiry, self-sense eval or journal compose run, it builds a want and asks the market in-process. Admission is unchanged: the bid runs in the background. The want reserves nothing while the run waits in the pool's queue. Its reservation starts at the first GPU grant and ends when the run ends (`services/orion-durable-runs/app/run_bids.py`).
- **cortex-orch bids before dispatch.** A tier 1 or 2 verb that is not on Juniper's turn bids with its `physiology` block (0.5 s timeout). The dispatch is byte-identical whatever the answer. The decision id is logged on the dispatch trace, and the reservation is released when the dispatch returns (`services/orion-cortex-orch/app/physiology_bid.py`).
- **Reservations stop biasing the market.** A shadow-funded want whose act never opens now holds capacity at most until the end of its own window. Its reservation is released when its act closes. A producer that reports its act's open reserves from the open, not from the decision (`orion/physiology/choice.py`).
- **The exploration floor is a weekly quota.** A Tier 2 family under 5% of Tier 2's funded wants over the rolling 7 days is funded when only the price said wait. The count comes from `want_ledger`. The C1 once-per-window limit is gone. R7 now passes.
- **A disagreement report.** `scripts/analysis/body_economy_shadow_vs_real.py` puts the shadow decision next to what really ran, per family, in the four-cell table Patch D needs.

## Outcome moved

- Before: `want_ledger` held 2 hand-run probe rows. No producer bid. After deploy: every want-family durable run and every self-initiated tier 1/2 verb dispatch writes a `want_ledger` row that joins to its `grammar_acts` row.
- R7: **FAIL → PASS** on the replay over real act history (10-16..10-27).
- First disagreement table (below): the agent lane reads at least 90% full in 88% of physiology ticks, so the market would defer most curiosity-family runs on `lane:agent` capacity. Most of those ran anyway after a median 1.3–1.6 h wait in the pool's own queue.

## Current architecture

- Patch C1 built the market (`POST /physiology/bid` on orion-durable-runs, `want_ledger`, the client). No producer called it.
- Patch C2 gave every cortex verb a `physiology` block. Nothing read it at dispatch.
- Reservations lasted one act-length (≤1 h) from the decision whether or not an act ever opened. In shadow, many funded wants never run, so they held capacity no real act used.
- The exploration floor funded a starved family at most once per lane window, counted from act history. R7 failed: 22 under-5% wants were deferred on price.
- `Market.context` rebuilt the context by hand and dropped `tier1_recent_use`.

## Architecture touched

- **orion-durable-runs:** new `app/run_bids.py`. `AdmissionRuntime.submit` / `lease` (grant) / `_terminal` call it. `app/physiology_market.py` seeds the floor from `want_ledger`, adds `mark_opened`, and keeps every context field.
- **orion-cortex-orch:** new `app/physiology_bid.py`. `main.handle` bids before `call_verb_runtime` and releases after it returns.
- **Shared:** `orion/physiology/choice.py` (reservation lifetime, weekly floor), `client.py` (`release`/`arelease`), `weights.py` (dead `funded_shares` removed).
- **Analysis:** the replay's R7 grading, the new disagreement report, and its eval wrapper.
- No schema, bus channel, route or sql-writer change. The release route already existed (C1).

## Files changed

- `orion/physiology/choice.py`: `Reservation.ttl_sec/opened/pending`, `decide(..., reserve_at_open=)`, `SYNTHETIC_PRODUCERS`, `ReservationBook.mark_opened/seed_funded/note_funded/funded_share`, `FundedWant`, the window-capped expiry and the weekly Tier 2 floor. `MarketContext.funded_share` removed.
- `orion/physiology/client.py`: `release` / `arelease` (POST `.../bid/{want_id}/release`, None on any failure).
- `orion/physiology/weights.py`: `funded_shares` removed (the floor no longer reads act history).
- `orion/physiology/tests/test_choice.py`, `test_weights.py`: the weekly quota, the rolling week and dedup, Tier 2 only, reservation lifetime, client release.
- `services/orion-durable-runs/app/run_bids.py` (new), `admission_runtime.py`, `main.py`, `physiology_market.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-durable-runs/tests/test_run_bids.py` (new; pure + real-Postgres lanes), `test_physiology_bid.py` (floor seeding, `mark_opened`, the `want_ledger` SQL on Postgres).
- `services/orion-cortex-orch/app/physiology_bid.py` (new), `main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `README.md`.
- `services/orion-cortex-orch/tests/test_physiology_shadow_bid.py` (new): turn paths never bid, end to end through `handle`.
- `scripts/analysis/body_economy_shadow_vs_real.py` (new), `eval_body_economy_shadow_vs_real.py` (new), `tests/test_body_economy_shadow_vs_real.py` (new).
- `scripts/analysis/replay_body_economy.py`, `tests/test_body_economy_replay_and_fit.py`: R7 on the weekly quota, graded literally.
- `config/physiology/contract.yaml`: registers `physiology_bid.py` and the report (display).
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: "Patch C3a results".
- `orion/schema_skew_discovery.py`: declares `VerbPhysiologyV1` as YAML that cortex-orch loads itself. cortex-orch now reads it, which the static gate "schema skew discovery" flagged.

## Schema / bus / API changes

- Added: none (no schema, channel or route). `orion.physiology.client.release/arelease` call the existing C1 route `POST /physiology/bid/{want_id}/release`.
- Removed: `MarketContext.funded_share`, `orion.physiology.weights.funded_shares` (in-repo only; every caller updated).
- Behavior changed: reservations expire at the end of the want's window before its act opens. The floor counts `want_ledger` funded wants, Tier 2 only.
- Compatibility notes: none on the bus. Deploy orion-durable-runs before orion-cortex-orch so the bid route is up before dispatch bids start. Until then the client returns None and dispatch is unaffected either way.

## Env/config changes

- Added keys: `ORION_PHYSIOLOGY_SHADOW_BIDS_ENABLED=true` (orion-durable-runs, orion-cortex-orch). In orion-cortex-orch, `ORION_PHYSIOLOGY_BID_URL` and `ORION_PHYSIOLOGY_BID_TIMEOUT_SEC` were already read by the client, and are now declared in `.env_example` and compose.
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: yes, both services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (the primary checkout's `services/orion-{durable-runs,cortex-orch}/.env` gained the keys).
- skipped keys requiring operator action: none. The sql-writer is not touched, so no hand edit.

## Body economy impact

- Adds a veto? No. Both bid sites only log; admission and dispatch are unchanged (tests compare the dispatched payload with and without a bid).
- Bypasses `bid()`? No: these ARE the first two producer bid sites.
- Changes what a price means? No. Changes what a reservation holds (window-capped before open) and how the floor counts (weekly, from `want_ledger`).
- Touches a spinal line? No. The new readers are registered as `display` (they gate nothing).

## Tests run

```text
orion/physiology/tests + scripts/analysis/tests + tests/scripts/test_schema_skew_discovery.py: 691 passed
services/orion-durable-runs/tests (incl. Postgres lanes on a throwaway postgres:16): 656 passed, 1 skipped
services/orion-cortex-orch/tests/test_physiology_shadow_bid.py: 24 passed
services/orion-cortex-orch/tests (full): 31 failed / 250 passed -- the same 31 fail on main (main fails 32);
  none touch this patch (test_workflow_lane compactor passes, test_mind_skip_logging MindRunResultV1 literal, ...)
make check-body-economy-contract: OK (79 state-reading modules, all registered)
make check-verb-physiology: OK (106 verbs)
orion-static-gates.yml, all 40 run steps locally: 39 PASS, 1 FAIL (schema skew: VerbPhysiologyV1) -> fixed, re-run PASS
check_service_env_compose_parity (durable-runs, cortex-orch): OK; check_env_template_parity: PASS
```

## Evals run

```text
scripts/analysis/eval_body_economy_shadow_vs_real.py (fixture lane): PASS
scripts/analysis/eval_body_economy_shadow_vs_real.py --live (7 days): invariants PASS; A3 0/0 (no live bid site yet)
replay_body_economy.py --wants-only (10-16..10-27, 727 wants): R7 PASS (was FAIL in C1); R2 FAIL (structural, unchanged),
  R5 not evaluable, R6 PASS.
  R7 weekly quota, stress run (every lane price 1.0), Tier 2: 57 floor funds, 0 hungry price deferrals under 5%.
  Honest caveat: the quota is enforced when a family bids. Between bids its trailing share can dip:
  curiosity_investigate and thought.dream_hop end the window at 4%, hub.retro_leisure at 1% (3 bids, demand-limited).
  The quota clause confirms the code (the grader recomputes the share the floor reads); the synthetic zero-use
  clause is the part that could fail (thought.painting ends 0.40 vs prior 1.0).

First disagreement table -- body_economy_shadow_vs_real.py --source replay, real durable admissions 10-20..10-27,
re-decided against the physiology_state in force at each admission:

| family | bids | funded | deferred | funded & ran | funded & not run (old stricter) | deferred & ran (old looser) | deferred & not run | no market | before act producer | median wait to open | top binding | Hub veto reasons (journald) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| durable.curiosity_investigate | 28 | 0 | 28 | 0 | 0 | 24 | 0 | 0 | 4 | 4,609 s | lane:agent 27 | daily_cap 92, cooldown 28, queue_backlog 10, arousal_strained 3 |
| durable.self_inquiry | 30 | 2 | 28 | 1 | 0 | 23 | 1 | 0 | 5 | 5,736 s | lane:agent 26 | cooldown 125, daily_cap 12, arousal_engaged 8 |
| durable.self_sense_eval | 49 | 4 | 45 | 4 | 0 | 21 | 1 | 18 | 5 | 5,276 s | lane:agent 27 | cooldown 129, daily_cap 12 |
| durable.journal_compose | 9 | 3 | 6 | 2 | 0 | 2 | 0 | 4 | 1 | 11 s | physiology_unavailable 4 | none logged |

Totals: 116 bids, 79 in the table, agreement 11%. 68 of the 70 "deferred & ran" are lane:agent capacity
deferrals of runs that waited a median 1.3-1.6 h in the pool's queue and then ran. That cell is the pool's queue,
not a veto disagreement. Live, lane:agent read >= 90% full in 88% of physiology ticks (3,198 ticks, 5 days).
22 "no market": physiology_state older than 360 s at the admission (gaps in the tick).
The live want_ledger holds 2 rows, both hand-run probes (excluded), until this deploys.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile (throwaway tag): OK; in the image app.run_bids, app.physiology_market,
  app.admission_runtime import.
docker build -f services/orion-cortex-orch/Dockerfile (throwaway tag): OK; app.main + app.physiology_bid import;
  load_verb_physiology('dream_cycle') -> tier=1 family='dream.cycle' lanes=['metacog'].
cortex-orch container -> http://orion-athena-durable-runs:8121/health: 200 (the bid URL is reachable on app-net).
Both throwaway images removed. Not deployed.
```

## Review findings fixed

The review was independent and read-only (orion-repo-agent). It confirmed findings 1-3 with probes.

1. **Finding:** Juniper's turn could still bid. cortex-exec's capability bridge re-enters cortex-orch inside a gateway turn (`source=cortex-exec`, `policy_dispatch_only`, `metadata.capability_bridge`, lane `mode_chat`), and `bid_skip_reason` returned None.
   - **Fix:** skip `nested_exec_call` for source cortex-exec, `policy_dispatch_only` or `capability_bridge`.
   - **Evidence:** the parametrized `test_bid_skip_reason` cases and the end-to-end `test_turn_and_interactive_dispatches_never_bid_end_to_end[cortex-exec-...]`.
2. **Finding:** cortex-exec's nested `self_study.reflect` call made a second bid for an act that had already bid.
   - **Fix:** the same skip.
   - **Evidence:** `test_bid_skip_reason[cortex-exec-self_study.reflect-...]`.
3. **Finding:** durable runs wait 1.3-1.6 h for a grant. That is longer than one act-length, so the reservation covered the wait and lapsed before the act ran.
   - **Fix:** `decide(..., reserve_at_open=True)` reserves PENDING. A pending reservation counts against no line until `mark_opened`, and lapses at the window's end. run_bids uses it, and so does the report's replay.
   - **Evidence:** `test_reserve_at_open_holds_nothing_until_the_act_opens`.
4. **Finding:** a client-side timeout left the market's reservation behind (release needed a decision).
   - **Fix:** `shadow_bid` returns `ShadowBid(want_id, decision)`. A bid that got no answer is released by want id.
   - **Evidence:** `test_self_initiated_tier1_verb_bids_and_dispatch_is_unchanged[None]`.
5. **Finding:** a grant or terminal that beat the background bid left a leaked reservation.
   - **Fix:** run_bids records opens and terminals for in-flight bids and applies them when the bid lands. A timed-out bid is released.
   - **Evidence:** `test_a_run_that_opens_or_ends_before_its_bid_lands_does_not_leak_a_reservation`.
6. **Finding:** hand-run probe rows counted toward the floor. The live ledger's one funded Tier 2 row is a deploy probe.
   - **Fix:** `SYNTHETIC_PRODUCERS` in choice.py, excluded in `FUNDED_SQL`, in `note_funded` and in the report's seed. The replay's own wants are renamed `replay.act_history`.
   - **Evidence:** `test_probe_bids_never_count_toward_the_floor`, plus a probe row in the Postgres `want_ledger` SQL test.
7. **Finding:** the R7 note claimed an independent check, but the quota clause holds by construction.
   - **Fix:** the note now says it confirms the code.
8. **Finding:** a missing verb YAML was logged as `physiology_invalid`, and the YAML was re-read on every dispatch.
   - **Fix:** `FileNotFoundError` now maps to `no_physiology`, and the lookup is `lru_cache`d.
   - **Evidence:** the assertion in `test_a_verb_with_no_or_tier0_physiology_never_bids`.

## Restart required

Deploy from the clean primary `main` checkout after merge, durable-runs first:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
```

## Risks / concerns

- **Severity: medium.** The verb families in the YAML and `families.yaml` do not match. Many tier 1/2 verb families (`self_study`, `concept_induction`, `execution_dispatch.*`, `thought.reverie`, ...) are not in `config/physiology/families.yaml`, so their dispatch bids come back `unknown_family`.
  - Mitigation: the report counts these as "no market", never as disagreements. Adding the families is a follow-up.
- **Severity: medium.** For the curiosity families, agreement is low by construction. The agent lane is saturated, so the market defers on capacity while the pool queues.
  - Mitigation: the report splits "deferred & ran" by reason and shows the wait. Patch D should read capacity deferrals as "queue", not "veto".
- **Severity: low.** The cortex-orch bid adds up to ~0.5–1 s to background dispatch latency, and never to turn paths. The live market answers in tens of ms (reviewer timing).
- **Severity: low.** Live A3 cannot be checked until both services deploy. The shadow-vs-real table above is a replay, not the live ledger: UNVERIFIED live.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2928

🤖 Generated with [Claude Code](https://claude.com/claude-code)
