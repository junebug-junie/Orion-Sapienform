## Summary

Orion's own chosen work now asks the body economy "may I?" at the two places it really starts, and the answer is written down next to what really happened. Nothing obeys the answer yet (SHADOW).

- **Durable runs bid at admission.** When orion-durable-runs accepts a curiosity investigate, self-inquiry, self-sense eval or journal compose run, it builds a want and asks the market in-process. Admission is unchanged: the bid runs in the background. The want's reservation is re-anchored when the run gets its first GPU grant and released when the run ends (`services/orion-durable-runs/app/run_bids.py`).
- **cortex-orch bids before dispatch.** A tier 1 or 2 verb that is not on Juniper's turn bids with its `physiology` block (0.5 s timeout). The dispatch is byte-identical whatever the answer. The decision id is logged on the dispatch trace, and the reservation is released when the dispatch returns (`services/orion-cortex-orch/app/physiology_bid.py`).
- **Reservations stop biasing the market.** A shadow-funded want whose act never opens now holds capacity only until the end of its own window, not for an hour. Its reservation is released when its act closes (`orion/physiology/choice.py`).
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

- `orion/physiology/choice.py`: `Reservation.ttl_sec/opened`, `ReservationBook.mark_opened/seed_funded/note_funded/funded_share`, `FundedWant`, the window-capped expiry and the weekly Tier 2 floor. `MarketContext.funded_share` removed.
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
TESTS_PLACEHOLDER
```

## Evals run

```text
EVALS_PLACEHOLDER
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

REVIEW_PLACEHOLDER

## Restart required

Deploy from the clean primary `main` checkout after merge, durable-runs first:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-orch up -d --build
```

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
