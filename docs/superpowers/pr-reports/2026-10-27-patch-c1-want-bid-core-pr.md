## Summary

Orion can now say "I would like to do this", and a market answers "fund it now" or "wait, because of X". The market only writes its answers down (SHADOW). Nothing obeys them yet.

- A want has a hunger that grows while it waits and resets when the act closes. It is ranked against its own family's last 30 days. It is weighted by a learned family weight. It pays the posted prices for its share of each budget it would use (`orion/physiology/{appetite,weights,choice}.py`).
- `POST /physiology/bid` on orion-durable-runs takes a `WantV1` and returns a `WantDecisionV1`. The decision has `shadow=true` and is published on `orion:physiology:want:decision`, which sql-writer stores in `want_ledger`.
- `orion/physiology/client.py` `bid(want)` uses a 0.5 s timeout. It returns `None` on any failure, and the caller keeps its own gate. It never fails open to "funded".
- `config/physiology/families.yaml` lists one family per act-ledger actor. Tier 0 families (Juniper's turns, perception) never bid. `actions.journal`, `dream.cycle` and `durable.journal_compose` are Tier 1. Everything else is Tier 2.
- Two strain fixes:
  - (a) Juniper's outreach cap is filled by Orion's own sends. That is self-spending, not the world pushing back, so it leaves the strain dial.
  - (b) A swap seat that ran its work while loaded and was unloaded afterwards no longer reads as "out of my hands".
- The replay runs R2, R5, R6 and R7 on synthetic wants built from real act history. A read-only script samples decisions against the live prices.

## Outcome moved

- Every want can now get an inspectable answer. The answer names the prices it saw, the net, the binding constraint, what it displaced and when to retry. Each answer is a `want_ledger` row, which is the input to the 7-day shadow-vs-real table in Patch C.
- Live 10-27, before this patch, strain named `attention:juniper`. That was Orion's own 4 of 4 sends, so after this patch it is listed in `strain.self_spent`. Strain also listed `lane:diffusion` and `lane:agent-gpu2` as `exogenous_bind` from finished work on unloaded swap seats. After this patch those prices decay to 0.

## Current architecture

Before this patch:
- Patch A gave Orion the act ledger (`grammar_acts`, with `cost_json`) and `act_downstream_use`.
- Patch B gave Orion the Body readings and shadow prices on `PhysiologyStateV1`, every ~120 s, in Redis `orion:physiology:latest`, with `GET /physiology/state`.
- Nothing turned "Orion wants X" into a priced decision.
- The strain dial counted every live price, including the Juniper outreach cap. `lane_ask` clamped any 0-slot seat with use in the window to `exogenous_bind`.

## Architecture touched

- **orion-durable-runs:**
  - new `app/physiology_market.py`: history reads, the weight cache and the reservation book;
  - routes `POST /physiology/bid`, `POST /physiology/bid/{want_id}/release` and `GET /physiology/reservations`;
  - `/health` → `physiology_market`.
- **orion-sql-writer:** the `WantLedgerSQL` model, a default route, a forced subscription and a migration.
- **Shared:**
  - `orion/schemas/physiology_want.py`, plus `self_spent` on `PhysiologyStrainV1`;
  - both schema registries and `inner_state_registry`;
  - `channels.yaml`;
  - the body-economy contract;
  - the metric lock.

## Files changed

- `orion/schemas/physiology_want.py`: `WantV1`, `WantDecisionV1`, the channel and the kind constants.
- `orion/schemas/physiology.py`: `PhysiologyStrainV1.self_spent` (consumer-first).
- `orion/physiology/families.py`, `config/physiology/families.yaml`: the families config and its loader.
- `orion/physiology/appetite.py`: intensity, the lane-hunger seed from B1's `deferrable_queued`/`deferrable_waiting_sec`, value_rank, graded, and the cost estimate (want → fixed → p90 → unpriced).
- `orion/physiology/weights.py`: shrinkage (κ=30). Sources are (a) downstream use and (b) Juniper replies. (c) is excluded. There is a 0.1/day drift cap, replayed day by day, and funded shares.
- `orion/physiology/choice.py`: the bid rule, placement, reservations, the Tier 1 reserved share, the exploration floor, retry_after and `build_context`.
- `orion/physiology/client.py`: `bid` and `abid`.
- `orion/physiology/body.py`, `budget.py`: strain fixes (a) and (b).
- `services/orion-durable-runs/{app/main.py,app/physiology_market.py,app/settings.py,.env_example,docker-compose.yml,Dockerfile,README.md}`.
- `services/orion-sql-writer/{app/models/want_ledger.py,app/models/__init__.py,app/worker.py,app/settings.py,.env_example,README.md}`, `services/orion-sql-db/manual_migration_want_ledger_v1.sql`.
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `orion/inner_state_registry.py`, `config/metrics/metric_definitions.lock.json`.
- `config/physiology/contract.yaml`: appetite, weights, choice, physiology_market and the sample script.
- `scripts/analysis/replay_body_economy.py`, `_body_economy_data.py`: the wants replay (`--wants`, `--wants-only`).
- `scripts/analysis/sample_body_economy_bids.py`: read-only live sample.
- Tests:
  - `orion/physiology/tests/{test_appetite,test_weights,test_choice}.py`;
  - updates to `test_body.py` and `test_budget.py`;
  - `services/orion-durable-runs/tests/test_physiology_bid.py`;
  - `services/orion-sql-writer/tests/test_want_ledger_sql_shape.py`;
  - `scripts/analysis/tests/test_body_economy_replay_and_fit.py`.
- `.github/workflows/{orion-sql-writer-tests,body-economy-analysis,orion-durable-runs-tests}.yml`: the new test and path triggers.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: "Patch C1 results".

## Schema / bus / API changes

- **Added:**
  - `WantV1` and `WantDecisionV1`, in `_REGISTRY` and `SCHEMA_REGISTRY` (both checked with `resolve()`);
  - channel `orion:physiology:want:decision` (durable-runs → sql-writer);
  - table `want_ledger`;
  - `POST /physiology/bid`, `POST /physiology/bid/{want_id}/release`, `GET /physiology/reservations`;
  - inner-state signal `physiology.want.decision.v1` (REHEARSAL);
  - `PhysiologyStrainV1.self_spent`.
- **Removed / renamed:** none.
- **Behavior changed:**
  - The strain dial skips `attention:juniper`.
  - A lane with 0 live slots whose window use already ran (`served_while_loaded`) is slack, not `exogenous_bind`.
  - A lease still open on a 0-slot seat (`held_without_live_slot`) is `no_reading`.
- **Compatibility:** `PhysiologyStrainV1` is `extra="forbid"`. **Deploy sql-writer before durable-runs.** If durable-runs deploys first, the old sql-writer rejects `physiology_state` rows until it is rebuilt.

## Env/config changes

- **Added keys** (durable-runs): `ORION_PHYSIOLOGY_BID_ENABLED=true`, `CHANNEL_PHYSIOLOGY_WANT_DECISION`, `ORION_PHYSIOLOGY_FAMILIES_PATH`, `ORION_PHYSIOLOGY_BID_HISTORY_TTL_SEC=300.0`, `ORION_PHYSIOLOGY_BID_STALE_AFTER_SEC=360.0`. The flag ships ON.
- **Changed values** (sql-writer): `SQL_WRITER_SUBSCRIBE_CHANNELS` gains `orion:physiology:want:decision`. `SQL_WRITER_ROUTE_MAP_JSON` gains `"physiology.want.decision.v1":"WantLedgerSQL"`.
- **`.env_example` updated:** yes, for both services, plus `settings.py` and compose.
- **Local `.env` synced:**
  - I ran `python scripts/sync_local_env_from_example.py orion-durable-runs orion-sql-writer`, which added the 5 durable-runs keys to the primary checkout's `.env`.
  - The sql-writer JSON members cannot be synced by that script. I merged them into the primary `.env` with the idempotent one-liner in "Restart required". A backup is at `/tmp/claude-1000/-mnt-scripts-Orion-Sapienform/5937d05a-adcc-4212-8c54-076555cffe99/scratchpad/sqlwriter.env.bak`.
  - Checked against this branch's templates: 0 missing members, 0 missing keys, for both services.
- **Skipped keys:** none. `DURABLE_RUNS_GRAPH_HOST` is reported as diverged; it is a pre-existing local override and I left it unchanged.

## Metric gate (WantDecisionV1's numbers)

- **Provenance:**
  - intensity comes from `grammar_acts.closed_at`;
  - value_rank comes from `grammar_acts.outcome_value` / `outcome_source`;
  - weight (a) comes from `act_downstream_use` (linked rows with the window closed);
  - weight (b) comes from resolver closes of `juniper_replied` acts;
  - prices come from `PhysiologyStateV1`.
- **Independence:** none of these is new signal. Each is a transform of the act ledger plus the posted prices.
  - Source (b) for `hub.outreach` uses the same resolver grade as its value_rank. value_rank is a within-family percentile, so the grade's level enters only through the weight.
  - (c), the personality graph, is excluded.
- **Theory anchors:**
  - intensity: the R2 drive template;
  - weights: value of information (Howard) for (a), social grounding for (b);
  - net: Lagrangian pricing, from Organ 2.
- **Live sanity:**
  - Weights are mostly priors today. Only outreach has evidence (Juniper replied to 63% of 43).
  - Hunger does return to rest: right after a close, intensity is 0 and the decision reads `sated` (35 such decisions in the replay).
- **Reversibility:** set `ORION_PHYSIOLOGY_BID_ENABLED=false`. No reader obeys the decisions.

## Replay (synthetic wants from real act history, 10-16..10-27 06:00Z, 601 wants)

`scripts/analysis/replay_body_economy.py --wants-only --acts-end 2026-10-27T06:00:00Z`. The live Choice code makes every decision. Lanes before 10-23 are replayed at 0 use and price 0, because no `physiology_state` exists before then.

- **R2 FAIL (structural).**
  - In the 10-18..22 heat, 52 of 52 non-GPU wants were funded (100%). Outside the heat, 145 wants were funded at 90%; the gap is `sated` wants and one floor fund. So the rate is non-zero but not reduced.
  - Heat is unpriced per act: `wh_by_host` is null on every act. So nothing in C1 can reduce a non-GPU want in heat. The design needs Organ 6 per-act heat cost first.
  - Juniper's turns: 100% by construction, because Tier 0 never bids.
  - GPU wants in the heat: only 3. The real system shed them, so few are in the history.
- **R5 NOT EVALUABLE.**
  - No lane price exists for the 10-21..22 baseline, and nothing bids before it leases (Patch E).
  - Baseline numbers: 728 background requests ended `unavailable` (712 shed) and 1,895 were granted.
  - Since 10-23, only 17 ticks carry lane prices. 12 of them had a price above 0, with a maximum of 0.17.
- **R6 PASS (a design property).**
  - 20 simultaneous bids at price 0 on an empty lane reserved at most 1.00x the lane, against a limit of 1.2x.
  - A mixed burst (every family x20, lanes half used, diffusion unloaded) also stayed at 1.00x at most.
  - It cannot fail by construction, because reservations are added only after the capacity check. It confirms that the code enforces the rule.
- **R7 FAIL (floor clause).**
  - Drift was 0.100/day at most. The cap is applied by construction.
  - A synthetic zero-use family (painting's 46 real rows, all set unused) ends at 0.43, against a prior of 1.0. That clause passes.
  - Under a stressed lane price of 1.0, the floor funded 47 wants, but 22 wants of under-5% families were deferred on price. The floor now funds a family at most once per lane window (review fix 6). Without that limit the live floor would fund every bid while act history lags. One fund an hour does not hold 5% for every family.
  - Literal funded shares in the stress run: dream.cycle 2.5%, journal_compose 1.2%, retro_coplay 1.8%, retro_leisure 0% (1 want). This needs a design call. Either the floor is a weekly quota tracked in the ledger (C2), or R7's clause is graded against families that bid every week.
  - The only live zero-use family (`hub.retro_leisure`) has 1 evidenced act, so it stays at its prior under the "fewer than 10" rule.

## Live sample (read-only, against live prices, tick 2026-10-27T06:33:45Z, 5 s old)

`scripts/analysis/sample_body_economy_bids.py`. It runs one want per family, in sequence, against a throwaway book. Strain was 0 with every live constraint slack; energy and lane:diffusion were `exogenous_bind` (the old rule, on the deployed code).

| family | tier | outcome | reason | lane | net | binding |
|---|---|---|---|---|---|---|
| hub.chat_reply / vision.council / perception | 0 | deferred | family_never_bids | - | - | - |
| actions.journal | 1 | funded | net_positive | metacog | 0.955 | - |
| dream.cycle | 1 | funded | exploration_floor | metacog | 0.000 | - |
| durable.journal_compose | 1 | funded | exploration_floor | fast | 0.000 | - |
| durable.curiosity_investigate | 2 | deferred | capacity_reserved | agent | 0.962 | lane:agent |
| durable.self_inquiry | 2 | deferred | capacity_reserved | agent | 0.869 | lane:agent |
| durable.self_sense_eval | 2 | deferred | capacity_reserved | agent | 0.929 | lane:agent |
| hub.outreach | 2 | deferred | capacity_reserved | - | 0.370 | lane:agent |
| hub.retro_coplay | 2 | funded | net_positive | - | 0.929 | - |
| hub.retro_leisure | 2 | funded | net_positive | - | 1.022 | - |
| hub.world_pulse_read | 2 | funded | net_positive | - | 0.610 | - |
| thought.dream_hop | 2 | funded | net_positive | diffusion | 0.385 | - |
| thought.painting | 2 | deferred | capacity_reserved | diffusion | 0.316 | lane:diffusion |
| topic_foundry.run | 2 | funded | net_positive | fast | 0.393 | - |

What the sample shows:
- lane:agent is about 86% held by curiosity holds, so new agent wants wait.
- The live seats on agent-gpu2 and hecate are unloaded, so the agent wants do not spill onto them.
- painting, dream_hop and topic_foundry have no measured slot time. They are priced at one full window, which is the upper bound.
- Every decision lists heat, energy and claude as unpriced.

## Tests run

```text
orion/physiology/tests + scripts/analysis/tests/test_body_economy_replay_and_fit.py: 165 passed
services/orion-durable-runs/tests: 535 passed, 108 skipped (Postgres-DSN tests run in CI)
services/orion-sql-writer CI unit list (incl. test_want_ledger_sql_shape.py): 513 passed / 205 passed
tests/test_agent_trace_schema_registry.py + test_route_map_completeness.py: 6 passed
orion-static-gates.yml, all 38 run steps locally: 38 PASS (definition drift after re-lock)
make check-body-economy-contract: OK (77 state-reading modules, all registered)
scripts/check_env_template_parity.py orion-sql-writer orion-durable-runs: PASS
```

## Evals run

```text
replay_body_economy.py --wants-only (R2/R5/R6/R7): verdicts above, written to
  /tmp/claude-1000/-mnt-scripts-Orion-Sapienform/5937d05a-adcc-4212-8c54-076555cffe99/scratchpad/replay/wants.{md,json}
sample_body_economy_bids.py: table above
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile (throwaway tag): OK; in the image, families.yaml loads (16 families) and app.physiology_market imports
docker build -f services/orion-sql-writer/Dockerfile (throwaway tag): OK; MODEL_MAP['WantLedgerSQL'] resolves
Both throwaway images were removed. Not deployed.
want_ledger is absent live (to_regclass NULL); the migration passes auto_apply_refusals with the table absent.
```

## Review findings fixed

The review was independent and read-only (orion-repo-agent). It confirmed findings 1-4 with probe scripts.

1. **Finding:** a GPU family with no slot history was funded free, with no charge and no capacity check. An empty act read funded everything.
   - **Fix:** price the want at one window (the upper bound) and place it, with the warning `lane_cost_unmeasured`. An empty `grammar_acts` read is now `physiology_unavailable`.
   - **Evidence:** `test_a_gpu_family_with_no_slot_history_is_priced_at_one_window_never_free`, and the empty-pool case in `test_never_funded_on_a_missing_market`.
2. **Finding:** a no-reading lane counted as empty.
   - **Fix:** such a lane is skipped in placement. If it is the only lane, or a fixed-placement lane, the want gets `physiology_unavailable`.
   - **Evidence:** `test_a_lane_with_no_reading_is_never_an_empty_lane`.
3. **Finding:** a want spilled onto an unloaded swap seat, evicting its card-mate unpriced.
   - **Fix:** an unloaded seat satisfies capacity only when it is the want's only lane. Otherwise the want is deferred with `only_room_on_unloaded_seat`.
   - **Evidence:** `test_full_live_seat_does_not_spill_onto_an_unloaded_swap_seat`.
4. **Finding:** the Tier 1 share was subtracted twice from Tier 2's line.
   - **Fix:** Tier 2's line is the budget minus the unused part of Tier 1's share (share - Tier 1 last-window use - Tier 1 reservations). The Tier 1 free pass also counts its measured use.
   - **Evidence:** `test_tier2_line_does_not_subtract_tier1_share_twice`.
5. **Finding:** the client could raise `http.client` errors.
   - **Fix:** it catches every exception.
   - **Evidence:** `test_client_returns_none_on_a_broken_http_response`.
6. **Finding:** the exploration floor had no limit live.
   - **Fix:** at most one floor fund per family per lane window, recorded in the book.
   - **Evidence:** `test_exploration_floor_funds_a_family_at_most_once_per_window`. The replay now runs the same rule, which turned R7's floor clause to FAIL.
7. **Finding:** retry_after used the wrong rate and promised times for non-falling prices.
   - **Fix:** a retry time is given only for a slack price. It uses the guaranteed slack decay, so it is an upper bound. Otherwise retry_after is None.
   - **Evidence:** test_price_deferral asserts both cases.
8. **Finding:** some replay checks pass by construction.
   - **Fix:** R6 and R7 now say so in their output. R6 gained a mixed multi-family burst with measured use and an unloaded seat. R2 is labelled structural.
9. **Finding:** deploy order.
   - **Fix:** sql-writer comes first in "Restart required".
10. **Finding (nit):** if sql-writer's `create_all` creates `want_ledger` before the migration runs, the migration is refused and the column is `json`, not `jsonb`.
    - **Fix:** apply the migration before the sql-writer rebuild. The post-pull applier does this; the manual command is below.
11. **Finding (nit):** performance.
    - **Fix:** weights are cached per history snapshot and UTC day; the rest of the context costs about 5 ms.
    - **Not fixed:** the history lock is still held across the DB read.
12. **Finding (nit):** the tests used a cwd-relative path.
    - **Fix:** the tests now load families.yaml with `Path(__file__)`.

Also fixed from my own replay and live sample:
- A hold longer than the window costs at most one window.
- Zero hunger with zero charge is now `sated`, not "price".
- A live seat wins over an unloaded one in placement.
- The exploration floor needs hunger above 0.

## Restart required

Deploy from the clean primary `main` after merge. Order matters: migration, then sql-writer, then durable-runs.

The sql-writer `.env` JSON edit. I already applied it to the primary checkout. It is idempotent, so it is safe to re-run:

```bash
cd /mnt/scripts/Orion-Sapienform && python3 -c 'import json,pathlib;p=pathlib.Path("services/orion-sql-writer/.env");E={"SQL_WRITER_SUBSCRIBE_CHANNELS":lambda v:v+[c for c in ["orion:physiology:want:decision"] if c not in v],"SQL_WRITER_ROUTE_MAP_JSON":lambda v:{**v,"physiology.want.decision.v1":"WantLedgerSQL"}};f=lambda l:(lambda k,v:k+"="+json.dumps(E[k](json.loads(v)),separators=(",",":")))(*l.split("=",1)) if l.split("=",1)[0] in E else l;p.write_text("\n".join(f(l) for l in p.read_text().split("\n")))'
```

The migration. The post-pull applier applies it automatically (additive, REQUIRED-BY orion-sql-writer). By hand:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_want_ledger_v1.sql
```

Then:

```bash
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
./scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The post-pull rebuild also lists orion-hub, because shared `orion/` code is baked into its image. No hub behaviour changes.

Live check after deploy:
- `curl -s localhost:8124/health | jq .physiology_market`;
- one `POST /physiology/bid`, then a `want_ledger` row;
- `GET /physiology/state` shows `strain.self_spent == ["attention:juniper"]`.

## Risks / concerns

- **Severity: medium.** R2 and R7 FAIL, and R5 is not evaluable.
  - R2 needs per-act heat cost (Organ 6).
  - R7's floor clause needs a design call: a weekly ledger quota, or a regraded clause.
- **Severity: medium.** Heat, energy and claude are unpriced on every want. Until per-act watt-hours exist, only lanes and Juniper's cap can defer a want.
- **Severity: low.** Reservations live in memory, so a durable-runs restart releases them. Shadow fundings reserve capacity that real acts never use, which biases later shadow decisions toward `capacity_reserved`.
- **Severity: low.** Weights are mostly priors: `act_downstream_use` is `no_link_path` for most families.
- **Severity: low.** No producer bids yet. C2 adds the cortex-orch dispatch bid; until then `want_ledger` rows come only from explicit calls.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2923

🤖 Generated with [Claude Code](https://claude.com/claude-code)
