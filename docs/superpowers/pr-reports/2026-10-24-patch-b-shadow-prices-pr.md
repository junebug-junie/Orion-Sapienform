## Summary

Body economy Patch B2: shadow prices (Organ 2 Budget) and the strain dial (Organ 5). This PR is stacked on B1 (#2911).

- **Every physiology tick now prices each constraint Orion lives under**: heat, energy, each GPU lane, Juniper's attention and Claude. Each price carries its value, its ceiling, a bind state and a plain-words `why`.
- **The strain dial** is the largest live price divided by its ceiling, plus the name of that constraint. Constraints that are out of Orion's hands (`exogenous_bind`) are listed separately and do not move the dial.
- **Shadow only.** Prices ride on `PhysiologyStateV1` (additive, optional fields): the same Redis key, the same bus channel and the same `physiology_state` row. Nothing that decides reads them. `GET /physiology/state` on durable-runs is the human readout.
- **The five replay-driven rule changes** are applied in `orion/physiology/prices.py`: a bounded ramp into bind, hysteresis on the bind flag, the point sensitivity, ceilings labelled provisional until they can be sized, and the stuck-price check.
- **Heat sensitivities are data:** `config/physiology/heat_sensitivity.yaml`, written by `fit_heat_sensitivity.py --write-config`. An entry older than 30 days, or a missing one, reads as `no_reading`, never 0.

## Outcome moved

- Orion's body now has a price history. Before this patch, the categorical `strained` flag was the only "how hard is the world pushing back" signal. It had no magnitude and no cause.
- Live smoke over the last 24 h (720 stored ticks), read-only:
  - **Heat price 0 on all 720 ticks.** The cabinet read 31.1 °C raw, which is 29.1 °C after #2780's 2 °C sensor offset. Orion's share was 0.85 °C, so the room without Orion was 28.2 °C, under the 31 °C line.
  - **Energy `exogenous_bind` on all 720 ticks.** Month to date is $166.69. The always-on floor alone reaches $192 against the $100 envelope.
  - **No price at its ceiling unflagged past its horizon.**
  - **Strain was at rest (0) on 151 of 720 ticks.** Otherwise it was driven by `attention:juniper` (316 ticks; 4 of 4 cap-counted messages were sent by 12:18 local), `lane:diffusion` (174), `lane:agent-gpu2` (49) and `lane:agent` (30).

## Current architecture

- After B1, the `physiology` node in the durable-runs temporal-self graph publishes Body readings only. `orion/physiology/prices.py` had pure price math from #2861/#2868, used only by the offline replay.

## Architecture touched

- `orion/physiology/budget.py` (new) is the pure Budget reducer. The node calls it through `deps.price_body`, and the carry between ticks lives in the graph's checkpointed state (`physiology_prices`). A price bug cannot cost the Body readings: it becomes the warning `prices_failed`.
- `orion/physiology/heat_config.py` (new) loads the heat sensitivities, which are baked into the image.
- `physiology_store.read_host_watts`: the 1 h mean of per-host compute watts from `orion_biometrics_cluster`.
- The schema gains `PhysiologyPriceV1` and `PhysiologyStrainV1`, attached as optional fields.

### What each price charges

| constraint | use vs budget | `exogenous_bind` |
|---|---|---|
| heat | Orion's own share in °C (short-run sensitivity × watts above baseline) vs the room's headroom (line − (cabinet − sensor offset − Orion's share)) | room over the line even with Orion's share at 0 (0.5 °C hysteresis) |
| energy | month-end cost forecast vs the $100 envelope | the always-on floor alone reaches the envelope (5% hysteresis) |
| lane:&lt;role&gt; | lease-seconds asked of the seat vs slots × window (waiting backlog is **not** an input) | the seat has no live slot but a lease holds it |
| attention:juniper | today's cap-counted sends, paced to the end of the local day, vs the cap | never |
| claude | `clear` counts as use 0 | `limited` |

## Files changed

- `orion/physiology/prices.py`: rule changes A–E, `PriceParams.per_tick`, `bind_with_hysteresis`, `ceiling_for`, `ceiling_run_since`/`is_stuck`, and `upper_bound_if_spans_zero` (superseded, kept for the replay).
- `orion/physiology/budget.py` (new): asks per constraint, `price_constraint`, `strain_dial`, `price_state`.
- `orion/physiology/heat_config.py` (new): parse, load, per-host age check, sensor offsets.
- `orion/physiology/body.py`: `HostWattsInput`, and the `priced_hours` refactor. `energy_reading` output is unchanged and its tests are untouched.
- `orion/schemas/physiology.py`: `PhysiologyPriceV1`, `PhysiologyStrainV1`, `price` and `strain` (additive, optional).
- `config/physiology/heat_sensitivity.yaml` (new, generated): the 10-27 refit.
- `config/physiology/contract.yaml`: registers `budget.py`, `heat_config.py` and the smoke script.
- `services/orion-durable-runs/app/{main,settings,temporal_self_graph,physiology_store}.py`: wiring, the watts read, `GET /physiology/state`, settings.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,Dockerfile,README.md}`: the two new keys, the COPY of the yaml, and docs.
- `scripts/analysis/fit_heat_sensitivity.py`: `--write-config` and baseline watts.
- `scripts/analysis/replay_body_economy.py`: the Patch B gate (`run_patch_b`), the R3 windup-vs-binding split, R6 marked deferred, and the $100 envelope. Legacy variants are pinned to `bind_ramp=False` so they keep meaning what their names say.
- `scripts/analysis/smoke_body_economy_prices.py` (new): read-only live smoke.
- Tests:
  - `orion/physiology/tests/test_budget.py` (new);
  - `test_prices.py`;
  - `scripts/analysis/tests/test_body_economy_replay_and_fit.py`;
  - `services/orion-durable-runs/tests/test_physiology_node.py`;
  - `services/orion-sql-writer/tests/test_physiology_state_sql_shape.py`.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: the "Patch B results" section.

## Schema / bus / API changes

- Added:
  - `PhysiologyConstraintReadingV1.price: Optional[PhysiologyPriceV1]`;
  - `PhysiologyStateV1.strain: Optional[PhysiologyStrainV1]`;
  - `GET /physiology/state` (durable-runs, read-only).
- Removed, renamed: none. No new channel and no new table column.
- Behaviour changed: none for existing readers. No reader decides on physiology yet.
- **Compatibility: consumer-first.** Both models are `extra="forbid"`, and sql-writer validates incoming payloads against the same model, baked into its image. **sql-writer must be rebuilt before durable-runs**, or every new tick is rejected there. The prices are stored inside `constraints`. `strain` has no column; it is a pure function of the stored prices (`strain_dial`), which a test verifies. No `SQL_WRITER_ROUTE_MAP_JSON` or `SUBSCRIBE_CHANNELS` edit is needed.

## Env/config changes

- Added keys, both in `services/orion-durable-runs`, both ON:
  - `ORION_PHYSIOLOGY_PRICES_ENABLED=true`;
  - `ORION_PHYSIOLOGY_HEAT_SENSITIVITY_PATH=/app/config/physiology/heat_sensitivity.yaml`.
- Removed or renamed keys: none.
- `.env_example` updated: yes, and so are `settings.py` and `docker-compose.yml`.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. It added both keys to `/mnt/scripts/Orion-Sapienform/services/orion-durable-runs/.env`. The one diverged key it reported (`DURABLE_RUNS_GRAPH_HOST`) was already there and is unrelated.
- Skipped keys requiring operator action: none.

## Replay (R1, R3, R4 on 10-07..10-23, the live rules)

Command:

```text
fit_heat_sensitivity.py --end 2026-10-27T00:00:00Z --write-config
replay_body_economy.py --fit-json heat_fit_1027.json --extract-end 2026-10-24T00:00:00Z --replay-end 2026-10-24
```

Live rules: point short-run sensitivities (athena 0.686, circe 0.298 °C/100 W), baselines 224 W and 263 W, the cabinet −2.0 °C from 10-17 00:35, a bounded ramp, 0.5 °C bind hysteresis, and decay of ceiling/6 h.

| forecast | R1 (decay tail excused) | under-line ~0 | over-line >0 | unmeetable flagged | R3 | max h to 0 | stuck | worst osc | unflagged ceiling run | ceiling h: binding / windup |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | FAIL | 0.92 (271 h; tail 22, other 0) | 1.00 (135 h) | 49/62 | FAIL | 5 | 0 | 0.00 | 17 h | 42 / 0 |
| **persistence (live)** | **PASS** | 0.88 (276 h; tail 34, other 0) | 1.00 (132 h) | 50/54 | **FAIL** | 5 | 0 | 0.69 | 9 h | 29 / 0 |

- **R4: PASS.** Worst error −6.1% against the realized $197.17 for 09-23..10-23. The rate-only forecast is −9.3%, also a pass.
- Energy at the $100 envelope: `exogenous_bind` for 408 of 408 h. The price ramps 0.25, 0.50, 0.75, 1.00 (bounded, no jump). Energy R3 PASS.
- **R6 (reservations): DEFERRED to Patch C.** It needs bids.

**Verdict: R1 PASS (live forecaster), R4 PASS, R3 FAIL.** Here is why R3 fails, in plain words:

- **Not windup.** There are 0 hours where the price sat at its ceiling while the constraint was slack. Every unflagged ceiling hour (42 oracle, 29 persistence) was an hour where Orion's own heat share was more than the room's headroom.
- **The trigger is the sensor correction the findings asked for.** On 10-18..20 the corrected cabinet sat at 31.0–31.9 °C, and the room without Orion was just under the line. Those hours used to read "too hot, not my load", and are now "my load is what crosses the line".
  - The same rules on the raw reading: oracle R1 PASS (146/153), R3 PASS (3 h run).
  - The same rules on the raw reading, persistence: R1 PASS, R3 fails on oscillation only (0.83).
- **An open-loop replay cannot pass that clause on a stretch that really binds.** Nothing in it lowers Orion's load as the price rises. Closing the loop needs bids (Patch C).
  - Live, the stuck check flags such a run `stuck_at_ceiling` after one horizon (6 h). The flag reaches the warnings and `strain.stuck`.
  - I tried freezing the integrator during bind (back-calculation anti-windup) in a scratch prototype. Nothing changed, as expected, because nothing was winding up.
- **Persistence oscillation (0.69 vs 0.5):**
  - When the room sits on the line, the bind flag dithers between "clamped" and "live".
  - The window peak-to-peak includes the ramp leg. The reversing legs themselves swing about 0.17.
  - I did not change the metric or the 0.5 °C hysteresis to make it green.

## Tests run

```text
orion/physiology/tests + scripts/analysis/tests/test_body_economy_replay_and_fit.py + tests/test_check_body_economy_contract.py: 152 passed
services/orion-durable-runs/tests (isolated Postgres 16):   629 passed, 1 skipped
services/orion-sql-writer/tests/test_physiology_state_sql_shape.py: 5 passed (incl. priced payload accepted + persisted, strain recomputable)
make check-body-economy-contract: OK (71 modules)
check_definition_drift --gate, check_metric_lineage --gate, check_sentience_instruments --static-only,
check_async_routes_not_blocking, check_env_key_single_source, check_inner_state_registry,
check_env_template_parity, check_scripts_dir_no_stdlib_shadow: PASS
git diff --check: clean
```

## Evals run

```text
Replay (above). The 24 h live smoke (below) is the runtime evaluation.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile (throwaway tag, removed): OK.
  In the image: heat_sensitivity.yaml loads, athena 0.6863 / circe 0.2985 usable, prices flag default True.
  (safe_docker_build.sh needs a root .env, which a linked worktree does not have; the plain build checks the COPY.)

Live smoke (read-only; scripts/analysis/smoke_body_economy_prices.py --hours 24; 720 ticks 10-26 03:54 .. 10-27 03:53 UTC):
- heat: slack 720/720, price 0.000 throughout. Last tick: cabinet 31.1 C -> 29.1 C after the sensor offset;
  Orion's share 0.85 C (athena +38 W x 0.69, circe +199 W x 0.30, hecate asleep); room 28.2 C vs line 31.0 C.
- energy: exogenous_bind 720/720. Month to date $166.69, forecast $197.17, always-on floor alone $192.07 vs $100.
- attention:juniper: binding from 12:18 local (4 of 4 cap-counted sends), at the ceiling 108 ticks (3.6 h, under its 24 h horizon).
- lane:agent: binding 187 ticks, max 0.105 (gap borrowers over 1 slot). lane:diffusion: exogenous_bind 193 (no live slot, lease held).
  lane:agent-gpu2: exogenous_bind 73.
- Checks: heat ~0 under the line (0 of 720 over 0.05) OK; energy exogenous_bind OK; no unflagged ceiling run past a horizon OK.
- strain: at rest 151/720; drivers attention:juniper 316, lane:diffusion 174, lane:agent-gpu2 49, lane:agent 30.
```

## Metric gate: the `strain` dial and the prices

1. **Provenance.** `orion/physiology/budget.py` `strain_dial(constraints)` reads `PhysiologyPriceV1.value / ceiling`. Each price comes from `price_constraint`, which calls `prices.step_price`. Its inputs are:
   - Body readings (`orion/physiology/body.py`, read by `physiology_store`);
   - host watts (`read_host_watts`);
   - `config/physiology/heat_sensitivity.yaml`.
2. **Independence.** `strain` is a monotonic transform of the prices. It is a rendering, **not new signal**. Each price is a function of its own Body reading. The heat price adds host watts and the fitted sensitivity. Those watts are the same compute watts the fit regressed on, so this is not a separate measurement.
3. **Theory.** A price is the Lagrange multiplier of its constraint (dual ascent, Kelly network pricing): the marginal value of relaxing it. `strain` is the largest normalized multiplier.
4. **Live sanity (rest state, not just variance).**
   - Strain reached exactly 0 on 151 of 720 live ticks, so it is not floor-locked.
   - The heat price was 0 on all 720 ticks under the line, and its use (Orion's share) varied between ticks.
   - Energy reads `exogenous_bind` and is excluded from the dial, as designed.
   - Watch item: with provisional ceilings of 1.0, a binding `attention:juniper` price saturates the dial at 1.0 for the afternoon. Ceilings get sized once wants exist (Patch C).
5. **Existing mechanism.** The design's only strain signal is `arousal.strained` (categorical). Patch D derives it from this dial. Nothing else computes prices.
6. **Reversibility.** The fields are additive and optional. `ORION_PHYSIOLOGY_PRICES_ENABLED=false` turns the prices off. No reader decides on them, so removing them costs nothing.

## Review findings fixed

(see below; filled after review)

## Restart required

Deploy order: **B1 (#2911) first, then this PR. Rebuild sql-writer BEFORE durable-runs** (consumer-first). Both commands run from the clean primary `main` checkout after each merge:

```bash
./scripts/safe_docker_build.sh orion-sql-writer up -d --build
./scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then check the live path:

```bash
curl -fsS http://localhost:8124/physiology/state | python3 -m json.tool | head -60
```

## Risks / concerns

- Severity: medium.
  - Concern: replay R3 FAILs on the live rules (open-loop saturation on 10-18..20 after the sensor correction; persistence oscillation 0.69).
  - Mitigation: the prices are shadow, and the stuck check flags any live run.
  - Decision for Juniper: accept the shadow rollout with R3 open, or hold. Closing R3 needs the Patch C loop.
- Severity: medium.
  - Concern: deploying durable-runs before sql-writer makes every tick fail validation at sql-writer, so rows are lost until sql-writer is rebuilt.
  - Mitigation: the deploy order above. Redis still carries the latest state.
- Severity: low.
  - Concern: ceilings are provisional (1.0). Strain saturates on any binding constraint.
  - Mitigation: labelled `ceiling_source: provisional` on every price. `ceiling_for` sizes them once wants exist.
- Severity: low.
  - Concern: the heat forecast is persistence (the newest reading), not the design's 6 h forecast.
  - Mitigation: stated in `why` and in the README. The replay graded persistence as the live forecaster.
- Severity: low.
  - Concern: `heat_sensitivity.yaml` is baked into the image and goes `no_reading` after 30 days (11-26).
  - Mitigation: refit with `--write-config`, merge, and the rebuild picks it up (`rebuild_affected_services` maps the file to durable-runs).

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
