# Cost views include the whole machine's overhead

Follows #2992, #2991 and #2987 (all three merged into main before this branch was pushed; it is rebased on that main).

## Summary

- Every cost number now means one thing: the whole machine's power at the wall above rest, fans and power-supply loss included. That is the unit an act is charged in since #2992.
- A card's watts, a container's CPU watts, Juniper's sponsored work and the painting run cost are parts. Each now shows its own watts and, beside it, an estimate of the whole-machine watts it drives: "gpu1 199 W, rest 42 W (≈368 W at the wall above rest)".
- The factor is fitted per machine on its own last 14 days, from the same readings and the same pinned rest the per-act meter uses. Live today: circe 2.34, athena 1.27. When a fit is not usable, #2992's circe fit (2.69) is used and the text says so.
- The dig text, the Hub Machines and Processes panels, the unit-cost table and the report fields all say the wall figures are ESTIMATES and say how the factor was found.
- The durable-runs image now contains `config/physiology/rest_body.yaml`. Before, the report in the container could not see it and fell back to its built-in defaults.

## Outcome moved

Before: "gpu1 199 W" (a part) and "this act cost 7 W·h" (the whole machine) sat side by side in different units. A reader added or compared them as if they were the same thing. Now each part also carries the whole-machine figure in the act's unit, and is labelled an estimate.

The painting unit cost moves from 0.27 W·h (GPU only) to about 0.63 W·h at the wall (circe, live 2026-10-30, 14 runs). Painting acts themselves still carry no whole-machine cost: 0 of 27 `thought.painting` acts in the last two days have one, because the act and its diffusion lease share no correlation id (the #2977 finding). So the scaled run cost is the painting cost until that link exists.

## Current architecture

- `GET /physiology/report` (orion-durable-runs, `app/physiology_report.py`) built `hosts` (#2991), `processes` (#2987 Hub panel on the existing section) and `unit_costs` from `orion_biometrics_summary`, `energy_run_cost` and `grammar_acts`.
- The per-act meter (`orion/energy/act_energy.py`, #2992) splits each second's chassis watts above the pinned rest over the load above idle. The 2.69 fit lived only in its docstring.
- Readers: dig text (`orion/physiology/report_text.py`, `python -P -m orion.dig body-economy`), Hub `#body-economy` (`static/js/body-economy.js`).

## Architecture touched

- New pure module `orion/energy/overhead.py`: the per-host fit and its fallback. One home for the 2.69 value.
- `physiology_report.py`: one new read (`HOSTS_OVERHEAD_SQL`, hourly means over 14 days, about 0.55 s live, cached 6 h), run when `hosts`, `processes` or `unit_costs` is asked. The pinned rest file and pinned parts are now also loaded for those sections.
- Report contract: new optional fields only (see below). No bus or channel change.

## Files changed

- `orion/energy/overhead.py`: the fit, the fallback and `Overhead.wall()`.
- `orion/energy/tests/test_overhead.py`: fit recovery, CPU-idle fill, every fallback reason, bad numbers.
- `orion/energy/act_energy.py`: docstring points at the view-side fit.
- `services/orion-durable-runs/app/physiology_report.py`: overhead read, cache, `overheads_by_host`, wall fields in hosts, processes and unit costs; `energy_run_cost` grouped by node.
- `services/orion-durable-runs/tests/test_physiology_report.py`: report fields, fallback, failed read, real-Postgres path (hourly fit on inserted history, run-cost node column).
- `services/orion-durable-runs/Dockerfile`: copies `rest_body.yaml`.
- `services/orion-durable-runs/README.md`, `services/orion-hub/README.md`: the method and fields.
- `orion/schemas/physiology_report.py`: optional fields.
- `orion/physiology/report_text.py`, `orion/dig/__main__.py`: dig text and help.
- `orion/physiology/tests/test_report_text.py`, `orion/dig/tests/test_dig.py`: text tests on a new live fixture.
- `orion/physiology/evals/fixtures/physiology_report_live_2026-10-30_overhead_24h.json`: live report, built read-only against production at 2026-10-30T04:37Z.
- `orion/physiology/evals/physiology_state_skill_eval.py`: new `overhead` check and scenario.
- `services/orion-hub/static/js/body-economy.js`, `body-economy.test.js`, `tests/fixtures/body_economy_report.json`, `tests/test_body_economy_browser_smoke.py`: Hub panels, fixture, unit and browser tests.
- `.github/workflows/orion-durable-runs-tests.yml`: runs on changes to `orion/energy/overhead.py`.

## Schema / bus / API changes

- Added (all optional, default None/0):
  - `HostPowerV1`: `overhead_factor`, `overhead_basis`, `overhead_fitted`, `overhead_r2`, `overhead_hours`, `load_above_idle_w`, `wall_est_parts_w`, `overhead_now_ratio`.
  - `HostCardV1`: `above_idle_w`, `watts_at_wall_est`.
  - `HostContainerV1`, `HostSponsoredV1`: `watts_at_wall_est`.
  - `ProcessV1`: `watts_at_wall_est`, `overhead_factor`, `overhead_basis`.
  - `UnitCostV1`: `wh_component`, `overhead_factor`, `overhead_basis`.
- Removed: none.
- Renamed (unit string): `one reverie_diffusion run` -> `one reverie_diffusion run (painting) on <node>` (`energy_run_cost` is now grouped by node).
- Behavior changed: that row's `wh`/`usd` are at the wall (status `derived`, priced at the newest marginal rate); the GPU-only figure is in `wh_component`. With no factor for its node the row is `not_measured`.
- Compatibility: models allow extra fields and readers treat absent fields as "no estimate". An older report renders exactly as before (tested in dig text and Hub).

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Metric quality gate

1. **Provenance.** `chassis_watts`, `gpu_watts_total`, `cpu_watts_total`, `cpu_idle_watts`, `gpuN_watts`, `container_cpu_watts.*`: `orion/telemetry/biometrics_pipeline.py` (iLO or wall PDU; nvidia-smi; RAPL; the fitted CPU idle), stored in `orion_biometrics_summary`. Rest: `config/physiology/rest_body.yaml` (Juniper approved 2026-10-30). GPU idle for the fit: the report's pinned p05 of `gpu_watts_total` over 2026-10-01..13 (`HOSTS_REST_SQL`). `energy_run_cost.energy_kwh`: `orion/energy/run_cost.py`, GPU above baseline.
2. **Independence.** Not new signal. Every wall figure is a derived view of the same chassis measurement the per-act meter already splits: k is fitted from chassis vs the same GPU and CPU readings, and a part's wall figure is that part's own reading times k. Nothing here should be fed into a model next to `chassis_watts` or the act costs as if it were a separate input.
3. **Theory.** Fans, PSU conversion loss, memory and VRMs draw in proportion to the component load that makes the heat (#2992's anchor; PSU loss is a near-constant fraction of DC load, fans follow heat). The slope through zero is the least-squares form of #2992's relation.
4. **Live data.** Read-only against production, 2026-10-16..30, 337 hourly means per host: circe k 2.34, R² 0.90; athena k 1.27, R² 0.98 (athena's overhead per watt is about half of circe's, so one constant would have over-charged athena's parts by about 2x). The fit reproduces #2992's order of size (2.69 used rolling card floors, which read higher and shrink the denominator). Rest point: a part at its idle reads exactly 0 W at the wall (live: circe gpu0 37.1 W at a 37.1 W rest → 0.0), so the view can read calm. A part with unknown idle reads None, never 0.
5. **Existing mechanism.** The only prior relation was the 2.69 in `act_energy.py`'s docstring. `heat_sensitivity.yaml`'s `baseline_watts_p10` is a rolling fit input, not an overhead. Nothing else maps component watts to wall watts.
6. **Reversibility.** Cheap. Optional report fields and one cached read. No stored table, no training default, no manifest. Removing the module and fields restores the old views.

## Tests run

```text
PYTHONPATH=. python -m pytest orion/physiology orion/dig orion/energy -q        436 passed
cd services/orion-durable-runs && ORION_ADMISSION_TEST_DSN=<throwaway postgres:16> python -m pytest tests -q
                                                                                 695 passed, 1 skipped (test_physiology_report.py 31 passed after the last fix)
node --test services/orion-hub/static/js/body-economy.test.js                    19 pass, 0 fail
python -m pytest services/orion-hub/tests/test_body_economy_routes.py tests/test_body_economy_browser_smoke.py
                                                                                 17 passed; browser smoke 3 passed (Chromium, playwright venv)
python scripts/check_inner_state_registry.py                                     OK
git diff --check                                                                 clean
```

## Evals run

```text
PYTHONPATH=. python orion/physiology/evals/physiology_state_skill_eval.py        passed: true
  (new live_overhead scenario: overview, all, hosts --host circe incl. the overhead check)
```

## Docker/build/smoke checks

```text
docker build -f services/orion-durable-runs/Dockerfile .                         built
docker run ... python -c "load_rest_body('/app/config/physiology/rest_body.yaml')"
  -> /app/config/physiology/rest_body.yaml exists; "pinned in config/physiology/rest_body.yaml"
Live read-only report (build_report against production, default_transaction_read_only=on):
  no read failed; first uncached call: overhead 0.55-0.79 s, total 1.7-2.3 s; cached after
```

Sample dig output for circe (live 2026-10-30T04:54Z, `body-economy hosts --host circe`; the fixture `orion/physiology/evals/fixtures/physiology_report_live_2026-10-30_overhead_24h.json` is this report):

```text
circe draws 1370 W now vs 541 W at rest (+829 W, about $76/month at this rate; my body at rest there costs about $50/month).
- made of (each part vs its own rest): GPUs 309 W (+115), CPU 120 W (+75), other 941 W (+661); fans 56%. Other = whole machine minus GPUs minus CPU: fans, PSU loss, memory, disks
- rest: pinned in config/physiology/rest_body.yaml: p05 chassis_watts over 2026-10-01..2026-10-13
- overhead: each watt of GPU/CPU load above idle draws about 2.34 W at the wall above rest (fans, power-supply loss); at-the-wall figures are ESTIMATES. fitted on circe 2026-10-16..2026-10-30: chassis above pinned rest 541 W = 2.34 x (GPU above 193 W + CPU above 100 W idle), 337 hourly means (337 with load), R^2 0.90; with an intercept: 2.05 x load + 76 W (always-on draw above the pinned rest, which this k charges...
- right now the parts' at-the-wall estimates add to 318 W against 829 W measured above rest (this sample's ratio 6.11, the fitted 2.34): fans and power-supply loss follow load over minutes to hours, so a part's figure is its average cost over hours, not a split of this second
- this week's lowest draw is 578 W, 37 W above the pinned rest: new always-on draw
- gpu0 37 W, rest 37 W (≈0 W at the wall above rest, est.), 39 °C (chat/experiment); held by curiosity.investigate runs 5.5 h, reading.turn runs 4.8 h, harness turns 2.8 h, cortex-exec 2.7 h
- gpu1 48 W, rest 42 W (≈14 W at the wall above rest, est.), 62 °C (agent/experiment); held by curiosity.investigate runs 10.8 h, reading.turn runs 7.8 h, cortex-exec 7.2 h, harness turns 4.7 h
- gpu2 151 W, rest 42 W (≈256 W at the wall above rest, est.), 59 °C (agent-gpu2/diffusion/experiment/world); held by reading.turn runs 7.7 h, self_sense_eval runs 4.2 h, cortex-exec 4.0 h, harness turns 117 min
- gpu3 74 W, rest 73 W (≈1 W at the wall above rest, est.), 52 °C (experiment/fast/kev/metacog); held by vision-council 9.2 h, orion-topic-foundry 4.7 h, AI Town 3.3 h, cortex-exec 90 min
- top CPU users (watts above CPU idle): dd62b3275f3b 7.4 W (≈17 W at the wall above rest, est.), 43baf7e0588d 5.4 W (≈13 W at the wall above rest, est.), (host, not in a container) 3.9 W (≈9 W at the wall above rest, est.), e5287fa21cba 2.1 W (≈5 W at the wall above rest, est.), 051e28075f1c 1.3 W (≈3 W at the wall above rest, est.)
- daily average (first and last day partial): 10-28 963 W, 10-29 979 W
```

## Review findings fixed

Code review ran in a subagent on the feature commit. No blockers.

- Finding: a run cost with no factor still filled `wh`/`usd` with the GPU-only figure (status `measured`), against the new rule that `wh`/`usd` mean the whole machine.
  - Fix: that row is now `not_measured`, `wh`/`usd` None, the GPU figure only in `wh_component`.
  - Evidence: `test_a_run_cost_with_no_factor_is_not_passed_off_as_whole_machine`.
- Finding: a failed pinned-rest read was reported only under `hosts`, and processes/unit costs then said "no GPU rest" (a data gap that did not exist).
  - Fix: the read is registered for every section it feeds; the fallback basis says "the pinned rest read failed".
  - Evidence: `test_a_failed_pinned_rest_read_is_said_as_a_read_failure_in_every_section_it_feeds`.
- Finding: the Hub said "(≈N W at the wall)", which reads as total wall draw.
  - Fix: "(≈N W at the wall above rest, est.)" in the Hub and in the dig text.
  - Evidence: JS tests, browser smoke, report_text tests.
- Finding: the through-zero fit pushes drift above the pinned rest into k, silently.
  - Fix: kept through zero on purpose (the meter charges all above-rest draw to load, and the goal is one unit), but the basis now also gives the fit with an intercept (circe 2.05 x load + 76 W), and the docstring says why.
  - Evidence: `test_drift_above_the_pinned_rest_lands_in_k_and_the_basis_says_how_much`.
- Finding: run-cost USD was `energy_run_cost.usd x k` (that USD is tiered on the billing cycle), and its baseline (the card's draw just before the run) is not the fitted idle.
  - Fix: priced at the newest marginal rate like the act rows; the source says the baseline differs and a warm card reads low.
  - Evidence: the full-report test checks `usd == wh / 1000 x rate` and the source words.
- Finding (nit): the cache kept the old window but the basis printed today's dates; an empty answer was cached for 6 h.
  - Fix: the window words travel with the cached rows; an empty answer is not cached.
- Finding (nits): R^2 form not said; "N hourly means" counted unloaded hours; the CPU-idle fill test could not tell median from mean; fitted/fallback read from a text prefix; the Hub hard-coded "14 days".
  - Fix: docstring says centred R^2; basis says "(N with load)"; new median-vs-mean test; `overhead_fitted` used where present; the Hub says "recent history" and the hover shows the exact window.
- Not changed (nit): a failed overhead read still says "read failed (overhead)" under processes and unit costs although they render with the fallback. The `read failed` prefix is what keeps a degraded report out of the route cache (`has_read_failure`), and the fallback basis on every row says the factor is the fallback.
- Not in the review, found in the live check after it: at one instant the parts' estimates do not split the measured surplus (two live samples two minutes apart: this second's ratio 1.23, then 6.11, against the fitted 2.34). Added `load_above_idle_w`, `wall_est_parts_w`, `overhead_now_ratio` per host, and a dig and Hub line when the parts' sum and the measured surplus differ by more than 25%. Evidence: report, text and JS tests; the live sample above shows the line.

## Restart required

Not deployed by this branch. After merge, the post-pull rebuild picks up both services; by hand from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: a part's wall figure is an hours-average. At one instant the parts' sum can be far from the measured surplus (live ratio 1.23 then 6.11 two minutes apart). Mitigation: every figure says "est."; the measured "above rest" stays the headline; the text and the Hub say when the two differ by more than 25%.
- Severity: low. Concern: the run-cost unit name changed to `one reverie_diffusion run (painting) on circe`. Only tests matched the old name (grepped). Anything that keys on `unit` must use the new name.
- Severity: low. Concern: the first uncached 24 h report took 2.3 s live (overhead read ~0.8 s, pinned parts ~1.3 s); both are cached for 6 h after that.
- Severity: low. Concern: the card estimates use each card's 7-day p05 idle (per-card watts began 10-29), while the fit uses the pinned GPU-total rest. The two agree live (circe card idles sum to 193.3 W vs 193.4 W pinned), but they can drift. Mitigation: pinning per-card idle in `rest_body.yaml` is #2992's open follow-up.
- Severity: low. Concern: `cpu_idle_watts` is only published since 2026-10-29; older hours use the host's own median of it (the setting is one value per host). Mitigation: said in the basis ("CPU above 100 W idle").
- Severity: low. Concern: Juniper's example "gpu3 261 W (≈700 W at the wall)" multiplied total card watts. This branch multiplies watts ABOVE IDLE, because a card's idle is already inside rest; multiplying the total would count rest twice. Mitigation: the text says "at the wall above rest".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2997

🤖 Generated with [Claude Code](https://claude.com/claude-code)
