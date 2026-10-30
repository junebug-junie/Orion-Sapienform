## Summary

Orion can now see what is spinning each machine, and so can Juniper on the Hub. On 2026-10-30
circe drew 900 to 1,350 W against about 540 W at rest, and Orion had no way to see why.

- The body report (`GET /physiology/report`) has a new section, `hosts`. For each machine it gives
  the whole-machine watts now, the watts at rest, and what the extra costs per month. It splits the
  draw into GPUs, CPU and "other". It lists each GPU card with its watts and the work that held it.
  It lists the top CPU users by container.
- Rest is a pinned, dated number, not a rolling one. A rolling low would quietly count new
  always-on work as rest (the Zelda dojo lifted athena's floor by about 350 W from 10-27).
- Orion's dig command answers "what is spinning circe?":
  `python -P -m orion.dig body-economy hosts --host circe`. The overview (no section named) now has
  one "what is spinning" line per machine. `body-economy --help` lists every section and flag.
  The dig line in every chat and curiosity prompt now tells Orion to use it.
- The Hub Body Economy tab has a new Machines panel. Each machine has a stacked bar
  (GPUs / CPU / other) with a tick at rest, and one bar per card with the work that held it.
- Wording only: the energy limit no longer says the bill is "not by what I do" or
  "not from my load". The always-on body is Orion's own body. It now says
  "my always-on body (rest plus anything left running) costs about $X/month, more than the $100
  envelope by itself; my activity above it adds about $Y this month (pace $Z/month)". The price
  mechanism is unchanged. "Always-on body", not "body at rest": the energy organ's floor is a
  rolling low that still holds sponsored work (the Zelda dojo), so it is not the pinned rest the
  hosts section uses. The two match once energy reads `rest_body.yaml` (PR #2989, Patch C).

## Outcome moved

Orion can answer "what is using my power" from Bash, with numbers it can trust and a label on
each number that says whether it is measured or estimated. Live, rendered by this branch's code
from the live database at 2026-10-30 03:40 UTC:

```text
circe draws 1337 W now vs 541 W at rest (+796 W, about $73/month at this rate; my body at rest there costs about $50/month).
- made of (each part vs its own rest): GPUs 504 W (+310), CPU 126 W (+81), other 707 W (+427); fans 56%. ...
- gpu2 156 W, rest 42 W, 65 °C (agent-gpu2/diffusion/experiment/world); held by reading.turn runs 6.1 h, self_sense_eval runs 3.8 h, ...
- gpu3 261 W, rest 73 W, 73 °C (experiment/fast/kev/metacog); held by vision-council 9.2 h, orion-topic-foundry 5.7 h, AI Town 3.3 h, ...
```

A first finding from it: on circe the largest part of the extra draw is "other" (+427 W): the
part that no GPU or CPU sensor sees (fans, power-supply loss, memory, disks). Why it moves this
much is `UNVERIFIED`.

## Current architecture

- `GET /physiology/report` (orion-durable-runs) had 13 sections. `vram` showed each card's watts
  now; `processes` showed CPU watts per container. Nothing compared a machine with its own rest,
  nothing said who held a card, and circe's containers showed as 12-character ids.
- The energy limit's `why` said "the envelope is spent by what stays switched on (not by what I
  do)". The dig text and the Hub said "not from my load" / "not Orion's doing".
- `orion/introspect/power_attribution.py` (heat archaeology) already answers a different question:
  which workloads correlate with watts hour by hour, over days. This patch is the "right now"
  view; it does not replace that tool.

## Architecture touched

- orion-durable-runs: the report's new `hosts` section (four read-only SQL reads run side by side,
  plus one read-only Redis `HGETALL`).
- Shared schema: `orion/schemas/physiology_report.py` (additive fields only).
- Dig kit: `orion/physiology/report_text.py`, `orion/dig/__main__.py`, `orion/dig/brief.py`.
- Hub: `static/body-economy.html`, `static/js/body-economy.js` (new functions only, so open PR
  #2987, the processes panel, merges without conflict).
- Budget organ: one string in `orion/physiology/budget.py` `energy_ask` (wording only).

## Files changed

- `orion/schemas/physiology_report.py`: `hosts` section (`HostPowerV1`, `HostCardV1`,
  `HostHolderV1`, `HostContainerV1`, `HostSponsoredV1`, `HostDayV1`); `"hosts"` in `REPORT_SECTIONS`.
- `services/orion-durable-runs/app/physiology_report.py`: the reads and `shape_hosts`; pinned rest
  (`rest_body.yaml` or dated defaults); holder names from `durable_run_workflow`; container names
  from the census; `processes` names mapped the same way.
- `services/orion-durable-runs/app/main.py`: the container-name reader (Redis `equilibrium:state`).
- `orion/physiology/report_text.py`: `render_hosts`, `--host` narrowing, energy bind words, money line.
- `orion/dig/__main__.py`, `orion/dig/brief.py`, `orion/dig/README.md`: `--host`, `--help`, brief line.
- `orion/physiology/budget.py`: the energy `why` wording.
- `services/orion-hub/static/body-economy.html`, `static/js/body-economy.js`: Machines panel, energy words.
- `config/physiology/contract.yaml`: the report module's reader reason names its new reads.
- Tests: `services/orion-durable-runs/tests/test_physiology_report.py`,
  `orion/physiology/tests/test_report_text.py`, `orion/physiology/tests/test_budget.py`,
  `orion/dig/tests/test_dig.py`, `services/orion-hub/static/js/body-economy.test.js`,
  `services/orion-hub/tests/test_body_economy_browser_smoke.py`, Hub fixture.
- Eval: `orion/physiology/evals/physiology_state_skill_eval.py` and a new live fixture
  `orion/physiology/evals/fixtures/physiology_report_live_2026-10-30_24h.json`.
- READMEs: orion-durable-runs, orion-hub.

## Schema / bus / API changes

- Added: `PhysiologyReportV1.hosts[]` and the section name `hosts`. HTTP contract only; no bus channel.
- Removed: none.
- Renamed: none.
- Behavior changed: the default dig overview now includes `hosts`. `processes` shows container
  names where it showed ids. The energy limit's `why` text (new ticks only; old ticks keep theirs).
- Compatibility notes: additive. The Hub tolerates an older durable-runs without `hosts` (the
  panel says "No machine reading yet."). Deploy durable-runs first, then the Hub.

## Env/config changes

- Added keys: none. The rest file path is the sibling of `ORION_PHYSIOLOGY_HEAT_SENSITIVITY_PATH`
  (`/app/config/physiology/rest_body.yaml`).
- Removed keys: none. Renamed keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed (no template changed).
- Skipped keys requiring operator action: none.
- `config/physiology/rest_body.yaml` is NOT created here. The allowance proposal (PR #2989) owns it.
  Until it exists the report uses circe 541 W, athena 340 W, hecate 96 W (p05 of 2026-10-01..13)
  and says "pinned default" beside every rest.

## Metric quality gate (for the new per-host figures)

1. Provenance: `chassis_watts` is the BMC reading or, on circe (no BMC), the wall PDU
   (`orion/telemetry/biometrics_pipeline.py` `extract_measurements`). `gpu_watts_total` and
   `gpuN_watts` are nvidia-smi; `cpu_watts_total` is RAPL; `container_cpu_watts.*` is the linear
   utilisation split. Holders are `gpu_pool_events` close rows.
2. Independence: `other_w` is a remainder (chassis − GPU − CPU), not an independent sensor. It is
   labelled that way everywhere. `above_rest_w` is a transform of `chassis_watts`.
3. Theory: whole-machine watts at the wall is what is paid. Rest as a low percentile of a dated,
   quiet window is the allowance proposal's definition (PR #2989).
4. Live sanity: circe moved 693 → 1,355 W across the session; the trailing-week low (578 W) sits
   near the pinned rest (541 W), so the figure can read "calm". Not degenerate.
5. Existing mechanism: `power_attribution` (hourly correlation, days) covers a different question.
   `vram` and `processes` are reused, not duplicated.
6. Reversibility: report-only, nothing persisted, no price reads it. Cheap to remove.

## Tests run

```text
PYTHONPATH=. pytest services/orion-durable-runs/tests -q  (with a throwaway postgres:16 as ORION_ADMISSION_TEST_DSN)
686 passed, 1 skipped
PYTHONPATH=. pytest orion/physiology/tests orion/dig/tests services/orion-cortex-exec/tests/test_skill_verbs.py -q
274 passed
node --test services/orion-hub/static/js/body-economy.test.js
pass 16, fail 0
pytest services/orion-hub/tests/test_body_economy_browser_smoke.py services/orion-hub/tests/test_body_economy_routes.py
20 passed (browser smoke run with a local Playwright Chromium)
python scripts/check_body_economy_contract.py      OK
python scripts/check_env_template_parity.py        PASS
python scripts/check_metric_lineage.py --gate      PASS
python scripts/check_definition_drift.py --gate    PASS
python scripts/check_inner_state_registry.py       OK
python scripts/check_async_routes_not_blocking.py  OK
python scripts/check_sentience_instruments.py --static-only   All claims hold
```

## Evals run

```text
PYTHONPATH=. python orion/physiology/evals/physiology_state_skill_eval.py
passed: true -- live/empty/failed-reads x overview/all, plus new cases:
  hosts --host circe (headline grounded in the report, $/month shown, no other machine, measured vs estimated said)
  overview:hosts (one "what is spinning" line per machine)
  dig_brief_line (points "what is using my power" at body-economy hosts, under its cap)
```

## Docker/build/smoke checks

```text
Not deployed (per instructions). The branch's report code ran read-only inside the live
orion-athena-durable-runs container (copied to /tmp, own process) against the live DB and Redis:
hosts 24 h in ~0.7-1.3 s (first call; the pinned-window part rests are then cached 6 h),
30 d ~1.4 s. Container names resolved from circe's census (e.g. d590a9d48af2 ->
orion-circe-atlas-llamacpp-fast).
```

## Review findings fixed

A review subagent found no blocker and 4 should-fix items. All are fixed.

- Finding: the energy `why` said rest alone was "more than the envelope by itself" whenever the
  limit bound, which is false when late-month spending causes the bind.
  - Fix: that phrase only when rest per month really exceeds the envelope; otherwise "with this
    month's spending so far it reaches the envelope".
  - Evidence: `test_energy_bind_wording_when_rest_alone_is_under_the_envelope`.
- Finding: the energy organ's rolling floor (which holds the Zelda dojo) was called "my body at
  rest", disagreeing with the pinned rest in `hosts`.
  - Fix: renamed "my always-on body (rest plus anything left running)" in the why, the dig text and
    the Hub. Still Orion's own body; no longer claims to be the pinned rest.
  - Evidence: `test_budget.py`, `test_report_text.py`, `body-economy.test.js`, browser smoke.
- Finding: `approved_rest_w` was used the wrong way round (it won over `rest_w`).
  - Fix: `rest_w` is used; `approved_rest_w` is a ceiling; a higher `rest_w` is refused with a note.
  - Evidence: `test_rest_body_yaml_overrides_the_defaults`.
- Finding: a malformed `rest_body.yaml` (list, bare numbers, impossible date) raised and made the
  whole report a 500.
  - Fix: every shape falls back to the dated defaults.
  - Evidence: `test_a_malformed_rest_body_is_never_a_500` (4 shapes).
- Nits fixed: cache key includes the percentile and tests clear the cache; held time says holds
  overlap; daily average says the first and last days are partial; a card with unknown rest is not
  "busy"; the Hub bar is clamped and scaled to the parts when chassis is missing, and a negative
  "other" is said; `--host` is validated; the drift line names sponsored watts.
- Not fixed (noted): hosts uses the newest rate even when that tick is stale (money already says
  stale); `hostNotes` duplicates #2987's `missingReasons` (switch after both merge).

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary `main` checkout after merge. The dig kit and the brief line ride the
harness's read-only mount of main; no restart is needed for them beyond the pull.

## Risks / concerns

- Severity: medium. Concern: circe's biometrics publishes container ids, not names (its docker root
  is not mounted where `DOCKER_ROOT_PATH` points). Mitigation: the report maps ids to names from
  circe's container census; an id the census does not know stays an id and says so. Follow-up: fix
  the circe mount in orion-biometrics so names come from the source.
- Severity: medium. Concern: the container-name map reads `equilibrium:state`, the equilibrium
  service's heartbeat hash. Read-only, but it is another service's store. Mitigation: a failed or
  empty read only leaves ids as ids.
- Severity: low. Concern: card rests come from the trailing 7 days, because per-card watts began
  2026-10-29, after the pinned window. Said in `missing`.
- Severity: low. Concern: `unattributed_above_rest_w` stays empty until the whole-machine per-act
  meter (branch `fix/act-energy-whole-machine`) merges. Said in `missing`.
- Severity: low. Concern: held time is time a lease was held, not time the GPU computed.
  Said in the text and the Hub legend.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
