# PR report: body-economy report API + `skills.physiology.state.v1`

## Summary

- Orion can now read their own body economy in plain words, in a chat turn and in a curiosity run. The
  command is `python -P -m orion.dig body-economy`. It says what binds now (heat, money, GPU lanes), what
  Orion wanted and did not get, how much GPU time was lost to failures, and which services are down.
- New read-only endpoint `GET /physiology/report` on orion-durable-runs returns one `PhysiologyReportV1`
  over 24 h, 7 d or 30 d. It has 12 sections: limits, wants, rhythm, money, unit costs, waste, queue wait,
  decisions, Claude, GPU memory, value per cost, services. The Hub body-economy page builds on the same shape.
- No section is ever silently empty. Each empty or partial section has a `missing` entry that says why
  (for example "prices began 2026-10-27", "per-card GPU memory is not reported yet").
- The skill verb named in the body-economy spec (Patch C) now exists: `skills.physiology.state.v1`, with a
  cortex-exec adapter for Skill Runner and verb plans.
- On/off switch: `ORION_PHYSIOLOGY_REPORT_ENABLED` (shipped on).

## Outcome moved

Before: Orion had no way to read their own body economy. The only read was `GET /physiology/state`: one
tick, for humans, with no history. Now Orion can ask "why did I wait", "what is binding", and "what did my
work cost", and the answer comes from the stored records.

Live check on 2026-10-29 (real data, read-only). The text Orion reads starts:

```text
Binding now: Energy money this month, Cabinet heat (my share against the soft line). Unknown (no price or no reading): Claude usage limits.
- Cabinet heat (my share against the soft line): BINDING, not from my load; 33.65 °C against a line of 31 °C. room over the line without Orion's load (not my load); cabinet 33.6 C; Orion's share 1.98 C ...
Money: $182.27 spent on electricity this month against a $100 envelope; forecast $196.87 by month end. What stays switched on alone reaches $193.80.
Waste: 40.5 h of GPU lease time ended without success (agent-gpu2 17.4 h, agent 12.8 h, chat 10.2 h, ...).
- durable-runs:d7d0c7d60c96 on agent: held 2.5 h, ended attempt_failed at 2026-10-29T18:58
```

The waste lines show today's incident: curiosity holds of about 8,500 s that ended in `attempt_failed`.

## Current architecture

- The physiology node writes one `physiology_state` row every ~120 s. Bids write `want_ledger` (shadow).
  Lease events go to `gpu_pool_events`.
- A unified chat turn and a curiosity run are both harness turns in orion-harness-governor. Each turn has
  Bash, the orion-introspect MCP tools, and the dig brief line.
- The model cannot call cortex-exec capability skills (`look_at_camera`, `show_biometrics_snapshot`) mid-turn.
  The capability bridge only runs under the agent supervisor, and that supervisor is retired
  (`supervisor.py` returns `agent_runtime_unavailable`).

## Architecture touched

- **orion-durable-runs**: new route `GET /physiology/report` and new module `app/physiology_report.py`,
  which does one SQL read and one pure shaping function per section. `Market.family_weights()` exposes the
  evidence the bids already use.
- **Contract**: `orion/schemas/physiology_report.py` (HTTP only, no bus channel), registered in
  `orion/schemas/registry.py`.
- **Renderer**: `orion/physiology/report_text.py`. It uses the stdlib only, and the dig kit and the
  cortex-exec adapter share it.
- **How Orion calls it**:
  - Chat turns and curiosity runs: `python -P -m orion.dig body-economy [<section>...] [--window 24h|7d|30d] [--family <f>]`
    in Bash. The dig brief line now says "your organs and body economy now".
  - Skill Runner and verb plans: `skills.physiology.state.v1` in cortex-exec, with skill_args `sections`,
    `window` and `family`.
- **Body-economy contract**: the three new readers are registered as `display` in
  `config/physiology/contract.yaml`. The verb is tier 0, family `autonomic`, the same as the other body
  reads. A read does not need a bid.

## Files changed

- `orion/schemas/physiology_report.py`: new `PhysiologyReportV1` contract.
- `orion/schemas/registry.py`: registry entries.
- `services/orion-durable-runs/app/physiology_report.py`: section SQL, shaping, assembly, cache.
- `services/orion-durable-runs/app/main.py`: the route.
- `services/orion-durable-runs/app/physiology_market.py`: `family_weights()`.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`: 4 report keys.
- `services/orion-durable-runs/README.md`: endpoint section.
- `services/orion-durable-runs/tests/test_physiology_report.py`: section, empty-state, failure, route
  and real-Postgres SQL tests.
- `orion/physiology/report_text.py`: fetch and render.
- `orion/physiology/tests/test_report_text.py`: renderer and fetch tests.
- `orion/physiology/evals/physiology_state_skill_eval.py` and `fixtures/physiology_report_live_2026-10-29_24h.json`:
  an end-to-end eval on real captured data.
- `orion/dig/__main__.py`, `orion/dig/brief.py`, `orion/dig/tests/test_dig.py`: the `body-economy` subcommand.
- `orion/cognition/verbs/skills.physiology.state.v1.yaml`: the verb.
- `services/orion-cortex-exec/app/verb_adapters.py`, `settings.py`, `.env_example`, `docker-compose.yml`,
  `tests/test_skill_verbs.py`: the adapter.
- `services/orion-harness-governor/.env_example`, `README.md`: `ORION_PHYSIOLOGY_REPORT_URL` for the dig kit.
- `config/physiology/contract.yaml`: the 3 display readers.
- `.github/workflows/dig-kit-tests.yml`, `orion-durable-runs-tests.yml`: CI path filters and a skill step.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: the results note.

## Schema / bus / API changes

- Added:
  - HTTP `GET /physiology/report` (orion-durable-runs).
  - Schema `PhysiologyReportV1` (`physiology.report.v1`).
  - Verb `skills.physiology.state.v1`.
  - Dig subcommand `body-economy`.
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing routes.
- Compatibility notes:
  - The top-level keys match the Hub contract. The additions beyond the base ask (`claude`, `vram`,
    `value_per_cost`, `services`) were approved by Juniper on 2026-10-29.
  - `services.basis` and `timing_ms` are extra, additive keys.

## Env/config changes

- Added keys:
  - orion-durable-runs: `ORION_PHYSIOLOGY_REPORT_ENABLED=true`, `ORION_PHYSIOLOGY_REPORT_CACHE_SEC=60.0`,
    `ORION_PHYSIOLOGY_REPORT_CONCURRENCY=3`, `ORION_PHYSIOLOGY_REPORT_GPU_POOL_PATH=/app/config/gpu_pool.yaml`.
  - orion-cortex-exec: `ORION_PHYSIOLOGY_REPORT_URL`, `ORION_PHYSIOLOGY_REPORT_TIMEOUT_SEC`.
  - orion-harness-governor: `ORION_PHYSIOLOGY_REPORT_URL`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes, for all three services.
- Local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. All 7 keys are present in
  the primary checkout's `.env` files.
- Skipped keys requiring operator action: none.

## Proposal-mode note (CLAUDE.md §0A: a new self-perception surface)

- **What capability changes.** Orion can read their own body economy on demand: limits, prices and their
  reasons, wants funded vs deferred, failed GPU time, queue waits, single decisions, Claude spend, value per
  cost, and services down. It is a read only. It changes no price, bid, gate or decision.
- **What data is touched (read-only):**
  - `physiology_state`, `want_ledger`, `gpu_pool_events`, `orion_energy_hourly`, `energy_run_cost`,
    `grammar_acts.cost_json`;
  - `dev_economics_ledger_log`: token counts only, no message text;
  - `orion_biometrics_summary`;
  - mesh-guardian rows in `notify_requests`.
  - Nothing is written. The dig kit writes only its own result file under `$ORION_DIG_HOME`.
- **Privacy boundary.**
  - The Claude section counts tokens from Juniper's Claude Code sessions on athena, and says so.
  - Per the standing rule there is no privacy boundary between Juniper and Orion. No message text is read.
- **What trace proves it worked:**
  - a dig result file `$ORION_DIG_HOME/dig/<day>/body-economy-<window>-N.json` in the governor;
  - a `Bash` tool_use of `orion.dig body-economy` in the harness transcripts;
  - durable-runs access logs for `GET /physiology/report`.
- **Dangerous failure mode.**
  - Orion could read a stale or partial report as the truth. For example: a `services.down` entry for a
    service that has since restarted (the guardian sends no recovery event), or a price read as a command.
  - Mitigations:
    - every section states its source and gaps;
    - the wants line says "shadow bids; nothing obeys them yet";
    - `services.basis` says a listed service may be up again.
- **How to disable it:**
  - `ORION_PHYSIOLOGY_REPORT_ENABLED=false` on orion-durable-runs. The route answers 503, and both the skill
    and the dig command tell Orion "the physiology report is disabled".
  - Remove the dig hint with `HARNESS_FCC_DIG_BRIEF_ENABLED=false`, which removes the whole dig line.

## Metric gate notes (sections that are new readings, not new metrics)

The report creates no new metric. It aggregates already-stored values. Two judgment calls:

- **`waste`**: lease-seconds of closes whose reason is not `ok`, `completed`, `generated`, `text_done` or
  `operator:*`. Recalls and retries count as waste, and the top rows show the reason. An expiry or abort
  writes two close rows for one lease, and the report counts that lease once. Live check: all 63 such
  `released` twins in 30 days had a matching `expired`/`aborted` row.
- **`rhythm.heavy`**: lease-seconds of `priority='background'` leases, spread over the hours each lease was
  held. This is the deferrable self-work class used by the S2 stopgap (#2873).

## Tests run

```text
services/orion-durable-runs: pytest tests/test_physiology_report.py tests/test_physiology_bid.py
  -> 17 passed (incl. test_report_sql_on_postgres against a throwaway postgres:16 container)
orion/physiology/tests (minimal deps: pytest pyyaml numpy pydantic) -> 169 passed
orion/dig/tests -> 40 passed
services/orion-cortex-exec: pytest tests/test_skill_verbs.py -> 47 passed
scripts/check_verb_physiology.py -> OK (107 verbs)
scripts/check_body_economy_contract.py -> OK (87 state-reading modules, all registered)
scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
PYTHONPATH=. python orion/physiology/evals/physiology_state_skill_eval.py -> passed
  scenarios: live 24h report (real 2026-10-29 data), all-empty report, failed lease-event reads,
  unreachable endpoint. Checks: every $ figure in the text is a value in the report; the headline names
  exactly the binding limits; every empty section shows its reason; overview <= 40 lines / 4000 chars;
  no None/nan/raw JSON; unreachable -> exit 1 with a plain reason. Mutation check: dropping reasons or
  changing a $ figure makes it fail.
```

## Docker/build/smoke checks

```text
Live DB timing (read-only, the real build_report on the live conjourney DB, concurrency 3, final code):
  24h 0.35 s, 7d 0.77 s, 30d 1.02 s   (H5: under 2 s)
  slowest reads at 30d: rhythm 0.94 s (per-hour spread), queue_wait 0.46 s, limits 0.43 s, waste 0.37 s.
The reviewer also ran it on the live DB inside a forced read-only session: 24h 245 ms, 30d 743 ms (before
the rhythm spread).
Network: orion-athena-harness-governor -> http://orion-athena-durable-runs:8121 reachable (0.018 s);
         orion-athena-cortex-exec -> same host reachable.
Not deployed (per task). The live route itself is UNVERIFIED until durable-runs is rebuilt.
```

## Review findings fixed

An independent review subagent read the full diff and ran the real SQL against the live DB in a read-only
session. It found no blockers, 8 should-fix findings and 9 nits. All the material ones are fixed:

- **Finding:** `family=` made queue_wait report a false claim. Most leases carry a holder name, not a want
  family, so the section came back empty with "no lease granted".
  - **Fix:** a new `family_source` field (`want_ledger` | `holder`). An empty filtered result now says
    "no granted lease is attributable to family X; only durable-run admissions carry a want family". A
    "partial" note appears when rows are named by holder. The wants and decisions reasons now say "for family X".
  - **Evidence:** `test_family_filter_says_which_sections_it_narrows`, `test_holder_named_queue_rows_are_marked`.
- **Finding:** most sections silently ignored the family filter.
  - **Fix:** a `*` missing entry names the whole-body sections. The text header names the filter, and
    value_per_cost now filters by family.
  - **Evidence:** the same test, plus `test_family_filter_is_named_in_the_header`.
- **Finding:** an old physiology tick was served as "now".
  - **Fix:** a tick older than 10 min adds a `stale: ... min old` reason to limits, money, claude and unit_costs.
  - **Evidence:** `test_stale_newest_tick_is_said_not_served_as_now`.
- **Finding:** the headline and lane summary counted "no price" as slack.
  - **Fix:** only `slack` counts as slack. No price and no reading are listed as "Unknown".
  - **Evidence:** `test_headline_never_counts_no_price_as_slack`.
- **Finding:** Claude showed 0 tokens when there was no data.
  - **Fix:** the token fields are now Optional and None when no row exists. The text says "token counts unknown".
  - **Evidence:** `test_no_token_rows_reads_unknown_not_zero`, `test_unknown_tokens_are_not_printed_as_zero`.
- **Finding:** an expiry or abort was counted twice in waste and rhythm, and `retried` never occurs.
  - **Fix:** the `released` twin with reason `expire:*`/`abort:*` is excluded, and `retried` is dropped.
  - **Evidence:** the Postgres test now inserts a twin pair and expects one 5 s count. Live check: 63/63 twins
    had a pair.
- **Finding:** rhythm put a whole 2.4 h hold into the hour it closed.
  - **Fix:** each lease is spread over the hours it was held (`generate_series`), clipped to the window.
  - **Evidence:** the Postgres test checks that an 8,500 s hold spans at least 3 hours, with no hour over 3,600 s.
- **Finding:** each request could take 3 of the 8 shared pool connections, the cache had no single flight,
  and it cached failed reads.
  - **Fix:**
    - one semaphore shared by all requests;
    - `ReportCache.get_or_build` gives single flight;
    - a report with a read or shaping failure is never cached.
  - **Evidence:** `test_cache_skips_failed_reports_and_shares_one_build`.
- **Nits fixed:**
  - partial days marked in `daily_kwh` (`partial`);
  - a shaping error becomes a `missing` reason, never a 500;
  - a helper-read failure is reported under each dependent section ("read failed (latest)");
  - vram names hosts with no fresh row, plus a "partial" note;
  - the services heading is now "reported stopped in the last 30 days";
  - waste says "ended without success", and operator benches are excluded;
  - the garbled queue-wait sentence is fixed;
  - the Claude session-count wording is fixed;
  - the cortex-exec text is capped at 40 lines / 4000 chars like the dig kit, and the undocumented
    `section` arg is dropped;
  - the cortex-exec adapter test runs in its own CI job, so its pins do not downgrade the dig job.
- **Not changed, on purpose:** `Market.family_weights` reuses the bid path's history cache, so a report
  read can warm that cache. That is the same read the bid path does. It gates nothing.

## Restart required

Deploy from the clean primary `main` checkout after merge:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

Smoke test after deploy:

```bash
curl -fsS 'http://localhost:8124/physiology/report?window=30d' | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['timing_ms'], d['missing'])"
docker exec orion-athena-harness-governor python -P -m orion.dig body-economy
```

## Risks / concerns

- Severity: low.
  - Concern: `gpu_pool_events` has no persisted hourly rollup. The waste, rhythm and queue-wait reads group
    raw events in SQL. Today 30 d takes ~1 s.
  - Mitigation: the 60 s cache and `timing_ms` per section. If 30 d grows past 2 s, add a rollup for that
    table.
- Severity: low.
  - Concern: `services.down` comes from guardian cards. There is no recovery event, so a restarted service
    stays listed until the 30-day lookback ends.
  - Mitigation: `basis` says this.
  - Follow-up: a guardian endpoint for the current stopped list.
- Severity: low.
  - Concern: `vram` and per-act `unit_costs` read fields that sibling agents are adding now
    (`gpuN_mem_*_mb`, `cost_json.wh_by_host` / `usd` / `heat_c_by_host`). Until those fields land, both say
    "not measured" with the reason.
  - The per-1k-token row is a TODO. It waits on the per-call token table, and its name is not on main yet.
- Severity: low.
  - Concern: the model can't call cortex-exec skill verbs from either live path, so the adapter is reachable
    only from Skill Runner and verb plans. Orion's live path is the dig command.
  - Follow-up: an orion-introspect MCP tool would give a typed tool call (the power_attribution recipe,
    about 29 files).

## Observed while testing (not changed here)

`hub.outreach` shadow funded decisions rose from 15 to 237 in the trailing 24 h within about 40 minutes on
2026-10-29 (~20:24 to ~21:02 UTC). Something is bidding outreach far more often than the daily cap allows.
This is a shadow decision, so nothing acted on it. It is worth a look by the body-economy owner.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2973

🤖 Generated with [Claude Code](https://claude.com/claude-code)
