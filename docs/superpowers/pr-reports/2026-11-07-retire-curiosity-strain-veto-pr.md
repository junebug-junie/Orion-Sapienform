## Summary

- Hub no longer holds the curiosity investigation line when arousal reads `strained`. The run's body-economy bid in durable-runs decides instead (`ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=on`, #3010). No flag brings the old hold back.
- Kept: the engaged cooldown stretch (investigation line) and self-inquiry's idle-only hold, which still holds on engaged and strained, because `durable.self_inquiry` bids do not bind yet.
- Orion's strain report no longer says "curiosity investigations wait".
- Contract, registry, README, settings, `.env_example` comments, and the reader replay eval now describe the new rule. The body economy design doc records the retirement with its evidence.

## Outcome moved

The first Patch D cutover is now complete: one of the strained vetoes is gone, and its reader is a bid. Before this patch, curiosity investigations were gated twice (Hub veto, then the bid).

## Current architecture

`scheduling_block_reason` (`services/orion-hub/scripts/curiosity_investigation.py`) returned `arousal_strained` for any line once every pacing gate had passed. Durable-runs also ran a binding bid on `durable.curiosity_investigate` since 10-30/31.

## Architecture touched

- Hub curiosity scheduling gate (investigation line only).
- Hub strain report text (`STRAIN_HOLDS`).
- `config/physiology/contract.yaml` row and `orion/inner_state_registry.py` consumer text (the module still reads arousal for the engaged and idle-only gates, so it stays registered).

## Files changed

- `services/orion-hub/scripts/curiosity_investigation.py`: strained branch scoped to the idle-only (self-inquiry) line only.
- `services/orion-hub/scripts/strain_report.py`: dropped "curiosity investigations wait".
- `services/orion-hub/tests/test_regulation_arousal_readers.py`: regression tests (strained passes the investigation line; strained reads exactly like unknown across a pacing grid; engaged and idle-only still hold).
- `services/orion-hub/tests/test_strain_report_outreach.py`: the report must not claim curiosity waits.
- `scripts/analysis/measure_arousal_reader_replay.py` + test: the replay's curiosity rule mirrors the shipped gate.
- `services/orion-hub/README.md`, `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example` (comment only), `config/physiology/contract.yaml`, `orion/inner_state_registry.py`: text.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: dated retirement line.

## Schema / bus / API changes

- Added: none
- Removed: Hub investigation-line block reason `arousal_strained` (the reason constant stays; self-inquiry, outreach and others still use it).
- Renamed: none
- Behavior changed: a strained arousal no longer blocks a scheduled curiosity investigation tick in Hub.
- Compatibility notes: none.

## Env/config changes

- Added/removed/renamed keys: none. `HUB_CURIOSITY_AROUSAL_ENABLED` now gates only the engaged cooldown stretch (comment updated).
- `.env_example` updated: comment only.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (no key changes).
- skipped keys requiring operator action: none.

## Live evidence (read-only)

- Replacement bid, 10-31 to 11-06 (`durable_resource_events`): 49 `run.economy_admitted` (17 `funded`/net_positive, 32 `queued_at_pool`/capacity_reserved), 0 `run.economy_held`.
- Flag live: `docker exec orion-athena-durable-runs env` shows `ORION_PHYSIOLOGY_BID_CURIOSITY_INVESTIGATE=on`.
- How often the Hub veto blocked curiosity over 14 days: **not recoverable directly.** Hub logs the reason only as `curiosity_investigation_blocked reason=...` in the container log (journald), and journald retains only from 2026-11-06 06:49 UTC (3.9 GB cap). In that ~21 h: 0 `arousal_strained` blocks (174 daily_cap, 28 cooldown, 28 arousal_engaged_cooldown). Proxy over 14 days from `temporal_self_event` `arousal_transition` rows: strained for 41.5 of 336 h (133 transitions); 2 of 94 curiosity runs started inside a strained window.
- Caveat (review): "0 held" was measured while the Hub veto still filtered strained periods, so it does not prove the bid holds under strain. What the market prices instead is the heat constraint and lane (GPU queue) prices. Post-deploy check: in the next strained window, count investigation runs started vs `run.economy_held`.

## Tests run

```text
services/orion-hub: tests/test_regulation_arousal_readers.py tests/test_regulation_must_not_readers.py
  tests/test_strain_report_outreach.py tests/test_curiosity_self_inquiry.py tests/test_curiosity_investigation.py
  tests/test_rest_drive_readers.py -> 330 passed (149 re-run after review fixes)
scripts/analysis/tests/test_measure_arousal_reader_replay.py tests/test_check_body_economy_contract.py -> 42 passed
orion/body_atlas/tests/test_claims.py tests/test_inner_state*.py tests/test_regulation*.py -> 184 passed
scripts/check_body_economy_contract.py -> OK (102 modules registered)
scripts/check_env_template_parity.py -> PASS
```

## Evals run

```text
scripts/analysis/measure_arousal_reader_replay.py unit tests (rule updated). Reviewer also ran
services/orion-hub/evals/test_strain_report_replay_eval -> pass.
```

## Docker/build/smoke checks

```text
None. No dependency, port, or compose change. Live behavior UNVERIFIED until Hub restarts.
```

## Review findings fixed

- Finding: design-doc evidence overclaimed ("0 held" measured while the Hub veto was still filtering strain).
  - Fix: caveat sentence + post-deploy check in the doc and this report.
  - Evidence: design doc retirement line.
- Finding: source-scan guard test was weaker than its name.
  - Fix: replaced with a behavioral grid test: strained == unknown on the investigation line.
  - Evidence: `test_strained_reads_like_no_arousal_on_the_investigation_line` passes.
- Finding: replay now understates pre-11-07 holds.
  - Fix: one-line note in `measure_arousal_reader_replay.py`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Hub only. Nothing else changes at runtime.

## Risks / concerns

- Severity: low. Concern: under strain, investigation runs now reach durable-runs; if the bid does not price strain's causes well, more runs start during heat/queue strain. Mitigation: the bid binds on heat and lane prices; post-deploy check above.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3159

🤖 Generated with [Claude Code](https://claude.com/claude-code)
