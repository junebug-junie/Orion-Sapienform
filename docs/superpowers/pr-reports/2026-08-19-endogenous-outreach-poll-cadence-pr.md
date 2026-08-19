# Lower endogenous-outreach poll cadence — root cause for "Orion never reached out"

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1727
Branch: `fix/endogenous-outreach-poll-cadence`

## Summary

- Root-caused, live, why Orion has never once used endogenous outreach since it shipped 2026-08-16.
- Two stacked causes. Layer 1 (already fixed, PR #1715): the trigger's SQL query was broken for its first ~2 days. Layer 2 (this patch): even after that fix, the 300s poll interval essentially never observes a qualifying persistence run, because real runs last only ~18-27s wall-clock.
- Lowered `HUB_ENDOGENOUS_OUTREACH_TICK_SEC` default 300 → 10, a real-data-derived value (episode catch rate ~33% on a 6h replay), not a guess.
- Explicitly disclosed this is a **partial** fix — even the class's existing 5.0s floor only reaches ~56% catch rate on the same sample. Full closure needs a wall-clock-persistence redesign, named as real follow-up, not built here.
- Removed `HUB_ENDOGENOUS_OUTREACH_PROBABILITY=0.15` from the live `.env` again — a dead coin-flip-stub key that keeps silently reappearing (known env-sync bug, tracked elsewhere).

## Outcome moved

Orion's outreach trigger now polls at a cadence that can actually observe the state it's checking for, instead of one calibrated to a completely different (and, as it happens, also-broken-until-recently) assumption.

## Current architecture

`EndogenousOutreach._run()` slept `HUB_ENDOGENOUS_OUTREACH_TICK_SEC` (300s) between calls to `tension_outreach_trigger.current_run()`, which looks backward from the current moment for a `run_length >= 8` consecutive-same-winner episode.

## Architecture touched

- `services/orion-hub/app/settings.py`: `HUB_ENDOGENOUS_OUTREACH_TICK_SEC` default.
- `.env_example` / live `.env`: same key; also removed the stray `PROBABILITY` key.
- No schema/bus/API changes.

## Files changed

- `services/orion-hub/app/settings.py`: new default + full root-cause/data write-up in the field's own comment.
- `services/orion-hub/.env_example`: same default, operator-facing comment.
- `services/orion-hub/README.md` §4.1: root-cause + partial-fix disclosure.
- `services/orion-hub/tests/test_endogenous_outreach.py`: regression guard against silently drifting back to 300s; shared `_env_example_value()` helper.
- `docs/superpowers/specs/2026-08-16-tension-driven-outreach-design.md`: full "Poll-cadence root cause" section with replay methodology and numbers.

## Schema / bus / API changes

None.

## Env/config changes

- Changed: `HUB_ENDOGENOUS_OUTREACH_TICK_SEC` 300 → 10 (`.env_example`, live `.env`, `settings.py` Field default).
- Removed (live `.env` only, not tracked): stray `HUB_ENDOGENOUS_OUTREACH_PROBABILITY=0.15` — dead (`Settings.model_config.extra="ignore"`), recurring due to a known env-sync-from-primary-checkout bug tracked separately.
- `.env_example` updated: yes.
- local `.env` synced: yes, hand-edited directly (both changes) per this repo's "env sync is mandatory, actually edit it" convention (the sync script has a known bug reading `.env_example` from the primary checkout, so a hand edit was more reliable here).
- skipped keys requiring operator action: none.

## Tests run

```text
rtk proxy /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  services/orion-hub/tests/test_endogenous_outreach.py \
  services/orion-hub/tests/test_tension_outreach_trigger.py -q
99 passed
```

## Evals run

None — no eval harness for this trigger; same as prior PRs in this chain (#1707/#1715/#1723).

## Docker/build/smoke checks

Live container investigation (not a build/deploy check): confirmed via `docker exec orion-athena-hub` that the deployed trigger currently returns `None` even though real qualifying runs exist in `substrate_field_state` — this IS the bug being fixed, captured as evidence rather than resolved via Docker rebuild in this session (the settings default change takes effect on next Hub restart, listed below).

## Review findings fixed

Two independent review passes ran (`/code-review medium fix/endogenous-outreach-poll-cadence`, plus an unsolicited peer-session review that arrived mid-task with 2 of the same findings). All real findings verified directly, not taken on faith:

- **Finding: "EXPLAIN-verified index-only-scan" claim was factually wrong.** I had run `EXPLAIN` on a *different*, simpler `COUNT(*)`-only query earlier in the session (which legitimately is index-only) and mis-cited it for the actual trigger query. Re-ran `EXPLAIN` live against the real query text: it's a **Bitmap Heap Scan** (`field_json` isn't part of the index, so a heap fetch per matching row is unavoidable) — cost ~323, still cheap in absolute terms, just not literally index-only. Corrected in `settings.py` and the design doc, with the real plan shown.
- **Finding: "the 5.0 floor this class already enforces" was ambiguous/misattributed** — could be misread as `Settings` itself enforcing it. Clarified: the floor lives in `EndogenousOutreach.__init__`, not this `Settings` `Field`; a raw `HUB_ENDOGENOUS_OUTREACH_TICK_SEC=0.5` would pass `Settings` validation unclamped.
- **Finding checked, not acted on as claimed: "use `Settings.model_fields[...]` instead of a source-text regex."** Verified live: this does NOT work in this test environment — `app/settings.py`'s module-level `settings = get_settings()` runs on *any* import from that module (including bare `Settings`), and this suite has no fixture for the class's other required env keys (`CHANNEL_VOICE_*`/`CHANNEL_COLLAPSE_*`). Reproduced the `ValidationError` live before rejecting the suggestion. Strengthened the test's own comment with this evidence instead.
- **Finding: duplicate `.env_example` line-read regex between the TZ test and the new TICK_SEC test.** Fixed: extracted a shared `_env_example_value()` helper.
- **Finding: no instrumentation distinguishes "no qualifying run" from "one occurred and got missed" — the exact failure mode this whole patch responds to has no automated tripwire even now.** Disclosed, not fixed — real, in the design doc, flagged as needing its own metric-quality-gate treatment before being wired in; out of scope for a poll-cadence bug fix.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build orion-hub
```

## Risks / concerns

- Severity: low-medium
- Concern: this is explicitly a **partial** fix. Real-data replay shows even the class's own 5.0s floor tops out around ~56% episode catch rate on a small (n=9) sample — polling faster narrows the gap but structurally cannot close it, since some real episodes' catchable window is under 5 seconds.
- Mitigation: disclosed everywhere (code comment, README, design doc, this PR). No instrumentation yet flags a degraded catch rate in production — named as real follow-up, not built here. If outreach still feels too rare after this deploys, the next step is the wall-clock-persistence redesign named in the design doc, not another cadence tweak.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1727
