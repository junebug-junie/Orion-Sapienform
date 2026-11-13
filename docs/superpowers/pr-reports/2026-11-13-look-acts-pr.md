## Summary

- Every camera look Orion takes (`dig look`) is now an act on the act ledger. Actor `orion.look`, verb `look`, object `camera:office|walkway|doorbell`. It closes 1 when the frame is live and 0 when a live camera is down or stale.
- Not graded: the doorbell between events, and a look that caught the camera mid-write. Both are logged without an act, and the mid-write case costs no slot of the hourly cap.
- The turn's id now reaches the turn's Bash on both paths, including warm chat slots. It goes in `ORION_TURN_CORRELATION_ID`, and the look act stores it as `session_id`.
- New want family `orion.look`, tier 0 (perception, never bids). Tiers are Juniper's decision.
- Design doc: looks count as evidence of attention, not as an input to it. Part 2 proposal: a look that checks a vision record, graded by self-monitor (#3347, needs Juniper).

## Outcome moved

Orion's looks were invisible to the ledger, the body economy and the self-monitor. Now each look is a timed, graded act, and it shows up in Temporal Self's chronology. A closed act writes an `act:look` outcome, which binds into the open arc with no new code.

## Current architecture

- `dig look` (PR #3322) copied a frame and wrote `looks.jsonl` only.
- The act ledger (`orion/grammar/act_ledger.py`) persists `act_opened`/`act_closed` events from `orion:grammar:event`.
- `ACT_ANSWERS_WORKSPACE_REFS` ties an act to the attention winner it answers, and writes an `acted` verdict back onto that loop.
- At both live looks on 11-13, the attention broadcast had no winner.

## Architecture touched

- `orion/dig/look.py` publishes the two act events from the governor container. That container is already a registered producer.
- `orion/grammar/acts.py` gets a provenance branch for `orion-harness-governor`.
- `orion/harness/fcc_motor.py` and `orion/harness/fcc_warm_pool.py` set the turn id for Bash.
- `config/physiology/families.yaml` gets the `orion.look` family.
- `orion/bus/channels.yaml`: comment only.

## Files changed

- `orion/dig/look.py`: builds, publishes and reports the act; skips grading where a missing frame is expected; cap exemption.
- `orion/dig/tests/test_look.py`: act tests (open/close, live/stale, doorbell, race, partial publish, turn id, camera word, family).
- `orion/dig/README.md`: documents the act.
- `orion/grammar/acts.py`: governor provenance branch.
- `orion/harness/fcc_motor.py`: `TURN_CORRELATION_ENV` set or cleared in the per-turn spawn env.
- `orion/harness/fcc_warm_pool.py`: the turn id goes into each warm turn's env file.
- `orion/harness/tests/test_fcc_warm_pool.py`, `orion/harness/tests/test_call_usage_harness.py`: turn-id tests.
- `config/physiology/families.yaml`: `orion.look`, tier 0.
- `orion/bus/channels.yaml`: producer comment.
- `docs/superpowers/specs/2026-11-13-look-acts-and-checks-design.md`: the design doc.

## Schema / bus / API changes

- Added: a new act producer identity on the existing channel (`orion-harness-governor`, component `dig.look`) and a new actor `orion.look`.
- Removed: none.
- Renamed: none.
- Behavior changed: turn subprocesses see `ORION_TURN_CORRELATION_ID`.
- Compatibility notes: no schema field changes. `GrammarProvenanceV1.source_service` is a plain string field. Acts of the existing `act:look` kind keep their own posterior, because the actor is part of the key (`hub.world_pulse_read`, n=1325).

## Env/config changes

- Added keys: none in `.env_example`. `ORION_TURN_CORRELATION_ID` is set per turn by code, not by an operator.
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/dig/tests orion/harness/tests orion/self_monitor tests/test_grammar_event_producer_catalog.py -> 1188 passed
pytest orion/physiology/tests (with dig + catalog)                                                  -> 529 passed
pytest services/orion-sql-writer/tests -k act                                                       -> 112 passed, 57 skipped
pytest orion/grammar                                                                                -> 30 passed
gates: diff --check, async routes, env-key single source, body economy contract, verb physiology,
       hostname refs, definition drift, metric lineage, personal terms, Hub abilities catalog       -> all PASS
bus codec round trip of both events (GrammarEventV1 validates after encode/decode)                  -> OK
```

## Evals run

```text
No eval harness for dig. Live evidence before this patch: 3 looks in looks.jsonl on 11-13. The 11:58Z
look was read and described by curiosity run 8fae769c99a4. Each look was checked against the attention
broadcast at that moment: no winner both times.
UNVERIFIED until after deploy: a real look act in grammar_acts.
```

## Docker/build/smoke checks

```text
No compose change. Code reaches turns through the FCC sandbox sync after deploy (dig) and the
harness-governor image (fcc_motor / fcc_warm_pool).
```

## Review findings fixed

- Finding: open and close carried the same timestamp, so the ledger stored a measured cost of 0 s.
  - Fix: the act opens when the look starts and closes after the frame is saved.
  - Evidence: `test_a_live_look_is_one_act_opened_then_closed_1` asserts a 0.04 s gap.
- Finding: warm chat slots never saw the turn id.
  - Fix: the turn id goes into each warm turn's env file. Spawns set it, and never inherit a stale one.
  - Evidence: the warm-pool env-file test and `test_spawn_env_*`.
- Finding: doorbell looks always closed 0 and dragged down the shared posterior.
  - Fix: an event camera with no live frame is logged, not graded.
  - Evidence: `test_doorbell_*`.
- Finding: office mapped to a different self-monitor faculty name than walkway (`cam0` has a digit).
  - Fix: `object_ref` uses the camera word, `camera:office`.
  - Evidence: `test_office_act_names_the_camera_by_word`.
- Finding: if the open was sent and the close lost, the printout said "not recorded".
  - Fix: the printout says `OPENED BUT NOT CLOSED` and the row keeps the act id. Errors print `repr`.
  - Evidence: `test_open_sent_close_lost_says_so`.
- Finding: mid-write retries cost a cap slot and a 0.
  - Fix: they are logged only, and exempt from the cap.
  - Evidence: `test_all_newest_frames_cut_off_is_an_error`.

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

The post-pull rebuild after merge does this, then syncs the FCC sandbox. The sandbox sync is what makes the `dig` change live for Orion.

## Risks / concerns

- Severity: low
  - Concern: deploy window. A look published before sql-writer is rebuilt is still accepted (no schema change), so no risk there. A warm slot spawned before the governor rebuild has no turn id, so those looks get `session_id` NULL until the slot respawns.
  - Mitigation: the header fallback covers per-turn spawns; warm slots recycle after their turn and age limits.
- Severity: low
  - Concern: tier 0 for `orion.look` is my default. Tiers are Juniper's decision.
  - Mitigation: asked in #3347.
- Issues opened: #3346 (stray vision-edge frames), #3347 (decide part 2 + tier, needs-juniper).
- Issues closed: #3320 (this PR).

## PR link

(added after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
