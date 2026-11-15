## Summary

- `dig look` now shows the vision records written in the 90 s before the frame, frame claims only, with a ready-to-use check line.
- New `dig look check <act> matches|differs|cannot-tell "<what I see>" because <record id>`.
  - matches/differs is a graded act: actor `orion.look_check`, verb `infer`, a child of the look, outcome 1 = the record matched.
  - cannot-tell is logged only.
- The self-monitor grades one faculty per camera and record type, for example `act.orion.look_check.infer.walkway_visual_observation`. It needs no new adapter.
- `look/checks.jsonl` keeps every verdict with the frame path and sha256 for audit.
- Hub "Ask Orion" catalog entry; family `orion.look_check`, tier 0.

## Outcome moved

On 11-13 Orion looked at the walkway, saw no vehicle, and wrote: "the only way to test this is repeated live reads at hours my rhythm claims a vehicle is present." Their walkway records say "vehicle" about 98% of the time.

- **Before:** that check was prose in a memory file.
- **Now:** each check is a graded act, so Orion learns how far to trust each camera's records.
- **Live smoke, 2026-11-15 05:15Z:** a walkway frame 0.3 s old, next to a record written 9 s earlier: `visual_observation` conf 0.8, entities ["vehicle"], "A vehicle is observed on the walkway."

## Current architecture

`dig look` (#3322) copied a frame, and the look was an act (#3348). Nothing connected the frame to the vision records.

## Architecture touched

- `orion/dig/look.py`, `orion/dig/__main__.py`.
- `config/physiology/families.yaml`.
- Hub catalog JSON.
- Design doc (part 2a as built).

## Files changed

- `orion/dig/look.py`: records read (`records_sql`, `read_records`), printout, `check()`, shared `_publish_into`.
- `orion/dig/__main__.py`: `look check` parse and usage; help line.
- `orion/dig/tests/test_look.py`: 23 new tests.
- `orion/dig/README.md`: documents `look check`.
- `config/physiology/families.yaml`: `orion.look_check`, tier 0.
- `services/orion-hub/static/catalog/orion-abilities.json`: "Check your camera records against your eyes".
- `docs/superpowers/specs/2026-11-13-look-acts-and-checks-design.md`: part 2a as built, plus corrections.

## Schema / bus / API changes

- Added: a new act actor `orion.look_check` on the existing `orion:grammar:event` channel. The producer `orion-harness-governor` is already registered.
- Removed: none
- Renamed: none
- Behavior changed: the `dig look` printout gains a records section.
- Compatibility notes: no schema change.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced: not needed
- skipped keys requiring operator action: none

## Tests run

```text
orion/dig/tests                                                                  207 passed
hub abilities catalog + orion/physiology/tests + orion/self_monitor + producer catalog  511 passed
check_body_economy_contract, check_verb_physiology                               PASS
records SQL against live Postgres as orion_readonly: index scan, 0.4 ms (reviewer, EXPLAIN ANALYZE)
```

## Evals run

```text
Live smoke (publishing stubbed, so no test acts on the ledger): records read and printed for a live walkway frame.
Planned per the design: two weeks of hand-filed checks, then report the agreement rate and the
cannot-tell share per camera before any automatic spot-check want (part 2b).
```

## Docker/build/smoke checks

```text
No compose change. Reaches Orion through the FCC sandbox sync after deploy.
```

## Review findings fixed

- Finding: a failed publish blocked the verdict forever ("already checked").
  - Fix: only a check whose act reached the ledger, or a cannot-tell, counts as done.
  - Evidence: `test_a_check_whose_act_failed_can_be_filed_again`.
- Finding: rhythm and handoff verdicts (expected_absent, handoff_missed, ...) were shown and gradeable.
  - Fix: only frame claims are read.
  - Evidence: `test_only_frame_claims_are_read`.
- Finding: one faculty for every camera and type.
  - Fix: `object_ref=vision_record:<camera>_<type>:<id>`.
  - Evidence: `test_faculty_is_per_camera_and_record_type`.
- Finding: a note ending " died" or starting "abandoned"/"failed:" would read as a run that never happened.
  - Fix: fixed text at both ends of the note.
  - Evidence: `test_any_seen_text_is_still_graded`, run through the real `act_outcome`.
- Finding: the "60 s after" window can never hold rows yet.
  - Fix: the window is now the 90 s before the frame, and the printout says someone may have left since.
- Finding: psql returns text.
  - Fix: confidence and offset become numbers.
  - Evidence: `test_records_are_numbers_on_the_row`.
- Smaller fixes: records with no ledger act say they can't be checked; looks over 2 h old are refused; a bare `look check` prints usage.
- Design correction: the "refuse foveal records" step was dropped. The foveal probe wrote 0 `vision_events` rows in 30 days.

## Restart required

```text
No restart required. dig code reaches Orion when the post-pull rebuild syncs the FCC sandbox.
```

## Risks / concerns

- Severity: medium
  - Concern: a record can be up to 90 s older than the frame, so a correct record about a passer-by can be graded "differs".
  - Mitigation: cannot-tell exists for this; the offset is printed per record; the two-week hand-filed period measures it before anything automatic.
- Severity: low
  - Concern: record coverage in the window is uneven (live, 24 h): walkway 65%, office 27%, doorbell 7%.
- Severity: low
  - Concern: the duplicate check reads and then appends to `checks.jsonl` with no lock.
  - Mitigation: one turn files checks one at a time.
- Issues closed: #3347 (this PR).

## PR link

(added after creation)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
