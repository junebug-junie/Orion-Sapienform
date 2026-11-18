## Summary

Orion asked 7 questions on #3698 (session thread cf5d4218097a). This PR fixes the 3 small ones and files issues for the rest.

- A spec that Hub cuts now ends on a line break. Its last line names the section the cut fell in and the sections left out. It also says where the full file is (`git show origin/<branch>:<path>`). Before, Orion got only "40000 of 64564 characters kept", cut mid-sentence.
- Each load of the mesh-edge guard logs its size and load time, so growth of the guarded set is visible.
- Claude's copy of a redacted consult opens with one line: how many passages were withheld and what the marker means. A consult with more than half its words withheld now stops unsent.

## Outcome moved

- Orion can see where a spec was cut and how to read the rest.
- Guard growth can be read from the logs. It was invisible before.
- Claude no longer gets bare blanks with no explanation. A brief that redaction has mostly emptied no longer goes out.

## Current architecture

- `orion/session_threads/core.py::cap_text` cut spec text at 40,000 chars at intake (`session_roundup_loop.py`) and added a counts-only line.
- `orion/memory/mesh_edge.py::load_guard` loaded the guard every 5 minutes with no size trace. It already deduped by normalized sentence.
- `services/orion-hub/scripts/consult_loop.py::_mesh_edge` replaced matching sentences with `[withheld: private memory]` and told only Hub chat the count.

## Architecture touched

Orion-hub (session-thread intake, consult loop) and the shared `orion/memory/mesh_edge.py`. No bus, schema or env change.

## Files changed

- `orion/session_threads/core.py`: adds `cap_spec`, a heading-aware and fence-aware cut line.
- `services/orion-hub/scripts/session_roundup_loop.py`: intake uses `cap_spec`, and the receipt wording is changed to match.
- `orion/memory/mesh_edge.py`: adds `source_rows` and `source_sentences`, plus `mesh_edge_guard_loaded` and `mesh_edge_guard_load_failed` log lines.
- `services/orion-hub/scripts/consult_loop.py`: adds the redaction note, `withheld_fraction`, and the `mesh_edge_mostly_withheld` stop (threshold 0.5).
- Tests: `services/orion-hub/tests/test_session_threads_core.py`, `test_session_roundup_loop.py`, `test_consult_loop.py`, `orion/memory/tests/test_mesh_edge.py`.

## Schema / bus / API changes

- Added: the consult `status_reason` value `mesh_edge_mostly_withheld`, a free-text column. No consumer matches on it.
- Removed / Renamed: none.
- Behavior changed: as listed in the Summary.
- Compatibility notes: none.

## Env/config changes

None. The threshold is a module constant (`MESH_EDGE_MAX_WITHHELD_FRACTION`).

## Tests run

```text
services/orion-hub: test_session_threads_core.py test_session_roundup_loop.py test_session_thread_routes.py test_consult_loop.py -> 148 passed, 28 skipped (DB-gated roundup tests; no ORION_MEMORY_EPISODE_TEST_DATABASE_URL)
orion/memory: tests/ -> 2058 passed
scripts/check_integration_seam.py OK; check_personal_terms OK; check_async_routes_not_blocking OK
Pre-existing on main (not this PR): tests/test_attention_loops_ui_smoke.py (2 fail on main too)
```

## Evals run

There is no eval harness for the cut line or the redaction note. Measured by hand instead:
- This spec (64,878 chars) with `cap_spec` at 40,000: the line names "Corrections rev 2" and lists 16 left-out sections.
- The live guard on 11-18: 1,348 rows, 1,983 sentences, 1,808 phrases, 13,588 shingles. The build takes 0.086 s, a 20k-char check 24 ms, and the SQL about 7 ms.

## Docker/build/smoke checks

Not run. This PR is not deployed.

## Review findings fixed

- Finding: a cut inside a code fence left Hub's note inside the open fence.
  - Fix: the fence is closed before the note, and a fence closes only on the marker that opened it.
  - Evidence: `test_a_cut_inside_a_code_fence_closes_the_fence_before_hubs_note`.
- Finding: the receipt said "cut to 40000 characters", but the cut is now on a line break.
  - Fix: the receipt now says "at most".
  - Evidence: the updated receipt assertion in `test_session_roundup_loop.py`.
- Finding: the generic "[Hub: " note prefix could strip an unrelated first paragraph on a resend.
  - Fix: the prefix is now `[Hub mesh-edge note: `.
- Not wired (review finding 1): the round-up budget cut still uses `cap_text`. Its 400-char slack in `_cut_to_fit` would not fit the longer note. This is left as it was on purpose.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build   # from clean primary main after merge
```

## Risks / concerns

- Severity: low. Concern: the 50% stop threshold is a judgment. No consult has run since 11-12, so no redacted consult exists to tune it on. Mitigation: the `withheld=` value is logged on every redaction.
- Severity: low. Concern: on a resend, `withheld_fraction` is measured against the already-redacted copy. If the guard grew between sends, the total withheld share can pass 50% without tripping the stop. Mitigation: a resend happens once at most (`MAX_CLAUDE_RESENDS = 1`).
- Severity: low. Concern: while the database is down, `mesh_edge_guard_load_failed` logs once per load attempt. Mitigation: none needed. It is the visible trace of the fail-closed hold.
- Issues opened: #3714 (supersede vote diversity, design), #3715 (supersede eval gate closed silently, follow-up), #3716 (recall caller identity, follow-up). Commented: #3661 (pending rows are served unlabelled). Closed: none.

UNVERIFIED live: all three changes. Hub is not rebuilt.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
