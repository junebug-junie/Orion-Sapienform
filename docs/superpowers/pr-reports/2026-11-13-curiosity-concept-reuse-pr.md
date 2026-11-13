# Curiosity runs see and reuse the concepts Orion already coined

## Summary

- Each curiosity run now sees a list of the concepts Orion already named in their own graph (`orion_worldview`). The list holds 15 concepts: the 10 most linked, then the 5 newest of the rest, so a new concept with one link still comes back. Each one shows its full `concept_id`, its name, and a note cut to 200 characters.
- The concept instruction changed. Orion first looks for an existing concept and binds it with `MATCH` on its `concept_id`. Orion makes a new one only when nothing fits, with `MERGE` on `concept_id`. The old instruction was a bare `CREATE`.
- A new example shows how to join a finding or prior to a concept with an `ABOUT` edge. The "connect it in the same breath" rule stays.
- When the graph cannot be read, the prompt says so. It never says "no concepts" in that case.
- This is step 2a of the core-memory-pages spec (PR #3319), "Decision per source", item 1.

## Outcome moved

Before: the prompt never showed a concept back, so every run started from nothing and coined new ones. Most concepts got one link and were never touched again.

Acceptance metric: concepts with 2 or more links.

- Baseline at 2026-11-13 when the task was written: 9 of 33.
- Baseline when this PR was built (same day, live read-only query): **9 of 35**. Two more concepts were coined in between, both with fewer than 2 links.
- Re-check after deploy: owned by the orchestrator, not done here. UNVERIFIED until then.

Query used for the baseline:

```text
MATCH (c:Concept) OPTIONAL MATCH (c)-[r]-() WITH c, count(r) AS deg
RETURN count(c), sum(CASE WHEN deg>=2 THEN 1 ELSE 0 END)
```

## Current architecture

- `orion/curiosity/worldview.py` reads Orion's graph read-only (`GRAPH.RO_QUERY`). `read_snapshot()` read priors, prior counts, the concept COUNT, settled priors and recent runs. It never read the concepts themselves.
- `orion/curiosity/kickoff_prompt.py` showed priors back with their full ids. For concepts it only gave a `CREATE (:Concept {...})` template.

## Architecture touched

- One more read-only query in `read_snapshot()`, inside the same try block. So a failed read makes the whole snapshot unavailable, the same as every other query in it.
- One new prompt section, placed right after the priors.
- No bus, schema, env, or service change. Hub picks it up on its next rebuild.

## Files changed

- `orion/curiosity/worldview.py`: `ConceptRecord`, `CONCEPTS_CYPHER`, `build_concepts()`, the `concepts` field on `WorldviewSnapshot`, and the read in `read_snapshot()`.
- `orion/curiosity/kickoff_prompt.py`: `_concepts_section()`, and the concept write instruction now says MATCH first, MERGE second.
- `orion/situational/context.py`: passes `read_concepts=False`, since the situation never shows concepts.
- `services/orion-cortex-exec/tests/test_situation_curiosity_reverie_context.py`: asserts that flag.
- `tests/test_curiosity_concept_reuse.py`: new tests.
- `docs/superpowers/pr-reports/2026-11-13-curiosity-concept-reuse-pr.md`: this report.

## Schema / bus / API changes

- Added: `WorldviewSnapshot.concepts` (in-process dataclass field, defaults to empty). No bus or registry schema.
- Removed: none.
- Renamed: none.
- Behavior changed: the kickoff prompt text. Orion is now asked to `MERGE` a new concept instead of `CREATE`.
- Compatibility notes: `read_snapshot()` gained `read_concepts: bool = True`. The situation context passes False. The Hub curiosity loop pays one more read of about 40 rows (about 11-15 ms live).

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, no template changed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest -q tests/test_curiosity_concept_reuse.py tests/test_curiosity_worldview.py tests/test_curiosity_peer_kickoff.py \
  tests/test_curiosity_supervisor.py tests/test_investigation_role_worldview.py tests/test_curiosity_hop_identity.py \
  tests/test_curiosity_unresolved_percepts.py tests/test_dream_hypotheses.py tests/test_curiosity_study_material.py \
  tests/test_curiosity_incident_report.py tests/test_curiosity_peer_patch0_acceptance.py orion/curiosity/tests \
  orion/orion_day/tests/test_corrections.py
  -> 616 passed, 29 skipped
(services/orion-hub) pytest -q tests/test_curiosity_*.py tests/test_orion_day_correction_listener.py evals/test_curiosity_focus_eval.py
  -> 628 passed, 1 skipped
(services/orion-cortex-exec) pytest -q tests/test_situation_curiosity_reverie_context.py -> 24 passed
PYTHONPATH=services/orion-curiosity-peer:. pytest -q services/orion-curiosity-peer/tests -> 79 passed
```

## Evals run

```text
services/orion-hub/evals/run_curiosity_focus_eval.py        -> PASS
services/orion-hub/evals/run_orion_day_correction_eval.py   -> PASS
services/orion-curiosity-peer/evals/run_contractor_peer_eval.py -> 4/4 passed
No eval measures concept reuse itself. The live measure is the acceptance metric above (2+ links), re-checked after deploy.
```

## Docker/build/smoke checks

```text
Live read-only check against orion-athena-falkordb (port 6380), from this worktree's code:
  CONCEPTS_CYPHER returned 40 rows (35 concepts; some runs have 2 TurnOutcomes).
  build_concepts() returned 15: the 9 concepts with 2+ links, then 6 recent one-link or zero-link concepts.
  The rendered concepts section is about 5,000 characters.
No Docker build: the change is pure Python inside the Hub image, no dependency or config change.
```

## Review findings fixed

Review ran in a subagent on commit 41439d88f. No must-fix findings. Fixed:

- Finding: ranking by links alone keeps the 9 well-linked concepts in place forever, so the one-link concepts this patch exists for never come back.
  - Fix: 5 of the 15 slots go to the newest concepts not already shown (`CONCEPT_FRESH_SLOTS`).
  - Evidence: `test_a_fresh_one_link_concept_is_shown_even_when_many_are_better_linked`; live list now holds 6 recent low-link concepts.
- Finding: the id-rules paragraph said Atlas concept ids never go in a MATCH, which could read as "concept ids never go in a MATCH".
  - Fix: the paragraph now says your own `concept_id:` lines DO go in a MATCH; the crystallization line says "memory crystallization".
  - Evidence: asserts in `test_write_instruction_reuses_before_it_creates`.
- Finding: the bare `MATCH (c:Concept ...)` template has no RETURN, so FalkorDB rejects it if pasted.
  - Fix: it now ends with `RETURN c.name, c.note`.
  - Evidence: same test.
- Finding: `_stamp_ms("inf")` raised OverflowError outside the try in `read_snapshot`, which would blank the worldview on one bad hand-written timestamp.
  - Fix: `_stamp_ms` catches OverflowError.
  - Evidence: `test_a_hand_written_infinite_timestamp_does_not_crash_the_read`.
- Finding: a MERGE that binds an existing concept writes nothing and the footprint shows no Concept; Orion may coin a new id to "fix" that.
  - Fix: one prompt line says that is correct and to draw the edge instead.
- Finding (nits, also fixed): unreadable-concepts message no longer claims a cause; a warning logs when the row limit is hit; forked concept ids are logged; a Prior ABOUT example was added with the arrow direction; a read-only prompt no longer asks for writes; the situation context (`orion/situational/context.py`) skips the concept read with `read_concepts=False`; the cypher column test is exact.
- Not changed: the self-inquiry prompt also reads concepts but does not render them. It shares one read helper with the main loop, so skipping it would mean another parameter path for about 11 ms per run.

## Restart required

Hub needs a rebuild to pick this up. That happens through the normal post-pull rebuild from main after merge. Not deployed by this PR.

## Risks / concerns

- Severity: low. Concern: the prompt grows by about 5,000 characters. Mitigation: capped at 15 concepts and 200 characters per note. Curiosity runs on the leased agent seat, not on the fast lanes.
- Severity: low. Concern: Orion may still coin a near-copy. Mitigation: the acceptance metric (concepts with 2+ links) will show it. The re-check is after deploy.
- Issues opened: none. Issues closed: none.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
