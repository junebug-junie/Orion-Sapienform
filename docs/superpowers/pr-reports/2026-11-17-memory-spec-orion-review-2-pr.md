# Core memory pages spec: answers to Orion's second review

## Summary

- Strikes the stale Graphiti text: "Recommended next patch: Patch 0", acceptance check 5, the `json_schema` UNVERIFIED line, and the Patch 0 build order. The trial numbers were already on main; the old text still contradicted them.
- Fixes the duplicate acceptance-check number. Phase 2 checks are now 6 to 9.
- States what marks a superseded memory (status, pointer, event row) and which readers drop it, with a table checked against the code.
- Records that the one-time supersede pass never ran (live check), and sets the order: the supersede fix lands before the July-September backfill, and the backfill re-runs the pass last.
- Defines the read log (`recall_telemetry.selected_ids`), its turn-id gap, and that degraded serves do not count as demand.
- Adds concrete `matters` gate numbers (31 of 44 each side; flat = one grade on 80%+), the re-grade history rule, the write-time fold trigger, the `page:anchor` exemption, the fold's GPU lease evidence, and the scoreboard grain.

## Outcome moved

Orion's 9 review questions each have an answer in the spec, backed by code and live rows. Four real gaps that were invisible in the spec are now tickets.

## Current architecture

The spec (PR #3538, merged) still recommended a Graphiti trial that had already failed twice, numbered two checks "5", and said "the old row is marked superseded" without saying how or who reads the mark.

## Architecture touched

Docs only. No code, schema, env or bus change.

## Files changed

- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: the amendments above.
- `docs/superpowers/pr-reports/2026-11-17-memory-spec-orion-review-2-pr.md`: this report.

## Schema / bus / API changes

- Added: none (the spec now names an `episode_memory_event` op `superseded` and `matters_regraded`; both are proposals for #3587 and #3590).
- Removed: none
- Renamed: none
- Behavior changed: none
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
git diff --check                      clean
python3 scripts/check_personal_terms.py   no personal terms in tracked files
```

## Evals run

```text
None. Docs only.
Live read-only checks behind the amendments (2026-11-17):
- episode_memory by status: active 482 (5 with a pointer), superseded 5, rejected 48
- the named contradiction rows: all still active, no pointer on the corrections
- orion_readonly SELECT: episode_memory f, episode_memory_referent f, self_concept_history t, recall_telemetry f
- recall_telemetry: 53 rows with lived ids in the first day; chat_quick + stance_react pairs 1-72 s apart
- gpu_pool_leases: one memory_distill lease from orion-durable-runs per fold (11-15, 11-16, 11-17)
- page:juniper v1->v2 similarity 0.902, v2->v3 0.910; 4 of 4 quotes kept each time
```

## Docker/build/smoke checks

```text
None. Docs only.
```

## Review findings fixed

- No code-review subagent ran: this PR changes one spec and adds this report. No code.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: high. Concern: lived-memory recall can serve an error beside its correction today, because the supersede pass never ran. Mitigation: #3587 (bug).
- Severity: medium. Concern: the "Hub page view shows every diff" mitigation for no-gate folds does not exist. Mitigation: #3588.
- Severity: medium. Concern: read-demand counts would double-count turns. Mitigation: #3589.
- Severity: low. Concern: `matters` gate waits on Juniper's eyeball pass. Mitigation: #3590 (needs-juniper).
- Severity: medium. Concern: Orion cannot read the tables that would let it check its own memory serves; the #3593 GRANT is merged but not applied. Mitigation: #3611 (needs-juniper).
- Severity: low. Concern: `page:anchor` is guarded on the bus path only and has no size cap. Mitigation: #3612.
- Second round (Orion's follow-ups): #3589 corrected. The double count came from AI Town `chat_quick` rows, not two rows per Hub turn. The spec states the idempotency rule for the supersede pass, and that superseded rows are kept out of both indexes. #3553 gained drift flags (v1 3670 to v3 3517 characters, 35 to 31 cites, 6 uncorrected cite drops).
- Third round: the 131 unexplained `stance_react` ids were one 15-minute non-turn burst (caller UNVERIFIED, logs lost on restart). The read-demand filter is now a join to the turn record. The AI Town page decision is flagged needs-juniper on #3589. #3611 is revised to a full `recall_telemetry` grant, because the same text is already readable in `chat_history_log`.
- Fourth round: corrected the live-proof check. No stored row holds the trip-as-home or pet-species error. The trip rows are true, and the dry run rightly kept them. Corrected the pair-detection text (cosine 0.80 alone qualifies). Write-time supersede applies without a precision sample and falls back to word-only matching when vectors are down (#3659). The confirmation backlog grows, and its rows are immune to supersede (#3661). 8% of chat-turn lived calls hit the rerank timeout (#3662).
- Fifth round (thread on PR #3677): Phase 2 births, archive and the read ledger had no owner; filed #3689. The `matters` fallback when the gate fails is now stated. `dig know` was checked: it goes through recall over HTTP and keeps an empty result apart from a failure. #3554 is marked needs-juniper, with an interim 8-word overlap gate as the recommendation. Juniper's quoted words stay pinned under a claim veto. Merged main, which brought in the Corrections rev 2 section from PR #3677; the conflict was resolved by keeping both sections.
- Issues opened: #3587, #3588, #3589, #3590, #3611, #3612, #3628, #3629, #3659, #3660, #3661, #3662, #3663, #3689. Comments added on #3463, #3537, #3553, #3567, #3589 and #3611. Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3591

🤖 Generated with [Claude Code](https://claude.com/claude-code)
