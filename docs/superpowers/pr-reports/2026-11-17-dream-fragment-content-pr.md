## Summary

- Dream replay fragments now carry what actually happened, not just a label. Metacog fragments get
  their time and their evidence ("exit_code=-9", "finalize cut after 207s"). Compaction-request
  fragments get their time and the reverie chain's own last thought in place of how the chain ended
  ("reverie_chain_settled").
- The cause of #3412 is found: the model on the dream route is honest, and the old 100% link rate
  was the defect. Evidence is on #3412.
- The pairing rule is the other half (it pairs the most unrelated fragments). That's a decision for
  Juniper: #3664.

## Outcome moved

The model can now check a link against facts and times. Before, it saw prose and labels.

Live, on the 11-17 19:08Z sleep, same model route:

| Fragments | Pairing | Linked |
|---|---|---|
| as stored (before) | current (max distance) | 0/3 |
| with time only | current | 0/3 |
| with time + content (this PR) | current | 0/3 |
| with time + content | close in time, different organs | 1/4 |

Sanity check, same route: 2 really-linked pairs → linked with specific claims; 1 unrelated → declined.

So this PR alone may leave the link count near 0. The count is not the target. The target is that
every claim the dream makes can be checked against its two refs, and that the stored pair answers
(PR #3636) show honest declines on real content. #3664 decides whether the pairing changes.

## Current architecture

`cycle_store.SOURCE_QUERIES` read `orion_metacog.summary` (model prose) and
`dream_compaction_request_queue.reason` (the chain's terminal reason) plus the loop title.
`replay.candidate_from_*` turned those into `ReplayItemV1.text`, the only text the recombine prompt sees.

## Architecture touched

orion-dream only: two source queries and two candidate builders. No schema, table, channel or env change.

## Files changed

- `services/orion-dream/app/cycle_store.py`: metacog query adds `occurred_at` and `evidence`
  (`what_changed->'evidence'`); compaction query adds `occurred_at` and the chain's last thought
  (`_chain_thought_join`, primary-key lookups).
- `services/orion-dream/app/replay.py`: `_when()` time prefix; metacog text = time + summary +
  evidence; compaction text = time + title + thought (falls back to the old reason).
- `services/orion-dream/tests/test_dream_cycle_v2.py`: text shape, ISO-string and bad times, rows
  without the new columns keep the old text, keys unchanged, SQL reads the content.

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: `ReplayItemV1.text` for metacog and compaction_request fragments is longer and
  starts with a time. `ref_id`, tags and dedupe keys are unchanged, so novelty and tiredness hold.
- Compatibility notes: stored cycles keep their old text; only new sleeps change.

## Env/config changes

None.

## Tests run

```text
services/orion-dream tests+evals: 406 passed, 9 skipped (after review fixes)
git diff --check: clean
```

## Evals run

The orion-dream evals ran in the suite above. The live experiments in "Outcome moved" are the
behavior eval for this change (scripts run read-only inside orion-athena-dream, then removed).

## Docker/build/smoke checks

Both new queries ran live in a `BEGIN READ ONLY` transaction against conjourney:
- They return the expected rows (metacog evidence such as "timeout open on
  orion:exec:request:LLMGatewayService; load 6.0/min vs usual 0.6/min"; compaction thoughts such as
  "The reading loop is stalling because the system is under acute strain from a GPU backlog").
- EXPLAIN ANALYZE: about 200 ms each.

## Review findings fixed

Review by a subagent (by hand; the skill was not loadable there). It checked the SQL against live
column types (`what_changed` jsonb; chain/thought ids text primary keys, so the LATERAL cannot
multiply rows) and found nothing material. Small findings:

- Finding: the overall 600-character clip could cut the end of the evidence (about 0.9% of live rows).
  - Fix: evidence gets only the room the summary leaves; no evidence if under 20 characters remain;
    the overall cap still applies.
  - Evidence: tests for a 272-character summary (evidence trimmed, close kept) and a 700-character
    summary (summary only, exactly 600).
- Finding: evidence already ending in "." produced "..".
  - Fix: `rstrip(". ")` before the closing stop. Evidence: test.
- Not changed: the story dream trims each fragment to 280 characters (`story.py ITEM_TEXT_CHARS`), so
  most evidence won't reach it. Recombine gets the full 600. The story's length is a separate choice.
- Checked sound: no stable-key consumer reads fragment text (keys use id/theme/dedupe_key); the echo
  guard is a verbatim-substring test, so longer text cannot cause wrong rejections; both scorecard
  arms draw from the same pool, so neither is favoured.

## Restart required

Merge, then the post-pull rebuild rebuilds orion-dream. Check after the next sleep:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select left(i->>'text',200) from dream_cycle, jsonb_array_elements(cycle_json->'replay') i order by started_at desc limit 5"
```

## Risks / concerns

- Severity: low. Concern: the reverie thought is model text, so it can be wrong. Mitigation: it is
  labelled by source kind and time, and a claim's refs point at the stored rows.
- Severity: medium. Concern: the link count may stay near 0 until #3664 is decided. Mitigation:
  #3412's done-when is revised to checkable claims and honest declines, not a count.
- Issues opened: #3664 (needs-juniper, pairing rule). Refs #3412.
