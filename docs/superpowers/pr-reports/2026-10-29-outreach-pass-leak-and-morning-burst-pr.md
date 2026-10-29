## Summary

- Orion's outreach prompts tell Orion to answer with the one word PASS when a message is not worth sending. Orion often thinks out loud first and then writes PASS on its own line. The check only accepted a reply that was the word PASS and nothing else. So Orion's private reasoning, ending in "PASS", went to Juniper as a chat message.
- This was most of the reading messages, not a few. In the last 30 days, 37 of 49 delivered reading-finding messages and 3 of 95 background outreach messages were declines of this shape.
- Now a line that holds only PASS is the decision, wherever it sits in the reply. So is PASS as the last word after a finished sentence ("Not worth interrupting for. PASS"). PASS inside a sentence is still a message. Every "Orion passed" decision row now says which shape it was (`pass_shape`: `bare`, `after_reasoning` or `trailing`).
- Reading findings wait out quiet hours and then all went out at once. On 2026-10-28, 14 messages arrived between 14:11 and 15:08 UTC. Now, after a reading message reaches Juniper, the next one waits 112.5 minutes. A PASS does not start the wait. A held finding stays in the queue.

## Outcome moved

- Juniper no longer gets Orion's "should I send this?" reasoning as a message.
- The morning pile is spread out. Replayed on the last 30 days of real sends: the most reading messages in any hour drops from 3 to 1 (after the PASS fix removes the declines). The longest delay a real message would get is 183 minutes.

## Current architecture

- `is_pass_response` (`services/orion-hub/scripts/endogenous_outreach.py`) was `text.strip().upper() == "PASS"`. All four outreach doors use it: the background tick (`maybe_outreach`), strain reports, and `offer_message` (curiosity findings and reading findings).
- Reading findings (`services/orion-hub/scripts/reading_finding_outreach.py`) skip the daily cap and cooldown on purpose (Juniper, 2026-10-13). Quiet hours hold them. At quiet-end, each 5-minute tick composed and sent every pending read, one after another.

## Where the PASS fix lives

The fix is in `is_pass_response`, not in the shared finalize chain (`orion/cognition/cortex_payload_extract`). PASS is a contract of the outreach prompts only. A normal chat turn is never asked for PASS. In 30 days, the only assistant chat rows with a PASS line were these outreach messages. One function covers all four outreach doors, so the background tick and the daydream-fed tick get the fix too (3 of the leaks came from there).

## Architecture touched

Hub only. No bus, schema or API change.

## Files changed

- `services/orion-hub/scripts/endogenous_outreach.py`: `pass_shape()` (three shapes) and `is_pass_response()`; `pass_shape` is added to the three `orion_passed` decision rows. `_publish_notification` is not touched.
- `services/orion-hub/scripts/reading_finding_outreach.py`: the spacing wait, `LAST_SENT_SQL` (the last send, read back from the decision log after a restart), the `reading_spacing` hold row, and a shared "record a block once" helper.
- `services/orion-hub/scripts/main.py`, `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: the new key.
- `services/orion-hub/README.md`: the spacing rule and the PASS rule.
- `orion/world_pulse_read/finding_prompt.py`, `orion/curiosity/outreach_prompt.py`: docstrings that described the old rule.
- `services/orion-hub/tests/test_endogenous_outreach.py`: the real 2026-10-28 15:08:09Z text and the two real trailing-PASS texts as regression tests, plus PASS-in-a-sentence cases that must stay messages.
- `services/orion-hub/tests/test_reading_finding_outreach.py`: a reasoned PASS is recorded as passed and never delivered; the morning pile goes out one at a time; a PASS does not start the wait; the gap survives a restart; 0 turns the wait off.
- `services/orion-hub/evals/run_outreach_pass_spacing_replay.py` and `evals/test_outreach_pass_spacing_replay_eval.py`: a read-only replay of real sends through the new rules, and its offline test.

## Schema / bus / API changes

- Added: decision rows may carry `pass_shape` (on `orion_passed`) and the new reason `reading_spacing` (with `spacing_wait_sec`, `min_spacing_sec`). `result_json` is free-form JSONB. No consumer counts reasons other than `sent`.
- Removed / renamed: none.
- Behavior changed: a reply with a PASS line is now a decline.
- Compatibility notes: none.

## Env/config changes

- Added keys: `HUB_READING_FINDING_OUTREACH_MIN_SPACING_SEC=6750` (0 turns the wait off).
- How 6750 was chosen: waking hours are 8 to 23 local (15 h). The busiest local day so far had 4 real reading messages (2026-10-28, after removing the PASS leaks). 15 h / (2 x 4) = 112.5 min. So twice the busiest day still fits in one day, and Juniper's "no cap" stands.

  ```sql
  select (decided_at at time zone 'America/Denver')::date as d, count(*) sends,
         sum(case when body ~ '(^|\n)[\s\*_`]*PASS[\s\*_`\.]*(\n|$)'
                    or body ~ '[.!?][ \t]+PASS[.!"*]*\s*$' then 0 else 1 end) real_msgs
  from (select d.decided_at,
               split_part(split_part(m.content, E'\n\nWhat the piece says -- ', 1), E'\n\nSource: ', 1) body
        from endogenous_outreach_decisions d
        join chat_message m on m.correlation_id = d.result_json->>'correlation_id' and m.role = 'assistant'
        where d.reason = 'sent' and d.result_json->>'source' = 'reading_finding_outreach') x
  group by 1 order by 1;
  -- real per day: 10-14 0, 10-15 1, 10-17 1, 10-23 2, 10-24 1, 10-25 1, 10-26 1, 10-27 1, 10-28 4
  ```
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (the key was added; its first value, 5400, was set by hand to the re-derived 6750, because the sync keeps existing values).
- skipped keys requiring operator action: none.

## Tests run

```text
cd services/orion-hub && pytest -q tests/test_reading_finding_outreach.py tests/test_endogenous_outreach.py \
  tests/test_strain_report_outreach.py tests/test_curiosity_investigation.py tests/test_outreach_provenance.py
481 passed (first run); after review fixes 453 passed on the reading, endogenous, curiosity, strain and replay-eval files
python scripts/check_env_template_parity.py            -> PASS
python scripts/check_settings_defaults.py orion-hub --example-drift -> OK
pytest tests/test_no_personal_terms.py                  -> 7 passed
```

## Evals run

```text
pytest services/orion-hub/evals/test_outreach_pass_spacing_replay_eval.py evals/test_reading_finding_eval.py -> 3 passed
PYTHONPATH=.:services/orion-hub python services/orion-hub/evals/run_outreach_pass_spacing_replay.py --days 30   (live, read-only)
curiosity_outreach: delivered=7 would_now_be_pass=0
endogenous: delivered=95 would_now_be_pass=3
reading_finding_outreach: delivered=49 would_now_be_pass=37
strain_report: delivered=2 would_now_be_pass=0
reading real messages=12 max/hour before=3 after=1 longest delay=182.9 min (spacing 6750s)
```

## Docker/build/smoke checks

```text
Not run. Pure Python change inside the Hub process; no dependency, port or compose change.
```

## Review findings fixed

A review subagent read the branch diff. It found no blockers. It re-ran the live replay. It scanned 90 days of delivered messages for false positives and found none.

- Finding (should): PASS as the last word after a sentence ("Not worth interrupting for. PASS") still leaked. 2 live sends had this shape.
  - Fix: a third shape, `trailing`. PASS must follow a sentence end and be the last word of the reply.
  - Evidence: both live texts are regression tests. "gates PASS on cadence" and "All PASS on the first try." stay messages. The replay now catches 37 of 49, not 35.
- Finding (nit): a reply that starts with a PASS line and then continues is dropped.
  - Fix: documented in `pass_shape`. It never happened live. The row keeps the shape.
- Finding (nit): the test alias `outreach_kw_last_sent` was misleading.
  - Fix: removed.
- Finding (nit): no test checked the filter in `LAST_SENT_SQL`.
  - Fix: `test_the_last_sent_query_counts_only_reading_sends`.
- Not fixed (nit): when spacing holds a bundle a second time, no second row is written. A finding held past the 48 h lookback ages out without a row. Quiet-hours holds already work the same way. With 4 real messages a day and 8 spacing slots, an age-out is unlikely.

## Restart required

Hub rebuild after merge, through the normal post-pull rebuild on the primary `main` checkout:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
```

## Risks / concerns

- Severity: low. Concern: a real message that contains a line holding only "PASS" would now be dropped as a decline. Mitigation: in 30 days no delivered outreach message had such a line except the 40 declines; the row is kept with its `pass_shape`, so a wrong drop is visible.
- Severity: low. Concern: on a heavy day, real messages wait up to 112.5 minutes each. Mitigation: the 48 h lookback holds them; set the key to 0 to turn the wait off.
- UNVERIFIED: the live path after deploy. Proof to look for: an `orion_passed` row with `pass_shape='after_reasoning'` for `source='reading_finding_outreach'`, and a `reading_spacing` row the morning after quiet hours.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2942

🤖 Generated with [Claude Code](https://claude.com/claude-code)
