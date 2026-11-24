## Summary

Why Prime cites so rarely, with live evidence, plus one small fix that makes a dropped cite readable.

- Thought act notes now name a dropped experience cite's id second, before the move (refusals and refs lists have no
  length cap). Before, the 500-character
  note cut the warning before the id (it happened on 2026-11-20 09:29Z), so an id mismatch could not be read back.
- `Applied.experience_unshown` carries the ids the answer cited that the frame did not show; when the now was not
  read at all, `experience_unread` is set and the note says "unchecked, the now was not read".
- Regression test (fails on origin/main's thought.py at merge time), replay golden re-recorded (2 parts changed, 0 after mapping).

Refs #3425. Refs #3434. Follows #3773 and #3797.

## Diagnosis (live, read-only)

All 73 Prime thought acts since 11-16 (`grammar_acts`, actor prime, verb infer), read back from their close notes:

| what the answer did | thoughts |
|---|---|
| proposed no experience cite at all | 70 |
| cited, stored (11-21 20:24Z: 2 gpu_wait; 11-24 05:03Z: 1 chat_turn) | 2 |
| cited, dropped as "not shown in this frame" (11-20 09:29Z; id cut from the note) | 1 |

1. Shown rows are stored for every thought (73 rows, 12 items on most).
2. The frame lists the items (`prime_now_read ... items=12`; frames rebuilt from real shown rows list all of them).
3. The model's answers almost never carry a cite. That is the main cause: 70 of 73.
4. The shown filter dropped 1 cite in 73. Its id was cut from the note, so an id-format mismatch is UNVERIFIED
   for that one. This PR makes the next one readable.
5. `_record_now_use` and the server work: both citing thoughts stored all their cites, against `since=` reads
   (`advances_cursor=false`), so the cursor concern from the #3501 review does not block cites.

What the citing thoughts share: in both, the cited experience was the direct subject of the thought (a GPU-wait
question cited gpu_wait rows; a consult promise cited the chat turn). In the non-citing thoughts the now held
telemetry unrelated to Prime's concerns (field dominance runs, GPU waits, reverie chains, prediction rows at
salience 0.00). Upstream cause filed as #3783: kinds with no salience (reading_turn, vision_percept) sort after
everything else, so two MemOnce reading turns were never shown while Prime held a MemOnce concern.

Live model samples (qwen3.8-flash-next, agent lane, operator hold, frames rebuilt from real now reads):

| prompt | frame | answers | valid JSON | answers with a kept cite |
|---|---|---|---|---|
| before #3773 | live as shown | 15 | 15 | 0 |
| #3773 (refs last) | live as shown | 15 | 9 | 0 |
| #3797 (refs before move, as live now) | live as shown | 6 | 6 | 0 |
| before #3773 | MemOnce readings pinned in | 3 | 3 | 0 |
| #3797 | MemOnce readings pinned in | 3 | 3 | 0 |
| #3797 + step 1 asks for the cite (candidate) | as shown and pinned | UNVERIFIED: queued for a GPU seat | - | - |

One JSON-mode call with the pre-#3773 prompt also came back as a gateway `structured_output_failed`, so invalid JSON
is not only a #3773 effect; the refs-last template made it much more frequent.

So even with an experience in the frame that bears directly on a held concern, the model does not cite it.

## Files changed

- `orion/prime/thought.py`: `Applied.experience_unshown` / `experience_unread`; `summary_note` names unshown ids second.
- `orion/prime/tests/test_prime_now_cites.py`: regression test.
- `services/orion-mind/tests/fixtures/prime_replay_golden.json`, `services/orion-mind/tests/test_prime_replay.py`: re-recorded.

## Schema / bus / API changes

None. The act note text gains one segment when a cite is dropped.

## Env/config changes

None.

## Tests run

```text
pytest orion/prime orion/integration services/orion-mind/tests -q -> 829 passed (main merged in)
new test with origin/main's thought.py -> 1 failed
```

## Evals run

```text
PYTHONPATH=. python orion/prime/evals/run_prime_frame_eval.py -> all PASS; live samples above
```

## Docker/build/smoke checks

Not run: text-only change to an act note.

## Review findings fixed

- Finding: the unshown ids sat after the move segment, which has no length cap (a refusal, a refs list), so the cut could still hide them.
  - Fix: the segment is now second, right after the concern counts; test with a 600-char refusal.
- Finding: with no now read, "not shown" hid the real reason.
  - Fix: `experience_unread`; the note says "cite(s) unchecked, the now was not read".
- Finding: `set(shown_experience)` rebuilt per cite.
  - Fix: built once.
- Finding: placeholders and "fails on main" wording in this report.
  - Fix: filled in; wording says which thought.py.

## Restart required

```bash
scripts/safe_docker_build.sh orion-mind up -d --build
```

## Risks / concerns

- Severity: medium. Concern: Prime will keep citing rarely until the now shows it things that bear on its concerns.
  Mitigation: #3783.
- Issues opened: #3783 (in #3797's session). Issues closed: none.

## Live check after deploy (UNVERIFIED until run)

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select opened_at, outcome_note from grammar_acts where actor='prime' and verb='infer' and outcome_note ilike '%not shown:%' order by opened_at desc limit 5"
```

Pass: any dropped cite shows its id in the note.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
