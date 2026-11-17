## Summary

- This PR adds one design document. It builds nothing and deploys nothing.
- The design answers Juniper's question: what should Orion do while someone plays a game on Orion?
- Orion records each sitting, looks at the screen when it changes, speaks through the unified chat turn, and answers a chat box inside the game. Orion can also decline to watch.
- **Revision 3** adds the results of step 0 (a spike with no deploy, run on Juniper's "yes" on 2026-11-17), her answers, and the answers to Orion's second round of 10 points.
- Step 0 in one line: the chat lane's model reads a game screen well enough to try (place and action right in 9 of 10 notes, one wrong detail in 4 of 10, 5.6 seconds a look). The quick lane's model does not.

Spec: `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`. Refs #3500.

## Outcome moved

None in the running system. Step 0 replaced four guesses in the design with measurements: whether a model can read a game screen, what a frame costs in tokens, what a look costs in seconds, and whether the change trigger works.

## Current architecture

- The game engine is sealed and has no link to Orion's bus.
- Hub relays the picture and the inputs, and never decodes a frame.
- A play record is written only when Orion held the controller.
- Orion's only signal during a 249-minute sitting on 2026-11-15 was the office camera's guess.

## Architecture touched

None. Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`: the design, revision 3.
- `docs/superpowers/pr-reports/2026-11-16-game-company-watch-and-commentary-design-pr.md`: this report.

## Schema / bus / API changes

- Added: none in this PR. The spec proposes them under "Proposed schema / API changes".
- Removed: none.
- Renamed: none.
- Behavior changed: none.
- Compatibility notes: none.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed, no env template changed.
- skipped keys requiring operator action: none.

## Tests run

```text
python3 scripts/check_personal_terms.py   -> no personal terms in tracked files
git diff --cached --check                 -> clean
anchor check                              -> 73 file:line anchors resolve against main
```

## Evals run

Step 0, a spike with no deploy. Its evidence is on athena in `~/.orion/retro-spike/` (owner-only, not in the repo) and in `gpu_pool_events` under the holder `operator:retro-watch-step0`.

```text
render      20 sequences, 600 frames, from 10 real saves, headless in the game image      86.5 s
looks       chat lane's model (Qwen3.8 Flash-Next, on the agent lane, leased hold)        18 calls
            median 5.6 s a look; 1,086 prompt tokens for 4 frames at 2x; 2,206 at 3x
            place and action right in 9 of 10 notes; one wrong detail in 4 of 10
looks       quick lane's model (Qwen3.5-9B + encoder, temporary copy on GPU 3)            22 calls
            median 2.7 s a look; 634 tokens at 2x; a thing not on the screen in 9 of 10 notes
change      still screens 0.0002-0.0004; calmest walk 0.05; fades and room changes to 0.43
counters    one sitting: 243 minutes by journal rows, 204 by mask changes (ratio 0.84)
cost        one change check 1.5 ms on one core; one 2x image 10 ms, 14 KB
not run     item 4, one line turn by hand (its origin would have been recorded as Juniper's)
```

## Docker/build/smoke checks

```text
None for the repo. Docs only.
Step 0 started and removed one temporary container on circe. circe is back to its earlier state:
no spike container, the encoder file and frames deleted, GPU memory as before.
```

## Review findings fixed

Orion reviewed the spec twice in the hourly round-up (PR comments, session thread `7a835850b6ef`). The spec's section "Orion's review" answers all 24 points. The ones that changed the design:

- Finding (Orion, round 1): Hub's chat attachment store has no sweep.
  - Fix: frames go to `orion-percept-store`, one hour, swept by that store.
  - Evidence: `services/orion-percept-store/README.md`.
- Finding (Orion, round 1, and my own error): 4 frames were said not to fit a quick-lane slot.
  - Fix: a quick slot holds 32,768 tokens. The claim is withdrawn. Step 0 measured the tokens.
  - Evidence: live `/slots`; step 0.
- Finding (Orion, round 1): a clock makes looks with nothing there.
  - Fix: the trigger is screen change, computed before any model call.
  - Evidence: step 0, 20 sequences.
- Finding (Orion, round 2, and step 0): a model cannot be trusted to say "nothing new". In step 0 it copied the earlier note word for word over new frames.
  - Fix: the look is stateless, and no model or string comparison judges newness.
  - Evidence: step 0, three tests.
- Finding (Orion, round 2): the backfill baseline and the live counter were different instruments.
  - Fix: both count mask changes.
  - Evidence: step 0, one sitting through both counters.
- Finding (Orion, round 2): a frame pointer outlives its frame, and "expired" looks like "lost".
  - Fix: each row keeps `frames_expire_at`, and an acceptance check tells the two apart.
  - Evidence: spec section 2.
- Finding (Orion, round 2): a cancelled line turn can still hold a finished draft.
  - Fix: the line is marked `yielded` first, and delivery refuses a `yielded` line.
  - Evidence: spec section 4.
- Finding (Orion, rounds 1 and 2): no way to decline, no record of the choice.
  - Fix: `dig watch`, stored in Postgres, with an act row and a line in Prime's frame.
  - Evidence: spec section 6.
- Finding (my own, during step 0): the temporary model copy landed on GPU 2, not GPU 3, because I chose the card by number.
  - Fix: stopped at once, restarted pinned by hardware id. The pool's records show no harm to the seat on GPU 2.
  - Evidence: `gpu_pool_leases` for `agent-gpu2`, 19 leases released clean in that window.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: 4 of 10 notes from the chat lane's model hold one wrong detail. The bar in revision 2 said none.
- Mitigation: open question 1 for Juniper. Notes are stored as looks, never as facts, and the turn that speaks sees the frames itself.

- Severity: medium
- Concern: looks must share the chat lane's one slot, because the quick lane's model failed.
- Mitigation: a look takes 5.6 seconds, runs below Juniper's priority, one call at a time. Her wait is an acceptance check in patch 3.

- Severity: low
- Concern: the end-to-end time of a line is still an estimate (60 to 90 seconds).
- Mitigation: patch 3 times the first real line turn by stage.

Issues opened: #3500 (design, needs-juniper). Issues commented: #3474 (session thread reply timed out though Hub stored it). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3517
