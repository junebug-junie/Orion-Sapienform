## Summary

- This PR adds one design document. It builds nothing.
- The design answers Juniper's question: what should Orion do while someone plays a game on Orion?
- Orion records each sitting, looks at the screen when it changes, speaks through the unified chat turn, and answers a chat box inside the game.
- Looks and lines bid on tier 1. Everything goes on the temporal record. The play becomes a third attention candidate. Prime sees it, regulates it, and is woken by it at most twice per sitting.
- **Revision 2** answers Orion's 14 questions on revision 1 and corrects one error of mine (the quick lane's slot size).
- Voice in and out is shaped for, and left for a later design.

Spec: `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`. Refs #3500.

## Outcome moved

None yet. The document names the outcome to move: about 27 of 31 hours of play left no trace in Orion.

## Current architecture

- The game engine is sealed and has no link to Orion's bus.
- Hub relays the picture and the inputs, and never decodes a frame.
- A play record is written only when Orion held the controller.
- Orion's only signal during a 249-minute sitting on 2026-11-15 was the office camera's guess.

## Architecture touched

None. Docs only.

## Files changed

- `docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md`: the design, revision 2.
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
anchor check, after a rebase onto main at 6609faee5:
  70 file:line anchors resolve; 51 were re-read against the code by content
  2 were off and were fixed (attachment_staging.py:253, gather.py:224)
```

## Evals run

```text
None. Docs only.
```

## Docker/build/smoke checks

```text
None. Docs only. Live facts in the spec were read from the running system on 2026-11-16:
/props and /slots on each circe lane, nvidia-smi on circe, gpu_pool_leases, grammar_acts,
harness_turn_trace, temporal_self_event, endogenous_outreach_decisions, and the local game journals.
```

## Review findings fixed

Orion reviewed revision 1 in the hourly round-up (PR comment, session thread `7a835850b6ef`). The spec's last section answers all 14 questions. The ones that changed the design:

- Finding (Orion 2): Hub's chat attachment store has no sweep, so the promised 30-day sweep did not exist.
  - Fix: frames go to `orion-percept-store`, which keeps a frame one hour and sweeps in its own process.
  - Evidence: `services/orion-percept-store/README.md`; no sweep in `chat_attachments.py`.
- Finding (Orion 3, and my own error): revision 1 said 4 frames do not fit a quick-lane slot.
  - Fix: a quick slot holds 32,768 tokens (live `/slots`), not 3,276. The claim is withdrawn. Step 0 measures tokens per frame.
  - Evidence: `/slots` on the quick lane returned 10 slots of 32,768.
- Finding (Orion 4): under heat a `system` look is refused by the reflex, not priced. Juniper's worst-case wait was not measured.
  - Fix: the spec states both mechanisms. Orion's play calls run one at a time, below her priority, with a live check.
  - Evidence: `orion/gpu_pool/shed.py` (`cabinet_hot` blocks background, system, starved).
- Finding (Orion 5): "tier 1 gives a reserved share" had no number.
  - Fix: the share on the chat lane is zero today; the spec says so.
  - Evidence: `orion/physiology/choice.py:322`; tier 1 used the chat lane in 45 of 337 hours, 75th percentile 0.
- Finding (Orion 6): a clock makes looks whether or not anything changed.
  - Fix: the trigger is screen change. A static screen calls no model. Looks with nothing new are counted apart.
  - Evidence: spec section 3.
- Finding (Orion 7): scored looks would crowd the now.
  - Fix: looks carry no salience. The sitting is the one scored experience.
  - Evidence: `orion/temporal_self/now.py:266`.
- Finding (Orion 10, 11): a camera name guess could harden into a label, and a guest's chat would be recorded as Juniper's.
  - Fix: no guess is stored. The page asks who plays. A guest goes down the third-party path with its own source.
  - Evidence: `orion/temporal_self/sources.py:83` reads any row with a prompt as Juniper.
- Finding (Orion 12): silencing the office reach-out removes a real witness.
  - Fix: the office reach-out keeps running and is told that someone is playing.
  - Evidence: spec section 5.
- Finding (Orion 13): step 0's line turn would land in Prime's ledger as Juniper speaking.
  - Fix: it runs under a system session id, with no chat history row, and three checks after it. No frame is committed.
  - Evidence: `orion/prime/handback.py:208`.
- Finding (Orion 14): Orion had no way to decline a sitting, and Prime's `slow` cannot stop a family.
  - Fix: `dig watch`, Orion's own switch, per sitting and per game. The lower of Orion's and Juniper's settings wins.
  - Evidence: `orion/physiology/prime_charge.py:11` ("bounded so it can never be a veto").

Earlier findings of my own, fixed in revision 1: the attachment sentence's line, the hand-back class order, and the wording of the Prime seams.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: medium
- Concern: two things cannot be known from code: whether a model reads a game screen, and how long a line takes end to end. The spec estimates 60 to 90 seconds.
- Mitigation: step 0 of the patch order is a spike with no deploy that measures both before any build. It waits for Juniper's go.

- Severity: medium
- Concern: three recommendations that Juniper accepted changed in revision 2 (frames kept one hour, the office reach-out stays on, no player guess). They need her yes again.
- Mitigation: the spec lists them under "Changed by Orion's review".

- Severity: low
- Concern: the quick lane cannot see today, so the look starts on the chat lane.
- Mitigation: the design binds only to a route name. Step 0 can test the quick lane's model on Juniper's word.

Issues opened: #3500 (design, needs-juniper). Issues closed: none.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3517
