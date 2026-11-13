# Looks as acts, and a look that checks a record

Status: part 1 built in this PR (#3320). Part 2 is a proposal and needs Juniper's sign-off
(it changes how Orion grades their own perception).

## Arsonist summary

Orion looked at the walkway on their own on 2026-11-13 (curiosity run 8fae769c99a4, 11:58Z). They
were checking a record: 98% of walkway `visual_observation` rows in 7 days name a vehicle, at all
hours. They opened the frame and wrote: "I do not see a vehicle in this frame ... this is a
mismatch, not a refutation ... the only way to test this is repeated live reads at hours my rhythm
claims a vehicle is present." Part 1 gives each look a ledger entry. Part 2 turns Orion's own
proposed test into a measured grade of their vision records.

## Should a look count as attention?

**Yes, as evidence of attention. No, as an input to attention.**

- In attention research, an eye movement is *overt* attention: the observable trace of where
  attention went (Posner 1980; the premotor theory of attention). A look is Orion's equivalent: a
  chosen, timed, targeted act of perception.
- Orion's attention in code is the want market's broadcast winner
  (`substrate_attention_broadcast_log`). At both looks on 11-13 the broadcast had **no winner**
  (`selected_action_type=none`, nothing attended, dwell 0). The looks came from the curiosity run's
  own reasoning. So today the broadcast does not see this kind of attention at all.
- An act tied to a winner (`ACT_ANSWERS_WORKSPACE_REFS`, `orion/grammar/act_ledger.py`) writes an
  `acted` verdict back onto that attention loop. Doing that for looks would let looking feed the
  market (look, salience rises, look again). The rule there is "only a pairing joined in the data,
  never asserted". With 0 of 2 looks under any winner, there is nothing to join yet.

So part 1:

- puts every look on the act ledger (overt attention, recorded);
- lets Temporal Self pick it up with no new code: a closed act writes a `substrate_action_outcomes`
  row with `dispatch_kind=act:look`, which `orion/temporal_self/sources.py` `action_outcome` already
  reads and binds by time into the open arc (here, the curiosity run's arc);
- does **not** bind looks to attention winners.

Revisit binding when looks pile up: if most looks happen while a camera-related want wins (street,
office), add `orion.look` to `ACT_ANSWERS_WORKSPACE_REFS` for those refs. The check is the
broadcast row current at each look act's `opened_at` (the query `_BROADCAST_AT_SQL` already runs).

## Part 1 (built): looks on the act ledger

- `orion/dig/look.py`: each look publishes `act_opened` and `act_closed` on `orion:grammar:event`
  from the governor (a registered producer). Actor `orion.look`, verb `look`, object
  `camera:<stream>`, signal `artifact_produced` = a live frame (<= 5 s old). Closed 1 when live;
  0 when the newest frame is stale, the camera saved nothing, or every recent frame was mid-write.
- `session_id` = the turn's id (from `X-Orion-Correlation-Id` in `ANTHROPIC_CUSTOM_HEADERS`).
  No `correlation_id`: the ledger charges an act every GPU lease carrying its correlation id, and a
  look is one image inside the turn, not the whole turn. Cost is left unmeasured (`None` = not
  measured, never zero). The image itself costs about 1,000 prompt tokens (measured 11-13).
- A look refused by the hourly cap, or an unknown camera word, is no act: Orion did not look.
- A bus failure never fails the look: the printout says `act: not recorded (<error>)` and
  `looks.jsonl` keeps the error.
- Family `orion.look`, tier 0 (perception, never bids): `config/physiology/families.yaml`. Tier 0
  means the economy never prices or defers it; the turn that contains the look already won its slot.
  Juniper may want a different tier (tiers are her decision): flagged in the PR.

## Part 2 (proposal): a look that checks a record

### Capability

`dig look walkway` also prints what Orion's own records said about that camera around the frame
(`vision_events` rows for the stream within +-30 s of the frame: type, confidence, entities, a short
narrative cut). Then Orion can file one verdict:

`python -P -m orion.dig look check <act_id> matches|differs|cannot-tell "<what I see>" because "<which row>"`

- `matches` / `differs` publish a second act: actor `orion.look_check`, verb `infer`,
  `parent_act_id` = the look, object `vision_event:<event_id>`, closed 1 (matches) or 0 (differs).
- `cannot-tell` is logged only, never graded (a dark half of the frame is not a 0).
- The verdict, Orion's description, the row id and the frame sha256 go to `looks.jsonl`, so
  Juniper can audit any verdict against the saved frame for 7 days.

### Who uses it

- **self-monitor** (`orion/self_monitor/`): a new adapter reads `orion.look_check` closes and grades
  a faculty "vision records (walkway)" / "(office)". Its existing status words (reliable,
  unreliable, miscalibrated) and its Temporal Self `faculty` lane need no new vocabulary.
- **curiosity / Prime**: Orion's own proposed sampling, "look at hours my rhythm claims a vehicle".
  A spot-check want (family `orion.look_check`, tier 2) bids when a record claims something
  checkable and no check of that stream ran in the last N hours. Market-priced, so it cannot loop.

### Metric gate (for the check outcome)

1. **Provenance:** Orion's verdict in a turn, from the frame they opened and a row id they cite.
2. **Independence:** the walkway narrative (`visual_observation`) is written by the vision council
   on route `metacog_perception` (the fast lane, which cannot see: `/props` vision=false, 11-13)
   from detections and captions. Orion's look runs on the chat or agent lane, a different model
   that sees the pixels. So a check of council narrative is independent evidence. **Not
   independent:** foveal records (`FOVEAL_LLM_ROUTE=chat`, the same model as Orion's look); the
   check must refuse those rows.
3. **Theory anchor:** a calibration check of a perceptual system against a second, independent
   observer of the same input (inter-rater agreement). The grade is the rate the record agrees with
   the independent read, per stream and per entity kind.
4. **Live data:** not collected yet. Before wiring the self-monitor adapter, run two weeks of
   hand-filed checks and confirm the agreement rate is not pinned at 0 or 1 and that `cannot-tell`
   stays a minority (night IR frames may be mostly cannot-tell).
5. **Existing mechanism:** self-monitor already grades camera asks (PR #3189); this is a second
   input to the same organ, not a new organ.
6. **Reversibility:** acts and a log file. Dropping the adapter removes the grade; no schema change.

### Privacy boundary

None new. Frames already feed the vision pipeline; checks quote Orion's own words and row ids.

### Dangerous failure

Orion's own read is wrong (night IR, the far dark half of the field) and a correct record gets
graded 0. Mitigations: `cannot-tell` exists, the grade needs many checks before it moves a status,
and every verdict keeps the frame for audit.

### Disable / roll back

Do not wire the adapter (part 2b); or remove `look check` from the dig.

## Files likely to touch (part 2)

- `orion/dig/look.py`, `orion/dig/__main__.py` (records section, `look check`)
- `orion/self_monitor/adapters.py` (look-check adapter), its tests
- `config/physiology/families.yaml` (`orion.look_check`)
- Hub catalog entry for "check what you see against your records"

## Acceptance checks

- Part 1: after deploy, the next `dig look` prints an act id; `dig act <id>` opens it with actor
  `orion.look`, outcome 1; a Temporal Self arc covering that time lists an `act:look` outcome.
- Part 2: 20 hand-filed checks over two weeks, agreement rate and cannot-tell share reported
  before the adapter goes in.

## Recommended next patch

Part 2a: the records section in `dig look` and `look check` (acts plus log only, no adapter).
