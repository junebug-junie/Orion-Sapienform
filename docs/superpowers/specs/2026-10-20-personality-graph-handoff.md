# Personality graph — session handoff (2026-10-20)

For the next agent picking up Orion's personality arc. Read this first, then the spec.

## What this arc is

Juniper's question: how do we converge one Orion personality across the many processes on the mesh, so it shows up and changes over time?

The answer being built is a **bottom-up personality graph**:

1. **Evidence.** It comes from Orion's own outputs in every lane (dream, journal, reverie, reading, day letter, chat, outreach, curiosity choices, self-inquiry seeds). Collection started on 2026-10-16. Nothing older was read in.
2. **Graph.** The evidence folds into weighted "dispositions": themes Orion keeps returning to.
3. **Read-back.** The strongest dispositions are read back into the lanes (chat stance, reverie, curiosity kickoff, outreach) as Orion's own record, never as tone instructions.

**Standing direction from Juniper (do not re-litigate):** bottom-up, not top-down.

- Do not precondition on old chats.
- Do not test or steer with tone constraints.
- The graph plus the read-back is the lever. Measurement and A/B tests come after.
- Memory: `feedback_personality_is_bottom_up_not_top_down.md`.

## Docs

- Spec (rev 2): `docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md`. Open as **PR #2726** (docs only, not merged). Theory anchors: CAPS, McAdams, Hebbian learning, Weber-Fechner, habituation.
- Proposal for Orion naming its own dispositions: `docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md`. Open as **PR #2742**, waiting on Juniper's 5 questions.

## Shipped and live

| PR | What |
|---|---|
| #2727 | `bad_tone` feedback tag (Hub thumbs-down), paired with `good_tone` |
| #2729 | P0/P1: tone dials (`orion/personality/tone_dials.py`, measurement only), choice contract `PersonalityChoiceV1` + reducers (`orion/personality/choices.py`) |
| #2738 | The loop: reducer `orion/personality/graph.py`, renderer `orion/personality/cue.py`, durable-runs thread `services/orion-durable-runs/app/personality_*.py`, routes `/personality/{frame,dispositions,evidence}`, read-back in orion-thought (stance + reverie) and orion-hub (curiosity + outreach), Hub Personality tab, tables `personality_*` |
| #2739 | Every evidence row records `meta.situation` (arousal, day phase, company) at its own time |
| #2743 | "When" column (situation mix per node) in the panel; reads wait for the Temporal Self chronicle watermark minus `COMPANY_CLOSE_GRACE` (180 s) so situations are final when stamped |
| #2746 | The choice line needs at least 5 choices and names the window it really covers |

### How the weights work (`graph.py`)

- **Weight** = sum over lanes of `base × min(1.5, ln(1 + lane_recent))`.
  - `lane_recent` is a per-lane hit counter. It fades with a 14-day half-life and is clamped at the cap.
  - The base is 1.0 for most lanes, 2.0 for curiosity choices and 0.5 for seeds.
  - In practice: a lane maxes out after about 4 recent hits, so weight reads as how many lanes keep returning to a theme.
- **Promotion:** a candidate becomes active when 2 or more non-seed lanes each have `lane_recent` of at least 1.5, and the node has at least 3 pieces of evidence.
- **Retirement** is terminal.
- **Measured live history behind these rules:** square-root gain let reverie's volume win. An uncapped `1/(1+recent)` still let a 2,600-hit flood lead. One stray echo hit used to promote a node.

## Live state (checked 2026-10-20 ~03:40 UTC)

- **Reducer:** 286 ticks, 0 failures, no warnings, no held sources.
- **Graph:** 4 active nodes, 88 candidates.
- **Evidence:** 962 rows. journal 444, reverie 399, reading 47, seed 27, chat 13, dream 12, curiosity choice 11, outreach 6, day letter 3. Of these, 923 carry a situation.
- **Active nodes:**
  1. Biometrics prediction error. Weight 4.94; reverie 28, journal 11, dream 3, outreach 1.
  2. Cabinet prediction error. Weight 3.58; reverie 29, journal 7, dream 1.
  3. Self-inquiry, "I am the seventy-fourth answer to the question I keep for myself…". Weight 3.49; journal 11, day letter 3, seed 8.
  4. Bus synaptic prediction error. Weight 2.94; reverie 35, journal 9.
- **Read-back verified live:**
  - orion-thought logs `personality_cue … rendered=True ids=disp-a6a805961f7be5d9`.
  - Outreach `grounding_summary` records the same node id (10-18).
  - Curiosity kickoff: **UNVERIFIED**. The Hub logs were lost on a restart and nothing persists that cue. Check `curiosity_personality_cue run=` in fresh Hub logs.
- **Tone tags:** still only 2 `good_tone` and 0 `bad_tone`. The tone baseline comparison needs at least 5 per side. Script: `scripts/analysis/measure_tone_dial_baseline.py`.

## Problems the live data shows (do these next, in this order)

1. **Orion sees only 1 of the 4 active themes.**
   - Cause: the cue is capped at 600 characters (`PERSONALITY_CUE_MAX_CHARS`), and each label line shows up to 120 characters plus lanes and counts. After the header and the choice line, only the top node fits.
   - Effect: the self-inquiry theme (#3), the only non-prediction-error node, never reaches Orion.
   - Fix in `orion/personality/cue.py`: shorten the shown label (about 60-70 characters) and/or trim the header, so 3-5 nodes fit. Keep the cap, or raise it deliberately; `PersonalityCueV1.text` has a `max_length`.
   - Verify against the live frame, then deploy durable-runs, thought and Hub, which all share the renderer.
2. **Correlated lanes promote prediction-error themes.**
   - 3 of the 4 active nodes are prediction-error narration backed by reverie plus the journal. Most of that journal is `source_kind=metacog`. Both narrate the same substrate signals, so "two lanes" is really one source.
   - This was flagged as a risk in #2738 and is now confirmed.
   - Next: measure lane co-occurrence per node from `personality_evidence` (4 days of data now). Then decide between down-weighting or merging correlated lane pairs (for example, treat metacog journal and reverie as one lane family for promotion) and leaving it.
   - Run the metric gate first. Do not hand-filter "machine text": that is top-down. Fix independence instead.
3. **PRs waiting on Juniper:** #2726 (spec, docs) and #2742 (naming proposal: 5 questions; recommended answers are in the PR). If #2742 is approved, its P0 is a read-only offline naming eval.
4. **Situation subnodes (CAPS if-then).** The data has accumulated since 10-16 23:36 (about 96% of evidence is stamped). Build the read-only analysis first: does each node's situation mix differ from the base rate? Only then add subnodes to the reducer and read-back.
5. **Later:**
   - The #2646 self card as a view over the graph, replacing the `orion_identity.yaml` identity blocks. This is an identity change and needs proposal mode.
   - Prediction/surprise via orion-prediction (categorical choices; data is still thin at about 3 per day).
   - Aspirations.

## Known residuals (documented, not bugs to chase)

- About 30 evidence rows from 10-16 23:36 to 10-17 00:26 keep early (possibly wrong) situations. Evidence is append-only.
- A camera producer dying silently is closed by the chronicle up to 1500 s late, backdated. This is not covered by the 180 s grace.
- `COMPANY_CLOSE_GRACE` mirrors vision-window's `WINDOW_PRESENCE_GRACE_SEC` by hand.
- Labels are excerpts of Orion's current (LLM-voiced) text. They are content, not style, but they carry the old voice.

## How to check things (read-only)

```bash
curl -fsS localhost:8124/health | python3 -c "import json,sys; print(json.load(sys.stdin)['personality'])"
curl -fsS localhost:8080/api/personality/cue        # exactly what the lanes render
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select status,count(*) from personality_disposition group by 1"
docker logs --since 6h orion-athena-thought 2>&1 | grep personality_cue
```

Hub panel: the **Personality** tab (`/personality`).

Real-Postgres tests: durable-runs `tests/test_personality_postgres.py` needs `ORION_ADMISSION_TEST_DSN`. Locally, start a throwaway pg16 (`/usr/lib/postgresql/16/bin/initdb` + `pg_ctl -o "-p <port> -k '' -c listen_addresses=127.0.0.1"`). Unix socket paths under the scratchpad are too long, so use TCP.

## Traps hit this arc

- `scripts/sync_local_env_from_example.py` skips keys outside `SYNC_PREFIXES`. Use `--all-keys`. It writes to the primary checkout's `.env`.
- `PersonalityFrameV1` and the nested models are `extra="forbid"`. A new field needs a consumer-first deploy (thought and Hub before durable-runs), or the readers' cues go dark with `invalid_frame`.
- The Hub `personality` page test needs the full Hub app (aiohttp). It skips in the slim reading workflow by design.
