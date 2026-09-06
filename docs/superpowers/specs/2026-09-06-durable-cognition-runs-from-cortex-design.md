# Durable cognition runs, kicked off from cortex

**Date:** 2026-09-06
**Status:** Design / proposal mode (CLAUDE.md sec 0A -- touches cognition loops; no code in this patch)
**Program:** Sentience Striving Program, Objective 3 -- step 3 of the ordering in
`docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md` ("The state machine")
**Prior decisions honoured:** `README.md` Long-Term ("Durable LangGraph-style planning for selected
workflows without replacing the existing verb/action spine"); knowledge-forge v0 non-goal
("LangGraph HITL workflows", deferred to v2); Juniper 2026-09-06: "cortex should still have the
kick off to know we are doing cognition with langgraph".

---

## Arsonist summary

Steps 1 and 2 shipped and moved their numbers (surface: PR #2124; bridge: PR #2126,
`goal_matched_no_loop` 42% -> 14%). The ordering said step 3 only opens if step 2 moves a number.
It did. But the thing step 3 is *for* changed while steps 1 and 2 were being built, and the
evidence came from the surface itself.

The surface's cross-lane join traced today's last curiosity run into cortex at 18:49:11Z (a
`cortex_turn` row with the run's correlation id) and then to nothing: no journal, no graph
write, no surface row. The hub was redeployed at ~19:01Z. Redis says three curiosity runs were
started today and the daily cap of three is spent; the journal says zero finished. A curiosity
investigation is Orion's longest single act of cognition -- 10 to 40 minutes, up to 2,400s of FCC
budget -- and it runs as one unbroken RPC inside a container that gets replaced several times a
day. Every replacement kills it, and nothing anywhere can pick it back up.

That is not a sequencing problem. It is a durability problem. The original step-3 framing --
"a state machine that sequences the three attention processes" -- would have been the third
parallel attention implementation this program has already warned itself about twice. The
narrower, older, already-decided framing is the right one: **durable, resumable runs for the
few workflows that are long enough to die, kicked off from cortex so cortex knows cognition is
happening.** The 30-second attention tick, reverie, and the goal producer stay exactly as they
are.

---

## Current architecture

### What a cognition run is today

Hub-owned saga (`orion/hub/turn_orchestrator.py::execute_unified_turn`): observation ->
pre-turn appraisal -> Thought stance RPC -> `HarnessRunRequestV1` to the harness governor over
bus RPC -> governor runs **one FCC subprocess per run** (`services/orion-harness-governor/app/
bus_listener.py::handle_harness_run_request`) -> `HarnessRunV1` back -> finalize legs in
cortex-exec (brain mode) -> frames. The governor has a cancel listener
(`HarnessRunCancelV1` kills the subprocess by correlation id) and **no checkpoint, no resume**.
The run's state lives in the subprocess's memory and the RPC's stack.

The curiosity investigation (`services/orion-hub/scripts/curiosity_investigation.py`) drives
that saga from Hub's own tick loop: kickoff prompt -> `execute_unified_turn` -> read Orion's own
graph -> publish attention-surface row -> journal -> optional outreach. It stamps the daily cap
and cooldown *before* the turn (`_record_investigation`) and refunds only on a clean
`CancelledError` (`_refund_investigation`, deliberately -- see its docstring for the two
measured incidents 2026-08-27/28 that motivated it).

### What cortex is, for this purpose

`orion-cortex-orch` is the intake for every cognition request (`orion:cortex:request`), routes
by verb (`app/decision_router.py`, `orion/cognition/verb_catalog.py`, catalog in
`orion/cognition/verbs/*.yaml`), and dispatches plans to `orion-cortex-exec`. The curiosity loop
does **not** enter through cortex-orch: it calls the Hub saga directly. So "cortex has the
kickoff" is not true for the one workflow most in need of durability.

### What already exists that this must not duplicate

- The unified turn IS a sequencer for one turn. Not replaced.
- The governor's cancel listener IS lifecycle control. Reused.
- `substrate_attention_schema` IS the place a run's state can be observed across lanes.
  Extended by one more producer, not a new table.
- `HarnessTurnTraceSQL` (`harness.run.v1` etc., via orion-sql-writer) IS the run trace.
  A checkpoint is not a trace and does not replace it.

### Live evidence, 2026-09-06

```text
redis  orion:curiosity:count:2026-09-06 = 3     (daily cap spent)
graph  latest TurnOutcome                  = 2026-09-05 16:53Z   (no run finished today)
journal self_study rows today              = 0
surface cortex_turn row, corr=uuid5(curiosity_investigation:2edafe3f4bd4) at 18:49:11Z, leg=stance_react
hub    container recreated ~19:01Z; docker logs from before are gone (also recreated 05:32Z)
```

Whether the three runs were killed by the two redeploys or failed on their own is
**UNVERIFIED** -- the logs that would say so did not survive the recreate. That is itself part
of the finding: a run whose only state is a process's memory leaves no evidence when the
process dies.

---

## Missing questions

**1. Which workflows are "long enough to die"?** Curiosity investigation (10-40 min) is the
only certain one. Self-study reflect (timeout raised to 480s in PR #2122) is borderline.
Endogenous outreach composes in one turn. Reverie chains are ~4 steps of ~90s and are already
persisted per thought. The answer decides whether this is one graph or a runner. Proposed
answer: **one workflow, curiosity, and a runner that could hold a second one later.** Do not
build the second one now.

**2. Where does a checkpoint live, and what is in it?** LangGraph's Postgres checkpointer
stores the whole graph state per super-step. For curiosity that state is: run_id,
correlation_id, kickoff prompt, material snapshot, worldview snapshot, the harness step
transcript so far, and which node is next. The FCC subprocess's own state is *not*
checkpointable -- a resumed run re-issues the harness request, it does not resume a subprocess.
So the resumable unit is a **node**, not a token. Acceptable: the expensive part of a killed run
is the material/worldview reads and the prompt (cheap) plus the FCC turn (expensive, restarted).
Resume saves the *cap slot* and the *continuation*, not the FCC minutes. That must be stated
plainly, or "resumable" will be read as "the LLM picks up mid-sentence".

**3. Does resuming re-spend the cap?** No -- the cap was spent at kickoff, the checkpoint
carries the run_id, a resume is the same run. This also retires the refund's dependence on a
clean `CancelledError`: a run that is checkpointed and not finished is simply resumed on the
next boot, whether or not the shutdown was graceful.

**4. HITL or not?** The knowledge-forge plan deferred "LangGraph HITL workflows" to v2. The
curiosity run has one natural interrupt point: before outreach (`_maybe_reach_out`), where
Orion has decided a finding is worth telling Juniper. A durable graph makes a human-in-the-loop
pause *possible* there. **Non-goal for the first patch** -- but the node boundary is placed so
that adding an interrupt later is one line, not a redesign.

**5. LangGraph, or a checkpoint table?** See "Options". This is the decision this document
exists to put in front of Juniper.

---

## Options

### A. LangGraph with the Postgres checkpointer (recommended, narrowly)

A new small service, `orion-workflow-runner`, owning the LangGraph dependency (it must not land
in hub or cortex-exec), a `StateGraph` for the curiosity run with nodes matching the existing
phases, `langgraph-checkpoint-postgres` against the existing `conjourney` database, and
resume-on-boot for any thread whose last checkpoint is not terminal.

- For: real durability with a maintained implementation; matches the README direction and the
  knowledge-forge v2 line verbatim; interrupts (HITL) come for free later; per-node retries are
  built in.
- Against: a new dependency and a new service; LangGraph's state model is a global dict, so the
  discipline of "nodes read/write named keys" has to be kept by review; the runner must be the
  only writer of its checkpoint tables.

### B. Hand-rolled checkpoint table in Hub

A `substrate_cognition_runs` table with `(run_id, phase, state_json)` written after each phase
of `_investigate`, and a resume branch on `start()`.

- For: no dependency, no service, ~150 lines.
- Against: reinvents a checkpointer badly (no interrupts, no per-node retry, no fan-out later),
  leaves the run inside Hub where it dies with Hub's redeploys, and does not put the kickoff in
  cortex.

### C. Do nothing; refund on SIGKILL

Make the refund robust to ungraceful shutdown (a boot-time sweep of "started, never finished").

- For: smallest possible change; recovers the cap.
- Against: recovers nothing else; the run still dies; the continuation note is still lost.

**Recommendation: A, scoped to the spike below, with C's boot-time sweep folded into the
runner's own resume-on-boot.** If the spike's checkpoint does not survive a container restart
end-to-end, stop and reconsider B.

---

## Proposed schema / API changes

### Kickoff from cortex

- New verb in `orion/cognition/verbs/`: `curiosity.investigate` (catalog entry only: name,
  description, the runner as its executor). `orion-cortex-orch`'s `decision_router` dispatches
  it to the runner over a new request channel instead of to cortex-exec.
- Hub's curiosity tick stops calling `execute_unified_turn` directly. It publishes an
  `orion:cortex:request` for `curiosity.investigate` with the same payload it builds today
  (material, worldview snapshot, run_id, correlation_id). **Cortex now sees every kickoff.**
  The tick loop keeps owning *when* (cooldown, cap, quiet hours) -- that is scheduling, not
  cognition.

### New channel + schemas (registered in both registry maps, `channels.yaml`)

```text
orion:workflow:run:request    WorkflowRunRequestV1    producer: orion-cortex-orch   consumer: orion-workflow-runner (single_consumer)
orion:workflow:run:state      WorkflowRunStateV1      producer: orion-workflow-runner consumer: orion-sql-writer, *
```

`WorkflowRunStateV1` (thin): `run_id`, `workflow` ("curiosity.investigate"), `thread_id`,
`node` (the node just completed), `next_node | None`, `status`
(`running|interrupted|completed|failed|resumed`), `resumed_from_node | None`, `correlation_id`,
`generated_at`. Every transition is one row in a new append-only table
`substrate_workflow_run_state` (sql-writer model + route + subscribe guard, same pattern as
PR #2124). LangGraph's own checkpoint tables are the runner's private storage, not a contract.

### The surface sees the sequencing

The runner is a fifth producer on `orion:attention:schema`, `process="workflow_run"`: one row
per node transition, `attended_id=run_id`, `attended_label=workflow`, `attention_reason=node
name`, `predicted_next=next_node`. This is the part the surface design promised: a state machine
that is *observable on the same table* as the processes it sequences, not a black box beside
them. Vocabulary stays process-owned; no shared enum.

### Nodes of the curiosity graph (matching today's phases, no new cognition)

```text
read_material -> read_worldview -> kickoff_prompt -> harness_turn -> read_turn_result
              -> publish_attention_row -> journal -> [decide_outreach] -> outreach -> done
```

`harness_turn` is the existing `HarnessRunRequestV1` RPC to the governor; on resume it is
re-issued with the same run_id (see MQ2). `decide_outreach` is the future interrupt point (MQ4).

### Env / config

Runner: `POSTGRES_URI`, `ORION_BUS_URL` (Tailscale IP, per AGENTS.md), `WORKFLOW_RUNNER_RESUME_ON_BOOT=true`,
`WORKFLOW_RUN_MAX_AGE_HOURS=24` (a checkpoint older than this is abandoned, not resumed --
a day-old curiosity run resuming into a different day's material is not continuity, it is a
ghost). Hub: `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=true` as the migration switch; `false` keeps
today's direct saga so the cutover is one flag.

---

## Files likely to touch

- `services/orion-workflow-runner/` (new): `app/main.py`, `app/graphs/curiosity.py`,
  `app/settings.py`, `requirements.txt` (langgraph, langgraph-checkpoint-postgres),
  `docker-compose.yml`, `.env_example`, `README.md`, `tests/`, `evals/`
- `orion/schemas/workflow_run.py` (new), `orion/schemas/registry.py` (both maps), `orion/bus/channels.yaml`
- `orion/cognition/verbs/curiosity_investigate.yaml` (new); `services/orion-cortex-orch/app/decision_router.py` (dispatch)
- `services/orion-hub/scripts/curiosity_investigation.py` (kickoff publish behind the flag; the phase functions become importable so the runner's nodes call the same code -- no second implementation)
- `services/orion-sql-writer/app/models/workflow_run_state.py` (+ worker/settings/env/tests, PR #2124 pattern)
- `orion/sentience_striving_program/instruments.yaml` (a `durable_runs` instrument with the claims below)
- `services/orion-sql-db/manual_migration_workflow_run_state.sql` only if sql-writer's `create_all` is not used for it

## Non-goals

- **Not a sequencer for the attention tick, reverie, or the goal producer.** Those stay on their
  own timers. The surface plus the bridge is how they relate.
- **Not a second implementation of the curiosity run.** The runner's nodes call the same
  functions Hub calls today; `curiosity_investigation.py`'s phases become importable, they are
  not copied.
- **Not HITL in the first patch** (MQ4). The interrupt point is placed, not wired.
- **Not a generic "cognition platform".** One workflow. A second one is a separate decision
  with its own evidence that it dies.
- **Not a change to what Orion thinks.** Same prompt, same material, same graph writes, same
  journal. This changes whether a run survives, and who saw it start.

## Acceptance checks

1. **A checkpoint survives a container restart, end-to-end, on the real rail.** Start a
   curiosity run through cortex; `docker compose restart` the runner between `kickoff_prompt`
   and `harness_turn`; the run finishes with the same `run_id`, one journal entry, one
   `TurnOutcome`, and `WorkflowRunStateV1` rows showing `resumed_from_node=harness_turn`.
   Named run id in the PR, before/after.
2. **Cortex sees the kickoff.** `orion:cortex:request` carries `curiosity.investigate` for every
   run; cortex-orch's decision log names it. Zero curiosity runs enter via the direct saga while
   the flag is on.
3. **The cap is not burned by a restart.** A day with a runner restart mid-run ends with
   `orion:curiosity:count:<day>` equal to the number of runs that *finished or failed on their
   own*, not the number started. Compare against today's 3-started/0-finished.
4. **The surface sees it.** `SELECT count(*) FROM substrate_attention_schema WHERE
   process='workflow_run'` is non-zero after one run, and `attention_reason` walks the node
   list in order for that `attended_id`.
5. **Nothing else moved.** Reverie, substrate, and cortex_turn lane counts per hour are
   unchanged before/after; the bridge's `voluntary_override_absent_reason` shares are unchanged.
   A change here means the runner touched something it should not have.
6. **Kill means kill.** With `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false` the runner receives
   nothing and the direct saga runs exactly as today (regression test on the flag).

## Failure modes that would be dangerous

- A resumed run re-issuing an *outreach* it already sent -> a message to Juniper twice. The
  `outreach` node must be idempotent on `run_id` (the outreach tag already is
  `uuid5(OUTREACH_TAG:run_id)`; the runner must check delivery before re-sending).
- A resume from a checkpoint whose material is stale -> Orion continues a thought about a
  world that moved. `WORKFLOW_RUN_MAX_AGE_HOURS` bounds it; the resumed prompt should say it was
  resumed and when.
- The runner becoming a second writer of curiosity state in redis/the graph -> two clocks. Only
  Hub's tick writes the cap/cooldown; the runner writes only its checkpoints and its state rows.

## Rollback

Flip `HUB_CURIOSITY_KICKOFF_VIA_CORTEX=false`; stop the runner. No table is read by anything on
the decision path; the checkpoint tables are the runner's alone.

---

## Recommended next patch

**The spike, and only the spike: prove acceptance check 1 on the real rail.** A runner service
with the curiosity graph's nodes calling Hub's existing phase functions, the Postgres
checkpointer, resume-on-boot, the `workflow_run` surface rows, and the cortex verb --
behind `HUB_CURIOSITY_KICKOFF_VIA_CORTEX` defaulting to `false`. Deploy it, run one
investigation through it, restart the runner mid-run, show the same run finishing. Then flip
the flag for one day and read acceptance checks 2-5.

If the restart test cannot be made to pass with LangGraph's checkpointer in one patch, that is
the answer to MQ5 and the fallback is Option B, which reuses everything above except the
dependency.

Two deterministic follow-ups belong to *this* branch, not the spike, and ship with this doc:

- `instruments.yaml`: `producers_live` re-recorded at 3 (curiosity lane 0 rows; the three runs
  today did not finish), and a new claim naming the cortex lane's thinness (871 rows, 2
  distinct narratives -- `select_actions` suppresses the same already-known target every turn).
- The surface design doc gets its 24h read.
