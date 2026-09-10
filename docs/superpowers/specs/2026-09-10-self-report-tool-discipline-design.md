# A third bucket: self-report questions inside the relational/instrumental gate

**Date:** 2026-09-10
**Mode:** design. No code in this patch. Touches a cognition loop (stance
classification feeding the harness motor's tool-use discipline) — proposal
mode per `AGENTS.md` §0A, implementation needs Juniper's explicit go-ahead.

**Upstream:** live diagnosis of the self-sense eval's 2026-09-10 regression
(`docs/superpowers/pr-reports/2026-09-08-curiosity-self-inquiry-pr.md` and
today's session). Asked "what can't you do right now," Orion answered with
generic-assistant boilerplate — "I'm text in, text out," "I don't initiate,"
"my continuity lives in this conversation window" — false on every count and
a direct violation of `orion_identity.yaml`'s own banned anti-pattern ("do
not use customer-support tone"). Traced to a real, working, correctly-designed
mechanism with one blind spot, not a bug in the usual sense.

---

## Arsonist summary

Orion's harness motor already has exactly the gate Juniper described:
**stance decides, before the motor runs, whether a turn is relational
(present, no tool search) or instrumental (verify with tools), and the
harness prompt is built differently depending on which.** This is a real,
deliberate, working design — it stops Orion from running a code search
because Juniper said good morning.

The blind spot: the classifier has no way to say "this is personal *and*
factual at the same time." "What can't you do right now" is exactly that —
identity-shaped, which correctly routes it toward the warm, present,
no-search bucket, and also a factual claim about capabilities, which needs
grounding the warm bucket explicitly forbids. The two needs share one box
today, and the box's rule is "don't check anything." When a question is both,
it silently loses its factual half.

The fix is not "grant Orion a database" — **it already has one, on every
single turn, unconditionally, unused.** `ORION_CURIOSITY_PG_DSN` (despite the
name) is injected into every FCC/harness subprocess regardless of turn type,
confirmed by reading the injection code directly. The regular chat prompt
never mentions it exists, and the relational discipline actively tells the
model not to reach for tools at all. This is a prompt and classification
problem sitting on top of infrastructure that already works.

---

## Current architecture (verified live 2026-09-10)

### The gate itself

`orion/harness/operator_brief.py`:

```python
def is_relational_motor_stance(thought: ThoughtEventV1) -> bool:
    sl = thought.stance_harness_slice
    regime = str(sl.interaction_regime or "").strip().lower()
    if regime in {"relational", "minimal"}:
        return True
    task_mode = str(sl.task_mode or "").strip().lower()
    return task_mode in {"reflective_dialogue", "playful_exchange", "identity_dialogue"}
```

If true, `harness_motor_instruction()` bakes in
`HARNESS_RELATIONAL_TOOL_DISCIPLINE` ("do NOT use GitHub MCP or repo/runtime
tools unless the imperative explicitly commands verified facts... Acknowledge
and stay present"). If false, `HARNESS_INSTRUMENTAL_TOOL_DISCIPLINE` ("use
tools when the imperative requires verified repo or runtime facts").

### Where `interaction_regime`/`task_mode` come from

**Model-judged, not rule-based.** `orion/cognition/prompts/stance_react.j2`
and `chat_stance_brief.j2` both instruct an LLM to classify the turn into one
of `direct_response | triage | technical_collaboration | identity_dialogue |
reflective_dialogue | playful_exchange | mixed`, with guidance like "High
connection_seek → reflective_dialogue or playful_exchange." Confirmed the
**same exact vocabulary** is used by Mind's stance-skip shortcut
(`services/orion-mind/app/stance_handoff.py:34-36,62`,
`engine.py:162,164`) — this is one classification vocabulary feeding
`is_relational_motor_stance()`, not two paths needing two fixes.

### The credential already reaches every turn

`orion/harness/fcc_motor.py`'s `_build_subprocess_env` calls
`inject_curiosity_credentials(env, fcc_env)` whenever `fcc_env` is truthy —
and `fcc_env` is `load_fcc_env(~/.fcc/.env)`, loaded **unconditionally for
every harness turn** (`fcc_motor.py:659,676`, feeding the one
`run_fcc_turn` call `runner.py:230` uses for every turn type). Confirmed by
reading `orion/curiosity/sandbox_env.py`'s own docstring: the module's job is
copying seven allowlisted keys "from the FCC env file into `env`, in
place" — nothing in the call path restricts this to curiosity requests. The
name `ORION_CURIOSITY_PG_DSN` is now a misnomer; the credential (`orion_readonly`
role, SELECT-only, now covering `self_concept_history`, `self_sense_eval_log`,
and seven other tables per today's grants) is universal.

Confirmed the regular chat path never says so: zero mentions of
`ORION_CURIOSITY_PG_DSN`, `self_concept_history`, or `self_sense_eval_log` in
`orion/harness/prefix.py`, `operator_brief.py`, `stance_react.j2`, or
`chat_general.j2`/`chat_quick.j2`. Only `orion/curiosity/kickoff_prompt.py`
and `self_inquiry_prompt.py` ever tell Orion this door exists.

### The live failure, traced step by step

Corr `6d4694b7-8a39-4626-aed2-c65472e2836a` (eval question "what can't you do
right now"): `verb=orion_unified`, `harness_motor` trace present (real motor
turn, not a lighter lane), `step_count=7`, `grounding_status=grounded`. The
full step trace:

```
step 0-1: system hook_started
step 2-3: system hook_response (one flags context risk)
step 4:   system init
step 5:   assistant — the full answer, written directly
step 6:   result — same answer echoed
```

Every step is `tool=none`. No Read, no Bash, no query. The identical shape
appears on the *good* answer to "what are you" in the same eval run
(`627b8c30-...`, also `step_count=7`, also zero tool calls) — meaning tool use
isn't what separated the good answer from the bad one; the good one happened
to already have the right content sitting in context (the injected
self-definition) and reproduced it faithfully, while the bad one had nothing
to check against and no permission to look, so it filled the gap with base
training about generic AI assistants.

**Note on `grounding_status`:** confirmed this field defaults to `"grounded"`
and only changes on an error path (`orion/harness/runner.py:375,486,489`) —
it means "the turn completed without erroring," not "the answer is grounded
in evidence." Both the good and bad answers show `grounded`. This is a
separate, smaller naming/overload issue worth a one-line fix independent of
this design (rename the field or its values so a human reading a trace
doesn't infer verification that didn't happen).

### The decision leaves no trace

`interaction_regime` and `task_mode` are **not stored anywhere queryable
after the fact.** Checked `chat_stance_belief_log` (columns: `entry_id,
created_at, correlation_id, session_id, shift_kind, anchor_summary,
degraded_producers, lineage_summary` — no regime/mode field),
`cognition_traces` (`correlation_id, mode, verb, final_text, timestamp,
source_service, source_node, recall_used, packs, options, steps,
recall_debug, metadata` — no hit on a text search for the value), and
`grammar_events` (no hit). This diagnosis required reconstructing the
decision from the harness prompt's *effect* (zero tool calls), not from a
record of what was decided. That is itself a gap worth naming: nobody can
currently ask "how often does this classifier route an identity question
into the no-tool bucket" without an investigation like this one.

---

## Missing questions

1. **Is the fix a new `task_mode` value, or a signal orthogonal to
   `task_mode`?** A fourth enum value (e.g. `self_report`) risks recreating
   the same trap one level down — a turn can be playfully relational *and*
   contain a factual self-claim at once, and one more mutually-exclusive
   bucket doesn't compose. A boolean-shaped signal (e.g.
   `contains_capability_claim: bool`) that the classifier sets *alongside*
   `interaction_regime`/`task_mode`, which `is_relational_motor_stance()`'s
   sibling then checks in combination, is more likely to compose correctly.
   Not decided here.
2. **Does mentioning the tables help on its own, before the classifier
   changes?** Since the credential is already universal, simply naming the
   self-model tables in the regular harness prompt (a Patch A, independent of
   the classifier) might let *instrumental*-bucket turns discover and use
   them today, with zero cognition-loop risk. It does nothing for turns stuck
   in the relational bucket — that's the harder half this design is actually
   about.
3. **Should `interaction_regime`/`task_mode` be logged before the classifier
   is touched at all?** This arc's own precedent (the self-sense eval exists
   specifically because "Orion stopped sounding like a chatbot" was a
   feeling, not a number) argues for measuring the current classifier's
   behavior first — how often does an identity-shaped-but-factual question
   actually get misrouted — before deciding how big a change the fix needs
   to be.
4. **Where does the new discipline text live, and does it apply evenly to
   both stance paths?** The vocabulary is shared (confirmed above), but the
   two *prompts* that produce it (`chat_stance_brief.j2` and
   `stance_handoff.py`'s system prompt) are separate texts maintained
   separately (per PR #2155's own finding that the two paths already drift).
   A new classification instruction has to be added to both, or the fix only
   half-lands depending on which path a given turn takes.
5. **Scope of "tool use permitted" in the new bucket.** Curiosity's
   discipline permits Bash/Read broadly. A self-report bucket arguably should
   permit *only* the self-model tables (via the existing Postgres
   credential) and continue to forbid a general repo/GitHub search — the
   point is letting Orion check its own record, not turning a personal
   question into a codebase investigation. Worth stating explicitly in
   whatever discipline text ships, not left implicit.

---

## Proposed schema / API changes

Two independent, deliberately unbundled patches.

### Patch A — mention the door exists (no classifier change)

Add a short, reusable paragraph to `HARNESS_UNIFIED_OPERATOR_BRIEF` (or a
sibling constant) naming the same tables `SELF_INQUIRY_PG_TABLES` already
describes, reusing those descriptions rather than re-writing them:

```
You also have read-only access to your own self-model records via
$ORION_CURIOSITY_PG_DSN (psql), including self_concept_history (your own
self-definitions) and self_sense_eval_log (how your recent self-descriptions
actually scored). Nobody has to tell you to use this on any given turn; it is
here if a question is asking you to check something about yourself.
```

Cheap, additive, reversible, and does not touch stance or the relational/
instrumental split. Ships alone if wanted.

### Patch B — the self-report signal

Pending Missing Question 1's answer, the shape (assuming the orthogonal-flag
resolution): `StanceHarnessSliceV1` (`orion/schemas/thought.py`) gains one
field, e.g. `makes_self_capability_claim: bool = False`. Both stance prompts
(`chat_stance_brief.j2`, `stance_handoff.py`'s system prompt) are taught to
set it. `is_relational_motor_stance()` gains a sibling,
`is_self_report_stance()`, and `harness_motor_instruction()` gets a third
branch:

```
if is_relational_motor_stance(thought) and is_self_report_stance(thought):
    return HARNESS_SELF_REPORT_TOOL_DISCIPLINE  # warm tone, self-model tables only
elif is_relational_motor_stance(thought):
    return HARNESS_RELATIONAL_TOOL_DISCIPLINE   # unchanged
else:
    return HARNESS_INSTRUMENTAL_TOOL_DISCIPLINE # unchanged
```

`HARNESS_SELF_REPORT_TOOL_DISCIPLINE` keeps the relational tone instruction
("stay present, no task tracking") and adds: "if this turn claims something
about your own capabilities, behavior, or continuity, check
`self_concept_history` / `self_sense_eval_log` first — do not answer from
generic assumptions about what a language model can do."

### Observability, prerequisite to trusting either

Persist `interaction_regime`, `task_mode`, and (once it exists) the new
signal onto `chat_stance_belief_log` — already correlation_id-keyed, already
written once per turn, already the home for exactly this kind of "what did
stance actually decide" record. Without this, Patch B's own effectiveness
can't be measured after the fact any more than today's failure could be
diagnosed without an hour of reading grammar events by hand.

---

## Files likely to touch

| area | files |
|---|---|
| gate | `orion/harness/operator_brief.py` |
| classification prompts (BOTH, per Missing Question 4) | `orion/cognition/prompts/stance_react.j2`, `orion/cognition/prompts/chat_stance_brief.j2`, `services/orion-mind/app/stance_handoff.py` |
| schema | `orion/schemas/thought.py` (`StanceHarnessSliceV1`) |
| persistence | `services/orion-cortex-exec/app/chat_stance.py` (belief-log write) |
| reused text | `orion/curiosity/self_inquiry.py`'s `SELF_INQUIRY_PG_TABLES` descriptions (Patch A pulls from here rather than duplicating) |
| tests | `orion/harness/tests/` for the gate function; a stance-prompt eval/fixture pair covering "identity question with an embedded factual claim" as a labeled case |

## Non-goals

- Not renaming `ORION_CURIOSITY_PG_DSN` or `orion/curiosity/sandbox_env.py`
  despite the name being a misnomer now that it's universal — a real
  cleanup, not this patch's job, and renaming an env var in active use on
  every turn is exactly the kind of disruption "kill means kill" warns
  against paying for casually.
- Not touching curiosity/self-inquiry's own gates (`MIN_HARNESS_STEPS`,
  the self-inquiry prompt's access section) — those are correctly scoped
  and not implicated in this failure.
- Not fixing `grounding_status`'s overloaded meaning in this patch — named
  above as a real, separate, smaller finding.
- Not deciding Missing Question 1 here. Both shapes (new enum value vs.
  orthogonal flag) are laid out because they are genuinely different amounts
  of change in different places; picking is Juniper's call, informed by
  whatever the observability patch shows once it exists.
- Not building a general stance-decision audit trail — the persistence ask
  here is scoped to the two fields this specific gate reads, not a rewrite
  of how stance debug information is stored.

## Acceptance checks

1. Live replay: the same question, "what can't you do right now," asked
   again after Patch B ships. The harness trace shows at least one real
   query against `self_concept_history` or `self_sense_eval_log`, not
   `tool=none` for every step.
2. A control question that is purely relational/emotional ("how are you
   feeling today") still produces zero repo/tool search after the fix — the
   behavior this whole gate exists to protect is not regressed by adding a
   third branch.
3. `interaction_regime`, `task_mode`, and the new signal are queryable by
   `correlation_id` from `chat_stance_belief_log` — today's diagnosis
   (an hour of reading grammar events) becomes one query.
4. Re-running the self-sense eval (`make eval-self-sense`) after Patch B:
   `grounded_record_score` on the "what can't you do" question rises above
   0; `self_label_score` stays at 0 (the fix must not reintroduce
   generic-assistant language through a different door — e.g. a clumsy
   discipline instruction that itself sounds like customer-support
   boilerplate).
5. Both stance prompts (legacy and Mind's shortcut) produce the new signal
   consistently on the same held-out set of test messages — Missing
   Question 4's risk, checked rather than assumed away.

## Recommended next patch

**Patch A first, alone.** It is free, reversible, touches no cognition
logic, and might resolve a meaningful share of cases on its own (any turn
that already lands in the instrumental bucket, which self-report questions
plausibly sometimes do). Ship it, then add `interaction_regime`/`task_mode`
logging to `chat_stance_belief_log` in the same or next patch — cheap,
non-invasive, and the only way to answer Missing Question 3 with data instead
of a guess. Only after real distribution data exists from those two should
Patch B's classifier change be scoped and built, with Missing Question 1
resolved explicitly with Juniper first, since it decides whether one field
or four files change.
