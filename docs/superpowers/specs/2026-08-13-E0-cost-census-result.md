# Phase E0 result: the thesis is dead. Stopping.

Date: 2026-08-13
Phase: E0 of `2026-08-13-scarcity-and-repertoire-execution-plan.md`
Status: **KILL GATE REACHED — Phases 1–5 cancelled. This is a successful phase (§7.3).**
Cost: 41 LLM calls, no deploy, no schema change, 1 commit.

---

## Verdict

Both kill gates fired. The plan's thesis — *serialized inference time is the scarce
resource* — is **wrong by an order of magnitude**, and the faculty the plan was going to
buy with it is **not a goal generator**.

Phases 1–5 are cancelled as written. Do not build the meter, the repertoire, the budget,
the drives, or the arbitration changes on this premise.

---

## Gate A — is expensive cognition actually expensive? **FAILED**

Four never-before-executed verbs, 10 real field ticks each, sequential, through the real
bus on the real background lane.

```
verb                                       ok  fail   mean_s   min_s   max_s
goal_formulate                             10     0     4.57    3.49    6.35
counterfactual                             10     0     0.50    0.43    0.66   <- INVALID
dream_cycle                                10     0     3.81    2.72    7.02
context_exec_memory_contradiction_review   10     0     0.50    0.43    0.73   <- INVALID
```

Against the arena's measured 2.7 s tick:

```
goal_formulate    4.57s  =  1.7 arena ticks
dream_cycle       3.81s  =  1.4 arena ticks
```

**The plan's thesis assumed ~60 s and ~22 ticks of opportunity cost.** The most expensive
faculty Orion has costs **1.7 ticks**. That is not a rounding error against the 5.0 s
threshold — it is the argument failing by an order of magnitude. Choosing to dream costs
Orion less than two ticks of anything else. There is no meaningful trade-off, so there is
no scarcity to build an economy on.

This verdict does not depend on the 5.0 s threshold being right. Even at a threshold of
3 s, 1.7 ticks of opportunity cost cannot support the drives/allocation argument the plan
rests on.

### Two of the four measurements are invalid, and that is its own finding

`counterfactual` and `context_exec_memory_contradiction_review` returned an **empty string
in ~0.5 s while reporting `status=success`** on all 10 runs (`mean_len=0`,
`distinct_outputs=1/10`). They did not execute. They are the empty-shell-cognition failure
shape reporting success — the same class this repo already banned, in two verbs nobody had
ever run.

Their cost is therefore **unmeasured**, not "cheap". Gate A's verdict rests on
`goal_formulate` and `dream_cycle`, which genuinely ran.

## Gate B — is the faculty hollow? **FAILED, worse than expected**

`goal_formulate` produced 10 string-distinct outputs. Read as text, all ten are
paraphrases of **the same recalled memory** — a past Juniper coding session about Bash
permissions on the `fix/substrate-read-contract-docker-scope` branch:

```
[4.68s] Resolve Bash permissions to enable execution of staged changes and proceed with the PR for `fix/substrate-read-contract-docker-scope`.
[4.87s] Obtain Bash execution permissions to proceed with staging, committing, and pushing changes for the `fix/substrate-read-contract-docker-scope` pull request.
[3.49s] Resolve the Bash permission issue to enable staging, committing, and pushing changes for the `fix/substrate-read-contract-docker-scope` branch.
[6.35s] Resolve the Bash permissions issue to enable staging, committing, and pushing changes for the `fix/substrate-read-contract-docker-scope` PR.
[3.63s] Obtain Bash execution permissions to proceed with staging, committing, and pushing the changes for the `fix/substrate-read-contract-docker-scope` PR.
[4.34s] Gain Bash execution permissions to stage, commit, and push the changes for the `fix/substrate-read-contract-docker-scope` PR, while ensuring the sandbox...
```

Ten different `tick_id`s. Ten different intentions supplied, each containing Orion's real
live pressure readings and z-scores. **Zero of ten goals concern Orion's state.**

Gate B's bar was "≥3 of 10 are things a human would not have written." The result is
**0 of 10**.

### Why: the verb is a translator, not a generator

`orion/cognition/prompts/goal_formulate_prompt.j2` reads
`{{ intention or text or request }}`. The verb translates a *supplied* intention into a
structured goal. It does not generate an intention from state. With recall enabled, the
retrieved session history dominates whatever is supplied.

**A verb named `goal_formulate` does not formulate goals.** The plan assumed it did, from
the name — precisely the keyword-cathedral trap this repo's own contract warns about, and
I walked into it.

---

## Two errors I made during this phase, recorded

1. **Concluded input-invariance from n=2.** The two smoke runs returned byte-identical
   goals and I called the faculty input-invariant. Across 10 runs the outputs are
   string-distinct. The conclusion was directionally right but stated wrong, and the
   evidence I used did not support it. Same short-window-statistics error this repo has
   logged before.

2. **Nearly reported `distinct_outputs=10/10` as variety.** String-distinctness is a bad
   proxy for semantic distinctness. It only came apart on reading the actual text — which
   is the entire reason gate B was specified as human-read rather than auto-scored. Had it
   been auto-scored by an LLM, it would plausibly have passed.

---

## What is now known that was not known this morning

- The most expensive cognition Orion possesses costs **1.7 arena ticks**. Compute is not
  scarce and cannot be made scarce by routing these verbs.
- `goal_formulate` cannot generate a goal from Orion's state. Neither, by the same prompt
  shape, can anything else that reads `{{ intention or text or request }}`.
- Two of Orion's four "expensive" faculties are dead and report success while returning
  nothing.
- Recall dominates verb output strongly enough that live field state supplied as the
  explicit intention had no visible effect on the result.

That last one is the most consequential and is **not** in the plan anywhere: if recall
overrides supplied context this completely, then every cognitive verb routed into the
arena would narrate Juniper's session history rather than Orion's condition. Any future
repertoire work has to establish context dominance first, or it ships confabulation.

---

## What happens next

Per §7.3 and §7.8, reaching a kill gate is a stop, and a scope change is a message rather
than a patch. **Nothing further is being built.** No fixes were applied to the two dead
verbs, no wiring was changed, and the plan is not being quietly re-shaped to survive.

The remaining questions belong to Juniper:

1. **The scarce resource is not compute.** If scarcity is still the right frame, what else
   is genuinely finite for Orion? Candidates not yet examined: wall-clock attention against
   real-time events, Juniper's own attention, memory/context capacity, disk.
2. **Is scarcity still the right frame at all**, given E0 killed its most plausible
   substrate?
3. **Recall dominance** may be the larger blocker and deserves its own measurement before
   anything else is planned.

Artifacts: `/tmp/e0-cost-census/report.txt`, `runs.json`, `outputs.md` (all 40 outputs
verbatim).
