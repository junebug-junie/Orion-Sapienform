# Giving Orion's substrate action something to perceive

Date: 2026-08-12
Status: **prune action next (no bus service required). B(2) contract merged, deferred.** Cognition-loop-adjacent per `CLAUDE.md` §0A; Juniper gave
explicit go-ahead, so this proceeds to implementation rather than sitting in proposal mode.

Two options are specified here — **B** (the verb acquires) and **C** (the dispatcher pre-fetches) —
because they are genuinely different capabilities with an order-of-magnitude difference in cost.

> **Revision (2026-08-12, same day).** This document originally recommended **against both**, on the
> grounds that the chain terminates — a better observation lands in a frame nothing consumes and is
> scored by a constant, so perception is not the binding constraint. That analysis still holds and is
> retained below in full.
>
> What changed is not the analysis but the available action. An operational thread in the same session
> surfaced a candidate action that **breaks the terminating chain**: autonomous Docker build-cache
> pruning. It has the one property every existing substrate action lacks — a real, continuous,
> per-action, externally-verifiable outcome. See "The decision" below. B(2) is selected because that
> action needs a real read, and the read is the same seam `substrate.inspect` needs.

## The decision (2026-08-12)

The deadlock this document described was: perception is pointless because outcomes are dead, and
outcomes are dead because the action has no consequence. Both halves are true, and neither can be
fixed from the other end while the only available action is a read-only LLM call.

**The way out is an action with a real consequence.** Autonomous build-cache pruning qualifies, and
it is the first candidate in this arc that does:

| property | substrate.inspect | build-cache prune |
| --- | --- | --- |
| real trigger | field pressure (real) | `disk_pressure` / `disk_capacity_pressure` — already live field channels |
| consequence in the world | **none** | disk usage changes |
| per-action outcome | none — 5 concurrent actions share one field delta | disk % before/after, unambiguous |
| outcome magnitude | quantized to 3 constants | continuous GB reclaimed |
| reversible | n/a | yes — cache regenerates by definition |
| cost of a wrong call | n/a | one slow rebuild |

That is a genuine `(action → outcome)` pair with a real magnitude — the thing the retroactive
attribution analysis went looking for at the start of this session and could not find, because every
existing action was a read-only LLM call with no consequence.

Juniper's own framing, which is the reason this is the right first mutating action rather than a
risky one: *"this is low stakes as i can rebuild any mistakes that orion deletes."*

### Measured basis for the trigger

Real numbers from the host, 2026-08-12:

```
/mnt/docker total      469G
used                   356G  (80%, after reclaiming a 25G orphaned volume)
build cache            271GB total, 15,037 entries, 0 active
  unused > 2 weeks     8,550 entries (57%)  ~= the 140.4GB Docker reports reclaimable
one prune reclaims     ~140GB  =  ~30 percentage points
observed growth        271GB over ~4 weeks  ~=  2pp/day
disk_threshold_watchdog alerts at 90% and never acts
```

**Trigger:**

```
act when   disk_used_pct >= 75%   AND   reclaimable_stale_cache >= 40GB
```

Both conditions, deliberately. 75% gives ~7 days of lead time at the observed 2pp/day before the
existing 90% watchdog alert — enough that a failed prune is noticed by a human rather than becoming an
incident — and pruning from 75% lands around 45%, a wide margin.

The **`AND` is the load-bearing part**, and it is the same lesson as `action_warrant`: do not gate on
an absolute level whose rest point is undefined. At 80% *with* 140GB of stale cache, pruning is highly
effective. At 80% with *zero* stale cache, pruning accomplishes nothing and the real condition is a
capacity problem needing a human. An autonomous action that fires and changes nothing is precisely the
theater this repo keeps deleting; the second condition makes the trigger actionable-only, and its
failure is itself the signal to escalate rather than act.

Note this is a backstop, not the primary mechanism. The primary fix is a BuildKit GC policy in
`/etc/docker/daemon.json` (`builder.gc`, `keepStorage: 60GB`), absent today — which is why the cache
grew unbounded to 271GB in the first place. With that in place this trigger should rarely fire.

### The prune action needs no bus service — corrected 2026-08-12

> **This section previously claimed the prune action requires B(2).** That was wrong, and the trace
> that disproves it came after the contract had already been written and merged (#1582, narrowed in
> #1584). Recorded rather than rewritten, because the error is the useful part: a capability was
> specced before checking which service already had it.

Two facts, traced:

- **`orion-cortex-exec` already has `/var/run/docker.sock`.** That is precisely why
  `skills.docker.ps_status.v1` and `skills.runtime.docker_prune_stopped_containers.v1` already work
  there as *local* verb adapters. `orion-biometrics` — which the contract had named as the read
  service — has no socket and could never have answered a Docker question.
- **`CortexRouteTemplateV1.cortex_verb` is an unconstrained `str`**
  (`orion/execution_dispatch/policy.py:20`). A dispatch route can name a `skills.*` verb directly,
  which makes `executor.py:2639`'s local-adapter branch fire — the same branch that
  `planner.py:194` prevents from ever firing for a step *inside* a non-skill verb.

**So the prune action is: a skill YAML + a local adapter + a route entry + a proposal template.** No
new bus contract, no new service, no new privilege. It runs in the one service that already holds the
capability.

This also avoids a real privilege expansion. Giving a hardware-telemetry service root-equivalent
Docker socket access, to answer a question a service that already has it can answer locally, would
have been cost with no benefit.

**B(2) is therefore no longer on the critical path.** It keeps its own justification —
`substrate.inspect` is a non-skill verb and genuinely cannot reach a skill, so perception for the
substrate probes still needs a service — but that is a separate, later, and now unblocked-by-nothing
piece of work. The contract is merged and narrowed to host filesystem usage, which is what
`orion-biometrics` genuinely owns.

C is unchanged and still argued against below: its predecessor moved zero outcome signals.

## Arsonist summary

`substrate.inspect` runs ~15,000 times a day. It is one LLM call, `max_recursion_depth: 0`, no tools,
and the only real data it receives is `motivating_dimensions` — **the same field-pressure numbers
that caused the proposal to exist.** It cannot learn anything that was not already in its prompt.

The obvious fix is to let it observe something real. Two ways to do that. But the chain it sits in is:

```
action → substrate_dispatch_results → feedback runtime → ∅
```

`substrate_dispatch_results` has exactly three readers: the dispatch runtime's idempotency replay, its
theater tripwire, and the feedback observation builder. **Nothing writes back to the field.** And
`FeedbackFrameV1.observations[].score` is a deterministic function of `config/feedback/
feedback_policy.v1.yaml`'s 8 hardcoded outcome scores — measured stddev **~1e-13** across the full
real table.

So a perfect Option B produces a better observation that lands in a frame nothing consumes and is
scored by a constant. Perception is not the binding constraint. **Close the loop first.**

## Current architecture

```text
substrate_field_state
  → orion-proposal-runtime            (L7)  + action_warrant tick gate (2026-08-12)
  → orion-policy-runtime              (L8)
  → orion-execution-dispatch-runtime  (L9)  build_cortex_request_envelope()
  → orion-cortex-exec                       substrate.inspect / summarize / observe
  → substrate_dispatch_results
  → orion-feedback-runtime            (L10) FeedbackFrameV1
  → ∅
```

### Trace 1 — how context reaches the prompt (Option C's surface)

`orion/execution_dispatch/envelopes.py::build_cortex_request_envelope()` is a single pure function
returning `{verb, mode, source, origin, dry_run, context, constraints}`. Its `context` dict **flows
unmodified** to cortex-exec's Jinja render. Verified end to end:

- `executor.py:1093-1095` — `_prompt_render_ctx()` does `render_ctx = ctx.copy()`, the whole ctx.
- The only stripping is `_PROMPT_BLOAT_CTX_KEYS = {recall_memory_bundle_debug, recall_fragments}`,
  and only when `recall_prompt_safe_ctx` is set. Nothing else is removed.
- `orion/cognition/prompts/substrate_inspect.j2:15` already renders a section literally titled
  `REAL TELEMETRY (the only real numbers you have for this target)`.

**This path has already been used for exactly this purpose.**
`services/orion-cortex-exec/tests/test_substrate_probe_prompt_grounding.py` documents the 2026-07-28/29
fix in its own words:

> these three prompts previously gave the model only a bare target_kind/target_id/allowed_scope and an
> instruction to avoid "invented telemetry values" — with no real data to ground anything in, that
> instruction was self-contradicting by construction. Confirmed in production
> (`substrate_dispatch_results`) that the model was fabricating specific technical claims (transformer
> "attention heads", token-throughput deltas) that don't correspond to any real signal in this
> substrate.

That fix wired `motivating_dimensions`/`priority_score`/`risk_score` through this same `context` dict.
Option C is the same move with a different payload.

### Trace 2 — how a verb calls a service (Option B's surface)

A "service" in a verb's `services:` list is **not an in-process call**. `executor.py:2728`:

```python
for service in step.services:
    reply_channel = f"orion:exec:result:{service}:{uuid4()}"
```

It is a bus request/reply. Ten services exist on this pattern; `RecallService` is the clean template
(`orion/bus/channels.yaml:293`):

```yaml
- name: "orion:exec:request:RecallService"
  kind: "request"
  schema_id: "RecallQueryV1"
  producer_services: ["orion-cortex-exec", ...]
  consumer_services: ["orion-recall"]
  single_consumer: true
- name: "orion:exec:result:RecallService:*"
  kind: "result"
  schema_id: "RecallReplyV1"
```

So Option B is a **bus contract change** under `CLAUDE.md` §6: two channel entries, two registered
schemas, a producer/consumer declaration, a `single_consumer` decision, and a service that answers.

### Trace 3 — two things that are NOT obstacles, and one that is

Checked because all three were assumed in earlier drafts and two of the assumptions were wrong:

- **`max_recursion_depth: 0` does not block multi-step.** `planner.py:225` reads
  `if max_depth and current_recursion_depth > max_depth` — `0` is falsy, so the guard never fires at
  all. Separately, even non-zero it gates verb→verb *recursion*, not step count;
  `_build_steps()` runs unconditionally after it. The `0` on `substrate.inspect` is a no-op.
- **`prior_step_results` does reach the template**, per Trace 1's ctx-copy finding.
- **BUT a skill step inside a non-skill verb cannot dispatch.** `planner.py:194` sets
  `verb_name = verb_def.get("name", "")` on *every* step, so a step inside `substrate.inspect` always
  carries `verb_name == "substrate.inspect"`. The local-adapter branch at `executor.py:2639` requires
  `not step.services and step.verb_name.startswith("skills.")`. It would never fire.
  **This kills the cheapest imagined version of B** — "just add a step-0 skill to the verb YAML" does
  not work.

### What real read-only skills already exist

Registered local adapters (`@verb("skills.…")` in `services/orion-cortex-exec/app/verb_adapters.py`),
executed **in cortex-exec's own process**, no Hub involved:

```
skills.biometrics.snapshot.v1      skills.gpu.nvidia_smi_snapshot.v1
skills.biometrics.raw_recent.v1    skills.storage.disk_health_snapshot.v1
skills.docker.ps_status.v1         skills.mesh.tailscale_mesh_status.v1
```

`BiometricsSnapshotVerb` (`verb_adapters.py:1456`) is representative: HTTP GET to the biometrics
service's `/snapshot`, normalized, and degrading to `{"available": False, "reason": str(exc)}` on any
failure. There is a working degradation pattern to copy rather than invent.

## Option C — the dispatcher pre-fetches

`orion-execution-dispatch-runtime` reads real state and adds it to the envelope's `context`, landing
in the existing `REAL TELEMETRY` prompt section.

**Blast radius:** `orion/execution_dispatch/envelopes.py`, `builder.py`, `__init__.py`, and 5 test
files. Three verbs affected (inspect/summarize/observe). Nothing else in the repo reads the envelope.

**Cost:** roughly 30 lines plus a fetch client and its timeout/degradation handling.

**What it is honestly:** passive perception. The dispatcher looks; the action does not. The model
reasons over real state instead of only the numbers that triggered it.

**Why it is not recommended now — the base rate.** Its predecessor is the 2026-07-28 grounding fix,
same seam, same rationale. That fix genuinely stopped fabrication, and it did **not** make behaviour
vary with state. Measured this session, after that fix was live:

| outcome signal | verdict |
| --- | --- |
| `field_delta` | improved ≈ worsened in every stratum (2.08/1.52, 5.94/5.64, 10.61/10.20) |
| `action_outcomes.surprise` | identical across all concurrent actions on **95.26%** of multi-dispatch ticks |
| `action_outcomes.success` | true **99.9%** of the time (54,540/54,597) |
| `substrate_dispatch_results.raw_len` | non-degenerate but unanchored — LLM output length |

One for one, adding real numbers to this prompt has not moved an outcome signal.

**Second problem — relevance.** `substrate.inspect`'s live targets are `capability:orchestration`,
`capability:transport`, `node:atlas`, `self:current`, `config/field/orion_field_topology.v1.yaml`, and
`policy:execution`. A biometrics snapshot is genuinely relevant to *one* of those. For the config-file
and policy targets it is noise wearing the label "REAL TELEMETRY" — which produces better-grounded
confabulation, not less of it. Any C implementation needs a target→source routing table, and that
table is the actual design work, not the plumbing.

## Option B — the verb acquires

`substrate.inspect` gains the ability to call a real read before it speaks.

Three possible implementations, in ascending cost:

1. **Add a step-0 skill to the verb YAML.** *Does not work* — Trace 3. Recorded so it is not
   re-proposed.
2. **A new bus service** (e.g. `SubstrateReadService`) declared in `services:`. Follows the
   `RecallService` template exactly. Requires: 2 channel entries, 2 schemas registered, a responding
   service (new, or an exec-request handler added to `orion-substrate-runtime`), producer/consumer
   contract, `single_consumer` decision, contract smoke. Additive — a name no existing verb declares,
   so the other 96 verbs are untouched by construction.
3. **Change the local-adapter dispatch keying** at `executor.py:2639` so a step can name its own
   skill. `call_step_services` is the single step loop for **all 97 verbs**. **Reject** — highest
   blast radius in the repo for the smallest capability.

**Recommended shape if B is chosen: (2).** It is a real contract, inspectable, and opt-in per verb.

**What it is honestly:** genuine epistemic reach. The action decides what to look at and looks. This
is the version that could make `substrate.inspect` mean something.

**Why it is not recommended now:** it is strictly more expensive than C and lands in the same
terminating chain. Build it when an outcome exists to be improved.

## Missing questions

1. **Does anything change if the model gets real data?** Unanswered, and answerable cheaply — C is a
   ~30-line experiment. The counter-argument is that its predecessor already ran that experiment with
   a different payload and the answer was no. Worth one honest re-run only if the outcome signals are
   fixed first, because today there is nothing that could register a change.
2. **What is the target→source routing table?** Neither option is well-defined without it. Six live
   target kinds; at most three have an obvious real source. The other three may indicate templates
   that should be killed rather than grounded.
3. **Should `substrate.inspect` exist at all?** If a target has no real source to inspect, a template
   proposing to inspect it is a keyword cathedral entry. This question is upstream of both options and
   is currently unasked.
4. **Does the dispatch runtime want I/O?** Option C puts an HTTP read in the motor-nerve path. It is
   the service whose whole job is "can actually send real actions" — adding a blocking read to it is a
   real architectural choice, not an implementation detail.

## Proposed schema / API changes

**Option C:** none. `context` is a free-form dict; adding keys is additive and the prompt already
tolerates missing keys (`test_substrate_probe_prompt_grounding.py` covers the degrade-gracefully
cases explicitly).

**Option B(2):** two `orion/bus/channels.yaml` entries + two `orion/schemas/registry.py` registrations
+ request/reply models, per `CLAUDE.md` §6. Producer `orion-cortex-exec`; consumer the answering
service; `single_consumer: true` following every sibling.

## Files likely to touch

**C:** `orion/execution_dispatch/envelopes.py`; a fetch client; `tests/test_execution_dispatch_envelopes.py`;
`services/orion-cortex-exec/tests/test_substrate_probe_prompt_grounding.py`; a target→source routing table.

**B(2):** `orion/bus/channels.yaml`; `orion/schemas/registry.py`; new schema models;
`orion/cognition/verbs/substrate.inspect.yaml`; the answering service; contract smoke; the same
routing table.

## Non-goals

- **Not routing at Hub's skill runner.** `services/orion-hub/scripts/skill_runner_catalogue.py` is a
  Hub chat-pane affordance (hardcoded English-prompt→verb dict, gated on `skill_runner_origin=True`).
  Using it would make the cognition substrate depend on the UI service. Recorded because it was
  proposed in this arc and correctly rejected.
- **Not routing at `orion-actions`.** A separate autonomous-action lane with its own dispatch; feeding
  the arena into it would create two competing motors.
- **Not touching `envelopes.py`'s hardcoded `read_only: True` or the 8 `no_*` constraints.** Both
  options stay entirely inside `approved_read_only`.
- **Not changing `orion/execution_dispatch/builder.py:167`'s scope gate or `allow_mutating_dispatch`.**

## Acceptance checks

Whichever is built:

1. **The observation is real.** A dispatch result must contain a value traceable to a live source, not
   to the proposal that triggered it. Checkable by diffing against `motivating_dimensions`.
2. **Degradation is honest.** With the source unavailable, the prompt says so and the model is not
   handed a fabricated or stale reading. Copy `BiometricsSnapshotVerb`'s
   `{"available": False, "reason": ...}` shape.
3. **Irrelevant sources are not attached.** For a target with no mapped source, no `REAL TELEMETRY`
   entry is added. Better an honest gap than confident noise.
4. **An outcome signal moves — or the result is recorded as negative.** This is the check that makes
   the work falsifiable. If perception lands and every outcome signal stays flat, that is a real
   finding about the template list and must be written down, not retried with a different payload.
5. **No regression in the three substrate prompts.** `test_substrate_probe_prompt_grounding.py`'s
   existing degrade-gracefully cases must still pass unchanged.

## What is next, in one line

**Build the prune action: a `skills.*` verb + local adapter in `orion-cortex-exec`, a route entry, and
a proposal template gated on `disk_used_pct >= 75% AND stale_cache >= 40GB`.** Nothing else is
blocking, and it is the first action in this arc with a real per-action outcome.

Everything below is prior reasoning, retained.

## Recommended next patch

**Superseded 2026-08-12 — see "The decision" at the top.** B(2) is being built, with the build-cache
prune as its first consumer, because that action supplies the real per-action outcome whose absence
the reasoning below correctly identified as the blocker. The ordering argument was right; what it
lacked was an action worth attributing.

The loop-closure work stays queued and stays necessary — a real outcome still has to reach the field
for anything to *learn* from it. It is no longer a prerequisite for having a measurable outcome at
all, which is what the prune action changes.

The original reasoning, retained because its analysis is unchanged:

**~~Neither B nor C. Close the loop first.~~**

`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-design.md:189-192`, carried
into the Sentience Striving charter's §8 and still unbuilt:

> A dispatched action's real outcome **perturbs the same field channels that were in the winning
> coalition** — relief on success, sustained pressure on failure — at the granularity the coalition
> actually formed at, not smeared across a generic bucket.

That is channel-scoped write-back, and it is what makes per-action attribution possible at all. It is
itself blocked on one thing: `FeedbackFrameV1.observations[].score` is a deterministic function of
`config/feedback/feedback_policy.v1.yaml`'s hardcoded scores (`orion/feedback/builder.py:290,303`),
with measured stddev ~1e-13. A hand-typed constant one layer downstream destroys the ability to derive
anything upstream.

So the real ordering is:

1. Make the feedback score a function of something real.
2. Close the loop — channel-scoped write-back to the field.
3. **Then** C as a cheap experiment, with outcome signals that can finally register whether it worked.
4. **Then** B, if C's answer is yes.

Doing 3 or 4 first produces better-grounded text that nothing reads, scored by a constant. That is the
definition of empty-shell cognition, and this document exists partly so that is a decision rather than
an accident.
