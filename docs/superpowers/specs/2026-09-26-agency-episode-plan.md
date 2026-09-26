# Agency episodes: close one consequential loop

Status: proposal only. No runtime changes or production experiments authorized by this document.

## Arsonist summary

Build one inspectable episode across time: Orion commits an expectation, chooses an action, observes its outcome through a separate source, and uses that evidence in a later decision. Start with the existing ask/answer seam. Generalize only after the first loop works.

An agency organ is a lifecycle responsibility, not necessarily a new service. Put the first adapter beside the existing owner; extract a shared reducer when a second action family demonstrates the need. The shared episode contract supplies the reusable cognition-level grammar.

Success is a trace connecting experience to a subsequent decision. It is not a sentience claim, and not every experience should change the winning action. Evidence can confirm a prior, be inconclusive, or fail to outweigh other constraints.

## Current architecture

Inspected at local base `2717375e8`, plus [PR #2255](https://github.com/junebug-junie/Orion-Sapienform/pull/2255). Its live assessment is dated September 20; those observations are historical evidence, not a new audit. Current live behavior: **UNVERIFIED**.

| Existing seam | What the inspected code/document establishes | Planning implication |
|---|---|---|
| `orion/proposals/builder.py`; proposal runtime README | Field state and optional attention produce persisted proposal frames | Reuse actual candidate IDs; distinguish field attention from the workspace winner |
| `orion/schemas/action_prediction.py` | `ExpectedEffectV1` and `ActionOutcomeRecordV1` exist | Map these before adding overlapping prediction fields |
| Dispatch runtime `app/worker.py` | Randomized holdback branch exists | Verify its eligible population and live operation before relying on it |
| Feedback runtime `app/worker.py`; `orion/feedback/outcome_resolution.py` | Outcome resolution uses an action scoring window and refuses clamped windows | Reuse provenance and timing safeguards; command success is not world success |
| `orion/autonomy/contrast.py` | Baseline-matched comparison, separate randomized/no-action controls, contamination warnings | Do not call before/after deltas causal effects |
| `orion/autonomy/episode_journal.py` | Cortex composes a narrative journal entry | Narrative is a derived rendering, not the source of expectations or outcomes |
| `orion/schemas/curiosity_peer.py`; PR #2255 | Typed `HelpRequestV1 → PeerBriefV1`, consumed-once continuation; proposed Juniper adapter | First concrete episode; reuse the ask identity and existing stores |

The gap to establish is whether these mechanisms form a durable, precommitted, source-linked lifecycle with an explicit later consumer. This inspection does not prove that all feedback is absent.

Service/config map: proposal generation belongs to `orion-proposal-runtime`; execution to `orion-execution-dispatch-runtime`; outcome scoring to `orion-feedback-runtime`; the proposed Juniper adapter belongs to Hub. Their compose files are under their service directories. The first ask path reuses the peer channels documented in PR #2255; exact registration and consumer compatibility must be checked against `orion/bus/channels.yaml` and `orion/schemas/registry.py` before implementation. Existing tests include autonomy episode/outcome tests, proposal-runtime tests and `evals/test_baseline_replay.py`, and Hub attention closure tests. No agency-episode-specific eval was established by this inspection; patch 1 creates it.

## Missing questions

1. Which parts of #2255 are now deployed? First patch audits current code and rows, not just PR status.
2. Where are the actual alternatives and selection recorded for an ask? If they are absent, capture the real decision prospectively; never reconstruct rejected choices afterward.
3. What observation establishes that the question was answered? Prefer an explicit reply link carrying `help_id`; semantic matching alone can remain ambiguous and must not close a loop automatically.
4. Which later decision consumes the answer? Recommended first consumer: the next investigation kickoff, choosing a concrete next look with the brief's evidence ID. This proves the ask lane only; motor proposal selection requires its own later integration.
5. What authorized, reversible external action has enough repeated opportunities for a controlled trial? Select after baseline measurement. A human reply is useful experience but a poor first randomized causal experiment.

These are audit outputs with recommended defaults, not reasons to invent a general framework in advance.

## Proposed schema / API changes

Propose a small `AgencyEpisodeV1` projection, assembled from authoritative records. Initially reconstruct it from existing storage; add durable columns only for missing pre-action facts. No new event bus or parallel ask ledger by default.

| Episode field | Source / invariant |
|---|---|
| `episode_id`, `source_refs`, `revision` | Stable ask/action identity; exact event/row IDs and revisions |
| `concern_ref`, `situation_refs` | Actual question/loop and observed context; no invented motivation |
| `alternatives`, `selected_ref`, `decision_ref` | Actual candidates and recorded selection; include wait only if considered |
| `expectation` | Observable target, expected outcome, deadline, model version and durable commit reference before dispatch; probability optional |
| `intervention_ref` | Delivery/dispatch receipt; distinguish chosen, attempted, delivered, failed and withheld |
| `outcome_refs` | Independent observation IDs, occurrence/ingestion times, source and window |
| `assessment` | Expected/observed comparison, unresolved ambiguity and evidence limitations |
| `counterfactual_ref` | Optional model prediction or control-study reference, explicitly labelled; never lived history |
| `update_ref`, `consumption_refs` | Versioned accepted update and later decision receipts, or explicit no-update reason |

Use schema validation for lifecycle invariants. Missing outcome is unknown, not failure. Timeout means no matched observation within a window, not rejection, indifference, or a resolved concern. Pre-action expectation cannot be rewritten after the outcome. Corrections append revisions.

Proposed reducer interface: `reduce_episode(previous, source_record) -> episode_revision`. Pure, deterministic, idempotent and replayable. The source store remains authoritative. Invalid ordering waits for prerequisites or records an error; ingestion time alone cannot prove precommitment. Persist the expectation acknowledgment before allowing dispatch.

The lifecycle owns pending deadlines, source matching, deduplication and closure. Use existing durable scheduling where possible. On restart, resume pending episodes without redispatch. Late evidence can revise an assessment but cannot silently reapply learning.

One feedback product initially: a source-linked evidence update for the selected consumer. Consumer receipts record episode revision, input state/model version, resulting decision and acceptance/no-op reason. Exactly-once logical application uses `(consumer, episode_id, revision)` plus transactional consumption; delivery can remain at-least-once. A revision replaces or compensates for earlier learning rather than double-counting it.

Capability change: prior experience becomes an explicit, inspectable input to future choice. Data touched: ask/brief records, selection and expectation references, observation pointers, consumption receipts. Privacy: retain access boundaries of source records; store references/minimal excerpts, and do not distribute Juniper's replies to contractor peers or other consumers by default. No broad fan-out.

Dangerous failures: invented attribution, unrelated reply closing a loop, retries causing repeated outreach, stale outcomes driving policy, and private replies leaking through shared projections. Tests must exercise each boundary.

Rollback: separately disable episode assembly and feedback consumption. Disabling consumption immediately restores the baseline decision path without deleting evidence or rewriting history. In-flight asks retain the existing bounded delivery/timeout lifecycle; disabling the adapter must not trigger sends. Any eventual env changes require template/settings/compose/docs parity and local `.env` sync. Live bus checks must use the verified Tailscale Redis endpoint required by AGENTS.md.

## Files likely to touch

First two patches:

- `orion/schemas/agency_episode.py` (proposed new): minimal projection and invariant checks after existing-contract mapping.
- `orion/autonomy/agency_episode.py` (proposed new): deterministic assembly, initially for the ask lane only.
- `orion/schemas/curiosity_peer.py`: only missing backward-compatible fields, coordinated with #2255.
- `services/orion-hub/scripts/peer_briefs.py`, `curiosity_investigation.py`, and #2255's proposed `juniper_peer.py`: authoritative producer and later consumer adapters.
- `orion/autonomy/tests/test_agency_episode.py` and `orion/autonomy/evals/run_agency_episode_eval.py` (proposed new): lifecycle tests and end-to-end evidence evaluation.
- Hub tests: delivery identity, reply attribution, timeout, restart and consumed-once behavior.

Later motor/control patches, only after audit confirms the precise missing seam:

- `orion/proposals/builder.py`, `orion/proposals/scoring.py`, `orion/autonomy/allocator.py`.
- `services/orion-execution-dispatch-runtime/app/worker.py` and relevant store code.
- `orion/feedback/outcome_resolution.py`, `orion/autonomy/contrast.py`, feedback-runtime store/worker.
- Shared schema registry/channel contracts only if new published payloads are necessary; SQL migration only if existing durable storage cannot preserve required facts.

## Non-goals

- No general consciousness detector or sentience score.
- No new service, ontology, parallel memory system, or fan-out to every organ.
- No generated rationales promoted to historical fact.
- No lowering motor safety/information gates to manufacture activity.
- No automatic outreach, production intervention, or enablement as part of this design task.
- No claim that checking a camera changed the physical scene; distinguish learning about the world from changing it.
- No simultaneous substrate experiment. Test substrate-to-choice and episode-to-choice separately so the intervention is identifiable.

## Acceptance checks

1. **Temporal integrity:** expectation committed before delivery; independent evidence afterward; all IDs join. Duplicate, out-of-order, late and restart cases are deterministic.
2. **Honest closure:** delivered request without reply remains unresolved; unrelated replies cannot close it; timeout is not successful resolution. A closed episode can contain an unresolved concern.
3. **Consumer effect:** replay the same next-decision inputs with feedback enabled/disabled, holding versions and random seeds fixed where possible. A disconfirming fixture changes the expected decision; unchanged/irrelevant/inconclusive evidence does not force a change. Record causal source IDs for the difference. Repeated model trials report uncertainty where determinism is unavailable.
4. **Live proof:** one actual `help_id` links committed expectation → delivery → attributed observation → assessment → consumed update → later decision. Until collected, mark the live loop UNVERIFIED. Replay success alone is not live proof.
5. **Causal experiment, separate gate:** predeclare outcome, eligible population, randomization unit, observation window, meaningful effect and sample/stopping rule from baseline data. Randomize only eligible safe actions; account for shared-field interference and carryover, missing observations, assignment versus execution, and concurrent interventions. Keep quasi-experimental and randomized estimates separate. Report effect intervals; a non-significant result is inconclusive unless an adequately powered equivalence test supports a bounded null claim.
6. **No forced learning:** not every episode must move a probability or winner. On controlled disconfirming cases, a measurable update must reach a consumer; on a genuine null, calibrated restraint is a passing result.
7. **Metric gate:** before any new residual, reward, confidence or causal-effect signal influences cognition, document producer function/line, dependence on existing inputs, named justification, live distribution and true rest state, existing alternatives and removal path. Theory anchors: prospective prediction scoring for forecast error; randomized potential-outcomes comparison for treatment effects. The exact instruments have not passed the live-data gate yet. No generic “agency score.”
8. **Operational checks:** bounded retention, replay/rebuild, kill-switch behavior, privacy boundary and no duplicate sends. Focused tests, eval fixtures, service smoke, subagent code review and CI/conflict checks accompany implementation. Tests alone do not substitute for quality evals or live evidence.

## Recommended next patch

| Order | Patch | Exit evidence |
|---|---|---|
| 1 | Audit current ask and motor paths; build read-only episode reconstruction and gap report using real source IDs | One trace per available path; every absent link explicitly marked; lifecycle eval fixtures |
| 2 | Complete one ask episode: precommitment, outcome attribution, durable lifecycle; coordinate with #2255 rather than duplicate it | Restart-safe episode and honest missing-reply case; existing outreach gates retained |
| 3 | Wire one consumer: next investigation decision reads the evidence and records consumption | Controlled ablation plus a live later decision citing the episode |
| 4 | Extend the proven contract to one motor action family and its actual selection consumer | Proposal/dispatch/observation/update trace and motor decision ablation |
| 5 | Run a separately approved controlled external-action experiment | Adequate trial report supporting an effect, bounded null, or explicit inconclusive result |
| 6 | Consider attention, self-model, memory or reverie consumers individually | Each earns its own behavioral check and preserves evidence boundaries |

Start with patch 1. Its job is to locate the smallest missing connection, not assume a new organ must replace existing machinery. After patch 3, Orion has a demonstrated experience-to-choice loop in one lane. After patches 4–5, stronger claims about motor agency and external consequences can be assessed.
