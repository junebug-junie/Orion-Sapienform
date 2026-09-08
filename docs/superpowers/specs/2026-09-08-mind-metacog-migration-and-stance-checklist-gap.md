# Mind's metacog migration, two incidents, and two open findings

**Date:** 2026-09-08
**Status:** Incidents closed and deployed. Two findings below are open decisions, not bugs — see each finding's own status line.
**Related PRs:** #2147 (route migration), #2150 (thinking-suppression fix)
**Related memory:** the two live incidents below are the concrete evidence behind the metric-quality-gate lesson already in `CLAUDE.md` 0A ("check whether it can ever return to a genuine rest/calm state").

## Arsonist summary

Chat turns were dying on ~97% of live turns (`mind_quality=fallback_contract_only`). Root cause: Mind's `semantic_synthesis` and `stance_handoff` LLM calls were pinned to the `quick` route (circe, 4 slots), which was saturated by unrelated traffic. Moved them to `metacog` (confirmed idle at the time), which triggered a second, unrelated failure — metacog's model does unsuppressed chain-of-thought unless explicitly told not to, and was burning its whole token budget on reasoning before ever emitting JSON. Both fixed and verified live. Along the way, a separate live incident (three services crash-looping on a stale build from an unrelated PR) was found and fixed in the same session. Two design findings surfaced during the investigation are recorded here as open, deliberately not acted on tonight.

## Incident 1 — `quick` lane saturation

**Symptom:** `docker logs orion-mind` showed `RPC timeout waiting on orion:mind:llm:reply:...` on `semantic_synthesis`, 60s budget exhausted, falling open to `mind_quality=fallback_contract_only` on ~97% of sampled live turns.

**Root cause:** `MIND_SEMANTIC_MODEL_ROUTE` and (in the live-deployed `.env`, diverged from its own `chat` code default) `MIND_STANCE_MODEL_ROUTE` both pointed at `quick` (circe port 8013, 4 llama.cpp slots), which was saturated — confirmed live via the gateway's own admission logs: `inflight=8 waiting=6-11`, 30-115s queue waits before a request even started. `metacog` (circe port 8012) was confirmed live-idle (4/4 free slots, zero real queued/overloaded events) at the time.

**Fix (PR #2147):** Moved `semantic_synthesis` and `stance_handoff` to `metacog`. Since metacog now carries live-turn traffic, retagged the real background callers already sharing that lane — `orion-vision-council`, `orion-memory-consolidation`'s turn-change classifier, and `orion-cortex-exec`'s daily metacog/journal draft pipeline — to `metacog_background`, so they yield slot slack via the gateway's existing `priority_admission.py` mechanism instead of competing evenly. Deliberately left `orion-thought`'s `reverie_metacog_background_enabled` (default off) untouched — a standing, deliberate Juniper experiment to observe reverie's unmitigated failure rate, not something this patch should silently flip.

**Residual risk flagged in the PR itself:** metacog's "idle" reading was one live snapshot, not a load test against 3 sequential Mind calls per turn plus background traffic. This risk materialized within the hour — see Incident 2.

## Incident 2 — metacog's unsuppressed thinking

**Symptom:** Immediately after Incident 1 deployed, chat turns kept failing — but differently. Replies now came back well inside the 60s budget (10-30s), yet still ended in `fallback_contract_only`, with `authorization_reasons: ["json_parse_failed", "json_parse_failed"]`.

**Root cause, confirmed live:** metacog's served model defaults to inline `<think>...</think>` chain-of-thought unless a caller explicitly disables it via `chat_template_kwargs.enable_thinking`. Confirmed three ways: the gateway's own `THINK_HOP` trace logs showed `inline_think_len` 2000-4000+ characters with empty final `content` on every failing call; the served chat template itself only suppresses thinking when `enable_thinking` is explicitly `false` (`{% if enable_thinking is defined and enable_thinking is false %}`); and `MindLLMClient.request_json`'s existing `thinking` parameter was a pure no-op — its old body (`if thinking: options["thinking"] = True`) set a key nothing in `orion-llm-gateway` ever reads. `active_frontier_judge` (already on metacog before this migration) only survives this because its token budget (3072) happens to outlast unsuppressed reasoning; `semantic_synthesis` (2048) and `stance_handoff` (1536) do not.

**Fix (PR #2150):** Fixed at the actual mechanism, not per call site. `MindLLMClient.request_json`'s `thinking=False` (the default, used by `semantic_synthesis` and `stance_handoff`) now actually sets `chat_template_kwargs.enable_thinking=False`; `thinking=True` (`active_frontier_judge`'s setting) leaves the model's own default alone, preserving its exact current behavior. A first-pass version of this fix hand-rolled the same dict at each of the two call sites; code review correctly flagged that as fixing the symptom twice instead of the cause once, and it was moved into the client before merge.

**Verified live:** two direct smoke-test calls to the rebuilt container's `/v1/mind/run`, both returning `mind_quality: meaningful_synthesis`, `authorized_for_stance_skip: true`, and genuine prompt-specific synthesis (not the system prompt's own few-shot example echoed back). Gateway trace confirmed `inline_think_len=0` on fresh calls.

## Incident 3 (unrelated) — three services on a stale build

Found mid-response to Incident 2: `orion-athena-attention-runtime`, `orion-athena-proposal-runtime`, and `orion-athena-feedback-runtime` were crash-looping on live `pydantic.ValidationError: Extra inputs are not permitted` for `sustained_load_pressure_channel` / `sustained_load_pressure_node_id` on the `extra="forbid"` `FieldStateV1` model.

**Root cause:** unrelated PR #2149 added those two fields to the shared `orion/schemas/field_state.py`. The deploy rebuilt the two *producers* (`hub`, `field-digester`) but not the three *consumers* — a consumer-first migration violated on a forbid model (see `feedback_additive_schema_fields_are_a_consumer_first_migration_on_forbid_models.md`). A first rebuild attempt from a stale worktree branch (cut before #2149 merged) silently used a pre-fix copy of the shared schema — caught by checking the file *inside* the freshly-built container rather than trusting the build log.

**Fix:** Rebuilt all three from a worktree cut fresh off current `main`. Verified live: real, successful ticks flowing (`attention_frame_saved`, `proposal_frame_saved`, `feedback_frame_saved` with actual outcome scoring), zero `ValidationError` since. No code changed — no PR, this was pure operational recovery.

## Open finding 1 — the active frontier's 13-field score vector is mostly decorative

**Status: PARKED. Not pursuing without a concrete failure case.**

`orion/mind/synthesis_v1.py`'s `AppraisalFeatureVectorV1` has 13 fields (`risk`, `freshness`, `confidence`, `actionability`, `unresolvedness`, `interaction_cost`, `evidence_strength`, `identity_relevance`, `redundancy_penalty`, `source_tag_penalty`, `source_corroboration`, `relationship_leverage`, `current_turn_relevance`). Live-checked: only `confidence` and `current_turn_relevance` are ever non-zero, because the system prompt's one worked example (`appraisal.py:80`) only ever demonstrates those two keys, and the model copies the example every call. Confirmed via repo-wide search: zero code anywhere reads any of the other 11 fields. `filter_active_frontier()` (`guardrails.py`) doesn't rank or select by `score` or `features` at all — it applies three content-validity gates (source-tag label, claim-id validity, evidence-ref validity) and truncates to 8 in whatever order the model already produced.

A tiered remediation plan was drafted (5 fields fixable deterministically today from data the model already produces; 4 needing real new plumbing; 2 genuinely blue-sky) and visualized as an artifact (`Two of Thirteen`, session-local, not committed to this repo). **Decision: do not build it.** Populating the dead fields in isolation would satisfy the schema without changing a single decision Orion makes — exactly the "keyword cathedral" / empty-shell-cognition pattern `CLAUDE.md` 0A warns against, since no consumer exists or is proposed alongside it. Revisit only if a specific, observed case surfaces where the frontier promoted or suppressed the wrong matter in a way a real score would have caught.

## Open finding 2 — two "decide Orion's stance" prompts, only one has real rules

**Status: Noted, not urgent. Revisit if the ratio below changes.**

`ChatStanceBrief` (the object that sets Orion's tone, question-asking behavior, and identity-foregrounding for a reply — read as primary controls by the actual answer-writing prompt, `orion/cognition/prompts/chat_general.j2`) can be produced two ways:

1. **Legacy path** — `orion/cognition/prompts/chat_stance_brief.j2`, in `orion-cortex-exec`. Contains an explicit, near-deterministic decision table keyed on two judgment axes (`interface_cost`, `connection_seek`): e.g. high interface-cost + low connection-seek forces `task_mode: direct_response` and a specific `response_priorities` list; high connection-seek forces `conversation_frame: reflective/playful_relational` and a different fixed list. Confirmed live: `chat_stance.py` has zero code computing or checking either axis — the entire table is prose in the prompt, executed by an LLM, with no backstop verifying compliance.
2. **Mind's shortcut** — `services/orion-mind/app/stance_handoff.py`'s `_STANCE_SYSTEM`. Produces the same schema, taken directly (skipping the legacy call entirely) when `mind_authorized_for_stance_skip=True`. Contains none of the above table — no mention of `interface_cost`, `connection_seek`, or the branching rules. It infers `task_mode`/`conversation_frame`/etc. from the active frontier with materially less guidance than the path it's replacing.

Incidents 1 and 2 make the shortcut path succeed more often (that was their point), which means more turns now take the *less-instructed* of the two paths for the same decision, not more. **Live-measured** (`mind_runs`, last 3 hours, across all Mind-invoking session types — see caveat below): 3 of 19 runs took the shortcut; 16 did not. Not yet the dominant path.

**Caveat on the measurement:** `mind_runs` session types observed in the last 24h (`orion_journal`, `orion_curiosity`, `orion_world_pulse_read`, `orion_world_pulse_read_stage2`, `orion_outreach`) could not be cleanly split into "live conversation with Juniper" versus "Orion's own autonomous background thinking" from this table alone — `request_summary_jsonb` does not carry the original message text for these rows. The 3-of-19 figure is across whichever mix of both actually occurred in that window, not isolated to human-facing chat.

**Recommendation if this is picked up later:** the two genuinely-irreducible judgment calls are `interface_cost` and `connection_seek` themselves — everything downstream of them in the legacy table (`task_mode`, `conversation_frame`, `interaction_regime`, the specific `response_priorities`/`response_hazards` lists) is a lookup, not a judgment, and could be ported into `stance_handoff.py` as an explicit rule set rather than left to the model's own unguided inference. Not done tonight because the shortcut isn't yet the dominant path and no live failure has been traced to this gap specifically.

## Loose thread — one anomalous near-instant `json_parse_failed`

**Status: open, not investigated, low priority.** One `orion_curiosity` run (2026-09-08 02:17:11 UTC) failed with the same error name as Incident 2 (`json_parse_failed`) but a materially different shape: `elapsed_ms: 105` — near-instant, not the 10-30 second unsuppressed-reasoning pattern. Single occurrence at time of writing. Worth a look if it recurs; not chased further tonight since it wasn't reproducible on demand and didn't match the fixed bug's signature.

## Non-goals (explicit)

- Do not build a ranking/selection step over `AppraisalFeatureVectorV1` without a concrete observed failure motivating it.
- Do not port the legacy stance decision-table into `stance_handoff.py` while the shortcut remains a minority path — re-measure first if this is revisited.
- Do not flip `orion-thought`'s `reverie_metacog_background_enabled` — that is Juniper's own standing experiment, unrelated to tonight's incidents.

## Acceptance checks (for whoever picks up either open finding)

- Open finding 1: before writing any code, produce at least one real `mind_runs` example where the frontier's selection was observably wrong and a real feature score would have caught it. No such example exists yet.
- Open finding 2: before porting the checklist, re-run the 3-of-19 measurement above with a wider window and, ideally, a way to isolate human-facing turns; port only if the shortcut's share has grown enough to matter.

## Verification record (incidents 1-3, tonight)

```text
PR #2147 — CI: Static repo gates PASS, hub-schedule-browser-smoke PASS. Merged.
PR #2150 — CI: Static repo gates PASS, hub-schedule-browser-smoke PASS. Merged.
  Tests: 89/89 passed (services/orion-mind/tests)
Live smoke (post #2150, direct /v1/mind/run calls):
  mind_quality: meaningful_synthesis (x2, distinct prompts)
  authorized_for_stance_skip: true
  gateway THINK_HOP inline_think_len: 0 (was 2000-4149 pre-fix)
Incident 3 rebuild: attention-runtime / proposal-runtime / feedback-runtime
  all three showing real successful ticks post-rebuild, zero ValidationError.
```
