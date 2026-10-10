## Summary

Orion's readings put concept nodes into the atlas but never connected them to anything: 211 reading concepts live, zero edges, because the reading adapter hard-codes `edges=[]` (`orion/substrate/adapters/world_pulse_read.py:63`) and nothing else ever made links for them. Orion spent a day waiting for links that could not arrive. This patch gives readings a way to link, on the rules Juniper set.

- **The text a read actually fetched is now kept.** On reading turns only, the harness copies each fetch result (what WebFetch handed the model, which is the fetch tool's digest of the page, not the page itself) to Hub. Hub stores it by its sha256 in the existing `reading_document_snapshot` table and the reading record keeps only the hash. The page text never goes into the stored handoff or to the chat UI.
- **Stage 2 can propose links.** Its prompt now lists the concepts this read produced (as the ids actually stored in the graph), up to 24 existing atlas concepts whose names appear in the read, and the retained text. It may return `relationship_claims`: subject, predicate, object, a one-line statement, and a supporting quote.
- **Code decides, not the model.** A claim is accepted as tentative (`provisional`) only when the quote is found word-for-word in the retained text and names the object, both ids are real stored nodes, and the predicate is one of `subtype_of / part_of / refines / associated_with / causes / co_occurs_with` with its domain rule. Anything else is recorded as `proposed` with no link. A made-up id or a predicate outside the list is refused and writes nothing at all. Every claim gets a receipt (what happened and why, plus the hash and byte range of the quote) stored with the Stage 2 result.
- **Hub writes the accepted links.** Hub runs the shared `AssertionProjector` for reading claims only, every Stage 2 tick. Memory's projector now applies only memory claims, because it cannot see reading nodes and would retry them forever.
- **Kill switch:** `HUB_WORLD_PULSE_READ_ASSERTIONS_ENABLED` (default `true`).

## Outcome moved

Before: a reading could never produce a relationship; the "anchor edges" Orion waited for had no producer. After: a reading whose retained text states a relationship to an existing concept produces a journaled, decided, projected, walkable link (proven end to end in tests and the eval). The live path is **UNVERIFIED**: no reading has completed since this was written (the reading queue is starved by GPU lease contention, a separate issue).

## Current architecture

- Stage 1 (`world_pulse_read_pipeline.py`) materialized reading concepts with no edges and recorded only `url / tool_name / content_chars` as read evidence; the fetched text was discarded.
- Stage 2 (`world_pulse_read_stage2.py`) summarized and tested priors; it never saw the fetched text or any existing concept ids.
- The #2515 assertion core (journal, projector, walkable gate, cognitive isolation) was live but only memory referents used it. `AssertionProjector.run_once` applied every pending decision regardless of producer, from a store primed with memory nodes only.

## Architecture touched

- Harness → Hub transport: `SourceFetchEvidenceV1.content_text` (transport only).
- Hub Stage 1: retains text (`orion/world_pulse_read/fetch_text.py`).
- Hub Stage 2: claim context, prompt block, validation, journal writes, projector runner (`orion/world_pulse_read/assertions.py`).
- Shared assertion core: `pending_decisions(proposal_actors=...)` and `AssertionProjector(proposal_actors=...)`.
- Memory consolidation: its projector passes `proposal_actors=("memory.referents",)`.

## Files changed

- `orion/schemas/reading.py`: `SourceFetchEvidenceV1.content_text` (transport only, omitted when unset); `content_sha256` now also keys a retained fetch digest.
- `orion/harness/reading_receipts.py`: copy the raw tool_result text (≤64K chars, else none) onto fetch evidence.
- `orion/hub/turn_orchestrator.py`: final frame carries `content_text` only for reading turns (`reading_only`).
- `orion/world_pulse_read/fetch_text.py` (new): retain/load texts by sha256 in `reading_document_snapshot`; `tool_digest` vs `source_text`; hash re-check on load.
- `orion/world_pulse_read/assertions.py` (new): subject lookup via the identity index, candidate retrieval, prompt block, `plan_claims` (the rule), `journal_claims`, `stored_object_lookup`.
- `orion/schemas/world_pulse_read.py`: `WorldPulseReadRelationshipClaimV1`, `WorldPulseReadClaimReceiptV1`, `READING_CLAIM_PREDICATES`, `relationship_claims` on the Stage 2 result, `strip_model_claim_receipts`.
- `orion/schemas/registry.py`: register the two new models.
- `orion/substrate/graph_journal.py`, `orion/substrate/assertion_projector.py`: `proposal_actors` filter.
- `services/orion-memory-consolidation/app/referent_projector.py`: apply memory claims only.
- `services/orion-hub/scripts/world_pulse_read_pipeline.py`: retain fetch text before persisting the handoff.
- `services/orion-hub/scripts/world_pulse_read_stage2.py`: claim context, prompt block, `_record_claims`, `_project_assertions` (every tick).
- `services/orion-hub/app/settings.py`, `.env_example`, `scripts/main.py`: `HUB_WORLD_PULSE_READ_ASSERTIONS_ENABLED`.
- `orion/schema_skew_discovery.py`: drop two `DECLARED_WRITERS` entries that went stale (Hub is now a discoverable writer of the journal models).
- Docs: `orion/substrate/README.md` Concepts rows, `orion/core/schemas/substrate_graph_journal.py` docstring, Hub README section.
- Tests/evals: `services/orion-hub/tests/test_world_pulse_read_assertion_links.py`, `reading_assertion_fakes.py`, `test_world_pulse_read_pipeline.py` (+1), `orion/harness/tests/test_reading_receipts.py` (+1), `orion/substrate/tests/test_graph_journal_pg.py` (+1), `services/orion-hub/evals/test_reading_assertion_eval.py`; CI wiring in `.github/workflows/orion-reading-tests.yml`.

## Schema / bus / API changes

- Added: `WorldPulseReadRelationshipClaimV1`, `WorldPulseReadClaimReceiptV1` (registered, `resolve()` verified); `WorldPulseReadStage2ResultV1.relationship_claims` (default empty); `SourceFetchEvidenceV1.content_text` (default None, omitted when unset); journal proposal actor `world_pulse_read_stage2`, decision policy `reading_quote_rule_v1`.
- Removed / renamed: none. No bus channel added (the journal stays Postgres-only, as in #2515).
- Behavior changed: memory's projector ignores non-memory claims; `content_sha256` appears on web-fetch evidence when text was retained.
- Compatibility:
  - `WorldPulseReadStage2ResultV1` is `extra="forbid"` and only Hub (and the `verify.py` CLI) validates it. Rolling Hub back after new rows exist makes the old repair path reject rows that carry claims: pause Stage 2 first (same rule the README already states for `read_evidence`).
  - `SourceFetchEvidenceV1` is `extra="ignore"`, so an old governor or old durable-runs silently drops `content_text`: reads still work, they just retain no text and produce no claims.
  - Old memory-consolidation (no actor filter) would try reading decisions and record `endpoint_missing` failures (transient, retried) while Hub applies them; noisy, not breaking. Deploy it first anyway.

## Env/config changes

- Added keys: `HUB_WORLD_PULSE_READ_ASSERTIONS_ENABLED=true` (services/orion-hub).
- Removed / renamed: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (`orion-hub: +HUB_WORLD_PULSE_READ_ASSERTIONS_ENABLED='true'`).
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-hub/tests/test_world_pulse_read_assertion_links.py   25 passed (after review fixes)
services/orion-hub tests (test_world_pulse_read_*, test_reading_*, test_turn_orchestrator_ws_frames, test_curiosity_investigation)
                                                                     467 passed, 65 skipped (RUN_READING_POSTGRES lanes; CI runs them)
orion/world_pulse_read/tests orion/harness/tests orion/introspect/tests tests/test_world_pulse_read_* tests/test_unified_turn_*
                                                                     800 passed, 2 skipped
orion/substrate/tests (with a throwaway Postgres 16 for the journal lane)
                                                                     1061 passed, 3 failed (test_felt_state_self_definition_lane.py, failing identically on main)
orion/substrate/tests/test_graph_journal_pg.py (real Postgres)       11 passed (incl. new proposal_actors filter test)
services/orion-memory-consolidation tests + evals                   287 passed, 13 skipped
Every python gate in orion-static-gates.yml                          27/27 pass (after removing the two stale DECLARED_WRITERS)
Mutation checks:
  semantic_projection admitted to COGNITIVE_EDGE_ROLES -> dynamics isolation test FAILS (load-bearing)
  proposal_actors filter removed -> memory projector test would apply the reading claim (asserted not to)
```

## Evals run

```text
pytest services/orion-hub/evals/test_reading_assertion_eval.py -s
reading_assertion_eval cases=8 linked=2 unsupported_links=0 supported_recall=1.00 refused_writes=0
  verbatim:accepted, verbatim_contrast:accepted, paraphrase:quote_not_found, whitespace_drift:quote_not_found,
  invented_object:unknown_object, operational_predicate:predicate_not_allowed,
  quote_about_something_else:quote_not_about_endpoints, one_word_quote:quote_too_short
```

Recorded model-shaped claims, not a live model: it measures the rule and wiring, not how often a real Stage 2 quotes exactly.

## Docker/build/smoke checks

```text
No image built. Live path UNVERIFIED: no reading has completed since this change
(queue starved by GPU lease contention). Live data checked read-only:
- Falkor orion_substrate: 211 world_pulse_read_pipeline concepts, all ids sub-concept-wp-read-*,
  identity keys concept|orion|world_pulse|label:<label> (the lookup this patch uses).
- Postgres conjourney: substrate_graph_journal and reading_document_snapshot both exist; hub's DSN can write them.
```

## Review findings fixed

Code review ran in a subagent on adad7d85d (2 MUST, 5 SHOULD, 6 NIT).

- Finding (MUST): one malformed claim (empty statement, or an extra key like `confidence`) failed the whole `extra="forbid"` Stage 2 result, losing the summary and priors and spending the Wallet B slot.
  - Fix: `_coerce_claim_list` validates each claim on its own, drops unknown keys and invalid claims with a warning.
  - Evidence: `test_one_bad_claim_does_not_discard_the_stage2_read`.
- Finding (MUST): any 4-word sentence from the page could back a link between any offered pair, including `causes`.
  - Fix: the quote must also name the object by its stored label, else `proposed` with reason `quote_not_about_endpoints`. The subject may be a pronoun. **This goes past Juniper's literal rule (a); it only makes acceptance stricter. Confirm or revert.**
  - Evidence: `test_a_real_quote_that_does_not_name_the_object_stays_proposed`; eval case `quote_about_something_else`.
- Finding (SHOULD): page text rode the governor-to-Hub payload on every turn, not just reading turns.
  - Fix: `ReadingReceiptTracker(retain_text=request.reading_only)`.
  - Evidence: `test_non_reading_turns_never_carry_fetched_text`.
- Finding (SHOULD): an old governor or durable-runs silently strips the text.
  - Fix: deploy order in the restart section and README; Hub logs `reading_claim_text_missing` per read with unretained web evidence.
- Finding (SHOULD): subject endpoints skipped the private-memory fence check.
  - Fix: `_stored_endpoint` refuses fenced nodes. Evidence: `test_a_fenced_memory_node_is_never_a_subject`.
- Finding (SHOULD): schema-skew gate lost its durable-runs writer for the journal models.
  - Not fixed: the tool is single-writer by design; left as a named follow-up (multi-writer `DECLARED_WRITERS`).
- Finding (SHOULD): old memory-consolidation would record `endpoint_missing` noise on reading decisions.
  - Fix: restart order puts memory-consolidation first.
- NITs fixed: quote stripped once before search; `accepted` counted only on a real append. Not fixed: `_claim_context` held on the instance (single caller); quote search covers the whole retained text, not only the 16K shown in the prompt (still verbatim); no retention/deletion policy yet for retained text (follow-up); the digest/document sha collision note.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-harness-governor up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build
cd /mnt/scripts/Orion-Sapienform && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
```

Production deploys from the primary checkout on main, after merge, in this order: memory's projector filter first, then the two services that carry the fetched text (governor, durable-runs), then Hub. No migration: both tables already exist live.

## Risks / concerns

- Severity: medium. Concern: the neighborhood read's default request only admits `provisional`/`canonical` nodes, and every live concept except 4 seeds is `proposed`, so a default `read_neighborhood` from a reading concept returns nothing even with a link. Mitigation: no production caller uses `read_neighborhood` yet; the Concept Atlas and Recall region reads (which do not filter node state) show the link. Tests read with `semantic_states` including `proposed`. Follow-up: decide whether the default should admit `proposed` endpoints.
- Severity: medium (needs Juniper). Concern: two acceptance checks beyond the literal rule: the quote must name the object, and be at least 4 words / 20 chars. Both only make auto-accept stricter.
- Severity: low. Concern: retained fetch text has no retention/deletion policy yet (design doc asks for one). Follow-up.
- Severity: low. Concern (was): a quote must be at least 4 words / 20 chars. Juniper's rule says "verbatim"; this adds that a single word is not a cited statement. Mitigation: documented, one constant.
- Severity: low. Concern: firecrawl results over 64K chars are not retained, so those reads cannot back a claim. Mitigation: logged by absence of `content_sha256`; raise the cap if live reads show it matters.
- Severity: low. Concern: `schema_skew_discovery` is single-writer per model; the durable-runs writer of the journal models is now uncovered by the live skew check. Mitigation: comment in place; the check itself is unchanged.

## PR link

(pending)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
