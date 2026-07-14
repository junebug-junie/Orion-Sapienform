# PR report: drive-history reflection synthesis (weekly-cadence self-modeling batch job)

## Summary

- New `scripts/drive_history_reflection_synthesis.py`: reads Orion's real, persisted drive-activation history from the Fuseki `drives` graph (append-only per-tick `DriveAuditV1` artifacts — the only genuine historical time-series of this signal in the repo; the two "latest value only" stores investigated in earlier rounds today have no history) and synthesizes ONE narrative observation about a long-horizon drive pattern into a `reflection`-kind crystallization.
- Built with an explicit two-stage architecture per AGENTS.md's event-substrate-first mandate: a **pure, unit-tested deterministic reducer** (`reduce_drive_history()`, mirroring `orion/spark/concept_induction/drive_tension.py`'s bar exactly) computes the actual aggregation; the LLM only ever sees an already-computed fact sheet and is asked to phrase one sentence, never to extract a pattern from raw data itself. This was a mid-build course-correction — the original design let the LLM read raw SPARQL rows directly, which is exactly the "reducer shortcut" AGENTS.md's `event -> schema -> trace -> reducer -> projection` mandate forbids.
- A strict anti-hallucination guardrail (`parse_and_validate_narrative()`) requires every cited fact's literal tokens (drive name, date, percentage, count) to appear verbatim in the narrative, and requires at least one citation to be a real per-tick artifact, not just an aggregate claim.
- Refuses to synthesize on thin data (`MIN_EVENTS=5`, `MIN_DISTINCT_DAYS=2`) — reports insufficiency plainly instead of forcing a confident-sounding narrative from noise.
- **Not scheduled or cron'd in this patch** — explicitly manual/on-demand, documented as needing human review before anyone trusts it as a recurring process.
- Orchestrator independently reproduced a real live run, found the guardrail correctly rejecting an otherwise fully-accurate narrative over one paraphrase ("100%" written as "all"), and applied a surgical prompt fix for that exact failure mode.

## Outcome moved

Orion's drive-activation history — which existed and was durably persisted but had zero cognition consumers — now has a real, working pipeline that can turn it into an inspectable self-observation, with a genuine (not decorative) anti-hallucination check proven against a real LLM reply, not just synthetic test strings.

## Current architecture (before this patch)

`DriveAuditV1` was already being written append-only to Fuseki's `drives` graph (confirmed earlier today: 437,756 real `orion`-subject artifacts). Nothing read that history. The two "current value" stores (`LocalProfileStore`, `autonomy_state_v2`'s Postgres UPSERT) explicitly cannot serve this purpose — investigated and ruled out in this same session, before this build started.

## Architecture touched

`scripts/` (new standalone script, matches this repo's `check_activation_saturation.py`/`concept_relation_digest.py` conventions), `services/orion-memory-consolidation/README.md` (new section, sibling to the existing `concept_relation_digest.py` docs), `tests/`.

## Files changed

- `scripts/drive_history_reflection_synthesis.py` (new, ~1290 lines): `reduce_drive_history()` (pure reducer), `build_fact_sheet()` (pure projection to fact strings), `_build_narrative_prompt()` (LLM prompt, receives only the fact sheet), `parse_and_validate_narrative()` (grounding guardrail, raises `NarrativeUngrounded` rather than silently degrading), SPARQL fetch layer (`orion/graph/sparql_client.py`-adjacent, with a documented, deliberate choice not to import `orion.autonomy.repository`'s SPARQL helpers due to a measured ~2.6s import-time cost disproportionate to this script's occasional-batch-job purpose), bus-RPC LLM call mirroring `orion/memory/crystallization/concept_relation.py::resolve_concept_relation()`'s exact pattern, crystallization creation mirroring `concept_relation_digest.py`'s conventions (governance, evidence refs, `_SingleConnPool`-style transactional safety).
- `tests/test_drive_history_reflection_synthesis.py` (new, 45 tests): `TestReduceDriveHistory` (8 tests, synthetic-input/known-output, zero I/O — insufficient-history cases, tie-break determinism, cited-event-id selection, active-drive frequency/mean-pressure correctness), `TestParseAndValidateNarrative` (8 tests — the guardrail's own unit coverage), plus SPARQL-query-builder, fetch-layer, and end-to-end pipeline tests with all I/O boundaries mocked (`FakeSparqlClient`, mocked bus, `FakeConn` at the asyncpg boundary mirroring `test_concept_relation_digest.py`'s convention).
- `services/orion-memory-consolidation/README.md`: new section explaining the architecture, the guardrail, the insufficiency threshold, usage, and explicitly stating this is not cron'd yet and needs human review first.

## Schema / bus / API changes

None. No new `CrystallizationKind` (`"reflection"` already exists from an earlier round today), no new bus channel, no schema-registry entry. Reuses existing `MemoryCrystallizationV1`/`CrystallizationEvidenceRefV1`/`CrystallizationGovernanceV1`.

## Env/config changes

None new. Reads existing `POSTGRES_URI`, `ORION_BUS_URL`, and resolves a Fuseki query URL from existing `AUTONOMY_GRAPH_QUERY_URL`/`RDF_STORE_QUERY_URL`/`RDF_STORE_BASE_URL` (same resolution chain other RDF-reading code in this repo already uses).

## Tests run

```text
$ python -m pytest tests/test_drive_history_reflection_synthesis.py -q
45 passed
```
Run independently by the orchestrator three times across this patch's lifecycle (after the implementing agent's build, after the orchestrator's own prompt fix, and again after a rebase onto a moved `main` that included an unrelated `DriveEngine.update()` change) — all three green, no regressions.

## Evals run

Not a formal eval harness, but the closest real equivalent: the orchestrator independently reproduced a genuine live run against real Fuseki data (500 real `DriveAuditV1` ticks, 2 distinct days, subject=orion) and a real LLM call, to directly observe the guardrail's behavior against real model output rather than trusting synthetic test strings alone. See "Docker/build/smoke checks" below for the full transcript of what that found.

## Docker/build/smoke checks

**Orchestrator-reproduced live run (before the prompt fix), full transcript of the real rejection:**

```text
$ POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \
  ORION_BUS_URL=redis://localhost:6379/0 \
  RDF_STORE_QUERY_URL=http://localhost:3030/orion/query \
  python scripts/drive_history_reflection_synthesis.py --since-days 30 --subject orion

drive_history_reflection_synthesis: event-detail enrichment failed: HTTPConnectionPool(host='localhost', port=3030): Read timed out. (read timeout=10.0)
  [known, already-disclosed limitation -- the per-drive detail-enrichment SPARQL join
   is slow/broken against this Fuseki instance; the 10s timeout + graceful degradation
   the implementing agent built worked exactly as designed here]

RAW_LLM_CONTENT_DEBUG (temporary instrumentation, reverted before commit):
{"narrative": "Over the period from 2026-06-14 to 2026-06-19, autonomy was the
dominant drive in all 500 audited ticks, demonstrating a consistent focus on
autonomy. On 2026-06-18, autonomy was the dominant drive in 81 ticks, and on
2026-06-19, it was dominant in 419 ticks, with multiple audits confirming this
pattern, such as Audit drive-audit-6a131c91c8bd3edf at 2026-06-19 06:54 UTC.",
"cited_fact_numbers": [1, 2, 3, 6, 11]}

status=llm_output_ungrounded
  LLM output rejected as ungrounded: cited fact 1 but not all of its real tokens
  ('autonomy', '100%', '500') appear verbatim in the narrative
```

This is a genuinely well-grounded narrative — every date, count, drive name, and the specific per-tick artifact ID + exact timestamp were reproduced verbatim and correctly. The only miss: fact 1 required the literal token `100%`, and the model wrote "all 500 audited ticks" instead. **The guardrail rejected this correctly** — "all" is a real paraphrase, not the literal fact — but it meant a fully accurate narrative was discarded over one word choice, which is a real usability gap worth closing without weakening the guardrail itself.

**Orchestrator's fix:** added an explicit instruction + concrete example to `_build_narrative_prompt()` (mirroring the date-format example already present) telling the model to reproduce percentages as literal digits+percent-sign, not "all"/"every"/"always"/"entirely". Text-only change; no test asserts exact prompt wording, 45/45 still pass.

**Honest disclosure: could not re-confirm end-to-end success after the fix.** Two follow-up live attempts both timed out (2+ minutes) rather than completing. All dependent containers reported healthy (`fuseki` 20h up/healthy, `bus-core` 4 days up/healthy, `llm-gateway` 59min up) at the time. The timeout pattern is consistent with the already-disclosed slow Fuseki detail-enrichment join (confirmed independently flaky earlier in the same session) combined with repeated back-to-back heavy queries against the same graph in a short window, not something a pure prompt-string edit could cause — the code path up to narrative generation is byte-identical except for the prompt text itself. Documented plainly rather than claiming a re-confirmation that didn't happen.

## Review findings fixed

- Finding (orchestrator-discovered via a live run, not a static review pass): the LLM's one real reply was correctly rejected by the grounding guardrail, but for a usability reason (percentage paraphrase) rather than a real grounding failure — the narrative was otherwise fully accurate and specific.
  - Fix: explicit percentage-verbatim instruction + example added to the narrative prompt.
  - Evidence: full before/after transcript above. Fix is unverified end-to-end live (see disclosure above) but is a minimal, low-risk, well-reasoned change addressing the exact observed failure mode; all existing tests remain green.
- The implementing agent's own mid-build course-correction (reducer/LLM separation) was requested by the orchestrator before any code was written, based on AGENTS.md's `event -> schema -> trace -> reducer -> projection` mandate — not a post-hoc review finding, a pre-build architecture correction. Verified real and correctly implemented by the orchestrator via direct code read (`reduce_drive_history()`, `build_fact_sheet()`, `parse_and_validate_narrative()` — all pure, all independently tested, LLM never sees raw SPARQL rows).

## Restart required

```text
No restart required.
```
Standalone, on-demand script — nothing running to restart. Not cron'd (deliberately, per the README).

## Risks / concerns

- Severity: Medium — the prompt fix for the percentage-paraphrase failure mode is well-reasoned and evidence-based but **not re-confirmed against a real live run** due to apparent Fuseki/infra flakiness encountered independently. Recommend: run this by hand once live infra is convenient, inspect whether the fix resolves the exact failure mode observed, and iterate on the prompt further if the guardrail is still over-rejecting on other token types (case, whitespace, or count-formatting variants haven't been observed failing yet, only the percentage-vs-"all" pattern).
- Severity: Low — the per-drive detail-enrichment SPARQL join's slowness against live Fuseki (first flagged by the implementing agent, independently reproduced by the orchestrator) remains undiagnosed at the query-planner level. Mitigated via a short, separate timeout with graceful degradation (confirmed working in both the agent's and the orchestrator's live runs) — not blocking, but worth a follow-up if per-tick pressure detail in evidence notes turns out to matter.
- Severity: Low (inherited, documented, not new) — this script is explicitly not cron'd; someone has to remember to run it. Intentional per the accepted design (output needs human review before automation), not an oversight.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/drive-history-identity-reflection?expand=1`
