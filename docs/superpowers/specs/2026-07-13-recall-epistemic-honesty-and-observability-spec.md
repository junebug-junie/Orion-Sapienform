# Recall epistemic honesty + observability: confidence assignment, contradiction-detection instrumentation, retrieval-event logging

**Date:** 2026-07-13
**Status:** Proposed (items 1, 3) — **item 2 fixed and live-verified same day**, see addendum below.
**Scope:** Three independent thin patches, one spec because they're the direct output of one investigation into whether Orion's memory recall is "bullshit RAG" theater. Companion to `2026-07-13-memory-recall-reinforcement-decay-wiring-spec.md` — that spec covers ranking (`activation`); this one covers what's actually *true* about a memory (`confidence`) and whether the system can prove any of this happened in the wild (`observability`).

---

## Arsonist summary

Three things were flagged as possible theater and investigated against live data, not assumed:

| # | Claim checked | Verdict | Evidence |
|---|---|---|---|
| 1 | Confidence field is meaningless | **Confirmed theater — genuine gap, not bad luck** | `CrystallizationConfidence` is a real 4-value enum (`certain/likely/possible/uncertain`) with a real downstream consumer (`salience.py::CONFIDENCE_BOOST`), but **zero code paths in the entire system ever set it explicitly** — Hub manual propose, consolidation-gate window intake, autonomy-episode intake, concept-relation resolution. All 164 rows in `memory_crystallizations` (not just today's 100 active) are `"likely"`. Since `CONFIDENCE_BOOST["likely"] = 0.05` is a flat constant, `score_salience()`'s confidence term has been silently inert since inception — one whole dimension of the salience formula has never varied. |
| 2 | Contradiction-detection is broken (like the duplicate-detection Jaccard bug found twice this session) | **Confirmed broken, then fixed and live-verified same day** — see addendum. Two independent bugs stacked: a live-`.env` config regression I caused earlier this session, plus a real missing dependency + missing Docker build toolchain that predates this session entirely. | `maybe_resolve_concept_relation()` never had a working candidate-retrieval path to run against on this host. Root-caused by direct live invocation of the real production function, not by reading code. |
| 3 | Real chat-time recall can't be verified in the wild | **Confirmed, clean gap** | `services/orion-recall/` (the actual chat-time recall path) writes **zero** retrieval-event history anywhere. The only writer, `insert_retrieval_event`, lives on the Hub API route real chat doesn't call. This blocks verifying *everything* built this session — including the reinforcement/decay spec's own acceptance checks — against real traffic instead of synthetic smoke tests. |

---

## Addendum: item 2 root cause and fix (2026-07-13, same day)

Investigated by directly invoking `process_consolidation_crystallization()` — the real production function, real settings, real Postgres pool, real bus RPC to the LLM gateway — with two crystallizations manually constructed to genuinely contradict each other. Not a code read, not a guess: ran it and watched it fail, then fixed each failure and re-ran until it produced a real, persisted `contradicts` link.

**Bug A — config regression, self-inflicted, same session.** `services/orion-memory-consolidation/.env`'s `CRYSTALLIZER_EMBED_HOST_URL` and `CHROMA_HOST` were reset to empty by the *old* (pre-fix) `sync_local_env_from_example.py` run earlier this session — the exact overwrite bug closed by `fix/env-sync-preserve-local-overrides` (PR #996). `fetch_similar_candidates()` returns `[]` immediately when either is empty (`candidate_retrieval.py:29`); `maybe_resolve_concept_relation()` short-circuits the moment candidates are empty — the LLM call never fires. Fixed: restored both values in the live `.env` (config-only, not committed — gitignored, matches the deployment-specific-override pattern already established for `GRAPHITI_BACKEND`).

**Bug B — missing dependency, pre-existing, not caused by anything this session did.** Even with config fixed, `chromadb` was never installed in `orion-memory-consolidation`'s Docker image — `ModuleNotFoundError` on import, silently caught and logged as `"chromadb not installed; skipping chroma query"` (`chroma_query.py:23`), so `fetch_similar_candidates()` degrades to `[]` exactly the same as bug A, just for a different reason. Fixed: added `chromadb==0.4.24` + `numpy==1.26.4` to `requirements.txt` (matching the pinned versions already used by `orion-dream`/`orion-rag`/`orion-vector-writer`, which all depend on the same package). Rebuilding surfaced **bug C**.

**Bug C — missing Docker build toolchain, same root cause as B.** `chromadb`'s transitive dependency `chroma-hnswlib` builds a native C extension; `orion-memory-consolidation`'s Dockerfile had no `gcc`/`g++`/`libpq-dev` installed (unlike the three sibling Dockerfiles that already carry this feature). Fixed: added the same `apt-get install gcc g++ libpq-dev` step already used by `orion-dream`'s Dockerfile.

**Live verification, end to end, real infrastructure:** proposed+approved a seed belief ("deploy at night, never during the day") via Hub. Directly invoked `process_consolidation_crystallization()` with a second crystallization asserting the opposite ("deploy during the day, never at night") on the same subject. Real Chroma vector search found the seed as a candidate; real embed host produced the query embedding; real LLM gateway RPC classified the relation as `contradicts` with confidence `0.95` (well clear of the `0.6` floor); a real `contradicts` row landed in `memory_crystallization_links`, confirmed by direct Postgres query, not by trusting the function's return value alone. Test crystallizations deprecated/rejected afterward — no permanent pollution.

**Fix scope:** `services/orion-memory-consolidation/Dockerfile`, `requirements.txt` — code-level, applies to any deployment that ever flips `CONCEPT_RELATION_RESOLUTION_ENABLED=true`, not just this host. The `.env` config restoration is host-specific and not part of the commit (gitignored).

---

## Non-goals

- No LLM call added anywhere in this patch (item 1's confidence inference is deterministic; item 2 adds logging around an *existing* LLM call, doesn't add a new one).
- Not changing `concept_relation.py`'s confidence floor (0.6) or its logic — item 2 is observation only, until instrumentation produces evidence a change is warranted.
- Not building a new metrics/observability service — item 3 reuses the existing `memory_crystallization_retrieval_events` table Hub already writes to; item 2's logging is structured log lines with correlation IDs, not a new store.
- Not touching `dynamics.reinforcement_count`'s existing recall_boost/reinforce split from the companion spec — item 1 reads `reinforcement_count` as an input, never writes to it.
- No retroactive backfill of confidence on the 164 existing rows — this patch changes what happens for new/future crystallizations and reinforcement events; existing rows stay `"likely"` until they're naturally reinforced or re-evaluated (backfill is a separate, explicit decision, not bundled here).

---

## Item 1: Confidence assignment (deterministic, at formation + on reinforcement)

**Problem:** confidence is a free field nobody fills in. Fix: compute it from signals that already exist, at the two points a crystallization's evidentiary basis actually changes — formation and `reinforce()` (recurrence). Explicitly **not** at `recall_boost()` (mere retrieval) — being looked up is not evidence the belief is true, only that it's relevant; conflating the two is the conformity-bias failure the companion spec already named as a hard invariant to avoid.

**Proposed heuristic** (`infer_confidence(crystallization) -> CrystallizationConfidence`, deterministic, no LLM):

| Condition | Assigned confidence |
|---|---|
| No evidence, or all `evidence.strength` avg < 0.3, `reinforcement_count == 0` | `uncertain` |
| 1 evidence source, moderate strength, `reinforcement_count == 0` | `possible` |
| 1-2 evidence sources, or `reinforcement_count` 1-2 | `likely` |
| 3+ evidence sources, or `reinforcement_count >= 3`, or (evidence_count >= 2 AND `reinforcement_count >= 1`) | `certain` |

This is a first-cut heuristic, not a claimed-correct final formula — thresholds need tuning against real post-deployment data (acceptance check below), same discipline as the companion spec's saturation gate.

**Call sites:**
- Formation: wherever `apply_salience()` is currently called (crystallization creation) — call `infer_confidence()` immediately before, so `score_salience()` reads a real value on the same pass, not a stale default.
- `dynamics.py::reinforce()`: after incrementing `reinforcement_count`, recompute confidence via the same function and set it if it changed. This is the one new touch to `dynamics.py`.

**Files:** `orion/memory/crystallization/salience.py` (new `infer_confidence()`, or a sibling module if salience.py shouldn't own it — check for circular import risk with `dynamics.py`), `orion/memory/crystallization/dynamics.py` (`reinforce()` call site), wherever `apply_salience()` is currently invoked at formation (`intake_pipeline.py` and siblings).

---

## Item 2: Contradiction-detection instrumentation (observe, don't fix blind)

**Problem:** zero `contradicts`/`supersedes` links exist. Could be genuine (small, non-conflicting dataset) or a silent threshold/wiring problem (precedent: duplicate-detection's Jaccard threshold, proven broken twice this session). No way to tell which without data.

**Fix:** log every `maybe_resolve_concept_relation()` invocation and its raw decision — relation, confidence, whether it cleared the 0.6 floor — regardless of outcome, including `same`/`unrelated`/below-floor cases that currently vanish silently. One structured log line per call, correlation-id tagged (matches AGENTS.md's own definition of valid runtime evidence: "log line with correlation ID").

**What this answers, that we can't answer today:**
- Is the function even being called at realistic volume, or rarely triggered (wiring/trigger gap)?
- When it's called, does it ever land on `contradicts`/`refines` *below* the 0.6 floor (threshold-tuning problem, same shape as the Jaccard bug)?
- Or does it consistently land on `same`/`unrelated` (genuinely no conflicts in this data — the innocent explanation)?

This is the whole patch for item 2 — no logic change, no threshold change, until the log data says one is warranted.

**Files:** `orion/memory/crystallization/concept_relation.py` (`maybe_resolve_concept_relation()` — add the log line before the branch on `decision.relation`).

---

## Item 3: Retrieval-event logging from real chat-time recall

**Problem:** `services/orion-recall/app/collectors/active_packet.py::fetch_active_packet_fragments()` — the function real chat turns actually call — never writes to `memory_crystallization_retrieval_events`. Only Hub's API route does, and real chat doesn't use that route.

**Fix:** call the same `insert_retrieval_event` write (or a thin wrapper) from `fetch_active_packet_fragments()` after `retrieve_active_packet()` returns, using the same table Hub already writes to — this is a shared conceptual event (a real retrieval happened), not a new schema. `orion-recall` already has Postgres access to the crystallization schema (it already imports `list_crystallizations`/`retrieve_active_packet` from the same package), so this is a query away, not a new connection.

**Why this is the highest-leverage of the three:** without it, the companion spec's acceptance check #1 (saturation gate measured over "a real usage window") and acceptance check #2 (head-to-head recall competition) have no real data to run against once `RECALL_GRAPHITI_IN_CHAT` and normal chat traffic are the source — only synthetic smoke bursts. This item is a prerequisite for verifying the *other* spec in production, not just a nice-to-have.

**Files:** `services/orion-recall/app/collectors/active_packet.py`, `orion/memory/crystallization/repository.py` (reuse `insert_retrieval_event`, don't duplicate it).

---

## Acceptance checks

1. **Item 1 — real variance, not just a different constant.** After deployment, query `select confidence, count(*) from memory_crystallizations where created_at > <deploy time> group by confidence`. Fail condition: 100% still land on one value (formula is degenerate in practice, not just in theory — needs real threshold tuning, not just existing).
2. **Item 1 — salience actually moves.** Spot-check: two otherwise-identical crystallizations with different evidence counts get different `confidence` and therefore different `salience` post-deploy. Today this is impossible by construction; must be possible after.
3. **Item 2 — the log answers the question within N days of real traffic.** After a real usage window, query the new log: report (a) call volume, (b) relation-decision distribution including sub-floor cases, (c) whether any sub-floor `contradicts`/`refines` decisions exist (evidence of a tuning problem) vs. none (supports the innocent explanation). This is a diagnostic report, not a pass/fail gate — the acceptance criterion is "we now have an answer," not "the answer is positive."
4. **Item 3 — real chat retrieval produces a real log row.** Drive one real chat turn through `chat_general` with recall active, then query `memory_crystallization_retrieval_events` for a row with a timestamp matching that turn. This is the actual "runtime truth" proof the whole session has been chasing — a real conversation, not a smoke script, producing an inspectable artifact.
5. No regression: existing tests for `score_salience()`, `apply_salience()`, `maybe_resolve_concept_relation()`, and Hub's `insert_retrieval_event` call site all still pass unchanged.

---

## Recommended next patch

**Item 3 first, alone if needed.** It's the cleanest, has zero design ambiguity, and every other verification claim this session has made or will make about live chat recall depends on it existing. Items 1 and 2 can follow in the same or a subsequent patch — 1 needs the heuristic thresholds validated against a real deploy (acceptance check 1), 2 needs a real traffic window to produce a verdict (acceptance check 3), and neither of those windows can start meaningfully until item 3 exists to observe them through.
