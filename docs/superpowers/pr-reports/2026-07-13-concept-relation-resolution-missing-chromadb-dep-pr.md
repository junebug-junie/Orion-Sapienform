# PR report: fix concept-relation resolution (contradiction detection) — missing chromadb dependency + build toolchain

## Summary

- `orion-memory-consolidation`'s contradiction/refines/same-belief detection (`maybe_resolve_concept_relation()`) has never worked on this host — zero `contradicts`/`supersedes` links exist anywhere in `memory_crystallization_links`, despite `CONCEPT_RELATION_RESOLUTION_ENABLED=true`.
- Root-caused by directly invoking the real production function against real infrastructure (not by reading code) and watching it fail, twice, for two independent reasons.
- **Bug A (config, already self-corrected, not part of this commit):** `CRYSTALLIZER_EMBED_HOST_URL`/`CHROMA_HOST` in `services/orion-memory-consolidation/.env` had been reset to empty by the pre-fix `sync_local_env_from_example.py` overwrite bug (PR #996) earlier this session. Restored live.
- **Bug B/C (code, this commit):** `chromadb` was never in `requirements.txt`, and the Dockerfile had no C compiler toolchain to build its native `chroma-hnswlib` dependency even if it were added. Both fixed, matching the exact pattern already used by `orion-dream`/`orion-rag`/`orion-vector-writer`.
- Live-verified end to end: real Chroma search + real embed host + real LLM gateway classification produced a genuine `contradicts` link with confidence 0.95, confirmed by direct Postgres query.

## Outcome moved

Contradiction detection goes from "flag flipped on, silently a no-op since inception" to "verified working against real infrastructure." This was previously misdiagnosed (by me, in an earlier pass this same session) as "not proven broken, insufficient evidence" — that conclusion was wrong because I hadn't yet actually run the function against live data; I only checked whether it was *called*, not whether its own dependencies were satisfiable.

## Current architecture (before this patch)

`maybe_resolve_concept_relation()` (`orion/memory/crystallization/concept_relation.py`) is called from `intake_pipeline.py` on real consolidation-gate ingestion. It calls `fetch_similar_candidates()` (`candidate_retrieval.py`) for vector-similarity candidates before ever attempting an LLM classification. `fetch_similar_candidates()` requires both `CRYSTALLIZER_EMBED_HOST_URL` and `CHROMA_HOST` non-empty, and requires the `chromadb` Python package importable — both were broken independently.

## Architecture touched

Single service, `orion-memory-consolidation` — Dockerfile and requirements.txt only. No schema, bus, or API contract changes. No change to `concept_relation.py`'s logic, thresholds, or LLM prompt.

## Files changed

- `services/orion-memory-consolidation/requirements.txt`: added `chromadb==0.4.24`, `numpy==1.26.4` (pinned to match the three sibling services that already depend on the same package — `numpy==1.26.4` specifically because `chromadb==0.4.24`'s transitive code path uses `np.float_`, removed in numpy 2.0).
- `services/orion-memory-consolidation/Dockerfile`: added `apt-get install gcc g++ libpq-dev` before `pip install`, matching `orion-dream`'s Dockerfile — `chroma-hnswlib` (chromadb's transitive dependency) builds a native C extension and has no prebuilt wheel for this base image.
- `docs/superpowers/specs/2026-07-13-recall-epistemic-honesty-and-observability-spec.md`: addendum documenting the corrected finding (this spec originally concluded item 2 was "not confirmed broken" — that was wrong, corrected here with the real root cause and live verification).

## Schema / bus / API changes

None.

## Env/config changes

- Added keys: none (no `.env_example` shape change — `CRYSTALLIZER_EMBED_HOST_URL`/`CHROMA_HOST`/`CONCEPT_RELATION_RESOLUTION_ENABLED` already existed).
- This host's live `.env` values (`CRYSTALLIZER_EMBED_HOST_URL=http://orion-athena-vector-host:8320/embedding`, `CHROMA_HOST=orion-athena-vector-db`) were restored as part of diagnosing this — gitignored, not part of this commit, already applied live and confirmed working in the verification below.
- `.env_example` intentionally still ships these two keys empty (existing comment: "even if `CONCEPT_RELATION_RESOLUTION_ENABLED` is flipped true without also configuring these, the feature degrades to a no-op") — that's the correct safe default for fresh deployments and is unchanged by this patch. Any deployment that flips `CONCEPT_RELATION_RESOLUTION_ENABLED=true` now also needs a working `chromadb` install (this patch) *and* real embed/chroma host values (operator's own `.env`, unchanged contract).

## Tests run

```text
$ source venv/bin/activate && python -m pytest services/orion-memory-consolidation/tests -q
71 passed, 14 warnings in 4.31s
```
No regression. No new unit test added for this fix — it's a dependency/build-toolchain fix, not a logic change; the meaningful verification is the live end-to-end run below, not a mock.

## Evals run

No eval harness exists for `orion-memory-consolidation`'s concept-relation resolution specifically. The live verification below is the closest equivalent and is the standard this fix is held to, matching how the duplicate-detection Jaccard bug was verified earlier this session (empirically, not by inspection).

## Docker/build/smoke checks

```text
$ docker compose --env-file .env --env-file services/orion-memory-consolidation/.env \
    -f services/orion-memory-consolidation/docker-compose.yml up -d --build
Successfully installed ... chroma-hnswlib-0.7.3 chromadb-0.4.24 numpy-2.5.1 ...
# (first build: numpy 2.5.1 pulled in transitively, chromadb import failed on np.float_ removal)

$ docker exec orion-athena-memory-consolidation python -c "import chromadb"
AttributeError: `np.float_` was removed in the NumPy 2.0 release.
# fixed by pinning numpy==1.26.4 explicitly, rebuilt:

$ docker exec orion-athena-memory-consolidation python -c "import chromadb; print(chromadb.__version__)"
chromadb ok 0.4.24

# Live end-to-end verification (real production function, real infra, not mocked):
# 1. Proposed+approved seed belief via Hub: "deploy orion services at night, never during the day"
# 2. Directly invoked process_consolidation_crystallization() inside the running container with a
#    second crystallization asserting the opposite claim on the same subject.
$ docker exec orion-athena-memory-consolidation python /tmp/test_contradiction_live.py
RESULT: (..., MemoryCrystallizationV1(... links=[CrystallizationLinkV1(
    target_crystallization_id='b65bc9ca-ca05-41fb-bab5-f4c1c2541267',
    relation='contradicts', confidence=0.95, note='concept_relation_llm')], ...), 'proposed')

# 3. Confirmed persisted (not just returned) via direct Postgres query:
$ psql -c "select from_crystallization_id, to_crystallization_id, relation, confidence, note
           from memory_crystallization_links where relation='contradicts';"
 429b4c0e-... | b65bc9ca-... | contradicts | 0.9499999999999999... | concept_relation_llm

# 4. Test crystallizations cleaned up (deprecated/rejected) after verification, not left as
#    permanent pollution:
$ curl .../crystallizations/429b4c0e.../reject -> 200
$ curl .../crystallizations/b65bc9ca.../deprecate -> 200
```

## Review findings fixed

Self-caught, not from a separate review pass: my own first two live-test attempts each surfaced a genuine problem (the numpy incompatibility, and a same-window Jaccard collision with a leftover row from an earlier broken test-harness attempt where a shell-quoting bug had passed an empty subject) — both root-caused and fixed before concluding the underlying feature itself works. No material findings against the actual Dockerfile/requirements.txt diff, which is a two-line, low-risk, purely additive change mirroring an already-established pattern in three sibling services.

## Restart required

Already executed live during this session:

```bash
docker compose --env-file .env --env-file services/orion-memory-consolidation/.env \
  -f services/orion-memory-consolidation/docker-compose.yml up -d --build
```

If merged and redeployed elsewhere: same command. `CONCEPT_RELATION_RESOLUTION_ENABLED` stays operator-controlled per `.env_example`'s existing default (`false`) — this patch only ensures the feature actually works on hosts that choose to enable it, it does not change the default.

## Risks / concerns

- Severity: Low — `chromadb==0.4.24`/`numpy==1.26.4` are the same versions already running in three other services on this host; no new version-compatibility surface introduced beyond what's already proven elsewhere.
- Severity: Low — this session's earlier env-sync-script bug (now fixed, PR #996) is what caused bug A in the first place. No further action needed here; that fix is already merged.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...fix/concept-relation-resolution-missing-chromadb-dep?expand=1`
