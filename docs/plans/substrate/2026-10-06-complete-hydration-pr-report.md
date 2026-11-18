## Summary

Falkor's complete-cache loader stopped at the server's 10,000-edge result cap and still reported success. This patch follows reading property graph design section 2 after #2497: scan all pages, validate the staged topology, and only then replace the cache. A failed scan now preserves the prior cache and successful-refresh cursor with an explicit stale receipt.

- Keyset pages use internal database IDs and continue until an empty page, including when the server cap is smaller than the request.
- Reject malformed results, invalid nodes/edges, duplicate stable IDs, incompatible lookup identities, missing/mismatched endpoints and observed concurrent local writes.
- Preserve all parallel edges with compatible lookup aliases and select a deterministic representative for the existing single-ID lookup.
- Expose scan start/end, coverage, staleness, counts and failure information on the local snapshot; preserve existing read return fields.
- Keep neighborhood queries independent of hydration. No new bus events, cognition metrics, env keys or dependencies.

## Runtime evidence

Read-only replay against the running Falkor container, using `GRAPH.RO_QUERY`:

| Reader | Nodes | Edges | Page queries | Elapsed |
|---|---:|---:|---:|---:|
| Deployed unpaged code | 4,973 | 10,000 | 4 unpaged queries | 5.924 s |
| New default, 1,000-row pages | 4,973 | 37,630 | 47 | 25.539 s |
| New explicit 5,000-row pages | 4,973 | 37,630 | 13 | 17.335 s |

Both new scans completed with intact endpoint/type validation and 16,269 additional compatible lookup aliases; every stable edge ID is retained. Baseline incorrectly reported success. Measurements were sequential on a live mutable graph, not a controlled latency benchmark. Default remains 1,000 rows; callers can choose a bounded page size explicitly.

No service deployment, graph repair, deletion, bus access or production write was performed. Deployed service execution of the new path remains **UNVERIFIED**.

## Scope and contracts

Shared substrate store contract plus Falkor backend/client, recording fixture, replay script, regressions, eval and existing CI workflow. `MaterializedSubstrateGraphState.scan_receipt` is optional for other backends; the routed store preserves the primary receipt. No schema registry/channel changes because no persisted or bus payload changes.

Full contract: `docs/plans/substrate/2026-10-06-complete-hydration-contract.md`.

No `.env_example` changes; local env sync is not needed. No env keys skipped. No dependency or Docker changes. The runtime settings file is `services/orion-substrate-runtime/app/settings.py`; there is no root service `settings.py`.

## Verification

- Focused store/backend/dynamics/client regressions: **182 passed** (9.33 seconds).
- Repository static gates: **143 passed** (substrate ladder, schema-skew discovery, SQL migration drift).
- Complete-scan dynamics eval: three page/cap combinations over 43 nodes and 173 edges; identical complete topology, pressure, activation and dormancy results; 3 pressure and 43 activation updates in each case.
- Existing hub-heavy neighborhood eval also rerun.
- Read-only live service smoke: complete scans described above.
- `git diff --check` passed. Existing RDFLib/Redis deprecation warnings only.
- Local evidence under `/tmp/substrate-complete-hydration/`: `live-scan.json`, `live-scan-5000.json`, `tests-final.txt`, `static-tests.txt`, `eval.json`, `neighborhood-eval.json`.

## Review findings fixed

- Finding: legacy native rewrite overwrote the staged deterministic edge representative.
  - Fix: preserve the validated smallest-ID representative through successful and failed rewrite.
  - Evidence: writable native/legacy alias regressions for both outcomes.
- Finding: Redis's row adapter discarded malformed rows before strict page validation.
  - Fix: reject malformed result sets/rows at the shared parser.
  - Evidence: regressions through the real client adapter prove the old cache and refresh cursor remain intact.
- Independent review used `/home/athena/.claude/plugins/cache/claude-plugins-official/superpowers/6.4.1/skills/requesting-code-review/SKILL.md`. Corrective re-review independently passed 108 tests and found no remaining material issues; ready to merge subject to CI.

## Limits and rollout

- Keyset coverage is explicitly **non-atomic**. Undetected external mutation/internal-ID reuse can produce a mixed-time view. Strict historical equivalence requires quiescent data or an atomic export.
- Complete scans cost more than truncated scans. Existing read/write loops can trigger another full refresh. Local consumer migration remains necessary and is the next recommended patch.
- Existing legacy rewrite/cleanup remains best-effort after validation; the read-only diagnostic skips it entirely.
- The stable representative for an ambiguous edge lookup may differ from the previous arbitrary last row. All parallel edges remain in the topology.
- SPARQL complete hydration, node-only consumer migrations, retained reading evidence, assertion lifecycle/projection and shared provenance interfaces remain subsequent patches.

Deployment, when approved, requires rebuilding the consumers from a merged worktree with the operator's existing local env files:

```sh
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-recall up -d --build
```

These commands have not been run. Rollback is a code rollback; durable graph contents need no reversal.

## PR

https://github.com/junebug-junie/Orion-Sapienform/pull/2500
