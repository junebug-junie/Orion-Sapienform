# Graphify local storage and unlimited graph loading

Generated graphs caused LFS uploads and merge conflicts, while Graphify and the legacy union helper imposed file/node ceilings. Graph, manifest, report, archived outputs, notes and caches now live on `/mnt/storage-warm/orion-graphify`; Git holds the implementation and documentation. Existing Git/LFS history is retained.

This follows PR #2199 and is initially based on `feat/general-reading`. Merge #2199 first, then this PR. It does not merge either PR automatically.

## Result and evidence

- The migrated graph retains **77,966 nodes, 169,111 links, 104 hyperedges**, with SHA-256 verification of every original file. Original bytes remain in warm-storage recovery directories. [Machine-readable evidence](2026-09-11-graphify-local-storage-evidence.json).
- Each worktree has independent output. A whole-directory symlink preserves native atomic file replacement. Explicit publication atomically selects an immutable checkpoint for service readers.
- The CLI’s 512 MiB ceiling and legacy merger’s 512 MiB/100,000-node ceilings are removed. Actual **537,919,569-byte** valid JSON was rejected by upstream, then loaded and merged successfully through Orion. A real query also succeeded with upstream’s configured cap set to one byte.
- Local checkpoints replace Git HEAD as recovery authority. Failed or corrupt updates retain rejected bytes and restore the complete accepted bundle. Native reflection/labels are checkpointed; watch locks individual rebuilds.
- Co-creation signals, self-study enrichment and Cortex Exec read the published bundle from a read-only mount. Readers pin one checkpoint for related reads and observe graph-only publications sharing a source commit.
- No new metric, bus event, schema or cognitive capability is introduced. Existing observations retain their producers, formulas and provenance; only artifact location and publication freshness change. The metric-definition drift gate reports zero changes.

## Validation

- Storage regression gate: 10 tests, including lossless migration, separate worktrees, worktree removal, retry after interrupted checkpoint, full-bundle recovery, publication isolation, native-command transaction behavior, watcher lock scope, and graphs above 100,000 nodes.
- Legacy union/LFS regression gate: 10 tests.
- Safe-update, integrity guard and env-sync regression tests: 44 tests.
- Consumer tests cover external storage and publication with unchanged source SHA. Cortex’s original 52-test suite passed against the real local graph; additional focused publication tests pass. A small deterministic fixture supports fresh CI clones without operator graph data; explicitly real-data smokes skip there.
- All three affected Docker service builds succeeded. Network-disabled, read-only container smokes exercised actual graph evidence loading. No production containers were restarted.
- Compose mount/parity gates, script shadowing, metric-definition drift, shell syntax and `git diff --check` pass. CI now refuses indexed `graphify-out` artifacts and runs the local-storage regression gate.
- Native graph refresh was attempted through the safe wrapper. Graphify refused a 77,966→77,858-node rebuild; the wrapper restored the complete original graph/report/manifest. A fresh extraction is **UNVERIFIED**; migration and artifact preservation are verified. The node-loss protection was retained.

## Review findings fixed

Review used the repository’s requesting-code-review skill in a subagent.

- Legitimate reflection/label changes could be undone by recovery. Fix: transact native mutations and checkpoint sidecars. Evidence: actual native `save-result`/`reflect` followed by the guard preserves learning; regression test also covers failure rollback.
- Publication could pair graph and report from different checkpoints. Fix: resolve one directory before both reads. Evidence: producer tests and container reads.
- Interrupted initialization could leave no initial checkpoint. Fix: retry checkpoint creation before linking/returning. Evidence: injected disk-full failure followed by successful retry.
- An idle watcher could hold the lock indefinitely. Fix: transact each rebuild callback. Evidence: watcher regression verifies unlocked observer lifetime and locked/checkpointed rebuild.
- Graph-only publications could be skipped when source SHA stays the same. Fix: observe checkpoint identity and structural values; include publication identity in Cortex memoization. Evidence: same-SHA publication regressions for both readers.

## Operations and rollback

The local launcher is installed on this host; the preserved graph is already on warm storage. Six graph path keys were synchronized into the three operator `.env` files and copied into the worktree for builds; none are tracked. Existing operator values outside those keys were preserved. Source-only historical branches keep the original CLI behavior until they receive the storage code.

After merge, initialize the checkout with `scripts/install_graphify_local.sh`. Recreate the readers from the merged worktree to apply their mounts:

```sh
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
scripts/safe_docker_build.sh orion-self-study-enrichment up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
```

To publish a reviewed local refresh: `python3 scripts/graphify_storage.py publish`. Roll back by atomically selecting an earlier retained checkpoint. Reinstall the launcher after a Graphify upgrade; the adapter is validated against installed Graphify 0.9.15. Local storage availability replaces LFS availability for graph-backed operations; a fresh host needs an existing bundle seed and never receives a fabricated empty graph. [Operations guide](../../graphify-local-storage.md).
