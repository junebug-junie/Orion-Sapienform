# Graphify local storage

Graphify outputs now live under `/mnt/storage-warm/orion-graphify`. They are ignored by Git, including archives and query notes. Existing Git/LFS history is retained without rewriting it. The legacy LFS merge helper remains available for old branches and recovery, with no byte or node ceiling.

Install from this repo with `scripts/install_graphify_local.sh`. It preserves the installed Graphify entry point as `graphify-orion-upstream`, installs a repo-aware launcher, and migrates the current output. Other repositories keep their original Graphify behavior. Run the installer again after upgrading Graphify. The adapter targets Graphify 0.9.15's security check: unlike its `GRAPHIFY_MAX_GRAPH_BYTES=0` behavior (which restores 512 MiB), Orion imposes no graph byte ceiling. JSON integrity validation and browser visualization bounds remain.

`graphify-out` is a whole-directory symlink, so native atomic file replacements continue to work. Every linked worktree gets its own directory keyed by its Git worktree metadata path. `main/graphify-out` belongs to the primary checkout; `worktrees/<id>/graphify-out` belongs to an individual worktree. A fresh worktree copies the published bundle. `ORION_GRAPHIFY_STORAGE_ROOT` overrides the root for another host or isolated tests. Create the root with ownership appropriate to the operator first; initialization fails if storage or a valid seed is unavailable. It never manufactures an empty graph.

Migration copies and SHA-256 verifies every file, including dated archives, notes and caches, before replacing the directory with a symlink. The original directory is moved into `original-*` beside the new bundle. These copies and historical LFS objects remain recovery sources; this patch does not prune them. Worktree deletion cannot remove the external data.

Use `scripts/safe_graphify_update.sh` for updates. A lock covers backup, update and validation. Accepted bundles receive retained local checkpoints; the session guard compares against that checkpoint instead of Git HEAD. A corrupt or unaccepted bundle is preserved in `unaccepted-*` before the full checkpoint is restored. Recovery snapshots have no automatic retention ceiling. Inspect status with:

```sh
python3 scripts/graphify_storage.py status
python3 scripts/check_graph_worktree_integrity.py --check-only --json
```

Service readers use an explicitly published, immutable checkpoint. Updating a worktree does not publish its graph. After reviewing the extraction, promote it with:

```sh
python3 scripts/graphify_storage.py publish
```

Publication atomically replaces `published/graphify-out`, a relative symlink into the accepted checkpoint. Co-creation signals, self-study enrichment and Cortex Exec mount the storage root read-only at `/graphify` so that symlink resolves inside their containers. Source checkouts remain mounted separately for code and Git evidence. New container mounts need a recreate using each service's normal `scripts/safe_docker_build.sh <service> up -d --build` command.

For a new host with no published bundle, restore an existing bundle (all three of graph.json, manifest.json and GRAPH_REPORT.md) onto disk, then run `python3 scripts/graphify_storage.py init --seed /path/to/bundle`. A bundle from historical Git/LFS must be fully smudged first. Run `publish` to make it available to readers. To roll back a publication, point `published/graphify-out` at an earlier retained checkpoint atomically; do not regenerate or delete graphs as a rollback.

Tests: `python3 -m unittest discover -s tests/scripts -p 'test_graphify_storage.py' -v`. The real-data smoke is `python3 scripts/graphify_storage.py status`, `graphify query 'Graphify storage and consumers'`, and the graph-backed tests in `services/orion-cortex-exec/tests/test_self_study_pass1.py`. Those two real-data tests explicitly skip on fresh CI checkouts without a local graph.

The CLI also checkpoints legitimate `reflect`, `label`, and other sidecar changes. `watch` holds the transaction lock only during each rebuild, so an idle watcher does not block queries. Readers distinguish new publications even when `built_at_commit` stays the same. Run the installed-Graphify integration eval with that installation’s Python interpreter: `python scripts/evals/graphify_local_storage_smoke.py` (real 513 MiB load/merge plus native reflection and recovery guard).
