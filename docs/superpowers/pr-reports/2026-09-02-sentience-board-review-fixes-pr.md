# PR #2030 — Sentience board: review findings, CI wiring, merge repair

https://github.com/junebug-junie/Orion-Sapienform/pull/2030

## Summary

PR #2026 merged to `main` while its review fixes were still in flight, so **main currently carries the board with every defect the review found.** This lands them, rebased onto current main.

- Four HIGH fixes, all verified live against the running system.
- Wires the gate into CI — the one finding review left unfixed, and the one that matters most.
- Repairs two Makefile bugs that `c34ad7475`'s merge introduced on main.

## Outcome moved

**On main right now, before this PR:** the board's headline fact (the retention ceiling) can never render inside the Hub container; one genuine drift blanks the whole board and names the wrong cause; a page load freezes every other Hub route; the `make` target exits 127; and nothing runs the gate at all.

## Review findings fixed

- Finding: `raw_connection()` is not autocommit — one failing claim aborts the transaction and every later claim fails with `current transaction is aborted`, so a single real drift renders the whole board ERROR under a false cause. Same call also dropped the read-only guarantee `instruments.py` asserts in its own docstring.
  - Fix: `isolation_level="AUTOCOMMIT"` + `default_transaction_read_only=on`.
  - Evidence: live — failing claim no longer poisons later ones (19,771 still returned); `CREATE TABLE` → `cannot execute CREATE TABLE in a read-only transaction`. Mutations M7/M8 red.
- Finding: both handlers `async def` while doing Postgres reads, a subprocess, and a ~4,300-file walk — freezing every other Hub route and websocket.
  - Fix: sync `def`, matching the sibling `attention_organ_routes` it claimed to mirror.
  - Evidence: `test_handlers_are_sync_so_they_cannot_block_the_event_loop`; mutation M6 red.
- Finding: `REPO_ROOT` ignored `ORION_REPO_ROOT`. Verified inside `orion-athena-hub`: `/app/services` does not exist, and `/app/scripts` is **Hub's own** scripts dir — so a `__file__`-derived root resolved paths against the wrong tree, and the retention ceiling could never render.
  - Fix: house convention (`ORION_REPO_ROOT=/repo`, already set in Hub's compose).
  - Evidence: `docker exec` confirming both paths; retention now resolves from `.env` (live) not `.env_example`.
- Finding: `storage_note` rendered only on the `row_count === null` branch, so it was dropped for every instrument that has rows — exactly where it matters. `goal_provenance`'s "singleton, no history by construction" note had never appeared on the page.
  - Fix: the note rides along in the "writing now" cell.
- Finding: `top1_concentration_root_caused` could never fire — an `absent_from_repo` claim whose own note says the follow-up would land under `docs/`, which the scan excludes. A detector structurally incapable of firing reports green forever.
  - Fix: converted to `kind: manual` (`"NOT ROOT-CAUSED"`) so it reads as unrun, not passed.
- Finding: false liveness. `prediction_error_domains` pointed at `substrate_field_state` (the whole-field per-tick snapshot) and the hand-run clustering probe at `substrate_attention_frames`; both rendered "124k rows, 1m ago" as if it were the instrument's own activity — the precise failure this board exists to catch.
  - Fix: reclassified `derived`, with an explicit INPUT-not-output note.
- Finding: `consumers_for()` called without `exclude_paths`, so a registry declaring a token could be reported as consuming it.
  - Fix: `exclude_paths` passed; docstring's parity claim with `check_metric_lineage.py` is now true.

## The finding review declined, fixed here

**Nothing ran the gate.** It was in no workflow, hook, or schedule — the manifest's stated purpose was made *possible*, not *delivered*. A gate nobody runs is the decoration this whole manifest argues against.

Adds `--static-only` and wires it into `orion-static-gates.yml`, whose own rule is DB-free gates only. SQL claims report **SKIPPED** — deliberately a third state: `HOLDS` would let an unrunnable gate read green, `ERROR` would make the lane permanently red and get it switched off. The static lane still catches a manifest naming a module or entrypoint that no longer exists, and an unlock narrative past its review window.

Also removed the ripgrep dependency that mode would have assumed — a pure-Python fallback, verified **byte-identical to ripgrep on a pattern with real hits**, not on an empty result that would pass even if broken.

## Two Makefile bugs from the merge, not from this rebase

Both confirmed by reading the `origin/main` blob directly:

- `check-sentience-instruments` gained a **stray second recipe line running the postgres headroom gate**.
- That stray line was `postgres-headroom`'s own new recipe. The merge kept PR #2029's comment explaining why the target must use `$(METRIC_PYTHON)` and dropped the code implementing it — leaving `.venv/bin/python`, exit 127 in every worktree, with a comment on main asserting the opposite. **Restored** — someone else's fix, silently reverted.

## Tests run

```text
orion/sentience_striving_program/tests            21 passed (was 14)
services/orion-hub/tests/test_sentience_program_api.py   9 passed (was 6)

make check-sentience-instruments           All claims hold.  exit 0
make check-sentience-instruments STATIC=1  All claims hold.  exit 0
make -n postgres-headroom                  resolves the main checkout's venv from a worktree
```

## Mutation testing

Eight mutations, each red, baseline green:

```text
M1 recorded 16 -> 5            DRIFT     (reproduces the real 2026-08-20 -> 09-02 drift)
M2 deleted symbol returns      DRIFT
M3 module path moved           MISSING
M4 review 609d stale           STALE
M5 duplicate router registration   FAIL
M6 handler made async again        FAIL
M7 read-only enforcement dropped   FAIL
M8 autocommit dropped              FAIL
```

## Env/config changes

None.

## Restart required

```bash
sudo docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build orion-athena-hub
```

Hub mounts the repo read-only from `ORION_HOST_REPO_ROOT`, so this needs merging before the container sees it.

## Risks / concerns

- Severity: low. Concern: the stale branch `feat/sentience-instrument-board` still exists on the remote — I recreated it by pushing before noticing #2026 had merged, so it diverges from what landed. Safe to delete; left alone rather than deleting a branch unasked.
- Severity: informational. Concern: `--static-only` cannot catch a drift in the SQL-backed claims; only the full local/cron run can. That is inherent to a DB-free CI lane, and SKIPPED says so on every run rather than implying coverage.

