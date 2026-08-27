# Key the dev-economics ledger scan by transcript path, not session_id

## Summary

- `_scan_totals()` keyed its per-tick baseline dict by `session_id`, which is not unique: a session and every subagent it dispatches share the parent's `session_id` while each owning its own transcript file and its own `SessionUsageRecord`.
- The dict comprehension therefore kept only the last record per `session_id` and silently discarded the rest — measured on the real transcript tree as **1250 records collapsing to 93 keys (1157 records, 92.6%, discarded)**, with one `session_id` owning 117 files.
- Changed the key to the record's **resolved** `transcript_path`, the only field that is genuinely one-per-record. After the fix the same tree keeps **1251 entries including 1158 subagent transcripts**.
- Added a guard so a *re-discovered* transcript seeds the baseline instead of republishing its entire history as one tick's spend.
- Nothing downstream needed changing: `aggregate_session_deltas` already splits on `is_subagent`, and the schema already carries `subagent_transcript_count`.
- Added a regression test that fails against the old keying (`assert 1 == 3`) and passes against the new.

## Outcome moved

The ledger stops systematically undercounting subagent token/cost usage — the exact failure `SessionUsageRecord`'s own docstring warns about ("folding subagent token/cost usage into 'not really part of the ledger' would make the ledger's own stated purpose (real $ cost accounting) systematically undercount").

The visible symptom nobody had traced: **`subagent_transcript_count` was exactly 0 on all 1197 live ticks in the preceding 14 days**, while `session_count` summed to 996. A schema field that had never once been non-zero in production.

## Current architecture

- Service: `orion-cocreation-signals`
- Producer: `app/producers/dev_economics.py`, `dev_economics_loop` on a ~900s cadence
- It holds each transcript file's last-observed cumulative totals in-process (`last_totals`) and publishes the real **delta** since the last tick, rather than window-filtering by `started_at` (a 2026-08-12 code review caught that approach dropping every multi-tick session after its first tick).
- Channel: `orion:substrate:dev_economics_ledger` → `DevEconomicsLedgerV1` → `dev_economics_ledger_log`

## Architecture touched

One function's dict key, plus the type annotations and docstrings that described it. No service boundary, contract, schema, env key, or bus channel changed.

## Files changed

- `services/orion-cocreation-signals/app/producers/dev_economics.py`: key `_scan_totals` by `transcript_path`; annotations `dict[str, ...]` → `dict[Path, ...]` at the three sites; diff-loop variable renamed; module docstring's "keyed by `session_id`" claim corrected; cold-start log field `session_count=` → `transcript_count=` (it counts files, not sessions).
- `services/orion-cocreation-signals/tests/test_dev_economics_producer.py`: new regression test; existing end-to-end test's `assert "s1" in current_totals` updated to the path key.

## Why it survived until now

Every other test in that file monkeypatches `_scan_totals` away, so the real dict keying was never exercised. The one genuinely-unmocked end-to-end test uses a **single** transcript file — no sibling to collide with. And `diff_session_record`'s `max(0, ...)` truncation guard (added for a real case: a transcript being rewritten between ticks) laundered the resulting negatives into clean zeros, so the loss never surfaced as an error.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: `DevEconomicsLedgerV1.subagent_transcript_count` will be non-zero for the first time, and `total_tokens` / `total_estimated_cost_usd` will step up substantially now that subagent deltas actually reach the aggregate. The event shape is unchanged; the magnitudes are corrected.
- Compatibility notes: historical `dev_economics_ledger_log` rows remain undercounts. Any trailing-window comparison spanning the deploy boundary will see a step change that is a **measurement correction, not a real spend increase**. See Risks.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: not applicable, no env surface touched
- local `.env` synced: not applicable, no env template changed
- skipped keys requiring operator action: none

## Tests run

```text
services/orion-cocreation-signals/tests/test_dev_economics_producer.py + orion/dev_economics/tests
71 passed, 18 warnings in 2.40s

Mutation checks (each guard reverted in turn, suite re-run):
  test_scan_totals_keeps_every_transcript_when_subagents_share_a_session_id
    AssertionError: assert 1 == 3
  test_score_tick_real_end_to_end_against_real_transcript_files
    AssertionError: PosixPath(...) in {'s1': ...}
  2 failed, 6 passed

  drop .resolve():
    FAILED test_scan_totals_does_not_double_count_a_symlinked_transcript
  disable the re-discovery guard:
    FAILED test_rediscovered_transcript_is_seeded_not_republished_as_growth
      AssertionError: assert 1000000000 == 0
```

## Live verification (real transcript tree, not a fixture)

```text
before (session_id key):     93 entries kept
after  (resolved path key):  1251 entries kept, 1158 of them subagent records
                             distinct session_ids still 93
                             visible cumulative tokens 15.21B -> 18.74B  (1.23x)
                             scan wall time 7.6s against a ~900s poll interval
```

These absolute counts are a live, growing, Claude-Code-pruned tree and will not
reproduce exactly — a measurement 40 minutes earlier read 1282/98. The collapse
and the ratio are the reproducible facts; the source docstring says so too.

```text
psql: SELECT count(*), count(*) FILTER (WHERE subagent_transcript_count > 0),
             sum(subagent_transcript_count), sum(session_count)
      FROM dev_economics_ledger_log WHERE observed_at > now() - interval '14 days';
  1197 | 0 | 0 | 996
```

## Deterministic gates run

```text
check_env_key_single_source                PASS
check_scripts_dir_no_stdlib_shadow         PASS
check_service_env_compose_parity  <svc>    PASS
check_settings_defaults           <svc>    n/a (service not in the checker's allowlist)
git diff --check                           clean
```

## Docker/build/smoke checks

Not rebuilt in this branch — the running container carries the old code until redeployed. Restart commands below.

## Review findings fixed

Review ran in a subagent per CLAUDE.md §12. It independently reproduced the bug against both the live tree and the live DB, confirmed the core fix, and returned four material findings — all fixed.

- Finding: **A first-seen path publishes its entire cumulative history as one tick's delta.** Correct for a genuinely new transcript, catastrophic for a re-discovered one — and this patch widened the exposure from 93 keys to ~1250. Two live mechanisms can transiently hide a file and hand it back: `os.walk(..., onerror=lambda _exc: None)` silently drops a whole subtree on any scandir error, and unresolvable absolute symlinks are skipped per-file until an operator fixes the mount this service's own compose comment tells them to fix. Largest single session subtree on the real tree is 1.28B tokens against an int4 ceiling of 2.15B.
  - Fix: added `baseline_taken_at` to `_score_tick` and a `_predates_baseline()` helper. A first-seen path whose transcript started *before* the previous scan seeds the baseline and contributes nothing to the event; one that started after is published in full. Records with no parseable `started_at` are treated as new (they carry no usage in practice, so they cannot manufacture a spike, whereas suppressing them could drop real spend). The loop advances `baseline_taken_at` in lockstep with `last_totals`, so a failed publish keeps both.
  - Evidence: mutation test — disabling the guard fails `test_rediscovered_transcript_is_seeded_not_republished_as_growth` with `assert 1000000000 == 0`.

- Finding: **Symlinked transcripts would now be double-counted.** Claude Code writes some cross-project subagent transcripts as absolute-path symlinks back into the same root, and `os.walk` yields a symlink-to-file in `filenames`, so one underlying file is walked twice under two spellings. `session_id` keying happened to collapse those; raw path keying would count the tokens twice. Latent today (0 symlinks in the tree right now) — but latent only because the mechanism the compose file is configured to enable is currently absent.
  - Fix: key on `r.transcript_path.resolve()`. Also covers hardlinks.
  - Evidence: mutation test — dropping `.resolve()` fails `test_scan_totals_does_not_double_count_a_symlinked_transcript`.

- Finding: **This report's headline risk number was wrong by ~8x, in the misleading direction.** It claimed "~10x" while its own Summary said 17.5%.
  - Fix: re-measured and corrected to 1.23x tokens / ~1.15-1.19x cost, with the method stated so it can be re-derived.
  - Evidence: `sum(all records) / sum(session_id-keyed records)` = 18,737,921,309 / 15,208,186,916 = 1.2321.

- Finding: **The "unbounded, no eviction" risk was not real.** The loop does `last_totals = current_totals` — a full replacement, so vanished paths are evicted immediately and the dict is bounded by files on disk, not history (~1.8 MB traced).
  - Fix: removed that risk; replaced it with the real growth curve (full re-parse of every transcript every tick) and the int4 column ceiling.
  - Evidence: `dev_economics.py` loop body; measured scan time 7.6s.

- Finding: **The source docstring froze non-reproducible numbers** (1282/98/20.49B) that contradicted this report's own verification block (1249/93/18.72B) and the reviewer's independent run (1250/93/18.73B).
  - Fix: docstring now cites 1250/93 and the 1.23x ratio, states explicitly that the tree is live and pruned so absolute counts will not reproduce, and gives the two-line re-measurement recipe.

- Finding: **The pre-existing test fakes returned `str` keys** against the real `dict[Path, ...]` contract — a fake that cannot represent the real return shape, which is exactly how this bug survived every test in the file.
  - Fix: added a `_totals(*records)` helper that keys by the record's own resolved `transcript_path`; all 7 hand-written literals replaced.

- Finding (nit): module docstring still read "per-session instead of per-repo-HEAD".
  - Fix: now "per-transcript-file".

Categories the review explicitly cleared: no consumer misbehaves on the step change (`orion/bus/channels.yaml:2596-2604` lists one consumer, `orion-sql-writer`, which persists unconditionally; the Hub route is pure display with no EWMA by design; `quota_budget.py` is a pure advisory library wired into no allocator); the cold-start log rename is safe (`cocreation_dev_economics_cold_start` appears in exactly one place — the line that emits it); deleted transcripts do not leak, and the patch actually removes a worse case (cross-file diffing when a session's winning file changed identity between ticks); no env, schema, channel, registry, or dependency surface changed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-cocreation-signals up -d --build
```

On restart the loop cold-starts (`last_totals is None` → scan, **no publish**), so the first tick after deploy does not emit a spurious catch-up spike. The first published delta covers only growth after the cold-start scan.

## Risks / concerns

- Severity: medium
  Concern: reported spend steps up at deploy. Measured against the real tree, the correction is **1.23x tokens (15.21B → 18.74B) and ~1.15-1.19x cost** — not the ~10x an earlier draft of this report claimed. Anything baselining on `dev_economics_ledger_log` across the boundary will read the correction as a modest spend increase.
  Mitigation: a known, dated, one-time measurement correction of roughly +20%. Note the deploy timestamp when interpreting any trailing window that spans it. `docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md` draws on this table; its conclusions should be re-read with a +20% correction, not a 10x one.

- Severity: low (follow-up, not fixed here)
  Concern: `dev_economics_ledger_log.total_tokens` is a Postgres `integer` (ceiling 2,147,483,647). Largest observed real tick is 129,974,054 — ~16x headroom, ~13x after this correction. The re-discovery guard removes the mechanism that could have blown straight through it, but the column is still narrower than the data it holds warrants.
  Mitigation: widen `total_tokens` / `total_cache_read_input_tokens` to `BigInteger`. Deliberately not done here — it is a live-table migration and belongs in its own patch.

- Severity: low (pre-existing, unchanged by this patch)
  Concern: `_scan_totals` fully re-parses every transcript on every tick — 7.6-32s wall time for ~1250 files, scaling with total corpus bytes forever, CPU-bound inside `asyncio.to_thread` alongside the service's other producers.
  Mitigation: comfortably inside the ~900s poll interval today. Worth an mtime-based skip if the corpus keeps growing. Note this is the real growth curve — `last_totals` itself is **not** unbounded: the loop replaces the dict wholesale each tick, so vanished files are evicted immediately, and the traced footprint is ~1.8 MB.

- Severity: informational
  Concern: this does **not** rehabilitate the dollar axis for budgeting. The `--allowance-usd` denominator was separately refuted against 15 real rate-limit events (`docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md`), and the per-session vs machine-wide windowing mismatch stands on its own. This fix makes the numbers less wrong; it does not make them a budget.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1916
