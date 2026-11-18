## Summary

Orion's nightly memory step retires an old memory when a newer one changes or corrects it. This PR makes that step careful, measured, and reversible, all on local models.

- **Classify, don't pair.** A local model names how the new memory relates to the old one: update, correction, refine, compatible, or unrelated. It answers in JSON three times at temperature 0.7. The step acts only when all three votes agree on update or correction, each says "same attribute", and each is at least 0.8 confident. Anything else abstains, which leaves both memories visible.
- **When a fact stopped being true.** A retired row now records `invalid_at`, `invalidated_by`, and `invalid_reason`. An update was true until the new memory's time, and a past-tense read can still reach it (`EPISODE_VALID_AT_SQL`). A correction was never true, and nothing serves it again. Undo clears all three.
- **Measured before trusted.** `scripts/memory_supersede_eval.py` scores the classifier on made-up fixtures in the repo and on real pairs kept host-side. Write time applies only while the latest recorded eval for the current check version passed. Without that, it proposes and logs.
- **Juniper's words win.** Suppose a new memory is in Juniper's voice and is backed by a verified quote from her own message. It acts when every vote says update or correction, at any confidence, and exactly one old row qualifies.
- **Audit trail (#3660).** Every run writes one append-only `memory_supersede_run` row: ids, which candidate legs fired, votes, verdicts, and what was applied. It holds no statement text. The pass also keeps a run-history file next to its manifest, and a per-watched-row trace.
- **Tighter candidates.** A cosine alone no longer makes a candidate pair. Every pair also needs a rare shared word or referent.

## Outcome moved

The step's precision is now measured, and write time trusts it only while the measurement holds. The same 47 labelled pairs were scored three times (26 made-up, 21 real), on the local `memory_distill` route (qwen3.8-flash-next):

| Checker | Acted | Precision | Recall | False acts (34 negatives) | Errors | Passes gate |
|---|---|---|---|---|---|---|
| v2, before (yes/no + "both true?") | 6 | 1.00 | 0.46 (6/13) | 0 | 3 (gateway 5xx, no retry) | no |
| v3 first cut (relation + 3 votes) | 12 | 0.75 | 0.69 | **3** (unanimous, conf >= 0.95) | 0 | no |
| **v3.1, shipped** (relation + checklist + 3 votes) | 8 | **1.00** | 0.62 (8/13) | **0** | 0 | **yes** |

- v3 without the checklist acted with confidence on three hard negatives: a plan against what happened (real), two PR counts on different days (real), and an explicit "in 2019" fact (made up). All three votes agreed every time. So votes alone do not make it safe. The checklist does: the model must say what A is and whether B's own words change it, and an update needs A to be a lasting state.
- Real data: 0 false acts on 16 real hard negatives (both watched trip rows against the home correction, the three pet pairs, 5 dry-run vetoes, 6 near-duplicates). It acted on 0 of Juniper's 5 real revisions. Those are mostly rewording fixes, and the model reads them as refine or compatible. Her confirmation-loop path already retires them, so this is a missed proposal, not a safety gap. The made-up set: 8/8 updates and corrections caught.
- **In-sample caveat, then an out-of-sample check.** The checklist was designed after seeing v3's three false acts on this same set, so the v3.1 row above is in-sample. Nine new pairs were then written that the design never saw (`fixtures/supersede_eval_heldout.jsonl`): a plan against its outcome, counts on different days, readings over different windows, a past period, a changed view, a getaway against home in Juniper's own words, added detail, plus 2 positives. Result: 0 false acts, 2/2 positives caught. They are part of every full run from now on.
- The v2 run used the made-up fixtures before their invented place names were changed to neutral ones. The cases are the same.
- Self-consistency at temperature 0.7 rarely splits on this model; nearly every vote carried confidence 0.95-1.0. The votes mostly guard against a one-off malformed or odd answer. They are not what fixed the false acts.

### Live-store dry run (v3.1, read-only, 2026-11-18 00:56-02:09Z)

`scripts/memory_supersede_pass.py dry-run` ran over the live store. It wrote nothing to the database. Its private files are in `/tmp/memory-supersede-pass-v3/` (directory mode 700).

- 485 live rows, 735 candidate pairs (cosine plus a shared rare word or referent), every one classified, 0 errors in the end. The first pass lost 61 calls: 55 dropped connections when the gateway went away at 01:45Z, and 6 HTTP 5xx. A cached re-run finished them. The pass now retries both a dropped connection and a 5xx up to 3 times. It does not retry a timeout.
- Final relation per pair (majority vote): unrelated 355, compatible 349, refine 20, correction 6, update 2, no majority 3.
- **Proposed: 2, both corrections, both unanimous at 3/3.** Neither is applied. The pass needs Juniper's review, and `apply` also needs a passing recorded eval. Both are borderline, and Juniper should read them before any apply:
  - The first is an energy-numbers correction. The newer row gives a whole-host figure; the older row gives an overhead figure.
  - The second is a "deployed" against "nothing deployed yet" pair that is two weeks apart. It is the same single pair the v2 check proposed on 11-17.
- The checklist blocked 5 pairs where the votes said correction or update. One of them was trip row 1 against the home correction.
- **Watched rows: all six stay active, and none is proposed.** The trip rows and the home correction were asked and abstained (checklist or compatible). The two pending pet rows are never candidates (the open-ask rule). The pet correction was asked 6 times: unrelated 3, compatible 3.
- #3660 live: the two dry runs left two lines in `memory_supersede_pass_runs.jsonl`. Each watched row's `trace` lists every pair it was in, with its legs and votes and no text.

## Current architecture

PR #3617 added write-time supersede: the distill's check node asked a two-question yes/no check per candidate pair. A cosine of 0.80 alone could make a candidate. A "yes" set `status = 'superseded'`, which is binary, with no notion of when the fact stopped being true. Precision was never measured. The pass manifest was overwritten on every run. PR #3669 (merged) already stopped degraded applies and closed the eval gate.

## Architecture touched

- `orion/memory/episode/supersede.py`: the v3 classifier (`classify_pair`, `Vote`, `Classification`), the Juniper clear-target path in `check_pairs`, bi-temporal `apply_link`/`undo_link`, `record_run`, and candidate legs.
- `orion/memory/episode/supersede_eval.py`, `scripts/memory_supersede_eval.py`, `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: the eval.
- `orion/memory/lived_memory.py`: `EPISODE_VALID_AT_SQL`, `fetch_valid_at` (past-tense read).
- `orion/memory/episode/store.py`: persist writes the run audit row in a SAVEPOINT.
- `services/orion-durable-runs/app/episode_distill_graph.py`: votes carry their temperature; persist passes the plan.
- `scripts/memory_supersede_pass.py`: v3 classifier, a vote-aware cache, run history and watched trace, and a version check on apply.
- Migration `services/orion-sql-db/manual_migration_episode_memory_temporal_v1.sql` (additive, `REQUIRED-BY: orion-durable-runs`), with a rollback and a grant file.

## Files changed

- `orion/memory/episode/supersede.py`: the classifier, validity marks, audit writer, and candidate legs.
- `orion/memory/episode/supersede_eval.py`: labelled items and scoring (`passed` = zero false acts, zero errors, enough real items).
- `orion/memory/episode/fixtures/supersede_eval_synthetic.jsonl`: 26 made-up pairs that cover every relation.
- `scripts/memory_supersede_eval.py`: `build`, which makes the real set (read-only DB) into `~/.orion/evals/` (mode 600), and `run`, which scores and with `--record-db` records the verdict.
- `scripts/memory_supersede_pass.py`: v3, history, trace, and the check-version refusal.
- `orion/memory/lived_memory.py`: the as-of reader.
- `orion/memory/episode/store.py`: the audit row.
- `services/orion-durable-runs/app/episode_distill_graph.py`, `settings.py`, `.env_example`, `docker-compose.yml`: temperature per vote; `MEMORY_SUPERSEDE_MAX_CHECKS` now counts model calls (default 36).
- `services/orion-sql-db/manual_migration_episode_memory_temporal_v1{,_rollback}.sql`, `scripts/sql/2026-11-18_grant_orion_readonly_supersede_runs.sql`: schema.
- `docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md`: "Corrections rev 2".
- Tests: `test_supersede.py`, `test_supersede_eval.py`, `test_supersede_pass_script.py`, `test_supersede_pg.py`.
- `.github/workflows/orion-memory-episode-tests.yml`: the path filter now includes the eval script.

## Schema / bus / API changes

- Added: `episode_memory.invalid_at`, `invalidated_by`, `invalid_reason`. Tables `memory_supersede_eval_run` and `memory_supersede_run`. Plan key `decisions`. Link fields `relation`, `path`, `votes`.
- Removed: the v2 yes/no check (`CHECK_PROMPT`, `BOTH_TRUE_PROMPT`, `run_check`) and `DENSE_ALONE`.
- Renamed: `CHECK_VERSION` `supersede_check.v2` -> `supersede_classify.v3`. The eval gate is per version, so v3 needs its own passing eval.
- Behavior changed: an update keeps the old row's `confirmation_state`; only a correction marks it `corrected`.
- Compatibility notes: a pre-v3 checkpoint's links read as `correction` / path `model`, but #3669 already refuses to apply them. A pass pairs file from v2 is refused by `apply`. No bus changes.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- Changed meaning: `MEMORY_SUPERSEDE_MAX_CHECKS` now caps model calls (1-3 per pair). The default went from 12 to 36 (`le` 90).
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: the script reports value drift but does not change existing values, so the local `services/orion-durable-runs/.env` was set to 36 by hand.
- skipped keys requiring operator action: none.

## Tests run

```text
ORION_MEMORY_EPISODE_TEST_DATABASE_URL=<throwaway postgres:16> PYTHONPATH=. pytest -q orion/memory/episode/tests orion/memory/tests
  2407 passed (Postgres-backed included, 0 skipped)
PYTHONPATH=.:services/orion-durable-runs pytest -q services/orion-durable-runs/tests/test_episode_distill_{graph,referent_policy,reconcile}.py
  25 passed
python3 scripts/check_migration_required_by.py      OK (2 new migration files)
python3 scripts/check_env_template_parity.py        PASS
```

New tests cover each refusal and rule:
- the degraded path refuses: index down, index raising, no index, a malformed or failed vote, a closed or stale eval gate, a pre-v3 checkpoint;
- votes: unanimous and confident acts; any dissent, a split, or low confidence abstains;
- Juniper's path: needs one clear target, is dissent-proof, and a budget cut turns it off;
- validity: an update's `invalid_at` and the past-tense read, a correction is never served, undo clears validity, an update never ends before its start, expired rows are left out;
- audit: one row per run, idempotent, no text, and a missing table does not cost the memories;
- the manifest keeps two history lines for two runs, and the watched trace has legs and votes but no text;
- pass apply refuses a word-only pair, a v2 pairs file, and a closed gate.

## Evals run

```text
.venv/bin/python scripts/memory_supersede_eval.py build      -> 21 real items (5 positives, 16 hard negatives) in ~/.orion/evals/ (600)
.venv/bin/python scripts/memory_supersede_eval.py run --checker v2   (archived pre-PR tree)  precision 1.00 recall 0.46 fp 0 errors 3   FAIL (errors)
.venv/bin/python scripts/memory_supersede_eval.py run --checker v3   (v3, no checklist)      precision 0.75 recall 0.69 fp 3           FAIL
.venv/bin/python scripts/memory_supersede_eval.py run --checker v3   (v3.1, this PR)         precision 1.00 recall 0.62 fp 0 errors 0  PASS
```
Not recorded with `--record-db`: the table does not exist until the migration is applied on deploy, and opening the gate is Juniper's call.

## Docker/build/smoke checks

```text
Not run: no Dockerfile or dependency change. The migration was applied to a throwaway postgres:16 by the Postgres-backed tests.
scripts/check_migration_required_by.py: OK. orion.sql_migration_drift.auto_apply_refusals(migration, existing_tables={'episode_memory'}) == [] (fits the post-pull auto-apply allowlist).
```

## Review findings fixed

- Finding: the pass's `--max-checks` split a row's candidate list, so Juniper's "one clear target" could pass on a cut list.
  - Fix: the cap now falls on whole rows.
  - Evidence: `test_a_budget_cut_disables_the_juniper_path`, plus the whole-row cap in `cmd_dry_run`.
- Finding: `build` read `/tmp/.../checks.csv`, which a v3 dry run rewrites in a new format.
  - Fix: `build` defaults to a frozen host-side copy of the 11-17 v2 file and skips rows in other formats.
  - Evidence: `test_build_skips_a_v3_format_checks_file`.
- Finding: the eval gate ignored the vote count and never expired.
  - Fix: the gate needs `votes = VOTES` and a run under 30 days old. `--record-db` accepts only a full v3 run on `memory_distill`. The run stores the gateway's model id.
  - Evidence: `test_eval_gate_reads_the_latest_finished_run_for_this_check_version` (PG) and `test_eval_gate_open_only_on_...`.
- Finding: a malformed answer was cached and replayed forever.
  - Fix: only parseable votes are cached.
  - Evidence: `test_a_malformed_answer_is_never_cached`.
- Finding: the as-of read ignored `expires_at`.
  - Fix: added the filter.
  - Evidence: the expired case in the PG update test.
- Finding: an update could end before the old fact began.
  - Fix: `invalid_at = max(new time, old start)`.
  - Evidence: `test_an_update_never_ends_before_the_old_fact_began`.
- Finding: pass `apply` ignored the eval gate.
  - Fix: `apply` refuses unless `--allow-no-eval` is given.
  - Evidence: `test_pass_apply_refuses_while_the_eval_gate_is_closed` (PG).
- Finding: `fetch_valid_at` has no live reader.
  - Fix: filed #3674 (route past-tense recall to it). The read is UNVERIFIED in chat until then.
  - Evidence: issue.
- Nits fixed:
  - `links_to_apply` keeps only links from the current version that carry a relation.
  - No index call is made once the call budget is spent.
  - Audit run ids carry a uuid suffix.
  - The wording now says model reasons may paraphrase a memory.
  - `passed` needs at least one true positive.
  - The K limit on Juniper's target is documented.
- Second review (the checklist commit):
  - Finding: clamping an update's `invalid_at` to the old start still left an empty window.
    - Fix: `apply_link` refuses an update whose new time is at or before the old start.
    - Evidence: `test_an_update_that_points_backwards_is_refused` (PG).
  - Finding: a malformed checklist field held the whole plan as an error.
    - Fix: such a vote is a non-acting abstain. Kind spacing is normalized. A bool or NaN confidence is an error.
    - Evidence: `test_missing_or_mixed_checklist_fields`.
  - Finding: the `model` column was missing for any hand-applied earlier copy of the migration.
    - Fix: added `ADD COLUMN IF NOT EXISTS`.
  - Finding: the eval evidence was in-sample.
    - Fix: added the held-out set (above).
  - Finding: there were test gaps.
    - Fix: added tests for Juniper's path still needing the checklist, a third vote failing the checklist, missing fields, update kinds, and the cache with an old-format answer.
  - Nits fixed: the gate also needs `route = memory_distill`, the abstain branches are merged, and the cap and cache-key comments are corrected.
- Earlier review on #3669 (the safety PR): four findings, all fixed there.

## Restart required

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

The post-pull rebuild applies the migration first (additive, REQUIRED-BY header). Then, by hand:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "GRANT SELECT ON public.memory_supersede_run, public.memory_supersede_eval_run TO orion_readonly;"
.venv/bin/python scripts/memory_supersede_eval.py build && .venv/bin/python scripts/memory_supersede_eval.py run --record-db
```

Write time stays proposal-only until that eval run is recorded and passed.

## Risks / concerns

- Severity: medium.
  - Concern: chat recall does not yet route past-tense questions to the as-of reader.
  - Mitigation: #3674. Until then an updated fact's history is reachable only by SQL (`EPISODE_VALID_AT_SQL`, orion_readonly).
- Severity: low.
  - Concern: no Hub view of the audit rows.
  - Mitigation: #3673. They are readable by SQL after the grant.
- Severity: low.
  - Concern: write time applies nothing until someone runs the eval with `--record-db`, and the eval needs the host-side labelled set.
  - Mitigation: these are the intended semantics (proposals and audit rows still accrue). The commands are under Restart required.
- Issues: Fixes #3659. Fixes #3660. Opened #3673 and #3674.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3677 (merged 2026-11-18 01:25Z). Follow-up with this report's dry-run section and the pass 5xx retry: see the next PR.
