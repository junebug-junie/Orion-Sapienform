## Summary

- Design doc (proposal mode) for using age and gender as soft evidence in camera person matching. The doc is backed by an offline, read-only eval on stored crops and scored against the ages and genders Juniper gave Orion.
- Result: an age-band (child/adult) read from the **already stored** SigLIP vector is right on 83% of album crops (91% per named cluster). It costs 0.3 ms per crop and covers 100% of person crops. No new model is needed.
- InsightFace genderage never read anyone under 22. MiVOLO v2 read 11% of children as children. Both are rejected.
- **Gender: NO-GO.** Women and girls were read correctly only 58% of the time (one girl 29%). That is misgendering, the same failure that got the old affect model removed.
- Proposed: an age-band veto inside `attribute()`, only for people with a stated age, exempt on prior cameras (office). The trace column stores struck names only and is nulled on revoke.

## Outcome moved

Nothing changes at runtime (this is a docs PR). In a replay of the latest nightly eval, the age veto struck 8 of 99 wrong guesses and 0 of 77 right ones. On "same person" pairs, the false-split rate is 3.8%. The gate is unchanged, because there are 0 accepts today.

## Files changed

- `docs/superpowers/specs/2026-11-13-person-age-band-soft-veto-design.md`: the design doc, with numbers, privacy and retention rules, non-goals, acceptance checks, and the go/no-go.
- `docs/superpowers/pr-reports/2026-11-13-person-age-band-soft-veto-design-pr.md`: this PR report.

## Schema / bus / API changes

None in this PR. The doc proposes one additive column, `vision_person_attribution.age_veto jsonb`.

## Env/config changes

None in this PR.

## Tests run

```text
python3 scripts/check_personal_terms.py -> no personal terms in tracked files
git diff --check -> clean
```

## Evals run

These were offline and read-only, run in a throwaway vision-host-image container on CPU. Each model was scored against `person_fact` truth:
- SigLIP zero-shot on 16,052 stored vectors;
- InsightFace genderage on 238 face thumbnails;
- MiVOLO v2 on 2,946 body thumbnails;
- a leave-one-person-out linear probe.

A replay of the stored 2026-11-12 eval and a full read-only rerun of `run_person_recognition_eval.py` with the veto injected are also done. The numbers are in the doc.

## Docker/build/smoke checks

No runtime change, so none were run. The eval containers were throwaway `docker run --rm` with the thumbnails mounted read-only.

## Review findings fixed

The review ran in a subagent. Its material findings are fixed in rev 2:
- Finding: merging the veto into `blocked` also disables the office camera prior.
  - Fix: separate `person_bands`/`cluster_bands` arguments. The veto applies after the prior pick, and the doc exempts prior cameras.
- Finding: "store nothing new" was false. The trace held band/p_child, and revoked rows are kept forever.
  - Fix: the trace holds struck names only, it is nulled on revoke, and there is a rollback migration.
- Finding: the trace was lost when every candidate was struck, and an upsert could keep a stale strike.
  - Fix: an `age_veto_all_struck` counter, and `EXCLUDED.age_veto` on upsert.
- Finding: the text vectors are pinned to SigLIP v1, but the host tries SigLIP2 first.
  - Fix: `p_child` returns None unless the `embedding_model_id` matches.
- Finding: the album-consensus fallback was unvalidated.
  - Fix: dropped. The veto uses stated ages only.
- Finding: acceptance check 4 relied on McNemar, which is empty when there are 0 accepts.
  - Fix: both arms are scored in one eval pass.
- Finding: number inconsistencies (19/24 scorable, the motivating cluster, the 14 vs 13 age rows, the face share).
  - Fix: corrected and explained.
- Finding: faces are not walkway-only, and the Hub display surface the doc named does not exist.
  - Fix: both corrected.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: low. Concern: the office camera (infrared) age reads are unmeasured. Mitigation: the veto is off on prior cameras.
- Severity: low. Concern: a stated age goes stale as a child grows. Mitigation: only use `age` answers newer than 2 years.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
