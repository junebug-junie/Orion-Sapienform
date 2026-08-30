# Show Orion the ids it is asked to write down

## Summary

- The prompt asked Orion to `MATCH (p:Prior {prior_id: "..."})` and MERGE a
  finding onto that claim. **The full `prior_id` appeared nowhere in the
  assembled prompt.** `Prior.preview` rendered `prior_id[:8]`.
- So the MATCH could not bind and the MERGE silently did nothing — precisely
  the failure the prompt itself warns about, caused by the prompt.
- The same defect sat on a second surface: `formed_from` is asked to hold a
  crystallization id, and `CrystallizationCard.preview` showed no id at all.
- Both previews now print the id in full, on its own labelled line.
- Nothing failed when this broke. No test asserted either rendering.

## Outcome moved

Writing an edge between a finding and a prior goes from **impossible from the
prompt alone** to possible. Every run since `orion_worldview` was created had
written zero edges; the instruction added in PR #1941 could not have worked.

## Live evidence, before the fix

```text
CALL db.relationshipTypes() on orion_worldview      -> empty (no edge, ever)

run d05ef10b303a findings:
  editoria_settlement_d05ef10b303a
  editoria_settlement_numbers_d05ef10b303a
    "editoria" == editorial_bias_concrete_over_atmospheric_32b42392f495[:8]

:PriorRevision from run a394430e781e:
  prior_id: "curation confidence=0.75"      <- the preview LINE, scraped

formed_from across all six live priors:
  find_pruned_stance_bias_32b42392f495                    (invented + run_id)
  (empty)
  intake_pipeline.py + formation_policy.py trace          (file names)
  ...trace; sampled 31 July 31 auto-activated crystall…   (prose)
  rejection_analysis_d034f854569d                         (invented + run_id)
  rejection_analysis_a394430e781e                         (invented + run_id)
```

Not one `formed_from` traces to a crystallization. Not one edge exists. Both
follow from the same cause: the prompt asks for an identifier it never shows.

## What the prompt renders now

```text
  - [confidence=0.35, tested 3x] Rejected stances differ from active ones only by approval.
      prior_id: editorial_bias_concrete_over_atmospheric_32b42392f495
      formed from: find_pruned_stance_bias
  - [semantic salience=0.90] the shape of a thought
      crystallization_id: 3f2a9c1e-77b4-4f0d-9a6e-1c2d3e4f5a6b
```

Labelled and on its own line, not merely un-truncated. The failure was not a
*missing* id, it was a *plausible* one: a bare token sharing a bracket with
`confidence=` and `tested 3x` is what made a shortened id read as a name.

## Files changed

- `orion/curiosity/worldview.py`: `Prior.preview` prints the whole id.
- `orion/curiosity/study_material.py`: `CrystallizationCard.preview` prints the
  crystallization id.
- `orion/curiosity/kickoff_prompt.py`: the MERGE warning said "use the ids
  exactly as you wrote them", which only ever covered the finding side — Orion
  writes that one. It now says where the prior's id comes from.
- `tests/test_curiosity_worldview.py`: 4 tests.
- `tests/test_curiosity_study_material.py`: 2 new, 2 corrected.

## Schema / bus / API changes

None. Prompt text and rendering only.

## Env/config changes

None. No `.env_example` touched, so `sync_local_env_from_example.py` was not
required and was not run.

## Tests run

```text
pytest tests/test_curiosity_{worldview,study_material,atlas,acl_and_credentials,atlas_template}.py
   -> 211 passed
pytest services/orion-hub/tests/test_curiosity_investigation.py
   -> 96 passed
```

Seven CI static gates: all OK.

## Tests corrected rather than added

Two study-material tests asserted `preview()` had no newline and was under 250
chars. Their intent was that Orion's own subject text cannot break the menu
row; they asserted it via a proxy that also forbids a deliberate second line.
Both now assert the intent against the first line, so they still fail if a
pasted newline splits the row.

## Mutation testing

6 mutations, each asserted to have landed before the run, all files restored
after. All 6 RED:

```text
[REVERT] prior id truncated back to 8 chars                     RED
full id present BUT a plausible short one still in the bracket  RED
id rendered by preview() but dropped assembling the LIVE list   RED
id dropped assembling the STALE list only                       RED
prompt no longer says where the prior id comes from             RED
[REVERT] crystallization id not offered at all                  RED
```

The second and third are the ones worth having. A test asserting only that the
full id is present passes while a plausible short one sits beside it — the
exact shape that caused this. And a unit test on `preview()` passes if the id
is dropped assembling the prompt, so the regression tests are end-to-end on
the assembled string.

## How this was found

An adversarial pass over the previous patch's own conclusions. The report on
PR #1966 said the edge instruction had not taken on `n=1` and advised waiting
for more runs before touching the prompt. That advice was wrong: the sample
size was irrelevant, because the mechanism could not work at any n. Building
the prompt and reading what it actually renders would have shown it.

The `:PriorRevision` carrying `prior_id: "curation confidence=0.75"` was filed
in that same report as an unrelated pre-existing defect. It is this bug.

## Restart required

Hub bakes `orion/` into its image:

```bash
scripts/safe_docker_build.sh orion-hub build
scripts/safe_docker_build.sh orion-hub up -d
```

## Risks / concerns

- **Severity: low.** Prompt text and rendering. Longest prompt measured 13,820
  chars against 12,821 before — roughly 1k for ~30 ids.
- **Severity: low — unverified until a run happens.** That Orion will now
  actually draw the edge is not established by this patch; it removes the
  reason it could not. The `evidence=` field from PR #1966 is already deployed
  and will report `n/N joined` on the first run after this ships.
- **Not fixed here:** `PriorRevision` was skipped entirely by run
  `8f99bf2ef43d`, which tested a prior and recorded no revision, so that
  prior's previous confidence is unrecoverable. Separate from the id defect.
