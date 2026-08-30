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

## Review findings fixed

Eight findings. All eight fixed. Every one was about a surface the fix should
have covered and did not — the reviewer found no defect in the change itself.

- **Finding: the fix created a new instance of the bug it fixes.** Printing
  `crystallization_id` hands Orion a citeable id for a Postgres row that is a
  node in neither graph, while the prompt's exception paragraph named the
  Atlas as the *only* property-instead-of-edge case. Orion could now MERGE
  onto a `:Crystallization` that does not exist — the same silent no-op,
  relocated onto the surface this PR opened.
  - Fix: the paragraph now names `crystallization_id` and `decision_id` as
    Postgres rows, property-only, "never in a MATCH".
  - Evidence: mutation reverting the whole paragraph → RED.

- **Finding: `RECENT_SETTLED_CYPHER` never returned `prior_id`,** and the
  settled list printed none — four lines above "Nothing stops you reopening
  one", which is a `MATCH (p:Prior {prior_id: "..."}) SET`. Same defect class,
  same function, left unfixed by the first pass.
  - Root cause of the miss: my adversarial probe passed a fake id into the
    `claim` slot of a `(claim, status)` tuple and read it back out of the
    rendered claim, so the surface cleared a check it had never been subject
    to. The second false-positive probe of this session.
  - Fix: the query returns the id, the tuple carries it, the list prints it.

- **Finding: `RelationCard.preview` was left out.** Relations are the other
  menu Orion picks from and the prompt asks for `evidence: "<ids, queries,
  rows you actually looked at>"`. It offered no id in the resolvable case.
  I had scoped it out on a grep for `decision_id` in the prompt, which the
  general wording does not contain.
  - Fix: prints `decision_id`.

- **Finding: `prior_id:` now appears in two visually identical forms** — the
  menu label and the CREATE template's `"<something unique>"` placeholder.
  Given the failure was "a string that resembles an identifier", that is not
  a detail.
  - Fix: the instruction names all three id labels and explicitly warns off
    the placeholder. Mutation removing that warning → RED.

- **Finding: the instruction is emitted on the empty-graph path** where no
  list is printed. Reworded as a legend ("`prior_id:` under a claim") that
  does not assert a list exists; `crystallization_id` and `decision_id` are
  printed on that path regardless.

- **Finding: crystallizations got the id but no prompt line** connecting it to
  the `formed_from` property that justified the change. Covered by the same
  rewrite.

- **Finding: `test_the_preview_never_offers_a_shortened_id_as_the_id` was
  narrower than its own docstring** — it asserted only `_LONG_ID[:8]`, so a
  suffix or hash prefix put back in the bracket would pass.
  - Fix: asserts every prefix and suffix from 4 chars up. Mutation using
    `prior_id[-8:]` → RED.

- **Finding: the label read `formed from:` while the property is
  `formed_from`.** Fixed. This one had no test at all — caught by mutation
  after the fix, not by the review that suggested it.

## Mutation testing, after review

9 mutations, all RED. Two came back GREEN on the first attempt:

- `formed_from` label — a genuinely weak spot. I changed the label on review
  advice and pinned nothing. Test added.
- the exception paragraph — my **mutation** was ineffective, not the test
  weak: it replaced only the paragraph's first line and left every asserted
  string in place. Re-run against the whole paragraph → RED.

## Observed, not fixed

`_priors_section` returns early when there is neither a live nor a stale prior
(`kickoff_prompt.py:132`), so a graph holding **only** closed priors never
renders the settled list — and cannot be told it has claims it could reopen.
Pre-existing, out of scope, recorded in the test that tripped over it.

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
