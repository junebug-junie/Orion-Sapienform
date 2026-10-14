## Summary

- **P0:** Orion's prompts stop calling Juniper "they". `stance_react.j2:131` now says "her own life"; the same slip in `chat_quick.j2` and orion-mind's stance handoff (`services/orion-mind/app/stance_handoff.py`, four places) is fixed too. Pronouns now have one source of truth, `config/person_model/people.yaml`, where every value cites where the person said it (Juniper: "My pronouns are she/her", chat 2026-09-01, plus her AI-town card).
- **P1:** new module `orion/person_model/`, parameterized by person node id. People are data, not code: a CI gate fails if any name or node id from `people.yaml` appears in module code. Four contracts registered (`PersonRefV1`, `PersonMentionV1`, `PersonLinkageDecisionV1`, `EntityMergeLogEntryV1`).
- **P2 (dry run only):** pinned GLiNER extractor chosen by a bake-off against BERT/RoBERTa/spaCy; a two-threshold Fellegi-Sunter linkage reducer; a merge-log format that can be undone exactly; and a read-only CLI that turned the live graph into one self-contained HTML review file for Juniper.
- Nothing was written to Postgres or FalkorDB. Applying her decisions is a later step.

## Outcome moved

- The one-time cleanup is a review Juniper can actually do: **3 cards need her call** out of 120 decisions (53 auto-merge, 64 auto-reject). File: `docs/superpowers/evals/2026-10-14-person-cleanup-review.html`.
- The three that need her: "juniper feld" (her full name; the card also covers the matching chat mentions), "Skye" vs "Skye Laflamme" (the card also decides where the graph nodes "skye" and "skye laflamme" go), and "Jay, Jay" (one name, two people; the "jay" node merge waits on her answer).
- Misgendering in the live chat prompt path is gone and has a regression test tied to the pronoun source of truth.

## Current architecture

The graph had 11 nodes for Juniper from four producers, no person nodes for her family, and a referent alias calling SLC "home". Pronouns were stored nowhere; prompts hardcoded "their" for her in places. No linkage, extractor, or merge-log code existed (spec: `docs/superpowers/specs/2026-10-14-person-model-and-identity-card-design.md`, PR #2646).

## Architecture touched

- New package `orion/person_model/` (person, extract, normalize, linkage + `linkage_config.yaml`, merge_log, review_html, tests, evals).
- New contracts `orion/schemas/person_model.py`, registered in `orion/schemas/registry.py` (both `_REGISTRY` and `SCHEMA_REGISTRY`, checked with `resolve()`).
- New data: `config/person_model/people.yaml` (pronouns), `config/person_model/stated_answers.yaml` (her answer: home is Ogden).
- New read-only CLI `scripts/run_person_linkage.py`.
- Prompt text: `stance_react.j2`, `chat_quick.j2`, `services/orion-mind/app/stance_handoff.py`.
- New CI workflow `.github/workflows/orion-person-model-tests.yml`.
- No bus channel, env key, compose, or service runtime change.

## Files changed

- `orion/cognition/prompts/stance_react.j2`, `orion/cognition/prompts/chat_quick.j2`, `services/orion-mind/app/stance_handoff.py`: she/her for Juniper.
- `config/person_model/people.yaml`: pronoun source of truth with evidence.
- `config/person_model/stated_answers.yaml`: home = Ogden, citing her answer on #2646.
- `orion/person_model/person.py`: `PersonRefV1` loader; stated-only pronouns; name fallback instead of guessing.
- `orion/person_model/extract.py`: pinned extractor, replay-stable mention rows.
- `orion/person_model/normalize.py`: folding, tokens, prompt-key harvest from the templates.
- `orion/person_model/linkage.py`, `linkage_config.yaml`: the reducer and its weights/thresholds, with load-time checks that no single feature can force a merge.
- `orion/person_model/merge_log.py`: entries list every write with its before-value; `unmerge_ops` inverts from the entry alone.
- `orion/person_model/review_html.py`: the review page (inline CSS/JS, CSV export, localStorage).
- `orion/person_model/evals/*`: bake-off runner + results, linkage eval, UI check, labels.
- `scripts/run_person_linkage.py`: read-only dry run.
- `orion/schemas/person_model.py`, `orion/schemas/registry.py`: contracts.
- `tests/test_interlocutor_pronouns_in_prompts.py`, `tests/test_person_model_schemas_registered.py`, `orion/person_model/tests/*`.
- `docs/superpowers/evals/2026-10-14-person-cleanup-review.html`, `...-decisions.csv`: the dry-run output.

## Schema / bus / API changes

- Added: `PersonRefV1` (`person.ref.v1`), `PersonMentionV1` (`person.mention.v1`), `PersonLinkageDecisionV1` (`person.linkage.decision.v1`), `EntityMergeLogEntryV1` (`person.entity_merge_log.v1`).
- Removed / renamed: none.
- Behavior changed: none at runtime; nothing publishes these yet (the dry run writes files only).
- Compatibility notes: purely additive.

## Env/config changes

- Added keys: none. Removed / renamed: none.
- `.env_example` updated: no (no service runtime changed).
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Extractor bake-off

55 of Juniper's chat prompts, 34 named-person spans, 19 relation phrases ("my wife"). The label file stores only row ids, a text hash, and offsets; the eval re-reads the text read-only and refuses if it drifted. Labels were made by the implementing agent. Person-mention scores, relaxed overlap matching, AI agents (Orion, Claude, Muse, Lark) not scored:

| Model | F1 at precision >= 0.95 (P / R) | Best F1 (P / R) | CPU ms/prompt |
|---|---|---|---|
| **gliner-multitask-large-v0.5, label "person name" (pinned, t=0.85)** | **0.871** (0.964 / 0.794) | 0.912 (0.912 / 0.912) | 253 |
| dslim/bert-base-NER | 0.871 (0.964 / 0.794) | 0.875 (0.933 / 0.824) | 40 |
| roberta-large-ner-english | never reaches 0.95 | 0.943 (0.917 / 0.971) | 133 |
| gliner_medium-v2.1 | never | 0.638 | 95 |
| gliner_large-v2.1 | never | 0.587 | 259 |
| gliner_multi_pii-v1 | 0.162 | 0.267 | 91 |
| spaCy en_core_web_sm | never | 0.582 | 11 |

- With the bare label "person", every GLiNER model tags pronouns ("I" 44 times, "you" 22): precision 0.29-0.51. "person name" is what makes GLiNER usable.
- Relation phrases do not work as spans: best relation-phrase F1 0.20. Relationships will come from the person-study line, not the extractor.
- Pinned model ties BERT at the gate and leads on best F1; BERT is 6x faster and is the swap if latency matters. 34 spans is small: one span moves F1 by about 3 points.
- Replay: every GLiNER model produced byte-identical mention rows on a second pass.

## Linkage calibration

- Thresholds: auto-merge at score >= 8.5, auto-reject at <= 2.0. No single feature exceeds 6.0, and the embedding is capped at 1.5, below the 6.5-wide band (checked on every config load). An exact name needs two corroborating features (the extractor reads it as a name, and it comes from a different producer) to merge on its own.
- Embedding similarity is mean-centered over all 2,867 graph labels (raw bge cosine has no calm state, PR #2647), and adds nothing when the two labels fold to the same text (the embedder lowercases and strips accents too, so it would double count).
- Provisional labels (agent, made after seeing the output, so not independent): auto-merge precision 1.0 (18/18), recall 0.90 (2 of 20 true matches sit in the review band). Rule outcomes took every clear non-match, leaving one scored non-match, so labels cannot place the upper line yet. 8.5 is a design choice until Juniper's exported CSV comes back; `evals/run_linkage_eval.py` re-derives it from that CSV.
- Expected ask volume for the cleanup: 3.

## Metric quality gate (centered embedding feature)

1. Provenance: vector-host `/embedding` (bge-large-en-v1.5), centered in `scripts/run_person_linkage.py` (`CenteredEmbedder`).
2. Independence: a transform of string meaning; overlaps the string features, so it is capped and zeroed on exact matches.
3. Theory: All-but-the-Top (Mu & Viswanath 2018), as measured on this host in PR #2647.
4. Live check: 2,867 live labels, 0 embedding errors. 3,000 random label pairs: raw cosine p50 0.535 / p95 0.617 (no calm state); centered p50 -0.008 / p95 0.139, so it rests at about 0. Only 1.1% of random pairs reach the 0.3 "weak" bin and 0.3% the 0.5 "agree" bin. Real pairs: "juniper" vs "juniper feld" 0.54, "skye" vs "Skye Laflamme" 0.63. Exact-string pairs read 1.000, which is why the feature is zeroed on exact matches.
5. Existing mechanism: `orion/world_pulse_read/resonance_similarity.py` (PR #2647) does the same centering for reading; not merged into a shared helper here.
6. Reversible: one weight in `linkage_config.yaml`; nothing persisted.

## Tests run

```text
PYTHONPATH=. python -m pytest -q orion/person_model/tests tests/test_interlocutor_pronouns_in_prompts.py tests/test_person_model_schemas_registered.py
  -> 59 passed, on Python 3.11 (the CI version) in a clean venv with only the CI deps
services/orion-mind: pytest tests/test_stance_system_examples.py tests/test_stance_handoff_soft_labels.py -> 11 passed
services/orion-thought: pytest tests/test_stance_react_prompt_hides_turn_id.py -> 1 passed
services/orion-cortex-exec: pytest tests/test_curiosity_disclosure_replay.py -> 6 passed
tests/test_agent_trace_schema_registry.py tests/test_schema_registry_import_light.py tests/test_recall_diagnostic_schema_registry.py -> passed
scripts/check_scripts_dir_no_stdlib_shadow.py -> clean; scripts/check_definition_drift.py --gate -> PASS
```

## Evals run

```text
orion/person_model/evals/run_extractor_bakeoff.py  -> table above (bakeoff_results_2026-10-14.json)
scripts/run_person_linkage.py --run 2026-10-14     -> 120 decisions: 53 / 3 / 64; decisions.csv sha256 ad2ac313a45089e9...
  rerun in a fresh dir (embeddings re-fetched)     -> byte-identical decisions.csv
orion/person_model/evals/run_linkage_eval.py       -> auto-merge precision 1.0 (provisional labels), recall 0.90, ask volume 3
orion/person_model/evals/check_review_html_ui.py   -> PASS csv_has_every_card, choice_and_note_exported,
                                                     persisted_after_reload, no_network_requests, no_page_errors,
                                                     default_choices_marked_defaulted, formula_cells_neutralized
```

## Docker/build/smoke checks

```text
No service image or compose file changed; no Docker build needed.
Read-only access used: Postgres (read-only session), FalkorDB GRAPH.RO_QUERY, vector-host /embedding.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...feat/person-model-p0-p2`. No blockers; it confirmed the CLI is read-only and the schemas are registered in both registries.

- Finding: exact label + extractor (8.0) auto-merged even from the same producer, contradicting the config comment.
  - Fix: upper line 8.5; exact + extractor + cross-producer (8.5) merges, same-producer goes to review.
  - Evidence: `test_exact_plus_extractor_from_the_same_producer_goes_to_review`.
- Finding: two different names in the same chat message scored +2.0 toward being one person.
  - Fix: a shared chat_prompt ref is now `same_message` (-1.0); only a shared grounding memory counts as agreement.
  - Evidence: `test_sharing_only_a_chat_message_is_counter_evidence`.
- Finding: a candidate claimed by two people was resolved silently by score.
  - Fix: goes to review as `multiple_anchor_claims`; when the two claimants are themselves an open review pair, it rides on that card.
  - Evidence: `test_two_people_claiming_one_candidate_go_to_review`, `test_claims_by_two_people_ride_on_their_pair_question`; live: "skye"/"skye laflamme" ride on the Skye card.
- Finding: the "Jay, Jay" hold matched ids only.
  - Fix: also matches folded labels.
  - Evidence: `test_split_hold_also_catches_graph_nodes_with_the_same_name`.
- Finding: a rerun was not guaranteed identical (a timed-out embedding silently scored as missing; centering mean included cache leftovers).
  - Fix: 3 retries, abort on any missing vector, mean over this run's population only.
  - Evidence: fresh-dir replay, `decisions.csv` byte-identical.
- Finding: the HTML saved and exported pre-approved defaults as her choices, so tuning would fit the thresholds to their own output.
  - Fix: only explicit choices are stored; CSV has a `defaulted` column; the eval ignores defaulted rows; saved choices apply only within the same group.
  - Evidence: UI check `default_choices_marked_defaulted`.
- Finding: CSV cells starting with = + - @ would run as spreadsheet formulas.
  - Fix: neutralized with a leading `'` (score column exempt).
  - Evidence: UI check `formula_cells_neutralized` (fixture label `=HYPERLINK(...)`).
- Finding: merge plan not exact when entries are undone one at a time; `merged_into` could be overwritten.
  - Fix: entries planned in sequence over an overlay; a second need for the same alias is a `claim_alias`; unmerge takes the still-active entries and keeps an alias another merge needs; the planner refuses a node already merged elsewhere.
  - Evidence: `test_undo_a_middle_entry_alone_keeps_what_the_other_needs` (both orders), `test_planner_refuses_a_node_already_merged_elsewhere`.
- Finding: nothing pinned the CLI as read-only.
  - Fix: static test (no `GRAPH.QUERY`, no SQL/Cypher write keywords, `readonly=True`, every store write raises).
  - Evidence: `test_cli_is_read_only_by_construction`.
- Nits fixed: home compares the town ("Salt Lake City, Utah" is flagged), "St.John" is not a code identifier, side-by-side guard uses the matched alias, raw JSON island, `not_home` refuses a missing alias, workflow push paths completed.
- Declined: (a) "a full name seen only outside chat should be scored, not linked as a facet": that would put labels like "claude motor" in front of her; a facet link is reversible and still attached to the right person (comment in `linkage.py`). (b) Checking surnames/aliases in the no-special-case gate: `people.yaml` carries no aliases yet. (c) The `postgres:postgres@localhost` default DSN: the same local default is used in 97 other scripts in this repo.

## Restart required

The prompt files are baked into the images (`COPY orion /app/orion`), so the pronoun fix reaches chat only after a rebuild of the three services that render them (cortex-exec family, thought, mind). After merge, from the primary checkout on main (one line):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && for s in orion-cortex-exec orion-thought orion-mind; do ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh $s up -d --build; done
```

Not run by this task. Until it runs, chat keeps the old wording; nothing else in this PR has a runtime effect (the person model is a dry-run CLI).

## Risks / concerns

- Severity: medium. Concern: the review HTML is committed to a public repository and includes short snippets of Juniper's chat and memories (family names, Amanda's health). Mitigation: snippets are windowed and email addresses masked; the same facts are already in the public spec. If she wants it private, delete the file and keep the `/tmp/person-linkage/` copy.
- Severity: medium. Concern: labels for the bake-off and the linkage eval were made by the agent, the latter after seeing the output. Mitigation: her exported CSV replaces them; thresholds are re-derived from it before any live apply.
- Severity: low. Concern: 18 auto-merges include links from her chat mentions (Amanda, Teddy, Addy...) to topic-modeler entities with the same name; the topic modeler's "amanda" could in principle be a different Amanda. Mitigation: nothing is applied; each merge is reversible from its log entry.
- Severity: low. Concern: the extractor stack (torch, gliner) is not in any service image yet. Mitigation: dry run only; P3+ decides where it runs.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2649

🤖 Generated with [Claude Code](https://claude.com/claude-code)
