# orion/person_model

One module for every person Orion comes to know, parameterized by a person node
id. Nobody is special in code: who the interlocutor is, and everyone's stated
pronouns, live in `config/person_model/people.yaml`. A CI gate
(`tests/test_no_person_special_cases.py`) fails if a name or node id from that
file appears in module code.

Spec: `docs/superpowers/specs/2026-10-14-person-model-and-identity-card-design.md` (rev 3).

## One record per person (2026-10-22, live in orion-sql-writer)

Spec: `docs/superpowers/specs/2026-10-22-people-one-record-design.md`.

| File | What it does |
|---|---|
| `record.py` | Pure planner: named camera clusters + "different people" verdicts + `people.yaml` -> `person` / `person_link` writes. Links only from typed names; a "different" verdict blocks a merge |
| `facts.py` + `categories.yaml` | Which question to ask about a person next, the wording ("you" for the interlocutor, never a pronoun for anyone else), answers kept verbatim as `person_fact` rows |
| `evals/run_person_record_backfill_eval.py` | Read-only dry run on live data: who would link to whom, conflicts, ignored names |

The writer is `services/orion-sql-writer/app/person_record.py` (runs inside the camera individuals cycle).

## What exists (P1 + P2, dry run only)

| File | What it does |
|---|---|
| `person.py` | `PersonRefV1` loader, stated-only pronouns, `possessive()` (falls back to the name, never guesses) |
| `extract.py` | Pinned GLiNER extractor (`knowledgator/gliner-multitask-large-v0.5` @ `7a95e16803`, label "person name", threshold 0.85); replay-stable `PersonMentionV1` rows |
| `normalize.py` | Folding, tokens, possessive heads, prompt-context keys harvested from the templates |
| `linkage.py` + `linkage_config.yaml` | Two-threshold Fellegi-Sunter reducer: rule outcomes (topic / junk / facet), then a score; auto-merge above upper, auto-reject below lower, the band between goes to the interlocutor |
| `merge_log.py` | Merge-log entries that list every write with its before-value; `unmerge_ops` inverts an entry exactly |
| `review_html.py` | One self-contained HTML review file (no network), CSV export |
| `evals/run_extractor_bakeoff.py` | GLiNER vs BERT/RoBERTa/spaCy on labeled chat prompts (labels store ids + offsets, not text) |
| `evals/run_linkage_eval.py` | Auto-merge precision gate, recall, review-band size, label-supported thresholds |

The dry-run CLI is `scripts/run_person_linkage.py` (read only: Postgres read-only
session, FalkorDB `GRAPH.RO_QUERY`, vector-host `/embedding`). There is no apply
mode yet; applying reviewed decisions is a later step.

## Personal data stays local

The repo is public, so real stated answers and eval labels (real names, chat
snippets) live in gitignored `*.local.*` files beside committed, redacted
`*.example.*` fixtures. `person.local_first()` reads the local file when it
exists (same idea as `.env` / `.env_example`):

| Committed (redacted / fake) | Real, gitignored, read first |
|---|---|
| `config/person_model/stated_answers.example.yaml` | `config/person_model/stated_answers.local.yaml` |
| `evals/cleanup_labels.example.jsonl` | `evals/cleanup_labels.local.jsonl` |
| `evals/mention_labels.example.jsonl` | `evals/mention_labels.local.jsonl` |

The real copies sit in the primary checkout. A new worktree has none, so the
dry run and evals fall back to the fake fixtures; copy the local files over (or
pass `--answers` / `--labels`) to run against real data. CI only uses the
fixtures. `tests/test_no_personal_terms.py` fails on any tracked personal term.

## Run

```bash
pip install -r orion/person_model/requirements-extract.txt   # plus the CPU torch wheel
PYTHONPATH=. python scripts/run_person_linkage.py --run 2026-10-14 \
    --html /tmp/person-linkage/person-cleanup-review.html  (never inside the repo: it holds chat snippets and the repo is public)
PYTHONPATH=. python orion/person_model/evals/run_linkage_eval.py \
    --decisions /tmp/person-linkage/2026-10-14/decisions.jsonl
PYTHONPATH=. python -m pytest orion/person_model/tests -q
```

## Not here yet

Attribute store and categories (P3/P4, with their first writer), the person-study
curiosity line (P4), cards (P5), readers (P6). Relation phrases ("my wife") are
not extracted: every GLiNER variant scored relation-phrase F1 <= 0.2 on the
labeled set, so relationships come from the person-study line instead.
