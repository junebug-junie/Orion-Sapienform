## Summary

- One table now says which kinds of cited piece each belief relation needs. The judge prompt prints that table, and the checker enforces the same table. "supports" needs a cited belief (`self_prior`) or a lived answer (`lived_answer`). "strains" needs a cited belief, because a strain sends that belief to be re-tested. A mismatch is refused with the reason `relation_ref_kind_mismatch`, which replaces `prior_relation_without_prior`.
- The quote check now ignores differences in typography but not in words: HTML entities, NFKC forms (non-breaking space, the single-character ellipsis), curly quotes, unicode dashes, zero-width characters. It applies this to both the quote and the read. It also accepts real spans joined by "..." when they appear in the read's own order. Each span must be at least 5 whole words, there can be at most 3 spans, and each must start within 600 characters of the one before.
- The labeling set no longer calls an unparseable answer "judged, judge not run". It now shows as `REJECTED (unparseable)`.
- Privacy: the labeling set quotes Juniper's memories and chats verbatim, and this repo is public. The three `docs/superpowers/evals/2026-10-13-resonance-labeling-set.*` files are untracked. The runner now writes them to `/tmp/reading-resonance-eval/labeling/` and refuses any path inside the checkout. The file shapes are in `.gitignore`, and a static-gate test fails if one is tracked again.
- Re-ran the judge on the same 29 reads, one at a time, gated on the GPU pool. Regenerated the labeling set outside the repo.

## Outcome moved

The judge run on the same 29 reads, same model, same `--until 2026-10-14T00:10:00+00:00`:

| | before | after |
|---|---|---|
| accepted | 16 (8 share / 7 note / 1 none) | 23 (11 share / 11 note / 1 none) |
| rejected | 13 | 6 |
| belief relation without a matching piece | 12 | 2 |
| quote not in read | 1 | 2 |
| unparseable / broke schema | 0 | 2 |
| summary sentence count | 2 | 1 |

What the 6 remaining rejections are:
- Two still set "supports" while citing only atlas concepts. The prompt fix took this from 12 to 2. The checker refuses the remaining two by name.
- One quote was rewritten by the judge, not copied. The read says "notably, she has previously served on this same board". The judge wrote "and had previously served on this same board as pro tempore member starting in 2025 — making this a reappointment". The rejection is correct.
- One quote splices three real fragments that sit about 1,900 characters apart in the read. The splice limit refuses it. That item also fails the relation check.
- Two answers never became a judgment. In one, the connection was over its 900-character limit even though the prompt states that limit. The other was cut off at the 900-token budget.
- One summary had the wrong number of sentences.

The one quote rejection from the first run (CRS-35) was a normalization gap, not an invention. The judge joined two exact sentences of the read with "...", 376 characters apart. It is accepted now.

## Current architecture

`check_judgment` in `orion/world_pulse_read/resonance.py` refused `bears_on_prior` whenever no `self_prior` was cited. The prompt said only "'supports' if it backs one", which the judge read as backing a concept. The quote matcher folded only whitespace and case.

## Architecture touched

Offline eval and judge contract only. Nothing is published on the bus, and there is no runtime consumer.

## Files changed

- `orion/schemas/reading_resonance.py`: `RELATION_REF_KINDS` table. The reason `prior_relation_without_prior` is renamed to `relation_ref_kind_mismatch`.
- `orion/world_pulse_read/resonance.py`: the prompt renders the table, the checker enforces it, plus quote normalization and the elision matcher.
- `orion/world_pulse_read/tests/test_resonance.py`: regressions built from the real rejected cases (3f821acc supports-over-concepts, b7d743d2 strains-on-a-concept, b2e0b70c supports-via-lived-answer, db13f467 elided quote). Also tests for splice limits, edge elisions, the dash/quote/entity folding, a paraphrase that must not count as a prompt echo, and a check that the table and the prompt match.
- `services/orion-hub/evals/run_reading_resonance_eval.py`: labeling outputs default to `/tmp/reading-resonance-eval/labeling/`, a path inside the repo is refused, and the unparseable row renders as rejected.
- `services/orion-hub/evals/test_reading_resonance_eval.py`: tests for the unparseable row and the in-repo refusal.
- `tests/test_no_private_eval_artifacts.py`, `.github/workflows/orion-static-gates.yml`, `.gitignore`: the privacy gate.
- `docs/superpowers/evals/2026-10-13-resonance-labeling-set.{md,csv,html}`: removed from git.

## Schema / bus / API changes

- Added: `RELATION_REF_KINDS` (a constant, not a model).
- Removed: none.
- Renamed: the reject reason `prior_relation_without_prior` is now `relation_ref_kind_mismatch`.
- Behavior changed: "supports" accepts a cited `lived_answer`. The quote match is normalized and accepts elisions.
- Compatibility notes: `ReadingResonanceCheckV1` is only built in memory by the eval, so no stored rows carry the old reason.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/world_pulse_read/tests/test_resonance.py services/orion-hub/evals -q  -> 95 passed, 3 skipped
pytest tests/test_no_private_eval_artifacts.py -q                                    -> 3 passed
```

## Evals run

```text
run_reading_resonance_eval.py --out-dir /tmp/reading-resonance-eval-v2 \
  --judge-url http://127.0.0.1:8210/v1/chat/completions --judge-model agent --until 2026-10-14T00:10:00+00:00
  judged=29 accepted=23 share=11 note=11 rejected=6   (sequential, pool-gated; finished in one pass, no pause)
wrote /tmp/reading-resonance-eval/labeling/2026-10-13-resonance-labeling-set.{md,csv,html}
```

The first run's packets, prompts and judgments under `/tmp/reading-resonance-eval/` were left untouched. Browser labels carry over: the HTML keys labels by `seed_id` under a key derived from the same 29 seeds.

## Docker/build/smoke checks

```text
None: offline eval code only.
```

## Review findings fixed

- Finding (SHOULD): the elision matcher would splice any number of far-apart 5-word spans into a claim the read never makes.
  - Fix: at most 3 spans, each starting within 600 normalized characters of the previous one. The real CRS-35 case skipped 376.
  - Evidence: `test_edge_elisions_are_dropped_and_far_spans_are_not_spliced`.
- Finding (SHOULD): a quote with a leading or trailing "..." was always refused.
  - Fix: an edge elision is stripped before matching.
  - Evidence: the same test.
- Finding (NIT): the elision pattern matched a lone "[..." or "...]".
  - Fix: the bracket form is matched as a pair.
- Finding (NIT): the new relation lines are now treated as prompt-echo detectors.
  - Evidence: `test_a_paraphrase_of_a_relation_line_is_not_a_prompt_echo`.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: high (privacy). Concern: the labeling-set files are still in this branch's earlier commits on the remote. Mitigation: removed at the tip. A local history purge plus a force-push command are handed to Juniper. Force pushes are hers to run.
- Severity: low. Concern: 2 of 29 answers broke the schema (the connection was too long, or the output was cut off at 900 tokens). Mitigation: they are labelled rejected. A follow-up could raise `--judge-max-tokens` or name schema violations separately from bad JSON.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2647

🤖 Generated with [Claude Code](https://claude.com/claude-code)
