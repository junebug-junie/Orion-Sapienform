## Summary

Patch A of reading resonance (design: PR #2642, `docs/superpowers/specs/2026-10-13-reading-resonance-design.md`). Read-only: no outreach, no bus publish, no writes to any live store.

- Measured why similarity scores looked meaningless (every read about 0.6 close to some belief, a Utah board appointment included): bge cosine has no calm state. Ruled out the missing query prefix and the pooling mode with numbers.
- Retrieval now scores by mean-centered, hub-corrected cosine, used only to order a loose candidate list inside fixed per-kind budgets. The score is never shown to the judge and never decides "interesting".
- Crosswalk searches, by meaning: live priors (with the Findings behind them and their revision trail), lived questions (with answers) and lived answers (with the answer they revised), atlas concepts (with 1-hop substrate neighbours and assertions), crystallized memories (re-read from Postgres), curiosity findings, dreams, earlier reads' open threads, open concern arcs.
- Judge contract (`ReadingResonanceJudgmentV1`, with a 3-5 sentence `article_summary` for Juniper) and deterministic checks: cited ref must be in the packet, quote must be 5+ whole words verbatim from the read, a "strains" must name a cited prior, no prompt echo.
- Replay eval wrote a 29-item labeling set for Juniper. **The judge step is UNVERIFIED**: prompts are built and stored, not run (see Risks).

## Outcome moved

- A printed similarity now has a calm state. Unrelated pairs rest at 0.08 instead of 0.66. The Utah read's nearest belief goes from 0.593 to 0.219, below the offer floor, so no belief is offered for it. The CoreWeave GPU reads keep 0.27-0.35.
- One memory used to show up for everything. It was in the top-2 of 8 of 29 reads; after the fix the worst offender is in 3 of 29.
- Juniper has a concrete set to label (`docs/superpowers/evals/2026-10-13-resonance-labeling-set.md` + `.csv`).

## Current architecture

A read ended at the Stage 2 summary. The only path back into Orion was an atlas link, which needs a quote that names the concept word for word (1 link ever). Concept search offered atlas concepts by raw bge cosine at 0.65. Nothing compared reads to beliefs, memories, or questions. vector-host serves `BAAI/bge-large-en-v1.5` with mean pooling and no query instruction (`services/orion-vector-host/app/embedder.py:52-66`; checked live, the served vector matches local mean pooling at cos 1.0000).

## Architecture touched

- New pure module `orion/world_pulse_read/resonance.py`: packet builder, judge prompt, `check_judgment`, `strained_prior`.
- New `orion/world_pulse_read/resonance_similarity.py`: centering, CSLS hub correction, the offer floor.
- New schemas `orion/schemas/reading_resonance.py`, registered in `orion/schemas/registry.py`.
- New live replay eval `services/orion-hub/evals/run_reading_resonance_eval.py`. It reads only: Postgres inside a READ ONLY transaction, Falkor through `GRAPH.RO_QUERY` only, Chroma `/get`, and vector-host `/embedding`, which persists nothing.
- No service runtime, bus, env, or compose change.

## Similarity diagnosis (the numbers)

Labelled-by-construction sanity sets. A "related" pair is a read and its own Stage 2 claims, a prior and its own Finding, or a lived question and its own answer. The "unrelated" pair swaps in a random partner (a different site, a different prior, a different question).

| set | n | raw related / unrelated / AUC | centered related / unrelated / AUC |
|---|---|---|---|
| read summary vs its own claims | 346 | 0.774 / 0.582 / 0.977 | 0.484 / 0.047 / 0.988 |
| prior vs its own Finding | 241 | 0.828 / 0.666 / 0.967 | 0.526 / 0.094 / 0.961 |
| lived question vs its own answer | 60 | 0.801 / 0.616 / 0.948 | 0.522 / 0.095 / 0.949 |

Belief sets pooled. Raw: related p10 0.726; unrelated p50 0.658, p95 0.748. Centered: related p10 0.246; unrelated p50 0.078, p95 0.310.

Offline variant probe. Same model files copied from the vector-host container, run in a `--network none` throwaway container; 1,522 texts; read-vs-own-claims set:

| config | related / unrelated / AUC |
|---|---|
| mean pooling (live) | 0.774 / 0.580 / 0.979 |
| mean + query prefix | 0.768 / 0.575 / 0.978 |
| CLS pooling | 0.721 / 0.507 / 0.955 |
| CLS + prefix | 0.711 / 0.495 / 0.957 |
| mean + centering | 0.497 / 0.077 / 0.988 |

Reading:
- The query prefix and CLS pooling are not the problem. vector-host stays unchanged and nothing needs re-embedding.
- Anisotropy is the problem. Every vector sits in one narrow cone, so unrelated text rests at about 0.6.
- Centering fixes the scale, not the ranking. AUC is unchanged, and the unrelated p95 sits above the related p10. With about 3,000 candidates per read, the unrelated tail outnumbers the real neighbours, so top-k is mostly adjacency. That is why a model must judge and Juniper's labels are the metric.
- What the Utah 0.62 actually was (offline, raw): the nearest prior was about `governor.reject` calls on stance crystallizations. The only thing they share with "Gov. Cox" is one word.

Metric gate (for the rank score):
1. Provenance: `resonance_similarity.py`, computed from vector-host bge vectors.
2. Independence: it is a transform of the same cosine, not new signal, and is labelled as such.
3. Theory: All-but-the-Top (Mu & Viswanath 2018) and CSLS (Conneau et al. 2018).
4. Live check: the tables above; the rest state is 0.08, not a floor artifact.
5. Existing mechanism: `concept_search.py` uses raw cosine at 0.65. Left unchanged and noted.
6. Reversibility: it is used only to order candidates, nothing is persisted, and it is cheap to remove.

## Crosswalk findings worth knowing

- **Reading-minted atlas concepts are islands.** The 260 `sub-concept-wp-read-*` nodes have 2 substrate edges between them, while `sub-concept-seed-orion` alone has 2,027. Atlas neighbourhoods are therefore almost always empty for reading concepts. The neighbourhood step is wired; there is nothing to walk yet.
- **Atlas hits are mostly earlier reads on the same topic.** Example: the Utah read lands on "utah-judicial-appointment-process" from an earlier Utah read. Concepts the read itself minted are excluded (matched on the read URL or root request in `evidence_refs_json`).
- **The one open concern arc has no label and no body.** It is not offered, because it would be an empty excerpt.
- **Most reads (NASA and Utah press releases) get no beliefs or memories above the floor.** Only CoreWeave, the YouTube tech-news read, llama.cpp and a few arXiv reads do. That fits the earlier rough read that 3-6 of 30 reads carry a real connection.

## Files changed

- `orion/schemas/reading_resonance.py`: packet, candidate (+neighbourhood), judgment, and check models; extra=forbid.
- `orion/schemas/registry.py`: registers the four top-level models. The nested `ResonanceNeighborV1` and `ResonanceCitationV1` ride inside them and are deliberately not registered.
- `orion/world_pulse_read/resonance_similarity.py`: centering, hub correction, floor (with the data it came from).
- `orion/world_pulse_read/resonance.py`: crosswalk packet, judge prompt (no scores, no example sentences), deterministic checks.
- `orion/world_pulse_read/tests/test_resonance.py`: 27 gate tests.
- `services/orion-hub/evals/run_reading_resonance_eval.py`: live read-only replay; writes the diagnosis, packets, prompts, and the labeling set.
- `services/orion-hub/evals/test_reading_resonance_eval.py`: offline behavioural eval built on a frozen CoreWeave case.
- `docs/superpowers/evals/2026-10-13-resonance-labeling-set.md`, `.csv`: 29 items for Juniper.
- `docs/superpowers/pr-reports/2026-10-13-reading-resonance-crosswalk-pr.md`: this report.

## Schema / bus / API changes

- Added: `ResonanceCandidateV1`, `ResonancePacketV1`, `ReadingResonanceJudgmentV1`, `ReadingResonanceCheckV1` (registry only).
- Removed: none.
- Renamed: none.
- Behavior changed: none at runtime. Nothing is published or consumed yet.
- Compatibility notes: purely additive. Patch B adds the bus finding kind on the ReadingFindingV1 seam, consumer-first.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template change)
- skipped keys requiring operator action: none

## Tests run

```text
.venv/bin/python -m pytest -q orion/world_pulse_read/tests/ services/orion-hub/evals/test_reading_resonance_eval.py
158 passed
.venv/bin/python -m pytest -q tests/test_world_pulse_read_schemas.py tests/test_schema_registry_import_light.py tests/test_agent_trace_schema_registry.py
passed
```

## Evals run

```text
PYTHONPATH=. .venv/bin/python services/orion-hub/evals/run_reading_resonance_eval.py
reads=29 pool=3041 (live, read-only); diagnosis table above;
attractor raw 8/29 -> centered+hub 3/29; packets=29 (5-20 candidates);
judge: UNVERIFIED (no --judge-url); prompts under /tmp/reading-resonance-eval/prompts (29, 232 KB)
wrote docs/superpowers/evals/2026-10-13-resonance-labeling-set.{md,csv}
```

To run the judge later, when gpu1 is idle: add `--judge-url http://<gateway>/v1/chat/completions --judge-model agent`. The labeling set then carries Orion's connection, the judge's confidence, and the rejected items (with reasons) automatically.

## Docker/build/smoke checks

```text
None needed: no service code, image, env, or compose change. The offline variant probe ran in a
throwaway `docker run --rm --network none` container of the vector-host image (not the live one).
```

## Review findings fixed

Code review ran in a subagent on commit 249a8bbe1.

- Finding: a `decision="none"` judgment could carry `strains` with an uncited, out-of-packet prior id, and `strained_prior()` would route a re-test for it.
  - Fix: the citation-in-packet and prior checks now run whatever the decision is, and `strained_prior` requires a non-none accepted decision.
  - Evidence: `test_none_decision_cannot_smuggle_a_retest_or_foreign_citation`.
- Finding: the quote check accepted any 8-character substring, so "and the" passed.
  - Fix: a quote must be 5+ whole words, matched on word boundaries.
  - Evidence: `test_short_or_mid_word_quote_rejected`.
- Finding: errors from Postgres, Falkor, or empty sets escaped without closing the connection, so "exit 2 on a store failure" was not true.
  - Fix: the whole replay runs in `async with conn.transaction(readonly=True)` inside try/except/finally, and empty-set guards were added.
  - Evidence: replay exit=0; code path reviewed.
- Finding: each read's own vector counted toward its own hub penalty, which demoted genuine one-off neighbours.
  - Fix: the hub score is now leave-one-out per read (`hub_from_sims` over the other reads).
  - Evidence: `test_hub_from_sims_leaves_out_what_the_caller_excludes`; replay re-run.
- Nits fixed: the same ref can no longer appear twice in a packet; JSON is parsed with `raw_decode`, so trailing braces in prose are fine; the echo check also covers `article_summary` and `how`; `redis.from_url` (keeps auth/db); `ensure_ascii=False` for Cypher params; no empty embed text; a dimension-mismatch guard; `budget=None` handled explicitly; the sentence splitter is documented as a heuristic.
- Not fixed (nit): the embed cache is keyed by text only. It is scratch under `/tmp` per run. Delete it after any vector-host model change.

## Restart required

```text
No restart required.
```

## Risks / concerns

- Severity: material.
  - Concern: the judge step is UNVERIFIED. The only agent route (`agent`) runs at `system` priority in `config/gpu_pool.yaml`, ahead of the background reading and curiosity holds. hecate's agent-deep was offline. Running 29 judge calls would have jumped the live queue. So the labeling set shows retrieval's best piece per kind, not Orion's judged connection, and it has no judge-rejected items yet.
  - Mitigation: run with `--judge-url` while gpu1 is idle, or add a background-priority agent route in patch B, then regenerate the set.
- Severity: medium.
  - Concern: the 0.25 offer floor was measured against this pool's mean vector.
  - Mitigation: patch B stores the mean with the index and re-measures when the pool shifts.
- Severity: low.
  - Concern: the hub penalty still counts sibling reads of the same thread (e.g. the 4 CoreWeave reads).
  - Mitigation: left as is, since it is visible in packets. Revisit once labels exist.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
