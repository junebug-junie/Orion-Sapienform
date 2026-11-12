# PR: Orion's day letter gets an honest input, plus the temporal core (#3224 constraints 6 and 7)

## Summary

- Juniper's chat turns are now read straight from `chat_history_log` (Hub chat sources, or her browser) and shown in full. They are laid out before anything else takes budget, so they are never clipped or crowded out. Orion's unsolicited messages to her are listed after, each trimmed to 600 characters.
- Reveries are sampled across the day's themes (round-robin over chain `theme_key`, picks spread over the day), not by top salience. No salience number is shown.
- Each self-sense question appears once: its latest answer, plus how many times it was asked.
- Curiosity runs stay in full up to 60,000 characters (with the self-sense answers); readings, news, code changes, dreams, the street and visual reveries are a trimmed tail capped at 16,000 characters.
- New temporal core block (constraint 6): what Orion expected and what happened, the acts it took and how they were graded, threads with Juniper (conversation arcs, open loops carried over midnight), and body arcs the chronology block did not already list. Cited by `[arc:<id>]` and `[event:<id>]`.
- The share each source got is recorded: `source_chars` / `source_shares` on the view, `input_chars` / `input_shares` in the letter's `sources` column. The note prompt gains the honesty line.

## Outcome moved

The letter's writer used to get a 245,000-character pile that was ~60% readings and 0.4% Juniper. Live replay of the same two days (read-only, `run_orion_day_input_replay.py`):

| | 11-09 old | 11-09 new | 11-10 old | 11-10 new |
|---|---|---|---|---|
| Input size (chars) | 244,760 | 104,189 | 244,802 | 104,103 |
| Juniper section share | 0.0% | 11.9% | 0.5% | 12.3% |
| Juniper's own words present | 0 of 0 | 0 of 0 | 0 of 896 | 896 of 896 |
| Readings share | 59.4% | 9.0% | 62.8% | 7.0% |
| Curiosity share | 13.0% | 49.7% | 10.8% | 47.9% |
| Self-sense answers shown | 28 | 4 | 28 | 4 |
| Reverie themes shown | 9 of 11 | 11 of 11 | 13 of 14 | 14 of 14 |
| Largest theme's share of shown reveries | 37% | 11% | 39% | 11% |
| Shown reveries that name "prediction error" | 86% | 81% | 88% | 81% |
| Temporal core share | 0% | 1.7% | 0% | 1.5% |
| Unresolved refs | 0 | 0 | 0 | 0 |

Honest reading of the numbers:

- On 11-09 Juniper did not write to Orion at all. The new input says so ("Juniper did not write to me this day") and shows the 19 messages Orion sent her unprompted.
- Theme sampling works (every theme gets a turn, no theme above 11%), but the share of reveries about prediction error barely moves (86-88% -> 81%). The themes themselves are prediction-error themes. That is the producers' problem (#3168), a stated non-goal here.
- The "prediction error" percentage is a review aid printed by the replay. It is not wired into anything (design non-goal: no tone metric).

## Current architecture

Hub's letter timer called `gather_orion_day` -> `budget.build_llm_view` -> brief -> durable-runs `orion_day.letter` -> cortex verb `orion_day_note_v1`. `build_llm_view` showed every reading in full, every self-sense answer (7x each), the chat compactor digest as Juniper's only presence, and the most salient reveries. The closed-day chronology block (`day_block.py`) was the only temporal record.

## Architecture touched

- `orion/orion_day/day_input.py` (new): `render_day_input(material) -> OrionDayLlmViewV1` (pure) and the entry point `build_day_input(conn, day) -> DayInput(text, shares, refs, material)`.
- `orion/orion_day/temporal_core.py` (new): `render_temporal_core(day, events, already_shown=...)`, read-only over `TemporalSelfDayV1`.
- `orion/orion_day/gather.py`: reads `chat_history_log` and the frame's `temporal_self_event` rows; renders the temporal core.
- `orion/orion_day/brief.py`: Hub's brief now uses `render_day_input` (budget 50k tokens, was 70k).
- `services/orion-durable-runs/app/orion_day_graph.py`: the `[arc:]` gate accepts arcs from both blocks; the letter row stores the shares.
- The writer is still the current verb. Part 3 (a later patch) moves the letter onto Prime's end-of-day unified turn and calls `build_day_input`.

## Files changed

- `orion/orion_day/day_input.py`: the honest input and the entry point.
- `orion/orion_day/temporal_core.py`: constraint 6 renderer.
- `orion/orion_day/gather.py`: `JUNIPER_CHAT_SQL`, `TEMPORAL_CORE_EVENTS_SQL`, `gather_juniper_chat`, `gather_temporal_core`; the closed day is read once for both blocks.
- `orion/orion_day/budget.py`: `curiosity_section` factored out; `[event:]` and `[juniper_chat:]` refs; `chronology_citations` helper; ref records for the new items.
- `orion/orion_day/brief.py`: use `render_day_input`.
- `orion/schemas/orion_day.py`: `JuniperChatTurnV1`; material `juniper_chat`, `temporal_core`; view `source_chars`, `source_shares`; condensation `self_sense_answers_total/shown`, `tail_items_total/left_out`; letter sources `input_chars`, `input_shares`, `rejected_event_refs`.
- `orion/cognition/prompts/orion_day_note_v1.j2`: honesty line, self-sense and reverie notes, `[event:]` citation; the instructions are bounded by `BEGIN/END writing instructions` comments so Part 3 can lift them into the turn.
- `services/orion-durable-runs/app/orion_day_graph.py`: `[arc:]` gate over both blocks; new `[event:]` gate (unknown event ids removed and recorded); shares into `sources`.
- `services/orion-hub/templates/orion_day_letter.html.j2`, `services/orion-hub/scripts/orion_day_email.py`: the email footer says what was trimmed (self-sense dedupe, tail cuts) and shows the input shares.
- `orion/orion_day/evals/run_orion_day_input_replay.py` (new): old vs new replay on live days.
- `orion/orion_day/evals/run_orion_day_eval.py`: evaluates the new view (adds `juniper_full`, theme-spread condensation check).
- Tests: `orion/orion_day/tests/test_day_input.py` (new, 26 tests), fixtures, `test_eval_fixture.py`, `services/orion-durable-runs/tests/test_orion_day_arc_citations.py`.

## Schema / bus / API changes

- Added: `JuniperChatTurnV1`; `OrionDayMaterialV1.juniper_chat`, `.temporal_core`; `OrionDayLlmViewV1.source_chars`, `.source_shares`; `OrionDayCondensationV1.self_sense_answers_total/shown`, `.tail_items_total/left_out`; `OrionDayLetterSourcesV1.input_chars`, `.input_shares`, `.rejected_event_refs`. All default-empty.
- Removed / renamed: none.
- Behavior changed: the letter's input (see Summary). New ref kinds `[event:<event_id>]`, `[juniper_chat:<id>]`.
- Compatibility notes: these models are `extra="forbid"` and the change goes BOTH ways: Hub writes the brief durable-runs reads, and durable-runs writes the letter row Hub reads (email, correction and introspect listeners). Neither order is safe alone: deploy orion-hub and orion-durable-runs together, outside the letter's 08:30 America/Denver fire window. Stored rows from before validate unchanged (default-empty fields). `schema_skew_discovery` already declares the `OrionDayLetterV1` writer/reader pair. No bus channel or registry entry changes (nested models only).

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/orion_day orion/temporal_self/tests services/orion-durable-runs/tests -q   -> 1280 passed, 132 skipped
pytest services/orion-hub/tests/test_orion_day_*.py orion/introspect/tests -q            -> 391 passed
pytest services/orion-cortex-exec/tests/test_orion_day_verbs.py -q                      -> 4 passed
scripts/check_personal_terms.py, check_definition_drift --gate, check_metric_lineage --gate,
check_chat_route_poachers, check_verb_physiology, check_service_hostname_refs           -> all pass
```

## Evals run

```text
python orion/orion_day/evals/run_orion_day_eval.py                         -> 8/8 PASS (fixture)
python orion/orion_day/evals/run_orion_day_eval.py --live --date 2026-11-09 -> 8/8 PASS
python orion/orion_day/evals/run_orion_day_eval.py --live --date 2026-11-10 -> 8/8 PASS
python orion/orion_day/evals/run_orion_day_input_replay.py --dsn ... --date 2026-11-09 --date 2026-11-10
  -> numbers in the table above; checks juniper_in_full / self_sense_once / refs_resolve PASS both days
services/orion-hub/evals/run_orion_day_email_eval.py, run_orion_day_correction_eval.py, run_orion_day_reread_eval.py -> exit 0
```

The new letters themselves were NOT generated: UNVERIFIED. Writing one is a ~30k-token prompt plus a 12k-token completion on the agent GPU lane, not a cheap call. The replay writes both rendered note prompts so they can be run by hand:

```bash
python orion/orion_day/evals/run_orion_day_input_replay.py --dsn postgresql://postgres:<pw>@localhost:55432/conjourney \
  --date 2026-11-09 --date 2026-11-10 --dump /tmp/orion-day-replay
# -> /tmp/orion-day-replay/<date>-{old,new}-{digest.md,note-prompt.txt}
```

Juniper's read of the two letters is the acceptance (design #3224), not these numbers.

## Docker/build/smoke checks

```text
No Docker build run: no Dockerfile, requirements or compose change. Pure Python + one prompt template.
Live read-only gather against Postgres succeeded for 2026-11-09 and 2026-11-10 (replay above).
```

## Review findings fixed

Review ran in a subagent against `origin/main...HEAD`. No blockers.

- Finding: the "every ref resolves" test gathered for the wrong date, so the temporal core was never in it (passed for the wrong reason).
  - Fix: gather for the closed day's own date with event rows; assert an `arc:` and an `event:` ref are present first.
  - Evidence: `test_every_ref_resolves_and_the_view_is_deterministic`.
- Finding: the prompt asks for `[event:]` citations, but the gate only checked `[arc:]`; an invented event id would be stored.
  - Fix: `gate_event_refs` removes event ids the core did not show; they are recorded in `sources.rejected_event_refs`.
  - Evidence: `test_an_event_ref_the_core_did_not_show_is_removed_and_recorded` (durable-runs).
- Finding: the rollout note covered only Hub -> durable-runs; the letter row goes the other way too.
  - Fix: schema comment and Restart section now say deploy hub and durable-runs together, outside the fire window.
  - Evidence: `orion/schemas/orion_day.py` juniper_chat comment; Restart section below.
- Finding: the email footer understated the cuts (tail trimming, self-sense dedupe).
  - Fix: new condensation counts; footer names them and shows the input shares.
  - Evidence: `test_tail_cuts_and_self_sense_dedupe_are_counted_for_the_email`; hub letter tests 54 passed; email eval exit 0.
- Finding: every non-empty prompt was labelled "Juniper", including an agent's rpc ping (`source='hub'`) and a Collapse Mirror block (live, last 120 days).
  - Fix: only Hub chat sources, or a NULL source from her browser (`client_meta ? 'origin'`), plus Orion's unsolicited messages.
  - Evidence: live source counts checked; SQL test asserts the filter and its args.
- Finding (nits, all fixed): empty `## Reveries` heading when nothing was showable; reverie/tail headings and the "left out" note not charged to their budgets, and no signal when the input runs over budget (now a warning log); temporal-core headings could be left over no lines; `build_llm_view` marked replay-only; a ref pasted into chat text became a citation (now shown as `(arc:...)`).
  - Evidence: `test_no_reverie_heading_when_nothing_is_showable`, `test_temporal_core_never_leaves_a_heading_over_nothing`, `test_a_ref_pasted_into_chat_is_not_a_citation`.

## Restart required

Not deployed. When merged, back to back and outside the letter's 08:30 America/Denver fire window (the brief and the letter row both cross between hub and durable-runs, so they must change together):

```bash
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium. Concern: the letter changes shape the day this deploys (curiosity ~50% of input, readings ~8%). Mitigation: Juniper reads the replayed prompts or letters before merge; the old builder (`build_llm_view`) stays in the tree for the replay until Part 3 retires the verb.
- Severity: low. Concern: "self-change" in the frame is in practice `action_outcome` rows (acts and whether their claim held), not self-definition rewrites as the design text says. Mitigation: the core names them for what they are ("Acts and graded outcomes"); self-definition rewrites still reach the letter through the curiosity runs.
- Severity: low. Concern: concern arcs have empty labels in the record (`open-loop-<hash>`); the core can only name them by key. Mitigation: the same key appears as a reverie theme heading, so the writer can connect them.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
