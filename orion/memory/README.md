# orion/memory

Memory intake, episode distillation (Stage 1), recall-purpose selection and
the voice renderer. Design: `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`
and `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`.

## Concepts

Only concepts with a real producer, a real consumer and a test are listed.
Keep this table in sync with the code in the same PR.

| Concept | Plain-English meaning | Producer | Consumer | Test |
|---|---|---|---|---|
| `render_memory` / `VoicedMemory` (`voice_render.py`) | Turns a remembered item into the one line Orion reads, saying whose thought it is | `chat_stance._project_reverie_glimpse` (latest reverie); `episode/report.py` (each episode memory) | The chat stance prompt (`reverie_glimpse` in `chat_stance_brief.j2`); the daily episode report file | `orion/memory/tests/test_voice_render.py`, `test_chat_stance_reverie_glimpse_projection.py`, `test_episode_report_pg.py` |
| Speaker `juniper` ("Juniper told me") | Juniper's own words. Only a `juniper_said` chat memory with a verified quote from one of her prompts | live distiller rows (27 on 2026-10-06) | report | matrix test |
| Speaker `together` ("Juniper and I worked out") | Something the two of them reached together. Only `worked_out_together` on chat or confirmation | the Stage 1 validator keeps this voice when both a prompt and a response quote verify | report | matrix test |
| Speaker `orion_to_juniper` ("I told Juniper") | Something Orion said in chat | live distiller rows (4) | report | matrix test |
| Speaker `orion_private` ("Something I was turning over on my own…, not something Juniper and I discussed") | A private thought: anything on an internal channel (reverie, curiosity, dream, journal, topic_model), whatever its voice claims. An informed prior, never a shared memory | reverie glimpse (every live reverie); distiller rows on internal channels | stance prompt; report | matrix test, `test_the_180_hecate_reveries_case`, `test_glimpse_is_never_presented_as_juniper_or_as_shared` |
| Speaker `orion_note` ("My own note…, not Juniper's words") | Fallback for any other mismatch, e.g. `juniper_said` without a verified quote. Always away from Juniper's voice | report rows that fail the checks above | report | `test_juniper_said_needs_a_verified_prompt_quote`, PG report test |
| Speakers `orion_read` ("I read") and `orion_self_knowledge` ("From my own code and docs") | Something Orion read, or knows from its own code | distiller schema allows these voices (validator keeps them only with response evidence) | report | `test_contract_table_rows` |
| "Unconfirmed, check with Juniper if natural:" prefix | A high-stakes memory Juniper has not confirmed yet | validator sets `pending_confirmation` for high stakes | report | `test_pending_marker` |
| Retrieval intent rules (`retrieval_intent.py`, `rule_id`) | Which purposeful recall a chat turn gets, from the stance model, the turn-change model, the same-turn LLM's typed reading of the message, or explicit ids. No word lists | `derive_retrieval_intent` | `run_pcr_phase3` in orion-cortex-exec: picks `chat.belief.<intent>.v1` or skips phase 3; `rule_id` goes to the recall request's `task_hints`, the `pcr_phase3_*` log line and `ctx.debug.pcr` | `tests/test_retrieval_intent.py`, `test_phase3_profile_follows_the_turn_not_the_frame` |
| `turn_names_person` / `turn_names_plan` / `turn_names_topic` | The referent rule: the message names a person → relational; a plan → procedural; anything else → semantic | `ctx["current_turn_llm_signals"]` (same-turn LLM, human turns only) | same as above | `test_turn_signal_referent_rule` |

Not here on purpose (no producer or consumer yet): reading titles and claim
status, graphify build dates and the "faded" marker in the renderer; the
intent → memory-kind table and the open-loop rules that need follow-up
referents (Stage 2 PR F).
