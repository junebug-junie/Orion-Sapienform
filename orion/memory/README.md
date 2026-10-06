# orion/memory

Memory intake, episode distillation (Stage 1), recall-purpose selection and
the voice renderer. Design: `docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md`
and `docs/superpowers/specs/2026-10-06-memory-stage2-referent-graph-design.md`.

## Concepts

Only concepts with a real producer, a real consumer and a test are listed.
Keep this table in sync with the code in the same PR.

| Concept | Plain-English meaning | Producer | Consumer | Test |
|---|---|---|---|---|
| `render_memory` / `VoicedMemory` (`voice_render.py`) | Turns a remembered item into the one line Orion reads, saying whose thought it is. Reuses the Stage 1 validator's own voice rule (`_supported_voice`), so it is never looser than the writer | `episode/report.py` (every episode memory, live daily); `chat_stance._project_reverie_glimpse` (legacy `chat_general` path only, see below) | The daily episode report file (live); `chat_stance_brief.j2`'s `reverie_glimpse` (legacy; no live turns on 2026-10-06) | `orion/memory/tests/test_voice_render.py`, `test_episode_report_pg.py`, `test_chat_stance_reverie_glimpse_projection.py` |
| "Juniper told me" | Juniper's own words: a chat memory whose voice survives as `juniper_said` with a verified quote from her prompt | live distiller rows (27 on 2026-10-06) | report | matrix test, `test_juniper_said_needs_a_verified_prompt_quote` |
| "Juniper and I worked out" | Reached together: `worked_out_together` on chat with verified quotes from both her prompt and Orion's reply (same rule as `validate.py`) | validator keeps this voice only with both quotes | report | `test_worked_out_together_needs_both_quotes_and_chat` |
| "I told Juniper" | Something Orion said in chat, with a verified quote | live `orion_thought`/chat rows (4) | report | matrix test |
| "…on my own…, not something Juniper and I discussed" | A private thought: anything on an internal channel (reverie, curiosity, dream, journal, topic_model), whatever its voice claims | internal-channel memories; the legacy reverie glimpse | report; legacy stance prompt | matrix test, `test_the_180_hecate_reveries_case` |
| "My own note…, not Juniper's words" | Fallback for anything else, e.g. `juniper_said` without a verified quote. Always away from Juniper | report rows failing the checks above | report | matrix test, PG report test |
| "Unconfirmed, check with Juniper if natural:" | A high-stakes memory Juniper has not confirmed, including one she was asked about and did not answer within 7 days | validator sets `pending_confirmation` for high stakes; `episode/confirmation.py` sets `unconfirmed` when the ask expires | report | `test_pending_marker_and_whitespace` |
| "…that Juniper rejected, not true" / "…that Juniper corrected, superseded" | A memory Juniper turned down or corrected; never shown as current truth | `confirmation_state` values in `episode_memory` | report | `test_rejected_and_corrected_are_never_plain_truth`, matrix test |

Not here on purpose: "I read" / "From my own code and docs" voices (the
validator turns `orion_read`/`orion_self_knowledge` into `orion_thought`
for chat episodes, so nothing produces them yet); reading titles, claim status
and the "faded" marker; retrieval-intent changes (moved to Stage 2 PR F, see
`docs/superpowers/pr-reports/2026-10-06-memory-voice-render-pr.md`).
