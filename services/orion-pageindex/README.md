# orion-pageindex

Standalone journals PageIndex service.

## API
- `POST /corpora/journals/rebuild`
- `GET /corpora/journals/status`
- `POST /corpora/journals/query`
- `POST /corpora/chat_episodes/rebuild`
- `GET /corpora/chat_episodes/status`
- `POST /corpora/chat_episodes/query`
- `GET /healthz`

## Journal Metadata Coverage
The corpus export includes both canonical journal fields and denormalized journal index metadata from `journal_entry_index`.

Exported metadata fields include:
- `trigger_kind`
- `trigger_summary`
- `conversation_frame`
- `task_mode`
- `identity_salience`
- `answer_strategy`
- `stance_summary`
- `active_identity_facets`
- `active_growth_axes`
- `active_relationship_facets`
- `social_posture`
- `reflective_themes`
- `active_tensions`
- `dream_motifs`
- `response_hazards`
