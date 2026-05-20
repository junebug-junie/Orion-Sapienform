# Orion Knowledge Forge — agent contract

You are maintaining a **research-to-execution compiler**, not a wiki.

## Authority

- `raw/` is immutable after ingest. Never edit existing raw files.
- Claims in `claims/` are the atomic truth layer.
- Specs in `specs/execution_ready/` are execution-facing. You may not edit them directly.
- Propose changes only via `reviews/pending/*.patch.md`.

## Ingest prompt (use when adding a raw source)

When given `raw/sources/<file>`:

1. Read the full source.
2. Extract atomic claims as YAML files under `claims/disputed/` first (set `status: speculative` or `status: disputed` in the YAML).
3. Each claim MUST include `source_refs` pointing at a `source:*` id.
4. Use typed relations only: `supports`, `contradicts`, `supersedes`, `depends_on`, `implements`, `tested_by`, `blocked_by`, `motivated_by`.
5. Update `wiki/concepts/<topic>.md` as a compiled view citing claim ids inline.
6. If a spec or ADR must change, write a patch in `reviews/pending/` — do not mutate `specs/execution_ready/`.
7. Append one line to `wiki/log.md`.

## Compile prompt (use when preparing Cursor work)

Given a spec id and task description:

1. Run: `python -m orion.knowledge_forge compile context-pack --spec <id> --task "<task>" --out context_packs/cursor/<slug>.md`
2. Hand the generated markdown to Cursor — not the whole wiki.
3. Include only accepted claims. Flag disputed claims in `reviews/pending/` instead.

## Forbidden

- Untyped `related` links
- Silent overwrite of `specs/execution_ready/`
- Feeding `wiki/` wholesale into implementation agents
- Treating wiki prose as authority over claims
