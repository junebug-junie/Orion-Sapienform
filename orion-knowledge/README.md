# Orion Knowledge Forge

Research-to-execution compiler for Orion design and implementation context.

## Authority layers (strongest → weakest)

1. **Raw sources** (`raw/`) — immutable inputs; never edited after ingest
2. **Claims** (`claims/`) — atomic, source-backed statements
3. **Decisions** (`decisions/`) — ADRs with rationale
4. **Specs** (`specs/`) — reviewed design/plan intent
5. **Context packs** (`context_packs/`) — task bundles for agents
6. **Wiki** (`wiki/`) — human-readable compiled views (disposable)

Code and tests are ground truth for runtime behavior.

## v0 workflow

1. Drop source: `raw/sources/YYYY-MM-DD-topic.md`
2. Extract claims (agent-assisted using `AGENTS.md` ingest rules)
3. Propose spec/decision patches → `reviews/pending/` (never direct overwrite of `execution_ready/`)
4. Human approves: `python -m orion.knowledge_forge review apply <patch_id>`
5. Compile pack: `python -m orion.knowledge_forge compile context-pack --spec spec:... --out context_packs/cursor/...`
6. Hand pack to Cursor/Codex — not the whole wiki

## Environment

Set `ORION_KNOWLEDGE_ROOT` when the corpus is not discoverable from the current working directory (containers, monorepo subdirs). Default: walk up from `cwd` to find `orion-knowledge/`. See repo root `.env_example`.

## CLI

```bash
cd /path/to/Orion-Sapienform
export ORION_KNOWLEDGE_ROOT="$(pwd)/orion-knowledge"  # optional
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge lint
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge review list
PYTHONPATH=. ./venv/bin/python -m orion.knowledge_forge compile context-pack --spec spec:substrate-tier-telemetry-v1
```
