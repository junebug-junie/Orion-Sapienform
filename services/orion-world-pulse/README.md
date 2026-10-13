# orion-world-pulse

Daily world digest: fetches approved sources (`config/world_pulse/sources.yaml`), clusters articles into digest cards, and publishes the digest that Hub turns into reading seeds.

Bring-up, source strategies, curation policy, and rollback: [`docs/world_pulse_dev.md`](../../docs/world_pulse_dev.md).

## Env keys of note

| Key | Purpose |
| --- | --- |
| `WORLD_PULSE_SOURCES_CONFIG_PATH` | Source registry path. |
| `ORION_ACTION_OUTCOME_DB_URL` | Postgres DSN for the curiosity gate's bus_synaptic surprise signal. |
| `WORLD_PULSE_READ_HISTORY_DB_URL` | Read-only DSN for `world_pulse_read_seed`. The arXiv source skips papers already queued or read. Blank disables that dedupe (logged). |

Full list with defaults: `.env_example` and `app/settings.py`.

## arXiv source

`arxiv_research` pulls up to 4 new papers a day from cs.AI, cs.LG, cs.CL, q-bio.NC via the official export API (>= 3 s between requests, honest User-Agent), deduped against what Orion already read. See the arXiv section of `docs/world_pulse_dev.md`.

## Tests and evals

```bash
PYTHONPATH=.:services/orion-world-pulse pytest services/orion-world-pulse/tests -q
PYTHONPATH=.:services/orion-world-pulse python services/orion-world-pulse/evals/run_arxiv_source_eval.py --dsn <dsn>
```
