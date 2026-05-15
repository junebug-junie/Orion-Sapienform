# RDF / GraphDB V1 Cutover Gap Patch — Autonomy/Stance Graph Read Escape Hatch

Execution plan (bounded to original cutover; does **not** address duplicate correlation IDs / multi-container request handling).

## Goal

Make GraphDB optional for `chat_quick` / stance: explicit `AUTONOMY_GRAPH_BACKEND=graphdb` to enable autonomy SPARQL reads; default (unset) disables them. Quick lane uses bounded subjects/subqueries and a ≤3s timeout. Preserve full graph stance for non-quick / explicit full flows.

## Tasks

1. **Preflight** — Confirm `orion/autonomy/repository.py` (`GraphAutonomyRepository`, fanout/short-circuit), `orion/substrate/relational/adapters/autonomy_ctx.py`, `services/orion-cortex-exec/app/chat_stance.py` (`_load_autonomy_state`), `orion/autonomy/fanout_policy.py`.

2. **`orion/autonomy/graph_gate.py`** — Resolve read plan: backend explicit `graphdb` vs default disabled; endpoint resolution (no auto-enable from `GRAPHDB_URL` alone); quick-lane detection (aligned with `FAST_SINGLE_PASS_CHAT_VERBS` + `chat_quick_full_stance`); `AUTONOMY_QUICK_GRAPH_*` envs; structured `autonomy_graph_backend_decision` logging.

3. **`orion/autonomy/repository.py`** — Optional `active_subqueries` on `GraphAutonomyRepository` / `build_autonomy_repository` to restrict SPARQL fan-out.

4. **`chat_stance.py` + `autonomy_ctx.py`** — Branch on read plan: disabled → no HTTP, degraded summary/debug fields; graphdb → pass timeout/subjects/subqueries from plan; decision + degraded log lines.

5. **Tests** — `orion/autonomy/tests/test_graph_gate.py`; extend `test_chat_stance_autonomy_plumbing.py` with `AUTONOMY_GRAPH_BACKEND=graphdb` + dummy `GRAPHDB_URL` where repository is mocked; repository subquery filter test if lightweight.

6. **Docs / env** — `docs/architecture/rdf_store_v1_cutover.md` new section; `services/orion-cortex-exec/.env_example` (+ `docker-compose.yml` if env passthrough needed).

## Acceptance

Matches user prompt acceptance criteria (no `autonomy_graph_subject_fanout_start` under V1-safe defaults; explicit graphdb + quick envs bounds fan-out and timeout).
