# PR: Remove vector retrieval from orion-recall

**Branch:** `feat/orion-recall-vector-amputation`  
**Base:** `main`

## Summary

Vector/Chroma semantic retrieval is fully **amputated** from `services/orion-recall`. Recall no longer imports `vector_adapter`, calls `fetch_vector_*`, or constructs candidates with `source: vector`. Vector services elsewhere in the mesh (`orion-vector-host`, `orion-vector-writer`, `orion-vector-db`) are untouched.

Diagnostics remain: `vector_policy` paths report `allowed: false`, `reason: removed_from_orion_recall`, and `backend_counts` keep `vector` / `vector_anchor` / `vector_exact_anchor` at **0**.

## Files changed

| Area | Change |
|------|--------|
| `app/worker.py` | Removed vector imports and all fetch branches; `backend_counts["vector"]` / `vector_anchor` pinned to 0; `source_gating["vector"]` = `removed_from_orion_recall` |
| `app/recall_v2.py` | Removed vector exact/semantic branches; zero vector backend counts |
| `app/collectors.py` | SQL + RDF only |
| `app/source_policy.py` | `recall_vector_allowed()` always false with `removed_from_orion_recall`; `build_vector_policy()` for debug |
| `app/settings.py` | `RECALL_ENABLE_VECTOR` default `false`; `RECALL_VECTOR_EMBEDDING_URL` default `None` |
| `app/storage/vector_adapter.py` | **Deleted** |
| `orion/recall/profiles/*.yaml` | `enable_vector: false`, `vector_top_k: 0`, `relevance.backend_weights.vector: 0.0` |
| `requirements.txt` | Removed `chromadb`, `numpy` |
| `.env_example` | Vector section marked legacy; `RECALL_ENABLE_VECTOR=false` |
| `tests/test_recall_vector_amputation.py` | **New** — import guard (subprocess), backend counts, policy reason |
| `tests/test_source_policy_vector.py` | Updated for permanent removal |
| `tests/test_vector_self_hit_suppression.py` | **Deleted** (adapter removed) |
| `README.md` | Header notes vector removal from recall |

## Acceptance greps

```bash
grep -R "fetch_vector" services/orion-recall/app          # (none)
grep -R "storage.vector_adapter" services/orion-recall/app  # (none)
```

Legacy `source == "vector"` handling in `fusion.py` / `render.py` / mutation-pressure paths remains for **already-ranked** items only — not candidate construction.

## Test plan

```bash
cd services/orion-recall
# Docker (matches CI image deps):
docker build -f Dockerfile ../..
docker run --rm -v "$PWD/../..:/repo" -w /repo/services/orion-recall <image> \
  sh -c 'pip install -q pytest && PYTHONPATH=/repo:/repo/services/orion-recall python3 -m pytest \
    tests/test_recall_vector_amputation.py \
    tests/test_source_policy_vector.py \
    tests/test_profile_runtime_knobs.py \
    tests/test_recall_v2_shadow.py -q'
```

- [ ] All four test modules pass (14 tests)
- [ ] `RECALL_ENABLE_VECTOR=true` in env does **not** re-enable vector fetch
- [ ] `reflect.v1` / `graphtri.v1` RPC recall returns no vector-sourced items
- [ ] `recall_debug.vector_policy.*.reason` == `removed_from_orion_recall`
- [ ] Hub/cortex-exec recall path still returns SQL/RDF bundles

## Local env

`.env_example` updated; sync local `.env`:

- `RECALL_ENABLE_VECTOR=false`
- `RECALL_VECTOR_EMBEDDING_URL=` (empty)

## Out of scope

- `services/orion-vector-host`, `orion-vector-writer`, `orion-vector-db`
- `orion/bus/channels.yaml` (no recall↔vector channel changes required)
- README Chroma debug snippets (legacy ops docs; follow-up trim optional)
