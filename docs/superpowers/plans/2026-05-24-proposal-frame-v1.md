# Proposal Frame v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 7 — convert `SelfStateV1` (+ optional attention/field) into deterministic `ProposalFrameV1` snapshots (possible actions only), persisted for inspection with no execution or bus publish.

**Architecture:** Shared schemas in `orion/schemas/proposal_frame.py`; pure synthesis in `orion/proposals/`; polling service `orion-proposal-runtime` idempotent per `source_self_state_id`; optional Hub `GET /api/substrate/proposals/latest`. Policy from `config/proposals/proposal_policy.v1.yaml`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, FastAPI/uvicorn, pytest, Postgres.

**Depends on:** Layer 6 on `main` (`SelfStateV1`, `substrate_self_state`, port 8118).

**Non-goals:** Layer 8 policy, cortex-exec, bus, LLM, operator notify, settings mutation.

---

## Worktree

```bash
cd /mnt/scripts/Orion-Sapienform
git worktree add .worktrees/feat-proposal-frame-v1 -b feat/proposal-frame-v1 main
```

**Port:** 8119. **Bus:** registry only; no `channels.yaml` changes. **Field SQL:** `WHERE tick_id = :tick_id` (not JSON path).

---

## File structure

| Path | Role |
|------|------|
| `orion/schemas/proposal_frame.py` | `ProposalCandidateV1`, `ProposalFrameV1` |
| `orion/schemas/registry.py` | Register schemas |
| `config/proposals/proposal_policy.v1.yaml` | Templates + thresholds |
| `orion/proposals/{policy,scoring,templates,builder}.py` | Synthesis |
| `services/orion-sql-db/manual_migration_proposal_frame_v1.sql` | DDL |
| `services/orion-proposal-runtime/` | Runtime service |
| `services/orion-hub/scripts/substrate_proposal_routes.py` | Debug API |
| `tests/test_proposal_*.py` | Unit tests |
| `scripts/smoke_proposal_frame_v1.sh` | Live SQL smoke |

---

## Implementation status

This plan was executed on branch `feat/proposal-frame-v1` (commits `138e1aac`, `75997c1f`). See `docs/superpowers/pr-reports/2026-05-24-proposal-frame-v1-pr.md` for PR summary and verification commands.
