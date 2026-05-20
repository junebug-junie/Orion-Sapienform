#!/usr/bin/env bash
# Orion Knowledge Forge CLI wrapper (replaces shell functions in .env — compose-safe).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ORION_KNOWLEDGE_ROOT="${ORION_KNOWLEDGE_ROOT:-$REPO_ROOT/orion-knowledge}"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
exec "$REPO_ROOT/venv/bin/python" -m orion.knowledge_forge "$@"
