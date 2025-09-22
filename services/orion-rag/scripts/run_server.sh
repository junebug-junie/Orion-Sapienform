#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
uvicorn app.server:app --reload
