#!/usr/bin/env bash
# PreToolUse gate in front of `graphify hook-guard <kind>`.
#
# graphify's hook-guard is a pure informational nudge (fails open, never
# blocks) telling the agent to run `graphify query` before grepping/reading
# source files -- useful when navigating/editing code, pure token overhead
# for a conversational/forensics turn with no code-navigation payoff.
#
# orion/harness/fcc_motor.py tags every FCC chat-turn subprocess with
# ORION_FCC_SUBPROCESS=1 (same marker used by session_start_agent_board.py /
# session_stop_agent_board.py). When set, skip the nudge entirely. Unlike
# destructive_git_guard.py (actual safety enforcement, left ungated even for
# FCC turns since HARNESS_FCC_WORKSPACE is the shared checkout), this hook
# has no enforcement value to preserve.
set -euo pipefail

if [ -n "${ORION_FCC_SUBPROCESS:-}" ]; then
  exit 0
fi

exec "${GRAPHIFY_HOOK_GUARD_BIN:-/home/athena/.local/bin/graphify}" hook-guard "$1"
