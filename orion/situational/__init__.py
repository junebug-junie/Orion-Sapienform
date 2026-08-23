"""Shared situational-awareness building blocks (time/day-phase, presence,
weather, perception, runtime self-context) used by more than one service.

Originally private to `services/orion-cortex-exec/app/` (situation.py,
perception_reader.py, session_turn_phase.py); relocated here so
`orion-hub`'s unified-turn path can build the same situation brief without
reaching into another service's app/ directory (AGENTS.md section 5).
"""
