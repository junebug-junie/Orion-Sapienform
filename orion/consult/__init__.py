"""Consult: Orion takes a plumbing question to Claude before it reaches Juniper.

Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.
Pure helpers live in ``orion.consult.core``; Hub runs the loop
(``services/orion-hub/scripts/consult_loop.py``); the room companion runs Claude's turns.
"""
