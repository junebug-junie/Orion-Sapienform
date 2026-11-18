"""The self-monitor organ (rung 6): Orion's track record of their own faculties.

Spec: docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md.
``adapters`` turn source rows into ``SettledBetV1``; ``record`` reduces them into one
``FacultyRecordV1`` per faculty; ``catalog`` names the faculties that cannot be graded yet and why;
``view`` is the reader contract (unknown changes nothing; one direction only); ``report`` prints
the live table (read-only): ``python -P -m orion.self_monitor.report``.
"""
