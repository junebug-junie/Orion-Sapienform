# orion/spark/concept_induction — drive/tension/goal code DELETED (2026-07-30)

The `DriveEngine`, `tensions.py` bucket-voting logic, `signal_drive_map.yaml`,
and `GoalProposalEngine` this file used to describe are gone. This directory's
2026-07-18 halt ("development halted, code stays until replacement wiring
lands" — see `orion/sentience_striving_program/README.md` §8) has been
followed through: the drive-pressure/goal-generation system was deleted
outright 2026-07-30 (Wave 1 of the drive-pressure/goal-generation deletion
sprint), not merely frozen. `drives.py`, `tensions.py`, `drive_tension.py`,
`drive_attribution.py`, `goals.py`, `goal_generator.py`, and `audit.py` no
longer exist in this directory. Orion lost live goal-proposal capability from
this path as a direct, accepted consequence (Juniper-confirmed); no
field-native replacement exists yet.

Concept extraction/clustering/embedding/dossier/identity/profile_repository/
falkor_materialization/graph_mapper/graph_query/summarizer — the actual
reason `bus_worker.py`'s `ConceptWorker` exists — are untouched and still
run unconditionally on every intake event, independent of the deleted code.

If you land here looking for the math-bug retrospective
(`orion/autonomy/drives_and_autonomy_retrospective.md` §5b-§5e) or the
drive-taxonomy audit, they are still real historical documents, but they
describe code that no longer runs. Do not resurrect `DriveEngine`,
`_update_drive_pressures`, or the `signal_drive_map.yaml` bucket-voting
mechanism from them without first reading the deletion's rationale
(`orion/sentience_striving_program/README.md` §8) and confirming a
field-native replacement is actually what's being built — grafting the old
math back in would recreate the exact "poorer reimplementation of the
already-live canonical pipeline" problem that got this halted in the first
place.

`services/orion-field-digester` shares the same decay-mechanism bug *class*
this directory used to have (injection cadence not reconciled against decay
rate) — check that service's own `CLAUDE.md` if debugging something
structurally similar there; it is unrelated to this deletion.
