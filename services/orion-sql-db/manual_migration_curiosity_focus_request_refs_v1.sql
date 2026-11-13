-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Curiosity focus refs (Orion Prime step 1, 2026-10-30): the ids a queued question cites, as a
-- JSON list of {"kind", "id"} (orion/refs.py RefV1; kinds "arc" and "act"). Each one opens with
-- `python -P -m orion.dig <kind> <id>`, and the run's focus section prints that command
-- (orion/curiosity/focus.py format_focus_section). Written by Hub's queue_curiosity_focus responder
-- (services/orion-hub/scripts/curiosity_focus_listener.py); every focus read in orion/curiosity/focus.py
-- selects it, so Hub needs this column before it starts.
--
-- Additive and nullable: rows written before it, and questions that cite nothing, read NULL.

ALTER TABLE curiosity_focus_request ADD COLUMN IF NOT EXISTS refs JSONB;
