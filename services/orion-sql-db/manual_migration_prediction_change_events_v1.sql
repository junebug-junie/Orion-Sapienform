-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: change events -- known moments the world changed under a series, used by the
-- forecaster as changepoints (effect=break) or masked-out windows (effect=exclude).
-- Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md ("Change events").
--
-- Additive only (post-pull auto-apply allowlist): one small table (no index needed) and the events found in
-- the athena biometrics history on 2026-10-20.

BEGIN;

CREATE TABLE IF NOT EXISTS prediction_change_event (
    event_id    uuid PRIMARY KEY,
    scope       varchar(120) NOT NULL,      -- a series_id, or '*' for every series
    effect      varchar(16)  NOT NULL CHECK (effect IN ('break', 'exclude')),
    started_at  timestamptz  NOT NULL,
    ended_at    timestamptz,                -- required for exclude, unused for break
    title       varchar(200) NOT NULL,
    notes       text         NOT NULL DEFAULT '',
    source      varchar(80)  NOT NULL,
    created_at  timestamptz  NOT NULL DEFAULT now(),
    retracted_at timestamptz,               -- set by DELETE /change-events/{id}; retracted events are ignored
    CHECK (effect <> 'exclude' OR (ended_at IS NOT NULL AND ended_at > started_at))
);

-- athena GPU0 swapped to the passive T10 (vision): gpu0_temp_c 70 -> 55 C between 18:54 and 19:00 UTC,
-- fan_pct_max floor ~44 -> ~65 %, idle chassis_watts ~385 -> ~425 W.
INSERT INTO prediction_change_event (event_id, scope, effect, started_at, ended_at, title, notes, source, created_at)
VALUES ('6d1f5c0a-6a11-4c6e-9d2b-0a1e5b7c1001', '*', 'break', '2026-10-15 18:57:00+00', NULL,
        'athena GPU0 -> T10 (vision)', 'fans and chassis power step up; more heat into the cabinet', 'seed-2026-10-20', '2026-10-20 06:00:00+00')
ON CONFLICT DO NOTHING;

-- Cabinet opened for the vision hardware bring-up: cabinet_lidar_mm erratic (157/204/35/125) and the
-- temperature dips to 25.5 C while open; settles at lidar ~94 mm (was ~116).
-- Corrected 2026-10-29 (Juniper): the sensor never moved. The ~2 C after 00:35 is real heat from the
-- vision bring-up; the lidar/magnetic change is something near the sensor (likely the new hardware).
-- ON CONFLICT DO NOTHING means an existing row keeps its old title: the live row is fixed by the
-- one-time UPDATE in docs/superpowers/pr-reports/2026-10-29-cabinet-sensor-offset-was-real-heat-pr.md.
INSERT INTO prediction_change_event (event_id, scope, effect, started_at, ended_at, title, notes, source, created_at)
VALUES ('6d1f5c0a-6a11-4c6e-9d2b-0a1e5b7c1002', '*', 'exclude', '2026-10-16 21:30:00+00', '2026-10-17 00:35:00+00',
        'cabinet open: vision bring-up', 'door-open disturbance; not representative of either regime', 'seed-2026-10-20', '2026-10-20 06:00:00+00')
ON CONFLICT DO NOTHING;
INSERT INTO prediction_change_event (event_id, scope, effect, started_at, ended_at, title, notes, source, created_at)
VALUES ('6d1f5c0a-6a11-4c6e-9d2b-0a1e5b7c1003', '*', 'break', '2026-10-17 00:35:00+00', NULL,
        'vision services brought up (real heat)', 'cabinet reads ~2 C higher after: real heat, not a sensor move (the sensor never moved; Juniper 2026-10-29). lidar 116 -> 94 mm, magnetic 36 -> 33.5 uT: something near the sensor changed, likely the new vision hardware.', 'seed-2026-10-20', '2026-10-20 06:00:00+00')
ON CONFLICT DO NOTHING;

COMMIT;

-- Undo: DROP TABLE prediction_change_event;
