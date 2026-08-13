"""Guards that the drive-audit SQL write path stays fully removed.

Was `test_drive_audit_sql_shape.py`, a shape-check suite for `DriveAuditSQL`'s
write path (MODEL_MAP registration, column shape, derivation logic,
insert-only idempotency). That whole write path was untangled and removed
2026-08-13 (docs/superpowers/pr-reports/
2026-08-13-untangle-drive-audit-sql-writer-pr.md), a same-day follow-up to
the Hub Drives Analytics tab removal
(docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-analytics-tab-pr.md)
that had already dropped the underlying `drive_audits` Postgres table and its
boot DDL but left the rest of the wiring in place, scoped out at the time
under a (subsequently-checked-and-wrong) belief that `DriveAuditSQL` shared a
`_JSONB` type declaration with other live sql-writer models -- it does not;
every other model file declares its own private copy of that one-line type
alias, so there was nothing to untangle.

This file replaces the old shape checks with the inverse: asserting the
model, its registrations, and the channel are all actually gone, so a future
change can't silently re-wire this without it being a deliberate, visible
diff here.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

CHANNEL = "orion:memory:drives:audit"


def test_drive_audit_sql_model_file_is_gone() -> None:
    assert not (SERVICE_ROOT / "app" / "models" / "drive_audit.py").exists()


def test_drive_audit_sql_not_exported_from_models_package() -> None:
    import app.models as models  # noqa: E402

    assert not hasattr(models, "DriveAuditSQL")
    assert "DriveAuditSQL" not in models.__all__


def test_drive_audit_sql_not_in_model_map_or_insert_only() -> None:
    from app.worker import INSERT_ONLY_MODELS, MODEL_MAP  # noqa: E402

    assert "DriveAuditSQL" not in MODEL_MAP
    assert all(m.__name__ != "DriveAuditSQL" for m in INSERT_ONLY_MODELS)


def test_route_map_no_longer_points_at_drive_audit_sql() -> None:
    from app.settings import DEFAULT_ROUTE_MAP  # noqa: E402

    assert "memory.drives.audit.v1" not in DEFAULT_ROUTE_MAP


def test_route_map_json_in_env_example_no_longer_has_drive_audit_entry() -> None:
    """DEFAULT_ROUTE_MAP is only the Python fallback -- `Settings.route_map`
    (app/settings.py) does `{**DEFAULT_ROUTE_MAP, **json.loads(sql_writer_route_map_json)}`,
    so SQL_WRITER_ROUTE_MAP_JSON's env value REPLACES/overrides the default per-key
    on top of it. Cleaning DEFAULT_ROUTE_MAP alone does not clean a stale key still
    sitting in the operator-facing env JSON (review finding: exactly this key was
    left behind in .env_example -- and the deployed .env -- the first time this
    file's route-map test was written)."""
    import json

    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_ROUTE_MAP_JSON="):
            route_map = json.loads(line.split("=", 1)[1])
            assert "memory.drives.audit.v1" not in route_map
            return
    raise AssertionError("SQL_WRITER_ROUTE_MAP_JSON not found in .env_example")


def test_settings_effective_route_map_no_longer_has_drive_audit_entry() -> None:
    """Exercises the actual runtime-effective path (`Settings().route_map`),
    not just the Python-side default, with an env override string standing in
    for `.env_example`'s current value -- catches a regression even if a
    future edit reintroduces the stale key only in one of the two places."""
    from app.settings import Settings  # noqa: E402

    settings = Settings(SQL_WRITER_ROUTE_MAP_JSON='{"some.other.kind.v1": "SomeOtherModel"}')
    assert "memory.drives.audit.v1" not in settings.route_map
    assert settings.route_map["some.other.kind.v1"] == "SomeOtherModel"


def test_channel_no_longer_in_settings_default_list() -> None:
    from app.settings import Settings  # noqa: E402

    default_channels = Settings.model_fields["sql_writer_subscribe_channels"].default
    assert CHANNEL not in default_channels


def test_channel_no_longer_in_env_example() -> None:
    env_example = (SERVICE_ROOT / ".env_example").read_text()
    for line in env_example.splitlines():
        if line.startswith("SQL_WRITER_SUBSCRIBE_CHANNELS="):
            assert CHANNEL not in line
            return
    raise AssertionError("SQL_WRITER_SUBSCRIBE_CHANNELS not found in .env_example")


def test_worker_no_longer_imports_drive_audit_v1() -> None:
    worker_py = (SERVICE_ROOT / "app" / "worker.py").read_text(encoding="utf-8")
    assert "DriveAuditSQL" not in worker_py
    assert "DriveAuditV1" not in worker_py
    assert "_apply_drive_audit_derivations" not in worker_py
