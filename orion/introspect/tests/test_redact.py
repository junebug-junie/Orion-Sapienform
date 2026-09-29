"""Log-safe exception detail: credentials never survive into responder logs."""
from orion.introspect.redact import safe_exception_detail


def test_dsn_password_is_redacted_but_host_kept():
    exc = RuntimeError('connection to postgresql://postgres:secret@db:5432/conjourney failed')
    detail = safe_exception_detail(exc)
    assert "secret" not in detail and "postgres:" not in detail.split("//", 1)[1]
    assert "postgresql://[REDACTED]@db:5432/conjourney" in detail


def test_driver_suffixed_dsn_is_redacted():
    exc = RuntimeError("engine postgresql+psycopg2://postgres:secret@db:5432/conjourney refused")
    detail = safe_exception_detail(exc)
    assert "secret" not in detail
    assert "postgresql+psycopg2://[REDACTED]@db:5432/conjourney" in detail
    assert "postgres+asyncpg://[REDACTED]@h/db" in safe_exception_detail(RuntimeError("postgres+asyncpg://u:p@h/db"))


def test_password_kv_forms_are_redacted():
    exc = RuntimeError("host=db password=hunter2 PWD='quoted pass' passwd=\"dq\" user=orion")
    detail = safe_exception_detail(exc)
    for leaked in ("hunter2", "quoted pass", "dq"):
        assert leaked not in detail
    assert "host=db" in detail and "user=orion" in detail
    assert detail.count("[REDACTED]") == 3


def test_newlines_flattened_and_truncated_after_redaction():
    exc = RuntimeError("postgres://u:p@h/db\n" + "x" * 1000)
    detail = safe_exception_detail(exc)
    assert "\n" not in detail and len(detail) == 300
    assert detail.startswith("postgres://[REDACTED]@h/db ")
    assert len(safe_exception_detail(exc, limit=1000)) == 1000
