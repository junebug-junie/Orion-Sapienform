from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

MODELS_DREAMS_PATH = SQL_WRITER_ROOT / "app" / "models" / "dreams.py"
SPEC = importlib.util.spec_from_file_location("sql_writer_models_dreams_tests", MODELS_DREAMS_PATH)
assert SPEC and SPEC.loader
dreams = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dreams)


def test_dream_date_is_not_unique() -> None:
    # Multiple dream artifacts can be persisted on the same calendar day.
    assert bool(dreams.Dream.__table__.c.dream_date.unique) is False
