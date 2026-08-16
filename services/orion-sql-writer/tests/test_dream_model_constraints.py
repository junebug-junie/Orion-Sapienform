from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SQL_WRITER_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SQL_WRITER_ROOT) not in sys.path:
    sys.path.insert(0, str(SQL_WRITER_ROOT))

# Imported normally, NOT via importlib.util.spec_from_file_location.
#
# This file used to load app/models/dreams.py directly under a synthetic
# module name. That re-executes the module body, so `class Dream(Base)` ran a
# second time against the SAME shared Base.metadata that app.models had
# already populated -- raising
#
#   InvalidRequestError: Table 'dreams' is already defined for this MetaData
#
# at COLLECTION time. Because a collection error is fatal, that aborted the
# entire orion-sql-writer suite: running `pytest services/orion-sql-writer/tests`
# from the repo root executed ZERO tests, while the file passed in isolation
# (nothing had imported app.models yet), which is why it survived so long.
#
# `Dream` is a normal export of app.models (app/models/__init__.py:13,93), so
# there was never a reason to bypass the package. The 15 sibling tests that
# importlib-load worker.py are unaffected: worker.py reaches its models through
# ordinary imports, which sys.modules dedupes.
from app.models import Dream  # noqa: E402


def test_dream_date_is_not_unique() -> None:
    # Multiple dream artifacts can be persisted on the same calendar day.
    assert bool(Dream.__table__.c.dream_date.unique) is False
