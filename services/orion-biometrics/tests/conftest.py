"""Session-wide test setup for orion-biometrics.

`app.settings.settings` is a module-level singleton built once, the first time `app.settings`
(or anything importing it, like `app.main`) is imported anywhere in the pytest process --
whichever test file collects first wins the env. `NODE_CATALOG_PATH` defaults to the
container path `/app/config/biometrics/node_catalog.yaml`, which does not exist outside a
built image, so any test file importing `app.main` needs the real repo path set BEFORE that
first import -- setting it inside the test module itself (as test_power_intent_handler_wiring.py
and test_measurements_by_node.py both already do, each independently) only wins the race when
that file happens to be the one pytest collects first. A conftest.py is imported before every
test module in its directory, so this is the one place that setting is guaranteed to land
first regardless of collection order.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault(
    "NODE_CATALOG_PATH", str(_REPO_ROOT / "config" / "biometrics" / "node_catalog.yaml")
)
