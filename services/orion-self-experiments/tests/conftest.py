import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(ROOT, "app")
if APP not in sys.path:
    sys.path.insert(0, APP)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("ORION_BUS_ENABLED", "false")
os.environ.setdefault("SELF_EXPERIMENTS_DISPATCH_ENABLED", "false")
