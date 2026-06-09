import sys
import os
import pytest

# Make app/ importable without hyphens (service lives in hyphenated directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
