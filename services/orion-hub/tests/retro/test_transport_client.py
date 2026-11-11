"""Keep the real-asset JavaScript regression suite in the normal pytest gate."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_browser_transport_recovery():
    if not shutil.which('node'): pytest.skip('node not installed')
    run = subprocess.run(['node', '--test', str(Path(__file__).with_name('transport_client.cjs'))],
                         capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
