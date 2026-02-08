import shutil
import subprocess
from pathlib import Path


def test_hub_app_js_parses():
    node = shutil.which("node")
    assert node is not None, "node is required to check Hub static JS syntax"
    repo_root = Path(__file__).resolve().parents[1]
    app_js = repo_root / "services" / "orion-hub" / "static" / "js" / "app.js"
    result = subprocess.run([node, "--check", str(app_js)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"node --check failed: {result.stderr or result.stdout}"
