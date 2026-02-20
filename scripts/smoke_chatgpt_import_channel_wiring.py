from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) in sys.path:
    sys.path.remove(str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.bus.catalog_loader import load_channel_catalog
from orion.schemas.registry import resolve as resolve_schema_id

IMPORTER_SCRIPT = REPO_ROOT / "scripts" / "import_chatgpt_export.py"

EXPECTED = {
    "orion:chat:gpt:log": "ChatGptMessageV1",
    "orion:chat:gpt:turn": "ChatGptLogTurnV1",
}


def _assert_channel_wiring() -> None:
    catalog = load_channel_catalog()
    for channel, schema_id in EXPECTED.items():
        entry = catalog.get(channel)
        if not entry:
            raise SystemExit(f"missing channel in catalog: {channel}")
        actual = entry.get("schema_id")
        print(f"{channel} => {actual}")
        if actual != schema_id:
            raise SystemExit(
                f"channel wiring mismatch for {channel}: expected {schema_id}, found {actual}"
            )
        # Ensure the shared schema registry can resolve each schema_id.
        resolve_schema_id(schema_id)


def _sample_export_file() -> tuple[Path, Path]:
    sample = [
        {
            "id": "conv-1",
            "title": "Smoke Test Conversation",
            "create_time": 1700000000.0,
            "update_time": 1700000001.0,
            "current_node": "assistant-node",
            "mapping": {
                "user-node": {
                    "id": "user-node",
                    "parent": None,
                    "message": {
                        "id": "msg-user",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0,
                        "content": {"content_type": "text", "parts": ["hello"]},
                        "metadata": {},
                    },
                },
                "assistant-node": {
                    "id": "assistant-node",
                    "parent": "user-node",
                    "message": {
                        "id": "msg-assistant",
                        "author": {"role": "assistant"},
                        "create_time": 1700000001.0,
                        "content": {"content_type": "text", "parts": ["hi there"]},
                        "metadata": {"model_slug": "gpt-test", "provider": "openai"},
                    },
                },
            },
        }
    ]
    temp_dir = Path(tempfile.mkdtemp(prefix="chatgpt-smoke-"))
    sample_path = temp_dir / "conversations.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    return sample_path, temp_dir


def _run_importer_dry_run(export_file: Path) -> None:
    cmd = [
        sys.executable,
        str(IMPORTER_SCRIPT),
        "--export",
        str(export_file),
        "--channel-log",
        "orion:chat:gpt:log",
        "--channel-turn",
        "orion:chat:gpt:turn",
        "--dry-run",
        "--limit",
        "1",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "importer dry-run failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print("importer dry-run succeeded")


def main() -> int:
    _assert_channel_wiring()
    export_path, temp_dir = _sample_export_file()
    try:
        _run_importer_dry_run(export_path)
    finally:
        export_path.unlink(missing_ok=True)
        temp_dir.rmdir()
    print("smoke_chatgpt_import_channel_wiring: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
