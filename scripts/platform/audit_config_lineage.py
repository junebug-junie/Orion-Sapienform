from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from scripts.platform._common import find_repo_root, read_text, write_json


def find_compose_files(repo_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in repo_root.rglob("docker-compose*.yml"):
        if p.is_file():
            out.append(p)
    for p in repo_root.rglob("docker-compose*.yaml"):
        if p.is_file():
            out.append(p)
    return sorted(set(out))


def compose_mentions_service(compose_text: str, service_name: str) -> bool:
    return re.search(r"^\s{2,}" + re.escape(service_name) + r"\s*:\s*$", compose_text, flags=re.MULTILINE) is not None


def compose_has_environment_block(compose_text: str, service_name: str) -> bool:
    m = re.search(r"^\s{2,}" + re.escape(service_name) + r"\s*:\s*$", compose_text, flags=re.MULTILINE)
    if not m:
        return False
    start = m.start()
    chunk = "\n".join(compose_text[start:].splitlines()[:300])
    return re.search(r"^\s+environment\s*:\s*$", chunk, flags=re.MULTILINE) is not None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/platform/audit_config_lineage.py <RUN_DIR>")
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    repo_root = find_repo_root(run_dir)

    services_dir = repo_root / "services"
    compose_files = find_compose_files(repo_root)
    compose_texts = [(p, read_text(p)) for p in compose_files]

    results: Dict[str, Any] = {}

    for svc_path in sorted([p for p in services_dir.iterdir() if p.is_dir()]):
        svc = svc_path.name
        env_example_exists = (svc_path / ".env_example").exists()
        dockerfile_exists = (svc_path / "Dockerfile").exists()
        requirements_exists = (svc_path / "requirements.txt").exists()
        settings_py_exists = (svc_path / "app" / "settings.py").exists()

        mentions = False
        env_block = False
        for cp, ct in compose_texts:
            if compose_mentions_service(ct, svc):
                mentions = True
                if compose_has_environment_block(ct, svc):
                    env_block = True

        missing = []
        if not env_example_exists:
            missing.append(".env_example")
        if not dockerfile_exists:
            missing.append("Dockerfile")
        if not requirements_exists:
            missing.append("requirements.txt")
        if not settings_py_exists:
            missing.append("app/settings.py")
        if mentions and not env_block:
            missing.append("compose environment block")
        if not mentions:
            missing.append("compose service stanza")

        results[svc] = {
            "env_example_exists": env_example_exists,
            "dockerfile_exists": dockerfile_exists,
            "requirements_exists": requirements_exists,
            "settings_py_exists": settings_py_exists,
            "compose_mentions_service": mentions,
            "compose_has_environment_block": env_block,
            "missing_items": missing,
        }

    write_json(run_dir / "reports" / "config_lineage.json", results)
    print(f"Wrote config lineage audit to {run_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
