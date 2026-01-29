import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM_PATH = REPO_ROOT / "orion" / "spark" / "telemetry" / "spark.py"

ALLOWED_DEFINITIONS = {
    "SparkTelemetryPayload": REPO_ROOT / "orion" / "schemas" / "telemetry" / "spark.py",
    "SparkStateSnapshotV1": REPO_ROOT / "orion" / "schemas" / "telemetry" / "spark.py",
    "SparkCandidateV1": REPO_ROOT / "orion" / "schemas" / "telemetry" / "spark_candidate.py",
    "SparkSignalV1": REPO_ROOT / "orion" / "schemas" / "telemetry" / "spark_signal.py",
}

DISALLOWED_IMPORT = "orion.spark.telemetry.spark"


def _iter_python_files() -> list[Path]:
    roots = [REPO_ROOT / "orion", REPO_ROOT / "services"]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            paths.append(path)
    return paths


def _parse_ast(path: Path) -> ast.Module:
    data = path.read_text(encoding="utf-8")
    return ast.parse(data, filename=str(path))


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


class TestSparkContractGate(unittest.TestCase):
    def test_no_duplicate_schema_definitions(self):
        offenders: dict[str, list[str]] = {name: [] for name in ALLOWED_DEFINITIONS}
        for path in _iter_python_files():
            tree = _parse_ast(path)
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in ALLOWED_DEFINITIONS:
                    allowed_path = ALLOWED_DEFINITIONS[node.name]
                    if path.resolve() != allowed_path.resolve():
                        offenders[node.name].append(_relative(path))

        failures = {name: paths for name, paths in offenders.items() if paths}
        if failures:
            details = "\n".join(
                f"{name}: {', '.join(paths)}" for name, paths in sorted(failures.items())
            )
            self.fail(f"Duplicate Spark schema class definitions detected:\n{details}")

    def test_shim_has_no_schema_class_defs(self):
        tree = _parse_ast(SHIM_PATH)
        shim_classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        forbidden = [name for name in shim_classes if name in ALLOWED_DEFINITIONS]
        if forbidden:
            self.fail(
                "Shadow shim must not define Spark schema classes; found: "
                + ", ".join(sorted(forbidden))
            )

    def test_no_shadow_imports(self):
        offenders: list[str] = []
        allow_paths = {_relative(SHIM_PATH), _relative(Path(__file__).resolve())}
        for path in _iter_python_files():
            rel = _relative(path)
            if rel in allow_paths:
                continue
            tree = _parse_ast(path)
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == DISALLOWED_IMPORT:
                    offenders.append(rel)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == DISALLOWED_IMPORT:
                            offenders.append(rel)
        if offenders:
            self.fail(
                "Shadow spark schema import detected in: " + ", ".join(sorted(set(offenders)))
            )

    def test_shim_identity(self):
        from orion.schemas.telemetry import spark as canonical
        from orion.spark.telemetry import spark as shim

        self.assertIs(shim.SparkTelemetryPayload, canonical.SparkTelemetryPayload)
        self.assertIs(shim.SparkStateSnapshotV1, canonical.SparkStateSnapshotV1)


if __name__ == "__main__":
    unittest.main()
